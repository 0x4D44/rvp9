//! Fast in-process correctness coverage for the focused realtime feature matrix.
//!
//! This deliberately uses a small deterministic clip instead of an external
//! corpus.  The matrix is broad enough to catch interactions between the
//! realtime presets, one-pass rate-control modes, cyclic-refresh AQ, temporal
//! denoising, and temporal SVC while keeping the test suitable for debug runs.

use rvp9_decoder::Decoder;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use std::num::NonZeroUsize;

const WIDTH: usize = 128;
const HEIGHT: usize = 64;
const FRAME_COUNT: usize = 12;
const TARGET_BITRATE_BPS: u32 = 600_000;

type SourceFrame = (Vec<u8>, Vec<u8>, Vec<u8>);

#[derive(Copy, Clone, Debug)]
enum Scenario {
    Baseline,
    Aq3,
    Denoiser,
    Temporal2,
    Temporal3,
    CombinedAq3DenoiserTemporal3,
}

impl Scenario {
    const ALL: [Self; 6] = [
        Self::Baseline,
        Self::Aq3,
        Self::Denoiser,
        Self::Temporal2,
        Self::Temporal3,
        Self::CombinedAq3DenoiserTemporal3,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Aq3 => "aq3",
            Self::Denoiser => "denoiser",
            Self::Temporal2 => "temporal2",
            Self::Temporal3 => "temporal3",
            Self::CombinedAq3DenoiserTemporal3 => "aq3+denoiser+t3",
        }
    }

    const fn temporal_layers(self) -> u32 {
        match self {
            Self::Temporal2 => 2,
            Self::Temporal3 | Self::CombinedAq3DenoiserTemporal3 => 3,
            Self::Baseline | Self::Aq3 | Self::Denoiser => 1,
        }
    }

    const fn aq_mode(self) -> AqMode {
        match self {
            Self::Aq3 | Self::CombinedAq3DenoiserTemporal3 => AqMode::CyclicRefresh,
            Self::Baseline | Self::Denoiser | Self::Temporal2 | Self::Temporal3 => AqMode::Off,
        }
    }

    const fn noise_sensitivity(self) -> u8 {
        match self {
            Self::Denoiser | Self::CombinedAq3DenoiserTemporal3 => 3,
            Self::Baseline | Self::Aq3 | Self::Temporal2 | Self::Temporal3 => 0,
        }
    }
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: WIDTH,
        u_stride: WIDTH / 2,
        v_stride: WIDTH / 2,
    }
}

fn source_frames() -> Vec<SourceFrame> {
    (0..FRAME_COUNT)
        .map(|frame_idx| {
            let y = (0..WIDTH * HEIGHT)
                .map(|index| {
                    let row = index / WIDTH;
                    let col = index % WIDTH;
                    let moving_col = (col + frame_idx * 5) % WIDTH;
                    let checker = ((moving_col / 16) + (row / 16)) & 1;
                    let noise = (col * 17 + row * 29 + frame_idx * 43 + col * row) & 31;
                    ((moving_col * 3 + row * 7 + checker * 52 + noise) & 0xff) as u8
                })
                .collect();
            let u = (0..(WIDTH / 2) * (HEIGHT / 2))
                .map(|index| {
                    let row = index / (WIDTH / 2);
                    let col = index % (WIDTH / 2);
                    (96 + ((col * 5 + row * 3 + frame_idx * 7) & 31)) as u8
                })
                .collect();
            let v = (0..(WIDTH / 2) * (HEIGHT / 2))
                .map(|index| {
                    let row = index / (WIDTH / 2);
                    let col = index % (WIDTH / 2);
                    (160 + ((col * 3 + row * 7 + frame_idx * 11) & 31)) as u8
                })
                .collect();
            (y, u, v)
        })
        .collect()
}

fn config(speed: i32, rc_mode: RcMode, scenario: Scenario) -> EncoderConfig {
    EncoderConfig {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        framerate: 30.0,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        // One keyframe followed by eleven inter frames exercises both packet
        // kinds and keeps the temporal pattern continuous across the clip.
        gop_length: FRAME_COUNT as u32 + 1,
        aq_mode: scenario.aq_mode(),
        use_prev_frame_mvs: false,
        speed,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: scenario.noise_sensitivity(),
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode,
        cq_level: 0,
        temporal_layers: scenario.temporal_layers(),
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    }
}

fn expected_temporal_id(temporal_layers: u32, frame_index: usize) -> u8 {
    if frame_index == 0 {
        return 0;
    }
    match temporal_layers {
        1 => 0,
        2 => (frame_index & 1) as u8,
        3 => [2, 1, 2, 0][(frame_index - 1) & 3],
        _ => unreachable!("the matrix only exercises one, two, and three layers"),
    }
}

fn visible_i420(frame: &rvp9_core::frame::FrameBuf) -> Vec<u8> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let mut output = Vec::with_capacity(width * height * 3 / 2);
    for (plane, row_width, rows) in [
        (frame.y_plane(), width, height),
        (frame.u_plane(), uv_width, uv_height),
        (frame.v_plane(), uv_width, uv_height),
    ] {
        for row in 0..rows {
            let offset = row * plane.stride();
            output.extend_from_slice(&plane.data()[offset..offset + row_width]);
        }
    }
    output
}

#[test]
fn realtime_feature_interaction_matrix_decodes_every_row() {
    let frames = source_frames();
    let mut exercised_rows = 0;

    for speed in [6, 7, 8] {
        for rc_mode in [RcMode::Cbr, RcMode::Vbr] {
            for scenario in Scenario::ALL {
                exercised_rows += 1;
                let label = format!("speed={speed} rc={rc_mode:?} scenario={}", scenario.label());
                let config = config(speed, rc_mode, scenario);
                let mut encoder = StreamEncoder::new(config).unwrap_or_else(|error| {
                    panic!("{label}: encoder construction failed: {error}")
                });
                // The matrix checks encoder feature interactions, not decoder
                // scheduling. Keep one deterministic decoder worker instead
                // of spawning the host's full parallelism for every row.
                let mut decoder = Decoder::with_threads(NonZeroUsize::MIN);
                let mut keyframes = 0;
                let mut interframes = 0;
                let mut shown_frames = 0;
                let mut layer_coverage = vec![false; config.temporal_layers as usize];
                let mut packets = Vec::with_capacity(FRAME_COUNT);
                let mut temporal_ids = Vec::with_capacity(FRAME_COUNT);
                let mut full_outputs = Vec::with_capacity(FRAME_COUNT);

                for (frame_index, (y, u, v)) in frames.iter().enumerate() {
                    let packet = encoder
                        .encode_frame(&yuv_view(y, u, v))
                        .unwrap_or_else(|error| {
                            panic!("{label} frame {frame_index}: encode failed: {error}")
                        });
                    assert!(
                        !packet.data.is_empty(),
                        "{label} frame {frame_index}: empty packet"
                    );
                    assert_eq!(
                        packet.actual_bits as usize,
                        packet.data.len() * 8,
                        "{label} frame {frame_index}: incorrect packet bit count"
                    );
                    assert_eq!(
                        packet.temporal_layer_id,
                        expected_temporal_id(config.temporal_layers, frame_index),
                        "{label} frame {frame_index}: unexpected temporal layer"
                    );
                    layer_coverage[packet.temporal_layer_id as usize] = true;

                    if packet.frame_kind.is_key() {
                        keyframes += 1;
                    } else {
                        interframes += 1;
                    }

                    let parsed = decoder.parse_packet(&packet.data).unwrap_or_else(|error| {
                        panic!("{label} frame {frame_index}: parse failed: {error}")
                    });
                    assert_eq!(
                        parsed.count(),
                        1,
                        "{label} frame {frame_index}: temporal stream must emit one frame per packet"
                    );
                    let output_index = parsed.output_frame_index();
                    assert_eq!(
                        output_index,
                        Some(0),
                        "{label} frame {frame_index}: packet output index must select its shown frame"
                    );

                    let mut output = None;
                    for (component_index, parsed_frame) in parsed.frames().iter().enumerate() {
                        let decoded = decoder.decode_frame(parsed_frame).unwrap_or_else(|error| {
                            panic!(
                                "{label} frame {frame_index} component {component_index}: decode failed: {error}"
                            )
                        });
                        assert_eq!(
                            (decoded.width(), decoded.height()),
                            (WIDTH as u32, HEIGHT as u32),
                            "{label} frame {frame_index} component {component_index}: decoded dimensions changed"
                        );
                        if output_index == Some(component_index) {
                            shown_frames += 1;
                            output = Some(visible_i420(&decoded));
                        }
                    }
                    packets.push(packet.data);
                    temporal_ids.push(packet.temporal_layer_id);
                    full_outputs.push(output.expect("shown packet must produce one output"));
                }

                assert_eq!(keyframes, 1, "{label}: expected one keyframe");
                assert_eq!(
                    interframes,
                    FRAME_COUNT - 1,
                    "{label}: expected inter coverage"
                );
                assert_eq!(
                    shown_frames, FRAME_COUNT,
                    "{label}: decoded output count changed"
                );
                assert!(
                    layer_coverage.iter().all(|covered| *covered),
                    "{label}: every configured temporal layer must be exercised"
                );

                // Dropping enhancement packets must leave every retained
                // operating point byte-identical to the same frames decoded
                // from the complete stream. This is the temporal-SVC
                // reference-lifetime invariant spatial work must not regress.
                for max_temporal_id in 0..config.temporal_layers.saturating_sub(1) as u8 {
                    let mut retained_decoder = Decoder::with_threads(NonZeroUsize::MIN);
                    for (frame_index, ((packet, temporal_id), full_output)) in packets
                        .iter()
                        .zip(&temporal_ids)
                        .zip(&full_outputs)
                        .enumerate()
                    {
                        if *temporal_id > max_temporal_id {
                            continue;
                        }
                        let parsed = retained_decoder.parse_packet(packet).unwrap_or_else(|error| {
                            panic!(
                                "{label} TL{max_temporal_id} frame {frame_index}: parse failed: {error}"
                            )
                        });
                        let decoded = retained_decoder
                            .decode_frame(&parsed.frames()[0])
                            .unwrap_or_else(|error| {
                                panic!(
                                    "{label} TL{max_temporal_id} frame {frame_index}: decode failed: {error}"
                                )
                            });
                        let retained_output = visible_i420(&decoded);
                        assert!(
                            retained_output.as_slice() == full_output.as_slice(),
                            "{label} TL{max_temporal_id} frame {frame_index}: retained reconstruction changed"
                        );
                    }
                }
            }
        }
    }

    assert_eq!(
        exercised_rows, 36,
        "the focused matrix must contain 36 rows"
    );
}
