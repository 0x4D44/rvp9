//! Vectorless coverage for filtering after scratch-backed tile decode.
//!
//! Source guards prove removal of the former dormant construction. They are
//! content checks, not a general proof of runtime aliasing soundness.

use rvp9_core::{bitstream::FrameKind, frame::FrameBuf};
use rvp9_decoder::Decoder;
use rvp9_encoder::{EncoderConfig, EncoderParallelism, StreamEncoder, YuvFrameRef};
use std::{num::NonZeroUsize, sync::atomic::Ordering};

fn assert_no_predecode_overlap(start: &str, end: &str) {
    let source = include_str!("../src/lib.rs");
    let body = source
        .split_once(start)
        .expect("decode method start")
        .1
        .split_once(end)
        .expect("decode method end")
        .0;
    for construction in [
        "SharedFrame",
        "RowMaskInputs",
        "Non420Inputs",
        "ColumnWavefrontLfState",
    ] {
        assert!(
            !body.contains(construction),
            "{start} retains pre-decode {construction}"
        );
    }
    assert!(
        !source.contains("OUTPUT_FRAME_LF_OVERLAP_ALIAS_SAFE"),
        "a boolean must not rearm the removed construction"
    );
}

#[test]
fn intra_driver_has_no_predecode_overlap_construction() {
    assert_no_predecode_overlap("fn decode_intra_frame_inner(", "fn decode_inter_frame(");
}

#[test]
fn inter_driver_has_no_predecode_overlap_construction() {
    assert_no_predecode_overlap("fn decode_inter_frame(", "fn commit_inter_segment_map(");
}

fn packets() -> Vec<Vec<u8>> {
    let mut config = EncoderConfig::new(512, 128);
    config.min_qindex = 100;
    config.max_qindex = 100;
    let mut encoder = StreamEncoder::new_with_parallelism(
        config,
        EncoderParallelism {
            max_threads: 1,
            requested_log2_tile_cols: 1,
        },
    )
    .expect("encoder");
    (0..2)
        .map(|frame| {
            let y: Vec<u8> = (0..512 * 128)
                .map(|i| {
                    let x = i % 512 + frame * 3;
                    let row = i / 512;
                    ((x * 5 + row * 3 + ((x / 13 + row / 11) % 2) * 45) % 220 + 16) as u8
                })
                .collect();
            let u: Vec<u8> = (0..256 * 64)
                .map(|i| (80 + (i % 256 + i / 256 + frame) % 80) as u8)
                .collect();
            let v: Vec<u8> = (0..256 * 64)
                .map(|i| (90 + (i % 256 * 3 + i / 256 + frame) % 70) as u8)
                .collect();
            encoder
                .encode_frame(&YuvFrameRef {
                    y: &y,
                    u: &u,
                    v: &v,
                    y_stride: 512,
                    u_stride: 256,
                    v_stride: 256,
                })
                .expect("encode")
                .data
        })
        .collect()
}

fn visible(frame: &FrameBuf) -> Vec<u8> {
    let mut pixels = Vec::new();
    for plane in [frame.y_plane(), frame.u_plane(), frame.v_plane()] {
        for row in 0..plane.height() {
            pixels.extend_from_slice(
                &plane.data()[row * plane.stride()..row * plane.stride() + plane.width()],
            );
        }
    }
    pixels
}

fn decode(decoder: &mut Decoder, packet: &[u8], no_filter: bool, inter: bool) -> Vec<u8> {
    let frames = decoder.parse_packet(packet).expect("parse");
    assert_eq!(frames.count(), 1);
    let mut frame = frames.into_iter().next().expect("one frame");
    assert_eq!(
        matches!(frame.header.kind, FrameKind::InterFrame { .. }),
        inter
    );
    let tail = frame.header.tail.as_mut().expect("coded frame");
    assert_eq!(
        tail.tile_info.log2_tile_cols, 1,
        "two tile columns required"
    );
    assert_eq!(tail.tile_info.log2_tile_rows, 0);
    assert!(tail.loop_filter.filter_level > 0, "fixture must request LF");
    if no_filter {
        tail.loop_filter.filter_level = 0;
    }
    visible(&decoder.decode_frame(&frame).expect("decode"))
}

#[test]
fn generated_key_and_inter_tiles_match_serial_and_are_filtered() {
    let packets = packets();
    let mut serial = Decoder::with_threads(NonZeroUsize::new(1).unwrap());
    let mut parallel = Decoder::with_threads(NonZeroUsize::new(2).unwrap());
    let before = rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(Ordering::Relaxed);
    for (index, packet) in packets.iter().enumerate() {
        let expected = decode(&mut serial, packet, false, index != 0);
        let actual = decode(&mut parallel, packet, false, index != 0);
        assert!(actual == expected, "frame {index}: parallel LF differs");
        // Rebuild the exact filtered reference history before suppressing LF
        // on this frame alone, so inter differences cannot come from its refs.
        let mut unfiltered = Decoder::with_threads(NonZeroUsize::new(1).unwrap());
        for previous in &packets[..index] {
            decode(&mut unfiltered, previous, false, false);
        }
        let without_lf = decode(&mut unfiltered, packet, true, index != 0);
        assert!(
            actual != without_lf,
            "frame {index}: post-decode LF must change visible pixels"
        );
    }
    assert!(
        rvp9_decoder::RVP9_G5_1_PARALLEL_JOBS.load(Ordering::Relaxed) >= before + 4,
        "both frames must submit two tile jobs"
    );
}
