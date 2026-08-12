//! Cross-decoder gate for the keyframe-only spatial access-unit plumbing.
//!
//! This is not the final spatial-SVC oracle: the production stream will use
//! inter-layer prediction and persistent reference slots. It does lock the
//! packet boundary all later spatial work builds on.

use std::path::{Path, PathBuf};
use std::process::Command;

use rvp9_decoder::Decoder;
use rvp9_encoder::{
    EncoderConfig, SpatialKeyframeEncoder, YuvFrameRef,
    svc::{InterLayerPred, SpatialSvcConfig},
};
use rvp9_harness::container::{IvfHeader, write_ivf};
use sha2::{Digest, Sha256};

const WIDTH: usize = 128;
const HEIGHT: usize = 64;

fn source(frame_index: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = (0..WIDTH * HEIGHT)
        .map(|index| {
            let row = index / WIDTH;
            let column = index % WIDTH;
            ((column * 13 + row * 7 + frame_index * 19) & 0xff) as u8
        })
        .collect();
    let uv_len = WIDTH * HEIGHT / 4;
    let u = (0..uv_len)
        .map(|index| ((index * 5 + frame_index * 11 + 73) & 0xff) as u8)
        .collect();
    let v = (0..uv_len)
        .map(|index| ((index * 3 + frame_index * 17 + 149) & 0xff) as u8)
        .collect();
    (y, u, v)
}

fn encode_packets() -> Vec<Vec<u8>> {
    let mut template = EncoderConfig::new(WIDTH as u32, HEIGHT as u32);
    template.target_bitrate_bps = 100_000;
    template.speed = 8;
    let mut encoder = SpatialKeyframeEncoder::new(
        template,
        SpatialSvcConfig {
            number_spatial_layers: 2,
            inter_layer_pred: InterLayerPred::On,
        },
    )
    .unwrap();
    (0..2)
        .map(|frame_index| {
            let (y, u, v) = source(frame_index);
            encoder
                .encode_keyframe_access_unit(&YuvFrameRef {
                    y: &y,
                    u: &u,
                    v: &v,
                    y_stride: WIDTH,
                    u_stride: WIDTH / 2,
                    v_stride: WIDTH / 2,
                })
                .unwrap()
                .data
        })
        .collect()
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

struct DecodedComponent {
    data: Vec<u8>,
    decoded: Vec<u8>,
    width: u16,
    height: u16,
}

fn decode_rvp9(packets: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>, Vec<DecodedComponent>) {
    let mut decoder = Decoder::new();
    let mut every_component = Vec::new();
    let mut top_components = Vec::new();
    let mut components = Vec::new();
    for packet in packets {
        let parsed = decoder.parse_packet(packet).unwrap();
        let top_index = parsed.output_frame_index().unwrap();
        for (component_index, component) in parsed.frames().iter().enumerate() {
            let data = component.full_frame_data.to_vec();
            let decoded = decoder.decode_frame(component).unwrap();
            let bytes = visible_i420(&decoded);
            every_component.extend_from_slice(&bytes);
            if component_index == top_index {
                top_components.extend_from_slice(&bytes);
            }
            components.push(DecodedComponent {
                data,
                decoded: bytes,
                width: decoded.width().try_into().unwrap(),
                height: decoded.height().try_into().unwrap(),
            });
        }
    }
    (every_component, top_components, components)
}

fn write_packets(path: &Path, packets: &[Vec<u8>], width: u16, height: u16) {
    let refs = packets.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut file = std::fs::File::create(path).unwrap();
    write_ivf(&mut file, &IvfHeader::vp9(width, height, 30, 1), &refs).unwrap();
}

fn temporary_paths(label: &str) -> (PathBuf, PathBuf) {
    let stem = format!("rvp9-spatial-keyframe-{}-{label}", std::process::id());
    (
        std::env::temp_dir().join(format!("{stem}.ivf")),
        std::env::temp_dir().join(format!("{stem}.yuv")),
    )
}

#[test]
#[ignore = "external spatial packet gate; requires ffmpeg"]
fn keyframe_spatial_packet_decodes_and_components_are_byte_exact_through_ffmpeg() {
    let packets = encode_packets();
    let (_, _, components) = decode_rvp9(&packets);
    let (ivf, _output) = temporary_paths("ffmpeg");
    write_packets(&ivf, &packets, WIDTH as u16, HEIGHT as u16);
    let process = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&ivf)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg must be installed for this gate");
    let _ = std::fs::remove_file(&ivf);
    assert!(
        process.status.success(),
        "ffmpeg failed to decode the packed access units: {}",
        String::from_utf8_lossy(&process.stderr)
    );

    for (index, component) in components.iter().enumerate() {
        let (ivf, output) = temporary_paths(&format!("ffmpeg-component-{index}"));
        write_packets(
            &ivf,
            std::slice::from_ref(&component.data),
            component.width,
            component.height,
        );
        let process = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&ivf)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&output)
            .output()
            .expect("ffmpeg must be installed for this gate");
        let ffmpeg = std::fs::read(&output).unwrap();
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&ivf);
        assert!(
            process.status.success(),
            "ffmpeg failed to decode component {index}: {}",
            String::from_utf8_lossy(&process.stderr)
        );
        assert_eq!(
            Sha256::digest(&ffmpeg),
            Sha256::digest(&component.decoded),
            "ffmpeg component {index} differs at {}x{} ({} vs {} bytes)",
            component.width,
            component.height,
            ffmpeg.len(),
            component.decoded.len()
        );
    }
}

#[test]
#[ignore = "external spatial packet gate; requires pinned vpxdec"]
fn keyframe_spatial_packet_top_layer_is_byte_exact_through_vpxdec() {
    let packets = encode_packets();
    let (_, top_components, _) = decode_rvp9(&packets);
    let (ivf, output) = temporary_paths("vpxdec");
    write_packets(&ivf, &packets, WIDTH as u16, HEIGHT as u16);
    let process = Command::new("vpxdec")
        .args(["--codec=vp9", "--i420", "-o"])
        .arg(&output)
        .arg(&ivf)
        .output()
        .expect("pinned vpxdec must be on PATH for this gate");
    assert!(
        process.status.success(),
        "vpxdec failed: {}",
        String::from_utf8_lossy(&process.stderr)
    );
    let vpxdec = std::fs::read(&output).unwrap();
    assert_eq!(Sha256::digest(&vpxdec), Sha256::digest(&top_components));
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(ivf);
}
