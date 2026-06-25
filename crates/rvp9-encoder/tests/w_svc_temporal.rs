// Integration test: stderr carries the per-leg verdict (same rationale
// as `cross_decoder_gate.rs`). The workspace's `print_stderr = "deny"`
// lint targets library code; for an integration-test binary it is the
// wrong rule.
#![allow(clippy::print_stderr)]

//! A18 (SVC) — TEMPORAL SVC integration PROOF.
//!
//! Wires the `rvp9-encoder::svc` scaffold (the libvpx
//! `vp9_svc_layercontext.c` port) into the encode path and proves the
//! integration is live, not stubbed.
//!
//! ## What temporal SVC is (and the wire-format truth)
//!
//! VP9 **temporal SVC** (frame-rate scalability) encodes each access
//! unit as a single standalone VP9 frame. Temporal layers do NOT pack
//! into a superframe — superframes pack only SPATIAL layers within an
//! access unit (libvpx `vp9_svc_layercontext.c:451`
//! `++cpi->svc.current_superframe` is per access unit; with one spatial
//! layer that is one frame). The `temporal_layer_id` is encoder-side
//! metadata (libvpx exposes it via `vpx_svc_layer_id`), NOT a VP9
//! bitstream field. What carries the layering on the wire is the
//! per-frame REFERENCE / REFRESH structure the temporal setters compute
//! (`set_flags_and_fb_idx_for_temporal_mode2`,
//! `vp9_svc_layercontext.c:626-684`): enhancement-layer frames are
//! non-reference (they refresh nothing the base layer reads), so the
//! base layer skips over them — the stream is temporally scalable.
//!
//! vpxdec decodes a temporal-SVC stream as a normal VP9 frame sequence
//! (it plays ALL layers), so the byte-exact oracle is the same shape as
//! the single-layer cross-decoder gate.
//!
//! ## Tests (run with `--ignored`)
//!
//! 1. [`svc_temporal_layer_id_pattern_is_live`] — anti-G3. Encodes with
//!    `temporal_layers = 2` and asserts the per-frame `temporal_layer_id`
//!    follows the libvpx 0-1-0-1 pattern (KF = tl0, then the inter
//!    frames alternate 0,1,0,1). ALSO asserts the SVC bitstream DIFFERS
//!    from the single-layer (`temporal_layers = 1`) encode of the same
//!    source — because the enhancement frames flip `refresh_frame_flags`
//!    from `0x01` to `0x00`, the bytes must change. A stubbed/inert SVC
//!    path (every frame tagged tl0, refresh unchanged) FAILS both arms.
//!
//! 2. [`svc_temporal_stream_byte_exact_through_vpxdec`] — the oracle.
//!    Encodes a temporal-SVC stream, wraps it in IVF, and asserts vpxdec
//!    (and rvp9-decoder) decode it byte-exact (all layers play).

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Shared geometry.
// ---------------------------------------------------------------------------

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 8;
const TARGET_BITRATE_BPS: u32 = 1_000_000;

// ---------------------------------------------------------------------------
// Helpers (inlined — mirror `cross_decoder_gate.rs`; that file's helpers
// are private to its own test binary).
// ---------------------------------------------------------------------------

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

/// Build an `EncoderConfig` for the gate geometry with the supplied
/// temporal-layer count (1 = single-layer, 2 = 0-1-0-1 temporal SVC).
fn svc_config(framerate: f64, temporal_layers: u32) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        gop_length: N_FRAMES + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        min_qindex: 4,
        max_qindex: 255,
        temporal_layers,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
    }
}

/// Encode the source frames with the given config; returns
/// `(packets, per_frame_temporal_layer_id)`.
fn encode_clip(
    cfg: EncoderConfig,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    let mut tids = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        tids.push(pkt.temporal_layer_id);
        packets.push(pkt.data);
    }
    (packets, tids)
}

fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Vec<ParsedFrame<'a>> {
    let parsed = dec.parse_packet(pkt).expect("parse_packet");
    parsed
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect()
}

fn yv12_to_i420_bytes(buf: &rvp9_core::frame::Yv12Buffer, w: u32, h: u32) -> Vec<u8> {
    let yp = buf.y_plane();
    let up = buf.u_plane();
    let vp = buf.v_plane();
    let w_us = w as usize;
    let h_us = h as usize;
    let uvw_us = w_us / 2;
    let uvh_us = h_us / 2;
    let mut out = Vec::with_capacity(w_us * h_us + 2 * uvw_us * uvh_us);
    for r in 0..h_us {
        let off = r * yp.stride();
        out.extend_from_slice(&yp.data()[off..off + w_us]);
    }
    for r in 0..uvh_us {
        let off = r * up.stride();
        out.extend_from_slice(&up.data()[off..off + uvw_us]);
    }
    for r in 0..uvh_us {
        let off = r * vp.stride();
        out.extend_from_slice(&vp.data()[off..off + uvw_us]);
    }
    out
}

/// Decode every packet through rvp9-decoder, concatenating I420 bytes.
fn rvp9_decode_all(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut dec = Decoder::new();
    let mut out = Vec::new();
    for pkt in packets {
        let parsed = decode_packet(&mut dec, pkt);
        for pf in &parsed {
            let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
            out.extend_from_slice(&yv12_to_i420_bytes(&buf, W, H));
        }
    }
    out
}

fn vpxdec_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    if Command::new("vpxdec").arg("--help").output().is_ok() {
        return Some(PathBuf::from("vpxdec"));
    }
    None
}

fn run_vpxdec(vpxdec: &Path, ivf: &Path, out: &Path) -> std::io::Result<Vec<u8>> {
    let status = Command::new(vpxdec)
        .args(["--codec=vp9", "--i420", "-o"])
        .arg(out)
        .arg(ivf)
        .output()?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "vpxdec failed: stderr={}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    std::fs::read(out)
}

fn write_ivf_packets(path: &Path, framerate: f64, packets: &[Vec<u8>]) {
    let mut f = std::fs::File::create(path).expect("create ivf");
    let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
    let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    write_ivf(&mut f, &header, &refs).expect("write_ivf");
}

type SourceFrame = (Vec<u8>, Vec<u8>, Vec<u8>);

fn load_source() -> Option<(Vec<SourceFrame>, f64)> {
    let y4m_path = probe_testsrc_720p()?;
    match y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => Some((f, fr)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// PROOF 1 — temporal_layer_id pattern is live + the SVC bitstream differs
// from the single-layer encode (anti-G3: a stubbed SVC path fails both).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "A18 SVC temporal — run with `--ignored` (needs testsrc corpus)"]
fn svc_temporal_layer_id_pattern_is_live() {
    let Some((frames, framerate)) = load_source() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };

    // Single-layer baseline (temporal_layers = 1).
    let (base_packets, base_tids) = encode_clip(svc_config(framerate, 1), &frames);
    // Temporal SVC (2 layers, libvpx 0-1-0-1 pattern).
    let (svc_packets, svc_tids) = encode_clip(svc_config(framerate, 2), &frames);

    eprintln!("single-layer temporal_layer_id : {base_tids:?}");
    eprintln!("temporal-SVC temporal_layer_id : {svc_tids:?}");

    // --- Anti-G3 arm 1: single-layer tags every frame tl0. ---
    assert!(
        base_tids.iter().all(|&t| t == 0),
        "single-layer encode must tag every frame as temporal_layer_id 0, got {base_tids:?}"
    );

    // --- Anti-G3 arm 2: the SVC layering pattern. ---
    // Frame 0 is the keyframe (always base layer tl0). The subsequent
    // inter frames follow libvpx's 0-1-0-1 pattern
    // (`set_flags_and_fb_idx_for_temporal_mode2:629-634`,
    // `tl = current_video_frame_in_layer & 1`). The base-layer frame
    // counter starts at 0 on the first inter frame, so the inter
    // sequence is tl = 0, 1, 0, 1, ...  Prefixed by the KF (tl0) the
    // full per-frame pattern over N_FRAMES is [0, 0, 1, 0, 1, 0, 1, 0].
    let mut expected = Vec::with_capacity(N_FRAMES as usize);
    expected.push(0u8); // KF
    for inter_idx in 0..(N_FRAMES as usize - 1) {
        expected.push((inter_idx & 1) as u8);
    }
    assert_eq!(
        svc_tids, expected,
        "temporal-SVC per-frame temporal_layer_id must follow the libvpx \
         0-1-0-1 pattern (KF tl0, then 0,1,0,1...)"
    );
    // At least one enhancement-layer (tl1) frame must exist — otherwise
    // the "pattern" is degenerate and the test proves nothing.
    assert!(
        svc_tids.contains(&1),
        "temporal-SVC encode must produce at least one tl1 (enhancement) frame"
    );

    // --- Anti-G3 arm 3: the bitstream actually changed. ---
    // The enhancement-layer frames flip `refresh_frame_flags` from 0x01
    // (single-layer: refresh LAST) to 0x00 (SVC: non-reference). That is
    // a wire-format change, so the SVC packets MUST differ from the
    // single-layer packets on the enhancement frames. If the SVC wiring
    // were inert, the byte streams would be identical.
    let base_total: usize = base_packets.iter().map(Vec::len).sum();
    let svc_total: usize = svc_packets.iter().map(Vec::len).sum();
    eprintln!("single-layer total bytes: {base_total}, temporal-SVC total bytes: {svc_total}");
    let differs = base_packets
        .iter()
        .zip(svc_packets.iter())
        .any(|(a, b)| a != b)
        || base_packets.len() != svc_packets.len();
    assert!(
        differs,
        "temporal-SVC bitstream must differ from the single-layer bitstream \
         (the enhancement frames flip refresh_frame_flags) — identical streams \
         mean the SVC wiring is inert"
    );

    eprintln!("PROOF: temporal-SVC layer-id pattern is live and changes the bitstream.");
}

// ---------------------------------------------------------------------------
// PROOF 2 — the temporal-SVC stream is byte-exact decodable by vpxdec
// (the real oracle: vpxdec plays all layers as a normal VP9 sequence).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "A18 SVC temporal oracle — run with `--ignored` (needs vpxdec + corpus)"]
fn svc_temporal_stream_byte_exact_through_vpxdec() {
    let Some((frames, framerate)) = load_source() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };

    // Encode the temporal-SVC stream.
    let (packets, tids) = encode_clip(svc_config(framerate, 2), &frames);
    eprintln!("temporal-SVC temporal_layer_id pattern: {tids:?}");
    assert!(
        tids.contains(&1),
        "expected an enhancement layer in the SVC encode"
    );

    // rvp9-decoder baseline (the spec reference). All layers play.
    let rvp9_yuv = rvp9_decode_all(&packets);
    let expected_frame_bytes = (W as usize) * (H as usize) * 3 / 2;
    let expected_total = expected_frame_bytes * (N_FRAMES as usize);
    assert_eq!(
        rvp9_yuv.len(),
        expected_total,
        "rvp9-decoder must produce all {N_FRAMES} frames from the SVC stream \
         (got {} bytes, want {expected_total})",
        rvp9_yuv.len()
    );

    // Wrap in IVF.
    let tmp_dir = std::env::temp_dir();
    let ivf_path = tmp_dir.join("rvp9_svc_temporal_gate.ivf");
    write_ivf_packets(&ivf_path, framerate, &packets);

    // vpxdec leg — the oracle.
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[SKIPPED] vpxdec not found at canonical path or on PATH; oracle leg skipped.");
        let _ = std::fs::remove_file(&ivf_path);
        return;
    };
    eprintln!("vpxdec available at: {}", vpxdec.display());
    let vpx_yuv_path = tmp_dir.join("rvp9_svc_temporal_gate_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_yuv_path)
        .expect("vpxdec must decode the temporal-SVC stream");

    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "vpxdec must decode all layers (size {} vs rvp9 {}); a size mismatch \
         means vpxdec dropped frames",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert!(
        vpx_yuv == rvp9_yuv,
        "temporal-SVC stream must be BYTE-EXACT between rvp9-decoder and vpxdec \
         (both VP9 spec implementations decoding the same standard frames)"
    );

    eprintln!("PROOF: temporal-SVC stream is byte-exact decodable by vpxdec (all layers play).");
    let _ = std::fs::remove_file(&vpx_yuv_path);
    let _ = std::fs::remove_file(&ivf_path);
}
