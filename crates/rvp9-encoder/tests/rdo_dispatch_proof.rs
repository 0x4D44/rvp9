// Integration test: stderr is the diagnostic mechanism; the workspace
// `print_stderr = "deny"` lint targets library code, not test binaries.
#![allow(clippy::print_stderr)]

//! W4 RDO-dispatch PROOF (2026-05-26).
//!
//! Proves that at `speed <= 4` the per-leaf inter mode decision genuinely
//! routes through the transform-domain RDO picker
//! (`rvp9_encoder::rdo::pick_inter_sb` driven by the real `handle_inter`),
//! and that the resulting bitstream still decodes through the rvp9
//! decoder to valid frames.
//!
//! Three assertions form the anti-cheat gate:
//!
//! 1. **Routing.** The global `RDO_DISPATCH_INVOCATIONS` counter advances
//!    on a `speed = 3` encode and stays put on a `speed = 7` encode. A
//!    dispatch that called RDO and then discarded its result for the RT
//!    path would still bump the counter — so this is paired with #3.
//! 2. **Block8x8 routing (W4 extend-to-8x8).** The
//!    `RDO_DISPATCH_INVOCATIONS_8X8` sub-counter advances on the
//!    `speed = 3` encode, proving the W4-extend commit actually routes
//!    8×8 leaves (not just 16/32) through RDO. The variance partitioner
//!    descends to 8×8 on textured regions of the testsrc clip, so the
//!    counter must advance.
//! 3. **Effect + decodability.** The `speed = 3` (RDO) and `speed = 7`
//!    (RT) byte streams DIFFER on a real-motion clip, AND BOTH decode
//!    through `rvp9-decoder` to the expected number of frames. If the
//!    RDO output were silently discarded for the RT result the two byte
//!    streams would be identical — the assertion would fail.
//!
//! The corpus (`testsrc_720p30.y4m`) is centre-cropped to 256x256; if it
//! is unavailable the test SKIPs (informational) rather than failing, so
//! it is safe to run in environments without the corpus.

use rvp9_decoder::Decoder;
use rvp9_encoder::rdo::inter_dispatch::{
    RDO_DISPATCH_INVOCATIONS, RDO_DISPATCH_INVOCATIONS_8X8,
};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 6;
const TARGET_BITRATE_BPS: u32 = 1_000_000;

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: W as usize,
        u_stride: (W / 2) as usize,
        v_stride: (W / 2) as usize,
    }
}

fn config_at_speed(framerate: f64, speed: i32) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: TARGET_BITRATE_BPS,
        gop_length: N_FRAMES + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Encode the clip at `speed`, returning the per-frame packet bytes,
/// the total RDO-dispatch invocations consumed by this encode, and the
/// Block8x8-only invocations (the sub-counter the W4-extend-to-8x8 proof
/// asserts on).
fn encode_at_speed(
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    framerate: f64,
    speed: i32,
) -> (Vec<Vec<u8>>, u64, u64) {
    let before = RDO_DISPATCH_INVOCATIONS.load(Ordering::Relaxed);
    let before_8x8 = RDO_DISPATCH_INVOCATIONS_8X8.load(Ordering::Relaxed);
    let mut enc = StreamEncoder::new(config_at_speed(framerate, speed)).expect("StreamEncoder::new");
    let mut packets = Vec::with_capacity(frames.len());
    for (y, u, v) in frames {
        let pkt = enc.encode_frame(&yuv_view(y, u, v)).expect("encode_frame");
        packets.push(pkt.data);
    }
    let after = RDO_DISPATCH_INVOCATIONS.load(Ordering::Relaxed);
    let after_8x8 = RDO_DISPATCH_INVOCATIONS_8X8.load(Ordering::Relaxed);
    (packets, after - before, after_8x8 - before_8x8)
}

/// Decode every packet through rvp9-decoder; return the number of frames
/// successfully decoded (panics on any decode error).
fn decode_all(packets: &[Vec<u8>]) -> usize {
    let mut dec = Decoder::new();
    let mut frames_out = 0usize;
    for pkt in packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        // Collect owned headers so the borrow on `pkt` is released.
        let owned: Vec<rvp9_decoder::ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| rvp9_decoder::ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        for pf in &owned {
            let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
            // Sanity: the decoded buffer has the configured luma size.
            assert!(buf.y_plane().data().len() >= (W as usize) * (H as usize));
            frames_out += 1;
        }
    }
    frames_out
}

/// Probe libvpx-stock's `vpxdec.exe` at the canonical sibling path, then
/// fall back to `vpxdec` on PATH. Returns `None` when neither exists.
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

/// Decode the rvp9-decoder I420 bytes for a stream (Y,U,V concatenated
/// per frame), so the RDO leg can be byte-compared against vpxdec.
fn rvp9_decode_i420(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut dec = Decoder::new();
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mut out = Vec::new();
    for pkt in packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        let owned: Vec<rvp9_decoder::ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| rvp9_decoder::ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        for pf in &owned {
            let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
            let yp = buf.y_plane();
            let up = buf.u_plane();
            let vp = buf.v_plane();
            for r in 0..h {
                let off = r * yp.stride();
                out.extend_from_slice(&yp.data()[off..off + w]);
            }
            for r in 0..uvh {
                let off = r * up.stride();
                out.extend_from_slice(&up.data()[off..off + uvw]);
            }
            for r in 0..uvh {
                let off = r * vp.stride();
                out.extend_from_slice(&vp.data()[off..off + uvw]);
            }
        }
    }
    out
}

/// Run `vpxdec --i420` on an IVF and return the raw I420 bytes.
fn run_vpxdec(vpxdec: &Path, ivf: &Path, out: &Path) -> std::io::Result<Vec<u8>> {
    let status = Command::new(vpxdec)
        .args(["--codec=vp9", "--i420", "-o"])
        .arg(out)
        .arg(ivf)
        .output()?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "vpxdec failed: {}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    std::fs::read(out)
}

/// Cross-decoder check on the RDO (speed=3) stream: when vpxdec is
/// available, assert it agrees byte-exactly with rvp9-decoder. When
/// vpxdec is missing the check is informational (SKIPPED).
fn verify_rdo_through_vpxdec(packets: &[Vec<u8>], framerate: f64) {
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[vpxdec leg SKIPPED] vpxdec not found; rvp9-decoder gate already green.");
        return;
    };
    let tmp = std::env::temp_dir();
    let ivf_path = tmp.join("rvp9_rdo_proof_speed3.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    let vpx_out = tmp.join("rvp9_rdo_proof_speed3_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out)
        .expect("vpxdec must accept the RDO (speed=3) bitstream");
    let rvp9_yuv = rvp9_decode_i420(packets);
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "RDO stream: vpxdec/rvp9 size mismatch ({} vs {})",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "RDO (speed=3) stream: vpxdec and rvp9-decoder disagree — \
         the speed<=4 RDO output is not a valid VP9 bitstream"
    );
    eprintln!("vpxdec leg: BYTE-EXACT on the RDO (speed=3) stream.");
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

#[test]
fn rdo_dispatch_routes_at_speed3_and_decodes() {
    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };
    let (frames, framerate) = match y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "[SKIPPED] testsrc only yielded {}/{} frames.",
                f.len(),
                N_FRAMES
            );
            return;
        }
        Err(e) => {
            eprintln!("[SKIPPED] testsrc read failed: {e}");
            return;
        }
    };

    // --- speed = 3: RDO picker active for 8x8 / 16x16 / 32x32 leaves ---
    let (rdo_packets, rdo_invocations, rdo_invocations_8x8) =
        encode_at_speed(&frames, framerate, 3);
    // --- speed = 7: realtime picker, RDO never invoked ---
    let (rt_packets, rt_invocations, rt_invocations_8x8) =
        encode_at_speed(&frames, framerate, 7);

    eprintln!("speed=3 RDO invocations: {rdo_invocations} (8x8: {rdo_invocations_8x8})");
    eprintln!("speed=7 RDO invocations: {rt_invocations} (8x8: {rt_invocations_8x8})");

    // Assertion #1 (routing): RDO ran at speed 3, not at speed 7.
    assert!(
        rdo_invocations > 0,
        "RDO dispatch was never invoked at speed=3 — the speed<=4 path did \
         not route through the RDO picker"
    );
    assert_eq!(
        rt_invocations, 0,
        "RDO dispatch was invoked at speed=7 — the gate leaked into the RT path"
    );
    // Assertion #1b (8x8 routing): the W4-extend-to-8x8 commit must route
    // Block8x8 leaves through RDO too. Real motion at 256x256 with the
    // variance partitioner produces 8x8 leaves on textured regions, so
    // the counter must advance. The speed=7 (RT) path must NOT touch the
    // 8x8 sub-counter (the same gate the total counter checks).
    assert!(
        rdo_invocations_8x8 > 0,
        "RDO dispatch was never invoked for a Block8x8 leaf at speed=3 — \
         the W4-extend-to-8x8 path is not routing 8x8 leaves through RDO"
    );
    assert_eq!(
        rt_invocations_8x8, 0,
        "RDO dispatch was invoked for a Block8x8 leaf at speed=7 — the \
         RDO gate leaked into the RT path for 8x8"
    );

    // Assertion #2 (effect): the two byte streams differ. Identical bytes
    // would mean the RDO result was discarded for the RT path (the
    // forbidden cheat), since the two pickers make genuinely different
    // mode decisions on real motion.
    let rdo_total: usize = rdo_packets.iter().map(Vec::len).sum();
    let rt_total: usize = rt_packets.iter().map(Vec::len).sum();
    eprintln!("speed=3 total bytes: {rdo_total}, speed=7 total bytes: {rt_total}");
    assert_ne!(
        rdo_packets, rt_packets,
        "speed=3 (RDO) and speed=7 (RT) produced byte-identical streams — \
         the RDO decision is not actually driving the bitstream"
    );

    // Assertion #2 (decodability / cross-decoder gate): BOTH streams
    // decode through rvp9-decoder to the full frame count.
    let rdo_decoded = decode_all(&rdo_packets);
    let rt_decoded = decode_all(&rt_packets);
    assert_eq!(
        rdo_decoded, N_FRAMES as usize,
        "speed=3 (RDO) stream did not decode to {N_FRAMES} frames (got {rdo_decoded})"
    );
    assert_eq!(
        rt_decoded, N_FRAMES as usize,
        "speed=7 (RT) stream did not decode to {N_FRAMES} frames (got {rt_decoded})"
    );

    // Cross-decoder gate on the RDO output: vpxdec must agree byte-exact
    // with rvp9-decoder (informational SKIP when vpxdec is unavailable).
    verify_rdo_through_vpxdec(&rdo_packets, framerate);

    eprintln!(
        "PROOF PASS: speed=3 routed {rdo_invocations} leaves through RDO \
         ({rdo_invocations_8x8} of them Block8x8); both streams decode \
         ({rdo_decoded}/{rt_decoded} frames); streams differ."
    );
}
