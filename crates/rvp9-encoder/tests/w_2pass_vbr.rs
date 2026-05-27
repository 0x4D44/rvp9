// Integration test: stderr is the report mechanism for the oracle leg.
#![allow(clippy::print_stderr)]

//! W_2PASS — Two-pass VBR PROOF.
//!
//! Two-pass VBR is the last rate-control mode to land (CBR / 1-pass VBR
//! / CQ / Q were already real). This test proves the bounded two-pass
//! implementation:
//!
//! 1. **Anti-G3 (stats drive pass 2).** A two-pass encode produces a
//!    per-frame Q sequence that DIFFERS from a one-pass encode of the
//!    SAME input. The first pass gathers per-frame `coded_error`; the
//!    second pass allocates bits proportional to that error and
//!    regulates Q to hit each frame's target. If pass 2 ignored the
//!    first-pass stats (e.g. silently ran the 1-pass path), the two Q
//!    sequences would be identical and this test would FAIL.
//!
//!    The input is engineered so per-frame complexity varies (alternating
//!    "busy" high-texture and "flat" low-texture frames), which makes the
//!    stats-driven allocation visibly non-uniform — busy frames earn more
//!    bits (lower Q), flat frames fewer (higher Q).
//!
//! 2. **Oracle (byte-exact through vpxdec).** The two-pass stream decodes
//!    byte-identically through our own decoder and through libvpx-stock
//!    `vpxdec.exe` (when present). The two-pass path emits standard VP9 —
//!    only the per-frame Q (and hence the residual coding) changes, the
//!    syntax is unchanged — so any spec-compliant decoder must agree.
//!
//! libvpx reference: per-SEQUENCE two-pass model in
//! `third_party/libvpx/vp9/encoder/vp9_firstpass.c` (`vp9_first_pass`,
//! line 1392; `gf_group->bit_allocation`, lines 2334-2440;
//! `get_twopass_worst_quality`, lines 1557-1626) +
//! `vp9/encoder/vp9_ratectrl.c` (`rc_pick_q_and_bounds_two_pass`,
//! line 1427; `pass == 2` dispatch, lines 1606-1608).
//! `FIRSTPASS_STATS` struct: `vp9_firstpass_stats.h:20-48`.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{
    AqMode, Deadline, EncodedFrame, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef,
    encode_two_pass,
};
use std::path::{Path, PathBuf};
use std::process::Command;

const W: u32 = 128;
const H: u32 = 128;
const N_FRAMES: usize = 9;
const FRAMERATE: f64 = 30.0;
const TARGET_BPS: u32 = 400_000;
const GOP: u32 = 9; // one KF at frame 0, then inter frames.

/// Tight-stride YUV backing for one synthetic frame.
struct Frame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Frame {
    fn view(&self) -> YuvFrameRef<'_> {
        YuvFrameRef {
            y: &self.y,
            u: &self.u,
            v: &self.v,
            y_stride: W as usize,
            u_stride: (W / 2) as usize,
            v_stride: (W / 2) as usize,
        }
    }
}

/// Build a synthetic sequence with deliberately varying per-frame
/// complexity so the first-pass `coded_error` distribution is non-flat.
///
/// Even frames are "busy" (high-frequency checkerboard-ish texture that
/// is expensive to code → high coded_error); odd frames are "flat"
/// (near-uniform → low coded_error). A small per-frame phase shift keeps
/// inter frames from being trivially predictable. This guarantees the
/// stats-driven allocation differs across frames, which is what makes the
/// 2-pass Q sequence diverge from the 1-pass one.
fn synthetic_frames() -> Vec<Frame> {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    (0..N_FRAMES)
        .map(|f| {
            let busy = f % 2 == 0;
            let phase = (f * 3) as i32;
            let mut y = vec![0u8; w * h];
            for row in 0..h {
                for col in 0..w {
                    let val = if busy {
                        // High-frequency texture: depends on both x and y
                        // at a fine pitch → lots of residual energy.
                        let t = ((row as i32 + phase) ^ (col as i32 * 2 + phase)) & 0xFF;
                        let stripe = (((row as i32 + col as i32 + phase) / 2) & 0x3F) as i32;
                        ((t + stripe) & 0xFF) as u8
                    } else {
                        // Flat-ish: a gentle gradient, cheap to code.
                        (96 + ((row + col + f) as i32 % 16)) as u8
                    };
                    y[row * w + col] = val;
                }
            }
            // Chroma: mild, constant-ish (keeps the luma the dominant
            // complexity signal, mirroring real content where luma drives
            // the rate).
            let u = vec![128u8; uvw * uvh];
            let v = vec![128u8; uvw * uvh];
            Frame { y, u, v }
        })
        .collect()
}

/// Base encoder config shared by both 1-pass and 2-pass runs (only
/// `rc_mode` differs — the 2-pass driver overrides it internally).
fn base_config(rc_mode: RcMode) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate: FRAMERATE,
        target_bitrate_bps: TARGET_BPS,
        gop_length: GOP,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        min_qindex: 4,
        max_qindex: 255,
    }
}

/// Run a one-pass encode and return the per-frame Q sequence + packets.
fn encode_one_pass(rc_mode: RcMode, frames: &[Frame]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut enc = StreamEncoder::new(base_config(rc_mode)).expect("StreamEncoder::new");
    let mut qs = Vec::with_capacity(frames.len());
    let mut packets = Vec::with_capacity(frames.len());
    for f in frames {
        let pkt = enc.encode_frame(&f.view()).expect("encode_frame");
        qs.push(pkt.q_index);
        packets.push(pkt.data);
    }
    (qs, packets)
}

/// Run the two-pass encode and return the per-frame Q sequence + packets.
fn encode_two_pass_seq(frames: &[Frame]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let views: Vec<YuvFrameRef<'_>> = frames.iter().map(Frame::view).collect();
    let encoded: Vec<EncodedFrame> =
        encode_two_pass(base_config(RcMode::Cbr), &views).expect("encode_two_pass");
    let qs = encoded.iter().map(|e| e.q_index).collect();
    let packets = encoded.into_iter().map(|e| e.data).collect();
    (qs, packets)
}

// ---------------------------------------------------------------------------
// PROOF #1 — Anti-G3: two-pass Q sequence differs from one-pass.
// ---------------------------------------------------------------------------

/// The two-pass per-frame Q sequence must DIFFER from the one-pass CBR Q
/// sequence on the same input. This is the load-bearing anti-G3 proof:
/// if pass 2 ignored the first-pass stats (or routed through the 1-pass
/// path), the sequences would match.
#[test]
fn two_pass_q_sequence_differs_from_one_pass_cbr() {
    let frames = synthetic_frames();
    let (q_cbr, _) = encode_one_pass(RcMode::Cbr, &frames);
    let (q_2pass, _) = encode_two_pass_seq(&frames);

    assert_eq!(q_cbr.len(), N_FRAMES);
    assert_eq!(q_2pass.len(), N_FRAMES);

    assert_ne!(
        q_cbr, q_2pass,
        "Anti-G3: two-pass Q sequence ({q_2pass:?}) must differ from \
         one-pass CBR ({q_cbr:?}) — if they match, pass 2 is ignoring \
         the first-pass stats (or silently running the 1-pass path)."
    );
}

/// Stronger anti-G3 — the two-pass per-frame Q is itself non-uniform
/// across frames of differing complexity: the busy (even) inter frames
/// earn more bits (lower Q) than the flat (odd) inter frames. This proves
/// the first-pass `coded_error` distribution actually drives the
/// allocation, not just "some difference from CBR".
#[test]
fn two_pass_allocates_lower_q_to_busier_frames() {
    let frames = synthetic_frames();
    let (q_2pass, _) = encode_two_pass_seq(&frames);

    // Compare the mean inter Q of busy (even-index, excluding the KF at
    // 0) vs flat (odd-index) frames. Busy frames carry more residual
    // energy → bigger first-pass coded_error → bigger bit allocation →
    // lower Q.
    let mut busy_q = Vec::new();
    let mut flat_q = Vec::new();
    for (i, &q) in q_2pass.iter().enumerate() {
        if i == 0 {
            continue; // keyframe — not part of the inter complexity test
        }
        if i % 2 == 0 {
            busy_q.push(u32::from(q));
        } else {
            flat_q.push(u32::from(q));
        }
    }
    assert!(!busy_q.is_empty() && !flat_q.is_empty(), "need both classes");
    let mean = |v: &[u32]| v.iter().sum::<u32>() as f64 / v.len() as f64;
    let busy_mean = mean(&busy_q);
    let flat_mean = mean(&flat_q);
    assert!(
        busy_mean < flat_mean,
        "stats-driven allocation must give busy frames lower Q than flat \
         frames: busy_mean_q={busy_mean:.1} (q={busy_q:?}), \
         flat_mean_q={flat_mean:.1} (q={flat_q:?}). If these are equal the \
         first-pass coded_error is not driving pass-2 Q."
    );
}

// ---------------------------------------------------------------------------
// PROOF #2 — Oracle: two-pass stream is byte-exact through vpxdec.
// ---------------------------------------------------------------------------

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
    let uvw = w_us / 2;
    let uvh = h_us / 2;
    let mut out = Vec::with_capacity(w_us * h_us + 2 * uvw * uvh);
    for r in 0..h_us {
        let off = r * yp.stride();
        out.extend_from_slice(&yp.data()[off..off + w_us]);
    }
    for r in 0..uvh {
        let off = r * up.stride();
        out.extend_from_slice(&up.data()[off..off + uvw]);
    }
    for r in 0..uvh {
        let off = r * vp.stride();
        out.extend_from_slice(&vp.data()[off..off + uvw]);
    }
    out
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

/// The two-pass stream must decode byte-identically through rvp9-decoder
/// and (when present) libvpx-stock vpxdec. The stream is standard VP9 —
/// two-pass only changes the per-frame Q, not the syntax — so any
/// conformant decoder must agree. `#[ignore]`-gated like the
/// cross-decoder gate (needs the external binary + writes temp files).
#[test]
#[ignore = "W_2PASS oracle — run with `--ignored` (needs vpxdec)"]
fn two_pass_stream_byte_exact_through_vpxdec() {
    use rvp9_harness::container::{IvfHeader, write_ivf};

    let frames = synthetic_frames();
    let (_q, packets) = encode_two_pass_seq(&frames);
    eprintln!("W_2PASS oracle: encoded {} two-pass packets", packets.len());

    // ---- rvp9-decoder baseline.
    let rvp9_yuv: Vec<u8> = {
        let mut dec = Decoder::new();
        let mut out = Vec::new();
        for pkt in &packets {
            for pf in &decode_packet(&mut dec, pkt) {
                let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
                out.extend_from_slice(&yv12_to_i420_bytes(&buf, W, H));
            }
        }
        out
    };
    let frame_bytes = (W as usize) * (H as usize) * 3 / 2;
    assert_eq!(
        rvp9_yuv.len(),
        frame_bytes * N_FRAMES,
        "rvp9-decoder output size mismatch"
    );

    // ---- IVF mux for the external decoder.
    let tmp_dir = std::env::temp_dir();
    let ivf_path = tmp_dir.join("rvp9_w_2pass_vbr.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, FRAMERATE.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }

    // ---- vpxdec leg.
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[skipped] vpxdec not found; rvp9-decoder leg passed.");
        let _ = std::fs::remove_file(&ivf_path);
        return;
    };
    let vpx_out = tmp_dir.join("rvp9_w_2pass_vbr_vpxdec.yuv");
    let vpx_yuv = run_vpxdec(&vpxdec, &ivf_path, &vpx_out).expect("vpxdec must decode the stream");
    assert_eq!(
        vpx_yuv.len(),
        rvp9_yuv.len(),
        "vpxdec output size mismatch (dropped frames?): vpxdec={}, rvp9={}",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
    assert_eq!(
        vpx_yuv, rvp9_yuv,
        "W_2PASS oracle: two-pass stream must be BYTE-EXACT through vpxdec \
         (it is standard VP9 syntax — only per-frame Q changes)."
    );
    eprintln!("W_2PASS oracle: BYTE-EXACT through vpxdec ({} bytes).", vpx_yuv.len());
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&vpx_out);
}

/// Sanity — the two-pass encode completes (no panic) and emits exactly
/// one packet per input frame. Guards the `RcMode::VbrTwoPass` dispatch
/// arm: before this wave it was the one unimplemented rate-control mode.
#[test]
fn two_pass_completes_and_emits_one_packet_per_frame() {
    let frames = synthetic_frames();
    let (qs, packets) = encode_two_pass_seq(&frames);
    assert_eq!(qs.len(), N_FRAMES, "one Q per frame");
    assert_eq!(packets.len(), N_FRAMES, "one packet per frame");
    for (i, p) in packets.iter().enumerate() {
        assert!(!p.is_empty(), "frame {i} packet must be non-empty");
    }
}
