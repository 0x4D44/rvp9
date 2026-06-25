// Integration test: stderr IS the output mechanism (the harness's job
// is to print a per-decoder verdict). The workspace's `print_stderr =
// "deny"` lint targets library code; for this binary that lint is the
// wrong rule.
#![allow(clippy::print_stderr)]

//! RC.4 — Cross-decoder gate (PLAN.md Stage 4a exit criterion #3).
//!
//! Encodes 8 frames from `tests/corpus/testsrc_720p30.y4m` (centre-cropped
//! to 256x256) through `rvp9-encoder::StreamEncoder`, wraps the produced
//! superframe packets in an IVF (DKIF) container, then decodes the same
//! IVF through:
//!
//! 1. `rvp9-decoder::Decoder` — our own decoder.
//! 2. libvpx-stock `vpxdec.exe` (when present at the canonical path).
//! 3. `ffmpeg` (when on PATH).
//!
//! Asserts the YUV output:
//! - rvp9-decoder vs vpxdec MUST be byte-exact (both implement the VP9
//!   spec; any divergence is a real bug on one side).
//! - rvp9-decoder vs ffmpeg first tries byte-exact; on failure falls
//!   back to Y-plane PSNR >= 40 dB and documents the divergence to
//!   stderr (ffmpeg's IDCT differs slightly from libvpx's on some
//!   builds — informational, not a gate failure on its own).
//!
//! Per-decoder skip: missing vpxdec or missing ffmpeg drops that leg
//! with a clear "[skipped]" message instead of failing the test. The
//! test is `#[ignore]`-gated; invoke explicitly via:
//!
//! ```text
//! cargo test --test cross_decoder_gate -- --ignored --nocapture
//! ```
//!
//! **F.1-F.3 status (2026-05-23).** This gate compares rvp9-decoder
//! vs. vpxdec on an *rvp9-encoded* bitstream — both decoders are
//! consuming the same syntax, so picker-side changes (F.1's
//! `block_yrd` rate-aware RD, F.2's `find_mv_refs` proper, F.3's
//! motion-aware partition) do NOT break this contract: the bitstream
//! is still valid VP9, and both decoders should agree on its YUV
//! output. The F.1 HLD's §7.5 forward-looking note ("byte-exact
//! foreman cross-decode #[ignore] during F.1-F.3") refers to a
//! *future* `rvp9-encoder vs libvpx-encoder produces bit-identical
//! bitstreams on a given input` gate — not currently in this file.
//! That gate re-establishes as the F.4 exit criterion per the HLD.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;

mod common;

// ---------------------------------------------------------------------------
// Decoder availability probes.
// ---------------------------------------------------------------------------

/// Probe for libvpx-stock's `vpxdec.exe` at the canonical sibling-tree
/// location. Returns `Some(path)` if available, `None` otherwise.
fn vpxdec_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    // Fall back to `vpxdec` on PATH.
    if Command::new("vpxdec").arg("--help").output().is_ok() {
        return Some(PathBuf::from("vpxdec"));
    }
    None
}

#[test]
#[ignore = "Stage 4a product cross-decoder gate; hard-fails on missing corpus/tools/env pins"]
fn cross_decoder_gate_stage4a_vbr_product() {
    let (baseline, candidate) =
        common::stage4a_product::run_stage4a_product_candidate_with_fresh_baseline();
    common::stage4a_product::assert_product_report_pass(&baseline);
    common::stage4a_product::assert_product_cross_decoder_pass(&candidate);
}

/// Probe for `ffmpeg` on PATH. Returns `Some("ffmpeg")` if `ffmpeg
/// -version` succeeds, else `None`.
fn ffmpeg_path() -> Option<PathBuf> {
    let out = Command::new("ffmpeg").arg("-version").output().ok()?;
    if out.status.success() {
        Some(PathBuf::from("ffmpeg"))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Build a strided YUV view over tight-stride backing buffers.
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

/// Decode one rvp9-encoder packet through the real decoder. Returns
/// owned `ParsedFrame`s so the borrow on `pkt` ends with the call.
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

/// Concatenate the tight-stride Y / U / V samples of a decoded YV12
/// buffer in I420 order (Y, then U, then V), matching what
/// `vpxdec --i420 -o out.yuv` and `ffmpeg -pix_fmt yuv420p` write.
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

/// Compute Y-plane PSNR (dB). Returns `100.0` on byte-exact match.
fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut sse: u64 = 0;
    let mut max_abs: i32 = 0;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = i32::from(*a) - i32::from(*b);
        if d.abs() > max_abs {
            max_abs = d.abs();
        }
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    let _ = max_abs;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Compute per-plane (max_abs_diff, sum_squared_error) for diagnostics
/// when a byte-exact comparison fails.
fn plane_diff_stats(a: &[u8], b: &[u8]) -> (i32, u64) {
    assert_eq!(a.len(), b.len());
    let mut max_abs: i32 = 0;
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i32::from(*x) - i32::from(*y);
        if d.abs() > max_abs {
            max_abs = d.abs();
        }
        sse += (d * d) as u64;
    }
    (max_abs, sse)
}

// ---------------------------------------------------------------------------
// External decoder invocation.
// ---------------------------------------------------------------------------

/// Invoke `vpxdec --i420 -o <out> <ivf>` and return the resulting raw
/// I420 bytes (Y/U/V concatenated, one frame after another).
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

/// Invoke `ffmpeg -y -i <ivf> -f rawvideo -pix_fmt yuv420p <out>` and
/// return the resulting raw bytes.
fn run_ffmpeg(ffmpeg: &Path, ivf: &Path, out: &Path) -> std::io::Result<Vec<u8>> {
    let status = Command::new(ffmpeg)
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(ivf)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(out)
        .output()?;
    if !status.status.success() {
        return Err(std::io::Error::other(format!(
            "ffmpeg failed: stderr={}",
            String::from_utf8_lossy(&status.stderr)
        )));
    }
    std::fs::read(out)
}

// ---------------------------------------------------------------------------
// Test.
// ---------------------------------------------------------------------------

/// RC.4 — Stage 4a cross-decoder exit gate.
///
/// See the module-level docs for the report semantics. Per-decoder skip
/// on missing binary; the test itself is informational on missing
/// decoders and PASSes only when every available decoder agrees on its
/// configured comparison gate.
#[test]
#[ignore = "RC.4 — cross-decoder gate; run with `--ignored`"]
fn cross_decoder_gate_vpxdec_ffmpeg() {
    // Frame config. 256x256 = 4x4 SB grid — same shape as bdrate_gate's
    // F12 / AQ.5 — small enough to encode in a few seconds, large enough
    // that the partition / mode / coef machinery actually fires.
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;
    const FRAMERATE: f64 = 30.0;
    const TARGET_BITRATE_BPS: u32 = 1_000_000;

    eprintln!("RC.4 Cross-Decoder Gate — Stage 4a Exit Criterion #3");
    eprintln!("====================================================");
    eprintln!("Frames: {N_FRAMES} @ {W}x{H} {FRAMERATE} fps, target {TARGET_BITRATE_BPS} bps");
    eprintln!();

    // ---- Source --------------------------------------------------------
    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available; cannot run gate.");
        return;
    };
    let (frames, framerate) = match y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "[SKIPPED] testsrc only yielded {}/{} frames; cannot run gate.",
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
    eprintln!("Source: {} (framerate {framerate} fps)", y4m_path.display());

    // ---- Encode ---------------------------------------------------------
    let cfg = EncoderConfig {
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
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    let mut encoded_packets: Vec<Vec<u8>> = Vec::with_capacity(N_FRAMES as usize);
    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        eprintln!(
            "  encoded frame {idx}: kind={:?} q={} bytes={}",
            pkt.frame_kind,
            pkt.q_index,
            pkt.data.len()
        );
        encoded_packets.push(pkt.data);
    }
    let total_bytes: usize = encoded_packets.iter().map(|p| p.len()).sum();
    eprintln!("Total encoded bytes: {total_bytes}");
    eprintln!();

    // ---- Wrap in IVF ----------------------------------------------------
    let tmp_dir = std::env::temp_dir();
    let ivf_path = tmp_dir.join("rvp9_cross_decoder_gate.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let packet_refs: Vec<&[u8]> = encoded_packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &packet_refs).expect("write_ivf");
    }
    eprintln!("IVF written to {}", ivf_path.display());

    // ---- Decode through rvp9-decoder (the baseline) --------------------
    let rvp9_yuv: Vec<u8> = {
        let mut dec = Decoder::new();
        let mut out: Vec<u8> =
            Vec::with_capacity((W as usize) * (H as usize) * 3 / 2 * N_FRAMES as usize);
        for pkt in &encoded_packets {
            let parsed = decode_packet(&mut dec, pkt);
            for pf in &parsed {
                let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
                out.extend_from_slice(&yv12_to_i420_bytes(&buf, W, H));
            }
        }
        out
    };
    let expected_frame_bytes = (W as usize) * (H as usize) * 3 / 2;
    let expected_total = expected_frame_bytes * (N_FRAMES as usize);
    assert_eq!(
        rvp9_yuv.len(),
        expected_total,
        "rvp9-decoder output size mismatch: got {}, want {}",
        rvp9_yuv.len(),
        expected_total
    );
    eprintln!(
        "rvp9-decoder: produced {} bytes ({} frames * {} bytes)",
        rvp9_yuv.len(),
        N_FRAMES,
        expected_frame_bytes
    );
    eprintln!();

    // ---- vpxdec leg ----------------------------------------------------
    let vpxdec = vpxdec_path();
    let mut vpxdec_failed = false;
    let vpxdec_verdict: String;
    if let Some(ref vpxdec) = vpxdec {
        eprintln!("vpxdec available at: {}", vpxdec.display());
        let vpx_yuv_path = tmp_dir.join("rvp9_cross_decoder_gate_vpxdec.yuv");
        match run_vpxdec(vpxdec, &ivf_path, &vpx_yuv_path) {
            Ok(vpx_yuv) => {
                if vpx_yuv.len() != rvp9_yuv.len() {
                    eprintln!(
                        "rvp9-decoder vs vpxdec: SIZE MISMATCH (vpxdec={} bytes, \
                         rvp9={} bytes; vpxdec dropped {} frames)",
                        vpx_yuv.len(),
                        rvp9_yuv.len(),
                        (rvp9_yuv.len().saturating_sub(vpx_yuv.len())) / expected_frame_bytes
                    );
                    vpxdec_verdict = format!(
                        "FAIL (size mismatch: {} vs {})",
                        vpx_yuv.len(),
                        rvp9_yuv.len()
                    );
                    vpxdec_failed = true;
                } else if vpx_yuv == rvp9_yuv {
                    eprintln!(
                        "rvp9-decoder vs vpxdec: BYTE-EXACT (both VP9 spec implementations agree)."
                    );
                    vpxdec_verdict = "BYTE-EXACT".to_string();
                } else {
                    // Diagnostics: per-frame Y-plane delta.
                    let (max_abs, sse) = plane_diff_stats(&rvp9_yuv, &vpx_yuv);
                    eprintln!(
                        "rvp9-decoder vs vpxdec: DIVERGENT (max_abs={max_abs}, sse={sse}, \
                         total_bytes={})",
                        rvp9_yuv.len()
                    );
                    vpxdec_verdict = format!("FAIL (divergent: max_abs={max_abs}, sse={sse})");
                    vpxdec_failed = true;
                }
                let _ = std::fs::remove_file(&vpx_yuv_path);
            }
            Err(e) => {
                eprintln!("rvp9-decoder vs vpxdec: vpxdec REJECTED the bitstream: {e}");
                vpxdec_verdict = format!("FAIL (vpxdec rejected: {e})");
                vpxdec_failed = true;
            }
        }
    } else {
        eprintln!("[skipped] vpxdec not found at canonical path or on PATH.");
        vpxdec_verdict = "SKIPPED-MISSING".to_string();
    }
    eprintln!();

    // ---- ffmpeg leg ----------------------------------------------------
    let ffmpeg = ffmpeg_path();
    let mut ffmpeg_failed = false;
    let ffmpeg_verdict: String;
    if let Some(ref ffmpeg) = ffmpeg {
        eprintln!("ffmpeg available on PATH.");
        let ff_yuv_path = tmp_dir.join("rvp9_cross_decoder_gate_ffmpeg.yuv");
        match run_ffmpeg(ffmpeg, &ivf_path, &ff_yuv_path) {
            Ok(ff_yuv) => {
                if ff_yuv.len() != rvp9_yuv.len() {
                    eprintln!(
                        "rvp9-decoder vs ffmpeg: SIZE MISMATCH (ffmpeg={} bytes, \
                         rvp9={} bytes; ffmpeg dropped {} frames)",
                        ff_yuv.len(),
                        rvp9_yuv.len(),
                        (rvp9_yuv.len().saturating_sub(ff_yuv.len())) / expected_frame_bytes
                    );
                    ffmpeg_verdict = format!(
                        "FAIL (size mismatch: {} vs {})",
                        ff_yuv.len(),
                        rvp9_yuv.len()
                    );
                    ffmpeg_failed = true;
                } else if ff_yuv == rvp9_yuv {
                    eprintln!("rvp9-decoder vs ffmpeg: BYTE-EXACT.");
                    ffmpeg_verdict = "BYTE-EXACT".to_string();
                } else {
                    // Fall back to PSNR — ffmpeg's libvpx-wrapper sometimes
                    // has subtly different IDCT rounding on certain
                    // platforms. Check Y-plane PSNR >= 40 dB on every
                    // frame.
                    let (max_abs, sse) = plane_diff_stats(&rvp9_yuv, &ff_yuv);
                    eprintln!(
                        "rvp9-decoder vs ffmpeg: byte-exact failed; diagnostics: \
                         max_abs={max_abs}, sse={sse}, total_bytes={}",
                        rvp9_yuv.len()
                    );
                    // Compute per-frame Y-plane PSNR.
                    let y_size = (W as usize) * (H as usize);
                    let frame_bytes = expected_frame_bytes;
                    let mut min_psnr = f64::INFINITY;
                    for i in 0..N_FRAMES as usize {
                        let off = i * frame_bytes;
                        let rvp9_y = &rvp9_yuv[off..off + y_size];
                        let ff_y = &ff_yuv[off..off + y_size];
                        let p = psnr_db(rvp9_y, ff_y);
                        eprintln!("  frame {i}: Y-plane PSNR = {p:.2} dB");
                        if p < min_psnr {
                            min_psnr = p;
                        }
                    }
                    eprintln!("  min Y-plane PSNR across {N_FRAMES} frames: {min_psnr:.2} dB");
                    if min_psnr >= 40.0 {
                        eprintln!(
                            "rvp9-decoder vs ffmpeg: PSNR-GATE PASS (min {min_psnr:.2} dB >= 40 dB)"
                        );
                        ffmpeg_verdict = format!("PSNR-GATE (min {min_psnr:.2} dB)");
                    } else {
                        eprintln!(
                            "rvp9-decoder vs ffmpeg: PSNR-GATE FAIL (min {min_psnr:.2} dB < 40 dB)"
                        );
                        ffmpeg_verdict = format!("FAIL (PSNR min {min_psnr:.2} dB < 40)");
                        ffmpeg_failed = true;
                    }
                }
                let _ = std::fs::remove_file(&ff_yuv_path);
            }
            Err(e) => {
                eprintln!("rvp9-decoder vs ffmpeg: ffmpeg REJECTED the bitstream: {e}");
                ffmpeg_verdict = format!("FAIL (ffmpeg rejected: {e})");
                ffmpeg_failed = true;
            }
        }
    } else {
        eprintln!("[skipped] ffmpeg not on PATH.");
        ffmpeg_verdict = "SKIPPED-MISSING".to_string();
    }
    eprintln!();

    // ---- Final report --------------------------------------------------
    eprintln!("Cross-decoder verification on {N_FRAMES}-frame {W}x{H} testsrc encode:");
    eprintln!("  rvp9-decoder vs vpxdec  : {vpxdec_verdict}");
    eprintln!("  rvp9-decoder vs ffmpeg  : {ffmpeg_verdict}");
    eprintln!();

    let vpxdec_skipped = vpxdec_verdict.starts_with("SKIPPED");
    let ffmpeg_skipped = ffmpeg_verdict.starts_with("SKIPPED");
    let any_external_ran = !vpxdec_skipped || !ffmpeg_skipped;
    let any_external_failed = vpxdec_failed || ffmpeg_failed;

    if !any_external_ran {
        eprintln!(
            "Stage 4a cross-decoder gate: PARTIAL (both external decoders unavailable; \
             only rvp9-decoder ran). Test is informational."
        );
    } else if any_external_failed {
        eprintln!(
            "Stage 4a cross-decoder gate: FAIL — at least one external decoder rejected \
             or diverged from the rvp9-encoder bitstream. See per-decoder diagnostics above."
        );
    } else if vpxdec_skipped || ffmpeg_skipped {
        eprintln!(
            "Stage 4a cross-decoder gate: PARTIAL (one external decoder unavailable; \
             available decoders agreed with rvp9-decoder)."
        );
    } else {
        eprintln!("Stage 4a cross-decoder gate: PASS (all available decoders agree)");
    }

    // Best-effort cleanup of the IVF.
    let _ = std::fs::remove_file(&ivf_path);

    // Fail the test if any available external decoder reported a mismatch
    // or rejected the bitstream — that's the gate's whole purpose. A
    // PARTIAL run (no external decoders available) passes informationally.
    assert!(
        !any_external_failed,
        "cross-decoder gate FAIL: vpxdec={vpxdec_verdict}, ffmpeg={ffmpeg_verdict}"
    );
}

/// RC.X1 — diagnostic dump for cross-decoder gate. Produces the same IVF
/// as the gate but PRESERVES it under the temp dir, plus per-frame
/// payloads (`/tmp/rc_x1_frame_N.bin`) and truncated cumulative IVFs
/// (`/tmp/rc_x1_thru_N.ivf`) so each prefix can be fed to vpxdec
/// independently. This is purely for human inspection.
#[test]
#[ignore = "RC.X1 diagnostic dump; run with `--ignored`"]
fn rc_x1_diagnostic_dump() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;
    const TARGET_BITRATE_BPS: u32 = 1_000_000;

    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };
    let (frames, framerate) =
        y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize).expect("y4m_read_cropped");
    assert_eq!(frames.len(), N_FRAMES as usize);

    let cfg = EncoderConfig {
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
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(N_FRAMES as usize);
    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        eprintln!(
            "  encoded frame {idx}: kind={:?} q={} bytes={} first16={:02x?}",
            pkt.frame_kind,
            pkt.q_index,
            pkt.data.len(),
            &pkt.data[..pkt.data.len().min(16)]
        );
        packets.push(pkt.data);
    }

    let tmp_dir = std::env::temp_dir();

    // Dump every frame's raw payload.
    for (idx, pkt) in packets.iter().enumerate() {
        let p = tmp_dir.join(format!("rc_x1_frame_{idx}.bin"));
        std::fs::write(&p, pkt).expect("write frame bin");
        eprintln!("  wrote {} ({} bytes)", p.display(), pkt.len());
    }

    // Write cumulative IVFs containing frames 0..=k for k = 0..N_FRAMES-1.
    let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
    for end in 1..=packets.len() {
        let ivf_path = tmp_dir.join(format!("rc_x1_thru_{}.ivf", end - 1));
        let refs: Vec<&[u8]> = packets[..end].iter().map(Vec::as_slice).collect();
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
        eprintln!("  wrote {} ({} packets)", ivf_path.display(), end);
    }

    // Also write the full IVF under a stable name.
    let full_ivf = tmp_dir.join("rc_x1_full.ivf");
    let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    {
        let mut f = std::fs::File::create(&full_ivf).expect("create full ivf");
        write_ivf(&mut f, &header, &refs).expect("write_ivf full");
    }
    eprintln!(
        "RC.X1 diagnostic dump complete. Inspect files under: {}",
        tmp_dir.display()
    );
}

/// RC.X1 — dump per-SB leaf layout for frame 1 (rvp9-decoder view).
#[test]
#[ignore = "RC.X1 SB-leaf dump; run with `--ignored`"]
fn rc_x1_sb_leaf_dump() {
    use rvp9_core::mode_info::BlockMode;

    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;
    const TARGET_BITRATE_BPS: u32 = 1_000_000;

    let Some(y4m_path) = probe_testsrc_720p() else {
        return;
    };
    let (frames, framerate) =
        y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize).expect("y4m_read_cropped");

    let cfg = EncoderConfig {
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
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        for pf in &parsed {
            let _ = dec.decode_frame(pf).expect("rvp9 decode_frame");
        }
        if idx != 1 {
            continue;
        }
        let grid = dec.last_frame_mi_grid();
        let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
        let mi_cols_us = mi_cols as usize;
        // For each 8x8 SB-relative position within each SB, print bsize + skip + is_inter.
        for sb_row in 0..((mi_rows as usize) / 8) {
            for sb_col in 0..(mi_cols_us / 8) {
                eprintln!("SB ({}, {}):", sb_row * 8, sb_col * 8);
                for r in 0..8 {
                    for c in 0..8 {
                        let mi_r = sb_row * 8 + r;
                        let mi_c = sb_col * 8 + c;
                        let cell = grid[mi_r * mi_cols_us + mi_c];
                        if let Some(mi) = cell {
                            let is_inter = matches!(mi.mode, BlockMode::Inter { .. });
                            eprint!(
                                " [{}.{}|{:?}|sk={}|i={}]",
                                r,
                                c,
                                mi.block_size,
                                u8::from(mi.skip),
                                u8::from(is_inter)
                            );
                        } else {
                            eprint!(" [{}.{}|None]", r, c);
                        }
                    }
                    eprintln!();
                }
            }
        }
    }
}

/// RC.X1 — decode each frame through rvp9-decoder and print summary stats
/// (per-frame mode/partition/skip breakdown) so we can spot anomalies in
/// our frame-1's emitted modes.
#[test]
#[ignore = "RC.X1 internal mi-grid dump; run with `--ignored`"]
fn rc_x1_internal_dump() {
    use rvp9_core::mode_info::BlockMode;

    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;
    const TARGET_BITRATE_BPS: u32 = 1_000_000;

    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m not available.");
        return;
    };
    let (frames, framerate) =
        y4m_read_cropped(&y4m_path, W, H, N_FRAMES as usize).expect("y4m_read_cropped");

    let cfg = EncoderConfig {
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
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    for (idx, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        for pf in &parsed {
            let _ = dec.decode_frame(pf).expect("rvp9 decode_frame");
        }

        // Inspect the just-decoded mi grid.
        let grid = dec.last_frame_mi_grid();
        let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
        let mut counts: std::collections::HashMap<String, u32> = Default::default();
        let mut skip_true = 0u32;
        let mut skip_false = 0u32;
        let mut mode_zero = 0u32;
        let mut mode_nz = 0u32;
        let mut nonref_zero = 0u32;
        let mut nonzero_mv = 0u32;
        for cell in grid {
            if let Some(c) = cell {
                let bs = format!("{:?}", c.block_size);
                *counts.entry(bs).or_insert(0) += 1;
                if c.skip {
                    skip_true += 1;
                } else {
                    skip_false += 1;
                }
                match c.mode {
                    BlockMode::Inter { mode, mvs, .. } => {
                        let mv = mvs[0];
                        if mv.col == 0 && mv.row == 0 {
                            mode_zero += 1
                        } else {
                            nonzero_mv += 1
                        }
                        // Track mode discriminator separately for diagnosis.
                        let mode_str = format!("{:?}", mode);
                        *counts.entry(mode_str).or_insert(0) += 1;
                    }
                    BlockMode::Intra { .. } => {
                        mode_nz += 1;
                    }
                }
            } else {
                nonref_zero += 1;
            }
        }
        eprintln!(
            "frame {idx}: kind={:?} q={} bytes={} mi_dims=({},{}) bsize_counts={:?} skip(true/false)=({}/{}) mv(zero/nonzero)=({}/{}) intra={} unpopulated={}",
            pkt.frame_kind,
            pkt.q_index,
            pkt.data.len(),
            mi_rows,
            mi_cols,
            counts,
            skip_true,
            skip_false,
            mode_zero,
            nonzero_mv,
            mode_nz,
            nonref_zero,
        );
    }
}
