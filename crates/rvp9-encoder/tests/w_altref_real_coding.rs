// Integration test: stderr IS the output mechanism (per-decoder verdict
// reporting), mirroring `cross_decoder_gate.rs`. The workspace's
// `print_stderr = "deny"` lint targets library code, not this binary.
#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

//! W-ALTREF — REAL ALTREF per-leaf coding PROOF (keystone gate).
//!
//! Upgrades the ARF from the retired ZEROMV-skip scaffold (slot 2 was a
//! verbatim LAST copy) to REAL per-leaf coding: the temporally-filtered
//! ARF source is run through the SAME real-RD partition + inter-mode
//! picker the show frames use (`pick_and_simulate_inter_encode_partitions`),
//! choosing intra or inter-from-LAST per leaf by RD and producing a
//! genuine reconstruction the decoder reproduces bit-for-bit.
//!
//! Oracles:
//!
//! 1. **vpxdec byte-exact round-trip (the critical oracle).** Encode an
//!    ALTREF-containing stream on content with motion, wrap in IVF, and
//!    decode through both rvp9-decoder and libvpx-stock `vpxdec`. The
//!    two MUST agree byte-exact — this is the first byte-exact gate on
//!    an rvp9-encoded ALTREF superframe. Reuses the exact `vpxdec
//!    --i420` invocation from `cross_decoder_gate.rs`. Per-decoder skip
//!    if vpxdec is unavailable (informational, like the sibling gate).
//!
//! 2. **Anti-G3: slot 2 differs from LAST.** The encoder exposes a
//!    witness (`diag_last_arf_real_coded`) recording whether the coded
//!    ARF reconstruction differs from the LAST it predicted against.
//!    The ZEROMV-skip scaffold made slot 2 == LAST verbatim; real
//!    coding makes it differ. This assertion FAILS if the real-coding
//!    wiring is stripped back to the copy.
//!
//! libvpx file:line citations:
//! - `vp9/encoder/vp9_encoder.c:5093-5142` — the ARF runs as a normal
//!   frame encode (same per-block RD path) with `cm->show_frame = 0`.
//! - `vp9/encoder/vp9_rdopt.c` — the per-block inter RD path the ARF
//!   reuses (our `pick_and_simulate_inter_encode_partitions`).
//! - `vp9/encoder/vp9_encoder.c:5034-5049` `set_ref_sign_bias` — the
//!   sign-bias rule wired into the header.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;

const W: u32 = 128;
const H: u32 = 128;

// ---------------------------------------------------------------------------
// Decoder probe + invocation — copied verbatim from cross_decoder_gate.rs so
// both gates use the identical vpxdec command line.
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

/// Invoke `vpxdec --codec=vp9 --i420 -o <out> <ivf>` (identical to the
/// cross-decoder gate) and return the raw I420 bytes.
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

// ---------------------------------------------------------------------------
// Helpers.
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

/// Concatenate the tight-stride Y/U/V of a decoded buffer in I420 order
/// (matches `vpxdec --i420`). Copied from `cross_decoder_gate.rs`.
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
// PROOF.
// ---------------------------------------------------------------------------

/// W-ALTREF — anti-G3: the coded ARF reconstruction (slot 2) must differ
/// from the LAST reconstruction it predicted against. This runs WITHOUT
/// any external decoder, so it is always live (not `#[ignore]`-gated) and
/// fails immediately if the real-coding wiring is reverted to the
/// ZEROMV-skip copy.
#[test]
fn altref_slot2_differs_from_last_recon() {
    // testsrc cropped to 128x128 gives real motion/texture so the ARF's
    // RD picker emits genuine residual; a flat synthetic frame could
    // coincidentally reconstruct equal to LAST.
    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available.");
        return;
    };
    let n_frames: u32 = 16;
    let (frames, framerate) = y4m_read_cropped(&y4m_path, W, H, n_frames as usize)
        .expect("y4m_read_cropped");
    assert_eq!(frames.len(), n_frames as usize);

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 1_000_000,
        // gop_length 16 → arf_interval = max(8, 2) = 8.
        gop_length: 16,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        // lag must be > arf_interval so the lookahead can peek a
        // neighbour for the temporal filter when the cadence fires.
        lag_in_frames: 9,
        enable_alt_ref: true,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    for (y, u, v) in &frames {
        let _ = enc
            .encode_frame_delayed(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame_delayed");
    }
    for r in enc.flush() {
        let _ = r.expect("flush ok");
    }

    let witness = enc.diag_last_arf_real_coded();
    eprintln!("W-ALTREF anti-G3 witness: diag_last_arf_real_coded = {witness:?}");
    assert_eq!(
        witness,
        Some(true),
        "ARF reconstruction (slot 2) must DIFFER from the LAST it predicted \
         against (real per-leaf coding). `Some(false)` means slot 2 is a \
         verbatim LAST copy (the retired ZEROMV-skip scaffold); `None` means \
         no ARF fired — check the cadence/lag config."
    );

    // Cross-check: the persisted slot-2 buffer is non-degenerate.
    let slot2 = enc.diag_altref_recon_y();
    assert_eq!(slot2.len(), (W * H) as usize, "slot 2 Y plane sized w*h");
    assert!(
        slot2.iter().any(|&p| p != 0),
        "slot 2 reconstruction must hold real pixels, not all-zero"
    );
}

/// W-ALTREF — the critical oracle: an ALTREF-containing stream must be
/// byte-exact decodable by vpxdec (round-trip vs rvp9-decoder). This is
/// the first byte-exact gate on an rvp9-encoded ALTREF superframe.
/// `#[ignore]`-gated like the cross-decoder gate (needs the libvpx binary).
#[test]
#[ignore = "W-ALTREF byte-exact vpxdec round-trip; run with `--ignored`"]
fn altref_stream_byte_exact_vpxdec() {
    eprintln!("W-ALTREF — ALTREF byte-exact vpxdec round-trip");
    eprintln!("=============================================");

    let Some(y4m_path) = probe_testsrc_720p() else {
        eprintln!("[SKIPPED] testsrc_720p30.y4m corpus not available; cannot run gate.");
        return;
    };
    let n_frames: u32 = 16;
    let (frames, framerate) = y4m_read_cropped(&y4m_path, W, H, n_frames as usize)
        .expect("y4m_read_cropped");
    assert_eq!(frames.len(), n_frames as usize);
    eprintln!("Source: {} ({n_frames} frames @ {W}x{H})", y4m_path.display());

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 1_000_000,
        gop_length: 16,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 9,
        enable_alt_ref: true,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    // Encode (delayed + flush, lag_in_frames > 0). Collect transport
    // packets in encode order (superframes carry [ARF, show]).
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for (y, u, v) in &frames {
        if let Some(pkt) = enc
            .encode_frame_delayed(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame_delayed")
        {
            packets.push(pkt.data);
        }
    }
    for r in enc.flush() {
        packets.push(r.expect("flush ok").data);
    }
    eprintln!("Encoded {} transport packets", packets.len());

    // Confirm a real-coded ARF actually fired (otherwise the gate would
    // pass trivially on a stream with no ALTREF).
    assert_eq!(
        enc.diag_last_arf_real_coded(),
        Some(true),
        "expected a real-coded ARF in this stream"
    );

    // Confirm at least one non-show ARF is present at the wire level.
    let mut dec_probe = Decoder::new();
    let mut non_show = 0u32;
    let mut show = 0u32;
    for pkt in &packets {
        for pf in dec_probe.parse_packet(pkt).expect("parse_packet").frames() {
            if pf.header.show_frame {
                show += 1;
            } else {
                non_show += 1;
            }
        }
    }
    eprintln!("wire: {show} show frames, {non_show} non-show (ARF) frames");
    assert!(non_show >= 1, "stream must carry at least one non-show ARF");
    assert_eq!(show, n_frames, "show frames must equal input frames");

    // ---- Wrap in IVF ---------------------------------------------------
    let tmp_dir = std::env::temp_dir();
    let ivf_path = tmp_dir.join("rvp9_w_altref_real_coding.ivf");
    {
        let mut f = std::fs::File::create(&ivf_path).expect("create ivf");
        let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &header, &refs).expect("write_ivf");
    }
    eprintln!("IVF written to {}", ivf_path.display());

    // ---- rvp9-decoder baseline (show frames only — vpxdec emits only
    //      visible frames) -------------------------------------------------
    let rvp9_yuv: Vec<u8> = {
        let mut dec = Decoder::new();
        let mut out: Vec<u8> = Vec::new();
        for pkt in &packets {
            let parsed = decode_packet(&mut dec, pkt);
            for pf in &parsed {
                let buf = dec.decode_frame(pf).expect("rvp9 decode_frame");
                // Only visible frames land in vpxdec's output; the
                // non-show ARF is decoded (advances refs/FC) but not
                // emitted to the YUV file.
                if pf.header.show_frame {
                    out.extend_from_slice(&yv12_to_i420_bytes(&buf, W, H));
                }
            }
        }
        out
    };
    let frame_bytes = (W as usize) * (H as usize) * 3 / 2;
    assert_eq!(
        rvp9_yuv.len(),
        frame_bytes * (n_frames as usize),
        "rvp9-decoder must emit exactly {n_frames} visible frames"
    );

    // ---- vpxdec leg ----------------------------------------------------
    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[SKIPPED] vpxdec not found; ALTREF byte-exact gate is informational.");
        let _ = std::fs::remove_file(&ivf_path);
        return;
    };
    eprintln!("vpxdec available at: {}", vpxdec.display());
    let vpx_yuv_path = tmp_dir.join("rvp9_w_altref_real_coding_vpxdec.yuv");
    let vpx_yuv = match run_vpxdec(&vpxdec, &ivf_path, &vpx_yuv_path) {
        Ok(b) => b,
        Err(e) => {
            // A real finding — report as a blocker, do not force it.
            let _ = std::fs::remove_file(&ivf_path);
            panic!(
                "BLOCKER: vpxdec REJECTED the ALTREF stream: {e}\n\
                 (the ALTREF superframe is not decodable by libvpx)"
            );
        }
    };

    let verdict = if vpx_yuv.len() != rvp9_yuv.len() {
        format!(
            "FAIL (size mismatch: vpxdec={} bytes, rvp9={} bytes)",
            vpx_yuv.len(),
            rvp9_yuv.len()
        )
    } else if vpx_yuv == rvp9_yuv {
        "BYTE-EXACT".to_string()
    } else {
        let (max_abs, sse) = plane_diff_stats(&rvp9_yuv, &vpx_yuv);
        format!("FAIL (divergent: max_abs={max_abs}, sse={sse})")
    };
    eprintln!("rvp9-decoder vs vpxdec on ALTREF stream: {verdict}");

    let _ = std::fs::remove_file(&vpx_yuv_path);
    let _ = std::fs::remove_file(&ivf_path);

    assert_eq!(
        verdict, "BYTE-EXACT",
        "ALTREF stream must be byte-exact decodable by vpxdec; got: {verdict}"
    );
}
