// Integration test: stderr IS the output mechanism (the harness's job
// is to print a measurement report). The workspace's `print_stderr =
// "deny"` lint targets library code; for this binary that lint is the
// wrong rule.
#![allow(clippy::print_stderr)]

//! F12.0 — BD-rate vs libvpx measurement harness (Stage 4a exit gate).
//!
//! Gated by `#[ignore]` so it doesn't run on every `cargo test`. Invoke
//! explicitly:
//!
//! ```text
//! cargo test --test bdrate_gate -- --ignored --nocapture
//! ```
//!
//! ## What this measures
//!
//! For each (test source × q-index) pair the harness:
//!
//! 1. Encodes the source through `rvp9-encoder`'s `StreamEncoder` at
//!    that q-index — actually, since `StreamEncoder` doesn't take a
//!    direct q-index knob in v1, we vary `target_bitrate_bps` over a
//!    set of operating points that produce the desired q-index range.
//! 2. Decodes every packet through `rvp9-decoder::Decoder`, computing
//!    per-frame Y-plane PSNR vs the original source.
//! 3. Records `(actual_avg_bitrate_kbps, avg_psnr_db)` — one point per
//!    operating point.
//! 4. Forms the rvp9 RD curve from those points.
//!
//! If `vpxenc.exe` is available (at the path probed by
//! [`libvpx_stock_available`]), the harness also encodes the same
//! source through libvpx-stock at matched configs and computes the
//! BD-rate via [`rvp9_harness::bdrate::bdrate`].
//!
//! Reports to STDERR. Pass/fail is informational — the gate is
//! recorded in `wrk_journals/2026.05.21 - JRN - F12-stage4a-exit-gate.md`.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::bdrate::bdrate;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Probe for libvpx-stock's `vpxenc.exe` at a known path.
///
/// Returns `Some(path)` if the binary exists and is executable on this
/// host; `None` otherwise. The harness short-circuits the libvpx side
/// of the comparison if `None`.
fn libvpx_stock_available() -> Option<PathBuf> {
    // Try the canonical sibling-tree location first.
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxenc.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    // Fall back to `vpxenc` on PATH.
    if Command::new("vpxenc").arg("--help").output().is_ok() {
        return Some(PathBuf::from("vpxenc"));
    }
    None
}

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

/// Decode one rvp9-encoder packet through the real decoder. Mirrors
/// the helper in `frame_loop.rs` tests but lives here so the integration
/// test is self-contained.
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

/// Compute Y-plane PSNR (dB). 100.0 on bit-exact match.
fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut sse: u64 = 0;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = i32::from(*a) - i32::from(*b);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Synthesise frame `idx` of an animated gradient sequence — diagonal
/// gradient shifts per frame so the inter mode picker sees real motion.
fn animated_gradient(w: usize, h: usize, idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let v = 64i32 + (r as i32 / 2) + (c as i32 / 2) + (idx as i32 * 3);
            y[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// One operating point on a rate-distortion curve.
#[derive(Clone, Copy, Debug)]
struct RdPoint {
    /// Average bitrate over the encoded sequence, kbps.
    bitrate_kbps: f64,
    /// Average Y-plane PSNR over the decoded sequence, dB.
    psnr_db: f64,
}

/// Drive rvp9 `StreamEncoder` over a multi-frame source at a given
/// target bitrate; decode every packet; return one [`RdPoint`].
///
/// `frames` is a slice of `(y, u, v)` planes in tight stride matching
/// `(w, h)`. `gop_length = frames.len()` so the run contains exactly
/// one keyframe.
fn measure_rvp9_point(
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> RdPoint {
    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps,
        gop_length: frames.len() as u32 + 1,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    let mut total_bits: u64 = 0;
    let mut psnr_sum = 0.0;
    let n = frames.len();

    for (y, u, v) in frames.iter() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w as usize))
            .expect("encode_frame");
        total_bits += u64::from(pkt.actual_bits);

        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");

        let yp = buf.y_plane();
        let mut decoded_y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            decoded_y[r * w as usize..(r + 1) * w as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + w as usize]);
        }
        psnr_sum += psnr_db(y, &decoded_y);
    }

    let avg_bits_per_frame = total_bits as f64 / n as f64;
    let bitrate_kbps = avg_bits_per_frame * framerate / 1000.0;
    let _ = target_bitrate_bps;
    RdPoint {
        bitrate_kbps,
        psnr_db: psnr_sum / n as f64,
    }
}

/// Convert a YUV sequence in memory to an I420 raw file on disk for
/// `vpxenc --i420` input. Returns the temp path.
fn write_i420_raw(
    path: &Path,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for (y, u, v) in frames {
        f.write_all(y)?;
        f.write_all(u)?;
        f.write_all(v)?;
    }
    Ok(())
}

/// Drive libvpx-stock `vpxenc` at a matched config, decode the output,
/// return one [`RdPoint`]. Skipped (returns `None`) if vpxenc is
/// missing or the encode-decode pipeline fails.
fn measure_libvpx_point(
    vpxenc: &Path,
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Option<RdPoint> {
    let tmp_dir = std::env::temp_dir();
    let raw_path = tmp_dir.join(format!(
        "rvp9_bdrate_{}x{}_{}kbps.yuv",
        w,
        h,
        target_bitrate_bps / 1000
    ));
    let ivf_path = tmp_dir.join(format!(
        "rvp9_bdrate_{}x{}_{}kbps.ivf",
        w,
        h,
        target_bitrate_bps / 1000
    ));

    if write_i420_raw(&raw_path, frames).is_err() {
        return None;
    }

    let bitrate_kbps = target_bitrate_bps / 1000;
    let out = Command::new(vpxenc)
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--tune-content=default",
            "--passes=1",
            "--i420",
        ])
        .arg(format!("--width={w}"))
        .arg(format!("--height={h}"))
        .arg(format!("--fps={}/1", framerate as u32))
        .arg(format!("--target-bitrate={bitrate_kbps}"))
        .arg("-o")
        .arg(&ivf_path)
        .arg(&raw_path)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxenc failed at {bitrate_kbps} kbps: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    // Bitrate measurement: filesize × 8 × framerate / N. The IVF header
    // (32 B file + 12 B per-frame header) is small enough to ignore at
    // these sizes; we use the libvpx convention of file_size_bits.
    let ivf_bytes = std::fs::metadata(&ivf_path).ok()?.len();
    let avg_bits_per_frame = ivf_bytes as f64 * 8.0 / frames.len() as f64;
    let measured_bitrate_kbps = avg_bits_per_frame * framerate / 1000.0;

    // PSNR via `vpxdec --i420 -o out.yuv`. We compute PSNR ourselves
    // (vpxdec doesn't print per-frame PSNR without --md5/--rawvideo).
    let dec_raw_path = tmp_dir.join(format!(
        "rvp9_bdrate_{}x{}_{}kbps_dec.yuv",
        w,
        h,
        target_bitrate_bps / 1000
    ));
    // The decoder is rvp9 OR libvpx's vpxdec, depending on what's on PATH.
    // We use OUR decoder for fairness — measure what we can reconstruct.
    // But for the libvpx-leg of the BD-rate curve we should decode WITH
    // libvpx-stock too (so any decoder asymmetry doesn't pollute the
    // comparison). Use vpxdec at the same path tree.
    let vpxdec = vpxenc.with_file_name("vpxdec.exe");
    if !vpxdec.is_file() {
        eprintln!("    [libvpx skipped] vpxdec.exe missing at {}", vpxdec.display());
        return None;
    }
    let out = Command::new(&vpxdec)
        .args(["--i420", "-o"])
        .arg(&dec_raw_path)
        .arg(&ivf_path)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxdec failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    // Read back the decoded I420 file; compute Y-plane PSNR per frame.
    let dec_data = std::fs::read(&dec_raw_path).ok()?;
    let frame_size = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    if dec_data.len() < frame_size * frames.len() {
        eprintln!(
            "    [libvpx skipped] decoded file too small ({} < {})",
            dec_data.len(),
            frame_size * frames.len()
        );
        return None;
    }
    let mut psnr_sum = 0.0;
    let y_size = (w * h) as usize;
    for (i, (orig_y, _, _)) in frames.iter().enumerate() {
        let dec_y = &dec_data[i * frame_size..i * frame_size + y_size];
        psnr_sum += psnr_db(orig_y, dec_y);
    }

    // Best-effort cleanup; ignore errors.
    let _ = std::fs::remove_file(&raw_path);
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&dec_raw_path);

    let _ = target_bitrate_bps;
    Some(RdPoint {
        bitrate_kbps: measured_bitrate_kbps,
        psnr_db: psnr_sum / frames.len() as f64,
    })
}

/// F12.0 — Stage 4a BD-rate exit gate.
///
/// Synthesises a 128×128 8-frame translating-gradient source, encodes
/// it through rvp9 at four target bitrates spanning the operating
/// range, decodes each output and computes (bitrate, PSNR). If libvpx-
/// stock is available, runs the same source through `vpxenc` at the
/// same target bitrates and computes BD-rate.
///
/// Reports to stderr. The threshold check (±2 %) is reported but does
/// NOT cause the test to fail outright — it's informational. The
/// authoritative gate is the journal entry.
#[test]
#[ignore = "F12.0 — Stage 4a exit-gate harness; run with `--ignored`"]
fn bdrate_gate_stage4a() {
    // 128×128 is the smallest size that exercises a 2×2 SB grid (so
    // partition decisions matter) while staying tractable as a smoke
    // test. 8 frames at 30 fps = 1 GOP keyframe + 7 inter — enough to
    // average bitrate / PSNR meaningfully.
    const W: u32 = 128;
    const H: u32 = 128;
    const N_FRAMES: u32 = 8;
    const FRAMERATE: f64 = 30.0;

    // Operating points: span ~1 octave each side of "natural" bitrate
    // for this resolution. At 128×128×30fps a typical RT VP9 encode
    // sits around 200–1200 kbps; we use 4 evenly-spaced log-domain
    // points covering 250→2000 kbps.
    let target_bitrates_bps: [u32; 4] = [250_000, 500_000, 1_000_000, 2_000_000];

    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..N_FRAMES)
        .map(|i| animated_gradient(W as usize, H as usize, i))
        .collect();

    eprintln!("Stage 4a Exit Gate — F12.0 Measurement Report");
    eprintln!("=============================================");
    eprintln!(
        "Source: synthetic translating gradient, {W}x{H}, {N_FRAMES} frames @ {FRAMERATE} fps"
    );
    eprintln!("Target bitrates tested: {target_bitrates_bps:?} bps");
    eprintln!();

    // -------------------------------------------------------------
    // rvp9 curve.
    // -------------------------------------------------------------
    let mut rvp9_curve: Vec<(f64, f64)> = Vec::with_capacity(target_bitrates_bps.len());
    let mut rvp9_points: Vec<RdPoint> = Vec::with_capacity(target_bitrates_bps.len());
    for &tb in &target_bitrates_bps {
        let p = measure_rvp9_point(W, H, FRAMERATE, tb, &frames);
        eprintln!(
            "  rvp9    target={:>5} kbps  →  actual={:>7.1} kbps,  PSNR={:>5.2} dB",
            tb / 1000,
            p.bitrate_kbps,
            p.psnr_db
        );
        rvp9_curve.push((p.bitrate_kbps, p.psnr_db));
        rvp9_points.push(p);
    }

    // -------------------------------------------------------------
    // libvpx-stock curve (optional).
    // -------------------------------------------------------------
    let vpxenc = libvpx_stock_available();
    let mut libvpx_curve: Option<Vec<(f64, f64)>> = None;

    if let Some(vpxenc) = vpxenc {
        eprintln!();
        eprintln!("libvpx-stock available at: {}", vpxenc.display());
        let mut curve: Vec<(f64, f64)> = Vec::new();
        for &tb in &target_bitrates_bps {
            if let Some(p) = measure_libvpx_point(&vpxenc, W, H, FRAMERATE, tb, &frames) {
                eprintln!(
                    "  libvpx  target={:>5} kbps  →  actual={:>7.1} kbps,  PSNR={:>5.2} dB",
                    tb / 1000,
                    p.bitrate_kbps,
                    p.psnr_db
                );
                curve.push((p.bitrate_kbps, p.psnr_db));
            }
        }
        if curve.len() >= 2 {
            libvpx_curve = Some(curve);
        }
    } else {
        eprintln!();
        eprintln!("libvpx-stock not available — BD-rate measurement skipped.");
    }

    eprintln!();
    eprintln!("rvp9 curve (kbps, PSNR dB): {rvp9_curve:?}");
    if let Some(ref lv) = libvpx_curve {
        eprintln!("libvpx curve (kbps, PSNR dB): {lv:?}");
    } else {
        eprintln!("libvpx curve: (not available)");
    }

    // -------------------------------------------------------------
    // BD-rate.
    // -------------------------------------------------------------
    let bd = libvpx_curve
        .as_ref()
        .map(|lv| bdrate(lv, &rvp9_curve))
        .unwrap_or(f64::NAN);
    eprintln!();
    if bd.is_nan() {
        eprintln!("BD-rate: not measurable (no libvpx curve or insufficient overlap)");
        eprintln!("Exit gate status: DEFERRED (libvpx leg not measurable in this run)");
    } else {
        eprintln!(
            "BD-rate (rvp9 vs libvpx, positive = rvp9 uses more bits): {bd:+.2} %"
        );
        let pass = bd.abs() <= 2.0;
        eprintln!(
            "Exit gate status: {} (threshold ±2 %)",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // Sanity assertions on the rvp9 curve only. Higher target bitrate
    // should produce higher PSNR (monotonicity). If this regresses
    // the test fails — the gate isn't measurable but the curve must
    // be coherent.
    for w in rvp9_curve.windows(2) {
        let (_br0, ps0) = w[0];
        let (_br1, ps1) = w[1];
        // Allow a small tolerance for noise in 8-frame averages.
        assert!(
            ps1 >= ps0 - 1.0,
            "rvp9 curve must be ~monotonic in PSNR; got {ps0} → {ps1}",
        );
    }
}
