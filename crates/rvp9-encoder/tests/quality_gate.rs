// Integration test: stderr IS the output mechanism (the harness's job
// is to print a measurement report). The workspace's `print_stderr =
// "deny"` lint targets library code; for this binary that lint is the
// wrong rule.
#![allow(clippy::print_stderr)]

//! F12.0 / F.0.1 — Quality vs libvpx measurement harness (Stage 4a
//! exit gate).
//!
//! Gated by `#[ignore]` so it doesn't run on every `cargo test`. Invoke
//! explicitly:
//!
//! ```text
//! cargo test --test quality_gate -- --ignored --nocapture
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
//! source through libvpx-stock at matched configs.
//!
//! ## Primary metric — PSNR-delta mean (F.0.1)
//!
//! After Track E closed the rate-control / structural confounds, rvp9's
//! PSNR curve still sits entirely below libvpx's on real content, so
//! the Bjøntegaard cubic has no PSNR overlap and BD-rate is structurally
//! undefined. F.0.1 promotes the matched-bitrate PSNR-delta mean to the
//! primary signal: gate PASS when
//! `mean_delta_db <= psnr_delta_threshold_db` AND
//! `count_in_range >= 2`. BD-rate is reported as a secondary line that
//! returns to a co-gate at F.4 once the curve overlap closes. See
//! `wrk_docs/Track_F/F0.1_HLD_psnr_delta_primary_gate.md` and
//! `wrk_journals/2026.05.21 - JRN - F12-stage4a-exit-gate.md` for the
//! full rationale.
//!
//! Reports to STDERR.

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::bdrate::{bdrate, psnr_delta_summary};
use rvp9_harness::y4m::{
    YuvFrameBuf, probe_foreman_cif, probe_testsrc_720p, read_cropped as y4m_read_cropped,
};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Track F pass/fail threshold on the foreman PSNR-delta mean (dB).
/// Updated per milestone — see the F.0.1 HLD §3.2 table.
/// 2026-05-23 (initial): +9.5 dB regression guard at the +9.07
/// baseline mean.
/// 2026-05-23 (F.4-2 activation): +9.1 dB after K1=4 activation
/// landed the F.4-1 sweep winner (foreman mean Δ +9.07 → +9.00 dB).
/// The +0.07 dB improvement is honest but tiny; the full +8 dB
/// closure to libvpx needs structural work (DV-F1-1 Hadamard port)
/// beyond F.4's parametric-tuning scope.
const TRACK_F_PSNR_DELTA_THRESHOLD_DB: f64 = 9.1;

/// testsrc is a content-artefact corner case where libvpx wins on
/// flat-content luck (78 dB on colour bars at 68 kbps — a regime no
/// realtime encoder follows without flat-content specific
/// optimisations per Track_F_baseline_numbers.md). This constant is a
/// COHERENCE SANITY CHECK, NOT A TRACK F GATE — its only job is to
/// catch wholesale rvp9 regression on testsrc. The Track F headline
/// metric lives on foreman.
const TRACK_F_PSNR_DELTA_THRESHOLD_TESTSRC_DB: f64 = 50.0;

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
/// one keyframe. `aq_mode` selects between Stage 4a baseline
/// ([`AqMode::Off`]) and Stage 4b cyclic refresh
/// ([`AqMode::CyclicRefresh`]).
fn measure_rvp9_point(
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
    aq_mode: AqMode,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> RdPoint {
    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps,
        gop_length: frames.len() as u32 + 1,
        aq_mode,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        min_qindex: 4,
        max_qindex: 255,
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
///
/// `aq_mode_int` is forwarded to `vpxenc --aq-mode=N`:
///
/// - `0` — adaptive-quantization off (libvpx baseline)
/// - `3` — cyclic refresh (libvpx's `CYCLIC_REFRESH_AQ`)
///
/// AQ.5 uses `0` and `3` for the Stage 4b exit-gate comparison.
fn measure_libvpx_point(
    vpxenc: &Path,
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
    aq_mode_int: u32,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Option<RdPoint> {
    let tmp_dir = std::env::temp_dir();
    let raw_path = tmp_dir.join(format!(
        "rvp9_bdrate_{}x{}_{}kbps_aq{}.yuv",
        w,
        h,
        target_bitrate_bps / 1000,
        aq_mode_int
    ));
    let ivf_path = tmp_dir.join(format!(
        "rvp9_bdrate_{}x{}_{}kbps_aq{}.ivf",
        w,
        h,
        target_bitrate_bps / 1000,
        aq_mode_int
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
        .arg(format!("--aq-mode={aq_mode_int}"))
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
        "rvp9_bdrate_{}x{}_{}kbps_aq{}_dec.yuv",
        w,
        h,
        target_bitrate_bps / 1000,
        aq_mode_int
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

/// Stage 4a quality gate body, parameterised over the source frames.
///
/// Pulled out so both [`quality_gate_stage4a`] (testsrc) and
/// [`quality_gate_stage4a_realcontent`] (foreman_cif) share the same
/// measurement-and-report code path while varying only the input clip
/// and the PSNR-delta threshold.
///
/// `psnr_delta_threshold_db` is the F.0.1 PASS threshold on the
/// in-range mean Δ (libvpx_interp − rvp9). PASS requires the mean to
/// be ≤ threshold AND `count_in_range >= 2`; otherwise FAIL (or
/// DEFERRED if no libvpx leg is available / too few in-range points).
fn run_stage4a_quality_gate(
    source_label: &str,
    frames: &[YuvFrameBuf],
    framerate: f64,
    target_bitrates_bps: &[u32],
    w: u32,
    h: u32,
    psnr_delta_threshold_db: f64,
) {
    let n_frames = frames.len();
    eprintln!("Stage 4a Exit Gate — F.0.1 Quality Measurement Report");
    eprintln!("==========================================================");
    eprintln!("Source: {source_label}");
    eprintln!("Frames: {n_frames}    Target bitrates: {target_bitrates_bps:?} bps");
    eprintln!("PSNR-delta PASS threshold: <= {psnr_delta_threshold_db:.2} dB");
    eprintln!();

    // -------------------------------------------------------------
    // rvp9 curve.
    // -------------------------------------------------------------
    let mut rvp9_curve: Vec<(f64, f64)> = Vec::with_capacity(target_bitrates_bps.len());
    for &tb in target_bitrates_bps {
        let p = measure_rvp9_point(w, h, framerate, tb, AqMode::Off, frames);
        eprintln!(
            "  rvp9    target={:>5} kbps  →  actual={:>7.1} kbps,  PSNR={:>5.2} dB",
            tb / 1000,
            p.bitrate_kbps,
            p.psnr_db
        );
        rvp9_curve.push((p.bitrate_kbps, p.psnr_db));
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
        for &tb in target_bitrates_bps {
            if let Some(p) = measure_libvpx_point(&vpxenc, w, h, framerate, tb, 0, frames) {
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
    // F.0.1 — PSNR-delta is the PRIMARY gate; BD-rate is reported as a
    // secondary informational line.
    //
    // After Track E's overshoot fix (E6/E7/E8) rvp9 operates at the
    // *correct* bitrates but with insufficient coding quality, so its
    // PSNR curve (24-27 dB on foreman) sits entirely below libvpx's
    // (28-37 dB). The Bjøntegaard cubic has no PSNR overlap to integrate,
    // so BD-rate is structurally NaN. F.0.1 promotes the matched-bitrate
    // PSNR-delta (libvpx_PSNR − rvp9_PSNR, interpolated at each rvp9
    // bitrate via `psnr_at_bitrate`) to the primary signal. BD-rate
    // returns as a co-gate at F.4 once block_yrd + find_mv_refs +
    // motion-aware partition close the curve gap. See
    // `wrk_docs/Track_F/F0.1_HLD_psnr_delta_primary_gate.md`.
    // -------------------------------------------------------------
    eprintln!();
    // Secondary line — BD-rate (informational only during F.1–F.3).
    let bd = libvpx_curve
        .as_ref()
        .map(|lv| bdrate(lv, &rvp9_curve))
        .unwrap_or(f64::NAN);
    if bd.is_nan() {
        eprintln!(
            "BD-rate (secondary, informational): not measurable \
             (no PSNR overlap or libvpx leg absent)"
        );
    } else {
        eprintln!(
            "BD-rate (secondary, informational; positive = rvp9 uses more bits): \
             {bd:+.2} %"
        );
    }

    // Primary metric — PSNR-delta summary.
    if let Some(ref lv) = libvpx_curve {
        eprintln!(
            "PRIMARY metric — matched-bitrate PSNR delta (libvpx_PSNR − rvp9_PSNR):"
        );
        eprintln!(
            "  Positive delta = rvp9 trails libvpx at the same bitrate (dB)."
        );
        let lv_br_min = lv.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let lv_br_max = lv.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        // Per-point table — preserves the E15 reporter format that the
        // baseline doc reproduces exactly.
        for &(br_r, psnr_r) in rvp9_curve.iter() {
            match rvp9_harness::bdrate::psnr_at_bitrate(lv, br_r) {
                Some(psnr_l) => {
                    let d = psnr_l - psnr_r;
                    eprintln!(
                        "    rvp9 {br_r:>7.1} kbps / {psnr_r:>5.2} dB  vs  \
                         libvpx-interp {psnr_l:>5.2} dB  →  Δ = {d:+5.2} dB"
                    );
                }
                None => {
                    eprintln!(
                        "    rvp9 {br_r:>7.1} kbps / {psnr_r:>5.2} dB  → \
                         OUT OF libvpx bitrate range [{lv_br_min:.1}, {lv_br_max:.1}] \
                         kbps, skipped"
                    );
                }
            }
        }

        match psnr_delta_summary(lv, &rvp9_curve) {
            Some(s) => {
                let skipped_suffix = if s.count_out_of_range == 0 {
                    String::new()
                } else {
                    format!(" ({} skipped out-of-range)", s.count_out_of_range)
                };
                eprintln!(
                    "  Summary across {} in-range point(s){}: mean Δ = {:+5.2} dB, \
                     max Δ = {:+5.2} dB, min Δ = {:+5.2} dB",
                    s.count_in_range,
                    skipped_suffix,
                    s.mean_delta_db,
                    s.max_delta_db,
                    s.min_delta_db,
                );
                // Gate decision. PASS requires both:
                //   (a) mean Δ <= threshold (the milestone metric), AND
                //   (b) >= 2 in-range points (a single-point mean is not
                //       a meaningful summary of curve-distance — see HLD §3.2).
                // DEFERRED (not FAIL) when (b) is unmet, because that
                // condition reflects oracle coverage, not encoder quality.
                if s.count_in_range < 2 {
                    eprintln!(
                        "Exit gate status: DEFERRED (only {} in-range point(s); \
                         need >= 2 for a meaningful mean — this reflects oracle \
                         coverage, not encoder quality)",
                        s.count_in_range
                    );
                } else if s.mean_delta_db <= psnr_delta_threshold_db {
                    eprintln!(
                        "Exit gate status: PASS (mean Δ {:+.2} dB <= threshold \
                         {:+.2} dB, count_in_range = {})",
                        s.mean_delta_db,
                        psnr_delta_threshold_db,
                        s.count_in_range,
                    );
                } else {
                    eprintln!(
                        "Exit gate status: FAIL (mean Δ {:+.2} dB > threshold \
                         {:+.2} dB, count_in_range = {})",
                        s.mean_delta_db,
                        psnr_delta_threshold_db,
                        s.count_in_range,
                    );
                }
            }
            None => {
                eprintln!(
                    "  No rvp9 operating points fell inside the libvpx bitrate range; \
                     PSNR-delta summary not computable."
                );
                eprintln!(
                    "Exit gate status: DEFERRED (no in-range points — this reflects \
                     oracle coverage, not encoder quality)"
                );
            }
        }
    } else {
        eprintln!(
            "PRIMARY metric: not measurable (no libvpx curve available in this run)"
        );
        eprintln!(
            "Exit gate status: DEFERRED (libvpx leg not measurable in this run)"
        );
    }

    // Sanity assertions on the rvp9 curve only. Higher target bitrate
    // should produce higher PSNR (monotonicity). If this regresses
    // the test fails — the gate isn't measurable but the curve must
    // be coherent. Real-content sources may exhibit a larger PSNR
    // dip than synthetic gradient at the low end due to motion-
    // dependent quantisation, so the tolerance below is widened a
    // touch relative to the synthetic test.
    for w in rvp9_curve.windows(2) {
        let (_br0, ps0) = w[0];
        let (_br1, ps1) = w[1];
        // Allow a small tolerance for noise in 8-frame averages.
        assert!(
            ps1 >= ps0 - 1.5,
            "rvp9 curve must be ~monotonic in PSNR; got {ps0} → {ps1}",
        );
    }
}

/// Load a y4m source for the quality gate, falling back to a synthetic
/// generator on any failure. Returns `(frames, framerate, label)` ready
/// for [`run_stage4a_quality_gate`].
fn load_y4m_or_synthetic(
    probed: Option<&Path>,
    w: u32,
    h: u32,
    n_frames: u32,
    framerate_hint: f64,
    synth_label: &str,
    synth: impl Fn(usize, usize, u32) -> YuvFrameBuf,
) -> (Vec<YuvFrameBuf>, f64, String) {
    if let Some(path) = probed {
        match y4m_read_cropped(path, w, h, n_frames as usize) {
            Ok((frames, fr)) if frames.len() == n_frames as usize => {
                let label = format!(
                    "y4m centre-crop from {} → {w}x{h} @ {fr} fps",
                    path.display()
                );
                return (frames, fr, label);
            }
            Ok((frames, _)) => {
                eprintln!(
                    "    [y4m fallback] only got {}/{} frames; using {synth_label}",
                    frames.len(),
                    n_frames
                );
            }
            Err(e) => {
                eprintln!("    [y4m fallback] read failed: {e}; using {synth_label}");
            }
        }
    }
    let frames: Vec<YuvFrameBuf> = (0..n_frames)
        .map(|i| synth(w as usize, h as usize, i))
        .collect();
    let label = format!(
        "{synth_label} (no y4m available), {w}x{h} @ {framerate_hint} fps"
    );
    (frames, framerate_hint, label)
}

/// F.0.1 — Stage 4a quality exit gate (testsrc 720p crop).
///
/// Encodes 64 frames of a 256×256 centre-crop of `testsrc_720p30.y4m`
/// through rvp9 at four target bitrates spanning the operating range,
/// decodes each output and computes (bitrate, PSNR). If libvpx-stock
/// is available, runs the same source through `vpxenc` at the same
/// target bitrates and reports both the PSNR-delta primary metric and
/// BD-rate secondary line.
///
/// testsrc is largely flat colour bars + simple motion, so both rvp9
/// and libvpx saturate at ~75 kbps regardless of target. The
/// [`TRACK_F_PSNR_DELTA_THRESHOLD_TESTSRC_DB`] threshold is a coherence
/// sanity check (+50 dB), NOT the Track F gate — the headline metric
/// lives on foreman via [`quality_gate_stage4a_realcontent`].
#[test]
#[ignore = "F.0.1 — Stage 4a quality-gate harness; run with `--ignored`"]
fn quality_gate_stage4a() {
    // 256×256 = 4×4 SB grid — exercises non-trivial partition decisions
    // and is wide enough that the AQ-off baseline RD curve has
    // meaningful slope. 64 frames at 30 fps = 1 GOP keyframe + 63 inter
    // — gives the RC.1 rate-correction feedback loop AND the
    // `avg_frame_qindex_inter` EWMA enough samples to escape the
    // worst_quality init clamp. The night-6 F12 midpoint at 16 frames
    // showed all four target bitrates producing the same actual bitrate
    // because the EWMA stayed pinned high. RC.6 confirmed that 60+
    // frames was required for measurability.
    // 256×256 = 4×4 SB grid — exercises non-trivial partition decisions
    // and is wide enough that the AQ-off baseline RD curve has
    // meaningful slope. 64 frames at 30 fps = 1 GOP keyframe + 63 inter
    // — gives the RC.1 rate-correction feedback loop AND the
    // `avg_frame_qindex_inter` EWMA enough samples to escape the
    // worst_quality init clamp. The night-6 F12 midpoint at 16 frames
    // showed all four target bitrates producing the same actual bitrate
    // because the EWMA stayed pinned high. RC.6 confirmed that 60+
    // frames was required for measurability.
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;
    const FRAMERATE_HINT: f64 = 30.0;

    // Operating points: 25-250 kbps spans 3 octaves at the LOW end of
    // the bitrate spectrum, where both encoders have to push Q high
    // enough that the R-D curves actually slope. RC.6 measured both
    // rvp9 and libvpx saturating at ~75 kbps / 59-82 dB on the
    // testsrc 256×256 corpus at the original 250-2000 kbps range:
    // the content is too easy at 256×256 to fill higher bitrates,
    // and both encoders just plateau. By targeting below the
    // saturation knee we get a measurable curve.
    let target_bitrates_bps: [u32; 4] = [25_000, 50_000, 100_000, 250_000];

    // Source selection: testsrc when available, fall back to in-test
    // synthetic animated gradient.
    let probed = probe_testsrc_720p();
    let (frames, framerate, source_label) = load_y4m_or_synthetic(
        probed.as_deref(),
        W,
        H,
        N_FRAMES,
        FRAMERATE_HINT,
        "synthetic animated-gradient",
        animated_gradient,
    );

    run_stage4a_quality_gate(
        &source_label,
        &frames,
        framerate,
        &target_bitrates_bps,
        W,
        H,
        TRACK_F_PSNR_DELTA_THRESHOLD_TESTSRC_DB,
    );
}

/// F.0.1 — Stage 4a quality exit gate on real-motion content
/// (foreman_cif). **THIS IS THE HEADLINE TRACK F GATE.**
///
/// Companion to [`quality_gate_stage4a`]: the testsrc clip is largely
/// flat colour bars and saturates at ~75 kbps PSNR plateaus regardless
/// of target bitrate. foreman_cif (CIF 352×288, 30 fps, 300 frames)
/// has head motion, panning camera motion, and texture (face, hair,
/// construction-site detail) that stresses DCT, MV search, and motion
/// compensation. The PSNR-delta mean here is the metric to drive down
/// at every Track F milestone — see
/// `wrk_docs/Track_F/F0.1_HLD_psnr_delta_primary_gate.md`.
///
/// Source: standard Hannover/derf-collection clip distributed by
/// xiph.org (`media.xiph.org/video/derf/y4m/foreman_cif.y4m`). The y4m
/// file is gitignored — see `tests/corpus/sources/README.md` for the
/// acquisition steps. If the file is missing the test skips early
/// without panicking.
#[test]
#[ignore = "F.0.1 — Stage 4a real-content quality gate; run with `--ignored`"]
fn quality_gate_stage4a_realcontent() {
    // 256×256 centre crop of CIF (352×288). The face fills the centre
    // of the frame so the crop captures the principal motion. Crop
    // offsets: x = (352-256)/2 = 48, y = (288-256)/2 = 16.
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;

    let Some(path) = probe_foreman_cif() else {
        eprintln!(
            "    [skip] foreman_cif.y4m not found; place it at tests/corpus/foreman_cif.y4m"
        );
        eprintln!(
            "    See tests/corpus/sources/README.md for acquisition steps."
        );
        return;
    };

    let (frames, framerate) = match y4m_read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "    [skip] foreman_cif only yielded {}/{} frames",
                f.len(),
                N_FRAMES
            );
            return;
        }
        Err(e) => {
            eprintln!("    [skip] foreman_cif read failed: {e}");
            return;
        }
    };

    // Same target-bitrate range as the testsrc test — keeps the two
    // gates directly comparable.
    let target_bitrates_bps: [u32; 4] = [25_000, 50_000, 100_000, 250_000];
    let source_label = format!(
        "foreman_cif y4m centre-crop from {} → {W}x{H} @ {framerate} fps",
        path.display()
    );
    run_stage4a_quality_gate(
        &source_label,
        &frames,
        framerate,
        &target_bitrates_bps,
        W,
        H,
        TRACK_F_PSNR_DELTA_THRESHOLD_DB,
    );
}

// =========================================================================
// AQ.5 — Stage 4b exit-gate harness: 4-curve AQ comparison.
//
// Measures the BD-rate gain from turning AQ on, separately for rvp9 and
// libvpx, then compares the two gains. PLAN.md Stage 4b exit gate:
// `|gain_rvp9 - gain_libvpx| <= 0.3 %`.
//
// The synthetic 128×128 gradient used by [`quality_gate_stage4a`] is
// degenerate (both encoders saturate at low bitrates — see the F12.0
// journal). AQ.5 needs RD curves with PSNR variation, so this harness
// can read real content from `tests/corpus/*.y4m` (when present) and
// falls back to a richer "textured-foreground" synthetic when the y4m
// path isn't viable.
// =========================================================================

/// Generate frame `idx` of a "textured-foreground over textured-background"
/// synthetic sequence. The background is a static 2D pseudo-noise field;
/// the foreground is a square that translates diagonally per frame, with
/// its own pseudo-noise modulation. Designed to produce non-degenerate
/// RD curves (real residual energy at every q-step) without needing
/// external corpus.
fn textured_foreground(w: usize, h: usize, idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];

    // Background: stable 2D hash giving ~uniform [80, 200] noise.
    // The encoder sees this as static (zero MV optimal).
    for r in 0..h {
        for c in 0..w {
            // Cheap, deterministic, decorrelated pseudo-random in [0, 255].
            let hash = (r.wrapping_mul(2654435761) ^ c.wrapping_mul(40503))
                .wrapping_mul(0x9E37_79B9);
            let v = 80 + ((hash >> 8) & 0x7F) as u8;
            y[r * w + c] = v;
        }
    }

    // Foreground square: translates by (3, 2) per frame, modulo (w, h),
    // and carries its own per-frame texture (so it's not pure ZEROMV).
    let fg_w = w / 4;
    let fg_h = h / 4;
    let fg_x = (idx as usize * 3) % w;
    let fg_y = (idx as usize * 2) % h;
    for r in 0..fg_h {
        let rr = (fg_y + r) % h;
        for c in 0..fg_w {
            let cc = (fg_x + c) % w;
            let hash = (rr.wrapping_mul(1664525) ^ cc.wrapping_mul(1013904223))
                .wrapping_add((idx as usize).wrapping_mul(31));
            let v = 60 + ((hash >> 4) & 0xFF) as u8 / 2;
            y[rr * w + cc] = v;
        }
    }

    // Chroma: gentle textured field, also non-trivial so encoder doesn't
    // hit a chroma-DC-only quality ceiling.
    let uvw = w / 2;
    let uvh = h / 2;
    let mut u = vec![128u8; uvw * uvh];
    let mut v = vec![128u8; uvw * uvh];
    for r in 0..uvh {
        for c in 0..uvw {
            let h2 = (r ^ (c << 1)).wrapping_mul(0xDEAD_BEEF) >> 12;
            // (h2 & 0x1F) is in 0..=31, so neither expression overflows.
            u[r * uvw + c] = 120u8.saturating_add((h2 & 0x1F) as u8);
            v[r * uvw + c] = 140u8.saturating_sub((h2 & 0x1F) as u8);
        }
    }

    (y, u, v)
}

/// Build a curve of RD points by sweeping `target_bitrates_bps` on the
/// rvp9 side at fixed `aq_mode`. Logs each point to stderr with the
/// supplied `label`.
fn rvp9_curve_sweep(
    label: &str,
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrates_bps: &[u32],
    aq_mode: AqMode,
    frames: &[YuvFrameBuf],
) -> Vec<(f64, f64)> {
    let mut curve = Vec::with_capacity(target_bitrates_bps.len());
    for &tb in target_bitrates_bps {
        let p = measure_rvp9_point(w, h, framerate, tb, aq_mode, frames);
        eprintln!(
            "  {label}  target={:>5} kbps  →  actual={:>7.1} kbps,  PSNR={:>6.2} dB",
            tb / 1000,
            p.bitrate_kbps,
            p.psnr_db,
        );
        curve.push((p.bitrate_kbps, p.psnr_db));
    }
    curve
}

/// Build a curve of RD points by sweeping `target_bitrates_bps` on the
/// libvpx side at fixed `aq_mode_int`. Returns `None` if any point
/// can't be measured (curve too short for BD-rate).
#[allow(clippy::too_many_arguments)] // Test helper; args mirror vpxenc CLI.
fn libvpx_curve_sweep(
    label: &str,
    vpxenc: &Path,
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrates_bps: &[u32],
    aq_mode_int: u32,
    frames: &[YuvFrameBuf],
) -> Option<Vec<(f64, f64)>> {
    let mut curve = Vec::new();
    for &tb in target_bitrates_bps {
        if let Some(p) = measure_libvpx_point(vpxenc, w, h, framerate, tb, aq_mode_int, frames) {
            eprintln!(
                "  {label}  target={:>5} kbps  →  actual={:>7.1} kbps,  PSNR={:>6.2} dB",
                tb / 1000,
                p.bitrate_kbps,
                p.psnr_db,
            );
            curve.push((p.bitrate_kbps, p.psnr_db));
        }
    }
    if curve.len() >= 2 {
        Some(curve)
    } else {
        None
    }
}

/// Stage 4b AQ-comparison gate body, parameterised over source frames.
///
/// Pulled out so both [`bdrate_gate_stage4b_aq`] (testsrc) and
/// [`bdrate_gate_stage4b_aq_realcontent`] (foreman_cif) share the same
/// 4-curve measurement, BD-rate computation, and reporting code path.
fn run_stage4b_aq_gate(
    source_label: &str,
    frames: &[YuvFrameBuf],
    framerate: f64,
    target_bitrates_bps: &[u32],
    w: u32,
    h: u32,
) {
    let n_frames = frames.len();
    eprintln!("Stage 4b Exit Gate — AQ.5 Measurement Report");
    eprintln!("=============================================");
    eprintln!("Source: {source_label}");
    eprintln!("Frames: {n_frames}    Target bitrates: {target_bitrates_bps:?} bps");
    eprintln!();

    // -------------------------------------------------------------
    // Curves 1 & 2 — rvp9 AQ off vs cyclic refresh.
    // -------------------------------------------------------------
    eprintln!("Curve 1: rvp9 AqMode::Off");
    let rvp9_off = rvp9_curve_sweep(
        "rvp9-off ",
        w,
        h,
        framerate,
        target_bitrates_bps,
        AqMode::Off,
        frames,
    );

    eprintln!("Curve 2: rvp9 AqMode::CyclicRefresh");
    let rvp9_aq = rvp9_curve_sweep(
        "rvp9-aq  ",
        w,
        h,
        framerate,
        target_bitrates_bps,
        AqMode::CyclicRefresh,
        frames,
    );

    // -------------------------------------------------------------
    // Curves 3 & 4 — libvpx aq-mode 0 vs 3 (optional).
    // -------------------------------------------------------------
    let vpxenc_opt = libvpx_stock_available();
    let (libvpx_off, libvpx_aq) = if let Some(vpxenc) = vpxenc_opt {
        eprintln!();
        eprintln!("libvpx-stock available at: {}", vpxenc.display());
        eprintln!("Curve 3: libvpx --aq-mode=0");
        let off = libvpx_curve_sweep(
            "libvpx-off",
            &vpxenc,
            w,
            h,
            framerate,
            target_bitrates_bps,
            0,
            frames,
        );
        eprintln!("Curve 4: libvpx --aq-mode=3");
        let aq = libvpx_curve_sweep(
            "libvpx-aq ",
            &vpxenc,
            w,
            h,
            framerate,
            target_bitrates_bps,
            3,
            frames,
        );
        (off, aq)
    } else {
        eprintln!();
        eprintln!("libvpx-stock not available — Curves 3/4 skipped.");
        (None, None)
    };

    // -------------------------------------------------------------
    // BD-rate gains.
    // -------------------------------------------------------------
    eprintln!();
    eprintln!("Curve summaries:");
    eprintln!("  rvp9_off  = {rvp9_off:?}");
    eprintln!("  rvp9_aq   = {rvp9_aq:?}");
    if let Some(ref c) = libvpx_off {
        eprintln!("  libvpx_off= {c:?}");
    }
    if let Some(ref c) = libvpx_aq {
        eprintln!("  libvpx_aq = {c:?}");
    }

    // bdrate(A, B): positive means B uses MORE bits than A at the same
    // PSNR; negative means B saves bits. We want "AQ-on saves bits vs
    // AQ-off" → bdrate(off, on) should be negative for a working AQ.
    let gain_rvp9 = bdrate(&rvp9_off, &rvp9_aq);
    let gain_libvpx = match (libvpx_off.as_ref(), libvpx_aq.as_ref()) {
        (Some(off), Some(aq)) => bdrate(off, aq),
        _ => f64::NAN,
    };

    eprintln!();
    eprintln!("AQ gain (negative = AQ saves bits at matched PSNR):");
    if gain_rvp9.is_nan() {
        eprintln!("  rvp9   AQ gain : NOT MEASURABLE (curves degenerate / no PSNR overlap)");
    } else {
        eprintln!("  rvp9   AQ gain : {gain_rvp9:+.3} %");
    }
    if gain_libvpx.is_nan() {
        eprintln!("  libvpx AQ gain : NOT MEASURABLE (libvpx missing or curves degenerate)");
    } else {
        eprintln!("  libvpx AQ gain : {gain_libvpx:+.3} %");
    }

    eprintln!();
    if gain_rvp9.is_finite() && gain_libvpx.is_finite() {
        let delta = gain_rvp9 - gain_libvpx;
        eprintln!(
            "Delta (gain_rvp9 - gain_libvpx) : {delta:+.3} %"
        );
        let abs_delta = delta.abs();
        let pass = abs_delta <= 0.3;
        eprintln!(
            "Stage 4b literal gate (|delta| <= 0.3 %): {}",
            if pass { "PASS" } else { "FAIL" }
        );
        eprintln!(
            "Stage 4b literal gate (|delta| <= 2.0 %, informational): {}",
            if abs_delta <= 2.0 { "PASS" } else { "FAIL" }
        );
    } else {
        eprintln!("Delta : NOT MEASURABLE");
        eprintln!("Stage 4b literal gate (|delta| <= 0.3 %): NOT MEASURABLE");
        eprintln!(
            "Reason: at least one side's BD-rate is NaN. Common cause is degenerate \
             RD curves on a synthetic source (both encoders saturate at the same PSNR \
             regardless of target bitrate). See \
             `wrk_journals/2026.05.21 - JRN - AQ.5-stage4b-exit-gate.md` for analysis."
        );
    }

    // Coherence reporting (non-fatal). The rvp9 RD curves on this
    // source are EXPECTED to be near-flat or weakly inverted: the
    // F9.0 rate controller can't push the q-index high enough to
    // consume the targeted bitrate (the actual bitrate plateaus
    // well below the target). The same pattern was documented in
    // F12.0's journal (rvp9 saturated at ~35 dB / 20 kbps regardless
    // of target bitrate on the synthetic gradient). The test does
    // NOT panic on non-monotonicity; the measurement infrastructure
    // is the deliverable, and the honest verdict goes to the journal.
    let mut max_inversion = 0.0_f64;
    for c in [&rvp9_off, &rvp9_aq] {
        for w in c.windows(2) {
            let (_b0, ps0) = w[0];
            let (_b1, ps1) = w[1];
            let drop = (ps0 - ps1).max(0.0);
            if drop > max_inversion {
                max_inversion = drop;
            }
        }
    }
    eprintln!();
    eprintln!(
        "Coherence report — max non-monotonic PSNR drop across the two rvp9 \
         curves: {max_inversion:.2} dB"
    );
    if max_inversion > 0.5 {
        eprintln!(
            "  NOTE: an inversion >0.5 dB indicates the rate controller is \
             saturating below the target bitrate (the actual bitrate plateau \
             documented in F12.0). BD-rate numbers above are reported but \
             carry low confidence."
        );
    }
}

/// AQ.5 — Stage 4b exit-gate harness (testsrc 720p crop).
///
/// Builds four RD curves:
///
/// 1. rvp9 + [`AqMode::Off`]
/// 2. rvp9 + [`AqMode::CyclicRefresh`]
/// 3. libvpx `--aq-mode=0`
/// 4. libvpx `--aq-mode=3`
///
/// Computes the AQ-on gain on each side (`bdrate(curve_on, curve_off)`)
/// and the gain delta. Reports honestly: PASS only if the delta is
/// finite and ≤ 0.3 %; otherwise NOT MEASURABLE with the reason.
///
/// testsrc saturates at low bitrates regardless of AQ — see the
/// companion [`bdrate_gate_stage4b_aq_realcontent`] test for the same
/// 4-curve comparison on a real-motion clip.
#[test]
#[ignore = "AQ.5 — Stage 4b exit-gate harness; run with `--ignored`"]
fn bdrate_gate_stage4b_aq() {
    // Operating point:
    //
    // - 256×256 frames are 4× the F12.0 area (still tractable for the
    //   v1 encoder, ~1-2s per frame at high bitrate in debug, ~100ms
    //   in release). 256 is a 64-multiple.
    // - 64 frames per run: 1 KF + 63 inter. RC.6 raised this from 8 so
    //   the `avg_frame_qindex_inter` EWMA can settle from the libvpx-
    //   faithful worst_quality=255 init, and cyclic refresh has many
    //   passes to cycle segments (default period is 4 frames → 16 full
    //   refresh cycles in this window).
    // - RC.X1 dropped the encoder's "natural" bitrate further. Targets
    //   ≥ 200 kbps now saturate at the same PSNR (61.17 dB) — the cubic
    //   fit becomes ill-conditioned on flat curves. Use the same low-end
    //   range as F12 to expose meaningful R-D slope.
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;
    const FRAMERATE_HINT: f64 = 30.0;
    let target_bitrates_bps: [u32; 4] = [25_000, 50_000, 100_000, 250_000];

    let probed = probe_testsrc_720p();
    let (frames, framerate, source_label) = load_y4m_or_synthetic(
        probed.as_deref(),
        W,
        H,
        N_FRAMES,
        FRAMERATE_HINT,
        "synthetic textured-foreground",
        textured_foreground,
    );

    run_stage4b_aq_gate(
        &source_label,
        &frames,
        framerate,
        &target_bitrates_bps,
        W,
        H,
    );
}

/// AQ.5 — Stage 4b exit-gate harness on real-motion content (foreman_cif).
///
/// Companion to [`bdrate_gate_stage4b_aq`]: same 4-curve cyclic-refresh
/// comparison, but running on the centre-crop of foreman_cif (real
/// camera motion, face/hair/background texture). testsrc plateaus at
/// the same PSNR regardless of AQ on/off because the content has no
/// "boring" regions for cyclic refresh to advantage; foreman exposes a
/// real working set for the AQ heuristic.
///
/// Source: see [`quality_gate_stage4a_realcontent`] for acquisition
/// notes. Skips early if the y4m is missing.
#[test]
#[ignore = "AQ.5 — Stage 4b real-content gate; run with `--ignored`"]
fn bdrate_gate_stage4b_aq_realcontent() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;

    let Some(path) = probe_foreman_cif() else {
        eprintln!(
            "    [skip] foreman_cif.y4m not found; place it at tests/corpus/foreman_cif.y4m"
        );
        eprintln!(
            "    See tests/corpus/sources/README.md for acquisition steps."
        );
        return;
    };

    let (frames, framerate) = match y4m_read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "    [skip] foreman_cif only yielded {}/{} frames",
                f.len(),
                N_FRAMES
            );
            return;
        }
        Err(e) => {
            eprintln!("    [skip] foreman_cif read failed: {e}");
            return;
        }
    };

    let target_bitrates_bps: [u32; 4] = [25_000, 50_000, 100_000, 250_000];
    let source_label = format!(
        "foreman_cif y4m centre-crop from {} → {W}x{H} @ {framerate} fps",
        path.display()
    );
    run_stage4b_aq_gate(
        &source_label,
        &frames,
        framerate,
        &target_bitrates_bps,
        W,
        H,
    );
}
