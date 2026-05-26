// E9 — matched-Q quality-gap decomposition harness. stderr IS the
// output mechanism (this binary's job is to print a measurement report).
#![allow(clippy::print_stderr)]

//! E9 — decompose the matched-bitrate coding-quality gap.
//!
//! Diagnostic-only. Drives foreman_cif (256×256 crop, 64 frames) through
//! BOTH encoders at a **fixed quantizer** (no rate control), at a few Q
//! points, so PSNR/bit differences isolate prediction + transform +
//! entropy from rate control. For each Q it captures, on the rvp9 side:
//!
//! - per-frame **prediction SSE** (Y plane, source − prediction BEFORE
//!   residual) via `rvp9_encoder::predstat`,
//! - per-frame **reconstruction SSE / PSNR** (decode + compare to source),
//! - per-category **bits** via `rvp9_core::entropy::bitacct`,
//! - **transform-size histogram** (Y plane).
//!
//! libvpx is driven at the matched internal qindex via `--end-usage=q
//! --min-q=N --max-q=N` (where the CLI quantizer N maps to internal
//! qindex ≈ 4·N per `quantizer_to_qindex[]`), with `--psnr` parsed for
//! the average PSNR and IVF size for bits.
//!
//! Run:
//! ```text
//! cargo test --release --test e9_quality_gap -- --ignored --nocapture
//! ```

use rvp9_core::entropy::bitacct;
use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{
    predstat, skipstat, AqMode, EncoderConfig, StreamEncoder, StreamFrameKind, YuvFrameRef,
};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped, YuvFrameBuf};
use std::path::{Path, PathBuf};
use std::process::Command;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 64;

const CAT_NAMES: [&str; bitacct::N_CATEGORIES] = [
    "partition", "skip", "is_inter", "tx_size", "mode+ref", "mv_delta", "coef", "seg_id", "other",
];

/// The rvp9 base_qindex points to sweep, in full VP9 0..255 qindex space.
/// 80 / 120 / 160 map exactly onto libvpx CLI quantizers 20 / 30 / 40
/// (`quantizer_to_qindex[N] == 4·N` over this range).
const Q_POINTS: [u8; 3] = [80, 120, 160];

fn yuv_view<'a>(f: &'a YuvFrameBuf, w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y: &f.0,
        u: &f.1,
        v: &f.2,
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

/// Y-plane SSE between two equal-length buffers.
fn sse_u8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i64::from(*x) - i64::from(*y);
        s += (d * d) as u64;
    }
    s
}

/// Convert a Y-plane SSE over `pixels` samples to PSNR (dB).
fn psnr_from_sse(sse: u64, pixels: u64) -> f64 {
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / pixels as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// One rvp9 measurement at a fixed Q.
struct Rvp9Result {
    /// Reconstruction PSNR over inter frames (Y), dB.
    inter_recon_psnr: f64,
    /// Reconstruction PSNR over ALL frames (Y), dB.
    all_recon_psnr: f64,
    /// Equivalent prediction PSNR over inter frames (from predstat SSE).
    inter_pred_psnr: f64,
    /// Equivalent prediction PSNR over the keyframe (from predstat SSE).
    kf_pred_psnr: f64,
    /// Average coded bits per inter frame.
    inter_bits: f64,
    /// Average total coded bits per frame (all frames).
    all_bits: f64,
    /// Coefficient-token bits per inter frame (bitacct).
    inter_coef_bits: f64,
    /// Y-plane tx-size histogram over inter frames: [4x4, 8x8, 16x16, 32x32].
    inter_tx_hist: [u64; 4],
    /// Y-plane tx-size histogram over the keyframe.
    kf_tx_hist: [u64; 4],
    /// Sum of nonzero coef tokens (eob) over non-skip inter leaves.
    inter_eob_sum: u64,
    /// Skip fraction over inter leaves.
    inter_skip_frac: f64,
}

fn measure_rvp9_fixed_q(fixed_q: u8, frames: &[YuvFrameBuf], framerate: f64) -> Rvp9Result {
    // Enable the diagnostic scaffolds and pin Q for this process.
    unsafe {
        std::env::set_var("RVP9_FIXED_Q", fixed_q.to_string());
        std::env::set_var("RVP9_E4_BITACCT", "1");
        std::env::set_var("RVP9_E4_SKIPSTAT", "1");
        std::env::set_var("RVP9_E9_PREDSTAT", "1");
    }
    bitacct::init_from_env();
    skipstat::init_from_env();
    predstat::init_from_env();

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 100_000, // unused: RVP9_FIXED_Q overrides Q
        gop_length: N_FRAMES + 1,    // one keyframe at frame 0
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    let y_pixels = (W * H) as u64;

    let mut inter_recon_sse = 0u64;
    let mut all_recon_sse = 0u64;
    let mut inter_frames = 0u64;
    let mut inter_pred_sse = 0u64;
    let mut inter_pred_px = 0u64;
    let mut kf_pred_sse = 0u64;
    let mut kf_pred_px = 0u64;
    let mut inter_bits_sum = 0u64;
    let mut all_bits_sum = 0u64;
    let mut inter_coef_bits = 0.0f64;
    let mut inter_tx_hist = [0u64; 4];
    let mut kf_tx_hist = [0u64; 4];
    let mut inter_eob_sum = 0u64;
    let mut inter_leaves = 0u64;
    let mut inter_skip = 0u64;

    for f in frames.iter() {
        let _ = bitacct::take_snapshot();
        let _ = skipstat::take_snapshot();
        let _ = predstat::take_snapshot();

        let pkt = enc
            .encode_frame(&yuv_view(f, W as usize))
            .expect("encode_frame");

        let cats = bitacct::take_snapshot();
        let skip = skipstat::take_snapshot();
        let pred = predstat::take_snapshot();

        // Decode + recon PSNR.
        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let yp = buf.y_plane();
        let mut decoded_y = vec![0u8; y_pixels as usize];
        for r in 0..H as usize {
            decoded_y[r * W as usize..(r + 1) * W as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
        }
        let recon_sse = sse_u8(&f.0, &decoded_y);

        all_recon_sse += recon_sse;
        all_bits_sum += u64::from(pkt.actual_bits);

        if pkt.frame_kind == StreamFrameKind::Inter {
            inter_frames += 1;
            inter_recon_sse += recon_sse;
            inter_pred_sse += pred.y_pred_sse;
            inter_pred_px += pred.y_pixels;
            for i in 0..4 {
                inter_tx_hist[i] += pred.txsize_hist[i];
            }
            inter_bits_sum += u64::from(pkt.actual_bits);
            inter_coef_bits += cats[bitacct::BitCategory::Coef as usize];
            inter_eob_sum += skip.nonskip_eob_sum;
            inter_leaves += skip.leaves;
            inter_skip += skip.skip_leaves;
        } else {
            kf_pred_sse += pred.y_pred_sse;
            kf_pred_px += pred.y_pixels;
            for i in 0..4 {
                kf_tx_hist[i] += pred.txsize_hist[i];
            }
        }
    }

    // Clear the override so it doesn't leak into other tests in-process.
    unsafe {
        std::env::remove_var("RVP9_FIXED_Q");
    }

    let n_inter = inter_frames.max(1) as f64;
    let n_all = frames.len() as f64;
    Rvp9Result {
        inter_recon_psnr: psnr_from_sse(inter_recon_sse, inter_frames * y_pixels),
        all_recon_psnr: psnr_from_sse(all_recon_sse, frames.len() as u64 * y_pixels),
        inter_pred_psnr: psnr_from_sse(inter_pred_sse, inter_pred_px.max(1)),
        kf_pred_psnr: psnr_from_sse(kf_pred_sse, kf_pred_px.max(1)),
        inter_bits: inter_bits_sum as f64 / n_inter,
        all_bits: all_bits_sum as f64 / n_all,
        inter_coef_bits: inter_coef_bits / n_inter,
        inter_tx_hist,
        kf_tx_hist,
        inter_eob_sum,
        inter_skip_frac: if inter_leaves > 0 {
            inter_skip as f64 / inter_leaves as f64
        } else {
            0.0
        },
    }
}

/// libvpx fixed-Q result: (avg PSNR dB over decoded frames, avg bits/frame).
struct LibvpxResult {
    psnr_db: f64,
    bits_per_frame: f64,
}

fn libvpx_available() -> Option<PathBuf> {
    let p = PathBuf::from(r"D:\language\libvpx-stock\vpxenc.exe");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

fn write_i420_raw(path: &Path, frames: &[YuvFrameBuf]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for (y, u, v) in frames {
        f.write_all(y)?;
        f.write_all(u)?;
        f.write_all(v)?;
    }
    Ok(())
}

/// Drive libvpx at fixed internal qindex `qindex` (mapped to CLI
/// quantizer `qindex/4`). Returns avg Y-PSNR + avg bits/frame.
fn measure_libvpx_fixed_q(
    vpxenc: &Path,
    qindex: u8,
    framerate: f64,
    frames: &[YuvFrameBuf],
) -> Option<LibvpxResult> {
    let cli_q = (qindex / 4).clamp(0, 63);
    let tmp = std::env::temp_dir();
    let raw = tmp.join(format!("e9_{W}x{H}_q{qindex}.yuv"));
    let ivf = tmp.join(format!("e9_{W}x{H}_q{qindex}.ivf"));
    let dec_raw = tmp.join(format!("e9_{W}x{H}_q{qindex}_dec.yuv"));

    if write_i420_raw(&raw, frames).is_err() {
        return None;
    }

    let out = Command::new(vpxenc)
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--passes=1",
            "--end-usage=q",
            "--i420",
            "--aq-mode=0",
            "--lag-in-frames=0",
        ])
        .arg(format!("--width={W}"))
        .arg(format!("--height={H}"))
        .arg(format!("--fps={}/1", framerate as u32))
        // libvpx --end-usage=q codes every frame at --cq-level with no
        // rate control. (`--min-q == --max-q` is rejected with "should
        // differ by at least 8"; cq-level alone is the fixed-Q knob.)
        .arg(format!("--cq-level={cli_q}"))
        // Force a single keyframe at frame 0 (match rvp9's GOP).
        .arg(format!("--kf-min-dist={}", N_FRAMES))
        .arg(format!("--kf-max-dist={}", N_FRAMES))
        .arg("-o")
        .arg(&ivf)
        .arg(&raw)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxenc failed q={qindex}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    let ivf_bytes = std::fs::metadata(&ivf).ok()?.len();
    let bits_per_frame = ivf_bytes as f64 * 8.0 / frames.len() as f64;

    // Decode with vpxdec and compute our own Y-PSNR for an apples-to-
    // apples comparison with the rvp9 leg.
    let vpxdec = vpxenc.with_file_name("vpxdec.exe");
    let out = Command::new(&vpxdec)
        .args(["--i420", "-o"])
        .arg(&dec_raw)
        .arg(&ivf)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxdec failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let dec_data = std::fs::read(&dec_raw).ok()?;
    let frame_size = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    let y_size = (W * H) as usize;
    if dec_data.len() < frame_size * frames.len() {
        return None;
    }
    let mut sse_total = 0u64;
    for (i, (orig_y, _, _)) in frames.iter().enumerate() {
        let dec_y = &dec_data[i * frame_size..i * frame_size + y_size];
        sse_total += sse_u8(orig_y, dec_y);
    }
    let psnr = psnr_from_sse(sse_total, frames.len() as u64 * y_size as u64);

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&dec_raw);

    Some(LibvpxResult {
        psnr_db: psnr,
        bits_per_frame,
    })
}

#[test]
#[ignore = "E9 — matched-Q quality-gap harness; run with `--ignored --nocapture`"]
fn e9_quality_gap_foreman() {
    let Some(path) = probe_foreman_cif() else {
        eprintln!("    [skip] foreman_cif.y4m not found at tests/corpus/foreman_cif.y4m");
        return;
    };
    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!("    [skip] foreman_cif only yielded {}/{} frames", f.len(), N_FRAMES);
            return;
        }
        Err(e) => {
            eprintln!("    [skip] foreman_cif read failed: {e}");
            return;
        }
    };

    eprintln!("E9 Matched-Q Quality-Gap Decomposition — foreman_cif 256x256, {N_FRAMES} frames");
    eprintln!("source: {} @ {framerate} fps", path.display());
    eprintln!("==================================================================");

    let vpxenc = libvpx_available();

    eprintln!();
    eprintln!("=== Matched-Q PSNR + bits table (per-frame averages) ===");
    eprintln!(
        "  Q(idx) | side   | recon_PSNR(all) | recon_PSNR(inter) | pred_PSNR(inter) | bits/frame(all) | bits/frame(inter)"
    );

    for &q in Q_POINTS.iter() {
        let r = measure_rvp9_fixed_q(q, &frames, framerate);
        eprintln!(
            "  {q:>3}    | rvp9   |     {:>7.2}     |      {:>7.2}      |     {:>7.2}      |     {:>8.0}    |     {:>8.0}",
            r.all_recon_psnr, r.inter_recon_psnr, r.inter_pred_psnr, r.all_bits, r.inter_bits
        );

        if let Some(ref vpxenc) = vpxenc {
            if let Some(lv) = measure_libvpx_fixed_q(vpxenc, q, framerate, &frames) {
                eprintln!(
                    "  {q:>3}    | libvpx |     {:>7.2}     |        --        |       --         |     {:>8.0}    |        --",
                    lv.psnr_db, lv.bits_per_frame
                );
                let dpsnr = lv.psnr_db - r.all_recon_psnr;
                let dbits = lv.bits_per_frame - r.all_bits;
                eprintln!(
                    "         | DELTA  |  libvpx-rvp9 PSNR = {dpsnr:+.2} dB   |   libvpx-rvp9 bits = {dbits:+.0} b/frame"
                );
            }
        }

        // Per-Q rvp9 detail block.
        eprintln!(
            "         detail: inter coef bits/frame = {:.0}  |  inter skip frac = {:.1}%  |  inter eob_sum = {}",
            r.inter_coef_bits,
            100.0 * r.inter_skip_frac,
            r.inter_eob_sum
        );
        let bits_per_eob = if r.inter_eob_sum > 0 {
            r.inter_coef_bits * (inter_count_estimate(&frames) as f64) / r.inter_eob_sum as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "         detail: coef bits / nonzero-coef (entropy efficiency proxy) = {bits_per_eob:.3} bits/coef"
        );
        eprintln!(
            "         detail: inter Y tx-size hist [4x4,8x8,16x16,32x32] = {:?}",
            r.inter_tx_hist
        );
        eprintln!(
            "         detail: KF Y tx-size hist    [4x4,8x8,16x16,32x32] = {:?}  |  KF pred_PSNR = {:.2} dB",
            r.kf_tx_hist, r.kf_pred_psnr
        );
        eprintln!();
    }

    eprintln!("Categories tapped by bitacct: {CAT_NAMES:?}");
    eprintln!("Note: rvp9 KF is pinned to the same fixed Q as inter (no KF qp boost),");
    eprintln!("      whereas libvpx may apply a KF delta even in --end-usage=q. The");
    eprintln!("      inter-frame columns are the cleaner matched-Q comparison.");
}

/// Estimate the inter-frame count (frames - 1 keyframe) for the
/// bits/coef proxy. Avoids re-plumbing the count out of the measurement.
fn inter_count_estimate(frames: &[YuvFrameBuf]) -> usize {
    frames.len().saturating_sub(1)
}
