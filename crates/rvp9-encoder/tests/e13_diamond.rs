//! E13 — diamond-search-quality oracle harness. Drives foreman_cif
//! (256x256 crop, 64 frames) at a fixed Q and accumulates the
//! `RVP9_E13_DIAMOND` per-leaf SAD oracle so we can see how much of the
//! reachable full-pel gain the diamond captures, plus the NEWMV RD-win
//! rate (the shipped-prediction gate). Diagnostic-only; stderr IS the
//! output. Run:
//!   cargo test --release --test e13_diamond -- --ignored --nocapture
#![allow(clippy::print_stderr)]

use rvp9_core::mv_search::e12stat;
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, StreamFrameKind, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped, YuvFrameBuf};

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 64;

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

/// Convert an average SAD-per-pixel to an equivalent PSNR proxy. SAD is
/// L1; we use the same |.|->dB mapping the brief reports against so the
/// numbers line up: PSNR ~ 20*log10(255 / mean_abs_diff).
fn sad_psnr(sad_sum: u64, leaves: u64, block_px: u64) -> f64 {
    if sad_sum == 0 || leaves == 0 {
        return 100.0;
    }
    let mean_abs = sad_sum as f64 / (leaves * block_px) as f64;
    if mean_abs <= 0.0 {
        return 100.0;
    }
    20.0 * (255.0 / mean_abs).log10()
}

#[test]
#[ignore = "E13 diamond oracle; run with --ignored --nocapture"]
fn e13_diamond_oracle_foreman() {
    let fixed_q: u8 = std::env::var("RVP9_FIXED_Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    unsafe {
        std::env::set_var("RVP9_FIXED_Q", fixed_q.to_string());
        std::env::set_var("RVP9_E13_DIAMOND", "1");
    }
    e12stat::init_from_env();

    let Some(path) = probe_foreman_cif() else {
        eprintln!("[E13 skip] foreman_cif.y4m not found");
        return;
    };
    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!("[E13 skip] foreman only yielded {}/{N_FRAMES}", f.len());
            return;
        }
        Err(e) => {
            eprintln!("[E13 skip] foreman read failed: {e}");
            return;
        }
    };

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 100_000,
        gop_length: N_FRAMES + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    let mut agg = e12stat::DiamondStats::default();
    for f in frames.iter() {
        let _ = e12stat::take_snapshot();
        let pkt = enc.encode_frame(&yuv_view(f, W as usize)).expect("encode_frame");
        let snap = e12stat::take_snapshot();
        if pkt.frame_kind == StreamFrameKind::Inter {
            agg.leaves += snap.leaves;
            agg.sad_zero += snap.sad_zero;
            agg.sad_newmv += snap.sad_newmv;
            agg.sad_oracle += snap.sad_oracle;
            agg.reached_oracle += snap.reached_oracle;
            agg.improved_on_zero += snap.improved_on_zero;
            agg.seeded_leaves += snap.seeded_leaves;
            agg.mv_mag_sum += snap.mv_mag_sum;
            agg.newmv_won += snap.newmv_won;
            agg.argmin_leaves += snap.argmin_leaves;
        }
    }
    unsafe {
        std::env::remove_var("RVP9_FIXED_Q");
    }

    eprintln!("E13 diamond-search-quality oracle — foreman 256x256, q={fixed_q}");
    eprintln!("  NEWMV leaves observed: {}", agg.leaves);
    if agg.leaves == 0 {
        eprintln!("  (no NEWMV leaves — picker never enters NEWMV search)");
        return;
    }
    // Per-leaf SAD averages (block sizes vary, so use the variable-block
    // proxy: SAD-per-pixel via the running sums isn't exact without
    // pixel counts, but the relative ratios are the load-bearing signal).
    let z = agg.sad_zero as f64 / agg.leaves as f64;
    let n = agg.sad_newmv as f64 / agg.leaves as f64;
    let o = agg.sad_oracle as f64 / agg.leaves as f64;
    eprintln!("  avg SAD/leaf: zero={z:.0}  newmv={n:.0}  oracle={o:.0}");
    let recover = if (z - o).abs() > 1.0 {
        (z - n) / (z - o)
    } else {
        f64::NAN
    };
    eprintln!(
        "  diamond-recovers-fraction (zero-newmv)/(zero-oracle) = {:.1}%  (1.0 = reaches oracle)",
        100.0 * recover
    );
    eprintln!(
        "  reached_oracle: {}/{} ({:.1}%)   improved_on_zero: {}/{} ({:.1}%)",
        agg.reached_oracle, agg.leaves,
        100.0 * agg.reached_oracle as f64 / agg.leaves as f64,
        agg.improved_on_zero, agg.leaves,
        100.0 * agg.improved_on_zero as f64 / agg.leaves as f64,
    );
    eprintln!(
        "  seeded(int_pro short-circuit): {}/{} ({:.1}%)   avg |mv| (full-pel L1) = {:.2}",
        agg.seeded_leaves, agg.leaves,
        100.0 * agg.seeded_leaves as f64 / agg.leaves as f64,
        agg.mv_mag_sum as f64 / agg.leaves as f64,
    );
    eprintln!(
        "  NEWMV won RD argmin: {}/{} inter leaves ({:.2}%)  <- shipped-prediction gate",
        agg.newmv_won, agg.argmin_leaves,
        if agg.argmin_leaves > 0 {
            100.0 * agg.newmv_won as f64 / agg.argmin_leaves as f64
        } else {
            0.0
        },
    );
    // Equivalent prediction-PSNR proxy assuming a 256-px (16x16) block;
    // foreman's variance partitioner is 8x8-dominated (64 px). Report
    // both anchors so the reader can map onto the E12 numbers.
    for (label, px) in [("64px(8x8)", 64u64), ("256px(16x16)", 256u64)] {
        eprintln!(
            "  SAD-PSNR proxy @{label}: zero={:.2}  newmv={:.2}  oracle={:.2} dB",
            sad_psnr(agg.sad_zero, agg.leaves, px),
            sad_psnr(agg.sad_newmv, agg.leaves, px),
            sad_psnr(agg.sad_oracle, agg.leaves, px),
        );
    }
}
