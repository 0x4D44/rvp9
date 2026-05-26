// E4 — overshoot bit-accounting harness. stderr is the output mechanism
// (this binary's job is to print a measurement report).
#![allow(clippy::print_stderr)]

//! E4 — pin the foreman bitrate-overshoot mechanism.
//!
//! Diagnostic-only. Drives foreman_cif the same way
//! `bdrate_gate::bdrate_gate_stage4a_realcontent` does (256×256 crop,
//! 64 frames, target 100 kbps), with the env-gated bit-accounting and
//! skip-stat scaffolds enabled, and produces:
//!
//! 1. A per-category bit breakdown averaged over inter frames
//!    (partition / skip / is-inter / tx_size / mode+ref / MV / coef /
//!    seg-id), from `rvp9_core::entropy::bitacct`.
//! 2. The rate-model prediction error per inter frame (RC predicted
//!    bits at the picked Q vs actual coded bits) and the Q ceiling the
//!    controller hits at the 100 kbps target, from the live
//!    `StreamEncoder::rate_control()` snapshot.
//! 3. Skip utilisation: fraction of inter leaves coded skip=1 vs
//!    skip=0, and the residual energy carried by the skip=0 leaves,
//!    from `rvp9_encoder::skipstat`.
//!
//! Run:
//! ```text
//! cargo test --test e4_overshoot_accounting -- --ignored --nocapture
//! ```

use rvp9_core::entropy::bitacct;
use rvp9_encoder::rate_control::FrameKind as RcFrameKind;
use rvp9_encoder::skipstat;
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, StreamFrameKind, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped, YuvFrameBuf};

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

const CAT_NAMES: [&str; bitacct::N_CATEGORIES] = [
    "partition", "skip", "is_inter", "tx_size", "mode+ref", "mv_delta", "coef", "seg_id", "other",
];

#[test]
#[ignore = "E4 — overshoot accounting harness; run with `--ignored --nocapture`"]
fn e4_overshoot_accounting_foreman_100k() {
    // Enable the diagnostic scaffolds for this process.
    unsafe {
        std::env::set_var("RVP9_E4_BITACCT", "1");
        std::env::set_var("RVP9_E4_SKIPSTAT", "1");
    }
    bitacct::init_from_env();
    skipstat::init_from_env();

    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;
    const TARGET_BPS: u32 = 100_000;

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

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: TARGET_BPS,
        gop_length: N_FRAMES + 1, // one keyframe at frame 0
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    eprintln!("E4 Overshoot Accounting — foreman_cif 256x256, {N_FRAMES} frames, target {} kbps", TARGET_BPS / 1000);
    eprintln!("source: {} @ {framerate} fps", path.display());
    eprintln!();

    // Per-inter-frame accumulators.
    let mut cat_sum = [0.0f64; bitacct::N_CATEGORIES];
    let mut inter_count = 0u64;
    let mut total_actual_inter_bits = 0u64;

    // Skip accumulators (inter frames).
    let mut tot_leaves = 0u64;
    let mut tot_skip = 0u64;
    let mut tot_nonskip = 0u64;
    let mut tot_nonskip_eob = 0u64;
    let mut tot_nonskip_tiny = 0u64;

    // RC prediction-error accumulators (inter frames).
    let mut ratio_sum = 0.0f64;
    let mut q_max_seen = 0u8;
    let mut q_sum = 0u64;

    eprintln!("Per-inter-frame RC trace:");
    eprintln!(" frame  q  q_ceil  target  predicted   actual   act/pred");

    for (i, f) in frames.iter().enumerate() {
        // Reset the thread-local snapshots so each frame's writes are
        // isolated. (encode_frame runs the tile writer on this thread.)
        let _ = bitacct::take_snapshot();
        let _ = skipstat::take_snapshot();

        let pkt = enc
            .encode_frame(&yuv_view(f, W as usize))
            .expect("encode_frame");

        let cats = bitacct::take_snapshot();
        let skip = skipstat::take_snapshot();

        let rc = enc.rate_control();
        // NOTE: after encode_frame, postencode_update has run, so the
        // correction factor reflects THIS frame's feedback. last_picked
        // / active_max / this_frame_target still reflect this frame's
        // pick (they are only overwritten by the NEXT pick_q_and_bounds).
        let q = rc.last_picked_qindex;
        let q_ceil = rc.active_max_qindex;
        let target = rc.this_frame_target;
        // predicted_bits_at_picked_q uses the *current* correction
        // factor (post-update). For the prediction-error finding we
        // want the predict-side estimate at the picked Q; the post-
        // update factor is the closest available read and is what the
        // feedback loop will carry into the next frame.
        let predicted = rc.predicted_bits_at_picked_q(RcFrameKind::Inter);

        if pkt.frame_kind == StreamFrameKind::Inter {
            for (acc, v) in cat_sum.iter_mut().zip(cats.iter()) {
                *acc += *v;
            }
            inter_count += 1;
            total_actual_inter_bits += u64::from(pkt.actual_bits);

            tot_leaves += skip.leaves;
            tot_skip += skip.skip_leaves;
            tot_nonskip += skip.nonskip_leaves;
            tot_nonskip_eob += skip.nonskip_eob_sum;
            tot_nonskip_tiny += skip.nonskip_tiny;

            let ratio = if predicted > 0 {
                f64::from(pkt.actual_bits) / predicted as f64
            } else {
                f64::NAN
            };
            ratio_sum += ratio;
            q_sum += u64::from(q);
            if q > q_max_seen {
                q_max_seen = q;
            }

            eprintln!(
                "  {i:>3}  {q:>3}  {q_ceil:>3}  {target:>6}  {predicted:>8}  {:>8}  {ratio:>6.3}",
                pkt.actual_bits
            );
        } else {
            eprintln!(
                "  {i:>3}  {q:>3}  {q_ceil:>3}  {target:>6}  {predicted:>8}  {:>8}   (KEY)",
                pkt.actual_bits
            );
        }
    }

    assert!(inter_count > 0, "expected at least one inter frame");
    let n = inter_count as f64;

    // -----------------------------------------------------------------
    // 1. Per-category bit breakdown (averaged over inter frames).
    // -----------------------------------------------------------------
    let total_cat_bits: f64 = cat_sum.iter().sum();
    let avg_actual = total_actual_inter_bits as f64 / n;
    eprintln!();
    eprintln!("=== 1. Per-category bit breakdown (inter frames) ===");
    eprintln!("  avg ACTUAL coded bits / inter frame : {avg_actual:.0}");
    eprintln!("  sum of category bit estimates / fr  : {:.0}  (entropy-cost model)", total_cat_bits / n);
    eprintln!();
    eprintln!("  category      bits/frame    % of coded");
    let mut order: Vec<usize> = (0..bitacct::N_CATEGORIES).collect();
    order.sort_by(|&a, &b| cat_sum[b].partial_cmp(&cat_sum[a]).unwrap());
    for idx in order {
        let bpf = cat_sum[idx] / n;
        let pct = if total_cat_bits > 0.0 {
            100.0 * cat_sum[idx] / total_cat_bits
        } else {
            0.0
        };
        eprintln!("  {:<12} {bpf:>10.1}    {pct:>8.2}%", CAT_NAMES[idx]);
    }

    // -----------------------------------------------------------------
    // 2. RC prediction error + Q ceiling.
    // -----------------------------------------------------------------
    eprintln!();
    eprintln!("=== 2. Rate-model prediction error + Q ceiling ===");
    eprintln!("  mean actual/predicted ratio (inter) : {:.3}", ratio_sum / n);
    eprintln!("  (>1 => RC under-predicts the cost => picks too-low a Q => overshoots)");
    eprintln!("  avg picked inter Q                  : {:.1}", q_sum as f64 / n);
    eprintln!("  max picked inter Q                  : {q_max_seen}");
    eprintln!("  config max_qindex (worst_quality)   : 255");
    eprintln!("  (if max picked Q < 255, the active_worst clamp is binding below the ceiling)");

    // -----------------------------------------------------------------
    // 3. Skip utilisation.
    // -----------------------------------------------------------------
    eprintln!();
    eprintln!("=== 3. Skip utilisation (inter leaves) ===");
    let skip_frac = if tot_leaves > 0 {
        100.0 * tot_skip as f64 / tot_leaves as f64
    } else {
        0.0
    };
    eprintln!("  total inter leaves          : {tot_leaves}");
    eprintln!("  skip=1 leaves               : {tot_skip}  ({skip_frac:.1}%)");
    eprintln!("  skip=0 leaves               : {tot_nonskip}");
    if tot_nonskip > 0 {
        eprintln!(
            "  mean total-eob / skip=0 leaf: {:.2}  (nonzero coefficients across the whole leaf)",
            tot_nonskip_eob as f64 / tot_nonskip as f64
        );
        eprintln!(
            "  skip=0 leaves with eob<=4   : {tot_nonskip_tiny}  ({:.1}% of skip=0 leaves)",
            100.0 * tot_nonskip_tiny as f64 / tot_nonskip as f64
        );
        eprintln!("  (a high tiny-eob fraction = blocks libvpx's RD skip decision would drop)");
    }
}
