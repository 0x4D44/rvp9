//! E12 — the decisive search-failure-vs-prediction-ceiling oracle
//! (diagnostic only, env-gated `RVP9_E12_ORACLE`).
//!
//! For sampled inter leaves on a chosen frame this module computes the
//! *prediction* PSNR (source vs motion-compensated prediction, BEFORE
//! any transform / quant — the pure MC quality) for three things:
//!
//!   (a) the zero MV `(0,0)` — what NEARESTMV-at-(0,0) gives;
//!   (b) the MV rvp9's real NEWMV diamond+sub-pel search returned;
//!   (c) an EXHAUSTIVE oracle: a dense full-pel grid (±`FULL_RANGE`)
//!       plus the 8 half-pel and quarter-pel neighbours of the best
//!       full-pel point, picking the MV that minimises prediction SSE.
//!
//! The verdict:
//!   * oracle-best ≫ rvp9-found  →  SEARCH FAILURE (the good MV exists,
//!     the search doesn't converge to it). The dump records whether the
//!     oracle-best MV is sub-pel.
//!   * oracle-best ≈ rvp9-found ≈ 26 dB  →  PREDICTION CEILING (no MV
//!     helps — reference cascade or needs-intra).
//!
//! The prediction is built with the SAME `inter_predictor` (identity
//! scale, EightTap) the real encoder reconstruction uses, so the
//! interpolation is byte-identical to the encoder's MC path — this also
//! doubles as the sub-pel-interpolation faithfulness check.
//!
//! Per-frame aggregation mirrors `predstat`: a thread-local accumulator
//! the harness snapshots once per encoded frame. No coding logic changes.

#![allow(clippy::print_stderr)]

use core::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::inter::filter::{SUBPEL_BITS, SUBPEL_MASK};
use rvp9_core::inter::scale::{ScaleFactors, inter_predictor};
use rvp9_core::mode_info::Mv;

static E12_ON: AtomicBool = AtomicBool::new(false);

/// Full-pel search half-range for the exhaustive oracle (`±FULL_RANGE`).
const FULL_RANGE: i32 = 16;
/// Cap the number of sampled leaves per frame so an exhaustive O(R^2)
/// grid stays cheap. Leaves are sampled uniformly via a stride.
const SAMPLE_STRIDE: usize = 7;
/// How many per-leaf rows to print verbatim (over the whole run).
const PRINT_LIMIT: usize = 24;

/// Per-frame aggregate of the oracle measurement (Y plane only).
#[derive(Copy, Clone, Debug, Default)]
pub struct E12Stats {
    /// Number of sampled leaves contributing to the sums below.
    pub n: u64,
    /// Σ prediction SSE at MV=(0,0).
    pub sse_zero: u64,
    /// Σ prediction SSE at the MV rvp9 actually SHIPS (the RD argmin
    /// winner — typically ZEROMV-at-(0,0) on foreman).
    pub sse_rvp9: u64,
    /// Σ prediction SSE at NEWMV's RAW search output (before the RD
    /// argmin rejects it). Discriminates search non-convergence from
    /// RD/cost-model rejection: if this ≈ oracle, the search converges
    /// fine and the RD picker is the culprit; if this ≈ shipped, the
    /// search itself fails.
    pub sse_newmv: u64,
    /// Σ prediction SSE at the best FULL-PEL-only oracle MV (phase 1
    /// only — no sub-pel refinement). Isolates whether the full-pel
    /// diamond stage or the sub-pel stage is the failing one.
    pub sse_oracle_fp: u64,
    /// Σ prediction SSE at the exhaustive oracle-best MV (full + sub-pel).
    pub sse_oracle: u64,
    /// Σ pixels covered (so PSNR can be computed area-weighted).
    pub px: u64,
    /// Of the sampled leaves, how many had an oracle-best MV that is
    /// sub-pel (i.e. `mv & 7 != 0` on either axis).
    pub oracle_subpel: u64,
    /// Of the sampled leaves, how many had the oracle-best ≥ 2 dB
    /// better than rvp9-found (a meaningful search-failure margin).
    pub oracle_beats_rvp9: u64,
}

thread_local! {
    static STATS: RefCell<E12Stats> = const { RefCell::new(E12Stats {
        n: 0, sse_zero: 0, sse_rvp9: 0, sse_newmv: 0, sse_oracle_fp: 0,
        sse_oracle: 0, px: 0, oracle_subpel: 0, oracle_beats_rvp9: 0,
    }) };
    static LEAF_SEQ: RefCell<usize> = const { RefCell::new(0) };
    static PRINTED: RefCell<usize> = const { RefCell::new(0) };
}

/// Enable the oracle if `RVP9_E12_ORACLE` is set and non-zero.
pub fn init_from_env() {
    let on = std::env::var_os("RVP9_E12_ORACLE").is_some_and(|v| v != "0");
    E12_ON.store(on, Ordering::Relaxed);
}

/// Whether the oracle is active.
#[inline]
pub fn enabled() -> bool {
    E12_ON.load(Ordering::Relaxed)
}

/// Take and reset the per-frame stats for the current thread.
pub fn take_snapshot() -> E12Stats {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        let snap = *s;
        *s = E12Stats::default();
        snap
    })
}

/// Prediction SSE for one `bw×bh` block at plane offset `(px, py)`,
/// for an MV in Q3 (1/8-pel) units, against the source plane. Returns
/// `None` if the 8-tap convolve read window would fall outside the
/// (un-padded) reference plane — the encoder's MV-limits machinery is
/// the border-extend equivalent, so OOB candidates are simply not
/// reachable and are skipped by the oracle.
#[allow(clippy::too_many_arguments)] // diagnostic helper; plane geometry is explicit
fn pred_sse(
    src: &[u8],
    src_stride: usize,
    refp: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    px: usize,
    py: usize,
    bw: usize,
    bh: usize,
    mv: Mv,
) -> Option<u64> {
    let sf = ScaleFactors::new(ref_w as u32, ref_h as u32, ref_w as u32, ref_h as u32);
    let total_x_q4 = sf.scaled_x_q4(px as i32, mv.col);
    let total_y_q4 = sf.scaled_y_q4(py as i32, mv.row);
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

    // 8-tap halo: 3 left/top, +4 (TAPS-1) extent when sub-pel.
    let row_pre = if subpel_y_q4 != 0 { -3i32 } else { 0 };
    let col_pre = if subpel_x_q4 != 0 { -3i32 } else { 0 };
    let win_y = int_y + row_pre;
    let win_x = int_x + col_pre;
    let win_h = bh as i32 + if subpel_y_q4 != 0 { 7 } else { 0 };
    let win_w = bw as i32 + if subpel_x_q4 != 0 { 7 } else { 0 };
    if win_x < 0
        || win_y < 0
        || win_x + win_w > ref_w as i32
        || win_y + win_h > ref_h as i32
    {
        return None;
    }

    let src_base = (win_y as usize) * ref_stride + (win_x as usize);
    let mut pred = [0u8; 64 * 64];
    inter_predictor(
        &refp[src_base..],
        ref_stride,
        &mut pred,
        bw, // dst_stride
        subpel_x_q4,
        subpel_y_q4,
        &sf,
        InterpFilter::EightTap,
        bw,
        bh,
        false,
    );

    let mut sse = 0u64;
    for r in 0..bh {
        let srow = (py + r) * src_stride + px;
        for c in 0..bw {
            let s = i64::from(src[srow + c]);
            let p = i64::from(pred[r * bw + c]);
            let d = s - p;
            sse += (d * d) as u64;
        }
    }
    Some(sse)
}

/// Exhaustive oracle: minimise prediction SSE over a dense full-pel
/// grid (`±FULL_RANGE`), then refine over the half- and quarter-pel
/// neighbours of the best full-pel point. Returns `(fp_best_mv,
/// fp_best_sse, best_mv, best_sse)` where the `fp_*` pair is the
/// full-pel-only winner (phase 1) and the unprefixed pair is the
/// full+sub-pel winner. Only considers in-bounds candidates.
#[allow(clippy::too_many_arguments)]
fn oracle_best(
    src: &[u8],
    src_stride: usize,
    refp: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    px: usize,
    py: usize,
    bw: usize,
    bh: usize,
) -> Option<(Mv, u64, Mv, u64)> {
    let mut best: Option<(Mv, u64)> = None;
    let consider = |mv: Mv, best: &mut Option<(Mv, u64)>| {
        if let Some(sse) = pred_sse(
            src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh, mv,
        ) {
            match best {
                Some((_, b)) if *b <= sse => {}
                _ => *best = Some((mv, sse)),
            }
        }
    };

    // Phase 1 — dense full-pel grid.
    for dy in -FULL_RANGE..=FULL_RANGE {
        for dx in -FULL_RANGE..=FULL_RANGE {
            let mv = Mv {
                row: (dy * 8) as i16,
                col: (dx * 8) as i16,
            };
            consider(mv, &mut best);
        }
    }
    let (fp, fp_sse) = best?;

    // Phase 2 — half-pel (±4 Q3) then quarter-pel (±2 Q3) neighbours of
    // the best full-pel point. The 8 surrounding offsets at each step.
    for &step in &[4i16, 2i16] {
        for &dy in &[-step, 0, step] {
            for &dx in &[-step, 0, step] {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let mv = Mv {
                    row: fp.row + dy,
                    col: fp.col + dx,
                };
                consider(mv, &mut best);
            }
        }
    }
    let (bmv, bsse) = best?;
    Some((fp, fp_sse, bmv, bsse))
}

fn psnr(sse: u64, px: u64) -> f64 {
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / px as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Record one sampled inter leaf. No-op unless enabled; only every
/// `SAMPLE_STRIDE`-th picker-supported leaf is actually evaluated (the
/// exhaustive grid is expensive). `rvp9_mv` is the MV rvp9's real NEWMV
/// search returned for this leaf (Q3 units).
#[allow(clippy::too_many_arguments)]
pub fn record_leaf(
    src: &[u8],
    src_stride: usize,
    refp: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    px: usize,
    py: usize,
    bw: usize,
    bh: usize,
    rvp9_mv: Mv,
    newmv_raw: Mv,
) {
    if !enabled() {
        return;
    }
    let do_it = LEAF_SEQ.with(|s| {
        let mut s = s.borrow_mut();
        let seq = *s;
        *s += 1;
        seq % SAMPLE_STRIDE == 0
    });
    if !do_it {
        return;
    }

    let sse_zero = match pred_sse(
        src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh, Mv::ZERO,
    ) {
        Some(v) => v,
        None => return,
    };
    let sse_rvp9 = pred_sse(
        src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh, rvp9_mv,
    )
    .unwrap_or(sse_zero);
    let sse_newmv = pred_sse(
        src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh, newmv_raw,
    )
    .unwrap_or(sse_rvp9);
    let Some((_fp_mv, sse_oracle_fp, omv, sse_oracle)) =
        oracle_best(src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh)
    else {
        return;
    };

    let blk_px = (bw * bh) as u64;
    let subpel = (omv.row & 7) != 0 || (omv.col & 7) != 0;
    let beats = psnr(sse_oracle, blk_px) - psnr(sse_rvp9, blk_px) >= 2.0;

    STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.n += 1;
        s.sse_zero += sse_zero;
        s.sse_rvp9 += sse_rvp9;
        s.sse_newmv += sse_newmv;
        s.sse_oracle_fp += sse_oracle_fp;
        s.sse_oracle += sse_oracle;
        s.px += blk_px;
        if subpel {
            s.oracle_subpel += 1;
        }
        if beats {
            s.oracle_beats_rvp9 += 1;
        }
    });

    PRINTED.with(|p| {
        let mut p = p.borrow_mut();
        if *p < PRINT_LIMIT {
            *p += 1;
            eprintln!(
                "E12 leaf px=({px},{py}) bs={bw}x{bh} | \
                 ZERO[{:.2}dB] shipped mv=({},{})[{:.2}dB] NEWMVraw mv=({},{})[{:.2}dB] \
                 oracle mv=({},{}) subpel={} [{:.2}dB]",
                psnr(sse_zero, blk_px),
                rvp9_mv.row,
                rvp9_mv.col,
                psnr(sse_rvp9, blk_px),
                newmv_raw.row,
                newmv_raw.col,
                psnr(sse_newmv, blk_px),
                omv.row,
                omv.col,
                subpel,
                psnr(sse_oracle, blk_px),
            );
        }
    });
}
