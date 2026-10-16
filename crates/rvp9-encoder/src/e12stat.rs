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
    let on = crate::diagnostic_env::var_os("RVP9_E12_ORACLE").is_some_and(|v| v != "0");
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
/// `None` for invalid block/plane geometry, incomplete source or
/// reference slice coverage, or when the 8-tap convolve read window
/// would fall outside the (un-padded) reference plane. The encoder's
/// MV-limits machinery is the border-extend equivalent, so OOB
/// candidates are simply not reachable and are skipped by the oracle.
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
    const MAX_VP9_DIM: usize = 65_536;

    if bw == 0 || bh == 0 || bw > 64 || bh > 64 {
        return None;
    }
    if ref_w == 0 || ref_h == 0 || ref_w > MAX_VP9_DIM || ref_h > MAX_VP9_DIM {
        return None;
    }
    if ref_stride < ref_w {
        return None;
    }

    // Validate the complete source block before any predictor work.
    // The last-row formula intentionally covers a tight final slice;
    // no trailing row padding is required.
    let src_row_end = px.checked_add(bw)?;
    if src_row_end > src_stride {
        return None;
    }
    let src_last_row = py.checked_add(bh.checked_sub(1)?)?;
    let src_end = src_last_row
        .checked_mul(src_stride)?
        .checked_add(src_row_end)?;
    if src_end > src.len() {
        return None;
    }

    let ref_w_u32 = u32::try_from(ref_w).ok()?;
    let ref_h_u32 = u32::try_from(ref_h).ok()?;
    let px_i32 = i32::try_from(px).ok()?;
    let py_i32 = i32::try_from(py).ok()?;

    // ScaleFactors below is identity-scaled. Prove the Q4 operations
    // before passing these coordinates through its narrowing i32 API.
    let q4_scale = 1i32.checked_shl(SUBPEL_BITS as u32)?;
    let checked_x_q4 = px_i32
        .checked_mul(q4_scale)?
        .checked_add(i32::from(mv.col))?;
    let checked_y_q4 = py_i32
        .checked_mul(q4_scale)?
        .checked_add(i32::from(mv.row))?;

    // Equal, VP9-bounded dimensions make this the identity scale. The
    // bound is important: ScaleFactors' shared constructor narrows its
    // shifted fixed-point numerator to i32 before division.
    let sf = ScaleFactors::new(ref_w_u32, ref_h_u32, ref_w_u32, ref_h_u32);
    if !sf.is_valid() || sf.is_scaled() {
        return None;
    }
    let total_x_q4 = sf.scaled_x_q4(px_i32, mv.col);
    let total_y_q4 = sf.scaled_y_q4(py_i32, mv.row);
    if total_x_q4 != checked_x_q4 || total_y_q4 != checked_y_q4 {
        return None;
    }
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

    // 8-tap halo: 3 left/top, +4 (TAPS-1) extent when sub-pel.
    let row_pre = if subpel_y_q4 != 0 { -3i32 } else { 0 };
    let col_pre = if subpel_x_q4 != 0 { -3i32 } else { 0 };
    let win_y = int_y.checked_add(row_pre)?;
    let win_x = int_x.checked_add(col_pre)?;
    let win_h = i32::try_from(bh)
        .ok()?
        .checked_add(if subpel_y_q4 != 0 { 7 } else { 0 })?;
    let win_w = i32::try_from(bw)
        .ok()?
        .checked_add(if subpel_x_q4 != 0 { 7 } else { 0 })?;
    let win_end_x = win_x.checked_add(win_w)?;
    let win_end_y = win_y.checked_add(win_h)?;
    let ref_w_i32 = i32::try_from(ref_w).ok()?;
    let ref_h_i32 = i32::try_from(ref_h).ok()?;
    if win_x < 0 || win_y < 0 || win_end_x > ref_w_i32 || win_end_y > ref_h_i32 {
        return None;
    }

    // Bound the exact rows/columns the copy or convolution kernel may
    // read. A caller may provide only the covered window, not a full
    // ref_w × ref_h allocation.
    let win_x_usize = usize::try_from(win_x).ok()?;
    let win_y_usize = usize::try_from(win_y).ok()?;
    let win_h_usize = usize::try_from(win_h).ok()?;
    let win_w_usize = usize::try_from(win_w).ok()?;
    let ref_base = win_y_usize
        .checked_mul(ref_stride)?
        .checked_add(win_x_usize)?;
    let ref_last_row = win_y_usize.checked_add(win_h_usize.checked_sub(1)?)?;
    let ref_end = ref_last_row
        .checked_mul(ref_stride)?
        .checked_add(win_x_usize)?
        .checked_add(win_w_usize)?;
    if ref_end > refp.len() {
        return None;
    }

    let mut pred = [0u8; 64 * 64];
    inter_predictor(
        refp.get(ref_base..ref_end)?,
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
        let srow = py
            .checked_add(r)?
            .checked_mul(src_stride)?
            .checked_add(px)?;
        for c in 0..bw {
            let s = i64::from(*src.get(srow.checked_add(c)?)?);
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
/// search returned for this leaf (Q3 units). A sampled leaf with invalid
/// block or plane geometry is skipped without accumulating statistics.
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
        src,
        src_stride,
        refp,
        ref_stride,
        ref_w,
        ref_h,
        px,
        py,
        bw,
        bh,
        Mv::ZERO,
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
    let Some((_fp_mv, sse_oracle_fp, omv, sse_oracle)) = oracle_best(
        src, src_stride, refp, ref_stride, ref_w, ref_h, px, py, bw, bh,
    ) else {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================================
    // Fixture helpers
    // ===========================================================

    /// Build a reference plane with a non-linear hash pattern. The
    /// pattern is chosen so that no integer translation up to
    /// `±FULL_RANGE` pels yields a coincident byte pattern — this
    /// makes MV=(0,0) the *unique* global minimum when src == refp.
    /// A purely linear `(a*y + b*x + c) mod 256` pattern would alias
    /// under translations that satisfy `a*dy + b*dx ≡ 0 (mod 256)`,
    /// so we use a multiplicative hash with avalanche.
    fn make_pattern_plane(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = (y as u32)
                    .wrapping_mul(0x9E37_79B1)
                    .wrapping_add((x as u32).wrapping_mul(0x85EB_CA77))
                    .wrapping_add(0x12_3456);
                let v = v.wrapping_mul(0xC2B2_AE3D);
                // Final XOR-fold so the top bits inform the bottom.
                let v = v ^ v.rotate_right(13);
                buf[y * w + x] = v as u8;
            }
        }
        buf
    }

    /// Build a shifted copy: out[y][x] = base[y+dy][x+dx]. Used to
    /// fabricate a "source" whose oracle-best MV is a known integer
    /// offset. Caller must keep (dy, dx) inside the base plane.
    fn shifted_plane(base: &[u8], w: usize, h: usize, dy: usize, dx: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..(h - dy) {
            for x in 0..(w - dx) {
                out[y * w + x] = base[(y + dy) * w + (x + dx)];
            }
        }
        out
    }

    /// Process-wide lock so tests that mutate `E12_ON` don't race
    /// each other (cargo test runs them on parallel threads but the
    /// atomic is shared). Pure functions that do not touch the atom
    /// (`pred_sse`, `oracle_best`, `psnr`) do not need to hold this.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Drop guard that restores the global enable atomic on test exit
    /// and keeps the per-process mutex held for the full test
    /// duration so concurrent record_leaf-based tests serialise.
    struct EnableGuard {
        prior: bool,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl EnableGuard {
        fn acquire() -> Self {
            let lock = TEST_LOCK
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            Self {
                prior: E12_ON.load(Ordering::Relaxed),
                _lock: lock,
            }
        }
    }
    impl Drop for EnableGuard {
        fn drop(&mut self) {
            E12_ON.store(self.prior, Ordering::Relaxed);
        }
    }

    /// Drain the thread-local STATS, LEAF_SEQ and PRINTED so the
    /// test starts from a known baseline regardless of in-thread
    /// ordering with other tests.
    fn reset_thread_local() {
        let _ = take_snapshot();
        LEAF_SEQ.with(|s| *s.borrow_mut() = 0);
        PRINTED.with(|p| *p.borrow_mut() = 0);
    }

    #[derive(Clone, Copy)]
    struct LeafCase<'a> {
        name: &'static str,
        src: &'a [u8],
        src_stride: usize,
        refp: &'a [u8],
        ref_stride: usize,
        ref_w: usize,
        ref_h: usize,
        px: usize,
        py: usize,
        bw: usize,
        bh: usize,
        rvp9_mv: Mv,
        newmv_raw: Mv,
    }

    impl LeafCase<'_> {
        fn record(self) {
            record_leaf(
                self.src,
                self.src_stride,
                self.refp,
                self.ref_stride,
                self.ref_w,
                self.ref_h,
                self.px,
                self.py,
                self.bw,
                self.bh,
                self.rvp9_mv,
                self.newmv_raw,
            );
        }
    }

    fn leaf_case<'a>(
        name: &'static str,
        src: (&'a [u8], usize),
        reference: (&'a [u8], usize, usize, usize),
        origin: (usize, usize),
        block: (usize, usize),
    ) -> LeafCase<'a> {
        LeafCase {
            name,
            src: src.0,
            src_stride: src.1,
            refp: reference.0,
            ref_stride: reference.1,
            ref_w: reference.2,
            ref_h: reference.3,
            px: origin.0,
            py: origin.1,
            bw: block.0,
            bh: block.1,
            rvp9_mv: Mv::ZERO,
            newmv_raw: Mv::ZERO,
        }
    }

    // ===========================================================
    // psnr — every arithmetic arm
    // ===========================================================

    #[test]
    fn psnr_zero_sse_returns_sentinel() {
        assert_eq!(psnr(0, 64), 100.0, "sse==0 takes the early-return arm");
    }

    #[test]
    fn psnr_worst_case_zero_db() {
        // MSE = 255^2 exactly → PSNR = 10 * log10(1) = 0.
        let px: u64 = 64;
        let sse = 255u64 * 255u64 * px;
        let got = psnr(sse, px);
        assert!(
            (got - 0.0).abs() < 1e-9,
            "expected 0 dB at MSE=255², got {got}",
        );
    }

    #[test]
    fn psnr_known_intermediate_value() {
        // SSE = px → MSE = 1 → PSNR = 10*log10(255² / 1) = 48.13080…
        let px: u64 = 64;
        let sse: u64 = px;
        let expected = 10.0 * (255.0_f64 * 255.0).log10();
        let got = psnr(sse, px);
        assert!(
            (got - expected).abs() < 1e-9,
            "expected {expected} dB, got {got}",
        );
    }

    // ===========================================================
    // pred_sse — every branch
    // ===========================================================

    #[test]
    fn pred_sse_zero_mv_on_identity_planes_is_zero() {
        // src == refp, no motion → pred matches src exactly.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let sse = pred_sse(&plane, w, &plane, w, w, h, 16, 16, 8, 8, Mv::ZERO)
            .expect("ZERO mv should be in-bounds for a centered block");
        assert_eq!(sse, 0, "identity prediction must give SSE = 0");
    }

    #[test]
    fn pred_sse_nonzero_mv_on_pattern_yields_positive_sse() {
        // src == refp, MV != 0 → prediction reads a shifted view of
        // refp; with our non-trivial pattern that's NOT equal to src.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        // mv.col = 16 in Q4 = 1 full pel to the right; subpel = 0.
        let mv = Mv { row: 0, col: 16 };
        let sse = pred_sse(&plane, w, &plane, w, w, h, 16, 16, 8, 8, mv)
            .expect("integer +1 pel inside a 64×64 plane is in-bounds");
        // Hand-compute: at every (r, c) the diff is
        //   src[16+r][16+c] - refp[16+r][17+c]
        // = (37r + 17(16+c) + 5) - (37r + 17(17+c) + 5)   mod 256
        // = -17  mod 256                                  → 239 (as u8 sub)
        // But we subtract as i64, so the diff is exactly -17, sse per
        // pixel = 289. Over 64 pels: 64 * 289 = 18_496.
        // BUT: the pattern is mod 256, so wrap-arounds appear. We
        // therefore compute the oracle sum directly here.
        let mut hand: u64 = 0;
        for r in 0..8 {
            for c in 0..8 {
                let s = i64::from(plane[(16 + r) * w + (16 + c)]);
                let p = i64::from(plane[(16 + r) * w + (17 + c)]);
                let d = s - p;
                hand += (d * d) as u64;
            }
        }
        assert_eq!(sse, hand, "pred_sse must equal the hand-computed SSE");
        assert!(
            sse > 0,
            "shifted view of non-trivial pattern cannot be zero"
        );
    }

    #[test]
    fn pred_sse_subpel_motion_triggers_halo() {
        // Sub-pel MV → row_pre/col_pre = -3 and win_w/h += 7.
        // We need px,py >= 3 and px+bw+7 <= ref_w, py+bh+7 <= ref_h
        // for the halo window to be in-bounds.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let mv = Mv { row: 8, col: 8 }; // half-pel diagonal (Q4)
        let sse = pred_sse(&plane, w, &plane, w, w, h, 16, 16, 8, 8, mv);
        assert!(
            sse.is_some(),
            "sub-pel halo at px=py=16, bw=bh=8 fits inside 64×64",
        );
        assert!(sse.unwrap() > 0, "half-pel filtered prediction != source");
    }

    #[test]
    fn pred_sse_subpel_x_only_triggers_col_halo_only() {
        // Sub-pel x but integer y → only col_pre / win_w extended.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let mv = Mv { row: 0, col: 8 };
        let sse = pred_sse(&plane, w, &plane, w, w, h, 16, 16, 8, 8, mv);
        assert!(sse.is_some());
    }

    #[test]
    fn pred_sse_subpel_y_only_triggers_row_halo_only() {
        // Integer x but sub-pel y → only row_pre / win_h extended.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let mv = Mv { row: 8, col: 0 };
        let sse = pred_sse(&plane, w, &plane, w, w, h, 16, 16, 8, 8, mv);
        assert!(sse.is_some());
    }

    #[test]
    fn pred_sse_returns_none_when_win_x_negative() {
        // Negative MV at left edge → win_x < 0.
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        // mv.col = -32 means int_x shift of -2 (Q4 → pel), at px=0
        // win_x = -2 → < 0.
        let mv = Mv { row: 0, col: -32 };
        assert!(
            pred_sse(&plane, w, &plane, w, w, h, 0, 16, 8, 8, mv).is_none(),
            "win_x < 0 must return None",
        );
    }

    #[test]
    fn pred_sse_returns_none_when_win_y_negative() {
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let mv = Mv { row: -32, col: 0 };
        assert!(
            pred_sse(&plane, w, &plane, w, w, h, 16, 0, 8, 8, mv).is_none(),
            "win_y < 0 must return None",
        );
    }

    #[test]
    fn pred_sse_returns_none_when_window_exceeds_ref_w() {
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        // Block at px = 56, bw = 8 → block ends at col 64. With
        // mv.col = 32 (= +2 pel), window starts at px+2 = 58 and
        // ends at 58+8 = 66 > 64.
        let mv = Mv { row: 0, col: 32 };
        assert!(
            pred_sse(&plane, w, &plane, w, w, h, 56, 16, 8, 8, mv).is_none(),
            "win_x + win_w > ref_w must return None",
        );
    }

    #[test]
    fn pred_sse_returns_none_when_window_exceeds_ref_h() {
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let mv = Mv { row: 32, col: 0 };
        assert!(
            pred_sse(&plane, w, &plane, w, w, h, 16, 56, 8, 8, mv).is_none(),
            "win_y + win_h > ref_h must return None",
        );
    }

    // ===========================================================
    // oracle_best — every branch
    // ===========================================================

    #[test]
    fn oracle_best_finds_zero_mv_on_identity_planes() {
        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let (fp_mv, fp_sse, best_mv, best_sse) =
            oracle_best(&plane, w, &plane, w, w, h, 16, 16, 8, 8)
                .expect("interior block has many in-bounds candidates");
        assert_eq!(best_mv, Mv::ZERO, "MV=ZERO is the unique global minimum");
        assert_eq!(best_sse, 0);
        assert_eq!(fp_mv, Mv::ZERO);
        assert_eq!(fp_sse, 0);
    }

    #[test]
    fn oracle_best_finds_translation_in_shifted_source() {
        let w = 64;
        let h = 64;
        let refp = make_pattern_plane(w, h);
        // src is the +1 pel down-right view of refp. Then the
        // matching MV is (16, 16) in Q4 (+1 pel each axis).
        let src = shifted_plane(&refp, w, h, 1, 1);

        // Sample inside the valid (no-edge) region so the matching MV
        // is in-bounds for the oracle window.
        let (_, _, best_mv, best_sse) = oracle_best(&src, w, &refp, w, w, h, 16, 16, 8, 8)
            .expect("centered block has the matching MV in-bounds");
        assert_eq!(best_mv, Mv { row: 16, col: 16 });
        assert_eq!(
            best_sse, 0,
            "exact translation must give zero prediction SSE",
        );
    }

    #[test]
    fn oracle_best_returns_none_when_no_candidate_is_in_bounds() {
        // A reference plane just barely smaller than the block. Even
        // MV=(0,0) needs a `bw × bh` window starting at (px, py);
        // pick px,py,bw,bh so that bw > ref_w.
        let plane = make_pattern_plane(4, 4); // 4×4 plane
        let result = oracle_best(&plane, 4, &plane, 4, 4, 4, 0, 0, 8, 8);
        assert!(
            result.is_none(),
            "no in-bounds candidate exists → phase-1 best? returns None",
        );
    }

    // ===========================================================
    // record_leaf — every branch
    // ===========================================================

    #[test]
    fn record_leaf_invalid_sampled_geometry_is_a_safe_noop() {
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);

        let plane = make_pattern_plane(64, 64);
        let zeros = vec![0; 64 * 64];
        let too_wide = vec![0; 65];
        let too_tall = vec![0; 65];
        let too_large = vec![0; 65 * 65];
        let short = [0];
        let wide_ref_stride = 70_000;
        let wide_ref = vec![0; 23 * wide_ref_stride + 24];
        let tall_ref_stride = 64;
        let tall_ref = vec![0; 23 * tall_ref_stride + 24];
        let mut cases = vec![
            leaf_case(
                "width above 64",
                (&too_wide, 65),
                (&too_wide, 65, 65, 1),
                (0, 0),
                (65, 1),
            ),
            leaf_case(
                "height above 64",
                (&too_tall, 1),
                (&too_tall, 1, 1, 65),
                (0, 0),
                (1, 65),
            ),
            leaf_case(
                "both dimensions above 64",
                (&too_large, 65),
                (&too_large, 65, 65, 65),
                (0, 0),
                (65, 65),
            ),
            leaf_case(
                "zero width",
                (&plane, 64),
                (&plane, 64, 64, 64),
                (16, 16),
                (0, 8),
            ),
            leaf_case(
                "zero height",
                (&plane, 64),
                (&plane, 64, 64, 64),
                (16, 16),
                (8, 0),
            ),
            leaf_case(
                "short source",
                (&short, 64),
                (&plane, 64, 64, 64),
                (16, 16),
                (8, 8),
            ),
            leaf_case(
                "short reference",
                (&plane, 64),
                (&short, 64, 64, 64),
                (16, 16),
                (8, 8),
            ),
            leaf_case(
                "source stride smaller than block",
                (&plane, 7),
                (&plane, 64, 64, 64),
                (0, 16),
                (8, 8),
            ),
            leaf_case(
                "zero source stride",
                (&plane, 0),
                (&plane, 64, 64, 64),
                (0, 16),
                (8, 8),
            ),
            leaf_case(
                "reference stride smaller than width",
                (&plane, 64),
                (&plane, 63, 64, 64),
                (16, 16),
                (8, 8),
            ),
            leaf_case(
                "zero reference stride",
                (&plane, 64),
                (&plane, 0, 64, 64),
                (16, 16),
                (8, 8),
            ),
            leaf_case(
                "zero reference width",
                (&zeros, 64),
                (&zeros, 64, 0, 64),
                (0, 0),
                (8, 8),
            ),
            leaf_case(
                "zero reference height",
                (&zeros, 64),
                (&zeros, 64, 64, 0),
                (0, 0),
                (8, 8),
            ),
            leaf_case(
                "usize::MAX x offset",
                (&zeros, 64),
                (&zeros, 64, 64, 64),
                (usize::MAX, 16),
                (8, 8),
            ),
            leaf_case(
                "usize::MAX y offset",
                (&zeros, 64),
                (&zeros, 64, 64, 64),
                (16, usize::MAX),
                (8, 8),
            ),
            leaf_case(
                "overflowing source stride product",
                (&zeros, usize::MAX),
                (&zeros, 64, 64, 64),
                (0, 2),
                (8, 8),
            ),
            leaf_case(
                "overflowing reference stride product",
                (&zeros, 64),
                (&zeros, usize::MAX, 64, 64),
                (0, 2),
                (8, 8),
            ),
            leaf_case(
                "reference width above VP9 maximum",
                (&zeros, 64),
                (&wide_ref, wide_ref_stride, 70_000, 64),
                (0, 0),
                (8, 8),
            ),
            leaf_case(
                "reference height above VP9 maximum",
                (&zeros, 64),
                (&tall_ref, tall_ref_stride, 64, 70_000),
                (0, 0),
                (8, 8),
            ),
        ];

        if let Ok(converted_dim) = usize::try_from(u64::from(u32::MAX) + 65) {
            cases.push(leaf_case(
                "reference width must not truncate to u32",
                (&zeros, 64),
                (&zeros, 64, converted_dim, 64),
                (16, 16),
                (8, 8),
            ));
        }
        if let Ok(large_offset) = usize::try_from(u64::from(u32::MAX) + 17) {
            cases.push(leaf_case(
                "x offset must not truncate to i32",
                (&zeros, 64),
                (&zeros, 64, 64, 64),
                (large_offset, 16),
                (8, 8),
            ));
            cases.push(leaf_case(
                "y offset must not truncate to i32",
                (&zeros, 64),
                (&zeros, 64, 64, 64),
                (16, large_offset),
                (8, 8),
            ));
        }

        let mut violations = Vec::new();
        for case in cases {
            reset_thread_local();
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| case.record()));
            let snap = take_snapshot();
            if caught.is_err() || snap.n != 0 {
                violations.push(format!(
                    "{}: catch_unwind().is_ok()={}, accumulated n={}",
                    case.name,
                    caught.is_ok(),
                    snap.n,
                ));
            }
        }
        assert!(
            violations.is_empty(),
            "invalid enabled sampled geometry must be a no-op without panic: {violations:?}",
        );
    }

    #[test]
    fn record_leaf_fractional_reference_halo_uses_slice_coverage() {
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let src = vec![0; 64 * 64];
        // This is exactly enough for the integer 8x8 block at (16,16),
        // but not for its 3-pixel fractional-filter halo.
        let ref_end = (16 + 8 - 1) * 64 + 16 + 8;
        let refp = vec![0; ref_end];
        let fractional_mv = Mv { row: 8, col: 8 };
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            record_leaf(
                &src,
                64,
                &refp,
                64,
                64,
                64,
                16,
                16,
                8,
                8,
                fractional_mv,
                fractional_mv,
            );
        }));
        assert!(
            caught.is_ok(),
            "fractional prediction outside the actual slice window must return None",
        );
        let snap = take_snapshot();
        assert_eq!(snap.n, 1, "invalid candidates fall back to the ZERO sample");
        assert_eq!(snap.sse_rvp9, snap.sse_zero);
        assert_eq!(snap.sse_newmv, snap.sse_rvp9);
    }

    #[test]
    fn record_leaf_no_op_when_disabled() {
        let _g = EnableGuard::acquire();
        E12_ON.store(false, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        record_leaf(&plane, w, &plane, w, w, h, 16, 16, 8, 8, Mv::ZERO, Mv::ZERO);
        let snap = take_snapshot();
        assert_eq!(snap.n, 0, "disabled record_leaf must not accumulate");
        assert_eq!(snap.sse_zero, 0);
        assert_eq!(snap.sse_oracle, 0);
        assert_eq!(snap.px, 0);
    }

    #[test]
    fn record_leaf_sample_stride_filters_most_calls() {
        // With SAMPLE_STRIDE = 7, only every 7th call evaluates the
        // oracle. We make 14 calls → expect exactly 2 accumulator
        // bumps (seq=0 and seq=7).
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        // Exercise the largest supported leaf with distinct strides.
        // `src` ends exactly after the final block byte, so the final
        // row proves the checked coverage uses `last_row * stride + width`.
        let w = 72;
        let h = 72;
        let visible = make_pattern_plane(w, h);
        let ref_stride = 80;
        let ref_len = (h - 1) * ref_stride + w;
        let mut refp = vec![0; ref_len];
        for y in 0..h {
            for x in 0..w {
                refp[y * ref_stride + x] = visible[y * w + x];
            }
        }
        let (px, py, bw, bh) = (3, 3, 64, 64);
        let src_stride = 70;
        let src_len = (py + bh - 1) * src_stride + px + bw;
        let mut src = vec![0; src_len];
        for r in 0..bh {
            for c in 0..bw {
                src[(py + r) * src_stride + px + c] = visible[(py + r) * w + px + c];
            }
        }
        for _ in 0..14 {
            record_leaf(
                &src,
                src_stride,
                &refp,
                ref_stride,
                w,
                h,
                px,
                py,
                bw,
                bh,
                Mv::ZERO,
                Mv::ZERO,
            );
        }
        let snap = take_snapshot();
        assert_eq!(snap.n, 2, "SAMPLE_STRIDE = 7 → 2 of 14 calls evaluated");
        // Both calls are identity (src=refp); SSE everywhere = 0 for
        // ZERO mv. Oracle agrees → sse_oracle = 0, oracle_subpel = 0.
        assert_eq!(snap.sse_zero, 0);
        assert_eq!(snap.sse_rvp9, 0);
        assert_eq!(snap.sse_newmv, 0);
        assert_eq!(snap.sse_oracle_fp, 0);
        assert_eq!(snap.sse_oracle, 0);
        assert_eq!(snap.px, 2 * 64 * 64);
        // Oracle's best MV is ZERO → not sub-pel; psnr(0,*) = 100,
        // so beats threshold (100 - 100 = 0 < 2 dB) is NOT met.
        assert_eq!(snap.oracle_subpel, 0);
        assert_eq!(snap.oracle_beats_rvp9, 0);
    }

    #[test]
    fn record_leaf_oob_zero_mv_short_circuits_silently() {
        // First call inside SAMPLE_STRIDE window with px so close to
        // the right edge that the ZERO-MV `pred_sse` itself is OOB →
        // the `match pred_sse(...) { Some(v) => v, None => return }`
        // arm fires for sse_zero.
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        // px=60, bw=8 → window ends at col 68 > 64 → OOB.
        record_leaf(&plane, w, &plane, w, w, h, 60, 16, 8, 8, Mv::ZERO, Mv::ZERO);
        let snap = take_snapshot();
        assert_eq!(
            snap.n, 0,
            "OOB ZERO-MV must early-return without bumping accumulator",
        );
    }

    #[test]
    fn record_leaf_oracle_short_circuits_when_plane_too_small() {
        // Make sure SAMPLE_STRIDE picks up this call (first call on
        // a fresh thread-local), then trigger oracle_best to return
        // None. We use a tiny reference plane (4×4) and a block size
        // (8×8) that has no in-bounds candidates anywhere; the
        // sse_zero pred_sse also returns None, so we never reach the
        // oracle. To hit the explicit oracle short-circuit, we need
        // sse_zero to succeed but oracle_best to fail.
        //
        // Trick: make the ref plane exactly big enough for ZERO MV
        // (so sse_zero = Some) but small enough that EVERY non-zero
        // oracle candidate is OOB. With ref_w = ref_h = 8 + 1 = 9
        // and px = py = 0, bw = bh = 8:
        //   ZERO MV → win=(0,0)..(8,8) ⊆ (0,0)..(9,9) ✓
        //   MV = (8, 8) → subpel halo, win_y = -3 → OOB
        //   MV = (-8, -8) → subpel halo, win_y = -3 → OOB
        //   MV = (16, 16) → int_y = 1, win_y = 1, win_h = 8,
        //                   win_y+win_h = 9 ≤ 9 ✓ — IN-BOUNDS.
        //
        // So a 9×9 plane still leaves at least one in-bounds candidate
        // besides (0,0). With ref_w = ref_h = 8 exactly:
        //   ZERO MV → win=(0,0)..(8,8) ⊆ (0,0)..(8,8) ✓
        //   MV = (16, 0) → int_y = 1, win_y = 1, win_y+win_h = 9 > 8 OOB.
        //   MV = (-16, 0) → int_y = -1, win_y = -1 < 0 OOB.
        //   All non-zero candidates have either subpel halo OOB or
        //   integer shift OOB.
        //
        // BUT this case keeps oracle returning Some(...) with MV=ZERO
        // because at least ZERO itself succeeds — so oracle never
        // returns None here. The accumulator DOES bump.
        //
        // To make oracle_best return None we need EVERY candidate to
        // be OOB — including ZERO. But then sse_zero also returns
        // None, taking the *earlier* None arm. We therefore cannot
        // reach the explicit `oracle_best == None` arm without
        // sse_zero also being None — the two failure regions
        // coincide. We document that here and assert the symmetric
        // behaviour: tiny plane where ZERO is OOB returns
        // accumulator unchanged.
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let small = make_pattern_plane(4, 4);
        record_leaf(&small, 4, &small, 4, 4, 4, 0, 0, 8, 8, Mv::ZERO, Mv::ZERO);
        let snap = take_snapshot();
        assert_eq!(
            snap.n, 0,
            "any None along the chain must abort without accumulating",
        );
    }

    #[test]
    fn record_leaf_records_oracle_subpel_and_beats_arms() {
        // We construct a scenario where the oracle's best MV is
        // sub-pel — by making src match refp at a *quarter*-pel
        // shift. With our integer-coordinate pattern that is hard
        // to make exact (no quarter-pel interpolation lookup), but
        // we can at least force `subpel` true by checking that the
        // oracle subpel-and-beats path executes.
        //
        // Instead, we exercise the simpler `oracle_beats_rvp9` arm:
        // shipped MV = ZERO (which has large SSE on our shifted
        // source) while the oracle finds a translation MV with SSE
        // = 0 — that gives ≥ 2 dB margin trivially (anything vs 100
        // dB).
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let refp = make_pattern_plane(w, h);
        let src = shifted_plane(&refp, w, h, 1, 1);

        // The sampled predictions exercise a valid half-pel halo, while
        // the exhaustive oracle finds the exact integer translation.
        let fractional_mv = Mv { row: 8, col: 8 };
        record_leaf(
            &src,
            w,
            &refp,
            w,
            w,
            h,
            16,
            16,
            8,
            8,
            fractional_mv,
            fractional_mv,
        );
        let snap = take_snapshot();
        assert_eq!(snap.n, 1);
        assert_eq!(snap.sse_oracle, 0, "exact-shift oracle has SSE=0");
        assert_eq!(snap.sse_oracle_fp, 0);
        assert!(
            snap.sse_rvp9 > 0,
            "half-pel prediction differs from shifted source"
        );
        assert!(snap.sse_newmv > 0, "half-pel NEWMV prediction is evaluated");
        assert_ne!(
            snap.sse_rvp9, snap.sse_zero,
            "valid fractional prediction must not fall back to ZERO",
        );
        assert_eq!(snap.sse_newmv, snap.sse_rvp9);
        assert!(snap.sse_zero > 0);
        // Oracle MV (16, 16) is integer-pel → 16 & 7 = 0 on both
        // axes, NOT sub-pel.
        assert_eq!(snap.oracle_subpel, 0);
        // psnr(rvp9, blk_px) is finite; psnr(0, blk_px) = 100 →
        // beats by ≥ 2 dB.
        assert_eq!(snap.oracle_beats_rvp9, 1);
        assert_eq!(snap.px, 8 * 8);
    }

    #[test]
    fn take_snapshot_resets_thread_local() {
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        record_leaf(&plane, w, &plane, w, w, h, 16, 16, 8, 8, Mv::ZERO, Mv::ZERO);
        let first = take_snapshot();
        assert_eq!(first.n, 1, "one accumulating call");
        let second = take_snapshot();
        assert_eq!(second.n, 0, "snapshot drains the thread-local");
        assert_eq!(second.sse_oracle, 0);
        assert_eq!(second.px, 0);
    }

    #[test]
    fn init_from_env_reflects_environment() {
        let _scope = crate::diagnostic_env::scope();
        let _g = EnableGuard::acquire();
        crate::diagnostic_env::remove_var("RVP9_E12_ORACLE");
        init_from_env();
        assert!(!enabled(), "unset env var → disabled");

        crate::diagnostic_env::set_var("RVP9_E12_ORACLE", "1");
        init_from_env();
        assert!(enabled(), "non-zero env var → enabled");

        crate::diagnostic_env::set_var("RVP9_E12_ORACLE", "0");
        init_from_env();
        assert!(!enabled(), "literal \"0\" → disabled");

        crate::diagnostic_env::remove_var("RVP9_E12_ORACLE");
    }

    #[test]
    fn record_leaf_uses_fallback_when_rvp9_or_newmv_oob() {
        // rvp9_mv and newmv_raw can both be OOB (the search returned
        // something outside the un-extended ref). pred_sse returns
        // None for them, so record_leaf falls back via `unwrap_or`
        // — sse_rvp9 falls back to sse_zero, sse_newmv to sse_rvp9.
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        // ZERO mv at (16,16) is in-bounds, so sse_zero succeeds.
        // rvp9_mv chosen with huge col so the window goes OOB.
        let rvp9_oob = Mv {
            row: 0,
            col: 32_000,
        };
        let newmv_oob = Mv {
            row: 32_000,
            col: 0,
        };
        record_leaf(
            &plane, w, &plane, w, w, h, 16, 16, 8, 8, rvp9_oob, newmv_oob,
        );
        let snap = take_snapshot();
        assert_eq!(snap.n, 1);
        // Both fall back: sse_rvp9 == sse_zero == 0, sse_newmv ==
        // sse_rvp9 == 0.
        assert_eq!(snap.sse_zero, 0);
        assert_eq!(snap.sse_rvp9, 0, "OOB rvp9_mv falls back to sse_zero");
        assert_eq!(snap.sse_newmv, 0, "OOB newmv_raw falls back to sse_rvp9");
    }

    #[test]
    fn record_leaf_print_limit_does_not_panic() {
        // Drive enough calls to push PRINTED past PRINT_LIMIT and
        // exercise the `if *p < PRINT_LIMIT` false arm. Each
        // accumulating call (every 7th) bumps PRINTED, so we need
        // SAMPLE_STRIDE * (PRINT_LIMIT + 1) calls to fully exhaust.
        let _g = EnableGuard::acquire();
        E12_ON.store(true, Ordering::Relaxed);
        reset_thread_local();

        let w = 64;
        let h = 64;
        let plane = make_pattern_plane(w, h);
        let n_calls = SAMPLE_STRIDE * (PRINT_LIMIT + 2);
        for _ in 0..n_calls {
            record_leaf(&plane, w, &plane, w, w, h, 16, 16, 8, 8, Mv::ZERO, Mv::ZERO);
        }
        let snap = take_snapshot();
        // PRINT_LIMIT + 2 accumulating calls → snap.n == PRINT_LIMIT + 2.
        assert_eq!(snap.n as usize, PRINT_LIMIT + 2);
        // PRINTED counter should be saturated at PRINT_LIMIT.
        let printed = PRINTED.with(|p| *p.borrow());
        assert_eq!(printed, PRINT_LIMIT, "PRINTED saturates at PRINT_LIMIT");
        // Reset PRINTED so we don't perturb the next test on this thread.
        PRINTED.with(|p| *p.borrow_mut() = 0);
    }
}
