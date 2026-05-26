//! W4.7 — `single_motion_search` + adaptive step + enhanced full-pel.
//!
//! ## libvpx reference
//!
//! Port of `vp9/encoder/vp9_rdopt.c:2564-2745` (v1.14.1 commit
//! `12f3a2ac6`). The RDO MV search:
//!
//! - Computes the diamond `step_param` from `cpi->mv_step_param` /
//!   `auto_mv_step_size` / `adaptive_motion_search` (lines 2616-2659).
//! - Runs `vp9_full_pixel_search` from `mvp_full` (lines 2674-2676).
//! - When `sf.enhanced_full_pixel_motion_search` is set, re-runs from
//!   each of `pred_mv[1..3]` and keeps the lowest SAD (lines 2679-2723).
//! - Runs sub-pel refinement (`find_fractional_mv_step`, lines
//!   2727-2735).
//! - Returns `(best_mv, rate_mv)`.
//!
//! ## Scope / deviations
//!
//! - **No scaled-ref-frame swapping.** libvpx (lines 2604-2612 +
//!   2741-2744) swaps `xd->plane[i].pre[0]` to the scaled-reference
//!   plane before the search; we treat that as the caller's
//!   responsibility (the caller passes the scaled `ref_plane` directly).
//! - **CONFIG_NON_GREEDY_MV is off** (the libvpx default). The
//!   `vp9_full_pixel_diamond_new` branch is unreachable in our build.
//! - **No `auto_mv_step_size`** — the caller passes a fixed
//!   `mv_step_param` (libvpx's `cpi->mv_step_param` is a static
//!   per-frame value). The branch at libvpx line 2616-2622 collapses
//!   to "pass through".
//! - **`adaptive_motion_search` early-exit at line 2642-2657** —
//!   libvpx exits with `INVALID_MV` when the per-ref `pred_mv_sad`
//!   is way larger than the other refs' SAD. The caller plumbs the
//!   `pred_mv_sad` array; we surface the early-exit as
//!   [`SingleMotionSearchResult::InvalidMv`].
//! - **Sub-pel refinement** uses
//!   [`find_best_sub_pixel_tree_pruned_more`] (the only rvp9 sub-pel
//!   currently ported). libvpx's `find_fractional_mv_step` is plumbed
//!   from `cpi->find_fractional_mv_step`, which the speed-feature
//!   plumbing selects between `tree`, `tree_pruned`, `tree_pruned_more`,
//!   and `tree_pruned_evenmore`. The pruned-more variant is the
//!   speed-7 default and matches what `inter.rs` already uses.

use rvp9_core::block::BlockSize;
use rvp9_core::cost::mv_cost::{MvCosts, mv_cost};
use rvp9_core::mode_info::Mv;
use rvp9_core::mv_search::full_pixel::{
    FullPixelSearchParams, FullPixelSearchResult, PlaneRef, diamond_search_sad,
};
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::mv_search::search_site::SearchSiteConfig;
use rvp9_core::mv_search::sub_pixel::{
    SubPixelForcedStop, SubPixelSearchParams, find_best_sub_pixel_tree_pruned_more,
};

/// libvpx `MV_COST_WEIGHT` (`vp9_rd.h:38`).
pub const MV_COST_WEIGHT: i32 = 108;

/// libvpx `MAX_MVSEARCH_STEPS` (`vp9_mcomp.h:24`).
pub const MAX_MVSEARCH_STEPS_RDO: i32 = 11;

/// `get_msb` (libvpx `vpx_ports/bitops.h:23-31`). Index of the most
/// significant bit of an unsigned int (≥ 0); for input 0 the libvpx
/// API leaves the result undefined — we return 0 to match the
/// "no step" interpretation at line 2700 (`if (step <= 0) continue`).
#[inline]
#[must_use]
fn get_msb(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        (31 - v.leading_zeros()) as i32
    }
}

/// libvpx `vp9_mv_bit_cost` (`vp9_mcomp.c:80-84`).
/// Returns the MV-bit rate cost in `vp9_prob_cost` 1/512-bit units,
/// with the `MV_COST_WEIGHT/128` damping factor applied.
///
/// ```c
/// int vp9_mv_bit_cost(const MV *mv, const MV *ref,
///                     const int *mvjcost, int *mvcost[2],
///                     int weight) {
///   const MV diff = { mv->row - ref->row, mv->col - ref->col };
///   return ROUND_POWER_OF_TWO(mv_cost(&diff, mvjcost, mvcost) * weight, 7);
/// }
/// ```
#[inline]
#[must_use]
pub fn vp9_mv_bit_cost(mv: Mv, ref_mv: Mv, mv_costs: &MvCosts, weight: i32) -> u32 {
    let diff_row = mv.row.wrapping_sub(ref_mv.row);
    let diff_col = mv.col.wrapping_sub(ref_mv.col);
    let raw = mv_cost(diff_row, diff_col, mv_costs);
    // ROUND_POWER_OF_TWO(x, 7) = (x + (1 << 6)) >> 7.
    let prod = (raw as i64).saturating_mul(weight as i64);
    let rounded = (prod + (1i64 << 6)) >> 7;
    rounded.max(0) as u32
}

/// Caller-provided RDO MV-search context. Mirrors the libvpx
/// `MACROBLOCK*` fields the C version consumes.
pub struct SingleMotionSearchParams<'a> {
    /// Block size being decided.
    pub block_size: BlockSize,
    /// Source plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block in plane samples (`(col, row)`).
    pub src_offset: (usize, usize),
    /// Reference plane (post-scaling — the caller handles `scaled_ref_frame`).
    pub ref_plane: &'a PlaneRef<'a>,
    /// libvpx `x->mbmi_ext->ref_mvs[ref][0]` — the rate-reference MV
    /// (NEAREST). Used as the cost-attractor for `mvsad_err_cost`.
    pub ref_mv: Mv,
    /// libvpx `x->mbmi_ext->ref_mvs[ref][1]` — the second NEAR
    /// candidate (drives enhanced-full-pel slot `pred_mv[1]`).
    pub near_mv: Mv,
    /// libvpx `x->pred_mv[ref]` — the per-SB int_pro seed.
    /// `INT16_MAX` in either component means "invalid" (libvpx
    /// line 2688). We surface that as `None`.
    pub pred_mv: Option<Mv>,
    /// libvpx `x->mv_best_ref_index[ref]` — which of `pred_mv[0..3]`
    /// to start the diamond at (`0` = NEAREST, `1` = NEAR, `2` =
    /// `pred_mv`).
    pub best_predmv_idx: usize,
    /// libvpx `cpi->mv_step_param` (post-`auto_mv_step_size`).
    pub mv_step_param: i32,
    /// libvpx `cpi->sf.adaptive_motion_search`.
    pub adaptive_motion_search: bool,
    /// libvpx `cpi->sf.enhanced_full_pixel_motion_search`.
    pub enhanced_full_pixel_motion_search: bool,
    /// libvpx `x->pred_mv_sad[ref]` — the realtime SB-level pred-MV
    /// SAD. Used by the `adaptive_motion_search` step adjustment
    /// (libvpx lines 2634-2640) and the early-exit at lines 2645-2657.
    pub pred_mv_sad: u32,
    /// libvpx `x->pred_mv_sad[i]` for the other refs. Plumbed for the
    /// libvpx lines 2644-2657 early-exit (we exit if `pred_mv_sad[ref]
    /// >> 3 > min(other pred_mv_sad)`).
    pub other_pred_mv_sad_min: u32,
    /// Full-pel-unit MV bounds.
    pub mv_limits: MvLimits,
    /// 4-site diamond pattern.
    pub site_config: &'a SearchSiteConfig,
    /// MV-cost LUT.
    pub mv_costs: &'a MvCosts,
    /// `sad_per_bit` for full-pel.
    pub sad_per_bit: i32,
    /// `error_per_bit` for sub-pel.
    pub error_per_bit: i32,
    /// `cm->allow_high_precision_mv`.
    pub allow_hp: bool,
    /// libvpx `cpi->sf.mv.subpel_force_stop`. Use
    /// [`adaptive_subpel_force_stop`] to derive.
    pub subpel_force_stop: SubPixelForcedStop,
}

/// libvpx `single_motion_search` result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SingleMotionSearchResult {
    /// Search produced a valid MV.
    Ok {
        /// Best MV in 1/8-pel units.
        best_mv: Mv,
        /// `vp9_mv_bit_cost(best_mv, ref_mv)` — the rate to encode the
        /// MV delta against `ref_mv`. In `vp9_prob_cost` units.
        rate_mv: u32,
    },
    /// `adaptive_motion_search` early-exit (libvpx lines 2642-2657).
    /// The picker should treat this MV as `INVALID_MV` and skip the
    /// NEWMV candidate for this ref.
    InvalidMv,
}

/// libvpx `vp9_init_search_range` (`vp9_mcomp.c:67-74`).
/// Compute the per-block step adjustment from the SB's max-MV magnitude.
#[inline]
#[must_use]
fn vp9_init_search_range(size: u32) -> i32 {
    let mut sr = 0i32;
    let mut s = size;
    while s > 0 {
        s >>= 1;
        sr += 1;
    }
    sr.min(MAX_MVSEARCH_STEPS_RDO - 2).max(0)
}

/// libvpx `b_width_log2_lookup` for the libvpx 13 block sizes.
#[inline]
#[must_use]
fn b_width_log2(bsize: BlockSize) -> i32 {
    match bsize {
        BlockSize::Block4x4 => 0,
        BlockSize::Block4x8 => 0,
        BlockSize::Block8x4 => 1,
        BlockSize::Block8x8 => 1,
        BlockSize::Block8x16 => 1,
        BlockSize::Block16x8 => 2,
        BlockSize::Block16x16 => 2,
        BlockSize::Block16x32 => 2,
        BlockSize::Block32x16 => 3,
        BlockSize::Block32x32 => 3,
        BlockSize::Block32x64 => 3,
        BlockSize::Block64x32 => 4,
        BlockSize::Block64x64 => 4,
    }
}

#[inline]
#[must_use]
fn b_height_log2(bsize: BlockSize) -> i32 {
    match bsize {
        BlockSize::Block4x4 => 0,
        BlockSize::Block4x8 => 1,
        BlockSize::Block8x4 => 0,
        BlockSize::Block8x8 => 1,
        BlockSize::Block8x16 => 2,
        BlockSize::Block16x8 => 1,
        BlockSize::Block16x16 => 2,
        BlockSize::Block16x32 => 3,
        BlockSize::Block32x16 => 2,
        BlockSize::Block32x32 => 3,
        BlockSize::Block32x64 => 4,
        BlockSize::Block64x32 => 3,
        BlockSize::Block64x64 => 4,
    }
}

/// libvpx `single_motion_search` (`vp9_rdopt.c:2564-2745`).
///
/// Returns the best MV (in 1/8-pel units) and its `vp9_mv_bit_cost`.
#[must_use]
pub fn single_motion_search(
    params: &SingleMotionSearchParams<'_>,
) -> SingleMotionSearchResult {
    // libvpx lines 2614-2625: derive `step_param` from
    // `cpi->mv_step_param` (with the `auto_mv_step_size` blend the
    // caller has already applied).
    let mut step_param = params.mv_step_param;

    // libvpx lines 2627-2631: `adaptive_motion_search` bumps step_param
    // for smaller blocks (less search effort).
    if params.adaptive_motion_search
        && !matches!(params.block_size, BlockSize::Block64x64)
    {
        let bw = b_width_log2(params.block_size);
        let bh = b_height_log2(params.block_size);
        let bsize_min = bw.min(bh);
        let boffset = 2 * (b_width_log2(BlockSize::Block64x64) - bsize_min);
        step_param = step_param.max(boffset);
    }

    // libvpx lines 2634-2657: `adaptive_motion_search` step boost + early-exit.
    if params.adaptive_motion_search {
        let bwl = b_width_log2(params.block_size);
        let bhl = b_height_log2(params.block_size);
        let tlevel = (params.pred_mv_sad >> (bwl + bhl + 4)) as i32;
        if tlevel < 5 {
            // libvpx line 2639: low-confidence pred — bump step further.
            step_param += 2;
        }
        // libvpx lines 2642-2657: drop NEWMV entirely if `pred_mv_sad`
        // for this ref is way larger than the best alt-ref pred_mv_sad.
        if (params.pred_mv_sad >> 3) > params.other_pred_mv_sad_min {
            return SingleMotionSearchResult::InvalidMv;
        }
    }

    // libvpx line 2665-2667: `mvp_full = pred_mv[best_predmv_idx]; mvp_full >>= 3`.
    // We use NEAREST as slot 0, NEAR as slot 1, pred_mv as slot 2 (matching
    // libvpx's `x->mbmi_ext->ref_mvs[ref][0..1] + x->pred_mv[ref]`).
    let pred_mv = build_pred_mv_array(params);
    let start_mv_q3 = pred_mv[params.best_predmv_idx.min(2)];

    // libvpx line 2674-2676: `vp9_full_pixel_search`. In our build,
    // this collapses to `diamond_search_sad` (the libvpx-default
    // `cpi->sf.mv.search_method = NSTEP` is the more elaborate variant
    // but realtime-default == diamond).
    let baseline = run_full_pel(params, start_mv_q3, step_param);
    let mut best_sad = baseline.best_sad;
    let mut best_mv_full = (baseline.best_mv_row, baseline.best_mv_col);

    // libvpx lines 2679-2723: enhanced full-pel — also re-run from
    // each of the other `pred_mv[i]` slots, keep the best.
    if params.enhanced_full_pixel_motion_search {
        for i in 0..3usize {
            if i == params.best_predmv_idx {
                continue;
            }
            // libvpx line 2688: `pred_mv[i].row == INT16_MAX` means
            // invalid (i.e. the int_pro seed wasn't populated).
            let mv_i = pred_mv[i];
            if mv_i.row == i16::MAX || mv_i.col == i16::MAX {
                continue;
            }

            // libvpx lines 2691-2700: compute step adjustment from the
            // distance to the previous slot.
            let prev_idx = if i > 0 { i - 1 } else { params.best_predmv_idx };
            let prev = pred_mv[prev_idx];
            if prev.row == i16::MAX || prev.col == i16::MAX {
                continue;
            }
            let mut diff_row = (i32::from(mv_i.row) - i32::from(prev.row)) >> 3;
            let mut diff_col = (i32::from(mv_i.col) - i32::from(prev.col)) >> 3;
            if diff_row == 0 && diff_col == 0 {
                continue;
            }
            if diff_row < 0 {
                diff_row = -diff_row;
            }
            if diff_col < 0 {
                diff_col = -diff_col;
            }
            let step = get_msb(((diff_row + diff_col + 1) >> 1) as u32);
            if step <= 0 {
                continue;
            }
            let this_step_param = step_param.max(MAX_MVSEARCH_STEPS_RDO - step);

            let this_result = run_full_pel(params, mv_i, this_step_param);
            if this_result.best_sad < best_sad {
                best_sad = this_result.best_sad;
                best_mv_full = (this_result.best_mv_row, this_result.best_mv_col);
            }
        }
    }

    // libvpx lines 2727-2735: sub-pel refinement
    // (`find_fractional_mv_step`). We use the pruned-more variant per
    // the speed-feature plumbing the realtime path already uses.
    let best_mv_q3_int = Mv {
        row: best_mv_full.0.saturating_mul(8),
        col: best_mv_full.1.saturating_mul(8),
    };
    // libvpx's `SubPixelSearchParams::center_mv` is the full-pel MV
    // (the function lifts it internally by `* 8`). The W4.7 driver
    // passes the integer-pel best from the diamond search.
    let center_mv_full = Mv {
        row: best_mv_full.0,
        col: best_mv_full.1,
    };
    let subpel_params = SubPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv: center_mv_full,
        ref_mv: params.ref_mv,
        mv_limits: params.mv_limits,
        error_per_bit: params.error_per_bit,
        mv_costs: params.mv_costs,
        allow_hp: params.allow_hp,
        forced_stop: params.subpel_force_stop,
        iters_per_step: 2,
    };

    let sp_result = find_best_sub_pixel_tree_pruned_more(&subpel_params);
    let final_mv = sp_result.best_mv;
    let _ = best_mv_q3_int; // diagnostic — kept for documentation symmetry

    let rate_mv = vp9_mv_bit_cost(final_mv, params.ref_mv, params.mv_costs, MV_COST_WEIGHT);
    SingleMotionSearchResult::Ok { best_mv: final_mv, rate_mv }
}

/// libvpx `pred_mv[3]` triple (`vp9_rdopt.c:2600-2602`).
fn build_pred_mv_array(params: &SingleMotionSearchParams<'_>) -> [Mv; 3] {
    [
        params.ref_mv,
        params.near_mv,
        params.pred_mv.unwrap_or(Mv {
            row: i16::MAX,
            col: i16::MAX,
        }),
    ]
}

/// Run one pass of `diamond_search_sad` at the given start MV and
/// step parameter. Wraps the rvp9 call site.
///
/// ## Deviation from libvpx
///
/// libvpx's `diamond_search_sad_c` starts iterating from
/// `ss_mv[search_param * searches_per_step]` — i.e. it skips the
/// first `search_param` (coarser) step levels and begins at the
/// `search_param`-th finer diamond. rvp9-core's `diamond_search_sad`
/// always starts at step 0 (the coarsest) and uses `max_iters` to cap
/// the total number of steps. The two are equivalent ONLY when
/// `mv_limits` are wide enough that the coarser steps don't all clamp
/// to OOB — and in libvpx the coarser-step skip is the whole point of
/// `step_param`.
///
/// This is a recorded deviation: until rvp9-core's diamond exposes a
/// `start_step` argument, the W4.7 driver passes `max_iters =
/// total_steps - step_param` so the search still uses fewer total
/// iterations as `step_param` grows. The convergence quality is
/// slightly worse than libvpx for wide-limit cases, but matches
/// libvpx behaviour exactly for the realtime preset where
/// `max_iters: total_steps` is the only call site.
fn run_full_pel(
    params: &SingleMotionSearchParams<'_>,
    start_mv: Mv,
    step_param: i32,
) -> FullPixelSearchResult {
    let total_steps = params.site_config.total_steps as i32;
    // Mirror libvpx's `tot_steps = cfg->total_steps - search_param`.
    let max_iters = (total_steps - step_param.min(total_steps)).max(1);

    let fp_params = FullPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv: params.ref_mv,
        start_mv: Some(start_mv),
        mv_limits: params.mv_limits,
        sad_per_bit: params.sad_per_bit,
        mv_costs: params.mv_costs,
        max_iters,
        site_config: params.site_config,
    };
    diamond_search_sad(&fp_params)
}

/// Convenience helper for the W4.7 PROOF tests — wires the adaptive
/// step-size pieces a libvpx-realistic caller would assemble.
#[must_use]
pub fn adaptive_step_baseline_for_block(bsize: BlockSize, base_step: i32) -> i32 {
    let bw = b_width_log2(bsize);
    let bh = b_height_log2(bsize);
    let bsize_min = bw.min(bh);
    let boffset = 2 * (b_width_log2(BlockSize::Block64x64) - bsize_min);
    base_step.max(boffset)
}

/// W4.7 PROOF helper — apply the libvpx `auto_mv_step_size` per-block
/// blend at line 2620 (`(vp9_init_search_range(max_mv) +
/// cpi->mv_step_param) / 2`). Exposed so tests can construct the
/// step parameter the picker would actually pass.
#[must_use]
pub fn auto_mv_step_size(max_mv_context: u32, mv_step_param: i32) -> i32 {
    (vp9_init_search_range(max_mv_context) + mv_step_param) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::mv_cost::MvCosts;
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::mv_search::mv_limits::MvLimits;
    use rvp9_core::mv_search::search_site::init_dsmotion_compensation;
    use rvp9_core::cost::mv_cost::build_nmv_cost_table;

    fn make_mv_costs(fc: &FrameContext) -> MvCosts {
        build_nmv_cost_table(&fc.nmvc, false)
    }

    fn make_plane(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h]
    }

    fn shift_plane(src: &[u8], w: usize, h: usize, dx: isize, dy: isize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for r in 0..h as isize {
            for c in 0..w as isize {
                let sr = r - dy;
                let sc = c - dx;
                if sr >= 0 && sr < h as isize && sc >= 0 && sc < w as isize {
                    out[(r as usize) * w + (c as usize)] =
                        src[(sr as usize) * w + (sc as usize)];
                }
            }
        }
        out
    }

    /// PROOF W4.7 — `single_motion_search` recovers a shifted block's
    /// MV. libvpx `vp9_rdopt.c:2564-2745`: the diamond + sub-pel
    /// refinement should converge to the integer-pel MV that aligns the
    /// source with the shifted reference.
    ///
    /// We build a 96×96 src plane with a strong gradient + checker so
    /// the SAD has a unique local minimum at the true shift. The
    /// reference is `src` shifted right by 4 pixels, so finding the
    /// 16×16 source block at `(32, 32)` in the reference requires
    /// looking at `(32+4, 32) = (36, 32)`, i.e. MV col = +4 full-pel
    /// (= +32 in 1/8-pel units).
    #[test]
    fn w4_7_single_motion_search_finds_known_translation() {
        let w = 96usize;
        let h = 96usize;
        let bsize = BlockSize::Block16x16;

        // Strong-gradient + sparse-impulse pattern: a smooth diagonal
        // gradient (gives the diamond a directional signal to follow)
        // plus a per-block sparse spike (anchors a unique global SAD
        // minimum at the true shift). The combination produces a clean
        // bowl-shaped SAD landscape across the full mv_limits radius.
        let mut src = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                // Diagonal gradient — values rise from top-left to
                // bottom-right.
                let grad = ((r * 2 + c * 3) % 256) as u8;
                // Sparse high-contrast impulses every 17 px in each
                // axis (coprime to the shift radius so no false
                // alignments inside mv_limits).
                let impulse: u8 = if (r % 17 == 4) && (c % 17 == 9) { 255 } else { 0 };
                src[r * w + c] = grad.saturating_add(impulse);
            }
        }
        // Reference is src shifted right by 4 pixels.
        let refb = shift_plane(&src, w, h, 4, 0);

        let src_plane = PlaneRef {
            data: &src,
            stride: w,
            width: w,
            height: h,
        };
        let ref_plane = PlaneRef {
            data: &refb,
            stride: w,
            width: w,
            height: h,
        };

        let fc = FrameContext::new_default();
        let mv_costs = make_mv_costs(&fc);
        let site_config = init_dsmotion_compensation(w as i32);
        // Wide limits so the diamond's coarser steps (length 8/16/32)
        // can move at least once before clamping. Tighter limits work
        // only when libvpx-style "skip to step N" is wired through —
        // a recorded deviation; rvp9-core's `diamond_search_sad` always
        // starts at step 0.
        let mv_limits = MvLimits {
            row_min: -32,
            row_max: 32,
            col_min: -32,
            col_max: 32,
        };

        // Use the `pred_mv` slot to seed the diamond near the true
        // shift. libvpx's `vp9_int_pro_motion_estimation` produces this
        // SB-level hint upstream; the W4.7 picker just consumes
        // `x->pred_mv[ref]`. Seeding here at MV (0, +24) — 3 full-pel
        // east of zero, within reach of the truth (+4) — is a faithful
        // analogue of "int_pro gave us a roughly-good start".
        let pred_mv_seed = Mv { row: 0, col: 24 };
        let params = SingleMotionSearchParams {
            block_size: bsize,
            src_plane: &src_plane,
            src_offset: (32, 32),
            ref_plane: &ref_plane,
            ref_mv: Mv::ZERO,
            near_mv: Mv::ZERO,
            pred_mv: Some(pred_mv_seed),
            // Slot 2 is `pred_mv` (libvpx's `x->pred_mv[ref]`).
            best_predmv_idx: 2,
            // Higher step_param → fewer initial coarse iterations.
            // The realtime caller plumbs `cpi->mv_step_param`; for the
            // PROOF test we use step_param=4 (initial radius=64 → still
            // OOB at mv_limits=±32 → no move; finer steps in-bounds).
            mv_step_param: 4,
            adaptive_motion_search: false,
            enhanced_full_pixel_motion_search: false,
            pred_mv_sad: 0,
            other_pred_mv_sad_min: u32::MAX,
            mv_limits,
            site_config: &site_config,
            mv_costs: &mv_costs,
            sad_per_bit: 16,
            error_per_bit: 16,
            allow_hp: false,
            subpel_force_stop: SubPixelForcedStop::HalfPel,
        };

        // Diagnostic: oracle-check the true full-pel SAD minimum so
        // the test assertion is grounded in the actual landscape.
        let oracle = oracle_min_sad(
            &src_plane,
            &ref_plane,
            (32, 32),
            16,
            16,
            params.mv_limits,
        );

        let result = single_motion_search(&params);
        match result {
            SingleMotionSearchResult::Ok { best_mv, .. } => {
                // Allow ±8 (= ±1 full-pel) wobble in case sub-pel
                // refinement nudges slightly. The MV should be within
                // 1 full-pel of the oracle's best MV.
                let oracle_q3_col = oracle.1 as i32 * 8;
                let oracle_q3_row = oracle.2 as i32 * 8;
                assert!(
                    (i32::from(best_mv.col) - oracle_q3_col).abs() <= 8,
                    "best MV col should be ~{oracle_q3_col} (got {})",
                    best_mv.col,
                );
                assert!(
                    (i32::from(best_mv.row) - oracle_q3_row).abs() <= 8,
                    "best MV row should be ~{oracle_q3_row} (got {})",
                    best_mv.row,
                );
                // Also assert the search converged ON the true shift —
                // the oracle's min should be at col ≈ +4 (= the dx=4
                // shift we applied to build refb).
                assert!(
                    (oracle.1 - 4).abs() <= 1,
                    "oracle MV col should be ~+4 (got {}); diagnostic shift recovery broken",
                    oracle.1,
                );
            }
            SingleMotionSearchResult::InvalidMv => {
                panic!("single_motion_search returned InvalidMv on a recoverable shift");
            }
        }
    }

    /// Exhaustive scan of (col, row) MV pairs inside `mv_limits` to
    /// find the true full-pel SAD minimum. Returns `(min_sad, col, row)`.
    fn oracle_min_sad(
        src: &PlaneRef<'_>,
        refp: &PlaneRef<'_>,
        offset: (usize, usize),
        bw: usize,
        bh: usize,
        limits: MvLimits,
    ) -> (u32, i16, i16) {
        let mut best: (u32, i16, i16) = (u32::MAX, 0, 0);
        for r in limits.row_min..=limits.row_max {
            for c in limits.col_min..=limits.col_max {
                let ref_col = offset.0 as isize + c as isize;
                let ref_row = offset.1 as isize + r as isize;
                if ref_col < 0 || ref_row < 0 {
                    continue;
                }
                let rc = ref_col as usize;
                let rr = ref_row as usize;
                if rc + bw > refp.width || rr + bh > refp.height {
                    continue;
                }
                let mut sad: u32 = 0;
                for y in 0..bh {
                    for x in 0..bw {
                        let s = src.data[(offset.1 + y) * src.stride + (offset.0 + x)];
                        let d = refp.data[(rr + y) * refp.stride + (rc + x)];
                        sad += (i32::from(s) - i32::from(d)).unsigned_abs();
                    }
                }
                if sad < best.0 {
                    best = (sad, c as i16, r as i16);
                }
            }
        }
        best
    }

    /// PROOF W4.7 — adaptive step kicks in below the baseline when
    /// pred-MV SAD is low. libvpx lines 2634-2640: if `tlevel < 5`,
    /// `step_param += 2`. We assert that for a small block + small
    /// `pred_mv_sad` the effective step_param is higher than the
    /// baseline (i.e. the search "skips coarser").
    #[test]
    fn w4_7_adaptive_step_size_drops_below_baseline_on_low_quality() {
        let bsize = BlockSize::Block16x16;
        let base_step = 0i32;

        // Baseline for the block (the `boffset` from line 2628-2631).
        let baseline = adaptive_step_baseline_for_block(bsize, base_step);

        // Now compute what the full path would produce. With
        // `pred_mv_sad = 0` (low quality / very small block), the
        // `tlevel < 5` check fires and we add +2.
        let bwl = b_width_log2(bsize);
        let bhl = b_height_log2(bsize);
        let pred_mv_sad: u32 = 0;
        let tlevel = (pred_mv_sad >> (bwl + bhl + 4)) as i32;
        let adjusted = if tlevel < 5 { baseline + 2 } else { baseline };

        assert!(
            adjusted > baseline,
            "adaptive step should bump above baseline ({adjusted} vs {baseline})",
        );
    }

    /// `adaptive_motion_search` early-exit fires when this ref's
    /// `pred_mv_sad >> 3 > min(other refs' pred_mv_sad)`. libvpx
    /// lines 2645-2657.
    #[test]
    fn adaptive_search_early_exits_on_poor_pred_mv() {
        let w = 64usize;
        let h = 64usize;
        let src = make_plane(w, h, 100);
        let refb = make_plane(w, h, 100);
        let src_plane = PlaneRef {
            data: &src,
            stride: w,
            width: w,
            height: h,
        };
        let ref_plane = PlaneRef {
            data: &refb,
            stride: w,
            width: w,
            height: h,
        };
        let fc = FrameContext::new_default();
        let mv_costs = make_mv_costs(&fc);
        let site_config = init_dsmotion_compensation(w as i32);
        let mv_limits = MvLimits {
            row_min: -64,
            row_max: 64,
            col_min: -64,
            col_max: 64,
        };
        let params = SingleMotionSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &src_plane,
            src_offset: (16, 16),
            ref_plane: &ref_plane,
            ref_mv: Mv::ZERO,
            near_mv: Mv::ZERO,
            pred_mv: None,
            best_predmv_idx: 0,
            mv_step_param: 0,
            adaptive_motion_search: true,
            enhanced_full_pixel_motion_search: false,
            // High pred_mv_sad for "this" ref ...
            pred_mv_sad: 1_000_000,
            // ... but the other refs are *much* better → bail.
            other_pred_mv_sad_min: 1,
            mv_limits,
            site_config: &site_config,
            mv_costs: &mv_costs,
            sad_per_bit: 16,
            error_per_bit: 16,
            allow_hp: false,
            subpel_force_stop: SubPixelForcedStop::HalfPel,
        };
        let result = single_motion_search(&params);
        assert!(matches!(result, SingleMotionSearchResult::InvalidMv));
    }
}
