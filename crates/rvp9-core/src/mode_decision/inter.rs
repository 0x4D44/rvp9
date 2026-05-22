//! Inter mode search — libvpx-faithful realtime port of
//! `vp9_pick_inter_mode`.
//!
//! Tries `{ZEROMV, NEARESTMV, NEWMV} × {LASTFRAME}` for a single
//! luma block, performs sub-pel reference fetch + forward DCT +
//! quantize + dequantize + inverse DCT, and picks the mode minimising
//! `D + lambda * R` where:
//!
//! - `D` is the sum-of-squared-error between the source and the
//!   reconstruction (post-recon SSE, the same metric as F8.0).
//! - `R` is the rate cost of `(inter_mode, ref_frame, mv)` in 1/512-bit
//!   `vp9_prob_cost` units. For `NEWMV` we add the MV-bit cost
//!   relative to the `NEAREST` reference MV; for `ZEROMV` / `NEARESTMV`
//!   only the inter-mode rate is charged.
//!
//! Reference: `vp9/encoder/vp9_pickmode.c:1300-1700`
//! (`vp9_pick_inter_mode` realtime variant).
//!
//! ## libvpx divergences (recorded)
//!
//! Path-A v1 deliberately ships a *minimal but libvpx-shaped* picker.
//! The deltas relative to the full libvpx realtime variant are:
//!
//! - **Ref frames**: LASTFRAME only. libvpx iterates LAST / GOLDEN /
//!   ALTREF and the ref-frame search is gated by `usable_ref_frame`
//!   (`vp9_pickmode.c:1429-1452`); we collapse to single-ref single-
//!   reference-frame to keep v1 tractable. GOLDEN / ALTREF land later.
//! - **Inter modes**: `{NEARESTMV, ZEROMV, NEWMV}` only. `NEARMV` is
//!   skipped — it shares the candidate-list mechanics of `NEARESTMV`
//!   but uses the second-nearest MV; the supervisor's spec explicitly
//!   defers it to v2.
//! - **No compound prediction**: single-ref blocks only. libvpx's
//!   `vp9_pickmode.c` realtime path also skips compound (it's an RDO
//!   speed feature gated by `usable_ref_frame > 1`).
//! - **No interpolation-filter search**: we always pick `EightTap`
//!   (libvpx's `INTERP_FILTER` default `EIGHTTAP = 0`). The decoder
//!   will read `interp_filter = EightTap` from the per-block header
//!   bits the encoder writes; switchable filter selection is an
//!   F8.x-deferred feature.
//! - **No TX-size search**: every block uses `max_txsize(block_size)`
//!   capped at `Tx32x32` (libvpx's `tx_size_search` in the realtime
//!   path is a per-block CBR speed feature; we always pick the maximum
//!   transform size that fits in the block).
//! - **Distortion metric**: post-recon SSE — the same divergence F8.0
//!   already documents. libvpx-realtime's `block_yrd`
//!   (`vp9_pickmode.c:1056-1110`) is a transform-coefficient-domain
//!   model that approximates SSE via the quantized-coefficient
//!   residual energy plus a quantizer-tuned scale; F8.1 computes the
//!   exact pixel-domain SSE after the forward / inverse round-trip.
//! - **No `skip_txfm[]` early-exit**: libvpx maintains a per-plane
//!   skip-transform short-circuit (re-uses the previous block's
//!   "all coefficients quantized to zero" flag) — F8.1 always runs the
//!   full round-trip so the SSE is exact.
//! - **Block sizes**: support `Block16x16`, `Block32x32`, `Block64x64`.
//!   Block16x16 / Block32x32 use a single transform per block;
//!   Block64x64 decomposes into 4 × `Tx32x32` transforms tiled in
//!   raster order. Smaller sizes can land later when the partition
//!   picker (F8.2) needs them.
//!
//! ## Algorithm
//!
//! For each `(mode, mv)` candidate in
//!
//! ```text
//! [(ZEROMV,  Mv::ZERO),
//!  (NEARESTMV, mv_candidates.nearest),
//!  (NEWMV,  diamond + sub-pel from `mv_candidates.nearest`)]
//! ```
//!
//! 1. Fetch the reference block at `src_offset + (mv >> 3)` with sub-pel
//!    interpolation phase `(mv & 7) << 1` (Q3 → Q4 conversion).
//! 2. Compute residual `src - pred`.
//! 3. Forward DCT (`fdct{8,16,32}x{8,16,32}` — DCT_DCT only since inter
//!    blocks never use ADST per libvpx).
//! 4. Forward quant (`vp9_quantize_b` / `vp9_quantize_b_32x32`).
//! 5. Reconstruct via inverse DCT + add to pred.
//! 6. Compute SSE between source and reconstruction — this is `D`.
//! 7. Rate `R = inter_mode_costs[ctx][mode.idx()] +
//!    (if NEWMV: mv_cost(diff, mv_costs))`.
//! 8. `rd_cost = ROUND_POWER_OF_TWO(R * lambda, 9) + (D << 7)` —
//!    libvpx `RDCOST` macro at `vp9_rd.h:29-30`.

use crate::bitstream::header::FrameKind;
use crate::block::{BlockSize, max_txsize};
use crate::cost::mv_cost::{MvCosts, mv_cost};
use crate::cost::{FrameTypeForRd, InterModeCosts, compute_rd_mult};
use crate::entropy::default_probs::INTER_MODE_CONTEXTS;
use crate::inter::filter::{EIGHTTAP_KERNEL, FILTER_TAPS};
use crate::mode_decision::rd_cost;
use crate::mode_info::{InterMode, Mv};
use crate::mv_search::full_pixel::{
    FullPixelSearchParams, PlaneRef, diamond_search_sad,
};
use crate::mv_search::mv_limits::MvLimits;
use crate::mv_search::search_site::SearchSiteConfig;
use crate::mv_search::sub_pixel::{
    SubPixelForcedStop, SubPixelSearchParams, find_best_sub_pixel_tree_pruned_more,
};
use crate::quant::forward::{
    PlaneQuantizers, scan_for_forward, vp9_quantize_b, vp9_quantize_b_32x32,
};
use crate::transforms::forward::{fdct8x8, fdct16x16, fdct32x32};
use crate::transforms::inverse::{idct8x8_64_add, idct16x16_256_add, idct32x32_1024_add};
use crate::transforms::scans::scan_for;
use crate::transforms::{TxSize, TxType};

/// `sad_per_bit` heuristic for full-pel MV search — libvpx's
/// `sadperbit16` is `2 * x->errorperbit` for the realtime path, where
/// `errorperbit ≈ rd_mult >> RD_EPB_SHIFT` (`vp9_rd.c:294-302`,
/// `vp9_rdopt.h:42`). For Path-A v1 we use a simpler constant that
/// keeps the cost balanced against the SAD magnitudes the diamond
/// search compares: an empirical small positive integer that lets the
/// dist-attractor win on shifted-ref input. Tuning lives later.
const DEFAULT_SAD_PER_BIT: i32 = 16;

/// `error_per_bit` for sub-pel refinement, analogous to
/// [`DEFAULT_SAD_PER_BIT`] but in the variance domain. libvpx derives
/// it as `(rd_mult * 1) >> RD_EPB_SHIFT` (`vp9_rd.c:341-345`); we
/// likewise pin a small constant to ensure sub-pel pulls toward the
/// minimum-variance position without being drowned out by MV cost.
const DEFAULT_ERROR_PER_BIT: i32 = 16;

/// MV candidate pair fed to [`pick_inter_mode_luma`].
///
/// Mirrors libvpx's `MV_REFERENCE_FRAME`-indexed `nearest_mv` /
/// `near_mv` slots from `find_mv_refs` after `lower_mv_precision`. Both
/// are in 1/8-pel units (the standard VP9 MV unit).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MvCandidates {
    /// `mv_ref_list[0]` — the nearest decoded MV. Used as the
    /// reference MV for `NEARESTMV` and as the search center for
    /// `NEWMV`.
    pub nearest: Mv,
    /// `mv_ref_list[1]` — the second-nearest decoded MV. Stored for
    /// completeness; F8.1 doesn't evaluate `NEARMV` so this slot is
    /// currently unused. Populated for forward compatibility with v2.
    pub near: Mv,
}

/// Inputs to [`pick_inter_mode_luma`].
///
/// Most fields mirror [`crate::mode_decision::intra::IntraModeSearchParams`]
/// — same block-size / source-plane / quantizer / qindex / frame-kind
/// plumbing — with the addition of:
///
/// - `ref_plane` — LASTFRAME's reconstructed Y plane.
/// - `mv_candidates` — `find_mv_refs` output (NEAREST + NEAR).
/// - `mv_limits` / `site_config` — F7.0 full-pel-search bounds + 4-site
///   diamond config. Caller computes these via
///   [`crate::mv_search::set_mv_search_range`] /
///   [`crate::mv_search::init_dsmotion_compensation`].
/// - `mv_costs` — F6.1 MV-bit-cost LUT.
/// - `inter_mode_costs` — F6.1 inter-mode cost table indexed by
///   `[ctx][mode_idx]` where `mode_idx = mode as u8 - NEARESTMV as u8`
///   (i.e. 0..=3 = NEAREST / NEAR / ZERO / NEW). Build via
///   [`crate::cost::inter_mode_costs`].
/// - `inter_mode_ctx` — the per-block inter-mode context
///   (`0..INTER_MODE_CONTEXTS`) computed by
///   [`crate::inter::get_mode_context`].
/// - `allow_hp` — frame-header `allow_high_precision_mv` bit.
pub struct InterModeSearchParams<'a> {
    /// Block size being decided. Drives the transform size via
    /// [`max_txsize`] and the SAD / sub-pel-variance kernel choice.
    pub block_size: BlockSize,
    /// Source luma plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block in plane samples (`(x, y)`).
    pub src_offset: (usize, usize),
    /// LASTFRAME's reconstructed luma plane. The picker reads from
    /// this plane at `src_offset + (mv >> 3)` plus a `±3`-pixel
    /// halo to cover the 8-tap interpolator footprint.
    pub ref_plane: &'a PlaneRef<'a>,
    /// `find_mv_refs` output — NEAREST and NEAR ref MVs in 1/8-pel
    /// units. NEAREST is used as the search center for NEWMV and as
    /// the reference MV for NEARESTMV's reconstruction.
    pub mv_candidates: &'a MvCandidates,
    /// Full-pel-unit MV bounds (from
    /// [`crate::mv_search::set_mv_search_range`]).
    pub mv_limits: MvLimits,
    /// 4-site diamond config (from
    /// [`crate::mv_search::init_dsmotion_compensation`], computed
    /// per-stride at the caller).
    pub site_config: &'a SearchSiteConfig,
    /// MV-bit-cost LUT (from F6.1 `build_nmv_cost_table`).
    pub mv_costs: &'a MvCosts,
    /// Inter-mode rate-cost table from
    /// [`crate::cost::inter_mode_costs`]. Indexed
    /// `[inter_mode_ctx][mode.idx()]`.
    pub inter_mode_costs: &'a InterModeCosts,
    /// Inter-mode context index — `0..INTER_MODE_CONTEXTS`. Computed
    /// by [`crate::inter::get_mode_context`].
    pub inter_mode_ctx: usize,
    /// Y-plane forward-quantizer (from
    /// `init_plane_quantizers(q_index, QuantizerDeltas::Y)`).
    pub quantizer: &'a PlaneQuantizers,
    /// Frame base qindex (0..255) — drives lambda.
    pub q_index: u8,
    /// `cm->allow_high_precision_mv` — gates 1/8-pel sub-pel refinement.
    pub allow_hp: bool,
    /// Frame kind — selects the RD-multiplier formula.
    pub frame_kind: FrameKind,
    /// SB-level NEWMV diamond-search seed (mirror of libvpx's
    /// `x->pred_mv[LAST_FRAME]`, populated by
    /// `vp9_int_pro_motion_estimation` per SB upstream of
    /// `vp9_pick_inter_mode`).
    ///
    /// When `Some`, this MV (in Q3 / 1/8-pel units) replaces
    /// `mv_candidates.nearest` as the diamond-search center for the
    /// NEWMV candidate. The NEWMV rate cost is still computed
    /// relative to `mv_candidates.nearest` (it's the ref MV the
    /// bitstream encodes the delta against).
    ///
    /// When `None`, falls back to seeding at
    /// `mv_candidates.nearest` (the pre-A3' behaviour).
    pub newmv_seed: Option<Mv>,
}

/// Result of [`pick_inter_mode_luma`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InterModeSearchResult {
    /// Winning inter mode.
    pub best_mode: InterMode,
    /// Winning MV (1/8-pel units). For `ZEROMV` this is `Mv::ZERO`;
    /// for `NEARESTMV` it's `mv_candidates.nearest`; for `NEWMV` it's
    /// the sub-pel-refined output of the F7.0 search.
    pub best_mv: Mv,
    /// `RDCOST(lambda, rate, distortion)` for the winning candidate.
    pub best_rd_cost: i64,
    /// Distortion (SSE between source and reconstruction).
    pub best_distortion: u32,
    /// Rate cost (mode + MV bits, in `vp9_prob_cost` units).
    pub best_rate: u32,
}

/// Run the libvpx-style inter-mode RD search for one luma block.
///
/// Evaluates `{ZEROMV, NEARESTMV, NEWMV}` × `{LASTFRAME}` and returns
/// the lowest-RD-cost winner. See module docs for the deliberate
/// divergences from libvpx's full realtime path.
#[must_use]
pub fn pick_inter_mode_luma(params: &InterModeSearchParams<'_>) -> InterModeSearchResult {
    debug_assert!(
        params.inter_mode_ctx < INTER_MODE_CONTEXTS,
        "inter_mode_ctx {} out of [0, {INTER_MODE_CONTEXTS})",
        params.inter_mode_ctx,
    );

    let lambda = compute_rd_mult(
        i32::from(params.q_index),
        frame_type_for_rd(params.frame_kind),
    );

    // Candidate 1: ZEROMV.
    let zero = evaluate_candidate(
        params,
        InterMode::ZeroMv,
        Mv::ZERO,
        params.mv_candidates.nearest,
        lambda,
    );

    // Candidate 2: NEARESTMV.
    let nearest = evaluate_candidate(
        params,
        InterMode::NearestMv,
        params.mv_candidates.nearest,
        params.mv_candidates.nearest,
        lambda,
    );

    // Candidate 3: NEWMV — full-pel diamond seeded at NEAREST, then
    // sub-pel tree-pruned-more refinement.
    let new_mv = run_newmv_search(params);
    let new = evaluate_candidate(
        params,
        InterMode::NewMv,
        new_mv,
        params.mv_candidates.nearest,
        lambda,
    );

    // Argmin.
    let mut best = zero;
    if nearest.best_rd_cost < best.best_rd_cost {
        best = nearest;
    }
    if new.best_rd_cost < best.best_rd_cost {
        best = new;
    }
    best
}

/// Diagnostic: same search as [`pick_inter_mode_luma`] but returns the
/// per-candidate breakdown in candidate order
/// `[ZEROMV, NEARESTMV, NEWMV]`. Used by the tests to sanity-check
/// the RD-ranking for the shifted-ref oracle.
#[must_use]
pub fn pick_inter_mode_luma_all_costs(
    params: &InterModeSearchParams<'_>,
) -> [InterModeSearchResult; 3] {
    let lambda = compute_rd_mult(
        i32::from(params.q_index),
        frame_type_for_rd(params.frame_kind),
    );
    let zero = evaluate_candidate(
        params,
        InterMode::ZeroMv,
        Mv::ZERO,
        params.mv_candidates.nearest,
        lambda,
    );
    let nearest = evaluate_candidate(
        params,
        InterMode::NearestMv,
        params.mv_candidates.nearest,
        params.mv_candidates.nearest,
        lambda,
    );
    let new_mv = run_newmv_search(params);
    let new = evaluate_candidate(
        params,
        InterMode::NewMv,
        new_mv,
        params.mv_candidates.nearest,
        lambda,
    );
    [zero, nearest, new]
}

/// Clamp a Q3 (1/8-pel) MV into the picker's full-pel
/// `mv_limits` envelope by capping `mv.col >> 3` and `mv.row >> 3`
/// to `[col_min, col_max] × [row_min, row_max]`. Used to keep the
/// SB-level int_pro seed inside the per-leaf diamond's reachable
/// range before threading it into [`diamond_search_sad`].
fn clamp_mv_to_limits(mv: Mv, limits: &MvLimits) -> Mv {
    // mv.col, mv.row are i16 in Q3; convert to full-pel and clamp.
    // Reconstruct a Q3 MV that lies on the clamped full-pel grid
    // (the diamond search operates in full-pel internally and
    // truncates the Q3 input via `>> 3`, so feeding back the
    // clamped Q3 == clamped_fp << 3 is exact).
    let col_fp = (i32::from(mv.col) >> 3).clamp(limits.col_min, limits.col_max);
    let row_fp = (i32::from(mv.row) >> 3).clamp(limits.row_min, limits.row_max);
    Mv {
        col: (col_fp * 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
        row: (row_fp * 8).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
    }
}

/// Run the F7.0 full-pel diamond + sub-pel refinement for NEWMV.
/// Returns the resulting MV in 1/8-pel units. The full-pel search is
/// seeded at `params.newmv_seed` when provided (libvpx's
/// `x->pred_mv[]` from `vp9_int_pro_motion_estimation`), otherwise
/// at `mv_candidates.nearest`. Either way, the seed is clamped to
/// the picker's `mv_limits` envelope before being fed into the
/// diamond search.
fn run_newmv_search(params: &InterModeSearchParams<'_>) -> Mv {
    let center_mv = clamp_mv_to_limits(
        params.newmv_seed.unwrap_or(params.mv_candidates.nearest),
        &params.mv_limits,
    );
    let full = diamond_search_sad(&FullPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv,
        mv_limits: params.mv_limits,
        sad_per_bit: DEFAULT_SAD_PER_BIT,
        mv_costs: params.mv_costs,
        max_iters: params.site_config.total_steps as i32,
        site_config: params.site_config,
    });

    // Hand off to sub-pel. The sub-pel search only supports the 5
    // square block sizes (Block4x4 / 8 / 16 / 32 / 64); F8.1 v1's
    // supported sizes (Block16x16 / Block32x32 / Block64x64) are all
    // in that set.
    let sub = find_best_sub_pixel_tree_pruned_more(&SubPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv: Mv {
            row: full.best_mv_row,
            col: full.best_mv_col,
        },
        ref_mv: params.mv_candidates.nearest,
        mv_limits: params.mv_limits,
        error_per_bit: DEFAULT_ERROR_PER_BIT,
        mv_costs: params.mv_costs,
        allow_hp: params.allow_hp,
        forced_stop: SubPixelForcedStop::EighthPel,
    });
    sub.best_mv
}

/// Evaluate one `(mode, mv)` candidate: fetch the prediction with sub-pel
/// interpolation, run the forward/inverse transform round-trip,
/// compute SSE + rate cost + RD cost.
///
/// `ref_mv` is the reference MV used for the NEWMV rate-cost
/// calculation (only relevant when `mode == InterMode::NewMv`).
fn evaluate_candidate(
    params: &InterModeSearchParams<'_>,
    mode: InterMode,
    mv: Mv,
    ref_mv: Mv,
    lambda: u32,
) -> InterModeSearchResult {
    let bw = usize::from(params.block_size.width());
    let bh = usize::from(params.block_size.height());

    // Build the prediction in a per-block scratch buffer.
    let mut pred = [0u8; 64 * 64];
    build_inter_prediction(
        params.ref_plane,
        params.src_offset.0,
        params.src_offset.1,
        mv,
        bw,
        bh,
        &mut pred,
        bw, // pred_stride
    );

    // Compute the reconstruction. For Block16x16 / Block32x32 the
    // block matches its max_txsize and we do one big transform.
    // For Block64x64 we tile 4 × 32x32 transforms.
    let tx_size = max_txsize(params.block_size);
    let mut recon = [0u8; 64 * 64];
    recon[..bw * bh].copy_from_slice(&pred[..bw * bh]);
    reconstruct_via_round_trip(params, &pred, &mut recon, bw, bh, tx_size);

    // Distortion = SSE(src, recon). For block sizes ≤ 64×64 the SSE
    // fits in u32 (max 64*64*255^2 = 266_342_400 < 2^32).
    let sse = sse_block(params.src_plane, params.src_offset, &recon, bw, bw, bh);
    let dist = sse.min(u32::MAX as u64) as u32;
    // A0 diagnostic 2: dump full 16-pixel sample comparison for an
    // arbitrary leaf when RVP9_DIAG_EVAL_FULL is set.
    if std::env::var("RVP9_DIAG_EVAL_FULL").is_ok() {
        #[allow(clippy::print_stderr)]
        {
            let (sx, sy) = params.src_offset;
            // Print src row 0 cols 0..16 vs recon row 0 cols 0..16.
            let row_off = sy * params.src_plane.stride + sx;
            let src_row = &params.src_plane.data[row_off..row_off + 16];
            let recon_row = &recon[..16];
            eprintln!("      FULL mode={:?} mv=({},{}) src_row0={:?} recon_row0={:?} bw={} sse={}",
                mode, mv.row, mv.col, src_row, recon_row, bw, sse);
            // Also row 5 (middle of small block).
            let row5_off = (sy + 5) * params.src_plane.stride + sx;
            let src_row5 = &params.src_plane.data[row5_off..row5_off + 16];
            let recon_row5 = &recon[5 * bw..5 * bw + 16];
            eprintln!("           src_row5={:?} recon_row5={:?}", src_row5, recon_row5);
        }
    }
    // A0 diagnostic — print per-candidate src/pred/recon comparison.
    if std::env::var("RVP9_DIAG_EVAL").is_ok() {
        #[allow(clippy::print_stderr)]
        {
            let (sx, sy) = params.src_offset;
            let src0 = params.src_plane.data[sy * params.src_plane.stride + sx];
            let src1 = params.src_plane.data[sy * params.src_plane.stride + sx + 1];
            let pred0 = pred[0];
            let pred1 = pred[1];
            let recon0 = recon[0];
            let recon1 = recon[1];
            let ref_x = (sx as i32) + (i32::from(mv.col) >> 3);
            let ref_y = (sy as i32) + (i32::from(mv.row) >> 3);
            let ref0 = if ref_x >= 0 && ref_y >= 0 {
                params.ref_plane.data[(ref_y as usize) * params.ref_plane.stride + (ref_x as usize)]
            } else { 0 };
            eprintln!(
                "      eval mode={:?} mv=({},{}) src[0..2]=({},{}) ref[0]={} pred[0..2]=({},{}) recon[0..2]=({},{}) sse={}",
                mode, mv.row, mv.col, src0, src1, ref0, pred0, pred1, recon0, recon1, sse
            );
        }
    }

    // Rate. Mode cost is a table lookup; NEWMV additionally charges
    // the MV-bit cost (libvpx `vp9_mv_bit_cost`).
    let mode_rate = params.inter_mode_costs[params.inter_mode_ctx][mode.idx()]
        .max(0) as u32;
    let mv_rate = if matches!(mode, InterMode::NewMv) {
        let diff_row = mv.row.wrapping_sub(ref_mv.row);
        let diff_col = mv.col.wrapping_sub(ref_mv.col);
        mv_cost(diff_row, diff_col, params.mv_costs).max(0) as u32
    } else {
        0
    };
    let rate = mode_rate.saturating_add(mv_rate);

    let this_rd = rd_cost(lambda, rate, dist);

    InterModeSearchResult {
        best_mode: mode,
        best_mv: mv,
        best_rd_cost: this_rd,
        best_distortion: dist,
        best_rate: rate,
    }
}

/// Build a `bw × bh` inter prediction at `src_offset + (mv >> 3)` in
/// the reference plane, with sub-pel interpolation determined by
/// `(mv.col & 7, mv.row & 7) << 1` (Q3 → Q4).
///
/// The output is written to `dst[..bh * dst_stride]`. For zero sub-pel
/// (i.e. `mv & 7 == 0` on both axes) this calls
/// [`crate::inter::filter::convolve_copy`]; otherwise it dispatches to
/// the horiz-only / vert-only / 2-D `convolve8*` kernels per libvpx
/// `inter_predictor` (`vp9_reconinter.h:23-32`).
///
/// **No border extension**: callers must ensure that
/// `src_offset + (mv >> 3) ± 3` stays inside the reference plane. The
/// MV-limits machinery (`set_mv_search_range`) is the encoder's
/// border-extend equivalent. F8.1 v1 doesn't run on edge blocks
/// directly; the partition picker (F8.2) will choose conservative
/// MV bounds for edge blocks.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
fn build_inter_prediction(
    ref_plane: &PlaneRef<'_>,
    src_x: usize,
    src_y: usize,
    mv: Mv,
    bw: usize,
    bh: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    // Integer-pel offset and sub-pel phase.
    // mv is in 1/8-pel units (Q3); the eight-tap filter uses
    // SUBPEL_BITS=4 phases (Q4), so the phase is `(mv & 7) * 2`.
    let int_row = (mv.row as i32) >> 3;
    let int_col = (mv.col as i32) >> 3;
    let subpel_y_q4 = ((mv.row as i32) & 7) << 1;
    let subpel_x_q4 = ((mv.col as i32) & 7) << 1;
    debug_assert!(
        (0..16).contains(&subpel_x_q4),
        "subpel_x_q4 out of range: {subpel_x_q4}",
    );
    debug_assert!(
        (0..16).contains(&subpel_y_q4),
        "subpel_y_q4 out of range: {subpel_y_q4}",
    );

    let ref_x = src_x as isize + int_col as isize;
    let ref_y = src_y as isize + int_row as isize;

    // The 8-tap kernel needs 3 columns / rows of padding on the
    // top-left side (libvpx convention: `src -= SUBPEL_TAPS/2 - 1`).
    // When subpel is zero on an axis, no padding is needed on that
    // axis, so we shrink the halo by branching on the phase.
    let x_nonzero = subpel_x_q4 != 0;
    let y_nonzero = subpel_y_q4 != 0;

    // Compute the read-window's top-left.
    let pad_x = if x_nonzero { (FILTER_TAPS / 2 - 1) as isize } else { 0 };
    let pad_y = if y_nonzero { (FILTER_TAPS / 2 - 1) as isize } else { 0 };
    let win_x = ref_x - pad_x;
    let win_y = ref_y - pad_y;

    // Bounds check (debug-only): the caller's MV limits should keep
    // the read window inside the reference plane. We don't
    // border-extend here.
    debug_assert!(
        win_x >= 0 && win_y >= 0,
        "inter pred read window out of bounds: ({win_x}, {win_y})",
    );
    let win_h = bh
        + if y_nonzero { FILTER_TAPS - 1 } else { 0 };
    let win_w = bw
        + if x_nonzero { FILTER_TAPS - 1 } else { 0 };
    debug_assert!(
        (win_x as usize) + win_w <= ref_plane.width,
        "inter pred read window beyond ref width: x={win_x} + w={win_w} > {}",
        ref_plane.width,
    );
    debug_assert!(
        (win_y as usize) + win_h <= ref_plane.height,
        "inter pred read window beyond ref height: y={win_y} + h={win_h} > {}",
        ref_plane.height,
    );

    let src_start = (win_y as usize) * ref_plane.stride + (win_x as usize);
    let src_slice = &ref_plane.data[src_start..];

    // Dispatch the kernel matching the (x_nonzero, y_nonzero) pair.
    // EightTap filter, F8.1 v1 hard-pins the default.
    let kernel = &EIGHTTAP_KERNEL;
    let filter_x = &kernel[subpel_x_q4 as usize];
    let filter_y = &kernel[subpel_y_q4 as usize];

    match (x_nonzero, y_nonzero) {
        (false, false) => {
            crate::inter::filter::convolve_copy(
                src_slice, ref_plane.stride, dst, dst_stride, bw, bh,
            );
        }
        (true, false) => {
            crate::inter::filter::convolve8_horiz(
                src_slice, ref_plane.stride, dst, dst_stride, filter_x, bw, bh,
            );
        }
        (false, true) => {
            crate::inter::filter::convolve8_vert(
                src_slice, ref_plane.stride, dst, dst_stride, filter_y, bw, bh,
            );
        }
        (true, true) => {
            crate::inter::filter::convolve8(
                src_slice,
                ref_plane.stride,
                dst,
                dst_stride,
                filter_x,
                filter_y,
                bw,
                bh,
            );
        }
    }
}

/// Run the forward/inverse-transform round-trip on `residual = src -
/// pred` and add the dequantized residual into `recon`. The caller
/// must initialise `recon` to `pred` (the inverse kernels are
/// `_add`-style: they write `recon += idct(dq)`).
///
/// For Block16x16 / Block32x32 this is one transform per block; for
/// Block64x64 we tile 4 × `Tx32x32` transforms in raster order. Smaller
/// block sizes (which would tile smaller TX kernels analogously) are
/// out of scope for F8.1 v1.
fn reconstruct_via_round_trip(
    params: &InterModeSearchParams<'_>,
    pred: &[u8],
    recon: &mut [u8],
    bw: usize,
    bh: usize,
    tx_size: TxSize,
) {
    let tx_side = tx_size.side();
    let n = tx_size.n_coeffs();
    let src = params.src_plane;
    let (sx, sy) = params.src_offset;

    // Tile the block into (bw / tx_side) × (bh / tx_side) transforms.
    // For supported sizes (16/32/64), tx_side ∈ {16, 32} and
    // bw % tx_side == bh % tx_side == 0.
    debug_assert_eq!(
        bw % tx_side,
        0,
        "block width {bw} not a multiple of tx_side {tx_side}",
    );
    debug_assert_eq!(
        bh % tx_side,
        0,
        "block height {bh} not a multiple of tx_side {tx_side}",
    );

    // Scratch buffers — sized for one transform.
    let mut residual = [0i16; 32 * 32];
    let mut coefs = [0i32; 1024];
    let mut qcoeff = [0i16; 1024];
    let mut dqcoeff = [0i16; 1024];
    let mut tx_recon = [0u8; 32 * 32];

    let scan = scan_for(tx_size, TxType::DctDct);
    let scan_slice = scan_for_forward(&scan);
    let dequant_pair = [params.quantizer.dequant[0], params.quantizer.dequant[1]];

    let n_tiles_x = bw / tx_side;
    let n_tiles_y = bh / tx_side;
    for ty in 0..n_tiles_y {
        for tx in 0..n_tiles_x {
            let off_x = tx * tx_side;
            let off_y = ty * tx_side;
            // 1. Build residual = src - pred for this tile.
            for r in 0..tx_side {
                for c in 0..tx_side {
                    let s = src.data[(sy + off_y + r) * src.stride + (sx + off_x + c)] as i16;
                    let p = pred[(off_y + r) * bw + (off_x + c)] as i16;
                    residual[r * tx_side + c] = s - p;
                }
            }

            // 2. Forward DCT — inter blocks always use DCT_DCT
            //    (libvpx convention; ADST is intra-only).
            match tx_size {
                TxSize::Tx8x8 => {
                    let mut buf = [0i32; 64];
                    fdct8x8(&residual[..64], &mut buf, tx_side);
                    coefs[..64].copy_from_slice(&buf);
                }
                TxSize::Tx16x16 => {
                    let mut buf = [0i32; 256];
                    fdct16x16(&residual[..256], &mut buf, tx_side);
                    coefs[..256].copy_from_slice(&buf);
                }
                TxSize::Tx32x32 => {
                    let mut buf = [0i32; 1024];
                    fdct32x32(&residual[..1024], &mut buf, tx_side);
                    coefs[..1024].copy_from_slice(&buf);
                }
                TxSize::Tx4x4 => {
                    // F8.1 v1 doesn't cover sub-8 blocks; would need
                    // fdct4x4 + idct4x4_16_add. Defer until needed.
                    debug_assert!(false, "F8.1 v1 doesn't support Tx4x4 tiles");
                    return;
                }
            }

            // 3. Forward quant.
            for q in qcoeff[..n].iter_mut() {
                *q = 0;
            }
            for d in dqcoeff[..n].iter_mut() {
                *d = 0;
            }
            let eob = if tx_size == TxSize::Tx32x32 {
                vp9_quantize_b_32x32(
                    &coefs[..1024],
                    false,
                    &params.quantizer.zbin,
                    &params.quantizer.round,
                    &params.quantizer.quant,
                    &params.quantizer.quant_shift,
                    &mut qcoeff[..1024],
                    &mut dqcoeff[..1024],
                    &dequant_pair,
                    scan_slice,
                )
            } else {
                vp9_quantize_b(
                    &coefs[..n],
                    n,
                    false,
                    &params.quantizer.zbin,
                    &params.quantizer.round,
                    &params.quantizer.quant,
                    &params.quantizer.quant_shift,
                    &mut qcoeff[..n],
                    &mut dqcoeff[..n],
                    &dequant_pair,
                    scan_slice,
                )
            };

            // 4. Reconstruct tile: tx_recon = pred + idct(dq).
            for r in 0..tx_side {
                for c in 0..tx_side {
                    tx_recon[r * tx_side + c] = pred[(off_y + r) * bw + (off_x + c)];
                }
            }
            if eob > 0 {
                let mut dq32 = [0i32; 1024];
                for (i, &v) in dqcoeff[..n].iter().enumerate() {
                    dq32[i] = v as i32;
                }
                match tx_size {
                    TxSize::Tx8x8 => {
                        let mut buf = [0i32; 64];
                        buf.copy_from_slice(&dq32[..64]);
                        // TODO(G4.2): route through SimdDispatch::idct8x8_add() once mode_decision threads &SimdDispatch.
                        idct8x8_64_add(&buf, &mut tx_recon[..64], tx_side);
                    }
                    TxSize::Tx16x16 => {
                        let mut buf = [0i32; 256];
                        buf.copy_from_slice(&dq32[..256]);
                        idct16x16_256_add(&buf, &mut tx_recon[..256], tx_side);
                    }
                    TxSize::Tx32x32 => {
                        let mut buf = [0i32; 1024];
                        buf.copy_from_slice(&dq32[..1024]);
                        idct32x32_1024_add(&buf, &mut tx_recon[..1024], tx_side);
                    }
                    TxSize::Tx4x4 => unreachable!(),
                }
            }

            // 5. Copy the tile back into `recon` at the right offset.
            for r in 0..tx_side {
                for c in 0..tx_side {
                    recon[(off_y + r) * bw + (off_x + c)] = tx_recon[r * tx_side + c];
                }
            }
        }
    }
}

/// SSE between source-plane block at `(sx, sy)` (stride
/// `src_plane.stride`) and reconstruction block `recon` (size `bw × bh`,
/// stride `recon_stride`). Returns the raw `u64` so the caller can
/// decide how to clamp.
fn sse_block(
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    recon: &[u8],
    recon_stride: usize,
    bw: usize,
    bh: usize,
) -> u64 {
    let (sx, sy) = src_offset;
    let mut sse: u64 = 0;
    for r in 0..bh {
        for c in 0..bw {
            let s = src_plane.data[(sy + r) * src_plane.stride + (sx + c)] as i32;
            let p = recon[r * recon_stride + c] as i32;
            let d = s - p;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// Map a [`FrameKind`] to the [`FrameTypeForRd`] enum used by
/// [`compute_rd_mult`]. Same convention as F8.0
/// (`mode_decision/intra.rs`): keyframes get the KF multiplier;
/// intra-only inherits the inter formula; inter frames use inter.
#[inline]
fn frame_type_for_rd(kind: FrameKind) -> FrameTypeForRd {
    match kind {
        FrameKind::KeyFrame => FrameTypeForRd::Key,
        FrameKind::ShowExistingFrame { .. }
        | FrameKind::IntraOnly { .. }
        | FrameKind::InterFrame { .. } => FrameTypeForRd::Inter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::mv_cost::build_nmv_cost_table;
    use crate::cost::symbol_costs::inter_mode_costs;
    use crate::entropy::frame_context::FrameContext;
    use crate::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
    use crate::bitstream::frame::{InterpFilter, MAX_INTER_REFS};
    use crate::mv_search::init_dsmotion_compensation;
    use crate::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    /// Build a minimal InterFrame `FrameKind` for tests — only the
    /// variant matters for [`pick_inter_mode_luma`]'s downstream
    /// dispatch (`frame_type_for_rd` collapses every non-keyframe to
    /// `FrameTypeForRd::Inter`).
    fn synthetic_inter_frame_kind() -> FrameKind {
        FrameKind::InterFrame {
            reset_frame_context: 0,
            refresh_frame_flags: 0,
            ref_frame_indices: [0u8; MAX_INTER_REFS],
            ref_frame_sign_biases: [false; MAX_INTER_REFS],
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
        }
    }

    /// Deterministic chaotic-content plane via xorshift32 so SAD-driven
    /// search has no accidental low-SAD attractors away from the true
    /// shift.
    fn make_plane(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; width * height];
        let mut state = seed.wrapping_add(1);
        for slot in buf.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *slot = (state & 0xff) as u8;
        }
        buf
    }

    /// Build the standard set of inputs to `pick_inter_mode_luma`.
    /// Caller supplies the source / reference planes and the block
    /// offset; we fill in the costs/limits/etc.
    struct TestHarness {
        costs: MvCosts,
        inter_costs: InterModeCosts,
        quantizer: PlaneQuantizers,
        cfg: SearchSiteConfig,
    }
    impl TestHarness {
        fn new(plane_w: usize, q_index: u8) -> Self {
            let fc = FrameContext::new_default();
            let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
            let inter_costs = inter_mode_costs(&fc);
            let quantizer =
                init_plane_quantizers(i32::from(q_index), QuantizerDeltas::Y);
            let cfg = init_dsmotion_compensation(plane_w as i32);
            Self {
                costs,
                inter_costs,
                quantizer,
                cfg,
            }
        }
    }

    /// Convenience: build an `InterModeSearchParams` from the harness
    /// and explicit per-test args. `mv_candidates` defaults to all-zero
    /// MVs (the standard NEWMV-from-zero seed). `mv_limits` is wide
    /// enough to cover the test's expected shift.
    #[allow(clippy::too_many_arguments)] // 9 args; test-only helper.
    fn make_params<'a>(
        harness: &'a TestHarness,
        block_size: BlockSize,
        src_plane: &'a PlaneRef<'a>,
        ref_plane: &'a PlaneRef<'a>,
        src_offset: (usize, usize),
        mv_candidates: &'a MvCandidates,
        mv_limits: MvLimits,
        q_index: u8,
    ) -> InterModeSearchParams<'a> {
        InterModeSearchParams {
            block_size,
            src_plane,
            src_offset,
            ref_plane,
            mv_candidates,
            mv_limits,
            site_config: &harness.cfg,
            mv_costs: &harness.costs,
            inter_mode_costs: &harness.inter_costs,
            inter_mode_ctx: 0,
            quantizer: &harness.quantizer,
            q_index,
            allow_hp: false,
            frame_kind: synthetic_inter_frame_kind(),
            newmv_seed: None,
        }
    }

    /// Source plane == reference plane → ZEROMV's distortion is 0
    /// (perfect match, residual is identically zero, no DCT round-trip
    /// loss). NEWMV may tie on distortion but loses on rate (NEWMV
    /// charges MV bits even at MV ≈ 0). ZEROMV must win.
    #[test]
    fn pick_inter_zero_mv_on_identical_ref() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 17);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        let mvc = MvCandidates::default();

        let params = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-32, 32, -32, 32),
            50,
        );

        let res = pick_inter_mode_luma(&params);
        // Both ZEROMV and NEARESTMV produce identical zero-MV
        // predictions here (mv_candidates.nearest is (0, 0)), and
        // both quantize to zero residual against an identical ref
        // → distortion is zero for both. The winner is the cheaper-
        // rate mode under the default inter_mode_probs at ctx=0; that
        // happens to be NEARESTMV (libvpx's tree puts NEAREST deeper
        // than ZERO but the default probability favours NEAREST). Any
        // zero-MV winner is correct; what matters is that NEWMV did
        // **not** win (which would have wasted bits encoding an MV
        // residual against an already-perfect prediction).
        assert!(
            matches!(res.best_mode, InterMode::ZeroMv | InterMode::NearestMv),
            "identical planes → expected ZEROMV or NEARESTMV, got {:?} (rd={}, dist={}, rate={})",
            res.best_mode, res.best_rd_cost, res.best_distortion, res.best_rate,
        );
        assert_eq!(res.best_mv, Mv::ZERO);
        assert_eq!(
            res.best_distortion, 0,
            "zero-MV winner on identical ref should have zero distortion",
        );
    }

    /// Source block at (20, 20) holds the reference's content
    /// at offset (col=23, row=20) — i.e. a 3-pixel rightward shift.
    /// NEWMV should recover MV ≈ (col=24, row=0) in 1/8-pel units
    /// (= 3 full pels) and win on distortion. We assert NEWMV's
    /// distortion is strictly less than ZEROMV's.
    #[test]
    fn pick_inter_finds_new_mv_on_shifted_ref() {
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 17);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        // Paint a 16×16 block at (src_col=20, src_row=20) in the
        // source with the reference's content at (col=23, row=20):
        // i.e. the source came from the reference shifted 3 cols right.
        let mut src_data = ref_data.clone();
        let bw = 16;
        let bh = 16;
        let src_col = 20;
        let src_row = 20;
        let shift_col: i16 = 3;
        let shift_row: i16 = 0;
        for i in 0..bh {
            for j in 0..bw {
                let ridx = (src_row + i + shift_row as usize) * pw
                    + (src_col + j + shift_col as usize);
                src_data[(src_row + i) * pw + (src_col + j)] = ref_data[ridx];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        let harness = TestHarness::new(pw, 50);
        let mvc = MvCandidates::default();
        let params = make_params(
            &harness,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (src_col, src_row),
            &mvc,
            MvLimits::new(-16, 16, -16, 16),
            50,
        );

        let all = pick_inter_mode_luma_all_costs(&params);
        let zero = &all[0];
        let new = &all[2];

        // The actual oracle: NEWMV's distortion must be substantially
        // lower than ZEROMV's (NEWMV finds the shift; ZEROMV is stuck
        // with the misaligned content).
        assert!(
            new.best_distortion < zero.best_distortion,
            "NEWMV dist ({}) should be < ZEROMV dist ({})",
            new.best_distortion, zero.best_distortion,
        );
        // The recovered MV must be near (col=24, row=0) in 1/8-pel
        // units (24 = 3 full pels × 8). Allow ±2 in case the sub-pel
        // refinement settles a quarter-pel off.
        assert!(
            (new.best_mv.col - 24).abs() <= 2,
            "NEWMV col {} should be within ±2 of 24 (1/8-pel)",
            new.best_mv.col,
        );
        assert!(
            new.best_mv.row.abs() <= 2,
            "NEWMV row {} should be within ±2 of 0 (1/8-pel)",
            new.best_mv.row,
        );

        // Overall winner should be NEWMV — its distortion drop swamps
        // the MV rate cost at q=50.
        let main = pick_inter_mode_luma(&params);
        assert_eq!(
            main.best_mode,
            InterMode::NewMv,
            "shifted-ref: expected NEWMV winner, got {:?} (rd={})",
            main.best_mode, main.best_rd_cost,
        );
    }

    /// Basic finite-RD-cost sanity check across the supported sizes.
    #[test]
    fn pick_inter_returns_finite_rd_cost() {
        let pw = 80;
        let ph = 80;
        let data = make_plane(pw, ph, 23);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 80);
        let mvc = MvCandidates::default();

        for bs in [BlockSize::Block16x16, BlockSize::Block32x32] {
            let params = make_params(
                &harness,
                bs,
                &plane,
                &plane,
                (8, 8),
                &mvc,
                MvLimits::new(-4, 4, -4, 4),
                80,
            );
            let res = pick_inter_mode_luma(&params);
            assert!(
                res.best_rd_cost > 0,
                "{bs:?}: rd_cost {} must be positive",
                res.best_rd_cost,
            );
            assert!(
                res.best_rd_cost < i64::MAX,
                "{bs:?}: rd_cost must be finite",
            );
            assert!(
                res.best_rate > 0,
                "{bs:?}: rate must be positive (mode cost > 0 always)",
            );
        }
    }

    /// All 3 candidates are evaluated — `pick_inter_mode_luma_all_costs`
    /// returns exactly 3 entries in `[ZEROMV, NEARESTMV, NEWMV]` order.
    /// Each entry's mode field must match the slot's intended mode.
    #[test]
    fn pick_inter_walks_3_candidates() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 9);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 60);
        let mvc = MvCandidates {
            nearest: Mv { row: 0, col: 8 }, // 1 full-pel right
            near: Mv::ZERO,
        };
        let params = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-8, 8, -8, 8),
            60,
        );
        let all = pick_inter_mode_luma_all_costs(&params);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].best_mode, InterMode::ZeroMv);
        assert_eq!(all[1].best_mode, InterMode::NearestMv);
        assert_eq!(all[2].best_mode, InterMode::NewMv);
        // Each must produce a finite rate / dist / rd entry.
        for (i, c) in all.iter().enumerate() {
            assert!(c.best_rd_cost > 0, "slot {i}: rd_cost {} must be > 0", c.best_rd_cost);
            assert!(c.best_rd_cost < i64::MAX, "slot {i}: rd_cost must be finite");
        }
    }

    /// Smoke: Block16x16 / Block32x32 / Block64x64 all return finite
    /// results without panicking. We use identical source/ref to keep
    /// distortion near zero — the focus is on the per-size dispatch.
    #[test]
    fn pick_inter_block_size_smoke() {
        let pw = 96;
        let ph = 96;
        let data = make_plane(pw, ph, 5);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        let mvc = MvCandidates::default();

        for bs in [
            BlockSize::Block16x16,
            BlockSize::Block32x32,
            BlockSize::Block64x64,
        ] {
            let params = make_params(
                &harness,
                bs,
                &plane,
                &plane,
                (16, 16),
                &mvc,
                MvLimits::new(-4, 4, -4, 4),
                50,
            );
            let res = pick_inter_mode_luma(&params);
            assert!(
                res.best_rd_cost > 0 && res.best_rd_cost < i64::MAX,
                "{bs:?}: rd_cost {} must be finite positive",
                res.best_rd_cost,
            );
            // Same-plane → ZEROMV's and NEARESTMV's predictions both
            // collapse to zero distortion; either is a valid winner
            // (rate-cost tiebreaker, see
            // `pick_inter_zero_mv_on_identical_ref` for details). The
            // failure mode we're actually guarding against is NEWMV
            // winning over a zero-MV mode (that would mean the search
            // wasted bits on a useless MV residual).
            assert!(
                matches!(res.best_mode, InterMode::ZeroMv | InterMode::NearestMv),
                "{bs:?} identical src/ref: expected ZEROMV or NEARESTMV winner, got {:?}",
                res.best_mode,
            );
            assert_eq!(res.best_distortion, 0);
        }
    }

    /// Mode-cost lookup uses the right context index — passing a
    /// different `inter_mode_ctx` should change the rate component.
    /// (Sanity: at least two contexts must yield different costs for
    /// the same winning mode.)
    #[test]
    fn pick_inter_mode_ctx_affects_rate() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 41);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 40);
        let mvc = MvCandidates::default();

        let params0 = InterModeSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &plane,
            src_offset: (16, 16),
            ref_plane: &plane,
            mv_candidates: &mvc,
            mv_limits: MvLimits::new(-4, 4, -4, 4),
            site_config: &harness.cfg,
            mv_costs: &harness.costs,
            inter_mode_costs: &harness.inter_costs,
            inter_mode_ctx: 0,
            quantizer: &harness.quantizer,
            q_index: 40,
            allow_hp: false,
            frame_kind: synthetic_inter_frame_kind(),
            newmv_seed: None,
        };
        let params6 = InterModeSearchParams {
            inter_mode_ctx: 6,
            ..params0
        };

        let r0 = pick_inter_mode_luma(&params0);
        let r6 = pick_inter_mode_luma(&params6);
        // The rate-cost row differs across contexts. With identical-
        // plane src/ref both ZEROMV and NEARESTMV produce zero-dist
        // predictions; the winning mode is whichever has the cheaper
        // rate at each context. Across ctx 0 and ctx 6 the rate-cost
        // breakdown differs, so either:
        //   - winner mode differs between the two contexts, OR
        //   - winner mode is identical but rate (and hence RD) differs.
        // We assert one of those two outcomes holds.
        assert!(
            matches!(r0.best_mode, InterMode::ZeroMv | InterMode::NearestMv),
            "ctx 0 winner must be a zero-MV mode, got {:?}",
            r0.best_mode,
        );
        assert!(
            matches!(r6.best_mode, InterMode::ZeroMv | InterMode::NearestMv),
            "ctx 6 winner must be a zero-MV mode, got {:?}",
            r6.best_mode,
        );
        assert_eq!(r0.best_distortion, r6.best_distortion);
        assert!(
            r0.best_mode != r6.best_mode || r0.best_rate != r6.best_rate,
            "ctx 0 and ctx 6 should produce a different (mode, rate) pair; \
             got both = ({:?}, {})",
            r0.best_mode, r0.best_rate,
        );
    }
}
