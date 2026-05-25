//! Inter mode search — libvpx-faithful realtime port of
//! `vp9_pick_inter_mode`.
//!
//! Tries `{ZEROMV, NEARESTMV, NEARMV, NEWMV} × {LASTFRAME}` for a
//! single luma block, performs sub-pel reference fetch + forward DCT +
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
//! - **Inter modes**: `{NEARESTMV, NEARMV, ZEROMV, NEWMV}`. NEARMV
//!   (A12, 2026-05-22) shares the candidate-list mechanics of
//!   NEARESTMV but uses the second-nearest MV
//!   (`MvCandidates.near`). The caller is responsible for populating
//!   `MvCandidates.near` from `find_mv_refs(NEARMV, ...)` — which
//!   does NOT early-break — so the picker's NEARMV reconstruction
//!   matches what the decoder will compute when reading the
//!   bitstream.
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
//!  (NEARMV, mv_candidates.near),
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
//!    (if NEWMV: ROUND_POWER_OF_TWO(mv_cost(diff, mv_costs) * 108, 7))`.
//!    The `108/128` damping mirrors libvpx `vp9_mv_bit_cost`
//!    (`vp9_mcomp.c:80-84`) with `MV_COST_WEIGHT = 108` (`vp9_rd.h:38`).
//! 8. `rd_cost = ROUND_POWER_OF_TWO(R * lambda, 9) + (D << 7)` —
//!    libvpx `RDCOST` macro at `vp9_rd.h:29-30`.

use crate::bitstream::frame::InterpFilter;
use crate::bitstream::header::FrameKind;
use crate::block::BlockSize;
use crate::cost::mv_cost::{MvCosts, mv_cost};
use crate::cost::{
    FrameTypeForRd, InterModeCosts, SwitchableInterpCosts,
    compute_rd_mult, get_intra_cost_penalty,
};
use crate::entropy::default_probs::INTER_MODE_CONTEXTS;
use crate::inter::filter::{FILTER_TAPS, kernel_for};
use crate::intra_pred::{
    ABOVE_UNAVAILABLE_FILL, DcAvailability, IntraMode, LEFT_UNAVAILABLE_FILL, PredCtx,
    intra_predict,
};
use crate::mode_decision::block_yrd::{
    BlockYrdResult, block_error_fp_i32, satd_i16,
};
use crate::mode_decision::rd_cost;
use crate::mode_info::{InterMode, Mv, RefFrame};
use crate::mv_search::full_pixel::{
    FullPixelSearchParams, PlaneRef, diamond_search_sad,
};
use crate::mv_search::mv_limits::MvLimits;
use crate::mv_search::search_site::SearchSiteConfig;
use crate::mv_search::sub_pixel::{
    SubPixelSearchParams, adaptive_subpel_force_stop,
    find_best_sub_pixel_tree_pruned_more,
};
use crate::mode_decision::hadamard::{hadamard_8x8, hadamard_16x16, hadamard_32x32};
use crate::quant::forward::{
    PlaneQuantizers, scan_for_forward, vp9_quantize_b, vp9_quantize_b_32x32,
    vp9_quantize_fp, vp9_quantize_fp_32x32,
};
use crate::transforms::forward::{fdct8x8, fdct16x16, fdct32x32};
use crate::transforms::inverse::{idct8x8_64_add, idct16x16_256_add, idct32x32_1024_add};
use crate::transforms::scans::scan_for;
use crate::transforms::{TxMode, TxSize, TxType};
use crate::mode_decision::model_rd::{model_rd_for_sb_y, model_rd_for_sb_y_large, model_rd_for_sb_uv, SkipTxfm};
use crate::Pixel;

/// F.1 — the default skip-prob context index used by
/// `evaluate_candidate`'s ported skip-flag tie-break
/// (`vp9_pickmode.c:2355-2362`). libvpx looks up the per-block skip
/// probability from `cm->fc->skip_probs[skip_ctx]` where `skip_ctx`
/// is `vp9_get_pred_context_skip` (derived from neighbour skip
/// flags). The picker doesn't carry frame context, so F.1 uses the
/// middle default (`DEFAULT_SKIP_PROBS[1] = 128`, neutral 50/50:
/// `vp9_cost_bit(128, 0) == vp9_cost_bit(128, 1) == 512` =
/// 1 bit in `vp9_prob_cost` units). Threading the real per-block
/// skip context is a F.4 follow-up.
const F1_DEFAULT_SKIP_CTX: usize = 1;

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
    /// `mv_ref_list[1]` — the second-nearest decoded MV. Used as the
    /// reference MV for `NEARMV` (A12, 2026-05-22). The caller must
    /// derive this from `find_mv_refs(NEARMV, ...)` (the NEARMV path
    /// disables `early_break` so the search continues past the first
    /// candidate). Setting `near == nearest` is also legal — it
    /// makes NEARMV identical to NEARESTMV in rate/distortion, and
    /// the picker's RD tiebreaker (mode-cost lookup) will pick the
    /// cheaper mode. The decoder will still read `find_mv_refs(NEARMV,
    /// ...)` for byte-exact reconstruction.
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
    /// Ref-frame rate cost in `vp9_prob_cost` units (1/512-bit). Added
    /// to every candidate's `rate` so the dual-ref `argmin(pick_last,
    /// pick_golden)` selector at the caller can compare RDs that
    /// include the wire-format ref-frame-bit cost difference between
    /// LAST and GOLDEN (and intra-vs-inter, if the caller wants).
    ///
    /// **A16 (2026-05-22)**: introduced to fix the dual-ref picker
    /// regression. Without this term the picker chose GOLDEN ~25-35%
    /// of leaves at high q (residuals collapse to zero, SSEs tie,
    /// missing rate term lets GOLDEN sneak ahead), regressing F12
    /// PSNR by ~10 dB. Mirrors libvpx
    /// `vp9_pickmode.c:2398` `this_rdc.rate += ref_frame_cost[ref_frame]`
    /// at picker level; the picker itself does NOT know which ref it
    /// is searching, so the caller computes the cost via
    /// libvpx-default probs (`vp9_pickmode.c:1215-1230` `init_ref_frame_cost`)
    /// and passes it in.
    ///
    /// Set to `0` to disable (mirrors pre-A16 behaviour). Single-ref-
    /// only callers can use this field to add the intra-vs-inter bit
    /// too, though the differential between LAST and GOLDEN only
    /// requires the single-ref subtree.
    pub ref_frame_rate_cost: u32,
    /// Per-context per-filter bit-cost table (G2). Replaces the flat
    /// `F5_SWITCHABLE_FILTER_RATE`. Build once per frame via
    /// [`crate::cost::switchable_interp_costs`].
    pub switchable_interp_costs: &'a SwitchableInterpCosts,
    /// Switchable-interpolation context (0..4) from above/left
    /// neighbour filters. Compute via
    /// [`crate::inter::get_pred_context_switchable_interp`] per leaf.
    /// Context 3 is the "no neighbour" sentinel.
    pub switchable_interp_ctx: usize,

    // --- G1: UV RD fields ---

    /// Source U plane (full frame buffer).
    pub src_u: &'a [u8],
    /// Source V plane (full frame buffer).
    pub src_v: &'a [u8],
    /// Reference U plane (full frame buffer, per-ref).
    pub ref_u: &'a [u8],
    /// Reference V plane (full frame buffer, per-ref).
    pub ref_v: &'a [u8],
    /// Stride of all UV planes (= w/2 for 4:2:0).
    pub uv_stride: usize,
    /// Chroma DC dequant (`dequant[0]` from UV quantizer).
    pub uv_dc_quant: u32,
    /// Chroma AC dequant (`dequant[1]` from UV quantizer).
    pub uv_ac_quant: u32,
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
    /// Distortion (SSE between source and reconstruction). When the
    /// libvpx skip-flag tie-break picks `skip = true` (F.1, per Q5)
    /// this is overridden with `sse_after_pred = sum((src - pred)^2)`
    /// to match the distortion the decoder will see.
    pub best_distortion: u32,
    /// Rate cost (mode + MV bits + ref-frame bits + F.1 coef-bit
    /// estimate, in `vp9_prob_cost` units). When the libvpx skip-flag
    /// tie-break picks `skip = true` this collapses to
    /// `vp9_cost_bit(skip_prob, 1)`.
    pub best_rate: u32,
    /// F.1 — coefficient-bit estimate that contributed to the
    /// candidate's RD. Sourced from
    /// [`crate::mode_decision::block_yrd::BlockYrdResult::coef_rate`].
    /// In `vp9_prob_cost` units (1/512-bit). **`u64` (not `u32`) since
    /// the night-8 saturation fix**: the underlying `block_yrd`
    /// accumulator was widened to preserve K1 calibration signal at
    /// low q. Callers add to a `u32` rate sum with a saturate-cast at
    /// the use site; this field surfaces the un-clipped value for the
    /// F.0.2 per-block oracle (`coef_bits_est = (coef_rate >> 9).min
    /// (u32::MAX as u64) as u32`). For skip-flag-true candidates this
    /// remains the pre-skip estimate.
    pub coef_rate: u64,
    /// F.1 — coefficient-domain squared error from
    /// [`crate::mode_decision::block_yrd::BlockYrdResult::coef_dist`].
    /// `i64` per HLD Q7 to avoid overflow on saturated 32×32
    /// quant-error blocks. **Not consumed by `rd_cost` in F.1** (Q3
    /// keeps pixel-SSE-of-recon as the RD distortion term, DV-F1-3).
    /// Surfaced for F.4 A/B experiments.
    pub coef_dist: i64,
    /// F.5 — interpolation filter chosen by the per-block filter
    /// search. One of `{EightTap, EightTapSmooth}` (libvpx-realtime
    /// 2-filter range). Defaults to `EightTap` for callers that don't
    /// hit the filter loop (e.g. zero-distortion / identical-ref
    /// shortcuts). The writer reads this field to emit the per-block
    /// switchable-filter bits.
    pub picked_interp_filter: InterpFilter,
    /// G4 — Y-plane variance from `model_rd_for_sb_y`. Surfaced for
    /// the encode-breakout test in `pick_inter_mode`.
    pub var_y: u32,
    /// G4 — Y-plane SSE from `model_rd_for_sb_y`. Surfaced for
    /// the encode-breakout test in `pick_inter_mode`.
    pub sse_y: u32,
    /// G6 — set when `model_rd_for_sb_y_large` detects a full skip
    /// (AcDc) on a block >= 32x32. When true, the mode loop may
    /// exit early after evaluating at least 2 candidates. Mirrors
    /// libvpx `early_term` at `vp9_pickmode.c:560-603`.
    pub early_term: bool,
}

/// Run the libvpx-style inter-mode RD search for one luma block.
///
/// Evaluates `{ZEROMV, NEARESTMV, NEARMV, NEWMV}` × `{LASTFRAME}` and
/// returns the lowest-RD-cost winner. See module docs for the
/// deliberate divergences from libvpx's full realtime path.
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

    // Candidate 3: NEARMV — uses `mv_candidates.near`
    // (`find_mv_refs(NEARMV).mvs[count-1]`). No MV delta is encoded
    // on the wire; the rate cost is just the inter-mode bits.
    let near = evaluate_candidate(
        params,
        InterMode::NearMv,
        params.mv_candidates.near,
        params.mv_candidates.near,
        lambda,
    );

    // Candidate 4: NEWMV — full-pel diamond seeded at NEAREST, then
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
    if near.best_rd_cost < best.best_rd_cost {
        best = near;
    }
    if new.best_rd_cost < best.best_rd_cost {
        best = new;
    }
    if crate::mv_search::e12stat::enabled() {
        crate::mv_search::e12stat::record_argmin(matches!(best.best_mode, InterMode::NewMv));
    }
    best
}

/// Diagnostic: same search as [`pick_inter_mode_luma`] but returns the
/// per-candidate breakdown in candidate order
/// `[ZEROMV, NEARESTMV, NEARMV, NEWMV]`. Used by the tests to
/// sanity-check the RD-ranking for the shifted-ref oracle.
#[must_use]
pub fn pick_inter_mode_luma_all_costs(
    params: &InterModeSearchParams<'_>,
) -> [InterModeSearchResult; 4] {
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
    let near = evaluate_candidate(
        params,
        InterMode::NearMv,
        params.mv_candidates.near,
        params.mv_candidates.near,
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
    [zero, nearest, near, new]
}

// ====================================================================
// Phase 2a — unified dual-ref interleaved picker.
// ====================================================================

/// Inputs to [`pick_inter_mode`].
///
/// Carries data for BOTH reference frames (LAST and GOLDEN) so the
/// picker can iterate libvpx's `RT_INTER_MODES` interleaved order in
/// a single call, replacing the old dual-`pick_inter_mode_luma` +
/// argmin pattern.
pub struct PickInterModeParams<'a> {
    /// Block size being decided.
    pub block_size: BlockSize,
    /// Source luma plane view.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block in plane samples `(x, y)`.
    pub src_offset: (usize, usize),

    // --- LAST ref ---
    /// LASTFRAME's reconstructed luma plane.
    pub last_ref_plane: &'a PlaneRef<'a>,
    /// LAST MV candidates from `find_mv_refs`.
    pub last_mv_candidates: &'a MvCandidates,

    // --- GOLDEN ref ---
    /// GOLDENFRAME's reconstructed luma plane.
    pub golden_ref_plane: &'a PlaneRef<'a>,
    /// GOLDEN MV candidates from `find_mv_refs`.
    pub golden_mv_candidates: &'a MvCandidates,

    // --- Shared ---
    /// Full-pel-unit MV bounds.
    pub mv_limits: MvLimits,
    /// 4-site diamond config.
    pub site_config: &'a SearchSiteConfig,
    /// MV-bit-cost LUT.
    pub mv_costs: &'a MvCosts,
    /// Inter-mode rate-cost table.
    pub inter_mode_costs: &'a InterModeCosts,
    /// Inter-mode context index (`0..INTER_MODE_CONTEXTS`).
    pub inter_mode_ctx: usize,
    /// Y-plane forward-quantizer.
    pub quantizer: &'a PlaneQuantizers,
    /// Frame base qindex (0..255).
    pub q_index: u8,
    /// `allow_high_precision_mv` from frame header.
    pub allow_hp: bool,
    /// Frame kind — selects the RD-multiplier formula.
    pub frame_kind: FrameKind,
    /// SB-level NEWMV diamond-search seed.
    pub newmv_seed: Option<Mv>,
    /// Ref-frame rate cost for LAST in `vp9_prob_cost` units.
    pub ref_frame_cost_last: u32,
    /// Ref-frame rate cost for GOLDEN in `vp9_prob_cost` units.
    pub ref_frame_cost_golden: u32,

    // --- OOB flags (from caller's mv_in_bounds check) ---
    /// Whether LAST NEAREST MV is within the picker's MV envelope.
    pub last_nearest_in_bounds: bool,
    /// Whether LAST NEAR MV is within the picker's MV envelope.
    pub last_near_in_bounds: bool,
    /// Whether GOLDEN NEAREST MV is within the picker's MV envelope.
    pub golden_nearest_in_bounds: bool,
    /// Whether GOLDEN NEAR MV is within the picker's MV envelope.
    pub golden_near_in_bounds: bool,

    // --- Intra fallback context (Phase 2b) ---
    //
    // Neighbour-edge data for the intra-in-inter search
    // (`vp9_pickmode.c:2517-2646`). When `intra_above_pixels` is
    // `Some`, the picker evaluates DC/V/H intra modes after the inter
    // loop and may pick `RefFrame::IntraFrame` if they beat the best
    // inter candidate.

    /// Reconstructed row above the block (≥ `bw` pixels), or `None` when
    /// the block is at the top frame edge.
    pub intra_above_pixels: Option<&'a [Pixel]>,
    /// Reconstructed column left of the block (≥ `bh` pixels), or
    /// `None` when at the left frame edge.
    pub intra_left_pixels: Option<&'a [Pixel]>,
    /// Top-left corner pixel. Ignored when both neighbours are
    /// unavailable.
    pub intra_top_left_pixel: Pixel,
    /// Inter-frame per-Y-mode cost table `[size_group][mode]`. Mirrors
    /// libvpx `cpi->mbmode_cost` (a.k.a.
    /// `intra_y_mode_costs_inter(fc)[size_group]`).
    pub intra_y_mode_costs: &'a [i32; 10],
    /// Ref-frame rate cost for `INTRA_FRAME` in `vp9_prob_cost` units.
    /// `vp9_cost_bit(intra_inter_p, false)`.
    pub ref_frame_cost_intra: u32,
    /// Per-context per-filter bit-cost table (G2). Replaces the flat
    /// `F5_SWITCHABLE_FILTER_RATE`. Build once per frame via
    /// [`crate::cost::switchable_interp_costs`].
    pub switchable_interp_costs: &'a SwitchableInterpCosts,
    /// Switchable-interpolation context (0..4) from above/left
    /// neighbour filters. Compute via
    /// [`crate::inter::get_pred_context_switchable_interp`] per leaf.
    pub switchable_interp_ctx: usize,

    // --- G1: UV RD fields ---

    /// Source U plane (full frame buffer).
    pub src_u: &'a [u8],
    /// Source U plane stride.
    pub src_u_stride: usize,
    /// Source V plane (full frame buffer).
    pub src_v: &'a [u8],
    /// Source V plane stride.
    pub src_v_stride: usize,
    /// LAST recon U plane (full frame buffer).
    pub last_ref_u: &'a [u8],
    /// LAST recon V plane (full frame buffer).
    pub last_ref_v: &'a [u8],
    /// GOLDEN recon U plane (full frame buffer).
    pub golden_ref_u: &'a [u8],
    /// GOLDEN recon V plane (full frame buffer).
    pub golden_ref_v: &'a [u8],
    /// Stride of all UV ref planes (= w/2 for 4:2:0).
    pub uv_stride: usize,
    /// Chroma DC dequant (`dequant[0]` from UV quantizer).
    pub uv_dc_quant: u32,
    /// Chroma AC dequant (`dequant[1]` from UV quantizer).
    pub uv_ac_quant: u32,

    // --- G4: encode breakout test fields ---

    /// Per-frame quality knob for the encode-breakout test
    /// (`vp9_pickmode.c:933-1036`). Typically 20 for realtime speed >= 5.
    /// Set to 0 to disable the breakout test.
    pub encode_breakout: u32,
    /// Y-plane DC dequant value (`dequant[0]`). Used by the
    /// encode-breakout threshold computation.
    pub dequant_dc: u16,
    /// Y-plane AC dequant value (`dequant[1]`). Used by the
    /// encode-breakout threshold computation.
    pub dequant_ac: u16,
}

/// Result of [`pick_inter_mode`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PickInterModeResult {
    /// Winning inter mode. Meaningless when `picked_ref_frame ==
    /// IntraFrame`; use [`picked_intra_mode`] instead.
    pub best_mode: InterMode,
    /// Winning MV (1/8-pel units). `Mv::ZERO` when intra wins.
    pub best_mv: Mv,
    /// `RDCOST(lambda, rate, distortion)` for the winning candidate.
    pub best_rd_cost: i64,
    /// Distortion (SSE between source and reconstruction).
    pub best_distortion: u32,
    /// Rate cost in `vp9_prob_cost` units.
    pub best_rate: u32,
    /// F.1 — coefficient-bit estimate.
    pub coef_rate: u64,
    /// F.1 — coefficient-domain squared error.
    pub coef_dist: i64,
    /// F.5 — interpolation filter chosen by per-block filter search.
    pub picked_interp_filter: InterpFilter,
    /// The chosen reference frame (LastFrame, GoldenFrame, or
    /// IntraFrame when the intra fallback wins).
    pub picked_ref_frame: RefFrame,
    /// Intra mode when `picked_ref_frame == IntraFrame`. One of
    /// `{Dc, V, H}` in v1. `None` for inter winners.
    pub picked_intra_mode: Option<IntraMode>,
    /// G4 — Y-plane variance from the winning candidate's
    /// `model_rd_for_sb_y`. Used by the encode-breakout test.
    pub var_y: u32,
    /// G4 — Y-plane SSE from the winning candidate's
    /// `model_rd_for_sb_y`. Used by the encode-breakout test.
    pub sse_y: u32,
}

/// libvpx `ref_mode_set` iteration order (first 8 of 12 entries,
/// without ALTREF). Reference: `vp9_pickmode.c:1239-1246`.
///
/// Note: the SVC variant (`ref_mode_set_svc`, lines 1249-1254)
/// differs — it places `(LAST, NEAR)` before `(GOLDEN, ZERO)`.
/// We follow the non-SVC order.
///
/// ```text
/// (LAST, ZERO), (LAST, NEAREST), (GOLDEN, ZERO), (LAST, NEAR),
/// (LAST, NEW),  (GOLDEN, NEAREST), (GOLDEN, NEAR), (GOLDEN, NEW)
/// ```
const RT_INTER_MODES: [(RefFrame, InterMode); 8] = [
    (RefFrame::LastFrame, InterMode::ZeroMv),
    (RefFrame::LastFrame, InterMode::NearestMv),
    (RefFrame::GoldenFrame, InterMode::ZeroMv),
    (RefFrame::LastFrame, InterMode::NearMv),
    (RefFrame::LastFrame, InterMode::NewMv),
    (RefFrame::GoldenFrame, InterMode::NearestMv),
    (RefFrame::GoldenFrame, InterMode::NearMv),
    (RefFrame::GoldenFrame, InterMode::NewMv),
];

/// Run the unified dual-ref inter-mode RD search for one luma block.
///
/// Iterates candidates in libvpx's `RT_INTER_MODES` interleaved order
/// across both LAST and GOLDEN reference frames, performing duplicate-MV
/// skip, OOB veto, and early-exit. Returns the lowest-RD-cost winner
/// with the chosen reference frame.
///
/// Replaces the old pattern of calling [`pick_inter_mode_luma`] twice
/// (once per ref) and taking the argmin at the caller.
#[must_use]
pub fn pick_inter_mode(params: &PickInterModeParams<'_>) -> PickInterModeResult {
    debug_assert!(
        params.inter_mode_ctx < INTER_MODE_CONTEXTS,
        "inter_mode_ctx {} out of [0, {INTER_MODE_CONTEXTS})",
        params.inter_mode_ctx,
    );

    let lambda = compute_rd_mult(
        i32::from(params.q_index),
        frame_type_for_rd(params.frame_kind),
    );

    // Per-ref safe MVs: clamp OOB candidates to ZERO so the
    // evaluator doesn't read outside the reference plane.
    let last_mvc_safe = MvCandidates {
        nearest: if params.last_nearest_in_bounds {
            params.last_mv_candidates.nearest
        } else {
            Mv::ZERO
        },
        near: if params.last_near_in_bounds {
            params.last_mv_candidates.near
        } else {
            Mv::ZERO
        },
    };
    let golden_mvc_safe = MvCandidates {
        nearest: if params.golden_nearest_in_bounds {
            params.golden_mv_candidates.nearest
        } else {
            Mv::ZERO
        },
        near: if params.golden_near_in_bounds {
            params.golden_mv_candidates.near
        } else {
            Mv::ZERO
        },
    };

    // mode_checked[ref_idx][mode_idx] — tracks which (ref, MV) pairs
    // have been evaluated for duplicate-MV skip. ref_idx: 0=LAST,
    // 1=GOLDEN. mode_idx: InterMode::idx() (0=NEAREST,1=NEAR,2=ZERO,3=NEW).
    let mut mode_checked: [[bool; 4]; 2] = [[false; 4]; 2];
    // frame_mv[ref_idx][mode_idx] — the MV used for each evaluated
    // (ref, mode) candidate. Populated on evaluation, read for dup check.
    let mut frame_mv: [[Mv; 4]; 2] = [[Mv::ZERO; 4]; 2];

    // NEWMV search results (lazily computed per ref frame).
    let mut newmv_last: Option<Mv> = None;
    let mut newmv_golden: Option<Mv> = None;

    let mut best: Option<PickInterModeResult> = None;

    for &(ref_frame, mode) in &RT_INTER_MODES {
        let ref_idx: usize = match ref_frame {
            RefFrame::LastFrame => 0,
            RefFrame::GoldenFrame => 1,
            _ => unreachable!(),
        };
        let mvc_safe = if ref_idx == 0 { &last_mvc_safe } else { &golden_mvc_safe };
        let ref_plane = if ref_idx == 0 { params.last_ref_plane } else { params.golden_ref_plane };
        let ref_frame_rate_cost = if ref_idx == 0 {
            params.ref_frame_cost_last
        } else {
            params.ref_frame_cost_golden
        };
        // G1 — per-ref UV planes for chroma RD.
        let (ref_u, ref_v) = if ref_idx == 0 {
            (params.last_ref_u, params.last_ref_v)
        } else {
            (params.golden_ref_u, params.golden_ref_v)
        };

        // OOB veto for NEAREST / NEAR: skip if the caller flagged the
        // MV as out-of-bounds.
        match (ref_idx, mode) {
            (0, InterMode::NearestMv) if !params.last_nearest_in_bounds => continue,
            (0, InterMode::NearMv) if !params.last_near_in_bounds => continue,
            (1, InterMode::NearestMv) if !params.golden_nearest_in_bounds => continue,
            (1, InterMode::NearMv) if !params.golden_near_in_bounds => continue,
            _ => {}
        }

        // Determine the candidate MV.
        let mv = match mode {
            InterMode::ZeroMv => Mv::ZERO,
            InterMode::NearestMv => mvc_safe.nearest,
            InterMode::NearMv => mvc_safe.near,
            InterMode::NewMv => {
                // Lazily run the NEWMV search for this ref frame.
                let cached = if ref_idx == 0 { &mut newmv_last } else { &mut newmv_golden };
                if cached.is_none() {
                    let single_params = InterModeSearchParams {
                        block_size: params.block_size,
                        src_plane: params.src_plane,
                        src_offset: params.src_offset,
                        ref_plane,
                        mv_candidates: mvc_safe,
                        mv_limits: params.mv_limits,
                        site_config: params.site_config,
                        mv_costs: params.mv_costs,
                        inter_mode_costs: params.inter_mode_costs,
                        inter_mode_ctx: params.inter_mode_ctx,
                        quantizer: params.quantizer,
                        q_index: params.q_index,
                        allow_hp: params.allow_hp,
                        frame_kind: params.frame_kind,
                        newmv_seed: params.newmv_seed,
                        ref_frame_rate_cost,
                        switchable_interp_costs: params.switchable_interp_costs,
                        switchable_interp_ctx: params.switchable_interp_ctx,
                        src_u: params.src_u,
                        src_v: params.src_v,
                        ref_u,
                        ref_v,
                        uv_stride: params.uv_stride,
                        uv_dc_quant: params.uv_dc_quant,
                        uv_ac_quant: params.uv_ac_quant,
                    };
                    *cached = Some(run_newmv_search(&single_params));
                }
                cached.unwrap()
            }
        };

        // Record the MV for this (ref, mode) slot.
        frame_mv[ref_idx][mode.idx()] = mv;

        // Duplicate-MV skip (libvpx `vp9_pickmode.c:2257-2268`):
        // skip if another mode for the same ref already checked the
        // same MV (restricted to zero-MV to match libvpx's TODO
        // comment about non-zero regression).
        let mut skip_this = false;
        if mode != InterMode::NearestMv {
            for other in [InterMode::NearestMv, InterMode::NearMv, InterMode::ZeroMv, InterMode::NewMv] {
                if other == mode { continue; }
                if mode_checked[ref_idx][other.idx()]
                    && frame_mv[ref_idx][mode.idx()] == frame_mv[ref_idx][other.idx()]
                    && frame_mv[ref_idx][other.idx()] == Mv::ZERO
                {
                    skip_this = true;
                    break;
                }
            }
        }
        // libvpx `vp9_pickmode.c:2286-2289`: skip any non-NEAREST mode
        // whose MV equals NEAREST's MV for the same ref (avoids
        // re-evaluating the exact same prediction).
        if mode != InterMode::NearestMv
            && mode_checked[ref_idx][InterMode::NearestMv.idx()]
            && mv == frame_mv[ref_idx][InterMode::NearestMv.idx()]
        {
            skip_this = true;
        }
        // libvpx `vp9_pickmode.c:2166` — `const_motion` NEARMV skip.
        // In `mv_refs_rt`, `const_motion=1` when the first nearest
        // candidate was NOT found among the 2 nearest spatial
        // neighbours. This makes NEARMV redundant because:
        //   (a) NEAR == NEAREST → caught by the NEAREST-equality
        //       skip above, or
        //   (b) NEAR == (0,0) → caught by the zero-MV dup check.
        // The only uncovered case (NEAR is non-zero, differs from
        // NEAREST, populated via diff-ref fallback) is too rare to
        // justify threading a new flag through `find_mv_refs`.
        // Intentionally omitted.
        if skip_this { continue; }

        // Build a single-ref params struct for the evaluator.
        let single_params = InterModeSearchParams {
            block_size: params.block_size,
            src_plane: params.src_plane,
            src_offset: params.src_offset,
            ref_plane,
            mv_candidates: mvc_safe,
            mv_limits: params.mv_limits,
            site_config: params.site_config,
            mv_costs: params.mv_costs,
            inter_mode_costs: params.inter_mode_costs,
            inter_mode_ctx: params.inter_mode_ctx,
            quantizer: params.quantizer,
            q_index: params.q_index,
            allow_hp: params.allow_hp,
            frame_kind: params.frame_kind,
            newmv_seed: params.newmv_seed,
            ref_frame_rate_cost,
            switchable_interp_costs: params.switchable_interp_costs,
            switchable_interp_ctx: params.switchable_interp_ctx,
            src_u: params.src_u,
            src_v: params.src_v,
            ref_u,
            ref_v,
            uv_stride: params.uv_stride,
            uv_dc_quant: params.uv_dc_quant,
            uv_ac_quant: params.uv_ac_quant,
        };

        let ref_mv = match mode {
            InterMode::NearMv => mvc_safe.near,
            _ => mvc_safe.nearest,
        };

        let candidate = evaluate_candidate(&single_params, mode, mv, ref_mv, lambda);

        mode_checked[ref_idx][mode.idx()] = true;

        let is_new_best = match best.as_ref() {
            None => true,
            Some(prev) => candidate.best_rd_cost < prev.best_rd_cost,
        };
        if is_new_best {
            best = Some(PickInterModeResult {
                best_mode: candidate.best_mode,
                best_mv: candidate.best_mv,
                best_rd_cost: candidate.best_rd_cost,
                best_distortion: candidate.best_distortion,
                best_rate: candidate.best_rate,
                coef_rate: candidate.coef_rate,
                coef_dist: candidate.coef_dist,
                picked_interp_filter: candidate.picked_interp_filter,
                picked_ref_frame: ref_frame,
                picked_intra_mode: None,
                var_y: candidate.var_y,
                sse_y: candidate.sse_y,
            });
        }

        // --- G4: encode breakout test ---
        //
        // After updating the best candidate, check if the prediction
        // alone is good enough to skip residual coding entirely.
        // When this fires, the mode loop breaks immediately -- no
        // further candidates are evaluated, and the intra fallback
        // is skipped.
        //
        // Port of `encode_breakout_test` (vp9_pickmode.c:933-1036).
        if params.encode_breakout > 0 {
            if let Some(ref b) = best {
                if is_new_best {
                    if crate::mode_decision::model_rd::encode_breakout_test(
                        b.var_y,
                        b.sse_y,
                        params.dequant_dc,
                        params.dequant_ac,
                        params.block_size,
                        b.best_mv,
                        params.encode_breakout,
                    ) {
                        break;
                    }
                }
            }
        }

        // G6 — early_term mode-loop exit (libvpx vp9_pickmode.c:2474-2478).
        // When the large model detects a full skip on a block >= 32x32,
        // exit after at least 2 candidates have been evaluated.
        if candidate.early_term {
            if let Some(ref b) = best {
                if b.best_rd_cost <= candidate.best_rd_cost || is_new_best {
                    // At least 2 candidates evaluated (this one + previous best),
                    // and the current or previous best has early_term set.
                    // Only break if we're past the first evaluated candidate.
                    if mode_checked.iter().flatten().filter(|&&c| c).count() > 1 {
                        break;
                    }
                }
            }
        }

        // Early-exit: if the best candidate so far is a perfect
        // skip (zero RD cost or skip_txfm == AcDc detected via zero
        // distortion + minimal rate), break early.
        if let Some(ref b) = best {
            if b.best_rd_cost == 0 {
                break;
            }
        }
    }

    // The loop iterates 8 non-empty candidates (LAST ZERO is always
    // reachable), so `best` is always populated.
    let mut best = best.expect("RT_INTER_MODES contains at least one evaluable candidate");

    // ================================================================
    // Phase 2b — intra-in-inter fallback (vp9_pickmode.c:2517-2646).
    //
    // After the inter mode loop, optionally evaluate DC/V/H intra
    // modes. If any intra mode beats the best inter candidate on RD,
    // the picker returns `RefFrame::IntraFrame`.
    // ================================================================
    let intra_cost_penalty = get_intra_cost_penalty(params.q_index, params.block_size);
    let inter_mode_thresh = rd_cost(lambda, intra_cost_penalty, 0);

    if best.best_rd_cost > inter_mode_thresh {
        let intra_result = evaluate_intra_fallback(params, lambda, intra_cost_penalty);
        if let Some(intra) = intra_result {
            if intra.best_rd_cost < best.best_rd_cost {
                // Intra wins on RD. Replace the inter result.
                //
                // **v1 wire-format note**: the caller's encoder
                // simulator path doesn't yet handle per-TX intra
                // prediction for intra-in-inter blocks. When the
                // caller sees `picked_ref_frame == IntraFrame`, it
                // falls back to encoding as the best inter candidate
                // (accepting the quality regression on those blocks).
                // The picker's intra evaluation is the measurement
                // signal; full wire-format support lands in a
                // follow-up.
                best = intra;
            }
        }
    }

    best
}

/// Intra modes evaluated in the intra-in-inter fallback.
/// Matches libvpx's `intra_mode_list[] = { DC_PRED, V_PRED, H_PRED,
/// TM_PRED }` at `vp9_pickmode.c:2556`, minus TM (deferred for v1
/// simplicity — see module-level doc).
const INTRA_IN_INTER_MODES: [IntraMode; 3] = [IntraMode::Dc, IntraMode::V, IntraMode::H];

/// Evaluate DC/V/H intra modes as a fallback inside `pick_inter_mode`.
///
/// Port of libvpx `vp9_pickmode.c:2517-2646`. For each candidate mode
/// this function:
///
/// 1. Builds the intra prediction from the reconstructed neighbour
///    edges provided in `params.intra_{above,left}_pixels`.
/// 2. Runs `model_rd_for_sb_y` to estimate variance-domain rate/dist
///    and detect skip-txfm blocks.
/// 3. When `skip_txfm != AcDc`, runs the full DCT round-trip via
///    `reconstruct_via_round_trip` for the coefficient-aware RD.
/// 4. Computes `rate = model_rd_rate + coef_rate + skip_cost +
///    mbmode_cost + intra_cost_penalty + ref_frame_cost_intra`.
/// 5. Returns the best intra candidate as a [`PickInterModeResult`]
///    with `picked_ref_frame = IntraFrame`, or `None` if no intra
///    params are available.
fn evaluate_intra_fallback(
    params: &PickInterModeParams<'_>,
    lambda: u32,
    intra_cost_penalty: u32,
) -> Option<PickInterModeResult> {
    let intra_costs = params.intra_y_mode_costs;

    // Skip for Block64x64 — the intra prediction kernels only handle
    // bs ∈ {4, 8, 16, 32}. libvpx's realtime path also caps intra
    // evaluation at `cpi->sf.max_intra_bsize` (typically BLOCK_32X32
    // for speed >= 5). Block64x64 intra-in-inter lands as a follow-up
    // with tiled 4×32x32 intra prediction.
    if params.block_size == BlockSize::Block64x64 {
        return None;
    }

    let bw = usize::from(params.block_size.width());
    let bh = usize::from(params.block_size.height());
    let tx_size = crate::block::max_txsize(params.block_size);
    let _tx_side = tx_size.side();

    let have_above = params.intra_above_pixels.is_some();
    let have_left = params.intra_left_pixels.is_some();

    // Materialise neighbour edges (libvpx 127-above / 129-left fill).
    let mut above_row = [ABOVE_UNAVAILABLE_FILL; 128];
    let mut left_col = [LEFT_UNAVAILABLE_FILL; 64];
    if let Some(above) = params.intra_above_pixels {
        let n = bw.min(above.len());
        above_row[..n].copy_from_slice(&above[..n]);
        let last = above_row[n.saturating_sub(1)];
        for slot in above_row.iter_mut().take(2 * bw).skip(bw) {
            *slot = last;
        }
    }
    if let Some(left) = params.intra_left_pixels {
        let n = bh.min(left.len());
        left_col[..n].copy_from_slice(&left[..n]);
    }
    let above_left = match (have_above, have_left) {
        (true, true) => params.intra_top_left_pixel,
        (true, false) => LEFT_UNAVAILABLE_FILL,
        (false, true) => ABOVE_UNAVAILABLE_FILL,
        (false, false) => ABOVE_UNAVAILABLE_FILL,
    };
    let dc_avail = DcAvailability {
        above_available: have_above,
        left_available: have_left,
    };
    let dispatch = crate::dsp::get();

    // Quantizer-derived thresholds for model_rd (same as inter path).
    let dc_zbin = params.quantizer.zbin[0] as i64;
    let ac_zbin = params.quantizer.zbin[1] as i64;
    let dc_thr = (dc_zbin * dc_zbin) >> 6;
    let ac_thr = (ac_zbin * ac_zbin) >> 6;
    let dc_quant = params.quantizer.dequant[0] as u32;
    let ac_quant = params.quantizer.dequant[1] as u32;
    let tx_mode = TxMode::TxModeSelect;

    let src_slice = &params.src_plane.data;
    let (sx, sy) = params.src_offset;
    let src_start = sy * params.src_plane.stride + sx;

    let mut best_intra: Option<PickInterModeResult> = None;

    for &mode in &INTRA_IN_INTER_MODES {
        // 1. Build prediction.
        let mut pred = [0u8; 64 * 64];
        let ctx = PredCtx {
            above: &above_row[..2 * bw],
            left: &left_col[..bh],
            above_left,
        };
        // intra_predict needs stride == bw for contiguous layout.
        intra_predict(mode, &mut pred[..bw * bh], bw, bw, ctx, dc_avail, &dispatch);

        // 2. Variance-domain RD model.
        let mrd = model_rd_for_sb_y(
            &src_slice[src_start..],
            params.src_plane.stride,
            &pred,
            bw,
            bw,
            bh,
            params.block_size,
            dc_quant,
            ac_quant,
            dc_thr,
            ac_thr,
            tx_mode,
        );

        let picked_tx_size = mrd.tx_size;

        // 3. DCT round-trip (or skip).
        let mut recon = [0u8; 64 * 64];
        recon[..bw * bh].copy_from_slice(&pred[..bw * bh]);

        let by = if mrd.skip_txfm == SkipTxfm::AcDc {
            BlockYrdResult { coef_rate: 0, coef_dist: 0, skippable: true }
        } else {
            // We need an InterModeSearchParams to call
            // `reconstruct_via_round_trip`. Build one using LAST ref
            // (arbitrary — the ref plane is not read for intra, but
            // the struct is needed for the quantizer/src fields).
            let single_params = InterModeSearchParams {
                block_size: params.block_size,
                src_plane: params.src_plane,
                src_offset: params.src_offset,
                ref_plane: params.last_ref_plane, // unused for intra
                mv_candidates: params.last_mv_candidates,
                mv_limits: params.mv_limits,
                site_config: params.site_config,
                mv_costs: params.mv_costs,
                inter_mode_costs: params.inter_mode_costs,
                inter_mode_ctx: params.inter_mode_ctx,
                quantizer: params.quantizer,
                q_index: params.q_index,
                allow_hp: params.allow_hp,
                frame_kind: params.frame_kind,
                newmv_seed: params.newmv_seed,
                ref_frame_rate_cost: 0,
                switchable_interp_costs: params.switchable_interp_costs,
                switchable_interp_ctx: params.switchable_interp_ctx,
                // G1: UV fields unused for intra fallback.
                src_u: params.src_u,
                src_v: params.src_v,
                ref_u: params.last_ref_u,
                ref_v: params.last_ref_v,
                uv_stride: params.uv_stride,
                uv_dc_quant: params.uv_dc_quant,
                uv_ac_quant: params.uv_ac_quant,
            };
            reconstruct_via_round_trip(
                &single_params, &pred, &mut recon, bw, bh, picked_tx_size,
            )
        };

        // 4. Distortion = SSE(src, recon).
        let sse = sse_block(params.src_plane, params.src_offset, &recon, bw, bw, bh);
        let dist = sse.min(u32::MAX as u64) as u32;

        // 5. Rate = mbmode_cost + ref_frame_cost[INTRA] +
        //           intra_cost_penalty + coef_rate.
        let mbmode_cost = intra_costs[mode as usize].max(0) as u32;
        let effective_coef_rate: u32 = by.coef_rate.min(u32::MAX as u64) as u32;
        let rate = mbmode_cost
            .saturating_add(params.ref_frame_cost_intra)
            .saturating_add(intra_cost_penalty)
            .saturating_add(effective_coef_rate);

        let this_rd = rd_cost(lambda, rate, dist);

        let is_new_best = match best_intra.as_ref() {
            None => true,
            Some(prev) => this_rd < prev.best_rd_cost,
        };
        if is_new_best {
            best_intra = Some(PickInterModeResult {
                best_mode: InterMode::ZeroMv, // placeholder for intra
                best_mv: Mv::ZERO,
                best_rd_cost: this_rd,
                best_distortion: dist,
                best_rate: rate,
                coef_rate: by.coef_rate,
                coef_dist: by.coef_dist,
                picked_interp_filter: InterpFilter::EightTap,
                picked_ref_frame: RefFrame::IntraFrame,
                picked_intra_mode: Some(mode),
                var_y: mrd.var_y,
                sse_y: mrd.sse_y,
            });
        }
    }

    best_intra
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
/// Returns the resulting MV in 1/8-pel units.
///
/// Two paths:
///
/// * **No seed** (`newmv_seed == None`): standard libvpx flow —
///   full-pel diamond search centred at `mv_candidates.nearest`
///   followed by sub-pel refinement. Mirrors
///   `vp9/encoder/vp9_pickmode.c:211-214` (`vp9_full_pixel_search`)
///   plus `:246-253` (`find_fractional_mv_step`).
///
/// * **With seed** (`newmv_seed == Some(int_pro_mv)`): SKIP the
///   full-pel diamond and use the int_pro MV directly. Mirrors
///   libvpx's `sb_use_mv_part` short-circuit at
///   `vp9_pickmode.c:207-210`:
///   ```c
///   if (x->sb_use_mv_part) {
///     tmp_mv->as_mv.row = x->sb_mvrow_part >> 3;
///     tmp_mv->as_mv.col = x->sb_mvcol_part >> 3;
///   } else {
///     vp9_full_pixel_search(...);
///   }
///   ```
///   Without this short-circuit, the diamond's internal
///   `mvsad_err_cost` (penalising MV distance from `ref_mv =
///   nearest`) drags the search back to `(0, 0)` whenever
///   `nearest = (0, 0)` (the typical case when neighbours haven't
///   picked NEWMV yet — chicken-and-egg). The int_pro estimate
///   already accounts for SAD-domain quality at the SB level, so
///   we trust it and only run sub-pel refinement around it.
fn run_newmv_search(params: &InterModeSearchParams<'_>) -> Mv {
    // Compute the full-pel-domain MV center for the sub-pel
    // refinement step. The `SubPixelSearchParams::center_mv` is a
    // FULL-PEL value (the sub-pel search lifts it via `* 8`
    // internally to start at the Q3 grid point).
    // libvpx-faithful realtime full-pel motion search: ALWAYS run the
    // diamond. The int_pro estimate (when present) and the NEAREST
    // candidate are *seeds* for the search start (`mvp_full`), NOT a
    // replacement for it — mirror of `combined_motion_search`
    // (`vp9_pickmode.c:154-261`) → `vp9_full_pixel_search`. The earlier
    // `sb_use_mv_part` short-circuit (E10) skipped the diamond whenever
    // int_pro produced any non-zero MV; in libvpx that short-circuit is
    // golden/altref-only (`vp9_pickmode.c:207-210` is gated by
    // `x->sb_use_mv_part`, set only for the GF/ARF int_pro fast path),
    // so the LAST-frame search must still run the diamond. E13: bypassing
    // it left 90% of NEWMV leaves stuck at a poor int_pro seed (+0.1 dB
    // over (0,0) vs +8 dB reachable).
    //
    // Cost attractor (`center_mv` / libvpx `ref_mv`) stays at NEAREST so
    // the rate term matches what the NEWMV bitstream cost will charge.
    // Start position (`start_mv` / libvpx `mvp_full`): evaluate SAD at
    // each MV-ref candidate (nearest, near) via `mv_pred_sad_seed` — port
    // of `vp9_mv_pred` (`vp9_rd.c:588-639`). Pick the candidate with the
    // lowest SAD as the diamond search seed. If an SB-level int_pro seed
    // is also available, keep whichever has the lower SAD.
    let ref_mv = clamp_mv_to_limits(params.mv_candidates.nearest, &params.mv_limits);
    let candidates = [params.mv_candidates.nearest, params.mv_candidates.near];
    let (pred_mv, pred_sad) = crate::mv_search::mv_pred::mv_pred_sad_seed(
        params.src_plane,
        params.ref_plane,
        params.src_offset,
        &candidates,
        params.block_size,
    );
    // Merge with SB-level int_pro seed: if int_pro exists, evaluate its
    // SAD too and keep whichever is better.
    let start_mv = if pred_sad < u32::MAX {
        // mv_pred found a viable seed.
        if let Some(int_pro) = params.newmv_seed {
            // Both seeds available — evaluate int_pro's SAD for comparison.
            let int_pro_fp_row = (int_pro.row + if int_pro.row >= 0 { 4 } else { 3 }) >> 3;
            let int_pro_fp_col = (int_pro.col + if int_pro.col >= 0 { 4 } else { 3 }) >> 3;
            let int_pro_sad = crate::mv_search::full_pixel::raw_sad_at_mv(
                params.block_size,
                params.src_plane,
                params.src_offset,
                params.ref_plane,
                int_pro_fp_row,
                int_pro_fp_col,
            )
            .unwrap_or(u32::MAX);
            if int_pro_sad < pred_sad {
                Some(clamp_mv_to_limits(int_pro, &params.mv_limits))
            } else {
                Some(clamp_mv_to_limits(pred_mv, &params.mv_limits))
            }
        } else {
            Some(clamp_mv_to_limits(pred_mv, &params.mv_limits))
        }
    } else {
        // mv_pred found nothing viable — fall back to int_pro seed.
        params.newmv_seed.map(|seed| clamp_mv_to_limits(seed, &params.mv_limits))
    };
    let full = diamond_search_sad(&FullPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv: ref_mv,
        start_mv,
        mv_limits: params.mv_limits,
        sad_per_bit: DEFAULT_SAD_PER_BIT,
        mv_costs: params.mv_costs,
        max_iters: params.site_config.total_steps as i32,
        site_config: params.site_config,
    });
    let center_mv_fp = Mv {
        row: full.best_mv_row,
        col: full.best_mv_col,
    };

    // E13 diamond-search-quality oracle (diagnostic-only, env-gated).
    // Records SAD@(0,0), SAD@diamond-output (NEWMVraw), and the
    // exhaustive full-pel minimum SAD over the same window, so the
    // harness can measure how much of the reachable full-pel gain the
    // diamond actually captures. No-op unless RVP9_E13_DIAMOND is set.
    if crate::mv_search::e12stat::enabled() {
        use crate::mv_search::full_pixel::{exhaustive_full_pel_min_sad, raw_sad_at_mv};
        let sad_zero = raw_sad_at_mv(
            params.block_size, params.src_plane, params.src_offset,
            params.ref_plane, 0, 0,
        )
        .unwrap_or(u32::MAX);
        let sad_newmv = raw_sad_at_mv(
            params.block_size, params.src_plane, params.src_offset,
            params.ref_plane, center_mv_fp.row, center_mv_fp.col,
        )
        .unwrap_or(u32::MAX);
        // Cap the exhaustive scan radius so the diagnostic stays
        // tractable; foreman's true motion is well within ±32 full-pel.
        let sad_oracle = exhaustive_full_pel_min_sad(
            params.block_size, params.src_plane, params.src_offset,
            params.ref_plane, &params.mv_limits, 48,
        )
        .map_or(u32::MAX, |(s, _, _)| s);
        let mv_mag = center_mv_fp.row.unsigned_abs() as u32
            + center_mv_fp.col.unsigned_abs() as u32;
        crate::mv_search::e12stat::record(
            sad_zero,
            sad_newmv,
            sad_oracle,
            params.newmv_seed.is_some(),
            mv_mag,
        );
    }

    // P14 — adaptive `subpel_force_stop` (libvpx
    // `vp9_pickmode.c:239-246`). For large full-pel MVs we stop at
    // 1/2-pel; for zero/small full-pel MVs we refine to 1/4-pel. The
    // libvpx speed-9 thresholds are reproduced bug-for-bug in
    // [`adaptive_subpel_force_stop`]. rvp9 locks the rt path to
    // cpu-used 6-8 and applies the adaptive selector unconditionally
    // — strictly this is wider than libvpx (which only enables it at
    // speed 9 via `vp9_speed_features.c:803-806`) but the rvp9 target
    // is a single rt config, so we follow the rt-leaning behaviour
    // here. EIGHTH_PEL is no longer reachable for NEWMV; libvpx's
    // speed-9 setter likewise never selects it for NEWMV.
    let forced_stop = adaptive_subpel_force_stop(center_mv_fp);
    let sub = find_best_sub_pixel_tree_pruned_more(&SubPixelSearchParams {
        block_size: params.block_size,
        src_plane: params.src_plane,
        src_offset: params.src_offset,
        ref_plane: params.ref_plane,
        center_mv: center_mv_fp,
        ref_mv: params.mv_candidates.nearest,
        mv_limits: params.mv_limits,
        error_per_bit: DEFAULT_ERROR_PER_BIT,
        mv_costs: params.mv_costs,
        allow_hp: params.allow_hp,
        forced_stop,
        // libvpx rt default: `sf.mv.subpel_search_level = 2`
        // (`vp9_speed_features.c:929`), enabling SECOND_LEVEL_CHECKS
        // after each pyramid level's FIRST_LEVEL_CHECKS.
        iters_per_step: 2,
    });
    sub.best_mv
}

/// F.5 — per-block interpolation-filter search set (unconditionally
/// active since Phase 2c). Libvpx-realtime iterates
/// `EIGHTTAP (0) .. EIGHTTAP_SMOOTH (1)`
/// (`vp9_pickmode.c:1512-1514`, `search_filter_ref`); the 2-filter set
/// is the team-lead-approved starting point.
///
/// The frame header's `interp_filter` is set to `Switchable` so the
/// per-block filter bits go on the wire; the picker's RD comparison
/// decides per-block which filter wins.
fn f5_filter_search_set() -> &'static [InterpFilter] {
    &[InterpFilter::EightTap, InterpFilter::EightTapSmooth]
}

/// Evaluate one `(mode, mv)` candidate: fetch the prediction with sub-pel
/// interpolation, run the forward/inverse transform round-trip,
/// compute SSE + rate cost + RD cost.
///
/// **F.5** — runs the per-block filter search loop (libvpx
/// `search_filter_ref`, `vp9_pickmode.c:1488-1573`). For each filter
/// in [`f5_filter_search_set`] (`[EightTap, EightTapSmooth]`) the
/// prediction is rebuilt and the full `block_yrd` + RD evaluation
/// runs; the lowest-RD winner is returned with
/// `picked_interp_filter` set accordingly.
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
    // F.5 — per-block filter search. Each filter rebuilds prediction +
    // residual + DCT + quant + idct + SSE; the picker takes the
    // lowest-RD winner. The MV is fixed for the duration of the loop
    // (mirror of libvpx's realtime `search_filter_ref`, which also
    // does MV search once outside the filter loop).
    let mut best: Option<InterModeSearchResult> = None;
    for &filter in f5_filter_search_set() {
        let candidate = evaluate_candidate_for_filter(params, mode, mv, ref_mv, lambda, filter);
        match best.as_ref() {
            Some(prev) if candidate.best_rd_cost >= prev.best_rd_cost => {}
            _ => best = Some(candidate),
        }
    }
    // The loop iterates `[EightTap, EightTapSmooth]` — always non-empty,
    // so `best` is always populated.
    best.expect("F.5 filter search set is non-empty")
}

/// F.5 — evaluate one `(mode, mv, filter)` triplet. The original
/// pre-F.5 `evaluate_candidate` body, with the filter parameter
/// threaded into `build_inter_prediction` and the switchable-filter
/// signalling rate added to the rate sum for non-default filters.
fn evaluate_candidate_for_filter(
    params: &InterModeSearchParams<'_>,
    mode: InterMode,
    mv: Mv,
    ref_mv: Mv,
    lambda: u32,
    filter: InterpFilter,
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
        filter,
    );

    // --- Variance-domain RD model (model_rd_for_sb_y port) ---
    //
    // Before the expensive DCT round-trip, run the Laplacian–Gaussian
    // variance model to (a) pick a TX size adaptively and (b) detect
    // blocks whose residual will quantize entirely to zero
    // (skip_txfm == AcDc). For such blocks we short-circuit: the
    // distortion is `sse << 4` (libvpx convention) and the rate
    // component from coefficients is zero.
    //
    // quant_thred = zbin^2; model_rd thresholds = quant_thred >> 6.
    let dc_zbin = params.quantizer.zbin[0] as i64;
    let ac_zbin = params.quantizer.zbin[1] as i64;
    let dc_thr = (dc_zbin * dc_zbin) >> 6;
    let ac_thr = (ac_zbin * ac_zbin) >> 6;
    let dc_quant = params.quantizer.dequant[0] as u32;
    let ac_quant = params.quantizer.dequant[1] as u32;

    // TxModeSelect lets `calculate_tx_size` pick between TX_8X8 and
    // `max_txsize(bsize)` based on the variance / SSE ratio, matching
    // libvpx's realtime path at cpu-used 6-8 (`vp9_pickmode.c`).
    let tx_mode = TxMode::TxModeSelect;

    let src_slice = &params.src_plane.data;
    let (sx, sy) = params.src_offset;
    let src_start = sy * params.src_plane.stride + sx;
    // G6: dispatch to the per-sub-block variant for blocks >= 32x32.
    let use_large = params.block_size >= BlockSize::Block32x32;
    let mrd = if use_large {
        model_rd_for_sb_y_large(
            &src_slice[src_start..],
            params.src_plane.stride,
            &pred,
            bw,
            bw,
            bh,
            params.block_size,
            dc_quant,
            ac_quant,
            dc_thr,
            ac_thr,
            tx_mode,
        )
    } else {
        model_rd_for_sb_y(
            &src_slice[src_start..],
            params.src_plane.stride,
            &pred,
            bw, // pred_stride
            bw,
            bh,
            params.block_size,
            dc_quant,
            ac_quant,
            dc_thr,
            ac_thr,
            tx_mode,
        )
    };

    // Use the model_rd TX size for the round-trip (replaces the
    // previous hard-coded `max_txsize(block_size)`).
    let tx_size = mrd.tx_size;

    // Compute the reconstruction. For Block16x16 / Block32x32 the
    // block matches its max_txsize and we do one big transform.
    // For Block64x64 we tile 4 × 32x32 transforms.
    //
    // **F.1**: the round-trip helper now also returns a libvpx-style
    // [`BlockYrdResult`] (coef-rate / coef-dist / skippable). The
    // `coef_rate` term is added to the rate sum below; `skippable`
    // feeds the libvpx skip-flag tie-break further down.
    let mut recon = [0u8; 64 * 64];
    recon[..bw * bh].copy_from_slice(&pred[..bw * bh]);

    // skip_txfm short-circuit (libvpx `vp9_pickmode.c:684-686`):
    // When model_rd says both DC and AC quantize to zero, the entire
    // residual vanishes. The prediction IS the reconstruction — `recon`
    // already holds a copy of `pred`. Skip the expensive DCT round-trip
    // and produce the same `BlockYrdResult` that `block_yrd` would for
    // an all-zero qcoeff block: coef_rate=0, coef_dist=0, skippable=true.
    let by = if mrd.skip_txfm == SkipTxfm::AcDc {
        BlockYrdResult { coef_rate: 0, coef_dist: 0, skippable: true }
    } else {
        reconstruct_via_round_trip(params, &pred, &mut recon, bw, bh, tx_size)
    };

    // Distortion = SSE(src, recon). For block sizes ≤ 64×64 the SSE
    // fits in u32 (max 64*64*255^2 = 266_342_400 < 2^32).
    //
    // A8 (2026-05-22) tried source-domain SSE (SSE(src, pred)) here
    // to activate NEWMV picks on textured motion content (foreman
    // gate). Result: NEWMV picks went 0 → 3788 (foreman_cif), but
    // foreman PSNR stayed locked at 16 dB (RC q=255 saturation is
    // the real blocker; source-domain SSE only helps at lower q,
    // where the picker already had room to differentiate) AND
    // testsrc 250-kbps PSNR regressed by 1.92 dB (high-bitrate
    // residuals survive quant; with source-domain SSE the picker
    // ignores the residual rate downstream cost and picks NEWMV
    // too aggressively). Reverted. Notes in
    // `wrk_docs/A8-source-sse-foreman-report.md`.
    //
    // A9 (2026-05-22) re-applied A8 *combined* with an inter q-cap
    // at 200 (`RateControlConfig::inter_max_qindex`). Foreman_cif
    // still locked at 16-17 dB even with q now in [135, 195] across
    // every inter frame and 6.2k NEWMV picks active. Per-frame
    // recon SSE on foreman drifts from 3.4M (KF) to 143M (frame 31)
    // while source-consecutive SSE stays at 6-17M — MC fundamentally
    // can't track the head/camera motion at the picker+RC settings
    // we have. testsrc 250-kbps regression preserved (-1.92 dB).
    // The 16-dB foreman floor is a deeper rate-model problem (the
    // bits_per_mb estimator badly under-predicts foreman's per-MB
    // cost, so even with a q-cap the encoder runs 10x the target
    // bitrate AND still can't make MC stick). Reverted. Notes in
    // the 2026-05-22 A9 entry of `notes/divergence-ledger.md`.
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
    // the MV-bit cost (libvpx `vp9_mv_bit_cost`,
    // `vp9_mcomp.c:80-84`):
    //
    //     ROUND_POWER_OF_TWO(mv_cost(diff) * MV_COST_WEIGHT, 7)
    //
    // with `MV_COST_WEIGHT = 108` (`vp9_rd.h:38`). The 108/128 ≈ 0.84
    // damps the raw `mv_cost` before it is added to `mode_rate` and
    // fed into RDCOST. Skipping the scaling over-charges MV bits by
    // 128/108 ≈ 1.185×, which biases the picker against NEWMV.
    let mode_rate = params.inter_mode_costs[params.inter_mode_ctx][mode.idx()]
        .max(0) as u32;
    let mv_rate = if matches!(mode, InterMode::NewMv) {
        let diff_row = mv.row.wrapping_sub(ref_mv.row);
        let diff_col = mv.col.wrapping_sub(ref_mv.col);
        let raw = mv_cost(diff_row, diff_col, params.mv_costs).max(0) as u64;
        // ROUND_POWER_OF_TWO(raw * 108, 7).
        let scaled = (raw * 108 + 64) >> 7;
        scaled.min(u32::MAX as u64) as u32
    } else {
        0
    };
    // A16 (2026-05-22) — add the per-ref-frame rate cost. For single-
    // ref dual-ref-picker callers this carries the LAST-vs-GOLDEN bit
    // differential (libvpx `vp9_pickmode.c:2398`,
    // `this_rdc.rate += ref_frame_cost[ref_frame]`). For pre-A16
    // single-ref-only callers the field defaults to zero and the RD
    // is unchanged (preserves byte-exact LAST-only output).
    //
    // **F.1**: `by.coef_rate` is the libvpx `block_yrd` coefficient-bit
    // estimate (SATD-of-qcoeff in `vp9_prob_cost` units, see
    // `mode_decision::block_yrd`). Adding it to the rate sum is the
    // primary F.1 lever — previously the picker compared (mode + mv +
    // ref) against (mode + mv + ref), and a candidate with a strictly
    // better prediction but a non-zero `mv_rate` would lose to a
    // rate-free ZEROMV / NEAREST with a large residual.
    //
    // **Default: DISABLED at F.1 sign-off (team-lead decision,
    // 2026-05-23).** Empirical finding during F.1 bring-up: foreman
    // PSNR-delta regresses +9.07 → +10.55 dB with `coef_rate` active
    // alone, with the 250 kbps operating point catastrophically
    // collapsing from 26.96 → 20.91 dB. Root cause: libvpx's
    // `<< (2 + VP9_PROB_COST_SHIFT)` SATD-to-prob-cost scale is
    // calibrated against libvpx's lambda + RD ranking. rvp9's lambda
    // (`compute_rd_mult`) was calibrated when the picker had no
    // `coef_rate` term. At higher bitrate operating points where
    // residuals are non-trivial, the libvpx-tuned scale overprices
    // candidates → picker falls back to worse modes → PSNR collapses.
    //
    // **F.4 calibrates lambda + the SATD-to-bits scale** as a pair
    // (`vp9_pickmode.c:RD_THRESH_*` and `compute_rd_mult` together).
    // Setting `RVP9_F1_COEF_RATE_ENABLE=1` activates `coef_rate` now
    // for diagnostic / A/B work.
    //
    // F.1 ships as INFRASTRUCTURE-ONLY: the `block_yrd` module exists,
    // `BlockYrdResult` flows through `reconstruct_via_round_trip`,
    // `InterModeSearchResult` carries `coef_rate` and `coef_dist`, and
    // the F.0.2 oracle's `coef_bits_est` field reads them. F.2 / F.3
    // land on this infrastructure; F.4 turns on the behaviour after
    // calibrating lambda.
    // F.4-2 (2026-05-23): `coef_rate` is now unconditionally part of
    // the rate sum (env-gate removed per F.4 HLD §4.2). The F.4-1
    // sweep (notes/F4-calibration-sweep.md, night-8 corrected) picked
    // K1=4 at `inter.rs::reconstruct_via_round_trip` as the winning
    // SATD-to-prob-cost scale exponent: foreman PSNR-delta drops
    // +9.07 → +9.00 dB (modest +0.07 dB improvement; full +8 dB
    // closure to libvpx needs structural work — DV-F1-1 Hadamard
    // port — beyond F.4-2 scope).
    //
    // `by.coef_rate` is `u64` to avoid the early saturation that
    // hid K1 calibration signal at low q (night-8 saturation fix).
    // Saturate-cast here at the honest cap site before adding to
    // the `u32` rate sum.
    let effective_coef_rate: u32 = by.coef_rate.min(u32::MAX as u64) as u32;

    // G2 — per-context switchable-filter signalling cost. Replaces
    // the flat `F5_SWITCHABLE_FILTER_RATE = 2048` constant with the
    // true per-context cost from `switchable_interp_costs[ctx][filter]`.
    // ALL filters (including EightTap) get their true cost; the old
    // `EightTap == 0` special-case is gone — the table already encodes
    // EightTap's cost (low in ctx=0 but nonzero).
    let filter_rate: u32 = params.switchable_interp_costs
        [params.switchable_interp_ctx][filter as usize]
        .max(0) as u32;

    // G1 — UV model-RD (libvpx `vp9_pickmode.c:2378-2393`).
    //
    // Build chroma MC predictions and run the variance-domain RD model
    // for both U and V planes. The UV rate+dist are added to the
    // luma total so the picker can distinguish candidates that differ
    // in chroma fidelity.
    //
    // Only runs when UV planes are provided (non-empty src_u). Test
    // harnesses pass empty slices to skip this path.
    let (uv_rate, uv_dist): (u32, i64) = if !params.src_u.is_empty()
        && !params.ref_u.is_empty()
        && params.uv_stride > 0
    {
        let uv_bw = bw >> 1;
        let uv_bh = bh >> 1;
        // Chroma offset = luma offset >> 1 (4:2:0).
        let uv_x = params.src_offset.0 >> 1;
        let uv_y = params.src_offset.1 >> 1;

        // Chroma MV: VP9 4:2:0 scales the luma Q3 MV by halving for
        // the half-resolution chroma plane. `build_inter_prediction`
        // interprets its MV in Q3 of the plane's native pixel grid,
        // so we pass `mv / 2` to get the correct integer + sub-pel
        // displacement on the chroma plane.
        let uv_mv = Mv {
            row: mv.row / 2,
            col: mv.col / 2,
        };

        // Build UV MC predictions into scratch buffers.
        let mut pred_u = [0u8; 32 * 32];
        let mut pred_v = [0u8; 32 * 32];
        {
            let ref_u_plane = PlaneRef {
                data: params.ref_u,
                stride: params.uv_stride,
                width: params.uv_stride, // full chroma width
                height: params.ref_u.len() / params.uv_stride,
            };
            build_inter_prediction(
                &ref_u_plane, uv_x, uv_y, uv_mv, uv_bw, uv_bh,
                &mut pred_u, uv_bw, filter,
            );
        }
        {
            let ref_v_plane = PlaneRef {
                data: params.ref_v,
                stride: params.uv_stride,
                width: params.uv_stride,
                height: params.ref_v.len() / params.uv_stride,
            };
            build_inter_prediction(
                &ref_v_plane, uv_x, uv_y, uv_mv, uv_bw, uv_bh,
                &mut pred_v, uv_bw, filter,
            );
        }

        // Source UV at the chroma offset.
        let src_u_start = uv_y * params.uv_stride + uv_x;
        let src_v_start = uv_y * params.uv_stride + uv_x;

        let uv_rd = model_rd_for_sb_uv(
            &params.src_u[src_u_start..], params.uv_stride,
            &pred_u, uv_bw,
            &params.src_v[src_v_start..], params.uv_stride,
            &pred_v, uv_bw,
            bw, bh, // luma dimensions — halved internally
            params.block_size,
            params.uv_dc_quant,
            params.uv_ac_quant,
        );

        (uv_rd.rate.max(0) as u32, uv_rd.dist)
    } else {
        (0, 0)
    };

    // Add UV dist to luma dist (libvpx `vp9_pickmode.c:2390-2393`).
    let dist = (dist as i64 + uv_dist).clamp(0, u32::MAX as i64) as u32;

    let rate = mode_rate
        .saturating_add(mv_rate)
        .saturating_add(params.ref_frame_rate_cost)
        .saturating_add(effective_coef_rate)
        .saturating_add(filter_rate)
        .saturating_add(uv_rate);

    // **F.1 — libvpx skip-flag tie-break (vp9_pickmode.c:2355-2362, Q5).**
    // Unconditionally active since Phase 2c.
    //
    // ```c
    // if (RDCOST(rdmult, rddiv, this_rdc.rate, this_rdc.dist) <
    //     RDCOST(rdmult, rddiv, 0, this_sse)) {
    //   this_rdc.rate += vp9_cost_bit(skip_prob, 0);
    // } else {
    //   this_rdc.rate = vp9_cost_bit(skip_prob, 1);
    //   this_rdc.dist = this_sse;
    //   x->skip_txfm[0] = SKIP_TXFM_AC_DC;
    // }
    // ```
    //
    // `this_sse` here is libvpx's caller-supplied `sse_y` from
    // `model_rd_for_sb_y` — pixel-domain `sum((src - pred)^2)`
    // (NOT post-recon). The picker uses pixel-SSE-of-recon as
    // its dist term (Q3), so we mirror libvpx's tie-break with
    // a freshly computed `sse_after_pred = sum((src - pred)^2)`.
    //
    // `skip_prob` — libvpx looks this up from the per-block
    // `skip_ctx` against `cm->fc->skip_probs[skip_ctx]`. The
    // picker doesn't carry frame context, so we use the middle
    // skip context default (`DEFAULT_SKIP_PROBS[1] = 128`,
    // neutral 50/50 — `vp9_cost_bit(128, 0) ==
    // vp9_cost_bit(128, 1) == 512` = 1 bit in 1/512 units).
    // Threading the real per-block skip context is a follow-up.
    let sse_after_pred = sse_block(
        params.src_plane,
        params.src_offset,
        &pred,
        bw,
        bw,
        bh,
    );
    let sse_after_pred_u32 = sse_after_pred.min(u32::MAX as u64) as u32;
    let skip_prob =
        crate::entropy::default_probs::DEFAULT_SKIP_PROBS[F1_DEFAULT_SKIP_CTX];
    let skip_false_cost = crate::cost::vp9_cost_bit(skip_prob, false);
    let skip_true_cost = crate::cost::vp9_cost_bit(skip_prob, true);

    let rd_encode = rd_cost(lambda, rate, dist);
    let rd_skip = rd_cost(lambda, 0, sse_after_pred_u32);
    let (final_rate, final_dist) = if rd_encode < rd_skip {
        // Cheaper to encode the residual. Pay the skip=false bit.
        (rate.saturating_add(skip_false_cost), dist)
    } else {
        // Cheaper to skip. The decoder will see `sse_after_pred`
        // as the distortion (no residual added on the inverse
        // path); the rate collapses to the skip=true bit.
        (skip_true_cost, sse_after_pred_u32)
    };
    let this_rd = rd_cost(lambda, final_rate, final_dist);

    // G6: set early_term when the large model detects a full skip on
    // a block >= 32x32.
    let early_term = use_large && mrd.skip_txfm == SkipTxfm::AcDc;

    InterModeSearchResult {
        best_mode: mode,
        best_mv: mv,
        best_rd_cost: this_rd,
        best_distortion: final_dist,
        best_rate: final_rate,
        coef_rate: by.coef_rate,
        coef_dist: by.coef_dist,
        picked_interp_filter: filter,
        var_y: mrd.var_y,
        sse_y: mrd.sse_y,
        early_term,
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
    filter: InterpFilter,
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
    // F.5 — the kernel is now per-block: the picker drives it via the
    // filter loop in `evaluate_candidate`; callers outside the picker
    // pass `EightTap` to preserve pre-F.5 behaviour.
    let kernel = kernel_for(filter);
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
///
/// **F.1**: also accumulates libvpx-style `block_yrd` outputs across
/// the per-tile loop and returns them in [`BlockYrdResult`]. The picker
/// uses `coef_rate` to add coefficient-bit pressure to its RD
/// comparison and `skippable` for the libvpx skip-flag tie-break
/// (`vp9_pickmode.c:2355-2362`). `coef_dist` is surfaced for F.4 A/B
/// experiments — the F.1 picker keeps pixel-SSE-of-recon as the RD
/// distortion term (DV-F1-3). Mirrors libvpx `block_yrd` at
/// `vp9_pickmode.c:718-844` v1.14.1 (`12f3a2ac6`). See
/// [`crate::mode_decision::block_yrd`] for the module-level reference.
fn reconstruct_via_round_trip(
    params: &InterModeSearchParams<'_>,
    pred: &[u8],
    recon: &mut [u8],
    bw: usize,
    bh: usize,
    tx_size: TxSize,
) -> BlockYrdResult {
    let tx_side = tx_size.side();
    let n = tx_size.n_coeffs();
    let src = params.src_plane;
    let (sx, sy) = params.src_offset;

    // libvpx block_yrd accumulators (see vp9_pickmode.c:733, 768, 805).
    let mut by_skippable = true;
    let mut by_eob_cost: u32 = 0;
    let mut by_rate_satd: u32 = 0;
    let mut by_dist: i64 = 0;

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

    // Hadamard + quantize_fp scratch for block_yrd rate/dist (Phase 4c).
    // The Hadamard transform produces i16; vp9_quantize_fp needs i32 input.
    let mut had_coeff_i16 = [0i16; 1024];
    let mut had_coeff = [0i32; 1024];
    let mut had_qcoeff = [0i16; 1024];
    let mut had_dqcoeff = [0i16; 1024];

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
                    return BlockYrdResult::default();
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

            // F.1 / Phase 4c — per-tile block_yrd accumulators via
            // Hadamard + quantize_fp (libvpx vp9_pickmode.c:768,
            // 804-805, 818-835). The Hadamard path produces the
            // rate/dist metrics; the DCT pipeline above stays for
            // reconstruction.
            //
            // 1. Hadamard transform on the residual.
            match tx_size {
                TxSize::Tx8x8 => {
                    hadamard_8x8(&residual[..64], tx_side, &mut had_coeff_i16[..64]);
                }
                TxSize::Tx16x16 => {
                    hadamard_16x16(&residual[..256], tx_side, &mut had_coeff_i16[..256]);
                }
                TxSize::Tx32x32 => {
                    hadamard_32x32(&residual[..1024], tx_side, &mut had_coeff_i16[..1024]);
                }
                TxSize::Tx4x4 => unreachable!(),
            }
            // Widen Hadamard i16 → i32 for quantize_fp input.
            for i in 0..n {
                had_coeff[i] = i32::from(had_coeff_i16[i]);
            }
            // 2. quantize_fp on Hadamard coefficients.
            let had_eob = if tx_size == TxSize::Tx32x32 {
                vp9_quantize_fp_32x32(
                    &had_coeff[..1024],
                    false,
                    &params.quantizer.round_fp,
                    &params.quantizer.quant_fp,
                    &mut had_qcoeff[..1024],
                    &mut had_dqcoeff[..1024],
                    &dequant_pair,
                    scan_slice,
                )
            } else {
                vp9_quantize_fp(
                    &had_coeff[..n],
                    n,
                    false,
                    &params.quantizer.round_fp,
                    &params.quantizer.quant_fp,
                    &mut had_qcoeff[..n],
                    &mut had_dqcoeff[..n],
                    &dequant_pair,
                    scan_slice,
                )
            };
            // 3. Accumulate block_yrd metrics from Hadamard-quantized
            //    coefficients (libvpx vp9_pickmode.c:804, 830-835).
            by_skippable &= had_eob == 0;
            by_eob_cost = by_eob_cost.saturating_add(1);
            if had_eob == 1 {
                by_rate_satd = by_rate_satd
                    .saturating_add(i32::from(had_qcoeff[0]).unsigned_abs());
            } else if had_eob > 1 {
                by_rate_satd =
                    by_rate_satd.saturating_add(satd_i16(&had_qcoeff[..n]));
            }
            by_dist = by_dist
                .saturating_add(block_error_fp_i32(&had_coeff[..n], &had_dqcoeff[..n]) >> 2);

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

    // libvpx vp9_pickmode.c:842-843 — scale rate_satd to vp9_prob_cost
    // units and add the per-tile EOB charge. The combined shift is
    // (2 + VP9_PROB_COST_SHIFT) = 11. Now matches libvpx exactly:
    // Hadamard + quantize_fp produces the SATD rate proxy at the same
    // scale libvpx uses, so K1=2 (the libvpx default) is correct.
    // **u64 (not u32) to preserve calibration signal**: pre-fix the
    // value saturated at u32::MAX here, which masked K1 sweeps — low-q
    // blocks (large qcoeffs ⇒ rate_satd > 2_097_152 ⇒ shifted_rate >
    // u32::MAX) all clipped to u32::MAX. Saturate-cast back to u32
    // happens in `evaluate_candidate` at the rate-sum site.
    let shifted_rate =
        (by_rate_satd as u64) << (2 + crate::cost::VP9_PROB_COST_SHIFT);
    let eob_term = (by_eob_cost as u64) << crate::cost::VP9_PROB_COST_SHIFT;
    let coef_rate = shifted_rate.saturating_add(eob_term);
    BlockYrdResult {
        coef_rate,
        coef_dist: by_dist,
        skippable: by_skippable,
    }
}

/// SSE between source-plane block at `(sx, sy)` (stride
/// `src_plane.stride`) and a destination block `dst` (size `bw × bh`,
/// stride `dst_stride`). Returns the raw `u64` so the caller can
/// decide how to clamp.
///
/// Used for both the post-recon distortion term (`dst = recon`) and
/// the F.1 skip-flag tie-break's pre-recon "would-be skip" SSE
/// (`dst = pred`, libvpx `vp9_pickmode.c:2355`).
fn sse_block(
    src_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    dst: &[u8],
    dst_stride: usize,
    bw: usize,
    bh: usize,
) -> u64 {
    let (sx, sy) = src_offset;
    let mut sse: u64 = 0;
    for r in 0..bh {
        for c in 0..bw {
            let s = src_plane.data[(sy + r) * src_plane.stride + (sx + c)] as i32;
            let d = dst[r * dst_stride + c] as i32;
            let diff = s - d;
            sse += (diff * diff) as u64;
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
    use crate::cost::symbol_costs::{inter_mode_costs, switchable_interp_costs};
    use crate::cost::SwitchableInterpCosts;
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
        sic: SwitchableInterpCosts,
    }
    impl TestHarness {
        fn new(plane_w: usize, q_index: u8) -> Self {
            let fc = FrameContext::new_default();
            let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
            let inter_costs = inter_mode_costs(&fc);
            let sic = switchable_interp_costs(&fc);
            let quantizer =
                init_plane_quantizers(i32::from(q_index), QuantizerDeltas::Y);
            let cfg = init_dsmotion_compensation(plane_w as i32);
            Self {
                costs,
                inter_costs,
                quantizer,
                cfg,
                sic,
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
            ref_frame_rate_cost: 0,
            switchable_interp_costs: &harness.sic,
            switchable_interp_ctx: 3, // no-neighbour sentinel
            // G1: UV fields — empty for luma-only unit tests.
            src_u: &[],
            src_v: &[],
            ref_u: &[],
            ref_v: &[],
            uv_stride: 0,
            uv_dc_quant: 0,
            uv_ac_quant: 0,
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
        // ZEROMV, NEARESTMV and NEARMV all produce identical zero-MV
        // predictions here (mv_candidates.nearest = .near = (0, 0)),
        // and all three quantize to zero residual against an identical
        // ref → distortion is zero for each. The winner is the
        // cheapest-rate mode under the default inter_mode_probs at
        // ctx=0. Any zero-MV winner is correct; what matters is that
        // NEWMV did **not** win (which would have wasted bits
        // encoding an MV residual against an already-perfect
        // prediction).
        assert!(
            matches!(
                res.best_mode,
                InterMode::ZeroMv | InterMode::NearestMv | InterMode::NearMv
            ),
            "identical planes → expected ZEROMV / NEARESTMV / NEARMV, got {:?} (rd={}, dist={}, rate={})",
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
        let new = &all[3];

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

    /// All 4 candidates are evaluated — `pick_inter_mode_luma_all_costs`
    /// returns exactly 4 entries in
    /// `[ZEROMV, NEARESTMV, NEARMV, NEWMV]` order. Each entry's mode
    /// field must match the slot's intended mode.
    #[test]
    fn pick_inter_walks_4_candidates() {
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
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].best_mode, InterMode::ZeroMv);
        assert_eq!(all[1].best_mode, InterMode::NearestMv);
        assert_eq!(all[2].best_mode, InterMode::NearMv);
        assert_eq!(all[3].best_mode, InterMode::NewMv);
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
            // Same-plane → ZEROMV / NEARESTMV / NEARMV predictions all
            // collapse to zero distortion; any of the three is a valid
            // winner (rate-cost tiebreaker, see
            // `pick_inter_zero_mv_on_identical_ref` for details). The
            // failure mode we're actually guarding against is NEWMV
            // winning over a zero-MV mode (that would mean the search
            // wasted bits on a useless MV residual).
            assert!(
                matches!(
                    res.best_mode,
                    InterMode::ZeroMv | InterMode::NearestMv | InterMode::NearMv
                ),
                "{bs:?} identical src/ref: expected ZEROMV / NEARESTMV / NEARMV winner, got {:?}",
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
        let src_data = make_plane(pw, ph, 41);
        // Reference: horizontally shifted copy of the source. The
        // ZEROMV prediction (offset 0) sees a large residual against
        // the shifted content, whose transform coefficients survive
        // quantization at low q_index, making `rd_encode < rd_skip`
        // so the mode-rate differential from inter_mode_ctx is visible
        // through the skip tie-break.
        let mut ref_data = vec![0u8; pw * ph];
        for r in 0..ph {
            for c in 0..pw {
                // Shift source right by 2 pixels (wrapping at row boundary).
                let sc = (c + 2) % pw;
                ref_data[r * pw + c] = src_data[r * pw + sc];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        // q_index = 10: low quantization so transform coefficients
        // survive and the encode path beats the skip path.
        let harness = TestHarness::new(pw, 10);
        let mvc = MvCandidates::default();

        let params0 = InterModeSearchParams {
            block_size: BlockSize::Block16x16,
            src_plane: &src_plane,
            src_offset: (16, 16),
            ref_plane: &ref_plane,
            mv_candidates: &mvc,
            mv_limits: MvLimits::new(-4, 4, -4, 4),
            site_config: &harness.cfg,
            mv_costs: &harness.costs,
            inter_mode_costs: &harness.inter_costs,
            inter_mode_ctx: 0,
            quantizer: &harness.quantizer,
            q_index: 10,
            allow_hp: false,
            frame_kind: synthetic_inter_frame_kind(),
            newmv_seed: None,
            ref_frame_rate_cost: 0,
            switchable_interp_costs: &harness.sic,
            switchable_interp_ctx: 3,
            src_u: &[],
            src_v: &[],
            ref_u: &[],
            ref_v: &[],
            uv_stride: 0,
            uv_dc_quant: 0,
            uv_ac_quant: 0,
        };
        let params6 = InterModeSearchParams {
            inter_mode_ctx: 6,
            ..params0
        };

        let r0 = pick_inter_mode_luma(&params0);
        let r6 = pick_inter_mode_luma(&params6);
        // Different inter_mode_ctx values produce different mode-rate
        // costs. With shifted src/ref content at low q, the transform
        // coefficients survive quantization so the skip tie-break
        // chooses the encode path, making the mode-cost differential
        // visible in `best_rate`.
        assert_ne!(
            r0.best_rate, r6.best_rate,
            "ctx 0 and ctx 6 must produce different rates via the inter_mode_costs row differential",
        );
    }

    // ====================================================================
    // F.5 — interpolation filter search loop tests.
    // ====================================================================

    /// F.5 — identical source and reference → ZEROMV predictions are
    /// pixel-perfect, residuals quantize to zero, distortion is zero
    /// for every filter. With the skip tie-break active (Phase 2c),
    /// both filters collapse to `(skip_true_cost, 0)` — identical
    /// rate and distortion. The picker's `evaluate_candidate` loop
    /// breaks ties in favour of the first-in-slice filter (EightTap).
    ///
    /// This is the "no incentive to switch" oracle: when prediction
    /// quality can't be improved by changing filters, EightTap wins.
    #[test]
    fn f5_picker_chooses_eighttap_on_identical_src_ref() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 71);
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
            MvLimits::new(-4, 4, -4, 4),
            50,
        );

        // Direct per-filter evaluation: both filters should produce
        // zero distortion (identical src/ref, integer-pel MV → both
        // filters copy the centre pixel).
        let lambda =
            compute_rd_mult(50, frame_type_for_rd(synthetic_inter_frame_kind()));
        let eight = evaluate_candidate_for_filter(
            &params,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );
        let smooth = evaluate_candidate_for_filter(
            &params,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTapSmooth,
        );
        // Distortion is zero on both (sub-pel phase is zero → both
        // filters tap-128 copy the centre pixel; the convolve_copy
        // fast path also runs).
        assert_eq!(eight.best_distortion, 0);
        assert_eq!(smooth.best_distortion, 0);
        // With the skip tie-break (Phase 2c), both candidates collapse
        // to `(skip_true_cost, 0)` — identical rate and RD. The
        // filter tags are still correctly set on each result.
        assert_eq!(
            eight.best_rate, smooth.best_rate,
            "identical-src/ref skip tie-break: both filters should produce \
             the same rate (skip_true_cost)",
        );
        assert_eq!(eight.picked_interp_filter, InterpFilter::EightTap);
        assert_eq!(smooth.picked_interp_filter, InterpFilter::EightTapSmooth);
    }

    /// F.5/G2 — invoking `evaluate_candidate_for_filter` with each
    /// filter in the search set produces DIFFERENT predictions (and
    /// hence different distortion / rate / RD) on non-trivial content
    /// with a sub-pel MV, and the returned `picked_interp_filter` tags
    /// match the requested filter. This is the "wiring is exercised"
    /// oracle — proves the SMOOTH kernel is actually reachable from
    /// the picker, the result struct carries the chosen filter, and
    /// the per-context switchable-filter cost is applied to all
    /// filters (G2 replaces the old flat constant).
    #[test]
    fn f5_per_filter_evaluation_differs_and_tags_picked_filter() {
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 211);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        // Source: shifted variant of ref so a non-trivial NEARESTMV
        // reconstruction can probe the kernels' differing taps.
        let mut src_data = ref_data.clone();
        let bw = 16;
        let bh = 16;
        let src_col = 20;
        let src_row = 20;
        // Quarter-pel right shift (col=2 in Q3 = 0.25 px). Sub-pel
        // phase = (2 & 7) * 2 = 4 → both filters use kernel row 4,
        // where EightTap and EightTapSmooth coefficients differ
        // markedly (Lagrangian preserves edges, SMOOTH attenuates).
        for i in 0..bh {
            for j in 0..bw {
                src_data[(src_row + i) * pw + (src_col + j)] =
                    ref_data[(src_row + i) * pw + (src_col + j + 1)];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        let harness = TestHarness::new(pw, 80);
        let mvc = MvCandidates {
            nearest: Mv { row: 0, col: 2 },
            near: Mv { row: 0, col: 2 },
        };
        let params = make_params(
            &harness,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (src_col, src_row),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            80,
        );

        let lambda =
            compute_rd_mult(80, frame_type_for_rd(synthetic_inter_frame_kind()));
        let eight = evaluate_candidate_for_filter(
            &params,
            InterMode::NearestMv,
            mvc.nearest,
            mvc.nearest,
            lambda,
            InterpFilter::EightTap,
        );
        let smooth = evaluate_candidate_for_filter(
            &params,
            InterMode::NearestMv,
            mvc.nearest,
            mvc.nearest,
            lambda,
            InterpFilter::EightTapSmooth,
        );

        // The two filters produce different predictions; at least one
        // RD-axis must differ. The exact direction is content-dependent
        // (Lagrangian vs low-pass on a per-pixel basis) so we don't
        // assert which wins, only that they're distinguishable — the
        // F.5 wiring is exercised.
        assert!(
            eight.best_distortion != smooth.best_distortion
                || eight.best_rd_cost != smooth.best_rd_cost,
            "EightTap and SMOOTH evaluations must differ on non-trivial \
             sub-pel content (dist {} vs {}, rd {} vs {})",
            eight.best_distortion,
            smooth.best_distortion,
            eight.best_rd_cost,
            smooth.best_rd_cost,
        );
        // G2 — SMOOTH pays `sic[ctx][SMOOTH]`; EightTap pays
        // `sic[ctx][EightTap]`. The total rate sum on each side is
        // `mode_rate + mv_rate + ref_frame_rate + coef_rate +
        // filter_rate`; mode / mv / ref-frame rates are identical
        // (same `(mode, mv)` across both filter evaluations), so the
        // residual difference collapses to
        // `|filter_cost_delta| - coef_rate_slack`.
        let ctx = 3usize; // no-neighbour sentinel used by make_params
        let fc = FrameContext::new_default();
        let sic = switchable_interp_costs(&fc);
        let filter_cost_delta = (sic[ctx][InterpFilter::EightTapSmooth as usize]
            - sic[ctx][InterpFilter::EightTap as usize])
            .unsigned_abs() as u64;
        let coef_rate_delta = smooth.coef_rate.abs_diff(eight.coef_rate);
        let observed_rate_delta =
            u64::from(smooth.best_rate.abs_diff(eight.best_rate));
        let expected_floor =
            filter_cost_delta.saturating_sub(coef_rate_delta);
        assert!(
            observed_rate_delta >= expected_floor,
            "G2 switchable-rate floor: |smooth.rate {} - eight.rate {}| = {} \
             must be >= filter_cost_delta {} - coef_rate_slack {} = {}",
            smooth.best_rate,
            eight.best_rate,
            observed_rate_delta,
            filter_cost_delta,
            coef_rate_delta,
            expected_floor,
        );
        // Picker-side filter is correctly tagged on each result.
        assert_eq!(eight.picked_interp_filter, InterpFilter::EightTap);
        assert_eq!(smooth.picked_interp_filter, InterpFilter::EightTapSmooth);
    }

    /// Argmin: NEAR wins when its MV points at the correct shift but
    /// NEAREST and ZERO point elsewhere. Exercises `pick_inter_mode_luma`
    /// line 429 True path.
    #[test]
    fn pick_inter_near_mv_wins_argmin_when_near_points_at_shift() {
        let pw = 64;
        let ph = 64;
        let ref_data = make_plane(pw, ph, 41);
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };

        // Build a source = ref shifted 3 px right at (20, 20).
        let mut src_data = ref_data.clone();
        let bw = 16;
        let bh = 16;
        let src_col = 20;
        let src_row = 20;
        let shift_col: usize = 3;
        for i in 0..bh {
            for j in 0..bw {
                src_data[(src_row + i) * pw + (src_col + j)] =
                    ref_data[(src_row + i) * pw + (src_col + j + shift_col)];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        // NEAREST = wrong (zero), NEAR = the correct shift (3 full
        // pels right = col 24 in Q3 = 1/8 pel). NEAR's prediction
        // matches the source → distortion ≈ 0 → wins argmin without
        // paying NEWMV's MV-bit cost.
        let mvc = MvCandidates {
            nearest: Mv::ZERO,
            near: Mv { row: 0, col: 24 },
        };
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
        let res = pick_inter_mode_luma(&params);
        // The winner must be NEAR (or NEW that happens to match —
        // but NEW pays MV bits and NEAR doesn't, so NEAR strictly
        // wins). The picker's argmin chain has NEAR's branch take
        // the True path because near.rd_cost < zero.rd_cost.
        assert!(
            matches!(res.best_mode, InterMode::NearMv | InterMode::NewMv),
            "expected NEAR or NEW winner on shifted-NEAR oracle, got {:?}",
            res.best_mode,
        );
        // NEAR's distortion must be very small (near-zero).
        let all = pick_inter_mode_luma_all_costs(&params);
        let near = &all[2];
        assert!(
            near.best_distortion < 1000,
            "NEAR with matching MV should have small distortion: {}",
            near.best_distortion,
        );
    }

    // ====================================================================
    // pick_inter_mode (dual-ref) tests
    // ====================================================================

    /// Build a [`PickInterModeParams`] with sensible defaults from a
    /// test harness. Both refs point at the same plane to keep
    /// distortion controllable.
    #[allow(clippy::too_many_arguments)] // test-only helper
    fn make_dual_ref_params<'a>(
        harness: &'a TestHarness,
        intra_y_costs: &'a [i32; 10],
        block_size: BlockSize,
        src_plane: &'a PlaneRef<'a>,
        ref_plane: &'a PlaneRef<'a>,
        src_offset: (usize, usize),
        last_mvc: &'a MvCandidates,
        golden_mvc: &'a MvCandidates,
        mv_limits: MvLimits,
        q_index: u8,
    ) -> PickInterModeParams<'a> {
        PickInterModeParams {
            block_size,
            src_plane,
            src_offset,
            last_ref_plane: ref_plane,
            last_mv_candidates: last_mvc,
            golden_ref_plane: ref_plane,
            golden_mv_candidates: golden_mvc,
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
            ref_frame_cost_last: 0,
            ref_frame_cost_golden: 0,
            last_nearest_in_bounds: true,
            last_near_in_bounds: true,
            golden_nearest_in_bounds: true,
            golden_near_in_bounds: true,
            intra_above_pixels: None,
            intra_left_pixels: None,
            intra_top_left_pixel: 128,
            intra_y_mode_costs: intra_y_costs,
            ref_frame_cost_intra: 0,
            switchable_interp_costs: &harness.sic,
            switchable_interp_ctx: 3,
            src_u: &[],
            src_v: &[],
            src_u_stride: 0,
            src_v_stride: 0,
            last_ref_u: &[],
            last_ref_v: &[],
            golden_ref_u: &[],
            golden_ref_v: &[],
            uv_stride: 0,
            uv_dc_quant: 0,
            uv_ac_quant: 0,
            encode_breakout: 0,
            dequant_dc: 0,
            dequant_ac: 0,
        }
    }

    /// `pick_inter_mode` with OOB flags: when `last_nearest_in_bounds`
    /// is false the picker must clamp the MV to ZERO before evaluating
    /// (`Mv::ZERO` placement at line 706). With identical src/ref the
    /// best mode is still zero-MV; we assert the result is finite and
    /// the chosen ref-frame is LAST (default selection on identical
    /// content).
    #[test]
    fn pick_inter_mode_oob_clamps_nearest_to_zero() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 33);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        // size_group for Block16x16 — pick any (we just need a 10-array slice).
        let intra_y_costs: [i32; 10] = intra_costs[0];
        // Both refs have non-zero NEAREST that the OOB flag forces
        // back to zero.
        let last_mvc = MvCandidates {
            nearest: Mv { row: 100, col: 100 }, // out of bounds
            near: Mv::ZERO,
        };
        let golden_mvc = MvCandidates::default();
        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &last_mvc,
            &golden_mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        params.last_nearest_in_bounds = false; // force the clamp
        let res = pick_inter_mode(&params);
        assert!(
            res.best_rd_cost > 0 && res.best_rd_cost < i64::MAX,
            "RD cost must be finite positive: {}",
            res.best_rd_cost,
        );
        // The OOB veto on NEAREST means NEAREST is skipped entirely
        // for LAST. ZERO LAST is the first candidate, so it gets
        // evaluated. Identical src/ref → ZERO wins.
        assert!(
            matches!(res.best_mode, InterMode::ZeroMv | InterMode::NearMv | InterMode::NewMv),
            "expected ZERO/NEAR/NEW with OOB-vetoed NEAREST, got {:?}",
            res.best_mode,
        );
    }

    /// `pick_inter_mode`: when `intra_above_pixels` is `Some` and a
    /// noisy block at high q_index makes inter expensive, the intra
    /// fallback gets evaluated. We assert that the picker either
    /// produces an intra winner or maintains a finite inter winner —
    /// the wiring is exercised either way and the path on line 982
    /// (intra threshold check) fires when above_pixels is provided.
    #[test]
    fn pick_inter_mode_intra_fallback_path_runs_when_above_left_present() {
        let pw = 64;
        let ph = 64;
        let src_data = make_plane(pw, ph, 91);
        let ref_data = make_plane(pw, ph, 17); // very different from src
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        let intra_y_costs: [i32; 10] = intra_costs[0];
        let mvc = MvCandidates::default();

        // Provide neighbour edges so the intra fallback evaluates.
        let above = [128u8; 64];
        let left = [128u8; 64];

        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (16, 16),
            &mvc,
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        params.intra_above_pixels = Some(&above);
        params.intra_left_pixels = Some(&left);

        let res = pick_inter_mode(&params);
        // The intra-fallback wiring executes when above/left are
        // present. The result must remain finite and one of the legal
        // ref-frame outcomes.
        assert!(res.best_rd_cost > 0 && res.best_rd_cost < i64::MAX);
        assert!(
            matches!(
                res.picked_ref_frame,
                RefFrame::LastFrame | RefFrame::GoldenFrame | RefFrame::IntraFrame
            ),
            "ref frame must be one of LAST/GOLDEN/INTRA, got {:?}",
            res.picked_ref_frame,
        );
    }

    /// `pick_inter_mode` with `block_size = Block64x64`: the intra
    /// fallback short-circuits (returns `None` at line 1040). Even
    /// with neighbour edges provided, the picked ref-frame must be
    /// inter (not intra), and the result remains finite.
    #[test]
    fn pick_inter_mode_intra_fallback_skipped_for_64x64() {
        let pw = 128;
        let ph = 128;
        let data = make_plane(pw, ph, 17);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 80);
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        let intra_y_costs: [i32; 10] = intra_costs[3]; // size group for 64x64
        let mvc = MvCandidates::default();
        let above = [128u8; 64];
        let left = [128u8; 64];

        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block64x64,
            &plane,
            &plane,
            (32, 32),
            &mvc,
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            80,
        );
        params.intra_above_pixels = Some(&above);
        params.intra_left_pixels = Some(&left);

        let res = pick_inter_mode(&params);
        // Block64x64 → intra fallback returns None → picker keeps the
        // inter winner. ref_frame must be LAST or GOLDEN, never
        // IntraFrame.
        assert!(
            matches!(
                res.picked_ref_frame,
                RefFrame::LastFrame | RefFrame::GoldenFrame
            ),
            "Block64x64 intra fallback is skipped → ref must be inter, got {:?}",
            res.picked_ref_frame,
        );
    }

    /// `pick_inter_mode` with all OOB flags false: every NEAREST/NEAR
    /// MV gets vetoed except ZERO, which is always reachable. The
    /// result must still be finite — exercising the OOB continue
    /// paths at lines 765-768.
    #[test]
    fn pick_inter_mode_all_oob_vetoes_keep_zero_path() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 7);
        let plane = PlaneRef {
            data: &data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 50);
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        let intra_y_costs: [i32; 10] = intra_costs[0];
        let mvc = MvCandidates {
            nearest: Mv { row: 50, col: 50 },
            near: Mv { row: -50, col: -50 },
        };
        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        params.last_nearest_in_bounds = false;
        params.last_near_in_bounds = false;
        params.golden_nearest_in_bounds = false;
        params.golden_near_in_bounds = false;

        let res = pick_inter_mode(&params);
        assert!(res.best_rd_cost > 0 && res.best_rd_cost < i64::MAX);
        // Identical src/ref → ZERO MV wins.
        assert_eq!(res.best_mv, Mv::ZERO);
    }

    /// `pick_inter_mode` with encode_breakout set but a shifted
    /// reference: the breakout's `var_y/sse_y` thresholds aren't met
    /// on the first ZERO candidate, so the function loops through
    /// the full RT_INTER_MODES. Exercises the encode_breakout
    /// gate-true-but-returns-false path (lines 925-934).
    #[test]
    fn pick_inter_mode_encode_breakout_does_not_fire_on_textured_block() {
        let pw = 64;
        let ph = 64;
        let src_data = make_plane(pw, ph, 11);
        let mut ref_data = src_data.clone();
        // Shift by 4 pixels to make ZERO prediction noticeably wrong.
        for r in 0..ph {
            for c in 4..pw {
                ref_data[r * pw + c] = src_data[r * pw + c - 4];
            }
        }
        let src_plane = PlaneRef {
            data: &src_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let ref_plane = PlaneRef {
            data: &ref_data,
            stride: pw,
            width: pw,
            height: ph,
        };
        let harness = TestHarness::new(pw, 30);
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        let intra_y_costs: [i32; 10] = intra_costs[0];
        let mvc = MvCandidates::default();
        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (16, 16),
            &mvc,
            &mvc,
            MvLimits::new(-8, 8, -8, 8),
            30,
        );
        // Breakout enabled but the shifted-ref content won't trip it.
        params.encode_breakout = 20;
        params.dequant_dc = 96;
        params.dequant_ac = 80;

        let res = pick_inter_mode(&params);
        assert!(res.best_rd_cost > 0 && res.best_rd_cost < i64::MAX);
    }

    /// `pick_inter_mode` with encode_breakout enabled and identical
    /// src/ref: the breakout fires after the first ZERO candidate
    /// (perfect prediction) and the mode loop exits early. We can't
    /// assert the early-exit count directly but the result must still
    /// be a zero-MV winner with zero distortion.
    #[test]
    fn pick_inter_mode_encode_breakout_fires_on_perfect_prediction() {
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
        let fc = FrameContext::new_default();
        let intra_costs = crate::cost::symbol_costs::intra_y_mode_costs_inter(&fc);
        let intra_y_costs: [i32; 10] = intra_costs[0];
        let mvc = MvCandidates::default();
        let mut params = make_dual_ref_params(
            &harness,
            &intra_y_costs,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        // Enable the breakout test with the typical realtime value.
        params.encode_breakout = 20;
        params.dequant_dc = 96;
        params.dequant_ac = 80;

        let res = pick_inter_mode(&params);
        assert_eq!(
            res.best_distortion, 0,
            "identical src/ref → zero distortion winner",
        );
        assert_eq!(res.best_mv, Mv::ZERO);
    }

    // ====================================================================
    // UV-RD path (G1) tests
    // ====================================================================

    /// `evaluate_candidate_for_filter` with non-empty UV planes: the
    /// chroma RD model runs and contributes to the rate sum. We
    /// compare against the same call with empty UV planes — the
    /// post-tie-break RD must differ (UV adds dist via `uv_dist`,
    /// which feeds into the skip tie-break comparison).
    ///
    /// To keep the tie-break from collapsing both runs to the skip
    /// path, we use a low q_index (residuals survive) and a sharply
    /// different source vs reference so the encode path beats skip.
    #[test]
    fn evaluate_candidate_uv_path_adds_rate() {
        let pw = 64;
        let ph = 64;
        // Source = noise pattern; Reference = shifted variant so the
        // residual is non-trivial and survives low-q quantization.
        let data_src = make_plane(pw, ph, 23);
        let mut data_ref = data_src.clone();
        // Shift by 1 pixel everywhere to create a stable residual.
        for r in 0..ph {
            for c in 1..pw {
                data_ref[r * pw + c] = data_src[r * pw + c - 1];
            }
        }
        let src_plane = PlaneRef {
            data: &data_src,
            stride: pw,
            width: pw,
            height: ph,
        };
        let ref_plane = PlaneRef {
            data: &data_ref,
            stride: pw,
            width: pw,
            height: ph,
        };
        // Low q so coefficients survive quantization.
        let harness = TestHarness::new(pw, 10);
        let mvc = MvCandidates::default();

        // UV planes: source vs ref with non-trivial residual.
        let data_u_src = make_plane(pw / 2, ph / 2, 41);
        let mut data_u_ref = data_u_src.clone();
        for r in 0..ph / 2 {
            for c in 1..pw / 2 {
                data_u_ref[r * (pw / 2) + c] = data_u_src[r * (pw / 2) + c - 1];
            }
        }
        let data_v_src = data_u_src.clone();
        let data_v_ref = data_u_ref.clone();

        let mut params_with_uv = make_params(
            &harness,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            10,
        );
        params_with_uv.src_u = &data_u_src;
        params_with_uv.src_v = &data_v_src;
        params_with_uv.ref_u = &data_u_ref;
        params_with_uv.ref_v = &data_v_ref;
        params_with_uv.uv_stride = pw / 2;
        params_with_uv.uv_dc_quant = 96;
        params_with_uv.uv_ac_quant = 80;

        let lambda =
            compute_rd_mult(10, frame_type_for_rd(synthetic_inter_frame_kind()));
        let with_uv = evaluate_candidate_for_filter(
            &params_with_uv,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );

        // Same call but with empty UV — must hit the False arm of the
        // UV-path gate (line 1727).
        let params_empty = make_params(
            &harness,
            BlockSize::Block16x16,
            &src_plane,
            &ref_plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            10,
        );
        let without_uv = evaluate_candidate_for_filter(
            &params_empty,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );

        // With UV residual present, at least one RD axis must differ.
        // (UV adds rate via uv_rate and dist via uv_dist.)
        assert!(
            with_uv.best_rate != without_uv.best_rate
                || with_uv.best_distortion != without_uv.best_distortion
                || with_uv.best_rd_cost != without_uv.best_rd_cost,
            "UV-enabled evaluation must differ from empty-UV: \
             rate {} vs {}, dist {} vs {}, rd {} vs {}",
            with_uv.best_rate, without_uv.best_rate,
            with_uv.best_distortion, without_uv.best_distortion,
            with_uv.best_rd_cost, without_uv.best_rd_cost,
        );
    }

    /// UV gate: src_u non-empty but ref_u empty → second AND fails
    /// (line 1728:12 F branch). The function must skip the UV path
    /// and return rate/dist matching the no-UV evaluation.
    #[test]
    fn evaluate_candidate_uv_skipped_when_ref_u_empty() {
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
        let data_u = make_plane(pw / 2, ph / 2, 41);

        let mut params = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        // src_u present but ref_u empty → UV path skipped.
        params.src_u = &data_u;
        params.src_v = &data_u;
        params.ref_u = &[];
        params.ref_v = &[];
        params.uv_stride = pw / 2;

        let lambda =
            compute_rd_mult(50, frame_type_for_rd(synthetic_inter_frame_kind()));
        let with_partial_uv = evaluate_candidate_for_filter(
            &params,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );
        // Compare to empty-UV: result must match (UV path took the
        // False branch).
        let params_empty = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        let no_uv = evaluate_candidate_for_filter(
            &params_empty,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );
        assert_eq!(with_partial_uv.best_rd_cost, no_uv.best_rd_cost);
        assert_eq!(with_partial_uv.best_rate, no_uv.best_rate);
    }

    /// UV gate: src_u and ref_u non-empty but uv_stride = 0 → third
    /// AND fails (line 1729:12 F branch).
    #[test]
    fn evaluate_candidate_uv_skipped_when_stride_zero() {
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
        let data_u = make_plane(pw / 2, ph / 2, 41);

        let mut params = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        params.src_u = &data_u;
        params.src_v = &data_u;
        params.ref_u = &data_u;
        params.ref_v = &data_u;
        params.uv_stride = 0; // ← false

        let lambda =
            compute_rd_mult(50, frame_type_for_rd(synthetic_inter_frame_kind()));
        let with_zero_stride = evaluate_candidate_for_filter(
            &params,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );
        let params_empty = make_params(
            &harness,
            BlockSize::Block16x16,
            &plane,
            &plane,
            (16, 16),
            &mvc,
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        let no_uv = evaluate_candidate_for_filter(
            &params_empty,
            InterMode::ZeroMv,
            Mv::ZERO,
            Mv::ZERO,
            lambda,
            InterpFilter::EightTap,
        );
        assert_eq!(with_zero_stride.best_rd_cost, no_uv.best_rd_cost);
    }

    /// F.5/G2 — identical source and reference: both filters produce
    /// zero distortion. The per-context switchable-filter cost
    /// (`switchable_interp_costs[ctx][SMOOTH] > [ctx][EightTap]`) makes
    /// `EightTap` win the RD comparison. This validates the "no
    /// incentive to switch" invariant: when prediction quality is
    /// identical, the cheaper default filter must be selected.
    #[test]
    fn f5_identical_src_ref_yields_eighttap_picked_filter() {
        let pw = 64;
        let ph = 64;
        let data = make_plane(pw, ph, 99);
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
            MvLimits::new(-4, 4, -4, 4),
            50,
        );
        let res = pick_inter_mode_luma(&params);
        assert_eq!(
            res.picked_interp_filter,
            InterpFilter::EightTap,
            "identical src/ref must select EightTap (Smooth penalized by switchable rate)",
        );
    }

}
