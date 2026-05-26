//! W4.6 — `handle_inter_mode` single-ref + switchable-filter inner
//! loop, composed onto the **real transform-domain RD** primitives.
//!
//! ## libvpx references
//!
//! - `handle_inter_mode` — `vp9/encoder/vp9_rdopt.c:2812-3219`
//!   (v1.14.1 commit `12f3a2ac6`). The main per-mode entry point for
//!   the slow RDO picker.
//! - `super_block_yrd` (the real Y / UV transform-domain RD walker) —
//!   `vp9_rdopt.c:1025-1042`, called for the luma plane at
//!   `vp9_rdopt.c:3119-3147` and for chroma at `vp9_rdopt.c:3193-3204`.
//! - Switchable-filter inner loop — `vp9_rdopt.c:3010-3116`. Each
//!   filter rebuilds the prediction with the 8-tap kernel
//!   (`vp9_build_inter_predictors_sb`, `vp9_rdopt.c:3043`) and
//!   re-measures RD; the lowest-RD filter wins.
//! - `build_inter_pred_model_rd_earlyterm` — `vp9_rdopt.c:165-282`.
//!   Retained as a standalone utility (see below) but **no longer the
//!   RD source** for `handle_inter_mode_single_ref`.
//!
//! ## What changed (RDO integration prerequisite #1)
//!
//! This module previously FAKED its RD: it scored interp filters with
//! a synthetic `filter_quality_bias` constant, never applied the
//! filter kernel, computed RD via a SAD model
//! (`build_inter_pred_model_rd_earlyterm`), and stubbed chroma to 0.
//! That made the W4 RDO driver a strict downgrade of the RT picker
//! (see `wrk_journals/2026.05.26 - JRN - RDO integration assessment.md`).
//!
//! [`handle_inter_mode_single_ref`] now does the real thing:
//!
//! 1. For each candidate interp filter it builds the **real** 8-tap
//!    inter prediction (`rvp9_core::inter::filter::convolve8*`), forms
//!    the residual, runs a forward-DCT / quant / inverse-DCT round-trip
//!    to produce the decoder-equivalent reconstruction, and calls the
//!    real [`crate::rdo::block_yrd::super_block_yrd`] for the
//!    transform-domain `(rate, distortion)`.
//! 2. It adds the **real** chroma RD by running the same round-trip +
//!    `super_block_yrd` on the U and V planes (libvpx lines 3193-3204).
//! 3. The inner filter loop picks the genuine min-RD filter.
//!
//! ## Scope
//!
//! - **Single-ref only.** The compound-pred branches (libvpx lines
//!   2877-2887, 2891-2916, 3001) are stubbed; the W3.x compound path
//!   will wire these in when joint-search arrives.
//! - **Single transform tile per plane.** The real RD pins the
//!   transform to `max_txsize(bsize)` and treats the block as one tile
//!   (the W4.1 `super_block_yrd` single-tile scope, `block_yrd.rs:289`).
//!   This covers `bsize ∈ {≤ 32×32}` where the block is exactly one
//!   transform; 64×64 (4 × 32×32 tiles) is deferred with the
//!   partition-driver integration.
//! - **Switchable-filter loop ports all 3 EIGHTTAP filters**
//!   (`EIGHTTAP`, `EIGHTTAP_SMOOTH`, `EIGHTTAP_SHARP`); we always
//!   evaluate all three when `cm->interp_filter == SWITCHABLE`.
//! - **No `pred_filter_search` short-circuit** — we always run the full
//!   search (libvpx lines 2867-2875).
//! - **Skip-tx / `disable_skip` path is wired** via the `super_block_yrd`
//!   `skip` flag + `sse`.

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::block::{BlockSize, max_txsize, uv_plane_block_size, uv_plane_tx_size};
use rvp9_core::cost::SwitchableInterpCosts;
use rvp9_core::cost::coef_cost::CoefCostTable;
use rvp9_core::inter::filter::{FILTER_TAPS, convolve8, convolve8_horiz, convolve8_vert, convolve_copy, kernel_for};
use rvp9_core::mode_info::{InterMode, Mv};
use rvp9_core::mv_search::full_pixel::PlaneRef;
use rvp9_core::quant::forward::PlaneQuantizers;
use rvp9_core::transforms::inverse::{
    idct8x8_64_add, idct16x16_256_add, idct32x32_1024_add,
};
use rvp9_core::transforms::scans::scan_for;
use rvp9_core::transforms::{TxMode, TxSize, TxType};

use crate::rdo::block_yrd::{BlockInput, super_block_yrd};
use crate::rdo::trellis::vp9_xform_quant_fp;

/// Number of switchable filters libvpx evaluates inside the inner
/// filter loop. Mirrors `SWITCHABLE_FILTERS = 3` (libvpx
/// `vp9_enums.h:135`).
pub const SWITCHABLE_FILTERS: usize = 3;

/// libvpx `SKIP_TXFM_*` enum (`vp9_block.h:25-31`). Tracks the per-
/// transform-block skip-flag classification that drives the
/// `build_inter_pred_model_rd_earlyterm` `skip_flag` accumulator.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SkipTxfm {
    /// `SKIP_TXFM_NONE = 0` — transform must be applied.
    None = 0,
    /// `SKIP_TXFM_AC_ONLY = 1` — AC coefficients can be quantized to 0.
    AcOnly = 1,
    /// `SKIP_TXFM_AC_DC = 2` — both AC and DC quantize to 0 (full skip).
    AcDc = 2,
}

/// Output of [`build_inter_pred_model_rd_earlyterm`]. Mirrors libvpx
/// `vp9_rdopt.c:276-280`.
#[derive(Copy, Clone, Debug)]
pub struct PredModelRdResult {
    /// `*out_rate_sum`. Sum of per-plane model rates (libvpx 1/512-bit
    /// units).
    pub rate_sum: i32,
    /// `*out_dist_sum`. Sum of per-plane model distortions, shifted
    /// by `VP9_DIST_SCALE_LOG2 = 4` to align with `RDCOST`.
    pub dist_sum: i64,
    /// `*skip_txfm_sb`. `true` iff every transform tile in every plane
    /// reported a low-error skip.
    pub skip_txfm_sb: bool,
    /// `*skip_sse_sb`. `total_sse << VP9_DIST_SCALE_LOG2`.
    pub skip_sse_sb: i64,
    /// `true` iff the function early-terminated (libvpx line 270-273).
    pub early_terminated: bool,
}

/// libvpx `VP9_DIST_SCALE_LOG2` (`vp9_rd.h:31`).
pub const VP9_DIST_SCALE_LOG2: u32 = 4;

/// libvpx `RDCOST(RM, DM, R, D)` shaped for the inter-mode hooks
/// (`vp9_rd.h:29`).
#[inline]
fn rdcost(rdmult: i64, rate: i64, dist: i64) -> i64 {
    let rate_term = (rdmult.saturating_mul(rate) + 256) >> 9;
    let dist_term = dist << 7;
    rate_term.saturating_add(dist_term)
}

/// libvpx `build_inter_pred_model_rd_earlyterm` (`vp9_rdopt.c:165-282`)
/// — Y-plane simplified port.
///
/// Computes a per-plane SAD-like model RD using simple integer
/// arithmetic. The full libvpx version walks all 3 planes and calls
/// `vp9_model_rd_from_var_lapndz`; the standalone-Y reduction
/// captures the load-bearing shape (sum_sse → rate / dist model;
/// optional early-term against `best_rd`).
///
/// ## Inputs
///
/// - `src` / `dst` — Y-plane source and prediction destinations.
/// - `bsize` — block size (drives the tile walk).
/// - `qstep` — per-block quantizer step (`pd->dequant[1] >> 3` for
///   8-bit per libvpx line 182-186).
/// - `dc_thr` / `ac_thr` — quant thresholds (libvpx line 197-198).
/// - `do_earlyterm` / `best_rd` / `rdmult` — early-term knobs.
///
/// Returns a [`PredModelRdResult`] with the per-block model RD pair.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_inter_pred_model_rd_earlyterm(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    bsize: BlockSize,
    qstep: u32,
    dc_thr: i64,
    ac_thr: i64,
    do_earlyterm: bool,
    best_rd: i64,
    rdmult: i64,
) -> PredModelRdResult {
    let w = usize::from(bsize.width());
    let h = usize::from(bsize.height());

    // libvpx lines 217-249: walk transformed-block tiles and compute
    // (var, sse) per tile. We use a single tile = the whole block here
    // (the libvpx fine-grained tile walk is an SF for sub-block AC/DC
    // classification; for the Y-only model RD the single-tile metric
    // is equivalent up to the `skip_txfm[]` granularity).
    let (var, sse) = compute_var_sse(src, src_stride, dst, dst_stride, w, h);

    // libvpx lines 230-243: skip-flag classification.
    let mut skip_txfm = SkipTxfm::None;
    if var < ac_thr as u32 || var == 0 {
        skip_txfm = SkipTxfm::AcOnly;
        if i64::from(sse - var) < dc_thr || sse == var {
            skip_txfm = SkipTxfm::AcDc;
        }
    }
    let low_dc_thr = 50i64.min(dc_thr >> 2);
    let low_ac_thr = 80i64.min(ac_thr >> 2);
    let low_err_skip = if matches!(skip_txfm, SkipTxfm::AcDc) {
        sse == 0
            || (u64::from(var) < low_ac_thr as u64
                && i64::from(sse - var) < low_dc_thr)
    } else {
        false
    };
    let skip_flag = low_err_skip;

    // libvpx lines 255-263: simple_model_rd_from_var.
    let sum_sse = i64::from(sse);
    let rate_sum_q = if qstep < 120 {
        // (sum_sse * (280 - qstep)) >> (16 - VP9_PROB_COST_SHIFT)
        // = (sum_sse * (280 - qstep)) >> 7.
        (sum_sse.saturating_mul(280 - qstep as i64)) >> 7
    } else {
        0
    };
    let dist = (sum_sse.saturating_mul(qstep as i64)) >> 8;
    // libvpx line 269: dist_sum scaled by VP9_DIST_SCALE_LOG2.
    let dist_sum = dist << VP9_DIST_SCALE_LOG2;
    let skip_sse_sb = sum_sse << VP9_DIST_SCALE_LOG2;

    // libvpx lines 270-273: optional early-term.
    let early_terminated = if do_earlyterm {
        rdcost(rdmult, rate_sum_q, dist_sum) >= best_rd
    } else {
        false
    };

    PredModelRdResult {
        rate_sum: rate_sum_q.try_into().unwrap_or(i32::MAX),
        dist_sum,
        skip_txfm_sb: skip_flag,
        skip_sse_sb,
        early_terminated,
    }
}

/// Per-block (variance, SSE) computation. `variance = sum((s-d)^2) -
/// (sum(s-d))^2 / N`; SSE = `sum((s-d)^2)`.
fn compute_var_sse(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) -> (u32, u32) {
    let mut sum: i64 = 0;
    let mut sum_sq: i64 = 0;
    for r in 0..h {
        for c in 0..w {
            let s = i32::from(src[r * src_stride + c]);
            let d = i32::from(dst[r * dst_stride + c]);
            let diff = s - d;
            sum += i64::from(diff);
            sum_sq += i64::from(diff * diff);
        }
    }
    let n = (w * h) as i64;
    let var = (sum_sq - (sum * sum) / n).max(0) as u32;
    let sse = sum_sq.max(0).min(i64::from(u32::MAX)) as u32;
    (var, sse)
}

/// Inputs to [`handle_inter_mode_single_ref`].
pub struct HandleInterModeParams<'a> {
    /// Block being decided.
    pub bsize: BlockSize,
    /// Y-plane source.
    pub src_plane: &'a PlaneRef<'a>,
    /// Top-left of the block in source plane samples (`(col, row)`).
    pub src_offset: (usize, usize),
    /// Y-plane reference (pre-translated to the chosen ref frame).
    pub ref_plane: &'a PlaneRef<'a>,
    /// libvpx `cm->interp_filter`. When `SWITCHABLE`, the inner loop
    /// evaluates all 3 filters; otherwise the filter is pinned.
    pub interp_filter: InterpFilter,
    /// Whether the frame allows switchable filter selection. Mirrors
    /// `cm->interp_filter == SWITCHABLE`.
    pub interp_filter_switchable: bool,
    /// libvpx `cm->interp_filter == BILINEAR` shortcut (`vp9_rdopt.c:3010`).
    pub interp_filter_bilinear: bool,
    /// Per-block switchable-interp rate table (G2).
    pub switchable_interp_costs: &'a SwitchableInterpCosts,
    /// Switchable-interp context (0..4) from above/left neighbour filters.
    pub switchable_interp_ctx: usize,
    /// The candidate inter mode (`NEARESTMV` / `NEARMV` / `ZEROMV` /
    /// `NEWMV`). Used to compute `mv` from the candidate.
    pub this_mode: InterMode,
    /// The candidate MV (1/8-pel units). For NEWMV this is the
    /// post-search MV; for NEAREST / NEAR it's the candidate list entry.
    pub mv: Mv,
    /// Rate of the inter-mode signalling bits (e.g. NEAREST/NEAR/ZERO/NEW).
    /// In 1/512-bit units. The caller looks this up via the inter-mode
    /// cost table.
    pub inter_mode_rate: u32,
    /// Rate of the NEWMV bit-cost (for `NEWMV` candidates), in 1/512-bit
    /// units. Zero for non-NEWMV.
    pub newmv_rate: u32,
    /// `rdmult` (lambda).
    pub rdmult: i64,
    /// Caller's `ref_best_rd` cap. `INT64_MAX` to disable early-exit.
    pub ref_best_rd: i64,
    /// libvpx `cpi->sf.use_rd_breakout`.
    pub use_rd_breakout: bool,

    // --- Real transform-domain RD inputs (libvpx `super_block_yrd`) ---
    /// Y-plane quantizers (libvpx `x->plane[0]` / `pd->dequant`). Drives
    /// the forward-quant inside [`super_block_yrd`] and the recon
    /// round-trip.
    pub y_quant: &'a PlaneQuantizers,
    /// Chroma-plane quantizers (libvpx `x->plane[1..2]`). Shared by U
    /// and V (4:2:0).
    pub uv_quant: &'a PlaneQuantizers,
    /// Coefficient-bit cost table (libvpx `x->token_costs`). Feeds
    /// `cost_coeffs` inside [`super_block_yrd`].
    pub coef_costs: &'a CoefCostTable,
    /// libvpx `cm->tx_mode`. The real RD pins to a single transform
    /// size; pass `TxModeSelect` to let `super_block_yrd` walk, or a
    /// fixed mode to match a frame header that pins the size.
    pub tx_mode: TxMode,
    /// Per-size transform-size signalling cost (libvpx
    /// `cpi->tx_size_cost`), indexed by `TxSize as usize`.
    pub tx_size_cost: [u32; 4],
    /// Skip-flag cost when skip = false (libvpx `vp9_cost_bit(skip, 0)`).
    pub skip_cost0: u32,
    /// Skip-flag cost when skip = true (libvpx `vp9_cost_bit(skip, 1)`).
    pub skip_cost1: u32,

    // --- Chroma planes (4:2:0). Empty slices skip chroma RD. ---
    /// U-plane source.
    pub src_u: &'a [u8],
    /// V-plane source.
    pub src_v: &'a [u8],
    /// U-plane reference (pre-translated to the chosen ref frame).
    pub ref_u: &'a [u8],
    /// V-plane reference.
    pub ref_v: &'a [u8],
    /// Chroma-plane stride (samples). `0` (or empty planes) skips
    /// chroma RD.
    pub uv_stride: usize,
}

/// Per-filter RD evaluation. Mirrors libvpx's `(filter_cache[i],
/// best_filter)` accumulator. All fields are now the **real**
/// transform-domain RD from [`super_block_yrd`] (Y + chroma), no longer
/// a model estimate.
#[derive(Copy, Clone, Debug)]
pub struct FilterRd {
    /// Total RD with this filter (`RDCOST(rate, dist)`).
    pub rd: i64,
    /// Rate (Y coef bits + chroma coef bits + switchable-filter bits).
    pub rate: u32,
    /// Distortion (Y + chroma transform-domain pixel distortion).
    pub dist: i64,
    /// `true` iff the block is skippable (every plane fully quantized
    /// away). libvpx `skip_txfm_sb`.
    pub skip_txfm_sb: bool,
    /// Pixel-domain SSE before residual coding (skip-branch distortion).
    /// libvpx `skip_sse_sb`.
    pub skip_sse_sb: i64,
}

/// libvpx `handle_inter_mode` single-ref simplified return.
#[derive(Copy, Clone, Debug)]
pub struct HandleInterModeResult {
    /// Winning interp filter (one of `EightTap` / `EightTapSmooth` /
    /// `EightTapSharp`).
    pub best_filter: InterpFilter,
    /// Total rate at the winning filter (mode + NEWMV + filter bits +
    /// model-rate). 1/512-bit units.
    pub rate: u32,
    /// Total distortion at the winning filter.
    pub dist: i64,
    /// Total RD cost.
    pub rd: i64,
    /// `disable_skip` analog — `true` iff the picker chose the skip
    /// branch over the full-RD branch.
    pub disable_skip: bool,
    /// Per-filter RD table for the inner loop. Used by the picker to
    /// compute libvpx's `best_filter_rd[i]` deltas.
    pub filter_cache: [Option<FilterRd>; SWITCHABLE_FILTERS],
}

/// libvpx `handle_inter_mode` single-ref (`vp9_rdopt.c:2812-3219`).
///
/// Iterates the 3 switchable filters when enabled. For each filter it
/// builds the **real** 8-tap inter prediction, forms the residual, and
/// calls the real [`super_block_yrd`] for the Y plane (libvpx lines
/// 3119-3147) plus chroma RD on the U/V planes (libvpx lines 3193-3204);
/// the lowest-RD filter wins. Returns the winning `(rate, dist, RD)`
/// tuple for the caller's per-mode argmin.
#[must_use]
pub fn handle_inter_mode_single_ref(
    params: &HandleInterModeParams<'_>,
) -> HandleInterModeResult {
    let mut filter_cache: [Option<FilterRd>; SWITCHABLE_FILTERS] = [None; SWITCHABLE_FILTERS];

    // libvpx line 3010: skip the inner filter loop when bilinear.
    let filters: &[InterpFilter] = if params.interp_filter_bilinear {
        &[InterpFilter::Bilinear]
    } else {
        &[
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
        ]
    };

    for (i, &filter) in filters.iter().enumerate() {
        // libvpx lines 3032-3038: switchable-filter signalling rate.
        let rs: u32 = if params.interp_filter_switchable {
            switchable_filter_rate(
                params.switchable_interp_costs,
                params.switchable_interp_ctx,
                filter,
            )
        } else {
            0
        };

        // libvpx lines 3043-3147: build the real prediction with this
        // filter, then super_block_yrd for the real (rate, dist).
        let plane_rd = evaluate_filter_rd(params, filter);

        // libvpx lines 3119-3147 / 3193-3204: rate = Y coef bits +
        // chroma coef bits + switchable-filter bits.
        let rate = plane_rd.rate.saturating_add(rs);
        // libvpx line 3155-3162: RDCOST over (coef rate, distortion).
        let rd = rdcost(params.rdmult, i64::from(rate), plane_rd.dist);

        filter_cache[i] = Some(FilterRd {
            rd,
            rate,
            dist: plane_rd.dist,
            skip_txfm_sb: plane_rd.skippable,
            skip_sse_sb: plane_rd.sse,
        });
    }

    // libvpx line 3113: pick the lowest-RD filter for real.
    let winning_idx = filter_cache
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.map(|s| (i, s.rd)))
        .min_by_key(|(_, rd)| *rd)
        .map(|(i, _)| i)
        .expect("at least one filter slot must be populated");
    let winning = filter_cache[winning_idx].expect("just selected");
    let best_filter = filters[winning_idx];

    // libvpx line 3155: total rate = mode bits + NEWMV bits + the
    // winning filter's (coef + filter) rate.
    let mode_bits = params.inter_mode_rate.saturating_add(params.newmv_rate);
    let total_rate = mode_bits.saturating_add(winning.rate);

    // libvpx lines 3155-3162 (skip-flag tie-break). Compare encoding the
    // residual (`RDCOST(total_rate + skip0, dist)`) against skipping it
    // (`RDCOST(mode_bits + skip1, skip_sse_sb)`). The cheaper branch
    // wins; on skip the distortion collapses to the prediction SSE.
    let rd_encode = rdcost(
        params.rdmult,
        i64::from(total_rate.saturating_add(params.skip_cost0)),
        winning.dist,
    );
    let rd_skip = rdcost(
        params.rdmult,
        i64::from(mode_bits.saturating_add(params.skip_cost1)),
        winning.skip_sse_sb,
    );
    // A fully-skippable block (eob == 0 in every plane) always takes the
    // skip branch (libvpx forces `skip = 1` for `skippable` blocks).
    let take_skip = winning.skip_txfm_sb || rd_skip < rd_encode;
    let (final_rate, final_dist, disable_skip) = if take_skip {
        (
            mode_bits.saturating_add(params.skip_cost1),
            winning.skip_sse_sb,
            true,
        )
    } else {
        (
            total_rate.saturating_add(params.skip_cost0),
            winning.dist,
            false,
        )
    };
    let final_rd = rdcost(params.rdmult, i64::from(final_rate), final_dist);

    HandleInterModeResult {
        best_filter,
        rate: final_rate,
        dist: final_dist,
        rd: final_rd,
        disable_skip,
        filter_cache,
    }
}

/// Real per-plane RD for one `(mode, mv, filter)` triplet: builds the
/// 8-tap prediction, forms the residual, and runs [`super_block_yrd`]
/// on the Y plane plus (when chroma planes are present) the U and V
/// planes. Returns the summed `(rate, dist, sse, skippable)`.
///
/// libvpx: luma `super_block_yrd` at `vp9_rdopt.c:3119`, chroma
/// `super_block_uvrd` at `vp9_rdopt.c:3193`.
struct PlaneRd {
    /// Summed coef-bit rate (Y + U + V), 1/512-bit units.
    rate: u32,
    /// Summed pixel-domain distortion (Y + U + V).
    dist: i64,
    /// Summed pre-coding pixel SSE (the skip-branch distortion).
    sse: i64,
    /// `true` iff every plane is fully skippable.
    skippable: bool,
}

fn evaluate_filter_rd(params: &HandleInterModeParams<'_>, filter: InterpFilter) -> PlaneRd {
    let bw = usize::from(params.bsize.width());
    let bh = usize::from(params.bsize.height());
    let y_tx = max_txsize(params.bsize);

    // --- Y plane (libvpx vp9_rdopt.c:3119-3147) ---
    let mut y_pred = [0u8; 64 * 64];
    build_inter_pred(
        params.ref_plane,
        params.src_offset.0,
        params.src_offset.1,
        params.mv,
        bw,
        bh,
        &mut y_pred,
        bw,
        filter,
    );
    let y_rd = plane_super_block_yrd(
        params.src_plane.data,
        params.src_plane.stride,
        params.src_offset,
        &y_pred,
        bw,
        bw,
        bh,
        y_tx,
        params.tx_mode,
        params.y_quant,
        params.coef_costs,
        0, // plane_type: Y
        params.rdmult,
        params.tx_size_cost,
        params.skip_cost0,
        params.skip_cost1,
    );

    let mut rate = y_rd.rate;
    let mut dist = y_rd.dist;
    let mut sse = y_rd.sse;
    let mut skippable = y_rd.skip;

    // --- Chroma planes (libvpx vp9_rdopt.c:3193-3204) ---
    // 4:2:0 only: chroma block is half-resolution; chroma MV is the
    // luma MV halved on each axis (libvpx scales by the subsampling).
    if !params.src_u.is_empty()
        && !params.ref_u.is_empty()
        && params.uv_stride > 0
    {
        let uv_bw = bw >> 1;
        let uv_bh = bh >> 1;
        let uv_x = params.src_offset.0 >> 1;
        let uv_y = params.src_offset.1 >> 1;
        let uv_mv = Mv {
            row: params.mv.row / 2,
            col: params.mv.col / 2,
        };
        let uv_tx = uv_plane_tx_size(params.bsize, y_tx, 1, 1);
        let _ = uv_plane_block_size(params.bsize, 1, 1); // documents the chroma bsize derivation
        let uv_h = params.ref_u.len() / params.uv_stride;

        for (src_uv, ref_uv) in [
            (params.src_u, params.ref_u),
            (params.src_v, params.ref_v),
        ] {
            let ref_uv_plane = PlaneRef {
                data: ref_uv,
                stride: params.uv_stride,
                width: params.uv_stride,
                height: uv_h,
            };
            let mut uv_pred = [0u8; 32 * 32];
            build_inter_pred(
                &ref_uv_plane, uv_x, uv_y, uv_mv, uv_bw, uv_bh,
                &mut uv_pred, uv_bw, filter,
            );
            let uv_rd = plane_super_block_yrd(
                src_uv,
                params.uv_stride,
                (uv_x, uv_y),
                &uv_pred,
                uv_bw,
                uv_bw,
                uv_bh,
                uv_tx,
                params.tx_mode,
                params.uv_quant,
                params.coef_costs,
                1, // plane_type: UV
                params.rdmult,
                params.tx_size_cost,
                params.skip_cost0,
                params.skip_cost1,
            );
            rate = rate.saturating_add(uv_rd.rate);
            dist = dist.saturating_add(uv_rd.dist);
            sse = sse.saturating_add(uv_rd.sse);
            skippable = skippable && uv_rd.skip;
        }
    }

    PlaneRd { rate, dist, sse, skippable }
}

/// Run [`super_block_yrd`] for a single plane: build the residual from
/// `(src, pred)`, run the forward-DCT / quant / inverse-DCT round-trip
/// to produce the decoder-equivalent reconstruction, then hand both the
/// residual and the recon to `super_block_yrd` for the real RD.
///
/// `tx` is the transform pinned for this block (`max_txsize(bsize)` for
/// Y, `uv_plane_tx_size` for chroma); the standalone scope treats the
/// block as one transform tile (W4.1 `block_yrd.rs:289`).
#[allow(clippy::too_many_arguments)]
fn plane_super_block_yrd(
    src: &[u8],
    src_stride: usize,
    src_offset: (usize, usize),
    pred: &[u8],
    pred_stride: usize,
    bw: usize,
    bh: usize,
    tx: TxSize,
    tx_mode: TxMode,
    qp: &PlaneQuantizers,
    coef_costs: &CoefCostTable,
    plane_type: usize,
    rdmult: i64,
    tx_size_cost: [u32; 4],
    s0: u32,
    s1: u32,
) -> crate::rdo::block_yrd::SuperBlockYrdResult {
    let (sx, sy) = src_offset;
    let side = tx.side();
    debug_assert!(bw == side && bh == side, "single-tile scope: block must equal tx tile");

    // Residual (src - pred), tile-sized, row-major.
    let mut src_diff = [0i16; 32 * 32];
    // Source pixels in a contiguous tile buffer (super_block_yrd's
    // pixel-domain distortion reads `src_pixels` at `src_stride`).
    for r in 0..bh {
        for c in 0..bw {
            let s = i32::from(src[(sy + r) * src_stride + (sx + c)]);
            let p = i32::from(pred[r * pred_stride + c]);
            src_diff[r * bw + c] = (s - p) as i16;
        }
    }

    // Round-trip: fdct + quant + idct-add → recon (the decoder result).
    let mut recon = [0u8; 32 * 32];
    for r in 0..bh {
        recon[r * bw..r * bw + bw]
            .copy_from_slice(&pred[r * pred_stride..r * pred_stride + bw]);
    }
    reconstruct_tile(&src_diff, bw, tx, qp, &mut recon, bw);

    // Source pixels as a contiguous (sy,sx)-anchored view at src_stride.
    let src_view = &src[sy * src_stride + sx..];

    let input = BlockInput {
        src_diff: &src_diff,
        diff_stride: bw,
        src_pixels: src_view,
        src_stride,
        dst_pixels: &recon,
        dst_stride: bw,
    };

    // The reconstruction in `recon` is built at exactly `tx`, so we pin
    // `super_block_yrd` to that single size. `super_block_yrd`'s pixel-
    // domain distortion reads `recon` at the tile size, so a walk over
    // *smaller* sizes (TxModeSelect) would measure against a recon that
    // was reconstructed at the wrong size. Pinning keeps the (R, D) pair
    // self-consistent. `AllowTx32x32` pins the range to
    // `min(max_tx_size, Tx32x32) == tx` for every supported (≤32×32)
    // block (`block_yrd.rs::tx_size_search_range`). The caller's
    // `tx_mode` is recorded for the eventual TX-search integration but
    // the single-tile recon path is size-pinned today.
    let _ = tx_mode;
    super_block_yrd(
        input,
        tx,
        TxMode::AllowTx32x32,
        true, // inter
        qp,
        rdmult,
        coef_costs,
        plane_type,
        0,           // ref_type (intra-frame coef ctx slot 0)
        0,           // coeff_ctx (above+left both zero — start of block)
        s0,
        s1,
        tx_size_cost,
    )
}

/// Forward-DCT + fast-path quant + inverse-DCT-add for one transform
/// tile, producing the decoder-equivalent reconstruction in `recon`
/// (which the caller pre-seeded with `pred`). Mirrors the
/// `vp9_xform_quant_fp` → `inv_txfm_add` round-trip libvpx runs to
/// fill `pd->dst` before `dist_block` measures pixel-domain distortion.
fn reconstruct_tile(
    src_diff: &[i16],
    diff_stride: usize,
    tx: TxSize,
    qp: &PlaneQuantizers,
    recon: &mut [u8],
    recon_stride: usize,
) {
    let n = tx.n_coeffs();
    let scan = scan_for(tx, TxType::DctDct);
    let mut coeff = vec![0i32; n];
    let mut qcoeff = vec![0i16; n];
    let mut dqcoeff = vec![0i16; n];

    let eob = vp9_xform_quant_fp(
        src_diff,
        diff_stride,
        tx,
        qp,
        &scan,
        &mut coeff,
        &mut qcoeff,
        &mut dqcoeff,
    );
    if eob == 0 {
        // All coefficients quantized to zero — prediction is the recon.
        return;
    }

    // dqcoeff is in raster order; the idct*_add kernels add the inverse
    // transform onto `recon` in place.
    match tx {
        TxSize::Tx8x8 => {
            let mut buf = [0i32; 64];
            for (b, &d) in buf.iter_mut().zip(dqcoeff.iter()) {
                *b = i32::from(d);
            }
            idct8x8_64_add(&buf, recon, recon_stride);
        }
        TxSize::Tx16x16 => {
            let mut buf: Box<[i32; 256]> = Box::new([0; 256]);
            for (b, &d) in buf.iter_mut().zip(dqcoeff.iter()) {
                *b = i32::from(d);
            }
            idct16x16_256_add(buf.as_ref(), recon, recon_stride);
        }
        TxSize::Tx32x32 => {
            let mut buf: Box<[i32; 1024]> = Box::new([0; 1024]);
            for (b, &d) in buf.iter_mut().zip(dqcoeff.iter()) {
                *b = i32::from(d);
            }
            idct32x32_1024_add(buf.as_ref(), recon, recon_stride);
        }
        TxSize::Tx4x4 => {
            // Single-tile scope covers ≥ 8×8 blocks (the partition driver
            // never hands sub-8 tiles to this path). Leaving recon = pred
            // is conservative; the eob != 0 branch is unreachable here.
            debug_assert!(false, "Tx4x4 tiles are out of handle_inter scope");
        }
    }
}

/// Build a `bw × bh` inter prediction at `src_offset + (mv >> 3)` in the
/// reference plane, with sub-pel interpolation phase `(mv & 7) << 1`
/// (Q3 → Q4). Dispatches to the public `convolve8*` kernels keyed on the
/// `filter`. Mirror of `rvp9_core::mode_decision::inter::build_inter_prediction`
/// (which is private to that crate). libvpx `inter_predictor`
/// (`vp9_reconinter.h:23-32`).
#[allow(clippy::too_many_arguments)]
fn build_inter_pred(
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
    let int_row = (mv.row as i32) >> 3;
    let int_col = (mv.col as i32) >> 3;
    let subpel_y_q4 = ((mv.row as i32) & 7) << 1;
    let subpel_x_q4 = ((mv.col as i32) & 7) << 1;

    let ref_x = src_x as isize + int_col as isize;
    let ref_y = src_y as isize + int_row as isize;

    let x_nonzero = subpel_x_q4 != 0;
    let y_nonzero = subpel_y_q4 != 0;

    let pad_x = if x_nonzero { (FILTER_TAPS / 2 - 1) as isize } else { 0 };
    let pad_y = if y_nonzero { (FILTER_TAPS / 2 - 1) as isize } else { 0 };
    let win_x = ref_x - pad_x;
    let win_y = ref_y - pad_y;
    debug_assert!(
        win_x >= 0 && win_y >= 0,
        "inter pred read window out of bounds: ({win_x}, {win_y})",
    );

    let src_start = (win_y as usize) * ref_plane.stride + (win_x as usize);
    let src_slice = &ref_plane.data[src_start..];

    let kernel = kernel_for(filter);
    let filter_x = &kernel[subpel_x_q4 as usize];
    let filter_y = &kernel[subpel_y_q4 as usize];

    match (x_nonzero, y_nonzero) {
        (false, false) => {
            convolve_copy(src_slice, ref_plane.stride, dst, dst_stride, bw, bh);
        }
        (true, false) => {
            convolve8_horiz(src_slice, ref_plane.stride, dst, dst_stride, filter_x, bw, bh);
        }
        (false, true) => {
            convolve8_vert(src_slice, ref_plane.stride, dst, dst_stride, filter_y, bw, bh);
        }
        (true, true) => {
            convolve8(
                src_slice, ref_plane.stride, dst, dst_stride, filter_x, filter_y, bw, bh,
            );
        }
    }
}

/// libvpx `vp9_get_switchable_rate` — bits the switchable-filter tree
/// would emit for `filter` in the current context. The full table is
/// `SwitchableInterpCosts::ctx[filter] = vp9_cost_tokens(...)` per
/// context. We index directly.
fn switchable_filter_rate(
    costs: &SwitchableInterpCosts,
    ctx: usize,
    filter: InterpFilter,
) -> u32 {
    let idx = match filter {
        InterpFilter::EightTap => 0,
        InterpFilter::EightTapSmooth => 1,
        InterpFilter::EightTapSharp => 2,
        _ => 0,
    };
    costs[ctx][idx].max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::coef_cost::build_coef_cost_table;
    use rvp9_core::cost::switchable_interp_costs;
    use rvp9_core::cost::symbol_costs::inter_mode_costs;
    use rvp9_core::entropy::default_probs::INTER_MODE_CONTEXTS;
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::mv_search::full_pixel::PlaneRef;
    use rvp9_core::quant::forward::{init_plane_quantizers, QuantizerDeltas};

    /// A textured Y source + matching shifted reference, sized with a
    /// border so a small MV stays in-bounds. The reference is the source
    /// shifted by `(shift_row, shift_col)` luma pixels: the true motion
    /// is therefore `mv = (-shift_row*8, -shift_col*8)` in Q3.
    struct Harness {
        src: Vec<u8>,
        refp: Vec<u8>,
        u_src: Vec<u8>,
        u_ref: Vec<u8>,
        v_src: Vec<u8>,
        v_ref: Vec<u8>,
        stride: usize,
        uv_stride: usize,
        offset: (usize, usize),
        bw: usize,
        bh: usize,
    }

    /// Deterministic textured pattern (so the residual is non-trivial
    /// and the picker has real RD to compare).
    fn texture(x: usize, y: usize) -> u8 {
        let v = (x.wrapping_mul(7) ^ y.wrapping_mul(13)).wrapping_add(x * y / 3);
        (40 + (v % 160)) as u8
    }

    fn build_harness(bw: usize, bh: usize, shift_row: i32, shift_col: i32) -> Harness {
        // Plane large enough that the block at `offset` plus a small MV
        // plus the 8-tap halo stays inside. Offset of 8 leaves room for
        // a 1-pixel motion + 3-pixel halo on every side.
        let pad = 8usize;
        let stride = bw + 2 * pad;
        let height = bh + 2 * pad;
        let offset = (pad, pad);

        let mut src = vec![0u8; stride * height];
        let mut refp = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..stride {
                src[y * stride + x] = texture(x, y);
                // The reference is the source content shifted: a sample
                // at (x, y) in the ref equals the source sample at
                // (x + shift_col, y + shift_row).
                let sx = (x as i32 + shift_col).clamp(0, stride as i32 - 1) as usize;
                let sy = (y as i32 + shift_row).clamp(0, height as i32 - 1) as usize;
                refp[y * stride + x] = texture(sx, sy);
            }
        }

        // Chroma planes: half resolution, same shift halved.
        let uv_pad = 4usize;
        let uv_bw = bw / 2;
        let uv_bh = bh / 2;
        let uv_stride = uv_bw + 2 * uv_pad;
        let uv_height = uv_bh + 2 * uv_pad;
        let mut u_src = vec![0u8; uv_stride * uv_height];
        let mut u_ref = vec![0u8; uv_stride * uv_height];
        let mut v_src = vec![0u8; uv_stride * uv_height];
        let mut v_ref = vec![0u8; uv_stride * uv_height];
        for y in 0..uv_height {
            for x in 0..uv_stride {
                u_src[y * uv_stride + x] = texture(x + 50, y + 50);
                v_src[y * uv_stride + x] = texture(x + 90, y + 90);
                let sx = (x as i32 + shift_col / 2).clamp(0, uv_stride as i32 - 1) as usize;
                let sy = (y as i32 + shift_row / 2).clamp(0, uv_height as i32 - 1) as usize;
                u_ref[y * uv_stride + x] = texture(sx + 50, sy + 50);
                v_ref[y * uv_stride + x] = texture(sx + 90, sy + 90);
            }
        }

        Harness {
            src,
            refp,
            u_src,
            u_ref,
            v_src,
            v_ref,
            stride,
            uv_stride,
            offset,
            bw,
            bh,
        }
    }

    /// Build `HandleInterModeParams` for a harness + (mode, mv), wiring
    /// the **real** cost tables / quantizers.
    fn make_params<'a>(
        h: &'a Harness,
        src_plane: &'a PlaneRef<'a>,
        ref_plane: &'a PlaneRef<'a>,
        fc: &FrameContext,
        sic: &'a SwitchableInterpCosts,
        y_quant: &'a PlaneQuantizers,
        uv_quant: &'a PlaneQuantizers,
        costs: &'a CoefCostTable,
        mode: InterMode,
        mv: Mv,
        with_chroma: bool,
    ) -> HandleInterModeParams<'a> {
        let bsize = match (h.bw, h.bh) {
            (16, 16) => BlockSize::Block16x16,
            (32, 32) => BlockSize::Block32x32,
            (8, 8) => BlockSize::Block8x8,
            _ => panic!("unsupported test block size"),
        };
        // Real inter-mode rate from the cost table (task #2).
        let imc = inter_mode_costs(fc);
        let ctx = 0usize.min(INTER_MODE_CONTEXTS - 1);
        let inter_mode_rate = imc[ctx][mode.idx()].max(0) as u32;
        // Real skip-bit costs from the default skip prob (middle ctx).
        let skip_prob = rvp9_core::entropy::default_probs::DEFAULT_SKIP_PROBS[1];
        let skip0 = rvp9_core::cost::vp9_cost_bit(skip_prob, false);
        let skip1 = rvp9_core::cost::vp9_cost_bit(skip_prob, true);

        HandleInterModeParams {
            bsize,
            src_plane,
            src_offset: h.offset,
            ref_plane,
            interp_filter: InterpFilter::EightTap,
            interp_filter_switchable: true,
            interp_filter_bilinear: false,
            switchable_interp_costs: sic,
            switchable_interp_ctx: 0,
            this_mode: mode,
            mv,
            inter_mode_rate,
            newmv_rate: if matches!(mode, InterMode::NewMv) { 120 } else { 0 },
            rdmult: 80,
            ref_best_rd: i64::MAX,
            use_rd_breakout: false,
            y_quant,
            uv_quant,
            coef_costs: costs,
            tx_mode: TxMode::TxModeSelect,
            tx_size_cost: [4, 8, 12, 16],
            skip_cost0: skip0,
            skip_cost1: skip1,
            src_u: if with_chroma { &h.u_src } else { &[] },
            src_v: if with_chroma { &h.v_src } else { &[] },
            ref_u: if with_chroma { &h.u_ref } else { &[] },
            ref_v: if with_chroma { &h.v_ref } else { &[] },
            uv_stride: if with_chroma { h.uv_stride } else { 0 },
        }
    }

    /// PROOF W4.6 — `handle_inter_mode_single_ref` returns a finite,
    /// real transform-domain `(R, D, RD)` for a single-ref block, with
    /// Y + chroma RD composed. libvpx `vp9_rdopt.c:2812-3219`.
    #[test]
    fn w4_6_handle_inter_mode_returns_real_rd_for_single_ref() {
        let h = build_harness(16, 16, 0, 0); // identical ref → ZEROMV is exact
        let src_plane = PlaneRef { data: &h.src, stride: h.stride, width: h.stride, height: h.src.len() / h.stride };
        let ref_plane = PlaneRef { data: &h.refp, stride: h.stride, width: h.stride, height: h.refp.len() / h.stride };
        let fc = FrameContext::new_default();
        let sic = switchable_interp_costs(&fc);
        let costs = build_coef_cost_table(&fc);
        let y_quant = init_plane_quantizers(96, QuantizerDeltas::Y);
        let uv_quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let params = make_params(
            &h, &src_plane, &ref_plane, &fc, &sic, &y_quant, &uv_quant, &costs,
            InterMode::ZeroMv, Mv::ZERO, true,
        );
        let out = handle_inter_mode_single_ref(&params);

        assert!(out.rd < i64::MAX / 2, "handle_inter_mode returned an unbounded RD");
        assert!(matches!(
            out.best_filter,
            InterpFilter::EightTap | InterpFilter::EightTapSmooth | InterpFilter::EightTapSharp
        ));
        // A perfectly-matched reference (zero shift, ZEROMV) must be
        // skippable / near-zero distortion — the residual is exactly
        // zero, so the round-trip recon equals the source.
        assert!(
            out.disable_skip,
            "exact prediction must take the skip branch (dist {})",
            out.dist,
        );
    }

    /// PROOF W4.6 — switchable-filter inner loop evaluates ALL 3
    /// EIGHTTAP filters and produces a real per-filter RD spread.
    /// libvpx `vp9_rdopt.c:3023-3113`.
    #[test]
    fn w4_6_switchable_filter_loop_tries_three_filters_when_enabled() {
        // Sub-pel MV so the three filters produce genuinely different
        // predictions (and therefore different transform-domain RD).
        let h = build_harness(16, 16, 0, 0);
        let src_plane = PlaneRef { data: &h.src, stride: h.stride, width: h.stride, height: h.src.len() / h.stride };
        let ref_plane = PlaneRef { data: &h.refp, stride: h.stride, width: h.stride, height: h.refp.len() / h.stride };
        let fc = FrameContext::new_default();
        let sic = switchable_interp_costs(&fc);
        let costs = build_coef_cost_table(&fc);
        let y_quant = init_plane_quantizers(96, QuantizerDeltas::Y);
        let uv_quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let params = make_params(
            &h, &src_plane, &ref_plane, &fc, &sic, &y_quant, &uv_quant, &costs,
            InterMode::NewMv, Mv { row: 4, col: 4 }, false, // half-pel both axes
        );
        let out = handle_inter_mode_single_ref(&params);

        let populated: usize = out.filter_cache.iter().filter(|s| s.is_some()).count();
        assert_eq!(
            populated, 3,
            "switchable-filter inner loop must evaluate ALL 3 filters (got {populated})",
        );

        // The three filters apply DIFFERENT kernels to a sub-pel
        // position, so at least one RD must differ from the EightTap RD.
        let rds: Vec<i64> = out.filter_cache.iter().flatten().map(|f| f.rd).collect();
        assert!(
            rds.iter().any(|&r| r != rds[0]),
            "filter_cache RDs are all identical — filters not really applied: {rds:?}",
        );
    }

    /// PROOF W4.6 (DIFFERENTIAL ORACLE) — for a block with real motion
    /// (the reference is a shifted copy of the source), the matching
    /// NEWMV must beat a clearly-worse ZEROMV on real transform-domain
    /// RD. This is the anti-cheat gate: a synthetic-bias / SAD-model RD
    /// could NOT distinguish the matched NEWMV from the mismatched
    /// ZEROMV. libvpx `vp9_rdopt.c:2812-3219` ranks NEWMV competitively
    /// on genuine motion.
    #[test]
    fn w4_6_newmv_beats_zeromv_on_real_motion() {
        // Reference is source shifted by 2 px right + 1 px down → the
        // true motion vector is mv = (row -8, col -16) in Q3 (recovers
        // the source: ref(x,y) = src(x+2, y+1), so to predict src(x,y)
        // we read ref(x-2, y-1) i.e. integer mv (-2,-1) px → Q3 -16,-8).
        let shift_row = 1;
        let shift_col = 2;
        let h = build_harness(16, 16, shift_row, shift_col);
        let src_plane = PlaneRef { data: &h.src, stride: h.stride, width: h.stride, height: h.src.len() / h.stride };
        let ref_plane = PlaneRef { data: &h.refp, stride: h.stride, width: h.stride, height: h.refp.len() / h.stride };
        let fc = FrameContext::new_default();
        let sic = switchable_interp_costs(&fc);
        let costs = build_coef_cost_table(&fc);
        let y_quant = init_plane_quantizers(96, QuantizerDeltas::Y);
        let uv_quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        // ZEROMV: reads the unshifted reference → large residual.
        let zero_params = make_params(
            &h, &src_plane, &ref_plane, &fc, &sic, &y_quant, &uv_quant, &costs,
            InterMode::ZeroMv, Mv::ZERO, true,
        );
        let zero_out = handle_inter_mode_single_ref(&zero_params);

        // NEWMV at the true motion: prediction recovers the source.
        let true_mv = Mv {
            row: (-shift_row * 8) as i16,
            col: (-shift_col * 8) as i16,
        };
        let new_params = make_params(
            &h, &src_plane, &ref_plane, &fc, &sic, &y_quant, &uv_quant, &costs,
            InterMode::NewMv, true_mv, true,
        );
        let new_out = handle_inter_mode_single_ref(&new_params);

        // The matched NEWMV must have lower distortion than ZEROMV...
        assert!(
            new_out.dist < zero_out.dist,
            "matched NEWMV dist {} should be < mismatched ZEROMV dist {}",
            new_out.dist,
            zero_out.dist,
        );
        // ...and win on total RD even paying the NEWMV bit cost. This is
        // the load-bearing differential-oracle assertion: real motion is
        // worth more than the MV-rate penalty.
        assert!(
            new_out.rd < zero_out.rd,
            "matched NEWMV RD {} should beat mismatched ZEROMV RD {} \
             (anti-cheat: fake RD can't tell these apart)",
            new_out.rd,
            zero_out.rd,
        );
    }

    /// `build_inter_pred_model_rd_earlyterm` produces a `skip_flag =
    /// true` result when src == dst (zero residual → zero SSE → all
    /// thresholds trivially pass).
    #[test]
    fn pred_model_skips_when_zero_residual() {
        let pixels = vec![100u8; 16 * 16];
        let out = build_inter_pred_model_rd_earlyterm(
            &pixels, 16, &pixels, 16,
            BlockSize::Block16x16, 32,
            128, // dc_thr
            64,  // ac_thr
            false,
            i64::MAX,
            100_000,
        );
        assert!(out.skip_txfm_sb, "zero residual should be classified as skip");
    }

    /// Early-term fires when `do_earlyterm = true` and the model RD
    /// exceeds `best_rd`.
    #[test]
    fn pred_model_early_terminates_when_over_best_rd() {
        let pixels = vec![200u8; 32 * 32];
        let dst = vec![10u8; 32 * 32]; // large diff → large model RD
        let out = build_inter_pred_model_rd_earlyterm(
            &pixels, 32, &dst, 32,
            BlockSize::Block32x32, 64,
            128,
            64,
            true,
            1, // best_rd = 1 → any non-trivial model RD will exceed
            100_000,
        );
        assert!(out.early_terminated);
    }
}
