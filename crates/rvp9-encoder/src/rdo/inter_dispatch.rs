//! Wave 4 RDO integration — per-leaf inter dispatch glue.
//!
//! Composes the now-real RDO primitives into a single per-leaf entry
//! the encoder hot path can call at `speed.unsigned_abs() <= 4` in
//! place of the realtime [`rvp9_core::mode_decision::inter::pick_inter_mode`]
//! picker. The output is a `(mode, mv, ref, filter)` tuple that flows
//! into the SAME shared emit TX walk
//! (`lib.rs::pick_and_simulate_inter_encode_partitions`), so the
//! coef-source bridge is shared and the decoder re-derives `best_ref_mv`
//! from the mi-grid identically (the byte-exactness invariant the RT
//! path already maintains).
//!
//! ## libvpx references
//!
//! - `vp9_rd_pick_inter_mode_sb` driver — `vp9/encoder/vp9_rdopt.c:3438-4187`.
//! - `rd_pick_sb_modes` dispatch — `vp9/encoder/vp9_encodeframe.c:1975-2107`.
//! - `single_motion_search` — `vp9_rdopt.c:2564-2745` (ported in
//!   [`crate::rdo::single_mv_search`]).
//! - `handle_inter_mode` — `vp9_rdopt.c:2812-3219` (ported, real
//!   transform-domain RD, in [`crate::rdo::handle_inter`]).
//!
//! ## What this glue builds (assessment items 1-4)
//!
//! 1. A [`HandleInterRunner`] factory per ref-frame, wired to the real
//!    [`handle_inter_mode_single_ref`] with real Y/UV quantizers, the
//!    [`CoefCostTable`], `tx_size_cost`, skip costs and U/V plane
//!    slices, and the real `inter_mode_costs` mode rate (NOT the
//!    standalone placeholder).
//! 2. Ref-MV assembly: pre-runs [`single_motion_search`] for NEWMV and
//!    threads the result into [`RefMvCandidates`].
//! 3. Intra fallback: surfaced but DISABLED (`i64::MAX/2`) — the
//!    minimal-real cut keeps the intra branch on the RT fallback path
//!    (see scope note below). The driver still returns an inter winner.
//! 4. Maps the RDO winner `(mode, mv, ref)` back to the leaf
//!    representation, recovering the winning interp filter via one final
//!    [`handle_inter_mode_single_ref`] call (the driver discards the
//!    per-mode filter; re-deriving it for the single winner is cheap).
//!
//! ## Scope (recorded deviations)
//!
//! - **Blocks ≤ 32×32 only** (`handle_inter`'s single transform-tile
//!   scope, [`crate::rdo::handle_inter`] module docs). 8×8 and 64×64
//!   fall back to the RT picker within the speed≤4 path; the caller
//!   gates on this via [`InterDispatchLeaf::rdo_supported`].
//! - **Single-ref.** `{LAST, GOLDEN}`; compound is out of W4 scope.
//! - **Intra fallback disabled** — see item 3. A leaf whose true best
//!   is intra is decided by the inter cross-product here; the RT path's
//!   richer intra-in-inter evaluation is not invoked in the RDO branch.

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::block::BlockSize;
use rvp9_core::cost::SwitchableInterpCosts;
use rvp9_core::cost::coef_cost::CoefCostTable;
use rvp9_core::cost::mv_cost::MvCosts;
use rvp9_core::cost::symbol_costs::InterModeCosts;
use rvp9_core::entropy::default_probs::INTER_MODE_CONTEXTS;
use rvp9_core::mode_info::{InterMode, Mv, RefFrame};
use rvp9_core::mv_search::full_pixel::PlaneRef;
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::mv_search::search_site::SearchSiteConfig;
use rvp9_core::mv_search::sub_pixel::SubPixelForcedStop;
use rvp9_core::quant::forward::PlaneQuantizers;
use rvp9_core::transforms::TxMode;

use crate::rdo::handle_inter::{
    HandleInterModeParams, HandleInterModeResult, handle_inter_mode_single_ref,
};
use crate::rdo::pick_inter_sb::{
    IntraFallback, PickInterModeSbParams, PickResult, RefFrameInput, RefMvCandidates,
    pick_inter_mode_sb,
};
use crate::rdo::ref_costs::{ReferenceMode, RefFrameContext, estimate_ref_frame_costs};
use crate::rdo::single_mv_search::{
    SingleMotionSearchParams, SingleMotionSearchResult, single_motion_search,
};

/// Global RDO-dispatch invocation counter. Incremented once per leaf the
/// dispatch actually decides (an inter winner produced). Used by the
/// PROOF test to confirm the speed≤4 path routes through this module.
pub static RDO_DISPATCH_INVOCATIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Per-ref-frame plane + candidate bundle the caller assembles from the
/// running mi-grid (same `find_mv_refs` it already runs for the RT path).
pub struct RefInputs<'a> {
    /// `true` when this ref-frame is available for the leaf.
    pub available: bool,
    /// Y reference plane (already the chosen ref frame's recon).
    pub ref_y: &'a PlaneRef<'a>,
    /// U reference plane (4:2:0).
    pub ref_u: &'a [u8],
    /// V reference plane (4:2:0).
    pub ref_v: &'a [u8],
    /// `find_mv_refs(NEARMV)`-derived NEAREST candidate (post
    /// `lower_mv_precision`). Identical to the RT path's `mvc.nearest`.
    pub nearest: Mv,
    /// NEAR candidate. Identical to the RT path's `mvc.near`.
    pub near: Mv,
    /// `true` iff `nearest` is inside `mv_limits` (no-border-extend
    /// envelope). When false the NEAREST candidate is vetoed (mirrors
    /// the RT path substituting `Mv::ZERO`).
    pub nearest_in_bounds: bool,
    /// `true` iff `near` is inside `mv_limits`.
    pub near_in_bounds: bool,
}

/// Inputs for a single leaf's RDO inter dispatch. Mirrors the per-leaf
/// context the RT loop already computes (plane views, offsets, cost
/// tables, quantizers, MV envelope).
pub struct InterDispatchLeaf<'a> {
    /// Block size (must be 16×16 or 32×32 — see [`Self::rdo_supported`]).
    pub bsize: BlockSize,
    /// Y source plane.
    pub src_y: &'a PlaneRef<'a>,
    /// Top-left of the block in luma samples `(col, row)`.
    pub src_offset: (usize, usize),
    /// U source plane.
    pub src_u: &'a [u8],
    /// V source plane.
    pub src_v: &'a [u8],
    /// Chroma stride (4:2:0).
    pub uv_stride: usize,
    /// LAST ref bundle.
    pub last: RefInputs<'a>,
    /// GOLDEN ref bundle.
    pub golden: RefInputs<'a>,
    /// Full-pel MV envelope (no-border-extend), shared with NEWMV search.
    pub mv_limits: MvLimits,
    /// Diamond search site config.
    pub site_config: &'a SearchSiteConfig,
    /// MV-cost LUT.
    pub mv_costs: &'a MvCosts,
    /// SB-level int_pro NEWMV seed (`x->pred_mv[ref]`), if any.
    pub newmv_seed: Option<Mv>,
    /// Y quantizers for this leaf's effective qindex.
    pub y_quant: &'a PlaneQuantizers,
    /// Chroma quantizers (U/V share, 4:2:0).
    pub uv_quant: &'a PlaneQuantizers,
    /// Coefficient-bit cost table.
    pub coef_costs: &'a CoefCostTable,
    /// Per-context inter-mode cost table (real, from `inter_mode_costs`).
    pub inter_mode_costs: &'a InterModeCosts,
    /// Inter-mode context (0..7) from `get_mode_context`.
    pub inter_mode_ctx: usize,
    /// Switchable-interp cost table.
    pub switchable_interp_costs: &'a SwitchableInterpCosts,
    /// Switchable-interp context (0..4).
    pub switchable_interp_ctx: usize,
    /// `rdmult` (lambda) for this leaf's qindex.
    pub rdmult: i64,
    /// `tx_size` signalling cost, indexed by `TxSize as usize`.
    pub tx_size_cost: [u32; 4],
    /// Skip-bit cost (skip = false).
    pub skip_cost0: u32,
    /// Skip-bit cost (skip = true).
    pub skip_cost1: u32,
    /// `error_per_bit` for the NEWMV sub-pel refinement.
    pub error_per_bit: i32,
    /// `sad_per_bit` for the NEWMV full-pel diamond.
    pub sad_per_bit: i32,
}

impl InterDispatchLeaf<'_> {
    /// `true` iff this leaf's block size is in `handle_inter`'s scope
    /// (single transform tile, ≤ 32×32). The caller falls back to the
    /// RT picker otherwise.
    #[must_use]
    pub fn rdo_supported(bsize: BlockSize) -> bool {
        matches!(bsize, BlockSize::Block16x16 | BlockSize::Block32x32)
    }
}

/// RDO dispatch winner mapped to the leaf representation.
#[derive(Copy, Clone, Debug)]
pub struct InterDispatchResult {
    /// Winning inter mode (post NEWMV→NEAREST/NEAR/ZERO rewrite).
    pub mode: InterMode,
    /// Winning MV (1/8-pel).
    pub mv: Mv,
    /// Winning ref-frame (`LastFrame` / `GoldenFrame`).
    pub ref_frame: RefFrame,
    /// Winning interp filter.
    pub interp_filter: InterpFilter,
}

/// Build the real per-mode inter-mode rate for `(mode, ctx)` from the
/// frame's `inter_mode_costs` table (libvpx `cost_mv_ref`). Replaces the
/// standalone driver's placeholder.
#[inline]
fn real_mode_rate(costs: &InterModeCosts, ctx: usize, mode: InterMode) -> u32 {
    let ctx = ctx.min(INTER_MODE_CONTEXTS - 1);
    costs[ctx][mode.idx()].max(0) as u32
}

/// Pre-run [`single_motion_search`] for the NEWMV candidate of one ref
/// frame. Returns `(Some(mv), rate_mv)` when the search succeeds and the
/// MV stays inside `mv_limits`; `(None, 0)` otherwise (drops NEWMV).
fn search_newmv(leaf: &InterDispatchLeaf<'_>, refs: &RefInputs<'_>) -> (Option<Mv>, u32) {
    // libvpx single_motion_search uses ref_mvs[ref][0] (= NEAREST) as the
    // cost-reference MV; NEAR seeds the enhanced full-pel slot.
    let params = SingleMotionSearchParams {
        block_size: leaf.bsize,
        src_plane: leaf.src_y,
        src_offset: leaf.src_offset,
        ref_plane: refs.ref_y,
        ref_mv: refs.nearest,
        near_mv: refs.near,
        pred_mv: leaf.newmv_seed,
        // Slot 2 (= pred_mv / int_pro seed) when a seed is present,
        // else slot 0 (NEAREST). Mirrors x->mv_best_ref_index.
        best_predmv_idx: if leaf.newmv_seed.is_some() { 2 } else { 0 },
        mv_step_param: 4,
        adaptive_motion_search: false,
        enhanced_full_pixel_motion_search: false,
        pred_mv_sad: 0,
        other_pred_mv_sad_min: u32::MAX,
        mv_limits: leaf.mv_limits,
        site_config: leaf.site_config,
        mv_costs: leaf.mv_costs,
        sad_per_bit: leaf.sad_per_bit,
        error_per_bit: leaf.error_per_bit,
        allow_hp: false,
        subpel_force_stop: SubPixelForcedStop::HalfPel,
    };
    match single_motion_search(&params) {
        SingleMotionSearchResult::Ok { best_mv, rate_mv } => {
            if mv_in_limits(best_mv, &leaf.mv_limits) {
                (Some(best_mv), rate_mv)
            } else {
                (None, 0)
            }
        }
        SingleMotionSearchResult::InvalidMv => (None, 0),
    }
}

/// `true` iff `mv` (1/8-pel) is inside the full-pel `limits` envelope.
#[inline]
fn mv_in_limits(mv: Mv, limits: &MvLimits) -> bool {
    let col_fp = i32::from(mv.col) >> 3;
    let row_fp = i32::from(mv.row) >> 3;
    col_fp >= limits.col_min
        && col_fp <= limits.col_max
        && row_fp >= limits.row_min
        && row_fp <= limits.row_max
}

/// Build a [`HandleInterModeParams`] for one `(mode, mv, filter-search)`
/// candidate against one ref-frame's planes.
#[allow(clippy::too_many_arguments)]
fn make_handle_params<'a>(
    leaf: &'a InterDispatchLeaf<'a>,
    refs: &'a RefInputs<'a>,
    mode: InterMode,
    mv: Mv,
    inter_mode_rate: u32,
    newmv_rate: u32,
) -> HandleInterModeParams<'a> {
    HandleInterModeParams {
        bsize: leaf.bsize,
        src_plane: leaf.src_y,
        src_offset: leaf.src_offset,
        ref_plane: refs.ref_y,
        interp_filter: InterpFilter::EightTap,
        interp_filter_switchable: true,
        interp_filter_bilinear: false,
        switchable_interp_costs: leaf.switchable_interp_costs,
        switchable_interp_ctx: leaf.switchable_interp_ctx,
        this_mode: mode,
        mv,
        inter_mode_rate,
        newmv_rate,
        rdmult: leaf.rdmult,
        ref_best_rd: i64::MAX,
        use_rd_breakout: false,
        y_quant: leaf.y_quant,
        uv_quant: leaf.uv_quant,
        coef_costs: leaf.coef_costs,
        tx_mode: TxMode::TxModeSelect,
        tx_size_cost: leaf.tx_size_cost,
        skip_cost0: leaf.skip_cost0,
        skip_cost1: leaf.skip_cost1,
        src_u: leaf.src_u,
        src_v: leaf.src_v,
        ref_u: refs.ref_u,
        ref_v: refs.ref_v,
        uv_stride: leaf.uv_stride,
    }
}

/// Run the RDO inter picker for one leaf. Returns the winning
/// `(mode, mv, ref, filter)` or `None` if no inter candidate succeeded
/// (caller falls back to RT, e.g. when both refs are OOB-only).
///
/// PRECONDITION: `InterDispatchLeaf::rdo_supported(leaf.bsize)`.
#[must_use]
pub fn pick_inter_sb_leaf(leaf: &InterDispatchLeaf<'_>) -> Option<InterDispatchResult> {
    debug_assert!(
        InterDispatchLeaf::rdo_supported(leaf.bsize),
        "RDO dispatch called on out-of-scope block size {:?}",
        leaf.bsize,
    );

    // Ref-frame signalling costs (real, libvpx estimate_ref_frame_costs).
    let ref_costs = estimate_ref_frame_costs(
        false,
        ReferenceMode::SingleReference,
        RefFrameContext::default(),
    );

    // Item 2 — pre-run NEWMV search per ref.
    let (last_newmv, last_newmv_rate) = if leaf.last.available {
        search_newmv(leaf, &leaf.last)
    } else {
        (None, 0)
    };
    let (golden_newmv, golden_newmv_rate) = if leaf.golden.available {
        search_newmv(leaf, &leaf.golden)
    } else {
        (None, 0)
    };

    // OOB veto: substitute Mv::ZERO for an out-of-bounds NEAREST/NEAR so
    // the runner's prediction never over-reads the plane (mirror of the
    // RT path's mv_in_bounds veto). The post-pick rewrite will still map
    // a NEW/NEAREST/NEAR winner onto a decodable MV; the writer derives
    // its own best_ref_mv from the mi-grid identically.
    let last_cands = RefMvCandidates {
        nearest: if leaf.last.nearest_in_bounds { leaf.last.nearest } else { Mv::ZERO },
        near: if leaf.last.near_in_bounds { leaf.last.near } else { Mv::ZERO },
        newmv: last_newmv,
        newmv_rate: last_newmv_rate,
    };
    let golden_cands = RefMvCandidates {
        nearest: if leaf.golden.nearest_in_bounds { leaf.golden.nearest } else { Mv::ZERO },
        near: if leaf.golden.near_in_bounds { leaf.golden.near } else { Mv::ZERO },
        newmv: golden_newmv,
        newmv_rate: golden_newmv_rate,
    };

    // Item 1 — HandleInterRunner factory per ref. Each closure builds
    // real HandleInterModeParams and runs the real
    // handle_inter_mode_single_ref. The driver passes its own
    // placeholder `mode_rate` (`_placeholder`); we OVERRIDE it with the
    // real `inter_mode_costs` lookup so the candidate ranking uses the
    // frame's adapted mode-rate table (assessment deviation #4 fix).
    let imc = leaf.inter_mode_costs;
    let ctx = leaf.inter_mode_ctx;
    let last_runner: Option<crate::rdo::pick_inter_sb::HandleInterRunner> =
        if leaf.last.available {
            let leaf_ref: &InterDispatchLeaf = leaf;
            let refs: &RefInputs = &leaf.last;
            Some(Box::new(move |mode: InterMode, mv: Mv, _placeholder: u32, newmv_rate: u32| {
                let mode_rate = real_mode_rate(imc, ctx, mode);
                let params = make_handle_params(leaf_ref, refs, mode, mv, mode_rate, newmv_rate);
                Some(handle_inter_mode_single_ref(&params))
            }))
        } else {
            None
        };
    let golden_runner: Option<crate::rdo::pick_inter_sb::HandleInterRunner> =
        if leaf.golden.available {
            let leaf_ref: &InterDispatchLeaf = leaf;
            let refs: &RefInputs = &leaf.golden;
            Some(Box::new(move |mode: InterMode, mv: Mv, _placeholder: u32, newmv_rate: u32| {
                let mode_rate = real_mode_rate(imc, ctx, mode);
                let params = make_handle_params(leaf_ref, refs, mode, mv, mode_rate, newmv_rate);
                Some(handle_inter_mode_single_ref(&params))
            }))
        } else {
            None
        };

    let per_ref = [
        // IntraFrame slot — unused (intra fallback disabled).
        RefFrameInput {
            available: false,
            candidates: RefMvCandidates::default(),
            handle_inter_runner: None,
            rdmult: leaf.rdmult,
        },
        RefFrameInput {
            available: leaf.last.available,
            candidates: last_cands,
            handle_inter_runner: last_runner,
            rdmult: leaf.rdmult,
        },
        RefFrameInput {
            available: leaf.golden.available,
            candidates: golden_cands,
            handle_inter_runner: golden_runner,
            rdmult: leaf.rdmult,
        },
        // AltrefFrame slot — unavailable (single-LAST/GOLDEN scheme).
        RefFrameInput {
            available: false,
            candidates: RefMvCandidates::default(),
            handle_inter_runner: None,
            rdmult: leaf.rdmult,
        },
    ];

    let params = PickInterModeSbParams {
        ref_costs,
        per_ref,
        // Item 3 — intra fallback disabled (minimal-real cut).
        intra_fallback: IntraFallback {
            rate: 0,
            dist: 0,
            rd: i64::MAX / 2,
        },
        is_src_frame_alt_ref: false,
        mode_counter: None,
    };

    let pick: PickResult = pick_inter_mode_sb(&params);
    if pick.picked_intra {
        // Should not happen with intra disabled, but if every inter
        // candidate was dropped (e.g. all OOB) the driver returns intra
        // → no usable inter winner. Fall back to RT.
        return None;
    }

    // Item 4 — recover the winning interp filter. The driver discards the
    // per-mode FilterRd; re-run handle_inter once for the single winner.
    let refs = match pick.ref_frame {
        RefFrame::LastFrame => &leaf.last,
        RefFrame::GoldenFrame => &leaf.golden,
        _ => return None,
    };
    let mode_rate = real_mode_rate(imc, ctx, pick.mode);
    let newmv_rate = if matches!(pick.mode, InterMode::NewMv) {
        if matches!(pick.ref_frame, RefFrame::LastFrame) { last_newmv_rate } else { golden_newmv_rate }
    } else {
        0
    };
    let winner_params = make_handle_params(leaf, refs, pick.mode, pick.mv, mode_rate, newmv_rate);
    let winner: HandleInterModeResult = handle_inter_mode_single_ref(&winner_params);

    RDO_DISPATCH_INVOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    Some(InterDispatchResult {
        mode: pick.mode,
        mv: pick.mv,
        ref_frame: pick.ref_frame,
        interp_filter: winner.best_filter,
    })
}
