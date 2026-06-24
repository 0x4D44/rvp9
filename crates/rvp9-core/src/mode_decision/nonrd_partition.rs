//! Motion-aware (RD-driven) partition descent — F.3 infrastructure.
//!
//! Ports the **square-only NONE-vs-SPLIT skeleton** of libvpx's
//! `nonrd_pick_partition` (`vp9/encoder/vp9_encodeframe.c:4922-5210`).
//! This is the layer that runs ON TOP of the variance fast-path
//! (`mode_decision::partition::choose_partitioning`, libvpx
//! `vp9_encodeframe.c:1261`) and decides whether to split a larger
//! block based on whether motion-compensated RD justifies it.
//!
//! ## Status — F.3 ships INFRASTRUCTURE-ONLY (2026-05-23)
//!
//! Mirrors the F.1 / F.2 pattern. The descent is implemented in full
//! here and exposed for unit testing, but the encoder's default
//! partition path is **unchanged** — the variance hint still drives
//! the picker. The new descent only runs when
//! `RVP9_F3_NONRD_PARTITION_ENABLE=1` is set (the gate lives in
//! `crates/rvp9-encoder/src/lib.rs` next to the picker harness).
//!
//! F.4 owns flipping the env gate ON as part of lambda recalibration
//! (see `notes/divergence-ledger.md` F.3 ship-state entry).
//!
//! ## Scope (F.3 first cut + A4-rect, 2026-05-27)
//!
//! Per the F.3 HLD §2.3 / §4.3 / §10, extended by A4-rect:
//!
//! - **Rectangular partitions ADDED (A4-rect).** At each square parent
//!   the descent now also evaluates PARTITION_HORZ (two `w × h/2`
//!   blocks stacked) and PARTITION_VERT (two `w/2 × h` blocks side by
//!   side), coding each rect as TWO sub-blocks at the rect size (NOT
//!   recursed) and summing their RD — mirroring libvpx
//!   `rd_pick_partition` (`vp9_encodeframe.c:4342-4436`). The
//!   `partition_{horz,vert}_allowed` size constraints
//!   (`vp9_encodeframe.c:4018-4021`, `bsize >= BLOCK_8X8`) are honoured
//!   via the `!bsize_at_floor` + `bsize <= max_partition_size` gates.
//!   Frame-edge `force_{horz,vert}_split` overrides are not modelled
//!   (the descent runs only on 64-aligned SBs; the picker callback's
//!   SB-edge guard makes any off-frame rect half infeasible).
//! - **Recursion caps at Block16x16.** The existing picker bails to
//!   ZERO+skip at Block8x8 (`crates/rvp9-encoder/src/lib.rs:2660-2711`);
//!   F.3 inherits that. Any 8×8 leaves come from the variance hint.
//! - **No ML partitioning.**
//! - **No PC_TREE.** rvp9 carries working state on the stack.
//!
//! ## Algorithm — `nonrd_pick_partition` skeleton
//!
//! Libvpx reference at `vp9_encodeframe.c:4922-5210`. The square-only
//! skeleton at each `bsize`:
//!
//! ```text
//! if partition_none_allowed:           // libvpx :5000
//!     candidate_none = mode_picker(bsize, sb_local_xy)
//!     this_rdc = rd_cost(rate + partition_cost[ctx][NONE], dist)
//!     if this_rdc < best:
//!         best = this_rdc
//!         best_partition = PARTITION_NONE
//!         // breakout (libvpx :5017-5028): if rate AND dist are both
//!         // far below the q-conditional thresholds, do_split = 0.
//! if do_split && bsize > Block16x16:   // libvpx :5037
//!     sum_rdc = partition_cost[ctx][SPLIT]
//!     for each of 4 children:
//!         sum_rdc += recurse(subsize, ...)
//!         if sum_rdc >= best:
//!             break       // libvpx :5042 — early-out
//!     if sum_rdc < best:
//!         best = sum_rdc
//!         best_partition = PARTITION_SPLIT
//! ```
//!
//! ## Testability
//!
//! The descent is parametric over a [`ModePickerFn`] callback so it can
//! be unit-tested with a synthetic picker that returns known RD costs
//! for known motion patterns. The encoder's integration in `lib.rs`
//! supplies a real callback driven by `pick_inter_mode_luma`.

use crate::block::{BlockSize, PartitionType};
use crate::cost::symbol_costs::PartitionCosts;
use crate::mode_decision::partition::{PartitionDecision, SB_8X8_GRID};
use crate::mode_decision::rd_cost;
use crate::mode_info::Mv;

/// Side length of the SB grid in 8×8 mi cells (8 cells = 64 px). Mirrors
/// `partition::SB_8X8_GRID` but kept private here to avoid a public
/// re-export.
const SB_GRID: usize = SB_8X8_GRID;

/// libvpx `PARTITION_NONE = 0` / `PARTITION_HORZ = 1` /
/// `PARTITION_VERT = 2` / `PARTITION_SPLIT = 3` — see
/// `vp9/common/vp9_enums.h:84-89`. The `partition_cost` table is indexed
/// `[ctx][PARTITION_*]` in the same order.
const PARTITION_NONE_IDX: usize = 0;
const PARTITION_HORZ_IDX: usize = 1;
const PARTITION_VERT_IDX: usize = 2;
const PARTITION_SPLIT_IDX: usize = 3;

/// libvpx `partition_plane_context` (`vp9_onyxc_int.h:451-462`) reduces
/// to `(left * 2 + above) + bsl * PARTITION_PLOFFSET` where `bsl =
/// mi_width_log2_lookup[bsize]` and `PARTITION_PLOFFSET = 4`.
///
/// `mi_width_log2_lookup`:
/// - `Block8x8`  → 0
/// - `Block16x16` → 1
/// - `Block32x32` → 2
/// - `Block64x64` → 3
fn bsl_for_partition_ctx(bsize: BlockSize) -> usize {
    match bsize {
        BlockSize::Block64x64 => 3,
        BlockSize::Block32x32 => 2,
        BlockSize::Block16x16 => 1,
        BlockSize::Block8x8 => 0,
        // Non-square / sub-8 sizes don't appear in the F.3 descent
        // (capped at Block16x16, square-only). Reaching here is a
        // programming error in the caller — assert loudly in debug
        // so it's caught at test time, and return 0 in release as a
        // safe default (the partition_cost lookup still indexes a
        // valid context row).
        other => {
            debug_assert!(
                false,
                "F.3 descent: unexpected non-square bsize {:?} reached \
                 partition_plane_context; descent only walks square sizes \
                 {{Block8x8, Block16x16, Block32x32, Block64x64}}",
                other,
            );
            0
        }
    }
}

/// Compute libvpx's `partition_plane_context` for the F.3 descent.
///
/// The `above`/`left` bits come from the `above_seg_context` /
/// `left_seg_context` arrays the encoder maintains. For the F.3
/// first-cut descent these contexts are bootstrapped from the variance
/// hint and updated as the descent decides each sub-block.
///
/// Returns an index in `[0, PARTITION_CONTEXTS)` where
/// `PARTITION_CONTEXTS = 16`.
#[inline]
#[must_use]
pub fn partition_plane_context(above: bool, left: bool, bsize: BlockSize) -> usize {
    let bsl = bsl_for_partition_ctx(bsize);
    let a = usize::from(above);
    let l = usize::from(left);
    (l * 2 + a) + bsl * 4
}

/// One NONE-candidate evaluation returned by [`ModePickerFn`].
///
/// `rate` and `dist` are in the same units the picker uses for its
/// own RD cost: `rate` in `vp9_prob_cost` units (1/512-bit) summing
/// `mode_rate + mv_rate + ref_frame_rate_cost + (F.1 coef_rate when
/// active)`; `dist` is the picker's distortion term (pixel-SSE of
/// recon in F.1's first cut).
///
/// `mv`, `mode`, `ref_frame` are surfaced so the caller can splat the
/// winning candidate back into its leaf list when PARTITION_NONE wins
/// at that bsize. The `ref_frame` is an opaque `u8` (caller-defined
/// encoding — typically `RefFrame as u8`) to keep this module free of
/// encoder-side dependencies.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PartitionRdCandidate {
    /// libvpx `this_rdc.rate` (`vp9_prob_cost` units, 1/512-bit).
    pub rate: u32,
    /// libvpx `this_rdc.dist` (sum-of-squared-error units).
    pub dist: u32,
    /// Picked inter MV in 1/8-pel units. `Mv::ZERO` for skip-fallback
    /// candidates.
    pub mv: Mv,
    /// Picked inter mode discriminant (caller-defined encoding — the
    /// integration in `lib.rs` uses `InterMode as u8`).
    pub mode: u8,
    /// Picked reference frame discriminant (caller-defined encoding —
    /// the integration uses `RefFrame as u8`).
    pub ref_frame: u8,
}

impl PartitionRdCandidate {
    /// Sentinel "no candidate" — used by the descent when the picker
    /// returns `None` (e.g. partition_none not allowed at this
    /// bsize/position).
    pub const IMPOSSIBLE: Self = Self {
        rate: u32::MAX,
        dist: u32::MAX,
        mv: Mv::ZERO,
        mode: 0,
        ref_frame: 0,
    };
}

/// Mode-picker callback — evaluates PARTITION_NONE at a (sub-block
/// origin, bsize) within the current SB.
///
/// - `sb_local_x_8x8`, `sb_local_y_8x8`: top-left of the sub-block in
///   8×8 mi units within the SB (`0..=7`).
/// - `bsize`: square block size being evaluated (`Block16x16` /
///   `Block32x32` / `Block64x64` — never `Block8x8`, the descent caps
///   at 16×16 per §4.3).
///
/// Returns `None` if NONE isn't a feasible candidate at this
/// (position, bsize) — e.g. the variance hint excluded this size at
/// this position. The descent then falls through to the SPLIT branch.
pub trait ModePickerFn {
    /// Evaluate PARTITION_NONE.
    fn pick(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
    ) -> Option<PartitionRdCandidate>;
}

/// Allow naked closures as mode pickers.
impl<F> ModePickerFn for F
where
    F: FnMut(usize, usize, BlockSize) -> Option<PartitionRdCandidate>,
{
    fn pick(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
    ) -> Option<PartitionRdCandidate> {
        (self)(sb_local_x_8x8, sb_local_y_8x8, bsize)
    }
}

/// A21 — partition-search breakout hook (libvpx `ml_predict_breakout`
/// gate, `vp9/encoder/vp9_encodeframe.c:4161-4168`).
///
/// After the descent evaluates PARTITION_NONE at a block and that NONE
/// candidate IMPROVES the running best RD, libvpx consults the ML
/// breakout model: if it votes "stop searching", `do_split` (and
/// `do_rect`) are forced to 0 — the SPLIT recursion at this block is
/// skipped entirely. This trait abstracts that vote so the encoder can
/// plug the real `ml_partition::ml_predict_breakout` (with real source
/// variance + AC quantiser + rdmult features) without pulling the
/// `ml_partition` module into `rvp9-core`.
///
/// The hook is consulted ONLY when a NONE candidate is feasible and won
/// the current best at this block (mirroring libvpx's placement inside
/// the `this_rdc.rdcost < best_rdc.rdcost` arm at
/// `vp9_encodeframe.c:4140-4168`). Returning `true` suppresses the SPLIT
/// branch for this block.
pub trait BreakoutPruner {
    /// Vote on whether to stop the partition search at this block.
    ///
    /// - `sb_local_x_8x8`, `sb_local_y_8x8`: top-left of the block in
    ///   8×8 mi units within the SB.
    /// - `bsize`: the block the NONE candidate was evaluated at.
    /// - `none_rate`, `none_dist`: the winning NONE candidate's rate
    ///   (`vp9_prob_cost` units, incl. `partition_cost[NONE]`) and
    ///   distortion (SSE) — libvpx's `this_rdc.rate` / `this_rdc.dist`
    ///   at `vp9_encodeframe.c:4165`.
    ///
    /// Returns `true` to force `do_split = 0` (skip the SPLIT recursion).
    fn breakout(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
        none_rate: u32,
        none_dist: u32,
    ) -> bool;
}

/// No-op breakout pruner — never votes to break out. Used as the
/// default when the speed feature `sf.rd_ml_partition.search_breakout`
/// is off, so the descent behaves exactly as the pre-A21 baseline.
pub struct NoBreakout;

impl BreakoutPruner for NoBreakout {
    #[inline]
    fn breakout(&mut self, _: usize, _: usize, _: BlockSize, _: u32, _: u32) -> bool {
        false
    }
}

/// The RD inputs `ml_prune_rect_partition` reads about the current
/// block, gathered by the descent AFTER it has evaluated PARTITION_NONE
/// and PARTITION_SPLIT (mirroring libvpx's call placement at
/// `vp9/encoder/vp9_encodeframe.c:4332-4339`, which runs after the
/// NONE/SPLIT arms have populated `pc_tree`).
#[derive(Copy, Clone, Debug)]
pub struct RectPruneRdContext {
    /// libvpx `best_rdc.rdcost` — the running best RD at this block
    /// (the `ref_rd` argument, `vp9_encodeframe.c:4338`). `None` when
    /// no arm was feasible (the rect pruner is then not consulted).
    pub best_rd_cost: Option<i64>,
    /// libvpx `pc_tree->none.rdcost` — the PARTITION_NONE arm's RD at
    /// this block, or `None` if NONE was infeasible.
    pub none_rd_cost: Option<i64>,
    /// libvpx `pc_tree->split[i]->none.rdcost` — each SPLIT child's own
    /// PARTITION_NONE-arm RD (NOT the child's best). `None` per slot if
    /// that child's NONE arm was infeasible.
    pub split_none_rd_cost: [Option<i64>; 4],
    /// libvpx `pc_tree->partitioning == PARTITION_NONE` — whether NONE
    /// is the partition currently winning at this block.
    pub partitioning_is_none: bool,
}

/// A21-rect — rectangular-partition pruning hook (libvpx
/// `ml_prune_rect_partition`, `vp9/encoder/vp9_encodeframe.c:3334-3425`;
/// gate at `:4332-4339`).
///
/// After the descent has evaluated PARTITION_NONE and PARTITION_SPLIT at
/// a block, libvpx consults the ML rect-prune model to decide whether
/// the PARTITION_HORZ / PARTITION_VERT arms are still worth searching,
/// ANDing the verdict into the pre-existing `partition_horz_allowed` /
/// `partition_vert_allowed` masks. This trait abstracts that vote so the
/// encoder can plug the real `ml_partition::ml_prune_rect_partition`
/// (with the real DC quantiser + RD features) without pulling the
/// `ml_partition` module into `rvp9-core`.
///
/// The hook is consulted ONLY at blocks where rect is otherwise eligible
/// (`do_split && !bsize_at_floor && rect_within_max`, i.e. libvpx's
/// `(partition_horz_allowed || partition_vert_allowed) && bsize >
/// BLOCK_8X8` at `:4334`). The returned `(allow_horz, allow_vert)` gate
/// the descent's rect evaluation: a `false` skips that rect arm's picker
/// invocations entirely.
pub trait RectPruner {
    /// Vote on which rectangular arms to keep at this block.
    ///
    /// - `sb_local_x_8x8`, `sb_local_y_8x8`: top-left of the block in
    ///   8×8 mi units within the SB.
    /// - `bsize`: the square parent being partitioned (16/32/64).
    /// - `rd`: the NONE/SPLIT RD context gathered so far.
    /// - `allow_horz_in`, `allow_vert_in`: the pre-existing arm masks
    ///   (both `true` at the descent's call site — the size gates are
    ///   handled by the caller before this hook).
    ///
    /// Returns `(allow_horz, allow_vert)`; each is the input AND the
    /// model verdict (libvpx `:3422-3423`).
    fn prune_rect(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
        rd: RectPruneRdContext,
        allow_horz_in: bool,
        allow_vert_in: bool,
    ) -> (bool, bool);
}

/// No-op rect pruner — keeps both arms (never prunes). Used as the
/// default when `sf.rd_ml_partition.prune_rect_thresh` disables the
/// model, so the descent behaves exactly as the pre-A21-rect baseline.
pub struct NoRectPrune;

impl RectPruner for NoRectPrune {
    #[inline]
    fn prune_rect(
        &mut self,
        _: usize,
        _: usize,
        _: BlockSize,
        _: RectPruneRdContext,
        allow_horz_in: bool,
        allow_vert_in: bool,
    ) -> (bool, bool) {
        (allow_horz_in, allow_vert_in)
    }
}

/// Allow naked closures as rect pruners (so the encoder can plug an
/// inline `ml_prune_rect_partition` adapter without a named struct).
impl<F> RectPruner for F
where
    F: FnMut(usize, usize, BlockSize, RectPruneRdContext, bool, bool) -> (bool, bool),
{
    fn prune_rect(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
        rd: RectPruneRdContext,
        allow_horz_in: bool,
        allow_vert_in: bool,
    ) -> (bool, bool) {
        (self)(
            sb_local_x_8x8,
            sb_local_y_8x8,
            bsize,
            rd,
            allow_horz_in,
            allow_vert_in,
        )
    }
}

/// Allow naked closures as breakout pruners (so the encoder can plug an
/// inline `ml_predict_breakout` adapter without a named struct).
impl<F> BreakoutPruner for F
where
    F: FnMut(usize, usize, BlockSize, u32, u32) -> bool,
{
    fn breakout(
        &mut self,
        sb_local_x_8x8: usize,
        sb_local_y_8x8: usize,
        bsize: BlockSize,
        none_rate: u32,
        none_dist: u32,
    ) -> bool {
        (self)(sb_local_x_8x8, sb_local_y_8x8, bsize, none_rate, none_dist)
    }
}

/// Caller-supplied bounds + parameters for one descent invocation.
pub struct NonRdDescentParams<'a> {
    /// libvpx `x->min_partition_size` — the descent never recurses
    /// below this. Floored at `Block16x16` per F.3 HLD R3 (the
    /// recursion-depth cap; the picker bails to ZERO+skip at 8×8
    /// regardless).
    pub min_partition_size: BlockSize,
    /// libvpx `x->max_partition_size` — PARTITION_NONE is never
    /// evaluated at sizes larger than this.
    pub max_partition_size: BlockSize,
    /// Partition-tree wire cost table (libvpx `cpi->partition_cost`).
    /// Indexed `[ctx][PARTITION_NONE | _HORZ | _VERT | _SPLIT]`. Build
    /// with [`crate::cost::symbol_costs::partition_costs`].
    pub partition_costs: &'a PartitionCosts,
    /// Lambda from `compute_rd_mult(q_index, frame_type)`. Used in
    /// [`rd_cost`] for ranking NONE vs SPLIT.
    pub rd_mult: u32,
    /// Phase A (`RVP9_P8`) gate. When `true`, `Block8x8` becomes a valid
    /// PARTITION_NONE leaf candidate and `Block16x16` may PARTITION_SPLIT
    /// into four `Block8x8` leaves (8×8 is the no-further-split floor).
    /// When `false`, the descent floors at `Block16x16` exactly
    /// as before (8×8 leaves only ever come from the variance hint).
    ///
    /// The 8×8 leaves the descent emits are picked + reconstructed by the
    /// existing per-leaf loop (the picker already handles `Block8x8`);
    /// this flag only widens the partition-search candidate set.
    pub allow_8x8_leaf: bool,
}

/// One leaf of the descent's output — a sub-block within the SB.
///
/// Block8x8 leaves are inherited from the variance hint (the descent
/// doesn't search them per §4.3); their `pick` field is whatever the
/// caller seeds (typically `IMPOSSIBLE` — the existing lib.rs picker
/// handles 8×8 directly).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NonRdLeaf {
    /// Leaf row inside the SB, in 8×8 mi units (`0..=7`).
    pub mi_row_in_sb: usize,
    /// Leaf column inside the SB, in 8×8 mi units (`0..=7`).
    pub mi_col_in_sb: usize,
    /// Leaf block size (square, one of `Block8x8`, `Block16x16`,
    /// `Block32x32`, `Block64x64`).
    pub bsize: BlockSize,
    /// Winning candidate at this leaf (when the descent searched it).
    /// Block8x8 leaves inherited from the variance hint carry
    /// [`PartitionRdCandidate::IMPOSSIBLE`].
    pub pick: PartitionRdCandidate,
}

/// Aggregate RD diagnostics for the `Block16x16 -> 4x Block8x8`
/// SPLIT arm. This is intentionally narrow: Stage 4a's current VBR
/// partition gap is that 8x8 is allowed by diagnostics but still does not
/// win, so the useful question is whether the SPLIT arm is infeasible,
/// pruned, RD-more-expensive than 16x16 NONE, or later displaced by rect.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NonRdSplitStats {
    /// Number of `Block16x16` nodes where a split to 8x8 was a legal
    /// candidate and the descent started evaluating it.
    pub block16_attempts: u32,
    /// Candidate `Block16x16` nodes where the breakout hook suppressed
    /// SPLIT before any 8x8 child was evaluated.
    pub block16_breakout_suppressed: u32,
    /// SPLIT attempts where all four 8x8 children were feasible.
    pub block16_feasible: u32,
    /// SPLIT attempts aborted because a child was infeasible or the
    /// libvpx-style early-out proved the partial SPLIT sum already worse
    /// than the current best.
    pub block16_infeasible_or_early: u32,
    /// SPLIT attempts aborted because at least one 8x8 child was infeasible.
    pub block16_child_infeasible: u32,
    /// SPLIT attempts aborted by the libvpx-style partial-RD early-out.
    pub block16_partial_rd_early_out: u32,
    /// Feasible SPLIT-vs-NONE comparisons where SPLIT beat NONE.
    pub block16_split_beats_none: u32,
    /// Feasible SPLIT-vs-NONE comparisons where NONE stayed cheaper.
    pub block16_split_loses_to_none: u32,
    /// SPLIT candidates that survived final selection at this node.
    pub block16_split_committed: u32,
    /// SPLIT candidates that beat NONE, but were later displaced by HORZ/VERT.
    pub block16_rect_overrides_split: u32,
    /// Feasible SPLIT attempts where NONE was unavailable, so no margin can
    /// be computed.
    pub block16_no_none: u32,
    /// Sum of `split_rdcost - none_rdcost` over comparable attempts.
    pub block16_margin_sum: i64,
    /// Minimum `split_rdcost - none_rdcost` margin observed.
    pub block16_margin_min: Option<i64>,
    /// Maximum `split_rdcost - none_rdcost` margin observed.
    pub block16_margin_max: Option<i64>,
}

impl NonRdSplitStats {
    /// Add another SB's split stats into this accumulator.
    pub fn accumulate(&mut self, other: &Self) {
        self.block16_attempts += other.block16_attempts;
        self.block16_breakout_suppressed += other.block16_breakout_suppressed;
        self.block16_feasible += other.block16_feasible;
        self.block16_infeasible_or_early += other.block16_infeasible_or_early;
        self.block16_child_infeasible += other.block16_child_infeasible;
        self.block16_partial_rd_early_out += other.block16_partial_rd_early_out;
        self.block16_split_beats_none += other.block16_split_beats_none;
        self.block16_split_loses_to_none += other.block16_split_loses_to_none;
        self.block16_split_committed += other.block16_split_committed;
        self.block16_rect_overrides_split += other.block16_rect_overrides_split;
        self.block16_no_none += other.block16_no_none;
        self.block16_margin_sum = self
            .block16_margin_sum
            .saturating_add(other.block16_margin_sum);
        if let Some(min) = other.block16_margin_min {
            self.block16_margin_min = Some(self.block16_margin_min.map_or(min, |m| m.min(min)));
        }
        if let Some(max) = other.block16_margin_max {
            self.block16_margin_max = Some(self.block16_margin_max.map_or(max, |m| m.max(max)));
        }
    }

    fn record_block16_margin(&mut self, margin: i64) {
        self.block16_margin_sum = self.block16_margin_sum.saturating_add(margin);
        self.block16_margin_min = Some(self.block16_margin_min.map_or(margin, |m| m.min(margin)));
        self.block16_margin_max = Some(self.block16_margin_max.map_or(margin, |m| m.max(margin)));
        if margin < 0 {
            self.block16_split_beats_none += 1;
        } else {
            self.block16_split_loses_to_none += 1;
        }
    }
}

/// Result of one SB descent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonRdPartitionResult {
    /// Per-8×8-cell leaf-size grid (same shape as
    /// [`PartitionDecision::cells`]). Lets the caller cheaply look up
    /// "what size is this cell part of".
    pub cells: [[BlockSize; SB_GRID]; SB_GRID],
    /// Unique leaves in raster order (top-left cell only per leaf).
    pub leaves: Vec<NonRdLeaf>,
    /// Narrow diagnostics for 16x16 SPLIT-to-8x8 decisions.
    pub split_stats: NonRdSplitStats,
}

impl NonRdPartitionResult {
    /// Convert a [`PartitionDecision`] from the variance hint into a
    /// "all NONE candidates impossible" placeholder result. Used as
    /// the initial state before the descent runs.
    #[must_use]
    pub fn from_variance_hint(hint: &PartitionDecision) -> Self {
        let mut leaves = Vec::with_capacity(64);
        let mut visited = [[false; SB_GRID]; SB_GRID];
        for r in 0..SB_GRID {
            for c in 0..SB_GRID {
                if visited[r][c] {
                    continue;
                }
                let bs = hint.cells[r][c];
                let cells_wide = match bs {
                    BlockSize::Block8x8 => 1usize,
                    BlockSize::Block16x16 => 2,
                    BlockSize::Block32x32 => 4,
                    BlockSize::Block64x64 => 8,
                    _ => 1, // defensive — the hint never emits non-square sizes
                };
                for dr in 0..cells_wide {
                    for dc in 0..cells_wide {
                        if r + dr < SB_GRID && c + dc < SB_GRID {
                            visited[r + dr][c + dc] = true;
                        }
                    }
                }
                leaves.push(NonRdLeaf {
                    mi_row_in_sb: r,
                    mi_col_in_sb: c,
                    bsize: bs,
                    pick: PartitionRdCandidate::IMPOSSIBLE,
                });
            }
        }
        Self {
            cells: hint.cells,
            leaves,
            split_stats: NonRdSplitStats::default(),
        }
    }
}

/// Derive `(min_partition_size, max_partition_size)` from the variance
/// hint, with the F.3 HLD R3 floor that never lets `max` drop below
/// `Block16x16`.
///
/// The variance partitioner can emit all-Block8x8 SBs at high q
/// (foreman 85 % case). Capping max at 8×8 would defeat F.3 entirely;
/// the R3 mitigation lifts the floor to 16×16 so the descent always
/// has at least one larger size to evaluate.
#[must_use]
pub fn variance_hint_bounds(hint: &PartitionDecision) -> (BlockSize, BlockSize) {
    let mut min_idx = u8::MAX;
    let mut max_idx = 0u8;
    for r in 0..SB_GRID {
        for c in 0..SB_GRID {
            let bs = hint.cells[r][c];
            let idx = bs as u8;
            if idx < min_idx {
                min_idx = idx;
            }
            if idx > max_idx {
                max_idx = idx;
            }
        }
    }
    // Indices: Block8x8=3, Block16x16=6, Block32x32=9, Block64x64=12.
    let min_bs_raw = bs_from_idx(min_idx);
    let max_bs_raw = bs_from_idx(max_idx);
    // F.3 HLD R3: floor BOTH bounds at Block16x16. The variance hint
    // can collapse to all-8×8 at high q (foreman) which would defeat
    // F.3 (the descent caps at Block16x16 per §4.3, so a
    // `max_partition_size = Block8x8` would mean NONE is never
    // allowed at any valid descent size). Lifting both bounds to a
    // minimum of Block16x16 keeps the RD-vs-variance comparison
    // alive.
    let floor = BlockSize::Block16x16 as u8;
    let max_bs = if (max_bs_raw as u8) < floor {
        BlockSize::Block16x16
    } else {
        max_bs_raw
    };
    let min_bs = if (min_bs_raw as u8) < floor {
        BlockSize::Block16x16
    } else {
        min_bs_raw
    };
    // Defensive: never let `min > max` (can happen if max was raw and
    // min got floored above it).
    let min_bs = if (min_bs as u8) > (max_bs as u8) {
        max_bs
    } else {
        min_bs
    };
    (min_bs, max_bs)
}

fn bs_from_idx(idx: u8) -> BlockSize {
    match idx {
        12 => BlockSize::Block64x64,
        9 => BlockSize::Block32x32,
        6 => BlockSize::Block16x16,
        3 => BlockSize::Block8x8,
        // Defensive fallback — variance hint is constrained to square
        // sizes {8, 16, 32, 64}, but if we ever see something else
        // (e.g. a future Block4x4 leaf), treat it as Block8x8.
        _ => BlockSize::Block8x8,
    }
}

/// The child block size produced by `PARTITION_SPLIT`. Mirrors libvpx
/// `get_subsize(bsize, PARTITION_SPLIT)` (`vp9_common_data.c`) — for
/// square sizes this just halves the side.
fn subsize_for_split(bsize: BlockSize) -> Option<BlockSize> {
    match bsize {
        BlockSize::Block64x64 => Some(BlockSize::Block32x32),
        BlockSize::Block32x32 => Some(BlockSize::Block16x16),
        BlockSize::Block16x16 => Some(BlockSize::Block8x8),
        _ => None,
    }
}

/// SB-internal stride (in 8×8 cells) between split children at
/// `bsize`.
fn split_step_8x8(bsize: BlockSize) -> usize {
    match bsize {
        BlockSize::Block64x64 => 4,
        BlockSize::Block32x32 => 2,
        BlockSize::Block16x16 => 1,
        _ => 1,
    }
}

/// The child block size produced by `PARTITION_HORZ` (top + bottom, each
/// `w × h/2`). Mirrors libvpx `get_subsize(bsize, PARTITION_HORZ)`
/// (`vp9_common_data.c` `subsize_lookup`). Only the square 64/32/16
/// parents the descent walks are listed — others return `None`.
fn subsize_for_horz(bsize: BlockSize) -> Option<BlockSize> {
    match bsize {
        BlockSize::Block64x64 => Some(BlockSize::Block64x32),
        BlockSize::Block32x32 => Some(BlockSize::Block32x16),
        BlockSize::Block16x16 => Some(BlockSize::Block16x8),
        _ => None,
    }
}

/// The child block size produced by `PARTITION_VERT` (left + right, each
/// `w/2 × h`). Mirrors libvpx `get_subsize(bsize, PARTITION_VERT)`.
fn subsize_for_vert(bsize: BlockSize) -> Option<BlockSize> {
    match bsize {
        BlockSize::Block64x64 => Some(BlockSize::Block32x64),
        BlockSize::Block32x32 => Some(BlockSize::Block16x32),
        BlockSize::Block16x16 => Some(BlockSize::Block8x16),
        _ => None,
    }
}

/// SB-internal step (in 8×8 cells) between the two rect halves at
/// `bsize`. For HORZ this is the row offset to the bottom half; for VERT
/// the column offset to the right half. Equal to `hbs` (half the parent
/// width in 8×8 cells).
fn rect_half_step_8x8(bsize: BlockSize) -> usize {
    match bsize {
        BlockSize::Block64x64 => 4,
        BlockSize::Block32x32 => 2,
        BlockSize::Block16x16 => 1,
        _ => 1,
    }
}

/// Run the F.3 motion-aware partition descent for one 64×64 SB.
///
/// The descent walks the SB top-down (64→32→16). At each `bsize` it
/// evaluates PARTITION_NONE via the caller-supplied mode picker, then
/// — if `bsize > min_partition_size` — recurses into 4 children and
/// compares the SPLIT sum against the NONE result.
///
/// The variance hint shapes the descent through `params.{min,max}_partition_size`
/// (libvpx's `x->min_partition_size` / `x->max_partition_size`). The
/// hint's cell-grid is fed in as `hint` so the descent can fall back
/// to the hint's choice for any sub-block where NONE is infeasible
/// (e.g. partial-edge SBs in non-multiple-of-64 frames — defensive,
/// not hit on standard SB grids).
///
/// The descent inherits the picker's Block8x8 bail (per HLD §4.3): if
/// the recursion would step below `Block16x16`, the variance hint's
/// leaves at that position are kept as-is.
#[must_use]
pub fn nonrd_pick_partition<P: ModePickerFn>(
    hint: &PartitionDecision,
    params: &NonRdDescentParams<'_>,
    picker: &mut P,
) -> NonRdPartitionResult {
    // No breakout pruning (pre-A21 behaviour). Equivalent to libvpx with
    // `sf.rd_ml_partition.search_breakout = 0`.
    nonrd_pick_partition_with_breakout(hint, params, picker, &mut NoBreakout)
}

/// As [`nonrd_pick_partition`], but with an A21 ML partition-search
/// breakout hook (libvpx `ml_predict_breakout` gate,
/// `vp9/encoder/vp9_encodeframe.c:4161-4168`).
///
/// The `breakout` pruner is consulted after a NONE candidate wins the
/// current best at a block; a `true` vote suppresses that block's SPLIT
/// recursion. Pass [`NoBreakout`] to disable (identical to
/// [`nonrd_pick_partition`]).
#[must_use]
pub fn nonrd_pick_partition_with_breakout<P: ModePickerFn, B: BreakoutPruner>(
    hint: &PartitionDecision,
    params: &NonRdDescentParams<'_>,
    picker: &mut P,
    breakout: &mut B,
) -> NonRdPartitionResult {
    // No rect pruning (pre-A21-rect behaviour). Equivalent to libvpx with
    // `sf.rd_ml_partition.prune_rect_thresh[*] < 0`.
    nonrd_pick_partition_with_pruners(hint, params, picker, breakout, &mut NoRectPrune)
}

/// As [`nonrd_pick_partition_with_breakout`], but ALSO with an A21-rect
/// ML rectangular-partition pruning hook (libvpx `ml_prune_rect_partition`,
/// `vp9/encoder/vp9_encodeframe.c:3334-3425`; gate at `:4332-4339`).
///
/// The `rect_prune` hook is consulted after NONE + SPLIT are evaluated
/// at a rect-eligible block; its `(allow_horz, allow_vert)` verdict
/// gates whether each rect arm's picker invocations run at all. Pass
/// [`NoRectPrune`] to disable (identical to
/// [`nonrd_pick_partition_with_breakout`]).
#[must_use]
pub fn nonrd_pick_partition_with_pruners<P: ModePickerFn, B: BreakoutPruner, R: RectPruner>(
    hint: &PartitionDecision,
    params: &NonRdDescentParams<'_>,
    picker: &mut P,
    breakout: &mut B,
    rect_prune: &mut R,
) -> NonRdPartitionResult {
    let mut state = DescentState {
        cells: [[BlockSize::Block8x8; SB_GRID]; SB_GRID],
        leaves: Vec::with_capacity(16),
        split_stats: NonRdSplitStats::default(),
    };
    // Seed `above`/`left` segment-context bits to 0. libvpx maintains
    // these across SBs (`vp9_encodeframe.c:4922` reads
    // `xd->above_seg_context` + `xd->left_seg_context`); for a
    // one-shot per-SB descent the worst case is mis-pricing
    // `partition_cost` by a few `vp9_prob_cost` units (1/512-bit)
    // per partition decision — well below the SSE-dominated `dist
    // << 7` term in `rd_cost` (each unit of `dist` contributes 128
    // to the RD sum, vs ≤ ~600 units total swing in
    // partition_cost across all 4 contexts).
    //
    // **F.4 activation note**: when the env gate flips ON, this
    // approximation becomes potentially load-bearing on
    // partition-cost-tied decisions. F.4 should either thread real
    // above/left contexts through the descent (mirroring libvpx's
    // `partition_plane_context` callsite at
    // `vp9_encodeframe.c:5009`) or empirically verify the
    // approximation against the F.0.2 oracle's partition-match%
    // metric. The infrastructure-only baseline accepts the
    // approximation as documented.
    let ctx_above = false;
    let ctx_left = false;
    let _result = descend(
        hint,
        params,
        picker,
        breakout,
        rect_prune,
        (0, 0),
        BlockSize::Block64x64,
        ctx_above,
        ctx_left,
        &mut state,
    );
    NonRdPartitionResult {
        cells: state.cells,
        leaves: state.leaves,
        split_stats: state.split_stats,
    }
}

struct DescentState {
    cells: [[BlockSize; SB_GRID]; SB_GRID],
    leaves: Vec<NonRdLeaf>,
    split_stats: NonRdSplitStats,
}

/// Recursive worker. Returns the `(rate, dist)` chosen at this
/// sub-block (which feeds the parent SPLIT sum) plus the best RD cost
/// — or `None` if no candidate was feasible (the parent treats this
/// child as "infeasible" and falls back to the variance hint).
#[allow(clippy::too_many_arguments)] // Mirrors libvpx
// `nonrd_pick_partition`'s recursive signature; each argument is
// load-bearing (hint, bounds, picker, origin, bsize, above/left ctx,
// state accumulator). Refactoring to a struct would just hide the
// stack frame.
fn descend<P: ModePickerFn, B: BreakoutPruner, R: RectPruner>(
    hint: &PartitionDecision,
    params: &NonRdDescentParams<'_>,
    picker: &mut P,
    breakout: &mut B,
    rect_prune: &mut R,
    origin_8x8: (usize, usize),
    bsize: BlockSize,
    above: bool,
    left: bool,
    state: &mut DescentState,
) -> Option<DescendResult> {
    let (row_8x8, col_8x8) = origin_8x8;

    // §4.3 cap: never descend below the floor. `Block8x8` is always the
    // no-further-split floor — even with Phase A on, an 8×8 leaf is a
    // terminal NONE (it never recurses into 4×4). When Phase A is OFF the
    // floor also includes `min_partition_size` (which the variance hint
    // keeps at `Block16x16`); with Phase A ON the call site lowers
    // `min_partition_size` to `Block8x8`, so `Block16x16` is no longer at
    // the floor and may PARTITION_SPLIT into four 8×8 children.
    let bsize_at_floor =
        (bsize as u8) <= (params.min_partition_size as u8) || matches!(bsize, BlockSize::Block8x8);

    // PARTITION_NONE allowed? Phase A (`allow_8x8_leaf`) lets `Block8x8`
    // be a NONE candidate; otherwise it is excluded (pre-Phase-A
    // behaviour: 8×8 leaves only come from the variance hint).
    let none_allowed = (bsize as u8) <= (params.max_partition_size as u8)
        && (bsize as u8) >= (params.min_partition_size as u8)
        && (params.allow_8x8_leaf || !matches!(bsize, BlockSize::Block8x8));

    let part_ctx = partition_plane_context(above, left, bsize);
    let none_cost = params.partition_costs[part_ctx][PARTITION_NONE_IDX].max(0) as u32;
    let split_cost = params.partition_costs[part_ctx][PARTITION_SPLIT_IDX].max(0) as u32;

    // Try PARTITION_NONE.
    let mut best_pick: Option<PartitionRdCandidate> = None;
    let mut best_rd_cost: i64 = i64::MAX;
    // A21-rect — the PARTITION_NONE arm's own RD cost (libvpx
    // `pc_tree->none.rdcost`), captured separately from `best_rd_cost`
    // so the rect pruner gets the NONE arm even after SPLIT/rect lower
    // the running best. `None` when NONE was infeasible.
    let mut none_rd_cost: Option<i64> = None;
    // A21 — `do_split` gate (libvpx `vp9_encodeframe.c:4004`,
    // `int do_split = bsize >= BLOCK_8X8`). The ML breakout pruner can
    // force this to `false` after a winning NONE pick, suppressing the
    // SPLIT recursion at this block.
    let mut do_split = true;
    if let Some(cand) = none_allowed
        .then(|| picker.pick(col_8x8, row_8x8, bsize))
        .flatten()
    {
        // Add partition_cost[NONE] to rate (libvpx :5010).
        let rate_total = cand.rate.saturating_add(none_cost);
        let rdc = rd_cost(params.rd_mult, rate_total, u64::from(cand.dist));
        none_rd_cost = Some(rdc);
        if rdc < best_rd_cost {
            best_rd_cost = rdc;
            best_pick = Some(PartitionRdCandidate {
                rate: rate_total,
                dist: cand.dist,
                mv: cand.mv,
                mode: cand.mode,
                ref_frame: cand.ref_frame,
            });
            // A21 — ML partition-search breakout. libvpx consults
            // `ml_predict_breakout` inside the `this_rdc.rdcost <
            // best_rdc.rdcost` arm (`vp9_encodeframe.c:4140-4168`):
            // when it votes to stop, `do_split = 0; do_rect = 0`. We
            // pass libvpx's `this_rdc.rate` (rate WITH the NONE
            // partition cost folded in, matching the `:4136`
            // `this_rdc.rate += partition_cost[NONE]` that precedes the
            // breakout call at `:4165`) and `this_rdc.dist`.
            if breakout.breakout(col_8x8, row_8x8, bsize, rate_total, cand.dist) {
                do_split = false;
            }
        }
    }

    // Try PARTITION_SPLIT — only if we have a child size to recurse
    // into AND we haven't hit the floor AND the breakout pruner didn't
    // veto the split.
    let mut split_pick: Option<(u32, u32)> = None; // (sum_rate, sum_dist)
    let mut split_children: [Option<DescendResult>; 4] = [None, None, None, None];
    let track_block16_split = matches!(bsize, BlockSize::Block16x16) && !bsize_at_floor;
    if track_block16_split && !do_split {
        state.split_stats.block16_breakout_suppressed += 1;
    }
    if let Some(child_bsize) = (do_split && !bsize_at_floor)
        .then(|| subsize_for_split(bsize))
        .flatten()
    {
        if track_block16_split {
            state.split_stats.block16_attempts += 1;
        }
        let step = split_step_8x8(bsize);
        let mut sum_rate: u32 = split_cost;
        let mut sum_dist: u32 = 0;
        let mut all_feasible = true;
        let mut child_infeasible = false;
        let mut partial_rd_early_out = false;
        // Per-quadrant `above`/`left` propagation. For first-cut
        // F.3 infrastructure we keep the simplified ctx flags
        // constant across quadrants — refinement is F.4 territory
        // when the descent actually drives behaviour.
        for (i, child) in split_children.iter_mut().enumerate() {
            let dr = (i / 2) * step;
            let dc = (i % 2) * step;
            let child_origin = (row_8x8 + dr, col_8x8 + dc);
            // SB-edge guard: a non-square frame could leave us with
            // a child that walks past the SB grid. On standard
            // 64-aligned input this never fires.
            if child_origin.0 >= SB_GRID || child_origin.1 >= SB_GRID {
                child_infeasible = true;
                all_feasible = false;
                break;
            }
            match descend(
                hint,
                params,
                picker,
                breakout,
                rect_prune,
                child_origin,
                child_bsize,
                above,
                left,
                state,
            ) {
                Some(r) => {
                    sum_rate = sum_rate.saturating_add(r.rate);
                    sum_dist = sum_dist.saturating_add(r.dist);
                    *child = Some(r);
                    // libvpx :5042 early-out — if SPLIT already
                    // worse than the current best, stop walking
                    // siblings.
                    let rdc_so_far = rd_cost(params.rd_mult, sum_rate, u64::from(sum_dist));
                    if rdc_so_far >= best_rd_cost && best_pick.is_some() {
                        // Mark remaining quadrants infeasible (so
                        // we don't try to commit them).
                        partial_rd_early_out = true;
                        all_feasible = false;
                        break;
                    }
                }
                None => {
                    // Child infeasible → SPLIT can't win at this
                    // level. Fall back to NONE if we have it.
                    child_infeasible = true;
                    all_feasible = false;
                    break;
                }
            }
        }
        if all_feasible {
            if track_block16_split {
                state.split_stats.block16_feasible += 1;
            }
            let rdc_split = rd_cost(params.rd_mult, sum_rate, u64::from(sum_dist));
            if track_block16_split {
                if let Some(none_rdc) = none_rd_cost {
                    state
                        .split_stats
                        .record_block16_margin(rdc_split - none_rdc);
                } else {
                    state.split_stats.block16_no_none += 1;
                }
            }
            if rdc_split < best_rd_cost {
                best_rd_cost = rdc_split;
                split_pick = Some((sum_rate, sum_dist));
            }
        } else if track_block16_split {
            state.split_stats.block16_infeasible_or_early += 1;
            if partial_rd_early_out {
                state.split_stats.block16_partial_rd_early_out += 1;
            } else if child_infeasible {
                state.split_stats.block16_child_infeasible += 1;
            }
        }
    }

    // A4-rect — PARTITION_HORZ / PARTITION_VERT.
    //
    // libvpx `rd_pick_partition` (`vp9_encodeframe.c:4342-4436`) codes
    // each rect as TWO sub-blocks at the rect subsize (NOT recursed):
    // HORZ = top+bottom (`w × h/2` each), VERT = left+right
    // (`w/2 × h` each). It sums the two sub-block RDs + the rect
    // partition_cost and keeps the result if it beats the running best.
    //
    // `partition_horz_allowed` / `partition_vert_allowed`
    // (`vp9_encodeframe.c:4018-4021`): both require `bsize >= BLOCK_8X8`
    // and (for 4:2:0, `yss == xss == 1`) the `yss <= xss` / `xss <= yss`
    // chroma-subsampling gates BOTH hold. The `force_{horz,vert}_split`
    // frame-edge overrides are not modelled here — the descent only runs
    // on 64-aligned SBs (the picker callback's SB-edge guard returns
    // `None` for any rect half that would walk past the frame, which the
    // sum below treats as infeasible, falling back to NONE/SPLIT).
    //
    // Rect is evaluated even when the breakout pruner vetoed SPLIT
    // (`do_split = false` also sets libvpx's `do_rect = 0`), to keep the
    // A21 semantics: when breakout fires, skip rect too.
    let mut rect_pick: Option<(PartitionType, PartitionRdCandidate, PartitionRdCandidate)> = None;
    // Rect is gated like PARTITION_NONE on the upper bound: never search
    // a rect at a parent LARGER than `max_partition_size`. The variance
    // hint already excluded that parent size, so its rect halves are not
    // worth ranking (and searching them would invoke the picker at sizes
    // the hint ruled out). The `bsize >= BLOCK_8X8` half of libvpx's
    // `partition_{horz,vert}_allowed` (`vp9_encodeframe.c:4019/4021`) is
    // covered by `!bsize_at_floor` (the descent floors at Block16x16).
    let rect_within_max = (bsize as u8) <= (params.max_partition_size as u8);
    if do_split && !bsize_at_floor && rect_within_max {
        // A21-rect — consult the ML rect-prune model (libvpx
        // `ml_prune_rect_partition`, `vp9_encodeframe.c:3334-3425`;
        // gate at `:4332-4339`). It runs AFTER NONE + SPLIT and ANDs
        // its verdict into the per-arm allow masks. libvpx's
        // `do_ml_rect_partition_pruning` gate requires both arms not
        // already force-disabled and `bsize > BLOCK_8X8` — here both
        // arms start `true` and `!bsize_at_floor` covers the size
        // floor. `best_rdc.rdcost` (= our running `best_rd_cost`, which
        // already absorbed NONE + SPLIT) is the `ref_rd` feature.
        //
        // `pc_tree->partitioning == PARTITION_NONE`: NONE is the
        // current winner iff it produced a pick AND SPLIT did not lower
        // the running best past it (`split_pick.is_none()`).
        let partitioning_is_none = best_pick.is_some() && split_pick.is_none();
        let rd_ctx = RectPruneRdContext {
            best_rd_cost: (best_rd_cost != i64::MAX).then_some(best_rd_cost),
            none_rd_cost,
            split_none_rd_cost: [
                split_children[0].and_then(|r| r.none_rd_cost),
                split_children[1].and_then(|r| r.none_rd_cost),
                split_children[2].and_then(|r| r.none_rd_cost),
                split_children[3].and_then(|r| r.none_rd_cost),
            ],
            partitioning_is_none,
        };
        let (allow_horz, allow_vert) =
            rect_prune.prune_rect(col_8x8, row_8x8, bsize, rd_ctx, true, true);

        // HORZ then VERT, in libvpx order. Each arm is gated by the
        // rect pruner's verdict (`allow_horz` / `allow_vert`); a pruned
        // arm skips ALL its picker invocations (libvpx's
        // `partition_horz_allowed` / `partition_vert_allowed` short-
        // circuit at `:4343` / `:4393`).
        let horz_step = rect_half_step_8x8(bsize);
        if allow_horz && let Some(sub) = subsize_for_horz(bsize) {
            let horz_cost = params.partition_costs[part_ctx][PARTITION_HORZ_IDX].max(0) as u32;
            // Top half at (row, col); bottom half at (row + hbs, col).
            if let (Some(top), Some(bot)) = (
                feasible_rect_pick(picker, col_8x8, row_8x8, sub),
                (row_8x8 + horz_step < SB_GRID)
                    .then(|| feasible_rect_pick(picker, col_8x8, row_8x8 + horz_step, sub))
                    .flatten(),
            ) {
                let sum_rate = top.rate.saturating_add(bot.rate).saturating_add(horz_cost);
                let sum_dist = top.dist.saturating_add(bot.dist);
                let rdc = rd_cost(params.rd_mult, sum_rate, u64::from(sum_dist));
                if rdc < best_rd_cost {
                    best_rd_cost = rdc;
                    rect_pick = Some((PartitionType::Horz, top, bot));
                }
            }
        }
        let vert_step = rect_half_step_8x8(bsize);
        if allow_vert && let Some(sub) = subsize_for_vert(bsize) {
            let vert_cost = params.partition_costs[part_ctx][PARTITION_VERT_IDX].max(0) as u32;
            // Left half at (row, col); right half at (row, col + hbs).
            if let (Some(left_pick), Some(right_pick)) = (
                feasible_rect_pick(picker, col_8x8, row_8x8, sub),
                (col_8x8 + vert_step < SB_GRID)
                    .then(|| feasible_rect_pick(picker, col_8x8 + vert_step, row_8x8, sub))
                    .flatten(),
            ) {
                let sum_rate = left_pick
                    .rate
                    .saturating_add(right_pick.rate)
                    .saturating_add(vert_cost);
                let sum_dist = left_pick.dist.saturating_add(right_pick.dist);
                let rdc = rd_cost(params.rd_mult, sum_rate, u64::from(sum_dist));
                if rdc < best_rd_cost {
                    best_rd_cost = rdc;
                    rect_pick = Some((PartitionType::Vert, left_pick, right_pick));
                }
            }
        }
    }

    // Commit the winner into the cell grid + leaves. Precedence follows
    // the running `best_rd_cost`: whichever candidate last lowered it
    // wins. NONE / SPLIT / HORZ / VERT are evaluated in that order and
    // each only records its pick when it strictly beats the best so far.
    // Rect is checked first because it can lower the running best after
    // SPLIT; otherwise SPLIT wins; otherwise NONE wins.
    if let Some((part, a, b)) = rect_pick {
        if track_block16_split && split_pick.is_some() {
            state.split_stats.block16_rect_overrides_split += 1;
        }
        // Rect wins — roll back any speculative SPLIT children and
        // commit the two rect leaves.
        let step = rect_half_step_8x8(bsize);
        let (sub, a_origin, b_origin) = match part {
            PartitionType::Horz => (
                subsize_for_horz(bsize).expect("horz subsize exists"),
                (row_8x8, col_8x8),
                (row_8x8 + step, col_8x8),
            ),
            PartitionType::Vert => (
                subsize_for_vert(bsize).expect("vert subsize exists"),
                (row_8x8, col_8x8),
                (row_8x8, col_8x8 + step),
            ),
            _ => unreachable!("rect_pick only carries Horz/Vert"),
        };
        let rect_cost = match part {
            PartitionType::Horz => params.partition_costs[part_ctx][PARTITION_HORZ_IDX],
            _ => params.partition_costs[part_ctx][PARTITION_VERT_IDX],
        }
        .max(0) as u32;
        commit_rect_leaves(
            state, row_8x8, col_8x8, bsize, sub, a_origin, a, b_origin, b,
        );
        let sum_rate = a.rate.saturating_add(b.rate).saturating_add(rect_cost);
        let sum_dist = a.dist.saturating_add(b.dist);
        return Some(DescendResult {
            rate: sum_rate,
            dist: sum_dist,
            rd_cost: best_rd_cost,
            none_rd_cost,
        });
    }
    if let Some((rate, dist)) = split_pick {
        if track_block16_split {
            state.split_stats.block16_split_committed += 1;
        }
        // SPLIT wins — commit children's leaves (already pushed during
        // recursion via their NONE branch). Nothing further to do at
        // this level beyond updating the parent's accounting.
        return Some(DescendResult {
            rate,
            dist,
            rd_cost: best_rd_cost,
            none_rd_cost,
        });
    }
    if let Some(pick) = best_pick {
        // NONE wins (or SPLIT/rect weren't feasible) — commit this block
        // as a single leaf, overwriting any children we tentatively
        // committed during recursion. Recursion only commits children
        // when their *own* NONE wins, so the worst case is a few
        // dropped leaves; we rebuild this sub-block cleanly here.
        commit_leaf(state, row_8x8, col_8x8, bsize, pick);
        return Some(DescendResult {
            rate: pick.rate,
            dist: pick.dist,
            rd_cost: best_rd_cost,
            none_rd_cost,
        });
    }

    // Neither NONE nor SPLIT nor rect was feasible at this level — fall
    // back to the variance hint. The hint's leaves at this position
    // become our leaves; the descent contributes no RD signal here
    // (parent's SPLIT sum will mark this as infeasible and the parent's
    // NONE wins by default).
    inherit_hint(hint, state, row_8x8, col_8x8, bsize);
    None
}

/// Evaluate the picker for one rect half at `(col_8x8, row_8x8)` with
/// rect size `sub`. Returns `None` when the picker reports the candidate
/// infeasible (e.g. the rect half walks past a frame edge). The rate /
/// dist are returned WITHOUT the rect partition cost folded in — the
/// caller sums the two halves then adds the single rect partition cost
/// once (matching libvpx `vp9_encodeframe.c:4345` / `:4355`, where
/// `part_mode_rate` is added once to the summed sub-block RD).
fn feasible_rect_pick<P: ModePickerFn>(
    picker: &mut P,
    col_8x8: usize,
    row_8x8: usize,
    sub: BlockSize,
) -> Option<PartitionRdCandidate> {
    picker.pick(col_8x8, row_8x8, sub)
}

/// Commit two rect sub-block leaves (HORZ top+bottom, or VERT
/// left+right) at rect size `sub`, rolling back any speculative SPLIT
/// children in the parent's region first (same invariant as
/// [`commit_leaf`]). The parent occupies the `bsize` span starting at
/// `(parent_row, parent_col)`; the two halves are at `a_origin` /
/// `b_origin` (in 8×8 mi units within the SB).
#[allow(clippy::too_many_arguments)]
fn commit_rect_leaves(
    state: &mut DescentState,
    parent_row: usize,
    parent_col: usize,
    bsize: BlockSize,
    sub: BlockSize,
    a_origin: (usize, usize),
    a_pick: PartitionRdCandidate,
    b_origin: (usize, usize),
    b_pick: PartitionRdCandidate,
) {
    let parent_span = cells_wide_for(bsize);
    // Drop any leaves tentatively committed inside the parent's region
    // during a now-losing SPLIT branch.
    state.leaves.retain(|leaf| {
        let lr = leaf.mi_row_in_sb;
        let lc = leaf.mi_col_in_sb;
        !(lr >= parent_row
            && lr < parent_row + parent_span
            && lc >= parent_col
            && lc < parent_col + parent_span)
    });
    // Splat each rect half's cells + push its leaf.
    let (sub_w, sub_h) = (sub.n_8x8_wide() as usize, sub.n_8x8_high() as usize);
    for &(origin, pick) in &[(a_origin, a_pick), (b_origin, b_pick)] {
        let (r0, c0) = origin;
        for dr in 0..sub_h {
            for dc in 0..sub_w {
                if r0 + dr < SB_GRID && c0 + dc < SB_GRID {
                    state.cells[r0 + dr][c0 + dc] = sub;
                }
            }
        }
        state.leaves.push(NonRdLeaf {
            mi_row_in_sb: r0,
            mi_col_in_sb: c0,
            bsize: sub,
            pick,
        });
    }
}

/// Result returned by [`descend`] for a sub-block — the totals the
/// parent needs to accumulate into its SPLIT sum.
#[derive(Copy, Clone, Debug)]
struct DescendResult {
    rate: u32,
    dist: u32,
    #[allow(dead_code)] // useful for diagnostics; tests don't read it
    rd_cost: i64,
    /// A21-rect — this sub-block's PARTITION_NONE-arm RD cost (libvpx
    /// `pc_tree->none.rdcost`). Surfaced so a parent's rect pruner can
    /// read each SPLIT child's NONE arm. `None` when NONE was
    /// infeasible at this sub-block.
    none_rd_cost: Option<i64>,
}

/// Commit a single leaf at `(row_8x8, col_8x8)` with size `bsize` and
/// the given `pick`, rolling back any descendant leaves that were
/// speculatively pushed during a now-losing SPLIT branch.
///
/// **Invariant**: when this parent's NONE wins, all four
/// `descend(child)` calls in the SPLIT branch have already run.
/// Whenever a child's own NONE won, that child pushed a leaf onto
/// `state.leaves` and overwrote cells in `state.cells`. Both must be
/// reverted before the parent's NONE commit, or the leaf list will
/// double-cover this region. The `retain` drops the speculative
/// leaves; the cell-splat above overwrites the speculative cells.
fn commit_leaf(
    state: &mut DescentState,
    row_8x8: usize,
    col_8x8: usize,
    bsize: BlockSize,
    pick: PartitionRdCandidate,
) {
    let span = cells_wide_for(bsize);
    // Splat the bsize into the cell grid — overwrites any
    // speculative-child cells from the losing SPLIT branch.
    for dr in 0..span {
        for dc in 0..span {
            if row_8x8 + dr < SB_GRID && col_8x8 + dc < SB_GRID {
                state.cells[row_8x8 + dr][col_8x8 + dc] = bsize;
            }
        }
    }
    // Drop any leaves we tentatively committed inside this region
    // during the SPLIT recursion that lost the RD comparison (see
    // invariant above).
    state.leaves.retain(|leaf| {
        let lr = leaf.mi_row_in_sb;
        let lc = leaf.mi_col_in_sb;
        !(lr >= row_8x8 && lr < row_8x8 + span && lc >= col_8x8 && lc < col_8x8 + span)
    });
    state.leaves.push(NonRdLeaf {
        mi_row_in_sb: row_8x8,
        mi_col_in_sb: col_8x8,
        bsize,
        pick,
    });
}

fn inherit_hint(
    hint: &PartitionDecision,
    state: &mut DescentState,
    row_8x8: usize,
    col_8x8: usize,
    bsize: BlockSize,
) {
    let span = cells_wide_for(bsize);
    // Drop any speculative leaves in this region.
    state.leaves.retain(|leaf| {
        let lr = leaf.mi_row_in_sb;
        let lc = leaf.mi_col_in_sb;
        !(lr >= row_8x8 && lr < row_8x8 + span && lc >= col_8x8 && lc < col_8x8 + span)
    });
    // Walk the hint's cell grid within the region and emit leaves
    // (top-left cell per unique leaf).
    let mut visited = [[false; SB_GRID]; SB_GRID];
    for r in row_8x8..row_8x8.saturating_add(span).min(SB_GRID) {
        for c in col_8x8..col_8x8.saturating_add(span).min(SB_GRID) {
            if visited[r][c] {
                continue;
            }
            let bs = hint.cells[r][c];
            let s = cells_wide_for(bs);
            for dr in 0..s {
                for dc in 0..s {
                    if r + dr < SB_GRID && c + dc < SB_GRID {
                        visited[r + dr][c + dc] = true;
                        state.cells[r + dr][c + dc] = bs;
                    }
                }
            }
            state.leaves.push(NonRdLeaf {
                mi_row_in_sb: r,
                mi_col_in_sb: c,
                bsize: bs,
                pick: PartitionRdCandidate::IMPOSSIBLE,
            });
        }
    }
}

fn cells_wide_for(bsize: BlockSize) -> usize {
    match bsize {
        BlockSize::Block64x64 => 8,
        BlockSize::Block32x32 => 4,
        BlockSize::Block16x16 => 2,
        BlockSize::Block8x8 => 1,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::symbol_costs::partition_costs;
    use crate::entropy::frame_context::FrameContext;

    /// Build a [`PartitionDecision`] with every 8×8 cell set to `bs`
    /// (and `bs` self-consistent — i.e. it tiles 64×64 cleanly).
    fn uniform_hint(bs: BlockSize) -> PartitionDecision {
        PartitionDecision {
            cells: [[bs; SB_GRID]; SB_GRID],
        }
    }

    /// Make a `NonRdDescentParams` with realistic-ish defaults for tests.
    fn make_params<'a>(
        partition_costs_ref: &'a PartitionCosts,
        min_bs: BlockSize,
        max_bs: BlockSize,
    ) -> NonRdDescentParams<'a> {
        NonRdDescentParams {
            min_partition_size: min_bs,
            max_partition_size: max_bs,
            partition_costs: partition_costs_ref,
            // Realistic-ish lambda from `compute_rd_mult(q=128, INTER)`.
            // Tests don't depend on the exact value; the rd_mult just
            // needs to be positive so the rate term participates.
            rd_mult: 200,
            // Default OFF — existing descent tests assert pre-Phase-A
            // behaviour (8×8 only from the variance hint).
            allow_8x8_leaf: false,
        }
    }

    /// **Test 1 — Flat block picks Block64x64.**
    /// Synthetic picker returns very low RD for `Block64x64 NONE`
    /// (cheap motion-comp fit) and progressively higher RD for smaller
    /// sizes. F.3 must emit `PARTITION_NONE` at 64×64.
    #[test]
    fn flat_block_picks_64x64() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block8x8); // worst-case hint
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker: NONE at any bsize returns rate=10, dist=1 (very
        // low) — the descent's SPLIT sum at 64×64 would be 4×(10,1)
        // = (40, 4) + split_cost; NONE total ≈ rate (10 + none_cost).
        // At lambda=200 the rate component is comparable across
        // levels, but `dist << 7` dominates, and 4 children's combined
        // dist is 4× the parent → NONE wins.
        let mut picker = |_x: usize, _y: usize, _bs: BlockSize| {
            Some(PartitionRdCandidate {
                rate: 10,
                dist: 1,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        assert_eq!(result.leaves.len(), 1, "expected one Block64x64 leaf");
        assert_eq!(result.leaves[0].bsize, BlockSize::Block64x64);
        assert_eq!(result.leaves[0].mi_row_in_sb, 0);
        assert_eq!(result.leaves[0].mi_col_in_sb, 0);
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block64x64);
            }
        }
    }

    /// **Test 2 — Translating block picks 32×32 (headline test).**
    /// Picker reports the lowest RD for a 32×32 region at the
    /// top-left quadrant of the SB (the "moving" block with a good
    /// NEWMV fit), modest RD for the other three 32×32 quadrants
    /// (static content reachable via ZEROMV), and high RD for the
    /// 64×64 NONE candidate (because a single MV can't cover both
    /// the moving + static halves). F.3 must split at 64×64 → 4
    /// Block32x32 leaves.
    #[test]
    fn translating_block_picks_32x32() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block8x8);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |x: usize, y: usize, bs: BlockSize| {
            let in_top_left = x < 4 && y < 4;
            match bs {
                BlockSize::Block64x64 => Some(PartitionRdCandidate {
                    // High dist — one MV can't fit the whole 64×64.
                    rate: 100,
                    dist: 100_000,
                    mv: Mv::ZERO,
                    mode: 0,
                    ref_frame: 0,
                }),
                BlockSize::Block32x32 => Some(PartitionRdCandidate {
                    rate: if in_top_left { 50 } else { 30 },
                    dist: if in_top_left { 100 } else { 500 },
                    mv: if in_top_left {
                        Mv { col: 16, row: 0 }
                    } else {
                        Mv::ZERO
                    },
                    mode: if in_top_left { 3 } else { 0 },
                    ref_frame: 0,
                }),
                BlockSize::Block16x16 => Some(PartitionRdCandidate {
                    // Splitting further is more expensive than 32×32.
                    rate: 80,
                    dist: 1_000,
                    mv: Mv::ZERO,
                    mode: 0,
                    ref_frame: 0,
                }),
                _ => None,
            }
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // Top-left quadrant must be a single Block32x32 leaf.
        let has_32x32_top_left = result.leaves.iter().any(|leaf| {
            leaf.mi_row_in_sb == 0
                && leaf.mi_col_in_sb == 0
                && matches!(leaf.bsize, BlockSize::Block32x32)
        });
        assert!(
            has_32x32_top_left,
            "expected Block32x32 leaf at top-left of SB, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        // Check the cell grid: top-left 4×4 cells must all be Block32x32.
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    result.cells[r][c],
                    BlockSize::Block32x32,
                    "cell ({},{}) should be Block32x32, got {:?}",
                    r,
                    c,
                    result.cells[r][c],
                );
            }
        }
    }

    /// **Test 3 — Recursion-depth cap holds.**
    /// Even with a picker that wants to split everywhere, the descent
    /// MUST NOT emit Block8x8 leaves of its own — those only come
    /// from the variance hint. With `min_partition_size = Block16x16`,
    /// the smallest descent-emitted leaf is Block16x16.
    #[test]
    fn recursion_caps_at_block16x16() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        // Variance hint says Block16x16 everywhere.
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker: low cost everywhere. At Block16x16 it's very low so
        // SPLIT (which would be 4× Block8x8, but we cap) is never
        // attempted below 16×16.
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let cost = match bs {
                BlockSize::Block16x16 => (1u32, 1u32),
                BlockSize::Block32x32 => (10_000, 1_000_000),
                BlockSize::Block64x64 => (10_000, 1_000_000),
                _ => (u32::MAX, u32::MAX),
            };
            Some(PartitionRdCandidate {
                rate: cost.0,
                dist: cost.1,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // No Block8x8 leaves should appear (the descent caps at 16×16
        // per HLD §4.3).
        for leaf in &result.leaves {
            assert!(
                !matches!(leaf.bsize, BlockSize::Block8x8),
                "descent must not emit Block8x8 leaves: got {:?}",
                leaf,
            );
        }
        // We expect 16 Block16x16 leaves (one per 16×16 cell).
        assert_eq!(
            result.leaves.len(),
            16,
            "expected 16 Block16x16 leaves, got {}",
            result.leaves.len(),
        );
        for leaf in &result.leaves {
            assert_eq!(leaf.bsize, BlockSize::Block16x16);
        }
    }

    /// **Phase A — `allow_8x8_leaf` SPLITs Block16x16 into four 8×8.**
    /// With the Phase A gate on (`min_partition_size = Block8x8`,
    /// `allow_8x8_leaf = true`, `max = Block16x16`), a 16×16 region whose
    /// four 8×8 children each fit much better than the single 16×16 NONE
    /// must SPLIT into four Block8x8 leaves. This proves the descent
    /// mechanics: 8×8 is a NONE candidate AND 16×16 can split to it.
    #[test]
    fn phase_a_splits_16x16_into_8x8() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let mut params = make_params(&pc, BlockSize::Block8x8, BlockSize::Block16x16);
        params.allow_8x8_leaf = true;
        // Picker: 16×16 NONE has high dist (a single MV can't cover the
        // sub-block motion); each 8×8 child fits cheaply. So SPLIT
        // (4× cheap 8×8) must beat 16×16 NONE at every 16×16 block.
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (rate, dist) = match bs {
                BlockSize::Block8x8 => (10u32, 1u32),
                BlockSize::Block16x16 => (40, 200_000),
                // Larger sizes excluded by max_partition_size, but give
                // them high cost so they never win if queried.
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate,
                dist,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // The whole SB is 16 × 16×16 regions; each must split into 4 8×8.
        // Expect 64 Block8x8 leaves and zero non-8×8 leaves.
        let n8 = result
            .leaves
            .iter()
            .filter(|l| matches!(l.bsize, BlockSize::Block8x8))
            .count();
        assert_eq!(
            n8,
            64,
            "Phase A: expected 64 descent-chosen Block8x8 leaves, got {n8} (leaves: {:?})",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block8x8, "every cell should be 8×8");
            }
        }
        assert_eq!(result.split_stats.block16_attempts, 16);
        assert_eq!(result.split_stats.block16_feasible, 16);
        assert_eq!(result.split_stats.block16_child_infeasible, 0);
        assert_eq!(result.split_stats.block16_partial_rd_early_out, 0);
        assert_eq!(result.split_stats.block16_split_beats_none, 16);
        assert_eq!(result.split_stats.block16_split_loses_to_none, 0);
        assert_eq!(result.split_stats.block16_split_committed, 16);
        assert_eq!(result.split_stats.block16_rect_overrides_split, 0);
    }

    /// Block16x16 SPLIT attempt whose first 8x8 child is infeasible.
    #[test]
    fn block16_split_stats_child_infeasible() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let mut params = make_params(&pc, BlockSize::Block8x8, BlockSize::Block16x16);
        params.allow_8x8_leaf = true;
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            if matches!(bs, BlockSize::Block8x8) {
                return None;
            }
            let (rate, dist) = match bs {
                BlockSize::Block16x16 => (10, 10),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate,
                dist,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);

        assert_eq!(result.split_stats.block16_attempts, 16);
        assert_eq!(result.split_stats.block16_feasible, 0);
        assert_eq!(result.split_stats.block16_infeasible_or_early, 16);
        assert_eq!(result.split_stats.block16_child_infeasible, 16);
        assert_eq!(result.split_stats.block16_partial_rd_early_out, 0);
        assert_eq!(
            result
                .leaves
                .iter()
                .filter(|l| matches!(l.bsize, BlockSize::Block16x16))
                .count(),
            16
        );
    }

    /// Block16x16 SPLIT attempt aborted by partial-RD early-out.
    #[test]
    fn block16_split_stats_partial_rd_early_out() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let mut params = make_params(&pc, BlockSize::Block8x8, BlockSize::Block16x16);
        params.allow_8x8_leaf = true;
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (rate, dist) = match bs {
                BlockSize::Block16x16 => (1u32, 1u32),
                BlockSize::Block8x8 => (10_000, 1_000_000_000),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate,
                dist,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);

        assert_eq!(result.split_stats.block16_attempts, 16);
        assert_eq!(result.split_stats.block16_feasible, 0);
        assert_eq!(result.split_stats.block16_infeasible_or_early, 16);
        assert_eq!(result.split_stats.block16_child_infeasible, 0);
        assert_eq!(result.split_stats.block16_partial_rd_early_out, 16);
        assert_eq!(
            result
                .leaves
                .iter()
                .filter(|l| matches!(l.bsize, BlockSize::Block16x16))
                .count(),
            16
        );
    }

    /// **Phase A OFF — 8×8 still never emitted by the descent.**
    /// Same picker that makes 8×8 attractive, but with `allow_8x8_leaf`
    /// off and `min_partition_size = Block16x16`: the descent must keep
    /// the pre-Phase-A floor (no descent-chosen 8×8).
    #[test]
    fn phase_a_off_keeps_16x16_floor() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block16x16);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (rate, dist) = match bs {
                BlockSize::Block8x8 => (10u32, 1u32),
                BlockSize::Block16x16 => (40, 200_000),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate,
                dist,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        for leaf in &result.leaves {
            assert!(
                !matches!(leaf.bsize, BlockSize::Block8x8),
                "Phase A OFF: descent must not emit Block8x8, got {leaf:?}",
            );
        }
    }

    /// **Test 4 — `variance_hint_bounds` floors `max` at Block16x16.**
    /// All-8×8 hint must produce `max_partition_size = Block16x16`
    /// (the R3 mitigation) rather than `Block8x8`.
    #[test]
    fn variance_hint_bounds_floors_max_at_16x16() {
        let hint = uniform_hint(BlockSize::Block8x8);
        let (min_bs, max_bs) = variance_hint_bounds(&hint);
        assert_eq!(min_bs, BlockSize::Block16x16);
        assert_eq!(max_bs, BlockSize::Block16x16);
    }

    /// `variance_hint_bounds` with a mixed-size hint returns the
    /// observed range.
    #[test]
    fn variance_hint_bounds_mixed() {
        let mut hint = uniform_hint(BlockSize::Block16x16);
        // Top-left 32×32 = one Block32x32 leaf.
        for r in 0..4 {
            for c in 0..4 {
                hint.cells[r][c] = BlockSize::Block32x32;
            }
        }
        let (min_bs, max_bs) = variance_hint_bounds(&hint);
        assert_eq!(min_bs, BlockSize::Block16x16);
        assert_eq!(max_bs, BlockSize::Block32x32);
    }

    /// **Test 5 — `partition_plane_context` formula matches libvpx.**
    /// libvpx `partition_plane_context` =
    /// `(left * 2 + above) + bsl * 4` with `bsl ∈ {0,1,2,3}` for
    /// `{8x8, 16x16, 32x32, 64x64}`.
    #[test]
    fn partition_plane_context_matches_libvpx_formula() {
        for &(bsize, expected_bsl) in &[
            (BlockSize::Block8x8, 0usize),
            (BlockSize::Block16x16, 1),
            (BlockSize::Block32x32, 2),
            (BlockSize::Block64x64, 3),
        ] {
            for (above, left) in [(false, false), (true, false), (false, true), (true, true)] {
                let ctx = partition_plane_context(above, left, bsize);
                let expected = (usize::from(left) * 2 + usize::from(above)) + expected_bsl * 4;
                assert_eq!(
                    ctx, expected,
                    "ctx mismatch for bsize={:?} above={} left={}: got {} expected {}",
                    bsize, above, left, ctx, expected,
                );
                assert!(ctx < 16, "ctx out of range: {}", ctx);
            }
        }
    }

    /// `from_variance_hint`: an all-Block64x64 hint produces exactly
    /// one leaf at the top-left, and every cell carries Block64x64
    /// after construction.
    #[test]
    fn from_variance_hint_all_64x64_yields_one_leaf() {
        let hint = uniform_hint(BlockSize::Block64x64);
        let result = NonRdPartitionResult::from_variance_hint(&hint);
        assert_eq!(result.leaves.len(), 1);
        assert_eq!(result.leaves[0].mi_row_in_sb, 0);
        assert_eq!(result.leaves[0].mi_col_in_sb, 0);
        assert_eq!(result.leaves[0].bsize, BlockSize::Block64x64);
        // Cells must mirror the hint.
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block64x64);
            }
        }
    }

    /// `from_variance_hint`: an all-Block8x8 hint produces 64 leaves
    /// (one per 8×8 cell). The `visited` map keeps each leaf unique.
    #[test]
    fn from_variance_hint_all_8x8_yields_64_leaves() {
        let hint = uniform_hint(BlockSize::Block8x8);
        let result = NonRdPartitionResult::from_variance_hint(&hint);
        assert_eq!(result.leaves.len(), 64);
        // Every leaf must be Block8x8 with `pick == IMPOSSIBLE`.
        for leaf in &result.leaves {
            assert_eq!(leaf.bsize, BlockSize::Block8x8);
            assert_eq!(leaf.pick, PartitionRdCandidate::IMPOSSIBLE);
        }
    }

    /// `from_variance_hint`: a mixed hint produces the right number
    /// of leaves: one 32×32 leaf (top-left) + 12 Block16x16 leaves
    /// (remaining 3 quadrants). Exercises the `visited[r][c]` skip
    /// branch (line 276) when entering an already-claimed cell.
    #[test]
    fn from_variance_hint_mixed_skips_visited_cells() {
        let mut hint = uniform_hint(BlockSize::Block16x16);
        // Top-left 4×4 cells (one Block32x32 leaf).
        for r in 0..4 {
            for c in 0..4 {
                hint.cells[r][c] = BlockSize::Block32x32;
            }
        }
        let result = NonRdPartitionResult::from_variance_hint(&hint);
        // 1 (32×32) + 12 (the 16×16 quadrants in rows 0..4 cols 4..8,
        // rows 4..8 cols 0..4, rows 4..8 cols 4..8).
        // Specifically: each 32×32 quadrant has 4 × Block16x16 leaves,
        // and three such quadrants → 12 leaves + 1 = 13.
        assert_eq!(result.leaves.len(), 13);
        // Top-left is a Block32x32 leaf at (0,0).
        let top_left = result
            .leaves
            .iter()
            .find(|l| l.mi_row_in_sb == 0 && l.mi_col_in_sb == 0)
            .expect("top-left leaf");
        assert_eq!(top_left.bsize, BlockSize::Block32x32);
    }

    /// `variance_hint_bounds`: defensive `min > max` clamp. Constructed
    /// case: we can't directly cause `min > max` since the floor lifts
    /// both equally — but the test demonstrates the canonical "all
    /// Block64x64" path where min == max and no defensive clamp fires.
    /// Combined with the all-8x8 test we cover the floor + clamp
    /// branches.
    #[test]
    fn variance_hint_bounds_all_64x64_returns_max_max() {
        let hint = uniform_hint(BlockSize::Block64x64);
        let (min_bs, max_bs) = variance_hint_bounds(&hint);
        assert_eq!(min_bs, BlockSize::Block64x64);
        assert_eq!(max_bs, BlockSize::Block64x64);
    }

    /// `nonrd_pick_partition`: picker returns `None` for every NONE
    /// candidate. The descent must fall back to the variance hint via
    /// `inherit_hint`, leaving the hint's leaves intact. Exercises:
    ///   - `none_allowed.then(...)` flatten returning None (lines
    ///     509-516 false path),
    ///   - `if let Some(child_bsize) = ...` False (line 532) at
    ///     Block16x16,
    ///   - `inherit_hint` body (lines 691-720).
    #[test]
    fn picker_returns_none_falls_back_to_hint() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker rejects every NONE candidate.
        let mut picker = |_x: usize, _y: usize, _bs: BlockSize| None;
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // With NONE infeasible at every level, the descent inherits
        // the hint's Block16x16 leaves throughout.
        assert_eq!(result.leaves.len(), 16);
        for leaf in &result.leaves {
            assert_eq!(leaf.bsize, BlockSize::Block16x16);
            assert_eq!(leaf.pick, PartitionRdCandidate::IMPOSSIBLE);
        }
        // Every cell must match the hint.
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block16x16);
            }
        }
    }

    /// `nonrd_pick_partition` with `max_partition_size = Block16x16`:
    /// the descent's call at Block64x64 has `none_allowed` False
    /// (bsize > max). The SPLIT branch fires, NONE is never tried
    /// at 64×64. Exercises line 498 False path.
    #[test]
    fn none_disallowed_when_bsize_exceeds_max() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block16x16);
        // Picker tracks every bsize it sees so the test can verify
        // that NONE was NEVER tried at Block32x32 or Block64x64.
        let seen = std::cell::RefCell::new(Vec::<BlockSize>::new());
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            seen.borrow_mut().push(bs);
            Some(PartitionRdCandidate {
                rate: 10,
                dist: 1,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let _ = nonrd_pick_partition(&hint, &params, &mut picker);
        let calls = seen.into_inner();
        // The picker should only ever be invoked at Block16x16 (since
        // 32×32 and 64×64 have bsize > max_partition_size).
        assert!(
            calls.iter().all(|&bs| bs == BlockSize::Block16x16),
            "picker should only see Block16x16, got: {calls:?}",
        );
        assert!(
            !calls.is_empty(),
            "picker must have been called at least once at Block16x16",
        );
    }

    /// `nonrd_pick_partition` with `min_partition_size = Block32x32`:
    /// at Block16x16 the descent's NONE check fails (bsize < min).
    /// The recursion below 32x32 is also capped (`bsize_at_floor`
    /// True). The end result: NONE wins at 32x32 (the smallest
    /// allowed). Exercises line 499 False path.
    #[test]
    fn none_disallowed_when_bsize_below_min() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block32x32);
        let params = make_params(&pc, BlockSize::Block32x32, BlockSize::Block64x64);
        // Picker returns very low cost only at sizes >= 32x32. At
        // smaller sizes it would crash — but we won't see those calls
        // because min_partition_size = Block32x32.
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| match bs {
            BlockSize::Block16x16 | BlockSize::Block8x8 => {
                panic!("picker invoked at bsize < min: {bs:?}")
            }
            _ => Some(PartitionRdCandidate {
                rate: 10,
                dist: 1,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            }),
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // The descent picks either Block64x64 (NONE wins at top) or
        // 4× Block32x32 (SPLIT wins). Either way, no leaves smaller
        // than 32×32.
        for leaf in &result.leaves {
            assert!(
                matches!(leaf.bsize, BlockSize::Block32x32 | BlockSize::Block64x64),
                "no leaf below min_partition_size: got {:?}",
                leaf.bsize,
            );
        }
    }

    /// `commit_leaf`: NONE wins at Block64x64 after some children
    /// have speculatively committed Block32x32 leaves during the
    /// SPLIT exploration. The `state.leaves.retain` body (lines
    /// 669-673) must drop the speculative leaves so we end up with
    /// exactly one Block64x64 leaf. Exercises the `retain` true
    /// branch (children removed).
    #[test]
    fn commit_leaf_drops_speculative_children() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker: very low cost at 64x64 → NONE wins. Modest cost at
        // 32x32 → SPLIT children also feasible (they get pushed as
        // speculative leaves), but their RD sum is worse than the
        // 64×64 NONE → commit_leaf rolls them back.
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                BlockSize::Block64x64 => (10u32, 1u32),
                BlockSize::Block32x32 => (100, 50_000),
                BlockSize::Block16x16 => (1000, 100_000),
                _ => (u32::MAX, u32::MAX),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // After commit_leaf rolls back the speculative SPLIT children,
        // we must end up with exactly one Block64x64 leaf.
        assert_eq!(
            result.leaves.len(),
            1,
            "commit_leaf must roll back speculative children, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        assert_eq!(result.leaves[0].bsize, BlockSize::Block64x64);
    }

    /// SPLIT loses on the final RD comparison (line 587 False
    /// branch): all four children evaluate successfully (no early-out)
    /// but the summed SPLIT cost exceeds the NONE best. The result
    /// must be a single NONE leaf at this level.
    #[test]
    fn split_loses_on_final_comparison() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker: NONE at 64×64 is very cheap. NONE at 32×32 is just
        // barely affordable per child (no early-out) but 4× exceeds
        // the 64×64 cost. NONE at 16×16 is similar.
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                BlockSize::Block64x64 => (10u32, 100u32),
                // Each 32x32 child is (rate=30, dist=80). rd per child
                // ≈ lambda*30/512 + 80<<7 = small. 4× still beats
                // 64×64? 4*(30, 80) = (120, 320). rd_split = ~256*128
                // + 120*lambda/512. We want this > 64x64 NONE rd:
                // 100<<7 = 12800. 320<<7 = 40960. Definitely larger.
                BlockSize::Block32x32 => (30, 80),
                BlockSize::Block16x16 => (40, 20),
                _ => (u32::MAX, u32::MAX),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // SPLIT loses → committed as a single Block64x64 leaf.
        assert_eq!(result.leaves.len(), 1);
        assert_eq!(result.leaves[0].bsize, BlockSize::Block64x64);
    }

    /// Early-out path (line 570): a SPLIT child's accumulated RD
    /// already exceeds the NONE best, so the remaining children are
    /// skipped. We can't observe the early-out directly, but we can
    /// verify the result is consistent with the NONE winner (i.e.
    /// SPLIT did not get committed even though some children were
    /// evaluated).
    #[test]
    fn split_early_out_when_partial_sum_exceeds_best() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        // Picker: extremely cheap NONE at 64×64 (best_rd_cost ≈ ~128),
        // very expensive children at 32×32. The first child's RD
        // alone exceeds the NONE best → early-out kicks in.
        let call_count = std::cell::Cell::new(0u32);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            call_count.set(call_count.get() + 1);
            let (r, d) = match bs {
                BlockSize::Block64x64 => (1u32, 1u32),
                BlockSize::Block32x32 => (10_000, 1_000_000_000),
                _ => (u32::MAX, u32::MAX),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // NONE at 64×64 must be the winner.
        assert_eq!(result.leaves.len(), 1);
        assert_eq!(result.leaves[0].bsize, BlockSize::Block64x64);
        // The picker was called for at least the 64×64 NONE + one
        // 32×32 NONE before early-out. The early-out invariant: total
        // picker calls is strictly less than the "all children fully
        // explored" count (4 children × 4 grandchildren = 16 at 32×32
        // level if every child fully recursed).
        assert!(
            call_count.get() < 20,
            "early-out should bound picker calls, got {}",
            call_count.get(),
        );
    }

    /// **Test 6 — NONE-vs-SPLIT RD comparison: SPLIT wins when
    /// children sum is cheaper.**
    /// Picker returns very high cost for the 64×64 NONE candidate but
    /// low cost for every 32×32 NONE candidate. F.3 must pick SPLIT
    /// at 64×64 (producing 4× Block32x32 leaves).
    #[test]
    fn split_wins_when_children_cheaper() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let cost = match bs {
                BlockSize::Block64x64 => (10_000u32, 1_000_000u32),
                BlockSize::Block32x32 => (10, 1),
                BlockSize::Block16x16 => (1_000, 100_000),
                _ => (u32::MAX, u32::MAX),
            };
            Some(PartitionRdCandidate {
                rate: cost.0,
                dist: cost.1,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // We expect 4 Block32x32 leaves at the SB corners.
        let n_32x32 = result
            .leaves
            .iter()
            .filter(|l| matches!(l.bsize, BlockSize::Block32x32))
            .count();
        assert_eq!(
            n_32x32,
            4,
            "expected 4 Block32x32 leaves after SPLIT, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
    }

    /// **A4-rect Test — PARTITION_HORZ wins on a horizontal edge.**
    /// Picker reports cheap RD for the two `64×32` HORZ halves and
    /// expensive RD for everything else (NONE at 64×64, the 32×32 SPLIT
    /// quadrants, and the 32×64 VERT halves). The descent must emit
    /// PARTITION_HORZ at 64×64 → two `Block64x32` leaves stacked.
    #[test]
    fn horz_wins_on_horizontal_edge() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                // Cheap HORZ halves.
                BlockSize::Block64x32 => (20u32, 5u32),
                // Everything else expensive so HORZ wins outright.
                BlockSize::Block64x64 => (10_000, 1_000_000),
                BlockSize::Block32x64 => (10_000, 1_000_000),
                BlockSize::Block32x32 => (10_000, 1_000_000),
                BlockSize::Block16x16 => (10_000, 1_000_000),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        // Two Block64x32 leaves: top at (0,0), bottom at (4,0).
        let horz: Vec<_> = result
            .leaves
            .iter()
            .filter(|l| matches!(l.bsize, BlockSize::Block64x32))
            .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb))
            .collect();
        assert_eq!(
            horz.len(),
            2,
            "expected exactly 2 Block64x32 HORZ leaves, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        assert!(horz.contains(&(0, 0)), "HORZ top leaf at (0,0) missing");
        assert!(horz.contains(&(4, 0)), "HORZ bottom leaf at (4,0) missing");
        // The full SB cell grid must be Block64x32 (both halves tile it).
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block64x32);
            }
        }
    }

    /// **A4-rect Test — PARTITION_VERT wins on a vertical edge.**
    /// Mirror of the HORZ test: cheap `32×64` VERT halves, expensive
    /// everything else. The descent must emit PARTITION_VERT at 64×64 →
    /// two `Block32x64` leaves side by side at (0,0) and (0,4).
    #[test]
    fn vert_wins_on_vertical_edge() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                BlockSize::Block32x64 => (20u32, 5u32),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        let vert: Vec<_> = result
            .leaves
            .iter()
            .filter(|l| matches!(l.bsize, BlockSize::Block32x64))
            .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb))
            .collect();
        assert_eq!(
            vert.len(),
            2,
            "expected exactly 2 Block32x64 VERT leaves, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        assert!(vert.contains(&(0, 0)), "VERT left leaf at (0,0) missing");
        assert!(vert.contains(&(0, 4)), "VERT right leaf at (0,4) missing");
        for row in &result.cells {
            for &c in row {
                assert_eq!(c, BlockSize::Block32x64);
            }
        }
    }

    /// **A4-rect Test — HORZ at 32×32 inside a SPLIT.**
    /// Force a 64→split, then make HORZ cheapest at each 32×32 quadrant
    /// (cheap `32×16` halves). Each quadrant must become two Block32x16
    /// leaves → 8 total. Exercises rect commit nested under SPLIT and the
    /// speculative-child rollback.
    #[test]
    fn horz_at_32x32_under_split() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                // 64×64 NONE and its rects expensive → SPLIT preferred.
                BlockSize::Block64x64 => (10_000, 10_000_000),
                BlockSize::Block64x32 => (10_000, 10_000_000),
                BlockSize::Block32x64 => (10_000, 10_000_000),
                // 32×32 NONE moderate; 32×16 HORZ halves cheapest.
                BlockSize::Block32x32 => (500, 50_000),
                BlockSize::Block32x16 => (20, 50),
                BlockSize::Block16x32 => (500, 50_000),
                // 16×16 expensive so quadrants don't sub-split further.
                BlockSize::Block16x16 => (10_000, 10_000_000),
                _ => (10_000, 10_000_000),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let result = nonrd_pick_partition(&hint, &params, &mut picker);
        let n_32x16 = result
            .leaves
            .iter()
            .filter(|l| matches!(l.bsize, BlockSize::Block32x16))
            .count();
        assert_eq!(
            n_32x16,
            8,
            "expected 8 Block32x16 HORZ leaves (2 per 32×32 quadrant), got: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        // No square 32×32 leaf should survive (every quadrant went HORZ).
        assert!(
            !result
                .leaves
                .iter()
                .any(|l| matches!(l.bsize, BlockSize::Block32x32)),
            "no Block32x32 leaf should remain when HORZ wins every quadrant",
        );
    }

    /// **A21-rect Test — RectPruner disallowing HORZ suppresses it.**
    /// Same HORZ-wins picker as `horz_wins_on_horizontal_edge`, but a
    /// `RectPruner` that vetoes the HORZ arm (`allow_horz = false`). The
    /// descent must NOT emit any `Block64x32` leaf — it falls back to
    /// the next-best arm (here NONE / SPLIT). The pruner must also have
    /// been CONSULTED (the recorded `(bsize, ctx)` proves real feature
    /// flow, not a stub). Anti-G3: a no-op hook would leave HORZ winning.
    #[test]
    fn rect_pruner_vetoing_horz_suppresses_it() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let mut picker = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                BlockSize::Block64x32 => (20u32, 5u32),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        // Pruner: veto HORZ everywhere, keep VERT. Record the
        // (bsize, partitioning_is_none, ref_rd-present) tuples it saw so
        // we can assert it received real RD context.
        let mut seen: Vec<(BlockSize, bool, bool)> = Vec::new();
        let mut pruner = |_x: usize,
                          _y: usize,
                          bs: BlockSize,
                          rd: RectPruneRdContext,
                          _ah: bool,
                          av: bool|
         -> (bool, bool) {
            seen.push((bs, rd.partitioning_is_none, rd.best_rd_cost.is_some()));
            (false, av) // disallow HORZ, keep VERT
        };
        let result = nonrd_pick_partition_with_pruners(
            &hint,
            &params,
            &mut picker,
            &mut NoBreakout,
            &mut pruner,
        );
        // HORZ was vetoed → no Block64x32 leaf can appear.
        assert!(
            !result
                .leaves
                .iter()
                .any(|l| matches!(l.bsize, BlockSize::Block64x32)),
            "HORZ vetoed by the rect pruner, but a Block64x32 leaf appeared: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
        // The pruner must have been consulted at least once at 64×64
        // with a present ref_rd (real RD context, not a stub).
        assert!(
            seen.iter()
                .any(|&(bs, _, ref_present)| bs == BlockSize::Block64x64 && ref_present),
            "rect pruner was never consulted at 64×64 with a ref_rd: {:?}",
            seen,
        );
    }

    /// **A21-rect Test — NoRectPrune is a transparent pass-through.**
    /// With the default `NoRectPrune`, the rect-pruner path must produce
    /// EXACTLY the same result as the plain breakout-only descent (the
    /// pre-A21-rect baseline) — proving the hook adds zero behaviour when
    /// inactive (gate-faithfulness: the sacred gate runs with rect-prune
    /// off and must stay byte-identical).
    #[test]
    fn no_rect_prune_matches_baseline() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        let hint = uniform_hint(BlockSize::Block16x16);
        let params = make_params(&pc, BlockSize::Block16x16, BlockSize::Block64x64);
        let picker_fn = |_x: usize, _y: usize, bs: BlockSize| {
            let (r, d) = match bs {
                BlockSize::Block64x32 => (20u32, 5u32),
                _ => (10_000, 1_000_000),
            };
            Some(PartitionRdCandidate {
                rate: r,
                dist: d,
                mv: Mv::ZERO,
                mode: 0,
                ref_frame: 0,
            })
        };
        let mut p1 = picker_fn;
        let baseline = nonrd_pick_partition(&hint, &params, &mut p1);
        let mut p2 = picker_fn;
        let with_pruner = nonrd_pick_partition_with_pruners(
            &hint,
            &params,
            &mut p2,
            &mut NoBreakout,
            &mut NoRectPrune,
        );
        assert_eq!(
            baseline, with_pruner,
            "NoRectPrune must be a transparent pass-through (baseline parity)",
        );
    }
}
