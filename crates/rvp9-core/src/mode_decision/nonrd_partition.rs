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
//! ## Scope (F.3 first cut)
//!
//! Per the F.3 HLD §2.3 / §4.3 / §10:
//!
//! - **Square partitions only.** No PARTITION_HORZ / PARTITION_VERT
//!   (libvpx defers these at `speed >= 5`,
//!   `vp9_encodeframe.c:4957`).
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

use crate::block::BlockSize;
use crate::cost::symbol_costs::PartitionCosts;
use crate::mode_decision::partition::{PartitionDecision, SB_8X8_GRID};
use crate::mode_decision::rd_cost;
use crate::mode_info::Mv;

/// Side length of the SB grid in 8×8 mi cells (8 cells = 64 px). Mirrors
/// `partition::SB_8X8_GRID` but kept private here to avoid a public
/// re-export.
const SB_GRID: usize = SB_8X8_GRID;

/// libvpx `PARTITION_NONE = 0` / `PARTITION_SPLIT = 3` — see
/// `vp9/common/vp9_enums.h:84-89`.
const PARTITION_NONE_IDX: usize = 0;
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

/// Result of one SB descent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonRdPartitionResult {
    /// Per-8×8-cell leaf-size grid (same shape as
    /// [`PartitionDecision::cells`]). Lets the caller cheaply look up
    /// "what size is this cell part of".
    pub cells: [[BlockSize; SB_GRID]; SB_GRID],
    /// Unique leaves in raster order (top-left cell only per leaf).
    pub leaves: Vec<NonRdLeaf>,
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
    let mut state = DescentState {
        cells: [[BlockSize::Block8x8; SB_GRID]; SB_GRID],
        leaves: Vec::with_capacity(16),
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
        (0, 0),
        BlockSize::Block64x64,
        ctx_above,
        ctx_left,
        &mut state,
    );
    NonRdPartitionResult {
        cells: state.cells,
        leaves: state.leaves,
    }
}

struct DescentState {
    cells: [[BlockSize; SB_GRID]; SB_GRID],
    leaves: Vec<NonRdLeaf>,
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
fn descend<P: ModePickerFn>(
    hint: &PartitionDecision,
    params: &NonRdDescentParams<'_>,
    picker: &mut P,
    origin_8x8: (usize, usize),
    bsize: BlockSize,
    above: bool,
    left: bool,
    state: &mut DescentState,
) -> Option<DescendResult> {
    let (row_8x8, col_8x8) = origin_8x8;

    // §4.3 cap: never descend below the floor. When we'd step below
    // Block16x16 (i.e. caller asked for Block8x8), we inherit the
    // variance hint's cells unchanged and return "infeasible" so the
    // parent's NONE comparison wins the tie if NONE was feasible.
    let bsize_at_floor = (bsize as u8) <= (params.min_partition_size as u8)
        || matches!(bsize, BlockSize::Block8x8);

    // PARTITION_NONE allowed?
    let none_allowed = (bsize as u8) <= (params.max_partition_size as u8)
        && (bsize as u8) >= (params.min_partition_size as u8)
        && !matches!(bsize, BlockSize::Block8x8);

    let part_ctx = partition_plane_context(above, left, bsize);
    let none_cost = params.partition_costs[part_ctx][PARTITION_NONE_IDX].max(0) as u32;
    let split_cost = params.partition_costs[part_ctx][PARTITION_SPLIT_IDX].max(0) as u32;

    // Try PARTITION_NONE.
    let mut best_pick: Option<PartitionRdCandidate> = None;
    let mut best_rd_cost: i64 = i64::MAX;
    if let Some(cand) = none_allowed
        .then(|| picker.pick(col_8x8, row_8x8, bsize))
        .flatten()
    {
        // Add partition_cost[NONE] to rate (libvpx :5010).
        let rate_total = cand.rate.saturating_add(none_cost);
        let rdc = rd_cost(params.rd_mult, rate_total, cand.dist);
        if rdc < best_rd_cost {
            best_rd_cost = rdc;
            best_pick = Some(PartitionRdCandidate {
                rate: rate_total,
                dist: cand.dist,
                mv: cand.mv,
                mode: cand.mode,
                ref_frame: cand.ref_frame,
            });
        }
    }

    // Try PARTITION_SPLIT — only if we have a child size to recurse
    // into AND we haven't hit the floor.
    let mut split_pick: Option<(u32, u32)> = None; // (sum_rate, sum_dist)
    let mut split_children: [Option<DescendResult>; 4] = [None, None, None, None];
    if let Some(child_bsize) = (!bsize_at_floor).then(|| subsize_for_split(bsize)).flatten() {
        let step = split_step_8x8(bsize);
        let mut sum_rate: u32 = split_cost;
        let mut sum_dist: u32 = 0;
        let mut all_feasible = true;
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
                all_feasible = false;
                break;
            }
            match descend(
                hint,
                params,
                picker,
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
                    let rdc_so_far = rd_cost(params.rd_mult, sum_rate, sum_dist);
                    if rdc_so_far >= best_rd_cost && best_pick.is_some() {
                        // Mark remaining quadrants infeasible (so
                        // we don't try to commit them).
                        all_feasible = false;
                        break;
                    }
                }
                None => {
                    // Child infeasible → SPLIT can't win at this
                    // level. Fall back to NONE if we have it.
                    all_feasible = false;
                    break;
                }
            }
        }
        if all_feasible {
            let rdc_split = rd_cost(params.rd_mult, sum_rate, sum_dist);
            if rdc_split < best_rd_cost {
                best_rd_cost = rdc_split;
                split_pick = Some((sum_rate, sum_dist));
            }
        }
    }

    // Commit the winner into the cell grid + leaves.
    if let Some((rate, dist)) = split_pick {
        // SPLIT wins — commit children's leaves (already pushed during
        // recursion via their NONE branch). Nothing further to do at
        // this level beyond updating the parent's accounting.
        return Some(DescendResult {
            rate,
            dist,
            rd_cost: best_rd_cost,
        });
    }
    if let Some(pick) = best_pick {
        // NONE wins (or SPLIT wasn't feasible) — commit this block as
        // a single leaf, overwriting any children we tentatively
        // committed during recursion. Recursion only commits children
        // when their *own* NONE wins, so the worst case is a few
        // dropped leaves; we rebuild this sub-block cleanly here.
        commit_leaf(state, row_8x8, col_8x8, bsize, pick);
        return Some(DescendResult {
            rate: pick.rate,
            dist: pick.dist,
            rd_cost: best_rd_cost,
        });
    }

    // Neither NONE nor SPLIT was feasible at this level — fall back
    // to the variance hint. The hint's leaves at this position become
    // our leaves; the descent contributes no RD signal here (parent's
    // SPLIT sum will mark this as infeasible and the parent's NONE
    // wins by default).
    inherit_hint(hint, state, row_8x8, col_8x8, bsize);
    None
}

/// Result returned by [`descend`] for a sub-block — the totals the
/// parent needs to accumulate into its SPLIT sum.
#[derive(Copy, Clone, Debug)]
struct DescendResult {
    rate: u32,
    dist: u32,
    #[allow(dead_code)] // useful for diagnostics; tests don't read it
    rd_cost: i64,
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
                let expected =
                    (usize::from(left) * 2 + usize::from(above)) + expected_bsl * 4;
                assert_eq!(
                    ctx, expected,
                    "ctx mismatch for bsize={:?} above={} left={}: got {} expected {}",
                    bsize, above, left, ctx, expected,
                );
                assert!(ctx < 16, "ctx out of range: {}", ctx);
            }
        }
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
            n_32x32, 4,
            "expected 4 Block32x32 leaves after SPLIT, got leaves: {:?}",
            result
                .leaves
                .iter()
                .map(|l| (l.mi_row_in_sb, l.mi_col_in_sb, l.bsize))
                .collect::<Vec<_>>(),
        );
    }
}
