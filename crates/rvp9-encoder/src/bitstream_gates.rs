//! W7.1-W7.3 — bitstream emission gates ported from libvpx.
//!
//! Three independent helpers, each a scalar reference port of a libvpx
//! `vp9_bitstream.c` / `vp9_encoder.c` / `vp9_subexp.c` block:
//!
//! - [`prob_diff_update_savings_search`] — model-free savings search
//!   that picks the best replacement probability for a given pair of
//!   `(zero_count, one_count)` branch counts. Mirrors libvpx
//!   `vp9_prob_diff_update_savings_search` (`vp9_subexp.c:117-139`).
//! - [`prob_diff_update_savings_search_model`] — the PIVOT_NODE
//!   variant that scores the full coefficient sub-tree under the
//!   Pareto-8 model (`VP9_PARETO8_FULL`) rather than a single binary
//!   branch. Mirrors libvpx
//!   `vp9_prob_diff_update_savings_search_model` (`vp9_subexp.c:141-180`).
//!   Used by libvpx's TWO_LOOP coef updater for `t == PIVOT_NODE`
//!   (`vp9_bitstream.c:569-575`) — the case the plain search cannot
//!   model because the pivot prob drives 8 downstream model nodes.
//! - [`build_coef_branch_cts`] — the per-(plane,ref,band,ctx)
//!   `build_tree_distribution` body (`vp9_bitstream.c:518-543`): walks
//!   the FULL 12-token coefficient distribution through
//!   `vp9_tree_probs_from_distribution` to produce the
//!   `[ENTROPY_NODES][2]` branch counts the model search consumes,
//!   plus the per-node "ideal" probabilities (`get_binary_prob`).
//! - [`update_coef_probs_common`] — for one TX size, runs the
//!   "TWO_LOOP" path of libvpx's coef-prob updater: first a dry-run
//!   scan to compute aggregate `savings`, then either gate=0 (no
//!   update) or gate=1 followed by per-slot updates. Mirrors libvpx
//!   `update_coef_probs_common` (`vp9_bitstream.c:545-699`,
//!   TWO_LOOP branch).
//! - [`tx_totals_gate_skips`] — outer-loop early-out used by libvpx
//!   `update_coef_probs` (`vp9_bitstream.c:683-699`): if the
//!   per-TX-size token total is `<= 20` the encoder skips the
//!   savings search entirely and writes a single `0` gate bit.
//! - [`fix_interp_filter`] — collapse a SWITCHABLE frame-level
//!   interp filter down to a fixed filter when the per-tile
//!   `interp_filter_selected` counts show only one filter was
//!   actually used. Mirrors libvpx `fix_interp_filter`
//!   (`vp9_bitstream.c:863-884`).
//! - [`get_refresh_mask`] — compute the 8-bit `refresh_frame_flags`
//!   uncompressed-header field, including the
//!   `preserve_existing_gf` ARF-overlap special case and the
//!   `multi_layer_arf` slot search. Mirrors libvpx
//!   `vp9_get_refresh_mask` (`vp9_bitstream.c:902-940`) and the
//!   adjacent `vp9_preserve_existing_gf`
//!   (`vp9_bitstream.h:41-44`) inline.
//!
//! ## Out of scope
//!
//! - HBD: every helper assumes 8-bit unsigned counts / probabilities.
//! - SIMD: scalar reference only.

use rvp9_core::bitstream::InterpFilter;
use rvp9_core::bitstream::frame::{REF_FRAMES, ReferenceMode};
use rvp9_core::bitstream::{
    CoefProbUpdatePlan, CoefSlotUpdate, MV_UPDATE_PROB, tx_mode_to_biggest_tx_size,
};
use rvp9_core::cost::coef_cost::{ENTROPY_NODES, ENTROPY_TOKENS, PIVOT_NODE};
use rvp9_core::cost::prob_cost::{VP9_PROB_COST_SHIFT, vp9_cost_one, vp9_cost_zero};
use rvp9_core::entropy::BoolWriter;
use rvp9_core::entropy::adapt::{
    VP9_INTER_MODE_TREE, VP9_INTRA_MODE_TREE, VP9_MV_CLASS_TREE, VP9_MV_CLASS0_TREE,
    VP9_MV_FP_TREE, VP9_MV_JOINT_TREE, VP9_PARTITION_TREE, VP9_SWITCHABLE_INTERP_TREE,
};
use rvp9_core::entropy::{
    BLOCK_SIZE_GROUPS, CLASS0_SIZE, COEF_BANDS, COEFF_CONTEXTS, Coef12Bank, CoeffProbsModel,
    DIFF_UPDATE_PROB, EobBranchCountsBlock, FrameContext, FrameCounts, INTER_MODE_CONTEXTS,
    INTRA_INTER_CONTEXTS, MODEL_NODES, MV_OFFSET_BITS, PARTITION_CONTEXTS, PLANE_TYPES,
    REF_CONTEXTS, REF_TYPES, SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
    TX_SIZE_CONTEXTS, TX_SIZES, UNCONSTRAINED_NODES, VP9_PARETO8_FULL, get_binary_prob, remap_prob,
    write_prob_diff_update,
};
use rvp9_core::transforms::TxMode;

// =====================================================================
// W7.1 — prob_diff_update_savings_search + update_coef_probs_common
// =====================================================================

/// `update_bits[delp]` — the wire-bit cost of the term-subexp payload
/// for a remapped `delp` value in `[0, 254]`. Verbatim from libvpx
/// `vp9_subexp.c:17-32`. Slot 255 is `0` (unused).
#[rustfmt::skip]
static UPDATE_BITS: [u8; 255] = [
    5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 0,
];

/// Minimum delp-bits constant — the search early-out floor. Mirrors
/// libvpx `MIN_DELP_BITS` (`vp9_subexp.c:33`).
const MIN_DELP_BITS: i64 = 5;

/// Cost of writing a `delp` payload, in fixed-point bit units. Mirrors
/// libvpx `prob_diff_update_cost` (`vp9_subexp.c:79-82`).
#[inline]
fn prob_diff_update_cost(newp: u8, oldp: u8) -> i64 {
    let delp = remap_prob(newp, oldp);
    (i64::from(UPDATE_BITS[delp as usize])) << VP9_PROB_COST_SHIFT
}

/// `cost_branch256(ct, p)` — the expected number of fixed-point bits
/// to encode a stream where `ct[0]` zeros and `ct[1]` ones appear
/// under probability `p`. Mirrors libvpx `cost_branch256`
/// (`vp9_cost.h:32-34`).
#[inline]
fn cost_branch256(ct: [u32; 2], p: u8) -> i64 {
    i64::from(ct[0]) * i64::from(vp9_cost_zero(p)) + i64::from(ct[1]) * i64::from(vp9_cost_one(p))
}

/// One slot's worth of the savings-search outcome. `best_newp` is the
/// best replacement probability the search found (always `oldp` if no
/// improvement); `savings` is the resulting rate savings (always
/// `>= 0`).
#[derive(Clone, Copy, Debug)]
pub struct SavingsSearch {
    /// Best replacement probability (the search defaults to `oldp`).
    pub best_newp: u8,
    /// Rate savings in fixed-point bit units (`>= 0`).
    pub savings: i64,
}

/// Walk every probability strictly between `oldp` and `start_newp`
/// (exclusive of `oldp`), pick the one with the largest non-negative
/// savings vs. the `oldp` baseline. Mirrors libvpx
/// `vp9_prob_diff_update_savings_search` (`vp9_subexp.c:117-139`).
///
/// The search direction (`step = -1` or `+1`) is set so the walk
/// approaches `oldp`. The walk halts at `newp == oldp` (exclusive).
/// If `start_newp == oldp` no search is performed.
///
/// `ct = [zero_count, one_count]`. `upd` is the gate probability
/// (libvpx uses `DIFF_UPDATE_PROB = 252`).
#[must_use]
pub fn prob_diff_update_savings_search(
    ct: [u32; 2],
    oldp: u8,
    start_newp: u8,
    upd: u8,
) -> SavingsSearch {
    let old_b = cost_branch256(ct, oldp);
    let upd_cost = i64::from(vp9_cost_one(upd)) - i64::from(vp9_cost_zero(upd));

    let mut best_savings: i64 = 0;
    let mut best_newp = oldp;

    // Early-out: libvpx `vp9_subexp.c:126` — give up if the baseline
    // cost isn't even enough to cover the gate + the cheapest payload.
    if old_b > upd_cost + (MIN_DELP_BITS << VP9_PROB_COST_SHIFT) {
        let step: i32 = if start_newp > oldp { -1 } else { 1 };
        let mut newp = start_newp as i32;
        while newp != oldp as i32 {
            let newp_u8 = newp as u8;
            let new_b = cost_branch256(ct, newp_u8);
            let update_b = prob_diff_update_cost(newp_u8, oldp) + upd_cost;
            let savings = old_b - new_b - update_b;
            if savings > best_savings {
                best_savings = savings;
                best_newp = newp_u8;
            }
            newp += step;
        }
    }

    SavingsSearch {
        best_newp,
        savings: best_savings,
    }
}

// ---------------------------------------------------------------------
// PIVOT_NODE model search + build_tree_distribution
// ---------------------------------------------------------------------
//
// These are the two libvpx pieces the original W7.1 port omitted (its
// documented deviation): the PIVOT_NODE model search needs the FULL
// 12-token coefficient distribution (it scores 8 downstream Pareto-8
// model nodes), and `build_tree_distribution` is what turns the raw
// token counts into the `[ENTROPY_NODES][2]` branch counts that search
// consumes. They are pure functions with no bitstream side effects;
// wiring them into the live compressed-header writer additionally needs
// full-12-token count plumbing + a two-pass encode (see the W-coef-prob
// journal). They land here as the verified building blocks.

// `vp9_coef_tree[TREE_SIZE(ENTROPY_TOKENS)]` from libvpx
// `vp9_tokenize.c:75-87`. libvpx stores leaves as `-token` and internal
// children as positive tree indices, using `tree[i] <= 0` to detect a
// leaf and `num_events[-tree[i]]` to read its count. That encoding is
// ambiguous in Rust for `-ZERO_TOKEN == 0` vs `-EOB_TOKEN == -11` once
// a leaf overlaps index 0, so we split the table in two:
//
// - `LEAF_TOKEN[p]` resolves the leaf token value at tree slot `p`
//   positionally (255 = internal node, recurse instead). This carries
//   the `-token` information unambiguously.
// - `COEF_TREE_CHILD[p]` holds the child tree index for the internal
//   slots; it is only read when `LEAF_TOKEN[p] == 255`.
//
// Slot index → libvpx tree row (each row = 2 slots = one parent):
//   0/1 EOB|→2   2/3 ZERO|→4   4/5 ONE|→6   6/7 →8|→12
//   8/9 TWO|→10  10/11 THREE|FOUR   12/13 →14|→16
//   14/15 CAT1|CAT2  16/17 →18|→20  18/19 CAT3|CAT4  20/21 CAT5|CAT6
#[rustfmt::skip]
static LEAF_TOKEN: [u8; 2 * (ENTROPY_TOKENS - 1)] = [
    11, 255,  // EOB_TOKEN(11) | internal→2
     0, 255,  // ZERO_TOKEN(0) | internal→4
     1, 255,  // ONE_TOKEN(1)  | internal→6
    255, 255, // internal→8    | internal→12
     2, 255,  // TWO_TOKEN(2)  | internal→10
     3,  4,   // THREE(3)      | FOUR(4)
    255, 255, // internal→14   | internal→16
     5,  6,   // CAT1(5)       | CAT2(6)
    255, 255, // internal→18   | internal→20
     7,  8,   // CAT3(7)       | CAT4(8)
     9, 10,   // CAT5(9)       | CAT6(10)
];

/// Child tree index for each internal slot (read only where
/// `LEAF_TOKEN == 255`). 255 marks "leaf slot, unused".
#[rustfmt::skip]
static COEF_TREE_CHILD: [u8; 2 * (ENTROPY_TOKENS - 1)] = [
    255,  2,
    255,  4,
    255,  6,
      8, 12,
    255, 10,
    255, 255,
     14, 16,
    255, 255,
     18, 20,
    255, 255,
    255, 255,
];

/// `convert_distribution` (`vp9_treewriter.c:34-52`) — recursively sum
/// the per-token event counts up the `VP9_COEF_TREE`, writing each
/// internal node's `[left, right]` split into `branch_ct`. Returns the
/// total events under subtree rooted at tree index `i`.
fn convert_distribution(
    i: usize,
    num_events: &[u32; ENTROPY_TOKENS],
    branch_ct: &mut [[u32; 2]],
) -> u32 {
    let left = if LEAF_TOKEN[i] != 255 {
        num_events[LEAF_TOKEN[i] as usize]
    } else {
        convert_distribution(COEF_TREE_CHILD[i] as usize, num_events, branch_ct)
    };
    let right = if LEAF_TOKEN[i + 1] != 255 {
        num_events[LEAF_TOKEN[i + 1] as usize]
    } else {
        convert_distribution(COEF_TREE_CHILD[i + 1] as usize, num_events, branch_ct)
    };
    branch_ct[i >> 1][0] = left;
    branch_ct[i >> 1][1] = right;
    left + right
}

/// The `build_tree_distribution` body for one `(plane, ref, band, ctx)`
/// slot. Mirrors libvpx `build_tree_distribution`
/// (`vp9_bitstream.c:518-543`):
///
/// 1. `vp9_tree_probs_from_distribution(vp9_coef_tree, branch_ct, counts)`
///    fills `branch_ct[ENTROPY_NODES][2]` from the full 12-token count
///    histogram.
/// 2. Node 0 (the EOB branch) is OVERWRITTEN: its `[1]` slot becomes
///    `eob_branch_total - branch_ct[0][0]` (the # of positions that
///    were NOT EOB), matching libvpx `:533-534`.
/// 3. The three UNCONSTRAINED_NODES "ideal" probabilities are
///    `get_binary_prob(branch_ct[m][0], branch_ct[m][1])` (`:535-538`).
///
/// `counts` is the full per-token histogram `[ENTROPY_TOKENS]`;
/// `eob_branch_total` is libvpx's `eob_branch_ct[i][j][k][l]`. Returns
/// the branch counts (all `ENTROPY_NODES`) and the 3 ideal node probs.
#[must_use]
pub fn build_coef_branch_cts(
    counts: &[u32; ENTROPY_TOKENS],
    eob_branch_total: u32,
) -> ([[u32; 2]; ENTROPY_NODES], [u8; UNCONSTRAINED_NODES]) {
    let mut branch_ct = [[0u32; 2]; ENTROPY_NODES];
    convert_distribution(0, counts, &mut branch_ct);
    // libvpx `:533-534`: node 0's "one" branch is the non-EOB count.
    branch_ct[0][1] = eob_branch_total.saturating_sub(branch_ct[0][0]);
    let mut probs = [0u8; UNCONSTRAINED_NODES];
    for m in 0..UNCONSTRAINED_NODES {
        probs[m] = get_binary_prob(branch_ct[m][0], branch_ct[m][1]);
    }
    (branch_ct, probs)
}

/// PIVOT_NODE savings search. Mirrors libvpx
/// `vp9_prob_diff_update_savings_search_model` (`vp9_subexp.c:141-180`).
///
/// Unlike the plain search, the pivot probability `oldp`/`newp` does not
/// score a single branch — it indexes `VP9_PARETO8_FULL[p - 1]`, an
/// 8-entry model that supplies the probabilities for the 8 downstream
/// model nodes. So the baseline / candidate costs sum the PIVOT node's
/// own branch cost PLUS the 8 model-node branch costs evaluated under
/// the model probs. `branch_ct` is the full `[ENTROPY_NODES][2]` from
/// [`build_coef_branch_cts`].
///
/// `start_newp` is the caller's "ideal" pivot prob (libvpx's `*bestp`
/// on entry, i.e. `new_coef_probs[..][PIVOT_NODE]`). `stepsize` is
/// libvpx `cpi->sf.coeff_prob_appx_step` (1 for the reference encoder).
#[must_use]
pub fn prob_diff_update_savings_search_model(
    branch_ct: &[[u32; 2]; ENTROPY_NODES],
    oldp: u8,
    start_newp: u8,
    upd: u8,
    stepsize: i32,
) -> SavingsSearch {
    debug_assert!(stepsize > 0);
    let step_sign: i64 = if start_newp > oldp { -1 } else { 1 };
    let step: i64 = i64::from(stepsize) * step_sign;
    let upd_cost = i64::from(vp9_cost_one(upd)) - i64::from(vp9_cost_zero(upd));

    // Baseline cost under `oldp`: the pivot branch + the 8 model nodes
    // scored with VP9_PARETO8_FULL[oldp - 1]. libvpx `vp9_subexp.c:152-155`.
    let oldplist = &VP9_PARETO8_FULL[(oldp as usize) - 1];
    let mut old_b = cost_branch256(branch_ct[PIVOT_NODE], oldp);
    for i in UNCONSTRAINED_NODES..ENTROPY_NODES {
        old_b += cost_branch256(branch_ct[i], oldplist[i - UNCONSTRAINED_NODES]);
    }

    let mut best_savings: i64 = 0;
    let mut best_newp = oldp;

    if old_b > upd_cost + (MIN_DELP_BITS << VP9_PROB_COST_SHIFT) {
        // libvpx `:163`: `for (newp = *bestp; (newp-oldp)*step_sign < 0; newp += step)`.
        let mut newp = i64::from(start_newp);
        while (newp - i64::from(oldp)) * step_sign < 0 {
            if !(1..=255).contains(&newp) {
                newp += step;
                continue;
            }
            let newplist = &VP9_PARETO8_FULL[(newp as usize) - 1];
            let mut new_b = cost_branch256(branch_ct[PIVOT_NODE], newp as u8);
            for i in UNCONSTRAINED_NODES..ENTROPY_NODES {
                new_b += cost_branch256(branch_ct[i], newplist[i - UNCONSTRAINED_NODES]);
            }
            let update_b = prob_diff_update_cost(newp as u8, oldp) + upd_cost;
            let savings = old_b - new_b - update_b;
            if savings > best_savings {
                best_savings = savings;
                best_newp = newp as u8;
            }
            newp += step;
        }
    }

    SavingsSearch {
        best_newp,
        savings: best_savings,
    }
}

const _: () = assert!(MODEL_NODES == ENTROPY_NODES - UNCONSTRAINED_NODES);

/// `tx_totals_gate(tx_total)` — the libvpx outer-loop gate that
/// suppresses the entire per-TX-size coef-prob update when there
/// weren't enough tokens this frame to feed a meaningful update.
/// Mirrors libvpx `vp9_bitstream.c:690`
/// (`cpi->td.counts->tx.tx_totals[tx_size] <= 20`).
///
/// Returns `true` when the encoder must skip the savings search and
/// emit a single `0` bit.
#[inline]
#[must_use]
pub fn tx_totals_gate_skips(tx_total: u32) -> bool {
    tx_total <= 20
}

/// `BAND_COEFF_CONTEXTS(band)` (`vp9_entropy.h:108`): band 0 uses only
/// the first 3 of `COEFF_CONTEXTS`, bands 1..5 use all 6.
#[inline]
const fn band_coeff_contexts(band: usize) -> usize {
    if band == 0 { 3 } else { COEFF_CONTEXTS }
}

/// A6 anti-G3 observability — number of per-TX-size coef-prob tables
/// the TWO_LOOP plan UPDATED (emitted gate bit 1 + ≥1 slot update)
/// across the encode. Incremented in [`build_coef_prob_update_plan`].
pub static COEF_PROB_TABLES_UPDATED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// A6 anti-G3 observability — number of per-TX-size coef-prob tables
/// the TWO_LOOP plan SUPPRESSED **despite at least one slot whose ideal
/// prob differs from `oldp`** (the aggregate `savings < 0`, so the whole
/// table is gated off). This is the case "Strategy B"
/// (force-update-on-diff) can NEVER produce: Strategy B emits gate bit 1
/// whenever ANY slot differs. A non-zero value here on a real encode is
/// the live-TWO_LOOP proof.
pub static COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// A6 — build the per-frame TWO_LOOP coefficient-probability update
/// plan from THIS frame's full 12-token counts. The live replacement
/// for "Strategy B" (force-update-on-diff): runs libvpx's
/// `update_coef_probs` (`vp9_bitstream.c:683-699`) +
/// `update_coef_probs_common` TWO_LOOP dry-run (`:556-593`) to decide,
/// per TX size and per slot, whether to signal a coefficient-prob
/// update and to what value.
///
/// Inputs:
/// - `full_counts` — the encoder-only full 12-token histogram
///   (`Coef12Bank`) gathered by the count pass.
/// - `eob_branch` — the per-`(tx,plane,ref)` EOB-branch tally from the
///   same count pass (libvpx `cm->counts.eob_branch`).
/// - `old_probs` — the PRE-frame coefficient probabilities
///   (`cpi->common.fc->coef_probs`, i.e. the decoder's FC bank state
///   = `decoder_fc.coef_probs`). These are libvpx's `oldp`.
/// - `tx_mode` — gates which TX sizes are in the wire format.
///
/// The returned [`CoefProbUpdatePlan`] is consumed by
/// `write_compressed_header(.., Some(&plan))`: the writer emits the
/// gate bits + per-slot updates and mutates its `pre`/`decoder_fc`
/// in place to the post-update probs (so the encoder mirror tracks the
/// decoder exactly). `new_coef_probs` and `frame_branch_ct` are derived
/// per slot from the counts via [`build_coef_branch_cts`] (libvpx
/// `build_tree_distribution`, `:518-543`).
#[must_use]
pub fn build_coef_prob_update_plan(
    full_counts: &Coef12Bank,
    eob_branch: &[[[EobBranchCountsBlock; REF_TYPES]; PLANE_TYPES]; TX_SIZES],
    old_probs: &[CoeffProbsModel; TX_SIZES],
    tx_mode: TxMode,
) -> CoefProbUpdatePlan {
    // `coeff_prob_appx_step = 1` — the reference TWO_LOOP step
    // (`vp9_speed_features.c:933`).
    const STEPSIZE: i32 = 1;
    let max_tx = tx_mode_to_biggest_tx_size(tx_mode);
    let zero_cost = i64::from(vp9_cost_zero(DIFF_UPDATE_PROB));

    let mut plan: CoefProbUpdatePlan = Default::default();
    for (tx_idx, tx_plan) in plan.iter_mut().enumerate() {
        if tx_idx > max_tx {
            // Not in the wire format; the writer skips these TX sizes
            // (asserts pre == post). Leave the plan entry default.
            continue;
        }

        // libvpx `update_coef_probs` outer gate (`vp9_bitstream.c:690`):
        // `tx.tx_totals[tx_size] <= 20` ⇒ write a single `0` and skip.
        // rvp9's `tx_totals` does not count Tx4x4 blocks, so we instead
        // gate on the total token mass for this TX size (the gate's
        // intent: "too few tokens to bother"). A zeroed table (the KF
        // pattern, or a TX size with no coded blocks) is suppressed.
        let mut tx_total: u32 = 0;
        for plane_counts in full_counts[tx_idx].iter().take(PLANE_TYPES) {
            for ref_counts in plane_counts.iter().take(REF_TYPES) {
                for (band, band_counts) in ref_counts.iter().enumerate().take(COEF_BANDS) {
                    for ctx_counts in band_counts.iter().take(band_coeff_contexts(band)) {
                        for &tok_count in ctx_counts.iter().take(ENTROPY_TOKENS) {
                            tx_total = tx_total.saturating_add(tok_count);
                        }
                    }
                }
            }
        }
        if tx_totals_gate_skips(tx_total) {
            tx_plan.suppress = true;
            tx_plan.slots.clear();
            continue;
        }

        // Per-slot dry run (libvpx `:560-586`): walk
        // plane × ref × band × band_coeff_contexts(band) ×
        // UNCONSTRAINED_NODES, accumulating `savings` and the per-slot
        // `(update, newp)` decision so the writer can replay it.
        let mut slots: Vec<CoefSlotUpdate> = Vec::new();
        let mut savings_total: i64 = 0;
        let mut updates: usize = 0;
        // anti-G3: did ANY slot's ideal prob differ from `oldp`? That is
        // exactly the condition under which Strategy B would force a
        // gate-1 update. If the table is then SUPPRESSED (savings < 0),
        // TWO_LOOP diverges from Strategy B.
        let mut any_candidate_diff = false;
        for plane in 0..PLANE_TYPES {
            for r in 0..REF_TYPES {
                for band in 0..COEF_BANDS {
                    for ctx in 0..band_coeff_contexts(band) {
                        let counts = &full_counts[tx_idx][plane][r][band][ctx];
                        let eob_total = eob_branch[tx_idx][plane][r][band][ctx];
                        // build_tree_distribution body: branch counts +
                        // the 3 "ideal" unconstrained probs.
                        let (branch_ct, ideal) = build_coef_branch_cts(counts, eob_total);
                        let oldp_slot = &old_probs[tx_idx][plane][r][band][ctx];
                        for t in 0..UNCONSTRAINED_NODES {
                            let oldp = oldp_slot[t];
                            let start_newp = ideal[t];
                            let search = if t == PIVOT_NODE {
                                prob_diff_update_savings_search_model(
                                    &branch_ct,
                                    oldp,
                                    start_newp,
                                    DIFF_UPDATE_PROB,
                                    STEPSIZE,
                                )
                            } else {
                                prob_diff_update_savings_search(
                                    branch_ct[t],
                                    oldp,
                                    start_newp,
                                    DIFF_UPDATE_PROB,
                                )
                            };
                            let u = search.savings > 0 && search.best_newp != oldp;
                            // A slot whose ideal (search) prob differs
                            // from oldp is one Strategy B would signal.
                            if start_newp != oldp || search.best_newp != oldp {
                                any_candidate_diff = true;
                            }
                            if u {
                                savings_total += search.savings - zero_cost;
                                updates += 1;
                            } else {
                                savings_total -= zero_cost;
                            }
                            slots.push(CoefSlotUpdate {
                                update: u,
                                newp: if u { search.best_newp } else { oldp },
                            });
                        }
                    }
                }
            }
        }

        // libvpx `:590`: `update[1] == 0 || savings < 0` ⇒ suppress.
        use std::sync::atomic::Ordering::Relaxed;
        if updates == 0 || savings_total < 0 {
            tx_plan.suppress = true;
            tx_plan.slots.clear();
            // anti-G3: a table SUPPRESSED despite a would-update slot is
            // the Strategy-B-impossible outcome.
            if any_candidate_diff {
                COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF.fetch_add(1, Relaxed);
            }
        } else {
            tx_plan.suppress = false;
            tx_plan.slots = slots;
            COEF_PROB_TABLES_UPDATED.fetch_add(1, Relaxed);
        }
    }
    plan
}

/// One per-slot count pair, ordered `[zero_count, one_count]`.
pub type BranchCt = [u32; 2];

/// W7.1 TWO_LOOP coef-prob updater for one TX size.
///
/// Inputs:
/// - `bc` — the compressed-header bit writer.
/// - `old_probs` — current per-slot probability (mutated in place
///   to track the post-update value).
/// - `new_probs` — the "ideal" per-slot probability the search
///   starts from (caller normally derives this from frame branch
///   counts).
/// - `branch_ct` — per-slot `[zero_count, one_count]`.
///
/// All three arrays have the same length `N`. `N` is set by the
/// caller; libvpx walks
/// `PLANE_TYPES × REF_TYPES × COEF_BANDS × BAND_COEFF_CONTEXTS(k) ×
/// UNCONSTRAINED_NODES`, but this helper is generic in `N` so the
/// tests can exercise it on smaller slices.
///
/// Wire shape mirrors libvpx `update_coef_probs_common`
/// (`vp9_bitstream.c:545-625`, TWO_LOOP case):
///
/// 1. Dry-run pass accumulates `savings` and counts how many slots
///    would update.
/// 2. If `updated == 0 || savings < 0`, emit gate=0 and return.
/// 3. Otherwise emit gate=1 then, per slot, emit a gate bit at
///    `DIFF_UPDATE_PROB`, and on `1` emit the `delp` payload.
///
/// Returns the number of slots that ended up emitting an update.
///
/// ## Deviation
///
/// libvpx's PIVOT_NODE / model-search branch (the
/// `vp9_prob_diff_update_savings_search_model` path) is not exercised
/// here; this helper only ports the simpler
/// `vp9_prob_diff_update_savings_search` path (the non-pivot node
/// case in libvpx's nested loop). The model search is a separate
/// W7.1 follow-up — see libvpx `vp9_subexp.c:141-181`.
pub fn update_coef_probs_common(
    bc: &mut BoolWriter<'_>,
    old_probs: &mut [u8],
    new_probs: &[u8],
    branch_ct: &[BranchCt],
) -> usize {
    assert_eq!(old_probs.len(), new_probs.len());
    assert_eq!(old_probs.len(), branch_ct.len());
    let n = old_probs.len();

    // Pre-compute per-slot savings + post-search newp so the dry-run
    // and the emit pass agree. libvpx repeats the search; we cache it
    // to avoid double work without changing wire output.
    let mut searches: Vec<SavingsSearch> = Vec::with_capacity(n);
    let mut savings_total: i64 = 0;
    let mut updates: usize = 0;
    let zero_cost = i64::from(vp9_cost_zero(DIFF_UPDATE_PROB));

    for i in 0..n {
        let s = prob_diff_update_savings_search(
            branch_ct[i],
            old_probs[i],
            new_probs[i],
            DIFF_UPDATE_PROB,
        );
        let u = (s.savings > 0 && s.best_newp != old_probs[i]) as u8;
        if u == 1 {
            savings_total += s.savings - zero_cost;
            updates += 1;
        } else {
            savings_total -= zero_cost;
        }
        searches.push(s);
    }

    // libvpx `vp9_bitstream.c:590`: skip the whole block.
    if updates == 0 || savings_total < 0 {
        bc.write_bit(0);
        return 0;
    }

    // Emit gate=1 + per-slot diff-updates. libvpx `vp9_bitstream.c:594-622`.
    bc.write_bit(1);
    let mut emitted = 0usize;
    for i in 0..n {
        let s = searches[i];
        let oldp = old_probs[i];
        let u = (s.savings > 0 && s.best_newp != oldp) as u8;
        bc.write(u, DIFF_UPDATE_PROB);
        if u == 1 {
            write_prob_diff_update(bc, s.best_newp, oldp);
            old_probs[i] = s.best_newp;
            emitted += 1;
        }
    }
    emitted
}

// =====================================================================
// Same-frame non-coefficient probability updates
// =====================================================================

#[inline]
fn update_binary_target(oldp: u8, ct: [u32; 2]) -> u8 {
    let ideal = get_binary_prob(ct[0], ct[1]);
    let search = prob_diff_update_savings_search(ct, oldp, ideal, DIFF_UPDATE_PROB);
    if search.savings > 0 {
        search.best_newp
    } else {
        oldp
    }
}

fn update_tree_target_impl(
    node: usize,
    tree: &[i8],
    counts: &[u32],
    old_probs: &[u8],
    target_probs: &mut [u8],
) -> u32 {
    let left = tree[node];
    let left_count = if left <= 0 {
        counts[(-left) as usize]
    } else {
        update_tree_target_impl(left as usize, tree, counts, old_probs, target_probs)
    };
    let right = tree[node + 1];
    let right_count = if right <= 0 {
        counts[(-right) as usize]
    } else {
        update_tree_target_impl(right as usize, tree, counts, old_probs, target_probs)
    };
    let prob_idx = node >> 1;
    target_probs[prob_idx] = update_binary_target(old_probs[prob_idx], [left_count, right_count]);
    left_count + right_count
}

fn update_tree_target(tree: &[i8], counts: &[u32], old_probs: &[u8], target_probs: &mut [u8]) {
    let _ = update_tree_target_impl(0, tree, counts, old_probs, target_probs);
}

#[inline]
fn update_mv_target(oldp: u8, ct: [u32; 2]) -> u8 {
    let newp = get_binary_prob(ct[0], ct[1]) | 1;
    let old_cost = cost_branch256(ct, oldp) + i64::from(vp9_cost_zero(MV_UPDATE_PROB));
    let new_cost = cost_branch256(ct, newp)
        + i64::from(vp9_cost_one(MV_UPDATE_PROB))
        + (7_i64 << VP9_PROB_COST_SHIFT);
    if old_cost > new_cost { newp } else { oldp }
}

fn update_mv_tree_target(tree: &[i8], counts: &[u32], old_probs: &[u8], target_probs: &mut [u8]) {
    fn visit(
        node: usize,
        tree: &[i8],
        counts: &[u32],
        old_probs: &[u8],
        target_probs: &mut [u8],
    ) -> u32 {
        let left = tree[node];
        let left_count = if left <= 0 {
            counts[(-left) as usize]
        } else {
            visit(left as usize, tree, counts, old_probs, target_probs)
        };
        let right = tree[node + 1];
        let right_count = if right <= 0 {
            counts[(-right) as usize]
        } else {
            visit(right as usize, tree, counts, old_probs, target_probs)
        };
        let prob_idx = node >> 1;
        target_probs[prob_idx] = update_mv_target(old_probs[prob_idx], [left_count, right_count]);
        left_count + right_count
    }
    let _ = visit(0, tree, counts, old_probs, target_probs);
}

/// Build the compressed-header probability target selected from this frame's
/// non-coefficient counts. Mirrors the current-frame updates in libvpx
/// `write_compressed_header`; coefficient updates remain on their dedicated
/// TWO_LOOP plan.
#[must_use]
pub fn build_noncoef_prob_update_target(
    pre: &FrameContext,
    counts: &FrameCounts,
    reference_mode: ReferenceMode,
    interp_switchable: bool,
    tx_mode_select: bool,
    allow_hp: bool,
) -> FrameContext {
    let mut target = pre.clone();

    if tx_mode_select {
        for ctx in 0..TX_SIZE_CONTEXTS {
            let c8 = counts.tx.p8x8[ctx];
            target.tx_probs.p8x8[ctx][0] =
                update_binary_target(pre.tx_probs.p8x8[ctx][0], [c8[0], c8[1]]);
            let c16 = counts.tx.p16x16[ctx];
            let b16 = [[c16[0], c16[1] + c16[2]], [c16[1], c16[2]]];
            for (slot, ct) in b16.into_iter().enumerate() {
                target.tx_probs.p16x16[ctx][slot] =
                    update_binary_target(pre.tx_probs.p16x16[ctx][slot], ct);
            }
            let c32 = counts.tx.p32x32[ctx];
            let b32 = [
                [c32[0], c32[1] + c32[2] + c32[3]],
                [c32[1], c32[2] + c32[3]],
                [c32[2], c32[3]],
            ];
            for (slot, ct) in b32.into_iter().enumerate() {
                target.tx_probs.p32x32[ctx][slot] =
                    update_binary_target(pre.tx_probs.p32x32[ctx][slot], ct);
            }
        }
    }

    for ctx in 0..SKIP_CONTEXTS {
        target.skip_probs[ctx] = update_binary_target(pre.skip_probs[ctx], counts.skip[ctx]);
    }
    for ctx in 0..INTER_MODE_CONTEXTS {
        update_tree_target(
            &VP9_INTER_MODE_TREE,
            &counts.inter_mode[ctx],
            &pre.inter_mode_probs[ctx],
            &mut target.inter_mode_probs[ctx],
        );
    }
    if interp_switchable {
        for ctx in 0..SWITCHABLE_FILTER_CONTEXTS {
            update_tree_target(
                &VP9_SWITCHABLE_INTERP_TREE,
                &counts.switchable_interp[ctx],
                &pre.switchable_interp_prob[ctx],
                &mut target.switchable_interp_prob[ctx],
            );
        }
    }
    for ctx in 0..INTRA_INTER_CONTEXTS {
        target.intra_inter_prob[ctx] =
            update_binary_target(pre.intra_inter_prob[ctx], counts.intra_inter[ctx]);
    }
    if reference_mode == ReferenceMode::Select {
        for ctx in 0..target.comp_inter_prob.len() {
            target.comp_inter_prob[ctx] =
                update_binary_target(pre.comp_inter_prob[ctx], counts.comp_inter[ctx]);
        }
    }
    if reference_mode != ReferenceMode::Compound {
        for ctx in 0..REF_CONTEXTS {
            for branch in 0..2 {
                target.single_ref_prob[ctx][branch] = update_binary_target(
                    pre.single_ref_prob[ctx][branch],
                    counts.single_ref[ctx][branch],
                );
            }
        }
    }
    if reference_mode != ReferenceMode::Single {
        for ctx in 0..REF_CONTEXTS {
            target.comp_ref_prob[ctx] =
                update_binary_target(pre.comp_ref_prob[ctx], counts.comp_ref[ctx]);
        }
    }
    for group in 0..BLOCK_SIZE_GROUPS {
        update_tree_target(
            &VP9_INTRA_MODE_TREE,
            &counts.y_mode[group],
            &pre.y_mode_prob[group],
            &mut target.y_mode_prob[group],
        );
    }
    for ctx in 0..PARTITION_CONTEXTS {
        update_tree_target(
            &VP9_PARTITION_TREE,
            &counts.partition[ctx],
            &pre.partition_prob[ctx],
            &mut target.partition_prob[ctx],
        );
    }

    update_mv_tree_target(
        &VP9_MV_JOINT_TREE,
        &counts.mv.joints,
        &pre.nmvc.joints,
        &mut target.nmvc.joints,
    );
    for component in 0..2 {
        let old = &pre.nmvc.comps[component];
        let out = &mut target.nmvc.comps[component];
        let ct = &counts.mv.comps[component];
        out.sign = update_mv_target(old.sign, ct.sign);
        update_mv_tree_target(
            &VP9_MV_CLASS_TREE,
            &ct.classes,
            &old.classes,
            &mut out.classes,
        );
        update_mv_tree_target(
            &VP9_MV_CLASS0_TREE,
            &ct.class0,
            &old.class0,
            &mut out.class0,
        );
        for bit in 0..MV_OFFSET_BITS {
            out.bits[bit] = update_mv_target(old.bits[bit], ct.bits[bit]);
        }
        for class0 in 0..CLASS0_SIZE {
            update_mv_tree_target(
                &VP9_MV_FP_TREE,
                &ct.class0_fp[class0],
                &old.class0_fp[class0],
                &mut out.class0_fp[class0],
            );
        }
        update_mv_tree_target(&VP9_MV_FP_TREE, &ct.fp, &old.fp, &mut out.fp);
        if allow_hp {
            out.class0_hp = update_mv_target(old.class0_hp, ct.class0_hp);
            out.hp = update_mv_target(old.hp, ct.hp);
        }
    }

    target
}

// =====================================================================
// W7.2 — fix_interp_filter
// =====================================================================

/// `interp_filter_selected[ctx][filter]` counters — per-tile,
/// summed across all tiles by the caller. Mirrors libvpx's
/// `cpi->td.counts->switchable_interp[ctx][filter]` shape after the
/// per-tile worker reduction.
pub type SwitchableInterpCounts = [[u32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS];

/// Outcome of running [`fix_interp_filter`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixInterpFilter {
    /// Frame stays SWITCHABLE — at least two filters were observed.
    KeepSwitchable,
    /// Only one filter was used — collapse to that fixed filter.
    Collapse(InterpFilter),
}

/// Collapse a SWITCHABLE frame-level interp filter to a fixed filter
/// when only one of the three switchable filters was actually picked
/// across all per-context counters. Mirrors libvpx
/// `fix_interp_filter` (`vp9_bitstream.c:863-884`).
///
/// On non-SWITCHABLE input, returns `KeepSwitchable` (the libvpx
/// helper just no-ops in that case; we treat that as "no change").
#[must_use]
pub fn fix_interp_filter(
    frame_filter: InterpFilter,
    counts: &SwitchableInterpCounts,
) -> FixInterpFilter {
    if frame_filter != InterpFilter::Switchable {
        return FixInterpFilter::KeepSwitchable;
    }

    // libvpx `vp9_bitstream.c:866-873`: per-filter sum across the
    // SWITCHABLE_FILTER_CONTEXTS counter rows; `c` counts how many
    // filters had a non-zero total.
    let mut per_filter = [0u32; SWITCHABLE_FILTERS];
    let mut nonzero = 0usize;
    for (i, slot) in per_filter.iter_mut().enumerate().take(SWITCHABLE_FILTERS) {
        let mut sum = 0u32;
        for row in counts.iter().take(SWITCHABLE_FILTER_CONTEXTS) {
            sum = sum.wrapping_add(row[i]);
        }
        *slot = sum;
        if sum > 0 {
            nonzero += 1;
        }
    }

    if nonzero == 1 {
        // libvpx `vp9_bitstream.c:874-882`: collapse to the one used
        // filter (first non-zero slot in declaration order).
        for (i, &sum) in per_filter.iter().enumerate().take(SWITCHABLE_FILTERS) {
            if sum > 0 {
                let filter = match i {
                    0 => InterpFilter::EightTap,
                    1 => InterpFilter::EightTapSmooth,
                    _ => InterpFilter::EightTapSharp,
                };
                return FixInterpFilter::Collapse(filter);
            }
        }
        unreachable!("nonzero == 1 but no per_filter slot > 0");
    }

    FixInterpFilter::KeepSwitchable
}

// =====================================================================
// W7.3 — vp9_get_refresh_mask (preserve_existing_gf + multi-ARF stack)
// =====================================================================

/// Inputs to [`get_refresh_mask`] — a lean shape mirroring the slice
/// of `VP9_COMP` that libvpx's helper reads. Bit-true behaviour
/// against libvpx requires the caller populate all fields the same
/// way the C encoder does; see `vp9_bitstream.c:902-940` and
/// `vp9_bitstream.h:41-44` for the exact relationships.
#[derive(Clone, Debug)]
pub struct RefreshMaskInputs {
    /// `cpi->refresh_last_frame` — bool flag.
    pub refresh_last_frame: bool,
    /// `cpi->refresh_golden_frame` — bool flag.
    pub refresh_golden_frame: bool,
    /// `cpi->refresh_alt_ref_frame` — bool flag.
    pub refresh_alt_ref_frame: bool,
    /// `cpi->rc.is_src_frame_alt_ref` — true on ARF-overlap frames.
    /// Drives the `vp9_preserve_existing_gf` branch.
    pub is_src_frame_alt_ref: bool,
    /// `cpi->use_svc`. Disables the preserve-GF branch when set.
    pub use_svc: bool,
    /// `cpi->lst_fb_idx` — last-frame slot in the 8-slot pool
    /// (`0..REF_FRAMES`).
    pub lst_fb_idx: u32,
    /// `cpi->gld_fb_idx` — golden-frame slot.
    pub gld_fb_idx: u32,
    /// `cpi->alt_fb_idx` — alt-ref slot.
    pub alt_fb_idx: u32,
    /// `cpi->multi_layer_arf` — enables the alt-slot search loop.
    pub multi_layer_arf: bool,
    /// `cpi->twopass.gf_group.arf_index_stack` — the multi-ARF stack
    /// in libvpx. Only the first `stack_size` entries are meaningful.
    pub arf_index_stack: Vec<u32>,
    /// `cpi->twopass.gf_group.stack_size`.
    pub stack_size: usize,
    /// SVC override: when `Some(slot_mask)`, libvpx
    /// `vp9_bitstream.c:933-935` short-circuits and returns the
    /// caller-supplied refresh slot mask directly.
    pub svc_update_buffer_slot: Option<u8>,
}

/// Outcome of [`get_refresh_mask`] — the 8-bit refresh mask plus the
/// `top_arf_idx` libvpx records on `cpi->twopass.gf_group`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefreshMask {
    /// The 8-bit `refresh_frame_flags` field.
    pub mask: u8,
    /// libvpx writes this back to
    /// `cpi->twopass.gf_group.top_arf_idx`. Always set even when the
    /// preserve-GF branch fires (libvpx sets it in the `else`
    /// branch only, but exposing it unconditionally simplifies the
    /// PROOF tests).
    pub top_arf_idx: u32,
}

/// Returns `true` when libvpx's `vp9_preserve_existing_gf` inline
/// returns `true`. Mirrors `vp9_bitstream.h:41-44`.
#[inline]
#[must_use]
pub fn preserve_existing_gf(inputs: &RefreshMaskInputs) -> bool {
    inputs.refresh_golden_frame && inputs.is_src_frame_alt_ref && !inputs.use_svc
}

/// Compute the 8-bit refresh mask. Mirrors libvpx
/// `vp9_get_refresh_mask` (`vp9_bitstream.c:902-940`).
///
/// Wire-output is the bitwise-OR of:
/// - `(refresh_last_frame as u8) << lst_fb_idx`
/// - `(refresh_golden_frame as u8) << gld_fb_idx`  (or `alt_fb_idx`
///   in the preserve-GF branch)
/// - `(refresh_alt_ref_frame as u8) << arf_idx`     (else branch only)
///
/// The `multi_layer_arf` branch searches for an ARF slot that is
/// neither LAST/GOLD/ALT nor already on the stack. libvpx searches
/// `0..REF_FRAMES` and picks the first viable index; if none is
/// found, falls back to `alt_fb_idx`. The encoder records the chosen
/// index as `top_arf_idx`.
#[must_use]
pub fn get_refresh_mask(inputs: &RefreshMaskInputs) -> RefreshMask {
    assert!((inputs.lst_fb_idx as usize) < REF_FRAMES);
    assert!((inputs.gld_fb_idx as usize) < REF_FRAMES);
    assert!((inputs.alt_fb_idx as usize) < REF_FRAMES);

    if preserve_existing_gf(inputs) {
        // libvpx `vp9_bitstream.c:903-915`: GOLDEN bit goes into the
        // ARF slot because the encoder is about to swap ARF/GF.
        let mask = ((inputs.refresh_last_frame as u8) << inputs.lst_fb_idx)
            | ((inputs.refresh_golden_frame as u8) << inputs.alt_fb_idx);
        return RefreshMask {
            mask,
            top_arf_idx: inputs.alt_fb_idx,
        };
    }

    // libvpx `vp9_bitstream.c:916-931`: multi-ARF slot search.
    let mut arf_idx = inputs.alt_fb_idx;
    if inputs.multi_layer_arf {
        for candidate in 0..(REF_FRAMES as u32) {
            if candidate == inputs.alt_fb_idx
                || candidate == inputs.lst_fb_idx
                || candidate == inputs.gld_fb_idx
            {
                continue;
            }
            let mut hit_stack = false;
            for s in 0..inputs.stack_size {
                if candidate == inputs.arf_index_stack[s] {
                    hit_stack = true;
                    break;
                }
            }
            if !hit_stack {
                arf_idx = candidate;
                break;
            }
        }
    }

    // libvpx `vp9_bitstream.c:933-935` — SVC bypass.
    if let Some(slot_mask) = inputs.svc_update_buffer_slot {
        return RefreshMask {
            mask: slot_mask,
            top_arf_idx: arf_idx,
        };
    }

    // libvpx `vp9_bitstream.c:936-938` — the standard 3-way OR.
    let mask = ((inputs.refresh_last_frame as u8) << inputs.lst_fb_idx)
        | ((inputs.refresh_golden_frame as u8) << inputs.gld_fb_idx)
        | ((inputs.refresh_alt_ref_frame as u8) << arf_idx);

    RefreshMask {
        mask,
        top_arf_idx: arf_idx,
    }
}

// =====================================================================
// Tests — six PROOF tests across W7.1 + W7.2 + W7.3 (W7.4 lives in
// loopfilter.rs).
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- W7.1 -----

    #[test]
    fn same_frame_noncoef_target_is_identity_without_symbols() {
        let pre = FrameContext::new_default();
        let target = build_noncoef_prob_update_target(
            &pre,
            &FrameCounts::default(),
            ReferenceMode::Single,
            true,
            true,
            true,
        );
        assert_eq!(target, pre);
    }

    #[test]
    fn same_frame_noncoef_target_updates_every_emitted_probability_family() {
        let pre = FrameContext::new_default();
        let mut counts = FrameCounts::default();
        counts.tx.p8x8[0] = [10_000, 1];
        counts.tx.p16x16[0] = [10_000, 1, 1];
        counts.tx.p32x32[0] = [10_000, 1, 1, 1];
        counts.skip[0] = [10_000, 1];
        counts.inter_mode[0][3] = 10_000;
        counts.switchable_interp[0][2] = 10_000;
        counts.intra_inter[0] = [10_000, 1];
        counts.single_ref[0][0] = [10_000, 1];
        counts.single_ref[0][1] = [1, 10_000];
        counts.y_mode[0][9] = 10_000;
        counts.partition[0][3] = 10_000;
        counts.mv.joints[3] = 10_000;
        counts.mv.comps[0].sign = [10_000, 1];
        counts.mv.comps[0].classes[10] = 10_000;
        counts.mv.comps[0].class0[1] = 10_000;
        counts.mv.comps[0].bits[0] = [10_000, 1];
        counts.mv.comps[0].class0_fp[0][3] = 10_000;
        counts.mv.comps[0].fp[3] = 10_000;
        counts.mv.comps[0].class0_hp = [10_000, 1];
        counts.mv.comps[0].hp = [1, 10_000];

        let target = build_noncoef_prob_update_target(
            &pre,
            &counts,
            ReferenceMode::Single,
            true,
            true,
            true,
        );

        assert_ne!(target.tx_probs, pre.tx_probs);
        assert_ne!(target.skip_probs, pre.skip_probs);
        assert_ne!(target.inter_mode_probs, pre.inter_mode_probs);
        assert_ne!(target.switchable_interp_prob, pre.switchable_interp_prob);
        assert_ne!(target.intra_inter_prob, pre.intra_inter_prob);
        assert_ne!(target.single_ref_prob, pre.single_ref_prob);
        assert_ne!(target.y_mode_prob, pre.y_mode_prob);
        assert_ne!(target.partition_prob, pre.partition_prob);
        assert_ne!(target.nmvc, pre.nmvc);
        assert_eq!(target.comp_inter_prob, pre.comp_inter_prob);
        assert_eq!(target.comp_ref_prob, pre.comp_ref_prob);
        assert_eq!(target.uv_mode_prob, pre.uv_mode_prob);
        assert_eq!(target.coef_probs, pre.coef_probs);
    }

    /// W7.1 PROOF — `vp9_prob_diff_update_savings_search`
    /// (`vp9_subexp.c:117-139`).
    ///
    /// Synthetic branch counts heavily skewed toward `0` make the
    /// "true" optimal probability much higher than the libvpx default
    /// `oldp = 128`. The search must walk down from a low `start_newp`
    /// toward `oldp` and pick something `< 128` that yields positive
    /// savings — i.e. a non-default `best_newp`. We assert
    /// `best_newp != oldp` AND `savings > 0`.
    #[test]
    fn w7_1_savings_search_finds_better_prob_for_skewed_counts() {
        // 10_000 zeros vs 50 ones — true ratio is ~99.5% zero, so
        // the best `p` is near 255. Walking from start_newp = 250
        // toward oldp = 128 must land on something in that high
        // band.
        let ct: BranchCt = [10_000, 50];
        let oldp: u8 = 128;
        let start_newp: u8 = 250;
        let out = prob_diff_update_savings_search(ct, oldp, start_newp, DIFF_UPDATE_PROB);
        assert_ne!(
            out.best_newp, oldp,
            "search must move away from oldp on heavily skewed counts"
        );
        assert!(
            out.savings > 0,
            "savings must be positive ({} ≤ 0)",
            out.savings
        );
        assert!(
            out.best_newp > 128,
            "skew is toward zero, best_newp should be high (got {})",
            out.best_newp
        );
    }

    /// W7.1 PROOF — `tx_totals_gate` (`vp9_bitstream.c:690-692`).
    ///
    /// `tx_total <= 20` ⇒ libvpx writes a single `0` gate bit and
    /// skips the savings search entirely. We model that by calling
    /// `tx_totals_gate_skips` on borderline + below-threshold values
    /// and confirming the gate decision.
    #[test]
    fn w7_1_update_coef_probs_common_emits_no_update_when_tx_totals_below_threshold() {
        // libvpx `vp9_bitstream.c:690` threshold is `<= 20`.
        assert!(tx_totals_gate_skips(0), "0 tokens must skip");
        assert!(tx_totals_gate_skips(20), "exactly 20 tokens must skip");
        assert!(
            !tx_totals_gate_skips(21),
            "21 tokens must NOT skip (gate fires only at <= 20)"
        );

        // Independent check on `update_coef_probs_common`: with all
        // counts == zero, `savings < 0` because every slot pays the
        // gate-zero cost with no income, so the function MUST emit
        // gate=0 and return `updates == 0`.
        let mut buf = vec![0u8; 256];
        let mut w = BoolWriter::start_encode(&mut buf);
        let mut old_probs: Vec<u8> = vec![128; 8];
        let new_probs: Vec<u8> = vec![128; 8];
        let branch_ct: Vec<BranchCt> = vec![[0, 0]; 8];
        let updates = update_coef_probs_common(&mut w, &mut old_probs, &new_probs, &branch_ct);
        let _ = w.stop_encode().unwrap();
        assert_eq!(updates, 0, "no-counts input must emit zero updates");
        // First bit on the wire must be `0` — but the bool writer
        // doesn't expose direct introspection; the surrogate check is
        // `old_probs` unchanged.
        assert_eq!(old_probs, vec![128u8; 8]);
    }

    /// W-coef-prob PROOF — `build_coef_branch_cts` /
    /// `build_tree_distribution` (`vp9_bitstream.c:518-543`,
    /// `vp9_treewriter.c:34-58`).
    ///
    /// DIFFERENTIAL ORACLE: the three UNCONSTRAINED_NODES branch counts
    /// the tree produces MUST equal the model-token derivation already
    /// used by `rvp9_core::entropy::adapt::adapt_coef_probs`
    /// (`adapt.rs:614-618`):
    ///   node0 (EOB):  [#eob,  eob_total - #eob]
    ///   node1 (ZERO): [#zero, #one + #two+]
    ///   node2 (ONE):  [#one,  #two+]
    /// where `#two+` sums TWO..CAT6. Two independent paths (full-12-token
    /// tree walk vs. model-token folding) must agree — that is the
    /// oracle. The test also asserts the 8 MODEL nodes (3..10) carry the
    /// within-TWO+ distribution the model search needs (non-zero here),
    /// which the model-token folding alone CANNOT recover.
    #[test]
    fn w_coef_build_tree_distribution_matches_model_token_derivation() {
        // Full 12-token histogram. Token values:
        // ZERO=0 ONE=1 TWO=2 THREE=3 FOUR=4 CAT1=5 CAT2=6 CAT3=7
        // CAT4=8 CAT5=9 CAT6=10 EOB=11.
        let counts: [u32; ENTROPY_TOKENS] = [
            900, // ZERO
            300, // ONE
            120, // TWO
            60,  // THREE
            30,  // FOUR
            20,  // CAT1
            10,  // CAT2
            8,   // CAT3
            4,   // CAT4
            2,   // CAT5
            1,   // CAT6
            500, // EOB
        ];
        // eob_branch_total = every position an EOB decision was
        // attempted = (#non-eob tokens) + (#eob). Non-eob = all but EOB.
        let non_eob: u32 = counts[0..11].iter().sum();
        let eob_total = non_eob + counts[11];

        let (branch_ct, probs) = build_coef_branch_cts(&counts, eob_total);

        // --- adapt.rs model-token derivation (the oracle) ---
        let n0 = counts[ZERO_TOKEN_IDX];
        let n1 = counts[ONE_TOKEN_IDX];
        let n2: u32 = counts[2..11].iter().sum(); // TWO..CAT6
        let neob = counts[11];
        let expect = [
            [neob, eob_total - neob], // node 0 (EOB)
            [n0, n1 + n2],            // node 1 (ZERO)
            [n1, n2],                 // node 2 (ONE / PIVOT)
        ];
        for m in 0..UNCONSTRAINED_NODES {
            assert_eq!(
                branch_ct[m], expect[m],
                "unconstrained node {m} branch count must match the \
                 adapt.rs model-token derivation"
            );
            assert_eq!(
                probs[m],
                get_binary_prob(expect[m][0], expect[m][1]),
                "node {m} ideal prob must be get_binary_prob of the branch ct"
            );
        }

        // --- model nodes 3..10 must be populated (the info the
        // 4-token model loses) ---
        let model_sum: u32 = (UNCONSTRAINED_NODES..ENTROPY_NODES)
            .map(|i| branch_ct[i][0] + branch_ct[i][1])
            .sum();
        assert!(
            model_sum > 0,
            "model nodes 3..10 must carry the within-TWO+ distribution \
             (got all-zero — the tree walk failed)"
        );
        // node 3 (LOW_VAL: TWO|THREE|FOUR vs CAT1..CAT6) split.
        // left = TWO+THREE+FOUR, right = CAT1..CAT6.
        let low = counts[2] + counts[3] + counts[4];
        let high: u32 = counts[5..11].iter().sum();
        assert_eq!(branch_ct[3], [low, high], "LOW_VAL node split");
    }

    /// W-coef-prob PROOF — `prob_diff_update_savings_search_model`
    /// (`vp9_subexp.c:141-180`), the PIVOT_NODE branch libvpx uses at
    /// `vp9_bitstream.c:569-575`.
    ///
    /// The ORIGINAL W7.1 port ADMITTED skipping this (its module
    /// deviation note). This test proves the model search is now
    /// exercised AND is genuinely different from the plain search: on
    /// the same pivot branch counts, the model search ALSO charges the 8
    /// downstream Pareto-8 nodes, so its savings/best_newp differ from
    /// what the plain (single-branch) search returns for the pivot node.
    #[test]
    fn w_coef_pivot_model_search_exercised_and_differs_from_plain() {
        // Build a full distribution where the pivot branch (ONE vs
        // TWO+) is HEAVILY skewed toward ONE so the ideal pivot prob is
        // far above the default 128 — and the 8 model nodes carry real
        // mass so the model search has downstream cost to optimise.
        let counts: [u32; ENTROPY_TOKENS] = [2000, 3000, 80, 40, 20, 12, 6, 4, 2, 1, 1, 1500];
        let non_eob: u32 = counts[0..11].iter().sum();
        let eob_total = non_eob + counts[11];
        let (branch_ct, probs) = build_coef_branch_cts(&counts, eob_total);

        let oldp: u8 = 128;
        let start_newp = probs[PIVOT_NODE]; // libvpx feeds new_coef_probs[..][PIVOT].
        assert_ne!(
            start_newp, oldp,
            "test setup: pivot ideal prob must differ from oldp so the \
             search has a range to walk (got {start_newp})"
        );
        let model = prob_diff_update_savings_search_model(
            &branch_ct,
            oldp,
            start_newp,
            DIFF_UPDATE_PROB,
            1, // stepsize = coeff_prob_appx_step (reference encoder = 1)
        );

        // The model search must MOVE the pivot prob and report savings.
        assert_ne!(
            model.best_newp, oldp,
            "model search must move the pivot prob on a skewed distribution"
        );
        assert!(
            model.savings > 0,
            "model search savings must be positive (got {})",
            model.savings
        );

        // Differential vs the PLAIN search on JUST the pivot branch
        // (node 2 = [#one, #two+]): the plain search ignores the 8
        // downstream model nodes, so its result generally differs. If
        // they coincided, the model port would be redundant — assert
        // they differ in savings OR best_newp to prove the model arm
        // adds information.
        let plain = prob_diff_update_savings_search(
            branch_ct[PIVOT_NODE],
            oldp,
            start_newp,
            DIFF_UPDATE_PROB,
        );
        assert!(
            model.savings != plain.savings || model.best_newp != plain.best_newp,
            "model search must differ from the plain pivot-branch search \
             (model: newp={} sav={}; plain: newp={} sav={}) — otherwise the \
             8 Pareto-8 model nodes are not being scored",
            model.best_newp,
            model.savings,
            plain.best_newp,
            plain.savings
        );

        // Zero-distribution sanity: no events ⇒ no savings, prob stays.
        let zero = [[0u32; 2]; ENTROPY_NODES];
        let none = prob_diff_update_savings_search_model(&zero, oldp, 200, DIFF_UPDATE_PROB, 1);
        assert_eq!(none.best_newp, oldp, "no events ⇒ keep oldp");
        assert_eq!(none.savings, 0, "no events ⇒ zero savings");
    }

    // Local token-index aliases for the test above (the production
    // constants live in rvp9_core::entropy::detokenize).
    const ZERO_TOKEN_IDX: usize = 0;
    const ONE_TOKEN_IDX: usize = 1;

    // ----- W7.2 -----

    /// W7.2 PROOF — `fix_interp_filter` (`vp9_bitstream.c:863-884`).
    ///
    /// When the per-context counters show only EIGHTTAP was picked
    /// (all blocks fired the `count[EIGHTTAP] > 0` row, every other
    /// row sums to zero), libvpx collapses the frame-level filter to
    /// EIGHTTAP. We assert `Collapse(EightTap)`.
    #[test]
    fn w7_2_interp_filter_selected_collapses_when_uniform() {
        let mut counts: SwitchableInterpCounts =
            [[0u32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS];
        // libvpx index 0 is EIGHTTAP. Spread some hits across the
        // contexts so the per-filter sum is > 0 in column 0 only.
        for (j, row) in counts
            .iter_mut()
            .enumerate()
            .take(SWITCHABLE_FILTER_CONTEXTS)
        {
            row[0] = 100 + j as u32;
        }
        let out = fix_interp_filter(InterpFilter::Switchable, &counts);
        assert_eq!(out, FixInterpFilter::Collapse(InterpFilter::EightTap));

        // Sanity: with two filters used, the helper must keep SWITCHABLE.
        counts[0][1] = 1;
        let out2 = fix_interp_filter(InterpFilter::Switchable, &counts);
        assert_eq!(out2, FixInterpFilter::KeepSwitchable);

        // And: a non-SWITCHABLE input is a no-op.
        let out3 = fix_interp_filter(InterpFilter::EightTap, &counts);
        assert_eq!(out3, FixInterpFilter::KeepSwitchable);
    }

    // ----- W7.3 -----

    /// W7.3 PROOF — `vp9_get_refresh_mask` preserve-GF branch
    /// (`vp9_bitstream.c:902-915` + `vp9_bitstream.h:41-44`).
    ///
    /// On an ARF-overlap show frame (`is_src_frame_alt_ref = true`)
    /// with `refresh_golden_frame = true` and `use_svc = false`,
    /// `vp9_preserve_existing_gf` returns true and the GOLDEN bit
    /// shifts into the ARF slot rather than the GF slot. We assert
    /// the mask bit appears at `alt_fb_idx` and NOT at `gld_fb_idx`.
    #[test]
    fn w7_3_get_refresh_mask_preserves_gf_when_arf_overlap() {
        let inputs = RefreshMaskInputs {
            refresh_last_frame: true,
            refresh_golden_frame: true,
            refresh_alt_ref_frame: false,
            is_src_frame_alt_ref: true,
            use_svc: false,
            lst_fb_idx: 0,
            gld_fb_idx: 1,
            alt_fb_idx: 2,
            multi_layer_arf: false,
            arf_index_stack: vec![],
            stack_size: 0,
            svc_update_buffer_slot: None,
        };
        assert!(preserve_existing_gf(&inputs));
        let out = get_refresh_mask(&inputs);
        // LAST bit at slot 0 → mask & 1 == 1.
        assert_eq!(out.mask & 0b0000_0001, 0b0000_0001);
        // GOLDEN bit at alt_fb_idx (2), NOT at gld_fb_idx (1).
        assert_eq!(out.mask & 0b0000_0100, 0b0000_0100);
        assert_eq!(out.mask & 0b0000_0010, 0);
        assert_eq!(out.top_arf_idx, 2);

        // Sanity: turn off ARF-overlap → standard 3-way OR fires; GF
        // bit must move back to gld_fb_idx (1).
        let inputs2 = RefreshMaskInputs {
            is_src_frame_alt_ref: false,
            ..inputs.clone()
        };
        assert!(!preserve_existing_gf(&inputs2));
        let out2 = get_refresh_mask(&inputs2);
        assert_eq!(out2.mask & 0b0000_0010, 0b0000_0010);

        // Sanity: multi-layer ARF search picks a slot outside
        // {lst, gld, alt} and outside the stack.
        let inputs3 = RefreshMaskInputs {
            refresh_alt_ref_frame: true,
            is_src_frame_alt_ref: false,
            multi_layer_arf: true,
            arf_index_stack: vec![3, 4],
            stack_size: 2,
            ..inputs.clone()
        };
        let out3 = get_refresh_mask(&inputs3);
        // Forbidden: 0, 1, 2 (lst/gld/alt) and 3, 4 (stack).
        // First free slot ⇒ 5.
        assert_eq!(out3.top_arf_idx, 5);
        assert_eq!(out3.mask & (1 << 5), 1 << 5);
    }
}
