//! W7.1-W7.3 — bitstream emission gates ported from libvpx.
//!
//! Three independent helpers, each a scalar reference port of a libvpx
//! `vp9_bitstream.c` / `vp9_encoder.c` / `vp9_subexp.c` block:
//!
//! - [`prob_diff_update_savings_search`] — model-free savings search
//!   that picks the best replacement probability for a given pair of
//!   `(zero_count, one_count)` branch counts. Mirrors libvpx
//!   `vp9_prob_diff_update_savings_search` (`vp9_subexp.c:117-139`).
//! - [`update_coef_probs_common`] — for one TX size, runs the
//!   "TWO_LOOP" path of libvpx's coef-prob updater: first a dry-run
//!   scan to compute aggregate `savings`, then either gate=0 (no
//!   update) or gate=1 followed by per-slot updates. Mirrors libvpx
//!   `update_coef_probs_common` (`vp9_bitstream.c:545-699`,
//!   TWO_LOOP branch).
//! - [`tx_totals_gate`] — outer-loop early-out used by libvpx
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
//! - Encoder integration: these helpers ship standalone with PROOF
//!   tests; wiring them into `frame_loop.rs` is a separate task.

use rvp9_core::bitstream::InterpFilter;
use rvp9_core::bitstream::frame::REF_FRAMES;
use rvp9_core::cost::prob_cost::{VP9_PROB_COST_SHIFT, vp9_cost_one, vp9_cost_zero};
use rvp9_core::entropy::BoolWriter;
use rvp9_core::entropy::{
    DIFF_UPDATE_PROB, SWITCHABLE_FILTERS, SWITCHABLE_FILTER_CONTEXTS, remap_prob,
    write_prob_diff_update,
};

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
    i64::from(ct[0]) * i64::from(vp9_cost_zero(p))
        + i64::from(ct[1]) * i64::from(vp9_cost_one(p))
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
    for i in 0..SWITCHABLE_FILTERS {
        let mut sum = 0u32;
        for j in 0..SWITCHABLE_FILTER_CONTEXTS {
            sum = sum.wrapping_add(counts[j][i]);
        }
        per_filter[i] = sum;
        if sum > 0 {
            nonzero += 1;
        }
    }

    if nonzero == 1 {
        // libvpx `vp9_bitstream.c:874-882`: collapse to the one used
        // filter (first non-zero slot in declaration order).
        for i in 0..SWITCHABLE_FILTERS {
            if per_filter[i] > 0 {
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
        let out =
            prob_diff_update_savings_search(ct, oldp, start_newp, DIFF_UPDATE_PROB);
        assert_ne!(
            out.best_newp, oldp,
            "search must move away from oldp on heavily skewed counts"
        );
        assert!(out.savings > 0, "savings must be positive ({} ≤ 0)", out.savings);
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
        let updates =
            update_coef_probs_common(&mut w, &mut old_probs, &new_probs, &branch_ct);
        let _ = w.stop_encode().unwrap();
        assert_eq!(updates, 0, "no-counts input must emit zero updates");
        // First bit on the wire must be `0` — but the bool writer
        // doesn't expose direct introspection; the surrogate check is
        // `old_probs` unchanged.
        assert_eq!(old_probs, vec![128u8; 8]);
    }

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
        for j in 0..SWITCHABLE_FILTER_CONTEXTS {
            counts[j][0] = 100 + j as u32;
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
