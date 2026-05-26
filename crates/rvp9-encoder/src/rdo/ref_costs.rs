//! W4.5 — `estimate_ref_frame_costs`.
//!
//! Computes the per-ref-frame signaling cost (in 1/512-bit
//! `vp9_prob_cost` units) for the slow RDO picker. Mirrors libvpx
//! `vp9/encoder/vp9_rdopt.c:2430-2488` (v1.14.1 commit `12f3a2ac6`).
//!
//! ## libvpx source
//!
//! ```c
//! static void estimate_ref_frame_costs(const VP9_COMMON *cm,
//!                                      const MACROBLOCKD *xd, int segment_id,
//!                                      unsigned int *ref_costs_single,
//!                                      unsigned int *ref_costs_comp,
//!                                      vpx_prob *comp_mode_p) {
//!   /* ... */
//!   ref_costs_single[INTRA_FRAME] = vp9_cost_bit(intra_inter_p, 0);
//!   /* ... */
//!   ref_costs_single[LAST_FRAME] = ref_costs_single[GOLDEN_FRAME] =
//!       ref_costs_single[ALTREF_FRAME] = base_cost;
//!   ref_costs_single[LAST_FRAME] += vp9_cost_bit(ref_single_p1, 0);
//!   ref_costs_single[GOLDEN_FRAME] += vp9_cost_bit(ref_single_p1, 1);
//!   ref_costs_single[ALTREF_FRAME] += vp9_cost_bit(ref_single_p1, 1);
//!   ref_costs_single[GOLDEN_FRAME] += vp9_cost_bit(ref_single_p2, 0);
//!   ref_costs_single[ALTREF_FRAME] += vp9_cost_bit(ref_single_p2, 1);
//!   /* ... */
//! }
//! ```
//!
//! ## Scope
//!
//! - Real per-block ref-frame cost. Replaces a constant approximation
//!   the encoder had been threading in.
//! - The `ref_frame_context` here is the libvpx `(intra_inter_ctx,
//!   single_ref_p1_ctx, single_ref_p2_ctx, comp_ref_ctx)` quadruple —
//!   the picker indexes per-ctx default-probs to derive `cost_bit(p, 0)`
//!   and `cost_bit(p, 1)`.
//! - Segment-level ref-frame override (`SEG_LVL_REF_FRAME`) is honoured
//!   per libvpx lines 2435-2440 (`seg_ref_active`): when set, all
//!   ref-frame costs are zero.
//! - `REFERENCE_MODE_SELECT` / `COMPOUND_REFERENCE` / `SINGLE_REFERENCE`
//!   are all ported. The single-ref subtree (the only one currently
//!   exercised by the W4.x picker) is the load-bearing path.

use rvp9_core::cost::vp9_cost_bit;
use rvp9_core::entropy::{
    DEFAULT_COMP_INTER_PROB, DEFAULT_COMP_REF_PROB, DEFAULT_INTRA_INTER_PROB,
    DEFAULT_SINGLE_REF_PROB,
};
use rvp9_core::mode_info::RefFrame;

/// libvpx `REFERENCE_MODE` (`vp9_enums.h:158-162`). Selects which of
/// the single-/compound-/select reference subtrees the picker walks.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceMode {
    /// `SINGLE_REFERENCE = 0` — only single-ref ref-frame bits emitted.
    SingleReference,
    /// `COMPOUND_REFERENCE = 1` — only compound-ref bits emitted.
    CompoundReference,
    /// `REFERENCE_MODE_SELECT = 2` — per-block comp_pred flag emitted,
    /// followed by either the single-ref or compound-ref subtree.
    ReferenceModeSelect,
}

/// Per-block neighbour-derived context fed into the ref-frame
/// probability lookups. Mirrors libvpx's
/// `vp9_get_pred_prob_single_ref_p{1,2}` / `vp9_get_pred_prob_comp_ref_p`
/// / `vp9_get_intra_inter_prob` / `vp9_get_reference_mode_prob`
/// context-derivation results — i.e. each context is the
/// `0..N_CONTEXTS` index into the corresponding default-prob table.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RefFrameContext {
    /// `intra_inter_ctx ∈ [0..INTRA_INTER_CONTEXTS = 4]`.
    pub intra_inter_ctx: u8,
    /// `single_ref_p1_ctx ∈ [0..REF_CONTEXTS = 5]`.
    pub single_ref_p1_ctx: u8,
    /// `single_ref_p2_ctx ∈ [0..REF_CONTEXTS = 5]`.
    pub single_ref_p2_ctx: u8,
    /// `comp_ref_ctx ∈ [0..REF_CONTEXTS = 5]`.
    pub comp_ref_ctx: u8,
    /// `comp_inter_ctx ∈ [0..COMP_INTER_CONTEXTS = 5]`. Only consumed
    /// when [`ReferenceMode::ReferenceModeSelect`] is set.
    pub comp_inter_ctx: u8,
}

/// libvpx ref-frame cost tuple. Indexes are by [`RefFrame`] semantics
/// (libvpx `MV_REFERENCE_FRAME`):
/// - `[INTRA_FRAME]` = single-ref intra signaling cost.
/// - `[LAST_FRAME]` / `[GOLDEN_FRAME]` / `[ALTREF_FRAME]` = single-ref
///   inter signaling cost.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RefFrameCosts {
    /// Single-ref costs: `[INTRA, LAST, GOLDEN, ALTREF]`.
    pub single: [u32; 4],
    /// Compound-ref costs: `[unused, LAST, GOLDEN, unused]` per libvpx
    /// (only `LAST` and `GOLDEN` slots are written; ALTREF is the
    /// implicit second ref).
    pub comp: [u32; 4],
    /// `comp_mode_p` — probability the per-block `comp_pred` flag is
    /// `1`. Used by the picker to add `vp9_cost_bit(comp_mode_p,
    /// comp_pred)` to the rate when `REFERENCE_MODE_SELECT` is set.
    /// `128` for the non-select cases (the default neutral prob libvpx
    /// uses when the comp-inter context isn't materialised).
    pub comp_mode_p: u8,
}

/// libvpx `estimate_ref_frame_costs` (`vp9_rdopt.c:2430-2488`).
///
/// `seg_ref_active` mirrors `segfeature_active(&cm->seg, segment_id,
/// SEG_LVL_REF_FRAME)`. When true (libvpx line 2435-2440), all
/// ref-frame costs are zero and `comp_mode_p = 128`.
///
/// `reference_mode` mirrors `cm->reference_mode`.
#[must_use]
pub fn estimate_ref_frame_costs(
    seg_ref_active: bool,
    reference_mode: ReferenceMode,
    ctx: RefFrameContext,
) -> RefFrameCosts {
    let mut out = RefFrameCosts::default();
    if seg_ref_active {
        // libvpx lines 2437-2440: all-zero costs, neutral comp_mode_p.
        out.comp_mode_p = 128;
        return out;
    }

    // libvpx line 2442: `intra_inter_p = vp9_get_intra_inter_prob`.
    let intra_inter_p =
        DEFAULT_INTRA_INTER_PROB[ctx.intra_inter_ctx as usize];

    // libvpx lines 2443-2450: comp_inter_p / comp_mode_p selection.
    let comp_inter_p = match reference_mode {
        ReferenceMode::ReferenceModeSelect => {
            DEFAULT_COMP_INTER_PROB[ctx.comp_inter_ctx as usize]
        }
        _ => 128,
    };
    out.comp_mode_p = if matches!(reference_mode, ReferenceMode::ReferenceModeSelect) {
        comp_inter_p
    } else {
        128
    };

    // libvpx line 2452: intra-frame ref cost = cost_bit(intra_inter, 0).
    out.single[RefFrame::IntraFrame as usize] = vp9_cost_bit(intra_inter_p, false);

    // libvpx lines 2454-2473: single-ref subtree.
    if !matches!(reference_mode, ReferenceMode::CompoundReference) {
        let ref_single_p1 = DEFAULT_SINGLE_REF_PROB[ctx.single_ref_p1_ctx as usize][0];
        let ref_single_p2 = DEFAULT_SINGLE_REF_PROB[ctx.single_ref_p2_ctx as usize][1];

        // libvpx line 2457: base_cost = cost_bit(intra_inter_p, 1).
        let mut base_cost = vp9_cost_bit(intra_inter_p, true);
        if matches!(reference_mode, ReferenceMode::ReferenceModeSelect) {
            // libvpx line 2460: add cost_bit(comp_inter_p, 0).
            base_cost += vp9_cost_bit(comp_inter_p, false);
        }

        let last = RefFrame::LastFrame as usize;
        let golden = RefFrame::GoldenFrame as usize;
        let altref = RefFrame::AltrefFrame as usize;
        out.single[last] = base_cost;
        out.single[golden] = base_cost;
        out.single[altref] = base_cost;
        // libvpx lines 2464-2468: p1/p2 splitter bits.
        out.single[last] += vp9_cost_bit(ref_single_p1, false);
        out.single[golden] += vp9_cost_bit(ref_single_p1, true);
        out.single[altref] += vp9_cost_bit(ref_single_p1, true);
        out.single[golden] += vp9_cost_bit(ref_single_p2, false);
        out.single[altref] += vp9_cost_bit(ref_single_p2, true);
    } else {
        // libvpx lines 2469-2473: 512-unit fallback when subtree is unused.
        out.single[RefFrame::LastFrame as usize] = 512;
        out.single[RefFrame::GoldenFrame as usize] = 512;
        out.single[RefFrame::AltrefFrame as usize] = 512;
    }

    // libvpx lines 2474-2486: compound-ref subtree.
    if !matches!(reference_mode, ReferenceMode::SingleReference) {
        let ref_comp_p = DEFAULT_COMP_REF_PROB[ctx.comp_ref_ctx as usize];
        let mut base_cost = vp9_cost_bit(intra_inter_p, true);
        if matches!(reference_mode, ReferenceMode::ReferenceModeSelect) {
            base_cost += vp9_cost_bit(comp_inter_p, true);
        }
        let last = RefFrame::LastFrame as usize;
        let golden = RefFrame::GoldenFrame as usize;
        out.comp[last] = base_cost + vp9_cost_bit(ref_comp_p, false);
        out.comp[golden] = base_cost + vp9_cost_bit(ref_comp_p, true);
    } else {
        out.comp[RefFrame::LastFrame as usize] = 512;
        out.comp[RefFrame::GoldenFrame as usize] = 512;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PROOF W4.5 — different `ref_frame_context` produces different
    /// ref-frame costs. libvpx `vp9_rdopt.c:2454-2473`: the per-ref
    /// cost is `base_cost + vp9_cost_bit(ref_single_p{1,2}, bit)`,
    /// where `ref_single_p{1,2}` is read from `DEFAULT_SINGLE_REF_PROB
    /// [ctx][...]`. Two distinct context tuples MUST yield two distinct
    /// single-ref cost vectors.
    #[test]
    fn w4_5_ref_frame_cost_changes_with_ref_frame_context() {
        let ctx_a = RefFrameContext {
            intra_inter_ctx: 0, // intra_inter_p = 9 (very biased to intra)
            single_ref_p1_ctx: 0, // p1 = 33
            single_ref_p2_ctx: 0, // p2 = 16
            comp_ref_ctx: 0,
            comp_inter_ctx: 0,
        };
        let ctx_b = RefFrameContext {
            intra_inter_ctx: 3, // intra_inter_p = 225 (heavily inter)
            single_ref_p1_ctx: 4, // p1 = 238
            single_ref_p2_ctx: 4, // p2 = 247
            comp_ref_ctx: 0,
            comp_inter_ctx: 0,
        };

        let costs_a =
            estimate_ref_frame_costs(false, ReferenceMode::SingleReference, ctx_a);
        let costs_b =
            estimate_ref_frame_costs(false, ReferenceMode::SingleReference, ctx_b);

        // The intra-vs-inter cost MUST differ — `intra_inter_ctx 0 →
        // p=9` (very biased toward `0`, i.e. cheap to encode `0` /
        // intra), vs `ctx 3 → p=225` (biased the other way). The
        // intra ref cost is `cost_bit(p, 0)`, which decreases as `p`
        // increases. So ctx_a (p=9) costs MUCH MORE than ctx_b (p=225)
        // for an intra signal:
        //   cost_bit(9, 0) ≈ 2473 (high),
        //   cost_bit(225, 0) ≈ 95   (low).
        assert!(
            costs_a.single[RefFrame::IntraFrame as usize]
                > costs_b.single[RefFrame::IntraFrame as usize],
            "intra cost: ctx_a {} should be > ctx_b {} (p=9 → high cost \
             for intra=0; p=225 → low cost)",
            costs_a.single[RefFrame::IntraFrame as usize],
            costs_b.single[RefFrame::IntraFrame as usize],
        );

        // The LAST/GOLDEN/ALTREF triples must also differ across contexts.
        assert_ne!(
            costs_a.single[RefFrame::LastFrame as usize],
            costs_b.single[RefFrame::LastFrame as usize],
            "LAST cost must change with single_ref context",
        );
        assert_ne!(
            costs_a.single[RefFrame::GoldenFrame as usize],
            costs_b.single[RefFrame::GoldenFrame as usize],
            "GOLDEN cost must change with single_ref context",
        );
    }

    /// `seg_ref_active = true` zeros all costs and sets `comp_mode_p = 128`.
    /// Mirrors libvpx lines 2437-2440.
    #[test]
    fn seg_ref_active_zeros_all_costs() {
        let ctx = RefFrameContext::default();
        let out =
            estimate_ref_frame_costs(true, ReferenceMode::ReferenceModeSelect, ctx);
        assert!(out.single.iter().all(|&v| v == 0));
        assert!(out.comp.iter().all(|&v| v == 0));
        assert_eq!(out.comp_mode_p, 128);
    }

    /// Single-ref subtree under `SingleReference` mode follows libvpx
    /// lines 2454-2468: `cost(LAST) = base + cost_bit(p1, 0)`.
    #[test]
    fn single_ref_cost_breakdown_matches_libvpx() {
        let ctx = RefFrameContext {
            intra_inter_ctx: 2,
            single_ref_p1_ctx: 2,
            single_ref_p2_ctx: 2,
            comp_ref_ctx: 0,
            comp_inter_ctx: 0,
        };
        let intra_inter_p = DEFAULT_INTRA_INTER_PROB[2];
        let p1 = DEFAULT_SINGLE_REF_PROB[2][0];
        let p2 = DEFAULT_SINGLE_REF_PROB[2][1];
        let expected_base = vp9_cost_bit(intra_inter_p, true);
        let expected_last = expected_base + vp9_cost_bit(p1, false);
        let expected_golden =
            expected_base + vp9_cost_bit(p1, true) + vp9_cost_bit(p2, false);
        let expected_altref =
            expected_base + vp9_cost_bit(p1, true) + vp9_cost_bit(p2, true);

        let out =
            estimate_ref_frame_costs(false, ReferenceMode::SingleReference, ctx);
        assert_eq!(out.single[RefFrame::LastFrame as usize], expected_last);
        assert_eq!(out.single[RefFrame::GoldenFrame as usize], expected_golden);
        assert_eq!(out.single[RefFrame::AltrefFrame as usize], expected_altref);
        assert_eq!(
            out.single[RefFrame::IntraFrame as usize],
            vp9_cost_bit(intra_inter_p, false),
        );
    }
}
