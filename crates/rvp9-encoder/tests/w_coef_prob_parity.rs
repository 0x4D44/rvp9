//! W-coef-prob — coefficient-probability rate-parity port status.
//!
//! GOAL (Arthur-approved A6): replace the compressed-header writer's
//! "Strategy B" force-update-on-diff coef-prob section
//! (`rvp9_core::bitstream::compressed_header::write_coef_probs_common`,
//! `compressed_header.rs:703`) with libvpx's TWO_LOOP rate-optimal
//! updater `update_coef_probs_common` (`vp9_bitstream.c:545-625`),
//! including the PIVOT_NODE model search
//! (`vp9_prob_diff_update_savings_search_model`, `vp9_subexp.c:141-180`).
//!
//! STATUS: **SCOPED — building blocks landed, full integration BLOCKED**
//! on count plumbing. See
//! `wrk_journals/2026.05.27 - JRN - coef-prob rate-parity landed.md`.
//!
//! ## Why blocked (code-verified)
//!
//! The TWO_LOOP updater's PIVOT_NODE branch
//! (`vp9_bitstream.c:569-575`) calls
//! `vp9_prob_diff_update_savings_search_model`, which scores the FULL
//! 12-token coefficient sub-tree (it indexes `VP9_PARETO8_FULL` for the
//! 8 downstream model nodes, `vp9_subexp.c:152-168`). libvpx builds
//! those branch counts in `build_tree_distribution`
//! (`vp9_bitstream.c:518-543`) from `cpi->td.rd_counts.coef_counts`,
//! which is `vp9_coeff_count` — the FULL 12-token histogram accumulated
//! during `vp9_tokenize.c::tokenize_b`.
//!
//! rvp9 today accumulates only the 4-token MODEL histogram
//! (`CoefCountsBlock = [..][UNCONSTRAINED_NODES + 1]` = ZERO/ONE/TWO/
//! EOB_MODEL, `detokenize.rs:206`) — the same shape the DECODER needs
//! for backward adaptation. That shape COLLAPSES TWO..CAT6 into one
//! TWO_TOKEN bucket, discarding exactly the within-TWO+ distribution
//! the 8 model nodes encode. So the PIVOT model search is not
//! computable from the counts rvp9 gathers.
//!
//! Two further obstacles compound it:
//!  1. ORDERING. libvpx writes the coef-prob header AFTER tokenizing
//!     the frame (counts ready). rvp9 writes the compressed header
//!     (`frame_loop.rs:3393`) BEFORE the tile encode that gathers
//!     counts (`frame_loop.rs:3460-3474`). A faithful port needs a
//!     count pass before the header → a two-pass tile encode on the
//!     byte-exact path.
//!  2. SHARED TYPE. `CoefCountsBank` lives in `rvp9-core` and is used
//!     by the decoder in ~10 sites; widening its token dimension to 12
//!     ripples into the decoder's backward-adapt path.
//!
//! Estimated count-plumbing alone (full-12-token bank + the existing
//! 7-function thread duplicated/widened + the two-pass reorder of both
//! `build_inter_packet_with_residuals` and `build_keyframe_packet`):
//! ~200-300 LOC, at/over the task's STOP threshold, modifying the only
//! byte-exact-through-vpxdec path. Per discipline ("no half-TWO_LOOP
//! that breaks the gate"), Strategy B stays live (it emits VALID,
//! byte-exact-decodable VP9) and the verified building blocks land in
//! `rvp9_encoder::bitstream_gates` for a future integration session.
//!
//! ## What this file proves (the landed, gate-safe deliverable)
//!
//! The two libvpx pieces the original W7.1 port ADMITTED skipping (its
//! PIVOT_NODE deviation) are now ported and exercised:
//!  - `build_coef_branch_cts` (= `build_tree_distribution`'s per-slot
//!    body) — DIFFERENTIAL-checked against the independent model-token
//!    derivation in `rvp9_core::entropy::adapt::adapt_coef_probs`.
//!  - `prob_diff_update_savings_search_model` (PIVOT) — exercised and
//!    shown to differ from the plain single-branch search (i.e. it
//!    really does score the 8 Pareto-8 model nodes).
//!
//! The byte-exact ORACLE for the live coef-prob path remains the sacred
//! `cross_decoder_gate` (rvp9-decoder + vpxdec agree byte-for-byte) —
//! it is GREEN on the Strategy-B stream, so the live path is valid.

use rvp9_core::cost::coef_cost::{ENTROPY_NODES, ENTROPY_TOKENS, PIVOT_NODE};
use rvp9_core::entropy::{DIFF_UPDATE_PROB, UNCONSTRAINED_NODES, get_binary_prob};
use rvp9_encoder::bitstream_gates::{
    build_coef_branch_cts, prob_diff_update_savings_search, prob_diff_update_savings_search_model,
};

/// PROOF — `build_tree_distribution` body fidelity
/// (`vp9_bitstream.c:518-543`).
///
/// Two independent derivations of the three UNCONSTRAINED_NODES branch
/// counts must agree:
///  - the full-12-token tree walk in `build_coef_branch_cts`
///    (`vp9_tree_probs_from_distribution`), and
///  - the 4-token MODEL folding already used by the backward-adapt
///    kernel (`rvp9_core::entropy::adapt::adapt_coef_probs`,
///    `adapt.rs:614-618`).
///
/// Agreement on two code paths written from opposite directions is the
/// oracle. The test also confirms the 8 MODEL nodes carry the
/// within-TWO+ mass (what the 4-token shape discards) — the data the
/// PIVOT model search consumes.
#[test]
fn build_tree_distribution_agrees_with_model_token_folding() {
    // Token order: ZERO ONE TWO THREE FOUR CAT1 CAT2 CAT3 CAT4 CAT5
    // CAT6 EOB.
    let counts: [u32; ENTROPY_TOKENS] = [1200, 420, 160, 80, 44, 26, 14, 9, 5, 3, 2, 700];
    let non_eob: u32 = counts[0..11].iter().sum();
    let eob_total = non_eob + counts[11];

    let (branch_ct, probs) = build_coef_branch_cts(&counts, eob_total);

    // adapt.rs derivation (independent oracle).
    let n0 = counts[0];
    let n1 = counts[1];
    let n2: u32 = counts[2..11].iter().sum(); // TWO..CAT6 folded
    let neob = counts[11];
    let expect = [[neob, eob_total - neob], [n0, n1 + n2], [n1, n2]];
    for m in 0..UNCONSTRAINED_NODES {
        assert_eq!(branch_ct[m], expect[m], "unconstrained node {m}");
        assert_eq!(
            probs[m],
            get_binary_prob(expect[m][0], expect[m][1]),
            "node {m} ideal prob"
        );
    }

    // Model nodes must be non-zero (the within-TWO+ distribution).
    let model_mass: u32 = (UNCONSTRAINED_NODES..ENTROPY_NODES)
        .map(|i| branch_ct[i][0] + branch_ct[i][1])
        .sum();
    assert!(model_mass > 0, "model nodes 3..10 must carry mass");
}

/// PROOF — the PIVOT_NODE model search is exercised AND adds
/// information over the plain search (`vp9_subexp.c:141-180` vs
/// `:117-139`). This is the exact piece the original W7.1 port skipped.
#[test]
fn pivot_model_search_scores_downstream_nodes() {
    // ONE-heavy pivot so the ideal pivot prob is far from the default.
    let counts: [u32; ENTROPY_TOKENS] = [2000, 3000, 80, 40, 20, 12, 6, 4, 2, 1, 1, 1500];
    let non_eob: u32 = counts[0..11].iter().sum();
    let eob_total = non_eob + counts[11];
    let (branch_ct, probs) = build_coef_branch_cts(&counts, eob_total);

    let oldp: u8 = 128;
    let start_newp = probs[PIVOT_NODE];
    assert_ne!(start_newp, oldp, "setup: pivot ideal must differ from oldp");

    let model =
        prob_diff_update_savings_search_model(&branch_ct, oldp, start_newp, DIFF_UPDATE_PROB, 1);
    assert_ne!(model.best_newp, oldp, "model search must move the prob");
    assert!(model.savings > 0, "model savings must be positive");

    let plain =
        prob_diff_update_savings_search(branch_ct[PIVOT_NODE], oldp, start_newp, DIFF_UPDATE_PROB);
    assert!(
        model.savings != plain.savings || model.best_newp != plain.best_newp,
        "model search must differ from plain (model adds the 8 Pareto-8 nodes)"
    );
}
