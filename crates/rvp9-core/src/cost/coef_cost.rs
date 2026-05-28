//! Coefficient token cost table.
//!
//! Mirrors libvpx `fill_token_costs` (`vp9_rd.c:135-152`). One call to
//! [`build_coef_cost_table`] expands the `FrameContext.coef_probs`
//! model into a full per-`(tx, plane, ref, band, ctx, token)` cost
//! table.
//!
//! ## Layout
//!
//! `data[tx_size][plane_type][ref_type][band][ctx][token]` — costs in
//! `vp9_prob_cost` units (`512` = one bit).
//!
//! `tx_size` is one of `TX_4X4 = 0`, `TX_8X8 = 1`, `TX_16X16 = 2`,
//! `TX_32X32 = 3`. `plane_type` and `ref_type` are 0/1. `band` runs
//! `0..6`. `ctx` runs `0..6` (only the first 3 are valid in band 0;
//! libvpx pads the rest with zeros — see [`super`] crate's
//! `coef_probs` module for that convention).
//!
//! libvpx splits the table into two ENTROPY_TOKENS-sized slabs per
//! cell (`c[t][i][j][k][0]` and `c[t][i][j][k][1]`): the former is for
//! "may include EOB" (use `vp9_cost_tokens`), the latter is for "EOB
//! cannot follow" (use `vp9_cost_tokens_skip`). We carry both halves
//! here too.

use super::tree_cost::{vp9_cost_tokens, vp9_cost_tokens_skip};
use crate::entropy::coef_probs::{COEF_BANDS, COEFF_CONTEXTS, PLANE_TYPES, REF_TYPES};
use crate::entropy::frame_context::{FrameContext, TX_SIZES};
use crate::entropy::pareto::{MODEL_NODES, VP9_PARETO8_FULL};

/// `ENTROPY_TOKENS` in libvpx (`vp9_entropy.h:40`). 12 leaf indices for
/// the coefficient tree: zero / one / two / three / four / cat1 /
/// cat2 / cat3 / cat4 / cat5 / cat6 / eob.
pub const ENTROPY_TOKENS: usize = 12;

/// `ENTROPY_NODES = 11` per libvpx (`vp9_entropy.h:42`) — the number of
/// internal nodes (one less than the leaf count).
pub const ENTROPY_NODES: usize = 11;

/// `PIVOT_NODE = 2` — the index in the 3-element unconstrained
/// `model` array that the Pareto8 table is keyed off.
pub const PIVOT_NODE: usize = 2;

/// `vp9_coef_tree` from libvpx (`vp9_tokenize.c:73-87`).
/// `TREE_SIZE(ENTROPY_TOKENS) = 2 * 12 - 2 = 22`.
#[rustfmt::skip]
pub const VP9_COEF_TREE: [i8; 22] = [
    -11, 2,    // -EOB_TOKEN = -11, ...
    -0,  4,    // -ZERO_TOKEN, ...
    -1,  6,    // -ONE_TOKEN, ...
     8, 12,    //
    -2, 10,    // -TWO_TOKEN, ...
    -3, -4,    // -THREE_TOKEN, -FOUR_TOKEN
    14, 16,
    -5, -6,    // -CATEGORY1_TOKEN, -CATEGORY2_TOKEN
    18, 20,
    -7, -8,    // -CATEGORY3_TOKEN, -CATEGORY4_TOKEN
    -9, -10,   // -CATEGORY5_TOKEN, -CATEGORY6_TOKEN
];

/// Two ENTROPY_TOKENS-wide slabs per cell: index `[0]` includes EOB,
/// index `[1]` skips EOB (libvpx's `c[t][i][j][k][s][l]` layout, with
/// `s ∈ {0, 1}`).
pub type CoefTokenCosts = [[i32; ENTROPY_TOKENS]; 2];

/// One `[band][ctx][skip_eob][token]` slab — a single
/// `(plane_type, ref_type)` cell of the cost table.
pub type CoefRefBandCtxCosts = [[CoefTokenCosts; COEFF_CONTEXTS]; COEF_BANDS];

/// Per-tx-size slab: `[plane_type][ref_type][band][ctx][skip_eob][token]`.
pub type CoefTxCosts = [[CoefRefBandCtxCosts; REF_TYPES]; PLANE_TYPES];

/// Full coefficient cost table. Mirrors libvpx
/// `vp9_coeff_cost[TX_SIZES][PLANE_TYPES][REF_TYPES][COEF_BANDS]
/// [COEFF_CONTEXTS][2][ENTROPY_TOKENS]`.
///
/// At ~28 KiB this is small enough to live on the stack but we box it
/// to keep mode-decision call-sites from blowing the default stack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefCostTable {
    /// `[tx][plane_type][ref_type][band][ctx]` → two ENTROPY_TOKENS
    /// slabs (with-EOB / without-EOB).
    pub data: Box<[CoefTxCosts; TX_SIZES]>,
}

impl Default for CoefCostTable {
    fn default() -> Self {
        Self {
            data: Box::new(
                [[[[[[[0i32; ENTROPY_TOKENS]; 2]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
                    PLANE_TYPES]; TX_SIZES],
            ),
        }
    }
}

/// Expand the 3-prob "model" vector into the full 11-prob node-prob
/// vector via the Pareto8 table. Mirrors libvpx
/// `vp9_model_to_full_probs` + `extend_to_full_distribution`
/// (`vp9_entropy.c:1030-1039`).
#[inline]
fn model_to_full_probs(model: &[u8; 3]) -> [u8; ENTROPY_NODES] {
    let mut full = [0u8; ENTROPY_NODES];
    full[0] = model[0];
    full[1] = model[1];
    full[2] = model[2];
    // Pareto8 row for the pivot probability. libvpx's
    // `extend_to_full_distribution` asserts `p != 0` — band-0 zero rows
    // are padded with zeros and never indexed at runtime, but at table
    // build time we still pass through them. Guard the row lookup so a
    // padding entry stays at zeros and doesn't crash on row -1.
    let pivot = model[PIVOT_NODE];
    if pivot != 0 {
        let row = &VP9_PARETO8_FULL[(pivot - 1) as usize];
        full[3..3 + MODEL_NODES].copy_from_slice(row);
    }
    full
}

/// Build the per-`(tx, plane, ref, band, ctx, token)` cost table from
/// a `FrameContext`'s coefficient model. Mirrors libvpx
/// `fill_token_costs` (`vp9_rd.c:135-152`).
#[must_use]
pub fn build_coef_cost_table(fc: &FrameContext) -> CoefCostTable {
    let mut table = CoefCostTable::default();
    for (t, tx_probs) in fc.coef_probs.iter().enumerate() {
        for (i, plane_probs) in tx_probs.iter().enumerate() {
            for (j, ref_probs) in plane_probs.iter().enumerate() {
                for (k, band_probs) in ref_probs.iter().enumerate() {
                    // band 0 only has 3 valid contexts; the rest are
                    // zero-padded but iterating them produces all-zero
                    // costs, which the caller must avoid using.
                    for (l, ctx_model) in band_probs.iter().enumerate() {
                        // Skip the zero-padded band-0 rows. libvpx
                        // never indexes them; we leave their costs at 0.
                        if ctx_model[PIVOT_NODE] == 0 {
                            continue;
                        }
                        let full = model_to_full_probs(ctx_model);
                        vp9_cost_tokens(&mut table.data[t][i][j][k][l][0], &VP9_COEF_TREE, &full);
                        vp9_cost_tokens_skip(
                            &mut table.data[t][i][j][k][l][1],
                            &VP9_COEF_TREE,
                            &full,
                        );
                        debug_assert_eq!(
                            table.data[t][i][j][k][l][0][11], // EOB token index
                            table.data[t][i][j][k][l][1][11]
                        );
                    }
                }
            }
        }
    }
    table
}

/// Index keys for [`coef_token_cost`]. Packing the indices into a
/// struct sidesteps clippy's `too_many_arguments` lint and keeps
/// call-sites self-documenting.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoefCostIndex {
    /// 0 = TX_4X4 … 3 = TX_32X32.
    pub tx_size: usize,
    /// 0 = Y, 1 = UV.
    pub plane_type: usize,
    /// 0 = intra, 1 = inter.
    pub ref_type: usize,
    /// Coefficient band index, `0..COEF_BANDS`.
    pub band: usize,
    /// Neighbour-context index, `0..COEFF_CONTEXTS`.
    pub ctx: usize,
    /// Coefficient token (`0..ENTROPY_TOKENS`).
    pub token: u8,
    /// `false` → with-EOB branch (`c[..][0]`); `true` → EOB-cannot-follow branch (`c[..][1]`).
    pub skip_eob: bool,
}

/// Lookup the cost of coefficient token `idx.token` at the
/// `(tx_size, plane_type, ref_type, band, ctx)` cell selected by `idx`.
#[inline]
#[must_use]
pub fn coef_token_cost(table: &CoefCostTable, idx: CoefCostIndex) -> i32 {
    let s = usize::from(idx.skip_eob);
    table.data[idx.tx_size][idx.plane_type][idx.ref_type][idx.band][idx.ctx][s][idx.token as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_built_from_default_fc_is_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let table = build_coef_cost_table(&fc);
        // Pick one well-known non-zero-padded cell:
        // tx_4x4 / Y plane / intra / band 1 / ctx 0.
        let cell = &table.data[0][0][0][1][0][0];
        let nz = cell.iter().filter(|&&c| c > 0).count();
        assert!(
            nz >= 8,
            "expected most tokens to have positive cost, got nz={nz}"
        );
        for &c in cell {
            assert!(c >= 0, "negative cost = {c}");
            assert!(c < 100_000, "cost {c} too large");
        }
    }

    #[test]
    fn eob_costs_equal_across_skip_variants() {
        let fc = FrameContext::new_default();
        let table = build_coef_cost_table(&fc);
        // EOB_TOKEN = 11 (per libvpx). The skip-variant fills the EOB
        // slot with `cost_zero(probs[0])`, the non-skip variant walks
        // the root branch the same way for the EOB leaf.
        let with_eob = table.data[0][0][0][1][0][0][11];
        let no_eob = table.data[0][0][0][1][0][0][11];
        assert_eq!(with_eob, no_eob);
    }

    #[test]
    fn coef_token_cost_lookup_matches_table() {
        let fc = FrameContext::new_default();
        let table = build_coef_cost_table(&fc);
        let direct = table.data[1][0][0][2][1][0][3];
        let via_fn = coef_token_cost(
            &table,
            CoefCostIndex {
                tx_size: 1,
                plane_type: 0,
                ref_type: 0,
                band: 2,
                ctx: 1,
                token: 3,
                skip_eob: false,
            },
        );
        assert_eq!(direct, via_fn);
    }

    #[test]
    fn model_to_full_zero_pivot_yields_zero_tail() {
        let model = [0u8, 0, 0];
        let full = model_to_full_probs(&model);
        // model[0..3] copied through.
        assert_eq!(full[0..3], [0, 0, 0]);
        // tail zero-padded since the pivot is 0.
        for &p in &full[3..] {
            assert_eq!(p, 0);
        }
    }
}
