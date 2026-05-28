//! Per-symbol cost tables built from a [`FrameContext`].
//!
//! Mirrors libvpx `fill_mode_costs` (`vp9_rd.c:92-133`) plus the
//! partition / switchable-interp / skip / tx-size cost loops in
//! `vp9_initialize_rd_consts` (`vp9_rd.c:396-`). Each function in here
//! returns one densely-typed cost table built from one
//! `FrameContext`-sub-tree at a time.
//!
//! All costs are in `vp9_prob_cost` units — `512` = one bit. Indexing
//! by leaf-index (mode / partition / filter) matches the corresponding
//! `vpx_tree_index` tree.

use super::tree_cost::vp9_cost_tokens;
use crate::cost::prob_cost::{vp9_cost_one, vp9_cost_zero};
use crate::entropy::adapt::{
    VP9_INTER_MODE_TREE, VP9_INTRA_MODE_TREE, VP9_PARTITION_TREE, VP9_SWITCHABLE_INTERP_TREE,
};
use crate::entropy::default_probs::{
    INTER_MODE_CONTEXTS, INTER_MODES, INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES,
    SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
use crate::entropy::frame_context::{FrameContext, TX_SIZES};
use crate::entropy::kf_probs::{VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB, VP9_KF_Y_MODE_PROB};

// libvpx `TX_SIZE_CONTEXTS = 2`. Already pinned in
// `bitstream::compressed_header::TX_SIZE_CONTEXTS`; we re-pin a local
// constant so this module has no upward dependency.
const TX_SIZE_CONTEXTS: usize = 2;

/// Per-context per-partition-type bit-cost table built from one
/// `[[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS]` probability bank.
pub type PartitionCosts = [[i32; PARTITION_TYPES]; PARTITION_CONTEXTS];

/// Per-context per-partition-type bit-cost table for keyframes —
/// derived from the static `VP9_KF_PARTITION_PROBS` rather than the
/// frame context.
pub type KfPartitionCosts = PartitionCosts;

/// Per-context per-inter-mode bit-cost table built from
/// `FrameContext::inter_mode_probs`.
pub type InterModeCosts = [[i32; INTER_MODES]; INTER_MODE_CONTEXTS];

/// Per-context per-filter bit-cost table built from
/// `FrameContext::switchable_interp_prob`.
pub type SwitchableInterpCosts = [[i32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS];

/// Per-context per-skip-flag bit-cost table built from
/// `FrameContext::skip_probs`. `[ctx][bit]`, two entries per context
/// (`bit = 0` = skip-not-used; `bit = 1` = skip-used).
pub type SkipCosts = [[i32; 2]; SKIP_CONTEXTS];

/// Per-`max_tx_size` × per-context × per-actual-tx-size cost table.
/// Mirrors libvpx `cpi->tx_size_cost[3][TX_SIZE_CONTEXTS][TX_SIZES]`.
///
/// Index `[max_tx_size - 1][ctx][actual_tx_size]`:
/// - `max_tx_size ∈ {TX_8X8, TX_16X16, TX_32X32} → outer index 0/1/2`.
/// - Only entries with `actual_tx_size ≤ max_tx_size` are populated
///   (libvpx layout has `TX_SIZES = 4` inner slots; unused trailing
///   slots are zero).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TxSizeCostTable {
    /// `[max - 1][ctx][actual]` — 3 max-sizes × 2 contexts × 4 actual.
    pub data: [[[i32; TX_SIZES]; TX_SIZE_CONTEXTS]; 3],
}

/// Per-Y-mode-context Y intra-mode cost vector for non-keyframes.
/// Mirrors libvpx `cpi->mbmode_cost` which is the size-group-1 row.
///
/// `[mode]`, one cost per [`INTRA_MODES`] leaf.
///
/// libvpx only builds the size-group-1 row (`fc->y_mode_prob[1]`) into
/// `mbmode_cost`. We expose all 4 block-size-group rows for
/// completeness; mode-decision picks the row matching the current
/// block-size group.
#[must_use]
pub fn intra_y_mode_costs_inter(fc: &FrameContext) -> [[i32; INTRA_MODES]; 4] {
    let mut out = [[0i32; INTRA_MODES]; 4];
    for (group, probs) in fc.y_mode_prob.iter().enumerate() {
        vp9_cost_tokens(&mut out[group], &VP9_INTRA_MODE_TREE, probs);
    }
    out
}

/// Per-Y-mode UV intra-mode cost table for non-keyframes. Mirrors the
/// `intra_uv_mode_cost[INTER_FRAME]` slice in libvpx
/// (`vp9_rd.c:107-108`). `[y_mode][uv_mode]`.
#[must_use]
pub fn intra_uv_mode_costs_inter(fc: &FrameContext) -> [[i32; INTRA_MODES]; INTRA_MODES] {
    let mut out = [[0i32; INTRA_MODES]; INTRA_MODES];
    for (i, probs) in fc.uv_mode_prob.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_INTRA_MODE_TREE, probs);
    }
    out
}

/// Per-`(above, left)` Y intra-mode cost table for keyframes. Mirrors
/// libvpx `cpi->y_mode_costs[i][j]` (`vp9_rd.c:96-101`), built from
/// `vp9_kf_y_mode_prob[above][left]`.
#[must_use]
pub fn intra_y_mode_costs_kf_table() -> [[[i32; INTRA_MODES]; INTRA_MODES]; INTRA_MODES] {
    let mut out = [[[0i32; INTRA_MODES]; INTRA_MODES]; INTRA_MODES];
    for (i, plane) in VP9_KF_Y_MODE_PROB.iter().enumerate() {
        for (j, probs) in plane.iter().enumerate() {
            vp9_cost_tokens(&mut out[i][j], &VP9_INTRA_MODE_TREE, probs);
        }
    }
    out
}

/// Convenience wrapper: cost vector for `(above, left)` Y-mode
/// neighbours under a keyframe.
#[must_use]
pub fn intra_y_mode_costs_kf(above: usize, left: usize) -> [i32; INTRA_MODES] {
    let mut out = [0i32; INTRA_MODES];
    vp9_cost_tokens(
        &mut out,
        &VP9_INTRA_MODE_TREE,
        &VP9_KF_Y_MODE_PROB[above][left],
    );
    out
}

/// Per-Y-mode UV intra-mode cost table for keyframes. Mirrors
/// libvpx `intra_uv_mode_cost[KEY_FRAME][i]` (`vp9_rd.c:105-106`),
/// built from the static `VP9_KF_UV_MODE_PROB`.
#[must_use]
pub fn intra_uv_mode_costs_kf() -> [[i32; INTRA_MODES]; INTRA_MODES] {
    let mut out = [[0i32; INTRA_MODES]; INTRA_MODES];
    for (i, probs) in VP9_KF_UV_MODE_PROB.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_INTRA_MODE_TREE, probs);
    }
    out
}

/// Per-context inter-mode cost table for non-keyframe inter blocks.
/// Mirrors libvpx `cpi->inter_mode_cost` built by
/// `vp9_build_inter_mode_cost` (`vp9_rd.c:387-394`).
///
/// Inner indexing matches `INTER_OFFSET(mode) = mode - NEARESTMV`,
/// i.e. `0 = NEARESTMV, 1 = NEARMV, 2 = ZEROMV, 3 = NEWMV`.
#[must_use]
pub fn inter_mode_costs(fc: &FrameContext) -> InterModeCosts {
    let mut out: InterModeCosts = [[0i32; INTER_MODES]; INTER_MODE_CONTEXTS];
    for (i, probs) in fc.inter_mode_probs.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_INTER_MODE_TREE, probs);
    }
    out
}

/// Per-context partition cost table for non-keyframes. Mirrors libvpx
/// `cpi->partition_cost[i]` (`vp9_rd.c:430-432`) — built from
/// `fc->partition_prob` since `frame_is_intra_only(cm)` is false.
#[must_use]
pub fn partition_costs(fc: &FrameContext) -> PartitionCosts {
    let mut out: PartitionCosts = [[0i32; PARTITION_TYPES]; PARTITION_CONTEXTS];
    for (i, probs) in fc.partition_prob.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_PARTITION_TREE, probs);
    }
    out
}

/// Per-context partition cost table for keyframes — uses the static
/// `VP9_KF_PARTITION_PROBS` rather than the frame context.
#[must_use]
pub fn partition_costs_kf() -> KfPartitionCosts {
    let mut out: KfPartitionCosts = [[0i32; PARTITION_TYPES]; PARTITION_CONTEXTS];
    for (i, probs) in VP9_KF_PARTITION_PROBS.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_PARTITION_TREE, probs);
    }
    out
}

/// Per-context switchable-interpolation-filter cost table. Mirrors
/// libvpx `cpi->switchable_interp_costs[i]` (`vp9_rd.c:111-114`).
#[must_use]
pub fn switchable_interp_costs(fc: &FrameContext) -> SwitchableInterpCosts {
    let mut out: SwitchableInterpCosts = [[0i32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS];
    for (i, probs) in fc.switchable_interp_prob.iter().enumerate() {
        vp9_cost_tokens(&mut out[i], &VP9_SWITCHABLE_INTERP_TREE, probs);
    }
    out
}

/// Per-context skip-bit cost table. `[ctx][bit] = bit_cost(skip_prob,
/// bit)`. Each skip bit is a leaf single-bit decision, so the cost is
/// just `cost_bit(prob, bit)`.
#[must_use]
pub fn skip_costs(fc: &FrameContext) -> SkipCosts {
    let mut out: SkipCosts = [[0i32; 2]; SKIP_CONTEXTS];
    for (i, &prob) in fc.skip_probs.iter().enumerate() {
        out[i][0] = vp9_cost_zero(prob) as i32;
        out[i][1] = vp9_cost_one(prob) as i32;
    }
    out
}

/// Per-`max_tx_size` × per-context × per-actual-size TX-size cost
/// table. Mirrors libvpx `vp9_initialize_rd_consts`
/// (`vp9_rd.c:116-132`).
///
/// VP9 encodes `actual_tx_size` as a sequence of binary decisions —
/// `(actual == max)?`, then `(actual == max - 1)?`, etc. — each under a
/// different probability from `tx_probs.p{32x32,16x16,8x8}[ctx][k]`.
/// For `actual = k`, all earlier bits are `1` (still bigger) and the
/// `k`-th bit is `0` (settled). The leaf cost is the sum of those.
#[must_use]
pub fn tx_size_costs(fc: &FrameContext) -> TxSizeCostTable {
    let mut out = TxSizeCostTable::default();
    for max in 1..TX_SIZES {
        // max_tx_size index: 1 = TX_8X8, 2 = TX_16X16, 3 = TX_32X32.
        for ctx in 0..TX_SIZE_CONTEXTS {
            let probs: &[u8] = match max {
                1 => &fc.tx_probs.p8x8[ctx],
                2 => &fc.tx_probs.p16x16[ctx],
                3 => &fc.tx_probs.p32x32[ctx],
                _ => unreachable!(),
            };
            for k in 0..=max {
                let mut cost: i32 = 0;
                let limit = if k == max { k } else { k + 1 };
                for (m, &p) in probs.iter().take(limit).enumerate() {
                    cost += if m == k {
                        vp9_cost_zero(p) as i32
                    } else {
                        vp9_cost_one(p) as i32
                    };
                }
                out.data[max - 1][ctx][k] = cost;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::frame_context::FrameContext;

    #[test]
    fn y_mode_costs_inter_default_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let table = intra_y_mode_costs_inter(&fc);
        for group in &table {
            for &c in group {
                assert!(c > 0, "intra-Y cost must be positive, got {c}");
                assert!(c < 100_000, "intra-Y cost suspiciously large: {c}");
            }
        }
    }

    #[test]
    fn uv_mode_costs_inter_default_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let table = intra_uv_mode_costs_inter(&fc);
        for row in &table {
            for &c in row {
                assert!(c > 0);
                assert!(c < 100_000);
            }
        }
    }

    #[test]
    fn partition_costs_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let pc = partition_costs(&fc);
        for ctx in &pc {
            for &c in ctx {
                assert!(c > 0);
            }
        }
        // Keyframe partition costs likewise.
        let pck = partition_costs_kf();
        for ctx in &pck {
            for &c in ctx {
                assert!(c > 0);
            }
        }
    }

    #[test]
    fn inter_mode_costs_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let table = inter_mode_costs(&fc);
        for ctx in &table {
            for &c in ctx {
                assert!(c > 0);
            }
        }
    }

    #[test]
    fn skip_costs_match_cost_bit() {
        let fc = FrameContext::new_default();
        let sc = skip_costs(&fc);
        for (i, &prob) in fc.skip_probs.iter().enumerate() {
            assert_eq!(sc[i][0], vp9_cost_zero(prob) as i32);
            assert_eq!(sc[i][1], vp9_cost_one(prob) as i32);
        }
    }

    #[test]
    fn switchable_interp_costs_finite_and_nonzero() {
        let fc = FrameContext::new_default();
        let table = switchable_interp_costs(&fc);
        for row in &table {
            for &c in row {
                assert!(c > 0);
            }
        }
    }

    #[test]
    fn tx_size_costs_default_nonzero_for_valid_cells() {
        let fc = FrameContext::new_default();
        let t = tx_size_costs(&fc);
        // For max == TX_32X32, all 4 actual sizes should produce
        // strictly positive cost (each path costs at least one bit).
        for ctx in 0..TX_SIZE_CONTEXTS {
            for actual in 0..=3usize {
                let c = t.data[2][ctx][actual];
                assert!(c > 0, "tx_size_cost[2][{ctx}][{actual}] = {c}");
            }
        }
    }

    #[test]
    fn kf_intra_mode_costs_finite() {
        let table = intra_y_mode_costs_kf_table();
        for plane in &table {
            for row in plane {
                for &c in row {
                    assert!(c > 0);
                    assert!(c < 200_000);
                }
            }
        }
        let uv = intra_uv_mode_costs_kf();
        for row in &uv {
            for &c in row {
                assert!(c > 0);
            }
        }
    }
}
