//! VP9 default probability tables (Task 2b.1 — small mode probs).
//!
//! Source: libvpx `vp9/common/vp9_entropymode.c` lines 162-321 (the
//! constants assigned in `init_mode_probs`). Each `vpx_prob` is a
//! `u8` in the `[0, 255]` range — `128` is the maximum-entropy point.
//!
//! ## Scope of this commit (Task 2b.1)
//!
//! Small `init_mode_probs` tables that fit easily in one module:
//!
//! - Intra-mode probs (non-keyframe): [`DEFAULT_IF_Y_PROBS`],
//!   [`DEFAULT_IF_UV_PROBS`].
//! - Inter-frame structural probs:
//!   [`DEFAULT_INTRA_INTER_PROB`], [`DEFAULT_COMP_INTER_PROB`],
//!   [`DEFAULT_COMP_REF_PROB`], [`DEFAULT_SINGLE_REF_PROB`].
//! - Inter-mode tree probs: [`DEFAULT_INTER_MODE_PROBS`].
//! - Partition probs (non-keyframe): [`DEFAULT_PARTITION_PROBS`].
//! - Skip / switchable-interp / tx_size probs:
//!   [`DEFAULT_SKIP_PROBS`], [`DEFAULT_SWITCHABLE_INTERP_PROB`],
//!   [`DEFAULT_TX_PROBS`].
//!
//! ## Not in this commit
//!
//! - Keyframe mode probs (`vp9_kf_y_mode_prob` is `10 * 10 * 9 =
//!   900` entries; `vp9_kf_uv_mode_prob` is `90`; `vp9_kf_partition_probs`
//!   is `48`) — Task 2b.2.
//! - Pareto8 distribution (`vp9_pareto8full` — 256 × 8 entries) —
//!   Task 2b.3.
//! - Coefficient probability tables (`default_coef_probs_*x*[]`) —
//!   Task 2b.4.
//! - MV probability tables (`default_nmv_context`) — Task 2b.5.
//!
//! ## Constants reference
//!
//! VP9 enum sizes from `vp9_enums.h`:
//!
//! - `INTRA_MODES = 10` (DC / V / H / D45 / D135 / D117 / D153 / D207
//!   / D63 / TM).
//! - `INTER_MODES = 4` (NEARESTMV / NEARMV / ZEROMV / NEWMV).
//! - `BLOCK_SIZE_GROUPS = 4` (block-size-grouped contexts for intra-Y
//!   mode probabilities).
//! - `PARTITION_CONTEXTS = 4 * 4 = 16` (4 block-sizes × 4
//!   above/left-split combinations).
//! - `INTRA_INTER_CONTEXTS = 4`, `COMP_INTER_CONTEXTS = 5`,
//!   `REF_CONTEXTS = 5`, `SKIP_CONTEXTS = 3`, `INTER_MODE_CONTEXTS = 7`.
//! - `SWITCHABLE_FILTERS = 3`, `SWITCHABLE_FILTER_CONTEXTS = 4`.

/// Number of distinct VP9 intra prediction modes
/// (`vp9_enums.h:127`).
pub const INTRA_MODES: usize = 10;

/// Number of VP9 inter modes (`vp9_enums.h:129`).
pub const INTER_MODES: usize = 4;

/// Number of block-size groups used by intra-Y mode probs
/// (`vp9_entropymode.h:23`).
pub const BLOCK_SIZE_GROUPS: usize = 4;

/// Number of partition-tree probability contexts
/// (`vp9_enums.h:74`).
pub const PARTITION_CONTEXTS: usize = 16;

/// Number of partition types (NONE / HORZ / VERT / SPLIT)
/// (`vp9_enums.h:68`).
pub const PARTITION_TYPES: usize = 4;

/// Number of intra/inter classification contexts
/// (`vp9_enums.h:137`).
pub const INTRA_INTER_CONTEXTS: usize = 4;

/// Number of compound-inter contexts (`vp9_enums.h:138`).
pub const COMP_INTER_CONTEXTS: usize = 5;

/// Number of reference-frame contexts (`vp9_enums.h:139`).
pub const REF_CONTEXTS: usize = 5;

/// Number of skip-flag contexts (`vp9_enums.h:131`).
pub const SKIP_CONTEXTS: usize = 3;

/// Number of inter-mode contexts (`vp9_enums.h:132`).
pub const INTER_MODE_CONTEXTS: usize = 7;

/// Number of switchable interpolation filters (`vp9_filter.h:26`).
pub const SWITCHABLE_FILTERS: usize = 3;

/// Number of contexts for the switchable-interp tree
/// (`vp9_filter.h:31`).
pub const SWITCHABLE_FILTER_CONTEXTS: usize = 4;

/// Default intra-Y mode probs for non-keyframes.
/// `INTRA_MODES - 1 = 9` probabilities per `BLOCK_SIZE_GROUPS = 4`
/// context. Mirrors libvpx `default_if_y_probs`
/// (`vp9_entropymode.c:162-167`).
#[rustfmt::skip]
pub static DEFAULT_IF_Y_PROBS: [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS] = [
    [65, 32, 18, 144, 162, 194, 41, 51, 98],   // block_size < 8x8
    [132, 68, 18, 165, 217, 196, 45, 40, 78],  // block_size < 16x16
    [173, 80, 19, 176, 240, 193, 64, 35, 46],  // block_size < 32x32
    [221, 135, 38, 194, 248, 121, 96, 85, 29], // block_size >= 32x32
];

/// Default intra-UV mode probs (non-keyframe). Per-Y-mode context.
/// `INTRA_MODES - 1 = 9` probs × `INTRA_MODES = 10` Y-mode contexts.
/// Mirrors libvpx `default_if_uv_probs`
/// (`vp9_entropymode.c:169-180`).
#[rustfmt::skip]
pub static DEFAULT_IF_UV_PROBS: [[u8; INTRA_MODES - 1]; INTRA_MODES] = [
    [120, 7, 76, 176, 208, 126, 28, 54, 103],  // y = dc
    [48, 12, 154, 155, 139, 90, 34, 117, 119], // y = v
    [67, 6, 25, 204, 243, 158, 13, 21, 96],    // y = h
    [97, 5, 44, 131, 176, 139, 48, 68, 97],    // y = d45
    [83, 5, 42, 156, 111, 152, 26, 49, 152],   // y = d135
    [80, 5, 58, 178, 74, 83, 33, 62, 145],     // y = d117
    [86, 5, 32, 154, 192, 168, 14, 22, 163],   // y = d153
    [85, 5, 32, 156, 216, 148, 19, 29, 73],    // y = d207
    [77, 7, 64, 116, 132, 122, 37, 126, 120],  // y = d63
    [101, 21, 107, 181, 192, 103, 19, 67, 125],// y = tm
];

/// Default partition probs (non-keyframe). 3 probs per
/// `PARTITION_CONTEXTS = 16` contexts (4 block-sizes × 4
/// above/left-split combinations). Mirrors libvpx
/// `default_partition_probs` (`vp9_entropymode.c:209-231`).
#[rustfmt::skip]
pub static DEFAULT_PARTITION_PROBS: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS] = [
    // 8x8 -> 4x4
    [199, 122, 141], // a/l both not split
    [147, 63, 159],  // a split, l not split
    [148, 133, 118], // l split, a not split
    [121, 104, 114], // a/l both split
    // 16x16 -> 8x8
    [174, 73, 87],
    [92, 41, 83],
    [82, 99, 50],
    [53, 39, 39],
    // 32x32 -> 16x16
    [177, 58, 59],
    [68, 26, 63],
    [52, 79, 25],
    [17, 14, 12],
    // 64x64 -> 32x32
    [222, 34, 30],
    [72, 16, 44],
    [58, 32, 12],
    [10, 7, 6],
];

/// Default inter-mode probs. 3 tree probs per
/// `INTER_MODE_CONTEXTS = 7` contexts. Mirrors libvpx
/// `default_inter_mode_probs` (`vp9_entropymode.c:233-242`).
#[rustfmt::skip]
pub static DEFAULT_INTER_MODE_PROBS: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS] = [
    [2, 173, 34], // 0 = both zero mv
    [7, 145, 85], // 1 = one zero mv + one a predicted mv
    [7, 166, 63], // 2 = two predicted mvs
    [7, 94, 66],  // 3 = one predicted/zero and one new mv
    [8, 64, 46],  // 4 = two new mvs
    [17, 81, 31], // 5 = one intra neighbour + x
    [25, 29, 30], // 6 = two intra neighbours
];

/// Default intra/inter classification prob (per context). Mirrors
/// libvpx `default_intra_inter_p` (`vp9_entropymode.c:266-268`).
pub static DEFAULT_INTRA_INTER_PROB: [u8; INTRA_INTER_CONTEXTS] = [9, 102, 187, 225];

/// Default compound-inter prob. Mirrors libvpx `default_comp_inter_p`
/// (`vp9_entropymode.c:270-272`).
pub static DEFAULT_COMP_INTER_PROB: [u8; COMP_INTER_CONTEXTS] = [239, 183, 119, 96, 41];

/// Default compound-ref prob. Mirrors libvpx `default_comp_ref_p`
/// (`vp9_entropymode.c:274-275`).
pub static DEFAULT_COMP_REF_PROB: [u8; REF_CONTEXTS] = [50, 126, 123, 221, 226];

/// Default single-ref probs. 2 per context. Mirrors libvpx
/// `default_single_ref_p` (`vp9_entropymode.c:277-279`).
#[rustfmt::skip]
pub static DEFAULT_SINGLE_REF_PROB: [[u8; 2]; REF_CONTEXTS] = [
    [33, 16],
    [77, 74],
    [142, 142],
    [172, 170],
    [238, 247],
];

/// Default skip-flag prob, per context. Mirrors libvpx
/// `default_skip_probs` (`vp9_entropymode.c:313`).
pub static DEFAULT_SKIP_PROBS: [u8; SKIP_CONTEXTS] = [192, 128, 64];

/// Default switchable interpolation-filter prob. 2 tree probs per
/// `SWITCHABLE_FILTER_CONTEXTS = 4`. Mirrors libvpx
/// `default_switchable_interp_prob` (`vp9_entropymode.c:315-321`).
#[rustfmt::skip]
pub static DEFAULT_SWITCHABLE_INTERP_PROB:
    [[u8; SWITCHABLE_FILTERS - 1]; SWITCHABLE_FILTER_CONTEXTS] = [
    [235, 162],
    [36, 255],
    [34, 3],
    [149, 144],
];

/// Default tx-size probs. Three sub-tables (16×16, 32×32, 8×8) with
/// per-context entries. Mirrors libvpx `struct tx_probs default_tx_probs`
/// (`vp9_entropymode.c:281-286` + `vp9_entropymode.h` struct definition).
///
/// Layout:
/// - `p32x32[2][3]`: per-context probs that the TX size is 32x32 /
///   16x16 / 8x8 (3 tree probs each).
/// - `p16x16[2][2]`: per-context probs that the TX size is 16x16 /
///   8x8 (2 tree probs each).
/// - `p8x8[2][1]`: per-context prob that the TX size is 8x8 (1 prob).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DefaultTxProbs {
    /// Per-context TX-size probs for 32×32 max-size contexts.
    pub p32x32: [[u8; 3]; 2],
    /// Per-context TX-size probs for 16×16 max-size contexts.
    pub p16x16: [[u8; 2]; 2],
    /// Per-context TX-size probs for 8×8 max-size contexts.
    pub p8x8: [[u8; 1]; 2],
}

/// Default tx_probs values from libvpx `vp9_entropymode.c:281-286`.
pub const DEFAULT_TX_PROBS: DefaultTxProbs = DefaultTxProbs {
    p32x32: [[3, 136, 37], [5, 52, 13]],
    p16x16: [[20, 152], [15, 101]],
    p8x8: [[100], [66]],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every default probability must lie in `[1, 255]` — libvpx never
    /// ships a `0` or `MAX_PROB`-as-sentinel default. (Note: `MAX_PROB =
    /// 255` is reserved as a "no update" sentinel in the segmentation
    /// header parser; in mode probabilities, 255 would mean
    /// "deterministically the right branch" — uncommon but not
    /// invalid.) This test just pins that defaults are non-zero.
    #[test]
    fn all_mode_probs_nonzero() {
        for row in DEFAULT_IF_Y_PROBS.iter() {
            for &p in row.iter() {
                assert!(p > 0, "DEFAULT_IF_Y_PROBS contains 0");
            }
        }
        for row in DEFAULT_IF_UV_PROBS.iter() {
            for &p in row.iter() {
                assert!(p > 0, "DEFAULT_IF_UV_PROBS contains 0");
            }
        }
        for &p in DEFAULT_INTRA_INTER_PROB.iter() {
            assert!(p > 0, "DEFAULT_INTRA_INTER_PROB contains 0");
        }
        for &p in DEFAULT_COMP_INTER_PROB.iter() {
            assert!(p > 0, "DEFAULT_COMP_INTER_PROB contains 0");
        }
    }

    /// Dimension sanity for the larger tables.
    #[test]
    fn table_shapes_match_libvpx() {
        assert_eq!(DEFAULT_IF_Y_PROBS.len(), BLOCK_SIZE_GROUPS);
        assert_eq!(DEFAULT_IF_Y_PROBS[0].len(), INTRA_MODES - 1);
        assert_eq!(DEFAULT_IF_UV_PROBS.len(), INTRA_MODES);
        assert_eq!(DEFAULT_IF_UV_PROBS[0].len(), INTRA_MODES - 1);
        assert_eq!(DEFAULT_PARTITION_PROBS.len(), PARTITION_CONTEXTS);
        assert_eq!(DEFAULT_PARTITION_PROBS[0].len(), PARTITION_TYPES - 1);
        assert_eq!(DEFAULT_INTER_MODE_PROBS.len(), INTER_MODE_CONTEXTS);
        assert_eq!(DEFAULT_INTER_MODE_PROBS[0].len(), INTER_MODES - 1);
        assert_eq!(DEFAULT_INTRA_INTER_PROB.len(), INTRA_INTER_CONTEXTS);
        assert_eq!(DEFAULT_COMP_INTER_PROB.len(), COMP_INTER_CONTEXTS);
        assert_eq!(DEFAULT_COMP_REF_PROB.len(), REF_CONTEXTS);
        assert_eq!(DEFAULT_SINGLE_REF_PROB.len(), REF_CONTEXTS);
        assert_eq!(DEFAULT_SKIP_PROBS.len(), SKIP_CONTEXTS);
        assert_eq!(
            DEFAULT_SWITCHABLE_INTERP_PROB.len(),
            SWITCHABLE_FILTER_CONTEXTS
        );
        assert_eq!(
            DEFAULT_SWITCHABLE_INTERP_PROB[0].len(),
            SWITCHABLE_FILTERS - 1
        );
    }

    /// Spot-check the spec-quoted values that libvpx burns into the
    /// init_mode_probs initializer.
    #[test]
    fn spec_quoted_spot_checks() {
        // Y-mode group 0 first prob (libvpx :163).
        assert_eq!(DEFAULT_IF_Y_PROBS[0][0], 65);
        // UV-mode given y=dc, first prob (libvpx :170).
        assert_eq!(DEFAULT_IF_UV_PROBS[0][0], 120);
        // Partition: 8x8 a/l both split (libvpx :215).
        assert_eq!(DEFAULT_PARTITION_PROBS[3], [121, 104, 114]);
        // Skip context 1 = 128 = VPX_PROB_HALF.
        assert_eq!(DEFAULT_SKIP_PROBS[1], 128);
        // tx_probs: p32x32[0] (libvpx :281-282).
        assert_eq!(DEFAULT_TX_PROBS.p32x32[0], [3, 136, 37]);
        // intra-inter: all four contexts.
        assert_eq!(DEFAULT_INTRA_INTER_PROB, [9, 102, 187, 225]);
    }
}
