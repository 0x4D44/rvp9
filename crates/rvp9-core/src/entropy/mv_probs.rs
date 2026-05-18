//! VP9 motion-vector probability tables (Task 2b.5).
//!
//! Mirrors libvpx `nmv_context` definition (`vp9_entropymv.h:88-102`)
//! plus the `default_nmv_context` initialiser
//! (`vp9_entropymv.c:30-54`).
//!
//! ## Structure
//!
//! Each MV is encoded as a 4-way "joint" (zero / vert-only /
//! horz-only / both) plus two `NmvComponent`s — one per axis. Per
//! component:
//!
//! - 1 bit sign
//! - 10 bits for the magnitude class (`MV_CLASSES - 1`, tree-coded
//!   over 11 classes)
//! - 1 bit for `class0` (selects which side of `CLASS0_SIZE = 2`)
//! - 10 bits for the per-class offset bits
//! - `CLASS0_SIZE × (MV_FP_SIZE - 1) = 2 × 3` fractional-pel probs
//!   for class 0
//! - 3 fractional-pel probs for other classes
//! - 1 bit class0 high-precision (1/8-pel) flag
//! - 1 bit high-precision flag for other classes

/// Number of MV joints (`vp9_entropymv.h:38`).
pub const MV_JOINTS: usize = 4;

/// Number of MV magnitude classes (`vp9_entropymv.h:55`).
pub const MV_CLASSES: usize = 11;

/// Number of bits at integer precision for class 0 (`vp9_entropymv.h:70`).
pub const CLASS0_BITS: usize = 1;

/// `1 << CLASS0_BITS` (`vp9_entropymv.h:71`).
pub const CLASS0_SIZE: usize = 1 << CLASS0_BITS;

/// `MV_CLASSES + CLASS0_BITS - 2 = 10` (`vp9_entropymv.h:72`).
pub const MV_OFFSET_BITS: usize = MV_CLASSES + CLASS0_BITS - 2;

/// Number of fractional-pel positions (`vp9_entropymv.h:73`).
pub const MV_FP_SIZE: usize = 4;

/// One MV-axis component probability set. Mirrors libvpx
/// `nmv_component` (`vp9_entropymv.h:88-97`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NmvComponent {
    /// Sign bit probability.
    pub sign: u8,
    /// Magnitude-class tree probabilities (one per branch).
    pub classes: [u8; MV_CLASSES - 1],
    /// Probabilities for the class-0 sub-tree.
    pub class0: [u8; CLASS0_SIZE - 1],
    /// Probabilities for the per-class offset bits.
    pub bits: [u8; MV_OFFSET_BITS],
    /// Class-0 fractional-pel tree probabilities (one set per
    /// `class0` value).
    pub class0_fp: [[u8; MV_FP_SIZE - 1]; CLASS0_SIZE],
    /// Fractional-pel tree probabilities for non-class-0 magnitudes.
    pub fp: [u8; MV_FP_SIZE - 1],
    /// Class-0 high-precision (1/8-pel) flag probability.
    pub class0_hp: u8,
    /// High-precision flag probability for non-class-0 magnitudes.
    pub hp: u8,
}

/// Full MV probability set: joint probabilities + a per-axis
/// component pair. Mirrors libvpx `nmv_context`
/// (`vp9_entropymv.h:99-102`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NmvContext {
    /// Probabilities for the 4-way joint tree.
    pub joints: [u8; MV_JOINTS - 1],
    /// Per-axis components: `[vertical, horizontal]`.
    pub comps: [NmvComponent; 2],
}

/// Default MV probabilities. Verbatim from libvpx
/// `default_nmv_context` (`vp9_entropymv.c:30-54`).
pub const DEFAULT_NMV_CONTEXT: NmvContext = NmvContext {
    joints: [32, 64, 96],
    comps: [
        // Vertical component.
        NmvComponent {
            sign: 128,
            classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
            class0: [216],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        },
        // Horizontal component.
        NmvComponent {
            sign: 128,
            classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
            class0: [208],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        },
    ],
};

/// Per-axis MV-component count bank for backward adaptation.
/// Mirrors libvpx `nmv_component_counts` (`vp9_entropymv.h:114-123`).
///
/// Each field counts how often a particular boolean choice was decoded
/// (or encoded). The decoder's per-block MV-read path increments these
/// when `counts.is_some()`; at frame end the merged probs feed
/// `vp9_adapt_mv_probs`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NmvComponentCounts {
    /// Sign bit observations (`[#neg, #pos]` mapping to libvpx's
    /// `sign[2]`, where index `0` is "positive / not negated" and
    /// index `1` is "negative", matching `s = v < 0`).
    pub sign: [u32; 2],
    /// Magnitude-class hits, one slot per `MV_CLASS_*`.
    pub classes: [u32; MV_CLASSES],
    /// Class-0 integer-bit hits.
    pub class0: [u32; CLASS0_SIZE],
    /// Per-bit binary observations for the non-class-0 magnitude tail.
    /// `bits[i][0]` is "saw a 0 at offset bit i"; `bits[i][1]` is "saw a 1".
    pub bits: [[u32; 2]; MV_OFFSET_BITS],
    /// Class-0 fractional-pel observations, one row per `class0` value.
    pub class0_fp: [[u32; MV_FP_SIZE]; CLASS0_SIZE],
    /// Non-class-0 fractional-pel observations.
    pub fp: [u32; MV_FP_SIZE],
    /// Class-0 high-precision bit observations (`[#0, #1]`).
    pub class0_hp: [u32; 2],
    /// Non-class-0 high-precision bit observations (`[#0, #1]`).
    pub hp: [u32; 2],
}

/// Full MV count bank: joint counts + per-axis component counts.
/// Mirrors libvpx `nmv_context_counts` (`vp9_entropymv.h:125-128`).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NmvCounts {
    /// Joint-tree leaf counts (`MV_JOINT_*`).
    pub joints: [u32; MV_JOINTS],
    /// Per-axis component counts: `[vertical, horizontal]`.
    pub comps: [NmvComponentCounts; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_constants_match_libvpx() {
        assert_eq!(MV_JOINTS, 4);
        assert_eq!(MV_CLASSES, 11);
        assert_eq!(CLASS0_BITS, 1);
        assert_eq!(CLASS0_SIZE, 2);
        assert_eq!(MV_OFFSET_BITS, 10);
        assert_eq!(MV_FP_SIZE, 4);
    }

    #[test]
    fn default_nmv_joints() {
        assert_eq!(DEFAULT_NMV_CONTEXT.joints, [32, 64, 96]);
    }

    #[test]
    fn default_nmv_vertical_component() {
        let v = &DEFAULT_NMV_CONTEXT.comps[0];
        assert_eq!(v.sign, 128);
        assert_eq!(v.classes, [224, 144, 192, 168, 192, 176, 192, 198, 198, 245]);
        assert_eq!(v.class0, [216]);
        assert_eq!(v.bits, [136, 140, 148, 160, 176, 192, 224, 234, 234, 240]);
        assert_eq!(v.class0_fp, [[128, 128, 64], [96, 112, 64]]);
        assert_eq!(v.fp, [64, 96, 64]);
        assert_eq!(v.class0_hp, 160);
        assert_eq!(v.hp, 128);
    }

    #[test]
    fn default_nmv_horizontal_component() {
        let h = &DEFAULT_NMV_CONTEXT.comps[1];
        assert_eq!(h.sign, 128);
        assert_eq!(h.classes, [216, 128, 176, 160, 176, 176, 192, 198, 198, 208]);
        assert_eq!(h.class0, [208]);
        // bits, class0_fp, fp, class0_hp, hp are identical to vertical.
        assert_eq!(h.bits, DEFAULT_NMV_CONTEXT.comps[0].bits);
        assert_eq!(h.class0_fp, DEFAULT_NMV_CONTEXT.comps[0].class0_fp);
        assert_eq!(h.fp, DEFAULT_NMV_CONTEXT.comps[0].fp);
        assert_eq!(h.class0_hp, 160);
        assert_eq!(h.hp, 128);
    }

    /// Sanity: every probability in the default context is non-zero.
    #[test]
    fn no_zero_probs() {
        for &p in DEFAULT_NMV_CONTEXT.joints.iter() {
            assert!(p > 0);
        }
        for comp in DEFAULT_NMV_CONTEXT.comps.iter() {
            assert!(comp.sign > 0);
            for &p in comp.classes.iter() {
                assert!(p > 0);
            }
            for &p in comp.class0.iter() {
                assert!(p > 0);
            }
            for &p in comp.bits.iter() {
                assert!(p > 0);
            }
            for row in comp.class0_fp.iter() {
                for &p in row.iter() {
                    assert!(p > 0);
                }
            }
            for &p in comp.fp.iter() {
                assert!(p > 0);
            }
            assert!(comp.class0_hp > 0);
            assert!(comp.hp > 0);
        }
    }
}
