//! Motion-vector cost machinery.
//!
//! Mirrors libvpx `build_nmv_component_cost_table`,
//! `vp9_build_nmv_cost_table`, the fixed SAD-domain tables used by
//! full-pixel search, and the joint-then-per-axis MV-cost computation in
//! `vp9_encode_mv` from `vp9/encoder/vp9_encodemv.c:69-237`.
//!
//! ## Structure
//!
//! - `joint_cost[MV_JOINTS]` — cost of each of the 4 MV joints
//!   (`ZERO / HNZ-VZ / HZ-VNZ / HNZ-VNZ`).
//! - `MvComponentCosts.cost[i + MV_MAX]` for `i ∈ [-MV_MAX, MV_MAX]`
//!   — cost of encoding a single MV component value `i` (full-pel
//!   or higher-precision; signed, in 1/8-pel units).
//!
//! Mode-decision computes the cost of an MV `(row, col)` relative to
//! a reference MV by:
//!
//! 1. Forming the diff `(mv.row - ref.row, mv.col - ref.col)`.
//! 2. Looking up the joint cost via [`mv_joint`] applied to the diff.
//! 3. Adding the relevant component cost(s) — vertical if the diff's
//!    row is non-zero, horizontal if the diff's col is non-zero.
//!
//! The high-precision (`usehp`) variant adds an extra hp bit per
//! non-zero component; libvpx uses it only when
//! `cm->allow_high_precision_mv && use_mv_hp(ref)`.

use super::prob_cost::{vp9_cost_one, vp9_cost_zero};
use super::tree_cost::vp9_cost_tokens;
use crate::entropy::adapt::{
    VP9_MV_CLASS_TREE, VP9_MV_CLASS0_TREE, VP9_MV_FP_TREE, VP9_MV_JOINT_TREE,
};
use crate::entropy::mv_probs::{
    CLASS0_BITS, CLASS0_SIZE, MV_CLASSES, MV_FP_SIZE, MV_JOINTS, MV_OFFSET_BITS, NmvComponent,
    NmvContext,
};

/// `MV_MAX_BITS = MV_CLASSES + CLASS0_BITS + 2` in libvpx
/// (`vp9_entropymv.h:75`). `= 14` for the default `(MV_CLASSES = 11,
/// CLASS0_BITS = 1)`.
pub const MV_MAX_BITS: usize = MV_CLASSES + CLASS0_BITS + 2;

/// Maximum absolute value of an encoded MV component, in 1/8-pel
/// units. `MV_MAX = (1 << MV_MAX_BITS) - 1 = 16383`.
pub const MV_MAX: i32 = (1 << MV_MAX_BITS) - 1;

/// Number of distinct signed MV-component values: `2 * MV_MAX + 1`.
pub const MV_VALS: usize = (MV_MAX as usize) * 2 + 1;

/// Upper bound on the magnitude of a single bit-cost entry — useful
/// for callers that want to budget. Conservatively each MV component
/// can cost up to ~20 internal-node decisions × 4096 cost-units each
/// ≈ 80_000 prob-cost units.
pub const MV_COST_MAX: i32 = 200_000;

/// Per-axis component cost table — cost of encoding an MV component
/// value `i ∈ [-MV_MAX, MV_MAX]` lives in `cost[i + MV_MAX as usize]`.
/// Mirrors the `mvcost[0]`/`mvcost[1]` arrays libvpx allocates
/// per encoder thread (`MACROBLOCK::nmvcost[2][MV_VALS]`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MvComponentCosts {
    /// Indexed by `i + MV_MAX` (so element 0 corresponds to MV = -MV_MAX,
    /// element `MV_MAX` corresponds to MV = 0, and the last to +MV_MAX).
    pub cost: Box<[i32]>,
}

impl MvComponentCosts {
    /// Lookup the cost of MV component value `value`. Returns 0 if
    /// `value == 0` (libvpx convention — encoding the zero diff under
    /// the joint tree handles the zero leaf, not the per-component
    /// table). Clamps to the table edge if `value` is out of range.
    #[inline]
    #[must_use]
    pub fn lookup(&self, value: i16) -> i32 {
        let v = value as i32;
        let clamped = v.clamp(-MV_MAX, MV_MAX);
        let idx = (clamped + MV_MAX) as usize;
        self.cost[idx]
    }
}

/// Joint + per-axis MV cost bank. Mirrors libvpx
/// `(nmvjointcost, nmvcost[2])` (`vp9_block.h` per-thread MB struct).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MvCosts {
    /// Cost of each of the 4 MV joints.
    pub joint_cost: [i32; MV_JOINTS],
    /// `[0]` is the vertical (row) axis, `[1]` is the horizontal (col)
    /// axis — matching `NmvContext::comps[0]` / `comps[1]`.
    pub component_cost: [MvComponentCosts; 2],
}

/// Compute the MV joint index for a diff MV. Mirrors libvpx
/// `vp9_get_mv_joint` (`vp9_entropymv.h:36-46`).
///
/// Joint layout:
/// - `MV_JOINT_ZERO = 0` — both axes zero.
/// - `MV_JOINT_HNZVZ = 1` — horizontal non-zero, vertical zero.
/// - `MV_JOINT_HZVNZ = 2` — horizontal zero, vertical non-zero.
/// - `MV_JOINT_HNZVNZ = 3` — both non-zero.
#[inline]
#[must_use]
pub fn mv_joint(row: i16, col: i16) -> usize {
    let v = row != 0;
    let h = col != 0;
    match (h, v) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

/// Build one axis's `MvComponentCosts` from a `NmvComponent`. Mirrors
/// libvpx `build_nmv_component_cost_table` (`vp9_encodemv.c:69-141`).
///
/// `usehp` controls whether the high-precision bit is included in the
/// per-value cost. Mode-decision passes `false` while searching at
/// 1/4-pel and `true` for the 1/8-pel refinement pass.
#[must_use]
pub fn build_nmv_component_cost_table(mvcomp: &NmvComponent, usehp: bool) -> MvComponentCosts {
    let mut cost = vec![0i32; MV_VALS].into_boxed_slice();

    let sign_cost = [
        vp9_cost_zero(mvcomp.sign) as i32,
        vp9_cost_one(mvcomp.sign) as i32,
    ];

    let mut class_cost = [0i32; MV_CLASSES];
    vp9_cost_tokens(&mut class_cost, &VP9_MV_CLASS_TREE, &mvcomp.classes);

    let mut class0_cost = [0i32; CLASS0_SIZE];
    vp9_cost_tokens(&mut class0_cost, &VP9_MV_CLASS0_TREE, &mvcomp.class0);

    let mut bits_cost = [[0i32; 2]; MV_OFFSET_BITS];
    for (slot, &p) in bits_cost.iter_mut().zip(mvcomp.bits.iter()) {
        slot[0] = vp9_cost_zero(p) as i32;
        slot[1] = vp9_cost_one(p) as i32;
    }

    let mut class0_fp_cost = [[0i32; MV_FP_SIZE]; CLASS0_SIZE];
    for (i, probs) in mvcomp.class0_fp.iter().enumerate() {
        vp9_cost_tokens(&mut class0_fp_cost[i], &VP9_MV_FP_TREE, probs);
    }

    let mut fp_cost = [0i32; MV_FP_SIZE];
    vp9_cost_tokens(&mut fp_cost, &VP9_MV_FP_TREE, &mvcomp.fp);

    // Always build the hp costs — libvpx does too, even when usehp is
    // false, to dodge a gcc uninitialized-warning.
    let class0_hp_cost = [
        vp9_cost_zero(mvcomp.class0_hp) as i32,
        vp9_cost_one(mvcomp.class0_hp) as i32,
    ];
    let hp_cost = [
        vp9_cost_zero(mvcomp.hp) as i32,
        vp9_cost_one(mvcomp.hp) as i32,
    ];

    let mv_max = MV_MAX;
    let zero_idx = mv_max as usize;
    cost[zero_idx] = 0;

    // MV_CLASS_0 (`o` is the offset within the 0-class range).
    // The libvpx loop iterates `o ∈ [0, CLASS0_SIZE << 3)` —
    // `CLASS0_SIZE = 2`, so 16 entries (8 fractional positions × 2
    // class0 positions). `v = o + 1` indexes into mvcost.
    for o in 0..(CLASS0_SIZE << 3) {
        let mut c = class_cost[0];
        let v = (o + 1) as i32;
        let d = o >> 3;
        let f = (o >> 1) & 3;
        c += class0_cost[d];
        c += class0_fp_cost[d][f];
        if usehp {
            let e = o & 1;
            c += class0_hp_cost[e];
        }
        cost[(zero_idx as i32 + v) as usize] = c + sign_cost[0];
        cost[(zero_idx as i32 - v) as usize] = c + sign_cost[1];
    }

    for (cls, &class_c) in class_cost.iter().enumerate().take(MV_CLASSES).skip(1) {
        for d in 0..(1u32 << cls) {
            let whole_cost = {
                let mut whole = class_c;
                let b = cls + CLASS0_BITS - 1;
                for (i, bc) in bits_cost.iter().take(b).enumerate() {
                    whole += bc[((d >> i) & 1) as usize];
                }
                whole
            };
            for f in 0..4u32 {
                let c = whole_cost + fp_cost[f as usize];
                let v =
                    ((CLASS0_SIZE as i32) << (cls + 2)) + ((d as i32) * 8) + ((f as i32) * 2) + 1;
                if usehp {
                    cost[(zero_idx as i32 + v) as usize] = c + hp_cost[0] + sign_cost[0];
                    cost[(zero_idx as i32 - v) as usize] = c + hp_cost[0] + sign_cost[1];
                    if v + 1 > mv_max {
                        break;
                    }
                    cost[(zero_idx as i32 + v + 1) as usize] = c + hp_cost[1] + sign_cost[0];
                    cost[(zero_idx as i32 - v - 1) as usize] = c + hp_cost[1] + sign_cost[1];
                } else {
                    cost[(zero_idx as i32 + v) as usize] = c + sign_cost[0];
                    cost[(zero_idx as i32 - v) as usize] = c + sign_cost[1];
                    if v + 1 > mv_max {
                        break;
                    }
                    cost[(zero_idx as i32 + v + 1) as usize] = c + sign_cost[0];
                    cost[(zero_idx as i32 - v - 1) as usize] = c + sign_cost[1];
                }
            }
        }
    }

    MvComponentCosts { cost }
}

/// Build the joint + per-axis MV cost bank from a full `NmvContext`.
/// Mirrors libvpx `vp9_build_nmv_cost_table` (`vp9_encodemv.c:232-237`).
#[must_use]
pub fn build_nmv_cost_table(ctx: &NmvContext, usehp: bool) -> MvCosts {
    let mut joint_cost = [0i32; MV_JOINTS];
    vp9_cost_tokens(&mut joint_cost, &VP9_MV_JOINT_TREE, &ctx.joints);
    let v_cost = build_nmv_component_cost_table(&ctx.comps[0], usehp);
    let h_cost = build_nmv_component_cost_table(&ctx.comps[1], usehp);
    MvCosts {
        joint_cost,
        component_cost: [v_cost, h_cost],
    }
}

/// Return the fixed motion-vector costs used by full-pixel SAD search.
///
/// Pinned libvpx builds these separately from the adaptive entropy costs via
/// `cal_nmvjointsadcost` and `cal_nmvsadcosts`
/// (`vp9/encoder/vp9_encoder.c:2230-2256`). The component table is identical
/// for both axes and even around zero. Its high-precision variant has the same
/// values, so one immutable process-wide table serves both cases.
#[must_use]
pub fn nmv_sad_costs() -> &'static MvCosts {
    static COSTS: std::sync::LazyLock<MvCosts> = std::sync::LazyLock::new(build_nmv_sad_cost_table);
    &COSTS
}

fn build_nmv_sad_cost_table() -> MvCosts {
    let mut cost = vec![0i32; MV_VALS].into_boxed_slice();
    let zero_idx = MV_MAX as usize;

    for magnitude in 1..=MV_MAX {
        // Keep libvpx's operation types: log2f returns f32, then the `.6`
        // double literal promotes the remaining arithmetic to f64 before the
        // final truncating cast to int.
        let log2 = ((8 * magnitude) as f32).log2();
        let value = (256.0 * (2.0 * (f64::from(log2) + 0.6))) as i32;
        cost[zero_idx + magnitude as usize] = value;
        cost[zero_idx - magnitude as usize] = value;
    }

    let axis = MvComponentCosts { cost };
    MvCosts {
        joint_cost: [600, 300, 300, 300],
        component_cost: [axis.clone(), axis],
    }
}

/// Cost of encoding a single MV component value `comp` under
/// `mv_comp_costs`. Convenience around the boxed slice.
#[inline]
#[must_use]
pub fn mv_component_cost(comp: i16, mv_comp_costs: &MvComponentCosts) -> i32 {
    mv_comp_costs.lookup(comp)
}

/// Cost of encoding a full MV diff `(row, col)` under `costs`.
/// Mirrors the inner of libvpx `vp9_encode_mv` (`vp9_encodemv.c:210-237`):
/// joint cost + per-non-zero-axis component cost.
#[inline]
#[must_use]
pub fn mv_cost(row: i16, col: i16, costs: &MvCosts) -> i32 {
    let j = mv_joint(row, col);
    let mut c = costs.joint_cost[j];
    // Joint bits 0 / 1 already encode (row != 0) / (col != 0)
    // partitions, so adding the per-axis cost iff non-zero matches the
    // libvpx call path:
    //   if (mv_joint_vertical(j)) encode_mv_component(diff.row, ...);
    //   if (mv_joint_horizontal(j)) encode_mv_component(diff.col, ...);
    // where `vertical` = (j & 2) and `horizontal` = (j & 1) per the
    // joint enum layout in vp9_entropymv.h.
    if row != 0 {
        c += costs.component_cost[0].lookup(row);
    }
    if col != 0 {
        c += costs.component_cost[1].lookup(col);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::mv_probs::DEFAULT_NMV_CONTEXT;

    #[test]
    fn zero_diff_costs_only_joint_zero() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let c = mv_cost(0, 0, &costs);
        // Both axes zero → joint = 0; component costs not added.
        assert_eq!(c, costs.joint_cost[0]);
        assert!(c > 0, "joint(0) should still cost at least one branch");
    }

    #[test]
    fn cost_grows_with_magnitude() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let small = mv_cost(1, 0, &costs);
        let mid = mv_cost(8, 0, &costs);
        let big = mv_cost(64, 0, &costs);
        let huge = mv_cost(1024, 0, &costs);
        assert!(small < mid, "cost(1) = {small} < cost(8) = {mid}");
        assert!(mid < big, "cost(8) = {mid} < cost(64) = {big}");
        assert!(big < huge, "cost(64) = {big} < cost(1024) = {huge}");
    }

    #[test]
    fn component_cost_table_shape() {
        let v_cost = build_nmv_component_cost_table(&DEFAULT_NMV_CONTEXT.comps[0], false);
        assert_eq!(v_cost.cost.len(), MV_VALS);
        // The zero slot is special-cased to zero by libvpx semantics.
        assert_eq!(v_cost.cost[MV_MAX as usize], 0);
        // Symmetric around zero up to sign-cost asymmetry (default
        // sign prob is 128, so positive and negative entries cost the
        // same).
        for v in 1..=64i32 {
            let pos = v_cost.cost[(MV_MAX + v) as usize];
            let neg = v_cost.cost[(MV_MAX - v) as usize];
            assert_eq!(pos, neg, "sign asymmetry at v={v}: pos={pos}, neg={neg}");
        }
    }

    #[test]
    fn highprec_increases_component_cost_when_v_minus_one_lsb_is_set() {
        let no_hp = build_nmv_component_cost_table(&DEFAULT_NMV_CONTEXT.comps[0], false);
        let hp = build_nmv_component_cost_table(&DEFAULT_NMV_CONTEXT.comps[0], true);
        // The high-precision variant adds an extra bit to non-zero
        // entries → strictly greater than the non-hp version.
        let v = 1i32;
        let n = no_hp.cost[(MV_MAX + v) as usize];
        let y = hp.cost[(MV_MAX + v) as usize];
        assert!(y > n, "hp cost ({y}) should exceed non-hp cost ({n})");
    }

    #[test]
    fn joint_cost_table_sane() {
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        for j in 0..MV_JOINTS {
            assert!(costs.joint_cost[j] > 0, "joint {j} cost should be positive");
        }
    }

    #[test]
    fn sad_cost_table_matches_pinned_libvpx() {
        let costs = nmv_sad_costs();
        assert_eq!(costs.joint_cost, [600, 300, 300, 300]);
        assert_eq!(costs.component_cost[0].cost.len(), MV_VALS);
        assert_eq!(costs.component_cost[0], costs.component_cost[1]);
        assert_eq!(costs.component_cost[0].lookup(0), 0);

        // Values produced by pinned libvpx's log2f-based constructor on the
        // acceptance host. These span the class boundaries and table edge.
        for (magnitude, expected) in [
            (1, 1843),
            (2, 2355),
            (3, 2654),
            (4, 2867),
            (7, 3280),
            (8, 3379),
            (16, 3891),
            (255, 5936),
            (1024, 6963),
            (MV_MAX as i16, 9011),
        ] {
            assert_eq!(costs.component_cost[0].lookup(magnitude), expected);
            assert_eq!(costs.component_cost[0].lookup(-magnitude), expected);
        }
    }

    #[test]
    fn sad_cost_table_is_process_wide() {
        assert!(std::ptr::eq(nmv_sad_costs(), nmv_sad_costs()));
    }
}
