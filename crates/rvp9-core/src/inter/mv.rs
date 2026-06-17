//! Per-block motion-vector decode (`read_mv` + `read_mv_component`).
//!
//! Mirrors libvpx `vp9_decodemv.c:235-285`:
//!
//! - `read_mv_component` — decode one signed MV-axis component
//!   (`row` or `col`) from the boolean coder against a single
//!   [`NmvComponent`] probability bank.
//! - `read_mv`            — decode a full motion-vector *difference*
//!   from the joint tree + two components, add it to the reference
//!   MV, and (optionally) bump the per-frame counts for backward
//!   probability adaptation.
//!
//! ## libvpx call site
//!
//! `read_mv` is called from `read_mv` (sub-8 path,
//! `vp9_decodemv.c:414`) and `assign_mv` (parent path,
//! `vp9_decodemv.c:438-491`) — once per `NEWMV` reference. For
//! `NEAREST/NEAR/ZERO` modes, libvpx skips this read entirely; the
//! caller is responsible for that gate.
//!
//! ## High-precision (1/8-pel) gate
//!
//! libvpx's `use_mv_hp(ref)` (`vp9_entropymv.h:30-33`) caps the
//! per-block HP flag whenever the absolute reference MV exceeds
//! 64 1/8-pel units on either axis. This file mirrors that exactly:
//! `read_mv` computes `use_hp = allow_hp && use_mv_hp(ref_mv)` and
//! threads it down to both component reads, regardless of which
//! axis the magnitude check failed on.
//!
//! ## Trees
//!
//! The four MV-related trees from `vp9_entropymv.c:14-28` are
//! reproduced here ([`MV_JOINT_TREE`], [`MV_CLASS_TREE`],
//! [`MV_CLASS0_TREE`], [`MV_FP_TREE`]) so the decode-side reader
//! and the (Phase B.x) encode-side writer share one canonical
//! definition. The leaf encoding mirrors libvpx's `-leaf` /
//! `2 * next_node` convention.

use crate::entropy::{
    BoolReader, BoolWriter, CLASS0_BITS, CLASS0_SIZE, MV_CLASSES, MV_FP_SIZE, MV_JOINTS,
    NmvComponent, NmvComponentCounts, NmvContext, NmvCounts,
};
use crate::mode_info::Mv;

/// libvpx `vp9_mv_joint_tree` (`vp9_entropymv.c:14-16`). 6-entry flat
/// tree decoding one of the 4 `MV_JOINT_*` leaves from 3 boolean
/// probabilities (`NmvContext::joints`).
#[rustfmt::skip]
pub static MV_JOINT_TREE: [i8; 2 * MV_JOINTS - 2] = [
    -(MvJoint::Zero as i8),    2,
    -(MvJoint::HnzVz as i8),   4,
    -(MvJoint::HzVnz as i8),   -(MvJoint::HnzVnz as i8),
];

/// libvpx `vp9_mv_class_tree` (`vp9_entropymv.c:18-23`). 20-entry
/// flat tree decoding one of the 11 `MV_CLASS_*` leaves from 10
/// boolean probabilities (`NmvComponent::classes`).
#[rustfmt::skip]
pub static MV_CLASS_TREE: [i8; 2 * MV_CLASSES - 2] = [
    -(MvClass::Class0 as i8), 2,
    -(MvClass::Class1 as i8), 4,
    6,                         8,
    -(MvClass::Class2 as i8), -(MvClass::Class3 as i8),
    10,                        12,
    -(MvClass::Class4 as i8), -(MvClass::Class5 as i8),
    -(MvClass::Class6 as i8), 14,
    16,                        18,
    -(MvClass::Class7 as i8), -(MvClass::Class8 as i8),
    -(MvClass::Class9 as i8), -(MvClass::Class10 as i8),
];

/// libvpx `vp9_mv_class0_tree` (`vp9_entropymv.c:25`). Single-bit
/// tree over the 2 class-0 integer-pel slots. The decoder doesn't
/// actually walk this with [`BoolReader::read_tree`] — libvpx reads
/// the bit directly via `vpx_read(r, mvcomp->class0[0])` — but the
/// tree is preserved here for the count-update path and for any
/// future writer that re-uses the `read_tree` machinery.
pub static MV_CLASS0_TREE: [i8; 2] = [0, -1];

/// libvpx `vp9_mv_fp_tree` (`vp9_entropymv.c:27-28`). 6-entry flat
/// tree over the 4 fractional-pel positions (1/4-pel quantisation
/// within an integer-pel cell). 2-3 bits each.
#[rustfmt::skip]
pub static MV_FP_TREE: [i8; 2 * MV_FP_SIZE - 2] = [
    0, 2,
    -1, 4,
    -2, -3,
];

/// VP9 MV joint discriminant. Numeric values match libvpx
/// `MV_JOINT_TYPE` (`vp9_entropymv.h:39-44`) so they index
/// [`NmvCounts::joints`] / [`MV_JOINT_TREE`] directly.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum MvJoint {
    /// `MV_JOINT_ZERO = 0` — both axes are zero (no residual MV).
    Zero = 0,
    /// `MV_JOINT_HNZVZ = 1` — vertical zero, horizontal nonzero.
    HnzVz = 1,
    /// `MV_JOINT_HZVNZ = 2` — horizontal zero, vertical nonzero.
    HzVnz = 2,
    /// `MV_JOINT_HNZVNZ = 3` — both axes nonzero.
    HnzVnz = 3,
}

impl MvJoint {
    /// Build an [`MvJoint`] from a 0..=3 raw value. Returns `None`
    /// for out-of-range input.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Zero),
            1 => Some(Self::HnzVz),
            2 => Some(Self::HzVnz),
            3 => Some(Self::HnzVnz),
            _ => None,
        }
    }

    /// libvpx `mv_joint_vertical` (`vp9_entropymv.h:46-48`): `true`
    /// when the vertical component is nonzero in this joint.
    #[inline]
    #[must_use]
    pub const fn vertical(self) -> bool {
        matches!(self, Self::HzVnz | Self::HnzVnz)
    }

    /// libvpx `mv_joint_horizontal` (`vp9_entropymv.h:50-52`): `true`
    /// when the horizontal component is nonzero in this joint.
    #[inline]
    #[must_use]
    pub const fn horizontal(self) -> bool {
        matches!(self, Self::HnzVz | Self::HnzVnz)
    }
}

/// VP9 MV magnitude class. Numeric values match libvpx
/// `MV_CLASS_TYPE` (`vp9_entropymv.h:56-68`) so they index
/// [`NmvComponentCounts::classes`] directly.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum MvClass {
    /// `(0, 2]` integer pels.
    Class0 = 0,
    /// `(2, 4]` integer pels.
    Class1 = 1,
    /// `(4, 8]` integer pels.
    Class2 = 2,
    /// `(8, 16]` integer pels.
    Class3 = 3,
    /// `(16, 32]` integer pels.
    Class4 = 4,
    /// `(32, 64]` integer pels.
    Class5 = 5,
    /// `(64, 128]` integer pels.
    Class6 = 6,
    /// `(128, 256]` integer pels.
    Class7 = 7,
    /// `(256, 512]` integer pels.
    Class8 = 8,
    /// `(512, 1024]` integer pels.
    Class9 = 9,
    /// `(1024, 2048]` integer pels.
    Class10 = 10,
}

impl MvClass {
    /// Build an [`MvClass`] from a 0..=10 raw value. Returns `None`
    /// for out-of-range input.
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Class0),
            1 => Some(Self::Class1),
            2 => Some(Self::Class2),
            3 => Some(Self::Class3),
            4 => Some(Self::Class4),
            5 => Some(Self::Class5),
            6 => Some(Self::Class6),
            7 => Some(Self::Class7),
            8 => Some(Self::Class8),
            9 => Some(Self::Class9),
            10 => Some(Self::Class10),
            _ => None,
        }
    }
}

/// libvpx `use_mv_hp` (`vp9_entropymv.h:30-33`). Returns `true` when
/// the absolute reference MV is small enough on **both** axes that
/// high-precision (1/8-pel) coding may be enabled for the residual.
///
/// The 64 1/8-pel threshold is 8 integer pels — beyond that libvpx
/// degrades the residual to 1/4-pel even if the frame header says
/// `allow_high_precision_mv == 1`.
#[inline]
#[must_use]
pub fn use_mv_hp(reference: Mv) -> bool {
    const THRESH: i32 = 64;
    (reference.row as i32).abs() < THRESH && (reference.col as i32).abs() < THRESH
}

/// Compute the [`MvJoint`] for a given MV difference. Mirrors libvpx
/// `vp9_get_mv_joint` (`vp9_entropymv.h:104-110`).
#[inline]
#[must_use]
pub fn get_mv_joint(diff: Mv) -> MvJoint {
    if diff.row == 0 {
        if diff.col == 0 {
            MvJoint::Zero
        } else {
            MvJoint::HnzVz
        }
    } else if diff.col == 0 {
        MvJoint::HzVnz
    } else {
        MvJoint::HnzVnz
    }
}

/// Decode one motion-vector axis component (`row` or `col`) from the
/// boolean coder. Mirrors libvpx `read_mv_component`
/// (`vp9_decodemv.c:235-265`).
///
/// The returned `i16` is the signed MV residual in 1/8-pel units
/// (precise composition `sign ? -mag : mag`, with
/// `mag = base + (d << 3) + (fr << 1) + hp + 1`):
///
/// 1. Read sign bit (`mvcomp.sign`).
/// 2. Read 11-way class via [`MV_CLASS_TREE`] against `mvcomp.classes`.
/// 3. If class 0: read one integer-pel bit (`mvcomp.class0[0]`).
///    Otherwise: read `class + CLASS0_BITS - 1` offset bits LSB-first
///    against `mvcomp.bits[..]`.
/// 4. Read fractional-pel tree (`class0_fp[d]` for class 0, else `fp`).
/// 5. If `use_hp`: read HP bit (`class0_hp` / `hp`); else hp = 1.
///
/// When `counts` is `Some`, increments the matching count slots
/// (sign, class, integer-bits, fractional, HP) so backward adaptation
/// can update the next frame's probabilities. Counts mirror libvpx's
/// `inc_mv_component` (`vp9_entropymv.c:111-137`) — i.e. the HP slot
/// is touched even when `use_hp == false`, with `incr = usehp` so
/// "no HP" turns into a zero-increment no-op. Our `Option<&mut ..>`
/// avoids the explicit `usehp` multiplier and just gates the HP
/// increment on `use_hp` directly.
#[inline]
pub fn read_mv_component(
    reader: &mut BoolReader<'_>,
    comp_probs: &NmvComponent,
    counts: Option<&mut NmvComponentCounts>,
    use_hp: bool,
) -> i16 {
    let sign = reader.read(comp_probs.sign);
    let mv_class_raw = reader.read_tree(&MV_CLASS_TREE, &comp_probs.classes);
    let mv_class = MvClass::from_u8(mv_class_raw as u8)
        .expect("MV_CLASS_TREE always returns a valid MV_CLASS_*");
    let is_class0 = matches!(mv_class, MvClass::Class0);

    // Integer-pel data + magnitude base.
    let (d, mag_base) = if is_class0 {
        let d = reader.read(comp_probs.class0[0]) as u32;
        (d, 0u32)
    } else {
        // `n = class + CLASS0_BITS - 1` bits decoded LSB-first.
        let n = mv_class as usize + CLASS0_BITS - 1;
        let mut d: u32 = 0;
        for i in 0..n {
            d |= u32::from(reader.read(comp_probs.bits[i])) << i;
        }
        // libvpx: `mag = CLASS0_SIZE << (class + 2)`.
        let base = (CLASS0_SIZE as u32) << (mv_class as u32 + 2);
        (d, base)
    };

    // Fractional-pel tree (2-3 bits, values 0..=3).
    let fp_probs: &[u8] = if is_class0 {
        &comp_probs.class0_fp[d as usize]
    } else {
        &comp_probs.fp
    };
    let fr = reader.read_tree(&MV_FP_TREE, fp_probs) as u32;

    // High-precision bit (1/8-pel). libvpx hard-codes hp = 1 when
    // `use_hp == false` so the final magnitude composition is
    // deterministic at 1/4-pel.
    let hp = if use_hp {
        let p = if is_class0 {
            comp_probs.class0_hp
        } else {
            comp_probs.hp
        };
        reader.read(p) as u32
    } else {
        1
    };

    // Bump counts (backward adaptation). Mirrors libvpx
    // `inc_mv_component` (`vp9_entropymv.c:111-136`).
    //
    // D.22 — libvpx ALWAYS bumps `(class0_)hp[e]` (the per-block call
    // site `vp9_inc_mv` hard-codes `usehp=1` regardless of the
    // frame-level `allow_high_precision_mv` flag; `vp9_entropymv.c:145`
    // / `:149`). When the block-level read was done with `use_hp=false`
    // (hp was hard-coded to 1, no bit consumed), libvpx still bumps
    // `hp[1]` because `e = o & 1` and `o = (d<<3) | (fr<<1) | 1` ends
    // with LSB=1. The previous gating on `use_hp` here under-counted
    // those slots, drifting `comp.hp` (and `class0_hp`) away from
    // libvpx after every frame that adapts mv probs with `allow_hp`
    // ultimately enabled — surfaces as a MV-decode divergence on the
    // first NEWMV in an altref frame (`vp90-2-09-aq2.webm` shown frame
    // 15, libvpx `f=16` altref).
    if let Some(c) = counts {
        c.sign[sign as usize] = c.sign[sign as usize].saturating_add(1);
        let cls = mv_class as usize;
        c.classes[cls] = c.classes[cls].saturating_add(1);
        if is_class0 {
            let di = d as usize;
            c.class0[di] = c.class0[di].saturating_add(1);
            c.class0_fp[di][fr as usize] = c.class0_fp[di][fr as usize].saturating_add(1);
            c.class0_hp[hp as usize] = c.class0_hp[hp as usize].saturating_add(1);
        } else {
            let n = mv_class as usize + CLASS0_BITS - 1;
            for i in 0..n {
                let bit = ((d >> i) & 1) as usize;
                c.bits[i][bit] = c.bits[i][bit].saturating_add(1);
            }
            c.fp[fr as usize] = c.fp[fr as usize].saturating_add(1);
            c.hp[hp as usize] = c.hp[hp as usize].saturating_add(1);
        }
    }

    // Compose `mag = base + ((d << 3) | (fr << 1) | hp) + 1`.
    // The `| 1` over hp is safe because `hp` is 0 or 1 and the
    // low bits of `d << 3 | fr << 1` are zero.
    let composed = mag_base + ((d << 3) | (fr << 1) | hp) + 1;
    // composed fits in 14 bits per libvpx `MV_IN_USE_BITS = 14`.
    // The sign branch produces a value in `[MV_LOW, MV_UPP] =
    // [-16384, 16383]`, well within i16.
    let signed = composed as i16;
    if sign != 0 { -signed } else { signed }
}

/// Decode a complete motion-vector residual from the boolean coder
/// and add it to the reference MV. Mirrors libvpx `read_mv`
/// (`vp9_decodemv.c:267-285`).
///
/// 1. Read the [`MvJoint`] via [`MV_JOINT_TREE`].
/// 2. If the joint says vertical is nonzero, decode the row component.
/// 3. If the joint says horizontal is nonzero, decode the col component.
/// 4. Increment the joint count slot (and per-component counts via
///    [`read_mv_component`]) when `counts.is_some()`.
/// 5. Return `ref_mv + diff` (wrapping i16 add — libvpx uses raw
///    `int16_t` so we preserve that mod-`2^16` behaviour).
///
/// `allow_hp` is the frame-level
/// `uh->allow_high_precision_mv`. The per-block HP gate
/// ([`use_mv_hp`]) is applied internally.
///
/// **Per-block call contract:** invoke once per `NEWMV` MV slot
/// (libvpx `vp9_decodemv.c:414` + `:438-491`). For
/// `NEAREST/NEAR/ZERO` modes the caller skips this entirely and
/// fills the MV from the reference list (B.2 `find_mv_refs` /
/// `mv_pred` is the layer that builds that list).
#[inline]
pub fn read_mv(
    reader: &mut BoolReader<'_>,
    ref_mv: Mv,
    allow_hp: bool,
    nmv_context: &NmvContext,
    mut counts: Option<&mut NmvCounts>,
) -> Mv {
    let joint_raw = reader.read_tree(&MV_JOINT_TREE, &nmv_context.joints);
    let joint =
        MvJoint::from_u8(joint_raw as u8).expect("MV_JOINT_TREE always returns a valid MV_JOINT_*");
    let use_hp = allow_hp && use_mv_hp(ref_mv);

    let mut diff = Mv::ZERO;
    if joint.vertical() {
        let row_counts = counts.as_mut().map(|c| &mut c.comps[0]);
        diff.row = read_mv_component(reader, &nmv_context.comps[0], row_counts, use_hp);
    }
    if joint.horizontal() {
        let col_counts = counts.as_mut().map(|c| &mut c.comps[1]);
        diff.col = read_mv_component(reader, &nmv_context.comps[1], col_counts, use_hp);
    }

    if let Some(c) = counts {
        let j = joint as usize;
        c.joints[j] = c.joints[j].saturating_add(1);
    }

    // libvpx adds with the C `int16_t` (`vp9_mv.h:24-27`) wrap; we
    // use `wrapping_add` to match.
    Mv {
        row: ref_mv.row.wrapping_add(diff.row),
        col: ref_mv.col.wrapping_add(diff.col),
    }
}

// ============================================================================
// Encoder-side primitives — write_mv_component / write_mv.
//
// Mirror images of `read_mv_component` / `read_mv` above. Used by the
// inter-block ModeInfo writer (S4.16b) and ultimately by the encoder
// to emit NEWMV MV-residual payloads. libvpx counterpart:
// `vp9_encodemv.c::encode_mv_component` (`vp9/encoder/vp9_encodemv.c:159-205`)
// and `vp9_encode_mv` (`vp9_encodemv.c:207-242`).
//
// Invariants the encoder must preserve so the decoder round-trips:
//   - `diff` (MV-axis residual) is in libvpx's `MV_LOW..=MV_UPP` range,
//     i.e. fits in 14 bits (`MV_IN_USE_BITS = 14`).
//   - When `use_hp == false`, `abs(diff)` must be EVEN — equivalently
//     `mag & 1 == 1` (because `composed = mag + 1`). libvpx's decoder
//     hard-codes the HP bit to 1 in that case; the encoder accordingly
//     emits no HP bit, and the only diff values that compose back
//     correctly are those whose LSB-of-offset is already 1.
// ============================================================================

/// Decompose an MV-axis residual magnitude `mag` (= `abs(diff) - 1`)
/// into its `(MvClass, mag_base)` pair, matching libvpx's
/// `vp9_get_mv_class` (`vp9/common/vp9_entropymv.c:43-66`).
///
/// `mag_base` is the floor of the magnitude window for that class,
/// derived via the same `(CLASS0_SIZE) << (class + 2)` formula the
/// reader uses (see [`read_mv_component`] line 272). For class 0 the
/// base is 0.
#[inline]
fn mv_class_and_base(mag: u32) -> (MvClass, u32) {
    // libvpx's `log_in_base_2[z >> 3]` table maps `z >> 3` to
    // `floor(log2(z >> 3))` for `z >= 8`, with a special-case
    // "class 10" branch for `z >= CLASS0_SIZE * 4096 == 8192`.
    // The test helper already does this; we replicate it here as a
    // small inline pure helper that matches the reader's `base`
    // formula bit-for-bit.
    if mag >= (CLASS0_SIZE as u32) * 4096 {
        let class = MvClass::Class10;
        let base = (CLASS0_SIZE as u32) << (class as u32 + 2);
        return (class, base);
    }
    let idx = mag >> 3;
    if idx == 0 {
        return (MvClass::Class0, 0);
    }
    let bits = 32 - idx.leading_zeros();
    let class = MvClass::from_u8((bits - 1) as u8).expect("mv_class_and_base: class in 0..=10");
    // Class0 is special-cased to base=0 (matches the reader: line 261-263
    // returns base=0 for is_class0 unconditionally). Without this guard,
    // a mag in 8..=15 (e.g. value=9 → mag=8 → idx=1 → bits=1 → class=0)
    // would compute `base = CLASS0_SIZE << 2 = 8`, which is wrong: class
    // 0 covers magnitudes 0..=15 with base=0.
    if matches!(class, MvClass::Class0) {
        return (class, 0);
    }
    let base = (CLASS0_SIZE as u32) << (class as u32 + 2);
    (class, base)
}

/// Walk a libvpx-format probability tree to find the bit sequence that
/// reaches the leaf with value `target`, and emit those bits against
/// `probs` via `bw.write`. Mirrors the inverse of
/// [`BoolReader::read_tree`] (`bool_reader.rs:158-167`).
///
/// `tree` and `probs` follow the same layout the reader walks: positive
/// entries are next-node indices; negative entries `-leaf` are terminal
/// leaves whose returned value is `leaf` (negation recovers the positive
/// enum value); `probs[i >> 1]` gates the branch decision at node `i`.
///
/// Panics in debug builds if `target` is unreachable from the root.
#[inline]
pub(crate) fn write_tree(bw: &mut BoolWriter<'_>, tree: &[i8], probs: &[u8], target: i8) {
    /// Recursively check whether `target` is reachable in the subtree
    /// rooted at `node`. Leaves are negative (`-target`); internal
    /// nodes are positive indices into `tree`.
    fn contains(tree: &[i8], node: i8, target: i8) -> bool {
        if node <= 0 {
            -node == target
        } else {
            let i = node as usize;
            contains(tree, tree[i], target) || contains(tree, tree[i + 1], target)
        }
    }
    let mut i: i32 = 0;
    loop {
        let left = tree[i as usize];
        let right = tree[i as usize + 1];
        let prob = probs[(i as usize) >> 1];
        let bit: u8 = if contains(tree, left, target) { 0 } else { 1 };
        bw.write(bit, prob);
        let next = if bit == 0 { left } else { right };
        if next <= 0 {
            debug_assert_eq!(-next, target, "write_tree: descended to wrong leaf");
            return;
        }
        i = next as i32;
    }
}

/// Encode one motion-vector axis component (`row` or `col`) into the
/// boolean writer. Mirrors libvpx `encode_mv_component`
/// (`vp9/encoder/vp9_encodemv.c:159-205`) — the inverse of
/// [`read_mv_component`].
///
/// `diff` is the signed MV residual in 1/8-pel units; must be nonzero
/// (the zero case is handled by the MV joint short-circuit in
/// [`write_mv`]) and fit in libvpx's 14-bit `MV_IN_USE_BITS` range.
///
/// 1. Emit sign bit (`comp_probs.sign`).
/// 2. Emit class via [`MV_CLASS_TREE`] against `comp_probs.classes`.
/// 3. If class 0: emit one integer-pel bit (`comp_probs.class0[0]`).
///    Otherwise: emit `class + CLASS0_BITS - 1` offset bits LSB-first
///    against `comp_probs.bits[..]`.
/// 4. Emit fractional-pel tree (`class0_fp[d]` for class 0, else `fp`).
/// 5. If `use_hp`: emit the HP bit (`class0_hp` / `hp`). Otherwise no
///    HP bit is emitted — the caller guarantees `mag & 1 == 1` (i.e.
///    `abs(diff)` is even) so the decoder's hard-coded `hp = 1` will
///    recompose the same value.
///
/// # Panics (debug only)
///
/// Panics if `diff == 0`, if `abs(diff)` exceeds libvpx's
/// `MV_IN_USE_BITS = 14` range, or if `!use_hp` but `mag & 1 != 1`.
#[inline]
pub fn write_mv_component(
    bw: &mut BoolWriter<'_>,
    diff: i16,
    comp_probs: &NmvComponent,
    use_hp: bool,
    comp_counts: Option<&mut NmvComponentCounts>,
) {
    debug_assert!(
        diff != 0,
        "write_mv_component: zero handled by joint short-circuit"
    );

    let sign: u8 = if diff < 0 { 1 } else { 0 };
    // `abs_diff` is `abs(diff)` widened to u32. `i16::unsigned_abs`
    // returns u16; widen so the subtraction below stays in u32.
    let abs_diff = u32::from(diff.unsigned_abs());
    // libvpx's `z = (sign ? -v : v) - 1`. `z` is the "mag" in
    // `read_mv_component` parlance.
    let mag = abs_diff - 1;
    debug_assert!(
        mag < (1u32 << 14),
        "write_mv_component: diff out of MV_IN_USE_BITS range (got abs={abs_diff})"
    );

    // Class + magnitude base.
    let (class, mag_base) = mv_class_and_base(mag);
    let is_class0 = matches!(class, MvClass::Class0);

    // libvpx: `o = mag - mag_base`, `d = o >> 3`, `f = (o >> 1) & 3`,
    // `e = o & 1`. For class 0, `mag_base = 0` so `o = mag`.
    let offset = mag - mag_base;
    let d = offset >> 3;
    let fr = (offset >> 1) & 3;
    let hp = offset & 1;

    // Sign bit.
    bw.write(sign, comp_probs.sign);

    // Class tree.
    write_tree(bw, &MV_CLASS_TREE, &comp_probs.classes, class as i8);

    // Integer-pel section.
    if is_class0 {
        // Single bit gates the 2-slot `class0` (CLASS0_SIZE = 2), matching
        // the reader's `r.read(comp_probs.class0[0])` (line 262).
        bw.write(d as u8, comp_probs.class0[0]);
    } else {
        let n = class as usize + CLASS0_BITS - 1;
        for i in 0..n {
            let bit = ((d >> i) & 1) as u8;
            bw.write(bit, comp_probs.bits[i]);
        }
    }

    // Fractional-pel tree.
    let fp_probs: &[u8] = if is_class0 {
        &comp_probs.class0_fp[d as usize]
    } else {
        &comp_probs.fp
    };
    write_tree(bw, &MV_FP_TREE, fp_probs, fr as i8);

    // High-precision bit (1/8-pel). Only emitted when `use_hp`; libvpx's
    // decoder hard-codes `hp = 1` otherwise, so the encoder must only
    // call us with `mag & 1 == 1` when `use_hp == false`.
    if use_hp {
        let p = if is_class0 {
            comp_probs.class0_hp
        } else {
            comp_probs.hp
        };
        bw.write(hp as u8, p);
    } else {
        debug_assert_eq!(
            hp, 1,
            "write_mv_component: use_hp=false requires LSB-of-offset == 1 (abs(diff) must be even)"
        );
    }

    // P0b — accumulate MV component counts for backward adaptation.
    if let Some(cc) = comp_counts {
        cc.sign[sign as usize] += 1;
        cc.classes[class as usize] += 1;
        if is_class0 {
            cc.class0[d as usize] += 1;
            cc.class0_fp[d as usize][fr as usize] += 1;
            // libvpx's `vp9_inc_mv` bumps HP counts unconditionally;
            // when no HP bit was coded, `hp == 1` by construction.
            cc.class0_hp[hp as usize] += 1;
        } else {
            let n = class as usize + CLASS0_BITS - 1;
            for i in 0..n {
                let bit = ((d >> i) & 1) as usize;
                cc.bits[i][bit] += 1;
            }
            cc.fp[fr as usize] += 1;
            // Keep writer-side counts identical to the decoder/read path.
            cc.hp[hp as usize] += 1;
        }
    }
}

/// Encode a complete motion-vector residual `(cur_mv - ref_mv)` into the
/// boolean writer. Mirrors libvpx `vp9_encode_mv`
/// (`vp9/encoder/vp9_encodemv.c:207-242`) — the inverse of
/// [`read_mv`].
///
/// 1. Compute `diff = cur_mv - ref_mv` (wrapping i16 sub).
/// 2. Emit the [`MvJoint`] via [`MV_JOINT_TREE`].
/// 3. If the joint says vertical is nonzero, emit the row component.
/// 4. If the joint says horizontal is nonzero, emit the col component.
///
/// `allow_hp` is the frame-level `uh->allow_high_precision_mv`. The
/// per-block HP gate ([`use_mv_hp`]) is applied internally against
/// `ref_mv` and threaded down to both component writes — matching the
/// reader's behaviour exactly.
///
/// **Per-block call contract:** invoke once per `NEWMV` MV slot. For
/// `NEAREST/NEAR/ZERO` modes the encoder skips this entirely — the MV
/// is implied by the reference list. Matches the read-side gate at
/// `vp9_decodemv.c:414` and `:438-491`.
///
/// # Panics (debug only)
///
/// Panics if any non-zero diff component falls outside libvpx's
/// `MV_IN_USE_BITS = 14` range, or if `!use_hp` (after the per-block
/// gate) is incompatible with the diff value (see
/// [`write_mv_component`]).
#[inline]
pub fn write_mv(
    bw: &mut BoolWriter<'_>,
    cur_mv: Mv,
    ref_mv: Mv,
    allow_hp: bool,
    nmv_context: &NmvContext,
    mut mv_counts: Option<&mut NmvCounts>,
) {
    let diff = Mv {
        row: cur_mv.row.wrapping_sub(ref_mv.row),
        col: cur_mv.col.wrapping_sub(ref_mv.col),
    };
    let joint = get_mv_joint(diff);

    // Joint tree.
    write_tree(bw, &MV_JOINT_TREE, &nmv_context.joints, joint as i8);
    // P0b — accumulate joint count.
    if let Some(c) = mv_counts.as_deref_mut() {
        c.joints[joint as usize] += 1;
    }

    let use_hp = allow_hp && use_mv_hp(ref_mv);

    if joint.vertical() {
        write_mv_component(
            bw,
            diff.row,
            &nmv_context.comps[0],
            use_hp,
            mv_counts.as_deref_mut().map(|c| &mut c.comps[0]),
        );
    }
    if joint.horizontal() {
        write_mv_component(
            bw,
            diff.col,
            &nmv_context.comps[1],
            use_hp,
            mv_counts.map(|c| &mut c.comps[1]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::DEFAULT_NMV_CONTEXT;

    // ---------- Helpers --------------------------------------------------

    /// Walk a libvpx-shaped boolean tree to find the bit sequence that
    /// reaches the target leaf. Returns `(bit, prob_index)` pairs in
    /// the order the writer must encode them.
    fn tree_path(tree: &[i8], probs: &[u8], target: u8) -> Vec<(u8, u8)> {
        fn contains(tree: &[i8], node: i8, target: u8) -> bool {
            if node <= 0 {
                (-node) as u8 == target
            } else {
                let i = node as usize;
                contains(tree, tree[i], target) || contains(tree, tree[i + 1], target)
            }
        }
        let mut out = Vec::new();
        let mut i: i32 = 0;
        loop {
            let left = tree[i as usize];
            let right = tree[i as usize + 1];
            let prob = probs[(i as usize) >> 1];
            let bit = if contains(tree, left, target) { 0 } else { 1 };
            out.push((bit, prob));
            let next = if bit == 0 { left } else { right };
            if next <= 0 {
                break;
            }
            i = next as i32;
        }
        out
    }

    /// Encode the bit/prob sequence for one component, then pad with
    /// max-entropy zero bits so the reader's `fill` always sees data.
    fn finish_buf(w: BoolWriter<'_>) -> usize {
        let mut w = w;
        for _ in 0..16 {
            w.write_bit(0);
        }
        w.stop_encode().expect("stop_encode")
    }

    /// Encode a single component value (in 1/8-pel signed Q3 units)
    /// against `comp` and `use_hp`. Mirrors the inverse of
    /// `read_mv_component`. Returns the encoded buffer.
    fn encode_component(value: i16, comp: &NmvComponent, use_hp: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);

            let sign = (value < 0) as u8;
            w.write(sign, comp.sign);

            // libvpx `inc_mv_component`: `z = (sign ? -v : v) - 1`.
            let mag = if value < 0 {
                (-(value as i32)) as u32
            } else {
                value as u32
            };
            assert!(mag >= 1, "encode_component requires nonzero magnitude");
            let z = mag - 1;

            // Compute the class via libvpx `vp9_get_mv_class` —
            // matches `log2(z >> 3)` for non-class-10.
            let class = if z >= (CLASS0_SIZE as u32 * 4096) {
                MvClass::Class10
            } else {
                // Re-derive `c = log_in_base_2[z >> 3]` cheaply for
                // the test by counting leading zeros — this matches
                // libvpx's table exactly for `z` in range.
                let idx = z >> 3;
                if idx == 0 {
                    MvClass::Class0
                } else {
                    let bits = 32 - idx.leading_zeros();
                    // `log_in_base_2[k]` returns `floor(log2(k))`
                    // for k >= 1.
                    MvClass::from_u8((bits - 1) as u8).expect("valid mv class")
                }
            };
            let mag_base: u32 = if class as u8 == 0 {
                0
            } else {
                (CLASS0_SIZE as u32) << (class as u32 + 2)
            };
            let offset = z - mag_base; // libvpx `o`

            // Encode the class via the class tree.
            for (bit, prob) in tree_path(&MV_CLASS_TREE, &comp.classes, class as u8) {
                w.write(bit, prob);
            }

            // libvpx: `d = o >> 3`, `f = (o >> 1) & 3`, `e = o & 1`.
            let d = offset >> 3;
            let f = (offset >> 1) & 3;
            let e = offset & 1;

            // Integer-pel section.
            if matches!(class, MvClass::Class0) {
                w.write(d as u8, comp.class0[0]);
            } else {
                let n = class as usize + CLASS0_BITS - 1;
                for i in 0..n {
                    let bit = ((d >> i) & 1) as u8;
                    w.write(bit, comp.bits[i]);
                }
            }

            // Fractional tree.
            let fp_probs: &[u8] = if matches!(class, MvClass::Class0) {
                &comp.class0_fp[d as usize]
            } else {
                &comp.fp
            };
            for (bit, prob) in tree_path(&MV_FP_TREE, fp_probs, f as u8) {
                w.write(bit, prob);
            }

            // HP bit — only emitted if use_hp is set; otherwise
            // libvpx's decoder hard-codes hp = 1, so callers must
            // pass values whose low bit is 1 in that case.
            if use_hp {
                let p = if matches!(class, MvClass::Class0) {
                    comp.class0_hp
                } else {
                    comp.hp
                };
                w.write(e as u8, p);
            } else {
                assert_eq!(e, 1, "use_hp=false requires LSB of offset to be 1");
            }

            finish_buf(w)
        };
        buf.truncate(size);
        buf
    }

    /// Encode the joint-tree path for `j` into the writer.
    fn encode_joint(w: &mut BoolWriter<'_>, joints: &[u8; 3], j: MvJoint) {
        for (bit, prob) in tree_path(&MV_JOINT_TREE, joints, j as u8) {
            w.write(bit, prob);
        }
    }

    /// Encode a complete MV residual against `ctx` + `use_hp`.
    fn encode_mv(diff: Mv, ctx: &NmvContext, use_hp: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            let j = get_mv_joint(diff);
            encode_joint(&mut w, &ctx.joints, j);

            if j.vertical() {
                let inner = encode_component(diff.row, &ctx.comps[0], use_hp);
                replay_into(&mut w, &inner, &ctx.comps[0], use_hp, diff.row);
            }
            if j.horizontal() {
                let inner = encode_component(diff.col, &ctx.comps[1], use_hp);
                replay_into(&mut w, &inner, &ctx.comps[1], use_hp, diff.col);
            }
            finish_buf(w)
        };
        buf.truncate(size);
        buf
    }

    /// We can't actually concatenate two separate `BoolWriter` outputs
    /// (the bool coder is stateful). Instead, replay the bit sequence
    /// that `encode_component` produced *against the same writer*. We
    /// do that by re-running the component encode inline.
    #[allow(unused_variables)]
    fn replay_into(
        w: &mut BoolWriter<'_>,
        _inner: &[u8],
        comp: &NmvComponent,
        use_hp: bool,
        value: i16,
    ) {
        // Same as `encode_component`, sans the final flush.
        let sign = (value < 0) as u8;
        w.write(sign, comp.sign);

        let mag = if value < 0 {
            (-(value as i32)) as u32
        } else {
            value as u32
        };
        assert!(mag >= 1);
        let z = mag - 1;
        let class = if z >= (CLASS0_SIZE as u32 * 4096) {
            MvClass::Class10
        } else {
            let idx = z >> 3;
            if idx == 0 {
                MvClass::Class0
            } else {
                let bits = 32 - idx.leading_zeros();
                MvClass::from_u8((bits - 1) as u8).unwrap()
            }
        };
        let mag_base: u32 = if class as u8 == 0 {
            0
        } else {
            (CLASS0_SIZE as u32) << (class as u32 + 2)
        };
        let offset = z - mag_base;

        for (bit, prob) in tree_path(&MV_CLASS_TREE, &comp.classes, class as u8) {
            w.write(bit, prob);
        }

        let d = offset >> 3;
        let f = (offset >> 1) & 3;
        let e = offset & 1;

        if matches!(class, MvClass::Class0) {
            w.write(d as u8, comp.class0[0]);
        } else {
            let n = class as usize + CLASS0_BITS - 1;
            for i in 0..n {
                let bit = ((d >> i) & 1) as u8;
                w.write(bit, comp.bits[i]);
            }
        }

        let fp_probs: &[u8] = if matches!(class, MvClass::Class0) {
            &comp.class0_fp[d as usize]
        } else {
            &comp.fp
        };
        for (bit, prob) in tree_path(&MV_FP_TREE, fp_probs, f as u8) {
            w.write(bit, prob);
        }

        if use_hp {
            let p = if matches!(class, MvClass::Class0) {
                comp.class0_hp
            } else {
                comp.hp
            };
            w.write(e as u8, p);
        }
    }

    // ---------- Tree-shape tests ----------------------------------------

    #[test]
    fn mv_joint_tree_shape() {
        assert_eq!(MV_JOINT_TREE.len(), 6);
        // node 0: [-ZERO=0, 2]  → with -0 == 0, leaf encoded as 0 means
        // "MV_JOINT_ZERO". The reader's `read_tree` returns `-i` on
        // leaves, so it correctly returns 0 here.
        assert_eq!(MV_JOINT_TREE[0], 0);
        assert_eq!(MV_JOINT_TREE[1], 2);
        assert_eq!(MV_JOINT_TREE[2], -1);
        assert_eq!(MV_JOINT_TREE[3], 4);
        assert_eq!(MV_JOINT_TREE[4], -2);
        assert_eq!(MV_JOINT_TREE[5], -3);
    }

    #[test]
    fn mv_class_tree_shape() {
        assert_eq!(MV_CLASS_TREE.len(), 20);
        assert_eq!(MV_CLASS_TREE[0], 0); // -CLASS_0
        assert_eq!(MV_CLASS_TREE[1], 2);
        assert_eq!(MV_CLASS_TREE[6], -2); // -CLASS_2
        assert_eq!(MV_CLASS_TREE[7], -3); // -CLASS_3
        assert_eq!(MV_CLASS_TREE[18], -9); // -CLASS_9
        assert_eq!(MV_CLASS_TREE[19], -10); // -CLASS_10
    }

    #[test]
    fn mv_fp_tree_shape() {
        assert_eq!(MV_FP_TREE.len(), 6);
        assert_eq!(MV_FP_TREE[0], 0);
        assert_eq!(MV_FP_TREE[1], 2);
        assert_eq!(MV_FP_TREE[2], -1);
        assert_eq!(MV_FP_TREE[3], 4);
        assert_eq!(MV_FP_TREE[4], -2);
        assert_eq!(MV_FP_TREE[5], -3);
    }

    // ---------- Pure logic tests -----------------------------------------

    #[test]
    fn use_mv_hp_threshold() {
        assert!(use_mv_hp(Mv { row: 0, col: 0 }));
        assert!(use_mv_hp(Mv { row: 63, col: -63 }));
        assert!(!use_mv_hp(Mv { row: 64, col: 0 }));
        assert!(!use_mv_hp(Mv { row: 0, col: -64 }));
        assert!(!use_mv_hp(Mv { row: 100, col: 100 }));
    }

    #[test]
    fn get_mv_joint_cases() {
        assert_eq!(get_mv_joint(Mv { row: 0, col: 0 }), MvJoint::Zero);
        assert_eq!(get_mv_joint(Mv { row: 0, col: 5 }), MvJoint::HnzVz);
        assert_eq!(get_mv_joint(Mv { row: 7, col: 0 }), MvJoint::HzVnz);
        assert_eq!(get_mv_joint(Mv { row: -3, col: 9 }), MvJoint::HnzVnz);
    }

    // ---------- Round-trip tests ----------------------------------------

    /// Each MV joint: encode the joint path (plus dummy components if
    /// needed) and verify the decoder returns the right joint via the
    /// final composed MV.
    #[test]
    fn read_mv_each_joint_round_trip() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let cases = [
            (Mv::ZERO, MvJoint::Zero),
            (Mv { row: 0, col: 9 }, MvJoint::HnzVz),
            (Mv { row: -11, col: 0 }, MvJoint::HzVnz),
            (Mv { row: 5, col: -3 }, MvJoint::HnzVnz),
        ];
        for (diff, expected_joint) in cases {
            // The HnzVz/HzVnz/HnzVnz cases require nonzero magnitudes
            // — all our test values satisfy that.
            assert_eq!(get_mv_joint(diff), expected_joint);
            let buf = encode_mv(diff, ctx, /* use_hp = */ true);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv(&mut r, Mv::ZERO, /* allow_hp = */ true, ctx, None);
            assert_eq!(got, diff, "joint case {expected_joint:?} → diff={diff:?}");
            assert!(!r.has_error(), "decoder over-ran buffer");
        }
    }

    /// Each MV_CLASS_0..MV_CLASS_10: pick a representative MV in that
    /// class, encode, decode, verify.
    #[test]
    fn read_mv_component_each_class_round_trip() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[0];
        // libvpx's `log_in_base_2` table makes the class boundaries
        // `z ∈ [0, 15]` → class 0, `z ∈ [16, 31]` → class 1, …,
        // `z ∈ [2^(c+3), 2^(c+4) - 1]` → class c (c ≥ 1), `z ≥ 8192`
        // → class 10. Pick a representative magnitude in the middle
        // of each window (value = mag = z + 1).
        let cases: &[(MvClass, i16)] = &[
            (MvClass::Class0, 4),      // z=3
            (MvClass::Class1, 24),     // z=23,   z>>3=2 → log=1
            (MvClass::Class2, 48),     // z=47,   z>>3=5 → log=2
            (MvClass::Class3, 96),     // z=95,   z>>3=11 → log=3
            (MvClass::Class4, 192),    // z=191,  z>>3=23 → log=4
            (MvClass::Class5, 384),    // z=383,  z>>3=47 → log=5
            (MvClass::Class6, 768),    // z=767,  z>>3=95 → log=6
            (MvClass::Class7, 1536),   // z=1535, z>>3=191 → log=7
            (MvClass::Class8, 3072),   // z=3071, z>>3=383 → log=8
            (MvClass::Class9, 6144),   // z=6143, z>>3=767 → log=9
            (MvClass::Class10, 12288), // z=12287 ≥ 8192 → class 10
        ];
        for &(expected_class, value) in cases {
            let buf = encode_component(value, comp, /* use_hp = */ true);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv_component(&mut r, comp, None, /* use_hp = */ true);
            assert_eq!(got, value, "class {expected_class:?} value={value}");
        }
    }

    /// Negative values round-trip identically to their positive
    /// counterparts plus the sign bit.
    #[test]
    fn read_mv_component_sign_bit() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[1];
        for value in [-1i16, -3, -15, -127, -1023] {
            let buf = encode_component(value, comp, /* use_hp = */ true);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv_component(&mut r, comp, None, true);
            assert_eq!(got, value);
        }
    }

    /// `allow_hp = false` caps HP off: the decoder must hard-code
    /// `hp = 1` even when the residual decode path crosses what
    /// would otherwise be an HP bit. Encoder for this case omits
    /// the HP bit entirely.
    #[test]
    fn read_mv_component_allow_hp_false_caps() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[0];
        // Pick a value whose composition has `e = 1` so the
        // encoder can omit the HP bit without changing the value.
        // value = 4 → z = 3 → class 0, d = 0, f = 1, e = 1 → mag =
        // ((0<<3) | (1<<1) | 1) + 1 = 4. ✓
        let value: i16 = 4;
        let buf = encode_component(value, comp, /* use_hp = */ false);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv_component(&mut r, comp, None, /* use_hp = */ false);
        assert_eq!(got, value);
    }

    /// `read_mv` round-trips with both `allow_hp` paths, exercising
    /// the per-block `use_mv_hp(ref)` gate too.
    #[test]
    fn read_mv_allow_hp_gate_on_ref() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        // Small ref → HP allowed.
        let diff = Mv { row: 3, col: -5 };
        let buf = encode_mv(diff, ctx, /* use_hp = */ true);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, Mv { row: 8, col: 0 }, true, ctx, None);
        assert_eq!(got, Mv { row: 11, col: -5 });

        // Large ref → HP disabled even with allow_hp=true. Pick a
        // residual whose LSB-of-offset is 1 so the encode-without-HP
        // path matches the encode-with-HP path bit-for-bit on the
        // value.
        let diff = Mv { row: 4, col: -4 }; // both even mag, e=1 for class-0
        let buf = encode_mv(diff, ctx, /* use_hp = */ false);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, Mv { row: 200, col: 0 }, true, ctx, None);
        assert_eq!(got, Mv { row: 204, col: -4 });
    }

    /// `read_mv` adds the residual to `ref_mv`, including signed
    /// wrap (i16 add). Uses a small ref so the HP gate stays on,
    /// matching the encoder's `use_hp = true`.
    #[test]
    fn read_mv_adds_to_ref() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let diff = Mv { row: 12, col: -4 };
        let buf = encode_mv(diff, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let ref_mv = Mv { row: 30, col: 50 };
        let got = read_mv(&mut r, ref_mv, true, ctx, None);
        assert_eq!(got, Mv { row: 42, col: 46 });
    }

    /// Counts are bumped correctly for a known MV.
    #[test]
    fn read_mv_counts_increment() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let diff = Mv { row: 0, col: 4 }; // joint=HnzVz, col class 0
        let buf = encode_mv(diff, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let mut counts = NmvCounts::default();
        let got = read_mv(&mut r, Mv::ZERO, true, ctx, Some(&mut counts));
        assert_eq!(got, diff);
        // Joint count bumped on HnzVz (= 1).
        assert_eq!(counts.joints[MvJoint::HnzVz as usize], 1);
        // No row component decoded, so no vertical counts touched.
        assert_eq!(counts.comps[0].sign, [0, 0]);
        assert_eq!(counts.comps[0].classes, [0; MV_CLASSES]);
        // Col component: sign = 0 (positive), class 0, d=0, f=1, e=1.
        // value=4 → z=3 → class 0, offset=3, d=0, f=1, e=1.
        assert_eq!(counts.comps[1].sign, [1, 0]);
        assert_eq!(counts.comps[1].classes[MvClass::Class0 as usize], 1);
        assert_eq!(counts.comps[1].class0[0], 1);
        assert_eq!(counts.comps[1].class0_fp[0][1], 1);
        assert_eq!(counts.comps[1].class0_hp[1], 1);
        // The non-class-0 slots remain zero.
        assert_eq!(counts.comps[1].fp, [0; MV_FP_SIZE]);
        for row in counts.comps[1].bits.iter() {
            assert_eq!(*row, [0, 0]);
        }
    }

    /// `MV_JOINT_ZERO` joint short-circuits both component reads;
    /// only the joint count is bumped.
    #[test]
    fn read_mv_zero_joint_short_circuits() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let buf = encode_mv(Mv::ZERO, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let mut counts = NmvCounts::default();
        let got = read_mv(
            &mut r,
            Mv { row: 50, col: 50 },
            true,
            ctx,
            Some(&mut counts),
        );
        assert_eq!(got, Mv { row: 50, col: 50 });
        assert_eq!(counts.joints, [1, 0, 0, 0]);
        // No per-component counts moved.
        for comp in counts.comps.iter() {
            assert_eq!(*comp, NmvComponentCounts::default());
        }
    }

    /// HP bit count increments unconditionally — mirrors libvpx's
    /// `inc_mv_component` hard-coding `usehp=1` at the call site
    /// (`vp9_entropymv.c:145`). When the block-level read had
    /// `use_hp=false`, the hp bit is hard-coded to 1, and slot 1 still
    /// gets bumped. The previous test gating hp counts on `use_hp` was
    /// wrong; under-counting drifted `comp.hp` away from libvpx after
    /// every frame that adapts with `allow_hp` enabled. See D.22.
    #[test]
    fn read_mv_component_hp_count_always_bumps() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        // First case: small ref → HP enabled. diff col=4 → hp slot 1.
        let buf = encode_mv(Mv { row: 0, col: 4 }, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let mut counts = NmvCounts::default();
        let _ = read_mv(&mut r, Mv::ZERO, true, ctx, Some(&mut counts));
        assert_eq!(counts.comps[1].class0_hp[1], 1);
        assert_eq!(counts.comps[1].class0_hp[0], 0);

        // Second case: large ref → HP disabled at read time. Encode
        // without HP bit. Libvpx still bumps `class0_hp[1]` because
        // the fallback `hp=1` is treated as observed (`e = o & 1 = 1`).
        let buf = encode_mv(Mv { row: 0, col: 4 }, ctx, false);
        let mut r = BoolReader::new(&buf).expect("init");
        let mut counts = NmvCounts::default();
        let _ = read_mv(
            &mut r,
            Mv { row: 200, col: 200 },
            true,
            ctx,
            Some(&mut counts),
        );
        assert_eq!(counts.comps[1].class0_hp, [0, 1]);
    }

    // ---------- write_mv_component / write_mv round-trip tests ----------
    //
    // These exercise the encoder-side primitives added in S4.16a. The
    // existing test helpers above (`encode_component`, `encode_mv`)
    // mirror the same algorithm; here we directly call the new public
    // API and verify the decoder recovers the input.

    /// Drive `write_mv_component` and flush. Returns the byte-tight
    /// buffer the reader can consume.
    fn write_component_buf(value: i16, comp: &NmvComponent, use_hp: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_mv_component(&mut w, value, comp, use_hp, None);
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().expect("stop_encode")
        };
        buf.truncate(size);
        buf
    }

    /// Drive `write_mv` and flush.
    fn write_mv_buf(cur_mv: Mv, ref_mv: Mv, ctx: &NmvContext, allow_hp: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_mv(&mut w, cur_mv, ref_mv, allow_hp, ctx, None);
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().expect("stop_encode")
        };
        buf.truncate(size);
        buf
    }

    /// `write_mv_component` round-trips every value in class 0 (mag in
    /// 1..=16), both signs, with `use_hp = true`. This is the densest
    /// coverage band; any decomposition bug surfaces here.
    #[test]
    fn write_mv_component_round_trips_class0_all_values() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[0];
        for mag in 1i16..=16 {
            for &value in &[mag, -mag] {
                let buf = write_component_buf(value, comp, true);
                let mut r = BoolReader::new(&buf).expect("init");
                let got = read_mv_component(&mut r, comp, None, true);
                assert_eq!(got, value, "class0 mag {mag}, sign-signed {value}");
                assert!(!r.has_error(), "decoder over-ran buffer");
            }
        }
    }

    /// `write_mv_component` with `use_hp = false` requires the LSB of
    /// `offset = mag` to be 1, i.e. `abs(value)` is even. Verify every
    /// such value in class 0 round-trips (the decoder hard-codes hp=1).
    #[test]
    fn write_mv_component_round_trips_no_hp() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[1];
        // Even abs values in 2..=16 → mag = abs-1 ∈ {1,3,5,...,15},
        // LSB = 1, decoder's hp=1 fallback reconstructs the same value.
        for mag in (2i16..=16).step_by(2) {
            for &value in &[mag, -mag] {
                let buf = write_component_buf(value, comp, false);
                let mut r = BoolReader::new(&buf).expect("init");
                let got = read_mv_component(&mut r, comp, None, false);
                assert_eq!(got, value, "no-hp class0 value={value}");
            }
        }
    }

    /// One representative value per `MvClass` (1..=10). Uses the same
    /// representative magnitudes as `read_mv_component_each_class_round_trip`.
    #[test]
    fn write_mv_component_round_trips_each_class_one_value() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[0];
        let cases: &[(MvClass, i16)] = &[
            (MvClass::Class0, 4),
            (MvClass::Class1, 24),
            (MvClass::Class2, 48),
            (MvClass::Class3, 96),
            (MvClass::Class4, 192),
            (MvClass::Class5, 384),
            (MvClass::Class6, 768),
            (MvClass::Class7, 1536),
            (MvClass::Class8, 3072),
            (MvClass::Class9, 6144),
            (MvClass::Class10, 12288),
        ];
        for &(expected_class, value) in cases {
            let buf = write_component_buf(value, comp, true);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv_component(&mut r, comp, None, true);
            assert_eq!(got, value, "class {expected_class:?} value={value}");
        }
    }

    /// Extreme values at the edges of `MV_IN_USE_BITS = 14` (i.e.
    /// `mag = 2^14 - 1`, so `abs(value) = 2^14 = 16384`... no — `MV_LOW
    /// = -16384`, `MV_UPP = 16383`, so absolute max value is 16383 and
    /// the corresponding `mag = 16382`. With class 10 (`mag >= 8192`),
    /// `(d << 3) | (fr << 1) | hp` covers offsets 0..(2^11+8190) — but
    /// the safe bound for the residual itself is 8191 on the
    /// `value` side at `MV_CLASS10` boundary tests in the wider
    /// codebase. Use ±8191 as documented.
    #[test]
    fn write_mv_component_round_trips_extreme_values() {
        let comp = &DEFAULT_NMV_CONTEXT.comps[0];
        for &value in &[8191i16, -8191] {
            let buf = write_component_buf(value, comp, true);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv_component(&mut r, comp, None, true);
            assert_eq!(got, value, "extreme value={value}");
        }
    }

    /// `cur_mv == ref_mv` → joint == Zero → no component bits emitted.
    /// The decoder must reconstruct `ref_mv` unchanged.
    #[test]
    fn write_mv_round_trips_zero_joint() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let ref_mv = Mv { row: 17, col: -29 };
        let buf = write_mv_buf(ref_mv, ref_mv, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, ref_mv, true, ctx, None);
        assert_eq!(got, ref_mv);
    }

    /// `diff = (0, k)` → joint HnzVz; only the col component is on the
    /// wire.
    #[test]
    fn write_mv_round_trips_hnz_vz() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let ref_mv = Mv { row: 0, col: 0 };
        let cur_mv = Mv { row: 0, col: 9 };
        let buf = write_mv_buf(cur_mv, ref_mv, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, ref_mv, true, ctx, None);
        assert_eq!(got, cur_mv);
    }

    /// `diff = (k, 0)` → joint HzVnz; only the row component is on the
    /// wire.
    #[test]
    fn write_mv_round_trips_hz_vnz() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let ref_mv = Mv { row: 0, col: 0 };
        let cur_mv = Mv { row: -11, col: 0 };
        let buf = write_mv_buf(cur_mv, ref_mv, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, ref_mv, true, ctx, None);
        assert_eq!(got, cur_mv);
    }

    /// `diff = (k, m)` → joint HnzVnz; both components on the wire.
    #[test]
    fn write_mv_round_trips_hnz_vnz() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let ref_mv = Mv { row: 5, col: -3 };
        let cur_mv = Mv { row: 12, col: -10 };
        let buf = write_mv_buf(cur_mv, ref_mv, ctx, true);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_mv(&mut r, ref_mv, true, ctx, None);
        assert_eq!(got, cur_mv);
    }

    /// Writer-side counts must mirror libvpx's unconditional HP-count
    /// bump even when the per-block `use_mv_hp(ref)` gate disables HP
    /// coding. Otherwise encoder-side frame-context adaptation drifts
    /// from external decoders once frame-level HP MVs are enabled.
    #[test]
    fn write_mv_counts_hp_even_when_block_gate_disables_hp() {
        let ctx = &DEFAULT_NMV_CONTEXT;
        let ref_mv = Mv { row: 200, col: 200 };

        let mut buf = vec![0u8; 256];
        let mut counts = NmvCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_mv(
                &mut w,
                Mv { row: 200, col: 204 },
                ref_mv,
                /* allow_hp = */ true,
                ctx,
                Some(&mut counts),
            );
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().expect("stop_encode");
        }
        assert_eq!(counts.comps[1].class0_hp, [0, 1]);

        let mut buf = vec![0u8; 256];
        let mut counts = NmvCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_mv(
                &mut w,
                Mv { row: 200, col: 224 },
                ref_mv,
                /* allow_hp = */ true,
                ctx,
                Some(&mut counts),
            );
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().expect("stop_encode");
        }
        assert_eq!(counts.comps[1].hp, [0, 1]);
    }

    /// 200-iter fuzz over random (`cur_mv`, `ref_mv`, `allow_hp`)
    /// triples. Uses a deterministic XorShift64 PRNG so failures are
    /// reproducible. The diff is constrained to libvpx's
    /// `MV_IN_USE_BITS = 14` range on each axis; when the per-block
    /// HP gate disables HP, we round `abs(diff)` to the nearest even
    /// value so the encoder's `use_hp == false` contract holds.
    #[test]
    fn write_mv_round_trips_fuzz() {
        let ctx = &DEFAULT_NMV_CONTEXT;

        /// XorShift64 from a 64-bit seed. Same shape as the lt_pred
        /// fuzz harness.
        struct XorShift64(u64);
        impl XorShift64 {
            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            fn next_i16_in_range(&mut self, lo: i16, hi: i16) -> i16 {
                let span = (hi as i32 - lo as i32 + 1) as u64;
                let r = self.next_u64() % span;
                (lo as i32 + r as i32) as i16
            }
        }

        let mut rng = XorShift64(0x00C0_FFEE_F00D_1234);

        for iter in 0..200 {
            // Pick a ref MV anywhere in i16 (allowed; the per-block
            // HP gate falls back to no-HP for large refs).
            let ref_mv = Mv {
                row: rng.next_i16_in_range(-2000, 2000),
                col: rng.next_i16_in_range(-2000, 2000),
            };

            // Pick a diff in libvpx's MV_LOW..=MV_UPP per axis. Each
            // axis residual is bounded by `2^14 - 1 == 16383`.
            let mut diff_row = rng.next_i16_in_range(-1024, 1024);
            let mut diff_col = rng.next_i16_in_range(-1024, 1024);

            // allow_hp toggle.
            let allow_hp = (iter & 1) == 0;
            let use_hp = allow_hp && use_mv_hp(ref_mv);

            // If HP is gated off at the block level, the encoder needs
            // `abs(diff)` to be even (LSB-of-offset = 1) on any nonzero
            // axis. Coerce by setting bit 0 of `abs(diff)` to 0.
            if !use_hp {
                for d in [&mut diff_row, &mut diff_col] {
                    if *d != 0 && (*d).unsigned_abs() & 1 == 1 {
                        // Snap to the next-larger-magnitude even value.
                        // E.g. 5 -> 6, -5 -> -6. Both stay nonzero.
                        if *d > 0 {
                            *d += 1;
                        } else {
                            *d -= 1;
                        }
                    }
                }
            }

            let cur_mv = Mv {
                row: ref_mv.row.wrapping_add(diff_row),
                col: ref_mv.col.wrapping_add(diff_col),
            };

            let buf = write_mv_buf(cur_mv, ref_mv, ctx, allow_hp);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_mv(&mut r, ref_mv, allow_hp, ctx, None);
            assert_eq!(
                got, cur_mv,
                "fuzz iter {iter}: ref={ref_mv:?} diff=({diff_row},{diff_col}) allow_hp={allow_hp}"
            );
            assert!(!r.has_error(), "decoder over-ran buffer (iter {iter})");
        }
    }

    /// Spot-check the private `mv_class_and_base` helper against the
    /// representative magnitudes used in [`read_mv_component_each_class_round_trip`].
    #[test]
    fn mv_class_and_base_spot_checks() {
        // (mag, expected_class, expected_base)
        let cases: &[(u32, MvClass, u32)] = &[
            (0, MvClass::Class0, 0),
            // 8..=15 covers the "class 0 with idx >= 1" subcase that
            // exposed the missing base-override bug — see comment in
            // `mv_class_and_base`.
            (8, MvClass::Class0, 0),
            (15, MvClass::Class0, 0),
            (16, MvClass::Class1, 16),
            (31, MvClass::Class1, 16),
            (32, MvClass::Class2, 32),
            (8191, MvClass::Class9, 4096),
            (8192, MvClass::Class10, 8192),
            (16383, MvClass::Class10, 8192),
        ];
        for &(mag, expected_class, expected_base) in cases {
            let (c, b) = mv_class_and_base(mag);
            assert_eq!(c, expected_class, "mag {mag} class mismatch");
            assert_eq!(b, expected_base, "mag {mag} base mismatch");
        }
    }
}
