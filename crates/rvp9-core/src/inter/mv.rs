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
    BoolReader, CLASS0_BITS, CLASS0_SIZE, MV_CLASSES, MV_FP_SIZE, MV_JOINTS, NmvComponent,
    NmvComponentCounts, NmvContext, NmvCounts,
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
            c.class0_fp[di][fr as usize] =
                c.class0_fp[di][fr as usize].saturating_add(1);
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
    let joint = MvJoint::from_u8(joint_raw as u8)
        .expect("MV_JOINT_TREE always returns a valid MV_JOINT_*");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{BoolWriter, DEFAULT_NMV_CONTEXT};

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
            (MvClass::Class0, 4),       // z=3
            (MvClass::Class1, 24),      // z=23,   z>>3=2 → log=1
            (MvClass::Class2, 48),      // z=47,   z>>3=5 → log=2
            (MvClass::Class3, 96),      // z=95,   z>>3=11 → log=3
            (MvClass::Class4, 192),     // z=191,  z>>3=23 → log=4
            (MvClass::Class5, 384),     // z=383,  z>>3=47 → log=5
            (MvClass::Class6, 768),     // z=767,  z>>3=95 → log=6
            (MvClass::Class7, 1536),    // z=1535, z>>3=191 → log=7
            (MvClass::Class8, 3072),    // z=3071, z>>3=383 → log=8
            (MvClass::Class9, 6144),    // z=6143, z>>3=767 → log=9
            (MvClass::Class10, 12288),  // z=12287 ≥ 8192 → class 10
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
        let got = read_mv(&mut r, Mv { row: 50, col: 50 }, true, ctx, Some(&mut counts));
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
        let _ = read_mv(&mut r, Mv { row: 200, col: 200 }, true, ctx, Some(&mut counts));
        assert_eq!(counts.comps[1].class0_hp, [0, 1]);
    }
}
