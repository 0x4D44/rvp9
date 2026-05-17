//! Per-8×8-block descriptor — the [`ModeInfo`] struct + the
//! [`InterMode`], [`RefFrame`], [`Mv`], and [`BlockMode`] types it
//! aggregates.
//!
//! Mirrors libvpx `MODE_INFO` (`vp9/common/vp9_blockd.h:74-97`) plus
//! the supporting `PREDICTION_MODE` / `MV_REFERENCE_FRAME` /
//! `int_mv` / `b_mode_info` definitions.
//!
//! ## Design note: split inter / intra modes
//!
//! libvpx unifies intra and inter modes under a single 4-bit
//! `PREDICTION_MODE` (`0..=13`). Rvp9 separates them via the
//! [`BlockMode`] enum:
//!
//! - [`BlockMode::Intra`] carries `y_mode` / `uv_mode` as
//!   [`IntraMode`] (the existing [`crate::intra_pred::IntraMode`]).
//! - [`BlockMode::Inter`] carries [`InterMode`] + `ref_frames` +
//!   motion vectors + interpolation filter.
//!
//! The two cases are mutually exclusive (a block's `ref_frame[0] ==
//! INTRA_FRAME` means intra; anything else is inter). The enum
//! discriminant makes that exclusion type-safe and lets the
//! per-block decoder pattern-match on `BlockMode` rather than
//! threading a separate `is_inter` boolean.

use crate::bitstream::InterpFilter;
use crate::block::{BlockSize, PARTITION_TYPES, PartitionType};
use crate::entropy::BoolReader;
use crate::intra_pred::IntraMode;
use crate::transforms::TxSize;

/// libvpx `vp9_partition_tree` (`vp9_entropymode.c:262-264`):
///
/// ```text
///         node 0
///        /       \
///    NONE         node 2
///                /      \
///            HORZ       node 4
///                       /     \
///                    VERT    SPLIT
/// ```
///
/// Flat representation: positive entries are next-node indices,
/// negative entries `-leaf` are `PartitionType` terminals.
#[rustfmt::skip]
pub static PARTITION_TREE: [i8; 2 * PARTITION_TYPES - 2] = [
    -(PartitionType::None as i8),
    2,
    -(PartitionType::Horz as i8),
    4,
    -(PartitionType::Vert as i8),
    -(PartitionType::Split as i8),
];

/// Decode one [`PartitionType`] from the boolean reader given a
/// 3-entry context probability slot. Mirrors libvpx
/// `read_partition` (`vp9_decodemv.c:467-479`) — the tree-decode
/// part. The caller provides the right context's probability slot
/// by computing the 16-way `partition_plane_context`.
///
/// `probs` is `&fc.partition_prob[ctx]` (3 probabilities).
///
/// For block sizes < 8×8 (where `Block8x8` is the smallest
/// partitionable parent), libvpx forces `PARTITION_NONE` and
/// skips the tree decode entirely. The caller handles that case.
#[inline]
#[must_use]
pub fn read_partition(r: &mut BoolReader<'_>, probs: &[u8; PARTITION_TYPES - 1]) -> PartitionType {
    let raw = r.read_tree(&PARTITION_TREE, probs);
    // Safety: PARTITION_TREE only contains negations of valid
    // PartitionType discriminants, so `from_u8` always succeeds.
    PartitionType::from_u8(raw as u8).expect("partition tree returns valid PartitionType")
}

/// VP9 inter-prediction mode. Mirrors libvpx's
/// `NEARESTMV..=NEWMV` (`vp9_enums.h:120-123`); numeric values
/// match libvpx (`10..=13`) so libvpx-pinned tables (e.g.
/// `vp9_inter_mode_tree`) can index by `as u8 - 10`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum InterMode {
    /// `NEARESTMV` — copy the nearest decoded MV (zero residual).
    NearestMv = 10,
    /// `NEARMV` — copy the second-nearest decoded MV.
    NearMv = 11,
    /// `ZEROMV` — motion vector is `(0, 0)`.
    ZeroMv = 12,
    /// `NEWMV` — explicit MV difference encoded in the bitstream.
    NewMv = 13,
}

/// Number of inter modes (= `INTER_MODES` in libvpx).
pub const NUM_INTER_MODES: usize = 4;

impl InterMode {
    /// Zero-based index `[0..=3]` used to index per-context
    /// probability tables (which omit the leading `NEARESTMV` =
    /// libvpx convention).
    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize - InterMode::NearestMv as usize
    }

    /// Reverse of [`idx`](Self::idx): build from a `0..=3` index.
    /// Returns `None` for out-of-range input.
    #[inline]
    #[must_use]
    pub const fn from_idx(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::NearestMv),
            1 => Some(Self::NearMv),
            2 => Some(Self::ZeroMv),
            3 => Some(Self::NewMv),
            _ => None,
        }
    }
}

/// Motion-vector reference frame. Mirrors libvpx
/// `MV_REFERENCE_FRAME` (`vp9_blockd.h:57-65`).
///
/// Numeric values match libvpx exactly. `NoRefFrame` covers libvpx
/// `NO_REF_FRAME = -1`; the three real reference frames are
/// `LastFrame / GoldenFrame / AltrefFrame` (1, 2, 3). Intra-frame
/// blocks use `IntraFrame = 0`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(i8)]
pub enum RefFrame {
    /// `NO_REF_FRAME = -1` — unused slot (e.g. single-ref block's
    /// `ref_frames[1]`).
    NoRefFrame = -1,
    /// `INTRA_FRAME = 0` — intra-coded block.
    IntraFrame = 0,
    /// `LAST_FRAME = 1` — most-recently-decoded frame.
    LastFrame = 1,
    /// `GOLDEN_FRAME = 2` — periodically-refreshed reference.
    GoldenFrame = 2,
    /// `ALTREF_FRAME = 3` — alternate-reference frame.
    AltrefFrame = 3,
}

impl RefFrame {
    /// Build a `RefFrame` from a raw signed byte. Returns `None`
    /// for values outside the legal `-1..=3` range.
    #[inline]
    #[must_use]
    pub const fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Self::NoRefFrame),
            0 => Some(Self::IntraFrame),
            1 => Some(Self::LastFrame),
            2 => Some(Self::GoldenFrame),
            3 => Some(Self::AltrefFrame),
            _ => None,
        }
    }

    /// `true` for any reference frame that isn't `IntraFrame` or
    /// `NoRefFrame` (i.e. a real inter reference).
    #[inline]
    #[must_use]
    pub const fn is_inter(self) -> bool {
        matches!(self, Self::LastFrame | Self::GoldenFrame | Self::AltrefFrame)
    }
}

/// Motion vector at 1/8-pel precision (`Q3` fixed-point). Mirrors
/// libvpx `MV` struct (`vp9_mv.h:24-27`): `row` and `col` are
/// signed offsets in 1/8-pel units.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct Mv {
    /// Vertical component (positive = downward).
    pub row: i16,
    /// Horizontal component (positive = rightward).
    pub col: i16,
}

impl Mv {
    /// The zero MV (`{0, 0}`). Returned by `ZEROMV` blocks.
    pub const ZERO: Self = Self { row: 0, col: 0 };
}

/// Per-4×4 sub-block information for sub-8×8 partitions. Mirrors
/// libvpx `b_mode_info` (`vp9_blockd.h:48-51`).
///
/// Sub-8 (`Block4x4 / Block4x8 / Block8x4`) blocks store their
/// per-4×4 prediction info here; the parent [`ModeInfo`]'s `mode`
/// field is then not used for Y prediction (libvpx uses
/// `mi->bmi[i].as_mode` instead — see `vp9_blockd.h:99-101`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct BmiInfo {
    /// Per-sub-block intra mode (for keyframes / intra blocks) OR
    /// the MV for the sub-block (for inter blocks). The decoder
    /// distinguishes via the parent block's [`BlockMode`].
    pub as_mode: u8,
    /// First / second MV — only used by inter sub-8 blocks.
    pub as_mv: [Mv; 2],
}

/// Block-level prediction mode discriminant. Mirrors the conceptual
/// is-intra-or-inter split that libvpx encodes via
/// `ref_frame[0] == INTRA_FRAME`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockMode {
    /// Intra block: separate Y and UV prediction modes (the
    /// chroma mode is conditional on the luma mode in the
    /// bitstream).
    Intra {
        /// Luma prediction mode.
        y_mode: IntraMode,
        /// Chroma prediction mode.
        uv_mode: IntraMode,
    },
    /// Inter block: prediction mode + references + motion vectors.
    Inter {
        /// Inter-prediction mode (NEARESTMV / NEARMV / ZEROMV /
        /// NEWMV).
        mode: InterMode,
        /// 2 reference frames; `ref_frames[1] = NoRefFrame` for
        /// single-reference blocks.
        ref_frames: [RefFrame; 2],
        /// Motion vectors corresponding to `ref_frames`.
        mvs: [Mv; 2],
        /// Sub-pel interpolation filter (only meaningful when the
        /// frame's `interp_filter == Switchable`).
        interp_filter: InterpFilter,
    },
}

impl BlockMode {
    /// `true` iff this is an intra block.
    #[inline]
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self, Self::Intra { .. })
    }

    /// `true` iff this is an inter block.
    #[inline]
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        matches!(self, Self::Inter { .. })
    }
}

/// Per-8×8-block descriptor produced by the per-block decoder
/// (Task 8.2+). Mirrors libvpx `MODE_INFO`
/// (`vp9_blockd.h:74-97`).
///
/// One `ModeInfo` covers an aligned 8×8 luma region; sub-8
/// partitions still get a single `ModeInfo` per parent 8×8, with
/// the per-4×4 sub-block details in `bmi[4]`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModeInfo {
    /// libvpx `sb_type` — the partition-leaf block size that this
    /// `ModeInfo` represents.
    pub block_size: BlockSize,
    /// Luma transform size for this block.
    pub tx_size: TxSize,
    /// `skip` flag — block has no encoded residual.
    pub skip: bool,
    /// Segmentation index (`0..=MAX_SEGMENTS = 8`).
    pub segment_id: u8,
    /// `seg_id_predicted` — only meaningful when the segmentation
    /// header has `temporal_update == true`.
    pub seg_id_predicted: bool,
    /// Block-level prediction mode (carries the intra/inter split).
    pub mode: BlockMode,
    /// Per-4×4 sub-block info for sub-8 partitions
    /// (`Block4x4 / Block4x8 / Block8x4`). Untouched for ≥8×8
    /// partitions (libvpx leaves it zeroed in that case).
    pub bmi: [BmiInfo; 4],
}

impl ModeInfo {
    /// Convenience: `true` iff this is an intra block.
    #[inline]
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        self.mode.is_intra()
    }

    /// Convenience: `true` iff this is an inter block.
    #[inline]
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        self.mode.is_inter()
    }

    /// libvpx `get_y_mode` (`vp9_blockd.h:99-101`). For sub-8
    /// blocks, returns the per-`bmi[block]` intra mode (parsed
    /// as a `u8` and converted back to [`IntraMode`]). For ≥8×8
    /// intra blocks, returns the parent's `y_mode`. For inter
    /// blocks, returns `None`.
    #[inline]
    #[must_use]
    pub fn y_mode_for_subblock(&self, block: usize) -> Option<IntraMode> {
        match &self.mode {
            BlockMode::Intra { y_mode, .. } => {
                if matches!(
                    self.block_size,
                    BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
                ) {
                    IntraMode::from_u8(self.bmi[block].as_mode)
                } else {
                    Some(*y_mode)
                }
            }
            BlockMode::Inter { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `InterMode` numeric values match libvpx.
    #[test]
    fn inter_mode_values() {
        assert_eq!(InterMode::NearestMv as u8, 10);
        assert_eq!(InterMode::NearMv as u8, 11);
        assert_eq!(InterMode::ZeroMv as u8, 12);
        assert_eq!(InterMode::NewMv as u8, 13);
        assert_eq!(NUM_INTER_MODES, 4);
    }

    /// `InterMode::idx` is the zero-based offset libvpx uses for
    /// `[INTER_MODES - 1]` prob tables.
    #[test]
    fn inter_mode_idx() {
        assert_eq!(InterMode::NearestMv.idx(), 0);
        assert_eq!(InterMode::NearMv.idx(), 1);
        assert_eq!(InterMode::ZeroMv.idx(), 2);
        assert_eq!(InterMode::NewMv.idx(), 3);
        // Round-trip.
        for v in 0..=3u8 {
            let m = InterMode::from_idx(v).expect("legal");
            assert_eq!(m.idx() as u8, v);
        }
        assert!(InterMode::from_idx(4).is_none());
    }

    /// `RefFrame` numeric values match libvpx exactly.
    #[test]
    fn ref_frame_values() {
        assert_eq!(RefFrame::NoRefFrame as i8, -1);
        assert_eq!(RefFrame::IntraFrame as i8, 0);
        assert_eq!(RefFrame::LastFrame as i8, 1);
        assert_eq!(RefFrame::GoldenFrame as i8, 2);
        assert_eq!(RefFrame::AltrefFrame as i8, 3);
    }

    /// `RefFrame::from_i8` round-trips legal values and rejects
    /// out-of-range.
    #[test]
    fn ref_frame_from_i8() {
        for v in -1..=3i8 {
            let r = RefFrame::from_i8(v).expect("legal");
            assert_eq!(r as i8, v);
        }
        assert!(RefFrame::from_i8(4).is_none());
        assert!(RefFrame::from_i8(-2).is_none());
    }

    /// `RefFrame::is_inter` returns true only for real inter refs.
    #[test]
    fn ref_frame_is_inter() {
        assert!(!RefFrame::NoRefFrame.is_inter());
        assert!(!RefFrame::IntraFrame.is_inter());
        assert!(RefFrame::LastFrame.is_inter());
        assert!(RefFrame::GoldenFrame.is_inter());
        assert!(RefFrame::AltrefFrame.is_inter());
    }

    /// `Mv::ZERO` is `{0, 0}`.
    #[test]
    fn mv_zero() {
        assert_eq!(Mv::ZERO, Mv { row: 0, col: 0 });
    }

    /// `BlockMode::is_intra` / `is_inter` discriminator.
    #[test]
    fn block_mode_is_intra_is_inter() {
        let intra = BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::V,
        };
        assert!(intra.is_intra());
        assert!(!intra.is_inter());

        let inter = BlockMode::Inter {
            mode: InterMode::ZeroMv,
            ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
            mvs: [Mv::ZERO, Mv::ZERO],
            interp_filter: InterpFilter::EightTap,
        };
        assert!(!inter.is_intra());
        assert!(inter.is_inter());
    }

    /// `ModeInfo::y_mode_for_subblock`: ≥8×8 intra blocks return
    /// the parent's y_mode for any sub-block index; sub-8 intra
    /// blocks return the per-bmi mode.
    #[test]
    fn y_mode_for_subblock_paths() {
        // ≥8×8 intra: returns parent y_mode.
        let mi_big_intra = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::D45,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for block in 0..4 {
            assert_eq!(
                mi_big_intra.y_mode_for_subblock(block),
                Some(IntraMode::D45)
            );
        }

        // Sub-8 intra: per-bmi mode.
        let mut bmi: [BmiInfo; 4] = Default::default();
        bmi[0].as_mode = IntraMode::H as u8;
        bmi[1].as_mode = IntraMode::V as u8;
        bmi[2].as_mode = IntraMode::Tm as u8;
        bmi[3].as_mode = IntraMode::Dc as u8;
        let mi_sub8_intra = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc, // ignored at sub-8
                uv_mode: IntraMode::Dc,
            },
            bmi,
        };
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(0), Some(IntraMode::H));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(1), Some(IntraMode::V));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(2), Some(IntraMode::Tm));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(3), Some(IntraMode::Dc));

        // Inter: always None.
        let mi_inter = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv { row: 8, col: -4 }, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        for block in 0..4 {
            assert_eq!(mi_inter.y_mode_for_subblock(block), None);
        }
    }

    /// `PARTITION_TREE` is the libvpx-shaped 6-entry table.
    /// Hand-verified against `vp9_entropymode.c:262-264`.
    #[test]
    fn partition_tree_shape() {
        assert_eq!(PARTITION_TREE.len(), 6);
        // node 0: [-NONE, 2]
        assert_eq!(PARTITION_TREE[0], -(PartitionType::None as i8));
        assert_eq!(PARTITION_TREE[1], 2);
        // node 2: [-HORZ, 4]
        assert_eq!(PARTITION_TREE[2], -(PartitionType::Horz as i8));
        assert_eq!(PARTITION_TREE[3], 4);
        // node 4: [-VERT, -SPLIT]
        assert_eq!(PARTITION_TREE[4], -(PartitionType::Vert as i8));
        assert_eq!(PARTITION_TREE[5], -(PartitionType::Split as i8));
    }

    /// `read_partition` round-trips every PartitionType for an
    /// arbitrary context. The boolean tree encodes:
    /// - PARTITION_NONE  via bit sequence `0`
    /// - PARTITION_HORZ  via `1 0`
    /// - PARTITION_VERT  via `1 1 0`
    /// - PARTITION_SPLIT via `1 1 1`
    #[test]
    fn read_partition_round_trip() {
        use crate::entropy::BoolWriter;

        // We need a specific probs layout: use p=128 (max-entropy)
        // for every node so the encoder bits encode directly.
        let probs: [u8; 3] = [128, 128, 128];

        // Build, for each PartitionType, a buffer with the
        // appropriate bit sequence; verify the decoder returns
        // the expected value.
        let cases = [
            (PartitionType::None, vec![(0u8, probs[0])]),
            (PartitionType::Horz, vec![(1, probs[0]), (0, probs[1])]),
            (PartitionType::Vert, vec![(1, probs[0]), (1, probs[1]), (0, probs[2])]),
            (PartitionType::Split, vec![(1, probs[0]), (1, probs[1]), (1, probs[2])]),
        ];

        for (expected, bits) in cases {
            let mut buf = vec![0u8; 64];
            let mut w = BoolWriter::start_encode(&mut buf);
            for (bit, prob) in &bits {
                w.write(*bit, *prob);
            }
            // Pad.
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_partition(&mut r, &probs);
            assert_eq!(got, expected, "round-trip {expected:?}");
        }
    }

    /// `ModeInfo::is_intra` / `is_inter` delegates to `BlockMode`.
    #[test]
    fn mode_info_is_intra_is_inter() {
        let intra_mi = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: true,
            segment_id: 3,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        assert!(intra_mi.is_intra());
        assert!(!intra_mi.is_inter());

        let inter_mi = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: false,
            segment_id: 0,
            seg_id_predicted: true,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        assert!(!inter_mi.is_intra());
        assert!(inter_mi.is_inter());
    }
}
