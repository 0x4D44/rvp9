//! VP9 block-partition primitives — the [`BlockSize`] enum plus its
//! size-in-{4×4-units, 8×8-units, pixels} lookup tables.
//!
//! Mirrors libvpx `BLOCK_SIZE` (`vp9/common/vp9_enums.h:46-61`) and the
//! `num_*_blocks_{wide,high}_lookup` tables
//! (`vp9/common/vp9_common_data.{h,c}`).
//!
//! Block sizes are the partition-level shape descriptors. VP9 has 13:
//! all combinations of side ∈ {4, 8, 16, 32, 64} that aren't degenerate
//! (i.e. not 4×16, 4×32, etc.). Each side is in **pixels**; a `BLOCK_4×4`
//! covers exactly one transform-4×4 of luma.
//!
//! Used by:
//! - intra-prediction (dispatch by block shape),
//! - the partition-tree probability layer,
//! - loop-filter edge-mask building (the immediate consumer — Task 6.3),
//! - per-block coefficient decoding.

/// VP9 block partition size. Each variant corresponds to a `width ×
/// height` pixel rectangle the codec's partition tree can produce.
/// Numeric values match libvpx `BLOCK_4X4 = 0 .. BLOCK_64X64 = 12`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum BlockSize {
    /// 4×4
    Block4x4 = 0,
    /// 4×8
    Block4x8 = 1,
    /// 8×4
    Block8x4 = 2,
    /// 8×8
    Block8x8 = 3,
    /// 8×16
    Block8x16 = 4,
    /// 16×8
    Block16x8 = 5,
    /// 16×16
    Block16x16 = 6,
    /// 16×32
    Block16x32 = 7,
    /// 32×16
    Block32x16 = 8,
    /// 32×32
    Block32x32 = 9,
    /// 32×64
    Block32x64 = 10,
    /// 64×32
    Block64x32 = 11,
    /// 64×64
    Block64x64 = 12,
}

/// Number of VP9 block sizes (`BLOCK_SIZES = 13` in libvpx).
pub const BLOCK_SIZES: usize = 13;

impl BlockSize {
    /// Numeric index for table indexing (`0..13`).
    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block4x8 => 4,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 => 8,
            Self::Block16x8 | Self::Block16x16 | Self::Block16x32 => 16,
            Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 32,
            Self::Block64x32 | Self::Block64x64 => 64,
        }
    }

    /// Height in pixels.
    #[inline]
    #[must_use]
    pub const fn height(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block8x4 => 4,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 => 8,
            Self::Block8x16 | Self::Block16x16 | Self::Block32x16 => 16,
            Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 32,
            Self::Block32x64 | Self::Block64x64 => 64,
        }
    }

    /// Width in 4×4 units. Mirrors libvpx
    /// `num_4x4_blocks_wide_lookup[BLOCK_SIZES]`
    /// (`vp9_common_data.c:14-19`).
    #[inline]
    #[must_use]
    pub const fn n_4x4_wide(self) -> u8 {
        self.width() >> 2
    }

    /// Height in 4×4 units. Mirrors libvpx
    /// `num_4x4_blocks_high_lookup[BLOCK_SIZES]`.
    #[inline]
    #[must_use]
    pub const fn n_4x4_high(self) -> u8 {
        self.height() >> 2
    }

    /// Width in 8×8 units. Mirrors libvpx
    /// `num_8x8_blocks_wide_lookup[BLOCK_SIZES]`
    /// (`vp9_common_data.c:21-26`). For sub-8 widths this returns 1
    /// (libvpx's "clamp to at least 1" convention).
    #[inline]
    #[must_use]
    pub const fn n_8x8_wide(self) -> u8 {
        let w = self.width();
        if w < 8 { 1 } else { w >> 3 }
    }

    /// Height in 8×8 units. Same clamp-to-1 rule as
    /// [`n_8x8_wide`](Self::n_8x8_wide).
    #[inline]
    #[must_use]
    pub const fn n_8x8_high(self) -> u8 {
        let h = self.height();
        if h < 8 { 1 } else { h >> 3 }
    }

    /// Side `log2` of the **width** (i.e. `log2(width)`). For 4 →
    /// 2; 64 → 6. Useful for shift-based index math.
    #[inline]
    #[must_use]
    pub const fn width_log2(self) -> u8 {
        let mut w = self.width();
        let mut log = 0;
        while w > 1 {
            w >>= 1;
            log += 1;
        }
        log
    }

    /// Side `log2` of the **height**.
    #[inline]
    #[must_use]
    pub const fn height_log2(self) -> u8 {
        let mut h = self.height();
        let mut log = 0;
        while h > 1 {
            h >>= 1;
            log += 1;
        }
        log
    }

    /// Build a `BlockSize` from a `width × height` pair if the pair
    /// is one of the 13 legal VP9 block sizes; `None` otherwise.
    /// Useful for cross-checks against libvpx test vectors.
    #[inline]
    #[must_use]
    pub const fn from_dimensions(w: u8, h: u8) -> Option<Self> {
        match (w, h) {
            (4, 4) => Some(Self::Block4x4),
            (4, 8) => Some(Self::Block4x8),
            (8, 4) => Some(Self::Block8x4),
            (8, 8) => Some(Self::Block8x8),
            (8, 16) => Some(Self::Block8x16),
            (16, 8) => Some(Self::Block16x8),
            (16, 16) => Some(Self::Block16x16),
            (16, 32) => Some(Self::Block16x32),
            (32, 16) => Some(Self::Block32x16),
            (32, 32) => Some(Self::Block32x32),
            (32, 64) => Some(Self::Block32x64),
            (64, 32) => Some(Self::Block64x32),
            (64, 64) => Some(Self::Block64x64),
            _ => None,
        }
    }
}

/// VP9 partition type. The 4-leaf decision the partition-tree
/// decoder writes for every superblock down to 8×8. Mirrors libvpx
/// `PARTITION_TYPE` (`vp9_enums.h:63-69`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum PartitionType {
    /// No split — the block stays at its parent size.
    None = 0,
    /// Horizontal split: top half + bottom half, each `parent.width
    /// × parent.height/2`.
    Horz = 1,
    /// Vertical split: left half + right half, each `parent.width/2
    /// × parent.height`.
    Vert = 2,
    /// 4-way split into quarter-sized sub-blocks.
    Split = 3,
}

/// Number of partition types (`PARTITION_TYPES = 4` in libvpx).
pub const PARTITION_TYPES: usize = 4;

impl PartitionType {
    /// Numeric index for table indexing (`0..4`).
    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Map a 2-bit value from the bitstream tree decoder to a
    /// `PartitionType`. Returns `None` for invalid values (which
    /// shouldn't occur in valid streams since the tree decoder
    /// always lands at one of 0..3).
    #[inline]
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Horz),
            2 => Some(Self::Vert),
            3 => Some(Self::Split),
            _ => None,
        }
    }
}

/// Given a parent [`BlockSize`] and a [`PartitionType`], the child
/// block size. Returns `None` for invalid combinations (e.g. you
/// can't split a `Block4x4` further; you can only `None`-leave it).
/// Mirrors libvpx `subsize_lookup[PARTITION_TYPES][BLOCK_SIZES]`
/// (`vp9_common_data.c:71-88`).
#[must_use]
pub const fn subsize(parent: BlockSize, part: PartitionType) -> Option<BlockSize> {
    use BlockSize::*;
    use PartitionType::*;
    let result = match (part, parent) {
        // PARTITION_NONE: child == parent.
        (None, p) => return Some(p),

        // PARTITION_HORZ.
        (Horz, Block8x8) => Block8x4,
        (Horz, Block16x16) => Block16x8,
        (Horz, Block32x32) => Block32x16,
        (Horz, Block64x64) => Block64x32,

        // PARTITION_VERT.
        (Vert, Block8x8) => Block4x8,
        (Vert, Block16x16) => Block8x16,
        (Vert, Block32x32) => Block16x32,
        (Vert, Block64x64) => Block32x64,

        // PARTITION_SPLIT.
        (Split, Block8x8) => Block4x4,
        (Split, Block16x16) => Block8x8,
        (Split, Block32x32) => Block16x16,
        (Split, Block64x64) => Block32x32,

        _ => return Option::None,
    };
    Some(result)
}

/// Maximum TX size for a given block size. Mirrors libvpx
/// `max_txsize_lookup[BLOCK_SIZES]` (`vp9_common_data.c:90-93`).
#[must_use]
pub const fn max_txsize(bs: BlockSize) -> crate::transforms::TxSize {
    use BlockSize::*;
    use crate::transforms::TxSize;
    match bs {
        Block4x4 | Block4x8 | Block8x4 => TxSize::Tx4x4,
        Block8x8 | Block8x16 | Block16x8 => TxSize::Tx8x8,
        Block16x16 | Block16x32 | Block32x16 => TxSize::Tx16x16,
        Block32x32 | Block32x64 | Block64x32 | Block64x64 => TxSize::Tx32x32,
    }
}

/// libvpx `ss_size_lookup[BLOCK_SIZES][2][2]`
/// (`vp9/common/vp9_common_data.c:110-126`). Given a luma `bsize` and
/// the plane's chroma subsampling `(ss_x, ss_y)`, returns the
/// corresponding plane block size, or `None` for the `BLOCK_INVALID`
/// entries (a luma shape too small to subdivide at that subsampling).
///
/// Index order matches libvpx exactly: `[bsize][ss_x][ss_y]`.
/// Transcribed VERBATIM from the libvpx literal table (NOT derived).
#[must_use]
pub const fn ss_size_lookup(bsize: BlockSize, ss_x: usize, ss_y: usize) -> Option<BlockSize> {
    use BlockSize::*;
    // [bsize][ss_x][ss_y]; None == BLOCK_INVALID.
    let table: [[[Option<BlockSize>; 2]; 2]; BLOCK_SIZES] = [
        // ss_x==0,ss_y==0 | ss_x==0,ss_y==1 || ss_x==1,ss_y==0 | ss_x==1,ss_y==1
        [[Some(Block4x4), None], [None, None]],
        [[Some(Block4x8), Some(Block4x4)], [None, None]],
        [[Some(Block8x4), None], [Some(Block4x4), None]],
        [[Some(Block8x8), Some(Block8x4)], [Some(Block4x8), Some(Block4x4)]],
        [[Some(Block8x16), Some(Block8x8)], [None, Some(Block4x8)]],
        [[Some(Block16x8), None], [Some(Block8x8), Some(Block8x4)]],
        [[Some(Block16x16), Some(Block16x8)], [Some(Block8x16), Some(Block8x8)]],
        [[Some(Block16x32), Some(Block16x16)], [None, Some(Block8x16)]],
        [[Some(Block32x16), None], [Some(Block16x16), Some(Block16x8)]],
        [[Some(Block32x32), Some(Block32x16)], [Some(Block16x32), Some(Block16x16)]],
        [[Some(Block32x64), Some(Block32x32)], [None, Some(Block16x32)]],
        [[Some(Block64x32), None], [Some(Block32x32), Some(Block32x16)]],
        [[Some(Block64x64), Some(Block64x32)], [Some(Block32x64), Some(Block32x32)]],
    ];
    table[bsize.idx()][ss_x][ss_y]
}

/// Chroma plane block size for a luma `bsize` at subsampling
/// `(ss_x, ss_y)`. Wraps [`ss_size_lookup`], mirroring libvpx
/// `get_plane_block_size` (`vp9_blockd.h:280-285`).
///
/// For the four sub-8 luma shapes (`Block4x4`/`Block4x8`/`Block8x4`)
/// the lookup is `BLOCK_INVALID` at `(1,1)`; libvpx (and the prior
/// rvp9 code) clamp the chroma plane to `Block4x4` there. We reproduce
/// that by falling back to `Block4x4` whenever the lookup is `None`.
#[must_use]
pub const fn uv_plane_block_size(bsize: BlockSize, ss_x: usize, ss_y: usize) -> BlockSize {
    match ss_size_lookup(bsize, ss_x, ss_y) {
        Some(bs) => bs,
        None => BlockSize::Block4x4,
    }
}

/// Chroma plane TX size: `min(luma_tx, max_txsize(uv_plane_block_size))`.
/// Mirrors libvpx `get_uv_tx_size` (`vp9_blockd.h:262-267`).
#[must_use]
pub const fn uv_plane_tx_size(
    bsize: BlockSize,
    luma_tx: crate::transforms::TxSize,
    ss_x: usize,
    ss_y: usize,
) -> crate::transforms::TxSize {
    let cap = max_txsize(uv_plane_block_size(bsize, ss_x, ss_y));
    if (luma_tx as u8) < (cap as u8) {
        luma_tx
    } else {
        cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libvpx-pinned enum values (`vp9_enums.h:46-58`).
    #[test]
    fn block_size_indices_match_libvpx() {
        assert_eq!(BlockSize::Block4x4 as u8, 0);
        assert_eq!(BlockSize::Block4x8 as u8, 1);
        assert_eq!(BlockSize::Block8x4 as u8, 2);
        assert_eq!(BlockSize::Block8x8 as u8, 3);
        assert_eq!(BlockSize::Block8x16 as u8, 4);
        assert_eq!(BlockSize::Block16x8 as u8, 5);
        assert_eq!(BlockSize::Block16x16 as u8, 6);
        assert_eq!(BlockSize::Block16x32 as u8, 7);
        assert_eq!(BlockSize::Block32x16 as u8, 8);
        assert_eq!(BlockSize::Block32x32 as u8, 9);
        assert_eq!(BlockSize::Block32x64 as u8, 10);
        assert_eq!(BlockSize::Block64x32 as u8, 11);
        assert_eq!(BlockSize::Block64x64 as u8, 12);
        assert_eq!(BLOCK_SIZES, 13);
    }

    /// Width and height are correct for every size.
    #[test]
    fn dimensions() {
        for (bs, w, h) in [
            (BlockSize::Block4x4, 4, 4),
            (BlockSize::Block4x8, 4, 8),
            (BlockSize::Block8x4, 8, 4),
            (BlockSize::Block8x8, 8, 8),
            (BlockSize::Block8x16, 8, 16),
            (BlockSize::Block16x8, 16, 8),
            (BlockSize::Block16x16, 16, 16),
            (BlockSize::Block16x32, 16, 32),
            (BlockSize::Block32x16, 32, 16),
            (BlockSize::Block32x32, 32, 32),
            (BlockSize::Block32x64, 32, 64),
            (BlockSize::Block64x32, 64, 32),
            (BlockSize::Block64x64, 64, 64),
        ] {
            assert_eq!(bs.width(), w, "{bs:?} width");
            assert_eq!(bs.height(), h, "{bs:?} height");
            assert_eq!(bs.n_4x4_wide(), w / 4);
            assert_eq!(bs.n_4x4_high(), h / 4);
            assert_eq!(bs.n_8x8_wide(), if w < 8 { 1 } else { w / 8 });
            assert_eq!(bs.n_8x8_high(), if h < 8 { 1 } else { h / 8 });
        }
    }

    /// `width_log2` / `height_log2` for every size.
    #[test]
    fn log2_dimensions() {
        for (bs, log_w, log_h) in [
            (BlockSize::Block4x4, 2, 2),
            (BlockSize::Block4x8, 2, 3),
            (BlockSize::Block8x4, 3, 2),
            (BlockSize::Block8x8, 3, 3),
            (BlockSize::Block16x16, 4, 4),
            (BlockSize::Block32x32, 5, 5),
            (BlockSize::Block64x64, 6, 6),
            (BlockSize::Block16x32, 4, 5),
            (BlockSize::Block32x64, 5, 6),
        ] {
            assert_eq!(bs.width_log2(), log_w, "{bs:?} width_log2");
            assert_eq!(bs.height_log2(), log_h, "{bs:?} height_log2");
        }
    }

    /// `from_dimensions` round-trips every legal pair.
    #[test]
    fn from_dimensions_round_trip() {
        for &bs in &[
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ] {
            let recovered = BlockSize::from_dimensions(bs.width(), bs.height())
                .expect("legal dimensions");
            assert_eq!(recovered, bs);
        }
    }

    /// `from_dimensions` rejects illegal pairs.
    #[test]
    fn from_dimensions_rejects_illegal() {
        // VP9 disallows 4×16, 4×32, 16×4, 16×64, etc.
        for (w, h) in [(4u8, 16), (4, 32), (16, 4), (16, 64), (64, 4), (3, 3), (5, 5)] {
            assert!(BlockSize::from_dimensions(w, h).is_none(), "({w}, {h})");
        }
    }

    /// idx round-trip just to confirm the explicit `as u8` cast works.
    #[test]
    fn idx_matches_repr() {
        assert_eq!(BlockSize::Block4x4.idx(), 0);
        assert_eq!(BlockSize::Block64x64.idx(), 12);
    }

    /// PartitionType enum values pin to libvpx.
    #[test]
    fn partition_type_indices_match_libvpx() {
        assert_eq!(PartitionType::None as u8, 0);
        assert_eq!(PartitionType::Horz as u8, 1);
        assert_eq!(PartitionType::Vert as u8, 2);
        assert_eq!(PartitionType::Split as u8, 3);
        assert_eq!(PARTITION_TYPES, 4);
    }

    /// `PartitionType::from_u8` round-trips every legal value.
    #[test]
    fn partition_type_from_u8() {
        assert_eq!(PartitionType::from_u8(0), Some(PartitionType::None));
        assert_eq!(PartitionType::from_u8(1), Some(PartitionType::Horz));
        assert_eq!(PartitionType::from_u8(2), Some(PartitionType::Vert));
        assert_eq!(PartitionType::from_u8(3), Some(PartitionType::Split));
        assert_eq!(PartitionType::from_u8(4), None);
        assert_eq!(PartitionType::from_u8(255), None);
    }

    /// `subsize` matches libvpx `subsize_lookup` for the 4 legal
    /// parent block sizes that can be split (8×8, 16×16, 32×32,
    /// 64×64).
    #[test]
    fn subsize_legal_parents() {
        use BlockSize::*;
        use PartitionType::*;

        for &parent in &[Block8x8, Block16x16, Block32x32, Block64x64] {
            // NONE: child == parent.
            assert_eq!(subsize(parent, None), Some(parent));
        }

        // HORZ (parent.width × parent.height/2).
        assert_eq!(subsize(Block8x8, Horz), Some(Block8x4));
        assert_eq!(subsize(Block16x16, Horz), Some(Block16x8));
        assert_eq!(subsize(Block32x32, Horz), Some(Block32x16));
        assert_eq!(subsize(Block64x64, Horz), Some(Block64x32));

        // VERT (parent.width/2 × parent.height).
        assert_eq!(subsize(Block8x8, Vert), Some(Block4x8));
        assert_eq!(subsize(Block16x16, Vert), Some(Block8x16));
        assert_eq!(subsize(Block32x32, Vert), Some(Block16x32));
        assert_eq!(subsize(Block64x64, Vert), Some(Block32x64));

        // SPLIT (quarter).
        assert_eq!(subsize(Block8x8, Split), Some(Block4x4));
        assert_eq!(subsize(Block16x16, Split), Some(Block8x8));
        assert_eq!(subsize(Block32x32, Split), Some(Block16x16));
        assert_eq!(subsize(Block64x64, Split), Some(Block32x32));
    }

    /// `subsize` returns None for non-square or sub-8 parents
    /// combined with HORZ / VERT / SPLIT (only PARTITION_NONE is
    /// legal there). Mirrors libvpx's BLOCK_INVALID entries.
    #[test]
    fn subsize_rejects_illegal_combinations() {
        use BlockSize::*;
        use PartitionType::*;

        // 4x4 is too small to split.
        assert_eq!(subsize(Block4x4, Horz), Option::None);
        assert_eq!(subsize(Block4x4, Vert), Option::None);
        assert_eq!(subsize(Block4x4, Split), Option::None);
        // Non-square parents (libvpx never asks subsize for these).
        assert_eq!(subsize(Block16x8, Horz), Option::None);
        assert_eq!(subsize(Block32x16, Split), Option::None);
        assert_eq!(subsize(Block16x32, Vert), Option::None);
        assert_eq!(subsize(Block8x16, Horz), Option::None);
        // But PARTITION_NONE works for all parents.
        assert_eq!(subsize(Block4x4, None), Some(Block4x4));
        assert_eq!(subsize(Block16x8, None), Some(Block16x8));
        assert_eq!(subsize(Block8x16, None), Some(Block8x16));
    }

    /// `max_txsize` pins libvpx `max_txsize_lookup`
    /// (`vp9_common_data.c:90-93`).
    #[test]
    fn max_txsize_pin_libvpx() {
        use BlockSize::*;
        use crate::transforms::TxSize;
        assert_eq!(max_txsize(Block4x4), TxSize::Tx4x4);
        assert_eq!(max_txsize(Block4x8), TxSize::Tx4x4);
        assert_eq!(max_txsize(Block8x4), TxSize::Tx4x4);
        assert_eq!(max_txsize(Block8x8), TxSize::Tx8x8);
        assert_eq!(max_txsize(Block8x16), TxSize::Tx8x8);
        assert_eq!(max_txsize(Block16x8), TxSize::Tx8x8);
        assert_eq!(max_txsize(Block16x16), TxSize::Tx16x16);
        assert_eq!(max_txsize(Block16x32), TxSize::Tx16x16);
        assert_eq!(max_txsize(Block32x16), TxSize::Tx16x16);
        assert_eq!(max_txsize(Block32x32), TxSize::Tx32x32);
        assert_eq!(max_txsize(Block32x64), TxSize::Tx32x32);
        assert_eq!(max_txsize(Block64x32), TxSize::Tx32x32);
        assert_eq!(max_txsize(Block64x64), TxSize::Tx32x32);
    }

    /// All 13 block sizes in index order, for table-pinning loops.
    const ALL_BLOCK_SIZES: [BlockSize; BLOCK_SIZES] = {
        use BlockSize::*;
        [
            Block4x4, Block4x8, Block8x4, Block8x8, Block8x16, Block16x8, Block16x16, Block16x32,
            Block32x16, Block32x32, Block32x64, Block64x32, Block64x64,
        ]
    };

    /// Pin EVERY cell of `ss_size_lookup` against the libvpx literal
    /// table (`vp9_common_data.c:110-126`): 13 bsizes × 4 ss combos.
    #[test]
    fn ss_size_lookup_pins_libvpx() {
        use BlockSize::*;
        // [bsize] -> [(ss_x, ss_y, expected)]; None == BLOCK_INVALID.
        // Transcribed cell-for-cell from the libvpx literal table.
        let expected: [[Option<BlockSize>; 4]; BLOCK_SIZES] = [
            // order within row: (0,0), (0,1), (1,0), (1,1)
            [Some(Block4x4), None, None, None],
            [Some(Block4x8), Some(Block4x4), None, None],
            [Some(Block8x4), None, Some(Block4x4), None],
            [Some(Block8x8), Some(Block8x4), Some(Block4x8), Some(Block4x4)],
            [Some(Block8x16), Some(Block8x8), None, Some(Block4x8)],
            [Some(Block16x8), None, Some(Block8x8), Some(Block8x4)],
            [Some(Block16x16), Some(Block16x8), Some(Block8x16), Some(Block8x8)],
            [Some(Block16x32), Some(Block16x16), None, Some(Block8x16)],
            [Some(Block32x16), None, Some(Block16x16), Some(Block16x8)],
            [Some(Block32x32), Some(Block32x16), Some(Block16x32), Some(Block16x16)],
            [Some(Block32x64), Some(Block32x32), None, Some(Block16x32)],
            [Some(Block64x32), None, Some(Block32x32), Some(Block32x16)],
            [Some(Block64x64), Some(Block64x32), Some(Block32x64), Some(Block32x32)],
        ];
        for (i, &bsize) in ALL_BLOCK_SIZES.iter().enumerate() {
            for (j, &(ss_x, ss_y)) in [(0, 0), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                assert_eq!(
                    ss_size_lookup(bsize, ss_x, ss_y),
                    expected[i][j],
                    "ss_size_lookup({bsize:?}, {ss_x}, {ss_y})"
                );
            }
        }
    }

    /// `uv_plane_block_size(b, 1, 1)` reproduces the prior 13-arm match
    /// (the old tile.rs / reconstruct_inter.rs `uv_plane_block_size`).
    #[test]
    fn uv_plane_block_size_420_reproduces_old_match() {
        use BlockSize::*;
        // The exact 13-arm match from the deleted helpers.
        let expected = |b: BlockSize| -> BlockSize {
            match b {
                Block4x4 | Block4x8 | Block8x4 | Block8x8 => Block4x4,
                Block8x16 => Block4x8,
                Block16x8 => Block8x4,
                Block16x16 => Block8x8,
                Block16x32 => Block8x16,
                Block32x16 => Block16x8,
                Block32x32 => Block16x16,
                Block32x64 => Block16x32,
                Block64x32 => Block32x16,
                Block64x64 => Block32x32,
            }
        };
        for &b in &ALL_BLOCK_SIZES {
            assert_eq!(uv_plane_block_size(b, 1, 1), expected(b), "{b:?}");
        }
    }

    /// 4:4:4 identity: `uv_plane_block_size(b, 0, 0) == b` for all b.
    #[test]
    fn uv_plane_block_size_444_identity() {
        for &b in &ALL_BLOCK_SIZES {
            assert_eq!(uv_plane_block_size(b, 0, 0), b, "{b:?}");
        }
    }

    /// `uv_plane_tx_size` at (1,1) caps to the 4:2:0 chroma block's
    /// max TX, matching `min(luma_tx, max_txsize(uv_bsize))`.
    #[test]
    fn uv_plane_tx_size_420_caps() {
        use BlockSize::*;
        use crate::transforms::TxSize;
        // Block16x16 -> uv Block8x8 -> max TX Tx8x8.
        assert_eq!(uv_plane_tx_size(Block16x16, TxSize::Tx16x16, 1, 1), TxSize::Tx8x8);
        assert_eq!(uv_plane_tx_size(Block16x16, TxSize::Tx4x4, 1, 1), TxSize::Tx4x4);
        // Block64x64 -> uv Block32x32 -> max TX Tx32x32 (no cap).
        assert_eq!(uv_plane_tx_size(Block64x64, TxSize::Tx32x32, 1, 1), TxSize::Tx32x32);
    }
}
