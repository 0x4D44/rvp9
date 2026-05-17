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
}
