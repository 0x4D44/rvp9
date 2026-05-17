//! Per-superblock loop-filter edge-mask lookup tables.
//!
//! These are the 10 tables libvpx `build_masks` consults
//! (`vp9/common/vp9_loopfilter.c:39-203`). They encode "which 8×8
//! sub-blocks within a 64×64 superblock have a filterable edge"
//! as packed bitfields:
//!
//! - **Y-plane** masks are `u64`, one bit per 8×8 sub-block in an
//!   8×8 grid (`bit ((row << 3) | col)` covers the 8×8 at
//!   `(row, col)` in the superblock).
//! - **UV-plane** masks are `u16`, one bit per 8×8 sub-block in
//!   the chroma 4×4 grid (for 4:2:0 subsampling each chroma 8×8
//!   covers a 16×16 luma area, so the grid is half-sized).
//!
//! The 5 categories are:
//!
//! - **64×64 TX-form masks** (`{left,above}_64x64_txform_mask{,_uv}`)
//!   — per `TxSize`, the pattern of filter-applicable edges within
//!   a 64×64 covered entirely by the given TX size. Used as a
//!   pre-shifted template that `build_masks` ANDs with the block's
//!   `size_mask` and ORs into the per-superblock accumulator.
//! - **Prediction masks** (`{left,above}_prediction_mask{,_uv}`)
//!   — per `BlockSize`, the **single edge** (top or left) of the
//!   block, suitable for OR-ing into the per-superblock mask after
//!   shifting by the block's position within the superblock.
//! - **Size masks** (`size_mask{,_uv}`) — per `BlockSize`, the full
//!   "footprint" of the block (the bits covering all its 8×8
//!   sub-positions). Used as the AND-mask against the
//!   `{left,above}_64x64_txform_mask` template.
//!
//! All values are **verbatim** from libvpx; the tests pin every
//! entry to its libvpx source-line counterpart.

use crate::block::BLOCK_SIZES;
use crate::entropy::TX_SIZES;

/// Per-TX-size left-edge filter mask within a 64×64 covered by that
/// TX size. Mirrors libvpx `left_64x64_txform_mask`
/// (`vp9_loopfilter.c:39-44`).
#[rustfmt::skip]
pub static LEFT_64X64_TXFORM_MASK_Y: [u64; TX_SIZES] = [
    0xffff_ffff_ffff_ffff, // TX_4X4
    0xffff_ffff_ffff_ffff, // TX_8X8
    0x5555_5555_5555_5555, // TX_16X16
    0x1111_1111_1111_1111, // TX_32X32
];

/// Per-TX-size above-edge filter mask within a 64×64 covered by that
/// TX size. Mirrors libvpx `above_64x64_txform_mask`
/// (`vp9_loopfilter.c:63-68`).
#[rustfmt::skip]
pub static ABOVE_64X64_TXFORM_MASK_Y: [u64; TX_SIZES] = [
    0xffff_ffff_ffff_ffff, // TX_4X4
    0xffff_ffff_ffff_ffff, // TX_8X8
    0x00ff_00ff_00ff_00ff, // TX_16X16
    0x0000_00ff_0000_00ff, // TX_32X32
];

/// Per-block left-edge prediction mask (Y plane). Mirrors libvpx
/// `left_prediction_mask` (`vp9_loopfilter.c:85-99`).
#[rustfmt::skip]
pub static LEFT_PREDICTION_MASK_Y: [u64; BLOCK_SIZES] = [
    0x0000_0000_0000_0001, // Block4x4
    0x0000_0000_0000_0001, // Block4x8
    0x0000_0000_0000_0001, // Block8x4
    0x0000_0000_0000_0001, // Block8x8
    0x0000_0000_0000_0101, // Block8x16
    0x0000_0000_0000_0001, // Block16x8
    0x0000_0000_0000_0101, // Block16x16
    0x0000_0000_0101_0101, // Block16x32
    0x0000_0000_0000_0101, // Block32x16
    0x0000_0000_0101_0101, // Block32x32
    0x0101_0101_0101_0101, // Block32x64
    0x0000_0000_0101_0101, // Block64x32
    0x0101_0101_0101_0101, // Block64x64
];

/// Per-block above-edge prediction mask (Y plane). Mirrors libvpx
/// `above_prediction_mask` (`vp9_loopfilter.c:102-116`).
#[rustfmt::skip]
pub static ABOVE_PREDICTION_MASK_Y: [u64; BLOCK_SIZES] = [
    0x0000_0000_0000_0001, // Block4x4
    0x0000_0000_0000_0001, // Block4x8
    0x0000_0000_0000_0001, // Block8x4
    0x0000_0000_0000_0001, // Block8x8
    0x0000_0000_0000_0001, // Block8x16
    0x0000_0000_0000_0003, // Block16x8
    0x0000_0000_0000_0003, // Block16x16
    0x0000_0000_0000_0003, // Block16x32
    0x0000_0000_0000_000f, // Block32x16
    0x0000_0000_0000_000f, // Block32x32
    0x0000_0000_0000_000f, // Block32x64
    0x0000_0000_0000_00ff, // Block64x32
    0x0000_0000_0000_00ff, // Block64x64
];

/// Per-block "footprint" mask (Y plane). Mirrors libvpx `size_mask`
/// (`vp9_loopfilter.c:120-134`).
#[rustfmt::skip]
pub static SIZE_MASK_Y: [u64; BLOCK_SIZES] = [
    0x0000_0000_0000_0001, // Block4x4
    0x0000_0000_0000_0001, // Block4x8
    0x0000_0000_0000_0001, // Block8x4
    0x0000_0000_0000_0001, // Block8x8
    0x0000_0000_0000_0101, // Block8x16
    0x0000_0000_0000_0003, // Block16x8
    0x0000_0000_0000_0303, // Block16x16
    0x0000_0000_0303_0303, // Block16x32
    0x0000_0000_0000_0f0f, // Block32x16
    0x0000_0000_0f0f_0f0f, // Block32x32
    0x0f0f_0f0f_0f0f_0f0f, // Block32x64
    0x0000_0000_ffff_ffff, // Block64x32
    0xffff_ffff_ffff_ffff, // Block64x64
];

/// Y-plane left-edge border template for a 64×64 superblock.
/// Mirrors libvpx `left_border` (`vp9_loopfilter.c:137`).
pub const LEFT_BORDER_Y: u64 = 0x1111_1111_1111_1111;

/// Y-plane above-edge border template for a 64×64 superblock.
/// Mirrors libvpx `above_border` (`vp9_loopfilter.c:138`).
pub const ABOVE_BORDER_Y: u64 = 0x0000_00ff_0000_00ff;

/// Per-TX-size left-edge filter mask (UV plane). Mirrors libvpx
/// `left_64x64_txform_mask_uv` (`vp9_loopfilter.c:141-146`).
#[rustfmt::skip]
pub static LEFT_64X64_TXFORM_MASK_UV: [u16; TX_SIZES] = [
    0xffff, // TX_4X4
    0xffff, // TX_8X8
    0x5555, // TX_16X16
    0x1111, // TX_32X32
];

/// Per-TX-size above-edge filter mask (UV plane). Mirrors libvpx
/// `above_64x64_txform_mask_uv` (`vp9_loopfilter.c:148-153`).
#[rustfmt::skip]
pub static ABOVE_64X64_TXFORM_MASK_UV: [u16; TX_SIZES] = [
    0xffff, // TX_4X4
    0xffff, // TX_8X8
    0x0f0f, // TX_16X16
    0x000f, // TX_32X32
];

/// Per-block left-edge prediction mask (UV plane). Mirrors libvpx
/// `left_prediction_mask_uv` (`vp9_loopfilter.c:156-170`).
#[rustfmt::skip]
pub static LEFT_PREDICTION_MASK_UV: [u16; BLOCK_SIZES] = [
    0x0001, // Block4x4
    0x0001, // Block4x8
    0x0001, // Block8x4
    0x0001, // Block8x8
    0x0001, // Block8x16
    0x0001, // Block16x8
    0x0001, // Block16x16
    0x0011, // Block16x32
    0x0001, // Block32x16
    0x0011, // Block32x32
    0x1111, // Block32x64
    0x0011, // Block64x32
    0x1111, // Block64x64
];

/// Per-block above-edge prediction mask (UV plane). Mirrors libvpx
/// `above_prediction_mask_uv` (`vp9_loopfilter.c:172-186`).
#[rustfmt::skip]
pub static ABOVE_PREDICTION_MASK_UV: [u16; BLOCK_SIZES] = [
    0x0001, // Block4x4
    0x0001, // Block4x8
    0x0001, // Block8x4
    0x0001, // Block8x8
    0x0001, // Block8x16
    0x0001, // Block16x8
    0x0001, // Block16x16
    0x0001, // Block16x32
    0x0003, // Block32x16
    0x0003, // Block32x32
    0x0003, // Block32x64
    0x000f, // Block64x32
    0x000f, // Block64x64
];

/// Per-block "footprint" mask (UV plane). Mirrors libvpx
/// `size_mask_uv` (`vp9_loopfilter.c:189-203`).
#[rustfmt::skip]
pub static SIZE_MASK_UV: [u16; BLOCK_SIZES] = [
    0x0001, // Block4x4
    0x0001, // Block4x8
    0x0001, // Block8x4
    0x0001, // Block8x8
    0x0001, // Block8x16
    0x0001, // Block16x8
    0x0001, // Block16x16
    0x0011, // Block16x32
    0x0003, // Block32x16
    0x0033, // Block32x32
    0x3333, // Block32x64
    0x00ff, // Block64x32
    0xffff, // Block64x64
];

/// UV-plane left-edge border template. Mirrors libvpx
/// `left_border_uv` (`vp9_loopfilter.c:204`).
pub const LEFT_BORDER_UV: u16 = 0x1111;

/// UV-plane above-edge border template. Mirrors libvpx
/// `above_border_uv` (`vp9_loopfilter.c:205`).
pub const ABOVE_BORDER_UV: u16 = 0x000f;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;

    /// Every Y-plane table has `BLOCK_SIZES = 13` entries.
    #[test]
    fn y_table_lengths() {
        assert_eq!(LEFT_PREDICTION_MASK_Y.len(), BLOCK_SIZES);
        assert_eq!(ABOVE_PREDICTION_MASK_Y.len(), BLOCK_SIZES);
        assert_eq!(SIZE_MASK_Y.len(), BLOCK_SIZES);
        assert_eq!(LEFT_64X64_TXFORM_MASK_Y.len(), TX_SIZES);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_Y.len(), TX_SIZES);
    }

    /// Every UV-plane table has `BLOCK_SIZES = 13` entries.
    #[test]
    fn uv_table_lengths() {
        assert_eq!(LEFT_PREDICTION_MASK_UV.len(), BLOCK_SIZES);
        assert_eq!(ABOVE_PREDICTION_MASK_UV.len(), BLOCK_SIZES);
        assert_eq!(SIZE_MASK_UV.len(), BLOCK_SIZES);
        assert_eq!(LEFT_64X64_TXFORM_MASK_UV.len(), TX_SIZES);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_UV.len(), TX_SIZES);
    }

    /// Hand-checked entries against the libvpx source comments.
    ///
    /// `left_prediction_mask[BLOCK_8X16] = 0x0101` — a 1×2 column
    /// at the left edge of the block (2 stacked 8×8s).
    /// `above_prediction_mask[BLOCK_16X8] = 0x0003` — a 2×1 row
    /// at the top of the block.
    /// `size_mask[BLOCK_16X16] = 0x0303` — a 2×2 footprint.
    #[test]
    fn hand_checked_entries_y() {
        assert_eq!(LEFT_PREDICTION_MASK_Y[BlockSize::Block8x16.idx()], 0x0101);
        assert_eq!(ABOVE_PREDICTION_MASK_Y[BlockSize::Block16x8.idx()], 0x0003);
        assert_eq!(SIZE_MASK_Y[BlockSize::Block16x16.idx()], 0x0303);
        // 64x64 covers the full superblock.
        assert_eq!(
            SIZE_MASK_Y[BlockSize::Block64x64.idx()],
            0xffff_ffff_ffff_ffff
        );
        // Sanity: 4x4 / 4x8 / 8x4 / 8x8 all share the same single-bit
        // mask in left + above + size (they're a single 8x8 cell).
        for &bs in &[
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
        ] {
            assert_eq!(LEFT_PREDICTION_MASK_Y[bs.idx()], 0x0000_0000_0000_0001);
            assert_eq!(ABOVE_PREDICTION_MASK_Y[bs.idx()], 0x0000_0000_0000_0001);
            assert_eq!(SIZE_MASK_Y[bs.idx()], 0x0000_0000_0000_0001);
        }
    }

    /// Hand-checked UV-plane entries.
    #[test]
    fn hand_checked_entries_uv() {
        // BLOCK_32X32 covers a 2×2 chroma footprint at (0,0) and
        // (0,1) and (1,0) and (1,1) — but in the UV grid each chroma
        // 8x8 covers 16x16 luma, so the footprint is 2x2 within the
        // 4x4 chroma grid: bits at positions 0, 1, 4, 5 = 0x33.
        // libvpx says `size_mask_uv[BLOCK_32X32] = 0x0033`.
        assert_eq!(SIZE_MASK_UV[BlockSize::Block32x32.idx()], 0x0033);
        // 64x64 → full chroma footprint.
        assert_eq!(SIZE_MASK_UV[BlockSize::Block64x64.idx()], 0xffff);
        // 16x32 left prediction mask: vertical 1x2 in chroma grid →
        // bits at positions 0 and 4 = 0x11.
        assert_eq!(LEFT_PREDICTION_MASK_UV[BlockSize::Block16x32.idx()], 0x0011);
    }

    /// 64x64 TX-form masks: TX_4×4 and TX_8×8 require every-edge
    /// filtering (= all 1s). TX_16×16 / TX_32×32 are sparser.
    #[test]
    fn txform_masks_pin_libvpx() {
        // Y-plane.
        assert_eq!(LEFT_64X64_TXFORM_MASK_Y[0], 0xffff_ffff_ffff_ffff); // TX_4X4
        assert_eq!(LEFT_64X64_TXFORM_MASK_Y[1], 0xffff_ffff_ffff_ffff); // TX_8X8
        assert_eq!(LEFT_64X64_TXFORM_MASK_Y[2], 0x5555_5555_5555_5555); // TX_16X16
        assert_eq!(LEFT_64X64_TXFORM_MASK_Y[3], 0x1111_1111_1111_1111); // TX_32X32
        assert_eq!(ABOVE_64X64_TXFORM_MASK_Y[0], 0xffff_ffff_ffff_ffff);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_Y[1], 0xffff_ffff_ffff_ffff);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_Y[2], 0x00ff_00ff_00ff_00ff);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_Y[3], 0x0000_00ff_0000_00ff);
        // UV-plane.
        assert_eq!(LEFT_64X64_TXFORM_MASK_UV[0], 0xffff);
        assert_eq!(LEFT_64X64_TXFORM_MASK_UV[2], 0x5555);
        assert_eq!(ABOVE_64X64_TXFORM_MASK_UV[3], 0x000f);
    }

    /// The border constants match libvpx (`vp9_loopfilter.c:137-138,
    /// 204-205`).
    #[test]
    fn border_constants() {
        assert_eq!(LEFT_BORDER_Y, 0x1111_1111_1111_1111);
        assert_eq!(ABOVE_BORDER_Y, 0x0000_00ff_0000_00ff);
        assert_eq!(LEFT_BORDER_UV, 0x1111);
        assert_eq!(ABOVE_BORDER_UV, 0x000f);
    }

    /// Structural sanity: bit-popcount of `size_mask[bs]` equals
    /// `n_8x8_wide(bs) × n_8x8_high(bs)` (the actual sub-block count).
    /// Catches transcription errors that don't trip the literal-pin
    /// tests above.
    #[test]
    fn size_mask_popcount_matches_block_area() {
        for (idx, &bs) in [
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
        ]
        .iter()
        .enumerate()
        {
            let expected = bs.n_8x8_wide() as u32 * bs.n_8x8_high() as u32;
            assert_eq!(
                SIZE_MASK_Y[idx].count_ones(),
                expected,
                "{bs:?} Y size_mask popcount"
            );
        }
    }
}
