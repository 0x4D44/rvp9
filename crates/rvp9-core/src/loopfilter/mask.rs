//! Per-superblock loop-filter mask struct + the `build_masks`
//! function family.
//!
//! Mirrors libvpx `LOOP_FILTER_MASK` (`vp9/common/vp9_loopfilter.h:60-68`)
//! and `build_masks` / `build_y_mask`
//! (`vp9/common/vp9_loopfilter.c:652-743`).
//!
//! The decoder/encoder walk the partition tree of a 64×64 superblock
//! and call [`build_masks`] (or its Y-plane-only variant
//! [`build_y_mask`]) for every leaf block. Each call ORs the block's
//! contribution into the per-superblock [`LoopFilterMask`].
//!
//! ## Decoupling from `MODE_INFO`
//!
//! libvpx threads a `MODE_INFO *mi` pointer through `build_masks`.
//! That struct doesn't exist in rvp9 yet (it's owned by Task 8.2+
//! when per-block decoding lands). For Task 6.3.3 we accept the
//! 8 fields `build_masks` actually reads as explicit arguments —
//! see [`BuildMaskInputs`]. The eventual `MODE_INFO`-aware caller
//! just packages those 8 values from its `MODE_INFO` and passes
//! them in.

use super::mask_tables::{
    ABOVE_64X64_TXFORM_MASK_UV, ABOVE_64X64_TXFORM_MASK_Y, ABOVE_PREDICTION_MASK_UV,
    ABOVE_PREDICTION_MASK_Y, LEFT_64X64_TXFORM_MASK_UV, LEFT_64X64_TXFORM_MASK_Y,
    LEFT_PREDICTION_MASK_UV, LEFT_PREDICTION_MASK_Y, SIZE_MASK_UV, SIZE_MASK_Y,
};
use crate::block::BlockSize;
use crate::entropy::TX_SIZES;
use crate::transforms::TxSize;

/// Per-superblock loop-filter mask. Mirrors libvpx
/// `LOOP_FILTER_MASK` (`vp9_loopfilter.h:60-68`) bit-for-bit.
///
/// Each `u64` in `left_y` / `above_y` covers an 8×8 grid in the
/// 64×64 superblock (one bit per 8×8 sub-block). Each `u16` in
/// `left_uv` / `above_uv` covers the 4×4 chroma grid for 4:2:0.
/// `lfl_y[64]` is the per-8×8 luma filter level (used by the
/// edge-filter dispatch loop).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LoopFilterMask {
    /// Per-TX-size mask of 8×8 sub-blocks whose **left edge** the
    /// luma loop filter must visit.
    pub left_y: [u64; TX_SIZES],
    /// Per-TX-size mask of 8×8 sub-blocks whose **top edge** the
    /// luma loop filter must visit.
    pub above_y: [u64; TX_SIZES],
    /// Mask of 8×8 sub-blocks with **interior 4×4 luma edges** to
    /// filter. Only populated for TX_4×4 blocks.
    pub int_4x4_y: u64,
    /// Chroma analogue of `left_y` (4:2:0; 4×4 chroma grid).
    pub left_uv: [u16; TX_SIZES],
    /// Chroma analogue of `above_y`.
    pub above_uv: [u16; TX_SIZES],
    /// Chroma analogue of `int_4x4_y`.
    pub int_4x4_uv: u16,
    /// Per-8×8-luma filter level (0 = no filter). Index `(row << 3)
    /// | col` for the 8×8 at `(row, col)` in the superblock.
    pub lfl_y: [u8; 64],
}

impl Default for LoopFilterMask {
    fn default() -> Self {
        Self::ZERO
    }
}

impl LoopFilterMask {
    /// All-zero mask. Used as the per-superblock starting state
    /// before walking the partition tree. libvpx achieves the same
    /// via `memset(lfm, 0, sizeof(*lfm))` in `vp9_reset_lfm`.
    pub const ZERO: Self = Self {
        left_y: [0; TX_SIZES],
        above_y: [0; TX_SIZES],
        int_4x4_y: 0,
        left_uv: [0; TX_SIZES],
        above_uv: [0; TX_SIZES],
        int_4x4_uv: 0,
        lfl_y: [0; 64],
    };
}

/// Per-block inputs to [`build_masks`]. The 8 fields a libvpx
/// `MODE_INFO` actually exposes to the mask-building code, broken
/// out so this module doesn't have to know about `MODE_INFO`.
#[derive(Copy, Clone, Debug)]
pub struct BuildMaskInputs {
    /// Partition-tree leaf block size.
    pub block_size: BlockSize,
    /// Luma transform size.
    pub tx_size_y: TxSize,
    /// Chroma transform size (4:2:0 derives via
    /// [`uv_txsize_420`]).
    pub tx_size_uv: TxSize,
    /// Resolved per-block filter level (`0..=63`; 0 ⇒ skip).
    /// libvpx `get_filter_level(lfi_n, mi)`.
    pub filter_level: u8,
    /// True iff the block is an inter block (i.e. `is_inter_block(mi)`
    /// in libvpx). Combined with `skip` to short-circuit the
    /// in-block edges.
    pub is_inter: bool,
    /// True iff `mi->skip` is set (no coefficients).
    pub skip: bool,
    /// Luma shift: position of the block within the 64×64 superblock
    /// in 8×8 units, encoded as `(row << 3) | col`. Matches libvpx's
    /// `shift_y` argument convention.
    pub shift_y: u32,
    /// UV shift: position within the chroma 4×4 grid of the
    /// superblock, encoded as `(row << 2) | col` for 4:2:0. Matches
    /// libvpx's `shift_uv`.
    pub shift_uv: u32,
}

/// 4:2:0 chroma TX-size lookup. `uv_txsize_420(bs, tx_y)` returns
/// the chroma TX size libvpx uses for the given luma block and
/// luma TX size.
///
/// libvpx represents this as `uv_txsize_lookup[bs][tx_y][1][1]`
/// (`vp9_common_data.c:128-222`), the `(ss_x=1, ss_y=1)` slice of
/// the 4-D table. That table is exactly
/// `min(tx_y, max_txsize(ss_size_lookup[bs][ss_x][ss_y]))` —
/// libvpx `get_uv_tx_size_impl` (`vp9_blockd.h:240-258`). We delegate
/// to the shared [`crate::block::uv_plane_tx_size`] with `(1, 1)` so
/// the geometry lives in one place. Generalizing the loopfilter
/// mask-build to non-4:2:0 (threading `(ss_x, ss_y)` through
/// `setup_mask`/`inputs_for`) is deferred to G7.1c — see notes in the
/// caller at `setup_mask.rs::inputs_for`.
#[inline]
#[must_use]
pub const fn uv_txsize_420(block_size: BlockSize, tx_size_y: TxSize) -> TxSize {
    crate::block::uv_plane_tx_size(block_size, tx_size_y, 1, 1)
}

/// OR a block's edge contributions into the per-superblock loop-filter
/// mask. Mirrors libvpx `build_masks` (`vp9_loopfilter.c:652-724`).
///
/// `lfm` accumulates contributions across every block in the
/// superblock — call once per partition-tree leaf, with `shift_y`
/// and `shift_uv` set to the leaf's position within the superblock.
///
/// Short-circuits on `filter_level == 0` (no filter for this block).
/// For inter blocks with `skip == true`, the transform-size edges
/// and interior-4x4 edges are skipped (block-prediction edges
/// still apply). This matches the libvpx semantics at
/// `vp9_loopfilter.c:699`.
pub fn build_masks(inputs: &BuildMaskInputs, lfm: &mut LoopFilterMask) {
    let filter_level = inputs.filter_level;
    if filter_level == 0 {
        return;
    }

    let bs = inputs.block_size.idx();
    let tx_y = inputs.tx_size_y as usize;
    let tx_uv = inputs.tx_size_uv as usize;

    // 1. Per-8x8 luma filter levels: paint `n_8x8_wide × n_8x8_high`
    //    cells starting at `shift_y` with `filter_level`. libvpx
    //    uses `memset` per row; we use a slice fill since the rows
    //    are contiguous within each 8-element luma row of the
    //    superblock.
    let w = inputs.block_size.n_8x8_wide() as usize;
    let h = inputs.block_size.n_8x8_high() as usize;
    let mut index = inputs.shift_y as usize;
    for _ in 0..h {
        // The lfl_y array is 8x8; each row is 8 entries. The shift_y
        // encoding is (row << 3) | col, so adding 8 walks to the next
        // luma row.
        lfm.lfl_y[index..index + w].fill(filter_level);
        index += 8;
    }

    // 2. Block-prediction (outer) edges, indexed by block size only.
    lfm.above_y[tx_y] |= ABOVE_PREDICTION_MASK_Y[bs] << inputs.shift_y;
    lfm.above_uv[tx_uv] |= ABOVE_PREDICTION_MASK_UV[bs] << inputs.shift_uv;
    lfm.left_y[tx_y] |= LEFT_PREDICTION_MASK_Y[bs] << inputs.shift_y;
    lfm.left_uv[tx_uv] |= LEFT_PREDICTION_MASK_UV[bs] << inputs.shift_uv;

    // 3. Skip-inter: no transform edges needed inside.
    if inputs.skip && inputs.is_inter {
        return;
    }

    // 4. Transform-size edges, masked to the block's footprint.
    lfm.above_y[tx_y] |= (SIZE_MASK_Y[bs] & ABOVE_64X64_TXFORM_MASK_Y[tx_y]) << inputs.shift_y;
    lfm.above_uv[tx_uv] |=
        (SIZE_MASK_UV[bs] & ABOVE_64X64_TXFORM_MASK_UV[tx_uv]) << inputs.shift_uv;
    lfm.left_y[tx_y] |= (SIZE_MASK_Y[bs] & LEFT_64X64_TXFORM_MASK_Y[tx_y]) << inputs.shift_y;
    lfm.left_uv[tx_uv] |= (SIZE_MASK_UV[bs] & LEFT_64X64_TXFORM_MASK_UV[tx_uv]) << inputs.shift_uv;

    // 5. Interior 4x4 edges (only for blocks using TX_4×4).
    if inputs.tx_size_y == TxSize::Tx4x4 {
        lfm.int_4x4_y |= SIZE_MASK_Y[bs] << inputs.shift_y;
    }
    if inputs.tx_size_uv == TxSize::Tx4x4 {
        // libvpx ANDs against 0xffff here; in Rust the u16 mask is
        // already 16-bit so the AND is a no-op and clippy flags it.
        lfm.int_4x4_uv |= SIZE_MASK_UV[bs] << inputs.shift_uv;
    }
}

/// Y-plane-only variant of [`build_masks`], used by libvpx for the
/// 4-way split inside an 8×8 block (each sub-8 partition contributes
/// only Y edges since the chroma 8×8 stays whole at 4:2:0). Mirrors
/// `build_y_mask` (`vp9_loopfilter.c:729-757`).
pub fn build_y_mask(inputs: &BuildMaskInputs, lfm: &mut LoopFilterMask) {
    let filter_level = inputs.filter_level;
    if filter_level == 0 {
        return;
    }

    let bs = inputs.block_size.idx();
    let tx_y = inputs.tx_size_y as usize;

    let w = inputs.block_size.n_8x8_wide() as usize;
    let h = inputs.block_size.n_8x8_high() as usize;
    let mut index = inputs.shift_y as usize;
    for _ in 0..h {
        lfm.lfl_y[index..index + w].fill(filter_level);
        index += 8;
    }

    lfm.above_y[tx_y] |= ABOVE_PREDICTION_MASK_Y[bs] << inputs.shift_y;
    lfm.left_y[tx_y] |= LEFT_PREDICTION_MASK_Y[bs] << inputs.shift_y;

    if inputs.skip && inputs.is_inter {
        return;
    }

    lfm.above_y[tx_y] |= (SIZE_MASK_Y[bs] & ABOVE_64X64_TXFORM_MASK_Y[tx_y]) << inputs.shift_y;
    lfm.left_y[tx_y] |= (SIZE_MASK_Y[bs] & LEFT_64X64_TXFORM_MASK_Y[tx_y]) << inputs.shift_y;

    if inputs.tx_size_y == TxSize::Tx4x4 {
        lfm.int_4x4_y |= SIZE_MASK_Y[bs] << inputs.shift_y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-zero default.
    #[test]
    fn loop_filter_mask_zero() {
        let m = LoopFilterMask::default();
        assert_eq!(m.left_y, [0u64; TX_SIZES]);
        assert_eq!(m.above_y, [0u64; TX_SIZES]);
        assert_eq!(m.int_4x4_y, 0);
        assert_eq!(m.left_uv, [0u16; TX_SIZES]);
        assert_eq!(m.above_uv, [0u16; TX_SIZES]);
        assert_eq!(m.int_4x4_uv, 0);
        assert_eq!(m.lfl_y, [0u8; 64]);
    }

    /// `uv_txsize_420` matches hand-picked libvpx entries.
    #[test]
    fn uv_txsize_420_hand_checked() {
        // BLOCK_8X8 / tx_y = TX_8X8 / [1][1] = TX_4X4.
        assert_eq!(
            uv_txsize_420(BlockSize::Block8x8, TxSize::Tx8x8),
            TxSize::Tx4x4
        );
        // BLOCK_16X16 / tx_y = TX_16X16 / [1][1] = TX_8X8.
        assert_eq!(
            uv_txsize_420(BlockSize::Block16x16, TxSize::Tx16x16),
            TxSize::Tx8x8
        );
        // BLOCK_32X32 / tx_y = TX_32X32 / [1][1] = TX_16X16.
        assert_eq!(
            uv_txsize_420(BlockSize::Block32x32, TxSize::Tx32x32),
            TxSize::Tx16x16
        );
        // BLOCK_64X64 / tx_y = TX_32X32 / [1][1] = TX_32X32.
        assert_eq!(
            uv_txsize_420(BlockSize::Block64x64, TxSize::Tx32x32),
            TxSize::Tx32x32
        );
        // BLOCK_4X4 / tx_y = TX_4X4 → TX_4X4.
        assert_eq!(
            uv_txsize_420(BlockSize::Block4x4, TxSize::Tx4x4),
            TxSize::Tx4x4
        );
    }

    /// `build_masks` with `filter_level = 0` is a no-op.
    #[test]
    fn build_masks_filter_level_zero_is_noop() {
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block16x16,
            tx_size_y: TxSize::Tx16x16,
            tx_size_uv: TxSize::Tx8x8,
            filter_level: 0,
            is_inter: false,
            skip: false,
            shift_y: 0,
            shift_uv: 0,
        };
        let mut lfm = LoopFilterMask::default();
        build_masks(&inputs, &mut lfm);
        assert_eq!(lfm, LoopFilterMask::default());
    }

    /// `build_masks` for a 64×64 intra block at shift=0 with
    /// `tx_size_y = TX_32X32`: the full superblock gets painted
    /// with `filter_level`, every `above_y/left_y[TX_32X32]` bit
    /// gets set per the libvpx tables, and `int_4x4_y` stays 0
    /// (no TX_4X4 interior).
    #[test]
    fn build_masks_64x64_intra_tx32() {
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block64x64,
            tx_size_y: TxSize::Tx32x32,
            tx_size_uv: TxSize::Tx32x32,
            filter_level: 33,
            is_inter: false,
            skip: false,
            shift_y: 0,
            shift_uv: 0,
        };
        let mut lfm = LoopFilterMask::default();
        build_masks(&inputs, &mut lfm);

        // Every lfl_y entry filled.
        assert_eq!(lfm.lfl_y, [33u8; 64]);

        // above_y / left_y[TX_32X32] should equal prediction-mask
        // OR (size_mask & txform_mask) for the 64x64. Hand-trace:
        // - above_prediction_mask[64x64] = 0xff
        // - size_mask[64x64] = ~0
        // - above_64x64_txform_mask[TX_32X32] = 0x000000ff_000000ff
        // - left_prediction_mask[64x64] = 0x01010101_01010101
        // - left_64x64_txform_mask[TX_32X32] = 0x11111111_11111111
        // Expected (with shift = 0):
        //   above_y[3] = 0xff | (~0 & 0x000000ff_000000ff)
        //              = 0xff | 0x000000ff_000000ff
        //              = 0x000000ff_000000ff
        //   left_y[3]  = 0x01010101_01010101
        //              | (~0 & 0x11111111_11111111)
        //              = 0x11111111_11111111
        assert_eq!(lfm.above_y[3], 0x0000_00ff_0000_00ff);
        assert_eq!(lfm.left_y[3], 0x1111_1111_1111_1111);
        // Other TX sizes untouched.
        assert_eq!(lfm.above_y[0], 0);
        assert_eq!(lfm.above_y[1], 0);
        assert_eq!(lfm.above_y[2], 0);
        assert_eq!(lfm.left_y[0], 0);
        assert_eq!(lfm.left_y[1], 0);
        assert_eq!(lfm.left_y[2], 0);
        // No 4x4 interior edges for TX_32X32.
        assert_eq!(lfm.int_4x4_y, 0);
    }

    /// `build_masks` for a 16×16 intra block at shift=0 with
    /// `tx_size_y = TX_4X4`: `int_4x4_y` gets the block's
    /// `size_mask` (= 0x0303 — a 2×2 footprint).
    #[test]
    fn build_masks_16x16_intra_tx4_sets_int_4x4_y() {
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block16x16,
            tx_size_y: TxSize::Tx4x4,
            tx_size_uv: TxSize::Tx4x4,
            filter_level: 16,
            is_inter: false,
            skip: false,
            shift_y: 0,
            shift_uv: 0,
        };
        let mut lfm = LoopFilterMask::default();
        build_masks(&inputs, &mut lfm);

        // int_4x4_y bit pattern = SIZE_MASK_Y[16x16] = 0x0303.
        assert_eq!(lfm.int_4x4_y, 0x0303);
        // int_4x4_uv with TX_4X4 chroma = SIZE_MASK_UV[16x16] = 0x0001.
        assert_eq!(lfm.int_4x4_uv, 0x0001);
    }

    /// Skip-inter block: the prediction edges still set; the
    /// transform-size edges and `int_4x4_*` stay zero.
    #[test]
    fn build_masks_skip_inter_only_pred_edges() {
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block32x32,
            tx_size_y: TxSize::Tx16x16,
            tx_size_uv: TxSize::Tx8x8,
            filter_level: 20,
            is_inter: true,
            skip: true,
            shift_y: 0,
            shift_uv: 0,
        };
        let mut lfm = LoopFilterMask::default();
        build_masks(&inputs, &mut lfm);

        // Prediction edges only:
        //   above_y[2] = ABOVE_PREDICTION_MASK_Y[32x32] = 0x0f
        //   left_y[2]  = LEFT_PREDICTION_MASK_Y[32x32]  = 0x01010101
        assert_eq!(lfm.above_y[2], 0x0f);
        assert_eq!(lfm.left_y[2], 0x0101_0101);
        // No TX-size edges added (skip-inter cut us off).
        // For a 32x32 with TX_16X16, the non-skip path would have
        // OR'd in (size_mask & txform_mask). Hand-check that
        // contribution is absent: size_mask[32x32] = 0x0f0f_0f0f,
        // above_64x64_txform_mask[TX_16X16] = 0x00ff_00ff_00ff_00ff.
        // We must NOT see the 0x0f0f intersection bits beyond
        // 0x0f.
        // No int 4x4 since tx != TX_4X4 anyway.
        assert_eq!(lfm.int_4x4_y, 0);
    }

    /// `build_y_mask` produces the Y subset of `build_masks`
    /// (same Y fields, UV all zero).
    #[test]
    fn build_y_mask_matches_build_masks_y_subset() {
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block16x16,
            tx_size_y: TxSize::Tx4x4,
            tx_size_uv: TxSize::Tx4x4,
            filter_level: 25,
            is_inter: false,
            skip: false,
            shift_y: 0,
            shift_uv: 0,
        };
        let mut full = LoopFilterMask::default();
        let mut y_only = LoopFilterMask::default();
        build_masks(&inputs, &mut full);
        build_y_mask(&inputs, &mut y_only);

        // Y fields equal.
        assert_eq!(y_only.left_y, full.left_y);
        assert_eq!(y_only.above_y, full.above_y);
        assert_eq!(y_only.int_4x4_y, full.int_4x4_y);
        assert_eq!(y_only.lfl_y, full.lfl_y);
        // UV fields stay default in y_only.
        assert_eq!(y_only.left_uv, [0u16; TX_SIZES]);
        assert_eq!(y_only.above_uv, [0u16; TX_SIZES]);
        assert_eq!(y_only.int_4x4_uv, 0);
    }

    /// `build_masks` at non-zero `shift_y`: the prediction mask
    /// gets shifted, the lfl_y fills happen at the right luma
    /// row position.
    #[test]
    fn build_masks_with_shift_y_offset() {
        // 16x16 block at row=2 col=4 (shift_y = (2 << 3) | 4 = 20).
        let inputs = BuildMaskInputs {
            block_size: BlockSize::Block16x16,
            tx_size_y: TxSize::Tx16x16,
            tx_size_uv: TxSize::Tx8x8,
            filter_level: 7,
            is_inter: false,
            skip: false,
            shift_y: 20,
            shift_uv: (1 << 2) | 2, // chroma row 1 col 2 = shift_uv 6
        };
        let mut lfm = LoopFilterMask::default();
        build_masks(&inputs, &mut lfm);

        // lfl_y: 2x2 block at (row=2, col=4) → positions:
        // (2, 4..6), (3, 4..6).
        assert_eq!(lfm.lfl_y[20], 7);
        assert_eq!(lfm.lfl_y[21], 7);
        assert_eq!(lfm.lfl_y[28], 7);
        assert_eq!(lfm.lfl_y[29], 7);
        // Outside the 2x2: still 0.
        assert_eq!(lfm.lfl_y[0], 0);
        assert_eq!(lfm.lfl_y[19], 0);
        assert_eq!(lfm.lfl_y[22], 0);

        // above_y[TX_16X16] should include ABOVE_PREDICTION_MASK_Y[16x16]
        // << 20 = 0x03 << 20.
        assert!(lfm.above_y[2] & (0x03 << 20) != 0);
    }
}
