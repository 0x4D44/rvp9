//! Per-64×64 superblock loop-filter mask builder — the partition-tree
//! walker that ties [`crate::mode_info::ModeInfo`] to [`build_masks`]
//! (Task 6.3.4).
//!
//! Mirrors libvpx `vp9_setup_mask` (`vp9_loopfilter.c:880-1010`) plus
//! the small helpers `get_filter_level` and `mode_lf_lut`
//! (`vp9_loopfilter.c:207-210, 233-236`).
//!
//! Given a `MODE_INFO` grid (one entry per 8×8 luma block), the
//! starting `(mi_row, mi_col)` of a 64×64 superblock, and the
//! resolved per-frame `[seg][ref][mode]` filter-level cube produced
//! by [`crate::loopfilter::resolve_filter_levels`], walks the
//! partition tree of the superblock (32 → 16 → 8 levels), invoking
//! [`build_masks`] / [`build_y_mask`] at each partition leaf to
//! OR-accumulate the leaf's edge contributions into the
//! [`LoopFilterMask`].
//!
//! The output `lfm` is **zeroed** at entry — callers do not need to
//! pre-zero. This matches libvpx's `vp9_zero(*lfm)` at the top of
//! `vp9_setup_mask` (`vp9_loopfilter.c:916`).
//!
//! The frame-edge adjustments performed by libvpx
//! `vp9_adjust_mask` (`vp9_loopfilter.c:766-878`) — TX_32X32 →
//! TX_16X16 merge, 4×4-border → 8×8 promotion, left-frame-edge
//! clear — are **not** done here. Those land in a separate Task
//! 6.3.5 mirroring `vp9_adjust_mask`.

use crate::block::BlockSize;
use crate::mode_info::{BlockMode, InterMode, ModeInfo, RefFrame};

use super::mask::{BuildMaskInputs, LoopFilterMask, build_masks, build_y_mask, uv_txsize_420};
use crate::loopfilter::FrameFilterLevels;

/// MI-grid side length of a 64×64 superblock (in 8×8 units).
/// Mirrors libvpx `MI_BLOCK_SIZE` (`vp9_enums.h:25`).
const MI_BLOCK_SIZE: u32 = 8;

/// libvpx `mode_lf_lut[mi->mode]` (`vp9_loopfilter.c:207-210`):
/// returns 0 for every intra mode + `ZEROMV`, 1 for the remaining
/// inter modes (`NEARESTMV` / `NEARMV` / `NEWMV`).
#[inline]
fn mode_lf_idx(mi: &ModeInfo) -> usize {
    match &mi.mode {
        BlockMode::Intra { .. } => 0,
        BlockMode::Inter { mode, .. } => match mode {
            InterMode::ZeroMv => 0,
            InterMode::NearestMv | InterMode::NearMv | InterMode::NewMv => 1,
        },
    }
}

/// libvpx `mi->ref_frame[0]` mapped to the cube's ref-axis index
/// (`INTRA_FRAME=0, LAST=1, GOLDEN=2, ALTREF=3`). Intra blocks map
/// to `INTRA_FRAME = 0`.
#[inline]
fn ref0_idx(mi: &ModeInfo) -> usize {
    match &mi.mode {
        BlockMode::Intra { .. } => 0,
        BlockMode::Inter { ref_frames, .. } => match ref_frames[0] {
            // NoRefFrame is illegal for ref_frames[0] of an inter
            // block; treat as INTRA_FRAME to keep the lookup
            // in-bounds rather than panic.
            RefFrame::NoRefFrame | RefFrame::IntraFrame => 0,
            RefFrame::LastFrame => 1,
            RefFrame::GoldenFrame => 2,
            RefFrame::AltrefFrame => 3,
        },
    }
}

/// libvpx `get_filter_level(lfi_n, mi)`
/// (`vp9_loopfilter.c:233-236`):
/// `lfi_n->lvl[mi->segment_id][mi->ref_frame[0]][mode_lf_lut[mi->mode]]`.
#[inline]
fn get_filter_level(levels: &FrameFilterLevels, mi: &ModeInfo) -> u8 {
    let seg = (mi.segment_id as usize).min(crate::loopfilter::MAX_SEGMENTS - 1);
    levels[seg][ref0_idx(mi)][mode_lf_idx(mi)]
}

/// Assemble a [`BuildMaskInputs`] for `mi` at the given superblock
/// shifts. Pulls the resolved filter level from the per-frame cube
/// and derives the chroma TX size via [`uv_txsize_420`].
#[inline]
fn inputs_for(mi: &ModeInfo, levels: &FrameFilterLevels, shift_y: u32, shift_uv: u32)
    -> BuildMaskInputs
{
    BuildMaskInputs {
        block_size: mi.block_size,
        tx_size_y: mi.tx_size,
        tx_size_uv: uv_txsize_420(mi.block_size, mi.tx_size),
        filter_level: get_filter_level(levels, mi),
        is_inter: mi.is_inter(),
        skip: mi.skip,
        shift_y,
        shift_uv,
    }
}

/// Walk the partition tree of one 64×64 superblock and populate
/// `lfm` with the per-edge bit-fields.
///
/// `mi8x8` is a row-major slice of the per-8×8 `ModeInfo` grid; cell
/// `(r, c)` lives at `mi8x8[r * mi_stride + c]`. `mi_row` /
/// `mi_col` are the top-left 8×8 coordinates of the superblock in
/// the MI grid; `mi_rows` / `mi_cols` are the full-frame grid
/// extents so the walker can skip out-of-frame quadrants past the
/// frame edge.
///
/// `levels` is the resolved `[seg][ref][mode]` filter-level cube
/// (typically the output of [`crate::loopfilter::resolve_filter_levels`]).
///
/// Mirrors libvpx `vp9_setup_mask` (`vp9_loopfilter.c:880-1010`).
///
/// # Panics
///
/// Panics if `mi8x8` is too short for the SB's MI cells (i.e. if
/// `(mi_row + r) * mi_stride + (mi_col + c)` is out of bounds for
/// any in-frame cell visited by the walker).
#[allow(clippy::too_many_arguments)]
pub fn setup_mask(
    mi_row: u32,
    mi_col: u32,
    mi8x8: &[ModeInfo],
    mi_stride: usize,
    mi_rows: u32,
    mi_cols: u32,
    levels: &FrameFilterLevels,
    lfm: &mut LoopFilterMask,
) {
    *lfm = LoopFilterMask::ZERO;

    let max_rows = mi_rows.saturating_sub(mi_row).min(MI_BLOCK_SIZE);
    let max_cols = mi_cols.saturating_sub(mi_col).min(MI_BLOCK_SIZE);

    if max_rows == 0 || max_cols == 0 {
        return;
    }

    // Index helper: absolute MI position → slice index.
    let idx_at = |r: u32, c: u32| -> usize {
        (mi_row + r) as usize * mi_stride + (mi_col + c) as usize
    };

    // Top-left MI of the superblock. libvpx asserts `mip[0] != NULL`
    // before the switch (`:917`); the analogous safety in rvp9 is
    // that the slice is in-bounds for the (0, 0) cell.
    let top = &mi8x8[idx_at(0, 0)];

    match top.block_size {
        // ------------------------------------------------------------------
        // 64×64 leaf.
        // ------------------------------------------------------------------
        BlockSize::Block64x64 => {
            build_masks(&inputs_for(top, levels, 0, 0), lfm);
        }
        // ------------------------------------------------------------------
        // 64×32 — top half then bottom half.
        // ------------------------------------------------------------------
        BlockSize::Block64x32 => {
            build_masks(&inputs_for(top, levels, 0, 0), lfm);
            if 4 < max_rows {
                let bot = &mi8x8[idx_at(4, 0)];
                build_masks(&inputs_for(bot, levels, 32, 8), lfm);
            }
        }
        // ------------------------------------------------------------------
        // 32×64 — left half then right half.
        // ------------------------------------------------------------------
        BlockSize::Block32x64 => {
            build_masks(&inputs_for(top, levels, 0, 0), lfm);
            if 4 < max_cols {
                let right = &mi8x8[idx_at(0, 4)];
                build_masks(&inputs_for(right, levels, 4, 2), lfm);
            }
        }
        // ------------------------------------------------------------------
        // Otherwise: walk the 4 32×32 quadrants. Mirrors libvpx
        // `default:` branch at `vp9_loopfilter.c:933-1007`.
        // ------------------------------------------------------------------
        _ => walk_four_32x32(mi_row, mi_col, mi8x8, mi_stride, max_rows, max_cols, levels, lfm),
    }
}

/// 32×32 quadrant offsets (within the SB, in 8×8 units): top-left,
/// top-right, bottom-left, bottom-right (libvpx Morton order).
const SHIFT_32_Y: [u32; 4] = [0, 4, 32, 36];
const SHIFT_32_UV: [u32; 4] = [0, 2, 8, 10];
/// 16×16 sub-quadrant offsets within a 32×32 quadrant.
const SHIFT_16_Y: [u32; 4] = [0, 2, 16, 18];
const SHIFT_16_UV: [u32; 4] = [0, 1, 4, 5];
/// 8×8 sub-quadrant offsets within a 16×16 quadrant.
const SHIFT_8_Y: [u32; 4] = [0, 1, 8, 9];

/// Walk the four 32×32 quadrants of an SB whose top-level partition
/// is **not** 64×64 / 64×32 / 32×64. Each quadrant is itself either
/// a 32-leaf (32x32, 32x16, 16x32) or recurses to 16×16 and beyond.
#[allow(clippy::too_many_arguments)]
fn walk_four_32x32(
    mi_row: u32,
    mi_col: u32,
    mi8x8: &[ModeInfo],
    mi_stride: usize,
    max_rows: u32,
    max_cols: u32,
    levels: &FrameFilterLevels,
    lfm: &mut LoopFilterMask,
) {
    let idx_at = |r: u32, c: u32| -> usize {
        (mi_row + r) as usize * mi_stride + (mi_col + c) as usize
    };

    for idx_32 in 0u32..4 {
        let q32_row = (idx_32 >> 1) << 2; // 0 or 4
        let q32_col = (idx_32 & 1) << 2; // 0 or 4
        if q32_col >= max_cols || q32_row >= max_rows {
            continue;
        }
        let shift_y_32 = SHIFT_32_Y[idx_32 as usize];
        let shift_uv_32 = SHIFT_32_UV[idx_32 as usize];

        let q32 = &mi8x8[idx_at(q32_row, q32_col)];
        match q32.block_size {
            BlockSize::Block32x32 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
            }
            BlockSize::Block32x16 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
                if q32_row + 2 < max_rows {
                    let bot = &mi8x8[idx_at(q32_row + 2, q32_col)];
                    build_masks(
                        &inputs_for(bot, levels, shift_y_32 + 16, shift_uv_32 + 4),
                        lfm,
                    );
                }
            }
            BlockSize::Block16x32 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
                if q32_col + 2 < max_cols {
                    let right = &mi8x8[idx_at(q32_row, q32_col + 2)];
                    build_masks(
                        &inputs_for(right, levels, shift_y_32 + 2, shift_uv_32 + 1),
                        lfm,
                    );
                }
            }
            _ => walk_four_16x16(
                mi_row,
                mi_col,
                mi8x8,
                mi_stride,
                max_rows,
                max_cols,
                q32_row,
                q32_col,
                shift_y_32,
                shift_uv_32,
                levels,
                lfm,
            ),
        }
    }
}

/// Walk the four 16×16 sub-quadrants of a 32×32 region whose
/// top-level partition is not a 32-leaf. `q32_row` / `q32_col` are
/// the 32-quadrant's top-left position within the SB (in 8×8
/// units); `shift_y_32` / `shift_uv_32` are the running shifts the
/// 32-level walker passed in.
#[allow(clippy::too_many_arguments)]
fn walk_four_16x16(
    mi_row: u32,
    mi_col: u32,
    mi8x8: &[ModeInfo],
    mi_stride: usize,
    max_rows: u32,
    max_cols: u32,
    q32_row: u32,
    q32_col: u32,
    shift_y_32: u32,
    shift_uv_32: u32,
    levels: &FrameFilterLevels,
    lfm: &mut LoopFilterMask,
) {
    let idx_at = |r: u32, c: u32| -> usize {
        (mi_row + r) as usize * mi_stride + (mi_col + c) as usize
    };

    for idx_16 in 0u32..4 {
        let q16_row = q32_row + ((idx_16 >> 1) << 1); // +0 or +2
        let q16_col = q32_col + ((idx_16 & 1) << 1); // +0 or +2
        if q16_col >= max_cols || q16_row >= max_rows {
            continue;
        }
        let shift_y_16 = shift_y_32 + SHIFT_16_Y[idx_16 as usize];
        let shift_uv_16 = shift_uv_32 + SHIFT_16_UV[idx_16 as usize];

        let q16 = &mi8x8[idx_at(q16_row, q16_col)];
        match q16.block_size {
            BlockSize::Block16x16 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
            }
            BlockSize::Block16x8 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
                if q16_row + 1 < max_rows {
                    let bot = &mi8x8[idx_at(q16_row + 1, q16_col)];
                    // 16×8 bottom-half is Y-only: libvpx tracks UV
                    // only at the first 8×8 of a 16×16 group.
                    build_y_mask(&inputs_for(bot, levels, shift_y_16 + 8, 0), lfm);
                }
            }
            BlockSize::Block8x16 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
                if q16_col + 1 < max_cols {
                    let right = &mi8x8[idx_at(q16_row, q16_col + 1)];
                    build_y_mask(&inputs_for(right, levels, shift_y_16 + 1, 0), lfm);
                }
            }
            _ => {
                // Sub-16 partition: 4 × 8×8 leaves (potentially each
                // itself split into 8×4 / 4×8 / 4×4). The first
                // 8×8 (idx_8 = 0) carries the chroma-mask update;
                // libvpx invokes `build_masks` (full) for it then
                // `build_y_mask` for the remaining three.
                // (`vp9_loopfilter.c:985-1000`.)
                let first = &mi8x8[idx_at(q16_row, q16_col)];
                build_masks(
                    &inputs_for(first, levels, shift_y_16 + SHIFT_8_Y[0], shift_uv_16),
                    lfm,
                );
                for idx_8 in 1u32..4 {
                    let q8_row = q16_row + (idx_8 >> 1);
                    let q8_col = q16_col + (idx_8 & 1);
                    if q8_col >= max_cols || q8_row >= max_rows {
                        continue;
                    }
                    let q8 = &mi8x8[idx_at(q8_row, q8_col)];
                    build_y_mask(
                        &inputs_for(q8, levels, shift_y_16 + SHIFT_8_Y[idx_8 as usize], 0),
                        lfm,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::InterpFilter;
    use crate::intra_pred::IntraMode;
    use crate::mode_info::Mv;
    use crate::transforms::TxSize;

    /// All-segments-all-cells `levels` table that maps every
    /// (seg, ref, mode) tuple to the same filter level. Useful for
    /// pinning the mask without dragging in segmentation deltas.
    fn flat_levels(level: u8) -> FrameFilterLevels {
        [[[level; crate::loopfilter::MAX_MODE_LF_DELTAS];
            crate::loopfilter::MAX_REF_FRAMES]; crate::loopfilter::MAX_SEGMENTS]
    }

    /// A 64×64 intra `ModeInfo` with TX_32X32 luma transforms.
    fn intra_64x64() -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// A 16×16 intra `ModeInfo` with TX_4X4 luma transforms.
    fn intra_16x16_tx4() -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// An 8×8 intra `ModeInfo` with TX_4X4 transforms.
    fn intra_8x8_tx4() -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// A 4×4 intra `ModeInfo` (a 4-way split inside an 8×8).
    fn intra_4x4() -> ModeInfo {
        ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// Replicate `mi` across an 8×8 MI grid (the libvpx convention
    /// for partition-tree leaves ≥ 8×8: every covered MI cell
    /// holds the same pointer).
    fn fill_grid(mi: ModeInfo) -> Vec<ModeInfo> {
        vec![mi; 64]
    }

    /// `mode_lf_idx` returns 0 for intra, 0 for `ZEROMV`, 1 for
    /// other inter modes (mirrors libvpx `mode_lf_lut`).
    #[test]
    fn mode_lf_idx_matches_libvpx() {
        let mi_intra = intra_64x64();
        assert_eq!(mode_lf_idx(&mi_intra), 0);

        let make_inter = |m: InterMode| ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: m,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        assert_eq!(mode_lf_idx(&make_inter(InterMode::ZeroMv)), 0);
        assert_eq!(mode_lf_idx(&make_inter(InterMode::NearestMv)), 1);
        assert_eq!(mode_lf_idx(&make_inter(InterMode::NearMv)), 1);
        assert_eq!(mode_lf_idx(&make_inter(InterMode::NewMv)), 1);
    }

    /// `ref0_idx` returns 0 for intra blocks, and the numeric value
    /// of `ref_frames[0]` for inter blocks (LAST=1 / GOLDEN=2 /
    /// ALTREF=3).
    #[test]
    fn ref0_idx_matches_libvpx() {
        let mi_intra = intra_64x64();
        assert_eq!(ref0_idx(&mi_intra), 0);

        let make_inter = |r: RefFrame| ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::ZeroMv,
                ref_frames: [r, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        assert_eq!(ref0_idx(&make_inter(RefFrame::LastFrame)), 1);
        assert_eq!(ref0_idx(&make_inter(RefFrame::GoldenFrame)), 2);
        assert_eq!(ref0_idx(&make_inter(RefFrame::AltrefFrame)), 3);
    }

    /// 64×64 single-partition case: a whole-SB intra block with
    /// TX_32X32. `setup_mask` must produce the same `lfm` as a
    /// direct `build_masks` call at shift = 0.
    #[test]
    fn setup_mask_64x64_single_block() {
        let grid = fill_grid(intra_64x64());
        let levels = flat_levels(33);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Expected: build_masks called once at shift = 0 on the
        // 64x64 intra block.
        let mut expected = LoopFilterMask::ZERO;
        build_masks(
            &inputs_for(&intra_64x64(), &levels, 0, 0),
            &mut expected,
        );

        assert_eq!(got, expected);
        // Hand-pin: every lfl_y entry is 33.
        assert_eq!(got.lfl_y, [33u8; 64]);
        // above_y / left_y at TX_32X32 (= idx 3) per hand-trace
        // (see existing `build_masks_64x64_intra_tx32` test in
        // `loopfilter/mask.rs`).
        assert_eq!(got.above_y[3], 0x0000_00ff_0000_00ff);
        assert_eq!(got.left_y[3], 0x1111_1111_1111_1111);
        // Other TX-size slots untouched.
        assert_eq!(got.above_y[0], 0);
        assert_eq!(got.above_y[1], 0);
        assert_eq!(got.above_y[2], 0);
        assert_eq!(got.left_y[0], 0);
        assert_eq!(got.left_y[1], 0);
        assert_eq!(got.left_y[2], 0);
    }

    /// 16×16 sub-partitioned case: every 16×16 in the SB is a
    /// separate intra block with TX_4×4. `setup_mask` must visit
    /// every 16-quadrant and OR in its contribution.
    ///
    /// Hand-trace: for the SIZE_MASK_Y[Block16x16] = 0x0303 footprint
    /// at every 16×16 quadrant (16 of them), the resulting
    /// `int_4x4_y` should be the union — 0xffff_ffff_ffff_ffff
    /// (every 8×8 cell has int-4×4 edges).
    #[test]
    fn setup_mask_16x16_partitioned() {
        let grid = fill_grid(intra_16x16_tx4());
        let levels = flat_levels(10);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // 16 × 16x16 leaves at SHIFT_32_Y + SHIFT_16_Y combinations.
        // Each leaf's SIZE_MASK_Y[16x16] = 0x0303 ORs into int_4x4_y
        // at its (shift_y_32 + shift_y_16). The union covers every
        // 8x8 in the 8x8 grid — 0xffff_ffff_ffff_ffff.
        assert_eq!(got.int_4x4_y, 0xffff_ffff_ffff_ffff);

        // Every lfl_y cell painted with 10.
        assert_eq!(got.lfl_y, [10u8; 64]);

        // above_y[TX_4X4] gets the prediction-mask (0x3 << shift)
        // OR'd in for each 16×16 at every quadrant, plus the
        // size×txform contribution (size_mask 0x0303 & TX_4X4 all-ones
        // = 0x0303, << shift). Hand-trace bit pattern:
        //   shifts visited = { (idx_32 << ?) + (idx_16 << ?) ... }
        // For TX_4×4, ABOVE_64X64_TXFORM_MASK_Y = ~0, so the OR
        // collapses to SIZE_MASK_Y[16x16] << shift = 0x0303 << shift.
        // Across all 16 leaves these bit-patterns tile the entire
        // 64-bit mask.
        assert_eq!(got.above_y[0], 0xffff_ffff_ffff_ffff);
        assert_eq!(got.left_y[0], 0xffff_ffff_ffff_ffff);

        // No TX_8X8 / TX_16X16 / TX_32X32 contributions.
        assert_eq!(got.above_y[1], 0);
        assert_eq!(got.above_y[2], 0);
        assert_eq!(got.above_y[3], 0);
        assert_eq!(got.left_y[1], 0);
        assert_eq!(got.left_y[2], 0);
        assert_eq!(got.left_y[3], 0);
    }

    /// 4×4-leaf case: a single 16×16 region (the top-left quadrant
    /// of the SB's top-left 32×32) is partitioned into 4 × 8×8s,
    /// and each 8×8 is further split into 4 × 4×4s. All other
    /// 16×16 quadrants stay 16×16.
    ///
    /// In libvpx terms: `mip[0]->sb_type == BLOCK_4x4` for every MI
    /// cell in the top-left 16×16 region. The walker recurses to
    /// the deepest level, invoking `build_y_mask` for idx_8 ∈
    /// {1,2,3} and `build_masks` for idx_8 = 0.
    ///
    /// We pin the lfl_y array: the first 16×16 region's 4 cells
    /// (positions 0/1/8/9) should all hold the 4×4 leaves' filter
    /// level.
    #[test]
    fn setup_mask_4x4_leaf_in_top_left_16x16() {
        // Fill the entire grid with 16×16 intra/TX_4×4 baseline...
        let mut grid = fill_grid(intra_16x16_tx4());
        // ...but override the top-left 16×16 region to be a
        // 4-way-split of 8×8s, each of which is a 4-way-split of
        // 4×4s. In the MI grid (one MI per 8×8), that means the
        // 4 MI cells covering the 16×16 each hold a Block4x4
        // ModeInfo (libvpx's sub-8 convention: the parent
        // partition's MODE_INFO is replicated to every covered
        // 8×8 MI cell).
        //
        // Cells (0,0), (0,1), (1,0), (1,1) all hold a Block4x4 MI.
        grid[0] = intra_4x4();
        grid[1] = intra_4x4();
        grid[8] = intra_4x4();
        grid[9] = intra_4x4();

        let levels = flat_levels(5);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // The first 16×16 sub-region is the 4 cells at MI positions
        // (0,0), (0,1), (1,0), (1,1) → lfl_y positions 0, 1, 8, 9.
        // Each is a 4×4 leaf with filter_level = 5.
        assert_eq!(got.lfl_y[0], 5);
        assert_eq!(got.lfl_y[1], 5);
        assert_eq!(got.lfl_y[8], 5);
        assert_eq!(got.lfl_y[9], 5);

        // The remaining 15 × 16×16 leaves keep their 10 baseline...
        // no, wait: we override `levels` to 5 across the board.
        // Those 15 leaves also have filter_level 5. lfl_y[10..]
        // (other 16×16 cells) all == 5.
        // Spot-check a few:
        assert_eq!(got.lfl_y[2], 5);
        assert_eq!(got.lfl_y[7], 5);
        assert_eq!(got.lfl_y[63], 5);

        // int_4x4_y must have bits set at the 4 cells covering
        // the 4×4-split region (0, 1, 8, 9). Each 4×4 leaf's
        // SIZE_MASK_Y[Block4x4] = 0x1, so the OR'd contributions
        // are bits at positions {0, 1, 8, 9}. The 15 other 16×16
        // leaves contribute their 0x0303 footprints at their
        // respective shifts.
        //
        // Bits 0, 1, 8, 9 are all already set by the 16×16 leaf at
        // those quadrants (in the no-override version). With the
        // 4×4 override, those 4 bits still get set (now via the
        // 4×4 SIZE_MASK_Y contributions). Net union remains all 64
        // bits set.
        assert_eq!(got.int_4x4_y, 0xffff_ffff_ffff_ffff);
    }

    /// Frame-edge case: at the bottom-right superblock of a small
    /// frame, `max_rows`/`max_cols` clamp the walk. Past-frame
    /// quadrants must be skipped.
    ///
    /// Setup: 4×4-mi-cell frame (32×32 pixels), starting the SB at
    /// (0, 0). Visible region = 4×4 MIs. Place an 8×8 intra block
    /// at (0,0); rest of the visible region also 8×8 intra blocks.
    ///
    /// Expected: only the top-left 4×4 MI region gets edges; the
    /// past-frame 32×32 quadrants (idx_32 = 1, 2, 3 — column ≥ 4
    /// or row ≥ 4) are all skipped.
    #[test]
    fn setup_mask_frame_edge_clamps_quadrants() {
        // Fill 8×8 grid with 8×8 intra blocks (sub-partition of
        // every 16×16 region). MI grid stride is 8, but mi_rows /
        // mi_cols are 4 — only the top-left 4×4 MI region is in
        // frame.
        let grid = fill_grid(intra_8x8_tx4());
        let levels = flat_levels(7);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 4, 4, &levels, &mut got);

        // Only the top-left 4×4 MI region (= one 32×32 quadrant)
        // has any lfl_y entries set. Outside that: 0.
        // Inside: 16 cells × 7.
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(got.lfl_y[r * 8 + c], 7, "in-frame (r,c) = ({r}, {c})");
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                if r >= 4 || c >= 4 {
                    assert_eq!(
                        got.lfl_y[r * 8 + c],
                        0,
                        "past-frame (r,c) = ({r}, {c}) should be 0"
                    );
                }
            }
        }
    }

    /// `filter_level == 0` cells are no-ops. Setting `levels` to
    /// all-zero must produce an all-zero `lfm` even with a fully
    /// populated MI grid.
    #[test]
    fn setup_mask_filter_level_zero_all_noop() {
        let grid = fill_grid(intra_16x16_tx4());
        let levels = flat_levels(0);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);
        assert_eq!(got, LoopFilterMask::ZERO);
    }

    /// `setup_mask` zeroes the `lfm` at entry (mirrors libvpx's
    /// `vp9_zero(*lfm)` at `:916`).
    #[test]
    fn setup_mask_zeros_lfm_on_entry() {
        // Pre-populate `lfm` with garbage; setup_mask should still
        // produce a "fresh" mask matching a from-zero call.
        let grid = fill_grid(intra_64x64());
        let levels = flat_levels(33);

        let mut prepopulated = LoopFilterMask {
            left_y: [0xdead_beef_dead_beef; crate::entropy::TX_SIZES],
            above_y: [0xdead_beef_dead_beef; crate::entropy::TX_SIZES],
            int_4x4_y: 0xdead_beef_dead_beef,
            left_uv: [0xbeef; crate::entropy::TX_SIZES],
            above_uv: [0xbeef; crate::entropy::TX_SIZES],
            int_4x4_uv: 0xbeef,
            lfl_y: [0xff; 64],
        };
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut prepopulated);

        let mut fresh = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut fresh);

        assert_eq!(prepopulated, fresh);
    }
}
