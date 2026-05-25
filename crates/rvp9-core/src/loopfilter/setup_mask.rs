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
/// `mi8x8` is a row-major slice of the per-8×8 `Option<ModeInfo>`
/// grid; cell `(r, c)` lives at `mi8x8[r * mi_stride + c]`.
/// Wave 5b commit 1 (`fa06a1f`) pre-fills the decoder's frame-wide MI
/// slab with `Some(default_padding_modeinfo())` before tile workers
/// start, so every cell the walker reads is guaranteed `Some(_)`; the
/// inline `.expect()` at the read sites is therefore "free" in the
/// branch-predictor sense and lets the walker borrow the grid
/// directly without a per-frame `Vec<ModeInfo>` materialisation.
///
/// `mi_row` / `mi_col` are the top-left 8×8 coordinates of the
/// superblock in the MI grid; `mi_rows` / `mi_cols` are the
/// full-frame grid extents so the walker can skip out-of-frame
/// quadrants past the frame edge.
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
/// any in-frame cell visited by the walker), or if any cell visited
/// by the walker is `None` (the Wave 5b pre-fill guarantees this
/// never happens at the decoder/encoder LF call sites).
#[allow(clippy::too_many_arguments)]
pub fn setup_mask(
    mi_row: u32,
    mi_col: u32,
    mi8x8: &[Option<ModeInfo>],
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
    // that the slice is in-bounds for the (0, 0) cell and that
    // Wave 5b's pre-fill made it `Some`.
    let top = mi8x8[idx_at(0, 0)]
        .as_ref()
        .expect("Wave 5b: frame_mi_grid pre-filled with Some");

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
                let bot = mi8x8[idx_at(4, 0)]
                    .as_ref()
                    .expect("Wave 5b: frame_mi_grid pre-filled with Some");
                build_masks(&inputs_for(bot, levels, 32, 8), lfm);
            }
        }
        // ------------------------------------------------------------------
        // 32×64 — left half then right half.
        // ------------------------------------------------------------------
        BlockSize::Block32x64 => {
            build_masks(&inputs_for(top, levels, 0, 0), lfm);
            if 4 < max_cols {
                let right = mi8x8[idx_at(0, 4)]
                    .as_ref()
                    .expect("Wave 5b: frame_mi_grid pre-filled with Some");
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
    mi8x8: &[Option<ModeInfo>],
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

        let q32 = mi8x8[idx_at(q32_row, q32_col)]
            .as_ref()
            .expect("Wave 5b: frame_mi_grid pre-filled with Some");
        match q32.block_size {
            BlockSize::Block32x32 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
            }
            BlockSize::Block32x16 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
                if q32_row + 2 < max_rows {
                    let bot = mi8x8[idx_at(q32_row + 2, q32_col)]
                        .as_ref()
                        .expect("Wave 5b: frame_mi_grid pre-filled with Some");
                    build_masks(
                        &inputs_for(bot, levels, shift_y_32 + 16, shift_uv_32 + 4),
                        lfm,
                    );
                }
            }
            BlockSize::Block16x32 => {
                build_masks(&inputs_for(q32, levels, shift_y_32, shift_uv_32), lfm);
                if q32_col + 2 < max_cols {
                    let right = mi8x8[idx_at(q32_row, q32_col + 2)]
                        .as_ref()
                        .expect("Wave 5b: frame_mi_grid pre-filled with Some");
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
    mi8x8: &[Option<ModeInfo>],
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

        let q16 = mi8x8[idx_at(q16_row, q16_col)]
            .as_ref()
            .expect("Wave 5b: frame_mi_grid pre-filled with Some");
        match q16.block_size {
            BlockSize::Block16x16 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
            }
            BlockSize::Block16x8 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
                if q16_row + 1 < max_rows {
                    let bot = mi8x8[idx_at(q16_row + 1, q16_col)]
                        .as_ref()
                        .expect("Wave 5b: frame_mi_grid pre-filled with Some");
                    // 16×8 bottom-half is Y-only: libvpx tracks UV
                    // only at the first 8×8 of a 16×16 group.
                    build_y_mask(&inputs_for(bot, levels, shift_y_16 + 8, 0), lfm);
                }
            }
            BlockSize::Block8x16 => {
                build_masks(&inputs_for(q16, levels, shift_y_16, shift_uv_16), lfm);
                if q16_col + 1 < max_cols {
                    let right = mi8x8[idx_at(q16_row, q16_col + 1)]
                        .as_ref()
                        .expect("Wave 5b: frame_mi_grid pre-filled with Some");
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
                let first = mi8x8[idx_at(q16_row, q16_col)]
                    .as_ref()
                    .expect("Wave 5b: frame_mi_grid pre-filled with Some");
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
                    let q8 = mi8x8[idx_at(q8_row, q8_col)]
                        .as_ref()
                        .expect("Wave 5b: frame_mi_grid pre-filled with Some");
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
    /// holds the same pointer). Wrapped in `Some(_)` to match the
    /// post-W7-C1 `&[Option<ModeInfo>]` signature of `setup_mask`.
    fn fill_grid(mi: ModeInfo) -> Vec<Option<ModeInfo>> {
        vec![Some(mi); 64]
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
        grid[0] = Some(intra_4x4());
        grid[1] = Some(intra_4x4());
        grid[8] = Some(intra_4x4());
        grid[9] = Some(intra_4x4());

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

    /// A partition leaf at `block_size` with `tx_size` and intra mode.
    /// Helper used by the partition-tree tests below; the existing
    /// helpers above are special-cased to `Block64x64 / 16x16 / 8x8 /
    /// 4x4` so this lets us cover the other shapes (32x32 / 32x16 /
    /// 16x32 / 64x32 / 32x64 / 16x8 / 8x16) without repeating the
    /// struct boilerplate.
    fn intra_leaf(block_size: BlockSize, tx_size: TxSize) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size,
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

    /// An inter leaf at `block_size` with `mode`/`ref_frame`. Used by
    /// the `get_filter_level_mi` resolution tests so we can pin which
    /// LUT slot (`[seg][ref][mode_lf_idx]`) the walker reads.
    fn inter_leaf(
        block_size: BlockSize,
        tx_size: TxSize,
        mode: InterMode,
        ref_frame: RefFrame,
    ) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode,
                ref_frames: [ref_frame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        }
    }

    /// 64×32 top-level partition: top half + bottom half. Covers the
    /// `BlockSize::Block64x32` arm of `setup_mask`.
    #[test]
    fn setup_mask_64x32_top_and_bottom_halves() {
        // Whole SB is one Block64x32 partition pair: cells (0..4, 0..8)
        // are the top 64x32, (4..8, 0..8) are the bottom 64x32.
        let mi = intra_leaf(BlockSize::Block64x32, TxSize::Tx16x16);
        let grid = fill_grid(mi);
        let levels = flat_levels(20);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Every lfl_y must be 20 (both halves paint all 64 cells).
        assert_eq!(got.lfl_y, [20u8; 64]);

        // Hand-pin against two `build_masks` calls at shifts (0, 0) and
        // (32, 8) on the 64×32 leaf — exactly what the arm does.
        let mut expected = LoopFilterMask::ZERO;
        build_masks(&inputs_for(&mi, &levels, 0, 0), &mut expected);
        build_masks(&inputs_for(&mi, &levels, 32, 8), &mut expected);
        assert_eq!(got, expected);
    }

    /// 64×32 with frame-bottom clamp: when only the top half is in
    /// frame (`max_rows == 4`), the `if 4 < max_rows` branch must
    /// short-circuit the bottom-half `build_masks` call.
    #[test]
    fn setup_mask_64x32_clipped_to_top_half_only() {
        let mi = intra_leaf(BlockSize::Block64x32, TxSize::Tx16x16);
        let grid = fill_grid(mi);
        let levels = flat_levels(20);

        // mi_rows = 4 ⇒ max_rows = 4 ⇒ `if 4 < 4` is false ⇒ bottom
        // half skipped. Only top-row lfl_y cells get painted.
        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 4, 8, &levels, &mut got);

        // Top 4 rows of the lfl_y grid painted = 20; rows 4..8 = 0.
        for r in 0..4 {
            for c in 0..8 {
                assert_eq!(got.lfl_y[r * 8 + c], 20, "(r,c) = ({r}, {c})");
            }
        }
        for r in 4..8 {
            for c in 0..8 {
                assert_eq!(
                    got.lfl_y[r * 8 + c],
                    0,
                    "past-frame (r,c) = ({r}, {c})"
                );
            }
        }
    }

    /// 32×64 top-level partition: left half + right half. Covers the
    /// `BlockSize::Block32x64` arm of `setup_mask`.
    #[test]
    fn setup_mask_32x64_left_and_right_halves() {
        let mi = intra_leaf(BlockSize::Block32x64, TxSize::Tx16x16);
        let grid = fill_grid(mi);
        let levels = flat_levels(20);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Both halves cover the entire SB → every lfl_y = 20.
        assert_eq!(got.lfl_y, [20u8; 64]);

        // Pin against two `build_masks` calls at shifts (0, 0) and
        // (4, 2) on the 32×64 leaf.
        let mut expected = LoopFilterMask::ZERO;
        build_masks(&inputs_for(&mi, &levels, 0, 0), &mut expected);
        build_masks(&inputs_for(&mi, &levels, 4, 2), &mut expected);
        assert_eq!(got, expected);
    }

    /// 32×64 with frame-right clamp: only the left half is in frame
    /// (`max_cols == 4`), the `if 4 < max_cols` branch must skip the
    /// right-half `build_masks` call.
    #[test]
    fn setup_mask_32x64_clipped_to_left_half_only() {
        let mi = intra_leaf(BlockSize::Block32x64, TxSize::Tx16x16);
        let grid = fill_grid(mi);
        let levels = flat_levels(20);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 4, &levels, &mut got);

        // Left 4 columns painted; cols 4..8 untouched.
        for r in 0..8 {
            for c in 0..4 {
                assert_eq!(got.lfl_y[r * 8 + c], 20, "(r,c) = ({r}, {c})");
            }
            for c in 4..8 {
                assert_eq!(
                    got.lfl_y[r * 8 + c],
                    0,
                    "past-frame (r,c) = ({r}, {c})"
                );
            }
        }
    }

    /// 32×32 quadrant partition: four `Block32x32` leaves at the four
    /// quadrants. Covers the `Block32x32` arm of `walk_four_32x32`
    /// (currently only the `_ =>` arm via Block16x16 / Block8x8 base
    /// grids was exercised).
    #[test]
    fn setup_mask_32x32_four_quadrants() {
        // Top-left of every 32x32 quadrant = Block32x32. The other
        // MIs in each quadrant don't matter — the walker only reads
        // `q32 = mi8x8[(q32_row, q32_col)]`.
        let mi32 = intra_leaf(BlockSize::Block32x32, TxSize::Tx16x16);
        let grid = fill_grid(mi32);
        let levels = flat_levels(15);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Each quadrant's `build_masks` paints its 4x4 (8x8 units)
        // region of the lfl_y grid. All 4 quadrants together =
        // every cell painted with 15.
        assert_eq!(got.lfl_y, [15u8; 64]);

        // Pin against direct build_masks calls at SHIFT_32_Y /
        // SHIFT_32_UV positions.
        let mut expected = LoopFilterMask::ZERO;
        for i in 0..4 {
            build_masks(
                &inputs_for(&mi32, &levels, SHIFT_32_Y[i], SHIFT_32_UV[i]),
                &mut expected,
            );
        }
        assert_eq!(got, expected);
    }

    /// 32×16 partition within a 32x32 quadrant: top half + bottom
    /// half. Covers the `Block32x16` arm of `walk_four_32x32` and
    /// its `q32_row + 2 < max_rows` branch.
    #[test]
    fn setup_mask_32x16_within_top_left_quadrant() {
        // Top-left 32x32 quadrant: Block32x16 partition pair. Other
        // quadrants stay Block32x32 so the walker can recurse there
        // too without surprising us.
        let mi32x16 = intra_leaf(BlockSize::Block32x16, TxSize::Tx8x8);
        let mi32 = intra_leaf(BlockSize::Block32x32, TxSize::Tx16x16);
        let mut grid = fill_grid(mi32);
        // Cells covered by the top-left 32x32 quadrant (rows 0..4, cols
        // 0..4): place the 32x16 partition at (0,0). The bottom half
        // is at (2, 0).
        for r in 0..4 {
            for c in 0..4 {
                grid[r * 8 + c] = Some(mi32x16);
            }
        }
        let levels = flat_levels(15);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // The top-left quadrant should have its top-half + bottom-half
        // contributions both applied (q32_row=0, q32_row+2=2 < 4).
        // Pin via direct build_masks calls.
        let mut expected = LoopFilterMask::ZERO;
        // Top-left quadrant: 32x16 top + 32x16 bottom.
        build_masks(&inputs_for(&mi32x16, &levels, 0, 0), &mut expected);
        build_masks(&inputs_for(&mi32x16, &levels, 16, 4), &mut expected);
        // Other 3 quadrants: each a Block32x32.
        for i in 1..4 {
            build_masks(
                &inputs_for(&mi32, &levels, SHIFT_32_Y[i], SHIFT_32_UV[i]),
                &mut expected,
            );
        }
        assert_eq!(got, expected);
    }

    /// 32×16 with the bottom half clipped off (`q32_row + 2 >=
    /// max_rows`). Tests the `if q32_row + 2 < max_rows` guard.
    #[test]
    fn setup_mask_32x16_clipped_bottom_half_skipped() {
        // mi_rows = 2 ⇒ max_rows = 2 ⇒ only the top half of the
        // top-left 32x16 fits. The bottom half (q32_row=0, +2=2 ≥
        // max_rows=2) must be skipped.
        let mi32x16 = intra_leaf(BlockSize::Block32x16, TxSize::Tx8x8);
        let grid = fill_grid(mi32x16);
        let levels = flat_levels(15);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 2, 8, &levels, &mut got);

        // Only the top 2 rows × first 4 cols of lfl_y (the top-half
        // 32x16) get painted. Rows 2..8 stay 0.
        for r in 0..2 {
            for c in 0..4 {
                assert_eq!(got.lfl_y[r * 8 + c], 15, "in-frame (r,c) = ({r}, {c})");
            }
        }
        for r in 2..8 {
            for c in 0..8 {
                assert_eq!(
                    got.lfl_y[r * 8 + c],
                    0,
                    "past-frame (r,c) = ({r}, {c})"
                );
            }
        }
    }

    /// 16×32 partition within a 32x32 quadrant: left half + right
    /// half. Covers the `Block16x32` arm of `walk_four_32x32`.
    #[test]
    fn setup_mask_16x32_within_top_left_quadrant() {
        let mi16x32 = intra_leaf(BlockSize::Block16x32, TxSize::Tx8x8);
        let mi32 = intra_leaf(BlockSize::Block32x32, TxSize::Tx16x16);
        let mut grid = fill_grid(mi32);
        // Top-left 32x32 quadrant gets a Block16x32 pair.
        for r in 0..4 {
            for c in 0..4 {
                grid[r * 8 + c] = Some(mi16x32);
            }
        }
        let levels = flat_levels(15);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        let mut expected = LoopFilterMask::ZERO;
        build_masks(&inputs_for(&mi16x32, &levels, 0, 0), &mut expected);
        build_masks(&inputs_for(&mi16x32, &levels, 2, 1), &mut expected);
        for i in 1..4 {
            build_masks(
                &inputs_for(&mi32, &levels, SHIFT_32_Y[i], SHIFT_32_UV[i]),
                &mut expected,
            );
        }
        assert_eq!(got, expected);
    }

    /// 16×32 with the right half clipped off (`q32_col + 2 >=
    /// max_cols`). Tests the `if q32_col + 2 < max_cols` guard.
    #[test]
    fn setup_mask_16x32_clipped_right_half_skipped() {
        let mi16x32 = intra_leaf(BlockSize::Block16x32, TxSize::Tx8x8);
        let grid = fill_grid(mi16x32);
        let levels = flat_levels(15);

        // max_cols = 2 ⇒ q32_col=0, +2=2 ≥ 2 ⇒ right half skipped.
        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 2, &levels, &mut got);

        // Only top-left 2x4 (rows 0..4, cols 0..2) painted.
        for r in 0..4 {
            for c in 0..2 {
                assert_eq!(got.lfl_y[r * 8 + c], 15, "in-frame (r,c) = ({r}, {c})");
            }
            for c in 2..8 {
                assert_eq!(
                    got.lfl_y[r * 8 + c],
                    0,
                    "past-frame (r,c) = ({r}, {c})"
                );
            }
        }
    }

    /// 16×8 partition within a 16×16 region: top half (Block16x8) +
    /// bottom half (Block16x8) via `build_y_mask`. Covers the
    /// `Block16x8` arm of `walk_four_16x16`.
    #[test]
    fn setup_mask_16x8_within_top_left_quadrant() {
        let mi16x8 = intra_leaf(BlockSize::Block16x8, TxSize::Tx8x8);
        let mi16 = intra_leaf(BlockSize::Block16x16, TxSize::Tx8x8);
        let mut grid = fill_grid(mi16);
        // Top-left 16x16 region (cells (0,0), (0,1), (1,0), (1,1))
        // gets a Block16x8 pair.
        grid[0] = Some(mi16x8);
        grid[1] = Some(mi16x8);
        grid[8] = Some(mi16x8);
        grid[9] = Some(mi16x8);
        let levels = flat_levels(12);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // The 16x8 arm: build_masks at (shift_y_16, shift_uv_16) for
        // the top, then build_y_mask at (shift_y_16 + 8, 0) for the
        // bottom (Y-only — chroma stays whole at 4:2:0).
        //
        // Pin lfl_y bit positions: top 16x8 paints cells (0,0)
        // (0,1) (lfl_y[0], [1]) with 12; bottom 16x8 paints (1,0)
        // (1,1) (lfl_y[8], [9]) with 12.
        assert_eq!(got.lfl_y[0], 12, "top-left 16x8 cell (0,0)");
        assert_eq!(got.lfl_y[1], 12, "top-left 16x8 cell (0,1)");
        assert_eq!(got.lfl_y[8], 12, "bot-left 16x8 cell (1,0)");
        assert_eq!(got.lfl_y[9], 12, "bot-left 16x8 cell (1,1)");
    }

    /// 16×8 with the bottom half clipped (`q16_row + 1 >= max_rows`).
    /// Tests the `if q16_row + 1 < max_rows` guard.
    #[test]
    fn setup_mask_16x8_clipped_bottom_skipped() {
        let mi16x8 = intra_leaf(BlockSize::Block16x8, TxSize::Tx8x8);
        let grid = fill_grid(mi16x8);
        let levels = flat_levels(12);

        // max_rows = 1 ⇒ only row 0 in frame. q16_row=0, +1=1 ≥ 1 ⇒
        // bottom-half build_y_mask skipped.
        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 1, 2, &levels, &mut got);

        // Cells (0,0) (0,1) painted; (1,0) (1,1) stay 0.
        assert_eq!(got.lfl_y[0], 12, "top in-frame (0,0)");
        assert_eq!(got.lfl_y[1], 12, "top in-frame (0,1)");
        assert_eq!(got.lfl_y[8], 0, "bot clipped (1,0)");
        assert_eq!(got.lfl_y[9], 0, "bot clipped (1,1)");
    }

    /// 8×16 partition within a 16×16 region: left half (Block8x16) +
    /// right half (Block8x16) via `build_y_mask`. Covers the
    /// `Block8x16` arm of `walk_four_16x16`.
    #[test]
    fn setup_mask_8x16_within_top_left_quadrant() {
        let mi8x16 = intra_leaf(BlockSize::Block8x16, TxSize::Tx8x8);
        let mi16 = intra_leaf(BlockSize::Block16x16, TxSize::Tx8x8);
        let mut grid = fill_grid(mi16);
        grid[0] = Some(mi8x16);
        grid[1] = Some(mi8x16);
        grid[8] = Some(mi8x16);
        grid[9] = Some(mi8x16);
        let levels = flat_levels(12);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Left 8x16 paints cells (0,0) + (1,0); right 8x16 paints
        // (0,1) + (1,1). Net: top-left 2x2 of lfl_y = 12.
        assert_eq!(got.lfl_y[0], 12, "left 8x16 (0,0)");
        assert_eq!(got.lfl_y[8], 12, "left 8x16 (1,0)");
        assert_eq!(got.lfl_y[1], 12, "right 8x16 (0,1)");
        assert_eq!(got.lfl_y[9], 12, "right 8x16 (1,1)");
    }

    /// 8×16 with the right half clipped (`q16_col + 1 >= max_cols`).
    /// Tests the `if q16_col + 1 < max_cols` guard.
    #[test]
    fn setup_mask_8x16_clipped_right_skipped() {
        let mi8x16 = intra_leaf(BlockSize::Block8x16, TxSize::Tx8x8);
        let grid = fill_grid(mi8x16);
        let levels = flat_levels(12);

        // max_cols = 1 ⇒ q16_col=0, +1=1 ≥ 1 ⇒ right half skipped.
        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 2, 1, &levels, &mut got);

        // Cells (0,0) (1,0) painted; (0,1) (1,1) stay 0.
        assert_eq!(got.lfl_y[0], 12, "left in-frame (0,0)");
        assert_eq!(got.lfl_y[8], 12, "left in-frame (1,0)");
        assert_eq!(got.lfl_y[1], 0, "right clipped (0,1)");
        assert_eq!(got.lfl_y[9], 0, "right clipped (1,1)");
    }

    /// Sub-16 partition with four 8×8 leaves (Block8x8 each). Covers
    /// the `walk_four_16x16` `_ =>` arm's `idx_8 in 1..4` loop with
    /// all positions in-frame. (The existing `setup_mask_4x4_leaf_*`
    /// test takes the same code path with Block4x4 leaves; this one
    /// pins Block8x8 leaves to ensure the 8×8 LUT shift bookkeeping
    /// is correct for non-trivial partitions.)
    #[test]
    fn setup_mask_8x8_four_leaves_within_top_left_16x16() {
        let mi8 = intra_leaf(BlockSize::Block8x8, TxSize::Tx8x8);
        let mi16 = intra_leaf(BlockSize::Block16x16, TxSize::Tx8x8);
        let mut grid = fill_grid(mi16);
        // Top-left 16x16 region split into 4 × 8x8.
        grid[0] = Some(mi8);
        grid[1] = Some(mi8);
        grid[8] = Some(mi8);
        grid[9] = Some(mi8);
        let levels = flat_levels(7);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Each 8x8 leaf paints one cell of lfl_y with 7. The 4 cells
        // covering the 16x16 region = positions 0, 1, 8, 9.
        assert_eq!(got.lfl_y[0], 7, "8x8 leaf at SB-cell (0,0)");
        assert_eq!(got.lfl_y[1], 7, "8x8 leaf at SB-cell (0,1)");
        assert_eq!(got.lfl_y[8], 7, "8x8 leaf at SB-cell (1,0)");
        assert_eq!(got.lfl_y[9], 7, "8x8 leaf at SB-cell (1,1)");
    }

    /// Sub-16 / sub-8 with idx_8 clipped off the frame: a 4×4 split
    /// at the very bottom-right of a tiny frame so `q8_row >=
    /// max_rows` triggers the inner-loop continue.
    #[test]
    fn setup_mask_8x8_sub16_clips_idx8_past_frame() {
        // max_rows = 1, max_cols = 2 ⇒ only the (0, 0) and (0, 1)
        // 8x8 cells are in frame. The sub-16 walker tries idx_8 ∈
        // {1, 2, 3}; idx_8 = 1 (q8_row=0, q8_col=1) is in frame and
        // contributes; idx_8 = 2 (q8_row=1, q8_col=0) and idx_8 = 3
        // (q8_row=1, q8_col=1) are past the bottom row and must be
        // skipped by the `if q8_row >= max_rows` guard.
        let mi8 = intra_leaf(BlockSize::Block8x8, TxSize::Tx8x8);
        let grid = fill_grid(mi8);
        let levels = flat_levels(7);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 1, 2, &levels, &mut got);

        // Only cells (0,0) (idx_8=0, build_masks) and (0,1) (idx_8=1,
        // build_y_mask) painted. (1,0) (1,1) stay 0.
        assert_eq!(got.lfl_y[0], 7, "idx_8=0 in-frame (0,0)");
        assert_eq!(got.lfl_y[1], 7, "idx_8=1 in-frame (0,1)");
        assert_eq!(got.lfl_y[8], 0, "idx_8=2 clipped (1,0)");
        assert_eq!(got.lfl_y[9], 0, "idx_8=3 clipped (1,1)");
    }

    /// `get_filter_level_mi` resolution for inter ZeroMv: should pull
    /// from `levels[seg][LastFrame=1][mode_lf_idx=0]` (ZEROMV LUT
    /// slot). Pin by setting a level only at that exact slot and
    /// confirming `setup_mask` paints lfl_y with it.
    #[test]
    fn setup_mask_inter_zeromv_resolves_to_lut_idx_0() {
        // Build a per-slot levels cube where only [0][LAST=1][0] = 25;
        // everything else stays 0. ZeroMv resolves to mode_lf_idx 0;
        // LAST resolves to ref_idx 1. So an inter-ZeroMv-LAST block
        // should produce filter_level 25, painting lfl_y.
        let mut levels = flat_levels(0);
        levels[0][1][0] = 25;

        let mi = inter_leaf(
            BlockSize::Block64x64,
            TxSize::Tx32x32,
            InterMode::ZeroMv,
            RefFrame::LastFrame,
        );
        let grid = fill_grid(mi);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Every cell painted with 25 — every read of the LUT hits
        // [0][1][0].
        assert_eq!(got.lfl_y, [25u8; 64]);
    }

    /// `get_filter_level_mi` resolution for inter NewMv: should pull
    /// from `levels[seg][LastFrame=1][mode_lf_idx=1]`. Setting only
    /// that slot must paint lfl_y; the [seg][LAST][0] slot must NOT
    /// be consulted.
    #[test]
    fn setup_mask_inter_newmv_resolves_to_lut_idx_1() {
        let mut levels = flat_levels(0);
        // Put a poison value at idx 0 to ensure NewMv does NOT read it.
        levels[0][1][0] = 99;
        levels[0][1][1] = 25;

        let mi = inter_leaf(
            BlockSize::Block64x64,
            TxSize::Tx32x32,
            InterMode::NewMv,
            RefFrame::LastFrame,
        );
        let grid = fill_grid(mi);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        // Expect 25 everywhere — NewMv pulled from idx 1, not 0.
        assert_eq!(got.lfl_y, [25u8; 64]);
    }

    /// Sub-16 / sub-8 with idx_8 clipped on the *column* axis:
    /// max_cols = 1 so `q8_col=1` (idx_8=1 or 3) is out of frame and
    /// the `q8_col >= max_cols` guard fires for those iterations.
    /// (The row-clip sibling `setup_mask_8x8_sub16_clips_idx8_past_frame`
    /// covers the q8_row branch.)
    #[test]
    fn setup_mask_8x8_sub16_clips_idx8_past_col_frame() {
        let mi8 = intra_leaf(BlockSize::Block8x8, TxSize::Tx8x8);
        let grid = fill_grid(mi8);
        let levels = flat_levels(7);

        // max_cols = 1 ⇒ only the (0,0) and (1,0) 8x8 cells fit in
        // the column dimension. idx_8 = 1 (q8_col = 1) and 3 (q8_col
        // = 1) trigger the `q8_col >= max_cols` continue.
        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 2, 1, &levels, &mut got);

        // Cells (0,0) painted (idx_8=0 via build_masks), (1,0)
        // painted (idx_8=2 via build_y_mask, q8_row=1<1 fails? no,
        // q8_row=1 < max_rows=2 is true, q8_col=0 < max_cols=1 true);
        // idx_8=1 and idx_8=3 (q8_col=1≥1) skipped.
        assert_eq!(got.lfl_y[0], 7, "idx_8=0 in-frame (0,0)");
        assert_eq!(got.lfl_y[8], 7, "idx_8=2 in-frame (1,0)");
        assert_eq!(got.lfl_y[1], 0, "idx_8=1 col-clipped (0,1)");
        assert_eq!(got.lfl_y[9], 0, "idx_8=3 col-clipped (1,1)");
    }

    /// `get_filter_level_mi` resolution for inter NearestMv with
    /// AltrefFrame ref: pulls from `levels[seg][ALTREF=3][1]`.
    #[test]
    fn setup_mask_inter_nearestmv_altref_resolves_correctly() {
        let mut levels = flat_levels(0);
        levels[0][3][1] = 18;

        let mi = inter_leaf(
            BlockSize::Block64x64,
            TxSize::Tx32x32,
            InterMode::NearestMv,
            RefFrame::AltrefFrame,
        );
        let grid = fill_grid(mi);

        let mut got = LoopFilterMask::ZERO;
        setup_mask(0, 0, &grid, 8, 8, 8, &levels, &mut got);

        assert_eq!(got.lfl_y, [18u8; 64]);
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
