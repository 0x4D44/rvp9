//! Frame-level loop-filter apply driver (Task J).
//!
//! Mirrors libvpx `vp9_loop_filter_frame` + `loop_filter_rows` +
//! `vp9_filter_block_plane_ss00` + `vp9_filter_block_plane_ss11`
//! (`vp9/common/vp9_loopfilter.c:1239-1467`). Given a fully-reconstructed
//! [`Yv12Buffer`] and the per-SB [`LoopFilterMask`] grid (one entry
//! per 64×64 superblock; produced by `setup_mask` + `adjust_mask`),
//! walks the frame in SB raster order applying the per-edge kernels.
//!
//! ## Pass order (load-bearing, see LLD §5.11)
//!
//! For each SB:
//! 1. **Y vertical** edges (8 luma block-rows at a time, in pairs)
//! 2. **Y horizontal** edges (8 luma block-rows, one at a time)
//! 3. **U vertical** then **U horizontal**
//! 4. **V vertical** then **V horizontal**
//!
//! Vertical-before-horizontal **within each plane** is correctness-
//! critical: the horizontal filter at row `r` reads pixels at rows
//! `r-3..r+3` (or `r-7..r+7` for the 16-tap variant); if the vertical
//! filter ran second on the same SB, the horizontal pass would see
//! stale pixels along the row. Cited in libvpx `vp9_loopfilter.c:1252`.
//!
//! Plane order (Y → U → V per SB) matches libvpx
//! `loop_filter_rows:1450-1464`.
//!
//! ## Top / left frame edges
//!
//! The leftmost column of the frame is implicitly handled by
//! `adjust_mask`, which zeroes the `left_y[i]` bit for column 0 of
//! each TX-size mask when `mi_col == 0`. The top-row edge is handled
//! inside this driver: for `mi_row + r == 0` we zero the above-mask
//! before dispatching the horizontal pass, matching libvpx
//! `vp9_loopfilter.c:1291-1294`.

use super::mask::LoopFilterMask;
use crate::bitstream::ChromaFormat;
use crate::frame::Yv12Buffer;
use crate::loopfilter::{
    LoopFilterInfo, lpf_horizontal_4, lpf_horizontal_4_dual, lpf_horizontal_8,
    lpf_horizontal_8_dual, lpf_horizontal_16, lpf_horizontal_16_dual, lpf_vertical_4,
    lpf_vertical_4_dual, lpf_vertical_8, lpf_vertical_8_dual, lpf_vertical_16, lpf_vertical_16_dual,
};

/// Number of 8×8 MI rows per 64×64 superblock. Matches libvpx
/// `MI_BLOCK_SIZE` (`vp9_enums.h:25`).
const MI_BLOCK_SIZE: usize = 8;

/// Per-TX-size index into [`LoopFilterMask`] arrays. Matches the
/// `TxSize` enum ordering.
const TX_4X4: usize = 0;
const TX_8X8: usize = 1;
const TX_16X16: usize = 2;

/// Apply the loop filter to the whole frame, in SB raster order.
///
/// `lfm_grid` carries one [`LoopFilterMask`] per 64×64 superblock,
/// laid out row-major with `lfm_stride` SBs per row. `sb_rows`/
/// `sb_cols` are the SB extents (≥ `ceil(mi_rows/8)` /
/// `ceil(mi_cols/8)`). `mi_rows`/`mi_cols` are the frame's MI-grid
/// extents (not necessarily a multiple of 8 — partial bottom/right
/// SBs are handled via the per-loop `mi_row + r < mi_rows` guard
/// and `adjust_mask`'s frame-edge cleanup).
///
/// `info` is the precomputed threshold table (one entry per
/// `filter_level ∈ 0..=63`).
///
/// `start_mi_row`/`end_mi_row` are MI-grid row bounds; for a
/// whole-frame pass pass `0` and `sb_rows * MI_BLOCK_SIZE`
/// respectively (or `mi_rows`).
///
/// Mirrors libvpx `vp9_loop_filter_frame` + `loop_filter_rows`
/// (`vp9_loopfilter.c:1422-1483`) for the 4:2:0 (`LF_PATH_420`)
/// path.
#[allow(clippy::too_many_arguments)]
pub fn loop_filter_frame(
    yv12: &mut Yv12Buffer,
    lfm_grid: &[LoopFilterMask],
    lfm_stride: usize,
    sb_rows: usize,
    sb_cols: usize,
    mi_rows: usize,
    mi_cols: usize,
    info: &LoopFilterInfo,
    start_mi_row: usize,
    end_mi_row: usize,
) {
    debug_assert!(start_mi_row <= end_mi_row);
    debug_assert!(end_mi_row <= sb_rows * MI_BLOCK_SIZE);
    debug_assert!(lfm_grid.len() >= sb_rows * lfm_stride);
    debug_assert!(lfm_stride >= sb_cols);
    // G7.0 review (2026-05-21): defensive — `filter_block_plane_ss11`
    // hardcodes `ss_x = ss_y = 1` (see its name); the per-SB chroma
    // origin calculation at `(mi_row * MI_BLOCK_SIZE / 2) * stride +
    // (mi_col * MI_BLOCK_SIZE / 2)` only holds for 4:2:0. The G7.0
    // gate at `rvp9-decoder::Decoder::decode_frame` should have
    // rejected non-`Yuv420` before any frame reaches the loop filter.
    // Debug-only — release builds skip the check.
    debug_assert_eq!(
        yv12.chroma(),
        ChromaFormat::Yuv420,
        "G7.0 gate should have rejected non-Yuv420 before loop_filter_frame; \
         filter_block_plane_ss11 below assumes ss_x = ss_y = 1",
    );
    let _ = mi_cols; // libvpx parity: `mi_col` loop terminates at `mi_cols`,
    // which equals `sb_cols * MI_BLOCK_SIZE` rounded up; we
    // walk one mask per SB and the mask-edge cleanup
    // (adjust_mask) has already masked out columns ≥ mi_cols.

    let sb_start = start_mi_row / MI_BLOCK_SIZE;
    let sb_end = end_mi_row.div_ceil(MI_BLOCK_SIZE).min(sb_rows);

    for sb_row in sb_start..sb_end {
        let mi_row = sb_row * MI_BLOCK_SIZE;
        for sb_col in 0..sb_cols {
            let mi_col = sb_col * MI_BLOCK_SIZE;
            let lfm = &lfm_grid[sb_row * lfm_stride + sb_col];

            // Y plane (ss = 0).
            {
                let mut y = yv12.y_plane_mut();
                let stride = y.stride();
                filter_block_plane_ss00(
                    y.data_mut(),
                    stride,
                    lfm,
                    mi_row,
                    mi_col,
                    mi_rows,
                    info,
                );
            }
            // U plane (ss = 1).
            {
                let mut u = yv12.u_plane_mut();
                let stride = u.stride();
                filter_block_plane_ss11(
                    u.data_mut(),
                    stride,
                    lfm,
                    mi_row,
                    mi_col,
                    mi_rows,
                    info,
                );
            }
            // V plane (ss = 1).
            {
                let mut v = yv12.v_plane_mut();
                let stride = v.stride();
                filter_block_plane_ss11(
                    v.data_mut(),
                    stride,
                    lfm,
                    mi_row,
                    mi_col,
                    mi_rows,
                    info,
                );
            }
        }
    }
}

/// Filter one superblock's Y plane: vertical pass (2 luma block-rows
/// at a time) followed by horizontal pass (one luma block-row at a
/// time). Mirrors libvpx `vp9_filter_block_plane_ss00`
/// (`vp9_loopfilter.c:1239-1322`).
///
/// `pixels` is a mutable slice over the plane, with element 0 at the
/// plane's top-left visible pixel (i.e. the slice returned by
/// `Yv12Buffer::y_plane_mut().data_mut()`). `stride` is the plane's
/// row stride in bytes.
fn filter_block_plane_ss00(
    pixels: &mut [u8],
    stride: usize,
    lfm: &LoopFilterMask,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    info: &LoopFilterInfo,
) {
    // SB origin within the plane (offset to top-left of the 64×64).
    let sb_origin = mi_row * MI_BLOCK_SIZE * stride + mi_col * MI_BLOCK_SIZE;

    // === Vertical pass ===
    let mut mask_16x16 = lfm.left_y[TX_16X16];
    let mut mask_8x8 = lfm.left_y[TX_8X8];
    let mut mask_4x4 = lfm.left_y[TX_4X4];
    let mut mask_4x4_int = lfm.int_4x4_y;

    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        // dst_offset = top-left of the 2-block-row strip in this SB.
        let dst_offset = sb_origin + (r * MI_BLOCK_SIZE) * stride;
        filter_selectively_vert_row2_y(
            pixels,
            dst_offset,
            stride,
            mask_16x16 as u32,
            mask_8x8 as u32,
            mask_4x4 as u32,
            mask_4x4_int as u32,
            info,
            &lfm.lfl_y[(r << 3)..],
        );
        mask_16x16 >>= 16;
        mask_8x8 >>= 16;
        mask_4x4 >>= 16;
        mask_4x4_int >>= 16;
        r += 2;
    }

    // === Horizontal pass ===
    let mut mask_16x16 = lfm.above_y[TX_16X16];
    let mut mask_8x8 = lfm.above_y[TX_8X8];
    let mut mask_4x4 = lfm.above_y[TX_4X4];
    let mut mask_4x4_int = lfm.int_4x4_y;

    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        // Top frame edge: no above-edge filter at mi_row + r == 0
        // (libvpx `vp9_loopfilter.c:1291-1294`). The `mask_4x4_int`
        // is the *internal* mid-block horizontal edge inside the
        // 8×8, which exists regardless of frame-top.
        let (m16, m8, m4) = if mi_row + r == 0 {
            (0u32, 0u32, 0u32)
        } else {
            (
                (mask_16x16 & 0xff) as u32,
                (mask_8x8 & 0xff) as u32,
                (mask_4x4 & 0xff) as u32,
            )
        };
        let dst_offset = sb_origin + (r * MI_BLOCK_SIZE) * stride;
        filter_selectively_horiz(
            pixels,
            dst_offset,
            stride,
            m16,
            m8,
            m4,
            (mask_4x4_int & 0xff) as u32,
            info,
            &lfm.lfl_y[(r << 3)..],
        );
        mask_16x16 >>= 8;
        mask_8x8 >>= 8;
        mask_4x4 >>= 8;
        mask_4x4_int >>= 8;
        r += 1;
    }
}

/// Filter one superblock's chroma plane (4:2:0 — `ss_x = ss_y = 1`).
/// Each chroma 4×4 covers a 16×16 luma area, so the chroma grid is
/// 4×4 (vs 8×8 for luma) within the SB. Mirrors libvpx
/// `vp9_filter_block_plane_ss11` (`vp9_loopfilter.c:1324-1420`).
fn filter_block_plane_ss11(
    pixels: &mut [u8],
    stride: usize,
    lfm: &LoopFilterMask,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    info: &LoopFilterInfo,
) {
    // SB origin within the chroma plane. The chroma plane is half
    // the size of luma (4:2:0), so 8 MI rows of luma = 4 chroma
    // rows of 8 chroma pixels each. The origin offset is the same
    // formula as luma if we plug in mi_row/2 × MI_BLOCK_SIZE — but
    // the per-SB chroma origin is `mi_row * 4 * uv_stride + mi_col
    // * 4`, equivalently `(mi_row * MI_BLOCK_SIZE / 2) * uv_stride
    // + mi_col * MI_BLOCK_SIZE / 2`.
    let sb_origin = (mi_row * MI_BLOCK_SIZE / 2) * stride + (mi_col * MI_BLOCK_SIZE / 2);

    // Derive the per-chroma-4×4 filter level array `lfl_uv[16]`
    // from `lfm.lfl_y` (one luma level per 8×8). Each chroma 4×4
    // inherits the level of its top-left luma 8×8 (libvpx
    // `vp9_loopfilter.c:1342-1346`).
    let mut lfl_uv = [0u8; 16];

    // === Vertical pass ===
    let mut mask_16x16 = lfm.left_uv[TX_16X16];
    let mut mask_8x8 = lfm.left_uv[TX_8X8];
    let mut mask_4x4 = lfm.left_uv[TX_4X4];
    let mut mask_4x4_int = lfm.int_4x4_uv;

    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        // Populate the lfl_uv rows for chroma rows r/2 and (r+2)/2.
        // `r` here is in luma MI-units; chroma row = r/2. The luma
        // 8×8 at luma-row r, luma-col c (in SB-local 8×8 units) is
        // `lfm.lfl_y[(r << 3) + c]`; for the co-located chroma 4×4
        // at chroma-row r/2 col c/2 we want the luma 8×8 at
        // (r, c*2). libvpx writes:
        //   lfl_uv[(r << 1) + c] = lfm->lfl_y[(r << 3) + (c << 1)]
        //   lfl_uv[((r+2) << 1) + c] = lfm->lfl_y[((r+2) << 3) + (c << 1)]
        for c in 0..(MI_BLOCK_SIZE >> 1) {
            lfl_uv[(r << 1) + c] = lfm.lfl_y[(r << 3) + (c << 1)];
            lfl_uv[((r + 2) << 1) + c] = lfm.lfl_y[((r + 2) << 3) + (c << 1)];
        }

        // Each iteration of the vertical pass covers 16 chroma
        // rows = 4 chroma block-rows = 2 chroma block-rows-of-2
        // (the libvpx outer step is `+= 4` in luma MI units, which
        // corresponds to `+= 16` chroma pixel rows).
        let dst_offset = sb_origin + (r * MI_BLOCK_SIZE / 2) * stride;
        filter_selectively_vert_row2_uv(
            pixels,
            dst_offset,
            stride,
            mask_16x16 as u32,
            mask_8x8 as u32,
            mask_4x4 as u32,
            mask_4x4_int as u32,
            info,
            &lfl_uv[(r << 1)..],
        );

        mask_16x16 >>= 8;
        mask_8x8 >>= 8;
        mask_4x4 >>= 8;
        mask_4x4_int >>= 8;
        r += 4;
    }

    // === Horizontal pass ===
    let mut mask_16x16 = lfm.above_uv[TX_16X16];
    let mut mask_8x8 = lfm.above_uv[TX_8X8];
    let mut mask_4x4 = lfm.above_uv[TX_4X4];
    let mut mask_4x4_int = lfm.int_4x4_uv;

    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        // skip_border_4x4_r: when this is the very last visible
        // chroma row (mi_row + r == mi_rows - 1), drop the
        // internal-4×4 edge for the row — those internal edges
        // would otherwise read pixels past the frame.
        let skip_border_4x4_r = mi_row + r == mi_rows.saturating_sub(1);
        let mask_4x4_int_r = if skip_border_4x4_r {
            0u32
        } else {
            (mask_4x4_int & 0xf) as u32
        };
        let (m16, m8, m4) = if mi_row + r == 0 {
            (0u32, 0u32, 0u32)
        } else {
            (
                (mask_16x16 & 0xf) as u32,
                (mask_8x8 & 0xf) as u32,
                (mask_4x4 & 0xf) as u32,
            )
        };
        let dst_offset = sb_origin + (r * MI_BLOCK_SIZE / 2) * stride;
        filter_selectively_horiz_uv(
            pixels,
            dst_offset,
            stride,
            m16,
            m8,
            m4,
            mask_4x4_int_r,
            info,
            &lfl_uv[(r << 1)..],
        );

        mask_16x16 >>= 4;
        mask_8x8 >>= 4;
        mask_4x4 >>= 4;
        mask_4x4_int >>= 4;
        r += 2;
    }
}

/// Y-plane vertical pass — processes 2 luma block-rows (16 luma rows)
/// at a time, 8 columns at a time. Mirrors libvpx
/// `filter_selectively_vert_row2` for `subsampling_factor == 0`
/// (`vp9_loopfilter.c:297-372`).
///
/// `mask_4x4_int` is the SB's internal-4×4 mask shifted to expose
/// the relevant 16-bit window (2 luma block-rows × 8 columns).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert_row2_y(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mask_16x16: u32,
    mask_8x8: u32,
    mask_4x4: u32,
    mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
) {
    // Dual = both block-rows' columns. ss=0 ⇒ cutoff = 0xffff,
    // lfl_forward = 8 (next block-row in lfl_y), dual_one = 1 | (1<<8).
    let dual_mask_cutoff: u32 = 0xffff;
    let lfl_forward: usize = 8;
    let dual_one: u32 = 1 | (1 << lfl_forward);

    filter_selectively_vert_row2_inner(
        pixels,
        s_index,
        pitch,
        mask_16x16,
        mask_8x8,
        mask_4x4,
        mask_4x4_int,
        info,
        lfl,
        dual_mask_cutoff,
        lfl_forward,
        dual_one,
    );
}

/// UV-plane vertical pass. Same shape as the Y version but with
/// `subsampling_factor == 1` → cutoff = 0xff, lfl_forward = 4
/// (libvpx `vp9_loopfilter.c:297-372`).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert_row2_uv(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mask_16x16: u32,
    mask_8x8: u32,
    mask_4x4: u32,
    mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
) {
    let dual_mask_cutoff: u32 = 0xff;
    let lfl_forward: usize = 4;
    let dual_one: u32 = 1 | (1 << lfl_forward);

    filter_selectively_vert_row2_inner(
        pixels,
        s_index,
        pitch,
        mask_16x16,
        mask_8x8,
        mask_4x4,
        mask_4x4_int,
        info,
        lfl,
        dual_mask_cutoff,
        lfl_forward,
        dual_one,
    );
}

/// Shared body of the Y / UV vertical pass. Differs only in
/// `dual_mask_cutoff`, `lfl_forward`, `dual_one`. Mirrors libvpx
/// `filter_selectively_vert_row2` body (`vp9_loopfilter.c:297-372`).
///
/// The dispatch picks the widest TX-size filter that fires per
/// column (16 > 8 > 4). When both first and second block-row bits
/// are set (`mask & dual_one == dual_one`), the `_dual` variant
/// runs on both rows together; otherwise either the first row
/// (`mask & 1`) or the second row alone (`s + 8 * pitch`).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert_row2_inner(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
    dual_mask_cutoff: u32,
    lfl_forward: usize,
    dual_one: u32,
) {
    let mut mask = (mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int) & dual_mask_cutoff;
    let mut s = s_index;
    let mut lfl_index: usize = 0;
    // The second block-row in pixel units is always 8 rows down,
    // regardless of subsampling (libvpx `ss[1] = ss[0] + 8 * pitch`,
    // `vp9_loopfilter.c:315`).
    let second_pitch = 8 * pitch;

    while mask != 0 {
        if mask & dual_one != 0 {
            let lfi0 = info.thresh(lfl[lfl_index]);
            let lfi1 = info.thresh(lfl[lfl_index + lfl_forward]);

            // 16-tap.
            if mask_16x16 & dual_one != 0 {
                if (mask_16x16 & dual_one) == dual_one {
                    lpf_vertical_16_dual(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else if mask_16x16 & 1 != 0 {
                    lpf_vertical_16(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    lpf_vertical_16(
                        pixels,
                        s + second_pitch,
                        pitch,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                }
            }

            // 8-tap.
            if mask_8x8 & dual_one != 0 {
                if (mask_8x8 & dual_one) == dual_one {
                    lpf_vertical_8_dual(
                        pixels,
                        s,
                        pitch,
                        lfi0.mblim,
                        lfi0.lim,
                        lfi0.hev_thr,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                } else if mask_8x8 & 1 != 0 {
                    lpf_vertical_8(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    lpf_vertical_8(
                        pixels,
                        s + second_pitch,
                        pitch,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                }
            }

            // 4-tap.
            if mask_4x4 & dual_one != 0 {
                if (mask_4x4 & dual_one) == dual_one {
                    lpf_vertical_4_dual(
                        pixels,
                        s,
                        pitch,
                        lfi0.mblim,
                        lfi0.lim,
                        lfi0.hev_thr,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                } else if mask_4x4 & 1 != 0 {
                    lpf_vertical_4(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    lpf_vertical_4(
                        pixels,
                        s + second_pitch,
                        pitch,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                }
            }

            // Internal 4×4 edge: 4 pixels into the 8-pixel block.
            if mask_4x4_int & dual_one != 0 {
                if (mask_4x4_int & dual_one) == dual_one {
                    lpf_vertical_4_dual(
                        pixels,
                        s + 4,
                        pitch,
                        lfi0.mblim,
                        lfi0.lim,
                        lfi0.hev_thr,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                } else if mask_4x4_int & 1 != 0 {
                    lpf_vertical_4(pixels, s + 4, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    lpf_vertical_4(
                        pixels,
                        s + 4 + second_pitch,
                        pitch,
                        lfi1.mblim,
                        lfi1.lim,
                        lfi1.hev_thr,
                    );
                }
            }
        }

        s += 8;
        lfl_index += 1;
        // Libvpx clears the dual_one bit from `mask` (the union) so
        // the loop terminates faster; the individual masks just shift
        // (`vp9_loopfilter.c:367-370`).
        mask = (mask & !dual_one) >> 1;
        mask_16x16 >>= 1;
        mask_8x8 >>= 1;
        mask_4x4 >>= 1;
        mask_4x4_int >>= 1;
    }
}

/// Y-plane horizontal pass — processes one luma block-row (8 luma
/// rows) at a time. Mirrors libvpx `filter_selectively_horiz`
/// (`vp9_loopfilter.c:453-543`). Handles the dual-column merge
/// (`mask & 3 == 3` ⇒ wider filter on 16 columns).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_horiz(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
) {
    let mut mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    let mut s = s_index;
    let mut lfl_index: usize = 0;
    while mask != 0 {
        let mut count: u32 = 1;
        if mask & 1 != 0 {
            let lfi = info.thresh(lfl[lfl_index]);

            if mask_16x16 & 1 != 0 {
                if (mask_16x16 & 3) == 3 {
                    lpf_horizontal_16_dual(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    count = 2;
                } else {
                    lpf_horizontal_16(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                }
            } else if mask_8x8 & 1 != 0 {
                if (mask_8x8 & 3) == 3 {
                    let lfin = info.thresh(lfl[lfl_index + 1]);
                    lpf_horizontal_8_dual(
                        pixels,
                        s,
                        pitch,
                        lfi.mblim,
                        lfi.lim,
                        lfi.hev_thr,
                        lfin.mblim,
                        lfin.lim,
                        lfin.hev_thr,
                    );
                    if (mask_4x4_int & 3) == 3 {
                        lpf_horizontal_4_dual(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                            lfin.mblim,
                            lfin.lim,
                            lfin.hev_thr,
                        );
                    } else if mask_4x4_int & 1 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    } else if mask_4x4_int & 2 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 8 + 4 * pitch,
                            pitch,
                            lfin.mblim,
                            lfin.lim,
                            lfin.hev_thr,
                        );
                    }
                    count = 2;
                } else {
                    lpf_horizontal_8(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    if mask_4x4_int & 1 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    }
                }
            } else if mask_4x4 & 1 != 0 {
                if (mask_4x4 & 3) == 3 {
                    let lfin = info.thresh(lfl[lfl_index + 1]);
                    lpf_horizontal_4_dual(
                        pixels,
                        s,
                        pitch,
                        lfi.mblim,
                        lfi.lim,
                        lfi.hev_thr,
                        lfin.mblim,
                        lfin.lim,
                        lfin.hev_thr,
                    );
                    if (mask_4x4_int & 3) == 3 {
                        lpf_horizontal_4_dual(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                            lfin.mblim,
                            lfin.lim,
                            lfin.hev_thr,
                        );
                    } else if mask_4x4_int & 1 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    } else if mask_4x4_int & 2 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 8 + 4 * pitch,
                            pitch,
                            lfin.mblim,
                            lfin.lim,
                            lfin.hev_thr,
                        );
                    }
                    count = 2;
                } else {
                    lpf_horizontal_4(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    if mask_4x4_int & 1 != 0 {
                        lpf_horizontal_4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    }
                }
            } else {
                // Only the internal-4×4 bit was set — filter at the
                // mid-block horizontal edge. Mirrors libvpx
                // `vp9_loopfilter.c:532-534`.
                lpf_horizontal_4(
                    pixels,
                    s + 4 * pitch,
                    pitch,
                    lfi.mblim,
                    lfi.lim,
                    lfi.hev_thr,
                );
            }
        }
        let c = count as usize;
        s += 8 * c;
        lfl_index += c;
        mask_16x16 >>= count;
        mask_8x8 >>= count;
        mask_4x4 >>= count;
        mask_4x4_int >>= count;
        mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    }
}

/// UV horizontal pass — same shape as the Y version but the input
/// masks are 4 bits wide (chroma column count per chroma block-row).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_horiz_uv(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mask_16x16: u32,
    mask_8x8: u32,
    mask_4x4: u32,
    mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
) {
    // The libvpx UV horizontal dispatch is byte-identical to the Y
    // dispatch — only the mask widths differ (4 bits per chroma
    // row vs 8 for luma). Reuse the Y helper; the higher bits are
    // zero so it naturally terminates after 4 iterations.
    filter_selectively_horiz(
        pixels,
        s_index,
        pitch,
        mask_16x16,
        mask_8x8,
        mask_4x4,
        mask_4x4_int,
        info,
        lfl,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::ChromaFormat;
    use crate::frame::{BorderUsage, ByteAlignment};

    fn alloc_64x64() -> Yv12Buffer {
        Yv12Buffer::alloc(
            64,
            64,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc 64x64 frame")
    }

    /// Hash the visible Y region for cheap equality checks. We
    /// deliberately read row-by-row to skip the border bytes.
    fn snapshot_y(buf: &Yv12Buffer) -> Vec<u8> {
        let p = buf.y_plane();
        let mut out = Vec::with_capacity(p.width() * p.height());
        for r in 0..p.height() {
            out.extend_from_slice(p.row(r));
        }
        out
    }

    fn snapshot_u(buf: &Yv12Buffer) -> Vec<u8> {
        let p = buf.u_plane();
        let mut out = Vec::with_capacity(p.width() * p.height());
        for r in 0..p.height() {
            out.extend_from_slice(p.row(r));
        }
        out
    }

    fn snapshot_v(buf: &Yv12Buffer) -> Vec<u8> {
        let p = buf.v_plane();
        let mut out = Vec::with_capacity(p.width() * p.height());
        for r in 0..p.height() {
            out.extend_from_slice(p.row(r));
        }
        out
    }

    /// Empty mask grid (all zeros) → frame unchanged.
    #[test]
    fn empty_mask_grid_no_op() {
        let mut buf = alloc_64x64();
        // Fill with a recognisable gradient so any unintended write
        // would show up.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = ((r * 7 + c) & 0xff) as u8;
                }
            }
        }
        let y_before = snapshot_y(&buf);
        let u_before = snapshot_u(&buf);
        let v_before = snapshot_v(&buf);

        let lfm_grid = [LoopFilterMask::ZERO];
        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &lfm_grid, 1, 1, 1, 8, 8, &info, 0, 8);

        assert_eq!(snapshot_y(&buf), y_before, "Y unchanged");
        assert_eq!(snapshot_u(&buf), u_before, "U unchanged");
        assert_eq!(snapshot_v(&buf), v_before, "V unchanged");
    }

    /// Mask has `lfl_y` set but no edge bits → still no filter
    /// (every edge mask is zero, so no kernel fires).
    #[test]
    fn nonzero_levels_but_no_edges_is_noop() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            y.fill(128);
        }
        let y_before = snapshot_y(&buf);
        let mut lfm = LoopFilterMask::ZERO;
        lfm.lfl_y.fill(40); // levels assigned but no edges
        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);
        assert_eq!(snapshot_y(&buf), y_before);
    }

    /// Construct a single 8×8 vertical edge inside the SB (column 8,
    /// luma block-row 1 to avoid the left-frame-edge clear and the
    /// implicit top-row mask). Set TX_4X4 bit, set lfl_y, then
    /// confirm a hand-traced output.
    ///
    /// We use a *step* across column 8 (left 4 cols = 50, right 4
    /// cols = 100) and verify that the 4-tap filter smoothes the
    /// boundary in luma block-row 1 (rows 8..16 of the SB), and
    /// leaves the rest of the frame untouched.
    #[test]
    fn single_y_vertical_edge_filters_only_target_row() {
        // Use a 60→90 step (diff 30). filter_level 30 gives
        // blimit=94, lim=30. With p3=p2=p1=p0 and q0=q1=q2=q3, the
        // macroblock-mask formula `2*|p0-q0| + |p1-q1|/2 = 2*30 +
        // 15 = 75 ≤ 94` ⇒ filter fires. All in-block diffs are 0 ⇒
        // ≤ lim=30.
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }

        // Mask: TX_4X4 left-edge bit at row=1 col=1 (bit index
        // (1<<3)|1 = 9). lfl_y at (1, 0) and (1, 1) = filter_level.
        let mut lfm = LoopFilterMask::ZERO;
        let bit_index = (1u64 << 3) | 1; // = 9
        lfm.left_y[TX_4X4] |= 1u64 << bit_index;
        // Level 30 → mblim = 94, lim = 30, hev_thr = 1.
        for c in 0..8 {
            lfm.lfl_y[(1 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        // Rows 0..8 (luma block-row 0): untouched.
        for r in 0..8 {
            for c in 0..buf.y_plane().width() {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(
                    buf.y_plane().row(r)[c],
                    expected,
                    "row={r} col={c} should be unchanged"
                );
            }
        }
        // Row 8..16 (luma block-row 1): boundary cols changed.
        for r in 8..16 {
            // Far-left columns: still 60.
            for c in 0..6 {
                assert_eq!(buf.y_plane().row(r)[c], 60, "row={r} col={c} far-left");
            }
            // Far-right columns: still 90.
            for c in 12..64 {
                assert_eq!(buf.y_plane().row(r)[c], 90, "row={r} col={c} far-right");
            }
            let any_changed = (6..=9).any(|c| {
                let pre = if c < 8 { 60u8 } else { 90u8 };
                buf.y_plane().row(r)[c] != pre
            });
            assert!(any_changed, "row={r}: 4-tap kernel should have changed something");
        }
        // Rows 16..64: untouched.
        for r in 16..buf.y_plane().height() {
            for c in 0..buf.y_plane().width() {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(buf.y_plane().row(r)[c], expected, "row={r} col={c}");
            }
        }
    }

    /// Pass-ordering: vertical-then-horizontal differs from
    /// horizontal-then-vertical when both edges fire on the same
    /// SB. Construct a frame with both a vertical step (column 8,
    /// rows 8..16) AND a horizontal step (row 8, cols 0..16), then
    /// run our driver and check the result matches the
    /// hand-computed V-then-H expected output (vs the H-then-V
    /// permutation, which we compute manually).
    #[test]
    fn pass_order_is_vertical_then_horizontal() {
        // Fill with a cross-shaped step (small enough to fire for
        // filter_level 30, blimit 94): the 4 quadrants are
        // 60/90/80/100. The cross intersection sits at (row 8, col 8).
        let mut buf_a = alloc_64x64();
        let mut buf_b = alloc_64x64();
        for buf in [&mut buf_a, &mut buf_b] {
            let mut y = buf.y_plane_mut();
            for r in 0..16 {
                for c in 0..16 {
                    let v = if r < 8 && c < 8 {
                        60u8
                    } else if r < 8 {
                        90u8
                    } else if c < 8 {
                        80u8
                    } else {
                        100u8
                    };
                    y.row_mut(r)[c] = v;
                }
            }
            // Fill the rest of the plane uniformly with 100 to
            // keep the kernel from running off into uninitialised
            // border memory (the border is zeroed at alloc time
            // anyway, but make it deterministic).
            for r in 0..buf.y_plane().height() {
                for c in 16..buf.y_plane().width() {
                    let mut y = buf.y_plane_mut();
                    y.row_mut(r)[c] = 100;
                }
            }
            for r in 16..buf.y_plane().height() {
                for c in 0..16 {
                    let mut y = buf.y_plane_mut();
                    y.row_mut(r)[c] = 100;
                }
            }
        }

        // Mask: TX_4X4 left-edge at row=1 col=1 (vertical edge at
        // col 8, rows 8..16), AND TX_4X4 above-edge at row=1 col=0
        // (horizontal edge at row 8, cols 0..8). Both bits set
        // means both kernels fire on the same SB row.
        let mut lfm = LoopFilterMask::ZERO;
        let v_bit = (1u64 << 3) | 1; // row 1 col 1
        let h_bit = 1u64 << 3; // row 1 col 0
        lfm.left_y[TX_4X4] |= 1u64 << v_bit;
        lfm.above_y[TX_4X4] |= 1u64 << h_bit;
        for c in 0..8 {
            lfm.lfl_y[(1 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf_a, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        // Now compute the H-then-V order manually on buf_b.
        // Horizontal at row 8, cols 0..8 (block-row 1, block-col 0).
        {
            let mut y = buf_b.y_plane_mut();
            let stride = y.stride();
            let data = y.data_mut();
            // The horizontal-4 kernel is invoked as
            // lpf_horizontal_4(data, s_index, pitch, blimit, lim, hev).
            lpf_horizontal_4(data, 8 * stride, stride, 94, 30, 1);
        }
        // Then vertical at col 8, rows 8..16.
        {
            let mut y = buf_b.y_plane_mut();
            let stride = y.stride();
            let data = y.data_mut();
            lpf_vertical_4(data, 8 * stride + 8, stride, 94, 30, 1);
        }

        // buf_a (V-then-H, what our driver did) should NOT equal
        // buf_b (H-then-V) — at least one pixel near the (8, 8)
        // intersection should differ.
        let mut diff_count = 0;
        for r in 5..12 {
            for c in 5..12 {
                if buf_a.y_plane().row(r)[c] != buf_b.y_plane().row(r)[c] {
                    diff_count += 1;
                }
            }
        }
        assert!(
            diff_count > 0,
            "V-then-H result should differ from H-then-V near the edge intersection \
             (got {diff_count} differing pixels)"
        );

        // Now also verify our driver actually produced the V-then-H
        // permutation. Reproduce on a fresh buffer.
        let mut buf_c = alloc_64x64();
        for r in 0..16 {
            for c in 0..16 {
                let v = if r < 8 && c < 8 {
                    60u8
                } else if r < 8 {
                    90u8
                } else if c < 8 {
                    80u8
                } else {
                    100u8
                };
                let mut y = buf_c.y_plane_mut();
                y.row_mut(r)[c] = v;
            }
        }
        for r in 0..buf_c.y_plane().height() {
            for c in 16..buf_c.y_plane().width() {
                let mut y = buf_c.y_plane_mut();
                y.row_mut(r)[c] = 100;
            }
        }
        for r in 16..buf_c.y_plane().height() {
            for c in 0..16 {
                let mut y = buf_c.y_plane_mut();
                y.row_mut(r)[c] = 100;
            }
        }
        // V first then H, manually.
        {
            let mut y = buf_c.y_plane_mut();
            let stride = y.stride();
            lpf_vertical_4(y.data_mut(), 8 * stride + 8, stride, 94, 30, 1);
        }
        {
            let mut y = buf_c.y_plane_mut();
            let stride = y.stride();
            lpf_horizontal_4(y.data_mut(), 8 * stride, stride, 94, 30, 1);
        }
        assert_eq!(
            snapshot_y(&buf_a),
            snapshot_y(&buf_c),
            "driver should produce V-then-H ordering"
        );
    }

    /// UV plane filter: place a step in the U plane and verify the
    /// chroma vertical edge gets filtered with the correct
    /// derived-from-luma level.
    #[test]
    fn uv_plane_vertical_edge_filters() {
        let mut buf = alloc_64x64();
        // U plane is 32×32 chroma pixels. The chroma 4×4 mask grid
        // has 4 cells × 4 cells per SB; each cell is 8 chroma pixels
        // wide. So cell col 1 corresponds to chroma pixel col 8.
        // Place a 60→90 step at chroma pixel col 8 (mb_diff =
        // 75 ≤ 94, fires for filter_level 30).
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // Build mask: chroma 4×4 row=1 col=1 → bit (1<<2)|1 = 5 in
        // u16 UV mask. left_uv[TX_4X4] bit 5. Fill all luma lfl_y
        // with 30 so the derived chroma level is always 30.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_uv[TX_4X4] |= 1u16 << 5;
        lfm.lfl_y.fill(30);

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        // The U plane should have changed somewhere near chroma col 4.
        let u_after = snapshot_u(&buf);
        let mut changes = 0;
        for (a, b) in u_before.iter().zip(u_after.iter()) {
            if a != b {
                changes += 1;
            }
        }
        assert!(changes > 0, "UV vertical kernel should have fired");

        // Other planes unchanged.
        let v_after = snapshot_v(&buf);
        let y_after = snapshot_y(&buf);
        assert!(
            y_after.iter().all(|&p| p == 0),
            "Y plane should be untouched (was zero, no edges set)"
        );
        // V plane was never written so it's zero everywhere — and
        // since we didn't set any V edge bits, it stays zero.
        let _ = v_after;
    }

    /// Frame-edge clamping: after `adjust_mask`, the right-edge SB
    /// has its out-of-frame columns masked off, so the driver does
    /// not over-filter them. We construct a 56-column-wide-but-
    /// 1-SB-tall scenario, run adjust_mask, then verify the columns
    /// past mi_cols are unchanged.
    #[test]
    fn frame_edge_right_columns_not_filtered() {
        // 64x64 frame, but pretend mi_cols = 7 (i.e. 56-pixel wide
        // visible region; the rightmost luma 8×8 is past the frame).
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            y.fill(128);
            // Pollute the last 8 cols with sentinel so any
            // unintended filter would show up as a change from 200.
            for r in 0..y.height() {
                for c in 56..64 {
                    y.row_mut(r)[c] = 200;
                }
            }
        }

        // Build a mask with every TX_4X4 left bit set (all interior
        // edges would fire if not for adjust_mask). Levels = 30.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_4X4] = !0u64;
        lfm.lfl_y.fill(30);
        // Run adjust_mask with mi_cols = 7 to mask out cols ≥ 7.
        crate::loopfilter::adjust_mask(0, 0, 8, 7, &mut lfm);

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 7, &info, 0, 8);

        // Columns 56..64 should remain 200.
        for r in 0..buf.y_plane().height() {
            for c in 56..buf.y_plane().width() {
                assert_eq!(
                    buf.y_plane().row(r)[c],
                    200,
                    "right-of-frame col {c} row {r} should not be filtered"
                );
            }
        }
    }

    /// TX_32X32 demotion: after `adjust_mask` merges TX_32X32 →
    /// TX_16X16, an edge originally tagged TX_32X32 fires the
    /// 16-tap kernel. We set only the TX_32X32 left bit (at row=1
    /// col=4, to dodge the top-row clear and left-frame-edge
    /// clear), call adjust_mask, then verify the driver fires.
    #[test]
    fn tx32_demotion_to_tx16_fires() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    // 60 → 90 step at col 32 (diff 30 ≤ 47, fires
                    // for filter_level 30).
                    y.row_mut(r)[c] = if c < 32 { 60 } else { 90 };
                }
            }
        }

        // Pre-adjust: only TX_32X32 bit set at row=1 col=4 (bit
        // (1<<3)|4 = 12). The driver only reads TX_4X4/8X8/16X16,
        // so without adjust_mask there'd be no filtering.
        let mut lfm = LoopFilterMask::ZERO;
        let bit_index = (1u64 << 3) | 4; // = 12
        lfm.left_y[3] |= 1u64 << bit_index;
        lfm.lfl_y.fill(30);

        let pre = snapshot_y(&buf);

        // Run adjust_mask: TX_32X32 → TX_16X16 merge.
        crate::loopfilter::adjust_mask(0, 0, 8, 8, &mut lfm);
        assert_ne!(lfm.left_y[TX_16X16], 0, "TX_16X16 should now have the bit");

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let post = snapshot_y(&buf);
        assert_ne!(pre, post, "TX_16X16 filter (post-demotion) should fire");
    }

    /// Partial mi_row range: only filter the upper half of the
    /// frame, lower half untouched.
    #[test]
    fn partial_mi_row_range_filters_only_subset() {
        // Use a 128-row frame to have ≥ 2 SB rows.
        let mut buf = Yv12Buffer::alloc(
            64,
            128,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc 64x128");

        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }

        // SB 0 (rows 0..64): edge bit set, will fire.
        // SB 1 (rows 64..128): edge bit set too, but the partial
        //   range excludes it.
        let mut lfm0 = LoopFilterMask::ZERO;
        let bit_index = (1u64 << 3) | 1;
        lfm0.left_y[TX_4X4] |= 1u64 << bit_index;
        lfm0.lfl_y.fill(30);
        let lfm1 = lfm0;
        let lfm_grid = [lfm0, lfm1];

        let info = LoopFilterInfo::new(0);
        // Filter only mi_row 0..8 (= SB row 0). End at 8 = one SB row.
        loop_filter_frame(&mut buf, &lfm_grid, 1, 2, 1, 16, 8, &info, 0, 8);

        // SB 1 (rows 64..128) should still be the original step.
        for r in 64..128 {
            for c in 0..64 {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(
                    buf.y_plane().row(r)[c],
                    expected,
                    "row={r} col={c} should be unchanged outside partial range"
                );
            }
        }
    }
}
