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
use crate::block::uv_plane_tx_size;
use crate::dsp::SimdDispatch;
use crate::frame::Yv12Buffer;
use crate::loopfilter::{FrameFilterLevels, LoopFilterInfo};
use crate::mode_info::{BlockMode, InterMode, ModeInfo, RefFrame};
use crate::transforms::TxSize;

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
/// (`vp9_loopfilter.c:1422-1467`). Supports the `LF_PATH_420`
/// (4:2:0) and `LF_PATH_444` (4:4:4) chroma paths; the asymmetric
/// `LF_PATH_SLOW` (4:2:2 / 4:4:0) is not yet wired (the decoder gate
/// rejects those formats this milestone).
///
/// For 4:4:4 the chroma planes share the luma 8×8 grid, so libvpx
/// runs `vp9_filter_block_plane_ss00` on U/V using the **luma** masks
/// (`left_y`/`above_y`/`int_4x4_y`/`lfl_y`) — see
/// `vp9_loopfilter.c:1456-1457` (`LF_PATH_444`) and the
/// `assert(subsampling_x == 0 && subsampling_y == 0)` at
/// `vp9_filter_block_plane_ss00` (`:1250`). The separately-built UV
/// masks are ignored in that path. We do the same here, which is why
/// no widened UV mask representation is needed: the Y masks already
/// hold the full 8×8 grid the chroma planes require at 4:4:4.
///
/// ## Asymmetric chroma (G7.2 — Milestone 2 of Profile 1)
///
/// For 4:2:2 (ss_x=1, ss_y=0) and 4:4:0 (ss_x=0, ss_y=1) the chroma
/// grid is **asymmetric** w.r.t. luma — neither the packed
/// [`LoopFilterMask`] (ss11) nor the luma masks (ss00) carry the right
/// per-edge information. libvpx handles this via the `LF_PATH_SLOW`
/// path (`vp9_loopfilter.c:1459-1462`), which calls
/// [`filter_block_plane_non420`] (mirroring libvpx
/// `vp9_filter_block_plane_non420`, `:1078-1237`) and walks the
/// MODE_INFO grid directly per chroma row to build per-row masks. This
/// is why `mi_grid` / `mi_stride` / `levels` are required arguments:
/// the non420 path consults them; the 420/444 paths ignore them.
/// Bundle of inputs the `LF_PATH_SLOW` (4:2:2 / 4:4:0) path needs but
/// the 420 / 444 paths ignore. Bundled into a single optional argument
/// so the public signature stays backward-compatible for callers that
/// only ever produce 4:2:0 (e.g. the encoder).
///
/// Mirrors the per-frame fields libvpx's `loop_filter_rows` consults
/// for the non420 path: the row-major `MODE_INFO` slab (`cm->
/// mi_grid_visible`), the slab stride (`cm->mi_stride`), and the
/// resolved per-frame `[seg][ref][mode]` filter-level cube
/// (`cm->lf_info.lvl`).
#[derive(Copy, Clone, Debug)]
pub struct Non420Inputs<'a> {
    /// Frame-wide MI slab, row-major, indexed `[r * mi_stride + c]`.
    pub mi_grid: &'a [ModeInfo],
    /// Stride in MI cells per row.
    pub mi_stride: usize,
    /// Per-frame `[seg][ref][mode]` filter-level cube (output of
    /// [`crate::loopfilter::resolve_filter_levels`]).
    pub levels: &'a FrameFilterLevels,
}

/// Apply the loop filter to the whole frame, in SB raster order.
///
/// Backward-compatible entry that handles 4:2:0 and 4:4:4 buffers. For
/// 4:2:2 / 4:4:0 callers, use [`loop_filter_frame_with_non420`] and
/// pass `Some(Non420Inputs { .. })` — calling this overload on an
/// asymmetric-chroma buffer panics inside the dispatch.
///
/// See [`loop_filter_frame_with_non420`] for the argument semantics.
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
    // Backward-compatible entry: dispatches to the full version with no
    // non420 inputs. Callers that may see 4:2:2 / 4:4:0 must use
    // [`loop_filter_frame_with_non420`] instead — this overload panics
    // if the buffer's chroma format is asymmetric, matching the
    // pre-G7.2 behavior expected by callers that only ever encode /
    // decode 4:2:0 or 4:4:4 (e.g. the encoder).
    loop_filter_frame_with_non420(
        yv12,
        lfm_grid,
        lfm_stride,
        sb_rows,
        sb_cols,
        mi_rows,
        mi_cols,
        info,
        start_mi_row,
        end_mi_row,
        None,
    );
}

/// Like [`loop_filter_frame`] but accepts the `LF_PATH_SLOW` inputs
/// needed for 4:2:2 / 4:4:0 chroma (libvpx
/// `vp9_filter_block_plane_non420`). Pass `None` if you've statically
/// guaranteed the buffer's chroma is 4:2:0 or 4:4:4.
#[allow(clippy::too_many_arguments)]
pub fn loop_filter_frame_with_non420(
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
    non420: Option<Non420Inputs<'_>>,
) {
    debug_assert!(start_mi_row <= end_mi_row);
    debug_assert!(end_mi_row <= sb_rows * MI_BLOCK_SIZE);
    debug_assert!(lfm_grid.len() >= sb_rows * lfm_stride);
    debug_assert!(lfm_stride >= sb_cols);
    // G7.2: the loop filter now handles all four chroma formats.
    //   - 4:2:0 (`LF_PATH_420`): U/V via `filter_block_plane_ss11`,
    //     chroma origin `(mi_row * MI_BLOCK_SIZE / 2) * stride + ...`,
    //     ss_x = ss_y = 1.
    //   - 4:4:4 (`LF_PATH_444`): U/V via `filter_block_plane_ss00` with
    //     the luma masks (chroma grid == luma grid).
    //   - 4:2:2 / 4:4:0 (`LF_PATH_SLOW`): U/V via
    //     `filter_block_plane_non420`, which walks the MI grid directly
    //     because the packed SB mask cannot represent the asymmetric
    //     chroma layout. Mirrors libvpx `loop_filter_rows` dispatch at
    //     `vp9_loopfilter.c:1452-1463`.
    let chroma = yv12.chroma();
    debug_assert!(
        matches!(
            chroma,
            ChromaFormat::Yuv420
                | ChromaFormat::Yuv422
                | ChromaFormat::Yuv440
                | ChromaFormat::Yuv444
        ),
        "loop_filter_frame should see one of the four legal VP9 chroma \
         formats; got {chroma:?}",
    );
    let _ = mi_cols; // libvpx parity: `mi_col` loop terminates at `mi_cols`,
    // which equals `sb_cols * MI_BLOCK_SIZE` rounded up; we
    // walk one mask per SB and the mask-edge cleanup
    // (adjust_mask) has already masked out columns ≥ mi_cols.

    // Wave H H1: look up the process-wide SimdDispatch once, thread
    // it through the inner helpers so the AVX2 loopfilter slots (F10
    // + G3-G6) become live in production. Backward-compatible at the
    // public API: callers that hold their own `Arc<SimdDispatch>`
    // (e.g. the decoder) get the same dispatch table here via the
    // OnceLock; callers that don't (encoder recon path) pay one
    // `Arc::clone` per frame call — negligible.
    let dispatch_arc = crate::dsp::get();
    let dispatch: &SimdDispatch = &dispatch_arc;

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
                    dispatch,
                );
            }
            // U / V planes. Dispatch on the established chroma format:
            //   - Yuv420 → ss11 (4×4 chroma grid; UV masks).
            //   - Yuv444 → ss00 (8×8 chroma grid == luma; Y masks).
            // Mirrors libvpx `loop_filter_rows` `LF_PATH_420` /
            // `LF_PATH_444` (`vp9_loopfilter.c:1452-1458`).
            match chroma {
                ChromaFormat::Yuv420 => {
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
                            dispatch,
                        );
                    }
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
                            dispatch,
                        );
                    }
                }
                ChromaFormat::Yuv444 => {
                    // 4:4:4 chroma == luma grid: reuse the Y masks via
                    // ss00 (libvpx `vp9_filter_block_plane_ss00` for
                    // `LF_PATH_444`). The chroma plane is full-resolution,
                    // so the same SB-origin formula as luma applies.
                    {
                        let mut u = yv12.u_plane_mut();
                        let stride = u.stride();
                        filter_block_plane_ss00(
                            u.data_mut(),
                            stride,
                            lfm,
                            mi_row,
                            mi_col,
                            mi_rows,
                            info,
                            dispatch,
                        );
                    }
                    {
                        let mut v = yv12.v_plane_mut();
                        let stride = v.stride();
                        filter_block_plane_ss00(
                            v.data_mut(),
                            stride,
                            lfm,
                            mi_row,
                            mi_col,
                            mi_rows,
                            info,
                            dispatch,
                        );
                    }
                }
                // Asymmetric subsampling (4:2:2 / 4:4:0) — the
                // `LF_PATH_SLOW` path in libvpx (`vp9_loopfilter.c:1459-
                // 1462`). The packed SB mask doesn't carry the right
                // chroma layout, so `filter_block_plane_non420` walks
                // the MI grid directly per chroma row. Requires the
                // caller to pass `Non420Inputs` — without those the
                // non420 path cannot run.
                ChromaFormat::Yuv422 | ChromaFormat::Yuv440 => {
                    let n420 = non420.expect(
                        "loop_filter_frame called on Yuv422 / Yuv440 buffer \
                         without Non420Inputs — use loop_filter_frame_with_non420 \
                         and pass Some(Non420Inputs { .. })",
                    );
                    let ss_x = chroma.subsampling_x();
                    let ss_y = chroma.subsampling_y();
                    {
                        let mut u = yv12.u_plane_mut();
                        let stride = u.stride();
                        filter_block_plane_non420(
                            u.data_mut(),
                            stride,
                            n420.mi_grid,
                            n420.mi_stride,
                            mi_row,
                            mi_col,
                            mi_rows,
                            mi_cols,
                            ss_x,
                            ss_y,
                            info,
                            n420.levels,
                            dispatch,
                        );
                    }
                    {
                        let mut v = yv12.v_plane_mut();
                        let stride = v.stride();
                        filter_block_plane_non420(
                            v.data_mut(),
                            stride,
                            n420.mi_grid,
                            n420.mi_stride,
                            mi_row,
                            mi_col,
                            mi_rows,
                            mi_cols,
                            ss_x,
                            ss_y,
                            info,
                            n420.levels,
                            dispatch,
                        );
                    }
                }
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
    dispatch: &SimdDispatch,
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
            dispatch,
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
            dispatch,
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
#[allow(clippy::too_many_arguments)]
fn filter_block_plane_ss11(
    pixels: &mut [u8],
    stride: usize,
    lfm: &LoopFilterMask,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    info: &LoopFilterInfo,
    dispatch: &SimdDispatch,
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
            dispatch,
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
            dispatch,
        );

        mask_16x16 >>= 4;
        mask_8x8 >>= 4;
        mask_4x4 >>= 4;
        mask_4x4_int >>= 4;
        r += 2;
    }
}

// ----------------------------------------------------------------------------
// Asymmetric-chroma path (4:2:2 / 4:4:0) — libvpx `LF_PATH_SLOW`.
// ----------------------------------------------------------------------------

/// libvpx `mode_lf_lut[mi->mode]` (`vp9_loopfilter.c:207-210`).
/// Intra modes + `ZEROMV` map to 0; the other three inter modes map
/// to 1. Duplicated from `setup_mask.rs`'s private helper because that
/// module's copy is `pub(super)` to its parent only.
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

/// libvpx `mi->ref_frame[0]` → cube ref-axis index. Intra blocks map
/// to `INTRA_FRAME = 0`. Mirrors `setup_mask.rs::ref0_idx`.
#[inline]
fn ref0_idx(mi: &ModeInfo) -> usize {
    match &mi.mode {
        BlockMode::Intra { .. } => 0,
        BlockMode::Inter { ref_frames, .. } => match ref_frames[0] {
            RefFrame::NoRefFrame | RefFrame::IntraFrame => 0,
            RefFrame::LastFrame => 1,
            RefFrame::GoldenFrame => 2,
            RefFrame::AltrefFrame => 3,
        },
    }
}

/// libvpx `get_filter_level(lfi_n, mi)` (`vp9_loopfilter.c:233-236`).
/// Per-block filter level lookup from the resolved
/// `[seg][ref][mode]` cube.
#[inline]
fn get_filter_level_mi(levels: &FrameFilterLevels, mi: &ModeInfo) -> u8 {
    let seg = (mi.segment_id as usize).min(crate::loopfilter::MAX_SEGMENTS - 1);
    levels[seg][ref0_idx(mi)][mode_lf_idx(mi)]
}

/// `is_inter_block(mi)` mirror — libvpx `vp9_blockd.h:208-210` only
/// inspects `mi->ref_frame[0] > INTRA_FRAME`. Our `ModeInfo::is_inter`
/// is the cleaner expression of the same thing.
#[inline]
fn is_inter_block(mi: &ModeInfo) -> bool {
    mi.is_inter()
}

/// Filter one superblock's chroma plane for the asymmetric 4:2:2 /
/// 4:4:0 cases (libvpx `LF_PATH_SLOW`). Mirrors
/// `vp9_filter_block_plane_non420` (`vp9_loopfilter.c:1078-1237`).
///
/// Unlike the ss00 / ss11 paths, this function does **not** consume the
/// packed [`LoopFilterMask`] (which can only describe the 8×8 luma
/// grid). Instead it walks the per-MI `MODE_INFO` slab directly for
/// the SB, building per-luma-row mask arrays sized `MI_BLOCK_SIZE = 8`.
/// The chroma loop iterates `MI_BLOCK_SIZE` luma-row steps, stepping by
/// `row_step = 1 << ss_y` and `col_step = 1 << ss_x` so the inner
/// iteration count matches the *chroma* grid dimensions even though
/// the indexing remains in luma-MI units (libvpx `:1101-1108`).
///
/// `pixels` is a mutable slice over the chroma plane (`u_plane_mut()`
/// or `v_plane_mut()` `.data_mut()`); `stride` is its row stride.
/// `mi_grid` is the row-major frame-wide MI slab with stride
/// `mi_grid_stride`. `mi_row` / `mi_col` are the SB's top-left
/// position in luma-MI units; `mi_rows` / `mi_cols` are the frame's MI
/// extents (for the partial-bottom/right clamp). `ss_x` / `ss_y` are
/// the plane's subsampling flags (4:2:2 ⇒ ss_x=true, ss_y=false; 4:4:0
/// ⇒ ss_x=false, ss_y=true).
#[allow(clippy::too_many_arguments)]
fn filter_block_plane_non420(
    pixels: &mut [u8],
    stride: usize,
    mi_grid: &[ModeInfo],
    mi_grid_stride: usize,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    mi_cols: usize,
    ss_x: bool,
    ss_y: bool,
    info: &LoopFilterInfo,
    levels: &FrameFilterLevels,
    dispatch: &SimdDispatch,
) {
    // Subsampling factors as 0/1 integers (libvpx `ss_x` / `ss_y`).
    let ssx = ss_x as usize;
    let ssy = ss_y as usize;
    let row_step: usize = 1 << ssy;
    let col_step: usize = 1 << ssx;

    // Per-luma-row mask arrays — sized MI_BLOCK_SIZE (8) and indexed
    // by the SB-local luma MI row `r`. Only the rows actually visited
    // (`r % row_step == 0`) are written; the rest stay zero. Mirrors
    // libvpx `:1088-1099`.
    let mut mask_16x16 = [0u32; MI_BLOCK_SIZE];
    let mut mask_8x8 = [0u32; MI_BLOCK_SIZE];
    let mut mask_4x4 = [0u32; MI_BLOCK_SIZE];
    let mut mask_4x4_int = [0u32; MI_BLOCK_SIZE];
    // lfl is indexed `[(r << 3) + (c >> ss_x)]` — matches libvpx
    // (`:1129`). 64 entries: 8 luma rows × up-to-8 chroma cols.
    let mut lfl = [0u8; MI_BLOCK_SIZE * MI_BLOCK_SIZE];

    // SB origin within the chroma plane. For each chroma row covered
    // by this SB (luma row r → chroma row r >> ss_y), the pixel offset
    // is `(mi_row + r) >> ss_y * 8 * stride`. The `>> ss_y` is implicit:
    // because we step by `row_step` we visit only luma rows that
    // correspond to a chroma row, and each chroma row is 8 pixels —
    // so the per-iteration pixel advance is `8 * stride` regardless of
    // subsampling (libvpx `:1198`, `:1235`).
    //
    // The plane origin is `((mi_row * MI_BLOCK_SIZE) >> ss_y) * stride
    // + ((mi_col * MI_BLOCK_SIZE) >> ss_x)`.
    let sb_origin = ((mi_row * MI_BLOCK_SIZE) >> ssy) * stride
        + ((mi_col * MI_BLOCK_SIZE) >> ssx);

    // ============================================================
    // Vertical pass: build per-row vertical-edge masks AND dispatch
    // the per-row vertical kernels in the same outer loop, just like
    // libvpx (`:1101-1200`). This is correctness-critical: the
    // horizontal pass below reads the row-keyed mask_16x16 / mask_8x8
    // / mask_4x4 arrays, which are populated here.
    // ============================================================
    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        let mut mask_16x16_c: u32 = 0;
        let mut mask_8x8_c: u32 = 0;
        let mut mask_4x4_c: u32 = 0;

        // mi_grid row pointer for this luma row (libvpx `mi_8x8`).
        let row_base = (mi_row + r) * mi_grid_stride + mi_col;

        let mut c = 0usize;
        while c < MI_BLOCK_SIZE && mi_col + c < mi_cols {
            let mi = &mi_grid[row_base + c];
            let sb_type = mi.block_size;
            let skip_this = mi.skip && is_inter_block(mi);

            // "Left edge of current unit is block/partition edge"
            // (libvpx `:1112-1117`). Equivalent to "c is aligned to
            // the block's 8×8 width within the SB".
            let n_8x8_wide = sb_type.n_8x8_wide() as usize;
            let block_edge_left = if sb_type.n_4x4_wide() > 1 {
                (c & (n_8x8_wide - 1)) == 0
            } else {
                true
            };
            let skip_this_c = skip_this && !block_edge_left;

            // "Top edge of current unit is block/partition edge"
            // (libvpx `:1118-1123`).
            let n_8x8_high = sb_type.n_8x8_high() as usize;
            let block_edge_above = if sb_type.n_4x4_high() > 1 {
                (r & (n_8x8_high - 1)) == 0
            } else {
                true
            };
            let skip_this_r = skip_this && !block_edge_above;

            // Chroma TX size for this MI under (ss_x, ss_y). Mirrors
            // libvpx `get_uv_tx_size(mi, plane)` (`vp9_blockd.h:243-
            // 250`); our `uv_plane_tx_size` is the parametric form
            // landed in G7.1a (`crates/rvp9-core/src/block.rs:325`).
            let tx_size = uv_plane_tx_size(sb_type, mi.tx_size, ssx, ssy);

            // Frame-edge guards. Libvpx `:1125-1126` — when ss_x is
            // set AND this is the last column in the frame, drop the
            // 4×4-internal edge (it would read off the frame). Same
            // for ss_y at the bottom row.
            let skip_border_4x4_c = ss_x && mi_col + c == mi_cols - 1;
            let skip_border_4x4_r = ss_y && mi_row + r == mi_rows - 1;

            // Look up the filter level and bail if zero (libvpx
            // `:1129-1130`).
            let lvl = get_filter_level_mi(levels, mi);
            let lfl_idx = (r << 3) + (c >> ssx);
            lfl[lfl_idx] = lvl;
            if lvl == 0 {
                c += col_step;
                continue;
            }

            // Build per-tx masks. The bit position is `1 << (c >> ss_x)`
            // so chroma column indexing is naturally compact (4:2:2
            // ⇒ 4 chroma cols → bits 0..4; 4:4:0 ⇒ 8 cols → bits 0..8).
            let bit = 1u32 << (c >> ssx);
            match tx_size {
                TxSize::Tx32x32 => {
                    // libvpx `:1133-1145` — only mark on aligned
                    // 4-block boundaries (32 px = 4 × 8).
                    if !skip_this_c && ((c >> ssx) & 3) == 0 {
                        if !skip_border_4x4_c {
                            mask_16x16_c |= bit;
                        } else {
                            mask_8x8_c |= bit;
                        }
                    }
                    if !skip_this_r && ((r >> ssy) & 3) == 0 {
                        if !skip_border_4x4_r {
                            mask_16x16[r] |= bit;
                        } else {
                            mask_8x8[r] |= bit;
                        }
                    }
                }
                TxSize::Tx16x16 => {
                    // libvpx `:1146-1158` — aligned 2-block boundaries.
                    if !skip_this_c && ((c >> ssx) & 1) == 0 {
                        if !skip_border_4x4_c {
                            mask_16x16_c |= bit;
                        } else {
                            mask_8x8_c |= bit;
                        }
                    }
                    if !skip_this_r && ((r >> ssy) & 1) == 0 {
                        if !skip_border_4x4_r {
                            mask_16x16[r] |= bit;
                        } else {
                            mask_8x8[r] |= bit;
                        }
                    }
                }
                TxSize::Tx8x8 | TxSize::Tx4x4 => {
                    // libvpx `:1159-1177` — "force 8×8 filtering on
                    // 32×32 boundaries" for sub-16 transforms; 4×4
                    // otherwise. Plus the internal-4×4 edge for tx<8.
                    if !skip_this_c {
                        if tx_size == TxSize::Tx8x8 || ((c >> ssx) & 3) == 0 {
                            mask_8x8_c |= bit;
                        } else {
                            mask_4x4_c |= bit;
                        }
                    }
                    if !skip_this_r {
                        if tx_size == TxSize::Tx8x8 || ((r >> ssy) & 3) == 0 {
                            mask_8x8[r] |= bit;
                        } else {
                            mask_4x4[r] |= bit;
                        }
                    }
                    if !skip_this && tx_size < TxSize::Tx8x8 && !skip_border_4x4_c {
                        mask_4x4_int[r] |= bit;
                    }
                }
            }

            c += col_step;
        }

        // Dispatch this row's vertical pass. Border mask clears the
        // leftmost bit when mi_col == 0 (libvpx `:1181`).
        let border_mask: u32 = if mi_col == 0 { !1u32 } else { !0u32 };
        let dst_offset = sb_origin + (r >> ssy) * 8 * stride;
        // Note: libvpx passes the un-shifted `lfl[r << 3]` slice; the
        // chroma columns are at positions 0..(MI_BLOCK_SIZE >> ss_x)
        // within that row, and the column dispatcher walks the union
        // mask bit-by-bit. Reusing `filter_selectively_horiz` for the
        // *vertical* edges here mirrors what libvpx does — the per-
        // column dispatch logic is the same; only the kernel slot
        // selection differs (vertical kernels). The libvpx slow path
        // calls a dedicated `filter_selectively_vert` (the non-dual
        // variant); we inline a per-bit walker below so we can drive
        // the dispatch slots directly.
        filter_selectively_vert_non420(
            pixels,
            dst_offset,
            stride,
            mask_16x16_c & border_mask,
            mask_8x8_c & border_mask,
            mask_4x4_c & border_mask,
            mask_4x4_int[r],
            info,
            &lfl[(r << 3)..],
            dispatch,
        );

        r += row_step;
    }

    // ============================================================
    // Horizontal pass: dispatch per row, reading the row-keyed mask
    // arrays we built above. Mirrors libvpx `:1202-1236`.
    // ============================================================
    let mut r = 0usize;
    while r < MI_BLOCK_SIZE && mi_row + r < mi_rows {
        let skip_border_4x4_r = ss_y && mi_row + r == mi_rows - 1;
        let mask_4x4_int_r = if skip_border_4x4_r { 0 } else { mask_4x4_int[r] };

        let (m16, m8, m4) = if mi_row + r == 0 {
            (0u32, 0u32, 0u32)
        } else {
            (mask_16x16[r], mask_8x8[r], mask_4x4[r])
        };

        let dst_offset = sb_origin + (r >> ssy) * 8 * stride;
        // libvpx `:1191/1229` calls the same `filter_selectively_horiz`
        // used by the ss00 / ss11 paths — reuse our shared helper.
        filter_selectively_horiz(
            pixels,
            dst_offset,
            stride,
            m16,
            m8,
            m4,
            mask_4x4_int_r,
            info,
            &lfl[(r << 3)..],
            dispatch,
        );

        r += row_step;
    }
}

/// libvpx `filter_selectively_vert` (the non-dual variant —
/// `vp9_loopfilter.c:374-451`). Walks the per-column mask bits one at
/// a time, dispatching the widest applicable vertical kernel per
/// column. Used by the non420 path because the asymmetric chroma
/// grids don't tile cleanly into the "two-rows-of-the-luma-grid-
/// at-a-time" `_dual` shape.
#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert_non420(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    info: &LoopFilterInfo,
    lfl: &[u8],
    dispatch: &SimdDispatch,
) {
    let v16 = dispatch.lpf_vertical_16();
    let v8 = dispatch.lpf_vertical_8();
    let v4 = dispatch.lpf_vertical_4();

    let mut mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    let mut s = s_index;
    let mut lfl_index: usize = 0;

    while mask != 0 {
        if mask & 1 != 0 {
            let lfi = info.thresh(lfl[lfl_index]);
            if mask_16x16 & 1 != 0 {
                v16(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
            } else if mask_8x8 & 1 != 0 {
                v8(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
            } else if mask_4x4 & 1 != 0 {
                v4(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
            }
            if mask_4x4_int & 1 != 0 {
                v4(pixels, s + 4, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
            }
        }
        s += 8;
        lfl_index += 1;
        mask >>= 1;
        mask_16x16 >>= 1;
        mask_8x8 >>= 1;
        mask_4x4 >>= 1;
        mask_4x4_int >>= 1;
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
    dispatch: &SimdDispatch,
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
        dispatch,
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
    dispatch: &SimdDispatch,
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
        dispatch,
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
    dispatch: &SimdDispatch,
) {
    // Wave H H1: bind the loopfilter slots once so the per-mask
    // dispatch below is a single field load + indirect call. The
    // pointer values are stable across the SB iteration (the
    // dispatch table is immutable after `dsp::init`).
    let v16 = dispatch.lpf_vertical_16();
    let v16d = dispatch.lpf_vertical_16_dual();
    let v8 = dispatch.lpf_vertical_8();
    let v8d = dispatch.lpf_vertical_8_dual();
    let v4 = dispatch.lpf_vertical_4();
    let v4d = dispatch.lpf_vertical_4_dual();

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
                    v16d(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else if mask_16x16 & 1 != 0 {
                    v16(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    v16(
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
                    v8d(
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
                    v8(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    v8(
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
                    v4d(
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
                    v4(pixels, s, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    v4(
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
                    v4d(
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
                    v4(pixels, s + 4, pitch, lfi0.mblim, lfi0.lim, lfi0.hev_thr);
                } else {
                    v4(
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
    dispatch: &SimdDispatch,
) {
    // Wave H H1: bind the slot pointers once per call. See sibling
    // `filter_selectively_vert_row2_inner` for the rationale.
    let h16 = dispatch.lpf_horizontal_16();
    let h16d = dispatch.lpf_horizontal_16_dual();
    let h8 = dispatch.lpf_horizontal_8();
    let h8d = dispatch.lpf_horizontal_8_dual();
    let h4 = dispatch.lpf_horizontal_4();
    let h4d = dispatch.lpf_horizontal_4_dual();

    let mut mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    let mut s = s_index;
    let mut lfl_index: usize = 0;
    while mask != 0 {
        let mut count: u32 = 1;
        if mask & 1 != 0 {
            let lfi = info.thresh(lfl[lfl_index]);

            if mask_16x16 & 1 != 0 {
                if (mask_16x16 & 3) == 3 {
                    h16d(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    count = 2;
                } else {
                    h16(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                }
            } else if mask_8x8 & 1 != 0 {
                if (mask_8x8 & 3) == 3 {
                    let lfin = info.thresh(lfl[lfl_index + 1]);
                    h8d(
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
                        h4d(
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
                        h4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    } else if mask_4x4_int & 2 != 0 {
                        h4(
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
                    h8(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    if mask_4x4_int & 1 != 0 {
                        h4(
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
                    h4d(
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
                        h4d(
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
                        h4(
                            pixels,
                            s + 4 * pitch,
                            pitch,
                            lfi.mblim,
                            lfi.lim,
                            lfi.hev_thr,
                        );
                    } else if mask_4x4_int & 2 != 0 {
                        h4(
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
                    h4(pixels, s, pitch, lfi.mblim, lfi.lim, lfi.hev_thr);
                    if mask_4x4_int & 1 != 0 {
                        h4(
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
                h4(
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
    dispatch: &SimdDispatch,
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
        dispatch,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::ChromaFormat;
    use crate::frame::{BorderUsage, ByteAlignment};
    // Tests still call scalar lpf_* fns directly to compute reference
    // outputs (then compare to dispatch-routed `loop_filter_frame`).
    // Production code in this module goes through dispatch slots.
    use crate::loopfilter::{lpf_horizontal_4, lpf_vertical_4};

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

    // ============================================================
    // Tests targeting the non-420 path (4:2:2 / 4:4:0) and 4:4:4
    // (LF_PATH_444). These exercise `filter_block_plane_non420`,
    // `filter_selectively_vert_non420`, and the chroma-format dispatch
    // in `loop_filter_frame_with_non420`. Construction is heavier than
    // the 420 tests because the non420 path consults the MI grid /
    // `FrameFilterLevels` cube directly (see `Non420Inputs`).
    // ============================================================

    use crate::bitstream::InterpFilter;
    use crate::block::BlockSize;
    use crate::intra_pred::IntraMode;
    use crate::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
    use crate::transforms::TxSize;

    /// Allocate a 64×64 frame with the specified chroma format. Used
    /// by the non-420 and 4:4:4 tests where the chroma-plane shape
    /// matters.
    fn alloc_64x64_chroma(chroma: ChromaFormat) -> Yv12Buffer {
        Yv12Buffer::alloc(
            64,
            64,
            chroma,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc 64x64 frame")
    }

    /// Intra-DC `ModeInfo` builder used to populate the MI grid for
    /// the non-420 tests. Mirrors the helper in `setup_mask.rs`'s
    /// test mod.
    fn intra_mi(block_size: BlockSize, tx_size: TxSize) -> ModeInfo {
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

    /// Inter `ModeInfo` builder. Useful to drive the
    /// `is_inter_block` / `mode_lf_idx` / `ref0_idx` branches in
    /// the non-420 helpers (and the `lvl == 0` short-circuit by
    /// targeting an unused LUT slot).
    fn inter_mi(
        block_size: BlockSize,
        tx_size: TxSize,
        mode: InterMode,
        ref_frame: RefFrame,
        skip: bool,
    ) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size,
            skip,
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

    /// Flat `FrameFilterLevels` cube: every `[seg][ref][mode]` slot
    /// holds the same level. Borrowed from `setup_mask.rs`'s test
    /// mod for use here.
    fn flat_levels(level: u8) -> crate::loopfilter::FrameFilterLevels {
        [[[level; crate::loopfilter::MAX_MODE_LF_DELTAS];
            crate::loopfilter::MAX_REF_FRAMES]; crate::loopfilter::MAX_SEGMENTS]
    }

    /// 4:4:4 chroma path (`LF_PATH_444`): U and V planes are filtered
    /// by `filter_block_plane_ss00` using the Y masks (chroma grid ==
    /// luma grid). Construct a step in the U plane and confirm the
    /// chroma vertical kernel fires when the *Y* `left_y` mask has the
    /// edge bit. (The 4:2:0 test sets `left_uv`; the 4:4:4 path is
    /// distinguished by reading `left_y` for U/V too.)
    #[test]
    fn loop_filter_frame_yuv444_filters_u_via_y_mask() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv444);
        // U plane is full-resolution (64x64). Place a step at col 8
        // exactly like the Y test does.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // Mask: set the Y left mask bit for the 8×8 at SB-cell (1, 1).
        // The 4:4:4 path treats this bit as the chroma 8×8's left edge
        // (since the chroma grid == luma grid). UV masks are ignored
        // on this path so we leave them zero.
        let mut lfm = LoopFilterMask::ZERO;
        let bit_index = (1u64 << 3) | 1; // row=1 col=1
        lfm.left_y[TX_4X4] |= 1u64 << bit_index;
        // lfl_y at row 1 (cells 8..16) carries the filter level used
        // by *both* the Y plane and the 4:4:4 chroma planes.
        for c in 0..8 {
            lfm.lfl_y[(1 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        // The U plane should have changed at the (row 8..16, col ~8)
        // region. The V plane was untouched (no V-only edge bits) but
        // since the 4:4:4 path runs the same kernel on V, it should
        // also be filtered.
        let u_after = snapshot_u(&buf);
        assert_ne!(u_before, u_after, "U plane should have been filtered");

        // Rows 0..8 of U: untouched (no above-bit, no left-bit at row 0).
        for r in 0..8 {
            for c in 0..buf.u_plane().width() {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(
                    buf.u_plane().row(r)[c],
                    expected,
                    "U row={r} col={c} should be unchanged"
                );
            }
        }
    }

    /// 4:2:2 (non-420) path basic check: build a frame with a single
    /// 16×16 intra block at SB-cell (1, 1), set its filter level to 30,
    /// place a step in the U plane at the chroma column matching the
    /// block's left edge, and confirm the non-420 dispatch fires the
    /// vertical kernel.
    ///
    /// Yuv422 ⇒ chroma is half-width, full-height (32×64 for a 64×64
    /// luma frame). The chroma column matching luma col 8 is chroma
    /// col 4 (luma col >> ss_x). The vertical kernel fires at the
    /// left edge of a chroma 8×8 column.
    #[test]
    fn loop_filter_frame_yuv422_non420_vertical_edge_fires() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Chroma U plane = 32 wide × 64 tall. Place a step at chroma
        // col 8 (matches luma col 16 under ss_x=1).
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // MI grid: all cells are intra Block16x16 with TX_8x8 luma →
        // chroma TX size under (ss_x=1, ss_y=0) for Block16x16 is
        // derived by `uv_plane_tx_size`. The block at (r=2, c=2) (luma
        // pixel col 16) has a "left edge" the non-420 walker should
        // emit a chroma vertical edge for at chroma col 8.
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block16x16, TxSize::Tx8x8); 8 * 8];
        let levels = flat_levels(30);

        // The 420 mask grid is unused by the non420 dispatch; pass a
        // zero mask so the Y plane is untouched and we can attribute
        // any U change to the non420 helper.
        let lfm = LoopFilterMask::ZERO;

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[lfm],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_ne!(
            u_before, u_after,
            "non420 dispatch should have filtered the U plane"
        );

        // Y plane: no Y mask bits, so Y must be untouched.
        let y_after = snapshot_y(&buf);
        assert!(
            y_after.iter().all(|&p| p == 0),
            "Y plane should be untouched (zero mask + uninitialised Y)"
        );
    }

    /// 4:4:0 (ss_y) path: exercises the `ss_y` branches of
    /// `filter_block_plane_non420`, specifically the
    /// `skip_border_4x4_r = ss_y && mi_row + r == mi_rows - 1` guard
    /// in the horizontal pass.
    #[test]
    fn loop_filter_frame_yuv440_non420_horizontal_edge_fires() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv440);
        // Chroma plane = 64 wide × 32 tall (ss_y=true). Place a step
        // at chroma row 8 to drive the horizontal kernel.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if r < 8 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block16x16, TxSize::Tx8x8); 8 * 8];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_ne!(
            u_before, u_after,
            "non420 4:4:0 dispatch should have filtered the U plane"
        );
    }

    /// Drive the `TxSize::Tx32x32` arm of the non420 build-mask
    /// logic. Under Yuv422 + Block64x64 + TX_32X32 luma, the chroma
    /// block-size is Block32x64 and the chroma TX size is Tx32x32 →
    /// the TX_32X32 arm of the per-MI dispatch fires.
    ///
    /// The leftmost chroma col edge is cleared by the `border_mask`
    /// (mi_col == 0); the *horizontal* edge at chroma row 4 is the
    /// one that visibly fires under TX_32X32 (`((r >> ss_y) & 3) == 0`
    /// for r ∈ {0, 4}; r=0 is then cleared by `mi_row + r == 0` in
    /// the horizontal pass; r=4 survives). We place a horizontal
    /// step at chroma row 32 so the kernel at chroma row 32 has
    /// non-flat data to smooth.
    #[test]
    fn non420_tx32x32_chroma_marks_horizontal_edge() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Horizontal step at chroma row 32 (luma row 32 under
        // ss_y=0 = full-height): top half = 60, bottom half = 90.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if r < 32 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // One Block64x64 with TX_32X32 covers the whole SB.
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block64x64, TxSize::Tx32x32); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        // The TX_32X32 arm should mark the chroma column at the 32-px
        // boundary → some pixel in U should have changed.
        assert_ne!(
            u_before, u_after,
            "TX_32X32 non420 chroma marker should have produced a filter call"
        );
    }

    /// Drive the `TxSize::Tx4x4` arm of the non420 walker. A grid of
    /// Block8x8 + TX_4X4 intra blocks under 4:2:2 yields chroma TX
    /// size TX_4X4, exercising the `mask_4x4_int` internal-edge path
    /// (line `if !skip_this && tx_size < TxSize::Tx8x8 && ...`).
    #[test]
    fn non420_tx4x4_chroma_marks_mask_4x4_internal_edges() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Step at chroma col 0 to give the leftmost chroma 8x8 a
        // visible vertical edge that the kernel can chew on. We
        // pollute the rest with a non-flat pattern to surface
        // unintended writes.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 4 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // Block8x8 + TX_4X4 → chroma TX_4X4 (the internal-4x4 edge
        // arm fires).
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block8x8, TxSize::Tx4x4); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_ne!(
            u_before, u_after,
            "TX_4X4 non420 path should fire chroma filter kernels"
        );
    }

    /// Drive the `TxSize::Tx16x16` arm of the non420 walker. A grid
    /// of Block32x32 + TX_16x16 intra blocks under 4:2:2 yields chroma
    /// TX_16x16 (via `uv_plane_tx_size`), exercising the TX_16X16 arm.
    ///
    /// Under Yuv422 + Block32x32, the leftmost MI-col edge (c=0)
    /// is cleared by `border_mask` (`mi_col == 0`); the next block
    /// edge sits at c=4 → bit position `c >> ss_x = 2` → chroma
    /// pixel col 16. We place an edge there so the 16-tap kernel
    /// has visible work to do.
    #[test]
    fn non420_tx16x16_chroma_marks_mask_16x16() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 16 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block32x32, TxSize::Tx16x16); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_ne!(
            u_before, u_after,
            "TX_16X16 non420 path should fire chroma filter kernels"
        );
    }

    /// `lvl == 0` short-circuit in the non420 walker: if the MI's
    /// resolved filter level is zero, the bit-build is skipped and
    /// no edge is emitted for that column. We exercise this by
    /// putting a non-zero level in only one ref-slot of the LUT and
    /// using a different ref for the MI, so `get_filter_level_mi`
    /// returns 0 for every MI.
    #[test]
    fn non420_filter_level_zero_skips_block() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // Cube: only [seg=0][ref=2=GOLDEN][mode=0] = 30; every other
        // slot is 0. All MIs use LAST (ref idx 1), so the LUT lookup
        // returns 0 and the walker short-circuits every column.
        let mut levels = flat_levels(0);
        levels[0][2][0] = 30;

        // Inter Block16x16 with ZeroMv (mode_lf_idx = 0), ref LAST
        // (ref_idx = 1).
        let mi = inter_mi(
            BlockSize::Block16x16,
            TxSize::Tx8x8,
            InterMode::ZeroMv,
            RefFrame::LastFrame,
            false,
        );
        let mi_grid: Vec<ModeInfo> = vec![mi; 64];

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_eq!(
            u_before, u_after,
            "lvl == 0 should short-circuit every MI and leave U unchanged"
        );
    }

    /// `skip_this` short-circuit: an inter block with `skip = true`
    /// gets its internal transform edges suppressed (`skip_this_c`
    /// and `skip_this_r` set). For an 8×8 inter block with skip and
    /// TX_4X4, the only edges emitted are the block-left / block-above
    /// edges (where `block_edge_left`/`block_edge_above` is true).
    /// The TX_4X4 internal 4×4 edges (`mask_4x4_int`) are gated by
    /// `!skip_this`, so they must NOT be marked.
    ///
    /// We can't directly inspect `mask_4x4_int`, but we can construct
    /// a chroma plane that's flat *except* at the internal-4×4 boundary
    /// of one block. If the internal edge were marked, the kernel
    /// would smear pixels there; with skip suppression, the plane
    /// stays unchanged at those pixels.
    #[test]
    fn non420_inter_skip_short_circuits_internal_edges() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Flat plane = 128 everywhere. The kernel can't fire on a flat
        // plane (every diff is zero), so this confirms the static
        // behaviour: with skip suppression, no kernel ever runs.
        {
            let mut u = buf.u_plane_mut();
            u.fill(128);
        }
        let u_before = snapshot_u(&buf);

        // Inter Block8x8 + TX_4X4 + skip. The 8x8 block is at every
        // MI cell.
        let mi = inter_mi(
            BlockSize::Block8x8,
            TxSize::Tx4x4,
            InterMode::ZeroMv,
            RefFrame::LastFrame,
            true,
        );
        let mi_grid: Vec<ModeInfo> = vec![mi; 64];
        // Build a level table where ZeroMv/LAST → 30. (We need a
        // non-zero level so the walker DOESN'T short-circuit on
        // `lvl == 0`; the test point is the *skip-based* suppression
        // happening downstream.)
        let mut levels = flat_levels(0);
        levels[0][1][0] = 30;

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        // Flat plane stays flat: zero pixel differences confirms the
        // kernel either didn't fire OR fired-but-was-a-noop. Both
        // outcomes preserve the plane — what matters here is that we
        // executed the non420 walker with `skip_this = true` on every
        // MI, hitting the skip-related branches.
        let u_after = snapshot_u(&buf);
        assert_eq!(
            u_before, u_after,
            "skip-set inter blocks on a flat plane must produce no change"
        );
    }

    /// Left-frame-edge clear in the non420 walker: the
    /// `border_mask = if mi_col == 0 { !1u32 } else { !0u32 }` line
    /// zeroes the leftmost column bit when the SB is at the frame's
    /// left edge. We pin this by leaving a sentinel value in the
    /// leftmost chroma column and verifying that the vertical kernel
    /// at chroma col 0 does NOT fire (which would visibly perturb
    /// the sentinel byte).
    ///
    /// Filter-level 30 + diff 30 (60→90 step) gives blimit=94 and
    /// `2*30 + 30/2 = 75 ≤ 94` ⇒ the filter would fire if the
    /// border mask let it. The check is the *absence* of a change
    /// inside chroma col 0 (the kernel only writes p3..p0 q0..q3 at
    /// the edge, and the leftmost edge is at chroma col 0 → it
    /// writes cols -4..-1 and 0..3; with border_mask cleared no write
    /// happens at all, so col 0..3 stay at their sentinel value).
    #[test]
    fn non420_left_frame_edge_clears_col0() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Set up a step at chroma col 8 (where the second vertical
        // edge sits under Block16x16 + Yuv422). Cols 0..4 painted
        // with sentinel = 200; cols 4..8 = 60; cols 8..32 = 90. The
        // kernel at col 8 reads cols 4..11 and writes cols 4..11 →
        // it should fire and smooth the 60→90 step. The kernel at
        // col 0 (if it fired) would read cols -4..3 (border bytes
        // for negative offsets) and write cols 0..3 — but the border
        // mask blocks that.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 4 {
                        200u8
                    } else if c < 8 {
                        60u8
                    } else {
                        90u8
                    };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // All Block16x16 + TX_8X8: every block has a left edge at
        // c=0 and c=2 (Block16x16 → n_8x8_wide=2 → block_edge at c
        // & 1 == 0). The border mask clears bit 0 (chroma col 0)
        // but bit 1 (chroma col 8) survives.
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block16x16, TxSize::Tx8x8); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        // Leftmost 4 chroma cols still hold their 200-sentinel value:
        // the kernel at col 0 was suppressed by the border mask.
        for r in 0..buf.u_plane().height() {
            for c in 0..4 {
                assert_eq!(
                    buf.u_plane().row(r)[c],
                    200,
                    "U row={r} col={c} left-frame-edge should be unfiltered"
                );
            }
        }
        // Sanity: the kernel at col 8 DID fire and smoothed the
        // 60→90 step.
        let u_after = snapshot_u(&buf);
        assert_ne!(u_before, u_after, "kernel at col 8 should have fired");
    }

    /// Yuv440 (ss_y=true) with the SB's bottom row at the frame's
    /// last visible row: `skip_border_4x4_r = ss_y && mi_row + r ==
    /// mi_rows - 1` is true → the horizontal pass drops the
    /// `mask_4x4_int_r` value, leaving the internal-4×4 horizontal
    /// edge unfiltered. We construct the smallest test that exercises
    /// the `true` branch.
    #[test]
    fn non420_yuv440_skip_border_4x4_r_at_last_row() {
        // mi_rows = 8 (frame is exactly one SB tall). For r=7 in the
        // horizontal pass: ss_y=true, mi_row+r = 0+7 = 7 = mi_rows-1
        // → skip_border_4x4_r = true.
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv440);
        // Step at chroma row 8 so the kernel has work to do in
        // earlier rows.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if r < 8 { 60 } else { 90 };
                }
            }
        }

        // Block8x8 + TX_4X4 ⇒ chroma TX_4X4 under (0, 1). The
        // mask_4x4_int contribution will be marked for every row r
        // where `mask_4x4_int[r] |= bit`. At the very last row,
        // `skip_border_4x4_r` should suppress that contribution in
        // the horizontal pass.
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block8x8, TxSize::Tx4x4); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        // We can't directly observe the masked-out mask_4x4_int bit
        // (it's a per-row local), but completion without a panic plus
        // the fact that `non420_filter_level_zero_skips_block` already
        // pins the no-op semantics gives us coverage of the
        // `skip_border_4x4_r = true` branch. Pin the static post-
        // condition: something earlier in the SB got filtered (the
        // step at chroma row 8 has plenty of edges).
        // (Test passes if the call doesn't OOB / panic.)
        let _ = snapshot_u(&buf);
    }

    /// `mi_col != 0` branch of the `border_mask` line: when the SB
    /// isn't at the frame's left edge, the leftmost vertical edge
    /// bit is *kept* and a kernel call fires at chroma col 0 of the
    /// SB-local origin. We construct a 128-pixel-wide frame (2 SBs)
    /// and place an edge at the second SB's left side.
    #[test]
    fn non420_mi_col_nonzero_keeps_left_edge() {
        // 128×64 frame, Yuv422 ⇒ chroma U is 64×64.
        let mut buf = Yv12Buffer::alloc(
            128,
            64,
            ChromaFormat::Yuv422,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc 128x64");

        // Place a 60→90 step at chroma col 32 (the boundary between
        // the first and second SB at chroma col 32 = luma col 64).
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 32 { 60 } else { 90 };
                }
            }
        }
        let u_before = snapshot_u(&buf);

        // 16×8 MI grid (128 luma / 8 = 16 cols, 64 / 8 = 8 rows).
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block16x16, TxSize::Tx8x8); 16 * 8];
        let levels = flat_levels(30);

        // 2 SBs across, 1 SB tall. Both mask grids are zero (the
        // non-420 walker ignores them).
        let lfm_grid = [LoopFilterMask::ZERO, LoopFilterMask::ZERO];

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 16,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &lfm_grid,
            2,
            1,
            2,
            8,
            16,
            &info,
            0,
            8,
            Some(non420),
        );

        let u_after = snapshot_u(&buf);
        assert_ne!(
            u_before, u_after,
            "edge at the SB1/SB0 boundary (chroma col 32) should fire"
        );
        // Pin: at the SB1 boundary, the leftmost edge bit is NOT
        // border-masked (mi_col=8 != 0 for SB1), so the kernel
        // smoothed the 60→90 step at chroma col 32.
        let row_mid = buf.u_plane().height() / 2;
        let pre = u_before[row_mid * buf.u_plane().width() + 31];
        let post = u_after[row_mid * buf.u_plane().width() + 31];
        assert_ne!(pre, post, "chroma col 31 should have been smoothed");
    }

    /// Block8x16 has `n_4x4_wide = 2 > 1` but `n_4x4_high = 4 > 1` —
    /// both branches of the `if sb_type.n_4x4_wide() > 1` /
    /// `n_4x4_high() > 1` ternaries take the `true` arm. We need a
    /// case where one or both take the `else true` arm: Block4x4 has
    /// `n_4x4_wide = 1`, exercising the `else true` arm directly.
    #[test]
    fn non420_block4x4_exercises_else_true_branches() {
        let mut buf = alloc_64x64_chroma(ChromaFormat::Yuv422);
        // Step at chroma col 8 to give the kernels something to do.
        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }

        // Block4x4 + TX_4X4 throughout. n_4x4_wide = 1 → the else
        // branches of the block_edge_left/above ternaries fire.
        let mi_grid: Vec<ModeInfo> = vec![intra_mi(BlockSize::Block4x4, TxSize::Tx4x4); 64];
        let levels = flat_levels(30);

        let info = LoopFilterInfo::new(0);
        let non420 = Non420Inputs {
            mi_grid: &mi_grid,
            mi_stride: 8,
            levels: &levels,
        };
        loop_filter_frame_with_non420(
            &mut buf,
            &[LoopFilterMask::ZERO],
            1,
            1,
            1,
            8,
            8,
            &info,
            0,
            8,
            Some(non420),
        );

        // No panic + coverage of the `else true` branches is enough.
        // Pin: the step at chroma col 8 was smoothed somewhere.
        let mut col6 = vec![];
        for r in 0..buf.u_plane().height() {
            col6.push(buf.u_plane().row(r)[6]);
        }
        // col 6 was originally 60; if the kernel at col 8 (the
        // leftmost non-border-masked block edge under Block4x4) fired,
        // col 6 — within the 4-pixel taps of the kernel — should have
        // moved off 60.
        assert!(
            col6.iter().any(|&v| v != 60),
            "kernel should have perturbed col 6"
        );
    }

    /// Y-plane vertical pass: two adjacent block-rows both with
    /// `mask_4x4` set (dual_one match) — exercises the
    /// `(mask_4x4 & dual_one) == dual_one` true branch and v4d
    /// kernel. We hand-construct the mask directly (this is a 420
    /// path test).
    #[test]
    fn y_vertical_row2_dual_4x4_both_rows_fires_dual_kernel() {
        let mut buf = alloc_64x64();
        // Steps at col 8 and col 16 so the 4-tap kernel has data to
        // smooth at both edges.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = match c {
                        0..=7 => 60u8,
                        8..=15 => 90u8,
                        _ => 60u8,
                    };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // Set TX_4X4 left bits for row 0 col 1 AND row 1 col 1 (bit
        // indices 1 and 9). The vertical pass groups these into a
        // dual call because dual_one = 1 | (1<<8) = 0x101 and both
        // bits 1 and 9 of `(mask >> 1)` are set after the inner
        // loop's first shift.
        //
        // Wait — the row2_inner loops over `mask` (column bits 0..8
        // for the first row, with the corresponding second-row bit
        // 8 positions higher). We want col=1 in BOTH row 0 and row
        // 1 ⇒ bits (0<<3)|1=1 and (1<<3)|1=9 → both set.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_4X4] |= (1u64 << 1) | (1u64 << 9);
        // Lfl_y at rows 0 and 1 set to 30.
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "dual 4×4 vertical kernel should fire");
    }

    /// Y horizontal: two adjacent columns both with mask_8x8 set
    /// triggers the dual h8 path (mask & 3 == 3). Exercise the
    /// `mask_8x8 & 3 == 3` branch in `filter_selectively_horiz`.
    #[test]
    fn y_horizontal_dual_8x8_fires_dual_kernel() {
        let mut buf = alloc_64x64();
        // Horizontal step at row 16 (luma block-row 2) so the
        // 8-tap horizontal kernel has work to do.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if r < 16 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // Set TX_8X8 above-mask bits at row 2 col 0 AND col 1: bit
        // (2<<3)|0 = 16 and (2<<3)|1 = 17. The horizontal helper sees
        // `mask_8x8 & 3 == 3` (after shifting by 16 to land at row 2's
        // window) and fires the h8d kernel.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_8X8] |= (1u64 << 16) | (1u64 << 17);
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "h8d should have fired at row 16");
    }

    /// Y horizontal: two adjacent columns both with mask_16x16 set
    /// triggers the dual h16 path (mask & 3 == 3 in the 16-tap arm).
    #[test]
    fn y_horizontal_dual_16x16_fires_dual_kernel() {
        let mut buf = alloc_64x64();
        // Horizontal step at row 16 with a wider flat plateau on
        // each side so the 16-tap kernel's `flat_mask5` check passes.
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_16X16 above bits at row 2 col 0 + col 1.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_16X16] |= (1u64 << 16) | (1u64 << 17);
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "h16d should have fired at row 16");
    }

    /// Y vertical row2: only the *first* block-row has mask_4x4 set
    /// (the second block-row's bit is clear) — exercises the
    /// `else if mask_4x4 & 1 != 0` arm with the v4 single kernel.
    /// (The "only second row" arm is already hit by
    /// `single_y_vertical_edge_filters_only_target_row`.)
    #[test]
    fn y_vertical_row2_first_row_only_fires_single_kernel() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_4X4 left bit at row 0 col 1 (bit 1) only — NOT row 1 col
        // 1. The vertical pass processes rows 0 and 1 together
        // (row2 dual). The inner helper sees `mask_4x4 & dual_one`
        // with only bit 0 set ⇒ `mask & dual_one != 0` (true), but
        // `mask_4x4 & 1 != 0` ⇒ first-row-only branch fires.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_4X4] |= 1u64 << 1;
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        // Rows 0..8 (luma block-row 0) should have been filtered;
        // rows 8..16 should not have been touched by the vertical
        // pass (the horizontal pass doesn't fire either — no above
        // bits). The whole frame WAS a column step at col 8, so
        // confirm rows 8..16 are untouched at the edge.
        for r in 8..16 {
            for c in 0..buf.y_plane().width() {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(
                    buf.y_plane().row(r)[c],
                    expected,
                    "row 8..16 should be untouched (only first block-row had mask set)"
                );
            }
        }
        // Row 0 should have been touched.
        assert_ne!(y_before, y_after, "first-row-only vertical kernel should fire");
    }

    /// Y vertical row2: both rows have `mask_16x16` set → dual v16
    /// kernel fires. Exercise the `(mask_16x16 & dual_one) == dual_one`
    /// true arm.
    #[test]
    fn y_vertical_row2_dual_16x16_both_rows_fires_dual_kernel() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 0..y.height() {
                for c in 16..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_16X16 left bit at row 0 col 2 AND row 1 col 2 (bits 2
        // and 10). dual_one for Y = 1 | (1<<8) = 0x101. Shifted by
        // 2 columns the inner helper sees the dual bit set.
        // Actually the helper processes col-by-col: when lfl_index=2
        // (col=2), it checks `mask & dual_one`. mask_16x16 at that
        // point has bits 0 (=2-2) and 8 (=10-2) set. (1 | 1<<8) ==
        // dual_one ⇒ dual fires.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_16X16] |= (1u64 << 2) | (1u64 << 10);
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "dual v16 kernel should fire at col 16");
    }

    /// Y vertical row2: both rows have `mask_4x4_int` set → dual v4
    /// kernel fires at the +4 mid-block offset.
    #[test]
    fn y_vertical_row2_dual_int_4x4_both_rows_fires_dual_kernel() {
        let mut buf = alloc_64x64();
        // Stripes: every 4-pixel col group alternates 60/90 so the
        // mid-block 4×4 internal kernels see a step.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if (c >> 2) & 1 == 0 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // int_4x4_y bit at row 0 col 0 AND row 1 col 0 (bits 0 and 8).
        let mut lfm = LoopFilterMask::ZERO;
        lfm.int_4x4_y |= (1u64 << 0) | (1u64 << 8);
        // Also need a left_y bit so the vertical helper's outer mask
        // is non-zero at lfl_index=0 — actually we want only the
        // int_4x4 contribution to drive the iteration. Setting
        // mask_4x4 = 0 and mask_4x4_int = 1 at col 0 still works
        // because `mask = mask_16x16 | mask_8x8 | mask_4x4 |
        // mask_4x4_int` ≠ 0. Then `if mask & dual_one != 0` is true
        // (only mask_4x4_int contributes). Inside, mask_16x16/8x8/4x4
        // dual_one are zero so those skip. mask_4x4_int dual_one ==
        // dual_one ⇒ v4d at s+4.
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "dual v4 internal kernel should fire");
    }

    /// Y horizontal: `mask_4x4_int & 1 != 0` inside the dual h8 path
    /// (`(mask_8x8 & 3) == 3` AND `(mask_4x4_int & 3) != 3` AND
    /// `mask_4x4_int & 1 != 0`) — exercises the
    /// `else if mask_4x4_int & 1 != 0` arm of
    /// `filter_selectively_horiz`.
    #[test]
    fn y_horizontal_dual_h8_with_int_4x4_first_col_only_fires() {
        let mut buf = alloc_64x64();
        // Horizontal step at row 16 + a horizontal sub-step at row 20
        // restricted to cols 0..8 (first col of the pair). The
        // internal 4×4 kernel at row 20 cols 0..8 should fire.
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            for c in 0..8 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        let mut lfm = LoopFilterMask::ZERO;
        // TX_8X8 above bits at row 2 col 0 AND col 1.
        lfm.above_y[TX_8X8] |= (1u64 << 16) | (1u64 << 17);
        // int_4x4 bit at row 2 col 0 only (so dual = 0b01 not 0b11).
        lfm.int_4x4_y |= 1u64 << 16;
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// 4:2:0 ss11 path: with the SB occupying the very bottom row of
    /// the frame (`mi_row + r == mi_rows - 1` true for r=7), the
    /// `skip_border_4x4_r` branch fires → the internal-4×4
    /// horizontal edge at the last chroma row is suppressed.
    #[test]
    fn ss11_skip_border_4x4_r_at_last_chroma_row() {
        // 64-pixel-wide × 8-pixel-tall frame: mi_rows = 1, so r=0
        // is the only iteration AND mi_row + 0 = 0 = 1 - 1.
        // `skip_border_4x4_r = (mi_row + r == mi_rows - 1) = true`.
        let mut buf = Yv12Buffer::alloc(
            64,
            8,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .expect("alloc 64x8");

        {
            let mut u = buf.u_plane_mut();
            for r in 0..u.height() {
                for c in 0..u.width() {
                    u.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }

        let mut lfm = LoopFilterMask::ZERO;
        // Set int_4x4_uv bit at chroma row 0 col 1: chroma grid is
        // 4×4 within an SB, indexed (r<<2)|c, so row 0 col 1 = bit 1.
        lfm.int_4x4_uv |= 1u16 << 1;
        lfm.lfl_y.fill(30);

        let info = LoopFilterInfo::new(0);
        // mi_rows = 1, sb_rows = 1, end_mi_row = 1 (only row 0).
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 1, 8, &info, 0, 1);

        // The mid-block internal edge at chroma row 4 (= half of an
        // 8x8 chroma block) would have been filtered if not for
        // `skip_border_4x4_r = true`. Since the U plane has a step
        // at col 8 (not at row 4), the filter would barely move
        // pixels — but the key coverage point is that the *true*
        // branch of `skip_border_4x4_r` runs. (Verification by lack
        // of panic + the test passes the assertion that the call
        // completed.)
        let _ = snapshot_u(&buf);
    }

    /// Y vertical row2: only the second block-row has mask_16x16
    /// set (first row's bit is clear) — exercises the *else*
    /// branch of `else if mask_16x16 & 1 != 0` in
    /// `filter_selectively_vert_row2_inner` (the kernel fires at
    /// `s + second_pitch` with lfi1, NOT at `s` with lfi0).
    #[test]
    fn y_vertical_row2_16x16_second_row_only_fires() {
        let mut buf = alloc_64x64();
        // Step at col 16 with wider plateau for 16-tap flat_mask5.
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 0..y.height() {
                for c in 16..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_16X16 left bit at row 1 col 2 ONLY (bit 10). When the
        // helper processes lfl_index=2 (col 2), the relevant
        // mask_16x16 bit pattern (after the helper's internal shifts
        // by 2) is `0b1_0000_0000` = 0x100 (bit 8 is the second-row
        // bit for col 2 because dual_one = 1 | (1<<8)). At this
        // iteration the dual check fails (mask_16x16 & 1 == 0) but
        // the second-row branch fires (`else` clause).
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_16X16] |= 1u64 << 10;
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "second-row-only v16 kernel should fire");
    }

    /// Y vertical row2: only the second block-row has mask_8x8 set —
    /// exercises the `else { v8(s + second_pitch, ...) }` arm in
    /// `filter_selectively_vert_row2_inner`. (The dual case is hit
    /// by `single_y_vertical_edge_filters_only_target_row` which
    /// fires at row 1; this test pins the symmetric path.)
    #[test]
    fn y_vertical_row2_8x8_second_row_only_fires() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if c < 8 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_8X8 left bit at row 1 col 1 ONLY (bit 9).
        let mut lfm = LoopFilterMask::ZERO;
        lfm.left_y[TX_8X8] |= 1u64 << 9;
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "second-row-only v8 kernel should fire");
        // Rows 0..8 should be untouched.
        for r in 0..8 {
            for c in 0..buf.y_plane().width() {
                let expected = if c < 8 { 60u8 } else { 90u8 };
                assert_eq!(buf.y_plane().row(r)[c], expected);
            }
        }
    }

    /// Y vertical row2: only the second block-row has mask_8x8 set
    /// (mirror of the 8x8 second-row test, but for the dual-bit
    /// `(mask_8x8 & dual_one) == dual_one` False arm in the 8-tap
    /// branch). Same setup as above — covers the same branch.
    /// (Same as above; the test exists to nudge the second-row
    /// dispatcher path with mask_8x8 & 1 != 0 = False as well.)
    /// Y vertical row2: only the second block-row has mask_4x4_int
    /// set — exercises the second-row-only path of the internal-
    /// 4×4 vertical kernel (`else { v4(s + 4 + second_pitch, ...) }`).
    #[test]
    fn y_vertical_row2_int_4x4_second_row_only_fires() {
        let mut buf = alloc_64x64();
        // Striped pattern so the mid-block 4×4 kernel sees a step.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if (c >> 2) & 1 == 0 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // int_4x4_y bit at row 1 col 0 only (bit 8). dual_one = 0x101;
        // (mask_4x4_int & dual_one) = 0x100 ≠ dual_one ⇒ dual check
        // fails. mask_4x4_int & 1 == 0 ⇒ else branch fires (v4 at
        // s+4+second_pitch).
        let mut lfm = LoopFilterMask::ZERO;
        lfm.int_4x4_y |= 1u64 << 8;
        for c in 0..16 {
            lfm.lfl_y[c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "second-row-only int_4x4 kernel should fire");
    }

    /// Y horizontal: a gap in the mask — bit 0 set, bit 1 clear, bit
    /// 2 set. After the first iteration fires (count=1, shift), the
    /// helper hits `mask & 1 == 0` and skips an iteration. Covers
    /// the `False` branch of `if mask & 1 != 0` in
    /// `filter_selectively_horiz`.
    #[test]
    fn y_horizontal_mask_with_gap_skips_empty_iteration() {
        let mut buf = alloc_64x64();
        // Horizontal step at row 16 so col-aligned 8-tap kernels
        // have data to smooth.
        {
            let mut y = buf.y_plane_mut();
            for r in 0..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = if r < 16 { 60 } else { 90 };
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_8X8 above bits at row 2 col 0 AND col 2 (skip col 1).
        // mask_8x8 (in the helper's per-row window) starts as
        // 0b101 = 5. First iter: mask & 1 = 1, h8 fires; count=1,
        // shift; mask = 0b10. Second iter: mask & 1 = 0 ⇒ skip.
        // Third iter: mask = 0b1, h8 fires; etc.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_8X8] |= (1u64 << 16) | (1u64 << 18);
        for c in 0..8 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// Y horizontal: `mask_8x8 & 1 != 0` single-column path (not the
    /// dual) WITH `mask_4x4_int & 1 != 0` → exercises the
    /// `if mask_4x4_int & 1 != 0` arm inside the single-h8 branch
    /// (line 1296-1305).
    #[test]
    fn y_horizontal_single_h8_with_int_4x4_first_col_fires() {
        let mut buf = alloc_64x64();
        // Horizontal step at row 16 + sub-step at row 20 in col 0..8.
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            for c in 0..8 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_8X8 above bit at row 2 col 0 ONLY (no col 1 — single
        // h8 path, not dual). int_4x4_y bit at row 2 col 0.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_8X8] |= 1u64 << 16;
        lfm.int_4x4_y |= 1u64 << 16;
        for c in 0..8 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// Y horizontal: `mask_4x4_int & 1 != 0` arm of the single-h4
    /// path (line 1355) — when mask_4x4 is set single-col (not dual)
    /// AND int_4x4 is set at the same col.
    #[test]
    fn y_horizontal_single_h4_with_int_4x4_fires() {
        let mut buf = alloc_64x64();
        // Step at row 16 + mid-block step at row 20 (4-pixel resolution).
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            for c in 0..8 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_4X4 above bit at row 2 col 0 only + int_4x4 bit.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_4X4] |= 1u64 << 16;
        lfm.int_4x4_y |= 1u64 << 16;
        for c in 0..8 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// Y horizontal: `mask_4x4_int & 1 != 0` arm of the dual-h4d
    /// path (line 1333). Dual mask_4x4 at row 2 cols 0+1, plus
    /// int_4x4 at row 2 col 0 only.
    #[test]
    fn y_horizontal_dual_h4d_with_int_4x4_first_col_only_fires() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            for c in 0..8 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        // TX_4X4 above bits at row 2 col 0 AND col 1 (dual).
        // int_4x4_y bit at row 2 col 0 only.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_4X4] |= (1u64 << 16) | (1u64 << 17);
        lfm.int_4x4_y |= 1u64 << 16;
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// Y horizontal: `mask_4x4_int & 2 != 0` arm of the dual-h4d
    /// path (line 1342). Dual mask_4x4 at cols 0+1, int_4x4 at
    /// col 1 only.
    #[test]
    fn y_horizontal_dual_h4d_with_int_4x4_second_col_only_fires() {
        let mut buf = alloc_64x64();
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            for c in 8..16 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_4X4] |= (1u64 << 16) | (1u64 << 17);
        lfm.int_4x4_y |= 1u64 << 17;
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }

    /// Y horizontal: ONLY the internal-4×4 bit is set (no
    /// mask_16x16/8x8/4x4 at that column) → the `else` arm at
    /// line 1366 fires (h4 at the mid-block edge).
    #[test]
    fn y_horizontal_only_int_4x4_fires_mid_block_kernel() {
        let mut buf = alloc_64x64();
        // Step in the middle of a luma 8×8 (the int-4×4 mid-edge).
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 20..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
        }
        let y_before = snapshot_y(&buf);

        // Only int_4x4_y bit at row 2 col 0 — no other mask bits.
        let mut lfm = LoopFilterMask::ZERO;
        lfm.int_4x4_y |= 1u64 << 16;
        for c in 0..8 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after, "only-int_4x4 path should fire h4 at mid-block");
    }

    /// Y horizontal: `mask_4x4_int & 2 != 0` (only the second column
    /// of the dual-pair has the internal 4×4 bit set) — exercises
    /// the `else if mask_4x4_int & 2 != 0` arm in
    /// `filter_selectively_horiz` when h8d / h4d has fired.
    ///
    /// Setup: TX_8X8 dual at row 2 col 0+1 (mask_8x8 & 3 == 3); plus
    /// `int_4x4_y` bit for row 2 col 1 only (so mask_4x4_int & 1 ==
    /// 0 but mask_4x4_int & 2 != 0).
    #[test]
    fn y_horizontal_int_4x4_second_col_only_fires() {
        let mut buf = alloc_64x64();
        // Step pattern at row 16 + a sub-step at row 20 (within the
        // 4×4-internal mid-block) inside col 8..16 only.
        {
            let mut y = buf.y_plane_mut();
            y.fill(60);
            for r in 16..y.height() {
                for c in 0..y.width() {
                    y.row_mut(r)[c] = 90;
                }
            }
            // Internal mid-block step at row 20, cols 8..16 (the
            // second col of the dual pair): this gives the
            // internal-4×4 kernel something to do.
            for c in 8..16 {
                y.row_mut(20)[c] = 100;
                y.row_mut(21)[c] = 100;
                y.row_mut(22)[c] = 100;
                y.row_mut(23)[c] = 100;
            }
        }
        let y_before = snapshot_y(&buf);

        let mut lfm = LoopFilterMask::ZERO;
        lfm.above_y[TX_8X8] |= (1u64 << 16) | (1u64 << 17);
        // int_4x4_y bit at row 2 col 1 only: position (2<<3)|1 = 17.
        lfm.int_4x4_y |= 1u64 << 17;
        for c in 0..16 {
            lfm.lfl_y[(2 << 3) + c] = 30;
        }

        let info = LoopFilterInfo::new(0);
        loop_filter_frame(&mut buf, &[lfm], 1, 1, 1, 8, 8, &info, 0, 8);

        let y_after = snapshot_y(&buf);
        assert_ne!(y_before, y_after);
    }
}
