//! W6.8 — TPL backward-propagation (`tpl_model_update` +
//! `tpl_model_store`).
//!
//! Faithful port of:
//!
//! - `tpl_model_store` — `vp9/encoder/vp9_tpl_model.c:372-390`
//!   (v1.14.1 commit `12f3a2ac6`). Broadcasts the "anchor" stat at
//!   `(mi_row, mi_col)` to every 8×8 sub-mi inside `bsize`,
//!   preserving each sub-mi's `mc_flow` / `mc_ref_cost` accumulators
//!   and recomputing `mc_dep_cost = intra_cost + mc_flow`.
//! - `tpl_model_update_b` + `tpl_model_update` —
//!   `vp9_tpl_model.c:419-489`. For each 8×8 sub-mi of the current
//!   frame, projects its `mc_dep_cost * (1 - inter_cost / intra_cost)`
//!   flow into the **reference frame**'s `TplDepStats` according to
//!   the block's MV, weighted by the 4-way overlap area calculated
//!   on the reference grid.
//!
//! ## Scope
//!
//! - 8×8-mi resolution propagation only, matching libvpx's
//!   `BLOCK_8X8` inner loop (`vp9_tpl_model.c:486`).
//! - Pure integer arithmetic — no float, no SSE. Mirrors libvpx
//!   exactly.
//! - **TODO(W6.9):** wire `tpl_model_update` into the per-frame
//!   `mc_flow_dispenser` driver; this batch ships propagation as a
//!   standalone primitive.

use rvp9_core::block::BlockSize;

use super::stats::{MI_SIZE, TplDepFrame, TplDepStats};

/// libvpx `round_floor` (`vp9_tpl_model.c:362-370`). Floor-division
/// that handles negative numerators in a way that matches libvpx's C
/// integer behaviour (C99 truncates toward zero, the libvpx routine
/// shifts toward `-inf`).
#[inline]
fn round_floor(ref_pos: i32, bsize_pix: i32) -> i32 {
    debug_assert!(bsize_pix > 0);
    if ref_pos < 0 {
        -(1 + (-ref_pos - 1) / bsize_pix)
    } else {
        ref_pos / bsize_pix
    }
}

/// libvpx `get_overlap_area` (`vp9_tpl_model.c:333-360`). Returns the
/// pixel-area overlap between the reference patch (its top-left at
/// `(ref_pos_row, ref_pos_col)`) and the grid block whose top-left
/// is `(grid_pos_row, grid_pos_col)`, for each of the 4 possible
/// quadrant overlaps (`block ∈ 0..4`).
#[inline]
fn get_overlap_area(
    grid_pos_row: i32,
    grid_pos_col: i32,
    ref_pos_row: i32,
    ref_pos_col: i32,
    block: u32,
    bw: i32,
    bh: i32,
) -> i32 {
    let (width, height) = match block {
        0 => (
            grid_pos_col + bw - ref_pos_col,
            grid_pos_row + bh - ref_pos_row,
        ),
        1 => (
            ref_pos_col + bw - grid_pos_col,
            grid_pos_row + bh - ref_pos_row,
        ),
        2 => (
            grid_pos_col + bw - ref_pos_col,
            ref_pos_row + bh - grid_pos_row,
        ),
        3 => (
            ref_pos_col + bw - grid_pos_col,
            ref_pos_row + bh - grid_pos_row,
        ),
        _ => unreachable!("get_overlap_area: block must be 0..4"),
    };
    width * height
}

/// libvpx `tpl_model_store` (`vp9_tpl_model.c:372-390`).
///
/// Broadcasts the "anchor" `TplDepStats` at `(mi_row, mi_col)` to
/// every 8×8 sub-mi inside the `bsize` superblock, preserving each
/// sub-mi's `mc_flow` / `mc_ref_cost` accumulators (those are written
/// by downstream frames' propagation passes) and recomputing
/// `mc_dep_cost = intra_cost + mc_flow`.
///
/// Mirrors the libvpx C exactly:
///
/// ```c
/// for (idy = 0; idy < mi_height; ++idy) {
///   for (idx = 0; idx < mi_width; ++idx) {
///     TplDepStats *tpl_ptr = &tpl_stats[(mi_row + idy) * stride
///                                       + mi_col + idx];
///     const int64_t mc_flow = tpl_ptr->mc_flow;
///     const int64_t mc_ref_cost = tpl_ptr->mc_ref_cost;
///     *tpl_ptr = *src_stats;
///     tpl_ptr->mc_flow = mc_flow;
///     tpl_ptr->mc_ref_cost = mc_ref_cost;
///     tpl_ptr->mc_dep_cost = tpl_ptr->intra_cost + tpl_ptr->mc_flow;
///   }
/// }
/// ```
pub fn tpl_model_store(tpl_frame: &mut TplDepFrame, mi_row: usize, mi_col: usize, bsize: BlockSize) {
    let mi_height = bsize.n_8x8_high() as usize;
    let mi_width = bsize.n_8x8_wide() as usize;
    let stride = tpl_frame.stride;
    let src_stats = *tpl_frame.stat(mi_row, mi_col);
    for idy in 0..mi_height {
        for idx in 0..mi_width {
            let dst_idx = (mi_row + idy) * stride + (mi_col + idx);
            let dst = &mut tpl_frame.tpl_stats[dst_idx];
            let preserved_mc_flow = dst.mc_flow;
            let preserved_mc_ref_cost = dst.mc_ref_cost;
            *dst = src_stats;
            dst.mc_flow = preserved_mc_flow;
            dst.mc_ref_cost = preserved_mc_ref_cost;
            dst.mc_dep_cost = dst.intra_cost + dst.mc_flow;
        }
    }
}

/// libvpx `tpl_model_update_b` (`vp9_tpl_model.c:419-473`).
///
/// Per-sub-mi propagation: takes the anchor's `TplDepStats` at
/// `(mi_row, mi_col)` and contributes `mc_flow` / `mc_ref_cost`
/// into the 4 overlapping cells of the reference frame's stats
/// grid, weighted by `overlap_area / pix_num`.
fn tpl_model_update_b(
    cur_stats: &TplDepStats,
    ref_frame: &mut TplDepFrame,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
) {
    let mv = cur_stats.mv;
    // libvpx C reads `mv.row >> 3` / `mv.col >> 3` — arithmetic shift
    // for sign extension. Our `i16` matches semantically.
    let mv_row = i32::from(mv.row) >> 3;
    let mv_col = i32::from(mv.col) >> 3;

    let mi_size = MI_SIZE as i32;
    let ref_pos_row = (mi_row as i32) * mi_size + mv_row;
    let ref_pos_col = (mi_col as i32) * mi_size + mv_col;

    // libvpx `bw = 4 << b_width_log2_lookup[bsize]` resolves to the
    // pixel side: for BLOCK_8X8 → 8, BLOCK_16X16 → 16, BLOCK_32X32 → 32.
    // rvp9's `BlockSize::width()` already returns pixels directly.
    let bw = i32::from(bsize.width());
    let bh = i32::from(bsize.height());

    let mi_height = bsize.n_8x8_high() as usize;
    let mi_width = bsize.n_8x8_wide() as usize;
    let pix_num = bw * bh;

    // Top-left of the grid block(s) in the reference frame, snapped
    // down to a multiple of `bh` / `bw`.
    let grid_pos_row_base = round_floor(ref_pos_row, bh) * bh;
    let grid_pos_col_base = round_floor(ref_pos_col, bw) * bw;

    // The reference frame's `mc_dep` flow is computed once per
    // current-frame block. Match libvpx's division semantics
    // (integer truncate-toward-zero).
    if cur_stats.intra_cost == 0 {
        // libvpx asserts `intra_cost >= 1` via the VPXMAX(.., 1) in
        // mode_estimation; defend against bad inputs in our scaffold
        // (used by tests that may feed synthetic zero stats).
        return;
    }
    let mc_flow = cur_stats.mc_dep_cost
        - (cur_stats.mc_dep_cost * cur_stats.inter_cost) / cur_stats.intra_cost;

    let ref_grid_h_pix = ref_frame.mi_rows as i32 * mi_size;
    let ref_grid_w_pix = ref_frame.mi_cols as i32 * mi_size;

    for block in 0..4u32 {
        let grid_pos_row = grid_pos_row_base + bh * ((block >> 1) as i32);
        let grid_pos_col = grid_pos_col_base + bw * ((block & 0x01) as i32);

        if grid_pos_row < 0
            || grid_pos_row >= ref_grid_h_pix
            || grid_pos_col < 0
            || grid_pos_col >= ref_grid_w_pix
        {
            continue;
        }

        let overlap_area = get_overlap_area(
            grid_pos_row,
            grid_pos_col,
            ref_pos_row,
            ref_pos_col,
            block,
            bw,
            bh,
        );
        debug_assert!(overlap_area >= 0);

        let ref_mi_row =
            (round_floor(grid_pos_row, bh) * (mi_height as i32)) as usize;
        let ref_mi_col =
            (round_floor(grid_pos_col, bw) * (mi_width as i32)) as usize;

        let stride = ref_frame.stride;
        for idy in 0..mi_height {
            for idx in 0..mi_width {
                let dst_idx = (ref_mi_row + idy) * stride + (ref_mi_col + idx);
                let dst = &mut ref_frame.tpl_stats[dst_idx];
                dst.mc_flow += (mc_flow * i64::from(overlap_area)) / i64::from(pix_num);
                dst.mc_ref_cost +=
                    ((cur_stats.intra_cost - cur_stats.inter_cost) * i64::from(overlap_area))
                        / i64::from(pix_num);
            }
        }
    }
}

/// libvpx `tpl_model_update` (`vp9_tpl_model.c:475-489`).
///
/// Walks every 8×8 sub-mi inside the `bsize` superblock of the
/// current frame and projects its dependency cost into
/// `ref_frame`'s stats grid via `tpl_model_update_b`.
///
/// `cur_frame` is the **current frame** whose stats we read; `ref_frame`
/// is the reference frame whose `mc_flow` / `mc_ref_cost` we update.
/// libvpx threads these via `tpl_frame[ref_frame_index]` and the
/// top-level `tpl_frame[]` array; in rvp9 the caller routes the pair
/// explicitly.
pub fn tpl_model_update(
    cur_frame: &TplDepFrame,
    ref_frame: &mut TplDepFrame,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
) {
    let mi_height = bsize.n_8x8_high() as usize;
    let mi_width = bsize.n_8x8_wide() as usize;
    let stride = cur_frame.stride;
    // We need to snapshot the cur stats first because tpl_model_update_b
    // borrows `ref_frame` mutably; if we passed `cur_frame` through it'd
    // be a borrow conflict in the (cur == ref) edge case.
    for idy in 0..mi_height {
        for idx in 0..mi_width {
            let cur_idx = (mi_row + idy) * stride + (mi_col + idx);
            let cur_stats = cur_frame.tpl_stats[cur_idx];
            tpl_model_update_b(
                &cur_stats,
                ref_frame,
                mi_row + idy,
                mi_col + idx,
                BlockSize::Block8x8,
            );
        }
    }
}
