//! F11.0 — encoder-side loop-filter wireup.
//!
//! Mirrors the decoder's `apply_loop_filter_to_frame` so the encoder's
//! reference reconstruction stays bit-exact with the decoder's
//! reconstruction even when the in-loop deblocking filter fires.
//!
//! Two public entry points:
//!
//! - [`pick_filter_level`] — coarse `vp9_pick_filter_level`
//!   approximation: `filter_level ≈ q_index / 2`. See "Divergences"
//!   below.
//! - [`apply_loop_filter_to_tight_recon`] — copies the encoder's
//!   tight-stride reconstruction planes into a temporary
//!   [`Yv12Buffer`], runs `setup_mask` + `adjust_mask` +
//!   `loop_filter_frame` (the same trio the decoder uses), and copies
//!   the filtered pixels back. The encoder owns the reconstruction as
//!   three tight-stride `Vec<u8>` blobs (one per plane); the loop
//!   filter operates on the strided `Yv12Buffer` shape — the
//!   copy-in/copy-out is the cheapest correct adapter.
//!
//! ## Header invariants
//!
//! The encoder emits `mode_ref_delta_enabled = false` and
//! `mode_ref_delta_update = false` in every uncompressed header. With
//! that header shape, the decoder's `merge_lf_params` sets
//! `lf_deltas.enabled = false` (the keyframe-reset `[1, 0, -1, -1]`
//! deltas are overwritten by this merge — `enabled` is set
//! unconditionally). Consequently `resolve_filter_levels` takes the
//! "no deltas" branch and produces a flat `[seg][ref][mode]` cube
//! where every cell equals `filter_level`. The encoder mirrors this
//! by passing `ResolvedLfDeltas::default()` (all-zero, `enabled =
//! false`).
//!
//! ## Divergences from libvpx
//!
//! - **Filter-level picking.** libvpx's `vp9_pick_filter_level`
//!   (`vp9_picklpf.c:191-294`) runs a filtered-vs-unfiltered SSE
//!   search bracketing the previous frame's `filter_level`. That
//!   requires re-running the loop filter at multiple candidate levels
//!   per frame — too heavy for v1. We use a libvpx-friendly proxy:
//!   `filter_level = (q_index >> 1).min(MAX_LOOP_FILTER)`. At
//!   typical Q indices (40..120) this lands in the
//!   `filter_level ∈ [20, 60]` range, which sits inside libvpx's
//!   own typical envelope for the same Q. Re-evaluate in F11.1 once
//!   PSNR oracles are stable.
//! - **No `lf_deltas` plumbing.** v1 emits zero ref/mode deltas,
//!   matching the encoder's "no segmentation, single-ref LAST"
//!   shape. F11.1+ may revisit if/when segmentation is wired.
//! - **No sharpness control.** `sharpness_level = 0` always.

use rvp9_core::bitstream::ChromaFormat;
use rvp9_core::block::{BlockSize, max_txsize};
use rvp9_core::frame::{BorderUsage, ByteAlignment, Yv12Buffer};
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::loopfilter::{
    LoopFilterInfo, LoopFilterMask, MAX_LOOP_FILTER, ResolvedLfDeltas, SegmentationLfView,
    adjust_mask, loop_filter_frame, resolve_filter_levels, setup_mask,
};
use rvp9_core::mode_info::{BlockMode, ModeInfo, Mv, RefFrame};
use rvp9_core::bitstream::InterpFilter;

use crate::{InterPartitionLeafDesc, PartitionLeafDesc};

/// MI-grid side length of a 64×64 superblock (in 8×8 units).
const SB_MI_SIZE: usize = 8;

/// libvpx-divergent filter-level picker. See module docs.
///
/// At `q_index == 0` returns `0` (signals "disable LF"). At `q_index
/// == 1` (the encoder's minimum non-lossless clamp) returns `0` too —
/// the `>> 1` halving rounds 1 down to 0. Callers must skip the LF
/// pass entirely on `filter_level == 0` to match libvpx's
/// `vp9_loopfilter.c:1422-1427` early-out.
#[must_use]
pub(crate) fn pick_filter_level(q_index: u8) -> u8 {
    (q_index >> 1).min(MAX_LOOP_FILTER)
}

/// Apply the loop filter in place to the encoder's tight-stride
/// reconstruction planes. After return, `recon_y/u/v` contain the
/// post-LF pixels — the same pixels the decoder will produce after
/// its own LF pass given a header that emits the same `filter_level`
/// + zero LF deltas + sharpness 0.
///
/// `mi_grid` carries one [`Option<ModeInfo>`] per 8×8 luma cell,
/// row-major with stride `mi_cols`. `None` cells are replaced with a
/// placeholder (cloned from the first populated cell, or a fallback
/// intra cell if the grid is somehow all-`None` — that path is
/// unreachable in the encoder's frame loop but keeps this function
/// total).
///
/// # Panics
///
/// Panics if `filter_level == 0` — callers must skip the LF pass
/// entirely in that case to match the decoder's early-out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_loop_filter_to_tight_recon(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) {
    assert!(
        filter_level > 0,
        "callers must skip the LF pass at filter_level == 0"
    );

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mi_rows = h / 8;
    let mi_cols = w / 8;

    // Allocate a fresh Yv12Buffer the LF kernels can write into. The
    // border is needed because `loop_filter_frame` indexes ±8 pixels
    // past visible columns/rows via `mb_lpf_vertical_edge_w` /
    // `mb_lpf_horizontal_edge_w`; the `Decoder` border (96 px)
    // matches what the decoder allocates for its own frame buffers.
    let mut yv12 = Yv12Buffer::alloc(
        width,
        height,
        ChromaFormat::Yuv420,
        BorderUsage::Decoder,
        ByteAlignment::Legacy,
    )
    .expect("Yv12Buffer alloc for LF pass");

    // Copy tight-stride encoder recon into the strided Yv12Buffer.
    {
        let mut yp = yv12.y_plane_mut();
        let stride = yp.stride();
        let data = yp.data_mut();
        for r in 0..h {
            data[r * stride..r * stride + w]
                .copy_from_slice(&recon_y[r * w..r * w + w]);
        }
    }
    {
        let mut up = yv12.u_plane_mut();
        let stride = up.stride();
        let data = up.data_mut();
        for r in 0..uvh {
            data[r * stride..r * stride + uvw]
                .copy_from_slice(&recon_u[r * uvw..r * uvw + uvw]);
        }
    }
    {
        let mut vp = yv12.v_plane_mut();
        let stride = vp.stride();
        let data = vp.data_mut();
        for r in 0..uvh {
            data[r * stride..r * stride + uvw]
                .copy_from_slice(&recon_v[r * uvw..r * uvw + uvw]);
        }
    }

    // Resolve the [seg][ref][mode] filter-level cube. The encoder
    // emits `mode_ref_delta_enabled = false` ⇒ the decoder's
    // `merge_lf_params` sets `lf_deltas.enabled = false` ⇒ every
    // (seg, ref, mode) cell collapses to `filter_level`. We mirror
    // that here.
    let levels = resolve_filter_levels(
        filter_level,
        ResolvedLfDeltas::default(),
        SegmentationLfView::default(),
    );
    let info = LoopFilterInfo::new(/* sharpness_level = */ 0);

    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);

    // Materialise the full-frame ModeInfo grid (one entry per 8×8
    // luma cell). `setup_mask` expects `&[ModeInfo]` indexed as
    // `mi[r*stride + c]`. The encoder's `mi_grid` is
    // `&[Option<ModeInfo>]`; backfill `None` cells with a
    // placeholder cloned from the first populated cell.
    let placeholder = mi_grid
        .iter()
        .find_map(|c| c.as_ref().cloned())
        .unwrap_or_else(default_placeholder_mode_info);
    let mut frame_mi: Vec<ModeInfo> = Vec::with_capacity(mi_grid.len().max(1));
    for cell in mi_grid {
        match cell {
            Some(mi) => frame_mi.push(*mi),
            None => frame_mi.push(placeholder),
        }
    }

    // Build per-SB masks, then dispatch the apply driver.
    let mut lfm_grid: Vec<LoopFilterMask> =
        vec![LoopFilterMask::default(); sb_rows * sb_cols];
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mi_row = sb_row * SB_MI_SIZE;
            let mi_col = sb_col * SB_MI_SIZE;
            let lfm = &mut lfm_grid[sb_row * sb_cols + sb_col];
            setup_mask(
                mi_row as u32,
                mi_col as u32,
                &frame_mi,
                mi_cols,
                mi_rows as u32,
                mi_cols as u32,
                &levels,
                lfm,
            );
            adjust_mask(
                mi_row as u32,
                mi_col as u32,
                mi_rows as u32,
                mi_cols as u32,
                lfm,
            );
        }
    }

    loop_filter_frame(
        &mut yv12, &lfm_grid, sb_cols, sb_rows, sb_cols, mi_rows, mi_cols, &info, 0, mi_rows,
    );

    // Copy filtered pixels back into the encoder's tight-stride
    // reconstruction.
    {
        let yp = yv12.y_plane();
        let stride = yp.stride();
        let data = yp.data();
        for r in 0..h {
            recon_y[r * w..r * w + w]
                .copy_from_slice(&data[r * stride..r * stride + w]);
        }
    }
    {
        let up = yv12.u_plane();
        let stride = up.stride();
        let data = up.data();
        for r in 0..uvh {
            recon_u[r * uvw..r * uvw + uvw]
                .copy_from_slice(&data[r * stride..r * stride + uvw]);
        }
    }
    {
        let vp = yv12.v_plane();
        let stride = vp.stride();
        let data = vp.data();
        for r in 0..uvh {
            recon_v[r * uvw..r * uvw + uvw]
                .copy_from_slice(&data[r * stride..r * stride + uvw]);
        }
    }
}

/// Placeholder ModeInfo cell. Used to backfill `None` slots in the
/// `mi_grid` that the encoder's partition picker didn't write. The
/// loop-filter partition walker only reads the top-left MI of each
/// partition leaf so this placeholder is never actually consulted in
/// practice; it just keeps the conversion total.
fn default_placeholder_mode_info() -> ModeInfo {
    ModeInfo {
        block_size: BlockSize::Block64x64,
        tx_size: max_txsize(BlockSize::Block64x64),
        skip: true,
        segment_id: 0,
        seg_id_predicted: false,
        mode: BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
        },
        bmi: Default::default(),
    }
}

/// Build a frame-wide `Option<ModeInfo>` grid from intra (keyframe)
/// partition leaves. Mirrors `build_kf_mi_grid` in the frame loop —
/// re-exposed here so the encoder LF wireup can consume the same
/// shape without duplicating layout knowledge.
pub(crate) fn build_kf_mi_grid_for_lf(
    mi_rows: u32,
    mi_cols: u32,
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> Vec<Option<ModeInfo>> {
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    let mut mi_grid: Vec<Option<ModeInfo>> = vec![None; mi_rows_us * mi_cols_us];

    let sb_cols = mi_cols_us / 8;
    let sb_rows = mi_rows_us / 8;
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let cell = ModeInfo {
                    block_size: leaf.leaf_bsize,
                    tx_size: max_txsize(leaf.leaf_bsize),
                    skip: false,
                    segment_id: 0,
                    seg_id_predicted: false,
                    mode: BlockMode::Intra {
                        y_mode: leaf.predicted_y_mode,
                        uv_mode: IntraMode::Dc,
                    },
                    bmi: Default::default(),
                };
                splat_leaf(
                    &mut mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf.leaf_bsize,
                    cell,
                );
            }
        }
    }
    mi_grid
}

/// Inter-shape equivalent of [`build_kf_mi_grid_for_lf`].
pub(crate) fn build_inter_mi_grid_for_lf(
    mi_rows: u32,
    mi_cols: u32,
    sb_partitions: &[Vec<InterPartitionLeafDesc>],
) -> Vec<Option<ModeInfo>> {
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    let mut mi_grid: Vec<Option<ModeInfo>> = vec![None; mi_rows_us * mi_cols_us];

    let sb_cols = mi_cols_us / 8;
    let sb_rows = mi_rows_us / 8;
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let cell = ModeInfo {
                    block_size: leaf.leaf_bsize,
                    tx_size: max_txsize(leaf.leaf_bsize),
                    skip: leaf.skip,
                    segment_id: 0,
                    seg_id_predicted: false,
                    mode: BlockMode::Inter {
                        mode: leaf.picked_mode,
                        ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                        mvs: [leaf.picked_mv, Mv::ZERO],
                        interp_filter: InterpFilter::EightTap,
                    },
                    bmi: Default::default(),
                };
                splat_leaf(
                    &mut mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf.leaf_bsize,
                    cell,
                );
            }
        }
    }
    mi_grid
}

fn splat_leaf(
    mi_grid: &mut [Option<ModeInfo>],
    mi_cols: usize,
    leaf_mi_row: usize,
    leaf_mi_col: usize,
    leaf_bsize: BlockSize,
    cell: ModeInfo,
) {
    let n_8x8_wide = leaf_bsize.n_8x8_wide() as usize;
    let n_8x8_high = leaf_bsize.n_8x8_high() as usize;
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let r = leaf_mi_row + dr;
            let c = leaf_mi_col + dc;
            mi_grid[r * mi_cols + c] = Some(cell);
        }
    }
}
