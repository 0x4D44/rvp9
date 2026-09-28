//! F11.0 — encoder-side loop-filter wireup.
//!
//! Mirrors the decoder's `apply_loop_filter_to_frame` so the encoder's
//! reference reconstruction stays bit-exact with the decoder's
//! reconstruction even when the in-loop deblocking filter fires.
//!
//! Two public entry points:
//!
//! - [`pick_filter_level`] — libvpx's realtime `LPF_PICK_FROM_Q`
//!   policy, with a diagnostic escape hatch for the legacy proxy.
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
//! The encoder enables libvpx's default mode/reference deltas. Keyframes
//! signal the reset values `[1, 0, -1, -1]`; following inter frames keep
//! them enabled without re-sending unchanged values. The encoder applies
//! the same resolved delta state to its reference reconstruction.
//!
//! ## Divergences from libvpx
//!
//! - **No full-image filter search.** Realtime cpu-used 6–8 selects
//!   `LPF_PICK_FROM_Q`, which rvp9 mirrors. The slower full-image and
//!   sub-image search policies remain outside the v1 realtime scope.
//! - **Fixed `lf_deltas`.** v1 uses libvpx's default reference deltas
//!   and zero mode deltas; it does not yet adapt them per frame.
//! - **No sharpness control.** `sharpness_level = 0` always.

use rvp9_core::bitstream::BitDepth;
use rvp9_core::bitstream::ChromaFormat;
use rvp9_core::block::{BlockSize, max_txsize};
use rvp9_core::frame::{BorderUsage, ByteAlignment, Yv12Buffer};
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::loopfilter::{
    LoopFilterInfo, LoopFilterMask, MAX_LOOP_FILTER, ResolvedLfDeltas, SegmentationLfView,
    adjust_mask, loop_filter_frame, resolve_filter_levels, setup_mask,
};

const LIBVPX_DEFAULT_LF_DELTAS: ResolvedLfDeltas = ResolvedLfDeltas {
    ref_deltas: [1, 0, -1, -1],
    mode_deltas: [0, 0],
    enabled: true,
};
use rvp9_core::mode_info::{BlockMode, ModeInfo};
use rvp9_core::quant::ac_quant;
use std::borrow::Cow;

use crate::PartitionLeafDesc;

/// MI-grid side length of a 64×64 superblock (in 8×8 units).
const SB_MI_SIZE: usize = 8;

/// LPFQ (Track A4, 2026-05-30) — libvpx-faithful `LPF_PICK_FROM_Q`.
///
/// Default ON; `RVP9_LPFQ=0` is the escape hatch that restores the
/// legacy `(q_index >> 1)` proxy byte-for-byte.
///
/// rvp9's legacy proxy `(q_index >> 1)` over-filters at mid/low q vs
/// libvpx's calibrated formula. The worst case is the KEYFRAME: at q≈57
/// the proxy yields filter_level ≈28 while libvpx yields ≈5, so rvp9
/// over-smooths the anchor reference and every inter frame inherits the
/// lost detail (lower PSNR). At high q (low bitrate) both saturate at 63.
///
/// **Measured (quality_gate cpu7):** foreman mean Δ +2.51 → +2.43 dB
/// (250k +0.27, KF-propagated gains at 25/100k). testsrc (synthetic,
/// sharp edges the over-filter blurred) STRICTLY better — 25k +1.83 dB,
/// 50k +1.03, 100k +3.19, all at equal-or-fewer bits.
#[inline]
fn lpfq_enabled() -> bool {
    std::env::var_os("RVP9_LPFQ").is_none_or(|v| v != "0")
}

/// Filter-level picker. With LPFQ on, ports libvpx's `LPF_PICK_FROM_Q`
/// (`vp9_picklpf.c`) — the method `set_rt_speed_feature` selects at
/// cpu-used 7 (`sf->lpf_pick = LPF_PICK_FROM_Q`):
///
/// ```text
/// q          = ac_quant(base_qindex)           // AC dequant value, not qindex
/// filt_guess = ROUND_POWER_OF_TWO(q*20723 + 1015158, 18)
/// if KEY_FRAME: filt_guess -= 4
/// filter_level = clamp(filt_guess, 0, 63)
/// ```
///
/// (8-bit, aq-off CBR path — the cyclic-refresh `5*g>>3` adjustment is
/// gated on `aq_mode == CYCLIC_REFRESH_AQ`, which rvp9's default op-point
/// is not.) With LPFQ off, falls back to the legacy `(q_index >> 1)`
/// proxy (byte-identical to the pre-LPFQ baseline).
///
/// At `q_index == 0` (lossless) returns `0` (signals "disable LF"). At
/// `q_index == 1` the legacy proxy returns `0`; the LPFQ formula returns
/// a small positive value — callers skip the LF pass entirely on
/// `filter_level == 0` to match libvpx's `vp9_loopfilter.c` early-out.
#[must_use]
pub(crate) fn pick_filter_level(q_index: u8, is_key: bool) -> u8 {
    if lpfq_enabled() {
        // Lossless (q_index == 0) → no loop filter. rvp9's encoder clamps
        // q_index to `min_qindex` (≥ 4 by default), so this is a defensive
        // guard preserving the "lossless ⇒ filter_level 0" invariant the LF
        // callers rely on (they skip the pass on filter_level == 0).
        if q_index == 0 {
            return 0;
        }
        // libvpx LPF_PICK_FROM_Q (8-bit aq-off CBR).
        let q = i32::from(ac_quant(i32::from(q_index), 0, BitDepth::Eight));
        let mut filt_guess = (q * 20723 + 1_015_158 + (1 << 17)) >> 18;
        if is_key {
            filt_guess -= 4;
        }
        filt_guess.clamp(0, i32::from(MAX_LOOP_FILTER)) as u8
    } else {
        (q_index >> 1).min(MAX_LOOP_FILTER)
    }
}

/// Apply the loop filter in place to the encoder's tight-stride
/// reconstruction planes. After return, `recon_y/u/v` contain the
/// post-LF pixels — the same pixels the decoder will produce after
/// its own LF pass given a header that emits the same `filter_level`,
/// default LF deltas, and sharpness 0.
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
    apply_loop_filter_to_tight_recon_parallel(
        width,
        height,
        filter_level,
        mi_grid,
        recon_y,
        recon_u,
        recon_v,
        None,
        1,
        1,
    );
}

/// Parallel product variant of [`apply_loop_filter_to_tight_recon`].
///
/// The supplied pool is long-lived and owned by the stream encoder. When at
/// least two work opportunities and two superblock rows are available, this
/// dispatches the core split-edge pass as one alias-safe scoped job. The core
/// path remains limited to one worker until it uses row-bounded plane slices.
/// Otherwise this uses the serial path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_loop_filter_to_tight_recon_parallel(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    worker_pool: Option<&mut rvp9_core::tile_threading::WorkerPool>,
    max_threads: usize,
    num_tile_cols: usize,
) {
    assert!(
        filter_level > 0,
        "callers must skip the LF pass at filter_level == 0"
    );

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let mi_rows = h.div_ceil(8);
    let mi_cols = w / 8;

    let available_lf_workers = max_threads
        .min(num_tile_cols.max(1))
        .min(mi_rows.div_ceil(SB_MI_SIZE))
        .max(1);
    // Keep the existing direct split-edge dispatch for complete SB rows.
    // Partial rows use the serial path below only when their padded backing
    // storage and MI geometry satisfy the helper's addressability contract.
    if height.is_multiple_of(64)
        && available_lf_workers >= 2
        && let Some(worker_pool) = worker_pool
    {
        worker_pool.scope(|scope| {
            scope.submit(move || {
                apply_loop_filter_to_tight_recon_split_edge(
                    width,
                    height,
                    filter_level,
                    mi_grid,
                    recon_y,
                    recon_u,
                    recon_v,
                    num_tile_cols,
                );
            });
        });
        return;
    }

    let storage_h = h.next_multiple_of(64);
    let direct_storage = (w * storage_h, uvw * (storage_h / 2), uvw * (storage_h / 2));
    // The split-edge helper currently derives its MI height with floor
    // division and its luma stride with whole 64-pixel columns. Restrict the
    // serial direct route to partial SB rows until those assumptions are true,
    // and require the backing planes to expose the complete final SB row that
    // its fixed-width kernels can address. Visible-only or irregular storage
    // keeps the bordered path.
    let direct_serial_eligible = !height.is_multiple_of(64)
        && width.is_multiple_of(64)
        && height.is_multiple_of(8)
        && (recon_y.len(), recon_u.len(), recon_v.len()) == direct_storage;
    if direct_serial_eligible {
        apply_loop_filter_to_tight_recon_split_edge_preserving_tails(
            width,
            height,
            filter_level,
            mi_grid,
            recon_y,
            recon_u,
            recon_v,
            num_tile_cols,
        );
        return;
    }

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
            data[r * stride..r * stride + w].copy_from_slice(&recon_y[r * w..r * w + w]);
        }
    }
    {
        let mut up = yv12.u_plane_mut();
        let stride = up.stride();
        let data = up.data_mut();
        for r in 0..uvh {
            data[r * stride..r * stride + uvw].copy_from_slice(&recon_u[r * uvw..r * uvw + uvw]);
        }
    }
    {
        let mut vp = yv12.v_plane_mut();
        let stride = vp.stride();
        let data = vp.data_mut();
        for r in 0..uvh {
            data[r * stride..r * stride + uvw].copy_from_slice(&recon_v[r * uvw..r * uvw + uvw]);
        }
    }

    // Resolve the [seg][ref][mode] filter-level cube with libvpx's
    // keyframe-reset mode/reference deltas. This is also the persistent
    // state used by the following inter frames.
    let levels = resolve_filter_levels(
        filter_level,
        LIBVPX_DEFAULT_LF_DELTAS,
        SegmentationLfView::default(),
    );
    let info = LoopFilterInfo::new(/* sharpness_level = */ 0);

    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);

    // The encoder's product grid builders splat every leaf across its full
    // footprint, so complete frames can go straight to the mask walker.
    // Retain the old placeholder completion only for incomplete/adversarial
    // grids instead of allocating and copying a second full grid every frame.
    let frame_mi_opt = complete_mi_grid_for_lf(mi_grid);

    // Build per-SB masks, then dispatch the serial apply driver.
    let mut lfm_grid: Vec<LoopFilterMask> = vec![LoopFilterMask::default(); sb_rows * sb_cols];
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mi_row = sb_row * SB_MI_SIZE;
            let mi_col = sb_col * SB_MI_SIZE;
            let lfm = &mut lfm_grid[sb_row * sb_cols + sb_col];
            setup_mask(
                mi_row as u32,
                mi_col as u32,
                &frame_mi_opt,
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
            recon_y[r * w..r * w + w].copy_from_slice(&data[r * stride..r * stride + w]);
        }
    }
    {
        let up = yv12.u_plane();
        let stride = up.stride();
        let data = up.data();
        for r in 0..uvh {
            recon_u[r * uvw..r * uvw + uvw].copy_from_slice(&data[r * stride..r * stride + uvw]);
        }
    }
    {
        let vp = yv12.v_plane();
        let stride = vp.stride();
        let data = vp.data();
        for r in 0..uvh {
            recon_v[r * uvw..r * uvw + uvw].copy_from_slice(&data[r * stride..r * stride + uvw]);
        }
    }
}

/// Ensure tight reconstruction storage covers the final superblock row.
///
/// Returns `true` when visible-length vectors were extended. Callers pass that
/// result to [`truncate_packet_overlap_recon`] after the independently
/// scheduled luma and chroma passes have joined. Reference-pool storage is
/// already padded and therefore remains padded across the overlap.
pub(crate) fn pad_packet_overlap_recon(
    width: u32,
    height: u32,
    recon_y: &mut Vec<u8>,
    recon_u: &mut Vec<u8>,
    recon_v: &mut Vec<u8>,
) -> bool {
    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let storage_h = h.next_multiple_of(64);
    let visible_lengths = (w * h, uvw * (h / 2), uvw * (h / 2));
    let storage_lengths = (w * storage_h, uvw * (storage_h / 2), uvw * (storage_h / 2));
    let lengths = (recon_y.len(), recon_u.len(), recon_v.len());
    if lengths == visible_lengths {
        recon_y.resize(storage_lengths.0, 0);
        recon_u.resize(storage_lengths.1, 0);
        recon_v.resize(storage_lengths.2, 0);
        true
    } else {
        assert_eq!(lengths, storage_lengths);
        false
    }
}

/// Restore vectors extended by [`pad_packet_overlap_recon`] to visible length.
pub(crate) fn truncate_packet_overlap_recon(
    width: u32,
    height: u32,
    recon_y: &mut Vec<u8>,
    recon_u: &mut Vec<u8>,
    recon_v: &mut Vec<u8>,
    truncate: bool,
) {
    if !truncate {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    recon_y.truncate(w * h);
    recon_u.truncate(uvw * (h / 2));
    recon_v.truncate(uvw * (h / 2));
}

fn apply_packet_overlap_plane(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    mut apply_sb: impl FnMut(
        &LoopFilterMask,
        usize,
        usize,
        usize,
        &LoopFilterInfo,
        &rvp9_core::dsp::SimdDispatch,
    ),
) {
    assert!(filter_level > 0);
    let mi_rows = (height as usize).div_ceil(8);
    let mi_cols = width as usize / 8;
    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);
    let levels = resolve_filter_levels(
        filter_level,
        LIBVPX_DEFAULT_LF_DELTAS,
        SegmentationLfView::default(),
    );
    let info = LoopFilterInfo::new(/* sharpness_level = */ 0);
    let frame_mi_opt = complete_mi_grid_for_lf(mi_grid);
    let row_inputs = rvp9_core::loopfilter::apply::RowMaskInputs {
        mi_grid: &frame_mi_opt,
        mi_stride: mi_cols,
        mi_rows,
        mi_cols,
        levels: &levels,
    };
    let dispatch = rvp9_core::dsp::get();
    let mut row_lfm = vec![LoopFilterMask::default(); sb_cols];
    for sb_row in 0..sb_rows {
        rvp9_core::loopfilter::apply::build_lfm_row(sb_row, sb_cols, &row_inputs, &mut row_lfm);
        let mi_row = sb_row * SB_MI_SIZE;
        for (sb_col, lfm) in row_lfm.iter().enumerate() {
            apply_sb(lfm, mi_row, sb_col * SB_MI_SIZE, mi_rows, &info, &dispatch);
        }
    }
}

/// Apply only the luma half of the packet-overlapped YUV420 loop filter.
pub(crate) fn apply_packet_overlap_luma(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_y: &mut [u8],
) {
    let stride = width as usize;
    apply_packet_overlap_plane(
        width,
        height,
        filter_level,
        mi_grid,
        |lfm, mi_row, mi_col, mi_rows, info, dispatch| {
            rvp9_core::loopfilter::apply::apply_one_sb_luma(
                recon_y, stride, lfm, mi_row, mi_col, mi_rows, info, dispatch,
            );
        },
    );
}

/// Apply only the chroma half of the packet-overlapped YUV420 loop filter.
pub(crate) fn apply_packet_overlap_chroma(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) {
    let stride = width as usize / 2;
    apply_packet_overlap_plane(
        width,
        height,
        filter_level,
        mi_grid,
        |lfm, mi_row, mi_col, mi_rows, info, dispatch| {
            rvp9_core::loopfilter::apply::apply_one_sb_chroma_yuv420(
                recon_u, recon_v, stride, lfm, mi_row, mi_col, mi_rows, info, dispatch,
            );
        },
    );
}

/// Run the alias-safe split-edge loop filter directly on tight reconstruction
/// planes. The wrapper owns exclusive plane borrows for the complete call and
/// invokes exactly one core worker; callers may place this call on a scoped
/// thread while independent coefficient/packet work runs elsewhere.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_loop_filter_to_tight_recon_split_edge(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    num_tile_cols: usize,
) {
    use rvp9_core::loopfilter::apply::{ColumnWavefrontLfState, RowMaskInputs, SharedFrame};
    use std::sync::atomic::AtomicI32;

    assert!(filter_level > 0);
    let w = width as usize;
    let h = height as usize;
    let mi_rows = h / 8;
    let mi_cols = w / 8;
    let levels = resolve_filter_levels(
        filter_level,
        LIBVPX_DEFAULT_LF_DELTAS,
        SegmentationLfView::default(),
    );
    let info = LoopFilterInfo::new(/* sharpness_level = */ 0);
    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);
    let frame_mi_opt = complete_mi_grid_for_lf(mi_grid);
    let cur_sb_col: Vec<AtomicI32> = (0..sb_rows).map(|_| AtomicI32::new(-1)).collect();
    let shared = if height.is_multiple_of(64) {
        SharedFrame::from_tight_yuv420(recon_y, recon_u, recon_v, w, h)
    } else {
        SharedFrame::from_tight_yuv420_with_sb_tail(recon_y, recon_u, recon_v, w, h)
    };
    let state = ColumnWavefrontLfState {
        shared,
        info: &info,
        row_inputs: RowMaskInputs {
            mi_grid: &frame_mi_opt,
            mi_stride: mi_cols,
            mi_rows,
            mi_cols,
            levels: &levels,
        },
        non420: None,
        sb_rows,
        sb_cols,
        cur_sb_col: &cur_sb_col,
        nsync: rvp9_core::loopfilter::apply::nsync_for_width(width),
        decode_progress: None,
        num_tile_cols: num_tile_cols as i32,
        lf_delay: 1,
        abort: None,
    };
    let simd_dispatch = rvp9_core::dsp::get();
    // SAFETY: this safe wrapper holds exclusive access to every tight plane
    // until the single worker returns. `run_split_edge_lf_worker` asserts the
    // one-worker invariant, so no overlapping mutable slice can be formed.
    unsafe {
        rvp9_core::loopfilter::apply::run_split_edge_lf_worker(&state, 0, 1, &simd_dispatch);
    }
}

/// Run the split-edge filter on a partial-SB padded reconstruction while
/// preserving the caller's addressable tail bytes. The core kernels may write
/// into the final padded chroma rows even though those rows are outside the
/// visible MI grid; the bordered adapter leaves them untouched.
#[allow(clippy::too_many_arguments)]
fn apply_loop_filter_to_tight_recon_split_edge_preserving_tails(
    width: u32,
    height: u32,
    filter_level: u8,
    mi_grid: &[Option<ModeInfo>],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    num_tile_cols: usize,
) {
    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let storage_h = h.next_multiple_of(64);
    let visible_y_len = w * h;
    let visible_uv_len = uvw * uvh;
    let storage_lengths = (w * storage_h, uvw * (storage_h / 2), uvw * (storage_h / 2));
    assert_eq!(
        (recon_y.len(), recon_u.len(), recon_v.len()),
        storage_lengths,
        "partial split-edge filter requires SB-rounded backing storage"
    );

    let tail_y_len = storage_lengths.0 - visible_y_len;
    let tail_uv_len = storage_lengths.1 - visible_uv_len;
    let mut tails = Vec::with_capacity(tail_y_len + 2 * tail_uv_len);
    tails.extend_from_slice(&recon_y[visible_y_len..]);
    tails.extend_from_slice(&recon_u[visible_uv_len..]);
    tails.extend_from_slice(&recon_v[visible_uv_len..]);

    apply_loop_filter_to_tight_recon_split_edge(
        width,
        height,
        filter_level,
        mi_grid,
        recon_y,
        recon_u,
        recon_v,
        num_tile_cols,
    );

    let (tail_y, tail_uv) = tails.split_at(tail_y_len);
    let (tail_u, tail_v) = tail_uv.split_at(tail_uv_len);
    recon_y[visible_y_len..].copy_from_slice(tail_y);
    recon_u[visible_uv_len..].copy_from_slice(tail_u);
    recon_v[visible_uv_len..].copy_from_slice(tail_v);
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

fn complete_mi_grid_for_lf(mi_grid: &[Option<ModeInfo>]) -> Cow<'_, [Option<ModeInfo>]> {
    if mi_grid.iter().all(Option::is_some) {
        return Cow::Borrowed(mi_grid);
    }

    let placeholder = mi_grid
        .iter()
        .find_map(|cell| *cell)
        .unwrap_or_else(default_placeholder_mode_info);
    Cow::Owned(
        mi_grid
            .iter()
            .map(|cell| Some(cell.unwrap_or(placeholder)))
            .collect(),
    )
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

    let sb_cols = mi_cols_us.div_ceil(8);
    let sb_rows = mi_rows_us.div_ceil(8);
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let cell = ModeInfo {
                    block_size: leaf.leaf_bsize,
                    // Track-F iter8 — the LF mask must read the same per-block
                    // tx_size the decoder derives from the bitstream
                    // (`leaf.picked_tx_size`), or the encoder's in-loop LF
                    // diverges from the decoder's and seeds drift.
                    tx_size: leaf.picked_tx_size,
                    skip: leaf.skip,
                    segment_id: 0,
                    seg_id_predicted: false,
                    mode: BlockMode::Intra {
                        y_mode: leaf.parent_y_mode(),
                        uv_mode: IntraMode::Dc,
                    },
                    bmi: leaf.bmi(),
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
    let mi_rows = mi_grid.len() / mi_cols;
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let r = leaf_mi_row + dr;
            let c = leaf_mi_col + dc;
            if r < mi_rows && c < mi_cols {
                mi_grid[r * mi_cols + c] = Some(cell);
            }
        }
    }
}

// =====================================================================
// W7.4 — search_filter_level + try_filter_frame
//
// Scalar reference port of libvpx `vp9_picklpf.c:46-157`
// (`try_filter_frame` + `search_filter_level`). The SSE-driven binary
// search bracketing the previous frame's filter level. Mirrors the
// libvpx algorithm bit-for-bit; the kernel called inside
// `try_filter_frame` is rvp9's [`apply_loop_filter_to_tight_recon`]
// rather than libvpx's `vp9_loop_filter_frame`, but both produce the
// same decoder-equivalent post-LF pixels under the encoder's
// "zero LF deltas + sharpness 0" header shape.
//
// Out of scope: SIMD (`vpx_get_y_sse_neon` etc), HBD
// (`vpx_highbd_get_y_sse`), and the libvpx `partial_frame` mode
// (which only filters one MB row — unused outside libvpx's lpf
// regression tests). Section-intra biasing is included with a
// caller-supplied rating — see [`SearchFilterLevelParams`].
// =====================================================================

/// Sum of squared error between two tight-stride 8-bit luma planes.
/// Mirrors libvpx `vpx_get_y_sse` for one plane only — the picker
/// uses luma-only SSE as its quality metric (`vp9_picklpf.c:66-69`).
#[allow(dead_code)]
fn y_sse_tight(a: &[u8], b: &[u8]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: i64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i32::from(*x) - i32::from(*y);
        acc += i64::from(d * d);
    }
    acc
}

/// Inputs to the LF search. The caller owns the unfiltered recon and
/// the source frame; the search probes candidate `filter_level`s by
/// re-running [`apply_loop_filter_to_tight_recon`] on a private copy
/// per candidate. The unfiltered recon is left untouched.
#[allow(dead_code)]
pub struct SearchFilterLevelParams<'a> {
    pub width: u32,
    pub height: u32,
    pub mi_grid: &'a [Option<ModeInfo>],
    /// Tight-stride 8-bit unfiltered Y plane (encoder reconstruction
    /// pre-LF).
    pub recon_y: &'a [u8],
    /// Tight-stride 8-bit unfiltered U plane.
    pub recon_u: &'a [u8],
    /// Tight-stride 8-bit unfiltered V plane.
    pub recon_v: &'a [u8],
    /// Tight-stride 8-bit source luma plane (the per-frame target).
    pub source_y: &'a [u8],
    /// `cpi->common.lf.last_filt_level` — the encoder's previous
    /// frame's filter level. Mirrors libvpx `lf->last_filt_level`
    /// (`vp9_picklpf.c:90`). On the first frame, callers pass `0`.
    pub last_filt_level: u8,
    /// Max filter level the picker may pick. Mirrors libvpx
    /// `get_max_filter_level` output (`vp9_picklpf.c:30-44`).
    /// Callers typically pass `MAX_LOOP_FILTER = 63`; pass a lower
    /// cap for two-pass intra-rich sections.
    pub max_filter_level: u8,
    /// True when `cpi->common.tx_mode != ONLY_4X4`. Drives libvpx's
    /// `bias >>= 1` step at `vp9_picklpf.c:117`.
    pub tx_mode_not_only_4x4: bool,
}

/// Inputs to [`try_filter_frame`].
#[allow(dead_code)]
struct TryFilterInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub mi_grid: &'a [Option<ModeInfo>],
    pub recon_y: &'a [u8],
    pub recon_u: &'a [u8],
    pub recon_v: &'a [u8],
    pub source_y: &'a [u8],
}

/// Run the loop filter at `filt_level` on a private copy of the
/// recon, return the luma SSE between the post-LF luma plane and the
/// source. Mirrors libvpx `try_filter_frame` (`vp9_picklpf.c:46-76`).
///
/// libvpx's version mutates `cm->frame_to_show` and then restores it
/// from `cpi->last_frame_uf`; we sidestep that by cloning the input
/// recon before each apply. The wire output is identical.
#[allow(dead_code)]
fn try_filter_frame(inputs: &TryFilterInputs<'_>, filt_level: u8) -> i64 {
    if filt_level == 0 {
        // libvpx `vp9_loopfilter.c:1422-1427` early-out: LF skipped.
        // Post-LF Y plane equals unfiltered Y plane.
        return y_sse_tight(inputs.recon_y, inputs.source_y);
    }

    // Clone all three planes (the LF kernel mutates U/V too).
    let mut y = inputs.recon_y.to_vec();
    let mut u = inputs.recon_u.to_vec();
    let mut v = inputs.recon_v.to_vec();

    apply_loop_filter_to_tight_recon(
        inputs.width,
        inputs.height,
        filt_level,
        inputs.mi_grid,
        &mut y,
        &mut u,
        &mut v,
    );

    y_sse_tight(&y, inputs.source_y)
}

/// `clamp(v, lo, hi)` saturating to the inclusive range.
#[allow(dead_code)]
#[inline]
fn clamp_u8(v: u8, lo: u8, hi: u8) -> u8 {
    v.clamp(lo, hi)
}

/// Binary search for the SSE-minimising `filter_level`. Mirrors
/// libvpx `search_filter_level` (`vp9_picklpf.c:78-157`).
///
/// Starts at `last_filt_level` (clamped to
/// `[0, max_filter_level]`), brackets with a step that halves when
/// the centre value wins. libvpx's "bias against raising LF" at
/// `vp9_picklpf.c:111-117` is mirrored here (with the
/// `section_intra_rating` knob folded into `max_filter_level` —
/// see `get_max_filter_level`, `vp9_picklpf.c:30-44`).
#[allow(dead_code)]
#[must_use]
pub fn search_filter_level(params: &SearchFilterLevelParams<'_>) -> u8 {
    let min_filter_level = 0u8;
    let max_filter_level = params.max_filter_level.min(MAX_LOOP_FILTER);

    let try_inputs = TryFilterInputs {
        width: params.width,
        height: params.height,
        mi_grid: params.mi_grid,
        recon_y: params.recon_y,
        recon_u: params.recon_u,
        recon_v: params.recon_v,
        source_y: params.source_y,
    };

    // libvpx `vp9_picklpf.c:90`: start at last_filt_level (clamped).
    let mut filt_mid = clamp_u8(params.last_filt_level, min_filter_level, max_filter_level);
    let mut filter_step: i32 = if filt_mid < 16 {
        4
    } else {
        (filt_mid / 4) as i32
    };

    // libvpx uses `int64_t ss_err[MAX_LOOP_FILTER + 1]` initialised
    // to -1 (memset 0xFF). We track "unset" with `Option`.
    let mut ss_err: [Option<i64>; (MAX_LOOP_FILTER as usize) + 1] =
        [None; (MAX_LOOP_FILTER as usize) + 1];

    // libvpx `vp9_picklpf.c:102`: prime with the centre value.
    let mut best_err = try_filter_frame(&try_inputs, filt_mid);
    let mut filt_best = filt_mid;
    ss_err[filt_mid as usize] = Some(best_err);

    let mut filt_direction: i32 = 0;

    while filter_step > 0 {
        let filt_high = ((filt_mid as i32) + filter_step).min(max_filter_level as i32) as u8;
        let filt_low = ((filt_mid as i32) - filter_step).max(min_filter_level as i32) as u8;

        // libvpx `vp9_picklpf.c:111`: bias against raising LF.
        let mut bias: i64 = (best_err >> (15 - (filt_mid as i32 / 8))) * filter_step as i64;
        if params.tx_mode_not_only_4x4 {
            bias >>= 1; // libvpx `vp9_picklpf.c:117`.
        }

        // Lower-side probe.
        if filt_direction <= 0 && filt_low != filt_mid {
            if ss_err[filt_low as usize].is_none() {
                ss_err[filt_low as usize] = Some(try_filter_frame(&try_inputs, filt_low));
            }
            let lo_err = ss_err[filt_low as usize].unwrap();
            if lo_err - bias < best_err {
                if lo_err < best_err {
                    best_err = lo_err;
                }
                filt_best = filt_low;
            }
        }

        // Upper-side probe.
        if filt_direction >= 0 && filt_high != filt_mid {
            if ss_err[filt_high as usize].is_none() {
                ss_err[filt_high as usize] = Some(try_filter_frame(&try_inputs, filt_high));
            }
            let hi_err = ss_err[filt_high as usize].unwrap();
            if hi_err < best_err - bias {
                best_err = hi_err;
                filt_best = filt_high;
            }
        }

        if filt_best == filt_mid {
            filter_step /= 2;
            filt_direction = 0;
        } else {
            filt_direction = if filt_best < filt_mid { -1 } else { 1 };
            filt_mid = filt_best;
        }
    }

    filt_best
}

#[cfg(test)]
mod w7_4_tests {
    use super::*;
    use rvp9_core::bitstream::InterpFilter;
    use rvp9_core::mode_info::{InterMode, Mv, RefFrame};

    #[test]
    fn complete_filter_grid_borrows_populated_input_and_repairs_holes() {
        let populated = build_intra_mi_grid(8, 8);
        assert!(matches!(
            complete_mi_grid_for_lf(&populated),
            Cow::Borrowed(_)
        ));

        let expected = populated[0];
        let mut incomplete = populated;
        incomplete[7] = None;
        incomplete[63] = None;
        let completed = complete_mi_grid_for_lf(&incomplete);
        assert!(matches!(completed, Cow::Owned(_)));
        assert_eq!(completed[7], expected);
        assert_eq!(completed[63], expected);

        let empty = vec![None; 4];
        let completed_empty = complete_mi_grid_for_lf(&empty);
        assert!(matches!(completed_empty, Cow::Owned(_)));
        assert!(completed_empty.iter().all(Option::is_some));
    }

    /// Build a contiguous block of frame data and a paired mi_grid
    /// for an `mi_rows × mi_cols` (in 8×8 units) frame. Every mi
    /// cell is its own Block8x8 intra DC leaf, so the LF gets one
    /// filter edge on every 8-pixel boundary — that's the density
    /// needed for the synthetic step-edge test to register a SSE
    /// improvement.
    fn build_intra_mi_grid(mi_rows: usize, mi_cols: usize) -> Vec<Option<ModeInfo>> {
        let cell = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: max_txsize(BlockSize::Block8x8),
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        vec![Some(cell); mi_rows * mi_cols]
    }

    /// Build a complete inter-frame MI grid with distinct mode/reference
    /// combinations. The loop-filter mask depends on the inter reference and
    /// mode deltas, so this keeps the differential fixture from accidentally
    /// proving only the all-intra path.
    fn build_mixed_inter_mi_grid(mi_rows: usize, mi_cols: usize) -> Vec<Option<ModeInfo>> {
        (0..mi_rows)
            .flat_map(|mi_row| {
                (0..mi_cols).map(move |mi_col| {
                    let variant = (mi_row * 3 + mi_col) % 4;
                    let (mode, ref_frames, mvs) = match variant {
                        0 => (
                            InterMode::ZeroMv,
                            [RefFrame::LastFrame, RefFrame::NoRefFrame],
                            [Mv::ZERO, Mv::ZERO],
                        ),
                        1 => (
                            InterMode::NearestMv,
                            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
                            [Mv { row: 0, col: 8 }, Mv::ZERO],
                        ),
                        2 => (
                            InterMode::NearMv,
                            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
                            [Mv { row: -8, col: 0 }, Mv::ZERO],
                        ),
                        _ => (
                            InterMode::NewMv,
                            [RefFrame::LastFrame, RefFrame::NoRefFrame],
                            [Mv { row: 8, col: -8 }, Mv::ZERO],
                        ),
                    };
                    Some(ModeInfo {
                        block_size: BlockSize::Block8x8,
                        tx_size: max_txsize(BlockSize::Block8x8),
                        skip: variant == 0,
                        segment_id: ((mi_row + mi_col) as u8) & 1,
                        seg_id_predicted: false,
                        mode: BlockMode::Inter {
                            mode,
                            ref_frames,
                            mvs,
                            interp_filter: if variant == 3 {
                                InterpFilter::EightTapSmooth
                            } else {
                                InterpFilter::EightTap
                            },
                        },
                        bmi: Default::default(),
                    })
                })
            })
            .collect()
    }

    /// Fill visible rows with small, filterable luma/chroma steps and retain
    /// deliberately distinct non-zero poison in the padded SB-row tails.
    fn padded_inter_recon_fixture(
        width: usize,
        height: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<Option<ModeInfo>>) {
        let storage_height = height.next_multiple_of(64);
        let uv_width = width / 2;
        let uv_visible_height = height / 2;
        let uv_storage_height = storage_height / 2;
        let mut y = vec![0u8; width * storage_height];
        let mut u = vec![0u8; uv_width * uv_storage_height];
        let mut v = vec![0u8; uv_width * uv_storage_height];

        for row in 0..height {
            for col in 0..width {
                let step = (((row / 8) + (col / 8)) & 1) as u8;
                y[row * width + col] = 112 + step * 12;
            }
        }
        for row in 0..uv_visible_height {
            for col in 0..uv_width {
                let step = (((row / 4) + (col / 4)) & 1) as u8;
                u[row * uv_width + col] = 96 + step * 12;
                v[row * uv_width + col] = 160 - step * 12;
            }
        }
        for (index, value) in y[width * height..].iter_mut().enumerate() {
            *value = 0x51u8.wrapping_add(index as u8 % 37);
        }
        for (index, value) in u[uv_width * uv_visible_height..].iter_mut().enumerate() {
            *value = 0xa2u8.wrapping_add(index as u8 % 29);
        }
        for (index, value) in v[uv_width * uv_visible_height..].iter_mut().enumerate() {
            *value = 0xe3u8.wrapping_add(index as u8 % 23);
        }

        let mi_grid = build_mixed_inter_mi_grid(height / 8, width / 8);
        assert!(mi_grid.iter().flatten().any(|mi| {
            matches!(
                mi.mode,
                BlockMode::Inter {
                    mode: InterMode::NewMv,
                    ..
                }
            )
        }));
        assert!(mi_grid.iter().flatten().any(|mi| {
            matches!(
                mi.mode,
                BlockMode::Inter {
                    ref_frames: [RefFrame::AltrefFrame, ..],
                    ..
                }
            )
        }));
        (y, u, v, mi_grid)
    }

    /// Generate a noisy recon + a clean (smooth) source. The encoder
    /// recon has small step edges at every 8-pixel column boundary
    /// that the loop filter can smooth toward the source. Step size
    /// is intentionally modest (±5 around the mean of 120) — the
    /// VP9 LF mask check rejects large `|p0 - q0|` differences, so
    /// the synthetic edges must sit inside `blimit` for the target
    /// `filter_level` to fire. `(|p0-q0|*2 + |p1-q1|/2) ≤ blimit`
    /// at `level=30` ⇒ blimit ≈ 24 ⇒ step ≤ 12 is the safe ceiling.
    fn noisy_64x64_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let w = 64usize;
        let h = 64usize;
        let mut recon_y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                // ±5 alternation in 8-pixel columns. The 8-pixel
                // period matches the SB's 8×8 partition leaves so
                // the LF kernel has one edge to attack at every
                // column boundary.
                recon_y[r * w + c] = if (c / 8) % 2 == 0 { 115 } else { 125 };
            }
        }
        let recon_u = vec![120u8; (w / 2) * (h / 2)];
        let recon_v = vec![120u8; (w / 2) * (h / 2)];
        // Source: solid 120 — the mean. LF smoothing should bring
        // recon closer to source.
        let source_y = vec![120u8; w * h];
        (recon_y, recon_u, recon_v, source_y)
    }

    #[test]
    fn split_edge_parallel_filter_matches_serial_planes_bit_exact() {
        for (w, h, filter_level) in [(64usize, 64usize, 63u8), (256, 128, 1), (1280, 704, 30)] {
            let mi_grid = build_intra_mi_grid(h / 8, w / 8);
            let mut source_y = vec![0u8; w * h];
            for r in 0..h {
                for c in 0..w {
                    source_y[r * w + c] = 120 + (((r / 8 + c / 8) & 1) as u8) * 10;
                }
            }
            let uvw = w / 2;
            let uvh = h / 2;
            let mut source_u = vec![0u8; uvw * uvh];
            let mut source_v = vec![0u8; uvw * uvh];
            for r in 0..uvh {
                for c in 0..uvw {
                    let step = (((r / 4 + c / 4) & 1) as u8) * 10;
                    source_u[r * uvw + c] = 115 + step;
                    source_v[r * uvw + c] = 125 - step;
                }
            }

            let mut serial_y = source_y.clone();
            let mut serial_u = source_u.clone();
            let mut serial_v = source_v.clone();
            apply_loop_filter_to_tight_recon(
                w as u32,
                h as u32,
                filter_level,
                &mi_grid,
                &mut serial_y,
                &mut serial_u,
                &mut serial_v,
            );

            let mut parallel_y = source_y.clone();
            let mut parallel_u = source_u.clone();
            let mut parallel_v = source_v.clone();
            let mut pool = rvp9_core::tile_threading::WorkerPool::new(
                std::num::NonZeroUsize::new(4).expect("four is nonzero"),
            );
            apply_loop_filter_to_tight_recon_parallel(
                w as u32,
                h as u32,
                filter_level,
                &mi_grid,
                &mut parallel_y,
                &mut parallel_u,
                &mut parallel_v,
                Some(&mut pool),
                4,
                4,
            );

            if filter_level == 30 {
                assert_ne!(serial_y, source_y, "fixture must exercise luma filtering");
            }
            assert_eq!(parallel_y, serial_y, "luma differs at {w}x{h}");
            assert_eq!(parallel_u, serial_u, "U differs at {w}x{h}");
            assert_eq!(parallel_v, serial_v, "V differs at {w}x{h}");
        }
    }

    #[test]
    fn serial_split_edge_filter_matches_bordered_fallback_for_padded_inter_recon() {
        let mut backing_mismatches = Vec::new();
        for (width, height) in [(1920usize, 1080usize), (320, 72)] {
            let (original_y, original_u, original_v, mi_grid) =
                padded_inter_recon_fixture(width, height);
            let mut direct_y = original_y.clone();
            let mut direct_u = original_u.clone();
            let mut direct_v = original_v.clone();
            apply_loop_filter_to_tight_recon_parallel(
                width as u32,
                height as u32,
                30,
                &mi_grid,
                &mut direct_y,
                &mut direct_u,
                &mut direct_v,
                None,
                1,
                1,
            );

            // Force the bordered reference path with visible-only vectors,
            // then put its filtered visible rows back beside the original
            // poisoned tails. Full backing-vector equality below therefore
            // checks both filter output and direct-path tail discipline.
            let mut fallback_y = original_y.clone();
            let mut fallback_u = original_u.clone();
            let mut fallback_v = original_v.clone();
            let mut visible_y = fallback_y[..width * height].to_vec();
            let mut visible_u = fallback_u[..(width / 2) * (height / 2)].to_vec();
            let mut visible_v = fallback_v[..(width / 2) * (height / 2)].to_vec();
            apply_loop_filter_to_tight_recon(
                width as u32,
                height as u32,
                30,
                &mi_grid,
                &mut visible_y,
                &mut visible_u,
                &mut visible_v,
            );
            fallback_y[..width * height].copy_from_slice(&visible_y);
            fallback_u[..(width / 2) * (height / 2)].copy_from_slice(&visible_u);
            fallback_v[..(width / 2) * (height / 2)].copy_from_slice(&visible_v);

            assert_ne!(
                &direct_y[..width * height],
                &original_y[..width * height],
                "{width}x{height} fixture must exercise luma filtering"
            );
            assert_ne!(
                &direct_u[..(width / 2) * (height / 2)],
                &original_u[..(width / 2) * (height / 2)],
                "{width}x{height} fixture must exercise U filtering"
            );
            assert_ne!(
                &direct_v[..(width / 2) * (height / 2)],
                &original_v[..(width / 2) * (height / 2)],
                "{width}x{height} fixture must exercise V filtering"
            );
            let first_y_diff = direct_y
                .iter()
                .zip(&fallback_y)
                .position(|(direct, fallback)| direct != fallback);
            let first_u_diff = direct_u
                .iter()
                .zip(&fallback_u)
                .position(|(direct, fallback)| direct != fallback);
            let first_v_diff = direct_v
                .iter()
                .zip(&fallback_v)
                .position(|(direct, fallback)| direct != fallback);
            if (first_y_diff, first_u_diff, first_v_diff) != (None, None, None) {
                backing_mismatches.push((width, height, first_y_diff, first_u_diff, first_v_diff));
            }
        }
        assert!(
            backing_mismatches.is_empty(),
            "complete backing differs: {backing_mismatches:?}"
        );
    }

    #[test]
    fn split_packet_overlap_filter_matches_serial() {
        for (w, h) in [(64usize, 72usize), (128, 128)] {
            let filter_level = 30u8;
            let mi_grid = build_intra_mi_grid(h / 8, w / 8);
            let mut source_y = vec![0u8; w * h];
            for r in 0..h {
                for c in 0..w {
                    source_y[r * w + c] = 115 + (((r / 8 + c / 8) & 1) as u8) * 10;
                }
            }
            let source_u = vec![120u8; (w / 2) * (h / 2)];
            let source_v = vec![120u8; (w / 2) * (h / 2)];

            let mut serial_y = source_y.clone();
            let mut serial_u = source_u.clone();
            let mut serial_v = source_v.clone();
            apply_loop_filter_to_tight_recon(
                w as u32,
                h as u32,
                filter_level,
                &mi_grid,
                &mut serial_y,
                &mut serial_u,
                &mut serial_v,
            );

            let mut overlap_y = source_y.clone();
            let mut overlap_u = source_u.clone();
            let mut overlap_v = source_v.clone();
            let truncate_overlap_recon = pad_packet_overlap_recon(
                w as u32,
                h as u32,
                &mut overlap_y,
                &mut overlap_u,
                &mut overlap_v,
            );
            apply_packet_overlap_luma(w as u32, h as u32, filter_level, &mi_grid, &mut overlap_y);
            apply_packet_overlap_chroma(
                w as u32,
                h as u32,
                filter_level,
                &mi_grid,
                &mut overlap_u,
                &mut overlap_v,
            );
            truncate_packet_overlap_recon(
                w as u32,
                h as u32,
                &mut overlap_y,
                &mut overlap_u,
                &mut overlap_v,
                truncate_overlap_recon,
            );

            assert_ne!(serial_y, source_y, "fixture must exercise luma filtering");
            assert_eq!(overlap_y, serial_y, "luma differs at {w}x{h}");
            assert_eq!(overlap_u, serial_u, "U differs at {w}x{h}");
            assert_eq!(overlap_v, serial_v, "V differs at {w}x{h}");
        }
    }

    #[test]
    fn packet_overlap_preserves_existing_reference_storage_lengths() {
        let (w, h) = (64usize, 72usize);
        let storage_h = h.next_multiple_of(64);
        let mut y = vec![0u8; w * storage_h];
        let mut u = vec![0u8; (w / 2) * (storage_h / 2)];
        let mut v = u.clone();

        let truncate = pad_packet_overlap_recon(w as u32, h as u32, &mut y, &mut u, &mut v);
        assert!(!truncate);
        truncate_packet_overlap_recon(w as u32, h as u32, &mut y, &mut u, &mut v, truncate);

        assert_eq!(y.len(), w * storage_h);
        assert_eq!(u.len(), (w / 2) * (storage_h / 2));
        assert_eq!(v.len(), (w / 2) * (storage_h / 2));
    }

    /// W7.4 PROOF — `search_filter_level` (`vp9_picklpf.c:78-157`).
    ///
    /// Set up a frame where the unfiltered recon is noisy step edges
    /// and the source is the recon mean. The LF kernel smooths step
    /// edges, so `filter_level > 0` must lower SSE vs. the
    /// `filter_level = 0` baseline. The binary search must therefore
    /// pick a non-zero level.
    #[test]
    fn w7_4_search_filter_level_finds_lower_sse_level() {
        let (recon_y, recon_u, recon_v, source_y) = noisy_64x64_frame();
        let mi_grid = build_intra_mi_grid(8, 8);
        let params = SearchFilterLevelParams {
            width: 64,
            height: 64,
            mi_grid: &mi_grid,
            recon_y: &recon_y,
            recon_u: &recon_u,
            recon_v: &recon_v,
            source_y: &source_y,
            last_filt_level: 20,
            max_filter_level: MAX_LOOP_FILTER,
            tx_mode_not_only_4x4: false,
        };
        let level = search_filter_level(&params);
        assert!(
            level > 0,
            "search must pick a non-zero filter level on noisy input (got {level})"
        );

        // Differential: SSE at picked level <= SSE at level 0.
        let try_inputs = TryFilterInputs {
            width: 64,
            height: 64,
            mi_grid: &mi_grid,
            recon_y: &recon_y,
            recon_u: &recon_u,
            recon_v: &recon_v,
            source_y: &source_y,
        };
        let sse_zero = try_filter_frame(&try_inputs, 0);
        let sse_picked = try_filter_frame(&try_inputs, level);
        assert!(
            sse_picked <= sse_zero,
            "picked level {level} should not be worse than 0 (SSE {} vs {})",
            sse_picked,
            sse_zero
        );
    }

    /// W7.4 PROOF — `try_filter_frame` (`vp9_picklpf.c:46-76`).
    ///
    /// Sane input (smooth source, smooth recon) returns a usable
    /// non-negative SSE. We also assert that `filt_level == 0`
    /// returns exactly `y_sse_tight(recon_y, source_y)` — i.e. the
    /// libvpx early-out skips the LF kernel.
    #[test]
    fn w7_4_try_filter_frame_returns_sse() {
        let (recon_y, recon_u, recon_v, source_y) = noisy_64x64_frame();
        let mi_grid = build_intra_mi_grid(8, 8);
        let try_inputs = TryFilterInputs {
            width: 64,
            height: 64,
            mi_grid: &mi_grid,
            recon_y: &recon_y,
            recon_u: &recon_u,
            recon_v: &recon_v,
            source_y: &source_y,
        };
        // Level-0 path: must equal raw SSE.
        let raw_sse = y_sse_tight(&recon_y, &source_y);
        let sse_zero = try_filter_frame(&try_inputs, 0);
        assert_eq!(
            sse_zero, raw_sse,
            "filt_level=0 must skip LF and return raw SSE"
        );
        assert!(sse_zero >= 0, "SSE must be non-negative");

        // Level-30 path: non-zero filter. Returns a usable
        // (non-negative) SSE. With the noisy step-edge recon and
        // smooth source, filtering should narrow the gap.
        let sse_filt = try_filter_frame(&try_inputs, 30);
        assert!(sse_filt >= 0, "filtered SSE must be non-negative");
        assert!(
            sse_filt < sse_zero,
            "filtered SSE ({sse_filt}) should be less than unfiltered ({sse_zero}) on step-edge input"
        );
    }
}
