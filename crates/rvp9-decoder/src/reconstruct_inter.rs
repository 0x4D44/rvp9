//! Per-inter-block reconstruction driver.
//!
//! Ports libvpx `dec_build_inter_predictors_sb` +
//! `build_inter_predictors` (`vp9/common/vp9_reconinter.c:126-207`) for
//! the decoder side. For one inter coding block at `(mi_row, mi_col)`
//! with [`ModeInfo`]-derived MV(s) / refs / filter, this driver:
//!
//! 1. Looks up each active reference frame via its
//!    `header.ref_frame_indices[i]` slot in the [`DecoderFramePool`].
//! 2. Builds a [`ScaleFactors`] for each ref against the current frame's
//!    dimensions (libvpx `vp9_setup_scale_factors_for_frame` at
//!    `vp9_decodeframe.c:1559-1606`).
//! 3. For each plane (Y, U, V) and each TX-sized sub-region inside the
//!    coding block, computes the plane-relative pixel origin, scales
//!    the MV into the ref's coord system, and calls
//!    [`inter_predictor`].
//! 4. If `mi.skip == false`, decodes the residual coefficients per TX
//!    position via [`decode_block_tokens`] and applies the IT-add
//!    on top of the MC output.
//!
//! ## Staged delivery
//!
//! - **D.2.a (this commit)** — single-ref, non-skip, `bsize >= 8×8`,
//!   no compound. Validates the ref-pool lookup / scale-factors /
//!   per-plane MC loop / residual decode wiring against a synthetic
//!   reference frame.
//! - **D.2.b** — `mi.skip == true`: MC output is the final
//!   reconstruction (no residual).
//! - **D.2.c** — compound prediction (both refs valid): first ref via
//!   [`inter_predictor`] into dst, second ref via averaging.
//! - **D.2.d** — sub-8 inter blocks (`Block4x4` / `Block4x8` /
//!   `Block8x4`): per-bmi MC.
//! - **D.2.e** — integration into [`crate::tile::decode_block`].
//!
//! ## Coordinate system
//!
//! All MV reads happen via the with-border view exposed by
//! [`Yv12Buffer::plane_with_border`]. The integer-pel base of each
//! motion-compensated sub-block can therefore fall up to `border`
//! pixels outside the visible region without faulting, provided
//! [`Yv12Buffer::extend_borders`] has been called on the reference
//! frame first.

use rvp9_core::bitstream::{ChromaFormat, InterpFilter};
use rvp9_core::block::{BlockSize, uv_plane_block_size, uv_plane_tx_size};
use rvp9_core::dsp::SimdDispatch;
use rvp9_core::entropy::{BoolReader, FrameContext};
use rvp9_core::entropy::adapt::CoefCountsBank;
use rvp9_core::frame::{BorderedPlaneRef, Yv12Buffer};
use rvp9_core::inter::mode_info::{InterModeReadParams, read_inter_block_mode_info};
use rvp9_core::inter::mv_ref::{BlockBounds, NeighbourLookup, PrevFrameMvRef};
use rvp9_core::inter::scale::ScaleFactors;
use rvp9_core::inter::filter::{SUBPEL_BITS, SUBPEL_MASK, SUBPEL_SHIFTS};
use rvp9_core::mode_info::{BlockMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::inverse::{
    idct4x4_1_add, idct8x8_1_add, idct16x16_1_add,
    idct32x32_1_add, iwht4x4_1_add, iwht4x4_16_add,
};
use rvp9_core::transforms::types::{TxSize, TxType};

use crate::detokenize::decode_block_tokens;
use crate::tile_ctx::TileCoefCtx;

/// Maximum coefficient count at any TX size (`Tx32x32 = 32 * 32 = 1024`).
const MAX_TX_COEFFS: usize = 1024;

/// Per-plane chroma subsampling derived from a [`ChromaFormat`].
/// Plane 0 (luma) is always full-resolution `(0, 0)`; chroma planes
/// take the format's `(subsampling_x, subsampling_y)`. Mirrors
/// libvpx's per-plane `pd->subsampling_{x,y}` setup. At
/// [`ChromaFormat::Yuv420`] this yields the historical `[0, 1, 1]`
/// arrays exactly.
#[inline]
fn plane_ss(chroma: ChromaFormat, plane: usize) -> (usize, usize) {
    if plane == 0 {
        (0, 0)
    } else {
        (chroma.subsampling_x() as usize, chroma.subsampling_y() as usize)
    }
}

/// Inputs that don't vary across the per-plane / per-TX loop inside
/// [`reconstruct_inter_block`]. Bundled to keep the per-block driver
/// signature compact.
#[allow(missing_docs)]
pub struct InterBlockInputs<'a> {
    /// The current frame's raw (crop / un-aligned) luma width (pixels)
    /// — used to build [`ScaleFactors`] for each ref. D.24: matches
    /// libvpx `cm->width` fed to `vp9_setup_scale_factors_for_frame`
    /// (`vp9_decodeframe.c:2928-2935`), NOT the aligned dim.
    pub cur_y_width: u32,
    /// The current frame's raw (crop / un-aligned) luma height (pixels).
    /// See `cur_y_width`.
    pub cur_y_height: u32,
    /// Per-ref reference frames. `refs[i]` is `Some(&buf)` if the
    /// corresponding `mi.ref_frames[i]` is an inter ref; `None` if it's
    /// `NoRefFrame`. Caller resolves the ref-pool lookup beforehand.
    pub refs: [Option<&'a Yv12Buffer>; 2],
    /// `(dc, ac)` dequantisers per plane (Y, U, V) for this block's
    /// segment.
    pub dequant_per_plane: [(i32, i32); 3],
    /// `true` ⇒ lossless mode (forces `TxType::DctDct`).
    pub lossless: bool,
    /// Frame-wide mi rows. Used for partial-bottom / partial-right edge
    /// clamping of the residual decode loop (D.25).
    pub mi_rows: usize,
    /// Frame-wide mi cols. See `mi_rows`.
    pub mi_cols: usize,
}

/// Reconstruct one inter coding block at frame-pixel origin
/// `(mi_row * 8, mi_col * 8)`. Mirrors libvpx
/// `dec_build_inter_predictors_sb` + the per-block residual path at
/// `vp9_decodeframe.c:455-505`.
///
/// # Parameters
///
/// - `reader`: bool-coder positioned at the first token bit of this
///   block. Advances by however many bits the per-plane residual
///   decode consumes. `mi.skip == true` ⇒ reader is not advanced
///   (Stage D.2.b).
/// - `fc`: per-frame probability bank.
/// - `coef_counts`: optional [`CoefCountsBank`] for backward-adaptation
///   accumulation. Stage 3 Task E.1.b wires this through the inter
///   recon residual decode loop; pass `None` to disable count
///   accumulation (e.g. in unit tests).
/// - `coef_ctx`: per-tile coef-context state for residual decode.
/// - `mi`: the [`ModeInfo`] resolved by the per-block mode reader. The
///   driver requires `mi.is_inter()`.
/// - `mi_row`, `mi_col`: tile-relative coding-block position in 8×8
///   MI-units (these must equal the frame-absolute position when
///   `tile_cols == tile_rows == 1`, the only multi-tile arrangement
///   the v1 floor supports — see [`crate::tile::TileDecodeError::MultiTileUnsupported`]).
/// - `mi_row_in_tile`, `mi_col_in_tile`: tile-relative (used by
///   the per-plane coef-context indices).
/// - `inputs`: shared per-frame / per-block parameters (see
///   [`InterBlockInputs`]).
/// - `dst_frame`: the current frame's destination buffer; the per-plane
///   MC writes land here, residual IT-add applies on top.
///
/// # Panics
///
/// - If `mi.is_inter() == false` (caller routing error).
/// - If `inputs.refs[i] == None` for any `i` where `mi.ref_frames[i]`
///   is an inter ref (caller must resolve before invocation).
/// - Inherits the assertions in [`decode_block_tokens`] and
///   [`inter_predictor`].
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
pub fn reconstruct_inter_block(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    mut coef_counts: Option<&mut CoefCountsBank>,
    coef_ctx: &mut TileCoefCtx,
    mi: &ModeInfo,
    mi_row: usize,
    mi_col: usize,
    _mi_row_in_tile: usize,
    mi_col_in_tile: usize,
    mi_row_in_sb: usize,
    inputs: &InterBlockInputs<'_>,
    dst_frame: &mut Yv12Buffer,
    dispatch: &SimdDispatch,
) {
    // G7.2: the per-plane MC / residual loops are `ss`-aware for all
    // four legal VP9 chroma formats (G7.1a parametric geometry / G7.1b
    // inter MC / G7.2 non420 loopfilter). Debug-only sanity check.
    debug_assert!(
        matches!(
            dst_frame.chroma(),
            ChromaFormat::Yuv420
                | ChromaFormat::Yuv422
                | ChromaFormat::Yuv440
                | ChromaFormat::Yuv444
        ),
        "reconstruct_inter_block: unexpected chroma {:?}",
        dst_frame.chroma(),
    );
    let (mode, ref_frames, mvs, interp_filter) = match mi.mode {
        BlockMode::Inter {
            mode,
            ref_frames,
            mvs,
            interp_filter,
        } => (mode, ref_frames, mvs, interp_filter),
        BlockMode::Intra { .. } => {
            panic!("reconstruct_inter_block called on intra block");
        }
    };
    let _ = mode; // mode not consumed in D.2.a (no per-bmi yet)

    let is_sub8 = (mi.block_size as u8) < (BlockSize::Block8x8 as u8);

    // Active references and per-ref scale factors. `is_compound` =>
    // both refs valid (libvpx `has_second_ref`). Single-ref blocks
    // only iterate ref 0.
    debug_assert!(ref_frames[0].is_inter(), "ref[0] must be inter");
    let is_compound = ref_frames[1].is_inter();

    let mut sfs: [Option<ScaleFactors>; 2] = [None, None];
    let num_refs = 1 + usize::from(is_compound);
    #[allow(clippy::needless_range_loop)] // indexes both `inputs.refs` and `sfs`
    for r in 0..num_refs {
        let ref_buf = inputs.refs[r]
            .expect("caller must resolve ref-pool slot for each active ref");
        // D.24: use crop (visible) dims, not aligned dims, to match
        // libvpx `vp9_setup_scale_factors_for_frame` which is called
        // with `ref_buf->buf->y_crop_width / y_crop_height` against
        // `cm->width / cm->height` (the raw, un-aligned dims) — see
        // `vp9_decodeframe.c:2928-2935`. For frames whose dims are
        // 8-multiples the two coincide; for resize vectors (e.g.
        // 282×173) they differ and the scale factor mismatches.
        let sf = ScaleFactors::new(
            ref_buf.width(),
            ref_buf.height(),
            inputs.cur_y_width,
            inputs.cur_y_height,
        );
        debug_assert!(
            sf.is_valid(),
            "ref dimensions must be in libvpx envelope"
        );
        sfs[r] = Some(sf);
    }

    let bsize = mi.block_size;
    let chroma = dst_frame.chroma();
    for plane in 0..3 {
        let (ss_x, ss_y) = plane_ss(chroma, plane);

        let plane_bsize = if plane == 0 {
            bsize
        } else {
            uv_plane_block_size(bsize, ss_x, ss_y)
        };
        let plane_tx = if plane == 0 {
            mi.tx_size
        } else {
            uv_plane_tx_size(bsize, mi.tx_size, ss_x, ss_y)
        };
        // libvpx pd->n4_w / n4_h (vp9_decodeframe.c:796-797). For sub-8
        // luma the coding-block-level iterations follow the parent's
        // `(2 >> ss)` extent (luma covers 2×2 4×4 cells; chroma covers
        // 1×1 under 4:2:0). For ≥8×8 the extent follows the plane_bsize
        // directly. This mirrors the intra path's pinning in tile.rs.
        let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let tx_step_4x4 = 1usize << (plane_tx as usize);

        // Plane-pixel coord of block's top-left.
        let plane_px_x = (mi_col * 8) >> ss_x;
        let plane_px_y = (mi_row * 8) >> ss_y;

        // 4×4 coordinates used by the per-plane coef-context indices.
        let plane_x4_base = (mi_col_in_tile * 2) >> ss_x;
        let plane_y4_in_sb_base = (mi_row_in_sb * 2) >> ss_y;

        // Pixel width / height of the block in this plane.
        let block_w_px = plane_bw_4x4 * 4;
        let block_h_px = plane_bh_4x4 * 4;

        // For sub-8 blocks libvpx (`build_inter_predictors_for_planes`,
        // `vp9_reconinter.c:223-229`) iterates the plane's `num_4x4`
        // grid, running a counter `i = y * num_4x4_w + x` as the chroma
        // sub-block index into `average_split_mvs`. The plane grid is
        // `(2 >> ss_x)` wide × `(2 >> ss_y)` high: luma 2×2, 4:2:0
        // chroma 1×1, 4:2:2 chroma 1w×2h, 4:4:0 chroma 2w×1h, 4:4:4
        // chroma 2×2. Each sub-block is 4×4. For ≥8×8 (`!is_sub8`) we
        // iterate once over the full block.
        let (mc_sub_iter_count_w, mc_sub_iter_count_h) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (1, 1)
        };
        let mc_w_px = if is_sub8 { 4 } else { block_w_px };
        let mc_h_px = if is_sub8 { 4 } else { block_h_px };

        // -- Motion compensation -------------------------------------
        //
        // libvpx `build_inter_predictors` (`vp9_reconinter.c:126-207`):
        // iterate over `1 + is_compound` refs. The first ref writes
        // its prediction into `dst`; the second (compound) ref runs
        // through the `_avg` kernels which read the in-place dst and
        // overwrite it with `(dst + new_pred + 1) >> 1`.

        // Destination view of the current frame's plane. We borrow the
        // PlaneMut for this iteration so the slice lives for the whole
        // per-plane loop body.
        let mut dst_plane = match plane {
            0 => dst_frame.y_plane_mut(),
            1 => dst_frame.u_plane_mut(),
            _ => dst_frame.v_plane_mut(),
        };
        let dst_stride = dst_plane.stride();
        let dst_slice: &mut [u8] = dst_plane.data_mut();

        // Iterate per-bmi for sub-8 luma; once for ≥8×8 / chroma. For
        // each (y, x) sub-position we run MC over `(mc_w_px, mc_h_px)`
        // at the matching pixel offset.
        for sub_y in 0..mc_sub_iter_count_h {
            for sub_x in 0..mc_sub_iter_count_w {
                // libvpx running counter `i = y * num_4x4_w + x`
                // (`vp9_reconinter.c:226-228`).
                let bmi_idx = sub_y * mc_sub_iter_count_w + sub_x;
                let sub_px_x = plane_px_x + sub_x * 4;
                let sub_px_y = plane_px_y + sub_y * 4;
                let sub_dst_offset = sub_px_y * dst_stride + sub_px_x;

                for r in 0..num_refs {
                    let ref_buf = inputs.refs[r].expect("ref already validated");
                    let sf = sfs[r].expect("sf already validated");

                    // MV selection (libvpx `build_inter_predictors`,
                    // `vp9_reconinter.c:140-142`): sub-8 uses
                    // `average_split_mvs(pd, mi, ref, block)` for ALL
                    // planes; ≥8×8 reads `mi->mv[r]` directly. The luma
                    // plane (ss == 0,0) hits `average_split_mvs` case 0,
                    // which returns the per-block bmi MV unchanged — so
                    // this is byte-identical to the old explicit
                    // `mi.bmi[bmi_idx].as_mv[r]` luma read.
                    let mv = if is_sub8 {
                        average_split_mvs(&mi.bmi, r, ss_x, ss_y, bmi_idx)
                    } else {
                        mvs[r]
                    };

                    // Convert luma-resolution Q3 (1/8-pel) to Q4
                    // (1/16-pel, ref-coord) via the uniform per-axis
                    // doubling `mv_q4 = mv * (1 << (1 - ss))` (libvpx
                    // `clamp_mv_to_umv_border_sb`, `vp9_reconinter.c:99-
                    // 100`). `average_split_mvs` returns at luma
                    // resolution for every subsampling, so the same
                    // scale path applies whether or not the block is
                    // sub-8 (collapsing the former `is_sub8 && plane>0`
                    // special-case: at 4:2:0 chroma `1 << (1-1) == 1`,
                    // a no-op, so the 4:2:0 output is unchanged).
                    let mv_q4_row = i32::from(mv.row) * (1i32 << (1 - ss_y as i32));
                    let mv_q4_col = i32::from(mv.col) * (1i32 << (1 - ss_x as i32));

                    // D.24: libvpx's `build_inter_predictors`
                    // (`vp9_reconinter.c:155-186`) uses an asymmetric
                    // mapping in the scaled path: the integer-pel base
                    // uses the *plane* coord (`scaled_buffer_offset(x_start,
                    // y_start, ...)` where `x_start = (-mb_to_left_edge) >>
                    // (3 + ss_x)` is in chroma pixels) while
                    // `vp9_scale_mv(mv_q4, mi_x, mi_y, sf)` uses *luma*
                    // pixel coords (`mi_x = mi_col * 8`) to compute the
                    // sub-pel mask offset. For unscaled refs the two
                    // collapse together (xs/ys == 16 and scaled is the
                    // identity), so we keep the cheap unified path. For
                    // scaled refs the chroma off-by-1 vs libvpx is the
                    // exact "wrong subpel filter row" symptom — fix
                    // mirrors libvpx 1:1.
                    let (total_x_q4, total_y_q4) = if sf.is_scaled() {
                        // mi_x / mi_y are libvpx's `mi_x + x`, `mi_y + y`
                        // — pixel coords in LUMA precision regardless of
                        // plane. The per-sub-block offset is in plane
                        // pixels (`sub_* * 4`); `<< ss` converts it back
                        // to luma pixels. For 4:2:0 chroma the grid is
                        // 1×1 so `sub_* == 0` and the term vanishes; for
                        // 4:2:2/4:4:0/4:4:4 the non-collapsed grid scales
                        // correctly.
                        let mi_x = (mi_col * 8) as i32 + (sub_x * 4) as i32 * (1i32 << ss_x);
                        let mi_y = (mi_row * 8) as i32 + (sub_y * 4) as i32 * (1i32 << ss_y);
                        let scaled_mv_col = mul_q14(mv_q4_col, sf.x_scale_fp)
                            + (mul_q14(mi_x << SUBPEL_BITS, sf.x_scale_fp) & SUBPEL_MASK);
                        let scaled_mv_row = mul_q14(mv_q4_row, sf.y_scale_fp)
                            + (mul_q14(mi_y << SUBPEL_BITS, sf.y_scale_fp) & SUBPEL_MASK);
                        let base_int_x = mul_q14(sub_px_x as i32, sf.x_scale_fp);
                        let base_int_y = mul_q14(sub_px_y as i32, sf.y_scale_fp);
                        (
                            (base_int_x << SUBPEL_BITS) + scaled_mv_col,
                            (base_int_y << SUBPEL_BITS) + scaled_mv_row,
                        )
                    } else {
                        (
                            sf.scaled_x_q4(sub_px_x as i32, mv_q4_col as i16),
                            sf.scaled_y_q4(sub_px_y as i32, mv_q4_row as i16),
                        )
                    };

                    let ref_plane: BorderedPlaneRef<'_> =
                        ref_buf.plane_with_border(plane);
                    let crop = RefPlaneCrop::for_plane(ref_buf, plane);

                    let dst_for_mc = &mut dst_slice[sub_dst_offset..];
                    let is_compound_avg = r > 0;

                    // D.19 — gated MC trace for aq2 divergence debug.
                    // Prints per-MC: mi, plane, ref, mv (Q3), block w/h,
                    // dst origin, total_x_q4/total_y_q4, int_x/int_y,
                    // subpel_x_q4/subpel_y_q4, ref_frame enum.
                    // D.25 — additional gate on `RVP9_TRACE_MC_FILTER=row,colmin,colmax`
                    // to scope the trace to specific failing SB rows. Used to
                    // localize tile_1x4 frame-3 partial-bottom MC divergence to
                    // the altref (frame 4) partition decode upstream.
                    let trace_mc_pass = {
                        let tf = crate::trace_flags::trace_flags();
                        if tf.trace_mc {
                            match tf.trace_mc_filter.as_deref() {
                                None => true,
                                Some(s) => {
                                    let parts: Vec<&str> = s.split(',').collect();
                                    if parts.len() == 3 {
                                        let r: usize = parts[0].parse().unwrap_or(0);
                                        let cmin: usize = parts[1].parse().unwrap_or(0);
                                        let cmax: usize = parts[2].parse().unwrap_or(0);
                                        mi_row == r && mi_col >= cmin && mi_col < cmax
                                    } else {
                                        true
                                    }
                                }
                            }
                        } else {
                            false
                        }
                    };
                    if trace_mc_pass {
                        let int_x = total_x_q4 >> SUBPEL_BITS;
                        let int_y = total_y_q4 >> SUBPEL_BITS;
                        let subpel_x = total_x_q4 & SUBPEL_MASK;
                        let subpel_y = total_y_q4 & SUBPEL_MASK;
                        eprintln!(
                            "RVP9_TRACE_MC mi=({},{}) plane={} ref={} bw={} bh={} dst=({},{}) \
                             mv=({},{}) mv_q4=({},{}) total=({},{}) int=({},{}) subpel=({},{}) \
                             rf={:?} sub_x={} sub_y={}",
                            mi_row, mi_col, plane, r, mc_w_px, mc_h_px,
                            sub_px_x, sub_px_y,
                            mv.row, mv.col, mv_q4_row, mv_q4_col,
                            total_x_q4, total_y_q4, int_x, int_y, subpel_x, subpel_y,
                            ref_frames[r], sub_x, sub_y,
                        );
                    }

                    // Phase D.5: route through `build_inter_predictor_with_mc_border`
                    // so wide-MV blocks that point beyond the
                    // pre-extended static border fall back to a
                    // scratch buffer with edge replication. Mirrors
                    // libvpx `dec_build_inter_predictors`
                    // (`vp9_decodeframe.c:660-697`).
                    build_inter_predictor_with_mc_border(
                        &ref_plane,
                        crop,
                        total_x_q4,
                        total_y_q4,
                        &sf,
                        interp_filter,
                        dst_for_mc,
                        dst_stride,
                        mc_w_px,
                        mc_h_px,
                        is_compound_avg,
                        dispatch,
                    );
                }
            }
        }

        // -- Residual decode + IT-add -------------------------------
        //
        // libvpx `decode_block` (vp9_decodeframe.c:455-505): when
        // `mi->skip` is set, the per-TX residual decode is bypassed
        // and the MC output is the final reconstruction. Caller-side
        // `dec_reset_skip_context` (vp9_decodeframe.c:783-790) zeroes
        // the per-plane above/left coef-context cells covering the
        // block — we do that inline here per-plane.

        if mi.skip {
            // Zero the (above_coef, left_coef) cells for this plane's
            // span. Mirrors libvpx's per-plane `dec_reset_skip_context`
            // splat.
            coef_ctx.set_coef_flags(
                plane,
                plane_x4_base,
                plane_y4_in_sb_base,
                plane_bw_4x4,
                plane_bh_4x4,
                false,
            );
        } else {
            let plane_dequant_pair = inputs.dequant_per_plane[plane];

            // D.25 — frame-edge clamp for partial-bottom / partial-right
            // blocks, mirroring libvpx `predict_recon_inter`
            // (`vp9_decodeframe.c:892-921`) which iterates with:
            //
            //   max_blocks_wide = n4_w + (mb_to_right_edge >> (5 + ss_x))
            //   max_blocks_high = n4_h + (mb_to_bottom_edge >> (5 + ss_y))
            //
            // (where the second term is only applied when mb_to_*_edge < 0
            // i.e. the block extends past the frame edge). Without this
            // clamp rvp9 reads coef tokens for the invisible-edge TX cells,
            // drifting the bool-coder cursor relative to libvpx for any
            // partial-row/col inter block. Surfaced as the SB row 16
            // (mi_rows - mi_row = 7) divergence on tile_1x4 / tile_1x8 at
            // 1920×1080 / 3840×2160. The intra path doesn't hit this
            // because keyframes are bit-exact today; libvpx applies the
            // same clamp in `predict_recon_intra`.
            let bw_mi = usize::from(bsize.n_8x8_wide());
            let bh_mi = usize::from(bsize.n_8x8_high());
            let mb_to_right_edge_px = if is_sub8 {
                // libvpx applies the partial-row/col clamp to the parent
                // 8×8 cell, so use bw=1/bh=1 here. Sub-8 blocks never
                // straddle a frame edge in practice (the parent 8×8 is
                // the unit of edge-clamping).
                (inputs.mi_cols as i64 - 1 - mi_col as i64) * 64
            } else {
                (inputs.mi_cols as i64 - bw_mi as i64 - mi_col as i64) * 64
            };
            let mb_to_bottom_edge_px = if is_sub8 {
                (inputs.mi_rows as i64 - 1 - mi_row as i64) * 64
            } else {
                (inputs.mi_rows as i64 - bh_mi as i64 - mi_row as i64) * 64
            };
            let max_blocks_wide = if mb_to_right_edge_px >= 0 {
                plane_bw_4x4
            } else {
                let clamp = (mb_to_right_edge_px >> (5 + ss_x)) as isize;
                ((plane_bw_4x4 as isize) + clamp).max(0) as usize
            };
            let max_blocks_high = if mb_to_bottom_edge_px >= 0 {
                plane_bh_4x4
            } else {
                let clamp = (mb_to_bottom_edge_px >> (5 + ss_y)) as isize;
                ((plane_bh_4x4 as isize) + clamp).max(0) as usize
            };

            // Iterate TX positions inside the plane's coding block.
            let mut loff = 0usize;
            while loff < max_blocks_high {
                let mut aoff = 0usize;
                while aoff < max_blocks_wide {
                    let tx_px_x = plane_px_x + aoff * 4;
                    let tx_px_y = plane_px_y + loff * 4;
                    let tx_dst_off = tx_px_y * dst_stride + tx_px_x;
                    let tx_dst = &mut dst_slice[tx_dst_off..];

                    let tx_type = TxType::DctDct; // inter blocks always use DctDct
                    let seg_eob = plane_tx.n_coeffs();
                    // R.10: MaybeUninit eliminates the 4KB stack zero per
                    // TX iteration. `decode_block_tokens` -> `decode_coefs`
                    // zeros `[..seg_eob]` internally (`detokenize.rs:305-307`),
                    // and only `[..seg_eob]` is read afterwards
                    // (`inverse_transform_add_dct(coefs, ...)` line 568,
                    // n = seg_eob in this caller per Tx*.n_coeffs()).
                    let mut dq_coefs_uninit: std::mem::MaybeUninit<[i32; MAX_TX_COEFFS]> =
                        std::mem::MaybeUninit::uninit();
                    // SAFETY: decode_block_tokens fully writes
                    // dq_coefs[..seg_eob] (via its zero pass); subsequent
                    // reads stay within that range.
                    let dq_coefs: &mut [i32; MAX_TX_COEFFS] =
                        unsafe { dq_coefs_uninit.assume_init_mut() };
                    let eob = decode_block_tokens(
                        reader,
                        fc,
                        coef_counts.as_deref_mut(),
                        coef_ctx,
                        plane,
                        plane_x4_base + aoff,
                        plane_y4_in_sb_base + loff,
                        plane_tx,
                        tx_type,
                        /* ref_type = */ 1, // inter
                        seg_eob,
                        plane_dequant_pair,
                        &mut dq_coefs[..seg_eob],
                    );

                    if eob > 0 {
                        // G2 (decoder-hardening V6 §G2): pass
                        // `inputs.lossless` so the 4×4 site can pick
                        // the inverse Walsh-Hadamard family
                        // (`iwht4x4_*_add`) instead of the standard
                        // IDCT when the frame is lossless. Larger TX
                        // sizes are not legal in lossless mode.
                        inverse_transform_add_dct(
                            plane_tx,
                            &dq_coefs[..seg_eob],
                            eob,
                            tx_dst,
                            dst_stride,
                            inputs.lossless,
                            dispatch,
                        );
                    }

                    aoff += tx_step_4x4;
                }
                loff += tx_step_4x4;
            }
        }
    }
}

/// End-to-end per-inter-block decode + reconstruction. Stitches the
/// libvpx call sequence at `vp9/decoder/vp9_decodeframe.c:455-505`
/// (the inter arm of `decode_block`) into a single entry point that
/// reads mode-info from the bool-coder then drives reconstruction.
///
/// The eventual inter-frame driver (the analogue of
/// [`crate::tile::decode_tile_intra`]) wires the per-plane segments,
/// edges, and neighbour lookups; this helper consolidates the
/// per-block step.
///
/// `mi` must have its non-inter fields (`block_size`, `tx_size`,
/// `skip`, `segment_id`) populated by the caller before invocation,
/// and `mi.mode` initialised to a placeholder — `read_inter_block_mode_info`
/// will fully populate `mi.mode` and `mi.bmi`.
///
/// `decode_block` integration (Stage D.2.e): the live inter-frame
/// driver ([`crate::Decoder::decode_inter_frame`]) calls this helper
/// from the inter branch of `decode_block`. `FrameKind::InterFrame`
/// is fully handled; only specific structural error cases (e.g. an
/// inter ref naming an empty `ref_frame_map` slot —
/// [`crate::DecodeError::EmptyInterRef`]) produce early returns
/// before this helper runs.
///
/// # Panics
///
/// - If `read_inter_block_mode_info` populates `mi.mode` with an
///   intra variant (would indicate a bitstream / context bug).
/// - Inherits panics from [`reconstruct_inter_block`].
#[allow(clippy::too_many_arguments)]
pub fn decode_and_reconstruct_inter_block<N: NeighbourLookup>(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    coef_ctx: &mut TileCoefCtx,
    params: &InterModeReadParams,
    neighbours: &N,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    mi: &mut ModeInfo,
    mi_row: usize,
    mi_col: usize,
    mi_row_in_tile: usize,
    mi_col_in_tile: usize,
    mi_row_in_sb: usize,
    inputs: &InterBlockInputs<'_>,
    dst_frame: &mut Yv12Buffer,
    dispatch: &SimdDispatch,
) {
    // 1. Read inter-block mode info (refs, mode, MVs, filter).
    read_inter_block_mode_info(
        reader,
        fc,
        /* counts = */ None,
        /* mode_counts = */ None,
        params,
        neighbours,
        mi.block_size,
        bounds,
        prev_mv,
        mi,
    );

    // 2. Per-block reconstruction.
    reconstruct_inter_block(
        reader,
        fc,
        /* coef_counts = */ None,
        coef_ctx,
        mi,
        mi_row,
        mi_col,
        mi_row_in_tile,
        mi_col_in_tile,
        mi_row_in_sb,
        inputs,
        dst_frame,
        dispatch,
    );
}

/// libvpx `round_mv_comp_q4` (`vp9_reconinter.c:61-63`). Sign-aware
/// "round to nearest" division of a Q3 MV-component sum of 4 by 4.
#[inline]
fn round_mv_comp_q4(value: i32) -> i32 {
    if value < 0 {
        (value - 2) / 4
    } else {
        (value + 2) / 4
    }
}

/// libvpx `round_mv_comp_q2` (`vp9_reconinter.c:77-79`). Sign-aware
/// "round to nearest" division of a Q3 MV-component sum of 2 by 2.
#[inline]
fn round_mv_comp_q2(value: i32) -> i32 {
    if value < 0 {
        (value - 1) / 2
    } else {
        (value + 1) / 2
    }
}

/// libvpx `mi_mv_pred_q4` (`vp9_reconinter.c:65-75`). Averages all four
/// bmi MVs for ref `idx`.
#[inline]
fn mi_mv_pred_q4(bmi: &[rvp9_core::mode_info::BmiInfo; 4], idx: usize) -> Mv {
    let row_sum = i32::from(bmi[0].as_mv[idx].row)
        + i32::from(bmi[1].as_mv[idx].row)
        + i32::from(bmi[2].as_mv[idx].row)
        + i32::from(bmi[3].as_mv[idx].row);
    let col_sum = i32::from(bmi[0].as_mv[idx].col)
        + i32::from(bmi[1].as_mv[idx].col)
        + i32::from(bmi[2].as_mv[idx].col)
        + i32::from(bmi[3].as_mv[idx].col);
    Mv {
        row: round_mv_comp_q4(row_sum) as i16,
        col: round_mv_comp_q4(col_sum) as i16,
    }
}

/// libvpx `mi_mv_pred_q2` (`vp9_reconinter.c:81-87`). Averages the two
/// bmi MVs `block0`/`block1` for ref `idx`.
#[inline]
fn mi_mv_pred_q2(
    bmi: &[rvp9_core::mode_info::BmiInfo; 4],
    idx: usize,
    block0: usize,
    block1: usize,
) -> Mv {
    let row_sum =
        i32::from(bmi[block0].as_mv[idx].row) + i32::from(bmi[block1].as_mv[idx].row);
    let col_sum =
        i32::from(bmi[block0].as_mv[idx].col) + i32::from(bmi[block1].as_mv[idx].col);
    Mv {
        row: round_mv_comp_q2(row_sum) as i16,
        col: round_mv_comp_q2(col_sum) as i16,
    }
}

/// libvpx `average_split_mvs` (`vp9_reconinter.c:112-124`). Selects the
/// chroma MV for a sub-8 block's chroma sub-block `block`, by averaging
/// the contributing luma bmi MVs according to the plane subsampling.
/// Returns the MV at **luma resolution** (Q3); the downstream
/// `mv * (1 << (1 - ss))` scaling (libvpx `clamp_mv_to_umv_border_sb`,
/// `vp9_reconinter.c:99-100`) converts it to the chroma Q4 value.
///
/// ```text
/// const int ss_idx = ((pd->subsampling_x > 0) << 1) | (pd->subsampling_y > 0);
/// switch (ss_idx) {
///   case 0: res = mi->bmi[block].as_mv[ref].as_mv;        break;  // 4:4:4
///   case 1: res = mi_mv_pred_q2(mi, ref, block, block + 2); break; // 4:4:0
///   case 2: res = mi_mv_pred_q2(mi, ref, block, block + 1); break; // 4:2:2
///   case 3: res = mi_mv_pred_q4(mi, ref);                  break;  // 4:2:0
/// }
/// ```
#[inline]
fn average_split_mvs(
    bmi: &[rvp9_core::mode_info::BmiInfo; 4],
    ref_idx: usize,
    ss_x: usize,
    ss_y: usize,
    block: usize,
) -> Mv {
    let ss_idx = ((ss_x > 0) as usize) << 1 | (ss_y > 0) as usize;
    match ss_idx {
        0 => bmi[block].as_mv[ref_idx],
        1 => mi_mv_pred_q2(bmi, ref_idx, block, block + 2),
        2 => mi_mv_pred_q2(bmi, ref_idx, block, block + 1),
        _ => mi_mv_pred_q4(bmi, ref_idx),
    }
}

/// Dispatch one inverse-transform-add for an inter TX block. Inter
/// blocks always use `TxType::DctDct` (libvpx
/// `vp9_decodeframe.c:235-280`: hybrid kernels exist only for intra Y);
/// 32×32 has no hybrid in any case.
///
/// `lossless` selects the inverse Walsh-Hadamard family
/// (`iwht4x4_*_add`) at Tx4x4 in place of the standard 4×4 IDCT, per
/// libvpx `inverse_transform_block_inter` (`vp9_decodeframe.c:266-294`).
/// Lossless mode is restricted to `TX_4X4`, so at the larger TX sizes
/// the flag is unused and a debug assertion catches caller bugs.
#[allow(clippy::too_many_arguments)] // dispatch arg threading through G4.x SIMD rollout
fn inverse_transform_add_dct(
    tx_size: TxSize,
    input: &[i32],
    eob: usize,
    dest: &mut [u8],
    stride: usize,
    lossless: bool,
    dispatch: &SimdDispatch,
) {
    debug_assert!(
        !lossless || tx_size == TxSize::Tx4x4,
        "VP9 lossless mode is restricted to TX_4X4; got tx_size={tx_size:?}"
    );
    match tx_size {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = input.try_into().expect("Tx4x4 input is 16 coefs");
            if lossless {
                if eob == 1 {
                    // Defensive: `iwht4x4_1_add` only sees `coefs[0]`; if a
                    // future detokenizer bug emitted `eob == 1` alongside a
                    // nonzero AC coefficient, that AC value would be
                    // silently dropped and the block would miscompute.
                    debug_assert!(
                        coefs[1..].iter().all(|&c| c == 0),
                        "iwht4x4_1_add dispatched with non-DC coefficients; \
                         eob == 1 must imply coefs[1..] are zero"
                    );
                    iwht4x4_1_add(coefs[0], dest, stride);
                } else {
                    iwht4x4_16_add(coefs, dest, stride);
                }
            } else if eob > 1 {
                // G4.2 — dispatch through SimdDispatch (mirror of the
                // G4.1 idct8x8 wiring two arms down).
                // G4.2: inter is DCT-DCT-only by spec (no ADST hybrids
                // reach here); when intra wires hybrid dispatch slots in
                // G4.3a, the inter path is already covered.
                (dispatch.idct4x4_add())(coefs, dest, stride);
            } else {
                idct4x4_1_add(coefs[0], dest, stride);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = input.try_into().expect("Tx8x8 input is 64 coefs");
            if eob == 1 {
                idct8x8_1_add(coefs[0], dest, stride);
            } else if eob <= 12 {
                // Task A — sparse Tx8x8 path. SSE2 kernel exploits the
                // eob<=12 contract (rows 4..7 of coefs are zero per the
                // bitstream), so the row pass runs on only 4 rows
                // instead of 8. Saves ~50% of the 1-D row-pass work on
                // this dispatch. AVX2 _64 still owns the dense
                // (eob > 12) path.
                (dispatch.idct8x8_12_add())(coefs, dest, stride);
            } else {
                // G4.1 — dispatch through SimdDispatch so the AVX2
                // implementation runs on supported CPUs.
                (dispatch.idct8x8_add())(coefs, dest, stride);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = input.try_into().expect("Tx16x16 input is 256 coefs");
            if eob == 1 {
                idct16x16_1_add(coefs[0], dest, stride);
            } else if eob <= 10 {
                // R.5 — sparse-block fast path mirroring the intra
                // path's wiring (reconstruct.rs:430). Top-left 4×4 of
                // input is non-zero (libvpx scan-order bucket); row
                // pass runs 4× instead of 16×.
                (dispatch.idct16x16_10_add())(coefs, dest, stride);
            } else if eob <= 38 {
                // R.5 — top-left 8×8 sparse fast path. Row pass runs
                // 8× instead of 16×.
                (dispatch.idct16x16_38_add())(coefs, dest, stride);
            } else {
                // G4.2 — dispatch through SimdDispatch so the AVX2
                // implementation runs on supported CPUs.
                (dispatch.idct16x16_add())(coefs, dest, stride);
            }
        }
        TxSize::Tx32x32 => {
            let coefs: &[i32; 1024] = input.try_into().expect("Tx32x32 input is 1024 coefs");
            if eob == 1 {
                idct32x32_1_add(coefs[0], dest, stride);
            } else if eob <= 34 {
                // R.5 — top-left 8×8 sparse fast path mirroring
                // reconstruct.rs:460. Row pass runs 8× instead of 32×.
                (dispatch.idct32x32_34_add())(coefs, dest, stride);
            } else if eob <= 135 {
                // R.5 — top-left 16×16 sparse fast path. Row pass
                // runs 16× instead of 32×.
                (dispatch.idct32x32_135_add())(coefs, dest, stride);
            } else {
                // G4-α — dispatch through SimdDispatch so the AVX2
                // implementation runs on supported CPUs.
                (dispatch.idct32x32_add())(coefs, dest, stride);
            }
        }
    }
}

/// Maximum dimension of the MC scratch buffer along either axis. The
/// largest block any one MC call writes is 64×64 (Y plane, ≥8×8
/// `bsize`), and the convolve8 footprint extends by 7 along each
/// padded axis. For *scaled* refs the per-sample step `x_step_q4` can
/// reach `2 * SUBPEL_SHIFTS = 32` (libvpx's 2× downscale floor —
/// `vp9_scale.h:61-65`), so `(w - 1) * 32 >> 4 + 1 + 8 ≈ 135` for
/// `w = 64`. Round up to 160 to match libvpx's
/// `extend_and_predict_buf` cell count (`vp9_decoder.h:58`).
///
/// Stored as a constant so the scratch is stack-allocated — at 160² =
/// 25600 bytes, comfortably fits within Rust's default 8 MiB stack
/// budget while avoiding allocator pressure on the hot per-block path.
const MC_BORDER_SCRATCH_MAX: usize = 160;

/// `VP9_INTERP_EXTEND` in libvpx (`yv12config.h:25`). The 8-tap
/// convolve8 kernel needs 7 trailing taps past the integer base
/// (`VP9_INTERP_EXTEND - 1 = 3` leading + `VP9_INTERP_EXTEND = 4`
/// trailing).
const VP9_INTERP_EXTEND: i32 = 4;

/// Visible (cropped) plane dimensions for one ref-plane. Mirrors
/// libvpx's `y_crop_width` / `uv_crop_width` (`yv12config.h:74-79`).
/// Captured so the MC-border scratch fill replicates from the cropped
/// edges exactly as libvpx's `build_mc_border` does.
#[derive(Copy, Clone)]
struct RefPlaneCrop {
    /// Visible width in pixels (Y = `crop_width`; UV = `(crop_width +
    /// ss_x) >> ss_x`).
    crop_w: usize,
    /// Visible height in pixels.
    crop_h: usize,
}

impl RefPlaneCrop {
    /// Resolve the crop dims for `plane` (0=Y, 1=U, 2=V) on `buf`,
    /// with chroma subsampling derived from the buffer's
    /// [`ChromaFormat`]. Chroma planes use libvpx's `(crop + ss) >> ss`
    /// (`yv12config.c:254-255`); reuses [`Yv12Buffer::chroma_width`] /
    /// [`Yv12Buffer::chroma_height`], which compute exactly this.
    fn for_plane(buf: &Yv12Buffer, plane: usize) -> Self {
        if plane == 0 {
            Self {
                crop_w: buf.width() as usize,
                crop_h: buf.height() as usize,
            }
        } else {
            Self {
                crop_w: buf.chroma_width() as usize,
                crop_h: buf.chroma_height() as usize,
            }
        }
    }
}

/// libvpx `build_mc_border` (`vp9_decodeframe.c:456-490`). Copy a
/// rectangle of `b_w × b_h` reference pixels starting at top-left
/// `(x, y)` in visible-region coordinates into `dst`, padding any
/// out-of-visible cells via edge-replication from the nearest visible
/// pixel.
///
/// `src` points to the visible top-left (`crop[0, 0]`), `src_stride`
/// is the underlying ref-plane stride, and `(w, h)` are the cropped
/// dimensions. `dst` has stride `dst_stride == b_w` (caller allocates
/// the scratch tightly packed).
///
/// Reads remain in-bounds on `src` because the caller passes a
/// `BorderedPlaneRef`-derived pointer; cells in the `[crop_w,
/// aligned_w + border)` slack also hold edge-replicated values after
/// `extend_borders`, so even when `b_w` would overshoot we never
/// observe an uninitialised cell.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's `build_mc_border` 1:1
fn build_mc_border(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    x: i32,
    mut y: i32,
    b_w: usize,
    b_h: usize,
    w: i32,
    h: i32,
) {
    // libvpx: `ref_row = src - x - y * src_stride;` — i.e. row 0 of
    // the visible region. We index `ref_row` via `(y_clamped *
    // src_stride + col)`.
    let mut y_clamped = if y >= h {
        h - 1
    } else if y > 0 {
        y
    } else {
        0
    };

    let mut dst_off = 0usize;
    for _ in 0..b_h {
        // libvpx: per-row, recompute left/right paddings (only `x`
        // varies; we keep `x` constant per call but the loop matches
        // libvpx's structure 1:1).
        let left = if x < 0 { (-x) as usize } else { 0 };
        let left = left.min(b_w);
        let right_raw = x + b_w as i32 - w;
        let right = if right_raw > 0 { right_raw as usize } else { 0 };
        let right = right.min(b_w);
        let copy = b_w.saturating_sub(left + right);

        let row_base = (y_clamped as usize) * src_stride;
        let row = &src[row_base..];

        if left > 0 {
            let fill = row[0];
            for c in 0..left {
                dst[dst_off + c] = fill;
            }
        }
        if copy > 0 {
            let src_start = (x + left as i32) as usize;
            dst[dst_off + left..dst_off + left + copy]
                .copy_from_slice(&row[src_start..src_start + copy]);
        }
        if right > 0 {
            let fill = row[(w - 1) as usize];
            for c in 0..right {
                dst[dst_off + left + copy + c] = fill;
            }
        }

        dst_off += dst_stride;
        y += 1;
        // libvpx: `if (y > 0 && y < h) ref_row += src_stride;`
        if y > 0 && y < h {
            y_clamped = y;
        }
    }
}

/// Run motion compensation for one inter sub-block, transparently
/// handling MVs that point beyond the reference frame's extended
/// border via a fallback scratch buffer.
///
/// Mirrors libvpx's `dec_build_inter_predictors`
/// (`vp9_decodeframe.c:576-712`) — the bounds check at lines 660-697
/// decides between the direct convolve and the `extend_and_predict`
/// scratch path. For decoder paths that pre-extend the reference
/// frame's static border (our case; `extend_borders` runs at frame
/// finalisation), the fast direct path covers all MVs whose convolve
/// footprint fits within the bordered region (`±32` for Y / `±16`
/// for UV); only the wider-MV tail spills into the scratch fallback.
///
/// `total_x_q4` / `total_y_q4` are the *full* Q4 sub-pel positions in
/// the reference frame (as returned by [`ScaleFactors::scaled_x_q4`] /
/// [`ScaleFactors::scaled_y_q4`]). `ref_plane` is the bordered view
/// of the ref plane; `crop` is the visible (cropped) dimensions of
/// the same plane.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
#[allow(clippy::too_many_arguments)]
fn build_inter_predictor_with_mc_border(
    ref_plane: &BorderedPlaneRef<'_>,
    crop: RefPlaneCrop,
    total_x_q4: i32,
    total_y_q4: i32,
    sf: &ScaleFactors,
    interp_filter: InterpFilter,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    is_compound_avg: bool,
    dispatch: &SimdDispatch,
) {
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

    // libvpx `vp9_decodeframe.c:660-678`: when the convolve actually
    // touches sub-pel taps on an axis (subpel != 0 or scaled), the
    // footprint extends by `VP9_INTERP_EXTEND - 1 = 3` leading +
    // `VP9_INTERP_EXTEND = 4` trailing samples past the integer-pel
    // base. `x_pad` / `y_pad` are 1 iff that axis is padded.
    let x_pad = subpel_x_q4 != 0 || sf.x_step_q4 != SUBPEL_SHIFTS as i32;
    let y_pad = subpel_y_q4 != 0 || sf.y_step_q4 != SUBPEL_SHIFTS as i32;

    // libvpx: `x1 = ((x0_16 + (w - 1) * xs) >> SUBPEL_BITS) + 1`. For
    // the unscaled fast path `xs == SUBPEL_SHIFTS` and `(w - 1) * xs >>
    // SUBPEL_BITS == w - 1`, collapsing to `int_x + w`. We compute the
    // general form so the scaled tail walks correctly.
    let xs = sf.x_step_q4;
    let ys = sf.y_step_q4;
    let last_x_q4 = total_x_q4 + (w as i32 - 1) * xs;
    let last_y_q4 = total_y_q4 + (h as i32 - 1) * ys;
    let mut x0 = int_x;
    let mut y0 = int_y;
    let mut x1 = (last_x_q4 >> SUBPEL_BITS) + 1;
    let mut y1 = (last_y_q4 >> SUBPEL_BITS) + 1;
    if x_pad {
        x0 -= VP9_INTERP_EXTEND - 1;
        x1 += VP9_INTERP_EXTEND;
    }
    if y_pad {
        y0 -= VP9_INTERP_EXTEND - 1;
        y1 += VP9_INTERP_EXTEND;
    }

    // Bordered-region bounds: `[-border, aligned + border)`. As long
    // as the footprint extreme cell `(x1, y1)` (= one-past-last touched
    // sample's index — libvpx's predicate checks `x1 > frame_width -
    // 1`, equivalent to `x1 >= frame_width`) sits inside the bordered
    // region, the direct path produces the same result as the scratch
    // path because `extend_borders` has already populated the bordered
    // region with edge-replicated values.
    let bordered_x_lo = -(ref_plane.border_w as i32);
    let bordered_y_lo = -(ref_plane.border_h as i32);
    let bordered_x_hi = (ref_plane.width + ref_plane.border_w) as i32;
    let bordered_y_hi = (ref_plane.height + ref_plane.border_h) as i32;

    let needs_scratch = x0 < bordered_x_lo
        || y0 < bordered_y_lo
        || x1 > bordered_x_hi
        || y1 > bordered_y_hi;

    if !needs_scratch {
        // Fast path: read directly from the bordered region. Apply
        // the same per-axis `-3` pre-shift as the legacy in-line MC
        // code so `src[0]` is the kernel's first-tap base.
        let row_pre = if y_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let col_pre = if x_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let row_base =
            (ref_plane.border_h as i32 + int_y + row_pre) as usize;
        let col_base =
            (ref_plane.border_w as i32 + int_x + col_pre) as usize;
        let src_base = row_base * ref_plane.stride + col_base;
        // D.19 — gated source-data trace for the fast path. Dumps the
        // first row of the src footprint (post pre-shift) so the
        // caller can diff against libvpx's mc trace.
        if crate::trace_flags::trace_flags().trace_mc_src {
            let mut row = String::new();
            for c in 0..18.min(ref_plane.data.len().saturating_sub(src_base)) {
                row.push_str(&format!("{:3} ", ref_plane.data[src_base + c]));
            }
            eprintln!(
                "RVP9_TRACE_MC_SRC int=({int_x},{int_y}) subpel=({subpel_x_q4},{subpel_y_q4}) \
                 row_pre={row_pre} col_pre={col_pre} src_base={src_base} stride={} \
                 fast src_row0=[{row}]",
                ref_plane.stride,
            );
        }
        // R.16: pass the caller's dispatch to skip the Arc::clone
        // inside inter_predictor.
        rvp9_core::inter::inter_predictor_with_dispatch(
            &ref_plane.data[src_base..],
            ref_plane.stride,
            dst,
            dst_stride,
            subpel_x_q4,
            subpel_y_q4,
            sf,
            interp_filter,
            w,
            h,
            is_compound_avg,
            dispatch,
        );
        return;
    }

    // Slow path: copy footprint into scratch with libvpx-style edge
    // replication, then convolve from there. The scratch buffer is
    // tightly packed: `b_h × b_w` cells with stride `b_w`. Libvpx
    // sizes the scratch as `b_w = x1 - x0 + 1, b_h = y1 - y0 + 1`
    // (`vp9_decodeframe.c:685-686`) — i.e. inclusive of both
    // endpoints; the convolve's 8-tap footprint requires reading one
    // sample past the last destination column.
    let b_w = (x1 - x0 + 1) as usize;
    let b_h = (y1 - y0 + 1) as usize;
    debug_assert!(
        b_w <= MC_BORDER_SCRATCH_MAX && b_h <= MC_BORDER_SCRATCH_MAX,
        "MC scratch footprint {b_w}×{b_h} exceeds MC_BORDER_SCRATCH_MAX = {MC_BORDER_SCRATCH_MAX}",
    );
    // R.9: 25 KB uninit scratch. `build_mc_border` writes every cell
    // of `&mut scratch[..scratch_used]` (left fill + copy + right fill
    // sum to exactly `b_w` per row × `b_h` rows = `scratch_used`), so
    // `assume_init_mut` is sound — every cell we read post-call has
    // been written. Replaces `[0u8; ...]` whose compiler-emitted
    // memset at this declaration was wasted work (the buffer is
    // immediately overwritten by build_mc_border).
    let mut scratch_uninit: std::mem::MaybeUninit<[u8; MC_BORDER_SCRATCH_MAX * MC_BORDER_SCRATCH_MAX]>
        = std::mem::MaybeUninit::uninit();
    // SAFETY: every byte of `scratch[..scratch_used]` is written by
    // `build_mc_border` below before any read; reads outside that
    // range never happen (the convolve only indexes `[..scratch_used]`).
    let scratch: &mut [u8; MC_BORDER_SCRATCH_MAX * MC_BORDER_SCRATCH_MAX] =
        unsafe { scratch_uninit.assume_init_mut() };
    let scratch_stride = b_w;
    let scratch_used = b_h * scratch_stride;

    // Source pointer at the visible top-left (= bordered top-left +
    // (border_h, border_w)).
    let visible_base =
        ref_plane.border_h * ref_plane.stride + ref_plane.border_w;
    let src_visible = &ref_plane.data[visible_base..];

    build_mc_border(
        src_visible,
        ref_plane.stride,
        &mut scratch[..scratch_used],
        scratch_stride,
        x0,
        y0,
        b_w,
        b_h,
        crop.crop_w as i32,
        crop.crop_h as i32,
    );

    // The scratch's top-left is the 8-tap convolve footprint's
    // top-left when the relevant axis is padded — libvpx's
    // `border_offset = y_pad * 3 * b_w + x_pad * 3` places the
    // destination origin (post-shift) inside the scratch, but our
    // `inter_predictor` expects pre-shifted `src` (i.e. the
    // first-tap base), so the right scratch offset is `0` on any
    // padded axis. On a non-padded axis (`*_pad == false`) the
    // destination origin coincides with the integer-pel base, and
    // libvpx skips the corresponding `-3` shift; we mirror that by
    // leaving the offset at 0 too.
    //
    // Why does this still work for the no-pad axis? `b_w = x1 - x0 +
    // 1 = w + 1` in that case (no `VP9_INTERP_EXTEND` widening), and
    // the kernel reads `c ∈ 0..w` only (since `subpel == 0` collapses
    // to `convolve_copy`/`convolve8_vert` which don't sweep the tap
    // window horizontally). The kernel never indexes past
    // `b_w - 1`.
    rvp9_core::inter::inter_predictor_with_dispatch(
        &scratch[..],
        scratch_stride,
        dst,
        dst_stride,
        subpel_x_q4,
        subpel_y_q4,
        sf,
        interp_filter,
        w,
        h,
        is_compound_avg,
        dispatch,
    );
}

/// libvpx's `scaled_x` / `scaled_y` (`vp9_scale.c:16-22`) and the
/// `(int64_t)val * sf->x_scale_fp >> REF_SCALE_SHIFT` expression
/// used inside `vp9_scale_mv` / `scaled_buffer_offset`. Widens the
/// multiply to 64-bit so `val * scale_fp` doesn't overflow at full-HD
/// scale factors before the Q14 shift collapses it back to i32.
#[inline]
fn mul_q14(val: i32, scale_fp: i32) -> i32 {
    ((i64::from(val) * i64::from(scale_fp)) >> 14) as i32
}

#[allow(dead_code)] // referenced in tests / placeholders
fn _ignored() {
    let _ = InterpFilter::EightTap;
    let _: RefFrame = RefFrame::LastFrame;
    let _: Mv = Mv::ZERO;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::bitstream::ChromaFormat;
    use rvp9_core::entropy::{BoolWriter, FrameContext};
    use rvp9_core::entropy::coef_probs::{COEF_BANDS, COEFF_CONTEXTS};
    use rvp9_core::frame::{BorderUsage, ByteAlignment, Yv12Buffer};
    use rvp9_core::mode_info::{BlockMode, BmiInfo, InterMode, ModeInfo, Mv, RefFrame};
    use rvp9_core::transforms::TxSize;
    use std::sync::Arc;

    /// G4.1 — local helper that returns the process-wide
    /// `Arc<SimdDispatch>` for the test arg. Goes through
    /// `rvp9_core::dsp::init()` so we share the OnceLock with
    /// production (caps detection runs internally). Call sites use
    /// `&test_dispatch()` to obtain a `&SimdDispatch` via deref
    /// coercion on `&Arc<SimdDispatch>`.
    fn test_dispatch() -> Arc<rvp9_core::dsp::SimdDispatch> {
        rvp9_core::dsp::init()
    }

    /// Build a `FrameContext` whose `coef_probs[tx][plane_type][1]`
    /// (inter `ref_type = 1`) is a flat-128 table, so the encoded
    /// "EOB at scan[0]" produces `eob = 0` (and the rest of the
    /// streams stay at default coefs).
    fn fc_flat_eob_inter() -> FrameContext {
        let mut fc = FrameContext::new_default();
        let triple = [128u8, 128, 128];
        let ctx_row = [triple; COEFF_CONTEXTS];
        let band_row = [ctx_row; COEF_BANDS];
        for tx in 0..4 {
            for plane_type in 0..2 {
                fc.coef_probs[tx][plane_type][1] = band_row;
            }
        }
        fc
    }

    /// Encode "EOB at scan[0]" under flat 128 probs.
    fn encode_eob_zero() -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            // One EOB read per TX position. We'll emit enough zeros to
            // cover up to a 32×32 block worth of 4×4 TX positions
            // (i.e. 64 positions). At p=128 a single 0-bit suffices
            // per position.
            for _ in 0..128 {
                w.write(0, 128);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().unwrap()
        };
        buf.truncate(n);
        buf
    }

    /// Build a `Yv12Buffer` filled with a repeating ramp pattern so we
    /// can verify the MC output landed at the right offset.
    fn make_ref(w: u32, h: u32) -> Yv12Buffer {
        let mut buf = Yv12Buffer::alloc(
            w,
            h,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        // Y plane: y * 31 + x ramps.
        {
            let mut y = buf.y_plane_mut();
            let stride = y.stride();
            let width = y.width();
            let height = y.height();
            let data = y.data_mut();
            for r in 0..height {
                for x in 0..width {
                    data[r * stride + x] = ((r * 31 + x * 7) & 0xff) as u8;
                }
            }
        }
        // U / V — simpler constants for sanity.
        buf.u_plane_mut().fill(0x40);
        buf.v_plane_mut().fill(0xC0);
        buf.extend_borders();
        buf
    }

    /// Build the `ModeInfo` for an inter block.
    fn inter_mi(bsize: BlockSize, tx_size: TxSize, mv: Mv) -> ModeInfo {
        ModeInfo {
            block_size: bsize,
            tx_size,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [mv, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: [BmiInfo::default(); 4],
        }
    }

    /// D.2.a smoke: single-ref, non-skip, NEARESTMV with zero MV, Block8x8.
    /// The MC output is a verbatim copy of the matching ref-frame region.
    /// We also confirm the residual decode runs without panicking when
    /// every TX position has `eob = 0` (so the IT-add is bypassed and
    /// `dst` ends up exactly equal to the ref-plane region).
    #[test]
    fn single_ref_zero_mv_block8x8_copies_ref() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w,
            cur_h,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        // Pre-fill dst with sentinels so missing writes are visible.
        dst.y_plane_mut().fill(0x11);
        dst.u_plane_mut().fill(0x22);
        dst.v_plane_mut().fill(0x33);

        let fc = fc_flat_eob_inter();
        let bytes = encode_eob_zero();
        let mut reader = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        let mi = inter_mi(BlockSize::Block8x8, TxSize::Tx8x8, Mv::ZERO);

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader,
            &fc,
            None,
            &mut coef_ctx,
            &mi,
            /* mi_row */ 1,
            /* mi_col */ 2,
            /* mi_row_in_tile */ 1,
            /* mi_col_in_tile */ 2,
            /* mi_row_in_sb */ 1,
            &inputs,
            &mut dst,
            &test_dispatch(),
        );

        // The block lands at pixel (mi_col*8, mi_row*8) = (16, 8) in
        // both ref and dst. With Mv = 0 + ScaleFactors::identity +
        // subpel = 0, the MC kernel is convolve_copy. Confirm the
        // 8x8 luma region of dst equals the same region of ref.
        let dst_y = dst.y_plane();
        let ref_y = ref_buf.y_plane();
        for row in 0..8 {
            let dst_row = &dst_y.data()[(8 + row) * dst_y.stride() + 16..];
            let ref_row = &ref_y.data()[(8 + row) * ref_y.stride() + 16..];
            assert_eq!(
                &dst_row[..8],
                &ref_row[..8],
                "row {row} mismatch in MC dst",
            );
        }
        // Sentinel bytes outside the block remain.
        assert_eq!(dst_y.data()[0], 0x11);

        // Chroma: 4x4 region at (8, 4) (each chroma pixel covers 2
        // luma pixels under 4:2:0).
        let dst_u = dst.u_plane();
        let ref_u = ref_buf.u_plane();
        for row in 0..4 {
            let dst_row = &dst_u.data()[(4 + row) * dst_u.stride() + 8..];
            let ref_row = &ref_u.data()[(4 + row) * ref_u.stride() + 8..];
            assert_eq!(&dst_row[..4], &ref_row[..4], "U row {row}");
        }
    }

    /// D.2.b: NEARESTMV + skip block reconstructs to exactly the
    /// ref-frame pixels (no residual, MC output is final). We verify
    /// against a Mv = 0 single-ref block where MC is `convolve_copy`,
    /// so the result is byte-equal to the corresponding ref-frame
    /// region. The bool-coder is NOT advanced (skip path doesn't read
    /// tokens) — we confirm by holding the reader's position before
    /// and after.
    #[test]
    fn skip_block_zero_mv_block16x16_copies_ref_no_token_consume() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        dst.y_plane_mut().fill(0xAB);
        dst.u_plane_mut().fill(0xCD);
        dst.v_plane_mut().fill(0xEF);

        let fc = fc_flat_eob_inter();
        // Reader holds *no* token bytes — the skip path mustn't read.
        let bytes = vec![0u8; 4];
        let mut reader = BoolReader::new(&bytes).unwrap();
        let pre_pos = reader.cursor_state().3;
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        // Pre-set some coef-context cells to non-zero so we can check
        // the reset splat after the skip block lands.
        coef_ctx.set_coef_flags(0, 0, 0, 4, 4, true);

        let mut mi = inter_mi(BlockSize::Block16x16, TxSize::Tx16x16, Mv::ZERO);
        mi.skip = true;

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader,
            &fc,
            None,
            &mut coef_ctx,
            &mi,
            /* mi_row */ 0,
            /* mi_col */ 0,
            /* mi_row_in_tile */ 0,
            /* mi_col_in_tile */ 0,
            /* mi_row_in_sb */ 0,
            &inputs,
            &mut dst,
            &test_dispatch(),
        );

        // Reader unchanged.
        let post_pos = reader.cursor_state().3;
        assert_eq!(pre_pos, post_pos, "skip path must not consume bits");

        // Coef-context cells in the spanned region are all zero.
        for x4 in 0..4 {
            assert_eq!(coef_ctx.above_coef(0, x4), 0, "Y above[{x4}] reset");
        }
        for y4 in 0..4 {
            assert_eq!(coef_ctx.left_coef(0, y4), 0, "Y left[{y4}] reset");
        }

        // Y region at (0,0) equals the corresponding ref region.
        let dst_y = dst.y_plane();
        let ref_y = ref_buf.y_plane();
        for row in 0..16 {
            let dst_row = &dst_y.data()[row * dst_y.stride()..];
            let ref_row = &ref_y.data()[row * ref_y.stride()..];
            assert_eq!(&dst_row[..16], &ref_row[..16], "Y row {row}");
        }
    }

    /// D.2.b: skip with a NON-zero MV (subpel) — the MC output still
    /// is the final reconstruction, but it now equals a `convolve8_*`
    /// of the ref pixels, not a direct copy.
    #[test]
    fn skip_block_subpel_mv_no_residual() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        dst.y_plane_mut().fill(0);

        let fc = fc_flat_eob_inter();
        let bytes = vec![0u8; 4];
        let mut reader = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        let mut mi = inter_mi(
            BlockSize::Block16x16,
            TxSize::Tx16x16,
            Mv { row: 0, col: 5 },
        );
        mi.skip = true;

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader, &fc, None, &mut coef_ctx, &mi,
            2, 2, 2, 2, 2, &inputs, &mut dst,
            &test_dispatch(),
        );

        // Some pixels in the 16x16 region at (16, 16) are non-zero
        // (MC output landed).
        let dst_y = dst.y_plane();
        let mut any = false;
        for row in 0..16 {
            for col in 0..16 {
                if dst_y.data()[(16 + row) * dst_y.stride() + 16 + col] != 0 {
                    any = true;
                    break;
                }
            }
        }
        assert!(any, "MC output must be non-zero");
    }

    /// D.2.c: compound prediction. Two refs filled with distinct
    /// constants (Y plane = 0x40 and 0xC0). The compound average for
    /// Mv = 0 is `(0x40 + 0xC0 + 1) >> 1 = 0x80` per pixel — that's
    /// the libvpx `convolve_avg` formula. The test confirms (a) ref 1
    /// went through the `_avg` kernel rather than overwriting ref 0's
    /// prediction and (b) the chroma path mirrors the same compound
    /// average.
    #[test]
    fn compound_zero_mvs_avg_const_refs_block8x8() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        // Two refs with distinct constant fills.
        let mut ref0 = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        ref0.y_plane_mut().fill(0x40);
        ref0.u_plane_mut().fill(0x10);
        ref0.v_plane_mut().fill(0x20);
        ref0.extend_borders();

        let mut ref1 = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        ref1.y_plane_mut().fill(0xC0);
        ref1.u_plane_mut().fill(0x30);
        ref1.v_plane_mut().fill(0x60);
        ref1.extend_borders();

        let mut dst = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        dst.y_plane_mut().fill(0x77); // sentinel; should be overwritten

        let fc = fc_flat_eob_inter();
        let bytes = encode_eob_zero();
        let mut reader = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        // Compound inter mi: both refs valid.
        let mut mi = inter_mi(BlockSize::Block8x8, TxSize::Tx8x8, Mv::ZERO);
        if let BlockMode::Inter { ref_frames, .. } = &mut mi.mode {
            *ref_frames = [RefFrame::LastFrame, RefFrame::GoldenFrame];
        }

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref0), Some(&ref1)],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader, &fc, None, &mut coef_ctx, &mi,
            1, 1, 1, 1, 1, &inputs, &mut dst,
            &test_dispatch(),
        );

        // Y block at pixel (8, 8): every cell = (0x40 + 0xC0 + 1) >> 1
        // = 0x80.
        let dst_y = dst.y_plane();
        for row in 0..8 {
            for col in 0..8 {
                let v = dst_y.data()[(8 + row) * dst_y.stride() + 8 + col];
                assert_eq!(v, 0x80, "Y[{row},{col}] compound avg");
            }
        }

        // U block at chroma pixel (4, 4): (0x10 + 0x30 + 1) >> 1 = 0x20.
        let dst_u = dst.u_plane();
        for row in 0..4 {
            for col in 0..4 {
                let v = dst_u.data()[(4 + row) * dst_u.stride() + 4 + col];
                assert_eq!(v, 0x20, "U[{row},{col}] compound avg");
            }
        }

        // V block: (0x20 + 0x60 + 1) >> 1 = 0x40.
        let dst_v = dst.v_plane();
        for row in 0..4 {
            for col in 0..4 {
                let v = dst_v.data()[(4 + row) * dst_v.stride() + 4 + col];
                assert_eq!(v, 0x40, "V[{row},{col}] compound avg");
            }
        }
    }

    /// D.2.e: end-to-end integration. Drive
    /// `decode_and_reconstruct_inter_block` against a hand-encoded
    /// stream that produces a single-ref ZEROMV Block16x16 with
    /// skip = true. The reconstructed dst region must match the ref
    /// region (because Mv = 0 + identity scale + skip = direct copy).
    #[test]
    fn integration_zeromv_skip_single_ref_block16x16_copies_ref() {
        use rvp9_core::bitstream::ReferenceMode;
        use rvp9_core::inter::mv_ref::{BlockBounds, NeighbourLookup, Position, sign_bias_from_header};
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::BlockMode;

        // Empty neighbours — all contexts collapse to frame-edge defaults.
        struct NoNeighbours;
        impl NeighbourLookup for NoNeighbours {
            fn get(&self, _pos: Position) -> Option<&ModeInfo> {
                None
            }
        }

        // 1. Build a frame context + inter params.
        let fc = fc_flat_eob_inter();
        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (fixed, var) =
            rvp9_core::inter::mode_info::setup_compound_reference_mode(&sign_bias);
        let params = InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref: fixed,
            comp_var_ref: var,
            sign_bias,
        };

        // 2. Hand-encode the stream: single-ref LAST + ZEROMV. The
        //    bitstream prefix (segment_id / skip / tx_size / is_inter)
        //    is read by `decode_block` (NOT by
        //    `read_inter_block_mode_info`), so we start the stream at
        //    the ref-frame bits.
        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref single_ref_p1 ctx = 2 (no neighbours), bit = 0 -> LAST.
        let p_single = fc.single_ref_prob[2][0];
        w.write(0, p_single);
        // Inter mode: ctx = 0 (BOTH_PREDICTED from no neighbours).
        // ZEROMV tree path = bit 0.
        let p_inter0 = fc.inter_mode_probs[0][0];
        w.write(0, p_inter0);
        // Pad and close.
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("stop_encode");
        buf.truncate(size);

        // 3. Build a ref frame with a recognisable pattern.
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        dst.y_plane_mut().fill(0xCC); // sentinel

        // 4. Skeleton mi — only block_size / tx_size / skip set; the
        //    inter reader fills in the rest.
        let mut mi = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: [BmiInfo::default(); 4],
        };

        // 5. Drive.
        let mut reader = BoolReader::new(&buf).expect("init");
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);
        let bounds = BlockBounds::new(
            /* mi_row */ 0, /* mi_col */ 0,
            /* mi_rows */ 8, /* mi_cols */ 8,
            /* bw */ 2, /* bh */ 2,
        );
        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        decode_and_reconstruct_inter_block(
            &mut reader,
            &fc,
            &mut coef_ctx,
            &params,
            &NoNeighbours,
            &bounds,
            None,
            &mut mi,
            /* mi_row */ 0,
            /* mi_col */ 0,
            /* mi_row_in_tile */ 0,
            /* mi_col_in_tile */ 0,
            /* mi_row_in_sb */ 0,
            &inputs,
            &mut dst,
            &test_dispatch(),
        );

        // 6. Confirm mi.mode is Inter with expected values.
        match mi.mode {
            BlockMode::Inter { mode, ref_frames, mvs, .. } => {
                assert_eq!(mode, InterMode::ZeroMv);
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                assert_eq!(mvs, [Mv::ZERO, Mv::ZERO]);
            }
            _ => panic!("expected Inter mode after decode"),
        }

        // 7. Reconstructed Y region at (0, 0) equals ref Y region.
        let dst_y = dst.y_plane();
        let ref_y = ref_buf.y_plane();
        for row in 0..16 {
            let dst_row = &dst_y.data()[row * dst_y.stride()..];
            let ref_row = &ref_y.data()[row * ref_y.stride()..];
            assert_eq!(&dst_row[..16], &ref_row[..16], "Y row {row}");
        }
    }

    /// D.2.d: sub-8 inter block (`Block4x4`) with four distinct
    /// per-bmi MVs. The luma plane runs four 4×4 MC passes — one per
    /// bmi — and the chroma plane runs ONE pass over a 4×4 region with
    /// the four MVs averaged. We use Mv = 0 on all bmis so each sub
    /// region is a copy of the ref. The test also confirms the parent
    /// `mi.skip = true` path skips residual decode.
    #[test]
    fn sub8_block4x4_per_bmi_mc_zero_mvs() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w, cur_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        dst.y_plane_mut().fill(0x55);
        dst.u_plane_mut().fill(0x66);
        dst.v_plane_mut().fill(0x77);

        let fc = fc_flat_eob_inter();
        let bytes = vec![0u8; 4];
        let mut reader = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        // Sub-8 mi with skip = true (avoid residual path; we exercise
        // it in the next test).
        let mut mi = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO; 2],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: [BmiInfo::default(); 4],
        };
        // All four bmis hold Mv = 0.
        for bmi in &mut mi.bmi {
            bmi.as_mv = [Mv::ZERO; 2];
        }

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader, &fc, None, &mut coef_ctx, &mi,
            /* mi_row */ 1, /* mi_col */ 1,
            1, 1, 1, &inputs, &mut dst,
            &test_dispatch(),
        );

        // Y region at pixel (8, 8) is 8×8 (4 bmis * 4×4 each).
        let dst_y = dst.y_plane();
        let ref_y = ref_buf.y_plane();
        for row in 0..8 {
            let dst_row = &dst_y.data()[(8 + row) * dst_y.stride() + 8..];
            let ref_row = &ref_y.data()[(8 + row) * ref_y.stride() + 8..];
            assert_eq!(&dst_row[..8], &ref_row[..8], "Y row {row}");
        }

        // Chroma U at (4, 4) — 4×4 region.
        let dst_u = dst.u_plane();
        let ref_u = ref_buf.u_plane();
        for row in 0..4 {
            let dst_row = &dst_u.data()[(4 + row) * dst_u.stride() + 4..];
            let ref_row = &ref_u.data()[(4 + row) * ref_u.stride() + 4..];
            assert_eq!(&dst_row[..4], &ref_row[..4], "U row {row}");
        }
    }

    /// D.2.d helper unit: `round_mv_comp_q4` matches libvpx's
    /// sign-aware "round to nearest" rule.
    #[test]
    fn round_mv_comp_q4_matches_libvpx() {
        assert_eq!(super::round_mv_comp_q4(0), 0);
        assert_eq!(super::round_mv_comp_q4(2), 1);
        assert_eq!(super::round_mv_comp_q4(4), 1);
        assert_eq!(super::round_mv_comp_q4(8), 2);
        assert_eq!(super::round_mv_comp_q4(-2), -1);
        assert_eq!(super::round_mv_comp_q4(-4), -1);
        assert_eq!(super::round_mv_comp_q4(-8), -2);
    }

    /// Reduction guarantee (G7.1b B4): at ss=(1,1) — 4:2:0 — the new
    /// `average_split_mvs` reproduces the OLD `avg_split_mv_q4_4x4`
    /// behaviour (the all-4 q4 average) for every chroma sub-block.
    /// libvpx `average_split_mvs` ss_idx == 3 (`vp9_reconinter.c:120`).
    #[test]
    fn average_split_mvs_420_matches_old_q4_avg() {
        let sets: [[Mv; 4]; 3] = [
            [
                Mv { row: 8, col: -4 },
                Mv { row: 4, col: -8 },
                Mv { row: 4, col: -4 },
                Mv { row: 8, col: -8 },
            ],
            [
                Mv { row: -3, col: 7 },
                Mv { row: -5, col: 1 },
                Mv { row: 2, col: -9 },
                Mv { row: 6, col: 3 },
            ],
            [
                Mv { row: 0, col: 0 },
                Mv { row: 1, col: -1 },
                Mv { row: -2, col: 2 },
                Mv { row: 0, col: 0 },
            ],
        ];
        for set in &sets {
            let bmi = [
                BmiInfo { as_mode: 0, as_mv: [set[0], Mv::ZERO] },
                BmiInfo { as_mode: 0, as_mv: [set[1], Mv::ZERO] },
                BmiInfo { as_mode: 0, as_mv: [set[2], Mv::ZERO] },
                BmiInfo { as_mode: 0, as_mv: [set[3], Mv::ZERO] },
            ];
            // Old behaviour: round_mv_comp_q4 over all-4 sum, for any block.
            let row_sum: i32 = set.iter().map(|m| i32::from(m.row)).sum();
            let col_sum: i32 = set.iter().map(|m| i32::from(m.col)).sum();
            let expected = Mv {
                row: super::round_mv_comp_q4(row_sum) as i16,
                col: super::round_mv_comp_q4(col_sum) as i16,
            };
            for block in 0..4 {
                assert_eq!(
                    super::average_split_mvs(&bmi, 0, 1, 1, block),
                    expected,
                    "4:2:0 block {block} set {set:?}"
                );
            }
        }
    }

    /// Differential values (G7.1b B4) — hand-derived from the libvpx
    /// `average_split_mvs` switch (`vp9_reconinter.c:112-124`) for
    /// ss_idx 0/1/2. bmi MVs (row, col) per the 2×2 sub-block layout:
    ///   bmi[0]=(8,-4) bmi[1]=(4,-8) bmi[2]=(4,-4) bmi[3]=(8,-8)
    #[test]
    fn average_split_mvs_differential_444_440_422() {
        let bmi = [
            BmiInfo { as_mode: 0, as_mv: [Mv { row: 8, col: -4 }, Mv::ZERO] },
            BmiInfo { as_mode: 0, as_mv: [Mv { row: 4, col: -8 }, Mv::ZERO] },
            BmiInfo { as_mode: 0, as_mv: [Mv { row: 4, col: -4 }, Mv::ZERO] },
            BmiInfo { as_mode: 0, as_mv: [Mv { row: 8, col: -8 }, Mv::ZERO] },
        ];

        // ss_idx 0 (4:4:4, ss=0,0): case 0 — returns bmi[block] unchanged.
        for block in 0..4 {
            assert_eq!(
                super::average_split_mvs(&bmi, 0, 0, 0, block),
                bmi[block].as_mv[0],
                "4:4:4 block {block}"
            );
        }

        // ss_idx 1 (4:4:0, ss_x=0 ss_y=1): case 1 — mi_mv_pred_q2(block, block+2).
        // block 0 -> avg{bmi0,bmi2}: row (8+4)=12 -> (12+1)/2=6; col (-4-4)=-8 -> (-8-1)/2=-4.
        assert_eq!(super::average_split_mvs(&bmi, 0, 0, 1, 0), Mv { row: 6, col: -4 });
        // block 1 -> avg{bmi1,bmi3}: row (4+8)=12 -> 6; col (-8-8)=-16 -> (-16-1)/2=-8.
        assert_eq!(super::average_split_mvs(&bmi, 0, 0, 1, 1), Mv { row: 6, col: -8 });

        // ss_idx 2 (4:2:2, ss_x=1 ss_y=0): case 2 — mi_mv_pred_q2(block, block+1).
        // block 0 -> avg{bmi0,bmi1}: row (8+4)=12 -> 6; col (-4-8)=-12 -> (-12-1)/2=-6 (int /2 truncates).
        assert_eq!(super::average_split_mvs(&bmi, 0, 1, 0, 0), Mv { row: 6, col: -6 });
        // block 1 -> avg{bmi1,bmi2}: row (4+4)=8 -> (8+1)/2=4; col (-8-4)=-12 -> -6.
        assert_eq!(super::average_split_mvs(&bmi, 0, 1, 0, 1), Mv { row: 4, col: -6 });
    }

    /// D.2.a coverage: non-zero MV with sub-pel offset. With identity
    /// scale and a fractional MV (Q3 = 3 ⇒ Q4 = 6 ⇒ subpel_x_q4 = 6),
    /// the MC kernel becomes `convolve8_horiz` at phase 6. We verify
    /// the dst region differs from the zero-MV copy (sanity) and that
    /// the residual decode still consumes the right number of bits.
    #[test]
    fn single_ref_subpel_mv_block16x16_runs() {
        let cur_w = 64u32;
        let cur_h = 64u32;
        let ref_buf = make_ref(cur_w, cur_h);
        let mut dst = Yv12Buffer::alloc(
            cur_w,
            cur_h,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .unwrap();
        dst.y_plane_mut().fill(0);

        let fc = fc_flat_eob_inter();
        let bytes = encode_eob_zero();
        let mut reader = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(cur_w as usize / 8);

        // Q3 mv (col=3, row=0) → Q4 (col=6, row=0). Identity scale.
        // subpel_x_q4 = 6, subpel_y_q4 = 0 → convolve8_horiz.
        let mi = inter_mi(
            BlockSize::Block16x16,
            TxSize::Tx16x16,
            Mv { row: 0, col: 3 },
        );

        let inputs = InterBlockInputs {
            cur_y_width: cur_w,
            cur_y_height: cur_h,
            refs: [Some(&ref_buf), None],
            dequant_per_plane: [(8, 16); 3],
            lossless: false,
            mi_rows: 1 << 16,
            mi_cols: 1 << 16,
        };

        reconstruct_inter_block(
            &mut reader,
            &fc,
            None,
            &mut coef_ctx,
            &mi,
            /* mi_row */ 2,
            /* mi_col */ 2,
            /* mi_row_in_tile */ 2,
            /* mi_col_in_tile */ 2,
            /* mi_row_in_sb */ 2,
            &inputs,
            &mut dst,
            &test_dispatch(),
        );

        // After convolve8_horiz the dst region must not be all-zero (we
        // pre-filled with zeros). At least one pixel must differ.
        let dst_y = dst.y_plane();
        let mut any_nonzero = false;
        for row in 0..16 {
            for col in 0..16 {
                let v = dst_y.data()[(16 + row) * dst_y.stride() + 16 + col];
                if v != 0 {
                    any_nonzero = true;
                    break;
                }
            }
            if any_nonzero {
                break;
            }
        }
        assert!(any_nonzero, "MC must have written non-zero pixels");
    }

    // --- Phase D.5: extend_mc_border ---------------------------------
    //
    // Tests the wide-MV scratch-buffer fallback in
    // `build_inter_predictor_with_mc_border`. The fast (direct) path
    // covers any footprint that fits within the bordered region
    // (`±32` for Y / `±16` for UV under the decoder border setting);
    // the slow (scratch) path fires only when the footprint extends
    // past that. Both paths must produce the same result as libvpx's
    // equivalent code path — edge-replication from the visible region.

    /// Build a tiny ref frame whose visible pixels carry distinct
    /// values so edge replication can be observed in test outputs.
    fn ref_with_distinct_edge_pixels(crop_w: u32, crop_h: u32) -> Yv12Buffer {
        let mut buf = Yv12Buffer::alloc(
            crop_w, crop_h, ChromaFormat::Yuv420,
            BorderUsage::Decoder, ByteAlignment::Legacy,
        ).unwrap();
        // Y plane: each pixel encodes its (clamped) position so the
        // edge-replicated cells differ from interior cells.
        {
            let mut y = buf.y_plane_mut();
            let stride = y.stride();
            let w = y.width();
            let h = y.height();
            let data = y.data_mut();
            for r in 0..h {
                for c in 0..w {
                    data[r * stride + c] = ((r * 13 + c * 5) & 0xff) as u8;
                }
            }
        }
        buf.u_plane_mut().fill(0x55);
        buf.v_plane_mut().fill(0xAA);
        buf.extend_borders();
        buf
    }

    /// Sanity: a fully-inside block with zero MV produces a verbatim
    /// copy of the ref pixels (the direct fast path).
    #[test]
    fn mc_border_direct_path_inside_frame() {
        use super::build_inter_predictor_with_mc_border;
        use rvp9_core::inter::scale::ScaleFactors;

        let ref_buf = ref_with_distinct_edge_pixels(64, 64);
        let ref_plane = ref_buf.plane_with_border(0);
        let crop = super::RefPlaneCrop::for_plane(&ref_buf, 0);
        let sf = ScaleFactors::new(64, 64, 64, 64);

        let mut dst = [0x99u8; 16 * 16];
        // sub_px = (8, 8), mv = 0 → total_*_q4 = (8 << 4, 8 << 4).
        let total_x_q4 = sf.scaled_x_q4(8, 0);
        let total_y_q4 = sf.scaled_y_q4(8, 0);

        build_inter_predictor_with_mc_border(
            &ref_plane,
            crop,
            total_x_q4,
            total_y_q4,
            &sf,
            InterpFilter::EightTap,
            &mut dst,
            16,
            8, 8,
            false,
            rvp9_core::dsp::get_ref(),
        );

        let ref_y = ref_buf.y_plane();
        for r in 0..8 {
            for c in 0..8 {
                let want = ref_y.data()[(8 + r) * ref_y.stride() + 8 + c];
                assert_eq!(dst[r * 16 + c], want, "dst[{r},{c}]");
            }
        }
    }

    /// Footprint within the static border (e.g. MV = -28 pixels = Q3 =
    /// -224) still uses the direct path. The bordered region already
    /// holds the replicated edge values so the result matches the
    /// scratch path bit-for-bit.
    #[test]
    fn mc_border_within_static_border_uses_direct_path() {
        use super::build_inter_predictor_with_mc_border;
        use rvp9_core::inter::scale::ScaleFactors;

        let ref_buf = ref_with_distinct_edge_pixels(64, 64);
        let ref_plane = ref_buf.plane_with_border(0);
        let crop = super::RefPlaneCrop::for_plane(&ref_buf, 0);
        let sf = ScaleFactors::new(64, 64, 64, 64);

        // Block at (0, 0), MV = (-28, -28) Q3 → Q4 = -56 each axis.
        // int_x = int_y = -56/16 = -4 (rounded toward -inf via shift =
        // -56 >> 4 = -4 with subpel = 8 ... actually -56 & 0xF = 8 in
        // two's complement: -56 = 0xFFFFFFC8 → & 0xF = 8). The
        // resulting footprint with both pads is x0 = -4 - 3 = -7,
        // x1 = (-56 + 7*16) >> 4 + 1 + 4 = (-56+112)>>4 + 5 = 3 + 5 =
        // 8. All within `[-32, 64+32)`.
        let total_x_q4 = sf.scaled_x_q4(0, -56);
        let total_y_q4 = sf.scaled_y_q4(0, -56);

        let mut dst_direct = [0u8; 8 * 8];
        build_inter_predictor_with_mc_border(
            &ref_plane,
            crop,
            total_x_q4,
            total_y_q4,
            &sf,
            InterpFilter::EightTap,
            &mut dst_direct,
            8,
            8, 8,
            false,
            rvp9_core::dsp::get_ref(),
        );

        // Smoke: result must contain at least one non-zero pixel
        // (interior of the ref is non-zero almost everywhere).
        assert!(dst_direct.iter().any(|&p| p != 0));
    }

    /// MV that points well beyond the static border (e.g. 200 pixels
    /// off the bottom-right corner of a 32×32 frame) fires the
    /// scratch path. All output pixels must equal the visible-corner
    /// pixel (edge-replication).
    #[test]
    fn mc_border_far_outside_static_border_uses_scratch() {
        use super::build_inter_predictor_with_mc_border;
        use rvp9_core::inter::scale::ScaleFactors;

        let ref_buf = ref_with_distinct_edge_pixels(32, 32);
        let ref_plane = ref_buf.plane_with_border(0);
        let crop = super::RefPlaneCrop::for_plane(&ref_buf, 0);
        let sf = ScaleFactors::new(32, 32, 32, 32);

        // Block at (0, 0) with MV pushing into far bottom-right
        // (Q3 = 200 * 8 = 1600 each axis). int_x = int_y = 1600 / 16 =
        // 100. With pads, x1 ≈ 100 + 8 + 4 = 112 — well past
        // `aligned + border = 32 + 32 = 64`, so scratch fires.
        let total_x_q4 = sf.scaled_x_q4(0, 1600);
        let total_y_q4 = sf.scaled_y_q4(0, 1600);

        let mut dst = [0u8; 8 * 8];
        build_inter_predictor_with_mc_border(
            &ref_plane,
            crop,
            total_x_q4,
            total_y_q4,
            &sf,
            InterpFilter::EightTap,
            &mut dst,
            8,
            8, 8,
            false,
            rvp9_core::dsp::get_ref(),
        );

        // All 8x8 pixels = bottom-right visible pixel of the ref.
        let ref_y = ref_buf.y_plane();
        let bottom_right = ref_y.data()[31 * ref_y.stride() + 31];
        for px in &dst {
            assert_eq!(*px, bottom_right, "scratch must replicate corner pixel");
        }
    }

    /// MV pointing far off the top-left corner: scratch must replicate
    /// from visible[0, 0].
    #[test]
    fn mc_border_top_left_scratch_replicates_corner() {
        use super::build_inter_predictor_with_mc_border;
        use rvp9_core::inter::scale::ScaleFactors;

        let ref_buf = ref_with_distinct_edge_pixels(32, 32);
        let ref_plane = ref_buf.plane_with_border(0);
        let crop = super::RefPlaneCrop::for_plane(&ref_buf, 0);
        let sf = ScaleFactors::new(32, 32, 32, 32);

        // Block at (0, 0) with MV pushing far into top-left
        // (Q3 = -200 * 8 = -1600). int_* = -100. With pads, x0 ≈
        // -100 - 3 = -103. `-32` < -103 → scratch fires.
        let total_x_q4 = sf.scaled_x_q4(0, -1600);
        let total_y_q4 = sf.scaled_y_q4(0, -1600);

        let mut dst = [0u8; 8 * 8];
        build_inter_predictor_with_mc_border(
            &ref_plane,
            crop,
            total_x_q4,
            total_y_q4,
            &sf,
            InterpFilter::EightTap,
            &mut dst,
            8,
            8, 8,
            false,
            rvp9_core::dsp::get_ref(),
        );

        let ref_y = ref_buf.y_plane();
        let top_left = ref_y.data()[0];
        for px in &dst {
            assert_eq!(*px, top_left, "top-left scratch must replicate corner pixel");
        }
    }

    /// Differential: scratch path and direct path produce identical
    /// output when both can read the same footprint. We choose an MV
    /// that lands the footprint inside the static border (so direct
    /// fires), then run the scratch fill manually against the same
    /// footprint and convolve; pixels must match.
    #[test]
    fn mc_border_scratch_matches_direct_for_inside_footprint() {
        use rvp9_core::inter::scale::ScaleFactors;
        use rvp9_core::inter::scale::inter_predictor;

        let ref_buf = ref_with_distinct_edge_pixels(64, 64);
        let ref_plane = ref_buf.plane_with_border(0);
        let crop = super::RefPlaneCrop::for_plane(&ref_buf, 0);
        let sf = ScaleFactors::new(64, 64, 64, 64);

        // Block at (16, 16), MV = (3, 5) Q3 → Q4 = (6, 10). Subpel
        // both axes; both pads fire. Footprint comfortably inside.
        let total_x_q4 = sf.scaled_x_q4(16, 10);
        let total_y_q4 = sf.scaled_y_q4(16, 6);

        // -- Direct path output (via the new helper).
        let mut dst_helper = [0u8; 8 * 8];
        super::build_inter_predictor_with_mc_border(
            &ref_plane,
            crop,
            total_x_q4,
            total_y_q4,
            &sf,
            InterpFilter::EightTap,
            &mut dst_helper,
            8,
            8, 8,
            false,
            rvp9_core::dsp::get_ref(),
        );

        // -- Same footprint via an explicit scratch fill + convolve.
        let int_x = total_x_q4 >> SUBPEL_BITS;
        let int_y = total_y_q4 >> SUBPEL_BITS;
        let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
        let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;
        let x0 = int_x - 3;
        let y0 = int_y - 3;
        // libvpx b_* = (x1 - x0 + 1) with x1 = int_x + w + 4 - 1 (the
        // last touched column = int_x + w + 3 inclusive of 7-tap tail
        // since x_pad fires); resulting b_w = w + 7 + 1 = 16.
        let b_w = 8 + 7 + 1;
        let b_h = 8 + 7 + 1;
        let mut scratch = [0u8; 16 * 16];
        // Visible top-left of ref_plane.
        let visible_base = ref_plane.border_h * ref_plane.stride + ref_plane.border_w;
        super::build_mc_border(
            &ref_plane.data[visible_base..],
            ref_plane.stride,
            &mut scratch,
            b_w,
            x0,
            y0,
            b_w,
            b_h,
            crop.crop_w as i32,
            crop.crop_h as i32,
        );

        let mut dst_scratch = [0u8; 8 * 8];
        inter_predictor(
            &scratch[..],
            b_w,
            &mut dst_scratch,
            8,
            subpel_x_q4,
            subpel_y_q4,
            &sf,
            InterpFilter::EightTap,
            8,
            8,
            false,
        );

        assert_eq!(dst_helper, dst_scratch);
    }

    /// `build_mc_border` directly: top-left replication (negative `x`
    /// and `y` start), with a 4-row × 4-col footprint reading from a
    /// known visible region.
    #[test]
    fn build_mc_border_replicates_top_left() {
        // 2×2 visible region; values: [[0xA0, 0xA1], [0xB0, 0xB1]].
        let src = [
            0xA0u8, 0xA1, 0x00, 0x00,
            0xB0,   0xB1, 0x00, 0x00,
            0x00,   0x00, 0x00, 0x00,
            0x00,   0x00, 0x00, 0x00,
        ];
        let mut dst = [0u8; 16]; // 4x4 footprint
        super::build_mc_border(
            &src,
            /* src_stride */ 4,
            &mut dst,
            /* dst_stride */ 4,
            /* x */ -2,
            /* y */ -2,
            /* b_w */ 4,
            /* b_h */ 4,
            /* w */ 2,
            /* h */ 2,
        );
        // Row -2 / -1 → clamp to row 0 → [0xA0, 0xA0, 0xA0, 0xA1]
        // (left=2 fills 0xA0, copy=2 reads [0,1]=A0,A1; right=0).
        // Wait: with x=-2, b_w=4, w=2: left = 2, right = max(0, -2+4-2) = 0.
        // copy = 4 - 2 - 0 = 2. left fill = row[0]=0xA0; copy = row[0..2]=A0,A1.
        // So expected row = [A0, A0, A0, A1].
        assert_eq!(&dst[0..4], &[0xA0, 0xA0, 0xA0, 0xA1]);
        assert_eq!(&dst[4..8], &[0xA0, 0xA0, 0xA0, 0xA1]);
        // Row 0 → [0xA0, 0xA0, 0xA0, 0xA1] same fill pattern.
        assert_eq!(&dst[8..12], &[0xA0, 0xA0, 0xA0, 0xA1]);
        // Row 1 → [0xB0, 0xB0, 0xB0, 0xB1].
        assert_eq!(&dst[12..16], &[0xB0, 0xB0, 0xB0, 0xB1]);
    }

    /// `build_mc_border` directly: bottom-right replication.
    #[test]
    fn build_mc_border_replicates_bottom_right() {
        let src = [
            0xA0u8, 0xA1, 0x00, 0x00,
            0xB0,   0xB1, 0x00, 0x00,
            0x00,   0x00, 0x00, 0x00,
            0x00,   0x00, 0x00, 0x00,
        ];
        let mut dst = [0u8; 16];
        super::build_mc_border(
            &src, 4, &mut dst, 4,
            /* x */ 0, /* y */ 0,
            /* b_w */ 4, /* b_h */ 4,
            /* w */ 2,  /* h */ 2,
        );
        // Row 0 (y=0): left=0, right=max(0, 0+4-2)=2, copy=2. fill = row[w-1]=row[1]=0xA1.
        //   → [0xA0, 0xA1, 0xA1, 0xA1].
        assert_eq!(&dst[0..4], &[0xA0, 0xA1, 0xA1, 0xA1]);
        // Row 1: [0xB0, 0xB1, 0xB1, 0xB1].
        assert_eq!(&dst[4..8], &[0xB0, 0xB1, 0xB1, 0xB1]);
        // Row 2 (y=2): y >= h → clamp to row h-1 = row 1.
        //   → [0xB0, 0xB1, 0xB1, 0xB1].
        assert_eq!(&dst[8..12], &[0xB0, 0xB1, 0xB1, 0xB1]);
        // Row 3: still clamped.
        assert_eq!(&dst[12..16], &[0xB0, 0xB1, 0xB1, 0xB1]);
    }
}
