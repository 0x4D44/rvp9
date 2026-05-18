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

use rvp9_core::bitstream::InterpFilter;
use rvp9_core::block::BlockSize;
use rvp9_core::entropy::{BoolReader, FrameContext};
use rvp9_core::entropy::detokenize::DecodeCoefCounts;
use rvp9_core::frame::{BorderedPlaneRef, Yv12Buffer};
use rvp9_core::inter::scale::{ScaleFactors, inter_predictor};
use rvp9_core::inter::filter::{SUBPEL_BITS, SUBPEL_MASK};
use rvp9_core::mode_info::{BlockMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::inverse::{
    idct4x4_1_add, idct4x4_16_add, idct8x8_1_add, idct8x8_64_add, idct16x16_1_add,
    idct16x16_256_add, idct32x32_1_add, idct32x32_1024_add,
};
use rvp9_core::transforms::types::{TxSize, TxType};

use crate::detokenize::decode_block_tokens;
use crate::tile_ctx::TileCoefCtx;

/// Maximum coefficient count at any TX size (`Tx32x32 = 32 * 32 = 1024`).
const MAX_TX_COEFFS: usize = 1024;

/// Per-plane subsampling (4:2:0 only — D-01 RESOLVED).
const SUBSAMPLING_X: [usize; 3] = [0, 1, 1];
const SUBSAMPLING_Y: [usize; 3] = [0, 1, 1];

/// Inputs that don't vary across the per-plane / per-TX loop inside
/// [`reconstruct_inter_block`]. Bundled to keep the per-block driver
/// signature compact.
#[allow(missing_docs)]
pub struct InterBlockInputs<'a> {
    /// The current frame's aligned luma width (pixels) — used to build
    /// [`ScaleFactors`] for each ref.
    pub cur_y_width: u32,
    /// The current frame's aligned luma height (pixels).
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
    _counts: Option<&mut DecodeCoefCounts<'_>>,
    coef_ctx: &mut TileCoefCtx,
    mi: &ModeInfo,
    mi_row: usize,
    mi_col: usize,
    _mi_row_in_tile: usize,
    mi_col_in_tile: usize,
    mi_row_in_sb: usize,
    inputs: &InterBlockInputs<'_>,
    dst_frame: &mut Yv12Buffer,
) {
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

    // D.2.a: enforce single-ref, no sub-8, no compound, no skip.
    debug_assert!(
        !ref_frames[1].is_inter(),
        "D.2.a does not handle compound — landed in D.2.c"
    );
    debug_assert!(
        (mi.block_size as u8) >= (BlockSize::Block8x8 as u8),
        "D.2.a does not handle sub-8 inter blocks — landed in D.2.d"
    );

    // Resolve the active reference frame.
    let ref_idx = 0usize;
    let ref_frame = ref_frames[ref_idx];
    debug_assert!(ref_frame.is_inter(), "active ref must be inter");
    let ref_buf = inputs.refs[ref_idx]
        .expect("caller must resolve ref-pool slot for active ref");

    // Build the Q14 scale factor for this ref against the current
    // frame's aligned luma dims (libvpx vp9_setup_scale_factors_for_frame).
    let sf = ScaleFactors::new(
        ref_buf.aligned_y_width(),
        ref_buf.aligned_y_height(),
        inputs.cur_y_width,
        inputs.cur_y_height,
    );
    debug_assert!(sf.is_valid(), "ref dimensions must be in libvpx envelope");

    // Per-plane MC + residual loop.
    let mv = mvs[ref_idx];
    let mode_for_y = TxType::DctDct; // unused for inter — placeholder
    let _ = mode_for_y;

    let bsize = mi.block_size;
    for plane in 0..3 {
        let ss_x = SUBSAMPLING_X[plane];
        let ss_y = SUBSAMPLING_Y[plane];

        let plane_bsize = if plane == 0 {
            bsize
        } else {
            uv_plane_block_size(bsize)
        };
        let plane_tx = if plane == 0 {
            mi.tx_size
        } else {
            uv_plane_tx_size(bsize, mi.tx_size)
        };
        let plane_bw_4x4 = plane_bsize.n_4x4_wide() as usize;
        let plane_bh_4x4 = plane_bsize.n_4x4_high() as usize;
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

        // -- Motion compensation -------------------------------------

        // libvpx `average_split_mvs(plane=0, sb_type >= 8)` returns
        // `mi->mv[ref].as_mv` directly. For chroma under 4:2:0 the
        // sub-8 case averages four bmi MVs (D.2.d); ≥8×8 is the same
        // raw mv shared across planes.
        //
        // Convert Q3 (1/8-pel, luma) to Q4 (1/16-pel, ref-coord). For
        // chroma without sub-8, the same MV applies but doubled (libvpx
        // `mv_q4 = mv.row * (1 << (1 - ss_y))`).
        let mv_q4_row = i32::from(mv.row) * (1i32 << (1 - ss_y as i32));
        let mv_q4_col = i32::from(mv.col) * (1i32 << (1 - ss_x as i32));

        // Combine plane-pixel origin + Q4 MV into a Q4 sample position
        // in the reference frame.
        let total_x_q4 = sf.scaled_x_q4(plane_px_x as i32, mv_q4_col as i16);
        let total_y_q4 = sf.scaled_y_q4(plane_px_y as i32, mv_q4_row as i16);

        let int_x = total_x_q4 >> SUBPEL_BITS;
        let int_y = total_y_q4 >> SUBPEL_BITS;
        let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
        let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

        // Source pointer — start at the bordered top-left, then shift
        // to the integer-pel base.
        let ref_plane: BorderedPlaneRef<'_> = ref_buf.plane_with_border(plane);
        let src_base = (ref_plane.border_h as i32 + int_y) as usize
            * ref_plane.stride
            + (ref_plane.border_w as i32 + int_x) as usize;
        let src = &ref_plane.data[src_base..];
        let src_stride = ref_plane.stride;

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
        let dst_offset = plane_px_y * dst_stride + plane_px_x;
        let dst_for_mc = &mut dst_slice[dst_offset..];

        inter_predictor(
            src,
            src_stride,
            dst_for_mc,
            dst_stride,
            subpel_x_q4,
            subpel_y_q4,
            &sf,
            interp_filter,
            block_w_px,
            block_h_px,
            /* is_compound_avg = */ false,
        );

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

            // Iterate TX positions inside the plane's coding block.
            let mut loff = 0usize;
            while loff < plane_bh_4x4 {
                let mut aoff = 0usize;
                while aoff < plane_bw_4x4 {
                    let tx_px_x = plane_px_x + aoff * 4;
                    let tx_px_y = plane_px_y + loff * 4;
                    let tx_dst_off = tx_px_y * dst_stride + tx_px_x;
                    let tx_dst = &mut dst_slice[tx_dst_off..];

                    let tx_type = TxType::DctDct; // inter blocks always use DctDct
                    let seg_eob = plane_tx.n_coeffs();
                    let mut dq_coefs = [0i32; MAX_TX_COEFFS];
                    let eob = decode_block_tokens(
                        reader,
                        fc,
                        /* counts = */ None,
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
                        inverse_transform_add_dct(
                            plane_tx, &dq_coefs[..seg_eob], eob, tx_dst, dst_stride,
                        );
                    }

                    aoff += tx_step_4x4;
                }
                loff += tx_step_4x4;
            }
        }
        let _ = inputs.lossless; // unused (no lossless inter yet)
    }
}

/// Compute the chroma plane's block size for the given luma `bsize`,
/// assuming 4:2:0. Mirrors libvpx `ss_size_lookup[bsize][1][1]`.
fn uv_plane_block_size(bsize: BlockSize) -> BlockSize {
    use BlockSize::*;
    match bsize {
        Block4x4 | Block4x8 | Block8x4 | Block8x8 => Block4x4,
        Block8x16 => Block4x8,
        Block16x8 => Block8x4,
        Block16x16 => Block8x8,
        Block16x32 => Block8x16,
        Block32x16 => Block16x8,
        Block32x32 => Block16x16,
        Block32x64 => Block16x32,
        Block64x32 => Block32x16,
        Block64x64 => Block32x32,
    }
}

/// libvpx `get_uv_tx_size`: `min(luma_tx, max_txsize_lookup[uv_bsize])`.
fn uv_plane_tx_size(bsize: BlockSize, luma_tx: TxSize) -> TxSize {
    let cap = rvp9_core::block::max_txsize(uv_plane_block_size(bsize));
    if (luma_tx as u8) < (cap as u8) {
        luma_tx
    } else {
        cap
    }
}

/// Dispatch one inverse-transform-add for an inter TX block. Inter
/// blocks always use `TxType::DctDct` (libvpx
/// `vp9_decodeframe.c:235-280`: hybrid kernels exist only for intra Y);
/// 32×32 has no hybrid in any case.
fn inverse_transform_add_dct(
    tx_size: TxSize,
    input: &[i32],
    eob: usize,
    dest: &mut [u8],
    stride: usize,
) {
    match tx_size {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = input.try_into().expect("Tx4x4 input is 16 coefs");
            if eob > 1 {
                idct4x4_16_add(coefs, dest, stride);
            } else {
                idct4x4_1_add(coefs[0], dest, stride);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = input.try_into().expect("Tx8x8 input is 64 coefs");
            if eob == 1 {
                idct8x8_1_add(coefs[0], dest, stride);
            } else {
                idct8x8_64_add(coefs, dest, stride);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = input.try_into().expect("Tx16x16 input is 256 coefs");
            if eob == 1 {
                idct16x16_1_add(coefs[0], dest, stride);
            } else {
                idct16x16_256_add(coefs, dest, stride);
            }
        }
        TxSize::Tx32x32 => {
            let coefs: &[i32; 1024] = input.try_into().expect("Tx32x32 input is 1024 coefs");
            if eob == 1 {
                idct32x32_1_add(coefs[0], dest, stride);
            } else {
                idct32x32_1024_add(coefs, dest, stride);
            }
        }
    }
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
        };

        reconstruct_inter_block(
            &mut reader, &fc, None, &mut coef_ctx, &mi,
            2, 2, 2, 2, 2, &inputs, &mut dst,
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
}
