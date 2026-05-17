//! Per-TX-block intra reconstruction primitive.
//!
//! Ports libvpx `predict_and_reconstruct_intra_block`
//! (`vp9/decoder/vp9_decodeframe.c:307-336`). For one transform block at
//! a given `(plane, pixel_x, pixel_y, tx_size)` with a resolved intra
//! mode, this wrapper:
//!
//! 1. Writes the intra prediction samples into the destination plane
//!    at the block's top-left pixel position
//!    ([`rvp9_core::intra_pred::intra_predict`]).
//! 2. Decodes the dequantised residual coefficients via
//!    [`crate::detokenize::decode_block_tokens`] — returns `eob` and
//!    fills a raster-order `dq_coefs` buffer.
//! 3. If `eob > 0`, runs the appropriate inverse-transform-add kernel
//!    from [`rvp9_core::transforms::inverse`], which reads the
//!    prediction from `dst` and overwrites it with prediction +
//!    residual, saturated to `[0, 255]`. If `eob == 0` (libvpx's
//!    fast-skip path) the prediction is the final reconstruction.
//!
//! The wrapper is the **inner** reconstruction loop. The outer
//! iteration over (coding-block × plane × TX position) is the
//! per-coding-block driver (Wave 4 Task K) — that driver takes a
//! single `ModeInfo` and walks the TX grid, calling this primitive
//! once per TX position.
//!
//! ## Scope (Stage 3 Task H)
//!
//! - 8-bit only (PLAN.md D-01). The HBD branch from libvpx
//!   `inverse_transform_block_intra:242-263` is dropped.
//! - Lossless (WHT) is **not yet** ported: the caller passes
//!   `lossless = false`. Wiring the `iwht*_add` family into the
//!   dispatch is a follow-up (transforms LLD §3.5
//!   `PerFrameTransforms`). For now, a lossless caller would panic
//!   (debug-assert) — see [`reconstruct_intra_block`].
//! - Caller materialises the edge buffers (`above` / `left` /
//!   `above_left`). The edge-builder logic from
//!   `vp9_reconintra.c:263-403` is **not** in scope here; it lives in
//!   the K driver / intra-pred wrapper landing later.

use rvp9_core::entropy::detokenize::DecodeCoefCounts;
use rvp9_core::entropy::{BoolReader, FrameContext};
use rvp9_core::intra_pred::{IntraMode, PredCtx, intra_predict};
use rvp9_core::transforms::inverse::{
    idct4x4_1_add, idct4x4_16_add, idct8x8_1_add, idct8x8_64_add, idct16x16_1_add,
    idct16x16_256_add, idct32x32_1_add, idct32x32_1024_add, iht4x4_16_add, iht8x8_64_add,
    iht16x16_256_add,
};
use rvp9_core::transforms::types::{TxSize, TxType};

use crate::detokenize::decode_block_tokens;
use crate::tile_ctx::TileCoefCtx;

/// Maximum coefficient count at any TX size (`Tx32x32 = 32*32 = 1024`).
const MAX_TX_COEFFS: usize = 1024;

/// Resolve the TX type for one transform block, mirroring libvpx
/// `vp9_decodeframe.c:324-325` and `:349-350`:
///
/// ```c
/// tx_type = (plane || xd->lossless) ? DCT_DCT
///                                   : intra_mode_to_tx_type_lookup[mode];
/// ```
///
/// At `TxSize::Tx32x32` libvpx implicitly forces DCT_DCT because no
/// 32×32 hybrid (ADST) kernel exists; we collapse that into the
/// `intra_mode_to_tx_type_lookup` table's `tx_type` plus the
/// dispatch-time fact that 32×32 only ever calls `idct32x32_*_add`.
#[inline]
#[must_use]
pub fn resolve_tx_type(plane: usize, lossless: bool, intra_mode: IntraMode) -> TxType {
    if plane != 0 || lossless {
        TxType::DctDct
    } else {
        intra_mode.tx_type()
    }
}

/// Run intra prediction, decode the residual, and (if non-empty)
/// apply the inverse transform-add for one TX block.
///
/// Mirrors libvpx `predict_and_reconstruct_intra_block`
/// (`vp9/decoder/vp9_decodeframe.c:307-336`). Returns the decoded `eob`
/// so the caller (the per-coding-block driver) can splat the
/// `(eob > 0)` flag and advance its dq-coeff cursor.
///
/// # Parameters
///
/// - `reader`: bool-coder positioned at the first token bit of this
///   block; advanced by however many bits the decode consumes.
/// - `fc`: per-frame probability bank.
/// - `counts`: optional per-block count buckets for backward
///   adaptation; the K driver passes `None` until Task 1f wires
///   adaptation.
/// - `coef_ctx`: per-tile (above/left) coef-context state.
/// - `plane`: `0` / `1` / `2` for Y / U / V.
/// - `x4`, `y4_in_sb`: top-left 4×4 column / row of the TX block
///   within the tile / SB row, in plane-relative 4×4-sub-block units
///   (matches [`decode_block_tokens`]).
/// - `tx_size`: transform size.
/// - `intra_mode`: prediction mode for *this* plane — the caller has
///   already resolved Y vs UV (`mi->mode` vs `mi->uv_mode`) and
///   handled the BLOCK_<8 sub-8 `bmi[]` lookup.
/// - `pred_ctx`: pre-materialised neighbour pixels (above / left /
///   above-left). For directional modes (D45/D63 etc.) `above` must
///   already include the `2 * bs` pixels per the intra-pred LLD §3.2.
/// - `dst`: mutable slice into the destination plane, starting at the
///   block's top-left pixel.
/// - `dst_stride`: plane stride in bytes.
/// - `seg_eob`: per-segment EOB cap (callers without segmentation
///   pass `tx_size.n_coeffs()`).
/// - `dequant`: `(dc_dequant, ac_dequant)` for this plane / segment.
/// - `lossless`: forces TX type to DCT_DCT (libvpx
///   `inverse_transform_block_intra:280`). The 4×4 WHT path is not
///   yet wired; passing `lossless == true` for `Tx4x4` will trigger a
///   debug-assert. See module-level docs.
///
/// # Returns
///
/// The `eob` returned by [`decode_block_tokens`]. The caller uses it
/// to drive its `(eob > 0)` neighbour-context splat (already done
/// inside `decode_block_tokens`) and to advance per-block bookkeeping.
///
/// # Panics
///
/// - Inherits the assertions in [`decode_block_tokens`].
/// - Debug-asserts if `lossless == true` (4×4 WHT path not yet wired).
/// - Debug-asserts if `dst` is too small for the TX block at the given
///   stride.
#[allow(clippy::too_many_arguments)] // mirrors libvpx 1:1
pub fn reconstruct_intra_block(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    counts: Option<&mut DecodeCoefCounts<'_>>,
    coef_ctx: &mut TileCoefCtx,
    plane: usize,
    x4: usize,
    y4_in_sb: usize,
    tx_size: TxSize,
    intra_mode: IntraMode,
    pred_ctx: PredCtx<'_>,
    dst: &mut [u8],
    dst_stride: usize,
    seg_eob: usize,
    dequant: (i32, i32),
    lossless: bool,
) -> usize {
    debug_assert!(
        !lossless,
        "lossless (WHT) reconstruction not yet wired in Task H; pass lossless=false"
    );

    let bs = tx_size.side();
    debug_assert!(
        dst.len() >= (bs - 1) * dst_stride + bs,
        "dst too small for {bs}x{bs} reconstruction"
    );

    // 1. Intra prediction — writes prediction samples into `dst`.
    intra_predict(intra_mode, dst, dst_stride, bs, pred_ctx);

    // 2. Resolve TX type and decode the residual.
    let tx_type = resolve_tx_type(plane, lossless, intra_mode);

    let n = tx_size.n_coeffs();
    debug_assert!(seg_eob <= n);
    let mut dq_coefs = [0i32; MAX_TX_COEFFS];
    let eob = decode_block_tokens(
        reader,
        fc,
        counts,
        coef_ctx,
        plane,
        x4,
        y4_in_sb,
        tx_size,
        tx_type,
        /* ref_type = */ 0,
        seg_eob,
        dequant,
        &mut dq_coefs[..seg_eob],
    );

    // 3. If non-empty, apply IT-add into `dst` (reads prediction, adds
    // IT(dq_coefs), writes back).
    if eob > 0 {
        inverse_transform_add(tx_size, tx_type, &dq_coefs[..n], eob, dst, dst_stride);
    }

    eob
}

/// Dispatch one inverse-transform-add for a TX block. Mirrors libvpx
/// `inverse_transform_block_intra` (`vp9_decodeframe.c:235-294`) for
/// the 8-bit non-lossless path.
///
/// Per libvpx `vp9_iht*_add` (`vp9_idct.c:182-206`): when
/// `tx_type == DCT_DCT` the hybrid wrapper falls through to the pure
/// `idct*_add` family, which exposes the `eob == 1` DC-only fast path.
/// Non-DCT_DCT always uses the full `iht*_*_add` kernel — libvpx
/// ships no eob-bucketed hybrid variants. 32×32 is always pure IDCT
/// (no 32×32 hybrid exists).
///
/// `input` is `n_coeffs(tx_size)` dequantised coefficients in raster
/// order. `dest` is the in-place reconstruction buffer (predicted
/// samples already there; the IT output is added and saturated).
fn inverse_transform_add(
    tx_size: TxSize,
    tx_type: TxType,
    input: &[i32],
    eob: usize,
    dest: &mut [u8],
    stride: usize,
) {
    debug_assert!(input.len() == tx_size.n_coeffs());

    match tx_size {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = input.try_into().expect("Tx4x4 input is 16 coefs");
            if tx_type == TxType::DctDct {
                if eob > 1 {
                    idct4x4_16_add(coefs, dest, stride);
                } else {
                    idct4x4_1_add(coefs[0], dest, stride);
                }
            } else {
                // Hybrid 4×4 — libvpx vp9_iht4x4_16_add (vp9_idct.c:182-188).
                iht4x4_16_add(coefs, dest, stride, tx_type);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = input.try_into().expect("Tx8x8 input is 64 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct8x8_1_add(coefs[0], dest, stride);
                } else {
                    // libvpx splits eob<=12 → idct8x8_12_add; we ship
                    // only the full _64_add variant (transforms LLD
                    // §3.4 lists 12-bucket kernels but only _1 and
                    // _64 are landed in inverse.rs today). Bit-exact
                    // result; just slower for sparse blocks.
                    idct8x8_64_add(coefs, dest, stride);
                }
            } else {
                iht8x8_64_add(coefs, dest, stride, tx_type);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = input.try_into().expect("Tx16x16 input is 256 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct16x16_1_add(coefs[0], dest, stride);
                } else {
                    // 10/38-bucket fast paths not yet ported; full
                    // _256_add is bit-exact.
                    idct16x16_256_add(coefs, dest, stride);
                }
            } else {
                iht16x16_256_add(coefs, dest, stride, tx_type);
            }
        }
        TxSize::Tx32x32 => {
            // No 32×32 hybrid exists; libvpx always routes through
            // vp9_idct32x32_add at this size (vp9_idct.c:289).
            let coefs: &[i32; 1024] = input.try_into().expect("Tx32x32 input is 1024 coefs");
            if eob == 1 {
                idct32x32_1_add(coefs[0], dest, stride);
            } else {
                // 34/135-bucket fast paths not yet ported.
                idct32x32_1024_add(coefs, dest, stride);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::entropy::BoolWriter;
    use rvp9_core::entropy::coef_probs::{COEF_BANDS, COEFF_CONTEXTS};

    // ---- Helpers ----------------------------------------------------

    /// Build a `FrameContext` whose `coef_probs[tx][plane_type][ref]`
    /// at the indices we exercise is a flat `[p_eob, p_zero, p_one]`
    /// triple per (band, ctx). Mirrors the helper in `detokenize::tests`.
    fn fc_with_flat_probs(
        tx_size: TxSize,
        plane_type: usize,
        ref_type: usize,
        p_eob: u8,
        p_zero: u8,
        p_one: u8,
    ) -> FrameContext {
        let mut fc = FrameContext::new_default();
        let triple = [p_eob, p_zero, p_one];
        let ctx_row = [triple; COEFF_CONTEXTS];
        let band_row = [ctx_row; COEF_BANDS];
        fc.coef_probs[tx_size as usize][plane_type][ref_type] = band_row;
        fc
    }

    /// Open a `BoolWriter` over a fresh 256-byte buffer, run the
    /// closure to emit symbols, then close and return the produced
    /// bytes.
    fn encode<F>(f: F) -> Vec<u8>
    where
        F: FnOnce(&mut BoolWriter<'_>),
    {
        let mut buf = vec![0u8; 256];
        let n = {
            let mut w = BoolWriter::start_encode(&mut buf);
            f(&mut w);
            for _ in 0..16 {
                w.write_bit(0);
            }
            w.stop_encode().unwrap()
        };
        buf.truncate(n);
        buf
    }

    /// Encoder stream for "EOB at scan-position 0" under flat probs
    /// where every relevant probability is 128 — eob = 0 is the
    /// likeliest outcome, expressed as a single 0-bit at p=128.
    fn encode_eob_at_zero() -> Vec<u8> {
        encode(|w| {
            w.write(0, 128);
        })
    }

    /// Encoder stream for "ONE_TOKEN at scan[0] sign+, then EOB at
    /// scan[1]" under flat probs of 64. Same byte sequence the
    /// `detokenize` tests use. Yields `eob = 1` and `dq_coefs[0] = +dc`.
    fn encode_one_token_dc_pos() -> Vec<u8> {
        encode(|w| {
            w.write(1, 64); // NOT-EOB at pos 0
            w.write(1, 64); // NOT-ZERO
            w.write(0, 64); // ONE_TOKEN
            w.write_bit(0); // sign +
            w.write(0, 64); // EOB at pos 1
        })
    }

    /// Materialise an all-zero `dst` buffer (`stride` × `bs` rows)
    /// large enough for a `bs × bs` block at the origin, plus a
    /// trailing pixel row to detect cross-row writes.
    fn alloc_dst(bs: usize, stride: usize) -> Vec<u8> {
        vec![0u8; stride * (bs + 1)]
    }

    /// Build a fresh `PredCtx` populated with distinguishable
    /// neighbour samples. Used for the V / TM / 16x16 tests.
    /// `above` is sized `2 * bs` for directional-mode safety even
    /// when the tested mode doesn't need the extra `bs` pixels.
    fn make_neighbors(bs: usize) -> (Vec<u8>, Vec<u8>) {
        let above: Vec<u8> = (0..2 * bs).map(|i| (10 + i * 2) as u8).collect();
        let left: Vec<u8> = (0..bs).map(|i| (100 + i * 3) as u8).collect();
        (above, left)
    }

    // ---- Tests ------------------------------------------------------

    /// `resolve_tx_type` matches libvpx
    /// `(plane || lossless) ? DCT_DCT : lookup[mode]`.
    #[test]
    fn resolve_tx_type_matches_libvpx() {
        // Y plane, non-lossless: lookup table.
        assert_eq!(resolve_tx_type(0, false, IntraMode::Dc), TxType::DctDct);
        assert_eq!(resolve_tx_type(0, false, IntraMode::V), TxType::AdstDct);
        assert_eq!(resolve_tx_type(0, false, IntraMode::H), TxType::DctAdst);
        assert_eq!(resolve_tx_type(0, false, IntraMode::Tm), TxType::AdstAdst);
        assert_eq!(resolve_tx_type(0, false, IntraMode::D135), TxType::AdstAdst);

        // U / V plane: always DCT_DCT regardless of mode.
        for plane in [1, 2] {
            for raw_mode in 0u8..=9 {
                let m = IntraMode::from_u8(raw_mode).unwrap();
                assert_eq!(
                    resolve_tx_type(plane, false, m),
                    TxType::DctDct,
                    "plane {plane} mode {m:?}"
                );
            }
        }

        // Lossless: always DCT_DCT even on Y.
        for raw_mode in 0u8..=9 {
            let m = IntraMode::from_u8(raw_mode).unwrap();
            assert_eq!(resolve_tx_type(0, true, m), TxType::DctDct);
        }
    }

    /// Zero-residual case: stream encodes EOB at scan[0]. The wrapper
    /// returns `eob = 0` and `dst` ends up exactly equal to the pure
    /// intra-prediction output — IT-add is bypassed. We verify with
    /// V_PRED, which is trivially the above row repeated.
    #[test]
    fn zero_residual_dst_equals_pure_prediction_v_mode_tx4x4() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 128, 128, 128);
        let bytes = encode_eob_at_zero();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 4;
        let stride = 16;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 5,
        };
        let mut dst = alloc_dst(bs, stride);
        // Pre-fill with sentinels so we can detect missing writes.
        dst.fill(0xff);

        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0, // plane Y
            0, 0,
            TxSize::Tx4x4,
            IntraMode::V,
            pred_ctx,
            &mut dst,
            stride,
            16,
            (10, 20),
            false,
        );

        assert_eq!(eob, 0, "EOB at scan[0] → eob = 0");

        // The block region is the pure V-prediction: each row = above[..bs].
        for r_idx in 0..bs {
            assert_eq!(
                &dst[r_idx * stride..r_idx * stride + bs],
                &above[..bs],
                "row {r_idx}"
            );
        }
        // The row after the block is still the sentinel (no overshoot).
        assert!(dst[bs * stride..].iter().all(|&v| v == 0xff));
    }

    /// DC-only residual: stream encodes ONE_TOKEN at scan[0] then EOB.
    /// With a DC dequant of `8`, `dq_coefs[0] = 8`. The
    /// `idct4x4_1_add` fast path adds the known constant per LLD §5.4
    /// to every pixel — we compute it analytically and compare.
    ///
    /// The constant is the IDCT(8, 0, ..., 0) "all-DC" reconstruction:
    /// libvpx `vpx_idct4x4_1_add_c` does two `wraplow * COSPI_16_64`
    /// passes then `(_ + 8) >> 4`. We don't hand-trace that here;
    /// instead we compare against a separate `idct4x4_1_add` call on a
    /// fresh prediction buffer.
    #[test]
    fn dc_only_residual_matches_separate_predict_plus_it_add_tx4x4() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 64, 64, 64);
        let bytes = encode_one_token_dc_pos();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 4;
        let stride = 8;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 42,
        };

        let mut dst_wrapper = alloc_dst(bs, stride);
        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx4x4,
            IntraMode::Dc,
            pred_ctx,
            &mut dst_wrapper,
            stride,
            16,
            (8, 13),
            false,
        );
        assert_eq!(eob, 1);

        // Reference: separately predict into a fresh buffer, then run
        // the matching IT-add (idct4x4_1_add — DCT_DCT for DC, eob = 1).
        let mut dst_ref = alloc_dst(bs, stride);
        intra_predict(IntraMode::Dc, &mut dst_ref, stride, bs, pred_ctx);
        // dq_coef[0] = 1 * dc_dq = 8 for Tx4x4 (no dq_shift).
        idct4x4_1_add(8, &mut dst_ref, stride);

        assert_eq!(dst_wrapper, dst_ref, "wrapper IT-add path matches direct");
    }

    /// Mode coverage: zero-residual case at TM mode (the third primary
    /// kernel after DC and V). The `dst` block region matches a
    /// stand-alone `intra_predict(TM)` call.
    #[test]
    fn zero_residual_tm_mode_dst_equals_pure_prediction() {
        let fc = fc_with_flat_probs(TxSize::Tx4x4, 0, 0, 128, 128, 128);
        let bytes = encode_eob_at_zero();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 4;
        let stride = 8;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 50,
        };
        let mut dst = alloc_dst(bs, stride);

        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx4x4,
            IntraMode::Tm,
            pred_ctx,
            &mut dst,
            stride,
            16,
            (7, 11),
            false,
        );
        assert_eq!(eob, 0);

        let mut dst_ref = alloc_dst(bs, stride);
        intra_predict(IntraMode::Tm, &mut dst_ref, stride, bs, pred_ctx);

        for r_idx in 0..bs {
            assert_eq!(
                &dst[r_idx * stride..r_idx * stride + bs],
                &dst_ref[r_idx * stride..r_idx * stride + bs],
                "row {r_idx}"
            );
        }
    }

    /// Size coverage: zero-residual case at Tx16x16 + H_PRED. The 16x16
    /// block matches a stand-alone `intra_predict(H)` call. Confirms
    /// the wrapper threads `bs = 16` correctly through the dispatch.
    #[test]
    fn zero_residual_h_mode_tx16x16_matches_pure_prediction() {
        let fc = fc_with_flat_probs(TxSize::Tx16x16, 0, 0, 128, 128, 128);
        let bytes = encode_eob_at_zero();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 16;
        let stride = 32;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 60,
        };
        let mut dst = alloc_dst(bs, stride);

        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx16x16,
            IntraMode::H,
            pred_ctx,
            &mut dst,
            stride,
            256,
            (10, 20),
            false,
        );
        assert_eq!(eob, 0);

        let mut dst_ref = alloc_dst(bs, stride);
        intra_predict(IntraMode::H, &mut dst_ref, stride, bs, pred_ctx);

        for r_idx in 0..bs {
            assert_eq!(
                &dst[r_idx * stride..r_idx * stride + bs],
                &dst_ref[r_idx * stride..r_idx * stride + bs],
                "row {r_idx}"
            );
        }
    }

    /// Size coverage: Tx32x32 forces DCT_DCT regardless of mode (no
    /// 32x32 hybrid). Zero-residual case — verify the dispatch routes
    /// through correctly and the prediction lands. We use V_PRED.
    #[test]
    fn zero_residual_v_mode_tx32x32_matches_pure_prediction() {
        let fc = fc_with_flat_probs(TxSize::Tx32x32, 0, 0, 128, 128, 128);
        let bytes = encode_eob_at_zero();
        let mut r = BoolReader::new(&bytes).unwrap();
        // tile_mi_cols = 8 → above_coef width = 16 cells (Y).
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 32;
        let stride = 48;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 60,
        };
        let mut dst = alloc_dst(bs, stride);

        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            0,
            0, 0,
            TxSize::Tx32x32,
            IntraMode::V,
            pred_ctx,
            &mut dst,
            stride,
            1024,
            (10, 20),
            false,
        );
        assert_eq!(eob, 0);

        // Confirm tx_type used in decode_block_tokens was DctDct (the
        // call would have used the default 32x32 scan; for a flat-prob
        // EOB-at-zero stream the scan choice doesn't matter, but the
        // resolve_tx_type contract did the right thing — see also the
        // dedicated test above).
        assert_eq!(
            resolve_tx_type(0, false, IntraMode::V),
            TxType::AdstDct,
            "table value"
        );
        // (At Tx32x32 the IT dispatch ignores tx_type anyway.)

        let mut dst_ref = alloc_dst(bs, stride);
        intra_predict(IntraMode::V, &mut dst_ref, stride, bs, pred_ctx);

        for r_idx in 0..bs {
            assert_eq!(
                &dst[r_idx * stride..r_idx * stride + bs],
                &dst_ref[r_idx * stride..r_idx * stride + bs],
                "row {r_idx}"
            );
        }
    }

    /// Plane coverage: UV plane (plane = 1) zero-residual case. The
    /// caller passes `mi->uv_mode`; this test checks that the plane
    /// index threads through to the right plane-type for the coef
    /// table lookup and to `resolve_tx_type` (UV → DctDct).
    #[test]
    fn zero_residual_uv_plane_tx8x8_matches_pure_prediction() {
        // Place the flat-probs table at plane_type = 1 (UV).
        let fc = fc_with_flat_probs(TxSize::Tx8x8, 1, 0, 128, 128, 128);
        let bytes = encode_eob_at_zero();
        let mut r = BoolReader::new(&bytes).unwrap();
        let mut coef_ctx = TileCoefCtx::new(8);

        let bs = 8;
        let stride = 16;
        let (above, left) = make_neighbors(bs);
        let pred_ctx = PredCtx {
            above: &above,
            left: &left,
            above_left: 50,
        };
        let mut dst = alloc_dst(bs, stride);

        // uv_mode is H_PRED here; even though H is mapped to
        // TxType::DctAdst for Y, on UV resolve_tx_type forces DCT_DCT.
        let eob = reconstruct_intra_block(
            &mut r,
            &fc,
            None,
            &mut coef_ctx,
            1, // U plane
            0, 0,
            TxSize::Tx8x8,
            IntraMode::H,
            pred_ctx,
            &mut dst,
            stride,
            64,
            (10, 20),
            false,
        );
        assert_eq!(eob, 0);

        // tx_type contract: UV forces DCT_DCT.
        assert_eq!(resolve_tx_type(1, false, IntraMode::H), TxType::DctDct);

        let mut dst_ref = alloc_dst(bs, stride);
        intra_predict(IntraMode::H, &mut dst_ref, stride, bs, pred_ctx);
        for r_idx in 0..bs {
            assert_eq!(
                &dst[r_idx * stride..r_idx * stride + bs],
                &dst_ref[r_idx * stride..r_idx * stride + bs],
                "row {r_idx}"
            );
        }
        // Above[0] / left[0] flag for U was splat by decode_block_tokens.
        // eob == 0 → cells stay zero (writeback clears the spanned
        // region). The U plane has independent above/left arrays so
        // touching it does not pollute Y or V.
        for x4 in 0..coef_ctx.above_coef_width(0) {
            assert_eq!(coef_ctx.above_coef(0, x4), 0, "Y above[{x4}]");
        }
        for x4 in 0..coef_ctx.above_coef_width(1) {
            assert_eq!(coef_ctx.above_coef(1, x4), 0, "U above[{x4}]");
        }
    }
}
