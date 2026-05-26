//! W4.2 — Distortion primitives for RDO.
//!
//! Two functions, both faithful ports from libvpx:
//!
//! - [`vp9_block_error`] — coefficient-domain squared error, with an
//!   `ssz` (sum of squared `coeff`) out-parameter. Used by the
//!   transform-domain `dist_block` path
//!   (`vp9/encoder/vp9_rdopt.c:561-`).
//! - [`pixel_sse`] — pixel-domain sum of squared error for a tx-block
//!   sized region. Used by the non-tx-domain `dist_block` path.
//!
//! ## libvpx references
//!
//! - `vp9_block_error_c` — `vp9/encoder/vp9_rdopt.c:319-332`,
//!   v1.14.1 commit `12f3a2ac6`.
//! - `pixel_sse` — `vp9/encoder/vp9_rdopt.c:521-559`.
//!
//! ## Scope
//!
//! - Scalar reference only. SIMD parity is Wave 6+.
//! - 8-bit only. HBD variants (`vp9_highbd_block_error*`) are out of
//!   scope.
//! - The libvpx `pixel_sse` dispatches through `cpi->fn_ptr[bsize].vf`
//!   to a SIMD `variance` kernel and walks "visible 4×4s" only when
//!   the block straddles a frame edge. Our scalar reference always
//!   walks every pixel in the supplied region; the visible-only logic
//!   needs frame-edge plumbing the W4.1-W4.4 batch deliberately does
//!   not expose. Callers must hand us only the visible region (the
//!   inter picker today already clamps src/dst to the visible window
//!   before computing SSE).

/// Coefficient-domain squared quantization error + sum of squared
/// `coeff`. Mirrors libvpx `vp9_block_error_c`
/// (`vp9/encoder/vp9_rdopt.c:319-332`):
///
/// ```c
/// int64_t vp9_block_error_c(const tran_low_t *coeff,
///                           const tran_low_t *dqcoeff,
///                           intptr_t block_size, int64_t *ssz) {
///     int64_t error = 0, sqcoeff = 0;
///     for (i = 0; i < block_size; i++) {
///         const int diff = coeff[i] - dqcoeff[i];
///         error += diff * diff;
///         sqcoeff += coeff[i] * coeff[i];
///     }
///     *ssz = sqcoeff;
///     return error;
/// }
/// ```
///
/// rvp9 carries the un-quantized `coeff` slot as `i32` (the forward
/// DCT writes `i32`); `dqcoeff` is `i16` (the dequantizer output).
/// The math fits comfortably in `i64`.
#[must_use]
pub fn vp9_block_error(coeff: &[i32], dqcoeff: &[i16]) -> (i64, i64) {
    debug_assert_eq!(coeff.len(), dqcoeff.len());
    let mut error: i64 = 0;
    let mut sqcoeff: i64 = 0;
    for (&c, &dq) in coeff.iter().zip(dqcoeff.iter()) {
        let diff = c - i32::from(dq);
        error += i64::from(diff) * i64::from(diff);
        sqcoeff += i64::from(c) * i64::from(c);
    }
    (error, sqcoeff)
}

/// Pixel-domain sum-of-squared-error over a `side × side` region.
///
/// Mirrors the inner kernel of libvpx `pixel_sse`
/// (`vp9/encoder/vp9_rdopt.c:521-559`) — the SIMD `vf` dispatch +
/// the visible-only 4×4 walk both collapse to "sum of `(src - dst)^2`
/// over the supplied area" once the visibility plumbing is removed.
///
/// `src` and `dst` are interpreted as `side` rows of `side` consecutive
/// `u8` samples, walking with `src_stride` / `dst_stride` between rows.
/// Both slices must be at least `(side - 1) * stride + side` long
/// (asserted in debug builds).
#[must_use]
pub fn pixel_sse(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    side: usize,
) -> u64 {
    debug_assert!(src.len() >= (side - 1) * src_stride + side);
    debug_assert!(dst.len() >= (side - 1) * dst_stride + side);

    let mut sse: u64 = 0;
    for r in 0..side {
        let src_row = &src[r * src_stride..r * src_stride + side];
        let dst_row = &dst[r * dst_stride..r * dst_stride + side];
        for (&s, &d) in src_row.iter().zip(dst_row.iter()) {
            let diff = i32::from(s) - i32::from(d);
            sse += (diff * diff) as u64;
        }
    }
    sse
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PROOF W4.2 — `vp9_block_error` returns `(0, ssz)` when `coeff
    /// == dqcoeff` for every position. libvpx
    /// `vp9/encoder/vp9_rdopt.c:319-332`: when `coeff[i] == dqcoeff[i]`
    /// the `diff * diff` term is identically zero so `error == 0`,
    /// while `sqcoeff` still accumulates the squared `coeff` values.
    #[test]
    fn w4_2_block_error_zero_on_identical() {
        // Use a non-trivial coefficient pattern so the `ssz` half is
        // also exercised (i.e. so we aren't just trivially passing on
        // both halves being zero).
        let coeff: Vec<i32> = (1..=64).collect();
        let dqcoeff: Vec<i16> = coeff.iter().map(|&v| v as i16).collect();
        let (error, ssz) = vp9_block_error(&coeff, &dqcoeff);
        assert_eq!(error, 0, "coeff == dqcoeff → error must be 0");
        // ssz = sum(i^2) for i in 1..=64 = 64*65*129/6 = 89440.
        assert_eq!(ssz, 89440, "ssz must accumulate sum of squared coeff");
    }

    /// PROOF W4.2 — `pixel_sse` returns 0 when `src == dst`. libvpx
    /// `pixel_sse` (`vp9_rdopt.c:521-559`) reduces to the SIMD
    /// `variance` kernel for a non-edge block; with identical inputs
    /// the per-pixel diff is zero everywhere so the SSE is zero.
    #[test]
    fn w4_2_pixel_sse_zero_on_identical() {
        // 8×8 region with arbitrary content, src == dst.
        let pixels: Vec<u8> = (0..64u8).collect();
        let sse = pixel_sse(&pixels, 8, &pixels, 8, 8);
        assert_eq!(sse, 0, "src == dst → SSE must be 0");
    }

    /// `vp9_block_error` correctly accumulates per-position squared
    /// differences. With one position differing by 3, error = 9.
    #[test]
    fn block_error_accumulates_squared_diff() {
        let mut coeff = vec![0i32; 16];
        coeff[5] = 10;
        let mut dqcoeff = vec![0i16; 16];
        dqcoeff[5] = 7; // diff = 3
        let (error, ssz) = vp9_block_error(&coeff, &dqcoeff);
        assert_eq!(error, 9, "one diff of 3 → error 9");
        assert_eq!(ssz, 100, "coeff[5]^2 = 100");
    }

    /// `pixel_sse` correctly accumulates over a strided region —
    /// 4×4 block, all `src` = 100, all `dst` = 90, diff = 10 each,
    /// SSE = 16 * 100 = 1600.
    #[test]
    fn pixel_sse_strided_accumulates() {
        let src = vec![100u8; 64];
        let dst = vec![90u8; 64];
        // Walk a 4×4 region with stride 8 (typical of a sub-block).
        let sse = pixel_sse(&src, 8, &dst, 8, 4);
        assert_eq!(sse, 16 * 100);
    }

    /// `pixel_sse` honours the stride argument — content past the
    /// edge of the `side × side` window is ignored.
    #[test]
    fn pixel_sse_ignores_padding_past_window() {
        // 4×4 window inside an 8-wide stride. Make the "outside the
        // window" pixels (columns 4..8) wildly different — we must NOT
        // see them in the result.
        let mut src = vec![100u8; 64];
        let mut dst = vec![100u8; 64];
        for r in 0..8 {
            for c in 4..8 {
                src[r * 8 + c] = 0;
                dst[r * 8 + c] = 255;
            }
        }
        let sse = pixel_sse(&src, 8, &dst, 8, 4);
        assert_eq!(sse, 0, "padding column should not enter the SSE");
    }
}
