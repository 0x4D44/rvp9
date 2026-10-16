//! Narrow cross-crate access to architecture-specific DSP kernels.
//!
//! The architecture modules themselves are crate-private because most of
//! their entry points rely on dispatch-established CPU and buffer invariants.
//! Only kernels needed by sibling workspace crates are exposed here.
//!
//! Architecture modules cannot be reached through the safe public API:
//!
//! ```compile_fail
//! use rvp9_core::dsp::aarch64;
//! ```

#[cfg(target_arch = "aarch64")]
pub use super::aarch64::{
    subtract_square_neon, subtract_square_neon_with_nonzero, vp9_quantize_fp_all_zero_neon,
};

#[cfg(target_arch = "x86_64")]
use crate::inter::filter::SUBPEL_SHIFTS;

/// Subtract a square prediction block, honoring the process scalar override.
#[cfg(target_arch = "x86_64")]
pub fn subtract_square_sse2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [i16],
    side: usize,
) {
    if super::get_ref().caps.sse2 {
        super::x86_64::subtract_square_sse2(src, src_stride, pred, pred_stride, dst, side);
    } else {
        let _ = subtract_square_scalar(src, src_stride, pred, pred_stride, dst, side);
    }
}

/// Subtract a square block and report nonzero residuals, honoring the scalar override.
#[cfg(target_arch = "x86_64")]
pub fn subtract_square_sse2_with_nonzero(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [i16],
    side: usize,
) -> bool {
    if super::get_ref().caps.sse2 {
        super::x86_64::subtract_square_sse2_with_nonzero(
            src,
            src_stride,
            pred,
            pred_stride,
            dst,
            side,
        )
    } else {
        subtract_square_scalar(src, src_stride, pred, pred_stride, dst, side)
    }
}

#[cfg(target_arch = "x86_64")]
fn subtract_square_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [i16],
    side: usize,
) -> bool {
    assert!(matches!(side, 8 | 16 | 32 | 64));
    assert!(src_stride >= side);
    assert!(pred_stride >= side);
    let span = |stride: usize| {
        (side - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(side))
            .expect("square input span overflow")
    };
    assert!(src.len() >= span(src_stride));
    assert!(pred.len() >= span(pred_stride));
    assert!(dst.len() >= side * side);
    let mut nonzero = false;
    for row in 0..side {
        for col in 0..side {
            let value =
                i16::from(src[row * src_stride + col]) - i16::from(pred[row * pred_stride + col]);
            dst[row * side + col] = value;
            nonzero |= value != 0;
        }
    }
    nonzero
}

/// Run the AVX2 vertical eight-tap convolution.
///
/// # Safety
///
/// The caller must establish AVX2 support. `src` must cover the complete
/// eight-tap footprint, and `dst` must cover `(h - 1) * dst_stride + w` bytes.
#[cfg(target_arch = "x86_64")]
pub unsafe fn convolve8_vert_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    super::x86_64::convolve8_vert_avx2(src, src_stride, dst, dst_stride, filter_y, w, h);
}

/// Run the AVX2 separable two-dimensional eight-tap convolution.
///
/// # Safety
///
/// The caller must establish AVX2 support. `src` must cover
/// `(h + 6) * src_stride + w + 7` bytes, `dst` must cover
/// `(h - 1) * dst_stride + w` bytes, and `w` and `h` must be in `1..=64`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolve8_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    super::x86_64::convolve8_avx2(src, src_stride, dst, dst_stride, filter_x, filter_y, w, h);
}

/// Run the AVX2 scaled two-dimensional eight-tap convolution.
///
/// # Safety
///
/// The caller must establish AVX2 support and the scalar scaled-convolution
/// footprint for all source and destination slices. Steps must be positive,
/// and `w` and `h` must be in `1..=64`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn convolve8_scaled_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    super::x86_64::convolve8_scaled_avx2(
        src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
    );
}

/// Run the AVX2 D117 intra predictor.
///
/// # Safety
///
/// The caller must establish AVX2 support. `bs` must be 4, 8, 16, or 32;
/// `above` and `left` must contain at least `bs` samples; and `dst` must cover
/// `(bs - 1) * stride + bs` bytes.
#[cfg(target_arch = "x86_64")]
pub unsafe fn d117_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    super::x86_64::d117_predictor_avx2(dst, stride, bs, above, left, above_left);
}

/// Run the AVX2 D153 intra predictor.
///
/// # Safety
///
/// The caller must establish AVX2 support. `bs` must be 4, 8, 16, or 32;
/// `above` and `left` must contain at least `bs` samples; and `dst` must cover
/// `(bs - 1) * stride + bs` bytes.
#[cfg(target_arch = "x86_64")]
pub unsafe fn d153_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    super::x86_64::d153_predictor_avx2(dst, stride, bs, above, left, above_left);
}
