//! SSE2 pixel subtraction for VP9 transform residuals.
//!
//! This mirrors the low-bit-depth portion of libvpx's
//! `vpx_subtract_block_sse2` for the 8-, 16-, 32-, and 64-pixel-wide
//! transform blocks used by mode decision. The byte subtraction is widened to
//! i16 before subtracting, so every result in `[-255, 255]` is represented
//! exactly. The nonzero variant folds the widened differences while storing
//! them, avoiding a second pass over the residual.

use core::arch::x86_64::{
    _mm_cmpeq_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
    _mm_setzero_si128, _mm_storeu_si128, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
};

fn checked_block_span(height: usize, stride: usize, width: usize) -> usize {
    height
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(stride))
        .and_then(|offset| offset.checked_add(width))
        .expect("subtraction block layout overflow")
}

/// SSE2 implementation of packed byte subtraction.
///
/// An 8-byte row uses one 8-byte load; larger supported rows are processed as
/// 16-byte chunks. The output rows are packed at the block width, even when
/// source and prediction strides include padding. All loads and stores are
/// unaligned because frame and temporary-plane strides are not required to be
/// aligned.
///
/// # Safety
///
/// The caller must guarantee SSE2 support. `src` and `pred` must each cover
/// `(height - 1) * stride + width` bytes, and `dst` must cover
/// `height * width` i16 values. `width` must be one of 8, 16, 32, or 64.
// G4: loads from `src` / `pred` / `dst` use _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
#[target_feature(enable = "sse2")]
unsafe fn subtract_block_sse2_inner<const TRACK_NONZERO: bool>(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    dst: *mut i16,
    width: usize,
    height: usize,
) -> bool {
    let zero = _mm_setzero_si128();
    let mut nonzero = zero;

    for row in 0..height {
        let src_row = src.add(row * src_stride);
        let pred_row = pred.add(row * pred_stride);
        let dst_row = dst.add(row * width);

        if width == 8 {
            let src_values = _mm_loadl_epi64(src_row.cast());
            let pred_values = _mm_loadl_epi64(pred_row.cast());
            let diff = _mm_sub_epi16(
                _mm_unpacklo_epi8(src_values, zero),
                _mm_unpacklo_epi8(pred_values, zero),
            );
            if TRACK_NONZERO {
                nonzero = _mm_or_si128(nonzero, diff);
            }
            _mm_storeu_si128(dst_row.cast(), diff);
        } else {
            for col in (0..width).step_by(16) {
                let src_values = _mm_loadu_si128(src_row.add(col).cast());
                let pred_values = _mm_loadu_si128(pred_row.add(col).cast());
                let src_lo = _mm_unpacklo_epi8(src_values, zero);
                let src_hi = _mm_unpackhi_epi8(src_values, zero);
                let pred_lo = _mm_unpacklo_epi8(pred_values, zero);
                let pred_hi = _mm_unpackhi_epi8(pred_values, zero);
                let diff_lo = _mm_sub_epi16(src_lo, pred_lo);
                let diff_hi = _mm_sub_epi16(src_hi, pred_hi);
                if TRACK_NONZERO {
                    nonzero = _mm_or_si128(nonzero, diff_lo);
                    nonzero = _mm_or_si128(nonzero, diff_hi);
                }
                _mm_storeu_si128(dst_row.add(col).cast(), diff_lo);
                _mm_storeu_si128(dst_row.add(col + 8).cast(), diff_hi);
            }
        }
    }

    !TRACK_NONZERO || _mm_movemask_epi8(_mm_cmpeq_epi16(nonzero, zero)) != 0xffff
}

/// Subtracts a packed prediction block with an 8/16/32/64-pixel row width.
#[cfg(target_arch = "x86_64")]
pub fn subtract_block_sse2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [i16],
    width: usize,
    height: usize,
) {
    assert!(matches!(width, 8 | 16 | 32 | 64));
    assert!(height > 0);
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= checked_block_span(height, src_stride, width));
    assert!(pred.len() >= checked_block_span(height, pred_stride, width));
    let dst_len = width
        .checked_mul(height)
        .expect("subtraction destination layout overflow");
    assert!(dst.len() >= dst_len);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove that every
    // unaligned input load and residual store stays within its slice.
    unsafe {
        let _ = subtract_block_sse2_inner::<false>(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            dst.as_mut_ptr(),
            width,
            height,
        );
    }
}

/// Subtracts a square 8/16/32/64 prediction block from source pixels.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
pub fn subtract_square_sse2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    dst: &mut [i16],
    side: usize,
) {
    assert!(matches!(side, 8 | 16 | 32 | 64));
    assert!(src_stride >= side);
    assert!(pred_stride >= side);
    assert!(src.len() >= checked_block_span(side, src_stride, side));
    assert!(pred.len() >= checked_block_span(side, pred_stride, side));
    assert!(dst.len() >= side * side);

    // SAFETY: the square dimensions and checked slice assertions above prove
    // every input load and residual store stays within its slice.
    unsafe {
        let _ = subtract_block_sse2_inner::<false>(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            dst.as_mut_ptr(),
            side,
            side,
        );
    }
}

/// Subtracts a square prediction block and reports whether any residual is
/// non-zero without reading the completed residual buffer again.
#[cfg(target_arch = "x86_64")]
pub fn subtract_square_sse2_with_nonzero(
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
    assert!(src.len() >= checked_block_span(side, src_stride, side));
    assert!(pred.len() >= checked_block_span(side, pred_stride, side));
    assert!(dst.len() >= side * side);

    // SAFETY: the square dimensions and checked slice assertions above prove
    // every input load and residual store; OR reduction observes each result.
    unsafe {
        subtract_block_sse2_inner::<true>(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            dst.as_mut_ptr(),
            side,
            side,
        )
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::*;

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    fn subtract_square_scalar(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        dst: &mut [i16],
        side: usize,
    ) {
        for row in 0..side {
            for col in 0..side {
                dst[row * side + col] = i16::from(src[row * src_stride + col])
                    - i16::from(pred[row * pred_stride + col]);
            }
        }
    }

    fn subtract_block_scalar(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        dst: &mut [i16],
        width: usize,
        height: usize,
    ) {
        for row in 0..height {
            for col in 0..width {
                dst[row * width + col] = i16::from(src[row * src_stride + col])
                    - i16::from(pred[row * pred_stride + col]);
            }
        }
    }

    #[test]
    fn differential_random_all_sizes_and_strides() {
        assert!(is_x86_feature_detected!("sse2"));
        let mut rng = Lcg32(0x5AB7_12AC);
        for side in [8, 16, 32, 64] {
            for jitter in 0..17 {
                let src_stride = side + jitter;
                let pred_stride = side + (jitter * 5 + 2) % 17;
                let src: Vec<u8> = (0..side * src_stride + 3).map(|_| rng.byte()).collect();
                let pred: Vec<u8> = (0..side * pred_stride + 5).map(|_| rng.byte()).collect();
                let mut expected = vec![i16::MIN; side * side + 7];
                let mut actual = expected.clone();
                subtract_square_scalar(
                    &src[3..],
                    src_stride,
                    &pred[5..],
                    pred_stride,
                    &mut expected[7..],
                    side,
                );
                subtract_square_sse2(
                    &src[3..],
                    src_stride,
                    &pred[5..],
                    pred_stride,
                    &mut actual[7..],
                    side,
                );
                assert_eq!(
                    actual, expected,
                    "SSE2 mismatch at {side}x{side}, stride jitter {jitter}"
                );

                actual.fill(i16::MIN);
                let reported_nonzero = subtract_square_sse2_with_nonzero(
                    &src[3..],
                    src_stride,
                    &pred[5..],
                    pred_stride,
                    &mut actual[7..],
                    side,
                );
                assert_eq!(actual, expected);
                assert_eq!(
                    reported_nonzero,
                    expected[7..7 + side * side].iter().any(|&v| v != 0),
                    "nonzero mismatch at {side}x{side}, stride jitter {jitter}"
                );
            }
        }
    }

    #[test]
    fn differential_rectangular_blocks_and_strides() {
        assert!(is_x86_feature_detected!("sse2"));
        let mut rng = Lcg32(0xC0DE_128A);
        for &(width, height) in &[
            (8, 16),
            (8, 32),
            (8, 64),
            (16, 8),
            (16, 32),
            (32, 8),
            (32, 16),
            (32, 64),
            (64, 8),
            (64, 32),
        ] {
            let src_stride = width + 5;
            let pred_stride = width + 9;
            let src: Vec<u8> = (0..(height - 1) * src_stride + width + 7)
                .map(|_| rng.byte())
                .collect();
            let pred: Vec<u8> = (0..(height - 1) * pred_stride + width + 11)
                .map(|_| rng.byte())
                .collect();
            let mut expected = vec![i16::MIN; width * height + 13];
            let mut actual = expected.clone();
            subtract_block_scalar(
                &src[7..],
                src_stride,
                &pred[11..],
                pred_stride,
                &mut expected[7..],
                width,
                height,
            );
            subtract_block_sse2(
                &src[7..],
                src_stride,
                &pred[11..],
                pred_stride,
                &mut actual[7..],
                width,
                height,
            );
            assert_eq!(actual, expected, "SSE2 mismatch at {width}x{height}");
        }
    }

    #[test]
    fn reject_overflowing_stride_layouts() {
        let side = 8;
        let overflowing_stride = usize::MAX / (side - 1);
        let pixels = [0u8; 64];
        let mut residual = [0i16; 64];

        fn assert_layout_overflow(operation: impl FnOnce()) {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation))
                .expect_err("overflowing layout must panic before entering SSE2");
            let message = if let Some(message) = panic.downcast_ref::<&str>() {
                *message
            } else if let Some(message) = panic.downcast_ref::<String>() {
                message.as_str()
            } else {
                "<non-string panic>"
            };
            assert!(
                message.contains("layout overflow"),
                "unexpected panic payload: {message}"
            );
        }

        for &src_stride in &[overflowing_stride, side] {
            for &pred_stride in &[overflowing_stride, side] {
                if src_stride == side && pred_stride == side {
                    continue;
                }
                assert_layout_overflow(|| {
                    subtract_square_sse2(
                        &pixels,
                        src_stride,
                        &pixels,
                        pred_stride,
                        &mut residual,
                        side,
                    );
                });
                assert_layout_overflow(|| {
                    subtract_square_sse2_with_nonzero(
                        &pixels,
                        src_stride,
                        &pixels,
                        pred_stride,
                        &mut residual,
                        side,
                    );
                });
                assert_layout_overflow(|| {
                    subtract_block_sse2(
                        &pixels,
                        src_stride,
                        &pixels,
                        pred_stride,
                        &mut residual,
                        side,
                        side,
                    );
                });
            }
        }
    }

    #[test]
    fn differential_extreme_values_and_zero_detection() {
        assert!(is_x86_feature_detected!("sse2"));
        for side in [8, 16, 32, 64] {
            let src = vec![0; side * side];
            let pred = vec![u8::MAX; side * side];
            let mut residual = vec![0; side * side];
            subtract_square_sse2(&src, side, &pred, side, &mut residual, side);
            assert!(residual.iter().all(|&value| value == -255));
            assert!(subtract_square_sse2_with_nonzero(
                &src,
                side,
                &pred,
                side,
                &mut residual,
                side
            ));

            subtract_square_sse2(&pred, side, &src, side, &mut residual, side);
            assert!(residual.iter().all(|&value| value == 255));
            assert!(subtract_square_sse2_with_nonzero(
                &pred,
                side,
                &src,
                side,
                &mut residual,
                side
            ));

            let plus_one_src = vec![128; side * side];
            let plus_one_pred = vec![127; side * side];
            assert!(subtract_square_sse2_with_nonzero(
                &plus_one_src,
                side,
                &plus_one_pred,
                side,
                &mut residual,
                side
            ));
            assert!(residual.iter().all(|&value| value == 1));

            let minus_one_src = vec![127; side * side];
            let minus_one_pred = vec![128; side * side];
            assert!(subtract_square_sse2_with_nonzero(
                &minus_one_src,
                side,
                &minus_one_pred,
                side,
                &mut residual,
                side
            ));
            assert!(residual.iter().all(|&value| value == -1));

            let equal = vec![127; side * side];
            assert!(!subtract_square_sse2_with_nonzero(
                &equal,
                side,
                &equal,
                side,
                &mut residual,
                side
            ));
            assert!(residual.iter().all(|&value| value == 0));
        }
    }
}
