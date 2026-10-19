//! NEON pixel subtraction for VP9 transform residuals.
//!
//! This mirrors libvpx's `vpx_subtract_block_neon` for the square 8/16/32/64
//! transforms used by rvp9's block-rate/distortion path.

use crate::dsp::bounds::strided_len;

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn subtract_square_neon_inner<const TRACK_NONZERO: bool>(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    dst: *mut i16,
    side: usize,
) -> bool {
    let mut nonzero = vdupq_n_u16(0);
    for row in 0..side {
        let src_row = src.add(row * src_stride);
        let pred_row = pred.add(row * pred_stride);
        let dst_row = dst.add(row * side);

        if side >= 32 {
            for col in (0..side).step_by(32) {
                let src_0 = vld1q_u8(src_row.add(col));
                let src_1 = vld1q_u8(src_row.add(col + 16));
                let pred_0 = vld1q_u8(pred_row.add(col));
                let pred_1 = vld1q_u8(pred_row.add(col + 16));
                let diff_0 = vsubl_u8(vget_low_u8(src_0), vget_low_u8(pred_0));
                let diff_1 = vsubl_u8(vget_high_u8(src_0), vget_high_u8(pred_0));
                let diff_2 = vsubl_u8(vget_low_u8(src_1), vget_low_u8(pred_1));
                let diff_3 = vsubl_u8(vget_high_u8(src_1), vget_high_u8(pred_1));
                if TRACK_NONZERO {
                    nonzero = vorrq_u16(
                        nonzero,
                        vorrq_u16(vorrq_u16(diff_0, diff_1), vorrq_u16(diff_2, diff_3)),
                    );
                }
                vst1q_s16(dst_row.add(col), vreinterpretq_s16_u16(diff_0));
                vst1q_s16(dst_row.add(col + 8), vreinterpretq_s16_u16(diff_1));
                vst1q_s16(dst_row.add(col + 16), vreinterpretq_s16_u16(diff_2));
                vst1q_s16(dst_row.add(col + 24), vreinterpretq_s16_u16(diff_3));
            }
        } else if side == 16 {
            let src_values = vld1q_u8(src_row);
            let pred_values = vld1q_u8(pred_row);
            let diff_0 = vsubl_u8(vget_low_u8(src_values), vget_low_u8(pred_values));
            let diff_1 = vsubl_u8(vget_high_u8(src_values), vget_high_u8(pred_values));
            if TRACK_NONZERO {
                nonzero = vorrq_u16(nonzero, vorrq_u16(diff_0, diff_1));
            }
            vst1q_s16(dst_row, vreinterpretq_s16_u16(diff_0));
            vst1q_s16(dst_row.add(8), vreinterpretq_s16_u16(diff_1));
        } else {
            let diff = vsubl_u8(vld1_u8(src_row), vld1_u8(pred_row));
            if TRACK_NONZERO {
                nonzero = vorrq_u16(nonzero, diff);
            }
            vst1q_s16(dst_row, vreinterpretq_s16_u16(diff));
        }
    }
    !TRACK_NONZERO || vmaxvq_u16(nonzero) != 0
}

/// Subtracts a square 8/16/32/64 prediction block from source pixels.
pub fn subtract_square_neon(
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
    assert!(src.len() >= strided_len(side - 1, src_stride, side));
    assert!(pred.len() >= strided_len(side - 1, pred_stride, side));
    assert!(dst.len() >= side * side);

    // SAFETY: AArch64 always provides NEON. The assertions prove that every
    // source load, prediction load, and residual store stays inside its slice.
    unsafe {
        let _ = subtract_square_neon_inner::<false>(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            dst.as_mut_ptr(),
            side,
        );
    }
}

/// Subtracts a square prediction block and reports whether any residual is
/// non-zero without reading the completed residual buffer again.
pub fn subtract_square_neon_with_nonzero(
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
    assert!(src.len() >= strided_len(side - 1, src_stride, side));
    assert!(pred.len() >= strided_len(side - 1, pred_stride, side));
    assert!(dst.len() >= side * side);

    // SAFETY: The assertions prove every input load and residual store. The
    // OR reduction observes the exact widened differences before they are
    // stored and cannot overflow.
    unsafe {
        subtract_square_neon_inner::<true>(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            dst.as_mut_ptr(),
            side,
        )
    }
}

#[cfg(all(test, not(miri)))]
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

    #[test]
    fn differential_random_all_sizes_and_strides() {
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
                subtract_square_neon(
                    &src[3..],
                    src_stride,
                    &pred[5..],
                    pred_stride,
                    &mut actual[7..],
                    side,
                );
                let reported_nonzero = subtract_square_neon_with_nonzero(
                    &src[3..],
                    src_stride,
                    &pred[5..],
                    pred_stride,
                    &mut actual[7..],
                    side,
                );
                assert_eq!(
                    actual, expected,
                    "mismatch at {side}x{side}, stride jitter {jitter}"
                );
                assert_eq!(
                    reported_nonzero,
                    expected[7..7 + side * side].iter().any(|&v| v != 0)
                );
            }
        }
    }

    #[test]
    fn differential_extreme_values() {
        for side in [8, 16, 32, 64] {
            let src = vec![0; side * side];
            let pred = vec![u8::MAX; side * side];
            let mut residual = vec![0; side * side];
            subtract_square_neon(&src, side, &pred, side, &mut residual, side);
            assert!(residual.iter().all(|&value| value == -255));
            assert!(subtract_square_neon_with_nonzero(
                &src,
                side,
                &pred,
                side,
                &mut residual,
                side
            ));

            subtract_square_neon(&pred, side, &src, side, &mut residual, side);
            assert!(residual.iter().all(|&value| value == 255));
            assert!(!subtract_square_neon_with_nonzero(
                &src,
                side,
                &src,
                side,
                &mut residual,
                side
            ));
        }
    }
}
