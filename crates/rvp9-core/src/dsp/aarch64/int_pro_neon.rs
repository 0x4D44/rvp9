//! NEON integral-projection row reduction for motion estimation.
//!
//! This mirrors libvpx `vpx_int_pro_row_neon` for the 16/32/64-high
//! projection blocks used by VP9's superblock motion seed.

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn vpx_int_pro_row_neon_inner(
    output: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    height: usize,
) {
    let mut row = 0usize;
    let mut sum_lo_0 = vdupq_n_u16(0);
    let mut sum_lo_1 = vdupq_n_u16(0);
    let mut sum_hi_0 = vdupq_n_u16(0);
    let mut sum_hi_1 = vdupq_n_u16(0);

    while row < height {
        let row_0 = vld1q_u8(reference.add(row * reference_stride));
        let row_1 = vld1q_u8(reference.add((row + 1) * reference_stride));
        let row_2 = vld1q_u8(reference.add((row + 2) * reference_stride));
        let row_3 = vld1q_u8(reference.add((row + 3) * reference_stride));

        sum_lo_0 = vaddq_u16(sum_lo_0, vaddl_u8(vget_low_u8(row_0), vget_low_u8(row_1)));
        sum_hi_0 = vaddq_u16(sum_hi_0, vaddl_u8(vget_high_u8(row_0), vget_high_u8(row_1)));
        sum_lo_1 = vaddq_u16(sum_lo_1, vaddl_u8(vget_low_u8(row_2), vget_low_u8(row_3)));
        sum_hi_1 = vaddq_u16(sum_hi_1, vaddl_u8(vget_high_u8(row_2), vget_high_u8(row_3)));
        row += 4;
    }

    let shift = vdupq_n_s16(-((height >> 5) as i16 + 3));
    let average_lo = vshlq_s16(vreinterpretq_s16_u16(vaddq_u16(sum_lo_0, sum_lo_1)), shift);
    let average_hi = vshlq_s16(vreinterpretq_s16_u16(vaddq_u16(sum_hi_0, sum_hi_1)), shift);
    vst1q_s16(output, average_lo);
    vst1q_s16(output.add(8), average_hi);
}

/// Reduces 16 reference columns over a 16/32/64-row projection window.
pub fn vpx_int_pro_row_neon(
    output: &mut [i16],
    reference: &[u8],
    reference_stride: usize,
    height: usize,
) {
    assert_eq!(output.len(), 16);
    assert!(matches!(height, 16 | 32 | 64));
    assert!(reference_stride >= 16);
    assert!(reference.len() >= (height - 1) * reference_stride + 16);

    // SAFETY: AArch64 always provides NEON. The assertions prove that every
    // 16-byte row load and both 8-lane output stores stay inside their slices.
    // A lane sums at most 64 * 255 = 16,320, within unsigned 16-bit range.
    unsafe {
        vpx_int_pro_row_neon_inner(
            output.as_mut_ptr(),
            reference.as_ptr(),
            reference_stride,
            height,
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn vpx_int_pro_col_neon_inner(reference: *const u8, width: usize) -> i32 {
    let mut sum = vpaddlq_u8(vld1q_u8(reference));
    let mut offset = 16usize;
    while offset < width {
        sum = vpadalq_u8(sum, vld1q_u8(reference.add(offset)));
        offset += 16;
    }
    i32::from(vaddvq_u16(sum))
}

/// Sums a 16-byte-multiple reference row for integer-projection motion search.
pub fn vpx_int_pro_col_neon(reference: &[u8], width: usize) -> i32 {
    assert!(width >= 16 && width.is_multiple_of(16));
    assert!(reference.len() >= width);

    // SAFETY: The assertions prove every 16-byte load. At the largest product
    // width, 64 * 255 = 16,320, so the u16 accumulator cannot overflow.
    unsafe { vpx_int_pro_col_neon_inner(reference.as_ptr(), width) }
}

#[target_feature(enable = "neon")]
unsafe fn vpx_vector_var_neon_inner(
    reference: *const i16,
    source: *const i16,
    width: usize,
    shift: u32,
) -> i32 {
    let mut sse = vdupq_n_s32(0);
    let mut total = vdupq_n_s16(0);
    let mut offset = 0usize;
    while offset < width {
        let reference_values = vld1q_s16(reference.add(offset));
        let source_values = vld1q_s16(source.add(offset));
        let diff = vsubq_s16(reference_values, source_values);
        let diff_lo = vget_low_s16(diff);
        let diff_hi = vget_high_s16(diff);
        sse = vmlal_s16(sse, diff_lo, diff_lo);
        sse = vmlal_s16(sse, diff_hi, diff_hi);
        total = vaddq_s16(total, diff);
        offset += 8;
    }

    let mean = i32::from(vaddvq_s16(total));
    vaddvq_s32(sse) - ((mean * mean) >> shift)
}

/// Computes projection-vector variance for 16-, 32-, or 64-wide motion
/// search vectors.
pub fn vpx_vector_var_neon(reference: &[i16], source: &[i16], bwl: u32) -> i32 {
    let width = 4usize << bwl;
    assert!(matches!(width, 16 | 32 | 64));
    assert!(reference.len() >= width);
    assert!(source.len() >= width);

    // SAFETY: The assertions prove each eight-lane load. Projection values
    // are in [0, 510], so every i16 lane sum and i32 squared sum stays in
    // range for the supported widths.
    unsafe { vpx_vector_var_neon_inner(reference.as_ptr(), source.as_ptr(), width, bwl + 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    fn scalar(output: &mut [i16], reference: &[u8], reference_stride: usize, height: usize) {
        let divisor = (height >> 1) as i32;
        for (column, slot) in output.iter_mut().enumerate() {
            let mut sum = 0i32;
            for row in 0..height {
                sum += i32::from(reference[row * reference_stride + column]);
            }
            *slot = (sum / divisor) as i16;
        }
    }

    #[test]
    fn differential_random_heights_strides_and_unaligned_starts() {
        let mut rng = Lcg32(0x1A7E_6A0B);
        for height in [16, 32, 64] {
            for jitter in 0..17 {
                let stride = 16 + jitter;
                let input: Vec<u8> = (0..height * stride + 3).map(|_| rng.byte()).collect();
                let mut expected = [0i16; 16];
                let mut actual = [0i16; 16];
                scalar(&mut expected, &input[3..], stride, height);
                vpx_int_pro_row_neon(&mut actual, &input[3..], stride, height);
                assert_eq!(
                    actual, expected,
                    "mismatch at height {height}, stride jitter {jitter}"
                );
            }
        }
    }

    #[test]
    fn differential_extreme_rows() {
        for height in [16, 32, 64] {
            for value in [0, u8::MAX] {
                let input = vec![value; 16 * height];
                let mut expected = [0i16; 16];
                let mut actual = [0i16; 16];
                scalar(&mut expected, &input, 16, height);
                vpx_int_pro_row_neon(&mut actual, &input, 16, height);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn int_pro_col_matches_scalar_random_unaligned_and_extreme_rows() {
        let mut rng = Lcg32(0x1c01_5eed);
        for width in [16, 32, 64] {
            for offset in 0..17 {
                let input: Vec<u8> = (0..width + offset).map(|_| rng.byte()).collect();
                let reference = &input[offset..];
                let expected: i32 = reference
                    .iter()
                    .take(width)
                    .map(|&value| i32::from(value))
                    .sum();
                assert_eq!(vpx_int_pro_col_neon(reference, width), expected);
            }
            for value in [0, u8::MAX] {
                assert_eq!(
                    vpx_int_pro_col_neon(&vec![value; width], width),
                    width as i32 * i32::from(value)
                );
            }
        }
    }

    fn scalar_vector_var(reference: &[i16], source: &[i16], bwl: u32) -> i32 {
        let width = 4usize << bwl;
        let mut sse = 0i32;
        let mut mean = 0i32;
        for index in 0..width {
            let diff = i32::from(reference[index]) - i32::from(source[index]);
            mean += diff;
            sse += diff * diff;
        }
        sse - ((mean * mean) >> (bwl + 2))
    }

    #[test]
    fn vector_var_matches_scalar_random_and_extreme_projections() {
        let mut rng = Lcg32(0x5ec7_0a11);
        for bwl in [2, 3, 4] {
            let width = 4usize << bwl;
            for _ in 0..256 {
                let reference: Vec<i16> = (0..width).map(|_| i16::from(rng.byte()) * 2).collect();
                let source: Vec<i16> = (0..width).map(|_| i16::from(rng.byte()) * 2).collect();
                assert_eq!(
                    vpx_vector_var_neon(&reference, &source, bwl),
                    scalar_vector_var(&reference, &source, bwl)
                );
            }
            for (reference_value, source_value) in [(0, 0), (0, 510), (510, 0), (510, 510)] {
                let reference = vec![reference_value; width];
                let source = vec![source_value; width];
                assert_eq!(
                    vpx_vector_var_neon(&reference, &source, bwl),
                    scalar_vector_var(&reference, &source, bwl)
                );
            }
        }
    }
}
