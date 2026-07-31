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
}
