//! NEON integral-projection row reduction for motion estimation.
//!
//! This mirrors libvpx `vpx_int_pro_row_neon` for the 16/32/64-high
//! projection blocks used by VP9's superblock motion seed.

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn int_pro_64x64_fused_neon_inner(
    ref_h: *mut i16,
    ref_v: *mut i16,
    src_h: *mut i16,
    src_v: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    source: *const u8,
    source_stride: usize,
) {
    // The reference projection windows form a cross inside this 128x128
    // footprint. The central 64x64 pixels contribute to both projections;
    // load them once and accumulate both the column and row reductions.
    {
        let mut ref_lo = [vdupq_n_u16(0); 8];
        let mut ref_hi = [vdupq_n_u16(0); 8];
        for row in (0..64).step_by(4) {
            let row_0 = reference.add((row + 32) * reference_stride);
            let row_1 = row_0.add(reference_stride);
            let row_2 = row_1.add(reference_stride);
            let row_3 = row_2.add(reference_stride);
            let mut row_sums = [vdupq_n_u16(0); 4];
            for chunk in 0..8 {
                let offset = chunk * 16;
                let r0 = vld1q_u8(row_0.add(offset));
                let r1 = vld1q_u8(row_1.add(offset));
                let r2 = vld1q_u8(row_2.add(offset));
                let r3 = vld1q_u8(row_3.add(offset));
                ref_lo[chunk] = vaddq_u16(
                    ref_lo[chunk],
                    vaddq_u16(
                        vaddl_u8(vget_low_u8(r0), vget_low_u8(r1)),
                        vaddl_u8(vget_low_u8(r2), vget_low_u8(r3)),
                    ),
                );
                ref_hi[chunk] = vaddq_u16(
                    ref_hi[chunk],
                    vaddq_u16(
                        vaddl_u8(vget_high_u8(r0), vget_high_u8(r1)),
                        vaddl_u8(vget_high_u8(r2), vget_high_u8(r3)),
                    ),
                );
                if (2..6).contains(&chunk) {
                    row_sums[0] = vpadalq_u8(row_sums[0], r0);
                    row_sums[1] = vpadalq_u8(row_sums[1], r1);
                    row_sums[2] = vpadalq_u8(row_sums[2], r2);
                    row_sums[3] = vpadalq_u8(row_sums[3], r3);
                }
            }
            for (offset, sum) in row_sums.into_iter().enumerate() {
                *ref_v.add(row + 32 + offset) = (vaddvq_u16(sum) >> 5) as i16;
            }
        }
        let shift = vdupq_n_s16(-5);
        for chunk in 0..8 {
            let lo = vshlq_s16(vreinterpretq_s16_u16(ref_lo[chunk]), shift);
            let hi = vshlq_s16(vreinterpretq_s16_u16(ref_hi[chunk]), shift);
            vst1q_s16(ref_h.add(chunk * 16), lo);
            vst1q_s16(ref_h.add(chunk * 16 + 8), hi);
        }
    }

    // The 32 rows above and below the central square contribute only to the
    // vertical reference projection. Their 64-pixel window begins 32 pixels
    // into the 128-pixel footprint.
    for row in 0..32 {
        for output_row in [row, row + 96] {
            let reference_row = reference.add(output_row * reference_stride + 32);
            let r0 = vld1q_u8(reference_row);
            let r1 = vld1q_u8(reference_row.add(16));
            let r2 = vld1q_u8(reference_row.add(32));
            let r3 = vld1q_u8(reference_row.add(48));
            let mut row_sum = vpaddlq_u8(r0);
            row_sum = vpadalq_u8(row_sum, r1);
            row_sum = vpadalq_u8(row_sum, r2);
            row_sum = vpadalq_u8(row_sum, r3);
            *ref_v.add(output_row) = (vaddvq_u16(row_sum) >> 5) as i16;
        }
    }

    // Source pixels feed both source projections. Four-row groups keep the
    // column accumulator dependency chain short, like libvpx's row kernel.
    {
        let mut src_lo = [vdupq_n_u16(0); 4];
        let mut src_hi = [vdupq_n_u16(0); 4];
        for row in (0..64).step_by(4) {
            let row_0 = source.add(row * source_stride);
            let row_1 = row_0.add(source_stride);
            let row_2 = row_1.add(source_stride);
            let row_3 = row_2.add(source_stride);
            let mut row_sums = [vdupq_n_u16(0); 4];
            for chunk in 0..4 {
                let offset = chunk * 16;
                let s0 = vld1q_u8(row_0.add(offset));
                let s1 = vld1q_u8(row_1.add(offset));
                let s2 = vld1q_u8(row_2.add(offset));
                let s3 = vld1q_u8(row_3.add(offset));
                src_lo[chunk] = vaddq_u16(
                    src_lo[chunk],
                    vaddq_u16(
                        vaddl_u8(vget_low_u8(s0), vget_low_u8(s1)),
                        vaddl_u8(vget_low_u8(s2), vget_low_u8(s3)),
                    ),
                );
                src_hi[chunk] = vaddq_u16(
                    src_hi[chunk],
                    vaddq_u16(
                        vaddl_u8(vget_high_u8(s0), vget_high_u8(s1)),
                        vaddl_u8(vget_high_u8(s2), vget_high_u8(s3)),
                    ),
                );
                row_sums[0] = vpadalq_u8(row_sums[0], s0);
                row_sums[1] = vpadalq_u8(row_sums[1], s1);
                row_sums[2] = vpadalq_u8(row_sums[2], s2);
                row_sums[3] = vpadalq_u8(row_sums[3], s3);
            }
            for (offset, sum) in row_sums.into_iter().enumerate() {
                *src_v.add(row + offset) = (vaddvq_u16(sum) >> 5) as i16;
            }
        }
        let shift = vdupq_n_s16(-5);
        for chunk in 0..4 {
            let lo = vshlq_s16(vreinterpretq_s16_u16(src_lo[chunk]), shift);
            let hi = vshlq_s16(vreinterpretq_s16_u16(src_hi[chunk]), shift);
            vst1q_s16(src_h.add(chunk * 16), lo);
            vst1q_s16(src_h.add(chunk * 16 + 8), hi);
        }
    }
}

/// Computes the four 64x64 integral-projection vectors from direct source and
/// reference footprints while loading their overlapping pixels once.
///
/// `reference` begins at the top-left of the 128x128 footprint containing the
/// horizontal 128x64 and vertical 64x128 projection windows. `source` begins
/// at the top-left of its 64x64 block. The output matches libvpx's
/// `vpx_int_pro_row_neon` and `vpx_int_pro_col_neon` reductions exactly.
#[allow(clippy::too_many_arguments)]
pub fn int_pro_64x64_fused_neon(
    ref_h: &mut [i16; 128],
    ref_v: &mut [i16; 128],
    src_h: &mut [i16; 64],
    src_v: &mut [i16; 64],
    reference: &[u8],
    reference_stride: usize,
    source: &[u8],
    source_stride: usize,
) {
    assert!(reference_stride >= 128);
    assert!(source_stride >= 64);
    let reference_required = 127usize
        .checked_mul(reference_stride)
        .and_then(|offset| offset.checked_add(128))
        .expect("reference projection footprint overflow");
    let source_required = 63usize
        .checked_mul(source_stride)
        .and_then(|offset| offset.checked_add(64))
        .expect("source projection footprint overflow");
    assert!(reference.len() >= reference_required);
    assert!(source.len() >= source_required);

    // SAFETY: AArch64 always provides NEON. The assertions prove every load in
    // the 128x128 reference and 64x64 source footprints and all fixed output
    // stores. Each u16 lane sums at most 64 * 255 = 16,320.
    unsafe {
        int_pro_64x64_fused_neon_inner(
            ref_h.as_mut_ptr(),
            ref_v.as_mut_ptr(),
            src_h.as_mut_ptr(),
            src_v.as_mut_ptr(),
            reference.as_ptr(),
            reference_stride,
            source.as_ptr(),
            source_stride,
        );
    }
}

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
unsafe fn vpx_int_pro_row_clamped_neon_inner(
    output: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    frame_height: usize,
    base_y: i32,
    height: usize,
) {
    let mut sum_lo_0 = vdupq_n_u16(0);
    let mut sum_lo_1 = vdupq_n_u16(0);
    let mut sum_hi_0 = vdupq_n_u16(0);
    let mut sum_hi_1 = vdupq_n_u16(0);
    for row in (0..height).step_by(4) {
        let clamped_row = |offset: usize| {
            (base_y + (row + offset) as i32).clamp(0, frame_height as i32 - 1) as usize
        };
        let row_0 = vld1q_u8(reference.add(clamped_row(0) * reference_stride));
        let row_1 = vld1q_u8(reference.add(clamped_row(1) * reference_stride));
        let row_2 = vld1q_u8(reference.add(clamped_row(2) * reference_stride));
        let row_3 = vld1q_u8(reference.add(clamped_row(3) * reference_stride));
        sum_lo_0 = vaddq_u16(sum_lo_0, vaddl_u8(vget_low_u8(row_0), vget_low_u8(row_1)));
        sum_hi_0 = vaddq_u16(sum_hi_0, vaddl_u8(vget_high_u8(row_0), vget_high_u8(row_1)));
        sum_lo_1 = vaddq_u16(sum_lo_1, vaddl_u8(vget_low_u8(row_2), vget_low_u8(row_3)));
        sum_hi_1 = vaddq_u16(sum_hi_1, vaddl_u8(vget_high_u8(row_2), vget_high_u8(row_3)));
    }
    let shift = vdupq_n_s16(-((height >> 5) as i16 + 3));
    let average_lo = vshlq_s16(vreinterpretq_s16_u16(vaddq_u16(sum_lo_0, sum_lo_1)), shift);
    let average_hi = vshlq_s16(vreinterpretq_s16_u16(vaddq_u16(sum_hi_0, sum_hi_1)), shift);
    vst1q_s16(output, average_lo);
    vst1q_s16(output.add(8), average_hi);
}

/// Reduces 16 columns while replicating rows outside the visible frame.
pub fn vpx_int_pro_row_clamped_neon(
    output: &mut [i16],
    reference: &[u8],
    reference_stride: usize,
    frame_height: usize,
    base_y: i32,
    height: usize,
) {
    assert_eq!(output.len(), 16);
    assert!(matches!(height, 16 | 32 | 64));
    assert!(frame_height > 0);
    assert!(reference_stride >= 16);
    assert!(reference.len() >= (frame_height - 1) * reference_stride + 16);
    // SAFETY: the assertions prove every clamped 16-byte row load and both
    // output stores. Lane sums remain below 64 * 255.
    unsafe {
        vpx_int_pro_row_clamped_neon_inner(
            output.as_mut_ptr(),
            reference.as_ptr(),
            reference_stride,
            frame_height,
            base_y,
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

    fn fused_projection_oracle(
        reference: &[u8],
        reference_stride: usize,
        source: &[u8],
        source_stride: usize,
    ) -> ([i16; 128], [i16; 128], [i16; 64], [i16; 64]) {
        let mut ref_h = [0i16; 128];
        let mut ref_v = [0i16; 128];
        let mut src_h = [0i16; 64];
        let mut src_v = [0i16; 64];
        for (column, slot) in ref_h.iter_mut().enumerate() {
            let sum: i32 = (32..96)
                .map(|row| i32::from(reference[row * reference_stride + column]))
                .sum();
            *slot = (sum >> 5) as i16;
        }
        for (row, slot) in ref_v.iter_mut().enumerate() {
            let sum: i32 = reference[row * reference_stride + 32..][..64]
                .iter()
                .map(|&value| i32::from(value))
                .sum();
            *slot = (sum >> 5) as i16;
        }
        for (column, slot) in src_h.iter_mut().enumerate() {
            let sum: i32 = (0..64)
                .map(|row| i32::from(source[row * source_stride + column]))
                .sum();
            *slot = (sum >> 5) as i16;
        }
        for (row, slot) in src_v.iter_mut().enumerate() {
            let sum: i32 = source[row * source_stride..][..64]
                .iter()
                .map(|&value| i32::from(value))
                .sum();
            *slot = (sum >> 5) as i16;
        }
        (ref_h, ref_v, src_h, src_v)
    }

    #[test]
    fn fused_64x64_projections_match_scalar_with_padding_and_extremes() {
        let mut rng = Lcg32(0x4d44_1088);
        for (reference_stride, source_stride) in [(128, 64), (137, 73), (192, 128)] {
            for pattern in 0..18 {
                let mut reference = vec![0u8; reference_stride * 128];
                let mut source = vec![0u8; source_stride * 64];
                let fill = |index: usize, rng: &mut Lcg32| match pattern {
                    0 => 0,
                    1 => u8::MAX,
                    2 => ((index / reference_stride) & 1) as u8 * u8::MAX,
                    _ => rng.byte(),
                };
                for (index, value) in reference.iter_mut().enumerate() {
                    *value = fill(index, &mut rng);
                }
                for (index, value) in source.iter_mut().enumerate() {
                    *value = fill(index, &mut rng);
                }

                let expected =
                    fused_projection_oracle(&reference, reference_stride, &source, source_stride);
                let mut actual = (
                    [i16::MIN; 128],
                    [i16::MIN; 128],
                    [i16::MIN; 64],
                    [i16::MIN; 64],
                );
                int_pro_64x64_fused_neon(
                    &mut actual.0,
                    &mut actual.1,
                    &mut actual.2,
                    &mut actual.3,
                    &reference,
                    reference_stride,
                    &source,
                    source_stride,
                );
                assert_eq!(
                    actual, expected,
                    "pattern={pattern} strides={reference_stride}/{source_stride}"
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
