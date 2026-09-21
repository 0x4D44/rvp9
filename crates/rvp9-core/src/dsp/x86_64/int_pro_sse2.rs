//! SSE2 integral-projection kernels for realtime motion estimation.

use core::arch::x86_64::*;

#[target_feature(enable = "sse2")]
unsafe fn vpx_int_pro_row_sse2_inner<const SHIFT: i32>(
    output: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let mut sum_lo_0 = zero;
    let mut sum_lo_1 = zero;
    let mut sum_hi_0 = zero;
    let mut sum_hi_1 = zero;

    for row in (0..height).step_by(4) {
        // G4: loads from `reference` use _mm_loadu_si128 (unaligned).
        let row_0 = _mm_loadu_si128(reference.add(row * reference_stride).cast());
        let row_1 = _mm_loadu_si128(reference.add((row + 1) * reference_stride).cast());
        let row_2 = _mm_loadu_si128(reference.add((row + 2) * reference_stride).cast());
        let row_3 = _mm_loadu_si128(reference.add((row + 3) * reference_stride).cast());

        sum_lo_0 = _mm_add_epi16(
            sum_lo_0,
            _mm_add_epi16(
                _mm_unpacklo_epi8(row_0, zero),
                _mm_unpacklo_epi8(row_1, zero),
            ),
        );
        sum_hi_0 = _mm_add_epi16(
            sum_hi_0,
            _mm_add_epi16(
                _mm_unpackhi_epi8(row_0, zero),
                _mm_unpackhi_epi8(row_1, zero),
            ),
        );
        sum_lo_1 = _mm_add_epi16(
            sum_lo_1,
            _mm_add_epi16(
                _mm_unpacklo_epi8(row_2, zero),
                _mm_unpacklo_epi8(row_3, zero),
            ),
        );
        sum_hi_1 = _mm_add_epi16(
            sum_hi_1,
            _mm_add_epi16(
                _mm_unpackhi_epi8(row_2, zero),
                _mm_unpackhi_epi8(row_3, zero),
            ),
        );
    }

    let average_lo = _mm_srli_epi16(_mm_add_epi16(sum_lo_0, sum_lo_1), SHIFT);
    let average_hi = _mm_srli_epi16(_mm_add_epi16(sum_hi_0, sum_hi_1), SHIFT);
    _mm_storeu_si128(output.cast(), average_lo);
    _mm_storeu_si128(output.add(8).cast(), average_hi);
}

/// Reduces 16 reference columns over a 16/32/64-row projection window.
pub fn vpx_int_pro_row_sse2(
    output: &mut [i16],
    reference: &[u8],
    reference_stride: usize,
    height: usize,
) {
    assert_eq!(output.len(), 16);
    assert!(matches!(height, 16 | 32 | 64));
    assert!(reference_stride >= 16);
    assert!(reference.len() >= (height - 1) * reference_stride + 16);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove every
    // unaligned 16-byte row load and both output stores. A lane sums at most
    // 64 * 255 = 16,320, within unsigned 16-bit range.
    unsafe {
        match height {
            16 => vpx_int_pro_row_sse2_inner::<3>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                height,
            ),
            32 => vpx_int_pro_row_sse2_inner::<4>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                height,
            ),
            64 => vpx_int_pro_row_sse2_inner::<5>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                height,
            ),
            _ => unreachable!("validated projection height"),
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn vpx_int_pro_row_clamped_sse2_inner<const SHIFT: i32>(
    output: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    frame_height: usize,
    base_y: i32,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let mut sum_lo_0 = zero;
    let mut sum_lo_1 = zero;
    let mut sum_hi_0 = zero;
    let mut sum_hi_1 = zero;

    for row in (0..height).step_by(4) {
        let clamped_row = |offset: usize| {
            (base_y + (row + offset) as i32).clamp(0, frame_height as i32 - 1) as usize
        };
        // G4: loads from `reference` use _mm_loadu_si128 (unaligned).
        let row_0 = _mm_loadu_si128(reference.add(clamped_row(0) * reference_stride).cast());
        let row_1 = _mm_loadu_si128(reference.add(clamped_row(1) * reference_stride).cast());
        let row_2 = _mm_loadu_si128(reference.add(clamped_row(2) * reference_stride).cast());
        let row_3 = _mm_loadu_si128(reference.add(clamped_row(3) * reference_stride).cast());

        sum_lo_0 = _mm_add_epi16(
            sum_lo_0,
            _mm_add_epi16(
                _mm_unpacklo_epi8(row_0, zero),
                _mm_unpacklo_epi8(row_1, zero),
            ),
        );
        sum_hi_0 = _mm_add_epi16(
            sum_hi_0,
            _mm_add_epi16(
                _mm_unpackhi_epi8(row_0, zero),
                _mm_unpackhi_epi8(row_1, zero),
            ),
        );
        sum_lo_1 = _mm_add_epi16(
            sum_lo_1,
            _mm_add_epi16(
                _mm_unpacklo_epi8(row_2, zero),
                _mm_unpacklo_epi8(row_3, zero),
            ),
        );
        sum_hi_1 = _mm_add_epi16(
            sum_hi_1,
            _mm_add_epi16(
                _mm_unpackhi_epi8(row_2, zero),
                _mm_unpackhi_epi8(row_3, zero),
            ),
        );
    }

    let average_lo = _mm_srli_epi16(_mm_add_epi16(sum_lo_0, sum_lo_1), SHIFT);
    let average_hi = _mm_srli_epi16(_mm_add_epi16(sum_hi_0, sum_hi_1), SHIFT);
    _mm_storeu_si128(output.cast(), average_lo);
    _mm_storeu_si128(output.add(8).cast(), average_hi);
}

/// Reduces 16 columns while replicating rows outside the visible frame.
pub fn vpx_int_pro_row_clamped_sse2(
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
        match height {
            16 => vpx_int_pro_row_clamped_sse2_inner::<3>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                frame_height,
                base_y,
                height,
            ),
            32 => vpx_int_pro_row_clamped_sse2_inner::<4>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                frame_height,
                base_y,
                height,
            ),
            64 => vpx_int_pro_row_clamped_sse2_inner::<5>(
                output.as_mut_ptr(),
                reference.as_ptr(),
                reference_stride,
                frame_height,
                base_y,
                height,
            ),
            _ => unreachable!("validated projection height"),
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn sum_64_bytes_sse2(input: *const u8) -> i32 {
    let zero = _mm_setzero_si128();
    let mut sum = zero;
    for offset in (0..64).step_by(16) {
        // G4: loads from `input` use _mm_loadu_si128 (unaligned).
        let values = _mm_loadu_si128(input.add(offset).cast());
        sum = _mm_add_epi64(sum, _mm_sad_epu8(values, zero));
    }
    let mut lanes = [0u64; 2];
    _mm_storeu_si128(lanes.as_mut_ptr().cast(), sum);
    (lanes[0] + lanes[1]) as i32
}

#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn int_pro_64x64_fused_sse2_inner(
    ref_h: *mut i16,
    ref_v: *mut i16,
    src_h: *mut i16,
    src_v: *mut i16,
    reference: *const u8,
    reference_stride: usize,
    source: *const u8,
    source_stride: usize,
) {
    let zero = _mm_setzero_si128();

    // The central 64 rows contribute to all 128 horizontal reference
    // projections. Their middle 64 columns also supply the vertical sums.
    let mut ref_cols = [zero; 16];
    for row in 0..64 {
        let row_ptr = reference.add((row + 32) * reference_stride);
        let mut row_sum = zero;
        for chunk in 0..8 {
            // G4: loads from `reference` use _mm_loadu_si128 (unaligned).
            let values = _mm_loadu_si128(row_ptr.add(chunk * 16).cast());
            ref_cols[chunk * 2] =
                _mm_add_epi16(ref_cols[chunk * 2], _mm_unpacklo_epi8(values, zero));
            ref_cols[chunk * 2 + 1] =
                _mm_add_epi16(ref_cols[chunk * 2 + 1], _mm_unpackhi_epi8(values, zero));
            if (2..6).contains(&chunk) {
                row_sum = _mm_add_epi64(row_sum, _mm_sad_epu8(values, zero));
            }
        }
        let mut lanes = [0u64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), row_sum);
        *ref_v.add(row + 32) = ((lanes[0] + lanes[1]) >> 5) as i16;
    }
    for (chunk, values) in ref_cols.into_iter().enumerate() {
        let normalized = _mm_srli_epi16(values, 5);
        _mm_storeu_si128(ref_h.add(chunk * 8).cast(), normalized);
    }

    // The upper and lower 32 rows contribute only to the 64-pixel vertical
    // reference projection centered in the 128-pixel footprint.
    for row in 0..32 {
        for output_row in [row, row + 96] {
            let row_ptr = reference.add(output_row * reference_stride + 32);
            *ref_v.add(output_row) = (sum_64_bytes_sse2(row_ptr) >> 5) as i16;
        }
    }

    // Every source pixel contributes to both source projections.
    let mut src_cols = [zero; 8];
    for row in 0..64 {
        let row_ptr = source.add(row * source_stride);
        let mut row_sum = zero;
        for chunk in 0..4 {
            // G4: loads from `source` use _mm_loadu_si128 (unaligned).
            let values = _mm_loadu_si128(row_ptr.add(chunk * 16).cast());
            src_cols[chunk * 2] =
                _mm_add_epi16(src_cols[chunk * 2], _mm_unpacklo_epi8(values, zero));
            src_cols[chunk * 2 + 1] =
                _mm_add_epi16(src_cols[chunk * 2 + 1], _mm_unpackhi_epi8(values, zero));
            row_sum = _mm_add_epi64(row_sum, _mm_sad_epu8(values, zero));
        }
        let mut lanes = [0u64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), row_sum);
        *src_v.add(row) = ((lanes[0] + lanes[1]) >> 5) as i16;
    }
    for (chunk, values) in src_cols.into_iter().enumerate() {
        let normalized = _mm_srli_epi16(values, 5);
        _mm_storeu_si128(src_h.add(chunk * 8).cast(), normalized);
    }
}

/// Computes the four 64x64 integral-projection vectors from overlapping direct
/// source and reference footprints.
#[allow(clippy::too_many_arguments)]
pub fn int_pro_64x64_fused_sse2(
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

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove every
    // unaligned load and fixed output store. A u16 lane sums at most
    // 64 * 255 = 16,320.
    unsafe {
        int_pro_64x64_fused_sse2_inner(
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

#[target_feature(enable = "sse2")]
unsafe fn vpx_vector_var_sse2_inner(
    reference: *const i16,
    source: *const i16,
    width: usize,
    shift: u32,
) -> i32 {
    let mut sse = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();
    for offset in (0..width).step_by(8) {
        // G4: loads from `reference` and `source` use _mm_loadu_si128 (unaligned).
        let reference_values = _mm_loadu_si128(reference.add(offset).cast());
        let source_values = _mm_loadu_si128(source.add(offset).cast());
        let diff = _mm_sub_epi16(reference_values, source_values);
        sse = _mm_add_epi32(sse, _mm_madd_epi16(diff, diff));
        total = _mm_add_epi16(total, diff);
    }

    let mut sse_lanes = [0i32; 4];
    let mut total_lanes = [0i16; 8];
    _mm_storeu_si128(sse_lanes.as_mut_ptr().cast(), sse);
    _mm_storeu_si128(total_lanes.as_mut_ptr().cast(), total);
    let sse = sse_lanes.into_iter().sum::<i32>();
    let mean = total_lanes.into_iter().map(i32::from).sum::<i32>();
    sse - ((mean * mean) >> shift)
}

/// Computes projection-vector variance for 16-, 32-, or 64-wide vectors.
pub fn vpx_vector_var_sse2(reference: &[i16], source: &[i16], bwl: u32) -> i32 {
    let width = 4usize << bwl;
    assert!(matches!(width, 16 | 32 | 64));
    assert!(reference.len() >= width);
    assert!(source.len() >= width);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove each
    // unaligned eight-lane load. Projection values are in [0, 510], so all
    // lane accumulators and the final i32 arithmetic remain in range.
    unsafe { vpx_vector_var_sse2_inner(reference.as_ptr(), source.as_ptr(), width, bwl + 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 24) as u8
        }
    }

    fn row_projection_oracle(
        reference: &[u8],
        reference_stride: usize,
        height: usize,
    ) -> [i16; 16] {
        let divisor = (height >> 1) as i32;
        let mut output = [0i16; 16];
        for (column, slot) in output.iter_mut().enumerate() {
            let sum: i32 = (0..height)
                .map(|row| i32::from(reference[row * reference_stride + column]))
                .sum();
            *slot = (sum / divisor) as i16;
        }
        output
    }

    #[test]
    fn row_projection_sse2_matches_scalar_random_strides_and_unaligned_starts() {
        let mut rng = Lcg32(0x1a7e_6a0b);
        for height in [16, 32, 64] {
            for jitter in 0..17 {
                let stride = 16 + jitter;
                for pattern in 0..18 {
                    let mut input = vec![0u8; height * stride + 3];
                    for (index, value) in input.iter_mut().enumerate() {
                        *value = match pattern {
                            0 => 0,
                            1 => u8::MAX,
                            _ => rng.byte(),
                        };
                        if pattern == 2 {
                            *value = ((index * 37 + index / stride * 19) & 0xff) as u8;
                        }
                    }
                    let reference = &input[3..];
                    let expected = row_projection_oracle(reference, stride, height);
                    let mut actual = [i16::MIN; 16];
                    vpx_int_pro_row_sse2(&mut actual, reference, stride, height);
                    assert_eq!(
                        actual, expected,
                        "height={height} stride={stride} pattern={pattern}"
                    );
                }
            }
        }
    }

    fn clamped_row_projection_oracle(
        reference: &[u8],
        reference_stride: usize,
        frame_height: usize,
        base_y: i32,
        height: usize,
    ) -> [i16; 16] {
        let divisor = (height >> 1) as i32;
        let mut output = [0i16; 16];
        for (column, slot) in output.iter_mut().enumerate() {
            let sum: i32 = (0..height)
                .map(|row| {
                    let y = (base_y + row as i32).clamp(0, frame_height as i32 - 1) as usize;
                    i32::from(reference[y * reference_stride + column])
                })
                .sum();
            *slot = (sum / divisor) as i16;
        }
        output
    }

    #[test]
    fn clamped_row_projection_sse2_matches_scalar_edge_rows() {
        let frame_height = 83;
        let mut rng = Lcg32(0x6a0b_4d44);
        for stride in [16, 17, 32, 104] {
            for pattern in 0..4 {
                let mut input = vec![0u8; frame_height * stride + 3];
                for (index, value) in input.iter_mut().enumerate() {
                    *value = match pattern {
                        0 => 0,
                        1 => u8::MAX,
                        2 => ((index * 37 + index / stride * 19) & 0xff) as u8,
                        _ => rng.byte(),
                    };
                }
                let reference = &input[3..];
                for height in [16, 32, 64] {
                    for base_y in [-70, -3, 0, 31, 80, 100] {
                        let expected = clamped_row_projection_oracle(
                            reference,
                            stride,
                            frame_height,
                            base_y,
                            height,
                        );
                        let mut actual = [i16::MIN; 16];
                        vpx_int_pro_row_clamped_sse2(
                            &mut actual,
                            reference,
                            stride,
                            frame_height,
                            base_y,
                            height,
                        );
                        assert_eq!(
                            actual, expected,
                            "stride={stride} pattern={pattern} height={height} base_y={base_y}"
                        );
                    }
                }
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
                int_pro_64x64_fused_sse2(
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
                    vpx_vector_var_sse2(&reference, &source, bwl),
                    scalar_vector_var(&reference, &source, bwl)
                );
            }
            for (reference_value, source_value) in [(0, 0), (0, 510), (510, 0), (510, 510)] {
                let reference = vec![reference_value; width];
                let source = vec![source_value; width];
                assert_eq!(
                    vpx_vector_var_sse2(&reference, &source, bwl),
                    scalar_vector_var(&reference, &source, bwl)
                );
            }
        }
    }
}
