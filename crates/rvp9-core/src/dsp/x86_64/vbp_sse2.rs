//! SSE2 reductions for the realtime variance-partition front end.
//!
//! The kernel mirrors the exact source-minus-predictor average grid used by
//! the AArch64 implementation and by pinned libvpx's x86 average primitive.
//! It keeps one pair of 64-bit accumulators per adjacent-cell pair, so each
//! source and predictor pixel is loaded once while the two rounded averages
//! remain independent.

use core::arch::x86_64::{
    _mm_add_epi64, _mm_loadu_si128, _mm_sad_epu8, _mm_setzero_si128, _mm_storeu_si128,
};

/// Compute the exact independently rounded source-minus-predictor average for
/// every 8x8 cell in a 64x64 block.
///
/// Cells are returned in raster order. Each plane may use an independent
/// stride, and each slice may begin at an arbitrary alignment.
// G4: loads from `source` / `predictor` use _mm_loadu_si128 (unaligned)
#[target_feature(enable = "sse2")]
unsafe fn avg_diff_8x8_grid_64x64_sse2_inner(
    source: *const u8,
    source_stride: usize,
    predictor: *const u8,
    predictor_stride: usize,
) -> [i16; 64] {
    let zero = _mm_setzero_si128();
    let mut averages = [0i16; 64];

    for cell_row in 0..8 {
        let mut source_pairs = [zero; 4];
        let mut predictor_pairs = [zero; 4];

        for row in 0..8 {
            let source_row = source.add((cell_row * 8 + row) * source_stride);
            let predictor_row = predictor.add((cell_row * 8 + row) * predictor_stride);

            for pair in 0..4 {
                let source_values = _mm_loadu_si128(source_row.add(pair * 16).cast());
                let predictor_values = _mm_loadu_si128(predictor_row.add(pair * 16).cast());
                source_pairs[pair] =
                    _mm_add_epi64(source_pairs[pair], _mm_sad_epu8(source_values, zero));
                predictor_pairs[pair] =
                    _mm_add_epi64(predictor_pairs[pair], _mm_sad_epu8(predictor_values, zero));
            }
        }

        for pair in 0..4 {
            let mut source_sums = [0u64; 2];
            let mut predictor_sums = [0u64; 2];
            _mm_storeu_si128(source_sums.as_mut_ptr().cast(), source_pairs[pair]);
            _mm_storeu_si128(predictor_sums.as_mut_ptr().cast(), predictor_pairs[pair]);

            for lane in 0..2 {
                let source_average = (source_sums[lane] + 32) >> 6;
                let predictor_average = (predictor_sums[lane] + 32) >> 6;
                averages[cell_row * 8 + pair * 2 + lane] =
                    (source_average as i32 - predictor_average as i32) as i16;
            }
        }
    }

    averages
}

/// Return the exact rounded source-minus-predictor average for every 8x8
/// cell in a complete 64x64 block.
#[must_use]
pub fn avg_diff_8x8_grid_64x64_sse2(
    source: &[u8],
    source_stride: usize,
    predictor: &[u8],
    predictor_stride: usize,
) -> [i16; 64] {
    #[cfg(all(test, not(miri)))]
    crate::dsp::force_scalar_tests::record("avg_diff_8x8_grid_64x64_sse2");
    assert!(source_stride >= 64);
    assert!(predictor_stride >= 64);
    assert!(source.len() >= 63 * source_stride + 64);
    assert!(predictor.len() >= 63 * predictor_stride + 64);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove every
    // unaligned 16-byte load stays within each complete 64x64 footprint.
    unsafe {
        avg_diff_8x8_grid_64x64_sse2_inner(
            source.as_ptr(),
            source_stride,
            predictor.as_ptr(),
            predictor_stride,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::avg_diff_8x8_grid_64x64_sse2;

    struct XorShift32(u32);

    impl XorShift32 {
        fn byte(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0 as u8
        }
    }

    fn scalar_grid(
        source: &[u8],
        source_stride: usize,
        predictor: &[u8],
        predictor_stride: usize,
    ) -> [i16; 64] {
        let mut averages = [0i16; 64];
        for cell_row in 0..8 {
            for cell_col in 0..8 {
                let mut source_sum = 0u32;
                let mut predictor_sum = 0u32;
                for row in 0..8 {
                    for col in 0..8 {
                        source_sum += u32::from(
                            source[(cell_row * 8 + row) * source_stride + cell_col * 8 + col],
                        );
                        predictor_sum += u32::from(
                            predictor[(cell_row * 8 + row) * predictor_stride + cell_col * 8 + col],
                        );
                    }
                }
                averages[cell_row * 8 + cell_col] =
                    (((source_sum + 32) >> 6) as i32 - ((predictor_sum + 32) >> 6) as i32) as i16;
            }
        }
        averages
    }

    fn assert_matches(
        source_stride: usize,
        predictor_stride: usize,
        source_offset: usize,
        predictor_offset: usize,
        source: &mut [u8],
        predictor: &mut [u8],
    ) {
        let source = &source[source_offset..];
        let predictor = &predictor[predictor_offset..];
        assert_eq!(
            avg_diff_8x8_grid_64x64_sse2(source, source_stride, predictor, predictor_stride,),
            scalar_grid(source, source_stride, predictor, predictor_stride),
            "SSE2 grid mismatch at source stride {source_stride}, predictor stride {predictor_stride}, offsets {source_offset}/{predictor_offset}",
        );
    }

    #[test]
    fn differential_grid_matches_scalar_for_padding_offsets_and_extremes() {
        assert!(is_x86_feature_detected!("sse2"));
        let layouts = [
            (64, 64, 0, 0),
            (65, 79, 3, 5),
            (80, 96, 7, 11),
            (1920, 64, 13, 17),
        ];
        let mut rng = XorShift32(0x9e37_79b9);

        for &(source_stride, predictor_stride, source_offset, predictor_offset) in &layouts {
            let source_len = source_offset + 63 * source_stride + 64 + 23;
            let predictor_len = predictor_offset + 63 * predictor_stride + 64 + 29;
            let mut source = vec![0u8; source_len];
            let mut predictor = vec![0u8; predictor_len];

            for &(source_value, predictor_value) in &[(0, 0), (0, 255), (255, 0), (255, 255)] {
                source.fill(source_value);
                predictor.fill(predictor_value);
                assert_matches(
                    source_stride,
                    predictor_stride,
                    source_offset,
                    predictor_offset,
                    &mut source,
                    &mut predictor,
                );
            }

            for row in 0..64 {
                for col in 0..64 {
                    source[source_offset + row * source_stride + col] =
                        if (row + col) & 1 == 0 { 0 } else { 255 };
                    predictor[predictor_offset + row * predictor_stride + col] =
                        if (row * 3 + col) & 1 == 0 { 0 } else { 255 };
                }
            }
            assert_matches(
                source_stride,
                predictor_stride,
                source_offset,
                predictor_offset,
                &mut source,
                &mut predictor,
            );

            for _ in 0..256 {
                for byte in &mut source {
                    *byte = rng.byte();
                }
                for byte in &mut predictor {
                    *byte = rng.byte();
                }
                assert_matches(
                    source_stride,
                    predictor_stride,
                    source_offset,
                    predictor_offset,
                    &mut source,
                    &mut predictor,
                );
            }
        }
    }
}
