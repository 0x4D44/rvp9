//! AArch64 reductions for the realtime variance-partition front end.

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn avg_diff_8x8_grid_64x64_neon_inner(
    source: *const u8,
    source_stride: usize,
    predictor: *const u8,
    predictor_stride: usize,
) -> [i16; 64] {
    let mut averages = [0i16; 64];
    for cell_row in 0..8 {
        let mut source_columns = [vdupq_n_u16(0); 8];
        let mut predictor_columns = [vdupq_n_u16(0); 8];
        for row in 0..8 {
            let source_row = (cell_row * 8 + row) * source_stride;
            let predictor_row = (cell_row * 8 + row) * predictor_stride;
            for pair in 0..4 {
                // SAFETY: The wrapper proves that each 16-byte row load is in
                // bounds. Both pointers originate from live immutable slices.
                let source_pair = unsafe { vld1q_u8(source.add(source_row + pair * 16)) };
                // SAFETY: Same bounds proof as the source load above.
                let predictor_pair = unsafe { vld1q_u8(predictor.add(predictor_row + pair * 16)) };
                source_columns[pair * 2] =
                    vaddw_u8(source_columns[pair * 2], vget_low_u8(source_pair));
                source_columns[pair * 2 + 1] =
                    vaddw_u8(source_columns[pair * 2 + 1], vget_high_u8(source_pair));
                predictor_columns[pair * 2] =
                    vaddw_u8(predictor_columns[pair * 2], vget_low_u8(predictor_pair));
                predictor_columns[pair * 2 + 1] = vaddw_u8(
                    predictor_columns[pair * 2 + 1],
                    vget_high_u8(predictor_pair),
                );
            }
        }
        for cell_col in 0..8 {
            let source_sum = vaddlvq_u16(source_columns[cell_col]);
            let predictor_sum = vaddlvq_u16(predictor_columns[cell_col]);
            let source_average = ((source_sum + 32) >> 6) as i32;
            let predictor_average = ((predictor_sum + 32) >> 6) as i32;
            averages[cell_row * 8 + cell_col] = (source_average - predictor_average) as i16;
        }
    }
    averages
}

/// Return the exact rounded source-minus-predictor average for every 8x8 cell
/// in a 64x64 block.
///
/// Cells are stored in raster order. Each plane may use an independent
/// stride, which lets the encoder consume a frame-backed source without first
/// materializing a tight copy.
#[must_use]
pub fn avg_diff_8x8_grid_64x64_neon(
    source: &[u8],
    source_stride: usize,
    predictor: &[u8],
    predictor_stride: usize,
) -> [i16; 64] {
    assert!(source_stride >= 64);
    assert!(predictor_stride >= 64);
    assert!(source.len() >= 63 * source_stride + 64);
    assert!(predictor.len() >= 63 * predictor_stride + 64);

    // SAFETY: AArch64 guarantees NEON. The assertions prove every 8-byte
    // load in the 64x64 footprints, and the immutable input slices remain
    // live for the duration of the call.
    unsafe {
        avg_diff_8x8_grid_64x64_neon_inner(
            source.as_ptr(),
            source_stride,
            predictor.as_ptr(),
            predictor_stride,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::avg_diff_8x8_grid_64x64_neon;

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
                        let source_pixel =
                            source[(cell_row * 8 + row) * source_stride + cell_col * 8 + col];
                        let predictor_pixel =
                            predictor[(cell_row * 8 + row) * predictor_stride + cell_col * 8 + col];
                        source_sum += u32::from(source_pixel);
                        predictor_sum += u32::from(predictor_pixel);
                    }
                }
                averages[cell_row * 8 + cell_col] =
                    (((source_sum + 32) >> 6) as i32 - ((predictor_sum + 32) >> 6) as i32) as i16;
            }
        }
        averages
    }

    fn fill_random(bytes: &mut [u8], mut state: u32) -> u32 {
        for byte in bytes {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = state as u8;
        }
        state
    }

    #[test]
    fn avg_diff_grid_matches_scalar_for_strides_and_extremes() {
        for (source_stride, predictor_stride) in [(64, 64), (65, 79), (80, 96), (1920, 64)] {
            let mut source = vec![0u8; 63 * source_stride + 64];
            let mut predictor = vec![0u8; 63 * predictor_stride + 64];
            for (source_value, predictor_value) in [(0, 0), (0, 255), (255, 0), (255, 255)] {
                source.fill(source_value);
                predictor.fill(predictor_value);
                assert_eq!(
                    avg_diff_8x8_grid_64x64_neon(
                        &source,
                        source_stride,
                        &predictor,
                        predictor_stride,
                    ),
                    scalar_grid(&source, source_stride, &predictor, predictor_stride),
                );
            }

            let mut state = 0x9e37_79b9;
            for _ in 0..256 {
                state = fill_random(&mut source, state);
                state = fill_random(&mut predictor, state);
                assert_eq!(
                    avg_diff_8x8_grid_64x64_neon(
                        &source,
                        source_stride,
                        &predictor,
                        predictor_stride,
                    ),
                    scalar_grid(&source, source_stride, &predictor, predictor_stride),
                );
            }
        }
    }
}
