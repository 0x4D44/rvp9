//! SSE2 8x8 absolute-difference min/max reduction.
//!
//! The primitive mirrors the low-bitdepth `vpx_minmax_8x8` operation used by
//! the realtime variance-based partition chooser. It keeps the exact scalar
//! byte results while reducing all eight rows in SIMD.

use core::arch::x86_64::{
    _mm_loadl_epi64, _mm_max_epu8, _mm_min_epu8, _mm_set1_epi8, _mm_setzero_si128,
    _mm_storeu_si128, _mm_sub_epi8,
};

#[target_feature(enable = "sse2")]
unsafe fn minmax_8x8_sse2_inner(
    first: *const u8,
    first_stride: usize,
    second: *const u8,
    second_stride: usize,
) -> (u8, u8) {
    let mut minimum = _mm_set1_epi8(-1);
    let mut maximum = _mm_setzero_si128();
    for row in 0..8 {
        // G4: loads from `first` / `second` use _mm_loadl_epi64 (unaligned)
        let first_row = _mm_loadl_epi64(first.add(row * first_stride).cast());
        let second_row = _mm_loadl_epi64(second.add(row * second_stride).cast());
        let low = _mm_min_epu8(first_row, second_row);
        let high = _mm_max_epu8(first_row, second_row);
        let difference = _mm_sub_epi8(high, low);
        minimum = _mm_min_epu8(minimum, difference);
        maximum = _mm_max_epu8(maximum, difference);
    }

    let mut minimum_lanes = [0u8; 16];
    let mut maximum_lanes = [0u8; 16];
    _mm_storeu_si128(minimum_lanes.as_mut_ptr().cast(), minimum);
    _mm_storeu_si128(maximum_lanes.as_mut_ptr().cast(), maximum);
    let mut minimum_value = u8::MAX;
    let mut maximum_value = 0;
    for lane in 0..8 {
        minimum_value = minimum_value.min(minimum_lanes[lane]);
        maximum_value = maximum_value.max(maximum_lanes[lane]);
    }
    (minimum_value, maximum_value)
}

/// Return the minimum and maximum absolute pixel difference across an 8x8
/// block.
#[must_use]
pub fn minmax_8x8_sse2(
    first: &[u8],
    first_stride: usize,
    second: &[u8],
    second_stride: usize,
) -> (u8, u8) {
    #[cfg(all(test, not(miri)))]
    crate::dsp::force_scalar_tests::record("minmax_8x8_sse2");
    assert!(first_stride >= 8);
    assert!(second_stride >= 8);
    assert!(first.len() >= 7 * first_stride + 8);
    assert!(second.len() >= 7 * second_stride + 8);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions prove every exact
    // eight-byte row load remains within its source slice.
    unsafe { minmax_8x8_sse2_inner(first.as_ptr(), first_stride, second.as_ptr(), second_stride) }
}

#[cfg(test)]
mod tests {
    use super::minmax_8x8_sse2;

    fn scalar_minmax(
        first: &[u8],
        first_stride: usize,
        second: &[u8],
        second_stride: usize,
    ) -> (u8, u8) {
        let mut minimum = u8::MAX;
        let mut maximum = 0;
        for row in 0..8 {
            for col in 0..8 {
                let difference =
                    first[row * first_stride + col].abs_diff(second[row * second_stride + col]);
                minimum = minimum.min(difference);
                maximum = maximum.max(difference);
            }
        }
        (minimum, maximum)
    }

    #[test]
    fn minmax_8x8_matches_scalar_across_strides_and_offsets() {
        for &(first_stride, second_stride) in &[(8, 8), (13, 19), (64, 64)] {
            let mut first = vec![0u8; 5 + 7 * first_stride + 8];
            let mut second = vec![0u8; 7 + 7 * second_stride + 8];
            for (index, value) in first.iter_mut().enumerate() {
                *value = (index.wrapping_mul(73).wrapping_add(19) & 0xff) as u8;
            }
            for (index, value) in second.iter_mut().enumerate() {
                *value = (index.wrapping_mul(151).wrapping_add(241) & 0xff) as u8;
            }
            first[5] = 0;
            second[7] = 255;

            let first = &first[5..];
            let second = &second[7..];
            assert_eq!(
                minmax_8x8_sse2(first, first_stride, second, second_stride),
                scalar_minmax(first, first_stride, second, second_stride),
            );
        }
    }
}
