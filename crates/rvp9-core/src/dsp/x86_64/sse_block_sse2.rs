//! SSE2 squared-error reduction for inter mode-decision blocks.
//!
//! The realtime picker needs only the raw sum of squared byte differences,
//! while the existing variance family also computes a signed sum.  This
//! kernel keeps the reduction to the value the picker consumes and therefore
//! avoids the extra sum lane of the generic variance dispatch.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16,
    _mm_setzero_si128, _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
};

/// Horizontal sum of four non-overflowing i32 lanes.
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn hsum_epi32(value: __m128i) -> u32 {
    let value = _mm_add_epi32(value, _mm_srli_si128(value, 8));
    let value = _mm_add_epi32(value, _mm_srli_si128(value, 4));
    _mm_cvtsi128_si32(value) as u32
}

/// SSE2 implementation of a strided byte-block SSE reduction.
///
/// The reduction is exact: each byte is widened to i16 before subtraction,
/// then `_mm_madd_epi16` squares adjacent differences into i32 lanes.  A
/// legal VP9 block is at most 64×64, so the total squared error fits in i32.
/// Widths are generic to cover the rectangular partition leaves; 16-byte and
/// 8-byte chunks use SIMD and a final 1–7 byte tail uses the scalar reference.
///
/// # Safety
///
/// SSE2 must be available.  Each input must cover
/// `(height - 1) * stride + width` bytes, and `width`/`height` must be non-zero.
// G4: loads from `src` / `pred` use _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
#[target_feature(enable = "sse2")]
unsafe fn sse_block_sse2_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let zero = _mm_setzero_si128();
    let mut total = _mm_setzero_si128();
    let mut scalar_total = 0u64;

    for row in 0..height {
        let src_row = src.add(row * src_stride);
        let pred_row = pred.add(row * pred_stride);
        let mut col = 0usize;

        while col + 16 <= width {
            let src_values = _mm_loadu_si128(src_row.add(col).cast());
            let pred_values = _mm_loadu_si128(pred_row.add(col).cast());
            let diff_lo = _mm_sub_epi16(
                _mm_unpacklo_epi8(src_values, zero),
                _mm_unpacklo_epi8(pred_values, zero),
            );
            let diff_hi = _mm_sub_epi16(
                _mm_unpackhi_epi8(src_values, zero),
                _mm_unpackhi_epi8(pred_values, zero),
            );
            total = _mm_add_epi32(total, _mm_madd_epi16(diff_lo, diff_lo));
            total = _mm_add_epi32(total, _mm_madd_epi16(diff_hi, diff_hi));
            col += 16;
        }

        if col + 8 <= width {
            let src_values = _mm_loadl_epi64(src_row.add(col).cast());
            let pred_values = _mm_loadl_epi64(pred_row.add(col).cast());
            let diff = _mm_sub_epi16(
                _mm_unpacklo_epi8(src_values, zero),
                _mm_unpacklo_epi8(pred_values, zero),
            );
            total = _mm_add_epi32(total, _mm_madd_epi16(diff, diff));
            col += 8;
        }

        while col < width {
            let diff = i32::from(*src_row.add(col)) - i32::from(*pred_row.add(col));
            scalar_total += (diff * diff) as u64;
            col += 1;
        }
    }

    u64::from(hsum_epi32(total)) + scalar_total
}

/// Reduce a strided byte block to its exact sum of squared differences.
#[cfg(target_arch = "x86_64")]
pub fn sse_block_sse2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    #[cfg(all(test, not(miri)))]
    crate::dsp::force_scalar_tests::record("sse_block_sse2");
    assert!(width > 0 && height > 0);
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= (height - 1) * src_stride + width);
    assert!(pred.len() >= (height - 1) * pred_stride + width);

    // SAFETY: SSE2 is mandatory on x86-64. The assertions above prove every
    // unaligned load stays within both input slices.
    unsafe {
        sse_block_sse2_inner(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            width,
            height,
        )
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::sse_block_sse2;

    fn scalar_sse(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        width: usize,
        height: usize,
    ) -> u64 {
        let mut total = 0u64;
        for row in 0..height {
            for col in 0..width {
                let diff = i32::from(src[row * src_stride + col])
                    - i32::from(pred[row * pred_stride + col]);
                total += (diff * diff) as u64;
            }
        }
        total
    }

    #[test]
    fn differential_all_vp9_shapes_and_strides() {
        assert!(is_x86_feature_detected!("sse2"));
        let shapes = [
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 64),
        ];
        for (width, height) in shapes {
            for jitter in 0..9 {
                let src_stride = width + jitter;
                let pred_stride = width + ((jitter * 3 + 1) % 9);
                let src: Vec<u8> = (0..height * src_stride + 7)
                    .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
                    .collect();
                let pred: Vec<u8> = (0..height * pred_stride + 13)
                    .map(|index| (index as u8).wrapping_mul(19).wrapping_add(73))
                    .collect();
                assert_eq!(
                    sse_block_sse2(&src, src_stride, &pred, pred_stride, width, height),
                    scalar_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "shape={width}x{height} jitter={jitter}",
                );
            }
        }
    }

    #[test]
    fn differential_extreme_values_and_zero_detection() {
        let width = 64;
        let height = 64;
        let src = vec![u8::MAX; width * height + 5];
        let pred = vec![0u8; width * height + 3];
        assert_eq!(
            sse_block_sse2(&src, width, &pred, width, width, height),
            u64::from(width as u32 * height as u32) * u64::from(u8::MAX) * u64::from(u8::MAX),
        );
        let equal = vec![127u8; width * height];
        assert_eq!(
            sse_block_sse2(&equal, width, &equal, width, width, height),
            0
        );
    }
}
