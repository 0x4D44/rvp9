//! NEON 8x8 absolute-difference min/max reduction.
//!
//! This is the low-bitdepth `vpx_minmax_8x8_neon` primitive used by
//! libvpx's realtime variance-based partition chooser.

use crate::dsp::bounds::strided_len;

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn minmax_8x8_neon_inner(
    first: *const u8,
    first_stride: usize,
    second: *const u8,
    second_stride: usize,
) -> (u8, u8) {
    let first01 = vcombine_u8(vld1_u8(first), vld1_u8(first.add(first_stride)));
    let first23 = vcombine_u8(
        vld1_u8(first.add(2 * first_stride)),
        vld1_u8(first.add(3 * first_stride)),
    );
    let first45 = vcombine_u8(
        vld1_u8(first.add(4 * first_stride)),
        vld1_u8(first.add(5 * first_stride)),
    );
    let first67 = vcombine_u8(
        vld1_u8(first.add(6 * first_stride)),
        vld1_u8(first.add(7 * first_stride)),
    );

    let second01 = vcombine_u8(vld1_u8(second), vld1_u8(second.add(second_stride)));
    let second23 = vcombine_u8(
        vld1_u8(second.add(2 * second_stride)),
        vld1_u8(second.add(3 * second_stride)),
    );
    let second45 = vcombine_u8(
        vld1_u8(second.add(4 * second_stride)),
        vld1_u8(second.add(5 * second_stride)),
    );
    let second67 = vcombine_u8(
        vld1_u8(second.add(6 * second_stride)),
        vld1_u8(second.add(7 * second_stride)),
    );

    let diff01 = vabdq_u8(first01, second01);
    let diff23 = vabdq_u8(first23, second23);
    let diff45 = vabdq_u8(first45, second45);
    let diff67 = vabdq_u8(first67, second67);
    let minimum = vminq_u8(vminq_u8(diff01, diff23), vminq_u8(diff45, diff67));
    let maximum = vmaxq_u8(vmaxq_u8(diff01, diff23), vmaxq_u8(diff45, diff67));

    (vminvq_u8(minimum), vmaxvq_u8(maximum))
}

/// Return the minimum and maximum absolute pixel difference across an 8x8
/// block.
#[must_use]
pub fn minmax_8x8_neon(
    first: &[u8],
    first_stride: usize,
    second: &[u8],
    second_stride: usize,
) -> (u8, u8) {
    assert!(first_stride >= 8);
    assert!(second_stride >= 8);
    assert!(first.len() >= strided_len(7, first_stride, 8));
    assert!(second.len() >= strided_len(7, second_stride, 8));

    // SAFETY: AArch64 includes NEON. The assertions prove that every exact
    // eight-byte row load remains within its source slice.
    unsafe { minmax_8x8_neon_inner(first.as_ptr(), first_stride, second.as_ptr(), second_stride) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::minmax_8x8_neon;

    fn scalar_minmax(
        first: &[u8],
        first_stride: usize,
        second: &[u8],
        second_stride: usize,
    ) -> (u8, u8) {
        let mut minimum = u8::MAX;
        let mut maximum = u8::MIN;
        for row in 0..8 {
            for col in 0..8 {
                let diff =
                    first[row * first_stride + col].abs_diff(second[row * second_stride + col]);
                minimum = minimum.min(diff);
                maximum = maximum.max(diff);
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
                minmax_8x8_neon(first, first_stride, second, second_stride),
                scalar_minmax(first, first_stride, second, second_stride),
            );
        }
    }
}
