//! Regression coverage for safe inverse-transform dispatch storage contracts.
use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn check_transform<const N: usize>(kernel: fn(&[i32; N], &mut [u8], usize), bs: usize) {
    let input = [0; N];
    // An overlapping layout stays within the allocation even before the fix.
    // It proves the release guard executes without invoking out-of-bounds access.
    let mut pixels = vec![128; bs * bs];
    assert!(
        catch_unwind(AssertUnwindSafe(|| kernel(&input, &mut pixels, 0))).is_err(),
        "safe transform must reject a stride shorter than its block width"
    );
    for (len, stride) in [(0, bs), (bs * bs - 1, bs), (bs * bs, usize::MAX)] {
        let mut pixels = vec![128; len];
        assert!(
            catch_unwind(AssertUnwindSafe(|| kernel(&input, &mut pixels, stride))).is_err(),
            "safe transform must reject short or overflowing storage"
        );
    }
}

#[test]
fn safety_bounds_sse2_transform_overlapping_stride() {
    check_transform(idct8x8_sse2_64::idct8x8_add_sse2_64, 8);
    check_transform(idct8x8_12_sse2::idct8x8_add_sse2_12, 8);
    check_transform(idct16x16_sse2_256::idct16x16_add_sse2_256, 16);
    check_transform(idct16x16_sse2_256::idct16x16_add_sse2_10, 16);
    check_transform(idct16x16_sse2_256::idct16x16_add_sse2_38, 16);
    check_transform(idct32x32_sse2_1024::idct32x32_add_sse2_1024, 32);
    check_transform(idct32x32_sse2_1024::idct32x32_add_sse2_34, 32);
    check_transform(idct32x32_sse2_1024::idct32x32_add_sse2_135, 32);
}

#[test]
fn safety_bounds_avx2_transforms_validate_before_isa_entry() {
    check_transform(idct4x4_avx2::idct4x4_add_avx2, 4);
    check_transform(idct32x32_avx2_1024::idct32x32_add_avx2_1024, 32);
    check_transform(idct32x32_avx2_135::idct32x32_add_avx2_135, 32);
    check_transform(
        |input, dst, stride| {
            iht4x4_avx2::iht4x4_add_avx2(input, dst, stride, crate::transforms::TxType::DctDct)
        },
        4,
    );
    check_transform(
        |input, dst, stride| {
            iht8x8_avx2::iht8x8_add_avx2(input, dst, stride, crate::transforms::TxType::DctDct)
        },
        8,
    );
    check_transform(
        |input, dst, stride| {
            iht16x16_avx2::iht16x16_add_avx2(input, dst, stride, crate::transforms::TxType::DctDct)
        },
        16,
    );
}

#[test]
fn safety_bounds_public_transform_getters() {
    let dispatch = crate::dsp::get();

    check_transform(dispatch.idct8x8_add(), 8);
    check_transform(dispatch.idct16x16_add(), 16);
    check_transform(dispatch.idct32x32_add(), 32);
}

#[test]
fn safety_bounds_dequant_rejects_bad_scratch_before_isa_entry() {
    let kernel = plane_dequant_finisher_avx2::plane_dequant_finisher_avx2;
    let mut output = [0; 8];
    assert!(
        catch_unwind(AssertUnwindSafe(|| kernel(
            &[0; 8],
            &[],
            &[1; 8],
            &[0; 8],
            0,
            &mut output
        )))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| kernel(
            &[8; 8],
            &[1; 8],
            &[1; 8],
            &[0; 8],
            0,
            &mut output
        )))
        .is_err()
    );
}
