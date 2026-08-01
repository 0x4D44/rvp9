//! Dense 8x8 inverse DCT and residual add for AArch64 NEON.
//!
//! This follows libvpx's `vpx_idct8x8_64_add_neon`: load and narrow the
//! 64 transform coefficients, run two transpose-plus-1D passes, then round,
//! add, clamp, and store eight destination rows.

use core::arch::aarch64::*;

use crate::transforms::constants::{
    COSPI_4_64, COSPI_8_64, COSPI_12_64, COSPI_16_64, COSPI_20_64, COSPI_24_64, COSPI_28_64,
};

/// Transpose eight vectors interpreted as an 8x8 matrix of i16 lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn transpose_8x8(io: &mut [int16x8_t; 8]) {
    let src = *io;

    let b0 = vtrn1q_s16(src[0], src[1]);
    let b1 = vtrn2q_s16(src[0], src[1]);
    let b2 = vtrn1q_s16(src[2], src[3]);
    let b3 = vtrn2q_s16(src[2], src[3]);
    let b4 = vtrn1q_s16(src[4], src[5]);
    let b5 = vtrn2q_s16(src[4], src[5]);
    let b6 = vtrn1q_s16(src[6], src[7]);
    let b7 = vtrn2q_s16(src[6], src[7]);

    let c0 = vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b2));
    let c1 = vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b2));
    let c2 = vtrn1q_s32(vreinterpretq_s32_s16(b1), vreinterpretq_s32_s16(b3));
    let c3 = vtrn2q_s32(vreinterpretq_s32_s16(b1), vreinterpretq_s32_s16(b3));
    let c4 = vtrn1q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b6));
    let c5 = vtrn2q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b6));
    let c6 = vtrn1q_s32(vreinterpretq_s32_s16(b5), vreinterpretq_s32_s16(b7));
    let c7 = vtrn2q_s32(vreinterpretq_s32_s16(b5), vreinterpretq_s32_s16(b7));

    io[0] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c0),
        vreinterpretq_s64_s32(c4),
    ));
    io[4] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c0),
        vreinterpretq_s64_s32(c4),
    ));
    io[1] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c2),
        vreinterpretq_s64_s32(c6),
    ));
    io[5] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c2),
        vreinterpretq_s64_s32(c6),
    ));
    io[2] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c1),
        vreinterpretq_s64_s32(c5),
    ));
    io[6] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c1),
        vreinterpretq_s64_s32(c5),
    ));
    io[3] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c3),
        vreinterpretq_s64_s32(c7),
    ));
    io[7] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c3),
        vreinterpretq_s64_s32(c7),
    ));
}

/// VP9's rounded two-output rotation, across eight independent lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn butterfly(a: int16x8_t, b: int16x8_t, c0: i16, c1: i16) -> (int16x8_t, int16x8_t) {
    let out0_lo = vmlsl_n_s16(vmull_n_s16(vget_low_s16(a), c0), vget_low_s16(b), c1);
    let out0_hi = vmlsl_n_s16(vmull_n_s16(vget_high_s16(a), c0), vget_high_s16(b), c1);
    let out1_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(a), c1), vget_low_s16(b), c0);
    let out1_hi = vmlal_n_s16(vmull_n_s16(vget_high_s16(a), c1), vget_high_s16(b), c0);

    (
        vcombine_s16(vrshrn_n_s32::<14>(out0_lo), vrshrn_n_s32::<14>(out0_hi)),
        vcombine_s16(vrshrn_n_s32::<14>(out1_lo), vrshrn_n_s32::<14>(out1_hi)),
    )
}

/// One 8-point inverse DCT for eight independent lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn idct8(io: &mut [int16x8_t; 8]) {
    let input = *io;

    // stage 1
    let (step1_4, step1_7) = butterfly(input[1], input[7], COSPI_28_64 as i16, COSPI_4_64 as i16);
    let (step1_5, step1_6) = butterfly(input[5], input[3], COSPI_12_64 as i16, COSPI_20_64 as i16);

    // stage 2
    let (step2_1, step2_0) = butterfly(input[0], input[4], COSPI_16_64 as i16, COSPI_16_64 as i16);
    let (step2_2, step2_3) = butterfly(input[2], input[6], COSPI_24_64 as i16, COSPI_8_64 as i16);
    let step2_4 = vaddq_s16(step1_4, step1_5);
    let step2_5 = vsubq_s16(step1_4, step1_5);
    let step2_6 = vsubq_s16(step1_7, step1_6);
    let step2_7 = vaddq_s16(step1_7, step1_6);

    // stage 3
    let step1_0 = vaddq_s16(step2_0, step2_3);
    let step1_1 = vaddq_s16(step2_1, step2_2);
    let step1_2 = vsubq_s16(step2_1, step2_2);
    let step1_3 = vsubq_s16(step2_0, step2_3);
    let (step1_5, step1_6) = butterfly(step2_6, step2_5, COSPI_16_64 as i16, COSPI_16_64 as i16);

    // stage 4
    io[0] = vaddq_s16(step1_0, step2_7);
    io[1] = vaddq_s16(step1_1, step1_6);
    io[2] = vaddq_s16(step1_2, step1_5);
    io[3] = vaddq_s16(step1_3, step2_4);
    io[4] = vsubq_s16(step1_3, step2_4);
    io[5] = vsubq_s16(step1_2, step1_5);
    io[6] = vsubq_s16(step1_1, step1_6);
    io[7] = vsubq_s16(step1_0, step2_7);
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_i32_as_i16(input: *const i32) -> int16x8_t {
    vcombine_s16(
        vmovn_s32(vld1q_s32(input)),
        vmovn_s32(vld1q_s32(input.add(4))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn add_and_store_row(residual: int16x8_t, dst: *mut u8) {
    let rounded = vrshrq_n_s16::<5>(residual);
    let prediction = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(dst)));
    vst1_u8(dst, vqmovun_s16(vaddq_s16(rounded, prediction)));
}

#[target_feature(enable = "neon")]
unsafe fn idct8x8_add_neon_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    let zero = vdupq_n_s16(0);
    let mut io = [zero; 8];
    for (row, slot) in io.iter_mut().enumerate() {
        *slot = load_i32_as_i16(input.as_ptr().add(row * 8));
    }

    for _ in 0..2 {
        transpose_8x8(&mut io);
        idct8(&mut io);
    }

    for (row, residual) in io.into_iter().enumerate() {
        add_and_store_row(residual, dst.as_mut_ptr().add(row * stride));
    }
}

/// Add a dense 8x8 inverse-DCT residual to an 8-bit prediction block.
///
/// This wrapper is installed only when runtime capability detection reports
/// NEON support. It validates the destination footprint before entering the
/// target-feature implementation.
pub fn idct8x8_add_neon(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    assert!(stride >= 8, "stride too small for 8x8 add");
    assert!(dst.len() >= 7 * stride + 8, "dst too small for 8x8 add");

    // SAFETY: `SimdDispatch::build` installs this wrapper only when runtime
    // capability detection reports NEON. AArch64 mandates NEON in practice;
    // direct tests also gate on `is_aarch64_feature_detected!("neon")`.
    unsafe { idct8x8_add_neon_inner(input, dst, stride) }
}

#[cfg(all(test, target_arch = "aarch64", not(miri)))]
mod tests {
    use super::*;
    use crate::transforms::inverse::idct8x8_64_add;
    use std::arch::is_aarch64_feature_detected;

    fn compare(input: &[i32; 64], stride: usize, skew: usize) {
        let len = skew + 7 * stride + 8;
        let mut neon = vec![0u8; len + 16];
        for (i, pixel) in neon.iter_mut().enumerate() {
            *pixel = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let mut scalar = neon.clone();
        idct8x8_add_neon(input, &mut neon[skew..len], stride);
        idct8x8_64_add(input, &mut scalar[skew..len], stride);
        assert_eq!(neon, scalar, "stride={stride} skew={skew}");
    }

    #[test]
    fn idct8x8_neon_directed_matches_scalar() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        compare(&[0; 64], 8, 0);
        for position in 0..64 {
            for value in [1, -1, 100, -100, 1000, -1000, 2047, -2048] {
                let mut input = [0; 64];
                input[position] = value;
                compare(&input, 8, 0);
            }
        }
    }

    #[test]
    fn idct8x8_neon_random_valid_inputs_match_scalar() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..512 {
            let mut input = [0i32; 64];
            for coefficient in &mut input {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *coefficient = ((state >> 17) as i32 & 0xfff) - 2048;
            }
            compare(&input, 8 + case % 17, case % 16);
        }
    }
}
