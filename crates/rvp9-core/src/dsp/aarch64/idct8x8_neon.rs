//! Dense and sparse 8x8 inverse DCT residual adds for AArch64 NEON.
//!
//! This follows libvpx's `vpx_idct8x8_64_add_neon`: load and narrow the
//! 64 transform coefficients, run two transpose-plus-1D passes, then round,
//! add, clamp, and store eight destination rows.

use crate::dsp::bounds::strided_len;

use core::arch::aarch64::*;

use crate::transforms::TxType;
use crate::transforms::constants::{
    COSPI_2_64, COSPI_4_64, COSPI_6_64, COSPI_8_64, COSPI_10_64, COSPI_12_64, COSPI_14_64,
    COSPI_16_64, COSPI_18_64, COSPI_20_64, COSPI_22_64, COSPI_24_64, COSPI_26_64, COSPI_28_64,
    COSPI_30_64,
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

/// Round two pairs of four-lane products with VP9's DCT constant shift.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn round_shift_pair(lo: int32x4_t, hi: int32x4_t) -> int16x8_t {
    vcombine_s16(vrshrn_n_s32::<14>(lo), vrshrn_n_s32::<14>(hi))
}

/// Compute the two signed ADST butterflies used by one stage.
///
/// The products stay in 32-bit lanes until after the add/subtract. This is
/// important for the full-range `tran_low_t` inputs: adding the i16 inputs
/// before multiplication would overflow and differ from libvpx's NEON path.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn iadst_butterfly(
    in0: int16x8_t,
    in1: int16x8_t,
    c0: i16,
    c1: i16,
) -> ([int32x4_t; 2], [int32x4_t; 2]) {
    let in0_lo = vget_low_s16(in0);
    let in0_hi = vget_high_s16(in0);
    let in1_lo = vget_low_s16(in1);
    let in1_hi = vget_high_s16(in1);

    let mut sum = [vmull_n_s16(in0_lo, c0), vmull_n_s16(in0_hi, c0)];
    sum[0] = vmlal_n_s16(sum[0], in1_lo, c1);
    sum[1] = vmlal_n_s16(sum[1], in1_hi, c1);

    let mut diff = [vmull_n_s16(in0_lo, c1), vmull_n_s16(in0_hi, c1)];
    diff[0] = vmlsl_n_s16(diff[0], in1_lo, c0);
    diff[1] = vmlsl_n_s16(diff[1], in1_hi, c0);

    (sum, diff)
}

/// Add or subtract two product pairs, then apply the DCT constant shift.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn round_shift_add(a: [int32x4_t; 2], b: [int32x4_t; 2], subtract: bool) -> int16x8_t {
    let (lo, hi) = if subtract {
        (vsubq_s32(a[0], b[0]), vsubq_s32(a[1], b[1]))
    } else {
        (vaddq_s32(a[0], b[0]), vaddq_s32(a[1], b[1]))
    };
    round_shift_pair(lo, hi)
}

/// One 8-point inverse ADST for eight independent lanes.
///
/// This is a direct NEON port of libvpx `vp9_iadst8` in
/// `vp9/common/arm/neon/vp9_iht_neon.h`. Every narrowing point matches the
/// scalar reference's `wraplow(dct_const_round_shift(...))` operation.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn iadst8(io: &mut [int16x8_t; 8]) {
    let input = *io;
    let x0 = input[7];
    let x1 = input[0];
    let x2 = input[5];
    let x3 = input[2];
    let x4 = input[3];
    let x5 = input[4];
    let x6 = input[1];
    let x7 = input[6];

    // Stage 1.
    let (s0, s1) = iadst_butterfly(x0, x1, COSPI_2_64 as i16, COSPI_30_64 as i16);
    let (s2, s3) = iadst_butterfly(x2, x3, COSPI_10_64 as i16, COSPI_22_64 as i16);
    let (s4, s5) = iadst_butterfly(x4, x5, COSPI_18_64 as i16, COSPI_14_64 as i16);
    let (s6, s7) = iadst_butterfly(x6, x7, COSPI_26_64 as i16, COSPI_6_64 as i16);

    let x0 = round_shift_add(s0, s4, false);
    let x1 = round_shift_add(s1, s5, false);
    let x2 = round_shift_add(s2, s6, false);
    let x3 = round_shift_add(s3, s7, false);
    let x4 = round_shift_add(s0, s4, true);
    let x5 = round_shift_add(s1, s5, true);
    let x6 = round_shift_add(s2, s6, true);
    let x7 = round_shift_add(s3, s7, true);

    // Stage 2. The first four lanes are plain i16 wraplow add/subtracts;
    // the latter four retain 32-bit products until their rounded narrowing.
    let t0 = x0;
    let t1 = x1;
    let t2 = x2;
    let t3 = x3;
    let (s4, s5) = iadst_butterfly(x4, x5, COSPI_8_64 as i16, COSPI_24_64 as i16);
    let (s7, s6) = iadst_butterfly(x7, x6, COSPI_24_64 as i16, COSPI_8_64 as i16);

    let x0 = vaddq_s16(t0, t2);
    let x1 = vaddq_s16(t1, t3);
    let x2 = vsubq_s16(t0, t2);
    let x3 = vsubq_s16(t1, t3);
    let x4 = round_shift_add(s4, s6, false);
    let x5 = round_shift_add(s5, s7, false);
    let x6 = round_shift_add(s4, s6, true);
    let x7 = round_shift_add(s5, s7, true);

    // Stage 3.
    let (x2, x3) = iadst_half_butterfly(x2, x3, COSPI_16_64 as i16);
    let (x6, x7) = iadst_half_butterfly(x6, x7, COSPI_16_64 as i16);

    *io = [
        x0,
        vnegq_s16(x4),
        x6,
        vnegq_s16(x2),
        x3,
        vnegq_s16(x7),
        x5,
        vnegq_s16(x1),
    ];
}

/// Final ADST butterfly, retaining 32-bit products until the rounded narrow.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn iadst_half_butterfly(a: int16x8_t, b: int16x8_t, c: i16) -> (int16x8_t, int16x8_t) {
    let a_lo = vmull_n_s16(vget_low_s16(a), c);
    let a_hi = vmull_n_s16(vget_high_s16(a), c);
    let b_lo = vmull_n_s16(vget_low_s16(b), c);
    let b_hi = vmull_n_s16(vget_high_s16(b), c);
    (
        round_shift_pair(vaddq_s32(a_lo, b_lo), vaddq_s32(a_hi, b_hi)),
        round_shift_pair(vsubq_s32(a_lo, b_lo), vsubq_s32(a_hi, b_hi)),
    )
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

#[target_feature(enable = "neon")]
unsafe fn idct8x8_12_add_neon_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    let zero = vdupq_n_s16(0);
    let mut io = [zero; 8];
    for (row, slot) in io[..4].iter_mut().enumerate() {
        *slot = load_i32_as_i16(input.as_ptr().add(row * 8));
    }

    // `eob <= 12` confines all nonzero scan positions to the first four
    // coefficient rows. Keeping rows 4..7 zero matches libvpx's sparse
    // caller contract while reusing the bit-exact dense vector stages.
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
    assert!(
        dst.len() >= strided_len(7, stride, 8),
        "dst too small for 8x8 add"
    );

    // SAFETY: `SimdDispatch::build` installs this wrapper only when runtime
    // capability detection reports NEON. AArch64 mandates NEON in practice;
    // direct tests also gate on `is_aarch64_feature_detected!("neon")`.
    unsafe { idct8x8_add_neon_inner(input, dst, stride) }
}

/// Add an `eob <= 12` sparse 8x8 inverse-DCT residual to an 8-bit block.
///
/// Coefficients in input rows 4..7 are deliberately ignored, matching
/// [`crate::transforms::inverse::idct8x8_12_add`] and libvpx's sparse kernel.
pub fn idct8x8_12_add_neon(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    assert!(stride >= 8, "stride too small for 8x8 add");
    assert!(
        dst.len() >= strided_len(7, stride, 8),
        "dst too small for 8x8 add"
    );

    // SAFETY: dispatch installs this wrapper only on AArch64 NEON targets.
    unsafe { idct8x8_12_add_neon_inner(input, dst, stride) }
}

/// Add a dense 8×8 inverse hybrid-transform residual to an 8-bit prediction.
///
/// This is the AArch64 counterpart of libvpx
/// `vp9_iht8x8_64_add_neon`. The `tx_type` selects DCT or ADST independently
/// for the two transform passes.
pub fn iht8x8_add_neon(input: &[i32; 64], dst: &mut [u8], stride: usize, tx_type: TxType) {
    assert!(stride >= 8, "stride too small for 8x8 add");
    assert!(
        dst.len() >= strided_len(7, stride, 8),
        "dst too small for 8x8 add"
    );

    // SAFETY: `SimdDispatch::build` installs this wrapper only when runtime
    // capability detection reports NEON. AArch64 mandates NEON in practice;
    // direct tests also gate on `is_aarch64_feature_detected!("neon")`.
    unsafe { iht8x8_add_neon_inner(input, dst, stride, tx_type) }
}

#[target_feature(enable = "neon")]
unsafe fn iht8x8_add_neon_inner(input: &[i32; 64], dst: &mut [u8], stride: usize, tx_type: TxType) {
    let zero = vdupq_n_s16(0);
    let mut io = [zero; 8];
    for (row, slot) in io.iter_mut().enumerate() {
        *slot = load_i32_as_i16(input.as_ptr().add(row * 8));
    }

    // The input transpose turns rows into independent transform lanes, as in
    // libvpx `vp9_iht8x8_64_add_neon`. The second transpose restores rows for
    // the column pass.
    transpose_8x8(&mut io);
    match tx_type {
        TxType::DctDct => {
            idct8(&mut io);
            transpose_8x8(&mut io);
            idct8(&mut io);
        }
        TxType::AdstDct => {
            idct8(&mut io);
            transpose_8x8(&mut io);
            iadst8(&mut io);
        }
        TxType::DctAdst => {
            iadst8(&mut io);
            transpose_8x8(&mut io);
            idct8(&mut io);
        }
        TxType::AdstAdst => {
            iadst8(&mut io);
            transpose_8x8(&mut io);
            iadst8(&mut io);
        }
    }

    for (row, residual) in io.into_iter().enumerate() {
        add_and_store_row(residual, dst.as_mut_ptr().add(row * stride));
    }
}

#[cfg(all(test, target_arch = "aarch64", not(miri)))]
mod tests {
    use super::*;
    use crate::transforms::inverse::{idct8x8_12_add, idct8x8_64_add, iht8x8_64_add};
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

    fn compare_12(input: &[i32; 64], stride: usize, skew: usize) {
        let len = skew + 7 * stride + 8;
        let mut neon = vec![0u8; len + 16];
        for (i, pixel) in neon.iter_mut().enumerate() {
            *pixel = (i as u8).wrapping_mul(41).wrapping_add(17);
        }
        let mut scalar = neon.clone();
        idct8x8_12_add_neon(input, &mut neon[skew..len], stride);
        idct8x8_12_add(input, &mut scalar[skew..len], stride);
        assert_eq!(neon, scalar, "sparse stride={stride} skew={skew}");
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

    #[test]
    fn idct8x8_12_neon_matches_scalar_and_ignores_tail_rows() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        let mut state = 0xa076_1d64_78bd_642fu64;
        for case in 0..512 {
            let mut input = [0i32; 64];
            for coefficient in &mut input[..32] {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *coefficient = ((state >> 17) as i32 & 0xfff) - 2048;
            }
            // The sparse contract ignores these rows; poison them so the test
            // catches an accidental dense load or dependence on scratch tail.
            for (index, coefficient) in input[32..].iter_mut().enumerate() {
                *coefficient = if index & 1 == 0 {
                    i16::MAX as i32
                } else {
                    i16::MIN as i32
                };
            }
            compare_12(&input, 8 + case % 19, case % 16);
        }
    }

    fn compare_iht(input: &[i32; 64], stride: usize, skew: usize, tx_type: TxType) {
        let len = skew + 7 * stride + 8;
        let mut neon = vec![0u8; len + 16];
        for (i, pixel) in neon.iter_mut().enumerate() {
            *pixel = (i as u8).wrapping_mul(53).wrapping_add(19);
        }
        let mut scalar = neon.clone();
        iht8x8_add_neon(input, &mut neon[skew..len], stride, tx_type);
        iht8x8_64_add(input, &mut scalar[skew..len], stride, tx_type);
        assert_eq!(
            neon, scalar,
            "stride={stride} skew={skew} tx_type={tx_type:?}"
        );
    }

    #[test]
    fn iht8x8_neon_all_tx_types_and_eob64_match_scalar() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        let tx_types = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut input = [0i32; 64];
        for coefficient in &mut input {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *coefficient = ((state >> 17) as i32 & 0xfff) - 2048;
        }
        // Keep the highest raster coefficient non-zero so this is a dense
        // eob=64 fixture rather than an accidental sparse case.
        input[63] = 173;
        for &tx_type in &tx_types {
            compare_iht(&input, 8, 0, tx_type);
            compare_iht(&input, 17, 7, tx_type);
        }
    }

    #[test]
    fn iht8x8_neon_extreme_inputs_match_scalar() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        let tx_types = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ];
        for &value in &[i16::MAX as i32, i16::MIN as i32] {
            let input = [value; 64];
            for &tx_type in &tx_types {
                compare_iht(&input, 8, 0, tx_type);
            }
        }

        let mut input = [0i32; 64];
        for (index, coefficient) in input.iter_mut().enumerate() {
            *coefficient = if index & 1 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        for &tx_type in &tx_types {
            compare_iht(&input, 23, 11, tx_type);
        }
    }

    #[test]
    fn iht8x8_neon_random_valid_inputs_match_scalar() {
        if !is_aarch64_feature_detected!("neon") {
            return;
        }

        let tx_types = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ];
        let mut state = 0xd1b5_4a32_9c7e_0183u64;
        for case in 0..256 {
            let mut input = [0i32; 64];
            for coefficient in &mut input {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *coefficient = ((state >> 17) as i32 & 0x1fff) - 4096;
            }
            for &tx_type in &tx_types {
                compare_iht(&input, 8 + case % 19, case % 16, tx_type);
            }
        }
    }
}
