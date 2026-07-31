//! AArch64 NEON 16x16 inverse DCT.
//!
//! The scalar VP9 transform is applied to four independent rows or columns
//! in parallel. Products widen to i64 before the exact `+8192 >> 14`
//! reduction, and every butterfly store wraps through i16.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;

use crate::transforms::constants::*;

type V = int32x4_t;

#[inline(always)]
unsafe fn add(a: V, b: V) -> V {
    vaddq_s32(a, b)
}

#[inline(always)]
unsafe fn sub(a: V, b: V) -> V {
    vsubq_s32(a, b)
}

#[inline(always)]
unsafe fn wrap16(value: V) -> V {
    vmovl_s16(vmovn_s32(value))
}

#[inline(always)]
unsafe fn wrap_add(a: V, b: V) -> V {
    wrap16(add(a, b))
}

#[inline(always)]
unsafe fn wrap_sub(a: V, b: V) -> V {
    wrap16(sub(a, b))
}

#[inline(always)]
unsafe fn mul_round_wrap(a: V, constant: i32) -> V {
    let bias = vdupq_n_s64(1 << 13);
    let lo = vaddq_s64(vmull_n_s32(vget_low_s32(a), constant), bias);
    let hi = vaddq_s64(vmull_n_s32(vget_high_s32(a), constant), bias);
    wrap16(vcombine_s32(vshrn_n_s64::<14>(lo), vshrn_n_s64::<14>(hi)))
}

#[inline(always)]
unsafe fn mul_add_round_wrap(a: V, ac: i32, b: V, bc: i32) -> V {
    let bias = vdupq_n_s64(1 << 13);
    let lo = vaddq_s64(
        vmlal_n_s32(vmull_n_s32(vget_low_s32(a), ac), vget_low_s32(b), bc),
        bias,
    );
    let hi = vaddq_s64(
        vmlal_n_s32(vmull_n_s32(vget_high_s32(a), ac), vget_high_s32(b), bc),
        bias,
    );
    wrap16(vcombine_s32(vshrn_n_s64::<14>(lo), vshrn_n_s64::<14>(hi)))
}

/// Four independent 1-D 16-point inverse DCTs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn idct16x4(input: &[V; 16], output: &mut [V; 16]) {
    let zero = vdupq_n_s32(0);
    let mut step1 = [zero; 16];
    let mut step2 = [zero; 16];

    // Stage 1.
    step1[0] = input[0];
    step1[1] = input[8];
    step1[2] = input[4];
    step1[3] = input[12];
    step1[4] = input[2];
    step1[5] = input[10];
    step1[6] = input[6];
    step1[7] = input[14];
    step1[8] = input[1];
    step1[9] = input[9];
    step1[10] = input[5];
    step1[11] = input[13];
    step1[12] = input[3];
    step1[13] = input[11];
    step1[14] = input[7];
    step1[15] = input[15];

    // Stage 2.
    step2[..8].copy_from_slice(&step1[..8]);
    step2[8] = mul_add_round_wrap(step1[8], COSPI_30_64, step1[15], -COSPI_2_64);
    step2[15] = mul_add_round_wrap(step1[8], COSPI_2_64, step1[15], COSPI_30_64);
    step2[9] = mul_add_round_wrap(step1[9], COSPI_14_64, step1[14], -COSPI_18_64);
    step2[14] = mul_add_round_wrap(step1[9], COSPI_18_64, step1[14], COSPI_14_64);
    step2[10] = mul_add_round_wrap(step1[10], COSPI_22_64, step1[13], -COSPI_10_64);
    step2[13] = mul_add_round_wrap(step1[10], COSPI_10_64, step1[13], COSPI_22_64);
    step2[11] = mul_add_round_wrap(step1[11], COSPI_6_64, step1[12], -COSPI_26_64);
    step2[12] = mul_add_round_wrap(step1[11], COSPI_26_64, step1[12], COSPI_6_64);

    // Stage 3.
    step1[..4].copy_from_slice(&step2[..4]);
    step1[4] = mul_add_round_wrap(step2[4], COSPI_28_64, step2[7], -COSPI_4_64);
    step1[7] = mul_add_round_wrap(step2[4], COSPI_4_64, step2[7], COSPI_28_64);
    step1[5] = mul_add_round_wrap(step2[5], COSPI_12_64, step2[6], -COSPI_20_64);
    step1[6] = mul_add_round_wrap(step2[5], COSPI_20_64, step2[6], COSPI_12_64);
    step1[8] = wrap_add(step2[8], step2[9]);
    step1[9] = wrap_sub(step2[8], step2[9]);
    step1[10] = wrap_sub(step2[11], step2[10]);
    step1[11] = wrap_add(step2[10], step2[11]);
    step1[12] = wrap_add(step2[12], step2[13]);
    step1[13] = wrap_sub(step2[12], step2[13]);
    step1[14] = wrap_sub(step2[15], step2[14]);
    step1[15] = wrap_add(step2[14], step2[15]);

    // Stage 4.
    step2[0] = mul_round_wrap(add(step1[0], step1[1]), COSPI_16_64);
    step2[1] = mul_round_wrap(sub(step1[0], step1[1]), COSPI_16_64);
    step2[2] = mul_add_round_wrap(step1[2], COSPI_24_64, step1[3], -COSPI_8_64);
    step2[3] = mul_add_round_wrap(step1[2], COSPI_8_64, step1[3], COSPI_24_64);
    step2[4] = wrap_add(step1[4], step1[5]);
    step2[5] = wrap_sub(step1[4], step1[5]);
    step2[6] = wrap_sub(step1[7], step1[6]);
    step2[7] = wrap_add(step1[6], step1[7]);
    step2[8] = step1[8];
    step2[9] = mul_add_round_wrap(step1[9], -COSPI_8_64, step1[14], COSPI_24_64);
    step2[14] = mul_add_round_wrap(step1[9], COSPI_24_64, step1[14], COSPI_8_64);
    step2[10] = mul_add_round_wrap(step1[10], -COSPI_24_64, step1[13], -COSPI_8_64);
    step2[13] = mul_add_round_wrap(step1[10], -COSPI_8_64, step1[13], COSPI_24_64);
    step2[11] = step1[11];
    step2[12] = step1[12];
    step2[15] = step1[15];

    // Stage 5.
    step1[0] = wrap_add(step2[0], step2[3]);
    step1[1] = wrap_add(step2[1], step2[2]);
    step1[2] = wrap_sub(step2[1], step2[2]);
    step1[3] = wrap_sub(step2[0], step2[3]);
    step1[4] = step2[4];
    step1[5] = mul_round_wrap(sub(step2[6], step2[5]), COSPI_16_64);
    step1[6] = mul_round_wrap(add(step2[5], step2[6]), COSPI_16_64);
    step1[7] = step2[7];
    step1[8] = wrap_add(step2[8], step2[11]);
    step1[9] = wrap_add(step2[9], step2[10]);
    step1[10] = wrap_sub(step2[9], step2[10]);
    step1[11] = wrap_sub(step2[8], step2[11]);
    step1[12] = wrap_sub(step2[15], step2[12]);
    step1[13] = wrap_sub(step2[14], step2[13]);
    step1[14] = wrap_add(step2[13], step2[14]);
    step1[15] = wrap_add(step2[12], step2[15]);

    // Stage 6.
    step2[0] = wrap_add(step1[0], step1[7]);
    step2[1] = wrap_add(step1[1], step1[6]);
    step2[2] = wrap_add(step1[2], step1[5]);
    step2[3] = wrap_add(step1[3], step1[4]);
    step2[4] = wrap_sub(step1[3], step1[4]);
    step2[5] = wrap_sub(step1[2], step1[5]);
    step2[6] = wrap_sub(step1[1], step1[6]);
    step2[7] = wrap_sub(step1[0], step1[7]);
    step2[8] = step1[8];
    step2[9] = step1[9];
    step2[10] = mul_round_wrap(sub(step1[13], step1[10]), COSPI_16_64);
    step2[13] = mul_round_wrap(add(step1[10], step1[13]), COSPI_16_64);
    step2[11] = mul_round_wrap(sub(step1[12], step1[11]), COSPI_16_64);
    step2[12] = mul_round_wrap(add(step1[11], step1[12]), COSPI_16_64);
    step2[14] = step1[14];
    step2[15] = step1[15];

    // Stage 7.
    output[0] = wrap_add(step2[0], step2[15]);
    output[1] = wrap_add(step2[1], step2[14]);
    output[2] = wrap_add(step2[2], step2[13]);
    output[3] = wrap_add(step2[3], step2[12]);
    output[4] = wrap_add(step2[4], step2[11]);
    output[5] = wrap_add(step2[5], step2[10]);
    output[6] = wrap_add(step2[6], step2[9]);
    output[7] = wrap_add(step2[7], step2[8]);
    output[8] = wrap_sub(step2[7], step2[8]);
    output[9] = wrap_sub(step2[6], step2[9]);
    output[10] = wrap_sub(step2[5], step2[10]);
    output[11] = wrap_sub(step2[4], step2[11]);
    output[12] = wrap_sub(step2[3], step2[12]);
    output[13] = wrap_sub(step2[2], step2[13]);
    output[14] = wrap_sub(step2[1], step2[14]);
    output[15] = wrap_sub(step2[0], step2[15]);
}

#[inline(always)]
unsafe fn transpose4(a: V, b: V, c: V, d: V) -> [V; 4] {
    let ab0 = vtrn1q_s32(a, b);
    let ab1 = vtrn2q_s32(a, b);
    let cd0 = vtrn1q_s32(c, d);
    let cd1 = vtrn2q_s32(c, d);
    [
        vcombine_s32(vget_low_s32(ab0), vget_low_s32(cd0)),
        vcombine_s32(vget_low_s32(ab1), vget_low_s32(cd1)),
        vcombine_s32(vget_high_s32(ab0), vget_high_s32(cd0)),
        vcombine_s32(vget_high_s32(ab1), vget_high_s32(cd1)),
    ]
}

#[target_feature(enable = "neon")]
unsafe fn idct16x16_add_inner(input: *const i32, dest: *mut u8, stride: usize, active_rows: usize) {
    let zero = vdupq_n_s32(0);
    let mut intermediate = [0i32; 256];

    // First pass: transpose four coefficient rows into four lane-wise
    // transforms. Rows outside the selected EOB bucket remain zero.
    for row in (0..active_rows).step_by(4) {
        let mut lane_in = [zero; 16];
        for column in (0..16).step_by(4) {
            let values = transpose4(
                vld1q_s32(input.add(row * 16 + column)),
                vld1q_s32(input.add((row + 1) * 16 + column)),
                vld1q_s32(input.add((row + 2) * 16 + column)),
                vld1q_s32(input.add((row + 3) * 16 + column)),
            );
            lane_in[column..column + 4].copy_from_slice(&values);
        }
        let mut lane_out = [zero; 16];
        idct16x4(&lane_in, &mut lane_out);
        for coefficient in (0..16).step_by(4) {
            let rows = transpose4(
                lane_out[coefficient],
                lane_out[coefficient + 1],
                lane_out[coefficient + 2],
                lane_out[coefficient + 3],
            );
            for (lane, &values) in rows.iter().enumerate() {
                vst1q_s32(
                    intermediate
                        .as_mut_ptr()
                        .add((row + lane) * 16 + coefficient),
                    values,
                );
            }
        }
    }

    // Second pass: four columns per transform. The final pixel add is
    // scalar because each destination row contains only four adjacent
    // bytes here; the transform arithmetic remains four-way vectorized.
    let mut residual = [0i32; 4];
    for column in (0..16).step_by(4) {
        let mut lane_in = [zero; 16];
        for (row, values) in lane_in.iter_mut().enumerate() {
            *values = vld1q_s32(intermediate.as_ptr().add(row * 16 + column));
        }
        let mut lane_out = [zero; 16];
        idct16x4(&lane_in, &mut lane_out);
        for (row, &values) in lane_out.iter().enumerate() {
            let rounded = vshrq_n_s32::<6>(add(values, vdupq_n_s32(32)));
            vst1q_s32(residual.as_mut_ptr(), rounded);
            for (lane, &value) in residual.iter().enumerate() {
                let pixel = dest.add(row * stride + column + lane);
                *pixel = ((*pixel as i32) + value).clamp(0, 255) as u8;
            }
        }
    }
}

pub(super) fn idct16x16_add(
    input: &[i32; 256],
    dest: &mut [u8],
    stride: usize,
    active_rows: usize,
) {
    assert!(stride >= 16);
    assert!(dest.len() >= stride * 15 + 16);
    assert!(matches!(active_rows, 4 | 8 | 16));
    // SAFETY: The bounds above prove every four-pixel destination group.
    // `input` provides all coefficient loads, and NEON is mandatory on
    // AArch64.
    unsafe { idct16x16_add_inner(input.as_ptr(), dest.as_mut_ptr(), stride, active_rows) };
}
