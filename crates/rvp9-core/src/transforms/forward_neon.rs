//! AArch64 NEON 32x32 forward DCT.
//!
//! The scalar VP9 transform is applied to four independent rows or columns
//! in parallel. Products widen to i64 before the exact `+8192 >> 14`
//! reduction, preserving the scalar transform's valid-residual result.

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
unsafe fn mul_round(a: V, constant: i32) -> V {
    let bias = vdupq_n_s64(1 << 13);
    let lo = vaddq_s64(vmull_n_s32(vget_low_s32(a), constant), bias);
    let hi = vaddq_s64(vmull_n_s32(vget_high_s32(a), constant), bias);
    vcombine_s32(vshrn_n_s64::<14>(lo), vshrn_n_s64::<14>(hi))
}

#[inline(always)]
unsafe fn mul_add_round(a: V, ac: i32, b: V, bc: i32) -> V {
    let bias = vdupq_n_s64(1 << 13);
    let lo = vaddq_s64(
        vmlal_n_s32(vmull_n_s32(vget_low_s32(a), ac), vget_low_s32(b), bc),
        bias,
    );
    let hi = vaddq_s64(
        vmlal_n_s32(vmull_n_s32(vget_high_s32(a), ac), vget_high_s32(b), bc),
        bias,
    );
    vcombine_s32(vshrn_n_s64::<14>(lo), vshrn_n_s64::<14>(hi))
}

/// Four independent, non-RD 1-D 32-point DCTs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct32x4(input: &[V; 32], output: &mut [V; 32]) {
    let zero = vdupq_n_s32(0);
    let mut step = [zero; 32];
    let mut out = [zero; 32];

    // Stage 1.
    for k in 0..16 {
        step[k] = add(input[k], input[31 - k]);
    }
    for k in 16..32 {
        step[k] = sub(input[31 - k], input[k]);
    }

    // Stage 2.
    for k in 0..8 {
        out[k] = add(step[k], step[15 - k]);
    }
    for k in 8..16 {
        out[k] = sub(step[15 - k], step[k]);
    }
    out[16] = step[16];
    out[17] = step[17];
    out[18] = step[18];
    out[19] = step[19];
    out[20] = mul_round(sub(step[27], step[20]), COSPI_16_64);
    out[21] = mul_round(sub(step[26], step[21]), COSPI_16_64);
    out[22] = mul_round(sub(step[25], step[22]), COSPI_16_64);
    out[23] = mul_round(sub(step[24], step[23]), COSPI_16_64);
    out[24] = mul_round(add(step[24], step[23]), COSPI_16_64);
    out[25] = mul_round(add(step[25], step[22]), COSPI_16_64);
    out[26] = mul_round(add(step[26], step[21]), COSPI_16_64);
    out[27] = mul_round(add(step[27], step[20]), COSPI_16_64);
    out[28] = step[28];
    out[29] = step[29];
    out[30] = step[30];
    out[31] = step[31];

    // Stage 3.
    step[0] = add(out[0], out[7]);
    step[1] = add(out[1], out[6]);
    step[2] = add(out[2], out[5]);
    step[3] = add(out[3], out[4]);
    step[4] = sub(out[3], out[4]);
    step[5] = sub(out[2], out[5]);
    step[6] = sub(out[1], out[6]);
    step[7] = sub(out[0], out[7]);
    step[8] = out[8];
    step[9] = out[9];
    step[10] = mul_round(sub(out[13], out[10]), COSPI_16_64);
    step[11] = mul_round(sub(out[12], out[11]), COSPI_16_64);
    step[12] = mul_round(add(out[12], out[11]), COSPI_16_64);
    step[13] = mul_round(add(out[13], out[10]), COSPI_16_64);
    step[14] = out[14];
    step[15] = out[15];
    step[16] = add(out[16], out[23]);
    step[17] = add(out[17], out[22]);
    step[18] = add(out[18], out[21]);
    step[19] = add(out[19], out[20]);
    step[20] = sub(out[19], out[20]);
    step[21] = sub(out[18], out[21]);
    step[22] = sub(out[17], out[22]);
    step[23] = sub(out[16], out[23]);
    step[24] = sub(out[31], out[24]);
    step[25] = sub(out[30], out[25]);
    step[26] = sub(out[29], out[26]);
    step[27] = sub(out[28], out[27]);
    step[28] = add(out[28], out[27]);
    step[29] = add(out[29], out[26]);
    step[30] = add(out[30], out[25]);
    step[31] = add(out[31], out[24]);

    // Stage 4.
    out[0] = add(step[0], step[3]);
    out[1] = add(step[1], step[2]);
    out[2] = sub(step[1], step[2]);
    out[3] = sub(step[0], step[3]);
    out[4] = step[4];
    out[5] = mul_round(sub(step[6], step[5]), COSPI_16_64);
    out[6] = mul_round(add(step[6], step[5]), COSPI_16_64);
    out[7] = step[7];
    out[8] = add(step[8], step[11]);
    out[9] = add(step[9], step[10]);
    out[10] = sub(step[9], step[10]);
    out[11] = sub(step[8], step[11]);
    out[12] = sub(step[15], step[12]);
    out[13] = sub(step[14], step[13]);
    out[14] = add(step[14], step[13]);
    out[15] = add(step[15], step[12]);
    out[16] = step[16];
    out[17] = step[17];
    out[18] = mul_add_round(step[18], -COSPI_8_64, step[29], COSPI_24_64);
    out[19] = mul_add_round(step[19], -COSPI_8_64, step[28], COSPI_24_64);
    out[20] = mul_add_round(step[20], -COSPI_24_64, step[27], -COSPI_8_64);
    out[21] = mul_add_round(step[21], -COSPI_24_64, step[26], -COSPI_8_64);
    out[22] = step[22];
    out[23] = step[23];
    out[24] = step[24];
    out[25] = step[25];
    out[26] = mul_add_round(step[26], COSPI_24_64, step[21], -COSPI_8_64);
    out[27] = mul_add_round(step[27], COSPI_24_64, step[20], -COSPI_8_64);
    out[28] = mul_add_round(step[28], COSPI_8_64, step[19], COSPI_24_64);
    out[29] = mul_add_round(step[29], COSPI_8_64, step[18], COSPI_24_64);
    out[30] = step[30];
    out[31] = step[31];

    // Stage 5.
    step[0] = mul_round(add(out[0], out[1]), COSPI_16_64);
    step[1] = mul_round(sub(out[0], out[1]), COSPI_16_64);
    step[2] = mul_add_round(out[2], COSPI_24_64, out[3], COSPI_8_64);
    step[3] = mul_add_round(out[3], COSPI_24_64, out[2], -COSPI_8_64);
    step[4] = add(out[4], out[5]);
    step[5] = sub(out[4], out[5]);
    step[6] = sub(out[7], out[6]);
    step[7] = add(out[7], out[6]);
    step[8] = out[8];
    step[9] = mul_add_round(out[9], -COSPI_8_64, out[14], COSPI_24_64);
    step[10] = mul_add_round(out[10], -COSPI_24_64, out[13], -COSPI_8_64);
    step[11] = out[11];
    step[12] = out[12];
    step[13] = mul_add_round(out[13], COSPI_24_64, out[10], -COSPI_8_64);
    step[14] = mul_add_round(out[14], COSPI_8_64, out[9], COSPI_24_64);
    step[15] = out[15];
    step[16] = add(out[16], out[19]);
    step[17] = add(out[17], out[18]);
    step[18] = sub(out[17], out[18]);
    step[19] = sub(out[16], out[19]);
    step[20] = sub(out[23], out[20]);
    step[21] = sub(out[22], out[21]);
    step[22] = add(out[22], out[21]);
    step[23] = add(out[23], out[20]);
    step[24] = add(out[24], out[27]);
    step[25] = add(out[25], out[26]);
    step[26] = sub(out[25], out[26]);
    step[27] = sub(out[24], out[27]);
    step[28] = sub(out[31], out[28]);
    step[29] = sub(out[30], out[29]);
    step[30] = add(out[30], out[29]);
    step[31] = add(out[31], out[28]);

    // Stage 6.
    out[0] = step[0];
    out[1] = step[1];
    out[2] = step[2];
    out[3] = step[3];
    out[4] = mul_add_round(step[4], COSPI_28_64, step[7], COSPI_4_64);
    out[5] = mul_add_round(step[5], COSPI_12_64, step[6], COSPI_20_64);
    out[6] = mul_add_round(step[6], COSPI_12_64, step[5], -COSPI_20_64);
    out[7] = mul_add_round(step[7], COSPI_28_64, step[4], -COSPI_4_64);
    out[8] = add(step[8], step[9]);
    out[9] = sub(step[8], step[9]);
    out[10] = sub(step[11], step[10]);
    out[11] = add(step[11], step[10]);
    out[12] = add(step[12], step[13]);
    out[13] = sub(step[12], step[13]);
    out[14] = sub(step[15], step[14]);
    out[15] = add(step[15], step[14]);
    out[16] = step[16];
    out[17] = mul_add_round(step[17], -COSPI_4_64, step[30], COSPI_28_64);
    out[18] = mul_add_round(step[18], -COSPI_28_64, step[29], -COSPI_4_64);
    out[19] = step[19];
    out[20] = step[20];
    out[21] = mul_add_round(step[21], -COSPI_20_64, step[26], COSPI_12_64);
    out[22] = mul_add_round(step[22], -COSPI_12_64, step[25], -COSPI_20_64);
    out[23] = step[23];
    out[24] = step[24];
    out[25] = mul_add_round(step[25], COSPI_12_64, step[22], -COSPI_20_64);
    out[26] = mul_add_round(step[26], COSPI_20_64, step[21], COSPI_12_64);
    out[27] = step[27];
    out[28] = step[28];
    out[29] = mul_add_round(step[29], COSPI_28_64, step[18], -COSPI_4_64);
    out[30] = mul_add_round(step[30], COSPI_4_64, step[17], COSPI_28_64);
    out[31] = step[31];

    // Stage 7.
    step[0] = out[0];
    step[1] = out[1];
    step[2] = out[2];
    step[3] = out[3];
    step[4] = out[4];
    step[5] = out[5];
    step[6] = out[6];
    step[7] = out[7];
    step[8] = mul_add_round(out[8], COSPI_30_64, out[15], COSPI_2_64);
    step[9] = mul_add_round(out[9], COSPI_14_64, out[14], COSPI_18_64);
    step[10] = mul_add_round(out[10], COSPI_22_64, out[13], COSPI_10_64);
    step[11] = mul_add_round(out[11], COSPI_6_64, out[12], COSPI_26_64);
    step[12] = mul_add_round(out[12], COSPI_6_64, out[11], -COSPI_26_64);
    step[13] = mul_add_round(out[13], COSPI_22_64, out[10], -COSPI_10_64);
    step[14] = mul_add_round(out[14], COSPI_14_64, out[9], -COSPI_18_64);
    step[15] = mul_add_round(out[15], COSPI_30_64, out[8], -COSPI_2_64);
    step[16] = add(out[16], out[17]);
    step[17] = sub(out[16], out[17]);
    step[18] = sub(out[19], out[18]);
    step[19] = add(out[19], out[18]);
    step[20] = add(out[20], out[21]);
    step[21] = sub(out[20], out[21]);
    step[22] = sub(out[23], out[22]);
    step[23] = add(out[23], out[22]);
    step[24] = add(out[24], out[25]);
    step[25] = sub(out[24], out[25]);
    step[26] = sub(out[27], out[26]);
    step[27] = add(out[27], out[26]);
    step[28] = add(out[28], out[29]);
    step[29] = sub(out[28], out[29]);
    step[30] = sub(out[31], out[30]);
    step[31] = add(out[31], out[30]);

    // Final bit-reversed permutation.
    output[0] = step[0];
    output[16] = step[1];
    output[8] = step[2];
    output[24] = step[3];
    output[4] = step[4];
    output[20] = step[5];
    output[12] = step[6];
    output[28] = step[7];
    output[2] = step[8];
    output[18] = step[9];
    output[10] = step[10];
    output[26] = step[11];
    output[6] = step[12];
    output[22] = step[13];
    output[14] = step[14];
    output[30] = step[15];
    output[1] = mul_add_round(step[16], COSPI_31_64, step[31], COSPI_1_64);
    output[17] = mul_add_round(step[17], COSPI_15_64, step[30], COSPI_17_64);
    output[9] = mul_add_round(step[18], COSPI_23_64, step[29], COSPI_9_64);
    output[25] = mul_add_round(step[19], COSPI_7_64, step[28], COSPI_25_64);
    output[5] = mul_add_round(step[20], COSPI_27_64, step[27], COSPI_5_64);
    output[21] = mul_add_round(step[21], COSPI_11_64, step[26], COSPI_21_64);
    output[13] = mul_add_round(step[22], COSPI_19_64, step[25], COSPI_13_64);
    output[29] = mul_add_round(step[23], COSPI_3_64, step[24], COSPI_29_64);
    output[3] = mul_add_round(step[24], COSPI_3_64, step[23], -COSPI_29_64);
    output[19] = mul_add_round(step[25], COSPI_19_64, step[22], -COSPI_13_64);
    output[11] = mul_add_round(step[26], COSPI_11_64, step[21], -COSPI_21_64);
    output[27] = mul_add_round(step[27], COSPI_27_64, step[20], -COSPI_5_64);
    output[7] = mul_add_round(step[28], COSPI_7_64, step[19], -COSPI_25_64);
    output[23] = mul_add_round(step[29], COSPI_23_64, step[18], -COSPI_9_64);
    output[15] = mul_add_round(step[30], COSPI_15_64, step[17], -COSPI_17_64);
    output[31] = mul_add_round(step[31], COSPI_31_64, step[16], -COSPI_1_64);
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

#[inline(always)]
unsafe fn pass0_round(value: V) -> V {
    let one = vdupq_n_s32(1);
    let positive = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vcgtq_s32(value, vdupq_n_s32(0))));
    vshrq_n_s32::<2>(add(add(value, one), positive))
}

#[inline(always)]
unsafe fn pass1_round_and_wrap(value: V) -> V {
    let one = vdupq_n_s32(1);
    let negative = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vcltq_s32(value, vdupq_n_s32(0))));
    let rounded = vshrq_n_s32::<2>(add(add(value, one), negative));
    vmovl_s16(vmovn_s32(rounded))
}

#[target_feature(enable = "neon")]
unsafe fn fdct32x32_inner(input: *const i16, output: *mut i32, stride: usize) {
    let zero = vdupq_n_s32(0);
    let mut intermediate = [0i32; 1024];

    // First pass: four source columns per vector lane.
    for column in (0..32).step_by(4) {
        let mut lane_in = [zero; 32];
        for (row, values) in lane_in.iter_mut().enumerate() {
            *values = vshlq_n_s32::<2>(vmovl_s16(vld1_s16(input.add(row * stride + column))));
        }
        let mut lane_out = [zero; 32];
        fdct32x4(&lane_in, &mut lane_out);
        for (coefficient, &values) in lane_out.iter().enumerate() {
            vst1q_s32(
                intermediate.as_mut_ptr().add(coefficient * 32 + column),
                pass0_round(values),
            );
        }
    }

    // Second pass: transpose four intermediate rows into four independent
    // transforms, then transpose each four-coefficient output group back.
    for row in (0..32).step_by(4) {
        let mut lane_in = [zero; 32];
        for column in (0..32).step_by(4) {
            let rows = transpose4(
                vld1q_s32(intermediate.as_ptr().add(row * 32 + column)),
                vld1q_s32(intermediate.as_ptr().add((row + 1) * 32 + column)),
                vld1q_s32(intermediate.as_ptr().add((row + 2) * 32 + column)),
                vld1q_s32(intermediate.as_ptr().add((row + 3) * 32 + column)),
            );
            lane_in[column..column + 4].copy_from_slice(&rows);
        }
        let mut lane_out = [zero; 32];
        fdct32x4(&lane_in, &mut lane_out);
        for coefficient in (0..32).step_by(4) {
            let rows = transpose4(
                pass1_round_and_wrap(lane_out[coefficient]),
                pass1_round_and_wrap(lane_out[coefficient + 1]),
                pass1_round_and_wrap(lane_out[coefficient + 2]),
                pass1_round_and_wrap(lane_out[coefficient + 3]),
            );
            for (lane, &values) in rows.iter().enumerate() {
                vst1q_s32(output.add((row + lane) * 32 + coefficient), values);
            }
        }
    }
}

pub(super) fn fdct32x32(input: &[i16], output: &mut [i32; 1024], stride: usize) {
    assert!(stride >= 32);
    assert!(input.len() >= 31 * stride + 32);
    // SAFETY: The bounds above prove every 4-value input load. `output`
    // provides all 1024 destinations, and NEON is mandatory on AArch64.
    unsafe { fdct32x32_inner(input.as_ptr(), output.as_mut_ptr(), stride) };
}
