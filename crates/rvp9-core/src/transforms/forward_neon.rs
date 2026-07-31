//! AArch64 NEON 16x16 and 32x32 forward DCTs.
//!
//! The 16x16 and 32x32 first passes apply the scalar VP9 schedules to eight
//! i16 lanes. Their second passes retain bounded opening stages before
//! widening to four i32 lanes ahead of overflow-sensitive work. Cosine
//! products widen one level before the exact `+8192 >> 14` reduction,
//! preserving the scalar transforms' valid-residual results.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;

use crate::transforms::constants::*;

type V = int32x4_t;
type V16 = int16x8_t;
type VPair = [V; 2];

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

#[inline(always)]
unsafe fn mul_add_round_fast32(a: V, ac: i32, b: V, bc: i32) -> V {
    vrshrq_n_s32::<14>(vmlaq_n_s32(vmulq_n_s32(a, ac), b, bc))
}

#[inline(always)]
unsafe fn wrap16(value: V) -> V {
    vmovl_s16(vmovn_s32(value))
}

#[inline(always)]
unsafe fn add16(a: V16, b: V16) -> V16 {
    vaddq_s16(a, b)
}

#[inline(always)]
unsafe fn sub16(a: V16, b: V16) -> V16 {
    vsubq_s16(a, b)
}

#[inline(always)]
unsafe fn mul_round16(a: V16, constant: i32) -> V16 {
    let constant = constant as i16;
    let bias = vdupq_n_s32(1 << 13);
    let lo = vaddq_s32(vmull_n_s16(vget_low_s16(a), constant), bias);
    let hi = vaddq_s32(vmull_high_n_s16(a, constant), bias);
    vcombine_s16(vshrn_n_s32::<14>(lo), vshrn_n_s32::<14>(hi))
}

#[inline(always)]
unsafe fn mul_add_round16(a: V16, ac: i32, b: V16, bc: i32) -> V16 {
    let ac = ac as i16;
    let bc = bc as i16;
    let bias = vdupq_n_s32(1 << 13);
    let lo = vaddq_s32(
        vmlal_n_s16(vmull_n_s16(vget_low_s16(a), ac), vget_low_s16(b), bc),
        bias,
    );
    let hi = vaddq_s32(vmlal_high_n_s16(vmull_high_n_s16(a, ac), b, bc), bias);
    vcombine_s16(vshrn_n_s32::<14>(lo), vshrn_n_s32::<14>(hi))
}

/// Width-generic 1-D 16-point DCT schedule.
#[inline(always)]
unsafe fn fdct16_body<T, Add, Sub, Mul, MulAdd, Wrap>(
    input: &[T; 16],
    output: &mut [T; 16],
    zero_wrap_and_input: (T, Wrap, bool),
    add: Add,
    sub: Sub,
    mul_round: Mul,
    mul_add_round: MulAdd,
) where
    T: Copy,
    Add: Fn(T, T) -> T,
    Sub: Fn(T, T) -> T,
    Mul: Fn(T, i32) -> T,
    MulAdd: Fn(T, i32, T, i32) -> T,
    Wrap: Fn(T) -> T,
{
    let (zero, wrap16, input_is_crossed) = zero_wrap_and_input;
    let mut in_half = [zero; 8];
    let mut step1 = [zero; 8];
    for index in 0..8 {
        if input_is_crossed {
            in_half[index] = input[index];
            step1[index] = input[index + 8];
        } else {
            in_half[index] = add(input[index], input[15 - index]);
            step1[index] = sub(input[7 - index], input[8 + index]);
        }
    }

    // Even coefficients: the fdct8 sub-block.
    let s0 = add(in_half[0], in_half[7]);
    let s1 = add(in_half[1], in_half[6]);
    let s2 = add(in_half[2], in_half[5]);
    let s3 = add(in_half[3], in_half[4]);
    let s4 = sub(in_half[3], in_half[4]);
    let s5 = sub(in_half[2], in_half[5]);
    let s6 = sub(in_half[1], in_half[6]);
    let s7 = sub(in_half[0], in_half[7]);

    let x0 = add(s0, s3);
    let x1 = add(s1, s2);
    let x2 = sub(s1, s2);
    let x3 = sub(s0, s3);
    output[0] = wrap16(mul_round(add(x0, x1), COSPI_16_64));
    output[4] = wrap16(mul_add_round(x3, COSPI_8_64, x2, COSPI_24_64));
    output[8] = wrap16(mul_round(sub(x0, x1), COSPI_16_64));
    output[12] = wrap16(mul_add_round(x3, COSPI_24_64, x2, -COSPI_8_64));

    let t2 = mul_round(sub(s6, s5), COSPI_16_64);
    let t3 = mul_round(add(s6, s5), COSPI_16_64);
    let x0 = add(s4, t2);
    let x1 = sub(s4, t2);
    let x2 = sub(s7, t3);
    let x3 = add(s7, t3);
    output[2] = wrap16(mul_add_round(x0, COSPI_28_64, x3, COSPI_4_64));
    output[6] = wrap16(mul_add_round(x2, COSPI_12_64, x1, -COSPI_20_64));
    output[10] = wrap16(mul_add_round(x1, COSPI_12_64, x2, COSPI_20_64));
    output[14] = wrap16(mul_add_round(x3, COSPI_28_64, x0, -COSPI_4_64));

    // Odd coefficients.
    let step2_2 = mul_round(sub(step1[5], step1[2]), COSPI_16_64);
    let step2_3 = mul_round(sub(step1[4], step1[3]), COSPI_16_64);
    let step2_4 = mul_round(add(step1[4], step1[3]), COSPI_16_64);
    let step2_5 = mul_round(add(step1[5], step1[2]), COSPI_16_64);

    let step3_0 = add(step1[0], step2_3);
    let step3_1 = add(step1[1], step2_2);
    let step3_2 = sub(step1[1], step2_2);
    let step3_3 = sub(step1[0], step2_3);
    let step3_4 = sub(step1[7], step2_4);
    let step3_5 = sub(step1[6], step2_5);
    let step3_6 = add(step1[6], step2_5);
    let step3_7 = add(step1[7], step2_4);

    let step2_1 = mul_add_round(step3_1, -COSPI_8_64, step3_6, COSPI_24_64);
    let step2_2 = mul_add_round(step3_2, COSPI_24_64, step3_5, COSPI_8_64);
    let step2_5 = mul_add_round(step3_2, COSPI_8_64, step3_5, -COSPI_24_64);
    let step2_6 = mul_add_round(step3_1, COSPI_24_64, step3_6, COSPI_8_64);

    let step1_0 = add(step3_0, step2_1);
    let step1_1 = sub(step3_0, step2_1);
    let step1_2 = add(step3_3, step2_2);
    let step1_3 = sub(step3_3, step2_2);
    let step1_4 = sub(step3_4, step2_5);
    let step1_5 = add(step3_4, step2_5);
    let step1_6 = sub(step3_7, step2_6);
    let step1_7 = add(step3_7, step2_6);

    output[1] = wrap16(mul_add_round(step1_0, COSPI_30_64, step1_7, COSPI_2_64));
    output[9] = wrap16(mul_add_round(step1_1, COSPI_14_64, step1_6, COSPI_18_64));
    output[5] = wrap16(mul_add_round(step1_2, COSPI_22_64, step1_5, COSPI_10_64));
    output[13] = wrap16(mul_add_round(step1_3, COSPI_6_64, step1_4, COSPI_26_64));
    output[3] = wrap16(mul_add_round(step1_3, -COSPI_26_64, step1_4, COSPI_6_64));
    output[11] = wrap16(mul_add_round(step1_2, -COSPI_10_64, step1_5, COSPI_22_64));
    output[7] = wrap16(mul_add_round(step1_1, -COSPI_18_64, step1_6, COSPI_14_64));
    output[15] = wrap16(mul_add_round(step1_0, -COSPI_2_64, step1_7, COSPI_30_64));
}

/// Four independent 1-D 16-point DCTs after their bounded cross-input stage.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct16x4_from_cross(input: &[V; 16], output: &mut [V; 16]) {
    fdct16_body(
        input,
        output,
        (vdupq_n_s32(0), |value| wrap16(value), true),
        |a, b| add(a, b),
        |a, b| sub(a, b),
        |a, constant| mul_round(a, constant),
        |a, ac, b, bc| mul_add_round(a, ac, b, bc),
    );
}

/// Eight independent bounded 1-D 16-point DCTs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct16x8(input: &[V16; 16], output: &mut [V16; 16]) {
    fdct16_body(
        input,
        output,
        (vdupq_n_s16(0), |value| value, false),
        |a, b| add16(a, b),
        |a, b| sub16(a, b),
        |a, constant| mul_round16(a, constant),
        |a, ac, b, bc| mul_add_round16(a, ac, b, bc),
    );
}

#[inline(always)]
unsafe fn fdct16_interpass_round16(value: V16) -> V16 {
    vshrq_n_s16::<2>(add16(value, vdupq_n_s16(1)))
}

/// Width-generic, non-RD 1-D 32-point DCT schedule.
///
/// The operation closures select either four i32 lanes for the overflow-safe
/// second pass or eight i16 lanes for the bounded first pass.
#[inline(always)]
unsafe fn fdct32_body<T, Add, Sub, Mul, MulAdd>(
    input: &[T; 32],
    output: &mut [T; 32],
    zero_and_stage2: (T, Option<&[T; 32]>),
    add: Add,
    sub: Sub,
    mul_round: Mul,
    mul_add_round: MulAdd,
) where
    T: Copy,
    Add: Fn(T, T) -> T,
    Sub: Fn(T, T) -> T,
    Mul: Fn(T, i32) -> T,
    MulAdd: Fn(T, i32, T, i32) -> T,
{
    let (zero, stage2_input) = zero_and_stage2;
    let mut step = [zero; 32];
    let mut out = [zero; 32];

    if let Some(stage2) = stage2_input {
        out.copy_from_slice(stage2);
    } else {
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
    }

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

/// Eight independent first-pass 1-D 32-point DCTs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct32x8_first_pass(input: &[V16; 32], output: &mut [V16; 32]) {
    fdct32_body(
        input,
        output,
        (vdupq_n_s16(0), None),
        |a, b| add16(a, b),
        |a, b| sub16(a, b),
        |a, constant| mul_round16(a, constant),
        |a, ac, b, bc| mul_add_round16(a, ac, b, bc),
    );
}

/// Eight overflow-safe second-pass transforms whose first two stages were
/// already evaluated in i16. The paired low/high vectors mirror libvpx's
/// `c_lo`/`c_hi` schedule so both four-lane groups advance together.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct32x8_from_stage2(input: &[VPair; 32], output: &mut [VPair; 32]) {
    let zero = vdupq_n_s32(0);
    fdct32_body(
        input,
        output,
        ([zero; 2], Some(input)),
        |a, b| [add(a[0], b[0]), add(a[1], b[1])],
        |a, b| [sub(a[0], b[0]), sub(a[1], b[1])],
        |a, constant| [mul_round(a[0], constant), mul_round(a[1], constant)],
        |a, ac, b, bc| {
            [
                mul_add_round_fast32(a[0], ac, b[0], bc),
                mul_add_round_fast32(a[1], ac, b[1], bc),
            ]
        },
    );
}

/// Evaluate stages 1 and 2 for eight second-pass transforms while their
/// values remain bounded to i16, matching libvpx's overflow boundary.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fdct32x8_second_pass_stage2(input: &[V16; 32], output: &mut [V16; 32]) {
    let zero = vdupq_n_s16(0);
    let mut step = [zero; 32];

    for k in 0..16 {
        step[k] = add16(input[k], input[31 - k]);
    }
    for k in 16..32 {
        step[k] = sub16(input[31 - k], input[k]);
    }

    for k in 0..8 {
        output[k] = add16(step[k], step[15 - k]);
    }
    for k in 8..16 {
        output[k] = sub16(step[15 - k], step[k]);
    }
    output[16] = step[16];
    output[17] = step[17];
    output[18] = step[18];
    output[19] = step[19];
    output[20] = mul_round16(sub16(step[27], step[20]), COSPI_16_64);
    output[21] = mul_round16(sub16(step[26], step[21]), COSPI_16_64);
    output[22] = mul_round16(sub16(step[25], step[22]), COSPI_16_64);
    output[23] = mul_round16(sub16(step[24], step[23]), COSPI_16_64);
    output[24] = mul_round16(add16(step[24], step[23]), COSPI_16_64);
    output[25] = mul_round16(add16(step[25], step[22]), COSPI_16_64);
    output[26] = mul_round16(add16(step[26], step[21]), COSPI_16_64);
    output[27] = mul_round16(add16(step[27], step[20]), COSPI_16_64);
    output[28] = step[28];
    output[29] = step[29];
    output[30] = step[30];
    output[31] = step[31];
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
unsafe fn transpose8(mut rows: [V16; 8]) -> [V16; 8] {
    let ab0 = vtrn1q_s16(rows[0], rows[1]);
    let ab1 = vtrn2q_s16(rows[0], rows[1]);
    let cd0 = vtrn1q_s16(rows[2], rows[3]);
    let cd1 = vtrn2q_s16(rows[2], rows[3]);
    let ef0 = vtrn1q_s16(rows[4], rows[5]);
    let ef1 = vtrn2q_s16(rows[4], rows[5]);
    let gh0 = vtrn1q_s16(rows[6], rows[7]);
    let gh1 = vtrn2q_s16(rows[6], rows[7]);

    let abcd0 = vtrn1q_s32(vreinterpretq_s32_s16(ab0), vreinterpretq_s32_s16(cd0));
    let abcd1 = vtrn1q_s32(vreinterpretq_s32_s16(ab1), vreinterpretq_s32_s16(cd1));
    let abcd2 = vtrn2q_s32(vreinterpretq_s32_s16(ab0), vreinterpretq_s32_s16(cd0));
    let abcd3 = vtrn2q_s32(vreinterpretq_s32_s16(ab1), vreinterpretq_s32_s16(cd1));
    let efgh0 = vtrn1q_s32(vreinterpretq_s32_s16(ef0), vreinterpretq_s32_s16(gh0));
    let efgh1 = vtrn1q_s32(vreinterpretq_s32_s16(ef1), vreinterpretq_s32_s16(gh1));
    let efgh2 = vtrn2q_s32(vreinterpretq_s32_s16(ef0), vreinterpretq_s32_s16(gh0));
    let efgh3 = vtrn2q_s32(vreinterpretq_s32_s16(ef1), vreinterpretq_s32_s16(gh1));

    rows[0] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(abcd0),
        vreinterpretq_s64_s32(efgh0),
    ));
    rows[1] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(abcd1),
        vreinterpretq_s64_s32(efgh1),
    ));
    rows[2] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(abcd2),
        vreinterpretq_s64_s32(efgh2),
    ));
    rows[3] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(abcd3),
        vreinterpretq_s64_s32(efgh3),
    ));
    rows[4] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(abcd0),
        vreinterpretq_s64_s32(efgh0),
    ));
    rows[5] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(abcd1),
        vreinterpretq_s64_s32(efgh1),
    ));
    rows[6] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(abcd2),
        vreinterpretq_s64_s32(efgh2),
    ));
    rows[7] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(abcd3),
        vreinterpretq_s64_s32(efgh3),
    ));
    rows
}

#[inline(always)]
unsafe fn pass0_round16(value: V16) -> V16 {
    let one = vdupq_n_s16(1);
    let positive = vreinterpretq_s16_u16(vshrq_n_u16::<15>(vcgtq_s16(value, vdupq_n_s16(0))));
    vshrq_n_s16::<2>(add16(add16(value, one), positive))
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
    let zero16 = vdupq_n_s16(0);
    let mut intermediate = [0i16; 1024];

    // The first pass stays within i16 for valid 8-bit residuals, matching
    // libvpx's eight-wide NEON kernel and halving the number of lane groups.
    for column in (0..32).step_by(8) {
        let mut lane_in = [zero16; 32];
        for (row, values) in lane_in.iter_mut().enumerate() {
            *values = vshlq_n_s16::<2>(vld1q_s16(input.add(row * stride + column)));
        }
        let mut lane_out = [zero16; 32];
        fdct32x8_first_pass(&lane_in, &mut lane_out);
        for (coefficient, &values) in lane_out.iter().enumerate() {
            vst1q_s16(
                intermediate.as_mut_ptr().add(coefficient * 32 + column),
                pass0_round16(values),
            );
        }
    }

    // Transpose eight intermediate rows at a time. The second pass evaluates
    // stages 1 and 2 in i16, then advances paired four-lane i32 groups
    // through the overflow-sensitive remainder.
    for row in (0..32).step_by(8) {
        let mut lane_in16 = [zero16; 32];
        for column in (0..32).step_by(8) {
            let rows = transpose8([
                vld1q_s16(intermediate.as_ptr().add(row * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 1) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 2) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 3) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 4) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 5) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 6) * 32 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 7) * 32 + column)),
            ]);
            lane_in16[column..column + 8].copy_from_slice(&rows);
        }

        let mut stage2 = [zero16; 32];
        fdct32x8_second_pass_stage2(&lane_in16, &mut stage2);
        let mut lane_in = [[zero; 2]; 32];
        for (source, destination) in stage2.iter().zip(lane_in.iter_mut()) {
            *destination = [vmovl_s16(vget_low_s16(*source)), vmovl_high_s16(*source)];
        }
        let mut lane_out = [[zero; 2]; 32];
        fdct32x8_from_stage2(&lane_in, &mut lane_out);
        for half in [0, 1] {
            for coefficient in (0..32).step_by(4) {
                let rows = transpose4(
                    pass1_round_and_wrap(lane_out[coefficient][half]),
                    pass1_round_and_wrap(lane_out[coefficient + 1][half]),
                    pass1_round_and_wrap(lane_out[coefficient + 2][half]),
                    pass1_round_and_wrap(lane_out[coefficient + 3][half]),
                );
                for (lane, &values) in rows.iter().enumerate() {
                    vst1q_s32(
                        output.add((row + half * 4 + lane) * 32 + coefficient),
                        values,
                    );
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn fdct16x16_inner(input: *const i16, output: *mut i32, stride: usize) {
    let zero = vdupq_n_s32(0);
    let zero16 = vdupq_n_s16(0);
    let mut intermediate = [0i16; 256];

    // The bounded first pass processes eight source columns per vector.
    for column in (0..16).step_by(8) {
        let mut lane_in = [zero16; 16];
        for (row, values) in lane_in.iter_mut().enumerate() {
            *values = vshlq_n_s16::<2>(vld1q_s16(input.add(row * stride + column)));
        }
        let mut lane_out = [zero16; 16];
        fdct16x8(&lane_in, &mut lane_out);
        for (coefficient, &values) in lane_out.iter().enumerate() {
            vst1q_s16(
                intermediate.as_mut_ptr().add(coefficient * 16 + column),
                values,
            );
        }
    }

    // Transpose eight coefficient rows at a time, apply the exact
    // `(x + 1) >> 2` reduction, and perform the bounded cross-input stage in
    // eight i16 lanes before widening the remaining transform to i32.
    for row in (0..16).step_by(8) {
        let mut lane_in16 = [zero16; 16];
        for column in (0..16).step_by(8) {
            let rows = transpose8([
                vld1q_s16(intermediate.as_ptr().add(row * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 1) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 2) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 3) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 4) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 5) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 6) * 16 + column)),
                vld1q_s16(intermediate.as_ptr().add((row + 7) * 16 + column)),
            ]);
            for (offset, values) in rows.into_iter().enumerate() {
                lane_in16[column + offset] = fdct16_interpass_round16(values);
            }
        }

        let max_abs = lane_in16.iter().fold(zero16, |maximum, &values| {
            vmaxq_s16(maximum, vqabsq_s16(values))
        });
        if vmaxvq_s16(max_abs) <= 2047 {
            // A coefficient-bound audit puts the schedule's peak at 16x its
            // input. This limit therefore keeps every add, subtract, and
            // rounded rotation inside signed i16 (16 * 2047 = 32752).
            let mut lane_out16 = [zero16; 16];
            fdct16x8(&lane_in16, &mut lane_out16);
            for coefficient in (0..16).step_by(8) {
                let rows = transpose8([
                    lane_out16[coefficient],
                    lane_out16[coefficient + 1],
                    lane_out16[coefficient + 2],
                    lane_out16[coefficient + 3],
                    lane_out16[coefficient + 4],
                    lane_out16[coefficient + 5],
                    lane_out16[coefficient + 6],
                    lane_out16[coefficient + 7],
                ]);
                for (lane, values) in rows.into_iter().enumerate() {
                    let destination = output.add((row + lane) * 16 + coefficient);
                    vst1q_s32(destination, vmovl_s16(vget_low_s16(values)));
                    vst1q_s32(destination.add(4), vmovl_high_s16(values));
                }
            }
        } else {
            let mut crossed = [zero16; 16];
            for index in 0..8 {
                crossed[index] = add16(lane_in16[index], lane_in16[15 - index]);
                crossed[index + 8] = sub16(lane_in16[7 - index], lane_in16[8 + index]);
            }

            for half in 0..2 {
                let mut lane_in = [zero; 16];
                for (source, destination) in crossed.iter().zip(lane_in.iter_mut()) {
                    *destination = if half == 0 {
                        vmovl_s16(vget_low_s16(*source))
                    } else {
                        vmovl_high_s16(*source)
                    };
                }
                let mut lane_out = [zero; 16];
                fdct16x4_from_cross(&lane_in, &mut lane_out);
                for coefficient in (0..16).step_by(4) {
                    let rows = transpose4(
                        lane_out[coefficient],
                        lane_out[coefficient + 1],
                        lane_out[coefficient + 2],
                        lane_out[coefficient + 3],
                    );
                    for (lane, &values) in rows.iter().enumerate() {
                        vst1q_s32(
                            output.add((row + half * 4 + lane) * 16 + coefficient),
                            values,
                        );
                    }
                }
            }
        }
    }
}

pub(super) fn fdct16x16(input: &[i16], output: &mut [i32; 256], stride: usize) {
    assert!(stride >= 16);
    assert!(input.len() >= 15 * stride + 16);
    // SAFETY: The bounds above prove every 4-value input load. `output`
    // provides all 256 destinations, and NEON is mandatory on AArch64.
    unsafe { fdct16x16_inner(input.as_ptr(), output.as_mut_ptr(), stride) };
}

pub(super) fn fdct32x32(input: &[i16], output: &mut [i32; 1024], stride: usize) {
    assert!(stride >= 32);
    assert!(input.len() >= 31 * stride + 32);
    // SAFETY: The bounds above prove every 4-value input load. `output`
    // provides all 1024 destinations, and NEON is mandatory on AArch64.
    unsafe { fdct32x32_inner(input.as_ptr(), output.as_mut_ptr(), stride) };
}
