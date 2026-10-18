//! SSE2 and AVX2 forward DCT kernels for the 8x8 and 16x16 transforms.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::transforms::constants::*;

#[inline(always)]
unsafe fn mul_round16(value: __m128i, constant: i32) -> __m128i {
    let constant = _mm_set1_epi16(constant as i16);
    let product_lo = _mm_mullo_epi16(value, constant);
    let product_hi = _mm_mulhi_epi16(value, constant);
    let bias = _mm_set1_epi32(1 << 13);
    let lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_unpacklo_epi16(product_lo, product_hi), bias),
        14,
    );
    let hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_unpackhi_epi16(product_lo, product_hi), bias),
        14,
    );
    _mm_packs_epi32(lo, hi)
}

#[inline(always)]
unsafe fn mul_add_round16(a: __m128i, ac: i32, b: __m128i, bc: i32) -> __m128i {
    let pair = ((bc as i16 as u16 as u32) << 16) | u32::from(ac as i16 as u16);
    let constants = _mm_set1_epi32(pair as i32);
    let bias = _mm_set1_epi32(1 << 13);
    let lo = _mm_madd_epi16(_mm_unpacklo_epi16(a, b), constants);
    let hi = _mm_madd_epi16(_mm_unpackhi_epi16(a, b), constants);
    let lo = _mm_srai_epi32(_mm_add_epi32(lo, bias), 14);
    let hi = _mm_srai_epi32(_mm_add_epi32(hi, bias), 14);
    _mm_packs_epi32(lo, hi)
}

#[inline(always)]
unsafe fn transpose8(mut rows: [__m128i; 8]) -> [__m128i; 8] {
    let ab0 = _mm_unpacklo_epi16(rows[0], rows[1]);
    let ab1 = _mm_unpackhi_epi16(rows[0], rows[1]);
    let cd0 = _mm_unpacklo_epi16(rows[2], rows[3]);
    let cd1 = _mm_unpackhi_epi16(rows[2], rows[3]);
    let ef0 = _mm_unpacklo_epi16(rows[4], rows[5]);
    let ef1 = _mm_unpackhi_epi16(rows[4], rows[5]);
    let gh0 = _mm_unpacklo_epi16(rows[6], rows[7]);
    let gh1 = _mm_unpackhi_epi16(rows[6], rows[7]);

    let abcd0 = _mm_unpacklo_epi32(ab0, cd0);
    let abcd1 = _mm_unpacklo_epi32(ab1, cd1);
    let abcd2 = _mm_unpackhi_epi32(ab0, cd0);
    let abcd3 = _mm_unpackhi_epi32(ab1, cd1);
    let efgh0 = _mm_unpacklo_epi32(ef0, gh0);
    let efgh1 = _mm_unpacklo_epi32(ef1, gh1);
    let efgh2 = _mm_unpackhi_epi32(ef0, gh0);
    let efgh3 = _mm_unpackhi_epi32(ef1, gh1);

    rows[0] = _mm_unpacklo_epi64(abcd0, efgh0);
    rows[1] = _mm_unpackhi_epi64(abcd0, efgh0);
    rows[2] = _mm_unpacklo_epi64(abcd2, efgh2);
    rows[3] = _mm_unpackhi_epi64(abcd2, efgh2);
    rows[4] = _mm_unpacklo_epi64(abcd1, efgh1);
    rows[5] = _mm_unpackhi_epi64(abcd1, efgh1);
    rows[6] = _mm_unpacklo_epi64(abcd3, efgh3);
    rows[7] = _mm_unpackhi_epi64(abcd3, efgh3);
    rows
}

/// Run one exact 8-point DCT on eight independent i16 lanes.
#[inline(always)]
unsafe fn fdct8x8_pass(input: &[__m128i; 8], output: &mut [__m128i; 8]) {
    let s0 = _mm_add_epi16(input[0], input[7]);
    let s1 = _mm_add_epi16(input[1], input[6]);
    let s2 = _mm_add_epi16(input[2], input[5]);
    let s3 = _mm_add_epi16(input[3], input[4]);
    let s4 = _mm_sub_epi16(input[3], input[4]);
    let s5 = _mm_sub_epi16(input[2], input[5]);
    let s6 = _mm_sub_epi16(input[1], input[6]);
    let s7 = _mm_sub_epi16(input[0], input[7]);

    let x0 = _mm_add_epi16(s0, s3);
    let x1 = _mm_add_epi16(s1, s2);
    let x2 = _mm_sub_epi16(s1, s2);
    let x3 = _mm_sub_epi16(s0, s3);
    output[0] = mul_round16(_mm_add_epi16(x0, x1), COSPI_16_64);
    output[2] = mul_add_round16(x3, COSPI_8_64, x2, COSPI_24_64);
    output[4] = mul_round16(_mm_sub_epi16(x0, x1), COSPI_16_64);
    output[6] = mul_add_round16(x3, COSPI_24_64, x2, -COSPI_8_64);

    let t2 = mul_round16(_mm_sub_epi16(s6, s5), COSPI_16_64);
    let t3 = mul_round16(_mm_add_epi16(s6, s5), COSPI_16_64);
    let x0 = _mm_add_epi16(s4, t2);
    let x1 = _mm_sub_epi16(s4, t2);
    let x2 = _mm_sub_epi16(s7, t3);
    let x3 = _mm_add_epi16(s7, t3);
    output[1] = mul_add_round16(x0, COSPI_28_64, x3, COSPI_4_64);
    output[3] = mul_add_round16(x2, COSPI_12_64, x1, -COSPI_20_64);
    output[5] = mul_add_round16(x1, COSPI_12_64, x2, COSPI_20_64);
    output[7] = mul_add_round16(x3, COSPI_28_64, x0, -COSPI_4_64);
}

#[inline(always)]
unsafe fn pass1_is_safe(rows: &[__m128i; 8]) -> bool {
    let lower = _mm_set1_epi16(-4096);
    let upper = _mm_set1_epi16(4095);
    rows.iter().all(|&row| {
        let outside = _mm_or_si128(_mm_cmpgt_epi16(row, upper), _mm_cmpgt_epi16(lower, row));
        _mm_movemask_epi8(outside) == 0
    })
}

#[target_feature(enable = "sse2")]
unsafe fn fdct8x8_inner(input: *const i16, output: *mut i32, stride: usize) -> bool {
    let zero = _mm_setzero_si128();
    let mut rows = [zero; 8];
    for (row, values) in rows.iter_mut().enumerate() {
        *values = _mm_slli_epi16(_mm_loadu_si128(input.add(row * stride).cast()), 2);
    }
    let rows = {
        let mut transformed = [zero; 8];
        fdct8x8_pass(&rows, &mut transformed);
        transformed
    };
    // The second pass uses the same i16 schedule as libvpx's lowbd kernel.
    // High-energy blocks take the scalar path before any narrowing can alter
    // the exact result, matching the AArch64 exact kernel's safety guard.
    if !pass1_is_safe(&rows) {
        return false;
    }
    let rows = transpose8(rows);
    let rows = {
        let mut transformed = [zero; 8];
        fdct8x8_pass(&rows, &mut transformed);
        transformed
    };
    let rows = transpose8(rows);
    for (row, values) in rows.into_iter().enumerate() {
        let negative = _mm_srai_epi16(values, 15);
        let reduced = _mm_srai_epi16(_mm_sub_epi16(values, negative), 1);
        let lo = _mm_unpacklo_epi16(reduced, _mm_srai_epi16(reduced, 15));
        let hi = _mm_unpackhi_epi16(reduced, _mm_srai_epi16(reduced, 15));
        let destination = output.add(row * 8);
        _mm_storeu_si128(destination.cast(), lo);
        _mm_storeu_si128(destination.add(4).cast(), hi);
    }
    true
}

pub(super) fn fdct8x8(input: &[i16], output: &mut [i32; 64], stride: usize) {
    #[cfg(all(test, not(miri)))]
    crate::dsp::force_scalar_tests::record("fdct8x8");
    assert!(stride >= 8);
    let required_input_len =
        super::fdct8x8_required_input_len(stride).expect("8x8 input span overflows usize");
    assert!(input.len() >= required_input_len);
    // SAFETY: The bounds above prove every eight-value input load and all
    // fixed output stores. SSE2 is mandatory on x86-64.
    if !unsafe { fdct8x8_inner(input.as_ptr(), output.as_mut_ptr(), stride) } {
        super::fdct8x8_scalar(input, output, stride);
    }
}

#[inline(always)]
unsafe fn fdct16x8(input: &[__m128i; 16], output: &mut [__m128i; 16]) {
    let zero = _mm_setzero_si128();
    let mut in_half = [zero; 8];
    let mut step1 = [zero; 8];
    for index in 0..8 {
        in_half[index] = _mm_add_epi16(input[index], input[15 - index]);
        step1[index] = _mm_sub_epi16(input[7 - index], input[8 + index]);
    }

    let s0 = _mm_add_epi16(in_half[0], in_half[7]);
    let s1 = _mm_add_epi16(in_half[1], in_half[6]);
    let s2 = _mm_add_epi16(in_half[2], in_half[5]);
    let s3 = _mm_add_epi16(in_half[3], in_half[4]);
    let s4 = _mm_sub_epi16(in_half[3], in_half[4]);
    let s5 = _mm_sub_epi16(in_half[2], in_half[5]);
    let s6 = _mm_sub_epi16(in_half[1], in_half[6]);
    let s7 = _mm_sub_epi16(in_half[0], in_half[7]);

    let x0 = _mm_add_epi16(s0, s3);
    let x1 = _mm_add_epi16(s1, s2);
    let x2 = _mm_sub_epi16(s1, s2);
    let x3 = _mm_sub_epi16(s0, s3);
    output[0] = mul_round16(_mm_add_epi16(x0, x1), COSPI_16_64);
    output[4] = mul_add_round16(x3, COSPI_8_64, x2, COSPI_24_64);
    output[8] = mul_round16(_mm_sub_epi16(x0, x1), COSPI_16_64);
    output[12] = mul_add_round16(x3, COSPI_24_64, x2, -COSPI_8_64);

    let t2 = mul_round16(_mm_sub_epi16(s6, s5), COSPI_16_64);
    let t3 = mul_round16(_mm_add_epi16(s6, s5), COSPI_16_64);
    let x0 = _mm_add_epi16(s4, t2);
    let x1 = _mm_sub_epi16(s4, t2);
    let x2 = _mm_sub_epi16(s7, t3);
    let x3 = _mm_add_epi16(s7, t3);
    output[2] = mul_add_round16(x0, COSPI_28_64, x3, COSPI_4_64);
    output[6] = mul_add_round16(x2, COSPI_12_64, x1, -COSPI_20_64);
    output[10] = mul_add_round16(x1, COSPI_12_64, x2, COSPI_20_64);
    output[14] = mul_add_round16(x3, COSPI_28_64, x0, -COSPI_4_64);

    let step2_2 = mul_round16(_mm_sub_epi16(step1[5], step1[2]), COSPI_16_64);
    let step2_3 = mul_round16(_mm_sub_epi16(step1[4], step1[3]), COSPI_16_64);
    let step2_4 = mul_round16(_mm_add_epi16(step1[4], step1[3]), COSPI_16_64);
    let step2_5 = mul_round16(_mm_add_epi16(step1[5], step1[2]), COSPI_16_64);

    let step3_0 = _mm_add_epi16(step1[0], step2_3);
    let step3_1 = _mm_add_epi16(step1[1], step2_2);
    let step3_2 = _mm_sub_epi16(step1[1], step2_2);
    let step3_3 = _mm_sub_epi16(step1[0], step2_3);
    let step3_4 = _mm_sub_epi16(step1[7], step2_4);
    let step3_5 = _mm_sub_epi16(step1[6], step2_5);
    let step3_6 = _mm_add_epi16(step1[6], step2_5);
    let step3_7 = _mm_add_epi16(step1[7], step2_4);

    let step2_1 = mul_add_round16(step3_1, -COSPI_8_64, step3_6, COSPI_24_64);
    let step2_2 = mul_add_round16(step3_2, COSPI_24_64, step3_5, COSPI_8_64);
    let step2_5 = mul_add_round16(step3_2, COSPI_8_64, step3_5, -COSPI_24_64);
    let step2_6 = mul_add_round16(step3_1, COSPI_24_64, step3_6, COSPI_8_64);

    let step1_0 = _mm_add_epi16(step3_0, step2_1);
    let step1_1 = _mm_sub_epi16(step3_0, step2_1);
    let step1_2 = _mm_add_epi16(step3_3, step2_2);
    let step1_3 = _mm_sub_epi16(step3_3, step2_2);
    let step1_4 = _mm_sub_epi16(step3_4, step2_5);
    let step1_5 = _mm_add_epi16(step3_4, step2_5);
    let step1_6 = _mm_sub_epi16(step3_7, step2_6);
    let step1_7 = _mm_add_epi16(step3_7, step2_6);

    output[1] = mul_add_round16(step1_0, COSPI_30_64, step1_7, COSPI_2_64);
    output[9] = mul_add_round16(step1_1, COSPI_14_64, step1_6, COSPI_18_64);
    output[5] = mul_add_round16(step1_2, COSPI_22_64, step1_5, COSPI_10_64);
    output[13] = mul_add_round16(step1_3, COSPI_6_64, step1_4, COSPI_26_64);
    output[3] = mul_add_round16(step1_3, -COSPI_26_64, step1_4, COSPI_6_64);
    output[11] = mul_add_round16(step1_2, -COSPI_10_64, step1_5, COSPI_22_64);
    output[7] = mul_add_round16(step1_1, -COSPI_18_64, step1_6, COSPI_14_64);
    output[15] = mul_add_round16(step1_0, -COSPI_2_64, step1_7, COSPI_30_64);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_round16_avx2(value: __m256i, constant: i32) -> __m256i {
    let constant = _mm256_set1_epi16(constant as i16);
    let product_lo = _mm256_mullo_epi16(value, constant);
    let product_hi = _mm256_mulhi_epi16(value, constant);
    let bias = _mm256_set1_epi32(1 << 13);
    let lo = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_unpacklo_epi16(product_lo, product_hi), bias),
        14,
    );
    let hi = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_unpackhi_epi16(product_lo, product_hi), bias),
        14,
    );
    _mm256_packs_epi32(lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_add_round16_avx2(a: __m256i, ac: i32, b: __m256i, bc: i32) -> __m256i {
    let pair = ((bc as i16 as u16 as u32) << 16) | u32::from(ac as i16 as u16);
    let constants = _mm256_set1_epi32(pair as i32);
    let bias = _mm256_set1_epi32(1 << 13);
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), constants);
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), constants);
    let lo = _mm256_srai_epi32(_mm256_add_epi32(lo, bias), 14);
    let hi = _mm256_srai_epi32(_mm256_add_epi32(hi, bias), 14);
    _mm256_packs_epi32(lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fdct16x16_pass_avx2(input: &[__m256i; 16], output: &mut [__m256i; 16]) {
    let zero = _mm256_setzero_si256();
    let mut in_half = [zero; 8];
    let mut step1 = [zero; 8];
    for index in 0..8 {
        in_half[index] = _mm256_add_epi16(input[index], input[15 - index]);
        step1[index] = _mm256_sub_epi16(input[7 - index], input[8 + index]);
    }

    let s0 = _mm256_add_epi16(in_half[0], in_half[7]);
    let s1 = _mm256_add_epi16(in_half[1], in_half[6]);
    let s2 = _mm256_add_epi16(in_half[2], in_half[5]);
    let s3 = _mm256_add_epi16(in_half[3], in_half[4]);
    let s4 = _mm256_sub_epi16(in_half[3], in_half[4]);
    let s5 = _mm256_sub_epi16(in_half[2], in_half[5]);
    let s6 = _mm256_sub_epi16(in_half[1], in_half[6]);
    let s7 = _mm256_sub_epi16(in_half[0], in_half[7]);

    let x0 = _mm256_add_epi16(s0, s3);
    let x1 = _mm256_add_epi16(s1, s2);
    let x2 = _mm256_sub_epi16(s1, s2);
    let x3 = _mm256_sub_epi16(s0, s3);
    output[0] = mul_round16_avx2(_mm256_add_epi16(x0, x1), COSPI_16_64);
    output[4] = mul_add_round16_avx2(x3, COSPI_8_64, x2, COSPI_24_64);
    output[8] = mul_round16_avx2(_mm256_sub_epi16(x0, x1), COSPI_16_64);
    output[12] = mul_add_round16_avx2(x3, COSPI_24_64, x2, -COSPI_8_64);

    let t2 = mul_round16_avx2(_mm256_sub_epi16(s6, s5), COSPI_16_64);
    let t3 = mul_round16_avx2(_mm256_add_epi16(s6, s5), COSPI_16_64);
    let x0 = _mm256_add_epi16(s4, t2);
    let x1 = _mm256_sub_epi16(s4, t2);
    let x2 = _mm256_sub_epi16(s7, t3);
    let x3 = _mm256_add_epi16(s7, t3);
    output[2] = mul_add_round16_avx2(x0, COSPI_28_64, x3, COSPI_4_64);
    output[6] = mul_add_round16_avx2(x2, COSPI_12_64, x1, -COSPI_20_64);
    output[10] = mul_add_round16_avx2(x1, COSPI_12_64, x2, COSPI_20_64);
    output[14] = mul_add_round16_avx2(x3, COSPI_28_64, x0, -COSPI_4_64);

    let step2_2 = mul_round16_avx2(_mm256_sub_epi16(step1[5], step1[2]), COSPI_16_64);
    let step2_3 = mul_round16_avx2(_mm256_sub_epi16(step1[4], step1[3]), COSPI_16_64);
    let step2_4 = mul_round16_avx2(_mm256_add_epi16(step1[4], step1[3]), COSPI_16_64);
    let step2_5 = mul_round16_avx2(_mm256_add_epi16(step1[5], step1[2]), COSPI_16_64);

    let step3_0 = _mm256_add_epi16(step1[0], step2_3);
    let step3_1 = _mm256_add_epi16(step1[1], step2_2);
    let step3_2 = _mm256_sub_epi16(step1[1], step2_2);
    let step3_3 = _mm256_sub_epi16(step1[0], step2_3);
    let step3_4 = _mm256_sub_epi16(step1[7], step2_4);
    let step3_5 = _mm256_sub_epi16(step1[6], step2_5);
    let step3_6 = _mm256_add_epi16(step1[6], step2_5);
    let step3_7 = _mm256_add_epi16(step1[7], step2_4);

    let step2_1 = mul_add_round16_avx2(step3_1, -COSPI_8_64, step3_6, COSPI_24_64);
    let step2_2 = mul_add_round16_avx2(step3_2, COSPI_24_64, step3_5, COSPI_8_64);
    let step2_5 = mul_add_round16_avx2(step3_2, COSPI_8_64, step3_5, -COSPI_24_64);
    let step2_6 = mul_add_round16_avx2(step3_1, COSPI_24_64, step3_6, COSPI_8_64);

    let step1_0 = _mm256_add_epi16(step3_0, step2_1);
    let step1_1 = _mm256_sub_epi16(step3_0, step2_1);
    let step1_2 = _mm256_add_epi16(step3_3, step2_2);
    let step1_3 = _mm256_sub_epi16(step3_3, step2_2);
    let step1_4 = _mm256_sub_epi16(step3_4, step2_5);
    let step1_5 = _mm256_add_epi16(step3_4, step2_5);
    let step1_6 = _mm256_sub_epi16(step3_7, step2_6);
    let step1_7 = _mm256_add_epi16(step3_7, step2_6);

    output[1] = mul_add_round16_avx2(step1_0, COSPI_30_64, step1_7, COSPI_2_64);
    output[9] = mul_add_round16_avx2(step1_1, COSPI_14_64, step1_6, COSPI_18_64);
    output[5] = mul_add_round16_avx2(step1_2, COSPI_22_64, step1_5, COSPI_10_64);
    output[13] = mul_add_round16_avx2(step1_3, COSPI_6_64, step1_4, COSPI_26_64);
    output[3] = mul_add_round16_avx2(step1_3, -COSPI_26_64, step1_4, COSPI_6_64);
    output[11] = mul_add_round16_avx2(step1_2, -COSPI_10_64, step1_5, COSPI_22_64);
    output[7] = mul_add_round16_avx2(step1_1, -COSPI_18_64, step1_6, COSPI_14_64);
    output[15] = mul_add_round16_avx2(step1_0, -COSPI_2_64, step1_7, COSPI_30_64);
}

#[target_feature(enable = "avx2")]
unsafe fn fdct16x16_first_pass_avx2(input: *const i16, intermediate: *mut i16, stride: usize) {
    let zero = _mm256_setzero_si256();
    let mut lane_in = [zero; 16];
    for (row, values) in lane_in.iter_mut().enumerate() {
        // G4: loads from `input` use _mm256_loadu_si256 (unaligned).
        *values = _mm256_slli_epi16(_mm256_loadu_si256(input.add(row * stride).cast()), 2);
    }
    let mut lane_out = [zero; 16];
    fdct16x16_pass_avx2(&lane_in, &mut lane_out);
    for (coefficient, values) in lane_out.into_iter().enumerate() {
        _mm256_storeu_si256(intermediate.add(coefficient * 16).cast(), values);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn fdct16x16_first_pass_sse2(input: *const i16, intermediate: *mut i16, stride: usize) {
    let zero = _mm_setzero_si128();
    for column in (0..16).step_by(8) {
        let mut lane_in = [zero; 16];
        for (row, values) in lane_in.iter_mut().enumerate() {
            // G4: loads from `input` use _mm_loadu_si128 (unaligned).
            *values = _mm_slli_epi16(_mm_loadu_si128(input.add(row * stride + column).cast()), 2);
        }
        let mut lane_out = [zero; 16];
        fdct16x8(&lane_in, &mut lane_out);
        for (coefficient, values) in lane_out.into_iter().enumerate() {
            _mm_storeu_si128(intermediate.add(coefficient * 16 + column).cast(), values);
        }
    }
}

pub(super) fn fdct16x16(input: &[i16], output: &mut [i32; 256], stride: usize) {
    #[cfg(all(test, not(miri)))]
    crate::dsp::force_scalar_tests::record("fdct16x16");
    assert!(stride >= 16);
    assert!(input.len() >= 15 * stride + 16);
    let mut intermediate = [0i16; 256];

    // SAFETY: The assertions prove every unaligned input load and the fixed
    // intermediate stores. The AVX2 call is gated by the runtime feature
    // check; valid 8-bit residuals keep either first-pass schedule in i16.
    unsafe {
        if is_x86_feature_detected!("avx2") {
            fdct16x16_first_pass_avx2(input.as_ptr(), intermediate.as_mut_ptr(), stride);
        } else {
            fdct16x16_first_pass_sse2(input.as_ptr(), intermediate.as_mut_ptr(), stride);
        }
    }

    for row in 0..16 {
        let row_base = row * 16;
        let read = |column: usize| (i32::from(intermediate[row_base + column]) + 1) >> 2;
        let in_half: [i64; 8] = [
            i64::from(read(0) + read(15)),
            i64::from(read(1) + read(14)),
            i64::from(read(2) + read(13)),
            i64::from(read(3) + read(12)),
            i64::from(read(4) + read(11)),
            i64::from(read(5) + read(10)),
            i64::from(read(6) + read(9)),
            i64::from(read(7) + read(8)),
        ];
        let step1: [i64; 8] = [
            i64::from(read(7) - read(8)),
            i64::from(read(6) - read(9)),
            i64::from(read(5) - read(10)),
            i64::from(read(4) - read(11)),
            i64::from(read(3) - read(12)),
            i64::from(read(2) - read(13)),
            i64::from(read(1) - read(14)),
            i64::from(read(0) - read(15)),
        ];
        super::fdct16_butterfly(&in_half, &step1, output, row_base);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdct8x8_sse2_matches_scalar_valid_residuals() {
        // Random and extreme inputs verify wrapper dispatch, including scalar fallback cases.
        let mut state = 0x8d31_e74a_25c6_09bf_u64;
        for case in 0..512 {
            let stride = 8 + case % 23;
            let mut residual = vec![0i16; 7 * stride + 8];
            for row in 0..8 {
                for column in 0..8 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    residual[row * stride + column] = (((state >> 32) % 511) as i16) - 255;
                }
            }
            let mut scalar = [0i32; 64];
            let mut vector = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, stride);
            fdct8x8(&residual, &mut vector, stride);
            assert_eq!(
                vector, scalar,
                "random residual case {case}, stride {stride}"
            );
        }

        for &value in &[-255i16, 0, 255] {
            let residual = [value; 64];
            let mut scalar = [0i32; 64];
            let mut vector = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, 8);
            // SAFETY: The arrays cover the 8x8 input and 64 outputs. SSE2 is mandatory on x86-64.
            let used_sse2 = unsafe { fdct8x8_inner(residual.as_ptr(), vector.as_mut_ptr(), 8) };
            assert_eq!(
                used_sse2,
                value == 0,
                "constant residual {value} expected {}",
                if value == 0 {
                    "SSE2"
                } else {
                    "scalar fallback"
                }
            );
            fdct8x8(&residual, &mut vector, 8);
            assert_eq!(vector, scalar, "constant residual {value} wrapper parity");
        }

        for axis in [0, 1] {
            let residual: [i16; 64] = std::array::from_fn(|index| {
                let coordinate = if axis == 0 { index / 8 } else { index % 8 };
                if coordinate & 1 == 0 { -255 } else { 255 }
            });
            let mut scalar = [0i32; 64];
            let mut vector = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, 8);
            // SAFETY: The arrays cover the 8x8 input and 64 outputs. SSE2 is mandatory on x86-64.
            let used_sse2 = unsafe { fdct8x8_inner(residual.as_ptr(), vector.as_mut_ptr(), 8) };
            assert!(
                !used_sse2,
                "axis {axis} residual extremes should take scalar fallback"
            );
            fdct8x8(&residual, &mut vector, 8);
            assert_eq!(
                vector, scalar,
                "axis {axis} residual extremes fallback parity"
            );
        }

        for index in 0..64 {
            let residual: [i16; 64] =
                std::array::from_fn(|position| if position == index { 255 } else { -255 });
            let mut scalar = [0i32; 64];
            let mut vector = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, 8);
            // SAFETY: The arrays cover the 8x8 input and 64 outputs. SSE2 is mandatory on x86-64.
            let used_sse2 = unsafe { fdct8x8_inner(residual.as_ptr(), vector.as_mut_ptr(), 8) };
            assert!(
                !used_sse2,
                "extreme impulse index {index} should take scalar fallback"
            );
            fdct8x8(&residual, &mut vector, 8);
            assert_eq!(
                vector, scalar,
                "extreme impulse index {index} fallback parity"
            );
        }
    }

    fn residual_8x8_with_stride(
        stride: usize,
        mut sample: impl FnMut(usize, usize) -> i16,
    ) -> Vec<i16> {
        let mut residual = vec![i16::MAX; 7 * stride + 8];
        for row in 0..8 {
            for column in 0..8 {
                residual[row * stride + column] = sample(row, column);
            }
        }
        residual
    }

    #[test]
    fn fdct8x8_sse2_near_guard_inputs_run_kernel_and_match_scalar() {
        let stride = 13;
        for value in [-181i16, 181] {
            let residual = residual_8x8_with_stride(stride, |_, _| value);
            let mut scalar = [0i32; 64];
            let mut vector = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, stride);
            // SAFETY: The strided input holds 8 rows of 8 values; output holds 64 coefficients.
            // The x86-64 target guarantees SSE2.
            let used_sse2 =
                unsafe { fdct8x8_inner(residual.as_ptr(), vector.as_mut_ptr(), stride) };
            assert!(
                used_sse2,
                "constant residual {value} should hit the inclusive pass-one guard near 4095"
            );
            assert_eq!(
                vector, scalar,
                "constant residual {value} SSE2 output near the pass-one guard"
            );
        }

        let alternating_columns =
            residual_8x8_with_stride(stride, |_, column| if column & 1 == 0 { 181 } else { -181 });
        let mut scalar = [0i32; 64];
        let mut vector = [0i32; 64];
        super::super::fdct8x8_scalar(&alternating_columns, &mut scalar, stride);
        // SAFETY: The strided input holds 8 rows of 8 values; output holds 64 coefficients.
        // The x86-64 target guarantees SSE2.
        let used_sse2 =
            unsafe { fdct8x8_inner(alternating_columns.as_ptr(), vector.as_mut_ptr(), stride) };
        assert!(
            used_sse2,
            "alternating-column ±181 residuals should exercise both pass-one signs near the guard"
        );
        assert_eq!(
            vector, scalar,
            "alternating-column ±181 residual SSE2 output near the pass-one guard"
        );
    }

    #[test]
    fn fdct8x8_sse2_outside_guard_inputs_fall_back_and_match_scalar() {
        let stride = 13;
        for value in [-182i16, 182] {
            let residual = residual_8x8_with_stride(stride, |_, _| value);
            let mut direct = [0i32; 64];
            // SAFETY: The strided input holds 8 rows of 8 values; output holds 64 coefficients.
            // The x86-64 target guarantees SSE2.
            let used_sse2 =
                unsafe { fdct8x8_inner(residual.as_ptr(), direct.as_mut_ptr(), stride) };
            assert!(
                !used_sse2,
                "constant residual {value} should be outside the pass-one guard"
            );

            let mut scalar = [0i32; 64];
            let mut dispatched = [0i32; 64];
            super::super::fdct8x8_scalar(&residual, &mut scalar, stride);
            fdct8x8(&residual, &mut dispatched, stride);
            assert_eq!(
                dispatched, scalar,
                "constant residual {value} wrapper fallback output"
            );
        }
    }

    fn pass1_is_safe_for_repeated_lane(value: i16) -> bool {
        // SAFETY: SSE2 is mandatory on x86-64, and set1 initializes every lane checked below.
        unsafe {
            let rows = [_mm_set1_epi16(value); 8];
            pass1_is_safe(&rows)
        }
    }

    #[test]
    fn fdct8x8_sse2_guard_includes_exact_signed_lane_boundaries() {
        assert!(pass1_is_safe_for_repeated_lane(-4096));
        assert!(pass1_is_safe_for_repeated_lane(4095));
        assert!(!pass1_is_safe_for_repeated_lane(-4097));
        assert!(!pass1_is_safe_for_repeated_lane(4096));
    }

    #[test]
    fn fdct16x16_dispatch_matches_scalar_valid_residuals() {
        let mut state = 0x25d6_37b4_1a90_ce83_u64;
        for case in 0..256 {
            let stride = 16 + case % 17;
            let mut residual = vec![0i16; 15 * stride + 16];
            for row in 0..16 {
                for column in 0..16 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    residual[row * stride + column] = (((state >> 32) % 511) as i16) - 255;
                }
            }
            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, stride);
            fdct16x16(&residual, &mut vector, stride);
            assert_eq!(
                vector, scalar,
                "random residual case {case}, stride {stride}"
            );
        }

        for &value in &[-255i16, 0, 255] {
            let residual = [value; 256];
            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, 16);
            fdct16x16(&residual, &mut vector, 16);
            assert_eq!(vector, scalar, "constant residual {value}");
        }

        for axis in [0, 1] {
            let residual: [i16; 256] = std::array::from_fn(|index| {
                let coordinate = if axis == 0 { index / 16 } else { index % 16 };
                if coordinate & 1 == 0 { -255 } else { 255 }
            });
            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, 16);
            fdct16x16(&residual, &mut vector, 16);
            assert_eq!(vector, scalar, "axis {axis} residual extremes");
        }
    }

    #[test]
    fn fdct16x16_avx2_first_pass_matches_sse2_and_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut state = 0x6a09_e667_f3bc_c908_u64;
        for case in 0..256 {
            let stride = 16 + case % 29;
            let mut residual = vec![0i16; 15 * stride + 16];
            for row in 0..16 {
                for column in 0..16 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    residual[row * stride + column] = (((state >> 32) % 511) as i16) - 255;
                }
            }

            let mut sse2_intermediate = [0i16; 256];
            let mut avx2_intermediate = [0i16; 256];
            // SAFETY: AVX2 was detected above; both input and intermediate
            // buffers satisfy the first-pass helper's fixed load/store shape.
            unsafe {
                fdct16x16_first_pass_sse2(
                    residual.as_ptr(),
                    sse2_intermediate.as_mut_ptr(),
                    stride,
                );
                fdct16x16_first_pass_avx2(
                    residual.as_ptr(),
                    avx2_intermediate.as_mut_ptr(),
                    stride,
                );
            }
            assert_eq!(
                avx2_intermediate, sse2_intermediate,
                "random intermediate case {case}, stride {stride}"
            );

            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, stride);
            fdct16x16(&residual, &mut vector, stride);
            assert_eq!(vector, scalar, "random final case {case}, stride {stride}");
        }

        for &value in &[-255i16, 0, 255] {
            let residual = [value; 256];
            let mut sse2_intermediate = [0i16; 256];
            let mut avx2_intermediate = [0i16; 256];
            // SAFETY: AVX2 was detected above and the fixed-size buffers are
            // valid for all first-pass loads and stores.
            unsafe {
                fdct16x16_first_pass_sse2(residual.as_ptr(), sse2_intermediate.as_mut_ptr(), 16);
                fdct16x16_first_pass_avx2(residual.as_ptr(), avx2_intermediate.as_mut_ptr(), 16);
            }
            assert_eq!(avx2_intermediate, sse2_intermediate, "constant {value}");

            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, 16);
            fdct16x16(&residual, &mut vector, 16);
            assert_eq!(vector, scalar, "constant final output {value}");
        }

        for axis in [0, 1] {
            let residual: [i16; 256] = std::array::from_fn(|index| {
                let coordinate = if axis == 0 { index / 16 } else { index % 16 };
                if coordinate & 1 == 0 { -255 } else { 255 }
            });
            let mut sse2_intermediate = [0i16; 256];
            let mut avx2_intermediate = [0i16; 256];
            // SAFETY: AVX2 was detected above and the fixed-size buffers are
            // valid for all first-pass loads and stores.
            unsafe {
                fdct16x16_first_pass_sse2(residual.as_ptr(), sse2_intermediate.as_mut_ptr(), 16);
                fdct16x16_first_pass_avx2(residual.as_ptr(), avx2_intermediate.as_mut_ptr(), 16);
            }
            assert_eq!(
                avx2_intermediate, sse2_intermediate,
                "axis {axis} intermediate"
            );

            let mut scalar = [0i32; 256];
            let mut vector = [0i32; 256];
            super::super::fdct16x16_scalar(&residual, &mut scalar, 16);
            fdct16x16(&residual, &mut vector, 16);
            assert_eq!(vector, scalar, "axis {axis} final output");
        }
    }
}
