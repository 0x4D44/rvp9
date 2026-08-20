//! AArch64 NEON unscaled VP9 8-tap interpolation.
//!
//! Each vector lane computes one destination pixel with the same i32
//! accumulation, `+64`, arithmetic `>> 7`, and u8 saturation as the scalar
//! oracle. The full 2-D kernel composes the exact horizontal and vertical
//! passes through a reusable per-thread u8 intermediate.

use core::arch::aarch64::*;
use core::arch::asm;

use crate::inter::filter::{convolve8, convolve8_horiz, convolve8_vert};

const INTER_STRIDE: usize = 64;
const INTER_ROWS: usize = 72;
const DOTPROD_PERMUTE: [u8; 48] = [
    0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9,
    10, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14,
];
const DOTPROD_TRANSPOSE: [u8; 32] = [
    0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14,
    22, 30, 7, 15, 23, 31,
];
const DOTPROD_MERGE_BLOCK: [u8; 48] = [
    1, 2, 3, 16, 5, 6, 7, 20, 9, 10, 11, 24, 13, 14, 15, 28, 2, 3, 16, 17, 6, 7, 20, 21, 10, 11,
    24, 25, 14, 15, 28, 29, 3, 16, 17, 18, 7, 20, 21, 22, 11, 24, 25, 26, 15, 28, 29, 30,
];

std::thread_local! {
    static CONV8_INTERMEDIATE: std::cell::RefCell<[u8; INTER_STRIDE * INTER_ROWS]> =
        const { std::cell::RefCell::new([0u8; INTER_STRIDE * INTER_ROWS]) };
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_eight(samples: [uint8x8_t; 8], filter: &[i16; 8]) -> uint8x8_t {
    let mut sum_lo = vdupq_n_s32(64);
    let mut sum_hi = vdupq_n_s32(64);
    for tap in 0..8 {
        let widened = vreinterpretq_s16_u16(vmovl_u8(samples[tap]));
        sum_lo = vmlal_n_s16(sum_lo, vget_low_s16(widened), filter[tap]);
        sum_hi = vmlal_n_s16(sum_hi, vget_high_s16(widened), filter[tap]);
    }
    let rounded = vcombine_s16(vshrn_n_s32::<7>(sum_lo), vshrn_n_s32::<7>(sum_hi));
    vqmovun_s16(rounded)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_four_repeated(src: *const u8) -> uint8x8_t {
    // SAFETY: callers guarantee four readable bytes at `src`; unaligned
    // access is deliberate because frame rows need not be word-aligned.
    let packed = unsafe { src.cast::<u32>().read_unaligned() };
    vreinterpret_u8_u32(vdup_n_u32(packed))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_four(dst: *mut u8, value: uint8x8_t) {
    let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(value));
    // SAFETY: callers guarantee four writable bytes at `dst`; unaligned
    // access is deliberate because destination rows need not be word-aligned.
    unsafe { dst.cast::<u32>().write_unaligned(packed) };
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_four_columns(samples: [uint8x8_t; 8], filter: &[i16; 8]) -> uint8x8_t {
    let mut sum = vdupq_n_s32(64);
    for tap in 0..8 {
        let widened = vreinterpretq_s16_u16(vmovl_u8(samples[tap]));
        sum = vmlal_n_s16(sum, vget_low_s16(widened), filter[tap]);
    }
    let rounded = vshrn_n_s32::<7>(sum);
    vqmovun_s16(vcombine_s16(rounded, rounded))
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot_lane<const LANE: usize>(
    mut sum: int32x4_t,
    samples: int8x16_t,
    filters: int8x16_t,
) -> int32x4_t {
    // SAFETY: `dotprod` is enabled for this function, every operand uses the
    // matching 128-bit SIMD register class, and `LANE` is instantiated only
    // as 0 or 1 below.
    unsafe {
        asm!(
            "sdot {sum:v}.4s, {samples:v}.16b, {filters:v}.4b[{lane}]",
            sum = inout(vreg) sum,
            samples = in(vreg) samples,
            filters = in(vreg) filters,
            lane = const LANE,
            options(pure, nomem, nostack),
        );
    }
    sum
}

#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn unsigned_dot_lane<const LANE: usize>(
    mut sum: int32x4_t,
    samples: uint8x16_t,
    filters: int8x16_t,
) -> int32x4_t {
    // SAFETY: `i8mm` is enabled for this function, every operand uses the
    // matching 128-bit SIMD register class, and `LANE` is instantiated only
    // as 0 or 1 below.
    unsafe {
        asm!(
            "usdot {sum:v}.4s, {samples:v}.16b, {filters:v}.4b[{lane}]",
            sum = inout(vreg) sum,
            samples = in(vreg) samples,
            filters = in(vreg) filters,
            lane = const LANE,
            options(pure, nomem, nostack),
        );
    }
    sum
}

#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn filter_eight_i8mm(
    samples: uint8x16_t,
    filters: int8x16_t,
    permute_0: uint8x16_t,
    permute_1: uint8x16_t,
    permute_2: uint8x16_t,
) -> uint8x8_t {
    let p0 = vqtbl1q_u8(samples, permute_0);
    let p1 = vqtbl1q_u8(samples, permute_1);
    let p2 = vqtbl1q_u8(samples, permute_2);
    let sum0 = unsigned_dot_lane::<0>(vdupq_n_s32(0), p0, filters);
    let sum0 = unsigned_dot_lane::<1>(sum0, p1, filters);
    let sum1 = unsigned_dot_lane::<0>(vdupq_n_s32(0), p1, filters);
    let sum1 = unsigned_dot_lane::<1>(sum1, p2, filters);
    vqrshrun_n_s16::<7>(vcombine_s16(vqmovn_s32(sum0), vqmovn_s32(sum1)))
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn filter_eight_dotprod(
    samples: uint8x16_t,
    filters: int8x8_t,
    correction: int32x4_t,
) -> uint8x8_t {
    let clamped = vreinterpretq_s8_u8(vsubq_u8(samples, vdupq_n_u8(128)));
    let p0 = vqtbl1q_s8(clamped, vld1q_u8(DOTPROD_PERMUTE.as_ptr()));
    let p1 = vqtbl1q_s8(clamped, vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(16)));
    let p2 = vqtbl1q_s8(clamped, vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(32)));
    let filters = vcombine_s8(filters, filters);
    let sum0 = dot_lane::<0>(correction, p0, filters);
    let sum0 = dot_lane::<1>(sum0, p1, filters);
    let sum1 = dot_lane::<0>(correction, p1, filters);
    let sum1 = dot_lane::<1>(sum1, p2, filters);
    vqrshrun_n_s16::<7>(vcombine_s16(vqmovn_s32(sum0), vqmovn_s32(sum1)))
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn transpose_concat_8x4(
    a0: int8x8_t,
    a1: int8x8_t,
    a2: int8x8_t,
    a3: int8x8_t,
) -> (int8x16_t, int8x16_t) {
    let samples = int8x16x2_t(vcombine_s8(a0, a1), vcombine_s8(a2, a3));
    (
        vqtbl2q_s8(samples, vld1q_u8(DOTPROD_TRANSPOSE.as_ptr())),
        vqtbl2q_s8(samples, vld1q_u8(DOTPROD_TRANSPOSE.as_ptr().add(16))),
    )
}

#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn transpose_concat_8x4_i8mm(
    a0: uint8x8_t,
    a1: uint8x8_t,
    a2: uint8x8_t,
    a3: uint8x8_t,
    transpose_0: uint8x16_t,
    transpose_1: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let samples = uint8x16x2_t(vcombine_u8(a0, a1), vcombine_u8(a2, a3));
    (
        vqtbl2q_u8(samples, transpose_0),
        vqtbl2q_u8(samples, transpose_1),
    )
}

#[inline]
#[target_feature(enable = "i8mm")]
unsafe fn filter_four_rows_i8mm(
    first_lo: uint8x16_t,
    second_lo: uint8x16_t,
    first_hi: uint8x16_t,
    second_hi: uint8x16_t,
    filters: int8x16_t,
) -> uint8x8_t {
    let sum_lo = unsigned_dot_lane::<0>(vdupq_n_s32(0), first_lo, filters);
    let sum_lo = unsigned_dot_lane::<1>(sum_lo, second_lo, filters);
    let sum_hi = unsigned_dot_lane::<0>(vdupq_n_s32(0), first_hi, filters);
    let sum_hi = unsigned_dot_lane::<1>(sum_hi, second_hi, filters);
    vqrshrun_n_s16::<7>(vcombine_s16(vqmovn_s32(sum_lo), vqmovn_s32(sum_hi)))
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn filter_four_rows_dotprod(
    first_lo: int8x16_t,
    second_lo: int8x16_t,
    first_hi: int8x16_t,
    second_hi: int8x16_t,
    filters: int8x16_t,
    correction: int32x4_t,
) -> uint8x8_t {
    let sum_lo = dot_lane::<0>(correction, first_lo, filters);
    let sum_lo = dot_lane::<1>(sum_lo, second_lo, filters);
    let sum_hi = dot_lane::<0>(correction, first_hi, filters);
    let sum_hi = dot_lane::<1>(sum_hi, second_hi, filters);
    vqrshrun_n_s16::<7>(vcombine_s16(vqmovn_s32(sum_lo), vqmovn_s32(sum_hi)))
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve8_horiz_i8mm_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    let filters8 = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let filters = vcombine_s8(filters8, filters8);
    let permute_0 = vld1q_u8(DOTPROD_PERMUTE.as_ptr());
    let permute_1 = vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(16));
    let permute_2 = vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(32));
    let vector_h = h & !3;

    for y in (0..vector_h).step_by(4) {
        let src_0 = src.add(y * src_stride);
        let src_1 = src_0.add(src_stride);
        let src_2 = src_1.add(src_stride);
        let src_3 = src_2.add(src_stride);
        let dst_0 = dst.add(y * dst_stride);
        let dst_1 = dst_0.add(dst_stride);
        let dst_2 = dst_1.add(dst_stride);
        let dst_3 = dst_2.add(dst_stride);
        for x in (0..w).step_by(8) {
            let d0 = filter_eight_i8mm(
                vld1q_u8(src_0.add(x)),
                filters,
                permute_0,
                permute_1,
                permute_2,
            );
            let d1 = filter_eight_i8mm(
                vld1q_u8(src_1.add(x)),
                filters,
                permute_0,
                permute_1,
                permute_2,
            );
            let d2 = filter_eight_i8mm(
                vld1q_u8(src_2.add(x)),
                filters,
                permute_0,
                permute_1,
                permute_2,
            );
            let d3 = filter_eight_i8mm(
                vld1q_u8(src_3.add(x)),
                filters,
                permute_0,
                permute_1,
                permute_2,
            );
            vst1_u8(dst_0.add(x), d0);
            vst1_u8(dst_1.add(x), d1);
            vst1_u8(dst_2.add(x), d2);
            vst1_u8(dst_3.add(x), d3);
        }
    }
    for y in vector_h..h {
        let src_row = src.add(y * src_stride);
        let dst_row = dst.add(y * dst_stride);
        for x in (0..w).step_by(8) {
            let value = filter_eight_i8mm(
                vld1q_u8(src_row.add(x)),
                filters,
                permute_0,
                permute_1,
                permute_2,
            );
            vst1_u8(dst_row.add(x), value);
        }
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve8_horiz_i8mm_4_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    h: usize,
) {
    let filters8 = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let filters = vcombine_s8(filters8, filters8);
    let permute_0 = vld1q_u8(DOTPROD_PERMUTE.as_ptr());
    let permute_1 = vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(16));
    let permute_2 = vld1q_u8(DOTPROD_PERMUTE.as_ptr().add(32));
    for y in 0..h {
        let src_row = src.add(y * src_stride);
        let samples = vcombine_u8(vld1_u8(src_row), load_four_repeated(src_row.add(8)));
        let value = filter_eight_i8mm(samples, filters, permute_0, permute_1, permute_2);
        store_four(dst.add(y * dst_stride), value);
    }
}

#[target_feature(enable = "dotprod")]
unsafe fn convolve8_horiz_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    let filters = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let correction = vdupq_n_s32(filter.iter().map(|&tap| i32::from(tap)).sum::<i32>() * 128);
    for y in 0..h {
        let src_row = src.add(y * src_stride);
        let dst_row = dst.add(y * dst_stride);
        for x in (0..w).step_by(8) {
            let samples = vld1q_u8(src_row.add(x));
            vst1_u8(
                dst_row.add(x),
                filter_eight_dotprod(samples, filters, correction),
            );
        }
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve8_vert_i8mm_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    let filters8 = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let filters = vcombine_s8(filters8, filters8);
    let transpose_0 = vld1q_u8(DOTPROD_TRANSPOSE.as_ptr());
    let transpose_1 = vld1q_u8(DOTPROD_TRANSPOSE.as_ptr().add(16));
    let merge_0 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr());
    let merge_1 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr().add(16));
    let merge_2 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr().add(32));
    let vector_h = h & !3;

    for x in (0..w).step_by(8) {
        if vector_h != 0 {
            let s0 = vld1_u8(src.add(x));
            let s1 = vld1_u8(src.add(src_stride + x));
            let s2 = vld1_u8(src.add(2 * src_stride + x));
            let s3 = vld1_u8(src.add(3 * src_stride + x));
            let s4 = vld1_u8(src.add(4 * src_stride + x));
            let s5 = vld1_u8(src.add(5 * src_stride + x));
            let s6 = vld1_u8(src.add(6 * src_stride + x));
            let zero = vdup_n_u8(0);

            let mut t0 = transpose_concat_8x4_i8mm(s0, s1, s2, s3, transpose_0, transpose_1);
            let mut t1 = transpose_concat_8x4_i8mm(s1, s2, s3, s4, transpose_0, transpose_1);
            let mut t2 = transpose_concat_8x4_i8mm(s2, s3, s4, s5, transpose_0, transpose_1);
            let mut t3 = transpose_concat_8x4_i8mm(s3, s4, s5, s6, transpose_0, transpose_1);
            let mut t4 = transpose_concat_8x4_i8mm(s4, s5, s6, zero, transpose_0, transpose_1);
            let mut t5 = transpose_concat_8x4_i8mm(s5, s6, zero, zero, transpose_0, transpose_1);
            let mut t6 = transpose_concat_8x4_i8mm(s6, zero, zero, zero, transpose_0, transpose_1);

            for y in (0..vector_h).step_by(4) {
                let s7 = vld1_u8(src.add((y + 7) * src_stride + x));
                let s8 = vld1_u8(src.add((y + 8) * src_stride + x));
                let s9 = vld1_u8(src.add((y + 9) * src_stride + x));
                let s10 = vld1_u8(src.add((y + 10) * src_stride + x));
                let t7 = transpose_concat_8x4_i8mm(s7, s8, s9, s10, transpose_0, transpose_1);

                let lo = uint8x16x2_t(t3.0, t7.0);
                t4.0 = vqtbl2q_u8(lo, merge_0);
                t5.0 = vqtbl2q_u8(lo, merge_1);
                t6.0 = vqtbl2q_u8(lo, merge_2);
                let hi = uint8x16x2_t(t3.1, t7.1);
                t4.1 = vqtbl2q_u8(hi, merge_0);
                t5.1 = vqtbl2q_u8(hi, merge_1);
                t6.1 = vqtbl2q_u8(hi, merge_2);

                let d0 = filter_four_rows_i8mm(t0.0, t4.0, t0.1, t4.1, filters);
                let d1 = filter_four_rows_i8mm(t1.0, t5.0, t1.1, t5.1, filters);
                let d2 = filter_four_rows_i8mm(t2.0, t6.0, t2.1, t6.1, filters);
                let d3 = filter_four_rows_i8mm(t3.0, t7.0, t3.1, t7.1, filters);
                vst1_u8(dst.add(y * dst_stride + x), d0);
                vst1_u8(dst.add((y + 1) * dst_stride + x), d1);
                vst1_u8(dst.add((y + 2) * dst_stride + x), d2);
                vst1_u8(dst.add((y + 3) * dst_stride + x), d3);

                t0 = t4;
                t1 = t5;
                t2 = t6;
                t3 = t7;
            }
        }

        for y in vector_h..h {
            let samples = core::array::from_fn(|tap| vld1_u8(src.add((y + tap) * src_stride + x)));
            vst1_u8(dst.add(y * dst_stride + x), filter_eight(samples, filter));
        }
    }
}

#[target_feature(enable = "i8mm")]
unsafe fn convolve8_vert_i8mm_4_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    h: usize,
) {
    let filters8 = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let filters = vcombine_s8(filters8, filters8);
    let transpose_0 = vld1q_u8(DOTPROD_TRANSPOSE.as_ptr());
    let transpose_1 = vld1q_u8(DOTPROD_TRANSPOSE.as_ptr().add(16));
    let merge_0 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr());
    let merge_1 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr().add(16));
    let merge_2 = vld1q_u8(DOTPROD_MERGE_BLOCK.as_ptr().add(32));
    let vector_h = h & !3;

    if vector_h != 0 {
        let s0 = load_four_repeated(src);
        let s1 = load_four_repeated(src.add(src_stride));
        let s2 = load_four_repeated(src.add(2 * src_stride));
        let s3 = load_four_repeated(src.add(3 * src_stride));
        let s4 = load_four_repeated(src.add(4 * src_stride));
        let s5 = load_four_repeated(src.add(5 * src_stride));
        let s6 = load_four_repeated(src.add(6 * src_stride));
        let zero = vdup_n_u8(0);

        let mut t0 = transpose_concat_8x4_i8mm(s0, s1, s2, s3, transpose_0, transpose_1);
        let mut t1 = transpose_concat_8x4_i8mm(s1, s2, s3, s4, transpose_0, transpose_1);
        let mut t2 = transpose_concat_8x4_i8mm(s2, s3, s4, s5, transpose_0, transpose_1);
        let mut t3 = transpose_concat_8x4_i8mm(s3, s4, s5, s6, transpose_0, transpose_1);
        let mut t4 = transpose_concat_8x4_i8mm(s4, s5, s6, zero, transpose_0, transpose_1);
        let mut t5 = transpose_concat_8x4_i8mm(s5, s6, zero, zero, transpose_0, transpose_1);
        let mut t6 = transpose_concat_8x4_i8mm(s6, zero, zero, zero, transpose_0, transpose_1);

        for y in (0..vector_h).step_by(4) {
            let s7 = load_four_repeated(src.add((y + 7) * src_stride));
            let s8 = load_four_repeated(src.add((y + 8) * src_stride));
            let s9 = load_four_repeated(src.add((y + 9) * src_stride));
            let s10 = load_four_repeated(src.add((y + 10) * src_stride));
            let t7 = transpose_concat_8x4_i8mm(s7, s8, s9, s10, transpose_0, transpose_1);

            let lo = uint8x16x2_t(t3.0, t7.0);
            t4.0 = vqtbl2q_u8(lo, merge_0);
            t5.0 = vqtbl2q_u8(lo, merge_1);
            t6.0 = vqtbl2q_u8(lo, merge_2);
            let hi = uint8x16x2_t(t3.1, t7.1);
            t4.1 = vqtbl2q_u8(hi, merge_0);
            t5.1 = vqtbl2q_u8(hi, merge_1);
            t6.1 = vqtbl2q_u8(hi, merge_2);

            let d0 = filter_four_rows_i8mm(t0.0, t4.0, t0.1, t4.1, filters);
            let d1 = filter_four_rows_i8mm(t1.0, t5.0, t1.1, t5.1, filters);
            let d2 = filter_four_rows_i8mm(t2.0, t6.0, t2.1, t6.1, filters);
            let d3 = filter_four_rows_i8mm(t3.0, t7.0, t3.1, t7.1, filters);
            store_four(dst.add(y * dst_stride), d0);
            store_four(dst.add((y + 1) * dst_stride), d1);
            store_four(dst.add((y + 2) * dst_stride), d2);
            store_four(dst.add((y + 3) * dst_stride), d3);

            t0 = t4;
            t1 = t5;
            t2 = t6;
            t3 = t7;
        }
    }

    for y in vector_h..h {
        let samples =
            core::array::from_fn(|tap| load_four_repeated(src.add((y + tap) * src_stride)));
        store_four(
            dst.add(y * dst_stride),
            filter_four_columns(samples, filter),
        );
    }
}

#[target_feature(enable = "neon")]
unsafe fn convolve8_vert_neon_4_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    h: usize,
) {
    for y in 0..h {
        let samples =
            core::array::from_fn(|tap| load_four_repeated(src.add((y + tap) * src_stride)));
        store_four(
            dst.add(y * dst_stride),
            filter_four_columns(samples, filter),
        );
    }
}

#[target_feature(enable = "dotprod")]
unsafe fn convolve8_vert_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    let filters8 = vmovn_s16(vld1q_s16(filter.as_ptr()));
    let filters = vcombine_s8(filters8, filters8);
    let correction = vdupq_n_s32(filter.iter().map(|&tap| i32::from(tap)).sum::<i32>() * 128);
    let range_limit = vdup_n_u8(128);
    let vector_h = h & !3;

    for x in (0..w).step_by(8) {
        for y in (0..vector_h).step_by(4) {
            let mut rows = [vdup_n_s8(0); 11];
            for (row, value) in rows.iter_mut().enumerate() {
                *value = vreinterpret_s8_u8(vsub_u8(
                    vld1_u8(src.add((y + row) * src_stride + x)),
                    range_limit,
                ));
            }

            let t0 = transpose_concat_8x4(rows[0], rows[1], rows[2], rows[3]);
            let t1 = transpose_concat_8x4(rows[1], rows[2], rows[3], rows[4]);
            let t2 = transpose_concat_8x4(rows[2], rows[3], rows[4], rows[5]);
            let t3 = transpose_concat_8x4(rows[3], rows[4], rows[5], rows[6]);
            let t4 = transpose_concat_8x4(rows[4], rows[5], rows[6], rows[7]);
            let t5 = transpose_concat_8x4(rows[5], rows[6], rows[7], rows[8]);
            let t6 = transpose_concat_8x4(rows[6], rows[7], rows[8], rows[9]);
            let t7 = transpose_concat_8x4(rows[7], rows[8], rows[9], rows[10]);

            let d0 = filter_four_rows_dotprod(t0.0, t4.0, t0.1, t4.1, filters, correction);
            let d1 = filter_four_rows_dotprod(t1.0, t5.0, t1.1, t5.1, filters, correction);
            let d2 = filter_four_rows_dotprod(t2.0, t6.0, t2.1, t6.1, filters, correction);
            let d3 = filter_four_rows_dotprod(t3.0, t7.0, t3.1, t7.1, filters, correction);
            vst1_u8(dst.add(y * dst_stride + x), d0);
            vst1_u8(dst.add((y + 1) * dst_stride + x), d1);
            vst1_u8(dst.add((y + 2) * dst_stride + x), d2);
            vst1_u8(dst.add((y + 3) * dst_stride + x), d3);
        }
        for y in vector_h..h {
            let samples = core::array::from_fn(|tap| vld1_u8(src.add((y + tap) * src_stride + x)));
            vst1_u8(dst.add(y * dst_stride + x), filter_eight(samples, filter));
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn convolve8_horiz_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let src_row = src.add(y * src_stride);
        let dst_row = dst.add(y * dst_stride);
        for x in (0..w).step_by(8) {
            let samples = core::array::from_fn(|tap| vld1_u8(src_row.add(x + tap)));
            vst1_u8(dst_row.add(x), filter_eight(samples, filter));
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn convolve8_vert_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let dst_row = dst.add(y * dst_stride);
        for x in (0..w).step_by(8) {
            let samples = core::array::from_fn(|tap| vld1_u8(src.add((y + tap) * src_stride + x)));
            vst1_u8(dst_row.add(x), filter_eight(samples, filter));
        }
    }
}

#[inline]
fn validate_horiz(
    src: &[u8],
    src_stride: usize,
    dst: &[u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    assert!(h > 0);
    assert!(w > 0 && w <= 64);
    assert!(src.len() >= (h - 1) * src_stride + w + 7);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
}

#[inline]
fn validate_vert(src: &[u8], src_stride: usize, dst: &[u8], dst_stride: usize, w: usize, h: usize) {
    assert!(h > 0);
    assert!(w > 0 && w <= 64);
    assert!(src.len() >= (h + 6) * src_stride + w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
}

/// NEON horizontal unscaled 8-tap convolution.
pub fn convolve8_horiz_neon(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    validate_horiz(src, src_stride, dst, dst_stride, w, h);
    if w == 4 {
        let narrow_filter = filter.iter().all(|&tap| i8::try_from(tap).is_ok());
        if std::arch::is_aarch64_feature_detected!("i8mm") && narrow_filter {
            // SAFETY: horizontal validation guarantees 11 readable bytes per
            // row, while the narrow loader reads exactly 8 + 4 bytes.
            unsafe {
                convolve8_horiz_i8mm_4_inner(
                    src.as_ptr(),
                    src_stride,
                    dst.as_mut_ptr(),
                    dst_stride,
                    filter,
                    h,
                );
            }
            return;
        }
    }
    if w < 8 {
        convolve8_horiz(src, src_stride, dst, dst_stride, filter, w, h);
        return;
    }
    let simd_w = w & !7;
    // SAFETY: Validation proves every eight- or sixteen-byte load and
    // eight-byte store is in-bounds.
    // SimdDispatch installs this wrapper only on NEON-capable AArch64 hosts.
    unsafe {
        let narrow_filter = filter.iter().all(|&tap| i8::try_from(tap).is_ok());
        if std::arch::is_aarch64_feature_detected!("i8mm") && narrow_filter {
            convolve8_horiz_i8mm_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        } else if std::arch::is_aarch64_feature_detected!("dotprod") && narrow_filter {
            convolve8_horiz_dotprod_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        } else {
            convolve8_horiz_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        }
    }
    if simd_w != w {
        convolve8_horiz(
            &src[simd_w..],
            src_stride,
            &mut dst[simd_w..],
            dst_stride,
            filter,
            w - simd_w,
            h,
        );
    }
}

/// NEON vertical unscaled 8-tap convolution.
pub fn convolve8_vert_neon(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter: &[i16; 8],
    w: usize,
    h: usize,
) {
    validate_vert(src, src_stride, dst, dst_stride, w, h);
    if w == 4 {
        let narrow_filter = filter.iter().all(|&tap| i8::try_from(tap).is_ok());
        // SAFETY: Vertical validation guarantees four readable source bytes
        // for each of the h + 7 rows and four writable destination bytes.
        unsafe {
            if std::arch::is_aarch64_feature_detected!("i8mm") && narrow_filter {
                convolve8_vert_i8mm_4_inner(
                    src.as_ptr(),
                    src_stride,
                    dst.as_mut_ptr(),
                    dst_stride,
                    filter,
                    h,
                );
            } else {
                convolve8_vert_neon_4_inner(
                    src.as_ptr(),
                    src_stride,
                    dst.as_mut_ptr(),
                    dst_stride,
                    filter,
                    h,
                );
            }
        }
        return;
    }
    if w < 8 {
        convolve8_vert(src, src_stride, dst, dst_stride, filter, w, h);
        return;
    }
    let simd_w = w & !7;
    // SAFETY: Validation proves every eight-byte load and store is in-bounds.
    // SimdDispatch installs this wrapper only on NEON-capable AArch64 hosts.
    unsafe {
        let narrow_filter = filter.iter().all(|&tap| i8::try_from(tap).is_ok());
        if std::arch::is_aarch64_feature_detected!("i8mm") && narrow_filter {
            convolve8_vert_i8mm_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        } else if std::arch::is_aarch64_feature_detected!("dotprod") && narrow_filter {
            convolve8_vert_dotprod_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        } else {
            convolve8_vert_inner(
                src.as_ptr(),
                src_stride,
                dst.as_mut_ptr(),
                dst_stride,
                filter,
                simd_w,
                h,
            );
        }
    }
    if simd_w != w {
        convolve8_vert(
            &src[simd_w..],
            src_stride,
            &mut dst[simd_w..],
            dst_stride,
            filter,
            w - simd_w,
            h,
        );
    }
}

/// NEON full 2-D separable unscaled 8-tap convolution.
#[allow(clippy::too_many_arguments)]
pub fn convolve8_neon(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!((1..=64).contains(&w));
    assert!((1..=64).contains(&h));
    if w < 8 && w != 4 {
        convolve8(src, src_stride, dst, dst_stride, filter_x, filter_y, w, h);
        return;
    }
    assert!(src.len() >= (h + 6) * src_stride + w + 7);
    assert!(dst.len() >= (h - 1) * dst_stride + w);

    CONV8_INTERMEDIATE.with_borrow_mut(|intermediate| {
        convolve8_horiz_neon(
            src,
            src_stride,
            intermediate,
            INTER_STRIDE,
            filter_x,
            w,
            h + 7,
        );
        convolve8_vert_neon(intermediate, INTER_STRIDE, dst, dst_stride, filter_y, w, h);
    });
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
    };

    const SHAPES: &[(usize, usize)] = &[
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
        (1, 1),
        (7, 3),
        (12, 5),
        (63, 9),
    ];

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 16) as u8
        }
    }

    #[test]
    fn single_axis_matches_scalar_all_filters_and_shapes() {
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut rng = Lcg32(0x434F_4E56);
        for &(w, h) in SHAPES {
            for kernel in kernels {
                for filter in kernel {
                    let src_stride_h = w + 7 + usize::from(rng.byte() & 7);
                    let src_stride_v = w + usize::from(rng.byte() & 7);
                    let dst_stride = w + usize::from(rng.byte() & 7);
                    let src_h: Vec<u8> = (0..((h - 1) * src_stride_h + w + 7))
                        .map(|_| rng.byte())
                        .collect();
                    let src_v: Vec<u8> = (0..((h + 6) * src_stride_v + w))
                        .map(|_| rng.byte())
                        .collect();
                    let dst_len = (h - 1) * dst_stride + w;
                    let mut scalar = vec![0u8; dst_len];
                    let mut neon = vec![0u8; dst_len];
                    convolve8_horiz(&src_h, src_stride_h, &mut scalar, dst_stride, filter, w, h);
                    convolve8_horiz_neon(&src_h, src_stride_h, &mut neon, dst_stride, filter, w, h);
                    assert_eq!(neon, scalar, "horizontal {w}x{h}, {filter:?}");

                    scalar.fill(0);
                    neon.fill(0);
                    convolve8_vert(&src_v, src_stride_v, &mut scalar, dst_stride, filter, w, h);
                    convolve8_vert_neon(&src_v, src_stride_v, &mut neon, dst_stride, filter, w, h);
                    assert_eq!(neon, scalar, "vertical {w}x{h}, {filter:?}");
                }
            }
        }
    }

    #[test]
    fn two_axis_matches_scalar_representative_filters() {
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut rng = Lcg32(0x325F_434F);
        for &(w, h) in SHAPES {
            for kernel_x in kernels {
                for kernel_y in kernels {
                    for &phase in &[0usize, 1, 4, 8, 12, 15] {
                        let src_stride = w + 7 + usize::from(rng.byte() & 7);
                        let dst_stride = w + usize::from(rng.byte() & 7);
                        let src: Vec<u8> = (0..((h + 6) * src_stride + w + 7))
                            .map(|_| rng.byte())
                            .collect();
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut scalar = vec![0u8; dst_len];
                        let mut neon = vec![0u8; dst_len];
                        convolve8(
                            &src,
                            src_stride,
                            &mut scalar,
                            dst_stride,
                            &kernel_x[phase],
                            &kernel_y[15 - phase],
                            w,
                            h,
                        );
                        convolve8_neon(
                            &src,
                            src_stride,
                            &mut neon,
                            dst_stride,
                            &kernel_x[phase],
                            &kernel_y[15 - phase],
                            w,
                            h,
                        );
                        assert_eq!(neon, scalar, "2-D {w}x{h}, phase {phase}");
                    }
                }
            }
        }
    }

    #[test]
    fn dispatch_routes_neon_and_scalar_caps() {
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::AArch64Caps;
        use crate::dsp::dispatch::{Convolve8OneAxisFn, Convolve8TwoAxisFn};

        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let neon_horiz: Convolve8OneAxisFn = convolve8_horiz_neon;
        let neon_vert: Convolve8OneAxisFn = convolve8_vert_neon;
        let neon_2d: Convolve8TwoAxisFn = convolve8_neon;
        assert_eq!(neon.convolve8_horiz() as usize, neon_horiz as usize);
        assert_eq!(neon.convolve8_vert() as usize, neon_vert as usize);
        assert_eq!(neon.convolve8() as usize, neon_2d as usize);

        let scalar = SimdDispatch::build(AArch64Caps {
            neon: false,
            dotprod: false,
        });
        let scalar_horiz: Convolve8OneAxisFn = convolve8_horiz;
        let scalar_vert: Convolve8OneAxisFn = convolve8_vert;
        let scalar_2d: Convolve8TwoAxisFn = convolve8;
        assert_eq!(scalar.convolve8_horiz() as usize, scalar_horiz as usize);
        assert_eq!(scalar.convolve8_vert() as usize, scalar_vert as usize);
        assert_eq!(scalar.convolve8() as usize, scalar_2d as usize);
    }
}
