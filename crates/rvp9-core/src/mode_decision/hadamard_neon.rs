//! AArch64 NEON Hadamard kernels.
//!
//! Integer-exact port of libvpx v1.14.1
//! `vpx_dsp/arm/hadamard_neon.c`.

use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
unsafe fn one_pass(a: &mut [int16x8_t; 8]) {
    let b0 = vaddq_s16(a[0], a[1]);
    let b1 = vsubq_s16(a[0], a[1]);
    let b2 = vaddq_s16(a[2], a[3]);
    let b3 = vsubq_s16(a[2], a[3]);
    let b4 = vaddq_s16(a[4], a[5]);
    let b5 = vsubq_s16(a[4], a[5]);
    let b6 = vaddq_s16(a[6], a[7]);
    let b7 = vsubq_s16(a[6], a[7]);

    let c0 = vaddq_s16(b0, b2);
    let c1 = vaddq_s16(b1, b3);
    let c2 = vsubq_s16(b0, b2);
    let c3 = vsubq_s16(b1, b3);
    let c4 = vaddq_s16(b4, b6);
    let c5 = vaddq_s16(b5, b7);
    let c6 = vsubq_s16(b4, b6);
    let c7 = vsubq_s16(b5, b7);

    a[0] = vaddq_s16(c0, c4);
    a[1] = vsubq_s16(c2, c6);
    a[2] = vsubq_s16(c0, c4);
    a[3] = vaddq_s16(c2, c6);
    a[4] = vaddq_s16(c3, c7);
    a[5] = vsubq_s16(c3, c7);
    a[6] = vsubq_s16(c1, c5);
    a[7] = vaddq_s16(c1, c5);
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn transpose_8x8(a: &mut [int16x8_t; 8]) {
    let b0e = vtrn1q_s16(a[0], a[1]);
    let b0o = vtrn2q_s16(a[0], a[1]);
    let b1e = vtrn1q_s16(a[2], a[3]);
    let b1o = vtrn2q_s16(a[2], a[3]);
    let b2e = vtrn1q_s16(a[4], a[5]);
    let b2o = vtrn2q_s16(a[4], a[5]);
    let b3e = vtrn1q_s16(a[6], a[7]);
    let b3o = vtrn2q_s16(a[6], a[7]);

    let c0e = vtrn1q_s32(vreinterpretq_s32_s16(b0e), vreinterpretq_s32_s16(b1e));
    let c0o = vtrn2q_s32(vreinterpretq_s32_s16(b0e), vreinterpretq_s32_s16(b1e));
    let c1e = vtrn1q_s32(vreinterpretq_s32_s16(b0o), vreinterpretq_s32_s16(b1o));
    let c1o = vtrn2q_s32(vreinterpretq_s32_s16(b0o), vreinterpretq_s32_s16(b1o));
    let c2e = vtrn1q_s32(vreinterpretq_s32_s16(b2e), vreinterpretq_s32_s16(b3e));
    let c2o = vtrn2q_s32(vreinterpretq_s32_s16(b2e), vreinterpretq_s32_s16(b3e));
    let c3e = vtrn1q_s32(vreinterpretq_s32_s16(b2o), vreinterpretq_s32_s16(b3o));
    let c3o = vtrn2q_s32(vreinterpretq_s32_s16(b2o), vreinterpretq_s32_s16(b3o));

    a[0] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c0e),
        vreinterpretq_s64_s32(c2e),
    ));
    a[4] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c0e),
        vreinterpretq_s64_s32(c2e),
    ));
    a[1] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c1e),
        vreinterpretq_s64_s32(c3e),
    ));
    a[5] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c1e),
        vreinterpretq_s64_s32(c3e),
    ));
    a[2] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c0o),
        vreinterpretq_s64_s32(c2o),
    ));
    a[6] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c0o),
        vreinterpretq_s64_s32(c2o),
    ));
    a[3] = vreinterpretq_s16_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(c1o),
        vreinterpretq_s64_s32(c3o),
    ));
    a[7] = vreinterpretq_s16_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(c1o),
        vreinterpretq_s64_s32(c3o),
    ));
}

#[target_feature(enable = "neon")]
unsafe fn hadamard_8x8_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The safe wrapper proves that each row has eight readable
    // values. AArch64 always provides NEON.
    let mut a = unsafe {
        [
            vld1q_s16(src),
            vld1q_s16(src.add(src_stride)),
            vld1q_s16(src.add(2 * src_stride)),
            vld1q_s16(src.add(3 * src_stride)),
            vld1q_s16(src.add(4 * src_stride)),
            vld1q_s16(src.add(5 * src_stride)),
            vld1q_s16(src.add(6 * src_stride)),
            vld1q_s16(src.add(7 * src_stride)),
        ]
    };
    // SAFETY: These helpers operate only on local NEON registers.
    unsafe {
        one_pass(&mut a);
        transpose_8x8(&mut a);
        one_pass(&mut a);
    }
    for (row, values) in a.into_iter().enumerate() {
        // SAFETY: The safe wrapper proves that coeff has 64 writable values.
        unsafe { vst1q_s16(coeff.add(row * 8), values) };
    }
}

#[target_feature(enable = "neon")]
unsafe fn hadamard_16x16_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The caller proves a complete 16x16 source and 256-value output.
    unsafe {
        hadamard_8x8_inner(src, src_stride, coeff);
        hadamard_8x8_inner(src.add(8), src_stride, coeff.add(64));
        hadamard_8x8_inner(src.add(8 * src_stride), src_stride, coeff.add(128));
        hadamard_8x8_inner(src.add(8 * src_stride + 8), src_stride, coeff.add(192));
    }

    for idx in (0..64).step_by(8) {
        // SAFETY: Each vector stays within the 256-value output.
        let (a0, a1, a2, a3) = unsafe {
            (
                vld1q_s16(coeff.add(idx)),
                vld1q_s16(coeff.add(64 + idx)),
                vld1q_s16(coeff.add(128 + idx)),
                vld1q_s16(coeff.add(192 + idx)),
            )
        };
        let b0 = vhaddq_s16(a0, a1);
        let b1 = vhsubq_s16(a0, a1);
        let b2 = vhaddq_s16(a2, a3);
        let b3 = vhsubq_s16(a2, a3);
        // SAFETY: Each vector stays within the 256-value output.
        unsafe {
            vst1q_s16(coeff.add(idx), vaddq_s16(b0, b2));
            vst1q_s16(coeff.add(64 + idx), vaddq_s16(b1, b3));
            vst1q_s16(coeff.add(128 + idx), vsubq_s16(b0, b2));
            vst1q_s16(coeff.add(192 + idx), vsubq_s16(b1, b3));
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn hadamard_32x32_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The caller proves a complete 32x32 source and 1024-value output.
    unsafe {
        hadamard_16x16_inner(src, src_stride, coeff);
        hadamard_16x16_inner(src.add(16), src_stride, coeff.add(256));
        hadamard_16x16_inner(src.add(16 * src_stride), src_stride, coeff.add(512));
        hadamard_16x16_inner(src.add(16 * src_stride + 16), src_stride, coeff.add(768));
    }

    for idx in (0..256).step_by(8) {
        // SAFETY: Each vector stays within the 1024-value output.
        let (a0, a1, a2, a3) = unsafe {
            (
                vld1q_s16(coeff.add(idx)),
                vld1q_s16(coeff.add(256 + idx)),
                vld1q_s16(coeff.add(512 + idx)),
                vld1q_s16(coeff.add(768 + idx)),
            )
        };
        let b0 = vshrq_n_s16::<1>(vhaddq_s16(a0, a1));
        let b1 = vshrq_n_s16::<1>(vhsubq_s16(a0, a1));
        let b2 = vshrq_n_s16::<1>(vhaddq_s16(a2, a3));
        let b3 = vshrq_n_s16::<1>(vhsubq_s16(a2, a3));
        // SAFETY: Each vector stays within the 1024-value output.
        unsafe {
            vst1q_s16(coeff.add(idx), vaddq_s16(b0, b2));
            vst1q_s16(coeff.add(256 + idx), vaddq_s16(b1, b3));
            vst1q_s16(coeff.add(512 + idx), vsubq_s16(b0, b2));
            vst1q_s16(coeff.add(768 + idx), vsubq_s16(b1, b3));
        }
    }
}

#[inline]
fn validate(src: &[i16], src_stride: usize, coeff: &[i16], size: usize) {
    assert!(src_stride >= size);
    assert!(src.len() >= (size - 1) * src_stride + size);
    assert!(coeff.len() >= size * size);
}

pub(super) fn hadamard_8x8(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    validate(src, src_stride, coeff, 8);
    // SAFETY: validate proves all pointer ranges; NEON is mandatory on AArch64.
    unsafe { hadamard_8x8_inner(src.as_ptr(), src_stride, coeff.as_mut_ptr()) };
}

pub(super) fn hadamard_16x16(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    validate(src, src_stride, coeff, 16);
    // SAFETY: validate proves all pointer ranges; NEON is mandatory on AArch64.
    unsafe { hadamard_16x16_inner(src.as_ptr(), src_stride, coeff.as_mut_ptr()) };
}

pub(super) fn hadamard_32x32(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    validate(src, src_stride, coeff, 32);
    // SAFETY: validate proves all pointer ranges; NEON is mandatory on AArch64.
    unsafe { hadamard_32x32_inner(src.as_ptr(), src_stride, coeff.as_mut_ptr()) };
}
