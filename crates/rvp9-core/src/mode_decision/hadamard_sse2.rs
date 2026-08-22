//! x86-64 SSE2 Hadamard kernels.
//!
//! Integer-exact low-bit-depth port of libvpx v1.14.1
//! `vpx_dsp/x86/avg_intrin_sse2.c`.

use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn one_pass(a: &mut [__m128i; 8], transpose: bool) {
    let b0 = _mm_add_epi16(a[0], a[1]);
    let b1 = _mm_sub_epi16(a[0], a[1]);
    let b2 = _mm_add_epi16(a[2], a[3]);
    let b3 = _mm_sub_epi16(a[2], a[3]);
    let b4 = _mm_add_epi16(a[4], a[5]);
    let b5 = _mm_sub_epi16(a[4], a[5]);
    let b6 = _mm_add_epi16(a[6], a[7]);
    let b7 = _mm_sub_epi16(a[6], a[7]);

    let c0 = _mm_add_epi16(b0, b2);
    let c1 = _mm_add_epi16(b1, b3);
    let c2 = _mm_sub_epi16(b0, b2);
    let c3 = _mm_sub_epi16(b1, b3);
    let c4 = _mm_add_epi16(b4, b6);
    let c5 = _mm_add_epi16(b5, b7);
    let c6 = _mm_sub_epi16(b4, b6);
    let c7 = _mm_sub_epi16(b5, b7);

    if transpose {
        let d0 = _mm_add_epi16(c0, c4);
        let d7 = _mm_add_epi16(c1, c5);
        let d3 = _mm_add_epi16(c2, c6);
        let d4 = _mm_add_epi16(c3, c7);
        let d2 = _mm_sub_epi16(c0, c4);
        let d6 = _mm_sub_epi16(c1, c5);
        let d1 = _mm_sub_epi16(c2, c6);
        let d5 = _mm_sub_epi16(c3, c7);

        let e0 = _mm_unpacklo_epi16(d0, d1);
        let e1 = _mm_unpacklo_epi16(d2, d3);
        let e2 = _mm_unpackhi_epi16(d0, d1);
        let e3 = _mm_unpackhi_epi16(d2, d3);
        let e4 = _mm_unpacklo_epi16(d4, d5);
        let e5 = _mm_unpacklo_epi16(d6, d7);
        let e6 = _mm_unpackhi_epi16(d4, d5);
        let e7 = _mm_unpackhi_epi16(d6, d7);

        let f0 = _mm_unpacklo_epi32(e0, e1);
        let f1 = _mm_unpacklo_epi32(e4, e5);
        let f2 = _mm_unpackhi_epi32(e0, e1);
        let f3 = _mm_unpackhi_epi32(e4, e5);
        let f4 = _mm_unpacklo_epi32(e2, e3);
        let f5 = _mm_unpacklo_epi32(e6, e7);
        let f6 = _mm_unpackhi_epi32(e2, e3);
        let f7 = _mm_unpackhi_epi32(e6, e7);

        a[0] = _mm_unpacklo_epi64(f0, f1);
        a[1] = _mm_unpackhi_epi64(f0, f1);
        a[2] = _mm_unpacklo_epi64(f2, f3);
        a[3] = _mm_unpackhi_epi64(f2, f3);
        a[4] = _mm_unpacklo_epi64(f4, f5);
        a[5] = _mm_unpackhi_epi64(f4, f5);
        a[6] = _mm_unpacklo_epi64(f6, f7);
        a[7] = _mm_unpackhi_epi64(f6, f7);
    } else {
        a[0] = _mm_add_epi16(c0, c4);
        a[7] = _mm_add_epi16(c1, c5);
        a[3] = _mm_add_epi16(c2, c6);
        a[4] = _mm_add_epi16(c3, c7);
        a[2] = _mm_sub_epi16(c0, c4);
        a[6] = _mm_sub_epi16(c1, c5);
        a[1] = _mm_sub_epi16(c2, c6);
        a[5] = _mm_sub_epi16(c3, c7);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_8x8_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The safe wrapper proves every unaligned row load and output store.
    let mut a = unsafe {
        [
            _mm_loadu_si128(src.cast()),
            _mm_loadu_si128(src.add(src_stride).cast()),
            _mm_loadu_si128(src.add(2 * src_stride).cast()),
            _mm_loadu_si128(src.add(3 * src_stride).cast()),
            _mm_loadu_si128(src.add(4 * src_stride).cast()),
            _mm_loadu_si128(src.add(5 * src_stride).cast()),
            _mm_loadu_si128(src.add(6 * src_stride).cast()),
            _mm_loadu_si128(src.add(7 * src_stride).cast()),
        ]
    };
    // SAFETY: The helper operates only on local SSE2 registers.
    unsafe {
        one_pass(&mut a, true);
        one_pass(&mut a, false);
        for (row, values) in a.into_iter().enumerate() {
            _mm_storeu_si128(coeff.add(row * 8).cast(), values);
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_8x8_subtract_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    coeff: *mut i16,
) {
    let zero = _mm_setzero_si128();
    let mut a = [zero; 8];
    // SAFETY: The safe wrapper proves every eight-byte load and output store.
    unsafe {
        for (row, values) in a.iter_mut().enumerate() {
            let source = _mm_loadl_epi64(src.add(row * src_stride).cast());
            let prediction = _mm_loadl_epi64(pred.add(row * pred_stride).cast());
            *values = _mm_sub_epi16(
                _mm_unpacklo_epi8(source, zero),
                _mm_unpacklo_epi8(prediction, zero),
            );
        }
        one_pass(&mut a, true);
        one_pass(&mut a, false);
        for (row, values) in a.into_iter().enumerate() {
            _mm_storeu_si128(coeff.add(row * 8).cast(), values);
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn half_add(a: __m128i, b: __m128i) -> __m128i {
    let one = _mm_set1_epi16(1);
    _mm_add_epi16(
        _mm_add_epi16(_mm_srai_epi16(a, 1), _mm_srai_epi16(b, 1)),
        _mm_and_si128(_mm_and_si128(a, b), one),
    )
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn half_sub(a: __m128i, b: __m128i) -> __m128i {
    let one = _mm_set1_epi16(1);
    _mm_sub_epi16(
        _mm_sub_epi16(_mm_srai_epi16(a, 1), _mm_srai_epi16(b, 1)),
        _mm_and_si128(_mm_andnot_si128(a, b), one),
    )
}

#[target_feature(enable = "sse2")]
unsafe fn combine_16(coeff: *mut i16) {
    // SAFETY: The caller provides a complete 256-coefficient output.
    unsafe {
        for idx in (0..64).step_by(8) {
            let a0 = _mm_loadu_si128(coeff.add(idx).cast());
            let a1 = _mm_loadu_si128(coeff.add(64 + idx).cast());
            let a2 = _mm_loadu_si128(coeff.add(128 + idx).cast());
            let a3 = _mm_loadu_si128(coeff.add(192 + idx).cast());
            let b0 = half_add(a0, a1);
            let b1 = half_sub(a0, a1);
            let b2 = half_add(a2, a3);
            let b3 = half_sub(a2, a3);
            _mm_storeu_si128(coeff.add(idx).cast(), _mm_add_epi16(b0, b2));
            _mm_storeu_si128(coeff.add(64 + idx).cast(), _mm_add_epi16(b1, b3));
            _mm_storeu_si128(coeff.add(128 + idx).cast(), _mm_sub_epi16(b0, b2));
            _mm_storeu_si128(coeff.add(192 + idx).cast(), _mm_sub_epi16(b1, b3));
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_16x16_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The caller proves complete 16x16 input and output buffers.
    unsafe {
        hadamard_8x8_inner(src, src_stride, coeff);
        hadamard_8x8_inner(src.add(8), src_stride, coeff.add(64));
        hadamard_8x8_inner(src.add(8 * src_stride), src_stride, coeff.add(128));
        hadamard_8x8_inner(src.add(8 * src_stride + 8), src_stride, coeff.add(192));
        combine_16(coeff);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_16x16_subtract_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    coeff: *mut i16,
) {
    // SAFETY: The caller proves complete 16x16 input and output buffers.
    unsafe {
        hadamard_8x8_subtract_inner(src, src_stride, pred, pred_stride, coeff);
        hadamard_8x8_subtract_inner(
            src.add(8),
            src_stride,
            pred.add(8),
            pred_stride,
            coeff.add(64),
        );
        hadamard_8x8_subtract_inner(
            src.add(8 * src_stride),
            src_stride,
            pred.add(8 * pred_stride),
            pred_stride,
            coeff.add(128),
        );
        hadamard_8x8_subtract_inner(
            src.add(8 * src_stride + 8),
            src_stride,
            pred.add(8 * pred_stride + 8),
            pred_stride,
            coeff.add(192),
        );
        combine_16(coeff);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn combine_32(coeff: *mut i16) {
    // SAFETY: The caller provides a complete 1024-coefficient output.
    unsafe {
        for idx in (0..256).step_by(8) {
            let a0 = _mm_loadu_si128(coeff.add(idx).cast());
            let a1 = _mm_loadu_si128(coeff.add(256 + idx).cast());
            let a2 = _mm_loadu_si128(coeff.add(512 + idx).cast());
            let a3 = _mm_loadu_si128(coeff.add(768 + idx).cast());
            let b0 = _mm_srai_epi16(half_add(a0, a1), 1);
            let b1 = _mm_srai_epi16(half_sub(a0, a1), 1);
            let b2 = _mm_srai_epi16(half_add(a2, a3), 1);
            let b3 = _mm_srai_epi16(half_sub(a2, a3), 1);
            _mm_storeu_si128(coeff.add(idx).cast(), _mm_add_epi16(b0, b2));
            _mm_storeu_si128(coeff.add(256 + idx).cast(), _mm_add_epi16(b1, b3));
            _mm_storeu_si128(coeff.add(512 + idx).cast(), _mm_sub_epi16(b0, b2));
            _mm_storeu_si128(coeff.add(768 + idx).cast(), _mm_sub_epi16(b1, b3));
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_32x32_inner(src: *const i16, src_stride: usize, coeff: *mut i16) {
    // SAFETY: The caller proves complete 32x32 input and output buffers.
    unsafe {
        hadamard_16x16_inner(src, src_stride, coeff);
        hadamard_16x16_inner(src.add(16), src_stride, coeff.add(256));
        hadamard_16x16_inner(src.add(16 * src_stride), src_stride, coeff.add(512));
        hadamard_16x16_inner(src.add(16 * src_stride + 16), src_stride, coeff.add(768));
        combine_32(coeff);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn hadamard_32x32_subtract_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    coeff: *mut i16,
) {
    // SAFETY: The caller proves complete 32x32 input and output buffers.
    unsafe {
        hadamard_16x16_subtract_inner(src, src_stride, pred, pred_stride, coeff);
        hadamard_16x16_subtract_inner(
            src.add(16),
            src_stride,
            pred.add(16),
            pred_stride,
            coeff.add(256),
        );
        hadamard_16x16_subtract_inner(
            src.add(16 * src_stride),
            src_stride,
            pred.add(16 * pred_stride),
            pred_stride,
            coeff.add(512),
        );
        hadamard_16x16_subtract_inner(
            src.add(16 * src_stride + 16),
            src_stride,
            pred.add(16 * pred_stride + 16),
            pred_stride,
            coeff.add(768),
        );
        combine_32(coeff);
    }
}

#[inline]
fn validate(src: &[i16], src_stride: usize, coeff: &[i16], size: usize) {
    assert!(src_stride >= size);
    assert!(src.len() >= (size - 1) * src_stride + size);
    assert!(coeff.len() >= size * size);
}

fn validate_subtract(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    coeff: &[i16],
    size: usize,
) {
    assert!(src_stride >= size);
    assert!(pred_stride >= size);
    assert!(src.len() >= (size - 1) * src_stride + size);
    assert!(pred.len() >= (size - 1) * pred_stride + size);
    assert!(coeff.len() >= size * size);
}

macro_rules! transform_wrapper {
    ($name:ident, $inner:ident, $size:literal) => {
        pub(super) fn $name(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
            validate(src, src_stride, coeff, $size);
            // SAFETY: x86-64 guarantees SSE2; validation proves every pointer range.
            unsafe { $inner(src.as_ptr(), src_stride, coeff.as_mut_ptr()) };
        }
    };
}

macro_rules! subtract_wrapper {
    ($name:ident, $inner:ident, $size:literal) => {
        pub(super) fn $name(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
            coeff: &mut [i16],
        ) {
            validate_subtract(src, src_stride, pred, pred_stride, coeff, $size);
            // SAFETY: x86-64 guarantees SSE2; validation proves every pointer range.
            unsafe {
                $inner(
                    src.as_ptr(),
                    src_stride,
                    pred.as_ptr(),
                    pred_stride,
                    coeff.as_mut_ptr(),
                )
            };
        }
    };
}

transform_wrapper!(hadamard_8x8, hadamard_8x8_inner, 8);
transform_wrapper!(hadamard_16x16, hadamard_16x16_inner, 16);
transform_wrapper!(hadamard_32x32, hadamard_32x32_inner, 32);
subtract_wrapper!(hadamard_8x8_subtract, hadamard_8x8_subtract_inner, 8);
subtract_wrapper!(hadamard_16x16_subtract, hadamard_16x16_subtract_inner, 16);
subtract_wrapper!(hadamard_32x32_subtract, hadamard_32x32_subtract_inner, 32);
