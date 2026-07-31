//! AArch64 NEON unscaled VP9 8-tap interpolation.
//!
//! Each vector lane computes one destination pixel with the same i32
//! accumulation, `+64`, arithmetic `>> 7`, and u8 saturation as the scalar
//! oracle. The full 2-D kernel composes the exact horizontal and vertical
//! passes through a reusable per-thread u8 intermediate.

use core::arch::aarch64::*;

use crate::inter::filter::{convolve8, convolve8_horiz, convolve8_vert};

const INTER_STRIDE: usize = 64;
const INTER_ROWS: usize = 72;

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
    if w < 8 {
        convolve8_horiz(src, src_stride, dst, dst_stride, filter, w, h);
        return;
    }
    let simd_w = w & !7;
    // SAFETY: Validation proves every eight-byte load and store is in-bounds.
    // SimdDispatch installs this wrapper only on NEON-capable AArch64 hosts.
    unsafe {
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
    if w < 8 {
        convolve8_vert(src, src_stride, dst, dst_stride, filter, w, h);
        return;
    }
    let simd_w = w & !7;
    // SAFETY: Validation proves every eight-byte load and store is in-bounds.
    // SimdDispatch installs this wrapper only on NEON-capable AArch64 hosts.
    unsafe {
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
    if w < 8 {
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

#[cfg(test)]
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

        let neon = SimdDispatch::build(AArch64Caps { neon: true });
        let neon_horiz: Convolve8OneAxisFn = convolve8_horiz_neon;
        let neon_vert: Convolve8OneAxisFn = convolve8_vert_neon;
        let neon_2d: Convolve8TwoAxisFn = convolve8_neon;
        assert_eq!(neon.convolve8_horiz() as usize, neon_horiz as usize);
        assert_eq!(neon.convolve8_vert() as usize, neon_vert as usize);
        assert_eq!(neon.convolve8() as usize, neon_2d as usize);

        let scalar = SimdDispatch::build(AArch64Caps { neon: false });
        let scalar_horiz: Convolve8OneAxisFn = convolve8_horiz;
        let scalar_vert: Convolve8OneAxisFn = convolve8_vert;
        let scalar_2d: Convolve8TwoAxisFn = convolve8;
        assert_eq!(scalar.convolve8_horiz() as usize, scalar_horiz as usize);
        assert_eq!(scalar.convolve8_vert() as usize, scalar_vert as usize);
        assert_eq!(scalar.convolve8() as usize, scalar_2d as usize);
    }
}
