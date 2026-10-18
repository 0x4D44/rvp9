//! 8-tap sub-pel convolution with compound average — x86_64 AVX2
//! (Wave J J6 / J7 / J8).
//!
//! The three `_avg` MC variants share one structure: produce the
//! motion-compensated prediction with the matching non-avg AVX2 kernel,
//! then round-average it against the existing `dst` (compound
//! prediction). The averaging step reuses [`convolve_avg_avx2`], whose
//! `dst = (dst + pred + 1) >> 1` is exactly libvpx's `ROUND_POWER_OF_TWO`
//! combine.
//!
//! | fn                          | prediction kernel        | scalar ref                          |
//! |-----------------------------|--------------------------|-------------------------------------|
//! | [`convolve8_avg_horiz_avx2`]| [`convolve8_horiz_avx2`] | `crate::inter::filter::convolve8_avg_horiz` |
//! | [`convolve8_avg_vert_avx2`] | [`convolve8_vert_avx2`]  | `crate::inter::filter::convolve8_avg_vert`  |
//! | [`convolve8_avg_avx2`]      | [`convolve8_avx2`]       | `crate::inter::filter::convolve8_avg`       |
//!
//! This composition is bit-identical to the scalar reference: the
//! scalar horiz/vert variants compute `round_avg(dst, round_clip(sum))`
//! per pixel, which equals computing the full clipped prediction first
//! and averaging afterwards (the per-pixel operations are independent).
//! The 2-D scalar `convolve8_avg` already follows the exact
//! convolve-to-scratch-then-average shape.
//!
//! The prediction scratch is `[u8; 64 * 64]` at stride `w` — large
//! enough for any VP9 inter block (`h * w ≤ 4096`).
//!
//! ## Contract
//!
//! Sub-pel filter rows only (taps fit i8) — inherited from the
//! prediction kernels.

#[cfg(target_arch = "x86_64")]
use super::{convolve_avg_avx2, convolve8_avx2, convolve8_horiz_avx2, convolve8_vert_avx2};

/// AVX2 8-tap horizontal convolution with compound average.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_avg_horiz_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(
        matches!(w, 4 | 8 | 16 | 32 | 64),
        "unsupported convolution width"
    );
    assert!((1..=64).contains(&h), "unsupported convolution height");
    let mut pred = [0u8; 64 * 64];
    convolve8_horiz_avx2(src, src_stride, &mut pred, w, filter_x, w, h);
    convolve_avg_avx2(&pred, w, dst, dst_stride, w, h);
}

/// AVX2 8-tap vertical convolution with compound average.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_avg_vert_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(
        matches!(w, 4 | 8 | 16 | 32 | 64),
        "unsupported convolution width"
    );
    assert!((1..=64).contains(&h), "unsupported convolution height");
    let mut pred = [0u8; 64 * 64];
    convolve8_vert_avx2(src, src_stride, &mut pred, w, filter_y, w, h);
    convolve_avg_avx2(&pred, w, dst, dst_stride, w, h);
}

/// AVX2 full 2-D 8-tap convolution with compound average.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
pub fn convolve8_avg_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(
        matches!(w, 4 | 8 | 16 | 32 | 64),
        "unsupported convolution width"
    );
    assert!((1..=64).contains(&h), "unsupported convolution height");
    let mut pred = [0u8; 64 * 64];
    convolve8_avx2(src, src_stride, &mut pred, w, filter_x, filter_y, w, h);
    convolve_avg_avx2(&pred, w, dst, dst_stride, w, h);
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Differential tests vs the scalar `_avg` references.
    use super::{convolve8_avg_avx2, convolve8_avg_horiz_avx2, convolve8_avg_vert_avx2};
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
        convolve8_avg, convolve8_avg_horiz, convolve8_avg_vert,
    };

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

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
    ];

    /// J6 — horizontal `_avg` differential. `dst` is pre-filled with
    /// random noise so the average step is genuinely exercised.
    #[test]
    fn avg_horiz_differential() {
        require_avx2!();
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0x2468_ACE0;
        for &(w, h) in SHAPES {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..8 {
                        let src_stride = w + 7 + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1) * src_stride + w + 7;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        for d in dst_scalar.iter_mut() {
                            *d = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_avx2 = dst_scalar.clone();
                        convolve8_avg_horiz(
                            &src,
                            src_stride,
                            &mut dst_scalar,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        convolve8_avg_horiz_avx2(
                            &src,
                            src_stride,
                            &mut dst_avx2,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        assert_eq!(dst_scalar, dst_avx2, "avg_horiz w={w} h={h} f={filter:?}");
                    }
                }
            }
        }
    }

    /// J7 — vertical `_avg` differential.
    #[test]
    fn avg_vert_differential() {
        require_avx2!();
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0x1111_2222;
        for &(w, h) in SHAPES {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..8 {
                        let src_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1 + 7) * src_stride + w;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        for d in dst_scalar.iter_mut() {
                            *d = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_avx2 = dst_scalar.clone();
                        convolve8_avg_vert(
                            &src,
                            src_stride,
                            &mut dst_scalar,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        convolve8_avg_vert_avx2(
                            &src,
                            src_stride,
                            &mut dst_avx2,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        assert_eq!(dst_scalar, dst_avx2, "avg_vert w={w} h={h} f={filter:?}");
                    }
                }
            }
        }
    }

    /// J8 — 2-D `_avg` differential with random (filter_x, filter_y).
    #[test]
    fn avg_2d_differential() {
        require_avx2!();
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0x3333_4444;
        for &(w, h) in SHAPES {
            for _iter in 0..128 {
                let kx = &kernels[(lcg_next(&mut state) as usize) % kernels.len()];
                let ky = &kernels[(lcg_next(&mut state) as usize) % kernels.len()];
                let filter_x = &kx[1 + (lcg_next(&mut state) as usize) % 15];
                let filter_y = &ky[1 + (lcg_next(&mut state) as usize) % 15];
                let src_stride = w + 7 + (lcg_next(&mut state) as usize & 0x7);
                let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                let src_len = (h - 1 + 7) * src_stride + w + 7;
                let dst_len = (h - 1) * dst_stride + w;
                let mut src = vec![0u8; src_len];
                for s in src.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut dst_scalar = vec![0u8; dst_len];
                for d in dst_scalar.iter_mut() {
                    *d = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut dst_avx2 = dst_scalar.clone();
                convolve8_avg(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    filter_x,
                    filter_y,
                    w,
                    h,
                );
                convolve8_avg_avx2(
                    &src,
                    src_stride,
                    &mut dst_avx2,
                    dst_stride,
                    filter_x,
                    filter_y,
                    w,
                    h,
                );
                assert_eq!(
                    dst_scalar, dst_avx2,
                    "avg_2d w={w} h={h} fx={filter_x:?} fy={filter_y:?}"
                );
            }
        }
    }

    /// Dispatch round-trip for all three avg slots.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);

        let fx = &EIGHTTAP_KERNEL[7];
        let fy = &EIGHTTAP_SMOOTH_KERNEL[9];
        let mut state: u32 = 0x5555_6666;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let s_stride = w + 7;
            let src_len = (h - 1 + 7) * s_stride + w + 7;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut seed = vec![0u8; h * w];
            for d in seed.iter_mut() {
                *d = (lcg_next(&mut state) & 0xFF) as u8;
            }

            // horiz
            let (mut a, mut b) = (seed.clone(), seed.clone());
            (d_avx2.convolve8_avg_horiz())(&src, s_stride, &mut a, w, fx, w, h);
            (d_scalar.convolve8_avg_horiz())(&src, s_stride, &mut b, w, fx, w, h);
            assert_eq!(a, b, "avg_horiz round-trip w={w} h={h}");

            // vert
            let (mut a, mut b) = (seed.clone(), seed.clone());
            (d_avx2.convolve8_avg_vert())(&src, s_stride, &mut a, w, fy, w, h);
            (d_scalar.convolve8_avg_vert())(&src, s_stride, &mut b, w, fy, w, h);
            assert_eq!(a, b, "avg_vert round-trip w={w} h={h}");

            // 2-D
            let (mut a, mut b) = (seed.clone(), seed.clone());
            (d_avx2.convolve8_avg())(&src, s_stride, &mut a, w, fx, fy, w, h);
            (d_scalar.convolve8_avg())(&src, s_stride, &mut b, w, fx, fy, w, h);
            assert_eq!(a, b, "avg_2d round-trip w={w} h={h}");
        }
    }
}
