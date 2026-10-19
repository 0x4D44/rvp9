//! AArch64 NEON four-reference sum-of-absolute-differences kernels.
//!
//! Source-faithful port of libvpx v1.14.1 `vpx_dsp/arm/sad4d_neon.c`
//! and `sad4d_neon_dotprod.c` for VP9's 13 block shapes.

use crate::dsp::bounds::strided_len;

use core::arch::aarch64::*;

use super::variance_neon::unsigned_dot;

#[target_feature(enable = "neon")]
unsafe fn sad4d_neon_inner(
    src: *const u8,
    src_stride: usize,
    refs: [*const u8; 4],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> [u32; 4] {
    match w {
        4 => {
            let mut sums = [vdupq_n_u16(0); 4];
            for row in (0..h).step_by(2) {
                let s0 = core::ptr::read_unaligned(src.add(row * src_stride) as *const u32);
                let s1 = core::ptr::read_unaligned(src.add((row + 1) * src_stride) as *const u32);
                let s = vcreate_u8(u64::from(s0) | (u64::from(s1) << 32));
                for (reference, sum) in refs.iter().zip(sums.iter_mut()) {
                    let r0 =
                        core::ptr::read_unaligned(reference.add(row * ref_stride) as *const u32);
                    let r1 = core::ptr::read_unaligned(
                        reference.add((row + 1) * ref_stride) as *const u32
                    );
                    let r = vcreate_u8(u64::from(r0) | (u64::from(r1) << 32));
                    *sum = vabal_u8(*sum, s, r);
                }
            }
            sums.map(|sum| vaddlvq_u16(sum))
        }
        8 => {
            let mut sums = [vdupq_n_u16(0); 4];
            for row in 0..h {
                let s = vld1_u8(src.add(row * src_stride));
                for (reference, sum) in refs.iter().zip(sums.iter_mut()) {
                    *sum = vabal_u8(*sum, s, vld1_u8(reference.add(row * ref_stride)));
                }
            }
            sums.map(|sum| vaddlvq_u16(sum))
        }
        16 | 32 | 64 => {
            let mut sums = [vdupq_n_u32(0); 4];
            for row in 0..h {
                let src_row = src.add(row * src_stride);
                for col in (0..w).step_by(16) {
                    let s = vld1q_u8(src_row.add(col));
                    for (reference, sum) in refs.iter().zip(sums.iter_mut()) {
                        let r = vld1q_u8(reference.add(row * ref_stride + col));
                        *sum = vpadalq_u16(*sum, vpaddlq_u8(vabdq_u8(s, r)));
                    }
                }
            }
            sums.map(|sum| vaddvq_u32(sum))
        }
        _ => unreachable!("VP9 SAD width must be 4, 8, 16, 32, or 64"),
    }
}

#[target_feature(enable = "dotprod")]
unsafe fn sad4d_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    refs: [*const u8; 4],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> [u32; 4] {
    debug_assert!(w >= 16);
    let mut sums = [vdupq_n_u32(0); 4];
    let ones = vdupq_n_u8(1);
    for row in 0..h {
        let src_row = src.add(row * src_stride);
        for col in (0..w).step_by(16) {
            let s = vld1q_u8(src_row.add(col));
            for (reference, sum) in refs.iter().zip(sums.iter_mut()) {
                let r = vld1q_u8(reference.add(row * ref_stride + col));
                *sum = unsigned_dot(*sum, vabdq_u8(s, r), ones);
            }
        }
    }
    sums.map(|sum| vaddvq_u32(sum))
}

#[inline]
fn validate(
    src: &[u8],
    src_stride: usize,
    refs: &[&[u8]; 4],
    ref_stride: usize,
    w: usize,
    h: usize,
) {
    assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    assert!(h > 0 && h.is_multiple_of(2));
    assert!(src.len() >= strided_len(h - 1, src_stride, w));
    for reference in refs {
        assert!(reference.len() >= strided_len(h - 1, ref_stride, w));
    }
}

#[inline]
fn sad4d_neon(
    src: &[u8],
    src_stride: usize,
    refs: &[&[u8]; 4],
    ref_stride: usize,
    out: &mut [u32; 4],
    w: usize,
    h: usize,
) {
    validate(src, src_stride, refs, ref_stride, w, h);
    let ref_ptrs = refs.map(|reference| reference.as_ptr());
    // SAFETY: Validation proves every load in the fixed block footprints.
    // SimdDispatch installs these wrappers only on NEON-capable hosts.
    *out = unsafe { sad4d_neon_inner(src.as_ptr(), src_stride, ref_ptrs, ref_stride, w, h) };
}

#[inline]
fn sad4d_dotprod(
    src: &[u8],
    src_stride: usize,
    refs: &[&[u8]; 4],
    ref_stride: usize,
    out: &mut [u32; 4],
    w: usize,
    h: usize,
) {
    validate(src, src_stride, refs, ref_stride, w, h);
    assert!(w >= 16);
    let ref_ptrs = refs.map(|reference| reference.as_ptr());
    // SAFETY: Validation proves every load in the fixed block footprints.
    // SimdDispatch installs these wrappers only when dotprod is present.
    *out = unsafe { sad4d_dotprod_inner(src.as_ptr(), src_stride, ref_ptrs, ref_stride, w, h) };
}

macro_rules! sad4d_neon_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("NEON four-reference SAD for a ", stringify!($w), "x", stringify!($h), " block.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            refs: &[&[u8]; 4],
            ref_stride: usize,
            out: &mut [u32; 4],
        ) {
            sad4d_neon(src, src_stride, refs, ref_stride, out, $w, $h);
        }
    };
}

macro_rules! sad4d_dotprod_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("Dot-product four-reference SAD for a ", stringify!($w), "x", stringify!($h), " block.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            refs: &[&[u8]; 4],
            ref_stride: usize,
            out: &mut [u32; 4],
        ) {
            sad4d_dotprod(src, src_stride, refs, ref_stride, out, $w, $h);
        }
    };
}

sad4d_neon_fn!(vpx_sad4x4x4d_neon, 4, 4);
sad4d_neon_fn!(vpx_sad4x8x4d_neon, 4, 8);
sad4d_neon_fn!(vpx_sad8x4x4d_neon, 8, 4);
sad4d_neon_fn!(vpx_sad8x8x4d_neon, 8, 8);
sad4d_neon_fn!(vpx_sad8x16x4d_neon, 8, 16);
sad4d_neon_fn!(vpx_sad16x8x4d_neon, 16, 8);
sad4d_neon_fn!(vpx_sad16x16x4d_neon, 16, 16);
sad4d_neon_fn!(vpx_sad16x32x4d_neon, 16, 32);
sad4d_neon_fn!(vpx_sad32x16x4d_neon, 32, 16);
sad4d_neon_fn!(vpx_sad32x32x4d_neon, 32, 32);
sad4d_neon_fn!(vpx_sad32x64x4d_neon, 32, 64);
sad4d_neon_fn!(vpx_sad64x32x4d_neon, 64, 32);
sad4d_neon_fn!(vpx_sad64x64x4d_neon, 64, 64);

sad4d_dotprod_fn!(vpx_sad16x8x4d_dotprod, 16, 8);
sad4d_dotprod_fn!(vpx_sad16x16x4d_dotprod, 16, 16);
sad4d_dotprod_fn!(vpx_sad16x32x4d_dotprod, 16, 32);
sad4d_dotprod_fn!(vpx_sad32x16x4d_dotprod, 32, 16);
sad4d_dotprod_fn!(vpx_sad32x32x4d_dotprod, 32, 32);
sad4d_dotprod_fn!(vpx_sad32x64x4d_dotprod, 32, 64);
sad4d_dotprod_fn!(vpx_sad64x32x4d_dotprod, 64, 32);
sad4d_dotprod_fn!(vpx_sad64x64x4d_dotprod, 64, 64);

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::dispatch::Sad4dFn;
    use crate::dsp::distortion;

    const SHAPES: &[(BlockSize, usize, usize, Sad4dFn, Option<Sad4dFn>)] = &[
        (BlockSize::Block4x4, 4, 4, vpx_sad4x4x4d_neon, None),
        (BlockSize::Block4x8, 4, 8, vpx_sad4x8x4d_neon, None),
        (BlockSize::Block8x4, 8, 4, vpx_sad8x4x4d_neon, None),
        (BlockSize::Block8x8, 8, 8, vpx_sad8x8x4d_neon, None),
        (BlockSize::Block8x16, 8, 16, vpx_sad8x16x4d_neon, None),
        (
            BlockSize::Block16x8,
            16,
            8,
            vpx_sad16x8x4d_neon,
            Some(vpx_sad16x8x4d_dotprod),
        ),
        (
            BlockSize::Block16x16,
            16,
            16,
            vpx_sad16x16x4d_neon,
            Some(vpx_sad16x16x4d_dotprod),
        ),
        (
            BlockSize::Block16x32,
            16,
            32,
            vpx_sad16x32x4d_neon,
            Some(vpx_sad16x32x4d_dotprod),
        ),
        (
            BlockSize::Block32x16,
            32,
            16,
            vpx_sad32x16x4d_neon,
            Some(vpx_sad32x16x4d_dotprod),
        ),
        (
            BlockSize::Block32x32,
            32,
            32,
            vpx_sad32x32x4d_neon,
            Some(vpx_sad32x32x4d_dotprod),
        ),
        (
            BlockSize::Block32x64,
            32,
            64,
            vpx_sad32x64x4d_neon,
            Some(vpx_sad32x64x4d_dotprod),
        ),
        (
            BlockSize::Block64x32,
            64,
            32,
            vpx_sad64x32x4d_neon,
            Some(vpx_sad64x32x4d_dotprod),
        ),
        (
            BlockSize::Block64x64,
            64,
            64,
            vpx_sad64x64x4d_neon,
            Some(vpx_sad64x64x4d_dotprod),
        ),
    ];

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    #[test]
    fn differential_random_all_shapes_and_strides() {
        let mut rng = Lcg32(0x5341_4434);
        for &(_, w, h, neon, dotprod) in SHAPES {
            for jitter in 0..9 {
                let src_stride = w + jitter;
                let ref_stride = w + (jitter * 5 + 3) % 9;
                let src: Vec<u8> = (0..((h - 1) * src_stride + w))
                    .map(|_| rng.byte())
                    .collect();
                let ref_bufs: [Vec<u8>; 4] = core::array::from_fn(|_| {
                    (0..((h - 1) * ref_stride + w))
                        .map(|_| rng.byte())
                        .collect()
                });
                let refs = ref_bufs.each_ref().map(Vec::as_slice);
                let mut expected = [0; 4];
                let mut actual = [0; 4];
                distortion::sad4d(&src, src_stride, &refs, ref_stride, w, h, &mut expected);
                neon(&src, src_stride, &refs, ref_stride, &mut actual);
                assert_eq!(
                    actual, expected,
                    "NEON mismatch for {w}x{h}, jitter {jitter}"
                );
                if let Some(dotprod) = dotprod {
                    dotprod(&src, src_stride, &refs, ref_stride, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "dotprod mismatch for {w}x{h}, jitter {jitter}"
                    );
                }
            }
        }
    }

    #[test]
    fn differential_edge_content() {
        for &(_, w, h, neon, dotprod) in SHAPES {
            let zeros = vec![0u8; w * h];
            let max = vec![255u8; w * h];
            let refs = [&zeros[..], &max[..], &zeros[..], &max[..]];
            let expected = [0, (w * h * 255) as u32, 0, (w * h * 255) as u32];
            let mut actual = [0; 4];
            neon(&zeros, w, &refs, w, &mut actual);
            assert_eq!(actual, expected);
            if let Some(dotprod) = dotprod {
                dotprod(&zeros, w, &refs, w, &mut actual);
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn dispatch_routes_every_shape() {
        let dotprod = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: true,
        });
        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let scalar = SimdDispatch::build(AArch64Caps {
            neon: false,
            dotprod: false,
        });
        for &(bsize, _, _, neon_fn, dotprod_fn) in SHAPES {
            let expected_dotprod = dotprod_fn.unwrap_or(neon_fn);
            assert_eq!(dotprod.sad4d(bsize) as usize, expected_dotprod as usize);
            assert_eq!(neon.sad4d(bsize) as usize, neon_fn as usize);
            assert_ne!(scalar.sad4d(bsize) as usize, neon_fn as usize);
        }
    }
}
