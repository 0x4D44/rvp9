//! AArch64 NEON sum-of-absolute-differences kernels.
//!
//! Source-faithful port of libvpx v1.14.1 `vpx_dsp/arm/sad_neon.c` for
//! VP9's 13 block shapes.

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn sad_neon_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    match w {
        4 => {
            let mut sum = vdupq_n_u16(0);
            for row in (0..h).step_by(2) {
                let s0 = core::ptr::read_unaligned(src.add(row * src_stride) as *const u32);
                let s1 = core::ptr::read_unaligned(src.add((row + 1) * src_stride) as *const u32);
                let r0 = core::ptr::read_unaligned(reference.add(row * ref_stride) as *const u32);
                let r1 =
                    core::ptr::read_unaligned(reference.add((row + 1) * ref_stride) as *const u32);
                let s = vcreate_u8(u64::from(s0) | (u64::from(s1) << 32));
                let r = vcreate_u8(u64::from(r0) | (u64::from(r1) << 32));
                sum = vabal_u8(sum, s, r);
            }
            vaddlvq_u16(sum)
        }
        8 => {
            let mut sum = vdupq_n_u16(0);
            for row in 0..h {
                let s = vld1_u8(src.add(row * src_stride));
                let r = vld1_u8(reference.add(row * ref_stride));
                sum = vabal_u8(sum, s, r);
            }
            vaddlvq_u16(sum)
        }
        16 => {
            let mut sum = vdupq_n_u16(0);
            for row in 0..h {
                let s = vld1q_u8(src.add(row * src_stride));
                let r = vld1q_u8(reference.add(row * ref_stride));
                sum = vpadalq_u8(sum, vabdq_u8(s, r));
            }
            vaddlvq_u16(sum)
        }
        32 => {
            let mut sums = [vdupq_n_u16(0), vdupq_n_u16(0)];
            for row in 0..h {
                let s = src.add(row * src_stride);
                let r = reference.add(row * ref_stride);
                let d0 = vabdq_u8(vld1q_u8(s), vld1q_u8(r));
                let d1 = vabdq_u8(vld1q_u8(s.add(16)), vld1q_u8(r.add(16)));
                sums[0] = vpadalq_u8(sums[0], d0);
                sums[1] = vpadalq_u8(sums[1], d1);
            }
            let mut sum = vpaddlq_u16(sums[0]);
            sum = vpadalq_u16(sum, sums[1]);
            vaddvq_u32(sum)
        }
        64 => {
            let mut sums = [
                vdupq_n_u16(0),
                vdupq_n_u16(0),
                vdupq_n_u16(0),
                vdupq_n_u16(0),
            ];
            for row in 0..h {
                let s = src.add(row * src_stride);
                let r = reference.add(row * ref_stride);
                for (chunk, sum) in sums.iter_mut().enumerate() {
                    let offset = chunk * 16;
                    let d = vabdq_u8(vld1q_u8(s.add(offset)), vld1q_u8(r.add(offset)));
                    *sum = vpadalq_u8(*sum, d);
                }
            }
            let mut sum = vpaddlq_u16(sums[0]);
            sum = vpadalq_u16(sum, sums[1]);
            sum = vpadalq_u16(sum, sums[2]);
            sum = vpadalq_u16(sum, sums[3]);
            vaddvq_u32(sum)
        }
        _ => unreachable!("VP9 SAD width must be 4, 8, 16, 32, or 64"),
    }
}

#[inline]
fn validate(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) {
    assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    assert!(h > 0 && h.is_multiple_of(2));
    assert!(src.len() >= (h - 1) * src_stride + w);
    assert!(reference.len() >= (h - 1) * ref_stride + w);
}

#[inline]
fn sad_neon(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    validate(src, src_stride, reference, ref_stride, w, h);
    // SAFETY: Validation proves every load in the fixed block footprint.
    // SimdDispatch installs these wrappers only on NEON-capable hosts.
    unsafe {
        sad_neon_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            w,
            h,
        )
    }
}

macro_rules! sad_neon_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("NEON SAD for a ", stringify!($w), "x", stringify!($h), " block.")]
        pub fn $name(src: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize) -> u32 {
            sad_neon(src, src_stride, reference, ref_stride, $w, $h)
        }
    };
}

sad_neon_fn!(vpx_sad4x4_neon, 4, 4);
sad_neon_fn!(vpx_sad4x8_neon, 4, 8);
sad_neon_fn!(vpx_sad8x4_neon, 8, 4);
sad_neon_fn!(vpx_sad8x8_neon, 8, 8);
sad_neon_fn!(vpx_sad8x16_neon, 8, 16);
sad_neon_fn!(vpx_sad16x8_neon, 16, 8);
sad_neon_fn!(vpx_sad16x16_neon, 16, 16);
sad_neon_fn!(vpx_sad16x32_neon, 16, 32);
sad_neon_fn!(vpx_sad32x16_neon, 32, 16);
sad_neon_fn!(vpx_sad32x32_neon, 32, 32);
sad_neon_fn!(vpx_sad32x64_neon, 32, 64);
sad_neon_fn!(vpx_sad64x32_neon, 64, 32);
sad_neon_fn!(vpx_sad64x64_neon, 64, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::dispatch::SadFn;
    use crate::dsp::distortion;

    const SHAPES: &[(BlockSize, usize, usize, SadFn)] = &[
        (BlockSize::Block4x4, 4, 4, vpx_sad4x4_neon),
        (BlockSize::Block4x8, 4, 8, vpx_sad4x8_neon),
        (BlockSize::Block8x4, 8, 4, vpx_sad8x4_neon),
        (BlockSize::Block8x8, 8, 8, vpx_sad8x8_neon),
        (BlockSize::Block8x16, 8, 16, vpx_sad8x16_neon),
        (BlockSize::Block16x8, 16, 8, vpx_sad16x8_neon),
        (BlockSize::Block16x16, 16, 16, vpx_sad16x16_neon),
        (BlockSize::Block16x32, 16, 32, vpx_sad16x32_neon),
        (BlockSize::Block32x16, 32, 16, vpx_sad32x16_neon),
        (BlockSize::Block32x32, 32, 32, vpx_sad32x32_neon),
        (BlockSize::Block32x64, 32, 64, vpx_sad32x64_neon),
        (BlockSize::Block64x32, 64, 32, vpx_sad64x32_neon),
        (BlockSize::Block64x64, 64, 64, vpx_sad64x64_neon),
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
        let mut rng = Lcg32(0x5341_4431);
        for &(_, w, h, simd) in SHAPES {
            for jitter in 0..9 {
                let src_stride = w + jitter;
                let ref_stride = w + (jitter * 5 + 3) % 9;
                let src: Vec<u8> = (0..((h - 1) * src_stride + w))
                    .map(|_| rng.byte())
                    .collect();
                let reference: Vec<u8> = (0..((h - 1) * ref_stride + w))
                    .map(|_| rng.byte())
                    .collect();
                assert_eq!(
                    simd(&src, src_stride, &reference, ref_stride),
                    distortion::sad(&src, src_stride, &reference, ref_stride, w, h),
                    "mismatch for {w}x{h}, jitter {jitter}"
                );
            }
        }
    }

    #[test]
    fn differential_edge_content() {
        for &(_, w, h, simd) in SHAPES {
            let zeros = vec![0u8; w * h];
            let max = vec![255u8; w * h];
            assert_eq!(simd(&zeros, w, &zeros, w), 0);
            assert_eq!(simd(&zeros, w, &max, w), (w * h * 255) as u32);
            assert_eq!(simd(&max, w, &zeros, w), (w * h * 255) as u32);
        }
    }

    #[test]
    fn dispatch_routes_every_shape() {
        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let scalar = SimdDispatch::build(AArch64Caps {
            neon: false,
            dotprod: false,
        });
        for &(bsize, _, _, simd) in SHAPES {
            assert_eq!(neon.sad(bsize) as usize, simd as usize);
            assert_ne!(scalar.sad(bsize) as usize, simd as usize);
        }
    }
}
