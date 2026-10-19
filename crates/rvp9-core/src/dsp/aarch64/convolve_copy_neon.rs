//! AArch64 NEON integer-pel block copy.
//!
//! Integer motion vectors need no interpolation. This mirrors pinned
//! libvpx's `vpx_convolve_copy_neon`: copy each active prediction row with
//! fixed-width loads and stores instead of calling `memmove` once per row.

use crate::dsp::bounds::strided_len;

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
unsafe fn convolve_copy_neon_inner(
    src: *const u8,
    src_stride: usize,
    dst: *mut u8,
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    match w {
        4 => {
            let mut row = 0;
            while row + 1 < h {
                let value_0 = core::ptr::read_unaligned(src.add(row * src_stride).cast::<u32>());
                let value_1 =
                    core::ptr::read_unaligned(src.add((row + 1) * src_stride).cast::<u32>());
                core::ptr::write_unaligned(dst.add(row * dst_stride).cast::<u32>(), value_0);
                core::ptr::write_unaligned(dst.add((row + 1) * dst_stride).cast::<u32>(), value_1);
                row += 2;
            }
            if row < h {
                let value = core::ptr::read_unaligned(src.add(row * src_stride).cast::<u32>());
                core::ptr::write_unaligned(dst.add(row * dst_stride).cast::<u32>(), value);
            }
        }
        8 => {
            let mut row = 0;
            while row + 1 < h {
                let value_0 = vld1_u8(src.add(row * src_stride));
                let value_1 = vld1_u8(src.add((row + 1) * src_stride));
                vst1_u8(dst.add(row * dst_stride), value_0);
                vst1_u8(dst.add((row + 1) * dst_stride), value_1);
                row += 2;
            }
            if row < h {
                vst1_u8(
                    dst.add(row * dst_stride),
                    vld1_u8(src.add(row * src_stride)),
                );
            }
        }
        16 => {
            let mut row = 0;
            while row + 1 < h {
                let value_0 = vld1q_u8(src.add(row * src_stride));
                let value_1 = vld1q_u8(src.add((row + 1) * src_stride));
                vst1q_u8(dst.add(row * dst_stride), value_0);
                vst1q_u8(dst.add((row + 1) * dst_stride), value_1);
                row += 2;
            }
            if row < h {
                vst1q_u8(
                    dst.add(row * dst_stride),
                    vld1q_u8(src.add(row * src_stride)),
                );
            }
        }
        32 => {
            let mut row = 0;
            while row + 1 < h {
                let src_0 = src.add(row * src_stride);
                let src_1 = src.add((row + 1) * src_stride);
                let dst_0 = dst.add(row * dst_stride);
                let dst_1 = dst.add((row + 1) * dst_stride);
                let value_00 = vld1q_u8(src_0);
                let value_01 = vld1q_u8(src_0.add(16));
                let value_10 = vld1q_u8(src_1);
                let value_11 = vld1q_u8(src_1.add(16));
                vst1q_u8(dst_0, value_00);
                vst1q_u8(dst_0.add(16), value_01);
                vst1q_u8(dst_1, value_10);
                vst1q_u8(dst_1.add(16), value_11);
                row += 2;
            }
            if row < h {
                let src_row = src.add(row * src_stride);
                let dst_row = dst.add(row * dst_stride);
                vst1q_u8(dst_row, vld1q_u8(src_row));
                vst1q_u8(dst_row.add(16), vld1q_u8(src_row.add(16)));
            }
        }
        64 => {
            for row in 0..h {
                let src_row = src.add(row * src_stride);
                let dst_row = dst.add(row * dst_stride);
                vst1q_u8(dst_row, vld1q_u8(src_row));
                vst1q_u8(dst_row.add(16), vld1q_u8(src_row.add(16)));
                vst1q_u8(dst_row.add(32), vld1q_u8(src_row.add(32)));
                vst1q_u8(dst_row.add(48), vld1q_u8(src_row.add(48)));
            }
        }
        _ => unreachable!("validated VP9 prediction width"),
    }
}

/// Copy an integer-pel VP9 prediction block with fixed-width NEON accesses.
pub fn convolve_copy_neon(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    if !matches!(w, 4 | 8 | 16 | 32 | 64) {
        crate::inter::filter::convolve_copy(src, src_stride, dst, dst_stride, w, h);
        return;
    }
    assert!(src_stride >= w);
    assert!(dst_stride >= w);
    assert!(src.len() >= strided_len(h.saturating_sub(1), src_stride, w));
    assert!(dst.len() >= strided_len(h.saturating_sub(1), dst_stride, w));

    // SAFETY: AArch64 dispatch installs this wrapper only when NEON is
    // available. The width match and slice-footprint assertions prove every
    // fixed-width row load and store remains in bounds.
    unsafe {
        convolve_copy_neon_inner(src.as_ptr(), src_stride, dst.as_mut_ptr(), dst_stride, w, h);
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    #[test]
    fn differential_all_vp9_shapes_and_strides() {
        let shapes = [
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
            (4, 5),
            (8, 5),
            (16, 5),
            (32, 5),
            (64, 5),
        ];
        let mut rng = Lcg32(0xC0FF_EE42);
        for (w, h) in shapes {
            for jitter in 0..17 {
                let src_stride = w + jitter;
                let dst_stride = w + (jitter * 5 + 3) % 17;
                let src_len = (h - 1) * src_stride + w;
                let dst_len = (h - 1) * dst_stride + w;
                let src: Vec<u8> = (0..src_len).map(|_| rng.byte()).collect();
                let mut expected: Vec<u8> = (0..dst_len).map(|_| rng.byte()).collect();
                let mut actual = expected.clone();

                crate::inter::filter::convolve_copy(
                    &src,
                    src_stride,
                    &mut expected,
                    dst_stride,
                    w,
                    h,
                );
                convolve_copy_neon(&src, src_stride, &mut actual, dst_stride, w, h);

                assert_eq!(actual, expected, "mismatch at {w}x{h}, jitter {jitter}");
            }
        }
    }

    #[test]
    fn dispatch_routes_neon_and_scalar_caps() {
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::AArch64Caps;

        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let scalar = SimdDispatch::build(AArch64Caps::default());
        let neon_fn: fn(&[u8], usize, &mut [u8], usize, usize, usize) = convolve_copy_neon;
        let scalar_fn: fn(&[u8], usize, &mut [u8], usize, usize, usize) =
            crate::inter::filter::convolve_copy;

        assert_eq!(neon.convolve_copy() as usize, neon_fn as usize);
        assert_eq!(scalar.convolve_copy() as usize, scalar_fn as usize);
    }
}
