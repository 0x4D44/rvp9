//! NEON sub-pixel variance for VP9's 16/32/64-wide block shapes.
//!
//! This follows libvpx `vpx_dsp/arm/subpel_variance_neon.c`, with one
//! deliberate fusion: the second bilinear pass feeds the `(sum, sse)`
//! reduction directly. Two rolling 64-byte rows replace libvpx's full
//! intermediate blocks, avoiding heap allocation and an extra pixel walk.

use core::arch::aarch64::*;

/// Apply one VP9 two-tap bilinear filter to 16 byte lanes.
///
/// libvpx stores taps scaled by 128. Every tap is a multiple of 16, so this
/// equivalent form uses weights `(8 - offset, offset)` and rounds by 3 bits.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_16(a: *const u8, b: *const u8, offset: usize) -> uint8x16_t {
    let av = vld1q_u8(a);
    if offset == 0 {
        return av;
    }
    let bv = vld1q_u8(b);
    if offset == 4 {
        return vrhaddq_u8(av, bv);
    }

    let f0 = vdup_n_u8((8 - offset) as u8);
    let f1 = vdup_n_u8(offset as u8);
    let lo = vmlal_u8(vmull_u8(vget_low_u8(av), f0), vget_low_u8(bv), f1);
    let hi = vmlal_u8(vmull_u8(vget_high_u8(av), f0), vget_high_u8(bv), f1);
    vcombine_u8(vrshrn_n_u16::<3>(lo), vrshrn_n_u16::<3>(hi))
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn horizontal_row(src: *const u8, dst: *mut u8, width: usize, xoffset: usize) {
    let mut col = 0;
    while col < width {
        let filtered = filter_16(src.add(col), src.add(col + 1), xoffset);
        vst1q_u8(dst.add(col), filtered);
        col += 16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn accumulate_16(pred: uint8x16_t, reference: *const u8, accum: &mut VarianceAccumulator) {
    accum.add(pred, vld1q_u8(reference));
}

struct VarianceAccumulator {
    sum_lo: int32x4_t,
    sum_hi: int32x4_t,
    sse_0: int32x4_t,
    sse_1: int32x4_t,
    sse_2: int32x4_t,
    sse_3: int32x4_t,
}

impl VarianceAccumulator {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn new() -> Self {
        Self {
            sum_lo: vdupq_n_s32(0),
            sum_hi: vdupq_n_s32(0),
            sse_0: vdupq_n_s32(0),
            sse_1: vdupq_n_s32(0),
            sse_2: vdupq_n_s32(0),
            sse_3: vdupq_n_s32(0),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add(&mut self, pred: uint8x16_t, reference: uint8x16_t) {
        let diff_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(pred), vget_low_u8(reference)));
        let diff_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(pred), vget_high_u8(reference)));

        self.sum_lo = vpadalq_s16(self.sum_lo, diff_lo);
        self.sum_hi = vpadalq_s16(self.sum_hi, diff_hi);
        self.sse_0 = vmlal_s16(self.sse_0, vget_low_s16(diff_lo), vget_low_s16(diff_lo));
        self.sse_1 = vmlal_s16(self.sse_1, vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        self.sse_2 = vmlal_s16(self.sse_2, vget_low_s16(diff_hi), vget_low_s16(diff_hi));
        self.sse_3 = vmlal_s16(self.sse_3, vget_high_s16(diff_hi), vget_high_s16(diff_hi));
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn finish(self) -> (i64, i64) {
        let sum = vaddvq_s32(vaddq_s32(self.sum_lo, self.sum_hi));
        let sse = vaddvq_s32(vaddq_s32(
            vaddq_s32(self.sse_0, self.sse_1),
            vaddq_s32(self.sse_2, self.sse_3),
        ));
        (i64::from(sum), i64::from(sse))
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_pixel_variance_neon_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    width: usize,
    height: usize,
) -> (u32, u32) {
    let mut accum = VarianceAccumulator::new();

    if xoffset == 0 {
        for row in 0..height {
            let top = src.add(row * src_stride);
            let bottom = top.add(src_stride);
            let ref_row = reference.add(row * ref_stride);
            let mut col = 0;
            while col < width {
                let pred = filter_16(top.add(col), bottom.add(col), yoffset);
                accumulate_16(pred, ref_row.add(col), &mut accum);
                col += 16;
            }
        }
    } else if yoffset == 0 {
        for row in 0..height {
            let src_row = src.add(row * src_stride);
            let ref_row = reference.add(row * ref_stride);
            let mut col = 0;
            while col < width {
                let pred = filter_16(src_row.add(col), src_row.add(col + 1), xoffset);
                accumulate_16(pred, ref_row.add(col), &mut accum);
                col += 16;
            }
        }
    } else {
        let mut row_a = [0u8; 64];
        let mut row_b = [0u8; 64];
        horizontal_row(src, row_a.as_mut_ptr(), width, xoffset);

        for row in 0..height {
            horizontal_row(
                src.add((row + 1) * src_stride),
                row_b.as_mut_ptr(),
                width,
                xoffset,
            );
            let ref_row = reference.add(row * ref_stride);
            let mut col = 0;
            while col < width {
                let pred = filter_16(row_a.as_ptr().add(col), row_b.as_ptr().add(col), yoffset);
                accumulate_16(pred, ref_row.add(col), &mut accum);
                col += 16;
            }
            core::mem::swap(&mut row_a, &mut row_b);
        }
    }

    let (sum, sse) = accum.finish();
    let variance = (sse - sum * sum / (width * height) as i64).max(0) as u32;
    (variance, sse as u32)
}

/// Generic safe NEON entry point used by the per-size wrappers.
#[allow(clippy::too_many_arguments)]
pub fn sub_pixel_variance_neon(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    width: usize,
    height: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(matches!(width, 16 | 32 | 64));
    assert!(matches!(height, 8 | 16 | 32 | 64));
    assert!(src_stride >= width + usize::from(xoffset != 0));
    assert!(ref_stride >= width);
    assert!(src.len() >= height * src_stride + width + usize::from(xoffset != 0));
    assert!(reference.len() >= (height - 1) * ref_stride + width);

    // SAFETY: AArch64 dispatch installs this wrapper only when NEON is
    // detected. The assertions above prove every vector load stays within
    // the supplied slices: horizontal filtering needs one extra source byte;
    // vertical filtering needs the extra source row; reference loads cover
    // exactly `width` bytes per output row.
    unsafe {
        sub_pixel_variance_neon_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
            width,
            height,
        )
    }
}

macro_rules! subpel_var_neon_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("NEON sub-pixel variance ", stringify!($w), "x", stringify!($h), ".")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            xoffset: usize,
            yoffset: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> (u32, u32) {
            sub_pixel_variance_neon(
                src, src_stride, reference, ref_stride, xoffset, yoffset, $w, $h,
            )
        }
    };
}

subpel_var_neon_fn!(vpx_sub_pixel_variance16x8_neon, 16, 8);
subpel_var_neon_fn!(vpx_sub_pixel_variance16x16_neon, 16, 16);
subpel_var_neon_fn!(vpx_sub_pixel_variance16x32_neon, 16, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x16_neon, 32, 16);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x32_neon, 32, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x64_neon, 32, 64);
subpel_var_neon_fn!(vpx_sub_pixel_variance64x32_neon, 64, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance64x64_neon, 64, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::distortion::sub_pixel_variance as scalar_svf;

    const SHAPES: &[(BlockSize, usize, usize)] = &[
        (BlockSize::Block16x8, 16, 8),
        (BlockSize::Block16x16, 16, 16),
        (BlockSize::Block16x32, 16, 32),
        (BlockSize::Block32x16, 32, 16),
        (BlockSize::Block32x32, 32, 32),
        (BlockSize::Block32x64, 32, 64),
        (BlockSize::Block64x32, 64, 32),
        (BlockSize::Block64x64, 64, 64),
    ];

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    #[test]
    fn differential_var_sse_all_offsets_and_shapes() {
        let mut rng = Lcg32(0xA6_4E0F);
        for &(_, width, height) in SHAPES {
            let src_stride = width + 17;
            let ref_stride = width + 11;
            let src: Vec<u8> = (0..height * src_stride + width + 1)
                .map(|_| rng.byte())
                .collect();
            let reference: Vec<u8> = (0..(height - 1) * ref_stride + width)
                .map(|_| rng.byte())
                .collect();
            for yoffset in 0..8 {
                for xoffset in 0..8 {
                    let scalar = scalar_svf(
                        &src, src_stride, &reference, ref_stride, xoffset, yoffset, width, height,
                    );
                    let neon = sub_pixel_variance_neon(
                        &src, src_stride, &reference, ref_stride, xoffset, yoffset, width, height,
                    );
                    assert_eq!(
                        scalar, neon,
                        "mismatch at {width}x{height}, offset ({xoffset},{yoffset})"
                    );
                }
            }
        }
    }

    #[test]
    fn differential_edge_content() {
        for &(_, width, height) in SHAPES {
            let src_stride = width + 1;
            let ref_stride = width;
            for &(src_value, ref_value) in &[(0u8, 0u8), (255, 255), (0, 255), (255, 0), (128, 17)]
            {
                let src = vec![src_value; height * src_stride + width + 1];
                let reference = vec![ref_value; height * ref_stride];
                for &(xoffset, yoffset) in &[(0, 0), (4, 4), (1, 7), (7, 1), (3, 5)] {
                    assert_eq!(
                        scalar_svf(
                            &src, src_stride, &reference, ref_stride, xoffset, yoffset, width,
                            height,
                        ),
                        sub_pixel_variance_neon(
                            &src, src_stride, &reference, ref_stride, xoffset, yoffset, width,
                            height,
                        ),
                        "edge mismatch at {width}x{height}, values \
                         ({src_value},{ref_value}), offset ({xoffset},{yoffset})"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatch_round_trip() {
        let neon = SimdDispatch::build(AArch64Caps { neon: true });
        let scalar = SimdDispatch::build(AArch64Caps { neon: false });
        let mut rng = Lcg32(0x0D15_A7C4);

        for &(bsize, width, height) in SHAPES {
            let src_stride = width + 9;
            let ref_stride = width + 5;
            let src: Vec<u8> = (0..height * src_stride + width + 1)
                .map(|_| rng.byte())
                .collect();
            let reference: Vec<u8> = (0..(height - 1) * ref_stride + width)
                .map(|_| rng.byte())
                .collect();
            let neon_fn = neon.subpel_variance(bsize).unwrap();
            let scalar_fn = scalar.subpel_variance(bsize).unwrap();
            for &(xoffset, yoffset) in &[(0, 0), (4, 4), (3, 5)] {
                assert_eq!(
                    neon_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    scalar_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    "dispatch mismatch at {bsize:?}, offset ({xoffset},{yoffset})"
                );
            }
        }
    }
}
