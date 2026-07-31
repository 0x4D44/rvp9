//! NEON variance `(sum, sse)` reductions for VP9's 8/16/32/64-wide blocks.
//!
//! The inter RD model consumes the integer-exact residual sum and squared
//! error before applying its scalar variance finisher. These kernels mirror
//! libvpx's NEON variance reduction while preserving rvp9's existing
//! `(i64, u64)` interface.

use core::arch::aarch64::*;
use core::arch::asm;

#[inline]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn unsigned_dot(
    mut sum: uint32x4_t,
    first: uint8x16_t,
    second: uint8x16_t,
) -> uint32x4_t {
    // SAFETY: `dotprod` is enabled and every operand uses a 128-bit SIMD
    // register. The instruction has no memory or control-flow effects.
    unsafe {
        asm!(
            "udot {sum:v}.4s, {first:v}.16b, {second:v}.16b",
            sum = inout(vreg) sum,
            first = in(vreg) first,
            second = in(vreg) second,
            options(pure, nomem, nostack),
        );
    }
    sum
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn accumulate_8(
    src: *const u8,
    pred: *const u8,
    sum: &mut int32x4_t,
    sse_lo: &mut int32x4_t,
    sse_hi: &mut int32x4_t,
) {
    let diff = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(src), vld1_u8(pred)));
    *sum = vpadalq_s16(*sum, diff);
    *sse_lo = vmlal_s16(*sse_lo, vget_low_s16(diff), vget_low_s16(diff));
    *sse_hi = vmlal_s16(*sse_hi, vget_high_s16(diff), vget_high_s16(diff));
}

#[target_feature(enable = "neon")]
unsafe fn var_sum_sse_neon_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (i64, u64) {
    let mut sum = vdupq_n_s32(0);
    let mut sse_lo = vdupq_n_s32(0);
    let mut sse_hi = vdupq_n_s32(0);

    for row in 0..height {
        let src_row = src.add(row * src_stride);
        let pred_row = pred.add(row * pred_stride);
        let mut col = 0;
        while col < width {
            accumulate_8(
                src_row.add(col),
                pred_row.add(col),
                &mut sum,
                &mut sse_lo,
                &mut sse_hi,
            );
            col += 8;
        }
    }

    (
        i64::from(vaddvq_s32(sum)),
        vaddvq_u32(vreinterpretq_u32_s32(vaddq_s32(sse_lo, sse_hi))) as u64,
    )
}

#[target_feature(enable = "dotprod")]
unsafe fn var_sum_sse_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (i64, u64) {
    let mut src_sum = vdupq_n_u32(0);
    let mut pred_sum = vdupq_n_u32(0);
    let mut sse = vdupq_n_u32(0);
    let ones = vdupq_n_u8(1);

    if width == 8 {
        for row in (0..height).step_by(2) {
            let src_bytes = vcombine_u8(
                vld1_u8(src.add(row * src_stride)),
                vld1_u8(src.add((row + 1) * src_stride)),
            );
            let pred_bytes = vcombine_u8(
                vld1_u8(pred.add(row * pred_stride)),
                vld1_u8(pred.add((row + 1) * pred_stride)),
            );
            let diff = vabdq_u8(src_bytes, pred_bytes);
            sse = unsigned_dot(sse, diff, diff);
            src_sum = unsigned_dot(src_sum, src_bytes, ones);
            pred_sum = unsigned_dot(pred_sum, pred_bytes, ones);
        }
    } else {
        for row in 0..height {
            let src_row = src.add(row * src_stride);
            let pred_row = pred.add(row * pred_stride);
            for col in (0..width).step_by(16) {
                let src_bytes = vld1q_u8(src_row.add(col));
                let pred_bytes = vld1q_u8(pred_row.add(col));
                let diff = vabdq_u8(src_bytes, pred_bytes);
                sse = unsigned_dot(sse, diff, diff);
                src_sum = unsigned_dot(src_sum, src_bytes, ones);
                pred_sum = unsigned_dot(pred_sum, pred_bytes, ones);
            }
        }
    }

    (
        i64::from(vaddvq_s32(vreinterpretq_s32_u32(vsubq_u32(
            src_sum, pred_sum,
        )))),
        u64::from(vaddvq_u32(sse)),
    )
}

/// Generic safe NEON `(sum, sse)` reduction.
pub fn var_sum_sse_neon(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (i64, u64) {
    assert!(matches!(width, 8 | 16 | 32 | 64));
    assert!(height > 0);
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= (height - 1) * src_stride + width);
    assert!(pred.len() >= (height - 1) * pred_stride + width);

    // SAFETY: AArch64 dispatch installs these wrappers only when NEON is
    // detected. The assertions prove that each 8-byte load stays within its
    // source slice. VP9's largest block has 4096 pixels, so every i32 lane
    // remains below its signed range for both sum and squared-error partials.
    unsafe {
        var_sum_sse_neon_inner(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            width,
            height,
        )
    }
}

/// Generic safe dot-product `(sum, sse)` reduction.
pub fn var_sum_sse_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (i64, u64) {
    assert!(matches!(width, 8 | 16 | 32 | 64));
    assert!(height > 0 && (width != 8 || height.is_multiple_of(2)));
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= (height - 1) * src_stride + width);
    assert!(pred.len() >= (height - 1) * pred_stride + width);

    // SAFETY: Dispatch installs these wrappers only when dotprod is present.
    // The assertions prove every 8- or 16-byte load. VP9's largest block has
    // 4096 pixels, so every four-way u32 partial remains below u32::MAX.
    unsafe {
        var_sum_sse_dotprod_inner(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            width,
            height,
        )
    }
}

macro_rules! var_sum_sse_neon_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("NEON (sum, sse) reduction for a ", stringify!($w), "x", stringify!($h), " block.")]
        pub fn $name(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> (i64, u64) {
            var_sum_sse_neon(src, src_stride, pred, pred_stride, $w, $h)
        }
    };
}

var_sum_sse_neon_fn!(var_sum_sse_8x4_neon, 8, 4);
var_sum_sse_neon_fn!(var_sum_sse_8x8_neon, 8, 8);
var_sum_sse_neon_fn!(var_sum_sse_8x16_neon, 8, 16);
var_sum_sse_neon_fn!(var_sum_sse_16x8_neon, 16, 8);
var_sum_sse_neon_fn!(var_sum_sse_16x16_neon, 16, 16);
var_sum_sse_neon_fn!(var_sum_sse_16x32_neon, 16, 32);
var_sum_sse_neon_fn!(var_sum_sse_32x16_neon, 32, 16);
var_sum_sse_neon_fn!(var_sum_sse_32x32_neon, 32, 32);
var_sum_sse_neon_fn!(var_sum_sse_32x64_neon, 32, 64);
var_sum_sse_neon_fn!(var_sum_sse_64x32_neon, 64, 32);
var_sum_sse_neon_fn!(var_sum_sse_64x64_neon, 64, 64);

macro_rules! var_sum_sse_dotprod_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("Dot-product (sum, sse) reduction for a ", stringify!($w), "x", stringify!($h), " block.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            pred: &[u8],
            pred_stride: usize,
        ) -> (i64, u64) {
            var_sum_sse_dotprod(src, src_stride, pred, pred_stride, $w, $h)
        }
    };
}

var_sum_sse_dotprod_fn!(var_sum_sse_8x4_dotprod, 8, 4);
var_sum_sse_dotprod_fn!(var_sum_sse_8x8_dotprod, 8, 8);
var_sum_sse_dotprod_fn!(var_sum_sse_8x16_dotprod, 8, 16);
var_sum_sse_dotprod_fn!(var_sum_sse_16x8_dotprod, 16, 8);
var_sum_sse_dotprod_fn!(var_sum_sse_16x16_dotprod, 16, 16);
var_sum_sse_dotprod_fn!(var_sum_sse_16x32_dotprod, 16, 32);
var_sum_sse_dotprod_fn!(var_sum_sse_32x16_dotprod, 32, 16);
var_sum_sse_dotprod_fn!(var_sum_sse_32x32_dotprod, 32, 32);
var_sum_sse_dotprod_fn!(var_sum_sse_32x64_dotprod, 32, 64);
var_sum_sse_dotprod_fn!(var_sum_sse_64x32_dotprod, 64, 32);
var_sum_sse_dotprod_fn!(var_sum_sse_64x64_dotprod, 64, 64);

/// NEON per-8x8-cell `(sum, sse)` reduction.
pub fn get_var_sse_8x8_neon(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i32, u32) {
    let (sum, sse) = var_sum_sse_neon(src, src_stride, pred, pred_stride, 8, 8);
    (sum as i32, sse as u32)
}

/// Dot-product per-8x8-cell `(sum, sse)` reduction.
pub fn get_var_sse_8x8_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i32, u32) {
    let (sum, sse) = var_sum_sse_dotprod(src, src_stride, pred, pred_stride, 8, 8);
    (sum as i32, sse as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::distortion;

    const SIZES: &[(BlockSize, usize, usize)] = &[
        (BlockSize::Block8x4, 8, 4),
        (BlockSize::Block8x8, 8, 8),
        (BlockSize::Block8x16, 8, 16),
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
    fn differential_random_all_sizes() {
        let mut rng = Lcg32(0x4E0F_5EED);
        for &(_, width, height) in SIZES {
            for jitter in 0..17 {
                let src_stride = width + jitter;
                let pred_stride = width + (jitter * 7 + 3) % 17;
                let src: Vec<u8> = (0..(height - 1) * src_stride + width)
                    .map(|_| rng.byte())
                    .collect();
                let pred: Vec<u8> = (0..(height - 1) * pred_stride + width)
                    .map(|_| rng.byte())
                    .collect();
                assert_eq!(
                    var_sum_sse_neon(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "mismatch at {width}x{height}, stride jitter {jitter}"
                );
                assert_eq!(
                    var_sum_sse_dotprod(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "dotprod mismatch at {width}x{height}, stride jitter {jitter}"
                );
            }
        }
    }

    #[test]
    fn differential_edge_content() {
        for &(_, width, height) in SIZES {
            let src_stride = width + 7;
            let pred_stride = width + 3;
            for &(src_value, pred_value) in &[(0, 0), (255, 255), (0, 255), (255, 0), (128, 17)] {
                let src = vec![src_value; (height - 1) * src_stride + width];
                let pred = vec![pred_value; (height - 1) * pred_stride + width];
                assert_eq!(
                    var_sum_sse_neon(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "edge mismatch at {width}x{height}, {src_value} vs {pred_value}"
                );
                assert_eq!(
                    var_sum_sse_dotprod(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "dotprod edge mismatch at {width}x{height}, {src_value} vs {pred_value}"
                );
            }
        }
    }

    #[test]
    fn dispatch_round_trip() {
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
        let dotprod_64: crate::dsp::dispatch::VarFn = var_sum_sse_64x64_dotprod;
        let dotprod_8: crate::dsp::dispatch::CellVarFn = get_var_sse_8x8_dotprod;
        let neon_64: crate::dsp::dispatch::VarFn = var_sum_sse_64x64_neon;
        assert_eq!(
            dotprod.variance(BlockSize::Block64x64) as usize,
            dotprod_64 as usize
        );
        assert_eq!(dotprod.get_var_sse_8x8() as usize, dotprod_8 as usize);
        assert_eq!(
            neon.variance(BlockSize::Block64x64) as usize,
            neon_64 as usize
        );
        let mut rng = Lcg32(0xD15A_5EED);

        for &(bsize, width, height) in SIZES {
            let src_stride = width + 5;
            let pred_stride = width + 2;
            let src: Vec<u8> = (0..(height - 1) * src_stride + width)
                .map(|_| rng.byte())
                .collect();
            let pred: Vec<u8> = (0..(height - 1) * pred_stride + width)
                .map(|_| rng.byte())
                .collect();
            assert_eq!(
                (neon.variance(bsize))(&src, src_stride, &pred, pred_stride),
                (scalar.variance(bsize))(&src, src_stride, &pred, pred_stride),
                "dispatch mismatch at {bsize:?}"
            );
            assert_eq!(
                (dotprod.variance(bsize))(&src, src_stride, &pred, pred_stride),
                (scalar.variance(bsize))(&src, src_stride, &pred, pred_stride),
                "dotprod dispatch mismatch at {bsize:?}"
            );
        }

        let src: Vec<u8> = (0..7 * 15 + 8).map(|_| rng.byte()).collect();
        let pred: Vec<u8> = (0..7 * 11 + 8).map(|_| rng.byte()).collect();
        assert_eq!(
            (neon.get_var_sse_8x8)(&src, 15, &pred, 11),
            (scalar.get_var_sse_8x8)(&src, 15, &pred, 11)
        );
        assert_eq!(
            (dotprod.get_var_sse_8x8)(&src, 15, &pred, 11),
            (scalar.get_var_sse_8x8)(&src, 15, &pred, 11)
        );
    }
}
