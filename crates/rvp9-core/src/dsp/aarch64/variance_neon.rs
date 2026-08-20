//! NEON variance `(sum, sse)` reductions for VP9's 4/8/16/32/64-wide blocks.
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

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_four_4byte_rows(src: *const u8, stride: usize) -> uint8x16_t {
    let mut rows = vdupq_n_u32(0);
    rows = vsetq_lane_u32::<0>(core::ptr::read_unaligned(src.cast()), rows);
    rows = vsetq_lane_u32::<1>(core::ptr::read_unaligned(src.add(stride).cast()), rows);
    rows = vsetq_lane_u32::<2>(core::ptr::read_unaligned(src.add(2 * stride).cast()), rows);
    rows = vsetq_lane_u32::<3>(core::ptr::read_unaligned(src.add(3 * stride).cast()), rows);
    vreinterpretq_u8_u32(rows)
}

#[target_feature(enable = "dotprod")]
unsafe fn var_sum_sse_4_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    height: usize,
) -> (i64, u64) {
    let mut src_sum = vdupq_n_u32(0);
    let mut pred_sum = vdupq_n_u32(0);
    let mut sse = vdupq_n_u32(0);
    let ones = vdupq_n_u8(1);

    for row in (0..height).step_by(4) {
        let src_bytes = load_four_4byte_rows(src.add(row * src_stride), src_stride);
        let pred_bytes = load_four_4byte_rows(pred.add(row * pred_stride), pred_stride);
        let diff = vabdq_u8(src_bytes, pred_bytes);
        sse = unsigned_dot(sse, diff, diff);
        src_sum = unsigned_dot(src_sum, src_bytes, ones);
        pred_sum = unsigned_dot(pred_sum, pred_bytes, ones);
    }

    (
        i64::from(vaddvq_s32(vreinterpretq_s32_u32(vsubq_u32(
            src_sum, pred_sum,
        )))),
        u64::from(vaddvq_u32(sse)),
    )
}

fn var_sum_sse_4_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    height: usize,
) -> (i64, u64) {
    assert!(matches!(height, 4 | 8));
    assert!(src_stride >= 4);
    assert!(pred_stride >= 4);
    assert!(src.len() >= (height - 1) * src_stride + 4);
    assert!(pred.len() >= (height - 1) * pred_stride + 4);

    // SAFETY: Dispatch installs these wrappers only when dotprod is present.
    // Exact-width unaligned 32-bit loads consume four live bytes per row;
    // four rows are packed before each dot-product reduction.
    unsafe {
        var_sum_sse_4_dotprod_inner(src.as_ptr(), src_stride, pred.as_ptr(), pred_stride, height)
    }
}

/// Dot-product `(sum, sse)` reduction for a 4x4 block.
pub fn var_sum_sse_4x4_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i64, u64) {
    var_sum_sse_4_dotprod(src, src_stride, pred, pred_stride, 4)
}

/// Dot-product `(sum, sse)` reduction for a 4x8 block.
pub fn var_sum_sse_4x8_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i64, u64) {
    var_sum_sse_4_dotprod(src, src_stride, pred, pred_stride, 8)
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

/// Reduce every 8×8 cell in one of the large-block model's supported grids.
///
/// The cells are finalized in raster order, matching the old per-cell loop;
/// the fused front end only shares the row traversal and keeps one accumulator
/// set per cell column. The returned totals therefore have the same reduction
/// order and integer values as [`crate::dsp::distortion::get_var_sse_8x8`].
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn var_sum_sse_8x8_grid_neon_stripe(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    row_base: usize,
    x_offset: usize,
    cell_base: usize,
    sse8x8: &mut [u32; 64],
    sum8x8: &mut [i32; 64],
) -> (u64, i64) {
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;

    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let mut sum2 = vdupq_n_s32(0);
    let mut sum3 = vdupq_n_s32(0);
    let mut sse0_lo = vdupq_n_s32(0);
    let mut sse1_lo = vdupq_n_s32(0);
    let mut sse2_lo = vdupq_n_s32(0);
    let mut sse3_lo = vdupq_n_s32(0);
    let mut sse0_hi = vdupq_n_s32(0);
    let mut sse1_hi = vdupq_n_s32(0);
    let mut sse2_hi = vdupq_n_s32(0);
    let mut sse3_hi = vdupq_n_s32(0);

    macro_rules! accumulate {
        ($src_row:ident, $pred_row:ident, $sum:ident, $sse_lo:ident, $sse_hi:ident, $offset:expr) => {{
            let diff = vreinterpretq_s16_u16(vsubl_u8(
                vld1_u8($src_row.add(x_offset + $offset)),
                vld1_u8($pred_row.add(x_offset + $offset)),
            ));
            $sum = vpadalq_s16($sum, diff);
            $sse_lo = vmlal_s16($sse_lo, vget_low_s16(diff), vget_low_s16(diff));
            $sse_hi = vmlal_s16($sse_hi, vget_high_s16(diff), vget_high_s16(diff));
        }};
    }

    for row in 0..8 {
        let src_row = src.add((row_base + row) * src_stride);
        let pred_row = pred.add((row_base + row) * pred_stride);
        accumulate!(src_row, pred_row, sum0, sse0_lo, sse0_hi, 0);
        accumulate!(src_row, pred_row, sum1, sse1_lo, sse1_hi, 8);
        accumulate!(src_row, pred_row, sum2, sse2_lo, sse2_hi, 16);
        accumulate!(src_row, pred_row, sum3, sse3_lo, sse3_hi, 24);
    }

    macro_rules! finish {
        ($offset:expr, $sum:ident, $sse_lo:ident, $sse_hi:ident) => {{
            let cell_sum = i64::from(vaddvq_s32($sum));
            let cell_sse = vaddvq_u32(vreinterpretq_u32_s32(vaddq_s32($sse_lo, $sse_hi))) as u32;
            sum8x8[cell_base + $offset] = cell_sum as i32;
            sse8x8[cell_base + $offset] = cell_sse;
            total_sum += cell_sum;
            total_sse += u64::from(cell_sse);
        }};
    }

    finish!(0, sum0, sse0_lo, sse0_hi);
    finish!(1, sum1, sse1_lo, sse1_hi);
    finish!(2, sum2, sse2_lo, sse2_hi);
    finish!(3, sum3, sse3_lo, sse3_hi);

    (total_sse, total_sum)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn var_sum_sse_8x8_grid_neon_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (u64, i64, [u32; 64], [i32; 64]) {
    let nh = height / 8;
    let nw = width / 8;
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;
    let mut sse8x8 = [0u32; 64];
    let mut sum8x8 = [0i32; 64];

    for cell_row in 0..nh {
        let row_base = cell_row * 8;
        let cell_base = cell_row * nw;
        let (stripe_sse, stripe_sum) = var_sum_sse_8x8_grid_neon_stripe(
            src,
            src_stride,
            pred,
            pred_stride,
            row_base,
            0,
            cell_base,
            &mut sse8x8,
            &mut sum8x8,
        );
        total_sse += stripe_sse;
        total_sum += stripe_sum;
        if width == 64 {
            let (stripe_sse, stripe_sum) = var_sum_sse_8x8_grid_neon_stripe(
                src,
                src_stride,
                pred,
                pred_stride,
                row_base,
                32,
                cell_base + 4,
                &mut sse8x8,
                &mut sum8x8,
            );
            total_sse += stripe_sse;
            total_sum += stripe_sum;
        }
    }

    (total_sse, total_sum, sse8x8, sum8x8)
}

/// Reduce a 32×32, 32×64, 64×32, or 64×64 grid of 8×8 cells with NEON.
///
/// Shapes outside this small set are rejected because the large-block model
/// only routes its four exact VP9 partition grids through this entry point.
#[must_use]
pub fn var_sum_sse_8x8_grid_neon(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (u64, i64, [u32; 64], [i32; 64]) {
    assert!(matches!(
        (width, height),
        (32, 32) | (32, 64) | (64, 32) | (64, 64)
    ));
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= (height - 1) * src_stride + width);
    assert!(pred.len() >= (height - 1) * pred_stride + width);

    // SAFETY: AArch64 dispatch installs this wrapper only when NEON is
    // available. The assertions prove every 8-byte load in both footprints;
    // the immutable slices remain live for the whole inner call.
    unsafe {
        var_sum_sse_8x8_grid_neon_inner(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            width,
            height,
        )
    }
}

#[inline]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn var_sum_sse_8x8_grid_dotprod_stripe(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    row_base: usize,
    x_offset: usize,
    cell_base: usize,
    sse8x8: &mut [u32; 64],
    sum8x8: &mut [i32; 64],
) -> (u64, i64) {
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;
    let ones = vdupq_n_u8(1);

    let mut src_sum0 = vdupq_n_u32(0);
    let mut src_sum1 = vdupq_n_u32(0);
    let mut src_sum2 = vdupq_n_u32(0);
    let mut src_sum3 = vdupq_n_u32(0);
    let mut pred_sum0 = vdupq_n_u32(0);
    let mut pred_sum1 = vdupq_n_u32(0);
    let mut pred_sum2 = vdupq_n_u32(0);
    let mut pred_sum3 = vdupq_n_u32(0);
    let mut sse0 = vdupq_n_u32(0);
    let mut sse1 = vdupq_n_u32(0);
    let mut sse2 = vdupq_n_u32(0);
    let mut sse3 = vdupq_n_u32(0);

    macro_rules! accumulate {
        ($src_row:ident, $pred_row:ident, $src_sum:ident, $pred_sum:ident, $sse:ident, $offset:expr) => {{
            let src_bytes = vcombine_u8(
                vld1_u8($src_row.add(x_offset + $offset)),
                vld1_u8($src_row.add(src_stride + x_offset + $offset)),
            );
            let pred_bytes = vcombine_u8(
                vld1_u8($pred_row.add(x_offset + $offset)),
                vld1_u8($pred_row.add(pred_stride + x_offset + $offset)),
            );
            let diff = vabdq_u8(src_bytes, pred_bytes);
            $sse = unsigned_dot($sse, diff, diff);
            $src_sum = unsigned_dot($src_sum, src_bytes, ones);
            $pred_sum = unsigned_dot($pred_sum, pred_bytes, ones);
        }};
    }

    for row in (0..8).step_by(2) {
        let src_row = src.add((row_base + row) * src_stride);
        let pred_row = pred.add((row_base + row) * pred_stride);
        accumulate!(src_row, pred_row, src_sum0, pred_sum0, sse0, 0);
        accumulate!(src_row, pred_row, src_sum1, pred_sum1, sse1, 8);
        accumulate!(src_row, pred_row, src_sum2, pred_sum2, sse2, 16);
        accumulate!(src_row, pred_row, src_sum3, pred_sum3, sse3, 24);
    }

    macro_rules! finish {
        ($offset:expr, $src_sum:ident, $pred_sum:ident, $sse:ident) => {{
            let cell_sum = i64::from(vaddvq_s32(vreinterpretq_s32_u32(vsubq_u32(
                $src_sum, $pred_sum,
            ))));
            let cell_sse = vaddvq_u32($sse);
            sum8x8[cell_base + $offset] = cell_sum as i32;
            sse8x8[cell_base + $offset] = cell_sse;
            total_sum += cell_sum;
            total_sse += u64::from(cell_sse);
        }};
    }

    finish!(0, src_sum0, pred_sum0, sse0);
    finish!(1, src_sum1, pred_sum1, sse1);
    finish!(2, src_sum2, pred_sum2, sse2);
    finish!(3, src_sum3, pred_sum3, sse3);

    (total_sse, total_sum)
}

#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn var_sum_sse_8x8_grid_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    pred: *const u8,
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (u64, i64, [u32; 64], [i32; 64]) {
    let nh = height / 8;
    let nw = width / 8;
    let mut total_sse = 0u64;
    let mut total_sum = 0i64;
    let mut sse8x8 = [0u32; 64];
    let mut sum8x8 = [0i32; 64];

    for cell_row in 0..nh {
        let row_base = cell_row * 8;
        let cell_base = cell_row * nw;
        let (stripe_sse, stripe_sum) = var_sum_sse_8x8_grid_dotprod_stripe(
            src,
            src_stride,
            pred,
            pred_stride,
            row_base,
            0,
            cell_base,
            &mut sse8x8,
            &mut sum8x8,
        );
        total_sse += stripe_sse;
        total_sum += stripe_sum;
        if width == 64 {
            let (stripe_sse, stripe_sum) = var_sum_sse_8x8_grid_dotprod_stripe(
                src,
                src_stride,
                pred,
                pred_stride,
                row_base,
                32,
                cell_base + 4,
                &mut sse8x8,
                &mut sum8x8,
            );
            total_sse += stripe_sse;
            total_sum += stripe_sum;
        }
    }

    (total_sse, total_sum, sse8x8, sum8x8)
}

/// Reduce a 32×32, 32×64, 64×32, or 64×64 grid of 8×8 cells with dotprod.
#[must_use]
pub fn var_sum_sse_8x8_grid_dotprod(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> (u64, i64, [u32; 64], [i32; 64]) {
    assert!(matches!(
        (width, height),
        (32, 32) | (32, 64) | (64, 32) | (64, 64)
    ));
    assert!(src_stride >= width);
    assert!(pred_stride >= width);
    assert!(src.len() >= (height - 1) * src_stride + width);
    assert!(pred.len() >= (height - 1) * pred_stride + width);

    // SAFETY: AArch64 dispatch installs this wrapper only when dotprod is
    // available. The assertions prove every 8-byte load in both footprints;
    // the target-feature inner call uses no memory beyond those footprints.
    unsafe {
        var_sum_sse_8x8_grid_dotprod_inner(
            src.as_ptr(),
            src_stride,
            pred.as_ptr(),
            pred_stride,
            width,
            height,
        )
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::distortion;

    const SIZES: &[(BlockSize, usize, usize)] = &[
        (BlockSize::Block4x4, 4, 4),
        (BlockSize::Block4x8, 4, 8),
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

    fn neon_for_shape(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        width: usize,
        height: usize,
    ) -> (i64, u64) {
        if width == 4 {
            distortion::var_sum_sse(src, src_stride, pred, pred_stride, width, height)
        } else {
            var_sum_sse_neon(src, src_stride, pred, pred_stride, width, height)
        }
    }

    fn dotprod_for_shape(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        width: usize,
        height: usize,
    ) -> (i64, u64) {
        match (width, height) {
            (4, 4) => var_sum_sse_4x4_dotprod(src, src_stride, pred, pred_stride),
            (4, 8) => var_sum_sse_4x8_dotprod(src, src_stride, pred, pred_stride),
            _ => var_sum_sse_dotprod(src, src_stride, pred, pred_stride, width, height),
        }
    }

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.0 >> 16) as u8
        }
    }

    fn scalar_grid(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        width: usize,
        height: usize,
    ) -> (u64, i64, [u32; 64], [i32; 64]) {
        let nw = width / 8;
        let nh = height / 8;
        let mut total_sse = 0u64;
        let mut total_sum = 0i64;
        let mut sse8x8 = [0u32; 64];
        let mut sum8x8 = [0i32; 64];
        for cell_row in 0..nh {
            for cell_col in 0..nw {
                let k = cell_row * nw + cell_col;
                let (sum, sse) = distortion::get_var_sse_8x8(
                    &src[cell_row * 8 * src_stride + cell_col * 8..],
                    src_stride,
                    &pred[cell_row * 8 * pred_stride + cell_col * 8..],
                    pred_stride,
                );
                sse8x8[k] = sse;
                sum8x8[k] = sum;
                total_sse += u64::from(sse);
                total_sum += i64::from(sum);
            }
        }
        (total_sse, total_sum, sse8x8, sum8x8)
    }

    fn grid_inputs(
        width: usize,
        height: usize,
        src_stride: usize,
        pred_stride: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        (
            (0..(height - 1) * src_stride + width)
                .map(|index| (index as u32).wrapping_mul(37) as u8)
                .collect(),
            (0..(height - 1) * pred_stride + width)
                .map(|index| (index as u32).wrapping_mul(61).wrapping_add(19) as u8)
                .collect(),
        )
    }

    #[test]
    fn fused_neon_grid_matches_scalar_for_all_large_shapes_and_content() {
        for &(width, height) in &[(32, 32), (32, 64), (64, 32), (64, 64)] {
            for &(src_stride, pred_stride) in &[
                (width, width),
                (width + 1, width + 13),
                (width + 17, width + 5),
            ] {
                let (mut src, mut pred) = grid_inputs(width, height, src_stride, pred_stride);
                for &(src_value, pred_value) in &[(0, 0), (0, 255), (255, 0), (255, 255)] {
                    src.fill(src_value);
                    pred.fill(pred_value);
                    assert_eq!(
                        var_sum_sse_8x8_grid_neon(
                            &src,
                            src_stride,
                            &pred,
                            pred_stride,
                            width,
                            height,
                        ),
                        scalar_grid(&src, src_stride, &pred, pred_stride, width, height),
                        "constant mismatch at {width}x{height}, strides {src_stride}/{pred_stride}",
                    );
                }

                for row in 0..height {
                    for col in 0..width {
                        src[row * src_stride + col] = if (row + col) & 1 == 0 { 255 } else { 0 };
                        pred[row * pred_stride + col] = 127;
                    }
                }
                assert_eq!(
                    var_sum_sse_8x8_grid_neon(&src, src_stride, &pred, pred_stride, width, height,),
                    scalar_grid(&src, src_stride, &pred, pred_stride, width, height),
                    "checkerboard mismatch at {width}x{height}, strides {src_stride}/{pred_stride}",
                );

                let mut rng = Lcg32(0xA8C0_8EED ^ width as u32 ^ (height as u32) << 8);
                for _ in 0..32 {
                    for byte in &mut src {
                        *byte = rng.byte();
                    }
                    for byte in &mut pred {
                        *byte = rng.byte();
                    }
                    assert_eq!(
                        var_sum_sse_8x8_grid_neon(
                            &src,
                            src_stride,
                            &pred,
                            pred_stride,
                            width,
                            height,
                        ),
                        scalar_grid(&src, src_stride, &pred, pred_stride, width, height),
                        "random mismatch at {width}x{height}, strides {src_stride}/{pred_stride}",
                    );

                    if std::arch::is_aarch64_feature_detected!("dotprod") {
                        assert_eq!(
                            var_sum_sse_8x8_grid_dotprod(
                                &src,
                                src_stride,
                                &pred,
                                pred_stride,
                                width,
                                height,
                            ),
                            scalar_grid(&src, src_stride, &pred, pred_stride, width, height),
                            "dotprod random mismatch at {width}x{height}, strides {src_stride}/{pred_stride}",
                        );
                    }
                }
            }
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
                    neon_for_shape(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "mismatch at {width}x{height}, stride jitter {jitter}"
                );
                assert_eq!(
                    dotprod_for_shape(&src, src_stride, &pred, pred_stride, width, height),
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
                    neon_for_shape(&src, src_stride, &pred, pred_stride, width, height),
                    distortion::var_sum_sse(&src, src_stride, &pred, pred_stride, width, height),
                    "edge mismatch at {width}x{height}, {src_value} vs {pred_value}"
                );
                assert_eq!(
                    dotprod_for_shape(&src, src_stride, &pred, pred_stride, width, height),
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
