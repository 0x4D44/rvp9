//! NEON sub-pixel variance for VP9's 8/16/32/64-wide block shapes.
//!
//! This follows libvpx `vpx_dsp/arm/subpel_variance_neon.c`, with one
//! deliberate fusion: the second bilinear pass feeds the `(sum, sse)`
//! reduction directly. Two rolling 64-byte rows replace libvpx's full
//! intermediate blocks, avoiding heap allocation and an extra pixel walk.

use core::arch::aarch64::*;

use super::variance_neon::unsigned_dot;

/// Apply one VP9 two-tap bilinear filter to eight byte lanes.
///
/// Eight-wide blocks need exact-width loads because callers are allowed to
/// provide only the live block footprint.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_8(a: *const u8, b: *const u8, offset: usize) -> uint8x8_t {
    let av = vld1_u8(a);
    if offset == 0 {
        av
    } else {
        filter_8_values(av, vld1_u8(b), offset)
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_8_values(av: uint8x8_t, bv: uint8x8_t, offset: usize) -> uint8x8_t {
    if offset == 0 {
        return av;
    }
    if offset == 4 {
        return vrhadd_u8(av, bv);
    }

    let f0 = vdup_n_u8((8 - offset) as u8);
    let f1 = vdup_n_u8(offset as u8);
    vrshrn_n_u16::<3>(vmlal_u8(vmull_u8(av, f0), bv, f1))
}

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
unsafe fn horizontal_row_8(src: *const u8, dst: *mut u8, xoffset: usize) {
    vst1_u8(dst, filter_8(src, src.add(1), xoffset));
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_8(value: uint8x8_t) -> uint8x16_t {
    vcombine_u8(value, vdup_n_u8(0))
}

struct VarianceAccumulator {
    sum_lo: int32x4_t,
    sum_hi: int32x4_t,
    sse_0: int32x4_t,
    sse_1: int32x4_t,
    sse_2: int32x4_t,
    sse_3: int32x4_t,
}

struct DotprodVarianceAccumulator {
    pred_sum: uint32x4_t,
    ref_sum: uint32x4_t,
    sse: uint32x4_t,
    ones: uint8x16_t,
}

impl DotprodVarianceAccumulator {
    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn new() -> Self {
        Self {
            pred_sum: vdupq_n_u32(0),
            ref_sum: vdupq_n_u32(0),
            sse: vdupq_n_u32(0),
            ones: vdupq_n_u8(1),
        }
    }

    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn add(&mut self, pred: uint8x16_t, reference: uint8x16_t) {
        let diff = vabdq_u8(pred, reference);
        self.pred_sum = unsigned_dot(self.pred_sum, pred, self.ones);
        self.ref_sum = unsigned_dot(self.ref_sum, reference, self.ones);
        self.sse = unsigned_dot(self.sse, diff, diff);
    }

    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn finish(self) -> (i64, i64) {
        let sum = vaddvq_s32(vreinterpretq_s32_u32(vsubq_u32(
            self.pred_sum,
            self.ref_sum,
        )));
        let sse = vaddvq_u32(self.sse);
        (i64::from(sum), i64::from(sse))
    }
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

macro_rules! sub_pixel_variance_body {
    ($accum:ident, $src:ident, $src_stride:ident, $reference:ident, $ref_stride:ident, $xoffset:ident, $yoffset:ident, $width:expr, $height:expr) => {{
        if $xoffset == 0 {
            for row in 0..$height {
                let top = $src.add(row * $src_stride);
                let bottom = top.add($src_stride);
                let ref_row = $reference.add(row * $ref_stride);
                let mut col = 0;
                while col < $width {
                    let pred = filter_16(top.add(col), bottom.add(col), $yoffset);
                    $accum.add(pred, vld1q_u8(ref_row.add(col)));
                    col += 16;
                }
            }
        } else if $yoffset == 0 {
            for row in 0..$height {
                let src_row = $src.add(row * $src_stride);
                let ref_row = $reference.add(row * $ref_stride);
                let mut col = 0;
                while col < $width {
                    let pred = filter_16(src_row.add(col), src_row.add(col + 1), $xoffset);
                    $accum.add(pred, vld1q_u8(ref_row.add(col)));
                    col += 16;
                }
            }
        } else {
            let mut row_a = [0u8; 64];
            let mut row_b = [0u8; 64];
            horizontal_row($src, row_a.as_mut_ptr(), $width, $xoffset);

            for row in 0..$height {
                horizontal_row(
                    $src.add((row + 1) * $src_stride),
                    row_b.as_mut_ptr(),
                    $width,
                    $xoffset,
                );
                let ref_row = $reference.add(row * $ref_stride);
                let mut col = 0;
                while col < $width {
                    let pred =
                        filter_16(row_a.as_ptr().add(col), row_b.as_ptr().add(col), $yoffset);
                    $accum.add(pred, vld1q_u8(ref_row.add(col)));
                    col += 16;
                }
                core::mem::swap(&mut row_a, &mut row_b);
            }
        }

        let (sum, sse) = $accum.finish();
        let variance = (sse - sum * sum / ($width * $height) as i64).max(0) as u32;
        (variance, sse as u32)
    }};
}

macro_rules! sub_pixel_variance_8_body {
    ($accum:ident, $src:ident, $src_stride:ident, $reference:ident, $ref_stride:ident, $xoffset:ident, $yoffset:ident, $height:expr) => {{
        if $xoffset == 0 {
            for row in 0..$height {
                let top = $src.add(row * $src_stride);
                let bottom = top.add($src_stride);
                let ref_row = $reference.add(row * $ref_stride);
                let pred = widen_8(filter_8(top, bottom, $yoffset));
                $accum.add(pred, widen_8(vld1_u8(ref_row)));
            }
        } else if $yoffset == 0 {
            for row in 0..$height {
                let src_row = $src.add(row * $src_stride);
                let ref_row = $reference.add(row * $ref_stride);
                let pred = widen_8(filter_8(src_row, src_row.add(1), $xoffset));
                $accum.add(pred, widen_8(vld1_u8(ref_row)));
            }
        } else {
            let mut row_a = [0u8; 8];
            let mut row_b = [0u8; 8];
            horizontal_row_8($src, row_a.as_mut_ptr(), $xoffset);

            for row in 0..$height {
                horizontal_row_8(
                    $src.add((row + 1) * $src_stride),
                    row_b.as_mut_ptr(),
                    $xoffset,
                );
                let pred = widen_8(filter_8(row_a.as_ptr(), row_b.as_ptr(), $yoffset));
                let ref_row = $reference.add(row * $ref_stride);
                $accum.add(pred, widen_8(vld1_u8(ref_row)));
                core::mem::swap(&mut row_a, &mut row_b);
            }
        }

        let (sum, sse) = $accum.finish();
        let variance = (sse - sum * sum / (8 * $height) as i64).max(0) as u32;
        (variance, sse as u32)
    }};
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_pixel_variance_8_neon_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    height: usize,
) -> (u32, u32) {
    let mut accum = VarianceAccumulator::new();
    sub_pixel_variance_8_body!(
        accum, src, src_stride, reference, ref_stride, xoffset, yoffset, height
    )
}

#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_pixel_variance_8_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    height: usize,
) -> (u32, u32) {
    let mut accum = DotprodVarianceAccumulator::new();
    debug_assert!(height.is_multiple_of(2));

    if xoffset == 0 {
        for row in (0..height).step_by(2) {
            let top = src.add(row * src_stride);
            let middle = top.add(src_stride);
            let bottom = middle.add(src_stride);
            let pred = vcombine_u8(
                filter_8(top, middle, yoffset),
                filter_8(middle, bottom, yoffset),
            );
            let reference_rows = vcombine_u8(
                vld1_u8(reference.add(row * ref_stride)),
                vld1_u8(reference.add((row + 1) * ref_stride)),
            );
            accum.add(pred, reference_rows);
        }
    } else if yoffset == 0 {
        for row in (0..height).step_by(2) {
            let first = src.add(row * src_stride);
            let second = src.add((row + 1) * src_stride);
            let pred = vcombine_u8(
                filter_8(first, first.add(1), xoffset),
                filter_8(second, second.add(1), xoffset),
            );
            let reference_rows = vcombine_u8(
                vld1_u8(reference.add(row * ref_stride)),
                vld1_u8(reference.add((row + 1) * ref_stride)),
            );
            accum.add(pred, reference_rows);
        }
    } else {
        let mut horizontal_first = filter_8(src, src.add(1), xoffset);
        for row in (0..height).step_by(2) {
            let middle = src.add((row + 1) * src_stride);
            let bottom = src.add((row + 2) * src_stride);
            let horizontal_middle = filter_8(middle, middle.add(1), xoffset);
            let horizontal_bottom = filter_8(bottom, bottom.add(1), xoffset);
            let pred = vcombine_u8(
                filter_8_values(horizontal_first, horizontal_middle, yoffset),
                filter_8_values(horizontal_middle, horizontal_bottom, yoffset),
            );
            let reference_rows = vcombine_u8(
                vld1_u8(reference.add(row * ref_stride)),
                vld1_u8(reference.add((row + 1) * ref_stride)),
            );
            accum.add(pred, reference_rows);
            horizontal_first = horizontal_bottom;
        }
    }

    let (sum, sse) = accum.finish();
    let variance = (sse - sum * sum / (8 * height) as i64).max(0) as u32;
    (variance, sse as u32)
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
    sub_pixel_variance_body!(
        accum, src, src_stride, reference, ref_stride, xoffset, yoffset, width, height
    )
}

#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_pixel_variance_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    width: usize,
    height: usize,
) -> (u32, u32) {
    let mut accum = DotprodVarianceAccumulator::new();
    sub_pixel_variance_body!(
        accum, src, src_stride, reference, ref_stride, xoffset, yoffset, width, height
    )
}

#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
unsafe fn sub_pixel_variance_32x32_dotprod_inner(
    src: *const u8,
    src_stride: usize,
    reference: *const u8,
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
) -> (u32, u32) {
    let mut accum = DotprodVarianceAccumulator::new();
    sub_pixel_variance_body!(
        accum, src, src_stride, reference, ref_stride, xoffset, yoffset, 32, 32
    )
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

/// Generic safe dot-product sub-pixel variance entry point.
#[allow(clippy::too_many_arguments)]
pub fn sub_pixel_variance_dotprod(
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

    // SAFETY: Dispatch selects this wrapper only when dotprod is present.
    // The slice bounds are identical to the NEON entry point above.
    unsafe {
        sub_pixel_variance_dotprod_inner(
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

macro_rules! subpel_var_dotprod_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("Dot-product sub-pixel variance ", stringify!($w), "x", stringify!($h), ".")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            xoffset: usize,
            yoffset: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> (u32, u32) {
            sub_pixel_variance_dotprod(
                src, src_stride, reference, ref_stride, xoffset, yoffset, $w, $h,
            )
        }
    };
}

/// NEON sub-pixel variance 8x8.
pub fn vpx_sub_pixel_variance8x8_neon(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    reference: &[u8],
    ref_stride: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(src_stride >= 8 + usize::from(xoffset != 0));
    assert!(ref_stride >= 8);
    assert!(src.len() >= 8 * src_stride + 8 + usize::from(xoffset != 0));
    assert!(reference.len() >= 7 * ref_stride + 8);

    // SAFETY: AArch64 dispatch selects this wrapper only when NEON is
    // present. The assertions prove every exact-width load, the optional
    // horizontal neighbour, and the vertical source row are live.
    unsafe {
        sub_pixel_variance_8_neon_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
            8,
        )
    }
}

/// NEON sub-pixel variance 8x16.
pub fn vpx_sub_pixel_variance8x16_neon(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    reference: &[u8],
    ref_stride: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(src_stride >= 8 + usize::from(xoffset != 0));
    assert!(ref_stride >= 8);
    assert!(src.len() >= 16 * src_stride + 8 + usize::from(xoffset != 0));
    assert!(reference.len() >= 15 * ref_stride + 8);

    // SAFETY: AArch64 dispatch selects this wrapper only when NEON is
    // present. The assertions prove every exact-width load, the optional
    // horizontal neighbour, and the vertical source row are live.
    unsafe {
        sub_pixel_variance_8_neon_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
            16,
        )
    }
}
subpel_var_neon_fn!(vpx_sub_pixel_variance16x8_neon, 16, 8);
subpel_var_neon_fn!(vpx_sub_pixel_variance16x16_neon, 16, 16);
subpel_var_neon_fn!(vpx_sub_pixel_variance16x32_neon, 16, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x16_neon, 32, 16);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x32_neon, 32, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance32x64_neon, 32, 64);
subpel_var_neon_fn!(vpx_sub_pixel_variance64x32_neon, 64, 32);
subpel_var_neon_fn!(vpx_sub_pixel_variance64x64_neon, 64, 64);

/// Dot-product sub-pixel variance 8x8.
pub fn vpx_sub_pixel_variance8x8_dotprod(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    reference: &[u8],
    ref_stride: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(src_stride >= 8 + usize::from(xoffset != 0));
    assert!(ref_stride >= 8);
    assert!(src.len() >= 8 * src_stride + 8 + usize::from(xoffset != 0));
    assert!(reference.len() >= 7 * ref_stride + 8);

    // SAFETY: Dispatch selects this wrapper only when dotprod is present.
    // The assertions prove the complete eight-wide source footprint.
    unsafe {
        sub_pixel_variance_8_dotprod_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
            8,
        )
    }
}

/// Dot-product sub-pixel variance 8x16.
pub fn vpx_sub_pixel_variance8x16_dotprod(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    reference: &[u8],
    ref_stride: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(src_stride >= 8 + usize::from(xoffset != 0));
    assert!(ref_stride >= 8);
    assert!(src.len() >= 16 * src_stride + 8 + usize::from(xoffset != 0));
    assert!(reference.len() >= 15 * ref_stride + 8);

    // SAFETY: Dispatch selects this wrapper only when dotprod is present.
    // The assertions prove the complete eight-wide source footprint.
    unsafe {
        sub_pixel_variance_8_dotprod_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
            16,
        )
    }
}
subpel_var_dotprod_fn!(vpx_sub_pixel_variance16x8_dotprod, 16, 8);
subpel_var_dotprod_fn!(vpx_sub_pixel_variance16x16_dotprod, 16, 16);
subpel_var_dotprod_fn!(vpx_sub_pixel_variance16x32_dotprod, 16, 32);
subpel_var_dotprod_fn!(vpx_sub_pixel_variance32x16_dotprod, 32, 16);
/// Dot-product sub-pixel variance 32x32.
pub fn vpx_sub_pixel_variance32x32_dotprod(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    reference: &[u8],
    ref_stride: usize,
) -> (u32, u32) {
    assert!(xoffset < 8 && yoffset < 8);
    assert!(src_stride >= 32 + usize::from(xoffset != 0));
    assert!(ref_stride >= 32);
    assert!(src.len() >= 32 * src_stride + 32 + usize::from(xoffset != 0));
    assert!(reference.len() >= 31 * ref_stride + 32);

    // SAFETY: Dispatch selects this wrapper only when dotprod is present. The
    // assertions prove the extra horizontal byte and vertical row are live.
    unsafe {
        sub_pixel_variance_32x32_dotprod_inner(
            src.as_ptr(),
            src_stride,
            reference.as_ptr(),
            ref_stride,
            xoffset,
            yoffset,
        )
    }
}
subpel_var_dotprod_fn!(vpx_sub_pixel_variance32x64_dotprod, 32, 64);
subpel_var_dotprod_fn!(vpx_sub_pixel_variance64x32_dotprod, 64, 32);
subpel_var_dotprod_fn!(vpx_sub_pixel_variance64x64_dotprod, 64, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::dsp::SimdDispatch;
    use crate::dsp::caps::AArch64Caps;
    use crate::dsp::distortion::sub_pixel_variance as scalar_svf;

    const SHAPES: &[(BlockSize, usize, usize)] = &[
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

    fn neon_for_shape(
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        offsets: (usize, usize),
        shape: (usize, usize),
    ) -> (u32, u32) {
        let (xoffset, yoffset) = offsets;
        let (width, height) = shape;
        if width == 8 {
            match height {
                8 => vpx_sub_pixel_variance8x8_neon(
                    src, src_stride, xoffset, yoffset, reference, ref_stride,
                ),
                16 => vpx_sub_pixel_variance8x16_neon(
                    src, src_stride, xoffset, yoffset, reference, ref_stride,
                ),
                _ => unreachable!(),
            }
        } else {
            sub_pixel_variance_neon(
                src, src_stride, reference, ref_stride, xoffset, yoffset, width, height,
            )
        }
    }

    fn dotprod_for_shape(
        src: &[u8],
        src_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        offsets: (usize, usize),
        shape: (usize, usize),
    ) -> (u32, u32) {
        let (xoffset, yoffset) = offsets;
        let (width, height) = shape;
        if width == 8 {
            match height {
                8 => vpx_sub_pixel_variance8x8_dotprod(
                    src, src_stride, xoffset, yoffset, reference, ref_stride,
                ),
                16 => vpx_sub_pixel_variance8x16_dotprod(
                    src, src_stride, xoffset, yoffset, reference, ref_stride,
                ),
                _ => unreachable!(),
            }
        } else {
            sub_pixel_variance_dotprod(
                src, src_stride, reference, ref_stride, xoffset, yoffset, width, height,
            )
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
                    let neon = neon_for_shape(
                        &src,
                        src_stride,
                        &reference,
                        ref_stride,
                        (xoffset, yoffset),
                        (width, height),
                    );
                    let dotprod = dotprod_for_shape(
                        &src,
                        src_stride,
                        &reference,
                        ref_stride,
                        (xoffset, yoffset),
                        (width, height),
                    );
                    assert_eq!(
                        scalar, neon,
                        "mismatch at {width}x{height}, offset ({xoffset},{yoffset})"
                    );
                    assert_eq!(
                        scalar, dotprod,
                        "dotprod mismatch at {width}x{height}, offset ({xoffset},{yoffset})"
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
                        neon_for_shape(
                            &src,
                            src_stride,
                            &reference,
                            ref_stride,
                            (xoffset, yoffset),
                            (width, height),
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
        let neon = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: false,
        });
        let dotprod = SimdDispatch::build(AArch64Caps {
            neon: true,
            dotprod: true,
        });
        let scalar = SimdDispatch::build(AArch64Caps {
            neon: false,
            dotprod: false,
        });
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
            let dotprod_fn = dotprod.subpel_variance(bsize).unwrap();
            let scalar_fn = scalar.subpel_variance(bsize).unwrap();
            for &(xoffset, yoffset) in &[(0, 0), (4, 4), (3, 5)] {
                assert_eq!(
                    neon_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    scalar_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    "dispatch mismatch at {bsize:?}, offset ({xoffset},{yoffset})"
                );
                assert_eq!(
                    dotprod_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    scalar_fn(&src, src_stride, xoffset, yoffset, &reference, ref_stride,),
                    "dotprod dispatch mismatch at {bsize:?}, offset ({xoffset},{yoffset})"
                );
            }
        }
    }
}
