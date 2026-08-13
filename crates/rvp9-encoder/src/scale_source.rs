//! Source-frame scaling for the real-time spatial-SVC pipeline.
//!
//! Pinned libvpx uses `vp9_scale_and_extend_frame` with phase 8 for the
//! canonical 2:1 spatial steps. This module keeps that operation separate
//! from reference-frame scaling: it filters the uncompressed I420 source
//! before a layer is encoded.

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::inter::filter::{BILINEAR_KERNEL, EIGHTTAP_SMOOTH_KERNEL, FILTER_BITS};

const FILTER_TAPS: usize = 8;
const FILTER_CENTER: isize = 3;
const HALF_PIXEL_PHASE: usize = 8;

/// A tightly packed 8-bit I420 frame produced by [`scale_i420_2x`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaledI420 {
    /// Luma width in pixels.
    pub width: usize,
    /// Luma height in pixels.
    pub height: usize,
    /// Tightly packed luma plane.
    pub y: Vec<u8>,
    /// Tightly packed 4:2:0 Cb plane.
    pub u: Vec<u8>,
    /// Tightly packed 4:2:0 Cr plane.
    pub v: Vec<u8>,
}

/// Source-scaling input error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ScaleSourceError {
    /// Canonical I420 2:1 scaling requires positive dimensions divisible by 4.
    InvalidDimensions,
    /// One or more input planes do not match the dimensions exactly.
    InvalidPlaneLength,
    /// Spatial SVC currently supports libvpx's bilinear and low-resolution
    /// eight-tap-smooth source filters.
    UnsupportedFilter,
}

impl std::fmt::Display for ScaleSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => f.write_str(
                "I420 source dimensions must be positive and divisible by 4 for 2x scaling",
            ),
            Self::InvalidPlaneLength => {
                f.write_str("I420 source plane lengths do not match the frame dimensions")
            }
            Self::UnsupportedFilter => {
                f.write_str("source scaling supports only bilinear or eight-tap-smooth filtering")
            }
        }
    }
}

impl std::error::Error for ScaleSourceError {}

/// Downscale one tightly packed 8-bit I420 frame by exactly 2× in each axis.
///
/// This is the canonical step used by two-layer `1/2, 1/1` and three-layer
/// `1/4, 1/2, 1/1` spatial SVC. A 4× base layer is formed by calling this
/// function twice, matching pinned libvpx's `vp9_svc_twostage_scale` path.
/// The filter is evaluated at subpixel phase 8, which is libvpx's default
/// symmetric averaging phase for spatial layers.
///
/// # Errors
///
/// Returns [`ScaleSourceError::InvalidDimensions`] unless the luma dimensions
/// are positive multiples of four, [`ScaleSourceError::InvalidPlaneLength`]
/// unless all planes are tightly packed with their exact I420 lengths, or
/// [`ScaleSourceError::UnsupportedFilter`] for filters outside the v1 spatial
/// source-scaling surface.
pub fn scale_i420_2x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
    filter: InterpFilter,
) -> Result<ScaledI420, ScaleSourceError> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(4)
        || !height.is_multiple_of(4)
        || width > isize::MAX as usize
        || height > isize::MAX as usize
    {
        return Err(ScaleSourceError::InvalidDimensions);
    }
    let y_len = width
        .checked_mul(height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_len = uv_width
        .checked_mul(uv_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    if y.len() != y_len || u.len() != uv_len || v.len() != uv_len {
        return Err(ScaleSourceError::InvalidPlaneLength);
    }

    let kernel = match filter {
        InterpFilter::Bilinear => &BILINEAR_KERNEL[HALF_PIXEL_PHASE],
        InterpFilter::EightTapSmooth => &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE],
        _ => return Err(ScaleSourceError::UnsupportedFilter),
    };
    let out_width = width / 2;
    let out_height = height / 2;
    Ok(ScaledI420 {
        width: out_width,
        height: out_height,
        y: scale_plane_2x(y, width, height, kernel)?,
        u: scale_plane_2x(u, uv_width, uv_height, kernel)?,
        v: scale_plane_2x(v, uv_width, uv_height, kernel)?,
    })
}

/// Downscale one tightly packed 8-bit I420 frame by exactly 4× with pinned
/// libvpx's non-normative resize filter.
///
/// Pinned libvpx falls back to `vp9_resize_plane` when the intermediate buffer
/// for its preferred two-stage SVC path has an aligned allocation larger than
/// the visible half-resolution image. Exact 4× scaling performs two clipped
/// symmetric down-by-two passes horizontally, then the same two passes
/// vertically. Keeping this separate from [`scale_i420_2x`] makes that
/// allocation-dependent compatibility rule explicit.
pub(crate) fn scale_i420_nonnormative_4x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> Result<ScaledI420, ScaleSourceError> {
    if width == 0 || height == 0 || !width.is_multiple_of(8) || !height.is_multiple_of(8) {
        return Err(ScaleSourceError::InvalidDimensions);
    }
    let y_len = width
        .checked_mul(height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_len = uv_width
        .checked_mul(uv_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    if y.len() != y_len || u.len() != uv_len || v.len() != uv_len {
        return Err(ScaleSourceError::InvalidPlaneLength);
    }

    Ok(ScaledI420 {
        width: width / 4,
        height: height / 4,
        y: scale_plane_nonnormative_4x(y, width, height)?,
        u: scale_plane_nonnormative_4x(u, uv_width, uv_height)?,
        v: scale_plane_nonnormative_4x(v, uv_width, uv_height)?,
    })
}

fn scale_plane_nonnormative_4x(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleSourceError> {
    if !width.is_multiple_of(4) || !height.is_multiple_of(4) {
        return Err(ScaleSourceError::InvalidDimensions);
    }
    let out_width = width / 4;
    let out_height = height / 4;
    let horizontal_len = out_width
        .checked_mul(height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut horizontal = vec![0u8; horizontal_len];
    let mut half_row = vec![0u8; width / 2];
    for row in 0..height {
        let input_row = &input[row * width..(row + 1) * width];
        down2_symeven(input_row, &mut half_row);
        down2_symeven(
            &half_row,
            &mut horizontal[row * out_width..(row + 1) * out_width],
        );
    }

    let mut output = vec![0u8; output_len];
    let mut column = vec![0u8; height];
    let mut half_column = vec![0u8; height / 2];
    let mut output_column = vec![0u8; out_height];
    for col in 0..out_width {
        for row in 0..height {
            column[row] = horizontal[row * out_width + col];
        }
        down2_symeven(&column, &mut half_column);
        down2_symeven(&half_column, &mut output_column);
        for row in 0..out_height {
            output[row * out_width + col] = output_column[row];
        }
    }
    Ok(output)
}

fn down2_symeven(input: &[u8], output: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the helper checks slice lengths before using vector loads,
        // handles both edges scalarly, and stores only complete eight-byte
        // output groups.
        unsafe { down2_symeven_neon(input, output) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        down2_symeven_scalar(input, output);
    }
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn down2_symeven_scalar(input: &[u8], output: &mut [u8]) {
    debug_assert!(input.len().is_multiple_of(2));
    debug_assert_eq!(output.len(), input.len() / 2);
    for (output_index, output_value) in output.iter_mut().enumerate() {
        *output_value = down2_symeven_sample(input, output_index);
    }
}

#[inline]
fn down2_symeven_sample(input: &[u8], output_index: usize) -> u8 {
    const FILTER: [i32; 4] = [56, 12, -3, -1];
    let last = input.len() - 1;
    let center = output_index * 2;
    let mut sum = 1 << (FILTER_BITS - 1);
    for (distance, coefficient) in FILTER.into_iter().enumerate() {
        let left = center.saturating_sub(distance);
        let right = center.saturating_add(1 + distance).min(last);
        sum += (i32::from(input[left]) + i32::from(input[right])) * coefficient;
    }
    (sum >> FILTER_BITS).clamp(0, 255) as u8
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn down2_symeven_neon(input: &[u8], output: &mut [u8]) {
    use core::arch::aarch64::{
        int16x8_t, int32x4_t, vaddl_u8, vcombine_s16, vdupq_n_s32, vget_high_s16, vget_low_s16,
        vld2_u8, vmlal_n_s16, vqmovn_s32, vqmovun_s16, vreinterpretq_s16_u16, vshrq_n_s32, vst1_u8,
    };

    debug_assert!(input.len().is_multiple_of(2));
    debug_assert_eq!(output.len(), input.len() / 2);
    let leading = output.len().min(2);
    for (output_index, output_value) in output.iter_mut().take(leading).enumerate() {
        *output_value = down2_symeven_sample(input, output_index);
    }

    let mut output_index = leading;
    while output_index + 8 <= output.len() && output_index * 2 + 20 <= input.len() {
        let input_index = output_index * 2;
        let mut acc_low: int32x4_t = vdupq_n_s32(1 << (FILTER_BITS - 1));
        let mut acc_high: int32x4_t = vdupq_n_s32(1 << (FILTER_BITS - 1));
        // SAFETY: the loop condition leaves three readable bytes before the
        // first requested sample and at least twenty bytes from `input_index`.
        // Every `vld2_u8` therefore reads sixteen initialized input bytes, and
        // the complete eight-byte store stays inside `output`.
        unsafe {
            for (distance, coefficient) in [56i16, 12, -3, -1].into_iter().enumerate() {
                let left = if distance == 0 {
                    vld2_u8(input.as_ptr().add(input_index)).0
                } else {
                    vld2_u8(input.as_ptr().add(input_index - distance)).0
                };
                let right = if distance == 0 {
                    vld2_u8(input.as_ptr().add(input_index)).1
                } else {
                    vld2_u8(input.as_ptr().add(input_index + 1 + distance)).0
                };
                let pair: int16x8_t = vreinterpretq_s16_u16(vaddl_u8(left, right));
                acc_low = vmlal_n_s16(acc_low, vget_low_s16(pair), coefficient);
                acc_high = vmlal_n_s16(acc_high, vget_high_s16(pair), coefficient);
            }
            let packed = vcombine_s16(
                vqmovn_s32(vshrq_n_s32::<7>(acc_low)),
                vqmovn_s32(vshrq_n_s32::<7>(acc_high)),
            );
            vst1_u8(output.as_mut_ptr().add(output_index), vqmovun_s16(packed));
        }
        output_index += 8;
    }
    while output_index < output.len() {
        output[output_index] = down2_symeven_sample(input, output_index);
        output_index += 1;
    }
}

fn scale_plane_2x(
    input: &[u8],
    width: usize,
    height: usize,
    kernel: &[i16; FILTER_TAPS],
) -> Result<Vec<u8>, ScaleSourceError> {
    if kernel == &BILINEAR_KERNEL[HALF_PIXEL_PHASE] {
        return scale_plane_bilinear_2x(input, width, height);
    }
    if kernel == &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE] {
        return scale_plane_smooth_2x(input, width, height);
    }
    scale_plane_2x_scalar(input, width, height, kernel)
}

fn scale_plane_bilinear_2x(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleSourceError> {
    let out_width = width / 2;
    let out_height = height / 2;
    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 always provides NEON. The input and output plane
        // lengths were validated by `scale_i420_2x`; the inner loop uses
        // exact-width chunks and leaves its tail to scalar code.
        unsafe {
            scale_plane_bilinear_2x_neon(
                input.as_ptr(),
                width,
                output.as_mut_ptr(),
                out_width,
                out_height,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scale_plane_bilinear_2x_scalar(input, width, out_width, out_height, &mut output);
    }
    Ok(output)
}

fn scale_plane_smooth_2x(
    input: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ScaleSourceError> {
    #[cfg(target_arch = "aarch64")]
    {
        scale_plane_smooth_2x_neon(
            input,
            width,
            height,
            &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE],
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scale_plane_2x_scalar(
            input,
            width,
            height,
            &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE],
        )
    }
}

#[cfg(target_arch = "aarch64")]
fn scale_plane_smooth_2x_neon(
    input: &[u8],
    width: usize,
    height: usize,
    kernel: &[i16; FILTER_TAPS],
) -> Result<Vec<u8>, ScaleSourceError> {
    let out_width = width / 2;
    let out_height = height / 2;
    let intermediate_height = height
        .checked_add(FILTER_TAPS - 1)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let intermediate_len = out_width
        .checked_mul(intermediate_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut intermediate = vec![0u8; intermediate_len];

    for source_y in 0..height {
        let source_row = &input[source_y * width..(source_y + 1) * width];
        let row_start = (source_y + FILTER_CENTER as usize) * out_width;
        let output_row = &mut intermediate[row_start..row_start + out_width];
        // SAFETY: AArch64 always provides NEON. Both slices cover the widths
        // passed to the inner routine, which keeps full vector reads inside
        // the visible row and handles replicated edges with scalar code.
        unsafe { scale_smooth_horizontal_row_neon(source_row, output_row, kernel) };
    }
    for row in 0..FILTER_CENTER as usize {
        intermediate.copy_within(
            FILTER_CENTER as usize * out_width..(FILTER_CENTER as usize + 1) * out_width,
            row * out_width,
        );
    }
    let last_visible_row = height + FILTER_CENTER as usize - 1;
    for row in height + FILTER_CENTER as usize..intermediate_height {
        intermediate.copy_within(
            last_visible_row * out_width..(last_visible_row + 1) * out_width,
            row * out_width,
        );
    }

    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];
    for y in 0..out_height {
        let intermediate_start = 2 * y * out_width;
        let output_start = y * out_width;
        // SAFETY: the intermediate contains `height + 7` complete rows. The
        // filter reads rows `2 * y..2 * y + 7`, and the destination pointer
        // addresses one complete output row.
        unsafe {
            scale_smooth_vertical_row_neon(
                intermediate.as_ptr().add(intermediate_start),
                out_width,
                output.as_mut_ptr().add(output_start),
                kernel,
            );
        }
    }
    Ok(output)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_smooth_samples_8_libvpx(
    samples: [core::arch::aarch64::uint8x8_t; FILTER_TAPS],
) -> core::arch::aarch64::uint8x8_t {
    use core::arch::aarch64::*;

    const COEFFICIENTS: [i16; FILTER_TAPS] = [-1, -4, 14, 55, 55, 14, -4, -1];
    let s0 = vreinterpretq_s16_u16(vmovl_u8(samples[0]));
    let s1 = vreinterpretq_s16_u16(vmovl_u8(samples[1]));
    let s2 = vreinterpretq_s16_u16(vmovl_u8(samples[2]));
    let s3 = vreinterpretq_s16_u16(vmovl_u8(samples[3]));
    let s4 = vreinterpretq_s16_u16(vmovl_u8(samples[4]));
    let s5 = vreinterpretq_s16_u16(vmovl_u8(samples[5]));
    let s6 = vreinterpretq_s16_u16(vmovl_u8(samples[6]));
    let s7 = vreinterpretq_s16_u16(vmovl_u8(samples[7]));
    let mut sum = vmulq_n_s16(s0, COEFFICIENTS[0]);
    sum = vmlaq_n_s16(sum, s1, COEFFICIENTS[1]);
    sum = vmlaq_n_s16(sum, s2, COEFFICIENTS[2]);
    sum = vmlaq_n_s16(sum, s5, COEFFICIENTS[5]);
    sum = vmlaq_n_s16(sum, s6, COEFFICIENTS[6]);
    sum = vmlaq_n_s16(sum, s7, COEFFICIENTS[7]);
    sum = vqaddq_s16(sum, vmulq_n_s16(s3, COEFFICIENTS[3]));
    sum = vqaddq_s16(sum, vmulq_n_s16(s4, COEFFICIENTS[4]));
    vqrshrun_n_s16::<7>(sum)
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_decimated_smooth_8_libvpx_neon(base: *const u8) -> core::arch::aarch64::uint8x8_t {
    use core::arch::aarch64::*;

    // SAFETY: callers prove that every sixteen-byte interleaved load stays
    // within its source row. The accumulation order matches pinned libvpx's
    // `scale_filter_8` and is exact for valid VP9 interpolation kernels.
    unsafe {
        filter_smooth_samples_8_libvpx([
            vld2_u8(base).0,
            vld2_u8(base.add(1)).0,
            vld2_u8(base.add(2)).0,
            vld2_u8(base.add(3)).0,
            vld2_u8(base.add(4)).0,
            vld2_u8(base.add(5)).0,
            vld2_u8(base.add(6)).0,
            vld2_u8(base.add(7)).0,
        ])
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_strided_smooth_8_libvpx_neon(
    base: *const u8,
    stride: usize,
) -> core::arch::aarch64::uint8x8_t {
    use core::arch::aarch64::*;

    // SAFETY: callers prove eight readable samples at each of the eight row
    // offsets. The shared arithmetic matches pinned libvpx's exact schedule.
    unsafe {
        filter_smooth_samples_8_libvpx([
            vld1_u8(base),
            vld1_u8(base.add(stride)),
            vld1_u8(base.add(2 * stride)),
            vld1_u8(base.add(3 * stride)),
            vld1_u8(base.add(4 * stride)),
            vld1_u8(base.add(5 * stride)),
            vld1_u8(base.add(6 * stride)),
            vld1_u8(base.add(7 * stride)),
        ])
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_smooth_horizontal_row_neon(
    source: &[u8],
    output: &mut [u8],
    kernel: &[i16; FILTER_TAPS],
) {
    let width = source.len();
    let mut x = 0;
    while x < 2.min(output.len()) {
        output[x] = smooth_horizontal_sample(source, x, kernel);
        x += 1;
    }
    // SAFETY: the vector bound leaves three samples before the first tap and
    // accounts for the extra byte read by the final interleaved load. Each
    // store writes exactly eight remaining output samples.
    unsafe {
        while x + 8 <= output.len() && 2 * x + 20 <= width {
            core::arch::aarch64::vst1_u8(
                output.as_mut_ptr().add(x),
                filter_decimated_smooth_8_libvpx_neon(source.as_ptr().add(2 * x - 3)),
            );
            x += 8;
        }
    }
    while x < output.len() {
        output[x] = smooth_horizontal_sample(source, x, kernel);
        x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_smooth_vertical_row_neon(
    intermediate: *const u8,
    width: usize,
    output: *mut u8,
    kernel: &[i16; FILTER_TAPS],
) {
    let mut x = 0;
    // SAFETY: the caller supplies eight readable intermediate rows and one
    // writable output row, all with `width` samples. Vector chunks and the
    // scalar tail partition that footprint exactly.
    unsafe {
        while x + 8 <= width {
            core::arch::aarch64::vst1_u8(
                output.add(x),
                filter_strided_smooth_8_libvpx_neon(intermediate.add(x), width),
            );
            x += 8;
        }
        while x < width {
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                sum += i32::from(*intermediate.add(tap * width + x)) * i32::from(coefficient);
            }
            *output.add(x) = round_and_clip(sum);
            x += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn smooth_horizontal_sample(source: &[u8], output_x: usize, kernel: &[i16; FILTER_TAPS]) -> u8 {
    let center = (output_x * 2) as isize;
    let mut sum = 0i32;
    for (tap, &coefficient) in kernel.iter().enumerate() {
        let source_x =
            (center + tap as isize - FILTER_CENTER).clamp(0, source.len() as isize - 1) as usize;
        sum += i32::from(source[source_x]) * i32::from(coefficient);
    }
    round_and_clip(sum)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_plane_bilinear_2x_neon(
    input: *const u8,
    width: usize,
    output: *mut u8,
    out_width: usize,
    out_height: usize,
) {
    use core::arch::aarch64::*;

    // SAFETY: the caller proves complete input and output plane footprints.
    // Each vector iteration consumes exactly 32 source bytes and produces 16
    // destination bytes; the scalar tail stays within the same rows.
    unsafe {
        for y in 0..out_height {
            let row0 = input.add(2 * y * width);
            let row1 = row0.add(width);
            let dst = output.add(y * out_width);
            let mut x = 0;
            while x + 16 <= out_width {
                let source_x = x * 2;
                let a = vld2q_u8(row0.add(source_x));
                let b = vld2q_u8(row1.add(source_x));
                let horizontal0 = vrhaddq_u8(a.0, a.1);
                let horizontal1 = vrhaddq_u8(b.0, b.1);
                vst1q_u8(dst.add(x), vrhaddq_u8(horizontal0, horizontal1));
                x += 16;
            }
            while x < out_width {
                let source_x = x * 2;
                let horizontal0 =
                    (u16::from(*row0.add(source_x)) + u16::from(*row0.add(source_x + 1)) + 1) >> 1;
                let horizontal1 =
                    (u16::from(*row1.add(source_x)) + u16::from(*row1.add(source_x + 1)) + 1) >> 1;
                *dst.add(x) = ((horizontal0 + horizontal1 + 1) >> 1) as u8;
                x += 1;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn scale_plane_bilinear_2x_scalar(
    input: &[u8],
    width: usize,
    out_width: usize,
    out_height: usize,
    output: &mut [u8],
) {
    for y in 0..out_height {
        for x in 0..out_width {
            let source_x = x * 2;
            let top = (u16::from(input[2 * y * width + source_x])
                + u16::from(input[2 * y * width + source_x + 1])
                + 1)
                >> 1;
            let bottom = (u16::from(input[(2 * y + 1) * width + source_x])
                + u16::from(input[(2 * y + 1) * width + source_x + 1])
                + 1)
                >> 1;
            output[y * out_width + x] = ((top + bottom + 1) >> 1) as u8;
        }
    }
}

fn scale_plane_2x_scalar(
    input: &[u8],
    width: usize,
    height: usize,
    kernel: &[i16; FILTER_TAPS],
) -> Result<Vec<u8>, ScaleSourceError> {
    let out_width = width / 2;
    let out_height = height / 2;

    // `vpx_convolve8_c` first filters horizontally into an 8-bit
    // intermediate, then filters that vertically. The vertical footprint for
    // output rows 0..out_height spans source rows -3..height+3. Materialise
    // exactly those rows with visible-edge replication, as libvpx's extended
    // YV12 source does.
    let intermediate_height = height
        .checked_add(FILTER_TAPS - 1)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let intermediate_len = out_width
        .checked_mul(intermediate_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut intermediate = vec![0u8; intermediate_len];
    for intermediate_y in 0..intermediate_height {
        let source_y =
            (intermediate_y as isize - FILTER_CENTER).clamp(0, height as isize - 1) as usize;
        let source_row = &input[source_y * width..(source_y + 1) * width];
        let output_row =
            &mut intermediate[intermediate_y * out_width..(intermediate_y + 1) * out_width];
        for (x, output) in output_row.iter_mut().enumerate() {
            let center = (x * 2) as isize;
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                let source_x =
                    (center + tap as isize - FILTER_CENTER).clamp(0, width as isize - 1) as usize;
                sum += i32::from(source_row[source_x]) * i32::from(coefficient);
            }
            *output = round_and_clip(sum);
        }
    }

    let output_len = out_width
        .checked_mul(out_height)
        .ok_or(ScaleSourceError::InvalidDimensions)?;
    let mut output = vec![0u8; output_len];
    for y in 0..out_height {
        for x in 0..out_width {
            let first_intermediate_y = y * 2;
            let mut sum = 0i32;
            for (tap, &coefficient) in kernel.iter().enumerate() {
                let sample = intermediate[(first_intermediate_y + tap) * out_width + x];
                sum += i32::from(sample) * i32::from(coefficient);
            }
            output[y * out_width + x] = round_and_clip(sum);
        }
    }
    Ok(output)
}

#[inline]
fn round_and_clip(sum: i32) -> u8 {
    ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    fn random_i420(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x5ca1_ab1eu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        };
        let y = (0..width * height).map(|_| next()).collect();
        let uv_len = (width / 2) * (height / 2);
        let u = (0..uv_len).map(|_| next()).collect();
        let v = (0..uv_len).map(|_| next()).collect();
        (y, u, v)
    }

    fn packed_hash(frame: &ScaledI420) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&frame.y);
        hasher.update(&frame.u);
        hasher.update(&frame.v);
        format!("{:x}", hasher.finalize())
    }

    #[test]
    fn constant_planes_stay_constant() {
        let width = 64;
        let height = 32;
        let y = vec![17; width * height];
        let u = vec![91; width * height / 4];
        let v = vec![233; width * height / 4];
        for filter in [InterpFilter::Bilinear, InterpFilter::EightTapSmooth] {
            let scaled = scale_i420_2x(&y, &u, &v, width, height, filter).unwrap();
            assert_eq!(scaled.width, 32);
            assert_eq!(scaled.height, 16);
            assert!(scaled.y.iter().all(|&sample| sample == 17));
            assert!(scaled.u.iter().all(|&sample| sample == 91));
            assert!(scaled.v.iter().all(|&sample| sample == 233));
        }
    }

    #[test]
    fn optimized_bilinear_plane_matches_scalar_across_shapes() {
        let mut state = 0x12b1_1ea2u32;
        for (width, height) in [(8, 8), (32, 12), (64, 36), (128, 64)] {
            let input = (0..width * height)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect::<Vec<_>>();
            let scalar =
                scale_plane_2x_scalar(&input, width, height, &BILINEAR_KERNEL[HALF_PIXEL_PHASE])
                    .unwrap();
            let optimized = scale_plane_bilinear_2x(&input, width, height).unwrap();
            assert_eq!(optimized, scalar, "{width}x{height}");
        }
    }

    #[test]
    fn optimized_smooth_plane_matches_scalar_across_shapes() {
        let mut state = 0x58a1_e5cau32;
        for (width, height) in [(8, 8), (20, 12), (64, 36), (128, 64)] {
            let input = (0..width * height)
                .map(|index| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    match index % 17 {
                        0 => 0,
                        1 => 255,
                        _ => state as u8,
                    }
                })
                .collect::<Vec<_>>();
            let scalar = scale_plane_2x_scalar(
                &input,
                width,
                height,
                &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE],
            )
            .unwrap();
            let optimized = scale_plane_smooth_2x(&input, width, height).unwrap();
            assert_eq!(optimized, scalar, "{width}x{height}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nonnormative_down2_neon_matches_scalar_across_shapes() {
        let mut state = 0x4d32_4e45u32;
        for length in [8usize, 16, 32, 40, 270, 540, 960, 1920] {
            let input = (0..length)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state as u8
                })
                .collect::<Vec<_>>();
            let mut expected = vec![0u8; length / 2];
            let mut actual = vec![0u8; length / 2];
            down2_symeven_scalar(&input, &mut expected);
            // SAFETY: the source and destination slices have the exact 2:1
            // lengths required by the NEON helper.
            unsafe { down2_symeven_neon(&input, &mut actual) };
            assert_eq!(actual, expected, "length={length}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn libvpx_neon_smooth_schedule_matches_scalar_extremes() {
        use core::arch::aarch64::vget_lane_u8;

        let kernel = &EIGHTTAP_SMOOTH_KERNEL[HALF_PIXEL_PHASE];
        for stride in [1usize, 2] {
            for mask in 0u16..256 {
                let mut source = [17u8; 22];
                for tap in 0..FILTER_TAPS {
                    source[tap * stride] = if mask & (1 << tap) == 0 { 0 } else { 255 };
                }
                // SAFETY: `source` covers the helper's eight-byte load at
                // every one of the eight tap offsets for both tested strides.
                let vector =
                    unsafe { filter_strided_smooth_8_libvpx_neon(source.as_ptr(), stride) };
                // SAFETY: lane zero is present in every uint8x8 vector.
                let actual = unsafe { vget_lane_u8::<0>(vector) };
                let expected = round_and_clip(
                    kernel
                        .iter()
                        .enumerate()
                        .map(|(tap, &coefficient)| {
                            i32::from(source[tap * stride]) * i32::from(coefficient)
                        })
                        .sum(),
                );
                assert_eq!(actual, expected, "stride={stride} mask={mask:#04x}");
            }
        }
    }

    #[test]
    fn validates_dimensions_planes_and_filter() {
        assert_eq!(
            scale_i420_2x(&[], &[], &[], 0, 4, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidDimensions)
        );
        assert_eq!(
            scale_i420_2x(&[], &[], &[], 6, 4, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidDimensions)
        );
        assert_eq!(
            scale_i420_2x(&[0; 64], &[0; 15], &[0; 16], 8, 8, InterpFilter::Bilinear),
            Err(ScaleSourceError::InvalidPlaneLength)
        );
        assert_eq!(
            scale_i420_2x(&[0; 64], &[0; 16], &[0; 16], 8, 8, InterpFilter::EightTap),
            Err(ScaleSourceError::UnsupportedFilter)
        );
    }

    #[test]
    fn random_720p_half_scale_matches_pinned_libvpx() {
        let (y, u, v) = random_i420(1280, 720);
        let bilinear = scale_i420_2x(&y, &u, &v, 1280, 720, InterpFilter::Bilinear).unwrap();
        assert_eq!(
            packed_hash(&bilinear),
            "2f2f77757ac5c0383bec6d7a4e7f162213373f34c4cd5408d12071d3e2934741"
        );
        let smooth = scale_i420_2x(&y, &u, &v, 1280, 720, InterpFilter::EightTapSmooth).unwrap();
        assert_eq!(
            packed_hash(&smooth),
            "5616cf7b5089ad6af0c6e47b97986bdd7758663bdde50b9eaf43a39c2b133d97"
        );
    }

    #[test]
    fn random_720p_two_stage_quarter_scale_matches_pinned_libvpx() {
        let (y, u, v) = random_i420(1280, 720);
        for (filter, expected) in [
            (
                InterpFilter::Bilinear,
                "a3ead90551cdf43d97a531a7bb3c6cab61fdb3fd9cb81e3058b800b8c261e470",
            ),
            (
                InterpFilter::EightTapSmooth,
                "7c115cdcc64fbb9594d4c575cd371ddab5b1ff09f0c3c786aa70b191912adc26",
            ),
        ] {
            let half = scale_i420_2x(&y, &u, &v, 1280, 720, filter).unwrap();
            let quarter =
                scale_i420_2x(&half.y, &half.u, &half.v, half.width, half.height, filter).unwrap();
            assert_eq!((quarter.width, quarter.height), (320, 180));
            assert_eq!(packed_hash(&quarter), expected);
        }
    }
}
