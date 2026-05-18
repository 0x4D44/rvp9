//! 8-tap separable sub-pel interpolation kernels (the `vpx_convolve8`
//! family) used by inter-prediction motion compensation.
//!
//! Mirrors libvpx's scalar reference at `vpx_dsp/vpx_convolve.c` plus
//! the four `InterpKernel` tables in `vp9/common/vp9_filter.c`:
//!
//! - [`EIGHTTAP_KERNEL`]        — `sub_pel_filters_8` (Lagrangian).
//! - [`EIGHTTAP_SMOOTH_KERNEL`] — `sub_pel_filters_8lp` (low-pass).
//! - [`EIGHTTAP_SHARP_KERNEL`]  — `sub_pel_filters_8s` (DCT-based).
//! - [`BILINEAR_KERNEL`]        — `bilinear_filters` (2-tap, padded).
//!
//! ## Scope (Task C.1)
//!
//! Pure compute kernels — no decoder integration, no per-arch SIMD.
//! Phase D will wire these into motion compensation.
//!
//! ## API shape
//!
//! The spec'd API is the *unscaled* convolve variant: filter taps are
//! a single `[i16; 8]` row (already indexed by sub-pel offset), and
//! the kernels do not step through `x_q4` / `y_q4` like libvpx's
//! reference. This matches the normative VP9 inter path — frame
//! scaling (and therefore non-unit `x_step_q4` / `y_step_q4`) is a
//! separate Phase D concern.
//!
//! Two-tap bilinear sits in the same `[i16; 8]` shape with the unused
//! taps zeroed, so a single kernel function handles both: the
//! multiply-by-zero is cheap and keeps the calling convention
//! uniform. Phase D may special-case `BILINEAR` for performance, but
//! it is not required for correctness.

use super::super::bitstream::frame::InterpFilter;

/// Filter taps per sub-pel kernel (libvpx `SUBPEL_TAPS`,
/// `vpx_dsp/vpx_filter.h:26`).
pub const FILTER_TAPS: usize = 8;

/// Sub-pel offset precision in bits (libvpx `SUBPEL_BITS`,
/// `vpx_dsp/vpx_filter.h:23`). Sub-pel positions span 0..16.
pub const SUBPEL_BITS: usize = 4;

/// Number of sub-pel filter rows per `InterpKernel`
/// (libvpx `SUBPEL_SHIFTS = 1 << SUBPEL_BITS`).
pub const SUBPEL_SHIFTS: usize = 1 << SUBPEL_BITS;

/// Alias for [`FILTER_TAPS`] — matches libvpx `SUBPEL_TAPS` naming.
pub const SUBPEL_TAPS: usize = FILTER_TAPS;

/// Fixed-point bits applied to the filter sum before the final
/// `clip_pixel` (libvpx `FILTER_BITS`, `vpx_dsp/vpx_filter.h:21`).
pub const FILTER_BITS: u32 = 7;

/// Bias added before the `>> FILTER_BITS` rounding shift
/// (libvpx `ROUND_POWER_OF_TWO`, `vpx_ports/mem.h:31`).
const FILTER_ROUNDING: i32 = 1 << (FILTER_BITS - 1);

/// 8-tap regular (Lagrangian) interpolation filter.
///
/// Verbatim transcription of libvpx `sub_pel_filters_8`
/// (`vp9/common/vp9_filter.c:28-38`).
pub static EIGHTTAP_KERNEL: [[i16; 8]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// 8-tap low-pass ("smooth") interpolation filter — `freqmultiplier =
/// 0.5`.
///
/// Verbatim transcription of libvpx `sub_pel_filters_8lp`
/// (`vp9/common/vp9_filter.c:54-64`).
pub static EIGHTTAP_SMOOTH_KERNEL: [[i16; 8]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 43, 4, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 18, 59, 51, 9, -4, 0],
    [-1, -4, 16, 57, 53, 12, -4, -1],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [-1, -4, 12, 53, 57, 16, -4, -1],
    [0, -4, 9, 51, 59, 18, -4, -1],
    [0, -4, 7, 49, 60, 21, -3, -2],
    [0, -4, 5, 46, 62, 24, -3, -2],
    [0, -4, 4, 43, 63, 26, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 1, 38, 64, 32, -1, -3],
];

/// 8-tap sharp (DCT-based) interpolation filter.
///
/// Verbatim transcription of libvpx `sub_pel_filters_8s`
/// (`vp9/common/vp9_filter.c:41-51`).
pub static EIGHTTAP_SHARP_KERNEL: [[i16; 8]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
    [-4, 11, -23, 80, 80, -23, 11, -4],
    [-4, 10, -21, 70, 90, -24, 11, -4],
    [-3, 9, -19, 59, 100, -24, 10, -4],
    [-3, 8, -16, 48, 108, -23, 10, -4],
    [-2, 6, -13, 37, 115, -20, 9, -4],
    [-2, 5, -10, 27, 121, -17, 7, -3],
    [-1, 3, -6, 17, 125, -13, 5, -2],
    [0, 1, -3, 8, 127, -7, 3, -1],
];

/// 2-tap bilinear filter, padded to 8 taps with leading/trailing
/// zeros so the shared 8-tap kernel handles it without a special
/// case.
///
/// Verbatim transcription of libvpx `bilinear_filters`
/// (`vp9/common/vp9_filter.c:15-25`).
pub static BILINEAR_KERNEL: [[i16; 8]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 0, 120, 8, 0, 0, 0],
    [0, 0, 0, 112, 16, 0, 0, 0],
    [0, 0, 0, 104, 24, 0, 0, 0],
    [0, 0, 0, 96, 32, 0, 0, 0],
    [0, 0, 0, 88, 40, 0, 0, 0],
    [0, 0, 0, 80, 48, 0, 0, 0],
    [0, 0, 0, 72, 56, 0, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 56, 72, 0, 0, 0],
    [0, 0, 0, 48, 80, 0, 0, 0],
    [0, 0, 0, 40, 88, 0, 0, 0],
    [0, 0, 0, 32, 96, 0, 0, 0],
    [0, 0, 0, 24, 104, 0, 0, 0],
    [0, 0, 0, 16, 112, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

/// Look up the static kernel table associated with the given filter
/// enum (libvpx `vp9_filter_kernels[]`, `vp9_filter.c:79-82`).
///
/// `InterpFilter::Switchable` is a frame-level sentinel — the
/// per-block decode resolves it to one of the other four before
/// reaching the convolver, so it has no kernel of its own. Calling
/// this with `Switchable` panics under `debug_assertions` and falls
/// back to [`EIGHTTAP_KERNEL`] in release; callers should never hit
/// that path.
#[must_use]
pub fn kernel_for(filter: InterpFilter) -> &'static [[i16; 8]; SUBPEL_SHIFTS] {
    match filter {
        InterpFilter::EightTap => &EIGHTTAP_KERNEL,
        InterpFilter::EightTapSmooth => &EIGHTTAP_SMOOTH_KERNEL,
        InterpFilter::EightTapSharp => &EIGHTTAP_SHARP_KERNEL,
        InterpFilter::Bilinear => &BILINEAR_KERNEL,
        InterpFilter::Switchable => {
            debug_assert!(false, "kernel_for(Switchable) — resolve per-block first");
            &EIGHTTAP_KERNEL
        }
    }
}

/// Apply rounding + clipping to a single 8-tap sum, matching libvpx
/// `clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS))`
/// (`vpx_dsp/vpx_convolve.c:36`).
#[inline]
fn round_clip(sum: i32) -> u8 {
    let rounded = (sum + FILTER_ROUNDING) >> FILTER_BITS;
    rounded.clamp(0, 255) as u8
}

/// Average two pixel candidates with rounding-up bias, matching
/// libvpx `ROUND_POWER_OF_TWO(a + b, 1)` (`vpx_convolve.c:59`).
#[inline]
fn round_avg(a: u8, b: u8) -> u8 {
    ((u32::from(a) + u32::from(b) + 1) >> 1) as u8
}

/// One 8-tap horizontal sample: reads `src[col + k]` for
/// `k ∈ 0..FILTER_TAPS`. Callers shift `src` left by `SUBPEL_TAPS/2
/// - 1 = 3` so the tap centre lines up with the destination column —
/// matching libvpx's `src -= SUBPEL_TAPS / 2 - 1` pre-shift
/// (`vpx_dsp/vpx_convolve.c:27`).
#[inline]
fn convolve_horiz_sample(src: &[u8], col: usize, filter: &[i16; 8]) -> i32 {
    let mut sum: i32 = 0;
    for k in 0..FILTER_TAPS {
        sum += i32::from(src[col + k]) * i32::from(filter[k]);
    }
    sum
}

/// One 8-tap vertical sample: reads `src[(row + k) * stride + col]`
/// for `k ∈ 0..FILTER_TAPS`. Callers shift `src` up by 3 rows.
#[inline]
fn convolve_vert_sample(
    src: &[u8],
    src_stride: usize,
    row: usize,
    col: usize,
    filter: &[i16; 8],
) -> i32 {
    let mut sum: i32 = 0;
    for k in 0..FILTER_TAPS {
        sum += i32::from(src[(row + k) * src_stride + col]) * i32::from(filter[k]);
    }
    sum
}

/// Horizontal 8-tap convolution writing `w × h` samples.
///
/// Mirrors libvpx `convolve_horiz` (`vpx_dsp/vpx_convolve.c:22-42`)
/// with `x_step_q4 == SUBPEL_SHIFTS` (no scaling). `filter_x` is the
/// already-resolved tap row — pick it from a kernel table with
/// `&KERNEL[x_offset_q4]`.
///
/// # Layout
///
/// libvpx applies `src -= SUBPEL_TAPS / 2 - 1` (i.e. shift left by
/// 3 columns) before entering the inner loop. This kernel matches
/// that convention: callers must pass `src` already shifted so that
/// `src[y * src_stride + x]` is the *first* of the eight taps that
/// produce the output at `dst[y * dst_stride + x]` (i.e. three
/// columns to the left of the destination origin).
///
/// Concretely, with a reference buffer `frame[stride * H]` and a
/// destination origin at `(ox, oy)`, pass
/// `&frame[oy * stride + ox - 3]` as `src`.
///
/// # Panics
///
/// Panics (via slice indexing) if `src` is shorter than the required
/// footprint (`(h - 1) * src_stride + w + FILTER_TAPS - 1`) or
/// `dst` is shorter than `(h - 1) * dst_stride + w`.
pub fn convolve8_horiz(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let src_row = &src[y * src_stride..];
        for x in 0..w {
            let sum = convolve_horiz_sample(src_row, x, filter_x);
            dst[y * dst_stride + x] = round_clip(sum);
        }
    }
}

/// Vertical 8-tap convolution writing `w × h` samples.
///
/// Mirrors libvpx `convolve_vert` (`vpx_dsp/vpx_convolve.c:67-88`)
/// with `y_step_q4 == SUBPEL_SHIFTS`. Callers must pre-shift `src`
/// up by `SUBPEL_TAPS / 2 - 1 = 3` rows so that
/// `src[y * src_stride + x]` is the first of the eight taps that
/// produce the output at `dst[y * dst_stride + x]` — analogous to
/// the horizontal convention.
pub fn convolve8_vert(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let sum = convolve_vert_sample(src, src_stride, y, x, filter_y);
            dst[y * dst_stride + x] = round_clip(sum);
        }
    }
}

/// Full 2-D separable 8-tap convolution: horizontal pass into a
/// stack scratch buffer, then vertical pass into `dst`.
///
/// Mirrors libvpx `vpx_convolve8_c` (`vpx_dsp/vpx_convolve.c:158-190`)
/// with both step sizes fixed at `SUBPEL_SHIFTS`. The scratch buffer
/// uses libvpx's worst-case `64 × 135` shape so any VP9 inter-block
/// size up to 64×64 fits without heap allocation. Callers requesting
/// larger blocks will panic — VP9's normative path never exceeds
/// 64×64.
///
/// # Layout
///
/// Callers must pre-shift `src` so that `src[0]` is the **top-left
/// corner of the 2-D filter footprint** — three columns to the left
/// *and* three rows above the destination origin. With a reference
/// buffer `frame[stride * H]` and a destination origin at `(ox, oy)`,
/// pass `&frame[(oy - 3) * stride + (ox - 3)]`. The kernel reads
/// `src[r * src_stride + c]` for `r ∈ 0..h + 7`, `c ∈ 0..w + 6`.
///
/// # Panics
///
/// Panics if `w > 64` or `h > 64`.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
pub fn convolve8(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(w <= 64, "convolve8: w must be <= 64");
    assert!(h <= 64, "convolve8: h must be <= 64");
    let intermediate_height = h + FILTER_TAPS - 1;
    let intermediate_stride = 64;
    let mut intermediate = [0u8; 64 * 135];

    // Horizontal pass: filter `intermediate_height` rows. `src` is
    // already pre-shifted up by 3 rows by the caller, so row 0 of
    // `src` corresponds to row 0 of the intermediate buffer.
    for y in 0..intermediate_height {
        let src_row = &src[y * src_stride..];
        for x in 0..w {
            let sum = convolve_horiz_sample(src_row, x, filter_x);
            intermediate[y * intermediate_stride + x] = round_clip(sum);
        }
    }

    // Vertical pass: convolve8_vert expects its `src` to be
    // pre-shifted up by 3 rows. Our intermediate buffer's row 0 IS
    // already three rows above the destination origin, so we hand
    // the buffer to convolve8_vert as-is.
    convolve8_vert(
        &intermediate,
        intermediate_stride,
        dst,
        dst_stride,
        filter_y,
        w,
        h,
    );
}

/// Horizontal `_avg`: convolve and average against the existing
/// `dst` pixel.
///
/// Mirrors libvpx `convolve_avg_horiz` (`vpx_convolve.c:44-65`).
pub fn convolve8_avg_horiz(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let src_row = &src[y * src_stride..];
        for x in 0..w {
            let sum = convolve_horiz_sample(src_row, x, filter_x);
            let pred = round_clip(sum);
            let idx = y * dst_stride + x;
            dst[idx] = round_avg(dst[idx], pred);
        }
    }
}

/// Vertical `_avg`: convolve and average against the existing `dst`
/// pixel.
///
/// Mirrors libvpx `convolve_avg_vert` (`vpx_convolve.c:90-114`).
pub fn convolve8_avg_vert(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let sum = convolve_vert_sample(src, src_stride, y, x, filter_y);
            let pred = round_clip(sum);
            let idx = y * dst_stride + x;
            dst[idx] = round_avg(dst[idx], pred);
        }
    }
}

/// Full 2-D `_avg`: convolve then average against existing `dst`.
///
/// Mirrors libvpx `vpx_convolve8_avg_c` (`vpx_convolve.c:192-204`):
/// runs the full 2-D convolution into a scratch buffer, then
/// averages that buffer into `dst`.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
pub fn convolve8_avg(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(w <= 64, "convolve8_avg: w must be <= 64");
    assert!(h <= 64, "convolve8_avg: h must be <= 64");
    let mut pred = [0u8; 64 * 64];
    let pred_stride = 64;
    convolve8(
        src,
        src_stride,
        &mut pred,
        pred_stride,
        filter_x,
        filter_y,
        w,
        h,
    );
    for y in 0..h {
        for x in 0..w {
            let idx = y * dst_stride + x;
            dst[idx] = round_avg(dst[idx], pred[y * pred_stride + x]);
        }
    }
}

/// Integer-pel passthrough: copy a `w × h` block from `src` to `dst`.
///
/// Mirrors libvpx `vpx_convolve_copy_c` (`vpx_convolve.c:206-223`).
/// Used when both sub-pel offsets are zero — no filtering needed.
pub fn convolve_copy(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let src_row = &src[y * src_stride..y * src_stride + w];
        let dst_row = &mut dst[y * dst_stride..y * dst_stride + w];
        dst_row.copy_from_slice(src_row);
    }
}

/// Integer-pel compound average: `dst = (dst + src + 1) >> 1`.
///
/// Mirrors libvpx `vpx_convolve_avg_c` (`vpx_convolve.c:225-242`).
pub fn convolve_avg(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        for x in 0..w {
            let s = src[y * src_stride + x];
            let d = &mut dst[y * dst_stride + x];
            *d = round_avg(*d, s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Filter-table transcription correctness ------------------------

    /// Identity row (sub-pel offset 0) is the integer-pel passthrough
    /// for every kernel: tap 3 is 128, all others are zero.
    fn assert_identity_row(kernel: &[[i16; 8]; SUBPEL_SHIFTS]) {
        assert_eq!(kernel[0], [0, 0, 0, 128, 0, 0, 0, 0]);
    }

    /// Every sub-pel row sums to exactly 128 (`1 << FILTER_BITS`)
    /// because the kernels are normalised so DC passes unchanged.
    fn assert_kernel_sums_to_128(kernel: &[[i16; 8]; SUBPEL_SHIFTS]) {
        for (i, row) in kernel.iter().enumerate() {
            let sum: i32 = row.iter().map(|&t| i32::from(t)).sum();
            assert_eq!(sum, 128, "kernel row {i} sums to {sum}, expected 128");
        }
    }

    #[test]
    fn eighttap_table_invariants() {
        assert_identity_row(&EIGHTTAP_KERNEL);
        assert_kernel_sums_to_128(&EIGHTTAP_KERNEL);
        // Spot-check a few rows verbatim against libvpx
        // `sub_pel_filters_8` (vp9_filter.c:28-38).
        assert_eq!(EIGHTTAP_KERNEL[1], [0, 1, -5, 126, 8, -3, 1, 0]);
        assert_eq!(EIGHTTAP_KERNEL[8], [-1, 6, -19, 78, 78, -19, 6, -1]);
        assert_eq!(EIGHTTAP_KERNEL[15], [0, 1, -3, 8, 126, -5, 1, 0]);
    }

    #[test]
    fn eighttap_smooth_table_invariants() {
        assert_identity_row(&EIGHTTAP_SMOOTH_KERNEL);
        assert_kernel_sums_to_128(&EIGHTTAP_SMOOTH_KERNEL);
        assert_eq!(EIGHTTAP_SMOOTH_KERNEL[1], [-3, -1, 32, 64, 38, 1, -3, 0]);
        assert_eq!(EIGHTTAP_SMOOTH_KERNEL[8], [-1, -4, 14, 55, 55, 14, -4, -1]);
        assert_eq!(EIGHTTAP_SMOOTH_KERNEL[15], [0, -3, 1, 38, 64, 32, -1, -3]);
    }

    #[test]
    fn eighttap_sharp_table_invariants() {
        assert_identity_row(&EIGHTTAP_SHARP_KERNEL);
        assert_kernel_sums_to_128(&EIGHTTAP_SHARP_KERNEL);
        assert_eq!(EIGHTTAP_SHARP_KERNEL[1], [-1, 3, -7, 127, 8, -3, 1, 0]);
        assert_eq!(EIGHTTAP_SHARP_KERNEL[8], [-4, 11, -23, 80, 80, -23, 11, -4]);
        assert_eq!(EIGHTTAP_SHARP_KERNEL[15], [0, 1, -3, 8, 127, -7, 3, -1]);
    }

    #[test]
    fn bilinear_table_invariants() {
        assert_identity_row(&BILINEAR_KERNEL);
        assert_kernel_sums_to_128(&BILINEAR_KERNEL);
        // Bilinear pads taps 0..3 and 5..8 with zeros — only taps 3
        // and 4 carry weight.
        for (i, row) in BILINEAR_KERNEL.iter().enumerate() {
            assert_eq!(row[0], 0, "bilinear[{i}][0] non-zero");
            assert_eq!(row[1], 0, "bilinear[{i}][1] non-zero");
            assert_eq!(row[2], 0, "bilinear[{i}][2] non-zero");
            assert_eq!(row[5], 0, "bilinear[{i}][5] non-zero");
            assert_eq!(row[6], 0, "bilinear[{i}][6] non-zero");
            assert_eq!(row[7], 0, "bilinear[{i}][7] non-zero");
        }
        assert_eq!(BILINEAR_KERNEL[8], [0, 0, 0, 64, 64, 0, 0, 0]);
    }

    #[test]
    fn kernel_for_dispatch() {
        // Pointer equality: `kernel_for` must hand back the exact
        // static table, not a copy.
        assert!(std::ptr::eq(
            kernel_for(InterpFilter::EightTap),
            &EIGHTTAP_KERNEL
        ));
        assert!(std::ptr::eq(
            kernel_for(InterpFilter::EightTapSmooth),
            &EIGHTTAP_SMOOTH_KERNEL
        ));
        assert!(std::ptr::eq(
            kernel_for(InterpFilter::EightTapSharp),
            &EIGHTTAP_SHARP_KERNEL
        ));
        assert!(std::ptr::eq(
            kernel_for(InterpFilter::Bilinear),
            &BILINEAR_KERNEL
        ));
    }

    // --- Algorithmic spot checks --------------------------------------

    /// Pad a `w x h` payload with a halo on each side so the 8-tap
    /// kernel can read its full footprint, and replicate the edge
    /// pixels into the halo. Returns `(buffer, stride, x_off,
    /// y_off)` where `(x_off, y_off)` is the destination origin
    /// within the buffer (the top-left of `payload`).
    fn pad_block(payload: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize, usize) {
        assert_eq!(payload.len(), w * h);
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut buf = vec![0u8; stride * padded_h];
        for y in 0..padded_h {
            for x in 0..stride {
                let cy = y.saturating_sub(pad).min(h - 1);
                let cx = x.saturating_sub(pad).min(w - 1);
                buf[y * stride + x] = payload[cy * w + cx];
            }
        }
        (buf, stride, pad, pad)
    }

    /// Slice offset for `convolve8_horiz` callers: shift left by 3
    /// columns from the destination origin.
    fn horiz_src_start(xo: usize, yo: usize, stride: usize) -> usize {
        yo * stride + xo - 3
    }

    /// Slice offset for `convolve8_vert` callers: shift up by 3 rows
    /// from the destination origin.
    fn vert_src_start(xo: usize, yo: usize, stride: usize) -> usize {
        (yo - 3) * stride + xo
    }

    /// Slice offset for `convolve8` callers: shift left 3 cols AND
    /// up 3 rows from the destination origin.
    fn convolve8_src_start(xo: usize, yo: usize, stride: usize) -> usize {
        (yo - 3) * stride + xo - 3
    }

    #[test]
    fn integer_pel_horizontal_is_identity() {
        // sub-pel offset 0 ⇒ filter = [0,0,0,128,0,0,0,0]; output is
        // a verbatim copy of the column-aligned source row.
        let w = 4;
        let h = 4;
        let payload: Vec<u8> = (0..16u8).collect();
        let (src, stride, xo, yo) = pad_block(&payload, w, h);
        let mut dst = vec![0u8; w * h];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            w,
            &EIGHTTAP_KERNEL[0],
            w,
            h,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn integer_pel_vertical_is_identity() {
        let w = 4;
        let h = 4;
        let payload: Vec<u8> = (0..16u8).collect();
        let (src, stride, xo, yo) = pad_block(&payload, w, h);
        let mut dst = vec![0u8; w * h];
        convolve8_vert(
            &src[vert_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            w,
            &EIGHTTAP_KERNEL[0],
            w,
            h,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn integer_pel_2d_is_identity() {
        let w = 8;
        let h = 8;
        let payload: Vec<u8> = (0..64u8).map(|v| v.wrapping_mul(3)).collect();
        let (src, stride, xo, yo) = pad_block(&payload, w, h);
        let mut dst = vec![0u8; w * h];
        convolve8(
            &src[convolve8_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            w,
            &EIGHTTAP_KERNEL[0],
            &EIGHTTAP_KERNEL[0],
            w,
            h,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn bilinear_half_pel_horiz_constant_input() {
        // Half-pel horizontal on a constant row: bilinear[8] = (64,
        // 64) sandwiched between zeros, so the output equals the
        // input.
        let payload = vec![100u8; 16];
        let (src, stride, xo, yo) = pad_block(&payload, 4, 4);
        let mut dst = vec![0u8; 16];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            4,
            &BILINEAR_KERNEL[8],
            4,
            4,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn bilinear_half_pel_horiz_ramp() {
        // Half-pel horizontal of [0, 10, 20, 30] sees tap[3] =
        // src[dest_x] and tap[4] = src[dest_x + 1] (edge-replicated
        // by `pad_block`). Output = (s[x] + s[x+1] + 1) >> 1.
        let row: [u8; 4] = [0, 10, 20, 30];
        let mut payload = Vec::with_capacity(16);
        for _ in 0..4 {
            payload.extend_from_slice(&row);
        }
        let (src, stride, xo, yo) = pad_block(&payload, 4, 4);
        let mut dst = vec![0u8; 16];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            4,
            &BILINEAR_KERNEL[8],
            4,
            4,
        );
        // Expected: edge-replicated right-neighbour at x = 3 ⇒ 30.
        // Hand-coded: dst[x] = (row[x] + row[x+1] + 1) >> 1 for the
        // half-pel bilinear filter [0,0,0,64,64,0,0,0].
        let expected_row: [u8; 4] = [5, 15, 25, 30];
        for y in 0..4 {
            assert_eq!(&dst[y * 4..(y + 1) * 4], &expected_row);
        }
    }

    #[test]
    fn eighttap_q1_near_integer_constant_input() {
        // q1/16 sub-pel offset on a uniform field should reproduce
        // the field exactly: each row sums to 128, so the convolution
        // of a constant K is `K * 128 / 128 = K` after the +64 round.
        let payload = vec![137u8; 64];
        let (src, stride, xo, yo) = pad_block(&payload, 8, 8);
        let mut dst = vec![0u8; 64];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            8,
            &EIGHTTAP_KERNEL[1],
            8,
            8,
        );
        assert_eq!(dst, payload);
    }

    #[test]
    fn eighttap_half_pel_hand_traced() {
        // q8/16 EIGHTTAP filter = [-1, 6, -19, 78, 78, -19, 6, -1].
        // Apply at an interior pixel of an input where every pixel is
        // unique, and hand-check one output.
        let w = 16;
        let h = 1;
        let payload: Vec<u8> = (0..16u8).map(|v| 20 + v * 5).collect();
        let (src, stride, xo, yo) = pad_block(&payload, w, h);
        let mut dst = vec![0u8; w * h];
        let f = &EIGHTTAP_KERNEL[8];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            w,
            f,
            w,
            h,
        );
        // Check x = 5 (well inside the interior, no edge replication
        // needed). Inputs: payload[2..10] = [30, 35, 40, 45, 50, 55,
        // 60, 65]. sum = -1*30 + 6*35 - 19*40 + 78*45 + 78*50 - 19*55
        // + 6*60 - 1*65.
        let taps = [30i32, 35, 40, 45, 50, 55, 60, 65];
        let sum: i32 = taps.iter().zip(f.iter()).map(|(s, t)| s * i32::from(*t)).sum();
        let expected = ((sum + 64) >> 7).clamp(0, 255) as u8;
        assert_eq!(dst[5], expected);
    }

    #[test]
    fn convolve8_matches_separable_passes() {
        // Differential: a full 2-D `convolve8` must match a horiz
        // pass into a scratch buffer + a vert pass over the result —
        // with the scratch buffer including the extra 7 rows of
        // vertical context.
        let w = 8;
        let h = 8;
        let mut rng_state = 0xdead_beefu32;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            (rng_state & 0xff) as u8
        };
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut src = vec![0u8; stride * padded_h];
        for cell in src.iter_mut() {
            *cell = next();
        }
        let filter_x = &EIGHTTAP_SHARP_KERNEL[5];
        let filter_y = &EIGHTTAP_SHARP_KERNEL[11];

        let mut dst_2d = vec![0u8; w * h];
        convolve8(
            &src[convolve8_src_start(pad, pad, stride)..],
            stride,
            &mut dst_2d,
            w,
            filter_x,
            filter_y,
            w,
            h,
        );

        // Reference: horiz pass into intermediate (h + 7 rows), then
        // vert pass over the intermediate.
        let intermediate_h = h + FILTER_TAPS - 1;
        let mut inter = vec![0u8; w * intermediate_h];
        // The horiz pass starts three rows above the dest origin.
        // We use convolve8_horiz directly, so its `src` must be
        // pre-shifted left by 3 cols; the `y = 0` row of `inter`
        // corresponds to `pad - 3` in `src`.
        convolve8_horiz(
            &src[(pad - 3) * stride + pad - 3..],
            stride,
            &mut inter,
            w,
            filter_x,
            w,
            intermediate_h,
        );
        // Vert pass: convolve8_vert expects `src` pre-shifted up by
        // 3 rows. `inter[0]` is already 3 rows above the dest
        // origin, so we pass it as-is.
        let mut dst_ref = vec![0u8; w * h];
        convolve8_vert(&inter, w, &mut dst_ref, w, filter_y, w, h);
        assert_eq!(dst_2d, dst_ref);
    }

    #[test]
    fn convolve8_avg_horiz_averages_against_existing_dst() {
        let w = 4;
        let h = 4;
        let payload: Vec<u8> = (0..16u8).map(|v| 10 + v).collect();
        let (src, stride, xo, yo) = pad_block(&payload, w, h);
        // Integer-pel filter so the prediction equals payload.
        let mut dst_plain = vec![0u8; w * h];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst_plain,
            w,
            &EIGHTTAP_KERNEL[0],
            w,
            h,
        );
        // Pre-fill the avg destination so we can check the merge.
        let mut dst_avg = vec![200u8; w * h];
        convolve8_avg_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst_avg,
            w,
            &EIGHTTAP_KERNEL[0],
            w,
            h,
        );
        for i in 0..w * h {
            let expected = ((u32::from(dst_plain[i]) + 200 + 1) >> 1) as u8;
            assert_eq!(dst_avg[i], expected);
        }
    }

    #[test]
    fn convolve8_avg_2d_matches_convolve_plus_avg() {
        // Differential: convolve8_avg must equal convolve8 into a
        // scratch buffer followed by convolve_avg.
        let w = 8;
        let h = 8;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut rng_state = 0x1234_5678u32;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            (rng_state & 0xff) as u8
        };
        let mut src = vec![0u8; stride * padded_h];
        for cell in src.iter_mut() {
            *cell = next();
        }
        let mut dst_avg = vec![0u8; w * h];
        for cell in dst_avg.iter_mut() {
            *cell = next();
        }
        let mut dst_ref = dst_avg.clone();
        let filter_x = &EIGHTTAP_SMOOTH_KERNEL[3];
        let filter_y = &EIGHTTAP_KERNEL[12];

        convolve8_avg(
            &src[convolve8_src_start(pad, pad, stride)..],
            stride,
            &mut dst_avg,
            w,
            filter_x,
            filter_y,
            w,
            h,
        );

        let mut pred = vec![0u8; w * h];
        convolve8(
            &src[convolve8_src_start(pad, pad, stride)..],
            stride,
            &mut pred,
            w,
            filter_x,
            filter_y,
            w,
            h,
        );
        convolve_avg(&pred, w, &mut dst_ref, w, w, h);
        assert_eq!(dst_avg, dst_ref);
    }

    #[test]
    fn convolve_copy_passthrough() {
        let w = 4;
        let h = 4;
        let src: Vec<u8> = (0..16u8).collect();
        let mut dst = vec![0u8; w * h];
        convolve_copy(&src, w, &mut dst, w, w, h);
        assert_eq!(dst, src);
    }

    #[test]
    fn convolve_avg_rounds_up() {
        // (3 + 4 + 1) >> 1 = 4; (5 + 5 + 1) >> 1 = 5.
        let src: Vec<u8> = vec![4u8; 4];
        let mut dst = vec![3u8, 5u8, 0u8, 255u8];
        convolve_avg(&src, 4, &mut dst, 4, 4, 1);
        assert_eq!(dst, vec![4, 5, 2, 130]);
    }

    #[test]
    fn clipping_at_negative_intermediate() {
        // EIGHTTAP_SHARP at q5/16 = [-4, 10, -23, 108, 48, -16, 8,
        // -3]. Feed a pattern where every negative-tap column is 255
        // and every positive-tap column is 0, driving the sum
        // strongly negative.
        let w = 1;
        let h = 1;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let mut src = vec![0u8; stride * (h + 2 * pad)];
        let f = &EIGHTTAP_SHARP_KERNEL[5];
        let yo = pad;
        let xo = pad;
        // src for convolve8_horiz must be pre-shifted left by 3, so
        // tap k reads src[xo - 3 + k] inside the row at yo.
        let row_base = yo * stride + xo - 3;
        for k in 0..FILTER_TAPS {
            src[row_base + k] = if f[k] < 0 { 255 } else { 0 };
        }
        let mut dst = vec![123u8; 1];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            1,
            f,
            1,
            1,
        );
        // Sum = 255 * (-4 - 23 - 16 - 3) = -11730.
        // (-11730 + 64) >> 7 = -92 → clamps to 0.
        assert_eq!(dst[0], 0);
    }

    #[test]
    fn clipping_at_positive_intermediate() {
        // Mirror image — push sum past 255 and check the upper
        // clamp fires.
        let w = 1;
        let h = 1;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let mut src = vec![0u8; stride * (h + 2 * pad)];
        let f = &EIGHTTAP_SHARP_KERNEL[5];
        let yo = pad;
        let xo = pad;
        let row_base = yo * stride + xo - 3;
        for k in 0..FILTER_TAPS {
            src[row_base + k] = if f[k] > 0 { 255 } else { 0 };
        }
        let mut dst = vec![0u8; 1];
        convolve8_horiz(
            &src[horiz_src_start(xo, yo, stride)..],
            stride,
            &mut dst,
            1,
            f,
            1,
            1,
        );
        // Sum = 255 * (10 + 108 + 48 + 8) = 44370.
        // (44370 + 64) >> 7 = 347 → clamps to 255.
        assert_eq!(dst[0], 255);
    }

    // --- Differential vs the libvpx scalar reference, randomised ------

    /// Verbatim Rust port of libvpx `convolve_horiz` with steps fixed
    /// to `SUBPEL_SHIFTS` (no scaling). Uses the same pre-shifted
    /// `src` convention as the production kernel (caller has already
    /// applied `src -= SUBPEL_TAPS / 2 - 1`). Lives only in test
    /// scope so a regression can never leak into production.
    fn libvpx_convolve8_horiz_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter: &[i16; 8],
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            for x in 0..w {
                let mut sum: i32 = 0;
                let base = y * src_stride + x;
                for k in 0..FILTER_TAPS {
                    sum += i32::from(src[base + k]) * i32::from(filter[k]);
                }
                let rounded = (sum + (1 << 6)) >> 7;
                dst[y * dst_stride + x] = rounded.clamp(0, 255) as u8;
            }
        }
    }

    fn libvpx_convolve8_vert_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter: &[i16; 8],
        w: usize,
        h: usize,
    ) {
        for x in 0..w {
            for y in 0..h {
                let mut sum: i32 = 0;
                for k in 0..FILTER_TAPS {
                    sum += i32::from(src[(y + k) * src_stride + x]) * i32::from(filter[k]);
                }
                let rounded = (sum + (1 << 6)) >> 7;
                dst[y * dst_stride + x] = rounded.clamp(0, 255) as u8;
            }
        }
    }

    /// libvpx `vpx_convolve8_c` reference. `src` must be pre-shifted
    /// up by 3 rows AND left by 3 columns from the destination
    /// origin (matching our `convolve8` API).
    #[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
    fn libvpx_convolve8_ref(
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        filter_x: &[i16; 8],
        filter_y: &[i16; 8],
        w: usize,
        h: usize,
    ) {
        let intermediate_h = h + FILTER_TAPS - 1;
        let mut inter = vec![0u8; 64 * intermediate_h];
        // Horizontal pass over `intermediate_h` rows; row 0 of inter
        // is the same row as row 0 of `src` (which is already 3 rows
        // above the dest origin).
        libvpx_convolve8_horiz_ref(src, src_stride, &mut inter, 64, filter_x, w, intermediate_h);
        // Vertical pass: inter row 0 is 3 rows above dest origin, so
        // it acts as the "pre-shifted-up" src for the vert kernel.
        libvpx_convolve8_vert_ref(&inter, 64, dst, dst_stride, filter_y, w, h);
    }

    #[test]
    fn differential_random_horiz() {
        // 100 random inputs across all four kernels and 16 sub-pel
        // offsets — confirms convolve8_horiz matches the libvpx
        // reference byte-for-byte.
        let w = 8;
        let h = 8;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut state = 0xC0FF_EE42u32;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let kernels: [&[[i16; 8]; SUBPEL_SHIFTS]; 4] = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        for iter in 0..100 {
            let mut src = vec![0u8; stride * padded_h];
            for cell in src.iter_mut() {
                *cell = (rand() & 0xff) as u8;
            }
            let kernel = kernels[(iter as usize) % 4];
            let subpel = (rand() as usize) % SUBPEL_SHIFTS;
            let filter = &kernel[subpel];
            let mut dst_kernel = vec![0u8; w * h];
            let mut dst_ref = vec![0u8; w * h];
            let src_off = horiz_src_start(pad, pad, stride);
            convolve8_horiz(
                &src[src_off..],
                stride,
                &mut dst_kernel,
                w,
                filter,
                w,
                h,
            );
            libvpx_convolve8_horiz_ref(
                &src[src_off..],
                stride,
                &mut dst_ref,
                w,
                filter,
                w,
                h,
            );
            assert_eq!(
                dst_kernel, dst_ref,
                "horiz mismatch at iter={iter} subpel={subpel}"
            );
        }
    }

    #[test]
    fn differential_random_vert() {
        let w = 8;
        let h = 8;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut state = 0xFEED_FACEu32;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let kernels: [&[[i16; 8]; SUBPEL_SHIFTS]; 4] = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        for iter in 0..100 {
            let mut src = vec![0u8; stride * padded_h];
            for cell in src.iter_mut() {
                *cell = (rand() & 0xff) as u8;
            }
            let kernel = kernels[(iter as usize) % 4];
            let subpel = (rand() as usize) % SUBPEL_SHIFTS;
            let filter = &kernel[subpel];
            let mut dst_kernel = vec![0u8; w * h];
            let mut dst_ref = vec![0u8; w * h];
            let src_off = vert_src_start(pad, pad, stride);
            convolve8_vert(
                &src[src_off..],
                stride,
                &mut dst_kernel,
                w,
                filter,
                w,
                h,
            );
            libvpx_convolve8_vert_ref(
                &src[src_off..],
                stride,
                &mut dst_ref,
                w,
                filter,
                w,
                h,
            );
            assert_eq!(
                dst_kernel, dst_ref,
                "vert mismatch at iter={iter} subpel={subpel}"
            );
        }
    }

    #[test]
    fn differential_random_2d() {
        let w = 8;
        let h = 8;
        let pad = 8usize;
        let stride = w + 2 * pad;
        let padded_h = h + 2 * pad;
        let mut state = 0xBADD_CAFEu32;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let kernels: [&[[i16; 8]; SUBPEL_SHIFTS]; 4] = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        for iter in 0..100 {
            let mut src = vec![0u8; stride * padded_h];
            for cell in src.iter_mut() {
                *cell = (rand() & 0xff) as u8;
            }
            let kernel = kernels[(iter as usize) % 4];
            let subpel_x = (rand() as usize) % SUBPEL_SHIFTS;
            let subpel_y = (rand() as usize) % SUBPEL_SHIFTS;
            let filter_x = &kernel[subpel_x];
            let filter_y = &kernel[subpel_y];
            let mut dst_kernel = vec![0u8; w * h];
            let mut dst_ref = vec![0u8; w * h];
            let src_off = convolve8_src_start(pad, pad, stride);
            convolve8(
                &src[src_off..],
                stride,
                &mut dst_kernel,
                w,
                filter_x,
                filter_y,
                w,
                h,
            );
            libvpx_convolve8_ref(
                &src[src_off..],
                stride,
                &mut dst_ref,
                w,
                filter_x,
                filter_y,
                w,
                h,
            );
            assert_eq!(
                dst_kernel, dst_ref,
                "2d mismatch at iter={iter} sx={subpel_x} sy={subpel_y}"
            );
        }
    }
}
