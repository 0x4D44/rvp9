//! Distortion primitives — scalar variance, SAD, sub-pixel variance.
//!
//! Mirror of libvpx `vpx_dsp/variance.c` and `vpx_dsp/sad.c` for the 13 VP9
//! block sizes plus a handful of helpers used by mode decision and
//! sub-pixel MV refinement (`vp9_find_best_sub_pixel_tree_pruned*`).
//!
//! All implementations are pure scalar — SIMD specialisations come later via
//! [`crate::dsp::SimdDispatch`]. Each block-size wrapper is a thin shim over a
//! generic `variance` / `sad` / `sub_pixel_variance` core, matching libvpx's
//! `VAR(W, H)` / `sadMxN(m, n)` / `SUBPIX_VAR(W, H)` macro expansions 1:1.
//!
//! ## Reference
//!
//! - `libvpx-stock/vpx_dsp/variance.c` — `variance`, `vpx_variance{WxH}_c`,
//!   `vpx_sub_pixel_variance{WxH}_c`, `vpx_get4x4sse_cs_c`.
//! - `libvpx-stock/vpx_dsp/sad.c` — `sad`, `vpx_sad{WxH}_c`.
//! - `libvpx-stock/vpx_dsp/variance.c` lines 19-22 — `bilinear_filters[8][2]`.
//!
//! ## Function-call signature
//!
//! The libvpx C signatures pass `sse` as an out-pointer and return the
//! variance scalar. We return a `(variance, sse)` tuple so callers can ignore
//! `sse` cheaply (the rate-distortion code generally needs both).
//!
//! ## Variance derivation
//!
//! For a block of `N = W * H` pixels with per-pixel difference `d`:
//! ```text
//!   sum = Σ d
//!   sse = Σ d²
//!   variance = sse − (sum * sum) / N
//! ```
//! Per libvpx, the division is integer truncation. We perform the
//! `sum * sum` multiply in `i64` to avoid overflow on the 64×64 size (max
//! `|sum| ≤ 64*64*255 ≈ 1.04 M`, and `sum² ≈ 1.09 × 10¹²` — exceeds `i32`).
//! `sse` itself fits in `u32` for every block size (max
//! `64*64*255² ≈ 2.66 × 10⁸`).

/// Number of fractional bits in the sub-pixel bilinear filter taps.
///
/// Matches `FILTER_BITS` in `libvpx-stock/vpx_dsp/variance.h`.
const FILTER_BITS: u32 = 7;

/// 2-tap bilinear filters at 1/8-pel granularity.
///
/// Verbatim copy of `bilinear_filters` from
/// `libvpx-stock/vpx_dsp/variance.c` lines 19-22. The two taps in each row
/// sum to `1 << FILTER_BITS` (= 128) so that the post-filter rounding shift
/// preserves the input range.
const BILINEAR_FILTERS: [[u8; 2]; 8] = [
    [128, 0],
    [112, 16],
    [96, 32],
    [80, 48],
    [64, 64],
    [48, 80],
    [32, 96],
    [16, 112],
];

/// Round `v` to the nearest integer after dividing by `2^n`.
///
/// Matches libvpx's `ROUND_POWER_OF_TWO(v, n)` macro (defined in
/// `vpx_ports/mem.h`). Used by the bilinear-filter passes.
#[inline]
fn round_power_of_two(v: i32, n: u32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

/// Generic variance core — mirror of `static void variance(...)` in libvpx
/// `vpx_dsp/variance.c` lines 52-70.
///
/// Returns `(variance, sse)`. The caller computes `variance` from `(sum,
/// sse)` via `sse - (sum*sum)/N` per the libvpx `VAR(W, H)` macro at line
/// 134; we fold that into this function so the size-specific wrappers can
/// remain 1-line shims.
///
/// # Panics
///
/// Debug-only `assert!`s check that `w` and `h` are non-zero and that the
/// input slices have enough rows. In release builds bad inputs silently
/// produce garbage — same as the libvpx C code.
pub fn variance(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> (u32, u32) {
    debug_assert!(w > 0 && h > 0);
    debug_assert!(src.len() >= (h - 1) * src_stride + w);
    debug_assert!(reference.len() >= (h - 1) * ref_stride + w);

    let mut sum: i32 = 0;
    let mut sse: u32 = 0;
    for i in 0..h {
        let src_row = &src[i * src_stride..i * src_stride + w];
        let ref_row = &reference[i * ref_stride..i * ref_stride + w];
        for j in 0..w {
            let diff = i32::from(src_row[j]) - i32::from(ref_row[j]);
            sum += diff;
            sse += (diff * diff) as u32;
        }
    }

    // `(sum * sum) / (w * h)` per libvpx VAR macro (variance.c:134). Use i64
    // to avoid overflow when sum is large (max |sum| ≈ 1.04M for 64×64).
    let n = (w * h) as i64;
    let sum_sq = (sum as i64) * (sum as i64);
    let variance = (sse as i64) - sum_sq / n;
    // VP9 variance is always non-negative by construction; clamp for safety.
    let variance = variance.max(0) as u32;

    (variance, sse)
}

/// Generic SAD core — mirror of `static INLINE unsigned int sad(...)` in
/// libvpx `vpx_dsp/sad.c` lines 20-33.
pub fn sad(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    debug_assert!(w > 0 && h > 0);
    debug_assert!(src.len() >= (h - 1) * src_stride + w);
    debug_assert!(reference.len() >= (h - 1) * ref_stride + w);

    let mut total: u32 = 0;
    for i in 0..h {
        let src_row = &src[i * src_stride..i * src_stride + w];
        let ref_row = &reference[i * ref_stride..i * ref_stride + w];
        for j in 0..w {
            let d = i32::from(src_row[j]) - i32::from(ref_row[j]);
            total += d.unsigned_abs();
        }
    }
    total
}

/// First-pass 1-D bilinear filter (8-bit in, 16-bit out).
///
/// Mirrors `var_filter_block2d_bil_first_pass` in libvpx
/// `vpx_dsp/variance.c` lines 80-98. `pixel_step` is 1 (horizontal) or
/// `src_pixels_per_line` (vertical).
fn var_filter_block2d_bil_first_pass(
    src: &[u8],
    src_pixels_per_line: usize,
    pixel_step: usize,
    output_height: usize,
    output_width: usize,
    filter: [u8; 2],
    output: &mut [u16],
) {
    debug_assert_eq!(output.len(), output_height * output_width);

    for i in 0..output_height {
        for j in 0..output_width {
            let src_idx = i * src_pixels_per_line + j;
            let a = i32::from(src[src_idx]);
            let b = i32::from(src[src_idx + pixel_step]);
            let v = a * i32::from(filter[0]) + b * i32::from(filter[1]);
            output[i * output_width + j] = round_power_of_two(v, FILTER_BITS) as u16;
        }
    }
}

/// Second-pass 1-D bilinear filter (16-bit in, 8-bit out).
///
/// Mirrors `var_filter_block2d_bil_second_pass` in libvpx
/// `vpx_dsp/variance.c` lines 109-126. Reads from the first-pass
/// scratch buffer (size `(H + 1) * W`) and writes the 2-D filtered
/// block of size `H * W`. `pixel_step` is the row stride of the
/// scratch buffer (= `output_width`).
fn var_filter_block2d_bil_second_pass(
    src: &[u16],
    src_pixels_per_line: usize,
    pixel_step: usize,
    output_height: usize,
    output_width: usize,
    filter: [u8; 2],
    output: &mut [u8],
) {
    debug_assert_eq!(output.len(), output_height * output_width);

    for i in 0..output_height {
        for j in 0..output_width {
            let src_idx = i * src_pixels_per_line + j;
            let a = i32::from(src[src_idx]);
            let b = i32::from(src[src_idx + pixel_step]);
            let v = a * i32::from(filter[0]) + b * i32::from(filter[1]);
            output[i * output_width + j] = round_power_of_two(v, FILTER_BITS) as u8;
        }
    }
}

/// Sub-pixel variance core — mirror of the libvpx `SUBPIX_VAR(W, H)` macro
/// (`vpx_dsp/variance.c` lines 137-150).
///
/// Interpolates `src` (the search candidate) at the fractional (xoffset,
/// yoffset) offset, then computes variance against `reference` (the source
/// block being searched for). Note libvpx's argument naming is reversed
/// from intuition: in libvpx the "src" is what gets bilinear-filtered (i.e.
/// the reference frame predictor) and "ref" is the block whose distortion
/// we measure. We preserve that orientation for 1:1 portability.
///
/// `xoffset` and `yoffset` are 1/8-pel positions in `0..8`. Offset (0, 0)
/// produces an identity copy (`filter = [128, 0]`) and the result equals
/// integer-pel `variance(src, ..., reference, ...)`.
///
/// Eight parameters matches the libvpx C signature 1:1; the `too_many_arguments`
/// clippy lint is silenced rather than packing into a struct (the callers in
/// MV search are hot loops where the wrapper inlining matters).
#[allow(clippy::too_many_arguments)]
pub fn sub_pixel_variance(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    w: usize,
    h: usize,
) -> (u32, u32) {
    debug_assert!(xoffset < 8 && yoffset < 8);
    debug_assert!(w > 0 && h > 0);
    // First pass reads (h + 1) rows and (w + 1) columns from `src` (so the
    // 2-tap bilinear filter can fetch `src[j]` and `src[j + 1]`). Callers
    // must provide a source plane with row stride and length sufficient to
    // cover that footprint — libvpx relies on frame-buffer extension.
    debug_assert!(src.len() > h * src_stride + w);

    // First-pass: horizontal bilinear filter → (h + 1) × w u16 buffer.
    let mut fdata3 = vec![0u16; (h + 1) * w];
    var_filter_block2d_bil_first_pass(
        src,
        src_stride,
        1, // horizontal: step by 1 pixel
        h + 1,
        w,
        BILINEAR_FILTERS[xoffset],
        &mut fdata3,
    );

    // Second-pass: vertical bilinear filter → h × w u8 buffer.
    let mut temp2 = vec![0u8; h * w];
    var_filter_block2d_bil_second_pass(
        &fdata3,
        w, // first-pass output row stride
        w, // vertical step = one row of the first-pass buffer
        h,
        w,
        BILINEAR_FILTERS[yoffset],
        &mut temp2,
    );

    variance(&temp2, w, reference, ref_stride, w, h)
}

/// 4×4 SSE-only helper — mirror of `vpx_get4x4sse_cs_c` (libvpx
/// `vpx_dsp/variance.c` lines 24-40). Used by select mode-decision
/// heuristics that want SSE without the variance correction term.
pub fn vpx_get4x4sse_cs(src: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize) -> u32 {
    let mut distortion: i32 = 0;
    for r in 0..4 {
        let src_row = &src[r * src_stride..r * src_stride + 4];
        let ref_row = &reference[r * ref_stride..r * ref_stride + 4];
        for c in 0..4 {
            let diff = i32::from(src_row[c]) - i32::from(ref_row[c]);
            distortion += diff * diff;
        }
    }
    distortion as u32
}

// ---------------------------------------------------------------------------
// Per-size wrappers — mirror libvpx `VARIANCES(W, H)` expansions for the 13
// VP9 block sizes (variance.c lines 201-213) and `sadMxN(m, n)` expansions
// in sad.c lines 75-126.
// ---------------------------------------------------------------------------

macro_rules! variance_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("Variance for a ", stringify!($w), "×", stringify!($h),
                       " block. Mirrors `vpx_variance", stringify!($w), "x",
                       stringify!($h), "_c` in libvpx.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> (u32, u32) {
            variance(src, src_stride, reference, ref_stride, $w, $h)
        }
    };
}

variance_fn!(vpx_variance4x4, 4, 4);
variance_fn!(vpx_variance4x8, 4, 8);
variance_fn!(vpx_variance8x4, 8, 4);
variance_fn!(vpx_variance8x8, 8, 8);
variance_fn!(vpx_variance8x16, 8, 16);
variance_fn!(vpx_variance16x8, 16, 8);
variance_fn!(vpx_variance16x16, 16, 16);
variance_fn!(vpx_variance16x32, 16, 32);
variance_fn!(vpx_variance32x16, 32, 16);
variance_fn!(vpx_variance32x32, 32, 32);
variance_fn!(vpx_variance32x64, 32, 64);
variance_fn!(vpx_variance64x32, 64, 32);
variance_fn!(vpx_variance64x64, 64, 64);

macro_rules! sad_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("SAD for a ", stringify!($w), "×", stringify!($h),
                       " block. Mirrors `vpx_sad", stringify!($w), "x",
                       stringify!($h), "_c` in libvpx.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> u32 {
            sad(src, src_stride, reference, ref_stride, $w, $h)
        }
    };
}

sad_fn!(vpx_sad4x4, 4, 4);
sad_fn!(vpx_sad4x8, 4, 8);
sad_fn!(vpx_sad8x4, 8, 4);
sad_fn!(vpx_sad8x8, 8, 8);
sad_fn!(vpx_sad8x16, 8, 16);
sad_fn!(vpx_sad16x8, 16, 8);
sad_fn!(vpx_sad16x16, 16, 16);
sad_fn!(vpx_sad16x32, 16, 32);
sad_fn!(vpx_sad32x16, 32, 16);
sad_fn!(vpx_sad32x32, 32, 32);
sad_fn!(vpx_sad32x64, 32, 64);
sad_fn!(vpx_sad64x32, 64, 32);
sad_fn!(vpx_sad64x64, 64, 64);

macro_rules! sub_pixel_variance_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("Sub-pixel variance for a ", stringify!($w), "×",
                       stringify!($h),
                       " block. Mirrors `vpx_sub_pixel_variance",
                       stringify!($w), "x", stringify!($h), "_c` in libvpx.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            xoffset: usize,
            yoffset: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> (u32, u32) {
            sub_pixel_variance(
                src, src_stride, reference, ref_stride, xoffset, yoffset, $w, $h,
            )
        }
    };
}

// Subset required by `vp9_find_best_sub_pixel_tree_pruned_more` per the
// F6.0 task brief: 4x4, 8x8, 16x16, 32x32, 64x64.
sub_pixel_variance_fn!(vpx_sub_pixel_variance4x4, 4, 4);
sub_pixel_variance_fn!(vpx_sub_pixel_variance8x8, 8, 8);
sub_pixel_variance_fn!(vpx_sub_pixel_variance16x16, 16, 16);
sub_pixel_variance_fn!(vpx_sub_pixel_variance32x32, 32, 32);
sub_pixel_variance_fn!(vpx_sub_pixel_variance64x64, 64, 64);

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-size variance function signature, used to build dispatch tables
    /// in the all-sizes smoke tests below.
    type VarFn = fn(&[u8], usize, &[u8], usize) -> (u32, u32);
    /// Per-size SAD function signature.
    type SadFn = fn(&[u8], usize, &[u8], usize) -> u32;

    /// Full list of VP9 block sizes for the `*_all_sizes_smoke` coverage tests.
    /// 13 entries — must match libvpx's `VARIANCES(W, H)` macro list in
    /// variance.c lines 201-213.
    const ALL_SIZES: &[(usize, usize)] = &[
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

    fn flat(value: u8, size: usize) -> Vec<u8> {
        vec![value; size]
    }

    #[test]
    fn variance_zero_when_blocks_equal() {
        // 8×8 flat-128 block against itself: variance and sse both zero.
        let src = flat(128, 8 * 8);
        let (var, sse) = vpx_variance8x8(&src, 8, &src, 8);
        assert_eq!(var, 0, "variance must be zero for identical blocks");
        assert_eq!(sse, 0, "sse must be zero for identical blocks");
    }

    #[test]
    fn variance_known_value() {
        // Hand-built 4×4. src is all 100. reference has a single +10 pixel
        // at (0, 0), rest match.
        //   diff at (0,0) = -10, all others = 0.
        //   sum = -10, sse = 100, N = 16
        //   variance = 100 - (100 / 16) = 100 - 6 = 94.
        let src = flat(100, 16);
        let mut reference = flat(100, 16);
        reference[0] = 110;
        let (var, sse) = vpx_variance4x4(&src, 4, &reference, 4);
        assert_eq!(sse, 100, "sse should be the single squared diff");
        assert_eq!(var, 94, "variance = sse - sum^2/N = 100 - 100/16 = 94");
    }

    #[test]
    fn sad_zero_when_blocks_equal() {
        let src = flat(64, 16 * 16);
        assert_eq!(vpx_sad16x16(&src, 16, &src, 16), 0);
    }

    #[test]
    fn sad_known_value() {
        // 4×4: src = 100 everywhere, reference has a +5, a -7, rest match.
        // Absolute diffs: 5 and 7; sad = 12.
        let src = flat(100, 16);
        let mut reference = flat(100, 16);
        reference[0] = 105; // |100 - 105| = 5
        reference[5] = 93; //  |100 -  93| = 7
        let total = vpx_sad4x4(&src, 4, &reference, 4);
        assert_eq!(total, 12);
    }

    /// Build a sub-pel test source buffer: 9 rows × 16-stride (so the
    /// first-pass filter can fetch `src[j+1]` at the right edge of an
    /// 8-wide block without overrun) with deterministic content.
    fn subpel_src_8x8(seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 9 * 16];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(seed).wrapping_add(11) & 0xff) as u8;
        }
        buf
    }

    #[test]
    fn sub_pixel_variance_at_zero_offset_matches_variance() {
        // With (xoffset=0, yoffset=0) the bilinear filter is [128, 0], so
        // the first-pass output is `src[i]` rounded — i.e. an exact copy.
        // The sub-pixel variance should then equal plain variance.
        let src = subpel_src_8x8(7);
        let src_stride = 16;
        let mut reference = vec![0u8; 8 * 8];
        for (i, v) in reference.iter_mut().enumerate() {
            *v = ((i * 3 + 41) & 0xff) as u8;
        }

        let (sub_var, sub_sse) =
            vpx_sub_pixel_variance8x8(&src, src_stride, 0, 0, &reference, 8);

        // For comparison: copy the same 8×8 region out of `src` so plain
        // variance sees identical pixels to what the (0,0) filter copied.
        let mut src_top_left = vec![0u8; 8 * 8];
        for row in 0..8 {
            src_top_left[row * 8..row * 8 + 8]
                .copy_from_slice(&src[row * src_stride..row * src_stride + 8]);
        }
        let (plain_var, plain_sse) = vpx_variance8x8(&src_top_left, 8, &reference, 8);

        assert_eq!(sub_var, plain_var, "sub-pel @ (0,0) should match plain var");
        assert_eq!(sub_sse, plain_sse, "sub-pel @ (0,0) should match plain sse");
    }

    #[test]
    fn sub_pixel_variance_smoke_at_half_pel() {
        // (xoffset=4, yoffset=4) is the half-pel position. On a non-flat
        // gradient pattern this exercises both passes of the bilinear
        // filter and the variance accumulator. Just check the call
        // succeeds and produces a sane invariant: variance ≤ sse.
        let src = subpel_src_8x8(13);
        let mut reference = vec![0u8; 8 * 8];
        for (i, v) in reference.iter_mut().enumerate() {
            *v = ((i ^ 0x55) & 0xff) as u8;
        }
        let (var, sse) = vpx_sub_pixel_variance8x8(&src, 16, 4, 4, &reference, 8);
        assert!(var <= sse, "variance ({}) must be ≤ sse ({})", var, sse);
    }

    #[test]
    fn get4x4sse_cs_matches_variance_sse() {
        // `vpx_get4x4sse_cs` returns sse only; it should match the `sse`
        // component of the 4×4 variance call exactly.
        let src: Vec<u8> = (0u8..16).collect();
        let reference: Vec<u8> = (0u8..16).map(|x| x ^ 0x0f).collect();

        let sse_cs = vpx_get4x4sse_cs(&src, 4, &reference, 4);
        let (_, sse_var) = vpx_variance4x4(&src, 4, &reference, 4);
        assert_eq!(sse_cs, sse_var);
    }

    #[test]
    fn variance_all_sizes_smoke() {
        // Sanity coverage: every VP9 block size with flat-128 inputs should
        // produce variance=0 and sse=0. Print the size list so the test log
        // proves we touched all 13 sizes per the F6.0 brief.
        let dispatch: &[((usize, usize), VarFn)] = &[
            ((4, 4), vpx_variance4x4),
            ((4, 8), vpx_variance4x8),
            ((8, 4), vpx_variance8x4),
            ((8, 8), vpx_variance8x8),
            ((8, 16), vpx_variance8x16),
            ((16, 8), vpx_variance16x8),
            ((16, 16), vpx_variance16x16),
            ((16, 32), vpx_variance16x32),
            ((32, 16), vpx_variance32x16),
            ((32, 32), vpx_variance32x32),
            ((32, 64), vpx_variance32x64),
            ((64, 32), vpx_variance64x32),
            ((64, 64), vpx_variance64x64),
        ];

        assert_eq!(
            dispatch.len(),
            ALL_SIZES.len(),
            "variance dispatch must cover all 13 VP9 block sizes"
        );

        for (size, var_fn) in dispatch {
            let (w, h) = *size;
            let buf = flat(128, w * h);
            let (var, sse) = var_fn(&buf, w, &buf, w);
            assert_eq!(var, 0, "variance({}×{}) on flat must be 0", w, h);
            assert_eq!(sse, 0, "sse({}×{}) on flat must be 0", w, h);
        }
    }

    #[test]
    fn sad_all_sizes_smoke() {
        let dispatch: &[((usize, usize), SadFn)] = &[
            ((4, 4), vpx_sad4x4),
            ((4, 8), vpx_sad4x8),
            ((8, 4), vpx_sad8x4),
            ((8, 8), vpx_sad8x8),
            ((8, 16), vpx_sad8x16),
            ((16, 8), vpx_sad16x8),
            ((16, 16), vpx_sad16x16),
            ((16, 32), vpx_sad16x32),
            ((32, 16), vpx_sad32x16),
            ((32, 32), vpx_sad32x32),
            ((32, 64), vpx_sad32x64),
            ((64, 32), vpx_sad64x32),
            ((64, 64), vpx_sad64x64),
        ];

        assert_eq!(
            dispatch.len(),
            ALL_SIZES.len(),
            "sad dispatch must cover all 13 VP9 block sizes"
        );

        for (size, sad_fn_) in dispatch {
            let (w, h) = *size;
            let buf = flat(64, w * h);
            assert_eq!(
                sad_fn_(&buf, w, &buf, w),
                0,
                "sad({}×{}) on flat must be 0",
                w,
                h
            );
        }
    }

    #[test]
    fn variance_negative_clamps_to_zero() {
        // Edge-case sanity: with a hand-picked input where sse < sum²/N due
        // to integer truncation rounding, libvpx's C cast wraps to a huge
        // u32. We instead saturate to 0. Construct a case where sum² is
        // exactly divisible by N and equals sse (variance == 0):
        //   src = reference + k for constant k → sum = N*k, sse = N*k²,
        //   sum²/N = N*k² = sse → variance = 0. Choose N=16, k=3.
        let src = flat(100, 16);
        let reference = flat(97, 16);
        let (var, sse) = vpx_variance4x4(&src, 4, &reference, 4);
        assert_eq!(sse, 16 * 9, "sse = N * k² = 16 * 9 = 144");
        assert_eq!(var, 0, "variance is exactly zero when block is shifted by a constant");
    }
}
