//! Scaled 8-tap 2-D sub-pel convolution — x86_64 AVX2 (Wave L L1).
//!
//! Scalar references: [`crate::inter::filter::convolve8_scaled`] and
//! [`crate::inter::filter::convolve8_scaled_avg`]. Bit-exact against
//! them (differential test is the oracle). Used when the reference
//! frame is a different size to the current frame (resize / spatial
//! SVC), so each output pixel samples a different sub-pel offset:
//! `x_q4 += x_step_q4` per column, `y_q4 += y_step_q4` per row.
//!
//! ## Why `madd_epi16` (i16), not `maddubs_epi16` (i8)
//!
//! The unscaled MC kernels pack taps into i8 for `_mm_maddubs_epi16`,
//! relying on the full-pel row (centre tap 128, out of i8 range) being
//! routed to `convolve_copy` upstream. The scaled path picks the filter
//! row **per sample** from `x_q4 & SUBPEL_MASK`, and sub-pel index 0
//! (the identity row `[0,0,0,128,0,0,0,0]`) is reachable whenever a
//! column lands on an integer position. So taps must be treated as i16.
//! `_mm_madd_epi16` over zero-extended-to-i16 source also reproduces the
//! scalar's exact non-saturating i32 accumulate — bit-exact by
//! construction, no saturation reasoning required.
//!
//! ## Structure
//!
//! - **Horizontal pass** (variable filter per column → a gather): one
//!   8-tap dot per output sample via `madd_epi16`, scalar round/clip
//!   into a `64 × 135` intermediate. Mirrors the scalar horizontal pass
//!   1:1.
//! - **Vertical pass**: loop-reordered to **row-outer** so the filter
//!   row + integer base are fixed across each output row (they depend
//!   only on `y`). That makes it a fixed-filter vertical FIR vectorised
//!   across columns (4 i32 lanes per group), `+64`, `>>7`, double-
//!   `packus` clamp. Reordering is bit-exact: every output sample is an
//!   independent read of the fully-built intermediate.
//!
//! ## Aligned-load discipline (HLD V6 §G4)
//!
//! All loads from `src` / intermediate / `dst` use unaligned intrinsics
//! (`_mm_loadl_epi64` / `_mm_loadu_si128` / `read_unaligned`).

#![cfg(target_arch = "x86_64")]

use crate::inter::filter::SUBPEL_SHIFTS;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_avg_epu8, _mm_cvtepu8_epi16, _mm_cvtepu8_epi32, _mm_cvtsi32_si128,
    _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16, _mm_mullo_epi32,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32, _mm_setzero_si128, _mm_shuffle_epi32,
    _mm_srai_epi32, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi64,
};

const SUBPEL_BITS: i32 = 4;
const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1;
const FILTER_TAPS: usize = 8;
const FILTER_BITS: i32 = 7;
const FILTER_ROUNDING: i32 = 1 << (FILTER_BITS - 1);
const INTER_STRIDE: usize = 64;
const INTER_CAP: usize = 64 * 135;

// Validate coordinates before the raw-pointer passes and fixed-size scratch.
fn validate_scaled_layout(
    src: (usize, usize),
    dst: (usize, usize),
    coordinates: [i32; 4],
    dimensions: (usize, usize),
) {
    let (src_len, src_stride) = src;
    let (dst_len, dst_stride) = dst;
    let [x0, x_step, y0, y_step] = coordinates;
    let (w, h) = dimensions;
    assert!(
        (1..=64).contains(&w) && (1..=64).contains(&h),
        "unsupported scaled dimensions"
    );
    fn last_coordinate(start: i32, step: i32, count: usize) -> i32 {
        assert!(start >= 0 && step >= 0, "negative scaled coordinate");
        // The loops also increment once after their final sample.
        assert!(
            step.checked_mul(count as i32)
                .and_then(|delta| start.checked_add(delta))
                .is_some(),
            "scaled coordinate overflow"
        );
        start
            .checked_add(
                step.checked_mul(count as i32 - 1)
                    .expect("scaled coordinate overflow"),
            )
            .expect("scaled coordinate overflow")
    }
    let last_x = last_coordinate(x0, x_step, w);
    let last_y = last_coordinate(y0, y_step, h);
    let ih = (last_y >> SUBPEL_BITS) as usize + FILTER_TAPS;
    assert!(ih <= 135, "scaled intermediate too tall");
    let source_width = (last_x >> SUBPEL_BITS) as usize + FILTER_TAPS;
    super::convolve_copy_avx2::validate_block(src_len, src_stride, source_width, ih);
    super::convolve_copy_avx2::validate_block(dst_len, dst_stride, w, h);
}

/// Scalar round + clip, identical to `crate::inter::filter::round_clip`
/// (which is private): `clip255((sum + 64) >> 7)`.
#[inline]
fn round_clip(sum: i32) -> u8 {
    ((sum + FILTER_ROUNDING) >> FILTER_BITS).clamp(0, 255) as u8
}

/// One 8-tap dot product: `Σ_{k} src[k] * filter[k]` as exact i32.
///
/// # Safety
/// AVX2 required. Reads 8 bytes at `src` and 8 i16 at `filter`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot8(src: *const u8, filter: *const i16) -> i32 {
    // 8 source bytes → zero-extended i16×8 (values 0..255, positive).
    let s8 = _mm_loadl_epi64(src as *const __m128i);
    let s16 = _mm_cvtepu8_epi16(s8);
    // 8 signed i16 taps (contiguous [i16; 8]).
    let f16 = _mm_loadu_si128(filter as *const __m128i);
    // 4 i32 pairwise products: [s0f0+s1f1, s2f2+s3f3, s4f4+s5f5, s6f6+s7f7].
    let prod = _mm_madd_epi16(s16, f16);
    // Horizontal-sum the 4 i32 lanes (exact i32 adds, no saturation).
    let hi = _mm_unpackhi_epi64(prod, prod);
    let s2 = _mm_add_epi32(prod, hi);
    let sh = _mm_shuffle_epi32(s2, 0x01);
    _mm_cvtsi128_si32(_mm_add_epi32(s2, sh))
}

/// Horizontal pass: fill `inter[0..ih][0..w]` (stride 64) from `src`.
#[inline]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn horiz_pass(
    src: &[u8],
    src_stride: usize,
    inter: &mut [u8],
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    w: usize,
    ih: usize,
) {
    let src_ptr = src.as_ptr();
    for y in 0..ih {
        let row = src_ptr.add(y * src_stride);
        let mut x_q4 = x0_q4;
        for x in 0..w {
            let int_off = (x_q4 >> SUBPEL_BITS) as usize;
            let filter = filters.get_unchecked((x_q4 & SUBPEL_MASK) as usize);
            let sum = dot8(row.add(int_off), filter.as_ptr());
            *inter.get_unchecked_mut(y * INTER_STRIDE + x) = round_clip(sum);
            x_q4 += x_step_q4;
        }
    }
}

/// Vertical pass over the intermediate into `dst`. Row-outer so the
/// filter row is fixed across each output row.
#[inline]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn vert_pass(
    inter: &[u8],
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    let inter_ptr = inter.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let k_round = _mm_set1_epi32(FILTER_ROUNDING);
    let mut y_q4 = y0_q4;
    for y in 0..h {
        let int_row = (y_q4 >> SUBPEL_BITS) as usize;
        let filter = filters.get_unchecked((y_q4 & SUBPEL_MASK) as usize);
        // Pre-broadcast the eight taps to i32 lanes.
        let mut taps = [_mm_setzero_si128(); FILTER_TAPS];
        for (t, &f) in taps.iter_mut().zip(filter.iter()) {
            *t = _mm_set1_epi32(i32::from(f));
        }
        let dst_row = dst_ptr.add(y * dst_stride);

        let mut x = 0usize;
        while x + 4 <= w {
            let mut acc = _mm_setzero_si128();
            for (k, &tap) in taps.iter().enumerate() {
                let p = inter_ptr.add((int_row + k) * INTER_STRIDE + x);
                let four = core::ptr::read_unaligned(p as *const i32);
                let v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(four));
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, tap));
            }
            acc = _mm_add_epi32(acc, k_round);
            acc = _mm_srai_epi32(acc, FILTER_BITS);
            // clamp 0..255 via i32→i16 then i16→u8 saturating packs.
            let packed = _mm_packus_epi16(_mm_packs_epi32(acc, acc), _mm_setzero_si128());
            let out = _mm_cvtsi128_si32(packed) as u32;
            core::ptr::write_unaligned(dst_row.add(x) as *mut u32, out);
            x += 4;
        }
        // Tail (w not a multiple of 4 — not reached for VP9 block sizes,
        // kept for generality).
        while x < w {
            let mut sum = 0i32;
            for (k, &fk) in filter.iter().enumerate() {
                let s = *inter.get_unchecked((int_row + k) * INTER_STRIDE + x);
                sum += i32::from(s) * i32::from(fk);
            }
            *dst.get_unchecked_mut(y * dst_stride + x) = round_clip(sum);
            x += 1;
        }
        y_q4 += y_step_q4;
    }
}

/// AVX2 scaled 2-D convolution inner kernel.
///
/// # Safety
/// AVX2 required. `w, h ∈ 1..=64`; `src` holds the scalar
/// `convolve8_scaled` footprint; intermediate height ≤ 135.
// G4: loads from `src` / intermediate / `dst` use _mm_loadl_epi64 /
// _mm_loadu_si128 / read_unaligned (unaligned)
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_scaled_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    debug_assert!(w <= 64 && h <= 64);
    let ih = (((h as i32 - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) as usize + FILTER_TAPS;
    debug_assert!(ih <= 135);
    let mut inter = [0u8; INTER_CAP];
    horiz_pass(
        src, src_stride, &mut inter, filters, x0_q4, x_step_q4, w, ih,
    );
    vert_pass(&inter, dst, dst_stride, filters, y0_q4, y_step_q4, w, h);
}

/// Safe wrapper for the AVX2 scaled 2-D convolution. Signature mirrors
/// the scalar [`crate::inter::filter::convolve8_scaled`].
#[allow(clippy::too_many_arguments)]
pub fn convolve8_scaled_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    validate_scaled_layout(
        (src.len(), src_stride),
        (dst.len(), dst_stride),
        [x0_q4, x_step_q4, y0_q4, y_step_q4],
        (w, h),
    );

    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper when
    // `caps.avx2 == true`. Caps are detected once at OnceLock init and
    // monotonic for the process lifetime. Tests gate every call on a
    // runtime `is_x86_feature_detected!("avx2")` check.
    unsafe {
        convolve8_scaled_avx2_inner(
            src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
        );
    }
}

/// AVX2 scaled 2-D `_avg`: convolve into scratch, then round-average
/// into `dst` (`_mm_avg_epu8` == `round_avg`).
///
/// # Safety
/// AVX2 required; same contract as [`convolve8_scaled_avx2_inner`].
// G4: loads from `dst` / scratch use _mm_loadu_si128 / _mm_loadl_epi64 /
// read_unaligned (unaligned)
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_scaled_avg_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    let mut pred = [0u8; 64 * 64];
    let pred_stride = 64;
    convolve8_scaled_avx2_inner(
        src,
        src_stride,
        &mut pred,
        pred_stride,
        filters,
        x0_q4,
        x_step_q4,
        y0_q4,
        y_step_q4,
        w,
        h,
    );
    let dst_ptr = dst.as_mut_ptr();
    let pred_ptr = pred.as_ptr();
    for y in 0..h {
        let drow = dst_ptr.add(y * dst_stride);
        let prow = pred_ptr.add(y * pred_stride);
        let mut x = 0usize;
        while x + 16 <= w {
            let d = _mm_loadu_si128(drow.add(x) as *const __m128i);
            let p = _mm_loadu_si128(prow.add(x) as *const __m128i);
            _mm_storeu_si128(drow.add(x) as *mut __m128i, _mm_avg_epu8(d, p));
            x += 16;
        }
        while x + 8 <= w {
            let d = _mm_loadl_epi64(drow.add(x) as *const __m128i);
            let p = _mm_loadl_epi64(prow.add(x) as *const __m128i);
            _mm_storel_epi64(drow.add(x) as *mut __m128i, _mm_avg_epu8(d, p));
            x += 8;
        }
        while x < w {
            let d = u32::from(*dst.get_unchecked(y * dst_stride + x));
            let p = u32::from(*pred.get_unchecked(y * pred_stride + x));
            *dst.get_unchecked_mut(y * dst_stride + x) = ((d + p + 1) >> 1) as u8;
            x += 1;
        }
    }
}

/// Safe wrapper for the AVX2 scaled 2-D `_avg` convolution. Signature
/// mirrors the scalar [`crate::inter::filter::convolve8_scaled_avg`].
#[allow(clippy::too_many_arguments)]
pub fn convolve8_scaled_avg_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filters: &[[i16; 8]; SUBPEL_SHIFTS],
    x0_q4: i32,
    x_step_q4: i32,
    y0_q4: i32,
    y_step_q4: i32,
    w: usize,
    h: usize,
) {
    validate_scaled_layout(
        (src.len(), src_stride),
        (dst.len(), dst_stride),
        [x0_q4, x_step_q4, y0_q4, y_step_q4],
        (w, h),
    );

    // SAFETY: see `convolve8_scaled_avx2` — installed only under
    // `caps.avx2`; tests gate on runtime AVX2 detection.
    unsafe {
        convolve8_scaled_avg_avx2_inner(
            src, src_stride, dst, dst_stride, filters, x0_q4, x_step_q4, y0_q4, y_step_q4, w, h,
        );
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for the scaled convolves.
    use super::{convolve8_scaled_avg_avx2, convolve8_scaled_avx2};
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
        SUBPEL_SHIFTS, convolve8_scaled, convolve8_scaled_avg,
    };

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    /// Intermediate height per the scalar contract.
    fn inter_h(h: usize, y0_q4: i32, y_step_q4: i32) -> usize {
        (((h as i32 - 1) * y_step_q4 + y0_q4) >> 4) as usize + 8
    }

    /// Differential vs scalar across the VP9 (w, h) matrix, several
    /// scale steps and sub-pel phases, and all four kernel tables.
    #[test]
    fn differential_all_filters() {
        require_avx2!();
        let shapes: &[(usize, usize)] = &[
            (4, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (32, 32),
            (64, 64),
        ];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        // Steps spanning down-scale (>16) and up-scale (<16) plus unit.
        let steps: &[i32] = &[16, 20, 24, 32, 12, 10];
        let mut state: u32 = 0x5CA1_ED00;
        for &(w, h) in shapes {
            for kernel in kernels {
                for &x_step in steps {
                    for &y_step in steps {
                        let x0 = (lcg_next(&mut state) & 0xF) as i32;
                        let y0 = (lcg_next(&mut state) & 0xF) as i32;
                        let ih = inter_h(h, y0, y_step);
                        if ih > 135 {
                            continue;
                        }
                        // Horizontal source span: max int_off + 8.
                        let max_xq4 = x0 + (w as i32 - 1) * x_step;
                        let src_w = (max_xq4 >> 4) as usize + 8;
                        let src_stride = src_w + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (ih - 1) * src_stride + src_w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut d_scalar = vec![0u8; dst_len];
                        let mut d_avx2 = vec![0u8; dst_len];
                        convolve8_scaled(
                            &src,
                            src_stride,
                            &mut d_scalar,
                            dst_stride,
                            kernel,
                            x0,
                            x_step,
                            y0,
                            y_step,
                            w,
                            h,
                        );
                        convolve8_scaled_avx2(
                            &src,
                            src_stride,
                            &mut d_avx2,
                            dst_stride,
                            kernel,
                            x0,
                            x_step,
                            y0,
                            y_step,
                            w,
                            h,
                        );
                        assert_eq!(
                            d_scalar, d_avx2,
                            "scaled mismatch w={w} h={h} xs={x_step} ys={y_step} x0={x0} y0={y0}"
                        );
                    }
                }
            }
        }
    }

    /// Differential for the `_avg` variant — seed `dst` with content so
    /// the round-average is exercised.
    #[test]
    fn differential_avg() {
        require_avx2!();
        let kernel = &EIGHTTAP_KERNEL;
        let mut state: u32 = 0xA17E_0001;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            for &step in &[16i32, 20, 24, 12] {
                let x0 = (lcg_next(&mut state) & 0xF) as i32;
                let y0 = (lcg_next(&mut state) & 0xF) as i32;
                let ih = inter_h(h, y0, step);
                if ih > 135 {
                    continue;
                }
                let max_xq4 = x0 + (w as i32 - 1) * step;
                let src_w = (max_xq4 >> 4) as usize + 8;
                let src_stride = src_w + 3;
                let dst_stride = w + 5;
                let src_len = (ih - 1) * src_stride + src_w;
                let mut src = vec![0u8; src_len];
                for s in src.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let dst_len = (h - 1) * dst_stride + w;
                let mut d_scalar = vec![0u8; dst_len];
                for v in d_scalar.iter_mut() {
                    *v = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut d_avx2 = d_scalar.clone();
                convolve8_scaled_avg(
                    &src,
                    src_stride,
                    &mut d_scalar,
                    dst_stride,
                    kernel,
                    x0,
                    step,
                    y0,
                    step,
                    w,
                    h,
                );
                convolve8_scaled_avg_avx2(
                    &src,
                    src_stride,
                    &mut d_avx2,
                    dst_stride,
                    kernel,
                    x0,
                    step,
                    y0,
                    step,
                    w,
                    h,
                );
                assert_eq!(
                    d_scalar, d_avx2,
                    "scaled_avg mismatch w={w} h={h} step={step}"
                );
            }
        }
    }

    /// Sanity: a constant source convolves to that constant (filters
    /// sum to 128, identity under round).
    #[test]
    fn constant_input() {
        require_avx2!();
        let _ = SUBPEL_SHIFTS;
        let kernel = &EIGHTTAP_KERNEL;
        for &(w, h) in &[(8usize, 8usize), (16, 16), (32, 32)] {
            let step = 20i32;
            let ih = inter_h(h, 0, step);
            let src_w = (((w as i32 - 1) * step) >> 4) as usize + 8;
            let src_stride = src_w;
            let src = vec![137u8; (ih - 1) * src_stride + src_w];
            let mut d = vec![0u8; w * h];
            convolve8_scaled_avx2(&src, src_stride, &mut d, w, kernel, 0, step, 0, step, w, h);
            assert!(d.iter().all(|&v| v == 137), "w={w} h={h} not constant");
        }
    }
}

#[cfg(test)]
mod layout_guard_tests {
    use super::validate_scaled_layout;

    #[test]
    fn scaled_layout_guard_accepts_exact_tail() {
        // Four outputs at phase zero, unit step: eleven input taps per axis.
        validate_scaled_layout((121, 11), (16, 4), [0, 16, 0, 16], (4, 4));
    }

    #[test]
    fn scaled_layout_guard_rejects_malformed_geometry() {
        let bad = [
            (0, 16, 0, 16, 0, 4),
            (-1, 16, 0, 16, 4, 4),
            (0, -1, 0, 16, 4, 4),
            (i32::MAX, 16, 0, 16, 4, 4),
            (0, 16, 0, i32::MAX, 4, 4),
            (0, 16, 0, 64, 4, 64),
        ];
        for (x0, xs, y0, ys, w, h) in bad {
            assert!(
                std::panic::catch_unwind(|| {
                    validate_scaled_layout((4096, 64), (4096, 64), [x0, xs, y0, ys], (w, h));
                })
                .is_err()
            );
        }
        assert!(
            std::panic::catch_unwind(|| {
                validate_scaled_layout((120, 11), (16, 4), [0, 16, 0, 16], (4, 4));
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                validate_scaled_layout((121, 11), (15, 4), [0, 16, 0, 16], (4, 4));
            })
            .is_err()
        );
    }
}
