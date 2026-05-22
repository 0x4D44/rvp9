//! 8-tap horizontal sub-pel convolution — x86_64 AVX2 implementation
//! (Wave J J3).
//!
//! The hot inner kernel of motion compensation: for every output pixel
//! `dst[y][x]`, convolve eight horizontally-adjacent source samples
//! with the resolved sub-pel filter row:
//!
//! ```text
//! dst[y][x] = clip255( ( Σ_{k=0..8} src[y][x+k] * filter_x[k] + 64 ) >> 7 )
//! ```
//!
//! Mirrors libvpx `convolve_horiz` (`vpx_dsp/vpx_convolve.c:22-42`) with
//! `x_step_q4 == SUBPEL_SHIFTS`. The scalar reference is
//! [`crate::inter::filter::convolve8_horiz`]; callers pre-shift `src`
//! left by three columns so `src[y*src_stride + x]` is the first of the
//! eight taps for output `(y, x)`.
//!
//! ## Algorithm (libvpx `convolve8_16` math, ported)
//!
//! Eight output columns are produced per 128-bit group. A 16-byte load
//! of the source row is shuffled four ways to gather the tap-pair
//! operands for each output lane:
//!
//! | shuffle mask          | lane `i` holds              | tap pair      |
//! |-----------------------|-----------------------------|---------------|
//! | `{0,1,1,2,…,7,8}`     | `(src[i],   src[i+1])`      | `(f0, f1)`    |
//! | `{2,3,3,4,…,9,10}`    | `(src[i+2], src[i+3])`      | `(f2, f3)`    |
//! | `{4,5,5,6,…,11,12}`   | `(src[i+4], src[i+5])`      | `(f4, f5)`    |
//! | `{6,7,7,8,…,13,14}`   | `(src[i+6], src[i+7])`      | `(f6, f7)`    |
//!
//! Each shuffled register feeds `_mm_maddubs_epi16(pixels_u8, taps_i8)`
//! (pixels unsigned, taps signed — the VP9 filter taps all fit in i8).
//! The four i16 partial sums are regrouped `(x0+x2)+(x1+x3)` with the
//! `+64` rounding bias folded in early and a saturating final add — the
//! exact order libvpx uses to keep every intermediate inside i16 — then
//! `srai 7` and `packus_epi16` (whose unsigned saturation performs the
//! `[0,255]` clip).
//!
//! ## Source over-read avoidance
//!
//! A 16-byte load at the right edge of the last row would read ~1 byte
//! past the tight scalar footprint (`(h-1)*src_stride + w + 7`). Those
//! bytes only ever feed output lanes `≥ w` (discarded), so they cannot
//! corrupt valid output — but the OOB access is still UB. To honor the
//! *exact* scalar contract (safe drop-in for any caller), each row's
//! `w + 7` source span is copied into a zeroed `[u8; 80]` stack scratch
//! and the SIMD loads read from the scratch. The copy is one ≤71-byte
//! `copy_from_slice` per row.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads (from the stack scratch) and stores (to `dst`) use
//! unaligned intrinsics. See `crates/rvp9-core/src/dsp/x86_64/mod.rs`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi16, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_maddubs_epi16,
    _mm_packus_epi16, _mm_set1_epi16, _mm_setr_epi8, _mm_shuffle_epi8, _mm_srai_epi16,
    _mm_storel_epi64,
};

/// Pack two adjacent filter taps into the replicated i8-pair form
/// consumed by `_mm_maddubs_epi16`'s signed operand. `lo` lands in the
/// low byte of every 16-bit lane, `hi` in the high byte.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair(lo: i16, hi: i16) -> __m128i {
    let packed = i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]);
    _mm_set1_epi16(packed)
}

/// Convolve eight output columns from a 16-byte source register.
///
/// `s` holds `src[x0 .. x0+16)`; returns an `__m128i` whose low eight
/// 16-bit lanes are the rounded/shifted (pre-clip) results for output
/// columns `x0 .. x0+8`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_8(s: __m128i, f01: __m128i, f23: __m128i, f45: __m128i, f67: __m128i) -> __m128i {
    let m01 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let m23 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);
    let m45 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12);
    let m67 = _mm_setr_epi8(6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14);

    let x0 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m01), f01);
    let x1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m23), f23);
    let x2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m45), f45);
    let x3 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m67), f67);

    // Regroup (x0+x2)+(x1+x3) — the only order that keeps every
    // intermediate inside i16 for all VP9 filters (libvpx note).
    let k_64 = _mm_set1_epi16(1 << 6);
    let mut sum1 = _mm_add_epi16(x0, x2);
    let sum2 = _mm_add_epi16(x1, x3);
    sum1 = _mm_add_epi16(sum1, k_64);
    sum1 = _mm_adds_epi16(sum1, sum2);
    _mm_srai_epi16(sum1, 7)
}

/// AVX2 implementation of the 8-tap horizontal sub-pel convolution.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve8_horiz_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1) * src_stride + w + 7` valid bytes
/// (the scalar `convolve8_horiz` footprint); `dst` at least
/// `(h - 1) * dst_stride + w`. `w` and `h` must be in `1..=64`.
// G4: loads from the source scratch use _mm_loadu_si128 (unaligned)
// G4: stores to `dst` use _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_horiz_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    debug_assert!((1..=64).contains(&w));
    debug_assert!((1..=64).contains(&h));
    debug_assert!(src.len() >= (h - 1) * src_stride + w + 7);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);
    // Contract: only sub-pel filter rows reach this kernel. The
    // `maddubs_epi16` signed operand requires every tap to fit in i8
    // (|tap| ≤ 127). The sole VP9 tap that violates this is the
    // full-pel centre tap 128 (filter row 0) — that case never reaches
    // here because `inter_predictor` routes zero sub-pel offsets to
    // `convolve_copy`. See module doc.
    debug_assert!(
        filter_x.iter().all(|&t| (-128..=127).contains(&t)),
        "convolve8_horiz_avx2: tap out of i8 range (full-pel row must use convolve_copy)"
    );

    let f01 = tap_pair(filter_x[0], filter_x[1]);
    let f23 = tap_pair(filter_x[2], filter_x[3]);
    let f45 = tap_pair(filter_x[4], filter_x[5]);
    let f67 = tap_pair(filter_x[6], filter_x[7]);

    for y in 0..h {
        // Copy this row's tap footprint into a zeroed scratch so the
        // 16-byte loads never read past the caller's buffer.
        let mut scratch = [0u8; 80];
        let base = y * src_stride;
        scratch[..w + 7].copy_from_slice(&src[base..base + w + 7]);
        let scratch_ptr = scratch.as_ptr();

        let dst_row = dst.as_mut_ptr().add(y * dst_stride);
        let mut x = 0usize;
        while x < w {
            let s = _mm_loadu_si128(scratch_ptr.add(x) as *const _);
            let res16 = convolve8_8(s, f01, f23, f45, f67);
            let res8 = _mm_packus_epi16(res16, res16);
            let remaining = w - x;
            if remaining >= 8 {
                _mm_storel_epi64(dst_row.add(x) as *mut _, res8);
            } else {
                // Narrow tail (only w == 4 reaches here). Store the
                // valid low bytes via a 32-bit write.
                debug_assert_eq!(remaining, 4);
                let four = _mm_cvtsi128_si32(res8) as u32;
                core::ptr::write_unaligned(dst_row.add(x) as *mut u32, four);
            }
            x += 8;
        }
    }
}

/// Safe wrapper for the AVX2 8-tap horizontal sub-pel convolution.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve8_horiz`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve8_horiz`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_horiz_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // time (see `crate::dsp::dispatch::init`) and are monotonic for the
    // process lifetime. Test paths gate every call on a runtime
    // `is_x86_feature_detected!("avx2")` check via `require_avx2!()`.
    unsafe {
        convolve8_horiz_avx2_inner(src, src_stride, dst, dst_stride, filter_x, w, h);
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve8_horiz_avx2`.
    use super::convolve8_horiz_avx2;
    use crate::inter::filter::{
        convolve8_horiz, BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL,
        EIGHTTAP_SMOOTH_KERNEL,
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

    /// Differential vs scalar across the VP9 (w, h) matrix, every
    /// sub-pel filter row, and all four kernel tables. Row 0 (the
    /// full-pel passthrough, centre tap 128) is skipped: it never
    /// reaches this kernel in production (`inter_predictor` routes
    /// zero sub-pel offsets to `convolve_copy`) and its 128 tap does
    /// not fit the `maddubs` i8 operand. The cartesian product over
    /// the remaining 15 rows × 4 kernels × shapes already exceeds 1024
    /// distinct configurations per shape.
    #[test]
    fn differential_all_filters() {
        require_avx2!();
        let shapes: &[(usize, usize)] = &[
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
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0x1234_5678;
        for &(w, h) in shapes {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..16 {
                        let src_stride = w + 7 + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1) * src_stride + w + 7;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        let mut dst_avx2 = vec![0u8; dst_len];
                        convolve8_horiz(&src, src_stride, &mut dst_scalar, dst_stride, filter, w, h);
                        convolve8_horiz_avx2(
                            &src,
                            src_stride,
                            &mut dst_avx2,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        assert_eq!(
                            dst_scalar, dst_avx2,
                            "mismatch w={w} h={h} filter={filter:?}"
                        );
                    }
                }
            }
        }
    }

    /// Tautology guard — the AVX2 path must actually be exercised on the
    /// test host, and a known sub-pel filter must produce a non-trivial
    /// (non-passthrough) result.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(is_x86_feature_detected!("avx2"));
        // EIGHTTAP row 8 is the half-pel filter — symmetric, definitely
        // not the identity. Feed a ramp and check output differs from a
        // plain copy of the centre tap.
        let filter = &EIGHTTAP_KERNEL[8];
        let w = 8;
        let h = 1;
        let src: Vec<u8> = (0..(w + 7)).map(|i| (i * 13 + 7) as u8).collect();
        let mut dst = vec![0u8; w];
        convolve8_horiz_avx2(&src, w + 7, &mut dst, w, filter, w, h);
        // Output must not equal src[3..3+w] (the integer-pel passthrough).
        assert_ne!(&dst[..], &src[3..3 + w], "half-pel filter must blur the ramp");
    }

    /// Round-trip through SimdDispatch with forced-AVX2 and
    /// forced-scalar caps — identical output on a deterministic input.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);

        let filter = &EIGHTTAP_SHARP_KERNEL[5];
        let mut state: u32 = 0xABCD_1234;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let stride = w + 7;
            let src_len = (h - 1) * stride + w + 7;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = vec![0u8; h * w];
            let mut dst_scalar = vec![0u8; h * w];
            (d_avx2.convolve8_horiz())(&src, stride, &mut dst_avx2, w, filter, w, h);
            (d_scalar.convolve8_horiz())(&src, stride, &mut dst_scalar, w, filter, w, h);
            assert_eq!(dst_avx2, dst_scalar, "dispatch round-trip mismatch w={w} h={h}");
        }
    }
}
