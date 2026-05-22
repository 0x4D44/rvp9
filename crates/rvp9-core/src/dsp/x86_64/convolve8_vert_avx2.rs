//! 8-tap vertical sub-pel convolution — x86_64 AVX2 implementation
//! (Wave J J4).
//!
//! Sibling of [`crate::dsp::x86_64::convolve8_horiz_avx2`]: for every
//! output pixel, convolve eight vertically-adjacent source samples with
//! the resolved sub-pel filter row:
//!
//! ```text
//! dst[y][x] = clip255( ( Σ_{k=0..8} src[(y+k)][x] * filter_y[k] + 64 ) >> 7 )
//! ```
//!
//! Mirrors libvpx `convolve_vert` (`vpx_dsp/vpx_convolve.c:67-88`) with
//! `y_step_q4 == SUBPEL_SHIFTS`. The scalar reference is
//! [`crate::inter::filter::convolve8_vert`]; callers pre-shift `src` up
//! by three rows so `src[(y)*src_stride + x]` is the first of the eight
//! taps for output `(y, x)`.
//!
//! ## Algorithm
//!
//! Columns are processed in 16-wide groups. For each output row, the
//! eight contributing source rows are loaded and consecutive rows are
//! interleaved byte-wise (`_mm_unpacklo_epi8` for the low 8 columns,
//! `_mm_unpackhi_epi8` for the high 8) so each 16-bit lane holds the
//! tap-pair operand `(src[row_k][c], src[row_{k+1}][c])`. Those feed the
//! same verified `convolve8_16` reduction as the horizontal kernel:
//! 4× `maddubs_epi16`, regrouped `(x0+x2)+(x1+x3)` with early `+64`
//! rounding and saturating final add, `srai 7`, then `packus_epi16`.
//!
//! ## Bounds
//!
//! Vertical column loads are exactly the group width (4 / 8 / 16 bytes),
//! so there is no horizontal over-read — the kernel reads `src`
//! directly with no scratch. The vertical extent (`h + 7` rows) is
//! covered by the scalar `convolve8_vert` footprint
//! (`(h - 1 + 7) * src_stride + w`).
//!
//! ## Contract
//!
//! Sub-pel filter rows only (taps fit i8). The full-pel row (centre tap
//! 128) is routed to `convolve_copy` upstream — see the horizontal
//! kernel's module doc.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `src` and stores to `dst` use unaligned intrinsics.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi16, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_maddubs_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_srai_epi16,
    _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
};

/// Pack two adjacent filter taps into the replicated i8-pair form
/// consumed by `_mm_maddubs_epi16`'s signed operand.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair(lo: i16, hi: i16) -> __m128i {
    _mm_set1_epi16(i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]))
}

/// Load `gw` source bytes (one column-group of one row) into the low
/// lanes of an `__m128i`. `gw ∈ {4, 8, 16}`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_cols(ptr: *const u8, gw: usize) -> __m128i {
    match gw {
        4 => _mm_cvtsi32_si128(core::ptr::read_unaligned(ptr as *const i32)),
        8 => _mm_loadl_epi64(ptr as *const _),
        _ => _mm_loadu_si128(ptr as *const _),
    }
}

/// Reduce four `maddubs` partial sums to the rounded/shifted (pre-clip)
/// i16 result, matching libvpx `convolve8_16` ordering exactly.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn combine(x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i) -> __m128i {
    let k_64 = _mm_set1_epi16(1 << 6);
    let mut sum1 = _mm_add_epi16(x0, x2);
    let sum2 = _mm_add_epi16(x1, x3);
    sum1 = _mm_add_epi16(sum1, k_64);
    sum1 = _mm_adds_epi16(sum1, sum2);
    _mm_srai_epi16(sum1, 7)
}

/// Vertical 8-tap reduction over eight source rows for one half
/// (low 8 columns if `!hi`, high 8 columns if `hi`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn vconv(
    r: &[__m128i; 8],
    f01: __m128i,
    f23: __m128i,
    f45: __m128i,
    f67: __m128i,
    hi: bool,
) -> __m128i {
    let (u0, u1, u2, u3) = if hi {
        (
            _mm_unpackhi_epi8(r[0], r[1]),
            _mm_unpackhi_epi8(r[2], r[3]),
            _mm_unpackhi_epi8(r[4], r[5]),
            _mm_unpackhi_epi8(r[6], r[7]),
        )
    } else {
        (
            _mm_unpacklo_epi8(r[0], r[1]),
            _mm_unpacklo_epi8(r[2], r[3]),
            _mm_unpacklo_epi8(r[4], r[5]),
            _mm_unpacklo_epi8(r[6], r[7]),
        )
    };
    combine(
        _mm_maddubs_epi16(u0, f01),
        _mm_maddubs_epi16(u1, f23),
        _mm_maddubs_epi16(u2, f45),
        _mm_maddubs_epi16(u3, f67),
    )
}

/// AVX2 implementation of the 8-tap vertical sub-pel convolution.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve8_vert_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1 + 7) * src_stride + w` valid bytes
/// (the scalar `convolve8_vert` footprint); `dst` at least
/// `(h - 1) * dst_stride + w`. `w` and `h` must be in `1..=64`.
// G4: loads from `src` use _mm_loadu_si128 / _mm_loadl_epi64 / read_unaligned (unaligned)
// G4: stores to `dst` use _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_vert_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    debug_assert!((1..=64).contains(&w));
    debug_assert!((1..=64).contains(&h));
    debug_assert!(src.len() >= (h - 1 + 7) * src_stride + w);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);
    debug_assert!(
        filter_y.iter().all(|&t| (-128..=127).contains(&t)),
        "convolve8_vert_avx2: tap out of i8 range (full-pel row must use convolve_copy)"
    );

    let f01 = tap_pair(filter_y[0], filter_y[1]);
    let f23 = tap_pair(filter_y[2], filter_y[3]);
    let f45 = tap_pair(filter_y[4], filter_y[5]);
    let f67 = tap_pair(filter_y[6], filter_y[7]);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let mut cg = 0usize;
    while cg < w {
        let gw = (w - cg).min(16);
        for y in 0..h {
            // Load the eight contributing source rows for this group.
            let mut r = [load_cols(src_ptr, gw); 8];
            for (k, slot) in r.iter_mut().enumerate() {
                *slot = load_cols(src_ptr.add((y + k) * src_stride + cg), gw);
            }
            let dst_row = dst_ptr.add(y * dst_stride + cg);
            if gw <= 8 {
                let res16 = vconv(&r, f01, f23, f45, f67, false);
                let res8 = _mm_packus_epi16(res16, res16);
                if gw == 8 {
                    _mm_storel_epi64(dst_row as *mut _, res8);
                } else {
                    debug_assert_eq!(gw, 4);
                    let four = _mm_cvtsi128_si32(res8) as u32;
                    core::ptr::write_unaligned(dst_row as *mut u32, four);
                }
            } else {
                let lo = vconv(&r, f01, f23, f45, f67, false);
                let hi = vconv(&r, f01, f23, f45, f67, true);
                let res16 = _mm_packus_epi16(lo, hi);
                _mm_storeu_si128(dst_row as *mut _, res16);
            }
        }
        cg += 16;
    }
}

/// Safe wrapper for the AVX2 8-tap vertical sub-pel convolution.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve8_vert`] points
/// at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve8_vert`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_vert_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // time and are monotonic for the process lifetime. Test paths gate
    // every call on a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe {
        convolve8_vert_avx2_inner(src, src_stride, dst, dst_stride, filter_y, w, h);
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve8_vert_avx2`.
    use super::convolve8_vert_avx2;
    use crate::inter::filter::{
        convolve8_vert, BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL,
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
    /// sub-pel filter row (skipping the full-pel row 0 — routed to
    /// `convolve_copy` upstream), and all four kernel tables.
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
        let mut state: u32 = 0x9E37_79B9;
        for &(w, h) in shapes {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..16 {
                        let src_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1 + 7) * src_stride + w;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        let mut dst_avx2 = vec![0u8; dst_len];
                        convolve8_vert(&src, src_stride, &mut dst_scalar, dst_stride, filter, w, h);
                        convolve8_vert_avx2(
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

    /// Tautology guard — AVX2 path exercised; a half-pel filter blurs.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(is_x86_feature_detected!("avx2"));
        let filter = &EIGHTTAP_KERNEL[8];
        let w = 8;
        let h = 4;
        let stride = w;
        let src_len = (h - 1 + 7) * stride + w;
        let src: Vec<u8> = (0..src_len).map(|i| (i * 7 + 11) as u8).collect();
        let mut dst = vec![0u8; w * h];
        convolve8_vert_avx2(&src, stride, &mut dst, w, filter, w, h);
        // Output row 0 must not equal src row 3 (the integer-pel pass).
        assert_ne!(&dst[..w], &src[3 * stride..3 * stride + w], "half-pel must blur");
    }

    /// Round-trip through SimdDispatch with forced-AVX2 / forced-scalar.
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

        let filter = &EIGHTTAP_SMOOTH_KERNEL[11];
        let mut state: u32 = 0x0BAD_F00D;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let stride = w;
            let src_len = (h - 1 + 7) * stride + w;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = vec![0u8; h * w];
            let mut dst_scalar = vec![0u8; h * w];
            (d_avx2.convolve8_vert())(&src, stride, &mut dst_avx2, w, filter, w, h);
            (d_scalar.convolve8_vert())(&src, stride, &mut dst_scalar, w, filter, w, h);
            assert_eq!(dst_avx2, dst_scalar, "dispatch round-trip mismatch w={w} h={h}");
        }
    }
}
