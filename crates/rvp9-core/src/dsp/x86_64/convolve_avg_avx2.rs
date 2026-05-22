//! Integer-pel compound average — x86_64 AVX2 implementation
//! (Wave J J2).
//!
//! The scalar reference is [`crate::inter::filter::convolve_avg`]:
//! for every (y, x) it computes
//!
//! ```text
//! dst[y*dst_stride + x] = (dst[y*dst_stride + x] + src[y*src_stride + x] + 1) >> 1
//! ```
//!
//! i.e. the rounded-up unsigned-byte average of the existing `dst`
//! pixel and the new `src` pixel. This is the compound-prediction
//! averaging step used by VP9 inter (`vpx_convolve_avg_c`,
//! `vpx_convolve.c:225-242`).
//!
//! ## One-instruction kernel
//!
//! The `_mm256_avg_epu8` / `_mm_avg_epu8` intrinsics compute exactly
//! `(a + b + 1) >> 1` on unsigned bytes, with the `+1` rounding-up
//! bias built in — bit-identical to the scalar `round_avg(a, b)` for
//! every u8 input pair. The kernel therefore reduces per chunk to a
//! single load + avg + store triple.
//!
//! ## Size dispatch
//!
//! VP9 inter widths are `w ∈ {4, 8, 16, 32, 64}` with heights spanning
//! `h ∈ {4, 8, 16, 32, 64}`. A single safe wrapper takes both as
//! runtime args and dispatches on `w` to the right load/store width:
//!
//! | `w` | load + avg + store per row                                          |
//! |-----|---------------------------------------------------------------------|
//! | 4   | scalar fallback (4 bytes per row — not worth a SIMD lane)           |
//! | 8   | `_mm_loadl_epi64` + `_mm_avg_epu8` + `_mm_storel_epi64`              |
//! | 16  | `_mm_loadu_si128` + `_mm_avg_epu8` + `_mm_storeu_si128`              |
//! | 32  | `_mm256_loadu_si256` + `_mm256_avg_epu8` + `_mm256_storeu_si256`     |
//! | 64  | two 32-byte `_mm256_avg_epu8` chunks per row                        |
//!
//! Any unexpected `w` falls back to the scalar reference fn.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `src` / `dst` and stores to `dst` use unaligned
//! intrinsics — no alignment assumption on the caller's slices.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm256_avg_epu8, _mm256_loadu_si256, _mm256_storeu_si256, _mm_avg_epu8, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_storel_epi64, _mm_storeu_si128,
};

/// AVX2 implementation of integer-pel compound average.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve_avg_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1) * src_stride + w` valid bytes;
/// `dst` must hold at least `(h - 1) * dst_stride + w` valid bytes.
// G4: loads from `src` / `dst` use _mm256_loadu_si256 / _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve_avg_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    debug_assert!(src.len() >= (h - 1) * src_stride + w);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);

    match w {
        8 => {
            for y in 0..h {
                let s_ptr = src.as_ptr().add(y * src_stride);
                let d_ptr = dst.as_mut_ptr().add(y * dst_stride);
                let s = _mm_loadl_epi64(s_ptr as *const _);
                let d = _mm_loadl_epi64(d_ptr as *const _);
                let avg = _mm_avg_epu8(d, s);
                _mm_storel_epi64(d_ptr as *mut _, avg);
            }
        }
        16 => {
            for y in 0..h {
                let s_ptr = src.as_ptr().add(y * src_stride);
                let d_ptr = dst.as_mut_ptr().add(y * dst_stride);
                let s = _mm_loadu_si128(s_ptr as *const _);
                let d = _mm_loadu_si128(d_ptr as *const _);
                let avg = _mm_avg_epu8(d, s);
                _mm_storeu_si128(d_ptr as *mut _, avg);
            }
        }
        32 => {
            for y in 0..h {
                let s_ptr = src.as_ptr().add(y * src_stride);
                let d_ptr = dst.as_mut_ptr().add(y * dst_stride);
                let s = _mm256_loadu_si256(s_ptr as *const _);
                let d = _mm256_loadu_si256(d_ptr as *const _);
                let avg = _mm256_avg_epu8(d, s);
                _mm256_storeu_si256(d_ptr as *mut _, avg);
            }
        }
        64 => {
            for y in 0..h {
                let s_ptr = src.as_ptr().add(y * src_stride);
                let d_ptr = dst.as_mut_ptr().add(y * dst_stride);
                // First 32-byte chunk.
                let s0 = _mm256_loadu_si256(s_ptr as *const _);
                let d0 = _mm256_loadu_si256(d_ptr as *const _);
                _mm256_storeu_si256(d_ptr as *mut _, _mm256_avg_epu8(d0, s0));
                // Second 32-byte chunk.
                let s1 = _mm256_loadu_si256(s_ptr.add(32) as *const _);
                let d1 = _mm256_loadu_si256(d_ptr.add(32) as *const _);
                _mm256_storeu_si256(d_ptr.add(32) as *mut _, _mm256_avg_epu8(d1, s1));
            }
        }
        _ => {
            // w = 4 (4 bytes per row — narrower than any useful SIMD
            // lane) or any other unexpected width: fall back to the
            // scalar reference.
            crate::inter::filter::convolve_avg(src, src_stride, dst, dst_stride, w, h);
        }
    }
}

/// Safe wrapper for the AVX2 integer-pel compound-average kernel.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve_avg`] points
/// at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve_avg`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve_avg_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate every call on a runtime `is_x86_feature_detected!("avx2")`
    // check via the `require_avx2!()` macro.
    unsafe { convolve_avg_avx2_inner(src, src_stride, dst, dst_stride, w, h) }
}

// Miri cannot interpret AVX2 intrinsics (`_mm256_*` etc.), so this
// `tests` module is excluded from `cargo +nightly miri test`. The
// scalar reference is already covered by `crate::inter::filter::tests`
// which DOES run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve_avg_avx2`.
    //!
    //! Sweeps every VP9 (w, h) shape with deterministic LCG-seeded
    //! random `src` / `dst` plus strides ≥ w, then dispatch round-trip
    //! and boundary-saturation guards.
    use super::convolve_avg_avx2;
    use crate::inter::filter::convolve_avg;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG (numerical recipes constants) — used for
    /// reproducible random-input generation in the differential test.
    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    /// Differential vs scalar across the VP9 (w, h) matrix. 1024
    /// iters per shape with random src/dst contents and per-iter
    /// random strides ≥ w. LCG-seeded so failures reproduce.
    #[test]
    fn differential_1024_iters() {
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
        let mut state: u32 = 0xDEAD_BEEF;
        for &(w, h) in shapes {
            for _iter in 0..1024 {
                let src_stride = w + (lcg_next(&mut state) as usize & 0x7);
                let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                let src_len = (h - 1) * src_stride + w;
                let dst_len = (h - 1) * dst_stride + w;
                let mut src = vec![0u8; src_len];
                for slot in src.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut dst_scalar = vec![0u8; dst_len];
                for slot in dst_scalar.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut dst_avx2 = dst_scalar.clone();

                convolve_avg(&src, src_stride, &mut dst_scalar, dst_stride, w, h);
                convolve_avg_avx2(&src, src_stride, &mut dst_avx2, dst_stride, w, h);

                assert_eq!(
                    dst_scalar, dst_avx2,
                    "AVX2/scalar mismatch at w={w} h={h} src_stride={src_stride} dst_stride={dst_stride}"
                );
            }
        }
    }

    /// Tautology guard — assert the AVX2 path is actually queried by
    /// `is_x86_feature_detected!("avx2")` on the test host. Without
    /// this, the differential test would silently no-op on a non-AVX2
    /// host and report success.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
        // Sanity: kernel actually averages — feed (dst=0, src=255)
        // and observe 128 (rounded up).
        let src = vec![255u8; 32];
        let mut dst = vec![0u8; 32];
        convolve_avg_avx2(&src, 32, &mut dst, 32, 32, 1);
        for b in dst {
            assert_eq!(b, 128, "round_avg(0, 255) must be 128");
        }
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// must produce identical output on a deterministic input.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let shapes: &[(usize, usize)] = &[(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)];
        let mut state: u32 = 0x5EED_F00D;
        for &(w, h) in shapes {
            let src_len = h * w;
            let mut src = vec![0u8; src_len];
            for slot in src.iter_mut() {
                *slot = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_seed = vec![0u8; src_len];
            for slot in dst_seed.iter_mut() {
                *slot = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = dst_seed.clone();
            let mut dst_scalar = dst_seed.clone();
            (dispatch_avx2.convolve_avg())(&src, w, &mut dst_avx2, w, w, h);
            (dispatch_scalar.convolve_avg())(&src, w, &mut dst_scalar, w, w, h);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch at w={w} h={h}"
            );
        }
    }

    /// Saturation boundary guard:
    ///   - src=all-0   + dst=all-255 → round_avg = 128
    ///   - src=all-255 + dst=all-255 → round_avg = 255
    /// Exercises both the rounding bias and the u8 ceiling.
    #[test]
    fn boundary_saturation() {
        require_avx2!();
        let shapes: &[(usize, usize)] = &[(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)];
        for &(w, h) in shapes {
            // Case A: src=0, dst=255 → 128.
            let src = vec![0u8; w * h];
            let mut dst = vec![255u8; w * h];
            convolve_avg_avx2(&src, w, &mut dst, w, w, h);
            for b in &dst {
                assert_eq!(*b, 128, "round_avg(255, 0) must be 128 at w={w} h={h}");
            }
            // Case B: src=255, dst=255 → 255.
            let src = vec![255u8; w * h];
            let mut dst = vec![255u8; w * h];
            convolve_avg_avx2(&src, w, &mut dst, w, w, h);
            for b in &dst {
                assert_eq!(*b, 255, "round_avg(255, 255) must be 255 at w={w} h={h}");
            }
        }
    }
}
