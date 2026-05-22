//! DC_PRED (left-only) intra-predictor — x86_64 AVX2 implementation
//! (Wave I I4, left variant).
//!
//! Structurally identical to
//! [`crate::dsp::x86_64::dc_predict_top_avx2`] — only the input edge
//! axis differs (`left` here, `above` there). The scalar reference is
//! [`crate::intra_pred::dc_left_predictor`].
//!
//! ## Algorithm
//!
//! ```text
//! count = bs                       // bs ∈ {4, 8, 16, 32}
//! sum   = Σ left[..bs]
//! dc    = (sum + count/2) / count  // half-up rounding
//!       = (sum + bs/2) >> log2(bs) // bs is power of 2
//! ```
//!
//! See [`crate::dsp::x86_64::dc_predict_top_avx2`] for the per-size
//! SAD-vs-zero sum strategy table; this kernel reuses the same shape
//! verbatim against `left` instead of `above`.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Unaligned loads/stores throughout.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_extract_epi16, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_sad_epu8, _mm_set1_epi8, _mm_setzero_si128, _mm_storel_epi64,
    _mm_storeu_si128, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_storeu_si256,
};

/// AVX2 implementation of DC_PRED with only `left` available,
/// size-parametric on `bs ∈ {4, 8, 16, 32}`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`dc_predict_left_avx2`]) when `caps.avx2 == true`.
///
/// `left` must have at least `bs` valid u8s. `dst` must hold at least
/// `(bs - 1) * stride + bs` bytes of valid storage. `bs` must be one
/// of `{4, 8, 16, 32}`.
// G4: loads from `left` use _mm_loadu_si128 / _mm_loadl_epi64 / _mm_cvtsi32_si128 + read_unaligned / _mm256_loadu_si256 (unaligned)
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dc_predict_left_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    left: &[u8],
) {
    debug_assert!(left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));

    match bs {
        4 => {
            let left_i32: i32 = core::ptr::read_unaligned(left.as_ptr() as *const i32);
            let v_left: __m128i = _mm_cvtsi32_si128(left_i32);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_left, zero);
            let sum = _mm_cvtsi128_si32(sad) as u32;
            let dc: u8 = ((sum + 2) >> 2) as u8;
            let row: u32 = (dc as u32) * 0x0101_0101;
            for r in 0..4 {
                core::ptr::write_unaligned(
                    dst.as_mut_ptr().add(r * stride) as *mut u32,
                    row,
                );
            }
        }
        8 => {
            let v_left: __m128i = _mm_loadl_epi64(left.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_left, zero);
            let sum = _mm_cvtsi128_si32(sad) as u32;
            let dc: u8 = ((sum + 4) >> 3) as u8;
            let row_v: __m128i = _mm_set1_epi8(dc as i8);
            for r in 0..8 {
                _mm_storel_epi64(
                    dst.as_mut_ptr().add(r * stride) as *mut __m128i,
                    row_v,
                );
            }
        }
        16 => {
            let v_left: __m128i = _mm_loadu_si128(left.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_left, zero);
            let lo = _mm_extract_epi16::<0>(sad) as u32;
            let hi = _mm_extract_epi16::<4>(sad) as u32;
            let sum = lo + hi;
            let dc: u8 = ((sum + 8) >> 4) as u8;
            let row_v: __m128i = _mm_set1_epi8(dc as i8);
            for r in 0..16 {
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(r * stride) as *mut __m128i,
                    row_v,
                );
            }
        }
        32 => {
            let v_left: __m256i = _mm256_loadu_si256(left.as_ptr() as *const __m256i);
            let zero = _mm256_setzero_si256();
            let sad: __m256i = _mm256_sad_epu8(v_left, zero);
            let sad_lo: __m128i = _mm256_castsi256_si128(sad);
            let sad_hi: __m128i = _mm256_extracti128_si256::<1>(sad);
            let s0 = _mm_extract_epi16::<0>(sad_lo) as u32;
            let s1 = _mm_extract_epi16::<4>(sad_lo) as u32;
            let s2 = _mm_extract_epi16::<0>(sad_hi) as u32;
            let s3 = _mm_extract_epi16::<4>(sad_hi) as u32;
            let sum = s0 + s1 + s2 + s3;
            let dc: u8 = ((sum + 16) >> 5) as u8;
            let row_v: __m256i = _mm256_set1_epi8(dc as i8);
            for r in 0..32 {
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(r * stride) as *mut __m256i,
                    row_v,
                );
            }
        }
        _ => unreachable!("dc_predict_left_avx2: bs must be in {{4, 8, 16, 32}}"),
    }
}

/// Safe wrapper for the AVX2 DC_PRED (left-only) predictor.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::dc_predict_left`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::intra_pred::dc_left_predictor`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn dc_predict_left_avx2(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { dc_predict_left_avx2_inner(dst, stride, bs, left) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::dc_predict_left_avx2;
    use crate::intra_pred::dc_left_predictor;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn run_scalar(bs: usize, left: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        dc_left_predictor(&mut dst, bs, bs, left);
        dst
    }

    fn run_avx2(bs: usize, left: &[u8]) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        dc_predict_left_avx2(&mut dst, bs, bs, left);
        dst
    }

    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut state: u32 = 0xC0FF_EE46;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let mut left = vec![0u8; bs];
                for slot in left.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let scalar = run_scalar(bs, &left);
                let avx2 = run_avx2(bs, &left);
                assert_eq!(
                    scalar, avx2,
                    "AVX2/scalar mismatch at bs={bs} left={left:?}"
                );
            }
        }
    }

    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
    }

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

        for bs in [4usize, 8, 16, 32] {
            let left: Vec<u8> = (0..bs).map(|i| (i as u32 * 29 + 11) as u8).collect();
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.dc_predict_left())(&mut dst_avx2, bs, bs, &left);
            (dispatch_scalar.dc_predict_left())(&mut dst_scalar, bs, bs, &left);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch at bs={bs}"
            );
        }
    }

    #[test]
    fn directed_boundary_inputs() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            for &l in &[0u8, 1, 127, 128, 255] {
                let left = vec![l; bs];
                let scalar = run_scalar(bs, &left);
                let avx2 = run_avx2(bs, &left);
                assert_eq!(scalar, avx2, "boundary mismatch bs={bs} left={l}");
            }
        }
    }
}
