//! DC_PRED (above-only) intra-predictor — x86_64 AVX2 implementation
//! (Wave I I4, top variant).
//!
//! Mirrors the size-parametric slot pattern of the I1/I2/I3 kernels
//! and shares the structural shape of `dc_predict_full_avx2` — the
//! only differences are halved input (only `above`, not `above + left`),
//! halved divisor (`bs` instead of `2 * bs`), and matched round bias
//! (`bs / 2`). The scalar reference is
//! [`crate::intra_pred::dc_top_predictor`].
//!
//! ## Algorithm
//!
//! ```text
//! count = bs                       // bs ∈ {4, 8, 16, 32}
//! sum   = Σ above[..bs]
//! dc    = (sum + count/2) / count  // half-up rounding
//!       = (sum + bs/2) >> log2(bs) // bs is power of 2
//! ```
//!
//! ## Size dispatch
//!
//! | `bs` | sum kernel             | round bias | shift | store width                       |
//! |------|------------------------|------------|-------|-----------------------------------|
//! | 4    | `_mm_sad_epu8` (4-byte payload) | + 2 | >> 2  | `write_unaligned::<u32>`          |
//! | 8    | `_mm_sad_epu8`         | + 4        | >> 3  | `_mm_storel_epi64` (64-bit)       |
//! | 16   | `_mm_sad_epu8` × 2 (low+high) | + 8 | >> 4 | `_mm_storeu_si128` (128-bit)     |
//! | 32   | `_mm256_sad_epu8` × 1  | + 16       | >> 5  | `_mm256_storeu_si256` (256-bit)   |
//!
//! See [`crate::dsp::x86_64::dc_predict_full_avx2`] for the SAD-vs-zero
//! trick used to convert an unsigned byte vector into a horizontal byte
//! sum in one instruction.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `above` use unaligned intrinsics. All stores into
//! `dst` use unaligned intrinsics. Grep sentinels per the family
//! convention.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_extract_epi16, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_sad_epu8, _mm_set1_epi8, _mm_setzero_si128, _mm_storel_epi64,
    _mm_storeu_si128, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_storeu_si256,
};

/// AVX2 implementation of DC_PRED with only `above` available,
/// size-parametric on `bs ∈ {4, 8, 16, 32}`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`dc_predict_top_avx2`]) when `caps.avx2 == true`.
///
/// `above` must have at least `bs` valid u8s. `dst` must hold at
/// least `(bs - 1) * stride + bs` bytes of valid storage. `bs` must
/// be one of `{4, 8, 16, 32}`.
// G4: loads from `above` use _mm_loadu_si128 / _mm_loadl_epi64 / _mm_cvtsi32_si128 + read_unaligned / _mm256_loadu_si256 (unaligned)
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dc_predict_top_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
) {
    debug_assert!(above.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));

    match bs {
        4 => {
            let above_i32: i32 = core::ptr::read_unaligned(above.as_ptr() as *const i32);
            let v_above: __m128i = _mm_cvtsi32_si128(above_i32);
            let zero = _mm_setzero_si128();
            // _mm_sad_epu8 on a 16-byte vector with only the low 4
            // bytes populated still works — the unused 12 high bytes
            // are zero and contribute nothing to the SAD lane sums.
            let sad: __m128i = _mm_sad_epu8(v_above, zero);
            let sum = _mm_cvtsi128_si32(sad) as u32;
            // count = 4, round bias = 2, shift = 2.
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
            let v_above: __m128i = _mm_loadl_epi64(above.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_above, zero);
            // sad has the 8-byte sum in lane 0; lane 4 sums the 8
            // zero bytes from the high half of the loaded register
            // and is therefore 0 — extracting just lane 0 suffices.
            let sum = _mm_cvtsi128_si32(sad) as u32;
            // count = 8, round bias = 4, shift = 3.
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
            let v_above: __m128i = _mm_loadu_si128(above.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_above, zero);
            // 16 bytes split into two 8-byte groups; SAD-vs-zero
            // returns two 16-bit partial sums in lanes 0 and 4.
            let lo = _mm_extract_epi16::<0>(sad) as u32;
            let hi = _mm_extract_epi16::<4>(sad) as u32;
            let sum = lo + hi;
            // count = 16, round bias = 8, shift = 4.
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
            let v_above: __m256i = _mm256_loadu_si256(above.as_ptr() as *const __m256i);
            let zero = _mm256_setzero_si256();
            let sad: __m256i = _mm256_sad_epu8(v_above, zero);
            // 32 bytes split into four 8-byte groups; SAD-vs-zero
            // returns four 16-bit partial sums in lanes 0/4/8/12
            // (= two __m128i halves, each with sums in lanes 0 and 4).
            let sad_lo: __m128i = _mm256_castsi256_si128(sad);
            let sad_hi: __m128i = _mm256_extracti128_si256::<1>(sad);
            let s0 = _mm_extract_epi16::<0>(sad_lo) as u32;
            let s1 = _mm_extract_epi16::<4>(sad_lo) as u32;
            let s2 = _mm_extract_epi16::<0>(sad_hi) as u32;
            let s3 = _mm_extract_epi16::<4>(sad_hi) as u32;
            let sum = s0 + s1 + s2 + s3;
            // count = 32, round bias = 16, shift = 5.
            let dc: u8 = ((sum + 16) >> 5) as u8;
            let row_v: __m256i = _mm256_set1_epi8(dc as i8);
            for r in 0..32 {
                _mm256_storeu_si256(
                    dst.as_mut_ptr().add(r * stride) as *mut __m256i,
                    row_v,
                );
            }
        }
        _ => unreachable!("dc_predict_top_avx2: bs must be in {{4, 8, 16, 32}}"),
    }
}

/// Safe wrapper for the AVX2 DC_PRED (above-only) predictor.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::dc_predict_top`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::intra_pred::dc_top_predictor`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn dc_predict_top_avx2(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { dc_predict_top_avx2_inner(dst, stride, bs, above) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::dc_predict_top_avx2;
    use crate::intra_pred::dc_top_predictor;

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

    fn run_scalar(bs: usize, above: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        dc_top_predictor(&mut dst, bs, bs, above);
        dst
    }

    fn run_avx2(bs: usize, above: &[u8]) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        dc_predict_top_avx2(&mut dst, bs, bs, above);
        dst
    }

    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut state: u32 = 0xC0FF_EE45;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let mut above = vec![0u8; bs];
                for slot in above.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let scalar = run_scalar(bs, &above);
                let avx2 = run_avx2(bs, &above);
                assert_eq!(
                    scalar, avx2,
                    "AVX2/scalar mismatch at bs={bs} above={above:?}"
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
            let above: Vec<u8> = (0..bs).map(|i| (i as u32 * 23 + 7) as u8).collect();
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.dc_predict_top())(&mut dst_avx2, bs, bs, &above);
            (dispatch_scalar.dc_predict_top())(&mut dst_scalar, bs, bs, &above);
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
            for &a in &[0u8, 1, 127, 128, 255] {
                let above = vec![a; bs];
                let scalar = run_scalar(bs, &above);
                let avx2 = run_avx2(bs, &above);
                assert_eq!(scalar, avx2, "boundary mismatch bs={bs} above={a}");
            }
        }
    }
}
