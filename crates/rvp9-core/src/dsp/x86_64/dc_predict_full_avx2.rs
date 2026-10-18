//! DC_PRED (both-neighbours available) intra-predictor — x86_64 AVX2
//! implementation (Wave I I4, full variant).
//!
//! Mirrors the safe-wrapper convention established by the I1 V_PRED
//! / I2 H_PRED / I3 TM_PRED kernels. The scalar reference is
//! [`crate::intra_pred::dc_predictor`].
//!
//! ## Algorithm
//!
//! DC_PRED (both neighbours) averages the `2 * bs` neighbour pixels
//! (`above[..bs] ++ left[..bs]`), rounds half-up, then fills the entire
//! `bs × bs` block with that constant byte:
//!
//! ```text
//! count = 2 * bs
//! sum   = Σ above[..bs] + Σ left[..bs]
//! dc    = (sum + count/2) / count        // half-up rounding
//! ```
//!
//! Because `count = 2 * bs ∈ {8, 16, 32, 64}` is always a power of two,
//! the division collapses to a right shift: `(sum + count/2) >> log2(count)`.
//!
//! ## Size dispatch
//!
//! `bs ∈ {4, 8, 16, 32}`. Single safe wrapper accepts `bs` as a runtime
//! arg and dispatches internally on it. Per-size shape:
//!
//! | `bs` | bytes summed | sum kernel             | round bias | shift | store width                       |
//! |------|--------------|------------------------|------------|-------|-----------------------------------|
//! | 4    | 8            | `_mm_sad_epu8` × 1     | + 4        | >> 3  | `write_unaligned::<u32>`          |
//! | 8    | 16           | `_mm_sad_epu8` × 1     | + 8        | >> 4  | `_mm_storel_epi64` (64-bit)       |
//! | 16   | 32           | `_mm256_sad_epu8` × 1  | + 16       | >> 5  | `_mm_storeu_si128` (128-bit)      |
//! | 32   | 64           | `_mm256_sad_epu8` × 2  | + 32       | >> 6  | `_mm256_storeu_si256` (256-bit)   |
//!
//! `_mm{,256}_sad_epu8(v, 0)` computes |v - 0| summed in 8-byte groups;
//! since `v` is unsigned, this is just the byte sum. The 128-bit form
//! produces two 16-bit lane sums in lanes 0 and 4; the 256-bit form
//! produces four 16-bit lane sums in lanes 0/4/8/12 (one per 8-byte
//! group). We combine these scalar partial sums with a couple of u32
//! adds — the divisor is `2*bs ≤ 64` so the full sum fits in u16 well
//! within u32's range.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `above` / `left` use unaligned intrinsics
//! (`_mm_loadu_si128` / `_mm_loadl_epi64` / `_mm_cvtsi32_si128`). All
//! stores into `dst` use `_mm256_storeu_si256` / `_mm_storeu_si128` /
//! `_mm_storel_epi64` / `write_unaligned`. The grep sentinels below
//! are checked by `scripts/check-simd-sentinels.sh` / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_extract_epi16, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_sad_epu8, _mm_set1_epi8, _mm_setzero_si128, _mm_storel_epi64,
    _mm_storeu_si128, _mm_unpacklo_epi32, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_sad_epu8,
    _mm256_set1_epi8, _mm256_setzero_si256, _mm256_storeu_si256,
};

/// AVX2 implementation of DC_PRED with both `above` and `left`
/// available, size-parametric on `bs ∈ {4, 8, 16, 32}`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`dc_predict_full_avx2`]) when `caps.avx2 == true`.
///
/// `above` and `left` must each have at least `bs` valid u8s. `dst`
/// must hold at least `(bs - 1) * stride + bs` bytes of valid storage.
/// `bs` must be one of `{4, 8, 16, 32}`.
// G4: loads from `above` / `left` use _mm_loadu_si128 / _mm_loadl_epi64 / _mm_cvtsi32_si128 + read_unaligned / _mm256_loadu_si256 (unaligned)
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dc_predict_full_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
) {
    debug_assert!(above.len() >= bs);
    debug_assert!(left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));

    match bs {
        4 => {
            // 8 bytes total: above[0..4] + left[0..4] packed into the
            // low 8 bytes of an __m128i, then _mm_sad_epu8 vs zero
            // returns the full byte-sum in the low 16 bits.
            let above_i32: i32 = core::ptr::read_unaligned(above.as_ptr() as *const i32);
            let left_i32: i32 = core::ptr::read_unaligned(left.as_ptr() as *const i32);
            let v_above = _mm_cvtsi32_si128(above_i32);
            let v_left = _mm_cvtsi32_si128(left_i32);
            let v_packed: __m128i = _mm_unpacklo_epi32(v_above, v_left);
            let zero = _mm_setzero_si128();
            let sum_v = _mm_sad_epu8(v_packed, zero);
            let sum = _mm_cvtsi128_si32(sum_v) as u32;
            // count = 8, round bias = 4, shift = 3.
            let dc: u8 = ((sum + 4) >> 3) as u8;
            let row: u32 = (dc as u32) * 0x0101_0101;
            for r in 0..4 {
                core::ptr::write_unaligned(dst.as_mut_ptr().add(r * stride) as *mut u32, row);
            }
        }
        8 => {
            // 16 bytes total: above[0..8] in low 8 bytes, left[0..8] in
            // high 8 bytes of one __m128i. _mm_sad_epu8 vs zero gives
            // two 16-bit partial sums (lanes 0 and 4); we add them.
            let v_above: __m128i = _mm_loadl_epi64(above.as_ptr() as *const __m128i);
            let v_left: __m128i = _mm_loadl_epi64(left.as_ptr() as *const __m128i);
            // Compose: low 8 bytes = above, high 8 bytes = left. Use
            // _mm_unpacklo_epi64 to interleave the two 64-bit halves.
            let v_packed: __m128i = core::arch::x86_64::_mm_unpacklo_epi64(v_above, v_left);
            let zero = _mm_setzero_si128();
            let sad: __m128i = _mm_sad_epu8(v_packed, zero);
            // sad has 16-bit sums in lanes 0 and 4 (= byte indices 0..2
            // and 8..10). Extract both as 16-bit values.
            let lo = _mm_extract_epi16::<0>(sad) as u32;
            let hi = _mm_extract_epi16::<4>(sad) as u32;
            let sum = lo + hi;
            // count = 16, round bias = 8, shift = 4.
            let dc: u8 = ((sum + 8) >> 4) as u8;
            let row_v: __m128i = _mm_set1_epi8(dc as i8);
            for r in 0..8 {
                _mm_storel_epi64(dst.as_mut_ptr().add(r * stride) as *mut __m128i, row_v);
            }
        }
        16 => {
            // 32 bytes total: above[0..16] + left[0..16] = one __m256i.
            // _mm256_sad_epu8 vs zero produces four 16-bit lane sums
            // (one per 8-byte group). Combine into one u32.
            let v_above: __m128i = _mm_loadu_si128(above.as_ptr() as *const __m128i);
            let v_left: __m128i = _mm_loadu_si128(left.as_ptr() as *const __m128i);
            // Compose into one __m256i: low 128 = above, high 128 = left.
            // `_mm256_set_m128i` is the conceptual op; use the
            // `castsi128_si256` + `inserti128` pair which is the
            // standardised form in `core::arch::x86_64`.
            let v_packed: __m256i =
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(v_above), v_left);
            let zero = _mm256_setzero_si256();
            let sad: __m256i = _mm256_sad_epu8(v_packed, zero);
            // Sum the four 16-bit lane partials. Extract the two
            // __m128i halves; each holds two 16-bit sums in lanes 0 and 4.
            let sad_lo: __m128i = _mm256_castsi256_si128(sad);
            let sad_hi: __m128i = _mm256_extracti128_si256::<1>(sad);
            let s0 = _mm_extract_epi16::<0>(sad_lo) as u32;
            let s1 = _mm_extract_epi16::<4>(sad_lo) as u32;
            let s2 = _mm_extract_epi16::<0>(sad_hi) as u32;
            let s3 = _mm_extract_epi16::<4>(sad_hi) as u32;
            let sum = s0 + s1 + s2 + s3;
            // count = 32, round bias = 16, shift = 5.
            let dc: u8 = ((sum + 16) >> 5) as u8;
            let row_v: __m128i = _mm_set1_epi8(dc as i8);
            for r in 0..16 {
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut __m128i, row_v);
            }
        }
        32 => {
            // 64 bytes total: above[0..32] is one __m256i, left[0..32]
            // is another. Two _mm256_sad_epu8 ops vs zero, then sum
            // the eight 16-bit partials.
            let v_above: __m256i = _mm256_loadu_si256(above.as_ptr() as *const __m256i);
            let v_left: __m256i = _mm256_loadu_si256(left.as_ptr() as *const __m256i);
            let zero = _mm256_setzero_si256();
            let sad_a: __m256i = _mm256_sad_epu8(v_above, zero);
            let sad_l: __m256i = _mm256_sad_epu8(v_left, zero);

            let a_lo: __m128i = _mm256_castsi256_si128(sad_a);
            let a_hi: __m128i = _mm256_extracti128_si256::<1>(sad_a);
            let l_lo: __m128i = _mm256_castsi256_si128(sad_l);
            let l_hi: __m128i = _mm256_extracti128_si256::<1>(sad_l);
            let s0 = _mm_extract_epi16::<0>(a_lo) as u32;
            let s1 = _mm_extract_epi16::<4>(a_lo) as u32;
            let s2 = _mm_extract_epi16::<0>(a_hi) as u32;
            let s3 = _mm_extract_epi16::<4>(a_hi) as u32;
            let s4 = _mm_extract_epi16::<0>(l_lo) as u32;
            let s5 = _mm_extract_epi16::<4>(l_lo) as u32;
            let s6 = _mm_extract_epi16::<0>(l_hi) as u32;
            let s7 = _mm_extract_epi16::<4>(l_hi) as u32;
            let sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
            // count = 64, round bias = 32, shift = 6.
            let dc: u8 = ((sum + 32) >> 6) as u8;
            let row_v: __m256i = _mm256_set1_epi8(dc as i8);
            for r in 0..32 {
                _mm256_storeu_si256(dst.as_mut_ptr().add(r * stride) as *mut __m256i, row_v);
            }
        }
        _ => unreachable!("dc_predict_full_avx2: bs must be in {{4, 8, 16, 32}}"),
    }
}

/// Safe wrapper for the AVX2 DC_PRED (full neighbours) predictor.
///
/// Signature is the flat `(dst, stride, bs, above, left)` form (cf.
/// [`crate::intra_pred::dc_predictor`] which takes a `PredCtx`). The
/// dispatch slot [`crate::dsp::SimdDispatch::dc_predict_full`] points
/// at this wrapper on AVX2 hosts and at the scalar adapter elsewhere.
/// `bs ∈ {4, 8, 16, 32}` for the AVX2 specialisations; `bs = 64`
/// (Block64x64) falls back to scalar — see the matching note in
/// [`crate::dsp::x86_64::dc_predict_left_avx2`].
#[cfg(target_arch = "x86_64")]
pub fn dc_predict_full_avx2(dst: &mut [u8], stride: usize, bs: usize, above: &[u8], left: &[u8]) {
    if bs == 64 {
        crate::dsp::validate_predictor_storage(dst.len(), stride, bs, above.len().min(left.len()))
            .unwrap_or_else(|reason| panic!("dc_predict_full_avx2: {reason}"));
        let ctx = crate::intra_pred::PredCtx {
            above,
            left,
            // dc_predictor (full-neighbours arm) does not read
            // `above_left`; supply any byte.
            above_left: 0,
        };
        crate::intra_pred::dc_predictor(dst, stride, bs, ctx);
        return;
    }
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len().min(left.len()))
        .unwrap_or_else(|reason| panic!("dc_predict_full_avx2: {reason}"));
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { dc_predict_full_avx2_inner(dst, stride, bs, above, left) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `dc_predict_full_avx2`.

    use super::dc_predict_full_avx2;
    use crate::intra_pred::{PredCtx, dc_predictor};

    #[test]
    #[should_panic(expected = "edge slice too short")]
    fn rejects_short_edge_before_avx2_inner() {
        let mut dst = [0u8; 16];
        let above = [0u8; 4];
        let left = [0u8; 3];
        dc_predict_full_avx2(&mut dst, 4, 4, &above, &left);
    }

    #[test]
    #[should_panic(expected = "unsupported block size")]
    fn rejects_invalid_block_size_before_avx2_inner() {
        dc_predict_full_avx2(&mut [], 4, 3, &[], &[]);
    }

    #[test]
    fn scalar_fallback_accepts_valid_64x64_layout() {
        let above = vec![41u8; 64];
        let left = vec![87u8; 64];
        let scalar = run_scalar(64, &above, &left);
        let mut actual = vec![0u8; 64 * 64];
        dc_predict_full_avx2(&mut actual, 64, 64, &above, &left);
        assert_eq!(actual, scalar);
    }

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG (Numerical Recipes constants).
    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn run_scalar(bs: usize, above: &[u8], left: &[u8]) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        let ctx = PredCtx {
            above,
            left,
            above_left: 0, // unused by DC_PRED
        };
        dc_predictor(&mut dst, bs, bs, ctx);
        dst
    }

    fn run_avx2(bs: usize, above: &[u8], left: &[u8]) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        dc_predict_full_avx2(&mut dst, bs, bs, above, left);
        dst
    }

    /// Differential fuzz: N=1024 LCG-seeded iterations per
    /// `bs ∈ {4, 8, 16, 32}`. Scalar and AVX2 must produce byte-equal
    /// output.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut state: u32 = 0xC0FF_EE44;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..1024 {
                let mut above = vec![0u8; bs];
                let mut left = vec![0u8; bs];
                for slot in above.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                for slot in left.iter_mut() {
                    *slot = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let scalar = run_scalar(bs, &above, &left);
                let avx2 = run_avx2(bs, &above, &left);
                assert_eq!(
                    scalar, avx2,
                    "AVX2/scalar mismatch at bs={bs} above={above:?} left={left:?}",
                );
            }
        }
    }

    /// Tautology guard — assert the AVX2 path is actually queried by
    /// `is_x86_feature_detected!("avx2")` on the test host.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// must produce identical output on a deterministic input.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        for bs in [4usize, 8, 16, 32] {
            let above: Vec<u8> = (0..bs).map(|i| (i as u32 * 7 + 11) as u8).collect();
            let left: Vec<u8> = (0..bs).map(|i| (i as u32 * 13 + 5) as u8).collect();
            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.dc_predict_full())(&mut dst_avx2, bs, bs, &above, &left);
            (dispatch_scalar.dc_predict_full())(&mut dst_scalar, bs, bs, &above, &left);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch at bs={bs}"
            );
        }
    }

    /// Directed all-zero / all-max / mid-grey / rounding-seam fixtures.
    #[test]
    fn directed_boundary_inputs() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            for &(a, l) in &[(0u8, 0u8), (255, 255), (128, 128), (0, 255), (255, 0)] {
                let above = vec![a; bs];
                let left = vec![l; bs];
                let scalar = run_scalar(bs, &above, &left);
                let avx2 = run_avx2(bs, &above, &left);
                assert_eq!(scalar, avx2, "boundary mismatch bs={bs} above={a} left={l}",);
            }
        }
    }
}
