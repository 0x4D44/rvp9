//! 32×32 inverse-DCT + add — EOB ≤ 135 sparse-block AVX2 fast path
//! (Wave F F9).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]) and the Wave E G4-α full
//! 32×32 rollout ([`crate::dsp::x86_64::idct32x32_avx2`]). The scalar
//! reference is [`crate::transforms::inverse::idct32x32_135_add`].
//!
//! Only **Stage 1** (the row pass) differs from the full
//! `idct32x32_add_avx2` finisher: this variant runs `idct32` over the
//! first 16 input rows only. Stage 2 (column pass) and Stage 3 (AVX2
//! finisher) are identical bytes — the per-row zero-skip from the full
//! kernel is replaced by a static row count of 16 because the caller
//! contract guarantees rows 16..32 are zero.
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same approach as every other AVX2 idct kernel: the i32 1D transform
//! math is identical to the scalar reference (1:1 reuse of
//! [`crate::transforms::inverse::idct32`]) and AVX2 accelerates only
//! the final `(out + 32) >> 6` + add-clamp-pack stage. The saturating
//! `_mm256_packs_epi32` in Stage 3 is fine because the saturation
//! point is the FINAL `(out + 32) >> 6` + clip — identical to
//! `idct32x32_avx2.rs`. No `wraplow` concerns because the butterfly
//! math is scalar.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadu_si128` (16 bytes per
//! half-row; two halves per row); stack scratch `final_out` is read via
//! `_mm256_loadu_si256`. The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadu_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_loadu_si256, _mm256_packs_epi32, _mm256_permute4x64_epi64,
    _mm256_set1_epi32, _mm256_srai_epi32,
};

use crate::transforms::inverse::idct32;

/// AVX2 implementation of the 32×32 inverse-DCT + residual add for the
/// `eob <= 135` sparse-block fast path.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct32x32_135_add_avx2`]) when `caps.avx2 == true`,
/// and caps are detected once at OnceLock init time.
///
/// `dest` must have at least `31 * stride + 32` bytes of valid storage.
///
/// **Caller contract** (mirrors libvpx): if used on a block whose true
/// eob exceeds 135, coefficients at indices `>=` row 16 are silently
/// ignored and the output diverges from the full kernel. Callers are
/// responsible for the eob bucket gate.
///
/// # Implementation
///
/// Row + column idct32 math uses the scalar reference (i32, bit-exact
/// `wraplow`). The row pass runs for the first 16 rows only — rows
/// 16..32 of `intermediate` stay at the zero initialiser per the
/// caller contract. AVX2 accelerates the final `(out + 32) >> 6` +
/// add + clip stage exactly as in
/// [`crate::dsp::x86_64::idct32x32_avx2`].
///
/// All loads / stores from / to `dest` are 16-byte (`__m128i`),
/// unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct32x32_135_add_avx2_inner(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 31 + 32, "dest too small for 32x32 add");

    // ----- Stage 1: row pass (scalar idct32, bit-exact) — first 16
    // rows only per the EOB ≤ 135 contract. Rows 16..32 of
    // `intermediate` remain at the zero initialiser, exactly as if
    // `idct32` had been applied to a zero input.
    let mut intermediate = [0i32; 1024];
    for r in 0..16 {
        let row_in: &[i32; 32] = input[r * 32..r * 32 + 32].try_into().unwrap();
        let row_out_mut: &mut [i32; 32] = (&mut intermediate[r * 32..r * 32 + 32])
            .try_into()
            .unwrap();
        idct32(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar idct32). Identical to the
    // full `idct32x32_avx2` finisher — all 32 columns.
    let mut final_out = [0i32; 1024];
    let mut col_in = [0i32; 32];
    let mut col_out = [0i32; 32];
    for c in 0..32 {
        for r in 0..32 {
            col_in[r] = intermediate[r * 32 + c];
        }
        idct32(&col_in, &mut col_out);
        for r in 0..32 {
            final_out[r * 32 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store.
    // Identical to the full `idct32x32_avx2` finisher — see that file
    // for the per-step rationale.
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..32 {
        let p0 = final_out.as_ptr().add(r * 32) as *const __m256i;
        let p1 = final_out.as_ptr().add(r * 32 + 8) as *const __m256i;
        let p2 = final_out.as_ptr().add(r * 32 + 16) as *const __m256i;
        let p3 = final_out.as_ptr().add(r * 32 + 24) as *const __m256i;
        let mut v0 = _mm256_loadu_si256(p0);
        let mut v1 = _mm256_loadu_si256(p1);
        let mut v2 = _mm256_loadu_si256(p2);
        let mut v3 = _mm256_loadu_si256(p3);
        v0 = _mm256_add_epi32(v0, round);
        v1 = _mm256_add_epi32(v1, round);
        v2 = _mm256_add_epi32(v2, round);
        v3 = _mm256_add_epi32(v3, round);
        v0 = _mm256_srai_epi32(v0, 6);
        v1 = _mm256_srai_epi32(v1, 6);
        v2 = _mm256_srai_epi32(v2, 6);
        v3 = _mm256_srai_epi32(v3, 6);

        let p0_packed = _mm256_packs_epi32(v0, v0);
        let p1_packed = _mm256_packs_epi32(v1, v1);
        let p2_packed = _mm256_packs_epi32(v2, v2);
        let p3_packed = _mm256_packs_epi32(v3, v3);
        let n0 = _mm256_permute4x64_epi64(p0_packed, 0xD8);
        let n1 = _mm256_permute4x64_epi64(p1_packed, 0xD8);
        let n2 = _mm256_permute4x64_epi64(p2_packed, 0xD8);
        let n3 = _mm256_permute4x64_epi64(p3_packed, 0xD8);
        let out_i16_a: __m128i = _mm256_castsi256_si128(n0);
        let out_i16_b: __m128i = _mm256_castsi256_si128(n1);
        let out_i16_c: __m128i = _mm256_castsi256_si128(n2);
        let out_i16_d: __m128i = _mm256_castsi256_si128(n3);

        // G4: loads from `dst` use _mm_loadu_si128 (unaligned)
        let dst_row_ptr_lo = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_row_ptr_hi = dest.as_mut_ptr().add(r * stride + 16) as *const __m128i;
        let dst_u8_lo = _mm_loadu_si128(dst_row_ptr_lo);
        let dst_u8_hi = _mm_loadu_si128(dst_row_ptr_hi);
        let dst_i16_a = _mm_unpacklo_epi8(dst_u8_lo, zero);
        let dst_i16_b = _mm_unpackhi_epi8(dst_u8_lo, zero);
        let dst_i16_c = _mm_unpacklo_epi8(dst_u8_hi, zero);
        let dst_i16_d = _mm_unpackhi_epi8(dst_u8_hi, zero);

        let sum_a = _mm_add_epi16(out_i16_a, dst_i16_a);
        let sum_b = _mm_add_epi16(out_i16_b, dst_i16_b);
        let sum_c = _mm_add_epi16(out_i16_c, dst_i16_c);
        let sum_d = _mm_add_epi16(out_i16_d, dst_i16_d);
        let sum_u8_lo = _mm_packus_epi16(sum_a, sum_b);
        let sum_u8_hi = _mm_packus_epi16(sum_c, sum_d);

        let dst_row_ptr_mut_lo = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        let dst_row_ptr_mut_hi = dest.as_mut_ptr().add(r * stride + 16) as *mut __m128i;
        _mm_storeu_si128(dst_row_ptr_mut_lo, sum_u8_lo);
        _mm_storeu_si128(dst_row_ptr_mut_hi, sum_u8_hi);
    }
}

/// Safe wrapper for the AVX2 `idct32x32_135_add` inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct32x32_135_add`], so the
/// `SimdDispatch::idct32x32_135_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_135_add_avx2(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct32x32_135_avx2::tests` gates each
    // call on a runtime `is_x86_feature_detected!("avx2")` check, so
    // calls from tests are equally sound.
    // The caller-side precondition is `dest.len() >= 31 * stride + 32`
    // — debug-asserted at the top of `idct32x32_135_add_avx2_inner`.
    unsafe { idct32x32_135_add_avx2_inner(input, dest, stride) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct32x32_135_add`.
    //!
    //! - **`idct32x32_135_scalar_eq_avx2_proptest`** — 128-case
    //!   proptest over inputs whose non-zero coefficients are confined
    //!   to the top-16 rows (the kernel's contract domain).
    //! - **`idct32x32_135_directed_per_row_count`** — directed sweep
    //!   over `K ∈ 1..=16` row counts, locking per-row arithmetic.
    //! - **`idct32x32_135_dispatch_tautology_guard`** — under forced
    //!   AVX2 caps the dispatch slot must NOT be the scalar fn
    //!   pointer (otherwise differential tests are vacuous).
    //!
    //! Every test gates on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::idct32x32_135_add_avx2;
    use crate::transforms::inverse::idct32x32_135_add;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_135_add_avx2(input, &mut dst, 32);
        dst
    }

    fn run_scalar(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_135_add(input, &mut dst, 32);
        dst
    }

    /// Strategy: inputs with non-zero coefficients confined to rows
    /// 0..16 (the kernel's contract domain). Rows 16..32 are forced
    /// to zero; rows 0..15 are dense i16-range.
    fn input_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-32768i32..=32767, 16 * 32).prop_map(|v| {
            let mut out = [0i32; 1024];
            out[..16 * 32].copy_from_slice(&v);
            out
        })
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 1024]> {
        prop::collection::vec(any::<u8>(), 1024).prop_map(|v| {
            let mut out = [0u8; 1024];
            out.copy_from_slice(&v);
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            max_shrink_iters: 512,
            .. ProptestConfig::default()
        })]

        #[test]
        fn idct32x32_135_scalar_eq_avx2_proptest(
            input in input_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let scalar = run_scalar(&input, &dst_init);
            let avx2 = run_avx2(&input, &dst_init);
            prop_assert_eq!(scalar, avx2);
        }
    }

    /// Directed per-row-count sweep. For each row count `K ∈ 1..=16`,
    /// fill rows 0..K with a deterministic pattern and compare scalar
    /// vs AVX2 outputs.
    #[test]
    fn idct32x32_135_directed_per_row_count() {
        require_avx2!();
        for k in 1..=16usize {
            let mut input = [0i32; 1024];
            for r in 0..k {
                for c in 0..32 {
                    input[r * 32 + c] = ((r as i32) * 199 + (c as i32) * 47) - 3000;
                }
            }
            let dst_init = [128u8; 1024];
            let scalar = run_scalar(&input, &dst_init);
            let avx2 = run_avx2(&input, &dst_init);
            assert_eq!(scalar, avx2, "AVX2 mismatch at k={k} populated rows");
        }
    }

    /// Tautology guard + dispatch round-trip: forced-AVX2 dispatch
    /// installs the AVX2 wrapper (NOT the scalar reference), and
    /// forced-scalar matches it for an input in the contract domain.
    #[test]
    fn idct32x32_135_dispatch_tautology_guard() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let scalar_fp: fn(&[i32; 1024], &mut [u8], usize) = idct32x32_135_add;
        assert_ne!(
            dispatch_avx2.idct32x32_135_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 1024];
        for r in 0..16 {
            for c in 0..32 {
                input[r * 32 + c] = ((r as i32) * 31 + (c as i32) * 17) % 4001 - 2000;
            }
        }
        let dst_init = [97u8; 1024];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct32x32_135_add())(&input, &mut dst_avx2, 32);
        (dispatch_scalar.idct32x32_135_add())(&input, &mut dst_scalar, 32);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
