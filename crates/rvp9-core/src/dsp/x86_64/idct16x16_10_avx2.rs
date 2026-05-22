//! 16×16 inverse-DCT + add — EOB ≤ 10 sparse-block fast path, x86_64
//! AVX2 implementation (Wave F F3).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]) and the G4.2 full-density
//! 16×16 kernel ([`crate::dsp::x86_64::idct16x16_avx2`]). The scalar
//! reference is [`crate::transforms::inverse::idct16x16_10_add`].
//!
//! ## Caller contract
//!
//! Identical to the scalar reference: when `eob <= 10` the non-zero
//! coefficients are confined to the top-left 4×4 sub-block of the input
//! (raster order), so the row 1D pass only runs on the first 4 input
//! rows and rows 4..16 of the input are assumed zero. Passing a block
//! whose true eob exceeds 10 silently drops coefficients in rows 4..16
//! and diverges from the full kernel — callers are responsible for the
//! eob bucket gate.
//!
//! ## Bit-exactness vs scalar reference
//!
//! Finisher-only port: the row pass (Stage 1) runs scalar `idct16` on
//! rows 0..4 with the intermediate buffer zero-initialised so rows
//! 4..16 produce all-zero column-pass inputs (matching scalar
//! `idct16x16_10_add` semantics). Stage 2 (column pass) is the full
//! scalar 16×idct16. Stage 3 (round + add + clip + pack) reuses the
//! AVX2 finisher from [`crate::dsp::x86_64::idct16x16_avx2`] unchanged.
//!
//! Since this is finisher-only (only Stage 1 row count reduced from 16
//! → 4) the saturating `_mm256_packs_epi32` in the finisher is correct
//! — it's the documented final `(out + 32) >> 6` + clip pack stage,
//! identical to the existing `idct16x16_avx2` kernel and bit-exact to
//! the scalar `(col_out + 32) >> 6` followed by `clip_pixel_add`.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadu_si128` (16 bytes per
//! row); stack scratch `final_out` is read via `_mm256_loadu_si256`.
//! The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadu_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_loadu_si256, _mm256_packs_epi32, _mm256_permute4x64_epi64,
    _mm256_set1_epi32, _mm256_srai_epi32,
};

use crate::transforms::inverse::idct16;

/// AVX2 implementation of the 16×16 inverse-DCT + residual add for the
/// `eob <= 10` sparse-block fast path.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct16x16_10_add_avx2`]) when `caps.avx2 == true`,
/// and caps are detected once at OnceLock init time.
///
/// `dest` must have at least `15 * stride + 16` bytes of valid storage.
///
/// # Implementation
///
/// Row pass runs scalar `idct16` on rows 0..4 only; the intermediate
/// buffer is zero-initialised so rows 4..16 feed zero through the
/// column pass (matching the scalar `idct16x16_10_add` semantics).
/// Column pass (Stage 2) and the AVX2 finisher (Stage 3) are identical
/// to [`crate::dsp::x86_64::idct16x16_avx2`].
///
/// All loads / stores from / to `dest` are 16-byte (`__m128i`),
/// unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct16x16_10_add_avx2_inner(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    // ----- Stage 1: row pass (scalar idct16, bit-exact) — rows 0..4 only.
    // The intermediate buffer MUST be zero-initialised so rows 4..16 feed
    // all-zero inputs through the column pass (caller-contract: input
    // rows 4..16 are assumed zero when eob <= 10).
    let mut intermediate = [0i32; 256];
    for r in 0..4 {
        let row_in: &[i32; 16] = input[r * 16..r * 16 + 16].try_into().unwrap();
        let row_out_mut: &mut [i32; 16] = (&mut intermediate[r * 16..r * 16 + 16])
            .try_into()
            .unwrap();
        idct16(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar idct16). -----
    // Produces 16 columns of 16 i32 outputs. Stored row-major into
    // `final_out[r * 16 + c]` so the AVX2 finisher can `loadu` two
    // halves per row.
    let mut final_out = [0i32; 256];
    let mut col_in = [0i32; 16];
    let mut col_out = [0i32; 16];
    for c in 0..16 {
        for r in 0..16 {
            col_in[r] = intermediate[r * 16 + c];
        }
        idct16(&col_in, &mut col_out);
        for r in 0..16 {
            final_out[r * 16 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // For each of the 16 rows:
    //   1. Load 16 i32 outputs (= 2 ymm loads of 8 i32 each) from
    //      `final_out` stack scratch.
    //   2. Add round +32, shift right arithmetic by 6.
    //   3. Saturate-pack i32 → i16. After permute(0xD8) the low 128 bits
    //      of each ymm hold 8 i16 lanes in natural order. Concatenate
    //      the two halves into one 16-lane register pair (low half +
    //      high half).
    //   4. Load 16 u8 dst bytes (one row), zero-extend to two i16 halves
    //      (lo: cols 0..8, hi: cols 8..16).
    //   5. Add transform output and saturate-pack i16 → u8.
    //   6. Store 16 u8 back.
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..16 {
        // Two ymm loads cover the 16 i32 outputs for this row.
        let row_lo_ptr = final_out.as_ptr().add(r * 16) as *const __m256i;
        let row_hi_ptr = final_out.as_ptr().add(r * 16 + 8) as *const __m256i;
        let mut lo_i32 = _mm256_loadu_si256(row_lo_ptr);
        let mut hi_i32 = _mm256_loadu_si256(row_hi_ptr);
        lo_i32 = _mm256_add_epi32(lo_i32, round);
        hi_i32 = _mm256_add_epi32(hi_i32, round);
        lo_i32 = _mm256_srai_epi32(lo_i32, 6);
        hi_i32 = _mm256_srai_epi32(hi_i32, 6);

        // Saturate-pack i32 → i16 with per-128-bit-lane behaviour, then
        // permute by 64-bit qwords (control 0xD8 = [0, 2, 1, 3]) so the
        // low 128 bits of each ymm hold 8 i16 in natural order. See
        // idct8x8_avx2.rs Stage 3 for the rationale.
        let lo_packed = _mm256_packs_epi32(lo_i32, lo_i32);
        let hi_packed = _mm256_packs_epi32(hi_i32, hi_i32);
        let lo_natural = _mm256_permute4x64_epi64(lo_packed, 0xD8);
        let hi_natural = _mm256_permute4x64_epi64(hi_packed, 0xD8);
        // 8 i16 lanes each (lo: cols 0..8, hi: cols 8..16).
        let out_i16_lo: __m128i = _mm256_castsi256_si128(lo_natural);
        let out_i16_hi: __m128i = _mm256_castsi256_si128(hi_natural);

        // (4) load 16 u8 from dst row (unaligned).
        // G4: loads from `dst` use _mm_loadu_si128 (unaligned)
        let dst_row_ptr = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadu_si128(dst_row_ptr);
        // Zero-extend u8 → i16. `unpacklo` takes bytes 0..8; `unpackhi`
        // takes bytes 8..16 — paired with `zero` so each byte becomes
        // a 16-bit zero-extended lane.
        let dst_i16_lo = _mm_unpacklo_epi8(dst_u8, zero);
        let dst_i16_hi = _mm_unpackhi_epi8(dst_u8, zero);

        // (5) add transform output and saturate-pack to u8.
        let sum_lo = _mm_add_epi16(out_i16_lo, dst_i16_lo);
        let sum_hi = _mm_add_epi16(out_i16_hi, dst_i16_hi);
        // `_mm_packus_epi16(a, b)` produces [a0..a7, b0..b7] saturated
        // to u8 — exactly the 16-byte row we want.
        let sum_u8 = _mm_packus_epi16(sum_lo, sum_hi);

        // (6) store 16 u8 back (unaligned).
        let dst_row_ptr_mut = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storeu_si128(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the AVX2 idct16x16_10 (EOB ≤ 10) inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct16x16_10_add`], so the
/// `SimdDispatch::idct16x16_10_add` slot can hold either.
///
/// **Caller contract:** same as the scalar reference — passing a block
/// whose true eob exceeds 10 silently diverges from the full kernel.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct16x16_10_add_avx2(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct16x16_10_avx2::tests` gates each call
    // on a runtime `is_x86_feature_detected!("avx2")` check, so calls
    // from tests are equally sound.
    unsafe { idct16x16_10_add_avx2_inner(input, dest, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct16x16_10_add`.
    //!
    //! Mirrors the G4.2 idct16x16 full-kernel test suite, but every
    //! `[i32; 256]` proptest input zeroes rows 4..16 (caller contract:
    //! the fast path silently ignores any non-zero data beyond row 4).
    //!
    //! Coverage:
    //!
    //! - **`idct16x16_10_scalar_eq_avx2_proptest`** — 256-case proptest
    //!   with uniform / sparse / high-magnitude / pathological strategies
    //!   for the top-4-row coefficients.
    //! - **`idct16x16_10_all_zero_input`** — zero coefficients leave dst
    //!   unchanged.
    //! - **`idct16x16_10_dc_only`** — single DC coefficient at position 0.
    //! - **`idct16x16_10_max_input`** — saturation stress with i16::MAX /
    //!   i16::MIN coefficients in rows 0..4 (rows 4..16 zero).
    //! - **`idct16x16_10_dispatch_round_trip`** — tautology guard +
    //!   forced-AVX2 vs forced-scalar dispatch produce identical output.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::idct16x16_10_add_avx2;
    use crate::transforms::inverse::idct16x16_10_add;
    use proptest::prelude::*;

    /// Top-K-row coefficient count for this fast path. `idct16x16_10`
    /// is the EOB ≤ 10 bucket → top-left 4×4 sub-block, which spans
    /// the first 4 rows of the raster-order input array. Rows K..16
    /// MUST be zero in every proptest input (the fast path silently
    /// ignores them and would diverge from the scalar reference if
    /// non-zero data is supplied there).
    const TOP_ROWS: usize = 4;
    const TOP_COEFFS: usize = TOP_ROWS * 16;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct16x16_10_add_avx2(input, &mut dst, 16);
        dst
    }

    fn run_scalar(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        let mut dst = *dst_init;
        idct16x16_10_add(input, &mut dst, 16);
        dst
    }

    fn assert_match(input: &[i32; 256], dst_init: &[u8; 256]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch (first divergence): scalar != avx2"
        );
    }

    // -----------------------------------------------------------------
    // Strategies — every `[i32; 256]` MUST zero rows TOP_ROWS..16
    // (caller contract). Coefficient data is generated for the first
    // TOP_COEFFS positions only.
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    /// Uniform random coefficients in the top-K rows; rows K..16 zero.
    fn uniform_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(uniform_coeff(), TOP_COEFFS).prop_map(|v| {
            let mut out = [0i32; 256];
            out[..TOP_COEFFS].copy_from_slice(&v);
            out
        })
    }

    /// 1..=8 sparse non-zero coefficients placed within the top-K rows.
    fn sparse_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec((0usize..TOP_COEFFS, uniform_coeff()), 1..=8).prop_map(|nz| {
            let mut out = [0i32; 256];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// High-magnitude coefficients (|c| ≥ 30000) in the top-K rows.
    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            TOP_COEFFS,
        )
        .prop_map(|v| {
            let mut out = [0i32; 256];
            out[..TOP_COEFFS].copy_from_slice(&v);
            out
        })
    }

    /// Pathological fixtures with `[i16::MAX]` / `[i16::MIN]` in the
    /// top-K rows only (rows K..16 zero per the caller contract).
    fn pathological_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![
            Just({
                let mut out = [0i32; 256];
                for slot in out[..TOP_COEFFS].iter_mut() {
                    *slot = i16::MAX as i32;
                }
                out
            }),
            Just({
                let mut out = [0i32; 256];
                for slot in out[..TOP_COEFFS].iter_mut() {
                    *slot = i16::MIN as i32;
                }
                out
            }),
            Just({
                let mut out = [0i32; 256];
                for (i, slot) in out[..TOP_COEFFS].iter_mut().enumerate() {
                    *slot = if i % 2 == 0 {
                        i16::MAX as i32
                    } else {
                        i16::MIN as i32
                    };
                }
                out
            }),
        ]
    }

    fn input_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            high_magnitude_strategy(),
            pathological_strategy(),
        ]
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 256]> {
        prop::collection::vec(any::<u8>(), 256).prop_map(|v| {
            let mut out = [0u8; 256];
            out.copy_from_slice(&v);
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn idct16x16_10_scalar_eq_avx2_proptest(
            input in input_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            // Sanity guard: every input strategy must zero rows K..16.
            for i in TOP_COEFFS..256 {
                prop_assert_eq!(input[i], 0, "test bug: row {} must be zero", i / 16);
            }
            let scalar = run_scalar(&input, &dst_init);
            let avx2 = run_avx2(&input, &dst_init);
            prop_assert_eq!(scalar, avx2);
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures.
    // -----------------------------------------------------------------

    /// All-zero coefficients leave dst unchanged.
    #[test]
    fn idct16x16_10_all_zero_input() {
        require_avx2!();
        let input = [0i32; 256];
        let dst_init = [42u8; 256];
        assert_match(&input, &dst_init);
    }

    /// DC-only: single coefficient at position 0.
    #[test]
    fn idct16x16_10_dc_only() {
        require_avx2!();
        let mut input = [0i32; 256];
        input[0] = 1234;
        let dst_init = [50u8; 256];
        assert_match(&input, &dst_init);
    }

    /// Directed pathological: `[i16::MAX; 4*16]` in rows 0..4, zero in
    /// rows 4..16. Density-stress at the K=10 fast-path's row limit.
    #[test]
    fn idct16x16_10_pathological_max_top_rows() {
        require_avx2!();
        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MAX as i32;
        }
        let dst_init = [0u8; 256];
        assert_match(&input, &dst_init);

        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MIN as i32;
        }
        let dst_init = [255u8; 256];
        assert_match(&input, &dst_init);
    }

    /// Tautology guard: forced-AVX2 dispatch slot must NOT be the
    /// scalar fn pointer, and the AVX2 wrapper must match the scalar
    /// reference on a non-trivial input.
    #[test]
    fn idct16x16_10_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        // Tautology guard: the AVX2 dispatch slot must NOT collapse to
        // the scalar reference. If they were equal the differential
        // below would be vacuously true.
        let scalar_fp: fn(&[i32; 256], &mut [u8], usize) = idct16x16_10_add;
        assert_ne!(
            dispatch_avx2.idct16x16_10_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 256];
        for (i, slot) in input[..TOP_COEFFS].iter_mut().enumerate() {
            *slot = ((i as i32) * 29) % 4001 - 2000;
        }
        let dst_init = [97u8; 256];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct16x16_10_add())(&input, &mut dst_avx2, 16);
        (dispatch_scalar.idct16x16_10_add())(&input, &mut dst_scalar, 16);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
