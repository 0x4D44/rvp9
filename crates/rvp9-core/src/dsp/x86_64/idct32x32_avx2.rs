//! 32×32 inverse-DCT + add — x86_64 AVX2 implementation (G4-α full
//! kernel rollout).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]) and the G4.2 16×16 rollout
//! ([`crate::dsp::x86_64::idct16x16_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::idct32x32_1024_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same approach as the other AVX2 idct kernels: the i32 1D transform
//! math is identical to the scalar reference (1:1 reuse of
//! [`crate::transforms::inverse::idct32`]) and AVX2 accelerates only
//! the final `(out + 32) >> 6` + add-clamp-pack stage. Each row of the
//! 32×32 block holds 32 i32 outputs (= 4 ymm lanes of 8 i32 each); the
//! finisher loads all four halves, applies round + shift + saturating
//! pack to i16, then loads the 32 dst u8 bytes (one row = two 16-byte
//! lanes), zero-extends to i16 halves, adds, and saturating-packs back
//! to u8.
//!
//! The per-row zero-skip optimisation from the scalar reference is
//! preserved here verbatim: if a row of the input is all-zero, the
//! row-pass call to `idct32` is skipped and the `intermediate` slot is
//! left at its zero initialiser.
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

/// AVX2 implementation of the 32×32 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct32x32_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `31 * stride + 32` bytes of valid storage.
///
/// # Implementation
///
/// Row + column idct32 math uses the scalar reference (i32, bit-exact
/// `wraplow`). AVX2 accelerates the final `(out + 32) >> 6` + add +
/// clip stage. Each of the 32 rows produces 32 i32 outputs → 4 ymm
/// loads, round-shift, pack to i16 (= 2 xmm of 16 i16 in natural
/// order), then add 32 zero-extended u8 dst bytes (two 16-byte halves)
/// and saturate-pack back.
///
/// All loads / stores from / to `dest` are 16-byte (`__m128i`),
/// unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct32x32_add_avx2_inner(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 31 + 32, "dest too small for 32x32 add");

    // ----- Stage 1: row pass (scalar idct32, bit-exact), with the
    // libvpx zero_coeff fast-skip preserved verbatim. -----
    let mut intermediate = [0i32; 1024];
    for r in 0..32 {
        let row_in: &[i32; 32] = input[r * 32..r * 32 + 32].try_into().unwrap();
        // OR-reduce to detect all-zero row (matches scalar reference).
        let any_nonzero = row_in.iter().any(|&v| v != 0);
        if any_nonzero {
            let row_out_mut: &mut [i32; 32] = (&mut intermediate[r * 32..r * 32 + 32])
                .try_into()
                .unwrap();
            idct32(row_in, row_out_mut);
        }
        // else: row stays at its zero initialiser.
    }

    // ----- Stage 2: column pass (scalar idct32). -----
    // Produces 32 columns of 32 i32 outputs. Stored row-major into
    // `final_out[r * 32 + c]` so the AVX2 finisher can `loadu` four
    // halves per row.
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

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // For each of the 32 rows:
    //   1. Load 32 i32 outputs (= 4 ymm loads of 8 i32 each) from
    //      `final_out` stack scratch.
    //   2. Add round +32, shift right arithmetic by 6.
    //   3. Saturate-pack i32 → i16. After permute(0xD8) the low 128 bits
    //      of each ymm hold 8 i16 lanes in natural order. We get four
    //      groups of 8 i16, which combine into two 16-lane xmm registers
    //      (one for cols 0..16, one for cols 16..32).
    //   4. Load 32 u8 dst bytes (one row = two _mm_loadu_si128), zero-
    //      extend to four i16 halves.
    //   5. Add transform output and saturate-pack i16 → u8 (two
    //      _mm_packus_epi16 calls produce the two 16-byte halves).
    //   6. Store 32 u8 back (two _mm_storeu_si128).
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..32 {
        // Four ymm loads cover the 32 i32 outputs for this row.
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

        // Saturate-pack i32 → i16 with per-128-bit-lane behaviour, then
        // permute by 64-bit qwords (control 0xD8 = [0, 2, 1, 3]) so the
        // low 128 bits of each ymm hold 8 i16 in natural order. See
        // idct8x8_avx2.rs Stage 3 for the rationale.
        let p0_packed = _mm256_packs_epi32(v0, v0);
        let p1_packed = _mm256_packs_epi32(v1, v1);
        let p2_packed = _mm256_packs_epi32(v2, v2);
        let p3_packed = _mm256_packs_epi32(v3, v3);
        let n0 = _mm256_permute4x64_epi64(p0_packed, 0xD8);
        let n1 = _mm256_permute4x64_epi64(p1_packed, 0xD8);
        let n2 = _mm256_permute4x64_epi64(p2_packed, 0xD8);
        let n3 = _mm256_permute4x64_epi64(p3_packed, 0xD8);
        // 8 i16 lanes each — group 0..8, 8..16, 16..24, 24..32.
        let out_i16_a: __m128i = _mm256_castsi256_si128(n0);
        let out_i16_b: __m128i = _mm256_castsi256_si128(n1);
        let out_i16_c: __m128i = _mm256_castsi256_si128(n2);
        let out_i16_d: __m128i = _mm256_castsi256_si128(n3);

        // (4) load 32 u8 from dst row (unaligned, two 16-byte halves).
        // G4: loads from `dst` use _mm_loadu_si128 (unaligned)
        let dst_row_ptr_lo = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_row_ptr_hi = dest.as_mut_ptr().add(r * stride + 16) as *const __m128i;
        let dst_u8_lo = _mm_loadu_si128(dst_row_ptr_lo);
        let dst_u8_hi = _mm_loadu_si128(dst_row_ptr_hi);
        // Zero-extend u8 → i16. `unpacklo` takes bytes 0..8; `unpackhi`
        // takes bytes 8..16 — paired with `zero` so each byte becomes
        // a 16-bit zero-extended lane.
        let dst_i16_a = _mm_unpacklo_epi8(dst_u8_lo, zero); // cols 0..8
        let dst_i16_b = _mm_unpackhi_epi8(dst_u8_lo, zero); // cols 8..16
        let dst_i16_c = _mm_unpacklo_epi8(dst_u8_hi, zero); // cols 16..24
        let dst_i16_d = _mm_unpackhi_epi8(dst_u8_hi, zero); // cols 24..32

        // (5) add transform output and saturate-pack to u8.
        let sum_a = _mm_add_epi16(out_i16_a, dst_i16_a);
        let sum_b = _mm_add_epi16(out_i16_b, dst_i16_b);
        let sum_c = _mm_add_epi16(out_i16_c, dst_i16_c);
        let sum_d = _mm_add_epi16(out_i16_d, dst_i16_d);
        // `_mm_packus_epi16(a, b)` produces [a0..a7, b0..b7] saturated
        // to u8 — exactly the 16-byte half-row we want.
        let sum_u8_lo = _mm_packus_epi16(sum_a, sum_b);
        let sum_u8_hi = _mm_packus_epi16(sum_c, sum_d);

        // (6) store 32 u8 back (unaligned, two 16-byte halves).
        let dst_row_ptr_mut_lo = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        let dst_row_ptr_mut_hi = dest.as_mut_ptr().add(r * stride + 16) as *mut __m128i;
        _mm_storeu_si128(dst_row_ptr_mut_lo, sum_u8_lo);
        _mm_storeu_si128(dst_row_ptr_mut_hi, sum_u8_hi);
    }
}

/// Safe wrapper for the AVX2 idct32x32 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct32x32_1024_add`], so the
/// `SimdDispatch::idct32x32_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_avx2(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct32x32_avx2::tests` gates each call on
    // a runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    // The caller-side precondition is `dest.len() >= 31 * stride + 32`
    // — debug-asserted at the top of `idct32x32_add_avx2_inner`.
    unsafe { idct32x32_add_avx2_inner(input, dest, stride) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct32x32_add`.
    //!
    //! Mirrors the G4.1 idct8x8 / G4.2 idct4x4 / G4.2 idct16x16 test
    //! suites. Coverage:
    //!
    //! - **`idct32x32_scalar_eq_avx2_proptest`** — 256-case proptest
    //!   with five stratified strategies (uniform, sparse, eob-bucket,
    //!   high-magnitude, pathological).
    //! - **`idct32x32_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (transform output is the zero matrix, +32 >> 6 = 0).
    //! - **`idct32x32_dc_only`** — single DC coefficient at position 0;
    //!   AVX2 matches scalar bit-for-bit.
    //! - **`idct32x32_single_coefficient_impulses`** — sweep one
    //!   non-zero coefficient across every position 0..1024 with several
    //!   magnitudes.
    //! - **`idct32x32_max_input`** — saturation stress with i16::MAX /
    //!   i16::MIN coefficients.
    //! - **`idct32x32_misaligned_dst_pointers`** — per-skew arithmetic
    //!   regression guard (kernel uses `_mm_loadu_si128`, unaligned).
    //! - **`idct32x32_dispatch_round_trip`** — forced-AVX2 vs
    //!   forced-scalar dispatch produce identical output.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::idct32x32_add_avx2;
    use crate::transforms::inverse::idct32x32_1024_add;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct32x32_add_avx2(input, &mut dst, 32);
        dst
    }

    fn run_scalar(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_1024_add(input, &mut dst, 32);
        dst
    }

    fn assert_match(input: &[i32; 1024], dst_init: &[u8; 1024]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch (first divergence): scalar != avx2"
        );
    }

    // -----------------------------------------------------------------
    // Strategies (same shape as idct16x16, sized for 1024 coeffs).
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(uniform_coeff(), 1024).prop_map(|v| {
            let mut out = [0i32; 1024];
            out.copy_from_slice(&v);
            out
        })
    }

    fn sparse_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec((0usize..1024, uniform_coeff()), 1..=16).prop_map(|nz| {
            let mut out = [0i32; 1024];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// K ∈ {1, 34, 135, 1024} non-zero coefficients in raster positions
    /// `0..K`. K=1 is the DC-only seam; K=34 / K=135 straddle libvpx's
    /// `idct32x32_34_add` / `_135_add` dispatcher boundaries (not yet
    /// ported but kept here for forward compatibility); K=1024 is full
    /// density.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop_oneof![Just(1usize), Just(34), Just(135), Just(1024)].prop_flat_map(|k| {
            prop::collection::vec(uniform_coeff(), k).prop_map(move |values| {
                let mut out = [0i32; 1024];
                for (i, val) in values.into_iter().enumerate() {
                    out[i] = val;
                }
                out
            })
        })
    }

    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            1024,
        )
        .prop_map(|v| {
            let mut out = [0i32; 1024];
            out.copy_from_slice(&v);
            out
        })
    }

    fn pathological_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop_oneof![
            (-32768i32..=32767).prop_map(|v| [v; 1024]),
            Just([i16::MIN as i32; 1024]),
            Just([i16::MAX as i32; 1024]),
            Just({
                let mut out = [0i32; 1024];
                for (i, slot) in out.iter_mut().enumerate() {
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

    fn input_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            eob_bucket_strategy(),
            high_magnitude_strategy(),
            pathological_strategy(),
        ]
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
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn idct32x32_scalar_eq_avx2_proptest(
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

    // -----------------------------------------------------------------
    // Directed fixtures.
    // -----------------------------------------------------------------

    /// All-zero coefficients leave dst unchanged.
    #[test]
    fn idct32x32_all_zero_input() {
        require_avx2!();
        let input = [0i32; 1024];
        let dst_init = [42u8; 1024];
        assert_match(&input, &dst_init);
    }

    /// DC-only: single coefficient at position 0.
    #[test]
    fn idct32x32_dc_only() {
        require_avx2!();
        let mut input = [0i32; 1024];
        input[0] = 1234;
        let dst_init = [50u8; 1024];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep across all 1024 positions.
    /// 1024 positions × 6 magnitudes = 6144 cases.
    #[test]
    fn idct32x32_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..1024 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 1024];
                input[pos] = val;
                let dst_init = [128u8; 1024];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Saturation stress: i16 extremes throughout.
    #[test]
    fn idct32x32_max_input() {
        require_avx2!();
        let input = [i16::MAX as i32; 1024];
        let dst_init = [0u8; 1024];
        assert_match(&input, &dst_init);

        let input = [i16::MIN as i32; 1024];
        let dst_init = [255u8; 1024];
        assert_match(&input, &dst_init);
    }

    /// Per-skew arithmetic regression guard. Kernel uses 16-byte
    /// `_mm_loadu_si128` so this isn't an alignment-fault test — it
    /// guards against per-offset stride miscomputes.
    #[test]
    fn idct32x32_misaligned_dst_pointers() {
        require_avx2!();
        // Strong synthetic input — eob ≈ 64 with mixed magnitudes.
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(64).enumerate() {
            *slot = (i as i32) * 199 - 6300;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            // Heap allocate over-sized so we can sub-slice at the
            // requested byte offset (32x32 block + max skew + 1 row
            // overshoot guard = 32 * 32 + 15 + 32 = 1071; round to
            // 1088).
            let mut buf: Box<[u8; 1088]> = Box::new([0u8; 1088]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;

            idct32x32_add_avx2(&input, &mut buf[skew..skew + 32 * 32], 32);
            idct32x32_1024_add(&input, &mut scalar_buf[skew..skew + 32 * 32], 32);
            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }

    /// Forced-AVX2 vs forced-scalar dispatch produces the same output.
    /// Also tautology-guard: under forced-AVX2 caps the dispatch slot
    /// must NOT be the scalar function pointer (otherwise the diff is
    /// trivially zero).
    #[test]
    fn idct32x32_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        // Tautology guard: the AVX2 dispatch slot must NOT collapse to
        // the scalar reference. If they were equal the differential
        // below would be vacuously true.
        let scalar_fp: fn(&[i32; 1024], &mut [u8], usize) = idct32x32_1024_add;
        assert_ne!(
            dispatch_avx2.idct32x32_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 29) % 4001 - 2000;
        }
        let dst_init = [97u8; 1024];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct32x32_add())(&input, &mut dst_avx2, 32);
        (dispatch_scalar.idct32x32_add())(&input, &mut dst_scalar, 32);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
