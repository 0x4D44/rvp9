//! 4×4 inverse-DCT + add — x86_64 AVX2 implementation (G4.2).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::idct4x4_16_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! As in G4.1: we keep the i32 1D transform math identical to the
//! scalar reference (1:1 reuse of [`crate::transforms::inverse::idct4`])
//! and AVX2-accelerate the final `(out + 8) >> 4` + add-clamp-pack
//! stage. The pilot proved this pattern; re-deriving the 4×4 butterfly
//! in 16-bit `_mm_madd_epi16` form is left as a future optimisation.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `dest` are 4-byte (one `i32` load per row via
//! `_mm_cvtsi32_si128`, sourcing memory via `read_unaligned`); the
//! 16-i32 stack-scratch `final_out` is read via `_mm256_loadu_si256`.
//! All ops on `dest` are unaligned. The grep sentinel below is
//! checked by `scripts/check-simd-sentinels.sh` / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_loadu_si256,
    _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srai_epi32,
    _mm_add_epi16, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_unpacklo_epi8,
};

use crate::transforms::inverse::idct4;

/// AVX2 implementation of the 4×4 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct4x4_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `3 * stride + 4` bytes of valid storage.
///
/// # Implementation
///
/// Row + column idct4 math uses the scalar reference (i32, bit-exact
/// `wraplow`). AVX2 accelerates the final `(out + 8) >> 4` + add +
/// clip stage by loading the 16 final i32 outputs into a single ymm
/// register, applying the round + shift + pack to i16, then writing
/// row-by-row (each row is 4 u8s) via `_mm_cvtsi32_si128` /
/// `_mm_cvtsi128_si32` so we don't over-read past the per-row 4-byte
/// span.
///
/// All loads / stores from / to `dest` are 4-byte (`i32`), unaligned.
/// See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct4x4_add_avx2_inner(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    // ----- Stage 1: row pass (scalar idct4, bit-exact). -----
    let mut intermediate = [0i32; 16];
    for r in 0..4 {
        let row_in: &[i32; 4] = input[r * 4..r * 4 + 4].try_into().unwrap();
        let row_out_mut: &mut [i32; 4] = (&mut intermediate[r * 4..r * 4 + 4]).try_into().unwrap();
        idct4(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar idct4). -----
    // Produces 4 columns of 4 i32 outputs. Stored in row-major
    // `final_out[r * 4 + c]` so the AVX2 finisher can `loadu` the
    // whole 16-i32 block in two ymm ops (or one if we pack tighter,
    // but we use two for symmetry with G4.1 8×8 layout).
    let mut final_out = [0i32; 16];
    for c in 0..4 {
        let col_in: [i32; 4] = [
            intermediate[c],
            intermediate[4 + c],
            intermediate[8 + c],
            intermediate[12 + c],
        ];
        let mut col_out = [0i32; 4];
        idct4(&col_in, &mut col_out);
        for r in 0..4 {
            final_out[r * 4 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // The 16 i32 outputs fit in two ymm registers (8 lanes each). We
    // load both halves, apply `(x + 8) >> 4`, saturate-pack to i16,
    // then peel off 4-byte rows for the add-and-store.
    //
    // We do two ymm loads (rows 0..2 and rows 2..4) and two pack
    // sequences. Each pack sequence produces 8 i16 lanes in natural
    // order (after `_mm256_permute4x64_epi64(_, 0xD8)`); a low 64-bit
    // extract gives us 4 i16 = one row.
    let round = _mm256_set1_epi32(8);
    let zero = _mm_setzero_si128();
    for half in 0..2 {
        // (1) unaligned load 8 i32 (= 2 rows) from `final_out` stack.
        let row_pair_ptr = final_out.as_ptr().add(half * 8) as *const __m256i;
        let mut out_i32 = _mm256_loadu_si256(row_pair_ptr);
        // (2) add round constant +8, shift right arithmetic by 4.
        out_i32 = _mm256_add_epi32(out_i32, round);
        out_i32 = _mm256_srai_epi32(out_i32, 4);
        // (3) saturate-pack i32 → i16. See idct8x8_avx2.rs Stage 3 for
        // the `_mm256_packs_epi32(x, x) + permute4x64(0xD8)` rationale.
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        // Low 128 bits = 8 × i16 in natural order: [row(2h)[0..4],
        // row(2h+1)[0..4]].
        let out_i16_pair: __m128i = _mm256_castsi256_si128(packed_natural);

        // Extract row a (low 64 bits = first 4 i16) and row b (high 64
        // bits = next 4 i16) by treating the __m128i as two 32-bit lo
        // halves of the residual. We avoid the extra shuffle by reusing
        // `_mm_unpacklo_epi8` on a zero-extended u8 row.
        for sub in 0..2 {
            let r = half * 2 + sub;
            // G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
            // Load 4 dst bytes as i32 (LE-ordered same as memory layout).
            let dst_row_start = r * stride;
            // SAFETY (within the unsafe fn): we read 4 bytes from
            // `dest[dst_row_start..dst_row_start + 4]`. The debug_assert
            // above bounds `dest.len() >= 3*stride + 4`.
            let raw: i32 =
                core::ptr::read_unaligned(dest.as_ptr().add(dst_row_start) as *const i32);
            let dst_u8 = _mm_cvtsi32_si128(raw);
            // Zero-extend the 4 u8 lanes to i16 (low 8 bytes of an
            // __m128i; only the low 4 are valid after the i32 load).
            let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);

            // Get the 4 transform i16 outputs for THIS row from the
            // packed pair. `out_i16_pair` holds 8 × i16 = 16 bytes;
            // row 0 is the low 4 lanes (bytes 0..8), row 1 is the high
            // 4 lanes (bytes 8..16). We need a per-sub extract; the
            // cleanest is a shift right by 64 for `sub == 1`.
            let row_i16: __m128i = if sub == 0 {
                out_i16_pair
            } else {
                // Move the high 64 bits down. `_mm_srli_si128` shifts
                // by *bytes*; 8 bytes = 4 i16 lanes.
                core::arch::x86_64::_mm_srli_si128::<8>(out_i16_pair)
            };

            // Add transform output and saturate-pack to u8. Only the
            // low 4 lanes are meaningful — the high 4 are garbage but
            // are not stored.
            let sum_i16 = _mm_add_epi16(row_i16, dst_i16);
            let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
            // Store the low 4 u8s back via _mm_cvtsi128_si32.
            let lo_i32 = _mm_cvtsi128_si32(sum_u8);
            core::ptr::write_unaligned(dest.as_mut_ptr().add(dst_row_start) as *mut i32, lo_i32);
        }
    }
}

/// Safe wrapper for the AVX2 idct4x4 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct4x4_16_add`], so the
/// `SimdDispatch::idct4x4_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct4x4_add_avx2(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. The unit-test path in
    // `idct4x4_avx2::tests` gates each call on a runtime
    // `is_x86_feature_detected!("avx2")` check, so calls from tests
    // are equally sound.
    unsafe { idct4x4_add_avx2_inner(input, dest, stride) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct4x4_add`.
    //!
    //! Mirrors the G4.1 idct8x8 test suite. EOB-seam set for idct4x4
    //! is `{1, 2, 16}` per `notes/g4-coordination.md`.

    use super::idct4x4_add_avx2;
    use crate::transforms::inverse::idct4x4_16_add;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 16], dst_init: &[u8; 16]) -> [u8; 16] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct4x4_add_avx2(input, &mut dst, 4);
        dst
    }

    fn run_scalar(input: &[i32; 16], dst_init: &[u8; 16]) -> [u8; 16] {
        let mut dst = *dst_init;
        idct4x4_16_add(input, &mut dst, 4);
        dst
    }

    fn assert_match(input: &[i32; 16], dst_init: &[u8; 16]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies (same shape as idct8x8 but sized for 16 coeffs)
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop::collection::vec(uniform_coeff(), 16).prop_map(|v| {
            let mut out = [0i32; 16];
            out.copy_from_slice(&v);
            out
        })
    }

    fn sparse_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop::collection::vec((0usize..16, uniform_coeff()), 1..=4).prop_map(|nz| {
            let mut out = [0i32; 16];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// K ∈ {1, 2, 16} non-zero coefficients in raster positions 0..K
    /// (matching the idct4x4 dispatcher-seam set per
    /// `notes/g4-coordination.md`). K=1 is the DC-only seam; K=16 is
    /// full density.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop_oneof![Just(1usize), Just(2), Just(16)].prop_flat_map(|k| {
            prop::collection::vec(uniform_coeff(), k).prop_map(move |values| {
                let mut out = [0i32; 16];
                for (i, val) in values.into_iter().enumerate() {
                    out[i] = val;
                }
                out
            })
        })
    }

    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            16,
        )
        .prop_map(|v| {
            let mut out = [0i32; 16];
            out.copy_from_slice(&v);
            out
        })
    }

    fn pathological_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop_oneof![
            (-32768i32..=32767).prop_map(|v| [v; 16]),
            Just([i16::MIN as i32; 16]),
            Just([i16::MAX as i32; 16]),
            Just({
                let mut out = [0i32; 16];
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

    fn input_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            eob_bucket_strategy(),
            high_magnitude_strategy(),
            pathological_strategy(),
        ]
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 16]> {
        prop::collection::vec(any::<u8>(), 16).prop_map(|v| {
            let mut out = [0u8; 16];
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
        fn idct4x4_scalar_eq_avx2_proptest(
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

    /// All-zero coefficients: dst unchanged after round-trip.
    #[test]
    fn idct4x4_all_zero_input() {
        require_avx2!();
        let input = [0i32; 16];
        let dst_init = [42u8; 16];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep across every position.
    #[test]
    fn idct4x4_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..16 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 16];
                input[pos] = val;
                let dst_init = [128u8; 16];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Dst-pointer-offset regression guard. Same approach as idct8x8:
    /// the kernel uses per-row 4-byte `_mm_cvtsi32_si128` /
    /// `_mm_cvtsi128_si32` loads/stores, so all addressing is naturally
    /// unaligned — this test catches per-skew arithmetic regressions
    /// rather than alignment-fault paths.
    #[test]
    fn idct4x4_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().take(8).enumerate() {
            *slot = (i as i32) * 91 - 400;
        }
        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 32]> = Box::new([0u8; 32]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(11);
            }
            let mut scalar_buf = *buf;
            // Each row of the 4×4 block lives at `dst[r * stride .. r * stride + 4]`
            // inside the working sub-slice. Stride is 4 (no row padding)
            // so the kernel touches `[skew .. skew + 4 * 3 + 4] = [skew, skew + 16)`.
            idct4x4_add_avx2(&input, &mut buf[skew..skew + 16], 4);
            idct4x4_16_add(&input, &mut scalar_buf[skew..skew + 16], 4);
            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }

    /// Round-trip through the dispatch: forced-AVX2 and forced-scalar
    /// must agree.
    #[test]
    fn idct4x4_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 23) % 4001 - 2000;
        }
        let dst_init = [97u8; 16];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct4x4_add())(&input, &mut dst_avx2, 4);
        (dispatch_scalar.idct4x4_add())(&input, &mut dst_scalar, 4);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
