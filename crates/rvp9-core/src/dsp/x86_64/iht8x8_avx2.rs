//! 8×8 hybrid inverse-transform + add — x86_64 AVX2 implementation
//! (Wave E G4-β).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::iht8x8_64_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same approach as the rest of the AVX2 inverse-transform family: the
//! i32 1D transform math is identical to the scalar reference (1:1
//! reuse of [`crate::transforms::inverse::idct8`] /
//! [`crate::transforms::inverse::iadst8`] per `tx_type`) and AVX2
//! accelerates only the final `(out + 16) >> 5` + add-clamp-pack stage.
//!
//! The `tx_type` argument picks the 1D row/column pair. Per the
//! `iht_8_dispatch` table in [`crate::transforms::inverse`] (libvpx
//! struct order is `{cols, rows}` — see Stage 3 stretch M.8 note):
//!
//! | `tx_type`   | row pass | col pass |
//! |-------------|----------|----------|
//! | `DctDct`    | `idct8`  | `idct8`  |
//! | `AdstDct`   | `idct8`  | `iadst8` |
//! | `DctAdst`   | `iadst8` | `idct8`  |
//! | `AdstAdst`  | `iadst8` | `iadst8` |
//!
//! `DctDct` is **not** delegated to the dedicated idct8x8 kernel — we
//! self-implement so the iht slot remains independently testable and
//! dispatch round-trip checks have a clean `assert_ne!` against the
//! scalar `iht8x8_64_add` reference.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `dest` are 8-byte (`_mm_loadl_epi64`); the
//! 64-i32 stack-scratch `final_out` is read via `_mm256_loadu_si256`.
//! The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_loadu_si256,
    _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srai_epi32,
    _mm_add_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_setzero_si128, _mm_storel_epi64,
    _mm_unpacklo_epi8,
};

use crate::transforms::inverse::{iadst8, idct8};
use crate::transforms::types::TxType;

/// Function pointer for a 1D 8-point inverse transform (`idct8` or
/// `iadst8`). Used as the element type of the `iht_8_pair` dispatch
/// tuple — kept as a `type` alias to satisfy
/// `clippy::type_complexity`.
type Xfm8 = fn(&[i32; 8], &mut [i32; 8]);

/// Pick the `(row_xform, col_xform)` pair for the iht8 dispatch table.
/// Bit-exact copy of `iht_8_dispatch` in
/// [`crate::transforms::inverse`] — replicated here to keep the
/// `iht_8_dispatch` helper a `transforms/inverse.rs` internal detail.
#[inline]
fn iht_8_pair(tx_type: TxType) -> (Xfm8, Xfm8) {
    match tx_type {
        TxType::DctDct => (idct8, idct8),
        TxType::AdstDct => (idct8, iadst8),
        TxType::DctAdst => (iadst8, idct8),
        TxType::AdstAdst => (iadst8, iadst8),
    }
}

/// AVX2 implementation of the 8×8 hybrid inverse-transform + residual
/// add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`iht8x8_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `7 * stride + 8` bytes of valid storage.
///
/// # Implementation
///
/// Row + column 1D math uses the scalar references picked by
/// `iht_8_pair(tx_type)` (i32, bit-exact). AVX2 accelerates the final
/// `(out + 16) >> 5` + add + clip stage. Each row holds 8 i32 outputs
/// → 1 ymm load, round-shift, pack to i16 (= 1 xmm of 8 i16 in
/// natural order), then add 8 zero-extended u8 dst bytes and
/// saturate-pack back to u8 (stored via `_mm_storel_epi64`).
///
/// All loads / stores from / to `dest` are 8-byte (`__m128i` low
/// half), unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iht8x8_add_avx2_inner(
    input: &[i32; 64],
    dest: &mut [u8],
    stride: usize,
    tx_type: TxType,
) {
    debug_assert!(dest.len() >= stride * 7 + 8, "dest too small for 8x8 add");

    let (row_xform, col_xform) = iht_8_pair(tx_type);

    // ----- Stage 1: row pass (scalar 1D, bit-exact). -----
    let mut intermediate = [0i32; 64];
    for r in 0..8 {
        let row_in: &[i32; 8] = input[r * 8..r * 8 + 8].try_into().unwrap();
        let row_out_mut: &mut [i32; 8] = (&mut intermediate[r * 8..r * 8 + 8]).try_into().unwrap();
        row_xform(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar 1D). -----
    // Produces 8 columns of 8 i32 outputs. Stored row-major into
    // `final_out[r * 8 + c]` so the AVX2 finisher can `loadu` a whole
    // row in one ymm op.
    let mut final_out = [0i32; 64];
    let mut col_in = [0i32; 8];
    let mut col_out = [0i32; 8];
    for c in 0..8 {
        for r in 0..8 {
            col_in[r] = intermediate[r * 8 + c];
        }
        col_xform(&col_in, &mut col_out);
        for r in 0..8 {
            final_out[r * 8 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // Identical mechanic to `idct8x8_add_avx2` (same per-row shape,
    // same `(out + 16) >> 5` final scaling).
    let round = _mm256_set1_epi32(16);
    let zero = _mm_setzero_si128();
    for r in 0..8 {
        let row_ptr = final_out.as_ptr().add(r * 8) as *const __m256i;
        // (1) unaligned load 8 i32 from `final_out` (stack scratch).
        let mut out_i32 = _mm256_loadu_si256(row_ptr);
        // (2) add round constant +16, shift right arithmetic by 5.
        out_i32 = _mm256_add_epi32(out_i32, round);
        out_i32 = _mm256_srai_epi32(out_i32, 5);
        // (3) saturate-pack i32 → i16 with the
        // `_mm256_packs_epi32(x, x) + permute4x64(0xD8)` idiom — see
        // `idct8x8_avx2.rs` Stage 3 for the lane-order rationale.
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        let out_i16_lo = _mm256_castsi256_si128(packed_natural);

        // (4) load 8 u8 from dst row (unaligned).
        // G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
        let dst_row_ptr = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadl_epi64(dst_row_ptr);
        // (5) zero-extend u8 → i16 (only the low 8 bytes are valid).
        let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);
        // Add transform output and saturate-pack to u8.
        let sum_i16 = _mm_add_epi16(out_i16_lo, dst_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
        // (6) store 8 u8 back (unaligned).
        let dst_row_ptr_mut = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storel_epi64(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the AVX2 iht8x8 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::iht8x8_64_add`], so the
/// `SimdDispatch::iht8x8_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn iht8x8_add_avx2(input: &[i32; 64], dest: &mut [u8], stride: usize, tx_type: TxType) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `iht8x8_avx2::tests` gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { iht8x8_add_avx2_inner(input, dest, stride, tx_type) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `iht8x8_add`.
    //!
    //! Mirrors the G4.1 idct8x8 test suite shape with the addition of a
    //! `tx_type` axis. Coverage spans all 4 tx_types per the Wave E
    //! G4-β acceptance criteria.
    //!
    //! - **`iht8x8_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (all four tx_types).
    //! - **`iht8x8_dc_only`** — single coefficient at position 0 (all
    //!   four tx_types).
    //! - **`iht8x8_single_coefficient_impulses`** — sweep one non-zero
    //!   coefficient across every position 0..64 (per tx_type).
    //! - **`iht8x8_max_input`** — i16::MAX / i16::MIN saturation stress.
    //! - **`iht8x8_misaligned_dst_pointers`** — per-skew regression
    //!   guard (kernel uses `_mm_loadl_epi64`, unaligned).
    //! - **`iht8x8_scalar_eq_avx2_proptest`** — 256-case proptest with
    //!   five stratified strategies × four tx_types.
    //! - **`iht8x8_dispatch_round_trip`** — forced-AVX2 vs forced-scalar
    //!   dispatch produce identical output (with tautology guard).
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::iht8x8_add_avx2;
    use crate::transforms::inverse::iht8x8_64_add;
    use crate::transforms::types::TxType;
    use proptest::prelude::*;

    const ALL_TX_TYPES: [TxType; 4] = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
    ];

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 64], dst_init: &[u8; 64], tx_type: TxType) -> [u8; 64] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        iht8x8_add_avx2(input, &mut dst, 8, tx_type);
        dst
    }

    fn run_scalar(input: &[i32; 64], dst_init: &[u8; 64], tx_type: TxType) -> [u8; 64] {
        let mut dst = *dst_init;
        iht8x8_64_add(input, &mut dst, 8, tx_type);
        dst
    }

    fn assert_match(input: &[i32; 64], dst_init: &[u8; 64], tx_type: TxType) {
        let scalar = run_scalar(input, dst_init, tx_type);
        let avx2 = run_avx2(input, dst_init, tx_type);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch for tx_type={tx_type:?}:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(uniform_coeff(), 64).prop_map(|v| {
            let mut out = [0i32; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    fn sparse_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec((0usize..64, uniform_coeff()), 1..=4).prop_map(|nz| {
            let mut out = [0i32; 64];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// K ∈ {1, 2, 12, 13, 64} non-zero coefficients in raster positions
    /// 0..K (matches `idct8x8` dispatcher seam set).
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![Just(1usize), Just(2), Just(12), Just(13), Just(64)].prop_flat_map(|k| {
            prop::collection::vec(uniform_coeff(), k).prop_map(move |values| {
                let mut out = [0i32; 64];
                for (i, val) in values.into_iter().enumerate() {
                    out[i] = val;
                }
                out
            })
        })
    }

    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            64,
        )
        .prop_map(|v| {
            let mut out = [0i32; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    fn pathological_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            (-32768i32..=32767).prop_map(|v| [v; 64]),
            Just([i16::MIN as i32; 64]),
            Just([i16::MAX as i32; 64]),
            Just({
                let mut out = [0i32; 64];
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

    fn input_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            eob_bucket_strategy(),
            high_magnitude_strategy(),
            pathological_strategy(),
        ]
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 64]> {
        prop::collection::vec(any::<u8>(), 64).prop_map(|v| {
            let mut out = [0u8; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    fn tx_type_strategy() -> impl Strategy<Value = TxType> {
        prop_oneof![
            Just(TxType::DctDct),
            Just(TxType::AdstDct),
            Just(TxType::DctAdst),
            Just(TxType::AdstAdst),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn iht8x8_scalar_eq_avx2_proptest(
            input in input_strategy(),
            dst_init in dst_strategy(),
            tx_type in tx_type_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let scalar = run_scalar(&input, &dst_init, tx_type);
            let avx2 = run_avx2(&input, &dst_init, tx_type);
            prop_assert_eq!(scalar, avx2);
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures (each spans all 4 tx_types).
    // -----------------------------------------------------------------

    #[test]
    fn iht8x8_all_zero_input() {
        require_avx2!();
        let input = [0i32; 64];
        let dst_init = [42u8; 64];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    #[test]
    fn iht8x8_dc_only() {
        require_avx2!();
        let mut input = [0i32; 64];
        input[0] = 1234;
        let dst_init = [50u8; 64];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    #[test]
    fn iht8x8_single_coefficient_impulses() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            for pos in 0..64 {
                for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                    let mut input = [0i32; 64];
                    input[pos] = val;
                    let dst_init = [128u8; 64];
                    assert_match(&input, &dst_init, tx_type);
                }
            }
        }
    }

    #[test]
    fn iht8x8_max_input() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            let input = [i16::MAX as i32; 64];
            let dst_init = [0u8; 64];
            assert_match(&input, &dst_init, tx_type);

            let input = [i16::MIN as i32; 64];
            let dst_init = [255u8; 64];
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// Per-skew arithmetic regression guard. Kernel uses
    /// `_mm_loadl_epi64` (8-byte unaligned) so this isn't an
    /// alignment-fault test — it guards against per-offset stride
    /// miscomputes.
    #[test]
    fn iht8x8_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().take(16).enumerate() {
            *slot = (i as i32) * 137 - 800;
        }

        for &tx_type in &ALL_TX_TYPES {
            for skew in [0usize, 1, 3, 7, 15] {
                // 8x8 block + max skew = 64 + 15 = 79; round to 96.
                let mut buf: Box<[u8; 96]> = Box::new([0u8; 96]);
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (i as u8).wrapping_mul(7);
                }
                let mut scalar_buf = *buf;

                iht8x8_add_avx2(&input, &mut buf[skew..skew + 8 * 8], 8, tx_type);
                iht8x8_64_add(&input, &mut scalar_buf[skew..skew + 8 * 8], 8, tx_type);
                assert_eq!(
                    &buf[..],
                    &scalar_buf[..],
                    "AVX2/scalar mismatch at skew={skew} tx_type={tx_type:?}"
                );
            }
        }
    }

    /// Forced-AVX2 vs forced-scalar dispatch produces the same output.
    /// Tautology guard: under forced-AVX2 caps the dispatch slot must
    /// NOT be the scalar function pointer (otherwise the diff is
    /// trivially zero).
    #[test]
    fn iht8x8_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        // Tautology guard.
        let scalar_fp: fn(&[i32; 64], &mut [u8], usize, TxType) = iht8x8_64_add;
        assert_ne!(
            dispatch_avx2.iht8x8_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 19) % 4001 - 2000;
        }
        let dst_init = [97u8; 64];
        for &tx_type in &ALL_TX_TYPES {
            let mut dst_avx2 = dst_init;
            let mut dst_scalar = dst_init;
            (dispatch_avx2.iht8x8_add())(&input, &mut dst_avx2, 8, tx_type);
            (dispatch_scalar.iht8x8_add())(&input, &mut dst_scalar, 8, tx_type);
            assert_eq!(dst_avx2, dst_scalar, "dispatch mismatch tx_type={tx_type:?}");
        }
    }
}
