//! 16×16 hybrid inverse-transform + add — x86_64 AVX2 implementation
//! (Wave F F5).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::iht16x16_256_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same approach as the rest of the AVX2 inverse-transform family: the
//! i32 1D transform math is identical to the scalar reference (1:1
//! reuse of [`crate::transforms::inverse::idct16`] /
//! [`crate::transforms::inverse::iadst16`] per `tx_type`) and AVX2
//! accelerates only the final `(out + 32) >> 6` + add-clamp-pack stage.
//!
//! The `tx_type` argument picks the 1D row/column pair. Per the
//! `iht_16_dispatch` table in [`crate::transforms::inverse`] (libvpx
//! struct order is `{cols, rows}` — see Stage 3 stretch M.8 note):
//!
//! | `tx_type`   | row pass  | col pass  |
//! |-------------|-----------|-----------|
//! | `DctDct`    | `idct16`  | `idct16`  |
//! | `AdstDct`   | `idct16`  | `iadst16` |
//! | `DctAdst`   | `iadst16` | `idct16`  |
//! | `AdstAdst`  | `iadst16` | `iadst16` |
//!
//! `DctDct` is **not** delegated to the dedicated `idct16x16_add_avx2`
//! kernel — we self-implement so the iht slot remains independently
//! testable and dispatch round-trip checks have a clean `assert_ne!`
//! against the scalar `iht16x16_256_add` reference.
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

use crate::transforms::inverse::{iadst16, idct16};
use crate::transforms::types::TxType;

/// Function pointer for a 1D 16-point inverse transform (`idct16` or
/// `iadst16`). Used as the element type of the `iht_16_pair` dispatch
/// tuple — kept as a `type` alias to satisfy
/// `clippy::type_complexity`.
type Xfm16 = fn(&[i32; 16], &mut [i32; 16]);

/// Pick the `(row_xform, col_xform)` pair for the iht16 dispatch table.
/// Bit-exact copy of `iht_16_dispatch` in
/// [`crate::transforms::inverse`] — replicated here to keep the
/// `iht_16_dispatch` helper a `transforms/inverse.rs` internal detail.
#[inline]
fn iht_16_pair(tx_type: TxType) -> (Xfm16, Xfm16) {
    // See note on `iht_4_dispatch` — libvpx struct order is `{cols, rows}`.
    match tx_type {
        TxType::DctDct => (idct16, idct16),
        TxType::AdstDct => (idct16, iadst16),
        TxType::DctAdst => (iadst16, idct16),
        TxType::AdstAdst => (iadst16, iadst16),
    }
}

/// AVX2 implementation of the 16×16 hybrid inverse-transform + residual
/// add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`iht16x16_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `15 * stride + 16` bytes of valid storage.
///
/// # Implementation
///
/// Row + column 1D math uses the scalar references picked by
/// `iht_16_pair(tx_type)` (i32, bit-exact). AVX2 accelerates the final
/// `(out + 32) >> 6` + add + clip stage. Each of the 16 rows produces
/// 16 i32 outputs → 2 ymm loads, round-shift, pack to i16 (= 1 xmm of
/// 16 i16 in natural order), then add 16 zero-extended u8 dst bytes
/// and saturating-packs back to u8.
///
/// All loads / stores from / to `dest` are 16-byte (`__m128i`),
/// unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iht16x16_add_avx2_inner(
    input: &[i32; 256],
    dest: &mut [u8],
    stride: usize,
    tx_type: TxType,
) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    let (row_xform, col_xform) = iht_16_pair(tx_type);

    // ----- Stage 1: row pass (scalar 1D, bit-exact). -----
    let mut intermediate = [0i32; 256];
    for r in 0..16 {
        let row_in: &[i32; 16] = input[r * 16..r * 16 + 16].try_into().unwrap();
        let row_out_mut: &mut [i32; 16] = (&mut intermediate[r * 16..r * 16 + 16])
            .try_into()
            .unwrap();
        row_xform(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar 1D). -----
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
        col_xform(&col_in, &mut col_out);
        for r in 0..16 {
            final_out[r * 16 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // Identical mechanic to `idct16x16_add_avx2` (same per-row shape,
    // same `(out + 32) >> 6` final scaling).
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

/// Safe wrapper for the AVX2 iht16x16 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::iht16x16_256_add`], so the
/// `SimdDispatch::iht16x16_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn iht16x16_add_avx2(input: &[i32; 256], dest: &mut [u8], stride: usize, tx_type: TxType) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `iht16x16_avx2::tests` gates each call on
    // a runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { iht16x16_add_avx2_inner(input, dest, stride, tx_type) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `iht16x16_add`.
    //!
    //! Mirrors the Wave E G4-β iht8x8 / G4.2 idct16x16 test suites with
    //! a `tx_type` axis. Coverage spans all 4 tx_types.
    //!
    //! - **`iht16x16_scalar_eq_avx2_proptest`** — 256-case proptest with
    //!   five stratified strategies (uniform, sparse, eob-bucket,
    //!   high-magnitude, pathological) × four tx_types.
    //! - **`iht16x16_all_zero_input`** — zero coefficients leave dst
    //!   unchanged.
    //! - **`iht16x16_dc_only`** — single DC coefficient at position 0.
    //! - **`iht16x16_single_coefficient_impulses`** — sweep one
    //!   non-zero coefficient across every position 0..256 with several
    //!   magnitudes (per tx_type).
    //! - **`iht16x16_max_input`** — saturation stress with i16::MAX /
    //!   i16::MIN coefficients (per tx_type).
    //! - **`iht16x16_pathological_per_tx_type`** — directed
    //!   `[i16::MAX, i16::MIN, ...]` alternating pattern (per tx_type).
    //! - **`iht16x16_misaligned_dst_pointers`** — per-skew arithmetic
    //!   regression guard (kernel uses `_mm_loadu_si128`, unaligned).
    //! - **`iht16x16_dispatch_round_trip`** — forced-AVX2 vs
    //!   forced-scalar dispatch produce identical output, with
    //!   tautology guard asserting the slot fn pointer is the AVX2
    //!   wrapper and NOT the scalar reference.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::iht16x16_add_avx2;
    use crate::transforms::inverse::iht16x16_256_add;
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

    fn run_avx2(input: &[i32; 256], dst_init: &[u8; 256], tx_type: TxType) -> [u8; 256] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        iht16x16_add_avx2(input, &mut dst, 16, tx_type);
        dst
    }

    fn run_scalar(input: &[i32; 256], dst_init: &[u8; 256], tx_type: TxType) -> [u8; 256] {
        let mut dst = *dst_init;
        iht16x16_256_add(input, &mut dst, 16, tx_type);
        dst
    }

    fn assert_match(input: &[i32; 256], dst_init: &[u8; 256], tx_type: TxType) {
        let scalar = run_scalar(input, dst_init, tx_type);
        let avx2 = run_avx2(input, dst_init, tx_type);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch for tx_type={tx_type:?} (first divergence): scalar != avx2"
        );
    }

    // -----------------------------------------------------------------
    // Strategies (same shape as idct16x16, sized for 256 coeffs).
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(uniform_coeff(), 256).prop_map(|v| {
            let mut out = [0i32; 256];
            out.copy_from_slice(&v);
            out
        })
    }

    fn sparse_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec((0usize..256, uniform_coeff()), 1..=8).prop_map(|nz| {
            let mut out = [0i32; 256];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// K ∈ {1, 10, 38, 256} non-zero coefficients in raster positions
    /// `0..K`. K=1 is the DC-only seam; K=10 / K=38 straddle libvpx's
    /// `idct16x16_10_add` / `_38_add` dispatcher boundaries; K=256 is
    /// full density.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![Just(1usize), Just(10), Just(38), Just(256)].prop_flat_map(|k| {
            prop::collection::vec(uniform_coeff(), k).prop_map(move |values| {
                let mut out = [0i32; 256];
                for (i, val) in values.into_iter().enumerate() {
                    out[i] = val;
                }
                out
            })
        })
    }

    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            256,
        )
        .prop_map(|v| {
            let mut out = [0i32; 256];
            out.copy_from_slice(&v);
            out
        })
    }

    fn pathological_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![
            (-32768i32..=32767).prop_map(|v| [v; 256]),
            Just([i16::MIN as i32; 256]),
            Just([i16::MAX as i32; 256]),
            Just({
                let mut out = [0i32; 256];
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

    fn input_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![
            uniform_strategy(),
            sparse_strategy(),
            eob_bucket_strategy(),
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
        fn iht16x16_scalar_eq_avx2_proptest(
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

    /// All-zero coefficients leave dst unchanged.
    #[test]
    fn iht16x16_all_zero_input() {
        require_avx2!();
        let input = [0i32; 256];
        let dst_init = [42u8; 256];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// DC-only: single coefficient at position 0.
    #[test]
    fn iht16x16_dc_only() {
        require_avx2!();
        let mut input = [0i32; 256];
        input[0] = 1234;
        let dst_init = [50u8; 256];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// Single-coefficient impulse sweep across all 256 positions.
    #[test]
    fn iht16x16_single_coefficient_impulses() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            for pos in 0..256 {
                for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                    let mut input = [0i32; 256];
                    input[pos] = val;
                    let dst_init = [128u8; 256];
                    assert_match(&input, &dst_init, tx_type);
                }
            }
        }
    }

    /// Saturation stress: i16 extremes throughout (per tx_type).
    #[test]
    fn iht16x16_max_input() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            let input = [i16::MAX as i32; 256];
            let dst_init = [0u8; 256];
            assert_match(&input, &dst_init, tx_type);

            let input = [i16::MIN as i32; 256];
            let dst_init = [255u8; 256];
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// Directed pathological per tx_type: alternating i16::MAX /
    /// i16::MIN throughout the 256-coefficient block.
    #[test]
    fn iht16x16_pathological_per_tx_type() {
        require_avx2!();
        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        let dst_init = [128u8; 256];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// Per-skew arithmetic regression guard. Kernel uses 16-byte
    /// `_mm_loadu_si128` so this isn't an alignment-fault test — it
    /// guards against per-offset stride miscomputes.
    #[test]
    fn iht16x16_misaligned_dst_pointers() {
        require_avx2!();
        // Strong synthetic input — eob ≈ 32 with mixed magnitudes.
        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().take(32).enumerate() {
            *slot = (i as i32) * 199 - 3100;
        }

        for &tx_type in &ALL_TX_TYPES {
            for skew in [0usize, 1, 3, 7, 15] {
                // Heap allocate over-sized so we can sub-slice at the
                // requested byte offset (16x16 block + max skew + 1 row
                // overshoot guard = 16 * 16 + 15 + 16 = 287; round to 320).
                let mut buf: Box<[u8; 320]> = Box::new([0u8; 320]);
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (i as u8).wrapping_mul(13);
                }
                let mut scalar_buf = *buf;

                iht16x16_add_avx2(&input, &mut buf[skew..skew + 16 * 16], 16, tx_type);
                iht16x16_256_add(&input, &mut scalar_buf[skew..skew + 16 * 16], 16, tx_type);
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
    /// NOT be the scalar function pointer AND MUST be the AVX2 wrapper
    /// (otherwise the diff is trivially zero).
    #[test]
    fn iht16x16_dispatch_round_trip() {
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
        let scalar_fp: fn(&[i32; 256], &mut [u8], usize, TxType) = iht16x16_256_add;
        let avx2_fp: fn(&[i32; 256], &mut [u8], usize, TxType) = iht16x16_add_avx2;
        assert_ne!(
            dispatch_avx2.iht16x16_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );
        assert_eq!(
            dispatch_avx2.iht16x16_add() as usize,
            avx2_fp as usize,
            "forced-AVX2 caps must install the iht16x16_add_avx2 wrapper"
        );

        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 29) % 4001 - 2000;
        }
        let dst_init = [97u8; 256];
        for &tx_type in &ALL_TX_TYPES {
            let mut dst_avx2 = dst_init;
            let mut dst_scalar = dst_init;
            (dispatch_avx2.iht16x16_add())(&input, &mut dst_avx2, 16, tx_type);
            (dispatch_scalar.iht16x16_add())(&input, &mut dst_scalar, 16, tx_type);
            assert_eq!(dst_avx2, dst_scalar, "dispatch mismatch tx_type={tx_type:?}");
        }
    }
}
