//! 4×4 hybrid inverse-transform + add — x86_64 AVX2 implementation
//! (Wave E G4-β).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]) and the G4.2 4×4 idct kernel
//! ([`crate::dsp::x86_64::idct4x4_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::iht4x4_16_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same approach as the rest of the AVX2 inverse-transform family: the
//! i32 1D transform math is identical to the scalar reference (1:1
//! reuse of [`crate::transforms::inverse::idct4`] /
//! [`crate::transforms::inverse::iadst4`] per `tx_type`) and AVX2
//! accelerates only the final `(out + 8) >> 4` + add-clamp-pack stage.
//!
//! The `tx_type` argument picks the 1D row/column pair. Per the
//! `iht_4_dispatch` table in [`crate::transforms::inverse`] (libvpx
//! struct order is `{cols, rows}` — see Stage 3 stretch M.8 note):
//!
//! | `tx_type`   | row pass | col pass |
//! |-------------|----------|----------|
//! | `DctDct`    | `idct4`  | `idct4`  |
//! | `AdstDct`   | `idct4`  | `iadst4` |
//! | `DctAdst`   | `iadst4` | `idct4`  |
//! | `AdstAdst`  | `iadst4` | `iadst4` |
//!
//! `DctDct` is **not** delegated to the dedicated idct4x4 kernel — we
//! self-implement so the iht slot remains independently testable and
//! dispatch round-trip checks have a clean `assert_ne!` against the
//! scalar `iht4x4_16_add` reference.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `dest` are 4-byte (one `i32` per row via
//! `read_unaligned` + `_mm_cvtsi32_si128`); the 16-i32 stack-scratch
//! `final_out` is read via `_mm256_loadu_si256`. The grep sentinel
//! below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_loadu_si256,
    _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_srai_epi32,
    _mm_add_epi16, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_unpacklo_epi8,
};

use crate::transforms::inverse::{iadst4, idct4};
use crate::transforms::types::TxType;

/// Function pointer for a 1D 4-point inverse transform (`idct4` or
/// `iadst4`). Used as the element type of the `iht_4_pair` dispatch
/// tuple — kept as a `type` alias to satisfy
/// `clippy::type_complexity`.
type Xfm4 = fn(&[i32; 4], &mut [i32; 4]);

/// Pick the `(row_xform, col_xform)` pair for the iht4 dispatch table.
/// Bit-exact copy of `iht_4_dispatch` in
/// [`crate::transforms::inverse`] — we replicate it here (rather than
/// adding a `pub` accessor) to keep the iht_4 dispatch table a
/// `crates/rvp9-core/src/transforms/inverse.rs` internal detail.
#[inline]
fn iht_4_pair(tx_type: TxType) -> (Xfm4, Xfm4) {
    match tx_type {
        TxType::DctDct => (idct4, idct4),
        TxType::AdstDct => (idct4, iadst4),
        TxType::DctAdst => (iadst4, idct4),
        TxType::AdstAdst => (iadst4, iadst4),
    }
}

/// AVX2 implementation of the 4×4 hybrid inverse-transform + residual
/// add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`iht4x4_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `3 * stride + 4` bytes of valid storage.
///
/// # Implementation
///
/// Row + column 1D math uses the scalar references picked by
/// `iht_4_pair(tx_type)` (i32, bit-exact). AVX2 accelerates the final
/// `(out + 8) >> 4` + add + clip stage by loading the 16 final i32
/// outputs into two ymm registers, applying the round + shift + pack
/// to i16, then writing row-by-row (each row is 4 u8s).
///
/// All loads / stores from / to `dest` are 4-byte (`i32`), unaligned.
/// See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn iht4x4_add_avx2_inner(
    input: &[i32; 16],
    dest: &mut [u8],
    stride: usize,
    tx_type: TxType,
) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    let (row_xform, col_xform) = iht_4_pair(tx_type);

    // ----- Stage 1: row pass (scalar 1D, bit-exact). -----
    let mut intermediate = [0i32; 16];
    for r in 0..4 {
        let row_in: &[i32; 4] = input[r * 4..r * 4 + 4].try_into().unwrap();
        let row_out_mut: &mut [i32; 4] = (&mut intermediate[r * 4..r * 4 + 4]).try_into().unwrap();
        row_xform(row_in, row_out_mut);
    }

    // ----- Stage 2: column pass (scalar 1D). -----
    // Produces 4 columns of 4 i32 outputs. Stored row-major into
    // `final_out[r * 4 + c]` so the AVX2 finisher loads two ymm halves
    // covering all 16 outputs.
    let mut final_out = [0i32; 16];
    for c in 0..4 {
        let col_in: [i32; 4] = [
            intermediate[c],
            intermediate[4 + c],
            intermediate[8 + c],
            intermediate[12 + c],
        ];
        let mut col_out = [0i32; 4];
        col_xform(&col_in, &mut col_out);
        for r in 0..4 {
            final_out[r * 4 + c] = col_out[r];
        }
    }

    // ----- Stage 3: AVX2-accelerated add + clamp + pack + store. -----
    //
    // The 16 i32 outputs fit in two ymm registers (8 lanes each). We
    // load both halves, apply `(x + 8) >> 4`, saturate-pack to i16,
    // then peel off 4-byte rows for the add-and-store. Same mechanic
    // as `idct4x4_add_avx2` (identical final scaling).
    let round = _mm256_set1_epi32(8);
    let zero = _mm_setzero_si128();
    for half in 0..2 {
        // (1) unaligned load 8 i32 (= 2 rows) from `final_out` stack.
        let row_pair_ptr = final_out.as_ptr().add(half * 8) as *const __m256i;
        let mut out_i32 = _mm256_loadu_si256(row_pair_ptr);
        // (2) add round constant +8, shift right arithmetic by 4.
        out_i32 = _mm256_add_epi32(out_i32, round);
        out_i32 = _mm256_srai_epi32(out_i32, 4);
        // (3) saturate-pack i32 → i16 with the `_mm256_packs_epi32(x, x)
        // + permute4x64(0xD8)` idiom — see `idct8x8_avx2.rs` Stage 3
        // for the lane-order rationale.
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        // Low 128 bits = 8 × i16 in natural order: [row(2h)[0..4],
        // row(2h+1)[0..4]].
        let out_i16_pair: __m128i = _mm256_castsi256_si128(packed_natural);

        for sub in 0..2 {
            let r = half * 2 + sub;
            // G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
            let dst_row_start = r * stride;
            let raw: i32 =
                core::ptr::read_unaligned(dest.as_ptr().add(dst_row_start) as *const i32);
            let dst_u8 = _mm_cvtsi32_si128(raw);
            let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);

            // Get the 4 transform i16 outputs for THIS row from the
            // packed pair (low 4 lanes = row 2h, high 4 lanes = row
            // 2h+1; shift by 8 bytes = 4 lanes to address the second
            // row).
            let row_i16: __m128i = if sub == 0 {
                out_i16_pair
            } else {
                core::arch::x86_64::_mm_srli_si128::<8>(out_i16_pair)
            };

            let sum_i16 = _mm_add_epi16(row_i16, dst_i16);
            let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
            let lo_i32 = _mm_cvtsi128_si32(sum_u8);
            core::ptr::write_unaligned(dest.as_mut_ptr().add(dst_row_start) as *mut i32, lo_i32);
        }
    }
}

/// Safe wrapper for the AVX2 iht4x4 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::iht4x4_16_add`], so the
/// `SimdDispatch::iht4x4_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn iht4x4_add_avx2(input: &[i32; 16], dest: &mut [u8], stride: usize, tx_type: TxType) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `iht4x4_avx2::tests` gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { iht4x4_add_avx2_inner(input, dest, stride, tx_type) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `iht4x4_add`.
    //!
    //! Mirrors the G4.1 / G4.2 idct test suites. Coverage matches the
    //! Wave E G4-β acceptance criteria: 7 named tests, all four
    //! `tx_type` variants exercised.
    //!
    //! - **`iht4x4_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (all four tx_types).
    //! - **`iht4x4_dc_only`** — single coefficient at position 0 (all
    //!   four tx_types).
    //! - **`iht4x4_single_coefficient_impulses`** — sweep one non-zero
    //!   coefficient across every position 0..16 (per tx_type).
    //! - **`iht4x4_max_input`** — i16::MAX / i16::MIN saturation stress.
    //! - **`iht4x4_misaligned_dst_pointers`** — per-skew regression
    //!   guard (kernel uses `_mm_cvtsi32_si128`, unaligned).
    //! - **`iht4x4_scalar_eq_avx2_proptest`** — 256-case proptest with
    //!   five stratified strategies × four tx_types.
    //! - **`iht4x4_dispatch_round_trip`** — forced-AVX2 vs forced-scalar
    //!   dispatch produce identical output (with tautology guard).
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::iht4x4_add_avx2;
    use crate::transforms::inverse::iht4x4_16_add;
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

    fn run_avx2(input: &[i32; 16], dst_init: &[u8; 16], tx_type: TxType) -> [u8; 16] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        iht4x4_add_avx2(input, &mut dst, 4, tx_type);
        dst
    }

    fn run_scalar(input: &[i32; 16], dst_init: &[u8; 16], tx_type: TxType) -> [u8; 16] {
        let mut dst = *dst_init;
        iht4x4_16_add(input, &mut dst, 4, tx_type);
        dst
    }

    fn assert_match(input: &[i32; 16], dst_init: &[u8; 16], tx_type: TxType) {
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

    /// K ∈ {1, 4, 16} non-zero coefficients in raster positions 0..K.
    /// K=1 is DC-only; K=16 is full density.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop_oneof![Just(1usize), Just(4), Just(16)].prop_flat_map(|k| {
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
        fn iht4x4_scalar_eq_avx2_proptest(
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
    fn iht4x4_all_zero_input() {
        require_avx2!();
        let input = [0i32; 16];
        let dst_init = [42u8; 16];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    #[test]
    fn iht4x4_dc_only() {
        require_avx2!();
        let mut input = [0i32; 16];
        input[0] = 1234;
        let dst_init = [50u8; 16];
        for &tx_type in &ALL_TX_TYPES {
            assert_match(&input, &dst_init, tx_type);
        }
    }

    #[test]
    fn iht4x4_single_coefficient_impulses() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            for pos in 0..16 {
                for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                    let mut input = [0i32; 16];
                    input[pos] = val;
                    let dst_init = [128u8; 16];
                    assert_match(&input, &dst_init, tx_type);
                }
            }
        }
    }

    #[test]
    fn iht4x4_max_input() {
        require_avx2!();
        for &tx_type in &ALL_TX_TYPES {
            let input = [i16::MAX as i32; 16];
            let dst_init = [0u8; 16];
            assert_match(&input, &dst_init, tx_type);

            let input = [i16::MIN as i32; 16];
            let dst_init = [255u8; 16];
            assert_match(&input, &dst_init, tx_type);
        }
    }

    /// Per-skew arithmetic regression guard. Kernel uses
    /// `_mm_cvtsi32_si128` (4-byte unaligned) so this isn't an
    /// alignment-fault test — it guards against per-offset stride
    /// miscomputes.
    #[test]
    fn iht4x4_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = (i as i32) * 113 - 600;
        }

        for &tx_type in &ALL_TX_TYPES {
            for skew in [0usize, 1, 3, 7, 15] {
                // 4x4 block + max skew = 16 + 15 = 31; round to 48.
                let mut buf: Box<[u8; 48]> = Box::new([0u8; 48]);
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = (i as u8).wrapping_mul(11);
                }
                let mut scalar_buf = *buf;

                iht4x4_add_avx2(&input, &mut buf[skew..skew + 4 * 4], 4, tx_type);
                iht4x4_16_add(&input, &mut scalar_buf[skew..skew + 4 * 4], 4, tx_type);
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
    fn iht4x4_dispatch_round_trip() {
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
        let scalar_fp: fn(&[i32; 16], &mut [u8], usize, TxType) = iht4x4_16_add;
        assert_ne!(
            dispatch_avx2.iht4x4_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 23) % 4001 - 2000;
        }
        let dst_init = [97u8; 16];
        for &tx_type in &ALL_TX_TYPES {
            let mut dst_avx2 = dst_init;
            let mut dst_scalar = dst_init;
            (dispatch_avx2.iht4x4_add())(&input, &mut dst_avx2, 4, tx_type);
            (dispatch_scalar.iht4x4_add())(&input, &mut dst_scalar, 4, tx_type);
            assert_eq!(dst_avx2, dst_scalar, "dispatch mismatch tx_type={tx_type:?}");
        }
    }
}
