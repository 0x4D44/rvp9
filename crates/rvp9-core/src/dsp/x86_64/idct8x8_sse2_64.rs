//! 8×8 inverse-DCT + add — x86_64 SSE2 i16-lane implementation
//! covering the dense path (the `idct8x8_add` slot when AVX2 is the
//! only alternative).
//!
//! This kernel is a line-by-line port of libvpx's
//! `vpx_idct8x8_64_add_sse2` (`third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:158-172`)
//! along with the inner 1-D core `vpx_idct8_sse2` (`inv_txfm_sse2.c:224-230`)
//! and the helper `idct8` from `inv_txfm_sse2.h:250-283`. It replaces
//! the slower i32-lane AVX2 path currently wired to `idct8x8_add`; the
//! i16-lane implementation packs 8 elements per `xmm` (vs 8 elements
//! per `ymm` in the AVX2 i32-lane path), so two stages of butterflies
//! pay one madd instead of two.
//!
//! Companion to [`super::idct8x8_12_sse2`] (the eob<=12 sparse path)
//! and [`super::idct16x16_sse2_256`] (the dense 16×16 SSE2 port). The
//! `butterfly` helper is reused from `idct8x8_12_sse2`, and
//! `load_input_data8` + `transpose_16bit_8x8` are reused from
//! `idct16x16_sse2_256` (both are `pub(crate)` after this port).
//!
//! ## Saturating-vs-modular semantic divergence
//!
//! Like the [`super::idct8x8_12_sse2`] and [`super::idct16x16_sse2_256`]
//! kernels, this implementation uses the dense `_mm_madd_epi16`
//! pipeline with `_mm_packs_epi32` (saturating, clamps to `±i16::MAX`)
//! at every wraplow site rather than the modular `wraplow` (sign-extend
//! low 16 bits) used by the i32-lane AVX2 kernels and the scalar
//! reference. The two policies agree on every in-spec bitstream
//! coefficient (and on the empirical 201-vector conformance corpus),
//! but diverge on synthetic adversarial inputs whose intermediate
//! magnitudes exceed `i16::MAX`.
//!
//! Per Task A precedent, the differential proptest at the bottom of
//! this file is bounded to ≤ i12 magnitudes (`[-2048, 2047]`); the
//! AVX2-style "pathological" strategies (i16-extreme alternating-sign
//! patterns) are **deliberately omitted** because they trip the
//! saturation divergence and would fail spuriously.
//!
//! ## Convention compatibility
//!
//! 1. The body uses `#[target_feature(enable = "sse2")]` on
//!    `unsafe fn`. SSE2 is architecturally universal on x86_64 (per
//!    SysV AMD64 / MS x64 ABIs), but tests still gate on
//!    `is_x86_feature_detected!("sse2")` belt-and-braces.
//! 2. A safe wrapper [`idct8x8_add_sse2_64`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant — see [`crate::dsp::dispatch`].
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadl_epi64` (8-byte, unaligned) via
//! `recon_and_store_row` (one 8-byte residual add per row × 8 rows).
//! The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128,
    _mm_srai_epi16, _mm_storel_epi64, _mm_sub_epi16, _mm_unpacklo_epi8,
};

use crate::transforms::constants::{
    COSPI_4_64, COSPI_8_64, COSPI_12_64, COSPI_16_64, COSPI_20_64, COSPI_24_64, COSPI_28_64,
};

#[cfg(target_arch = "x86_64")]
use super::idct8x8_12_sse2::butterfly;
#[cfg(target_arch = "x86_64")]
use super::idct16x16_sse2_256::{load_input_data8, transpose_16bit_8x8};

// ---------------------------------------------------------------------
// Inner 1-D core (line-by-line port of libvpx `idct8`).
// ---------------------------------------------------------------------

/// `idct8(in, out)` — 1-D 8-point inverse DCT, processing 8 independent
/// instances in parallel (one per i16 lane).
///
/// Line-by-line port of libvpx `inv_txfm_sse2.h:250-283`. Built from
/// `butterfly` calls (reused from Task A) plus `_mm_add_epi16` /
/// `_mm_sub_epi16` sum/difference combinations.
///
/// Note: this is structurally identical to the `idct8_sse2` helper used
/// inside [`super::idct8x8_12_sse2`] for the column pass. We could in
/// principle share that helper, but the _12 module marks it
/// module-private; duplicating here keeps the two SSE2 kernels
/// self-contained, mirroring libvpx's own organisation.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2 (universal on
/// x86_64; gated at dispatch-build time via `caps.sse2`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct8(in_: &[__m128i; 8], out: &mut [__m128i; 8]) {
    // stage 1
    let (s1_4, s1_7) = butterfly(in_[1], in_[7], COSPI_28_64 as i16, COSPI_4_64 as i16);
    let (s1_5, s1_6) = butterfly(in_[5], in_[3], COSPI_12_64 as i16, COSPI_20_64 as i16);

    // stage 2 — note libvpx's butterfly call signature `&step2[1], &step2[0]`
    // (output order flipped) for the cospi_16,cospi_16 pair.
    let (s2_1, s2_0) = butterfly(in_[0], in_[4], COSPI_16_64 as i16, COSPI_16_64 as i16);
    let (s2_2, s2_3) = butterfly(in_[2], in_[6], COSPI_24_64 as i16, COSPI_8_64 as i16);

    let s2_4 = _mm_add_epi16(s1_4, s1_5);
    let s2_5 = _mm_sub_epi16(s1_4, s1_5);
    let s2_6 = _mm_sub_epi16(s1_7, s1_6);
    let s2_7 = _mm_add_epi16(s1_7, s1_6);

    // stage 3
    let st1_0 = _mm_add_epi16(s2_0, s2_3);
    let st1_1 = _mm_add_epi16(s2_1, s2_2);
    let st1_2 = _mm_sub_epi16(s2_1, s2_2);
    let st1_3 = _mm_sub_epi16(s2_0, s2_3);
    let (st1_5, st1_6) = butterfly(s2_6, s2_5, COSPI_16_64 as i16, COSPI_16_64 as i16);

    // stage 4
    out[0] = _mm_add_epi16(st1_0, s2_7);
    out[1] = _mm_add_epi16(st1_1, st1_6);
    out[2] = _mm_add_epi16(st1_2, st1_5);
    out[3] = _mm_add_epi16(st1_3, s2_4);
    out[4] = _mm_sub_epi16(st1_3, s2_4);
    out[5] = _mm_sub_epi16(st1_2, st1_5);
    out[6] = _mm_sub_epi16(st1_1, st1_6);
    out[7] = _mm_sub_epi16(st1_0, s2_7);
}

/// `recon_and_store_row(dest, in_x)` — final-rounding (+16, >>5) + 8-byte
/// residual add for one row of the 8×8 output.
///
/// Mirrors libvpx `inv_txfm_sse2.h:135-142` (`recon_and_store`) plus
/// the `round_shift_8x8` per-row body. Uses `_mm_add_epi16` (NOT the
/// saturating `_mm_adds_epi16`) for the +16 rounding — matches libvpx
/// `round_shift_8x8` in `inv_txfm_sse2.h:144-165` exactly.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `dest..dest+8` is a valid, readable + writable u8 region.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn recon_and_store_row(dest: *mut u8, in_x: __m128i) {
    let final_rounding = _mm_set1_epi16(1 << 4);
    let zero = _mm_setzero_si128();
    let mut t = _mm_add_epi16(in_x, final_rounding);
    t = _mm_srai_epi16(t, 5);

    let d_u8 = _mm_loadl_epi64(dest as *const __m128i);
    let d_i16 = _mm_unpacklo_epi8(d_u8, zero);
    let sum_i16 = _mm_add_epi16(t, d_i16);
    let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
    _mm_storel_epi64(dest as *mut __m128i, sum_u8);
}

/// Inner SSE2 implementation of the dense 8×8 inverse-DCT + residual
/// add. Two-pass `idct8` (row pass + column pass) with a `transpose`
/// between, mirroring libvpx `vpx_idct8x8_64_add_sse2` /
/// `vpx_idct8_sse2`.
///
/// Note: libvpx's `vpx_idct8_sse2` performs the transpose-then-idct8
/// pair, called twice. The first call transposes the row layout
/// (so the inner `idct8` runs over rows-as-columns), and the second
/// call transposes again (back to the row layout) and runs the column
/// pass. The end result is row pass + column pass with the same 1-D
/// transform, exactly as the scalar reference does.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2 (universal on x86_64).
/// - `dst` has at least `7 * stride + 8` bytes of valid storage.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct8x8_add_sse2_64_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 7 + 8, "dst too small for 8x8 add");

    let zero = _mm_setzero_si128();
    let mut in_: [__m128i; 8] = [zero; 8];

    // ---- Load 8 rows × 8 i16 lanes via saturating-pack from i32. ----
    //
    // Mirrors libvpx `load_buffer_8x8` (`inv_txfm_sse2.c:146-156`) with
    // the HBD-branch `load_input_data8` (i32 → i16 saturating-pack).
    let input_ptr = input.as_ptr();
    in_[0] = load_input_data8(input_ptr.add(0 * 8));
    in_[1] = load_input_data8(input_ptr.add(1 * 8));
    in_[2] = load_input_data8(input_ptr.add(2 * 8));
    in_[3] = load_input_data8(input_ptr.add(3 * 8));
    in_[4] = load_input_data8(input_ptr.add(4 * 8));
    in_[5] = load_input_data8(input_ptr.add(5 * 8));
    in_[6] = load_input_data8(input_ptr.add(6 * 8));
    in_[7] = load_input_data8(input_ptr.add(7 * 8));

    // ---- 2-D pass — libvpx `vpx_idct8_sse2` loop body, twice. ----
    //
    // Each call is `transpose_16bit_8x8` + `idct8`. The first call
    // makes lanes correspond to rows (and registers to columns) for
    // the row pass; the second call swaps back for the column pass.
    for _ in 0..2 {
        let src = in_;
        transpose_16bit_8x8(&src, &mut in_);
        let pre = in_;
        idct8(&pre, &mut in_);
    }

    // ---- write_buffer_8x8: per-row (+16, >>5) + residual add. ----
    let dest_ptr = dst.as_mut_ptr();
    recon_and_store_row(dest_ptr.add(0 * stride), in_[0]);
    recon_and_store_row(dest_ptr.add(1 * stride), in_[1]);
    recon_and_store_row(dest_ptr.add(2 * stride), in_[2]);
    recon_and_store_row(dest_ptr.add(3 * stride), in_[3]);
    recon_and_store_row(dest_ptr.add(4 * stride), in_[4]);
    recon_and_store_row(dest_ptr.add(5 * stride), in_[5]);
    recon_and_store_row(dest_ptr.add(6 * stride), in_[6]);
    recon_and_store_row(dest_ptr.add(7 * stride), in_[7]);
}

/// Safe wrapper for the SSE2 i16-lane dense `idct8x8_64_add` kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct8x8_64_add`] and the existing
/// AVX2 wrapper [`super::idct8x8_avx2::idct8x8_add_avx2`], so the
/// `SimdDispatch::idct8x8_add` slot can hold any of them.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly for SSE2-vs-scalar differential
/// testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct8x8_add_sse2_64(coeffs: &[i32; 64], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is architecturally universal on
    // x86_64 (mandated by both SysV AMD64 and MS x64 ABIs), so the
    // gate is essentially free. The unit-test path gates each call on
    // a runtime `is_x86_feature_detected!("sse2")` check.
    unsafe { idct8x8_add_sse2_64_inner(coeffs, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Differential tests for `idct8x8_add_sse2_64`.
    //!
    //! All proptest strategies bound coefficients to ≤ i12 magnitude so
    //! that every 1-D `idct8` butterfly intermediate stays within i16
    //! range — the SSE2 kernel's saturating `_mm_packs_epi32` then
    //! agrees bit-exactly with the modular `wraplow`-truncating scalar
    //! reference. The pathological "i16 extremes" strategies are
    //! deliberately omitted because they trip the saturation
    //! divergence (see module docstring).
    //!
    //! Sub-tests:
    //! 1. Directed fixtures (all-zero, DC impulse, single-coef sweep).
    //! 2. Proptest (i12 envelope, full 64-coef density).
    //! 3. Dst-pointer-offset regression guard.

    use super::*;
    use crate::transforms::inverse::idct8x8_64_add;
    use proptest::prelude::*;

    macro_rules! require_sse2 {
        () => {
            if !is_x86_feature_detected!("sse2") {
                return;
            }
        };
    }

    fn run_sse2(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        assert!(is_x86_feature_detected!("sse2"));
        let mut dst = *dst_init;
        idct8x8_add_sse2_64(input, &mut dst, 8);
        dst
    }

    fn run_scalar(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        let mut dst = *dst_init;
        idct8x8_64_add(input, &mut dst, 8);
        dst
    }

    // -----------------------------------------------------------------
    // Directed fixtures.
    // -----------------------------------------------------------------

    #[test]
    fn idct8x8_sse2_64_all_zero_input() {
        require_sse2!();
        let input = [0i32; 64];
        let dst_init = [42u8; 64];
        let sse2 = run_sse2(&input, &dst_init);
        let scalar = run_scalar(&input, &dst_init);
        assert_eq!(sse2, scalar);
        // All-zero input ⇒ rounding (+16 >> 5) == 0, so dst unchanged.
        assert_eq!(sse2, dst_init);
    }

    /// DC-only impulse — only `input[0]` non-zero.
    #[test]
    fn idct8x8_sse2_64_dc_impulse() {
        require_sse2!();
        for &val in &[100i32, -100, 2000, -2000, 1, -1] {
            let mut input = [0i32; 64];
            input[0] = val;
            let dst_init = [128u8; 64];
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar(&input, &dst_init);
            assert_eq!(sse2, scalar, "val={val}");
        }
    }

    /// Single-coefficient impulse sweep over every position.
    #[test]
    fn idct8x8_sse2_64_single_coefficient_impulses() {
        require_sse2!();
        for pos in 0..64 {
            for &val in &[100i32, -100, 1000, -1000, 1, -1] {
                let mut input = [0i32; 64];
                input[pos] = val;
                let dst_init = [128u8; 64];
                let sse2 = run_sse2(&input, &dst_init);
                let scalar = run_scalar(&input, &dst_init);
                assert_eq!(sse2, scalar, "pos={pos} val={val}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Proptest strategies — i12 envelope, where the saturating-pack
    // and modular wraplow agree bit-exactly.
    // -----------------------------------------------------------------

    /// `uniform_strategy_12` — full-density i12 coefficients in every
    /// position.
    fn uniform_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(-2048i32..=2047, 64).prop_map(|v| {
            let mut out = [0i32; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    /// `sparse_strategy_12` — 1..=8 non-zero coefs at arbitrary raster
    /// positions.
    fn sparse_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec((0usize..64, -2048i32..=2047), 1..=8).prop_map(|nz| {
            let mut out = [0i32; 64];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 64]> {
        prop::collection::vec(any::<u8>(), 64).prop_map(|v| {
            let mut out = [0u8; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct8x8_sse2_64_eq_scalar_uniform_12(
            input in uniform_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }

        #[test]
        fn idct8x8_sse2_64_eq_scalar_sparse_12(
            input in sparse_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }
    }

    // -----------------------------------------------------------------
    // Dst-pointer-offset correctness regression guard. Mirrors the
    // AVX2 idct8x8 test of the same name. The kernel uses
    // `_mm_loadl_epi64` (8-byte, alignment-free) so this is a per-
    // offset arithmetic regression guard, NOT an alignment-fault test.
    // -----------------------------------------------------------------

    #[test]
    fn idct8x8_sse2_64_misaligned_dst_pointers() {
        require_sse2!();
        // Synthetic input — moderate-magnitude across the full 8×8.
        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 17) - 400;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 80]> = Box::new([0u8; 80]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;

            idct8x8_add_sse2_64(&input, &mut buf[skew..skew + 64], 8);
            idct8x8_64_add(&input, &mut scalar_buf[skew..skew + 64], 8);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "SSE2/scalar mismatch at skew={skew}: per-offset arithmetic regression"
            );
        }
    }
}
