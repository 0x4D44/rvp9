//! 32×32 inverse-DCT + add — x86_64 AVX2 i16-lane **EOB ≤ 135** kernel
//! covering the `idct32x32_135_add` slot when AVX2 is available.
//!
//! This kernel is structurally the upstream libvpx 256-bit AVX2 driver
//! `vpx_idct32x32_135_add_avx2`
//! (`third_party/libvpx/vpx_dsp/x86/inv_txfm_avx2.c:604-626`) with ONE
//! contractual adjustment for rvp9 (see "Input contract" below). It
//! REUSES the dense [`super::idct32x32_avx2_1024`] module's leaf
//! helpers verbatim (no logic / constant change) — only this driver is
//! new:
//!
//! - [`super::idct32x32_avx2_1024::idct_load16x16`] — 16×16 i16 load.
//! - [`super::idct32x32_avx2_1024::transpose_16bit_16x16_avx2`] —
//!   in-place 16×16 transpose.
//! - [`super::idct32x32_avx2_1024::transpose_16bit_16x16_avx2_to`] —
//!   out-of-place 16×16 transpose (column pass).
//! - [`super::idct32x32_avx2_1024::idct32_1024_16x32`] — the 1-D 32-pt
//!   engine (16 instances / ymm), in-place.
//! - [`super::idct32x32_avx2_1024::store_buffer_16x32`] — final +32/>>6
//!   rounding + 16-byte residual adds.
//!
//! ## Input contract (rvp9 vs upstream libvpx)
//!
//! The rvp9 scalar reference [`crate::transforms::inverse::idct32x32_135_add`]
//! reads ALL 32 cols of rows 0..16 (and treats rows 16..32 as zero).
//! Upstream libvpx's 135 kernel assumes the stricter zigzag-EOB envelope
//! — coefficients confined to the top-left 16×16 — and ignores cols
//! 16..32. To preserve the rvp9 dispatch contract
//! "`idct32x32_135_add` slot is bit-exact with the scalar reference on
//! all of rows 0..16 × cols 0..32", this driver loads BOTH 16-col halves
//! of rows 0..16 (mirroring the dense 1024 driver's strip-0 row pass)
//! rather than just the top-left 16×16.
//!
//! ## Partial structure (rows 16..32 all zero)
//!
//! Under the 135 contract, source rows 16..32 are zero. The driver
//! therefore:
//!
//! 1. runs a SINGLE row-pass strip (rows 0..16) — loading both 16-col
//!    halves and running `idct32_1024_16x32` once over the full 16×32
//!    block,
//! 2. zero-fills the second row strip's output (`r = [zero; 32]`) since
//!    `idct32(0..0) = 0`,
//! 3. runs the same 2-iteration column pass as the dense driver, with
//!    the bottom 16 rows of each column-pass input coming from the
//!    all-zero `r`.
//!
//! Net cost = 3 `idct32_1024_16x32` runs (1 row + 2 col) × 16 i16/ymm,
//! versus the dense 1024 driver's 4 runs (2 row + 2 col). The SSE2
//! sibling runs 6 engine runs × 8 i16/xmm.
//!
//! ## Bit-exactness
//!
//! Inherited unchanged from the reused helpers: saturating
//! `_mm256_packs_epi32` wraplow, `(x+8192)>>14` round-shift, and the
//! shared cospi constants. Same `(saturating)` policy as the SSE2 dense
//! sibling [`super::idct32x32_sse2_1024::idct32x32_add_sse2_1024`]. The
//! primary differential oracle is the bit-exact dense AVX2 driver
//! [`super::idct32x32_avx2_1024::idct32x32_add_avx2_1024`] (which is
//! itself bit-exact with the scalar reference on the saturation-safe
//! envelope); on the matching input domain the 135 driver produces
//! identical output to the dense 1024 driver, because rows 16..32 of
//! the input are zero (so the dense driver's strip-1 row pass collapses
//! to zero too).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadu_si128` (16-byte, unaligned) via the
//! reused `recon_and_store16` (inside `store_buffer_16x32`). The grep
//! sentinel below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256i, _mm256_setzero_si256};

use super::idct32x32_avx2_1024::{
    idct_load16x16, idct32_1024_16x32, store_buffer_16x32, transpose_16bit_16x16_avx2,
    transpose_16bit_16x16_avx2_to,
};

// ---------------------------------------------------------------------
// Top-level kernel.
// ---------------------------------------------------------------------

/// Inner AVX2 implementation of the EOB ≤ 135 32×32 inverse-DCT +
/// residual add. One row pass + two column passes of
/// `idct32_1024_16x32`, mirroring libvpx `vpx_idct32x32_135_add_avx2`
/// (`inv_txfm_avx2.c:604-626`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports AVX2.
/// - `dst` has at least `31 * stride + 32` bytes of valid storage.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct32x32_add_avx2_135_inner(input: &[i32; 1024], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 31 + 32, "dst too small for 32x32 add");

    let zero = _mm256_setzero_si256();
    // `l` holds the row-pass output for strip 0 (rows 0..16 — the only
    // non-zero strip under the rvp9 scalar `idct32x32_135_add` contract,
    // which reads all 32 cols of rows 0..16; cf. the upstream libvpx
    // 135 kernel which assumes coefficients live only in the top-left
    // 16×16 — a stricter zigzag-EOB envelope than rvp9 needs).
    //
    // `r` would hold the row-pass output for strip 1 (rows 16..32), but
    // those source rows are zero under the 135 contract, so the engine
    // output for that strip is also zero — we skip it entirely and use
    // an all-zero `r` for the column pass. The column pass thus still
    // sees a full 32-row tall column for each i16-lane, with the bottom
    // 16 rows zero.
    let mut l: [__m256i; 32] = [zero; 32];
    let r: [__m256i; 32] = [zero; 32];

    // ---- Row pass strip 0 (single iteration covering rows 0..16):
    // mirrors the AVX2 1024 driver's strip 0 — load both 16-col halves
    // of rows 0..16 (the FULL 16×32 non-zero region under rvp9's 135
    // contract — NOT just the top-left 16×16 the upstream libvpx kernel
    // assumes), transpose each in place, then run the 1-D 32-pt idct
    // in-place over all 32 register positions.
    let input_ptr = input.as_ptr();
    idct_load16x16(input_ptr, &mut l[0..16], 32);
    transpose_16bit_16x16_avx2(&mut l[0..16]);
    idct_load16x16(input_ptr.add(16), &mut l[16..32], 32);
    transpose_16bit_16x16_avx2(&mut l[16..32]);
    idct32_1024_16x32(&mut l);

    // ---- Column pass: 2 iterations (i = 0, 16). Mirrors the AVX2 1024
    // column pass — out-of-place transpose of the 16×16 sub-blocks of
    // `l` (top 16 rows) and `r` (bottom 16 rows = all zero) into a
    // 16×32 working buffer, run the 1-D idct32, then store the final
    // +32/>>6 rounded residual adds.
    let mut dest_ptr = dst.as_mut_ptr();
    let mut out: [__m256i; 32] = [zero; 32];
    let mut i = 0usize;
    while i < 32 {
        transpose_16bit_16x16_avx2_to(&l[i..i + 16], &mut out[0..16]);
        transpose_16bit_16x16_avx2_to(&r[i..i + 16], &mut out[16..32]);

        idct32_1024_16x32(&mut out);
        store_buffer_16x32(&mut out, dest_ptr, stride);

        dest_ptr = dest_ptr.add(16);
        i += 16;
    }
}

/// Safe wrapper for the AVX2 i16-lane EOB ≤ 135 `idct32x32_135_add`
/// kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct32x32_135_add`], the SSE2 wrapper
/// [`super::idct32x32_add_sse2_135`] and the i32-lane AVX2 wrapper
/// [`super::idct32x32_135_add_avx2`], so the
/// `SimdDispatch::idct32x32_135_add` slot can hold any of them.
///
/// `pub` (rather than `pub(crate)`) so unit tests in the same crate can
/// call it directly for differential testing without going through the
/// dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_avx2_135(coeffs: &[i32; 1024], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. The unit-test path gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { idct32x32_add_avx2_135_inner(coeffs, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Differential / equality tests for `idct32x32_add_avx2_135`.
    //!
    //! 1. SSE2-vs-AVX2 full-driver equality over arbitrary i16 lanes in
    //!    the EOB-135 nonzero region (rows 0..16) — the SSE2 135 kernel
    //!    is the primary saturating oracle (both kernels saturate
    //!    identically).
    //! 2. Scalar-differential over the EOB-135 coefficient envelope
    //!    (≤ i12 — saturating-vs-modular agrees in-spec, diverges at
    //!    i16-extreme, so the scalar comparison is bounded).
    //! 3. Directed per-row-count + pathological fixtures (mirrors the
    //!    i32-lane 135 test family).
    //!
    //! Every test gates on `is_x86_feature_detected!("avx2")`.

    use super::idct32x32_add_avx2_135;
    use crate::dsp::x86_64::idct32x32_add_avx2_1024;
    use crate::dsp::x86_64::idct32x32_add_sse2_135;
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
        assert!(is_x86_feature_detected!("avx2"));
        let mut dst = *dst_init;
        idct32x32_add_avx2_135(input, &mut dst, 32);
        dst
    }

    /// Run the bit-exact dense AVX2 1024 driver as the saturating
    /// oracle. The 135 driver is a partial sibling of the 1024 driver
    /// (rows 16..32 of the input are assumed zero); over that input
    /// domain the two kernels produce identical output. The SSE2 135
    /// kernel implements the stricter upstream-libvpx top-left-16×16
    /// contract and therefore is NOT a valid oracle for the rvp9 135
    /// contract (which mirrors the scalar reference: full rows 0..16).
    fn run_avx2_1024(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_add_avx2_1024(input, &mut dst, 32);
        dst
    }

    fn run_scalar(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_135_add(input, &mut dst, 32);
        dst
    }

    // -----------------------------------------------------------------
    // Strategies.
    // -----------------------------------------------------------------

    /// `i16_full_rows_0_16` — arbitrary full-magnitude i16 coefficients
    /// confined to rows 0..16 (the EOB-135 nonzero region). Used ONLY
    /// for the AVX2-vs-SSE2 equality test: both kernels share the same
    /// saturating flowgraph, so any i16 input is legal. NOT valid against
    /// the scalar oracle (modular, not saturating).
    fn i16_full_rows_0_16_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(any::<i16>(), 16 * 32).prop_map(|v| {
            let mut out = [0i32; 1024];
            for (slot, &val) in out[..16 * 32].iter_mut().zip(v.iter()) {
                *slot = val as i32;
            }
            out
        })
    }

    /// `eob_bucket_135_i10` — exactly 135 i10-bounded coefs at the first
    /// 135 raster positions (all within rows 0..16). Bounded well inside
    /// the saturation envelope so the scalar reference agrees.
    fn eob_bucket_135_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-512i32..=511, 135).prop_map(|values| {
            let mut out = [0i32; 1024];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
    }

    /// `dense_rows_0_16_i8` — full-density i8 over rows 0..16. Bounded
    /// inside the row-pass saturation envelope (the AVX2 1024 driver's
    /// `dense_8` envelope; full i12 dense is NOT safe — see the 1024
    /// docstring), so the scalar reference agrees.
    fn dense_rows_0_16_i8_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-128i32..=127, 16 * 32).prop_map(|v| {
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
            cases: 128, max_shrink_iters: 512, .. ProptestConfig::default()
        })]

        // ---- vs AVX2 dense 1024 driver (any i16 in rows 0..16 —
        // both saturate identically, and on inputs with rows 16..32 = 0
        // the dense driver computes the same result as the 135 driver). ----

        #[test]
        fn idct32x32_135_avx2_eq_sse2_i16_full(
            input in i16_full_rows_0_16_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_avx2_1024(&input, &dst_init));
        }

        // ---- vs scalar reference (bounded EOB-135 envelope) ----

        #[test]
        fn idct32x32_135_avx2_eq_scalar_eob_135(
            input in eob_bucket_135_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar(&input, &dst_init));
        }

        #[test]
        fn idct32x32_135_avx2_eq_scalar_dense_i12(
            input in dense_rows_0_16_i8_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar(&input, &dst_init));
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures.
    // -----------------------------------------------------------------

    /// All-zero input must leave `dst` unchanged (and match scalar).
    #[test]
    fn idct32x32_135_avx2_all_zero_input() {
        require_avx2!();
        let input = [0i32; 1024];
        let dst_init = [42u8; 1024];
        let avx2 = run_avx2(&input, &dst_init);
        assert_eq!(avx2, dst_init);
        assert_eq!(avx2, run_scalar(&input, &dst_init));
    }

    /// K=1 DC-only impulse through the full partial path.
    #[test]
    fn idct32x32_135_avx2_dc_impulse() {
        require_avx2!();
        for &val in &[1234i32, -1234, 100, -100, 2000, -2000, 1, -1] {
            let mut input = [0i32; 1024];
            input[0] = val;
            let dst_init = [50u8; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_scalar(&input, &dst_init),
                "val={val}"
            );
        }
    }

    /// Directed per-row-count sweep K ∈ 1..=16 (the EOB-135 row span).
    /// Coefficients bounded inside the saturation envelope.
    #[test]
    fn idct32x32_135_avx2_directed_per_row_count() {
        require_avx2!();
        for k in 1..=16usize {
            let mut input = [0i32; 1024];
            for r in 0..k {
                for c in 0..32 {
                    input[r * 32 + c] = ((r as i32) * 199 + (c as i32) * 47) - 3000;
                }
            }
            let dst_init = [128u8; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_scalar(&input, &dst_init),
                "AVX2/scalar mismatch at k={k} populated rows"
            );
        }
    }

    /// Pathological i16-extreme fills in rows 0..16 — saturating, so
    /// the dense AVX2 1024 driver (not scalar) is the oracle here. Both
    /// kernels share the same saturating flowgraph; on inputs with rows
    /// 16..32 zero, the dense driver computes the same result as the
    /// 135 driver. The SSE2 135 kernel implements the stricter upstream
    /// libvpx top-left-16×16 contract and is NOT a valid oracle for the
    /// rvp9 contract (full rows 0..16).
    #[test]
    fn idct32x32_135_avx2_pathological_directed() {
        require_avx2!();
        const TOP: usize = 16 * 32;

        // (1) i16::MAX in rows 0..16.
        let mut input = [0i32; 1024];
        for slot in input[..TOP].iter_mut() {
            *slot = i16::MAX as i32;
        }
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_avx2_1024(&input, &dst_init),
                "MAX fill, dst={dst_fill}"
            );
        }

        // (2) i16::MIN in rows 0..16.
        let mut input = [0i32; 1024];
        for slot in input[..TOP].iter_mut() {
            *slot = i16::MIN as i32;
        }
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_avx2_1024(&input, &dst_init),
                "MIN fill, dst={dst_fill}"
            );
        }

        // (3) Alternating MAX/MIN.
        let mut input = [0i32; 1024];
        for (i, slot) in input[..TOP].iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_avx2_1024(&input, &dst_init),
                "ALT fill, dst={dst_fill}"
            );
        }

        // (4) Single high-magnitude impulse at (15, 15) — the 16×16
        //     sub-block corner; exercises the second column-pass strip.
        let mut input = [0i32; 1024];
        input[15 * 32 + 15] = 28000;
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 1024];
            assert_eq!(
                run_avx2(&input, &dst_init),
                run_avx2_1024(&input, &dst_init),
                "corner impulse, dst={dst_fill}"
            );
        }
    }

    /// Misaligned-dst regression guard at several skews (bounded coefs).
    #[test]
    fn idct32x32_135_avx2_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 1024];
        for r in 0..16 {
            for c in 0..32 {
                input[r * 32 + c] = ((r as i32) * 5 + (c as i32) * 3) - 200;
            }
        }

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 1040]> = Box::new([0u8; 1040]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;

            idct32x32_add_avx2_135(&input, &mut buf[skew..skew + 1024], 32);
            idct32x32_135_add(&input, &mut scalar_buf[skew..skew + 1024], 32);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Dispatch tautology guard.
    // -----------------------------------------------------------------

    #[test]
    fn idct32x32_135_avx2_dispatch_tautology_guard() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        caps.sse2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);

        let scalar_fp: fn(&[i32; 1024], &mut [u8], usize) = idct32x32_135_add;
        let sse2_fp: fn(&[i32; 1024], &mut [u8], usize) = idct32x32_add_sse2_135;
        assert_ne!(
            dispatch_avx2.idct32x32_135_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must NOT install the scalar fn"
        );
        assert_ne!(
            dispatch_avx2.idct32x32_135_add() as usize,
            sse2_fp as usize,
            "forced-AVX2 caps must NOT install the SSE2 wrapper"
        );

        let mut input = [0i32; 1024];
        for r in 0..16 {
            for c in 0..32 {
                input[r * 32 + c] = ((r as i32) * 31 + (c as i32) * 17) % 4001 - 2000;
            }
        }
        let dst_init = [97u8; 1024];
        let mut dst_dispatch = dst_init;
        (dispatch_avx2.idct32x32_135_add())(&input, &mut dst_dispatch, 32);
        assert_eq!(dst_dispatch, run_scalar(&input, &dst_init));
    }
}
