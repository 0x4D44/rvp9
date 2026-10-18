//! `plane_dequant_finisher` — x86_64 AVX2 implementation (Wave H H6).
//!
//! Sibling of [`crate::quant::plane_dequant_finisher`] (the scalar
//! reference). Drives pass 2 of the Wave H H5 split detokenize loop in
//! [`crate::entropy::detokenize::decode_coefs`]: gather
//! `(val[i], dqv[i])` from the per-block scratch arrays, multiply,
//! arithmetic-right-shift by `dq_shift`, conditionally negate by
//! `sign[i]`, and scatter the result to `dq_coefs_out[raster[i]]`.
//!
//! ## Vectorisation
//!
//! 8 i32 lanes per iteration; tail of `< 8` entries falls through to
//! the scalar reference. The hot path is:
//!
//! 1. `_mm_loadu_si128` 8 i16 lanes of `val`, sign-extend to 8 i32
//!    lanes via `_mm256_cvtepi16_epi32` on the low 128 bits.
//! 2. Same for `dqv`.
//! 3. `_mm256_mullo_epi32` — the product fits in i32 (max
//!    `16450 * 32767 ≈ 5.4e8`, comfortably below `i32::MAX ≈ 2.15e9`).
//! 4. `_mm256_sra_epi32` by `dq_shift` (`0` or `1` per VP9 spec).
//! 5. Sign-conditional negate: load 8 sign bytes, expand to an i32
//!    `_mm256i` of `{0, 1}`, then compute `sign_arg = 1 - 2*sign` so
//!    `sign=0 → +1` and `sign=1 → -1`, then `_mm256_sign_epi32(mag,
//!    sign_arg)` to flip lanes whose sign byte is set.
//! 6. Scatter via 8 extract-and-store steps (AVX2 has no native i32
//!    scatter; AVX-512 would use `_mm512_i32scatter_epi32`).
//!
//! The output array is pre-zeroed by the caller (the
//! `dq_coefs_out[..seg_eob]` loop at the top of `decode_coefs`), so
//! the scatter is a pure WRITE — no read-modify-write hazard.
//!
//! ## Bit-exactness vs scalar reference
//!
//! `_mm256_mullo_epi32` returns the low 32 bits of the i32×i32 → i64
//! product. Because `(val * dqv)` always fits in i32 (per the bound
//! above), the low 32 bits ARE the full result. `_mm256_sra_epi32` is
//! the lane-wise arithmetic right shift matching scalar `>>` on i32.
//! `_mm256_sign_epi32(a, b)`: per Intel docs, `b > 0 → a`, `b < 0 →
//! -a`, `b == 0 → 0`. We feed `b = +1` for `sign=0` and `b = -1` for
//! `sign=1`, never `0` — so the `-a` lane matches the scalar `-mag`
//! exactly and the `+a` lane is just `a`.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads are from `scratch_*` arrays (unaligned scratch vectors);
//! the kernel does NOT read from `dq_coefs_out` (the output is
//! pre-zeroed and write-only). The grep sentinel below documents the
//! actual load discipline used in the inner kernel.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_extract_epi16, _mm_extract_epi32, _mm_loadl_epi64,
    _mm_loadu_si128, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
    _mm256_cvtepu8_epi32, _mm256_extracti128_si256, _mm256_mullo_epi32, _mm256_set1_epi32,
    _mm256_sign_epi32, _mm256_sra_epi32, _mm256_sub_epi32,
};

/// AVX2 inner kernel. Processes `len = scratch_raster.len()` entries
/// in batches of 8; the `< 8` tail falls back to the scalar reference.
///
/// # Safety
///
/// All four scratch slices must have the same length. Every raster
/// index in `scratch_raster[..]` must be `< dq_coefs_out.len()`. The
/// CPU must support AVX2 (caller is the safe wrapper, which is only
/// installed by `SimdDispatch::build` when `caps.avx2 == true`).
// G4: loads from scratch_* use _mm_loadu_si128 / _mm_loadl_epi64
// (unaligned); dq_coefs_out is write-only (pre-zeroed by caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn plane_dequant_finisher_avx2_inner(
    scratch_raster: &[u16],
    scratch_dqv: &[i16],
    scratch_val: &[i16],
    scratch_sign: &[u8],
    dq_shift: u32,
    dq_coefs_out: &mut [i32],
) {
    debug_assert_eq!(scratch_raster.len(), scratch_dqv.len());
    debug_assert_eq!(scratch_raster.len(), scratch_val.len());
    debug_assert_eq!(scratch_raster.len(), scratch_sign.len());

    let len = scratch_raster.len();
    let n_vec = len & !7; // floor(len / 8) * 8

    // dq_shift is a scalar count; AVX2 SRA uses an __m128i count.
    let shift_count: __m128i = _mm_cvtsi32_si128(dq_shift as i32);

    // `one_v` lane-constant for the (1 - 2*sign) sign-arg formula.
    let one_v: __m256i = _mm256_set1_epi32(1);

    let val_ptr = scratch_val.as_ptr();
    let dqv_ptr = scratch_dqv.as_ptr();
    let sign_ptr = scratch_sign.as_ptr();
    let raster_ptr = scratch_raster.as_ptr();

    let mut i = 0usize;
    while i < n_vec {
        // ---- Load val[i..i+8] as 8 i16 lanes, sign-extend to i32. ----
        let val_i16: __m128i = _mm_loadu_si128(val_ptr.add(i) as *const __m128i);
        let val_i32: __m256i = _mm256_cvtepi16_epi32(val_i16);

        // ---- Load dqv[i..i+8] as 8 i16 lanes, sign-extend to i32. ----
        let dqv_i16: __m128i = _mm_loadu_si128(dqv_ptr.add(i) as *const __m128i);
        let dqv_i32: __m256i = _mm256_cvtepi16_epi32(dqv_i16);

        // ---- Multiply lane-wise (low 32 bits of i64 product; fits). ----
        let prod: __m256i = _mm256_mullo_epi32(val_i32, dqv_i32);

        // ---- Arithmetic right shift by dq_shift. ----
        let mag: __m256i = _mm256_sra_epi32(prod, shift_count);

        // ---- Sign-conditional negate. ----
        // Load 8 sign bytes as zero-extended i32 lanes (sign ∈ {0,1}).
        let sign_u8: __m128i = _mm_loadl_epi64(sign_ptr.add(i) as *const __m128i);
        let sign_i32: __m256i = _mm256_cvtepu8_epi32(sign_u8);
        // Build `sign_arg = 1 - 2*sign`, lane-wise: sign=0 → +1, sign=1 → -1.
        let two_sign: __m256i = _mm256_add_epi32(sign_i32, sign_i32);
        let sign_arg: __m256i = _mm256_sub_epi32(one_v, two_sign);

        // _mm256_sign_epi32: sign_arg > 0 → mag; sign_arg < 0 → -mag.
        // sign_arg is exactly +1 or -1 here, never 0.
        let signed: __m256i = _mm256_sign_epi32(mag, sign_arg);

        // ---- Scatter ----
        // AVX2 has no i32 scatter; split into two __m128i halves and
        // extract+store 8 lanes serially. The output is pre-zeroed by
        // the caller so each write is the final value at that index.
        let lo: __m128i = _mm256_castsi256_si128(signed);
        let hi: __m128i = _mm256_extracti128_si256(signed, 1);

        // Raster indices are 8 u16 values, one __m128i load.
        let raster_v: __m128i = _mm_loadu_si128(raster_ptr.add(i) as *const __m128i);

        // Extract 8 raster indices and 8 i32 magnitudes; store.
        // `_mm_extract_epi16` returns i32 zero-extended from u16.
        let r0 = _mm_extract_epi16(raster_v, 0) as usize;
        let r1 = _mm_extract_epi16(raster_v, 1) as usize;
        let r2 = _mm_extract_epi16(raster_v, 2) as usize;
        let r3 = _mm_extract_epi16(raster_v, 3) as usize;
        let r4 = _mm_extract_epi16(raster_v, 4) as usize;
        let r5 = _mm_extract_epi16(raster_v, 5) as usize;
        let r6 = _mm_extract_epi16(raster_v, 6) as usize;
        let r7 = _mm_extract_epi16(raster_v, 7) as usize;

        let v0 = _mm_extract_epi32(lo, 0);
        let v1 = _mm_extract_epi32(lo, 1);
        let v2 = _mm_extract_epi32(lo, 2);
        let v3 = _mm_extract_epi32(lo, 3);
        let v4 = _mm_extract_epi32(hi, 0);
        let v5 = _mm_extract_epi32(hi, 1);
        let v6 = _mm_extract_epi32(hi, 2);
        let v7 = _mm_extract_epi32(hi, 3);

        // Direct writes: dq_coefs_out is pre-zeroed and every raster
        // index lies inside `0..seg_eob <= dq_coefs_out.len()` by the
        // detokenize caller contract.
        debug_assert!(r0 < dq_coefs_out.len());
        debug_assert!(r1 < dq_coefs_out.len());
        debug_assert!(r2 < dq_coefs_out.len());
        debug_assert!(r3 < dq_coefs_out.len());
        debug_assert!(r4 < dq_coefs_out.len());
        debug_assert!(r5 < dq_coefs_out.len());
        debug_assert!(r6 < dq_coefs_out.len());
        debug_assert!(r7 < dq_coefs_out.len());
        *dq_coefs_out.get_unchecked_mut(r0) = v0;
        *dq_coefs_out.get_unchecked_mut(r1) = v1;
        *dq_coefs_out.get_unchecked_mut(r2) = v2;
        *dq_coefs_out.get_unchecked_mut(r3) = v3;
        *dq_coefs_out.get_unchecked_mut(r4) = v4;
        *dq_coefs_out.get_unchecked_mut(r5) = v5;
        *dq_coefs_out.get_unchecked_mut(r6) = v6;
        *dq_coefs_out.get_unchecked_mut(r7) = v7;

        i += 8;
    }

    // ---- Tail: < 8 entries. ----
    if i < len {
        crate::quant::plane_dequant_finisher(
            &scratch_raster[i..],
            &scratch_dqv[i..],
            &scratch_val[i..],
            &scratch_sign[i..],
            dq_shift,
            dq_coefs_out,
        );
    }
}

/// Safe wrapper for the AVX2 `plane_dequant_finisher` kernel.
///
/// Signature mirrors [`crate::quant::plane_dequant_finisher`]. The
/// dispatch slot
/// [`crate::dsp::SimdDispatch::plane_dequant_finisher`] points at this
/// wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn plane_dequant_finisher_avx2(
    scratch_raster: &[u16],
    scratch_dqv: &[i16],
    scratch_val: &[i16],
    scratch_sign: &[u8],
    dq_shift: u32,
    dq_coefs_out: &mut [i32],
) {
    assert_eq!(
        scratch_raster.len(),
        scratch_dqv.len(),
        "dequant scratch length mismatch"
    );
    assert_eq!(
        scratch_raster.len(),
        scratch_val.len(),
        "dequant scratch length mismatch"
    );
    assert_eq!(
        scratch_raster.len(),
        scratch_sign.len(),
        "dequant scratch length mismatch"
    );
    assert!(
        scratch_raster
            .iter()
            .all(|&r| usize::from(r) < dq_coefs_out.len()),
        "dequant raster index out of bounds"
    );
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Slice-length parity and raster-index
    // bounds are checked above before any vector load or unchecked
    // scatter store can execute.
    unsafe {
        plane_dequant_finisher_avx2_inner(
            scratch_raster,
            scratch_dqv,
            scratch_val,
            scratch_sign,
            dq_shift,
            dq_coefs_out,
        )
    }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::quant` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `plane_dequant_finisher`.

    use super::plane_dequant_finisher_avx2;
    use crate::quant::plane_dequant_finisher;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Run scalar and AVX2 over the same scratch input and return both
    /// output buffers for comparison.
    fn run_pair(
        scratch_raster: &[u16],
        scratch_dqv: &[i16],
        scratch_val: &[i16],
        scratch_sign: &[u8],
        dq_shift: u32,
        out_len: usize,
    ) -> (Vec<i32>, Vec<i32>) {
        let mut a = vec![0i32; out_len];
        let mut s = vec![0i32; out_len];
        plane_dequant_finisher(
            scratch_raster,
            scratch_dqv,
            scratch_val,
            scratch_sign,
            dq_shift,
            &mut s,
        );
        plane_dequant_finisher_avx2(
            scratch_raster,
            scratch_dqv,
            scratch_val,
            scratch_sign,
            dq_shift,
            &mut a,
        );
        (s, a)
    }

    /// Empty input is a no-op for both paths.
    #[test]
    fn empty_input_no_op() {
        require_avx2!();
        let (s, a) = run_pair(&[], &[], &[], &[], 0, 1024);
        assert_eq!(s, a);
        assert_eq!(s, vec![0i32; 1024]);
    }

    /// Sub-vector input (< 8 entries) exercises the scalar tail.
    #[test]
    fn tail_only_path() {
        require_avx2!();
        let raster = [3u16, 7, 11, 0, 5];
        let dqv = [177i16, 231, 100, 4, 1828];
        let val = [1i16, 2, 16450, 67, 11];
        let sign = [0u8, 1, 0, 1, 0];
        let (s, a) = run_pair(&raster, &dqv, &val, &sign, 0, 32);
        assert_eq!(s, a);
    }

    /// A multiple-of-8 input exercises the AVX2 main loop with no
    /// scalar tail. Use distinct raster indices to verify the scatter.
    #[test]
    fn multiple_of_eight() {
        require_avx2!();
        let raster: Vec<u16> = (0..16).collect();
        let dqv: Vec<i16> = (0..16).map(|i| 50 + i as i16).collect();
        let val: Vec<i16> = (0..16).map(|i| 1 + i as i16).collect();
        let sign: Vec<u8> = (0..16).map(|i| (i & 1) as u8).collect();
        let (s, a) = run_pair(&raster, &dqv, &val, &sign, 0, 64);
        assert_eq!(s, a);
    }

    /// dq_shift = 1 exercises the Tx32x32 path.
    #[test]
    fn dq_shift_one_path() {
        require_avx2!();
        let raster: Vec<u16> = (0..11).collect();
        let dqv: Vec<i16> = (0..11).map(|i| 1000 + i as i16 * 7).collect();
        let val: Vec<i16> = (0..11).map(|i| 7 + i as i16 * 3).collect();
        let sign: Vec<u8> = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let (s, a) = run_pair(&raster, &dqv, &val, &sign, 1, 64);
        assert_eq!(s, a);
    }

    /// Maximum values across a representative VP9 range — confirm no
    /// i32 overflow in the AVX2 path.
    #[test]
    fn max_values_no_overflow() {
        require_avx2!();
        // CAT6_MIN_VAL + (2^14 - 1) = 67 + 16383 = 16450 is the
        // max `val`; AC dequant max is 1828. (sign bit independent.)
        let n = 16usize;
        let raster: Vec<u16> = (0..n as u16).collect();
        let dqv: Vec<i16> = vec![1828i16; n];
        let val: Vec<i16> = vec![16450i16; n];
        let sign: Vec<u8> = (0..n).map(|i| (i & 1) as u8).collect();
        let (s, a) = run_pair(&raster, &dqv, &val, &sign, 1, n);
        assert_eq!(s, a);
        // Confirm magnitude is plausible (val * dqv >> 1).
        let expected_mag = (16450 * 1828) >> 1;
        assert_eq!(s[0], expected_mag);
        assert_eq!(s[1], -expected_mag);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Differential proptest over random scratch arrays of length
        /// up to 32 (covers the typical 4x4..16x16 EOB range) and
        /// dq_shift in {0, 1}.
        #[test]
        fn scalar_eq_avx2_proptest(
            len in 0usize..=32,
            seed in any::<u32>(),
            dq_shift in 0u32..=1,
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            // Deterministic LCG fill keyed on `seed`. The raster
            // indices are forced distinct in [0, 64) so the scatter
            // pattern is exercised without aliasing.
            let mut s = seed;
            let mut next = || {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                s
            };
            // Distinct rasters by shuffle.
            let mut raster: Vec<u16> = (0..64u16).collect();
            for i in (1..raster.len()).rev() {
                let j = (next() as usize) % (i + 1);
                raster.swap(i, j);
            }
            raster.truncate(len);
            let dqv: Vec<i16> = (0..len).map(|_| {
                // dqv values are non-negative per VP9 spec, range
                // 4..=1828 for AC, 4..=1336 for DC. We sample inside
                // [0, 1828] (0 included as an edge case).
                (next() % 1829) as i16
            }).collect();
            let val: Vec<i16> = (0..len).map(|_| {
                // val ∈ [1, 16450].
                (1 + (next() % 16450)) as i16
            }).collect();
            let sign: Vec<u8> = (0..len).map(|_| (next() & 1) as u8).collect();

            let (sc, av) = run_pair(&raster, &dqv, &val, &sign, dq_shift, 64);
            prop_assert_eq!(sc, av);
        }
    }

    /// Tautology guard: the scalar reference must actually WRITE to
    /// the output buffer when given non-empty input — proves we're
    /// not silently returning the zeroed init.
    #[test]
    fn tautology_guard_scalar_writes() {
        let raster = [5u16, 2];
        let dqv = [177i16, 231];
        let val = [3i16, 11];
        let sign = [0u8, 1];
        let mut out = vec![0i32; 16];
        plane_dequant_finisher(&raster, &dqv, &val, &sign, 0, &mut out);
        assert_ne!(out[5], 0, "scalar finisher must write the raster[0] index");
        assert_ne!(out[2], 0, "scalar finisher must write the raster[1] index");
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// dispatches produce identical results.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps);
        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_s);

        let raster: Vec<u16> = (0..12).collect();
        let dqv: Vec<i16> = (0..12).map(|i| 100 + i as i16 * 5).collect();
        let val: Vec<i16> = (0..12).map(|i| 1 + i as i16).collect();
        let sign: Vec<u8> = (0..12).map(|i| (i & 1) as u8).collect();

        let mut a = vec![0i32; 16];
        let mut s = vec![0i32; 16];
        (d_avx2.plane_dequant_finisher())(&raster, &dqv, &val, &sign, 0, &mut a);
        (d_scalar.plane_dequant_finisher())(&raster, &dqv, &val, &sign, 0, &mut s);
        assert_eq!(a, s);
    }
}
