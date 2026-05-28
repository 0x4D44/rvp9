//! 16×16 inverse-DCT + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor (Wave F F2)**: both the row pass and the
//! column pass are now fully vectorised in AVX2 with 32-bit lanes,
//! following the pattern proven by the 8×8 G4-γ kernel in
//! [`crate::dsp::x86_64::idct8x8_avx2`]. The previous "finisher-only"
//! G4.2 pilot kept the scalar [`crate::transforms::inverse::idct16`]
//! call site and only AVX2-accelerated the final round/clamp/pack
//! stage. That pattern is preserved in the EOB fast paths
//! ([`crate::dsp::x86_64::idct16x16_10_avx2`],
//! [`crate::dsp::x86_64::idct16x16_38_avx2`]) and in
//! [`crate::dsp::x86_64::iht16x16_avx2`], all of which still call the
//! scalar `idct16` for the row pass.
//!
//! ## Bit-exactness vs scalar reference
//!
//! VP9 conformance requires bit-exact match to libvpx's C reference.
//! [`crate::transforms::inverse::idct16x16_256_add`] does row + column
//! 1-D `idct16` in i32 with explicit `wraplow` (`int16_t`-truncation)
//! at every intermediate store, then a final `(out + 32) >> 6`
//! ROUND_POWER_OF_TWO with `clip_pixel_add`. The vector form mirrors
//! every scalar operation in [`crate::transforms::inverse::idct16`]
//! 1:1 across 8 lanes (one lane per row of the 8-row sub-block we're
//! transforming), running the row pass twice (rows 0..8 and rows
//! 8..16) and the column pass twice (cols 0..8 and cols 8..16).
//!
//! Operand-range note: every intermediate is the result of either a
//! `wraplow` (hence i16-range) or an unwrapped raw input coefficient
//! (also spec-bounded to i16). Cospi constants are `< 2^15`. A single
//! product `wraplow × cospi` fits comfortably in `i32` (≈ 2^29). The
//! sum/difference of two such products fits in `i32` with margin
//! (≈ 2^30 < 2^31). [`_mm256_mullo_epi32`] returns the low 32 bits of
//! the product, exact for in-range operands. Therefore the scalar
//! reference's `i64` intermediate arithmetic can be folded down to
//! `i32` without divergence — same as in the 8×8 kernel.
//!
//! `wraplow` is `x as i16 as i32` (modular truncation, **not**
//! saturation). The vector equivalent is `slli_epi32(_, 16)` then
//! `srai_epi32(_, 16)` — see [`wrap`]. The use of [`_mm256_packs_epi32`]
//! (saturating) is restricted to the final round/clip/pack stage where
//! saturation is documented spec behaviour.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadu_si128` (16 bytes per
//! row); coefficient buffer reads via `_mm256_loadu_si256`. The grep
//! sentinel below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadu_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_packs_epi32,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_slli_epi32,
    _mm256_srai_epi32, _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_2_64, COSPI_4_64, COSPI_6_64, COSPI_8_64, COSPI_10_64, COSPI_12_64, COSPI_14_64,
    COSPI_16_64, COSPI_18_64, COSPI_20_64, COSPI_22_64, COSPI_24_64, COSPI_26_64, COSPI_28_64,
    COSPI_30_64,
};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers — duplicated verbatim from `idct8x8_avx2.rs`
// to keep this kernel self-contained. All are `#[inline]` so the
// compiler folds them away.
// ---------------------------------------------------------------------

/// `dct_const_round_shift(x) = (x + 8192) >> 14`, lane-parallel.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn drs(x: __m256i) -> __m256i {
    let round = _mm256_set1_epi32(1 << 13);
    _mm256_srai_epi32(_mm256_add_epi32(x, round), 14)
}

/// `wraplow(x) = x as i16 as i32`, lane-parallel. Sign-extends the low
/// 16 bits of each lane back to 32 bits — i.e., MODULAR i16 truncation,
/// not saturation. This is the critical correctness helper: replacing
/// it with `_mm256_packs_epi32` (saturating) diverges from the C
/// reference on pathological inputs.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. Product fits in i32 for in-range operands (see module
/// docstring).
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m256i, c: i32) -> __m256i {
    _mm256_mullo_epi32(x, _mm256_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 8×8 i32 transpose using a 3-stage unpack pyramid.
// Duplicated from `idct8x8_avx2::transpose8x8_i32`. We use this as the
// building block to transpose the 16×8 / 8×16 sub-blocks of the full
// 16×16 transform.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose8x8_i32(
    r0: __m256i,
    r1: __m256i,
    r2: __m256i,
    r3: __m256i,
    r4: __m256i,
    r5: __m256i,
    r6: __m256i,
    r7: __m256i,
) -> [__m256i; 8] {
    let a0 = _mm256_unpacklo_epi32(r0, r1);
    let a1 = _mm256_unpackhi_epi32(r0, r1);
    let a2 = _mm256_unpacklo_epi32(r2, r3);
    let a3 = _mm256_unpackhi_epi32(r2, r3);
    let a4 = _mm256_unpacklo_epi32(r4, r5);
    let a5 = _mm256_unpackhi_epi32(r4, r5);
    let a6 = _mm256_unpacklo_epi32(r6, r7);
    let a7 = _mm256_unpackhi_epi32(r6, r7);

    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);

    let c0 = _mm256_permute2x128_si256(b0, b4, 0x20);
    let c1 = _mm256_permute2x128_si256(b1, b5, 0x20);
    let c2 = _mm256_permute2x128_si256(b2, b6, 0x20);
    let c3 = _mm256_permute2x128_si256(b3, b7, 0x20);
    let c4 = _mm256_permute2x128_si256(b0, b4, 0x31);
    let c5 = _mm256_permute2x128_si256(b1, b5, 0x31);
    let c6 = _mm256_permute2x128_si256(b2, b6, 0x31);
    let c7 = _mm256_permute2x128_si256(b3, b7, 0x31);

    [c0, c1, c2, c3, c4, c5, c6, c7]
}

// ---------------------------------------------------------------------
// Vector 1D idct16. Each `__m256i` holds 8 i32 lanes corresponding to 8
// independent rows / columns being transformed in parallel; we take 16
// input ymm (one per input position 0..16) and return 16 output ymm
// (one per output position 0..16). The arithmetic mirrors
// `idct16` in `crate::transforms::inverse` line-by-line, including the
// stage-1 bit-reversed input gather (libvpx `idct16_c` :562-577).
//
// 7-stage butterfly:
//   stage 1 — bit-reversed input load (input[0,8,4,12,2,10,6,14,
//                                            1,9,5,13,3,11,7,15])
//   stage 2 — high-frequency cospi butterflies for positions 8..15
//   stage 3 — mid-frequency cospi butterflies for 4..7 and the
//             8..15 add/sub fan-out
//   stage 4 — DC + cospi-24/8 + 4..7 fan-out + 9/10/13/14 mixing
//   stage 5 — 0..3 cross-add + cospi-16 for 5,6 + 8..15 fan-out
//   stage 6 — 0..7 fan-out + cospi-16 for 10..13
//   stage 7 — final 0..15 cross-add
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct16_avx2(v: [__m256i; 16]) -> [__m256i; 16] {
    // ----- stage 1: bit-reversed input gather. -----
    // step1[k] = input[bitrev(k)]; values come straight from `v` with
    // no arithmetic — the gather is just a renaming of references.
    let s1_0 = v[0];
    let s1_1 = v[8];
    let s1_2 = v[4];
    let s1_3 = v[12];
    let s1_4 = v[2];
    let s1_5 = v[10];
    let s1_6 = v[6];
    let s1_7 = v[14];
    let s1_8 = v[1];
    let s1_9 = v[9];
    let s1_10 = v[5];
    let s1_11 = v[13];
    let s1_12 = v[3];
    let s1_13 = v[11];
    let s1_14 = v[7];
    let s1_15 = v[15];

    // ----- stage 2: high-frequency cospi butterflies. -----
    // step2[0..8] = step1[0..8] (pass-through, no rename needed since
    // these aren't reassigned at this stage).
    // step2[8]  = wrap(drs(step1[8]  * C30 - step1[15] * C2))
    // step2[15] = wrap(drs(step1[8]  * C2  + step1[15] * C30))
    let t1 = _mm256_sub_epi32(mul_const(s1_8, COSPI_30_64), mul_const(s1_15, COSPI_2_64));
    let t2 = _mm256_add_epi32(mul_const(s1_8, COSPI_2_64), mul_const(s1_15, COSPI_30_64));
    let s2_8 = wrap(drs(t1));
    let s2_15 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s1_9, COSPI_14_64), mul_const(s1_14, COSPI_18_64));
    let t2 = _mm256_add_epi32(mul_const(s1_9, COSPI_18_64), mul_const(s1_14, COSPI_14_64));
    let s2_9 = wrap(drs(t1));
    let s2_14 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s1_10, COSPI_22_64), mul_const(s1_13, COSPI_10_64));
    let t2 = _mm256_add_epi32(mul_const(s1_10, COSPI_10_64), mul_const(s1_13, COSPI_22_64));
    let s2_10 = wrap(drs(t1));
    let s2_13 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s1_11, COSPI_6_64), mul_const(s1_12, COSPI_26_64));
    let t2 = _mm256_add_epi32(mul_const(s1_11, COSPI_26_64), mul_const(s1_12, COSPI_6_64));
    let s2_11 = wrap(drs(t1));
    let s2_12 = wrap(drs(t2));

    // step2[0..8] pass through from step1; alias for readability.
    let s2_0 = s1_0;
    let s2_1 = s1_1;
    let s2_2 = s1_2;
    let s2_3 = s1_3;
    let s2_4 = s1_4;
    let s2_5 = s1_5;
    let s2_6 = s1_6;
    let s2_7 = s1_7;

    // ----- stage 3 -----
    // step1[0..4] = step2[0..4] (pass-through).
    let s3_0 = s2_0;
    let s3_1 = s2_1;
    let s3_2 = s2_2;
    let s3_3 = s2_3;

    // step1[4] = wrap(drs(step2[4] * C28 - step2[7] * C4))
    // step1[7] = wrap(drs(step2[4] * C4  + step2[7] * C28))
    let t1 = _mm256_sub_epi32(mul_const(s2_4, COSPI_28_64), mul_const(s2_7, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(s2_4, COSPI_4_64), mul_const(s2_7, COSPI_28_64));
    let s3_4 = wrap(drs(t1));
    let s3_7 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s2_5, COSPI_12_64), mul_const(s2_6, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(s2_5, COSPI_20_64), mul_const(s2_6, COSPI_12_64));
    let s3_5 = wrap(drs(t1));
    let s3_6 = wrap(drs(t2));

    // 8..15 fan-out (add/sub pairs).
    let s3_8 = wrap(_mm256_add_epi32(s2_8, s2_9));
    let s3_9 = wrap(_mm256_sub_epi32(s2_8, s2_9));
    // step1[10] = wrap(-step2[10] + step2[11]) = wrap(step2[11] - step2[10])
    let s3_10 = wrap(_mm256_sub_epi32(s2_11, s2_10));
    let s3_11 = wrap(_mm256_add_epi32(s2_10, s2_11));
    let s3_12 = wrap(_mm256_add_epi32(s2_12, s2_13));
    let s3_13 = wrap(_mm256_sub_epi32(s2_12, s2_13));
    let s3_14 = wrap(_mm256_sub_epi32(s2_15, s2_14));
    let s3_15 = wrap(_mm256_add_epi32(s2_14, s2_15));

    // ----- stage 4 -----
    // step2[0,1] = wrap(drs((step1[0] ± step1[1]) * C16))
    let t1 = mul_const(_mm256_add_epi32(s3_0, s3_1), COSPI_16_64);
    let t2 = mul_const(_mm256_sub_epi32(s3_0, s3_1), COSPI_16_64);
    let s4_0 = wrap(drs(t1));
    let s4_1 = wrap(drs(t2));

    // step2[2] = wrap(drs(step1[2] * C24 - step1[3] * C8))
    // step2[3] = wrap(drs(step1[2] * C8  + step1[3] * C24))
    let t1 = _mm256_sub_epi32(mul_const(s3_2, COSPI_24_64), mul_const(s3_3, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s3_2, COSPI_8_64), mul_const(s3_3, COSPI_24_64));
    let s4_2 = wrap(drs(t1));
    let s4_3 = wrap(drs(t2));

    let s4_4 = wrap(_mm256_add_epi32(s3_4, s3_5));
    let s4_5 = wrap(_mm256_sub_epi32(s3_4, s3_5));
    // step2[6] = wrap(-step1[6] + step1[7]) = wrap(step1[7] - step1[6])
    let s4_6 = wrap(_mm256_sub_epi32(s3_7, s3_6));
    let s4_7 = wrap(_mm256_add_epi32(s3_6, s3_7));

    let s4_8 = s3_8;
    let s4_15 = s3_15;
    // step2[9]  = wrap(drs(-step1[9]  * C8  + step1[14] * C24))
    // step2[14] = wrap(drs( step1[9]  * C24 + step1[14] * C8))
    let t1 = _mm256_sub_epi32(mul_const(s3_14, COSPI_24_64), mul_const(s3_9, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s3_9, COSPI_24_64), mul_const(s3_14, COSPI_8_64));
    let s4_9 = wrap(drs(t1));
    let s4_14 = wrap(drs(t2));
    // step2[10] = wrap(drs(-step1[10] * C24 - step1[13] * C8))
    // step2[13] = wrap(drs(-step1[10] * C8  + step1[13] * C24))
    //
    // Rewritten without leading negation:
    //   step2[10] = wrap(drs(-(step1[10] * C24 + step1[13] * C8)))
    //   step2[13] = wrap(drs( step1[13] * C24 - step1[10] * C8))
    let sum = _mm256_add_epi32(mul_const(s3_10, COSPI_24_64), mul_const(s3_13, COSPI_8_64));
    let t1 = _mm256_sub_epi32(_mm256_set1_epi32(0), sum);
    let t2 = _mm256_sub_epi32(mul_const(s3_13, COSPI_24_64), mul_const(s3_10, COSPI_8_64));
    let s4_10 = wrap(drs(t1));
    let s4_13 = wrap(drs(t2));
    let s4_11 = s3_11;
    let s4_12 = s3_12;

    // ----- stage 5 -----
    let s5_0 = wrap(_mm256_add_epi32(s4_0, s4_3));
    let s5_1 = wrap(_mm256_add_epi32(s4_1, s4_2));
    let s5_2 = wrap(_mm256_sub_epi32(s4_1, s4_2));
    let s5_3 = wrap(_mm256_sub_epi32(s4_0, s4_3));
    let s5_4 = s4_4;
    // step1[5] = wrap(drs((step2[6] - step2[5]) * C16))
    // step1[6] = wrap(drs((step2[5] + step2[6]) * C16))
    let t1 = mul_const(_mm256_sub_epi32(s4_6, s4_5), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s4_5, s4_6), COSPI_16_64);
    let s5_5 = wrap(drs(t1));
    let s5_6 = wrap(drs(t2));
    let s5_7 = s4_7;

    let s5_8 = wrap(_mm256_add_epi32(s4_8, s4_11));
    let s5_9 = wrap(_mm256_add_epi32(s4_9, s4_10));
    let s5_10 = wrap(_mm256_sub_epi32(s4_9, s4_10));
    let s5_11 = wrap(_mm256_sub_epi32(s4_8, s4_11));
    // step1[12] = wrap(-step2[12] + step2[15]) = wrap(step2[15] - step2[12])
    let s5_12 = wrap(_mm256_sub_epi32(s4_15, s4_12));
    // step1[13] = wrap(-step2[13] + step2[14]) = wrap(step2[14] - step2[13])
    let s5_13 = wrap(_mm256_sub_epi32(s4_14, s4_13));
    let s5_14 = wrap(_mm256_add_epi32(s4_13, s4_14));
    let s5_15 = wrap(_mm256_add_epi32(s4_12, s4_15));

    // ----- stage 6 -----
    let s6_0 = wrap(_mm256_add_epi32(s5_0, s5_7));
    let s6_1 = wrap(_mm256_add_epi32(s5_1, s5_6));
    let s6_2 = wrap(_mm256_add_epi32(s5_2, s5_5));
    let s6_3 = wrap(_mm256_add_epi32(s5_3, s5_4));
    let s6_4 = wrap(_mm256_sub_epi32(s5_3, s5_4));
    let s6_5 = wrap(_mm256_sub_epi32(s5_2, s5_5));
    let s6_6 = wrap(_mm256_sub_epi32(s5_1, s5_6));
    let s6_7 = wrap(_mm256_sub_epi32(s5_0, s5_7));
    let s6_8 = s5_8;
    let s6_9 = s5_9;
    // step2[10] = wrap(drs((-step1[10] + step1[13]) * C16)) = wrap(drs((step1[13] - step1[10]) * C16))
    // step2[13] = wrap(drs(( step1[10] + step1[13]) * C16))
    let t1 = mul_const(_mm256_sub_epi32(s5_13, s5_10), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_10, s5_13), COSPI_16_64);
    let s6_10 = wrap(drs(t1));
    let s6_13 = wrap(drs(t2));
    // step2[11] = wrap(drs((-step1[11] + step1[12]) * C16)) = wrap(drs((step1[12] - step1[11]) * C16))
    // step2[12] = wrap(drs(( step1[11] + step1[12]) * C16))
    let t1 = mul_const(_mm256_sub_epi32(s5_12, s5_11), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_11, s5_12), COSPI_16_64);
    let s6_11 = wrap(drs(t1));
    let s6_12 = wrap(drs(t2));
    let s6_14 = s5_14;
    let s6_15 = s5_15;

    // ----- stage 7: final outputs. -----
    let o0 = wrap(_mm256_add_epi32(s6_0, s6_15));
    let o1 = wrap(_mm256_add_epi32(s6_1, s6_14));
    let o2 = wrap(_mm256_add_epi32(s6_2, s6_13));
    let o3 = wrap(_mm256_add_epi32(s6_3, s6_12));
    let o4 = wrap(_mm256_add_epi32(s6_4, s6_11));
    let o5 = wrap(_mm256_add_epi32(s6_5, s6_10));
    let o6 = wrap(_mm256_add_epi32(s6_6, s6_9));
    let o7 = wrap(_mm256_add_epi32(s6_7, s6_8));
    let o8 = wrap(_mm256_sub_epi32(s6_7, s6_8));
    let o9 = wrap(_mm256_sub_epi32(s6_6, s6_9));
    let o10 = wrap(_mm256_sub_epi32(s6_5, s6_10));
    let o11 = wrap(_mm256_sub_epi32(s6_4, s6_11));
    let o12 = wrap(_mm256_sub_epi32(s6_3, s6_12));
    let o13 = wrap(_mm256_sub_epi32(s6_2, s6_13));
    let o14 = wrap(_mm256_sub_epi32(s6_1, s6_14));
    let o15 = wrap(_mm256_sub_epi32(s6_0, s6_15));

    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

// ---------------------------------------------------------------------
// Pass helper: run idct16 on 8 rows in parallel.
//
// Input layout: 8 ymm where ymm[k] holds input position k across the 8
// rows being transformed (one row per lane). This is what comes out of
// `transpose8x8_i32` applied to a row-major 8×8 sub-block.
//
// For a 16-element transform on 8 rows we need 16 ymm of input — split
// as `lo[0..8]` (positions 0..8) and `hi[0..8]` (positions 8..16). We
// return them in the same split layout.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct16_pass_8rows(lo: [__m256i; 8], hi: [__m256i; 8]) -> ([__m256i; 8], [__m256i; 8]) {
    let v: [__m256i; 16] = [
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3], hi[4],
        hi[5], hi[6], hi[7],
    ];
    let out = idct16_avx2(v);
    let out_lo: [__m256i; 8] = [
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
    ];
    let out_hi: [__m256i; 8] = [
        out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15],
    ];
    (out_lo, out_hi)
}

/// AVX2 implementation of the 16×16 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct16x16_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `15 * stride + 16` bytes of valid storage.
///
/// # Implementation
///
/// **G4-γ full-1D** structure:
///
/// 1. Row pass over rows 0..8: load 8 rows × 2 ymm each (lo / hi
///    halves), transpose each 8×8 quadrant so each ymm holds one
///    input position across 8 rows, call [`idct16_pass_8rows`], then
///    transpose back to row-major into a stack scratch buffer.
/// 2. Row pass over rows 8..16: same again, lower half of scratch.
/// 3. Column pass over cols 0..8: load 16 i32 per column-strip from
///    scratch (= 2 ymm per row × 8 rows? No — we load 8 columns × 16
///    rows; arrange as 16 ymm of 8 lanes each by reading row-aligned),
///    transpose, idct16, transpose back into a per-row finisher input.
/// 4. Column pass over cols 8..16: same again.
/// 5. AVX2 add-clamp-pack finisher: each row's 16 outputs become a
///    saturated u8 row added to dst.
///
/// All loads/stores from / to `dest` are 16-byte (`__m128i`), unaligned.
/// See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct16x16_add_avx2_inner(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    debug_assert!(
        dest.len() >= stride * 15 + 16,
        "dest too small for 16x16 add"
    );

    // Stack scratch for the intermediate (post-row-pass) 16×16 matrix
    // in row-major i32 layout. The column pass reads from this; the
    // finisher reads the column-pass output directly out of registers.
    let mut intermediate = [0i32; 256];

    // ----- Row pass: process 8 rows at a time. -----
    for row_group in 0..2 {
        let row_base = row_group * 8;

        // Load 8 input rows × 2 ymm each. Each ymm = 8 i32 (input
        // positions 0..8 for `lo`, 8..16 for `hi`).
        let in_ptr = input.as_ptr().add(row_base * 16) as *const __m256i;
        let r0_lo = _mm256_loadu_si256(in_ptr.add(0));
        let r0_hi = _mm256_loadu_si256(in_ptr.add(1));
        let r1_lo = _mm256_loadu_si256(in_ptr.add(2));
        let r1_hi = _mm256_loadu_si256(in_ptr.add(3));
        let r2_lo = _mm256_loadu_si256(in_ptr.add(4));
        let r2_hi = _mm256_loadu_si256(in_ptr.add(5));
        let r3_lo = _mm256_loadu_si256(in_ptr.add(6));
        let r3_hi = _mm256_loadu_si256(in_ptr.add(7));
        let r4_lo = _mm256_loadu_si256(in_ptr.add(8));
        let r4_hi = _mm256_loadu_si256(in_ptr.add(9));
        let r5_lo = _mm256_loadu_si256(in_ptr.add(10));
        let r5_hi = _mm256_loadu_si256(in_ptr.add(11));
        let r6_lo = _mm256_loadu_si256(in_ptr.add(12));
        let r6_hi = _mm256_loadu_si256(in_ptr.add(13));
        let r7_lo = _mm256_loadu_si256(in_ptr.add(14));
        let r7_hi = _mm256_loadu_si256(in_ptr.add(15));

        // Transpose two 8×8 sub-blocks so each ymm holds one input
        // position across the 8 rows. `lo_cols[k]` = input position k
        // (k ∈ 0..8) across 8 rows; `hi_cols[k]` = input position k+8.
        let lo_cols = transpose8x8_i32(r0_lo, r1_lo, r2_lo, r3_lo, r4_lo, r5_lo, r6_lo, r7_lo);
        let hi_cols = transpose8x8_i32(r0_hi, r1_hi, r2_hi, r3_hi, r4_hi, r5_hi, r6_hi, r7_hi);

        // Apply idct16 over 8 rows in parallel.
        let (out_lo, out_hi) = idct16_pass_8rows(lo_cols, hi_cols);

        // Transpose back to row-major. After this, `lo_rows[r]` =
        // output positions 0..8 of intermediate row r, `hi_rows[r]` =
        // output positions 8..16.
        let lo_rows = transpose8x8_i32(
            out_lo[0], out_lo[1], out_lo[2], out_lo[3], out_lo[4], out_lo[5], out_lo[6], out_lo[7],
        );
        let hi_rows = transpose8x8_i32(
            out_hi[0], out_hi[1], out_hi[2], out_hi[3], out_hi[4], out_hi[5], out_hi[6], out_hi[7],
        );

        // Store into intermediate (row-major).
        let out_ptr = intermediate.as_mut_ptr().add(row_base * 16) as *mut __m256i;
        // Per row r in [0..8]: lo at offset 2r, hi at offset 2r+1.
        // (Each row = 16 i32 = 2 ymm.)
        for r in 0..8 {
            core::arch::x86_64::_mm256_storeu_si256(out_ptr.add(2 * r), lo_rows[r]);
            core::arch::x86_64::_mm256_storeu_si256(out_ptr.add(2 * r + 1), hi_rows[r]);
        }
    }

    // ----- Column pass: process 8 columns at a time. -----
    //
    // For each 8-column strip we need to load 16 rows × 8 i32 = 16 ymm.
    // Each ymm = one row of intermediate at columns `col_base..col_base+8`.
    // We then transpose into 8 ymm "column-vectors" where ymm[k] = row k
    // across the 8 columns — but wait, idct16 operates on a 16-element
    // input. So for a column pass on cols `c..c+8` we want 16 ymm where
    // ymm[k] = intermediate[k*16 + c .. k*16 + c+8], i.e., position k
    // (k = row index) across 8 cols. That's a 16-row × 8-col layout
    // already — no transpose needed at all between row pass and column
    // pass loads!
    //
    // After idct16 produces 16 ymm output (one per output row, 8 cols
    // in lanes), we hand to the finisher which expects per-row layout.
    let mut col_outputs: [__m256i; 32] = [_mm256_set1_epi32(0); 32];
    for col_group in 0..2 {
        let col_base = col_group * 8;

        // Load 16 rows of 8 i32 each from intermediate at column offset
        // `col_base`. Each row's i32 are contiguous in memory at
        // `intermediate[r * 16 + col_base ..]`, so a single 32-byte
        // unaligned load picks up 8 i32 per row.
        let mut col_in: [__m256i; 16] = [_mm256_set1_epi32(0); 16];
        for r in 0..16 {
            let p = intermediate.as_ptr().add(r * 16 + col_base) as *const __m256i;
            col_in[r] = _mm256_loadu_si256(p);
        }

        // Apply idct16 across the 16 input-position registers. Output
        // is 16 ymm: out[k] = output row k across the 8 columns being
        // processed.
        let col_out = idct16_avx2(col_in);

        // Stash into col_outputs[col_group*16 .. col_group*16 + 16].
        // We'll merge the two halves per-row in the finisher.
        for r in 0..16 {
            col_outputs[col_group * 16 + r] = col_out[r];
        }
    }

    // ----- Finisher: per row, combine cols 0..8 and cols 8..16, round,
    // shift, pack, add to dst, saturate, store. -----
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..16 {
        let lo_i32 = col_outputs[r]; // cols 0..8 of row r
        let hi_i32 = col_outputs[16 + r]; // cols 8..16 of row r

        let lo_i32 = _mm256_srai_epi32(_mm256_add_epi32(lo_i32, round), 6);
        let hi_i32 = _mm256_srai_epi32(_mm256_add_epi32(hi_i32, round), 6);

        // Saturate-pack i32 → i16 with per-128-bit-lane behaviour, then
        // permute by 64-bit qwords (control 0xD8 = [0, 2, 1, 3]) so the
        // low 128 bits of each ymm hold 8 i16 in natural order.
        let lo_packed = _mm256_packs_epi32(lo_i32, lo_i32);
        let hi_packed = _mm256_packs_epi32(hi_i32, hi_i32);
        let lo_natural = _mm256_permute4x64_epi64(lo_packed, 0xD8);
        let hi_natural = _mm256_permute4x64_epi64(hi_packed, 0xD8);
        let out_i16_lo: __m128i = _mm256_castsi256_si128(lo_natural);
        let out_i16_hi: __m128i = _mm256_castsi256_si128(hi_natural);

        // Load 16 u8 from dst row (unaligned).
        // G4: loads from `dst` use _mm_loadu_si128 (unaligned)
        let dst_row_ptr = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadu_si128(dst_row_ptr);
        let dst_i16_lo = _mm_unpacklo_epi8(dst_u8, zero);
        let dst_i16_hi = _mm_unpackhi_epi8(dst_u8, zero);

        let sum_lo = _mm_add_epi16(out_i16_lo, dst_i16_lo);
        let sum_hi = _mm_add_epi16(out_i16_hi, dst_i16_hi);
        let sum_u8 = _mm_packus_epi16(sum_lo, sum_hi);

        let dst_row_ptr_mut = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storeu_si128(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the AVX2 idct16x16 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct16x16_256_add`], so the
/// `SimdDispatch::idct16x16_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct16x16_add_avx2(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct16x16_avx2::tests` gates each call on
    // a runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { idct16x16_add_avx2_inner(input, dest, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct16x16_add`.
    //!
    //! G4-γ refactor coverage (mirrors the G4.1 idct8x8 / G4.2 idct4x4
    //! and G4-γ idct8x8 test suites):
    //!
    //! - **`idct16x16_scalar_eq_avx2_proptest`** — 256-case proptest
    //!   with five stratified strategies (uniform, sparse, eob-bucket,
    //!   high-magnitude, pathological).
    //! - **`idct16x16_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (transform output is the zero matrix, +32 >> 6 = 0).
    //! - **`idct16x16_dc_only`** — single DC coefficient at position 0.
    //! - **`idct16x16_single_coefficient_impulses`** — sweep one
    //!   non-zero coefficient across every position 0..256.
    //! - **`idct16x16_max_input`** — saturation stress with i16::MAX /
    //!   i16::MIN coefficients.
    //! - **`idct16x16_misaligned_dst_pointers`** — per-skew arithmetic
    //!   regression guard (kernel uses `_mm_loadu_si128`, unaligned).
    //! - **`idct16x16_dispatch_round_trip`** — forced-AVX2 vs
    //!   forced-scalar dispatch produce identical output.
    //! - **`idct16x16_full1d_pathological_directed`** — G4-γ directed
    //!   regression: row/col-dominant pathological stress patterns that
    //!   exercise the transpose, row-pass butterfly, and column-pass
    //!   butterfly with high-magnitude alternating signs.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::idct16x16_add_avx2;
    use crate::transforms::inverse::idct16x16_256_add;
    use proptest::prelude::*;

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
        idct16x16_add_avx2(input, &mut dst, 16);
        dst
    }

    fn run_scalar(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        let mut dst = *dst_init;
        idct16x16_256_add(input, &mut dst, 16);
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
    // Strategies (same shape as idct8x8 / idct4x4, sized for 256 coeffs).
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

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn idct16x16_scalar_eq_avx2_proptest(
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
    // Directed fixtures (≥4 per acceptance criteria).
    // -----------------------------------------------------------------

    /// All-zero coefficients leave dst unchanged.
    #[test]
    fn idct16x16_all_zero_input() {
        require_avx2!();
        let input = [0i32; 256];
        let dst_init = [42u8; 256];
        assert_match(&input, &dst_init);
    }

    /// DC-only: single coefficient at position 0.
    #[test]
    fn idct16x16_dc_only() {
        require_avx2!();
        let mut input = [0i32; 256];
        input[0] = 1234;
        let dst_init = [50u8; 256];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep across all 256 positions.
    #[test]
    fn idct16x16_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..256 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 256];
                input[pos] = val;
                let dst_init = [128u8; 256];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Saturation stress: i16 extremes throughout.
    #[test]
    fn idct16x16_max_input() {
        require_avx2!();
        let input = [i16::MAX as i32; 256];
        let dst_init = [0u8; 256];
        assert_match(&input, &dst_init);

        let input = [i16::MIN as i32; 256];
        let dst_init = [255u8; 256];
        assert_match(&input, &dst_init);
    }

    /// Per-skew arithmetic regression guard. Kernel uses 16-byte
    /// `_mm_loadu_si128` so this isn't an alignment-fault test — it
    /// guards against per-offset stride miscomputes.
    #[test]
    fn idct16x16_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().take(32).enumerate() {
            *slot = (i as i32) * 199 - 3100;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 320]> = Box::new([0u8; 320]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;

            idct16x16_add_avx2(&input, &mut buf[skew..skew + 16 * 16], 16);
            idct16x16_256_add(&input, &mut scalar_buf[skew..skew + 16 * 16], 16);
            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }

    /// Forced-AVX2 vs forced-scalar dispatch produces the same output.
    /// Tautology-guard: under forced-AVX2 caps the dispatch slot must
    /// NOT be the scalar function pointer (otherwise the diff is
    /// trivially zero).
    #[test]
    fn idct16x16_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let scalar_fp: fn(&[i32; 256], &mut [u8], usize) = idct16x16_256_add;
        assert_ne!(
            dispatch_avx2.idct16x16_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 29) % 4001 - 2000;
        }
        let dst_init = [97u8; 256];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct16x16_add())(&input, &mut dst_avx2, 16);
        (dispatch_scalar.idct16x16_add())(&input, &mut dst_scalar, 16);
        assert_eq!(dst_avx2, dst_scalar);
    }

    /// **G4-γ directed regression**: pathological row/col-dominant
    /// stress patterns. The full-1D refactor splits work across the
    /// row pass, the column pass, and two transposes; a bug in any of
    /// these would corrupt outputs on extreme inputs. This test
    /// exercises:
    ///
    /// 1. Row 0 = ramp of high-magnitude positive values
    /// 2. Row 8 = high-magnitude alternating sign (straddles the
    ///    8-row group boundary — checks that the second row-group
    ///    iteration doesn't leak state)
    /// 3. Column 0 = ramp via the column-pass (only set via column-0
    ///    coefficients which the row pass routes through DC bin)
    /// 4. Column 8 = high-magnitude alternating sign (straddles the
    ///    8-col group boundary)
    /// 5. Single impulse at (15, 15) — the most-significant position;
    ///    exercises the "odd half" of the butterfly across BOTH the
    ///    row pass (within row 15) and column pass (within col 15).
    #[test]
    fn idct16x16_full1d_pathological_directed() {
        require_avx2!();

        // (1) Row 0 = ramp of high-magnitude positive values.
        let mut input = [0i32; 256];
        for c in 0..16 {
            input[c] = (c as i32) * 2000 + 100;
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (2) Row 8 = high-magnitude alternating sign (boundary test).
        let mut input = [0i32; 256];
        for c in 0..16 {
            input[8 * 16 + c] = if c & 1 == 0 { 28000 } else { -28000 };
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (3) Column 0 = ramp via the column-pass.
        let mut input = [0i32; 256];
        for r in 0..16 {
            input[r * 16] = (r as i32) * 2000 + 100;
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (4) Column 8 = high-magnitude alternating sign.
        let mut input = [0i32; 256];
        for r in 0..16 {
            input[r * 16 + 8] = if r & 1 == 0 { 28000 } else { -28000 };
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (5) Single impulse at (15, 15) — odd-half stress.
        let mut input = [0i32; 256];
        input[15 * 16 + 15] = 12345;
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (6) Combined: row 0 ramp + row 8 alternating + impulse.
        let mut input = [0i32; 256];
        for c in 0..16 {
            input[c] = (c as i32) * 2000 + 100;
            input[8 * 16 + c] = if c & 1 == 0 { 28000 } else { -28000 };
        }
        input[15 * 16 + 15] = 12345;
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }
    }

    /// **G4-γ helper sanity**: standalone transpose unit test.
    ///
    /// Builds a 16×16 i32 matrix with `m[r][c] = r * 16 + c`, runs the
    /// per-quadrant transpose used by the kernel (two 8×8 transposes
    /// per group + two row-group iterations equivalent to a single
    /// 16×16 transpose), and confirms `m_transposed[r][c] == c * 16 + r`.
    ///
    /// This catches transpose bugs (lane-permute misorder, quadrant
    /// swap miss, etc.) without dragging the full idct16 butterfly
    /// into the diagnosis path.
    #[test]
    fn idct16x16_transpose_sanity() {
        require_avx2!();
        // SAFETY: gated on AVX2 detection.
        unsafe { idct16x16_transpose_sanity_inner() }
    }

    // G4: loads from `dst` use n/a (test helper — operates on stack scratch only)
    #[target_feature(enable = "avx2")]
    unsafe fn idct16x16_transpose_sanity_inner() {
        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

        // Build a 16×16 row-major matrix where m[r * 16 + c] = r*16 + c.
        let mut src = [0i32; 256];
        for r in 0..16 {
            for c in 0..16 {
                src[r * 16 + c] = (r * 16 + c) as i32;
            }
        }

        // Apply the same transpose procedure the kernel uses:
        //   for each 8-row group:
        //     load 8 rows × 2 ymm
        //     transpose two 8×8 quadrants (lo and hi separately)
        //     transpose back into row-major
        // A double-transpose is identity — but we WANT a single
        // transpose for the sanity check. So we do:
        //   - 8×8 transpose of UL quadrant (rows 0..8, cols 0..8) →
        //     becomes UL of result (rows 0..8, cols 0..8 transposed)
        //   - 8×8 transpose of UR quadrant (rows 0..8, cols 8..16) →
        //     becomes LL of result (rows 8..16, cols 0..8 transposed)
        //   - 8×8 transpose of LL quadrant (rows 8..16, cols 0..8) →
        //     becomes UR of result (rows 0..8, cols 8..16 transposed)
        //   - 8×8 transpose of LR quadrant (rows 8..16, cols 8..16) →
        //     becomes LR of result (rows 8..16, cols 8..16 transposed)
        let mut dst = [0i32; 256];
        for row_group in 0..2usize {
            for col_group in 0..2usize {
                // Load 8 rows of 8 i32 from src[row_group*8 + r][col_group*8 ..]
                let mut rs: [__m256i; 8] = [super::_mm256_set1_epi32(0); 8];
                for r in 0..8 {
                    let p = src.as_ptr().add((row_group * 8 + r) * 16 + col_group * 8)
                        as *const __m256i;
                    rs[r] = _mm256_loadu_si256(p);
                }
                let cols =
                    super::transpose8x8_i32(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6], rs[7]);
                // Store into the swapped quadrant of dst:
                //   destination row group = col_group
                //   destination col group = row_group
                for k in 0..8 {
                    let p = dst
                        .as_mut_ptr()
                        .add((col_group * 8 + k) * 16 + row_group * 8)
                        as *mut __m256i;
                    _mm256_storeu_si256(p, cols[k]);
                }
            }
        }

        // Verify dst[r*16+c] == src[c*16+r] for all r,c.
        for r in 0..16 {
            for c in 0..16 {
                assert_eq!(
                    dst[r * 16 + c],
                    (c * 16 + r) as i32,
                    "transpose mismatch at (r={r}, c={c})"
                );
            }
        }
    }
}
