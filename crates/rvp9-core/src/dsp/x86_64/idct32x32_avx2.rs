//! 32×32 inverse-DCT + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor at 32×32 (Wave F F8)**: both the row pass and
//! the column pass are now fully vectorised in AVX2 with 32-bit lanes,
//! following the pattern proven by [`crate::dsp::x86_64::idct8x8_avx2`]
//! (Wave E G4-γ) and [`crate::dsp::x86_64::idct16x16_avx2`] (Wave F F2).
//! The previous "finisher-only" G4-α pilot kept the scalar
//! [`crate::transforms::inverse::idct32`] call site and only
//! AVX2-accelerated the final round/clamp/pack stage. That pattern is
//! preserved for the EOB fast paths
//! ([`crate::dsp::x86_64::idct32x32_34_avx2`],
//! [`crate::dsp::x86_64::idct32x32_135_avx2`]) which still call the
//! scalar `idct32` for the row pass.
//!
//! ## Bit-exactness vs scalar reference
//!
//! VP9 conformance requires bit-exact match to libvpx's C reference.
//! [`crate::transforms::inverse::idct32x32_1024_add`] does row + column
//! 1-D `idct32` in i32 with explicit `wraplow` (`int16_t`-truncation)
//! at every intermediate store, then a final `(out + 32) >> 6`
//! ROUND_POWER_OF_TWO with `clip_pixel_add`. The vector form mirrors
//! every scalar operation in [`crate::transforms::inverse::idct32`]
//! 1:1 across 8 lanes (one lane per row of the 8-row sub-block we're
//! transforming), running the row pass four times (rows 0..8, 8..16,
//! 16..24, 24..32) and the column pass four times (cols 0..8, 8..16,
//! 16..24, 24..32).
//!
//! Operand-range note: every intermediate is the result of either a
//! `wraplow` (hence i16-range) or an unwrapped raw input coefficient
//! (also spec-bounded to i16). Cospi constants are `< 2^15`. A single
//! product `wraplow × cospi` fits in `i32` (≈ 2^29 max).
//! [`_mm256_mullo_epi32`] returns the low 32 bits of the product, exact
//! for in-range operands. The sum/difference of two such products fits
//! in `i32` with margin (≈ 2^30 < 2^31). Therefore the scalar
//! reference's `i64` intermediate arithmetic can be folded down to `i32`
//! without divergence — same as in the 8×8 and 16×16 kernels.
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
//! half-row; two halves per row); coefficient buffer reads via
//! `_mm256_loadu_si256`. The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadu_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_packs_epi32,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_slli_epi32,
    _mm256_srai_epi32, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64,
};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers — duplicated verbatim from `idct8x8_avx2.rs`
// (and `idct16x16_avx2.rs`) to keep this kernel self-contained.
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
// 8×8 i32 transpose using a 3-stage unpack pyramid. Duplicated from
// `idct8x8_avx2::transpose8x8_i32`. Used as the building block to
// transpose the 8×8 sub-blocks of the full 32×32 transform.
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
// Vector 1D idct32. Each `__m256i` holds 8 i32 lanes corresponding to 8
// independent rows / columns being transformed in parallel; we take 32
// input ymm (one per input position 0..32) and return 32 output ymm
// (one per output position 0..32). The arithmetic mirrors
// `idct32` in `crate::transforms::inverse` line-by-line, including the
// stage-1 bit-reversed input gather for positions 0..16 (libvpx
// `idct32_c` :817-873) and the cosine butterflies for positions 16..32.
//
// 8-stage butterfly:
//   stage 1 — bit-reversed input load for 0..16; cosine butterflies for
//             16..32 (high frequencies, cospi 1/3/5/7/9/.../31)
//   stage 2 — cosine butterflies for 8..16 (cospi 2/6/10/.../30) +
//             16..32 add/sub fan-out
//   stage 3 — cospi-4/12/20/28 for 4..8; 8..16 fan-out; cospi-4/12/20/28
//             for 16..32 (with double-negation patterns at 18,22)
//   stage 4 — DC + cospi-8/24 for 2,3; 4..8 fan-out; cospi-8/24 for
//             9,10,13,14 (with double-negation at 10); 16..32 add/sub
//             fan-out (4-wide)
//   stage 5 — 0..4 cross-add; cospi-16 for 5,6; 8..16 fan-out; cospi-8/24
//             for 18,19,28,29 + cospi-8/24 for 20,21,26,27 (with
//             double-negation at 20,21)
//   stage 6 — 0..8 fan-out; cospi-16 for 10..13; 16..32 fan-out (8-wide)
//   stage 7 — cospi-16 for 20..27; 0..16 cross-add (similar to idct16
//             stage 7)
//   final   — 32-wide cross add/sub
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_avx2(v: [__m256i; 32]) -> [__m256i; 32] {
    // Reusable zero for double-negation patterns.
    let zero = _mm256_set1_epi32(0);

    // ----- stage 1 -----
    // step1[0..16] = input[bitrev(k)] for k in 0..16.
    let s1_0 = v[0];
    let s1_1 = v[16];
    let s1_2 = v[8];
    let s1_3 = v[24];
    let s1_4 = v[4];
    let s1_5 = v[20];
    let s1_6 = v[12];
    let s1_7 = v[28];
    let s1_8 = v[2];
    let s1_9 = v[18];
    let s1_10 = v[10];
    let s1_11 = v[26];
    let s1_12 = v[6];
    let s1_13 = v[22];
    let s1_14 = v[14];
    let s1_15 = v[30];

    // step1[16..32] = high-frequency cosine butterflies.
    let t1 = _mm256_sub_epi32(mul_const(v[1], COSPI_31_64), mul_const(v[31], COSPI_1_64));
    let t2 = _mm256_add_epi32(mul_const(v[1], COSPI_1_64), mul_const(v[31], COSPI_31_64));
    let s1_16 = wrap(drs(t1));
    let s1_31 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[17], COSPI_15_64), mul_const(v[15], COSPI_17_64));
    let t2 = _mm256_add_epi32(mul_const(v[17], COSPI_17_64), mul_const(v[15], COSPI_15_64));
    let s1_17 = wrap(drs(t1));
    let s1_30 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[9], COSPI_23_64), mul_const(v[23], COSPI_9_64));
    let t2 = _mm256_add_epi32(mul_const(v[9], COSPI_9_64), mul_const(v[23], COSPI_23_64));
    let s1_18 = wrap(drs(t1));
    let s1_29 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[25], COSPI_7_64), mul_const(v[7], COSPI_25_64));
    let t2 = _mm256_add_epi32(mul_const(v[25], COSPI_25_64), mul_const(v[7], COSPI_7_64));
    let s1_19 = wrap(drs(t1));
    let s1_28 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[5], COSPI_27_64), mul_const(v[27], COSPI_5_64));
    let t2 = _mm256_add_epi32(mul_const(v[5], COSPI_5_64), mul_const(v[27], COSPI_27_64));
    let s1_20 = wrap(drs(t1));
    let s1_27 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[21], COSPI_11_64), mul_const(v[11], COSPI_21_64));
    let t2 = _mm256_add_epi32(mul_const(v[21], COSPI_21_64), mul_const(v[11], COSPI_11_64));
    let s1_21 = wrap(drs(t1));
    let s1_26 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[13], COSPI_19_64), mul_const(v[19], COSPI_13_64));
    let t2 = _mm256_add_epi32(mul_const(v[13], COSPI_13_64), mul_const(v[19], COSPI_19_64));
    let s1_22 = wrap(drs(t1));
    let s1_25 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(v[29], COSPI_3_64), mul_const(v[3], COSPI_29_64));
    let t2 = _mm256_add_epi32(mul_const(v[29], COSPI_29_64), mul_const(v[3], COSPI_3_64));
    let s1_23 = wrap(drs(t1));
    let s1_24 = wrap(drs(t2));

    // ----- stage 2 -----
    // step2[0..8] pass-through from step1[0..8].
    let s2_0 = s1_0;
    let s2_1 = s1_1;
    let s2_2 = s1_2;
    let s2_3 = s1_3;
    let s2_4 = s1_4;
    let s2_5 = s1_5;
    let s2_6 = s1_6;
    let s2_7 = s1_7;

    // step2[8..16] = cospi-2/6/10/.../30 butterflies on step1[8..16].
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

    // step2[16..32] = add/sub fan-out (2-wide pairs).
    let s2_16 = wrap(_mm256_add_epi32(s1_16, s1_17));
    let s2_17 = wrap(_mm256_sub_epi32(s1_16, s1_17));
    // step2[18] = wrap(-step1[18] + step1[19]) = wrap(step1[19] - step1[18])
    let s2_18 = wrap(_mm256_sub_epi32(s1_19, s1_18));
    let s2_19 = wrap(_mm256_add_epi32(s1_18, s1_19));
    let s2_20 = wrap(_mm256_add_epi32(s1_20, s1_21));
    let s2_21 = wrap(_mm256_sub_epi32(s1_20, s1_21));
    let s2_22 = wrap(_mm256_sub_epi32(s1_23, s1_22));
    let s2_23 = wrap(_mm256_add_epi32(s1_22, s1_23));
    let s2_24 = wrap(_mm256_add_epi32(s1_24, s1_25));
    let s2_25 = wrap(_mm256_sub_epi32(s1_24, s1_25));
    let s2_26 = wrap(_mm256_sub_epi32(s1_27, s1_26));
    let s2_27 = wrap(_mm256_add_epi32(s1_26, s1_27));
    let s2_28 = wrap(_mm256_add_epi32(s1_28, s1_29));
    let s2_29 = wrap(_mm256_sub_epi32(s1_28, s1_29));
    let s2_30 = wrap(_mm256_sub_epi32(s1_31, s1_30));
    let s2_31 = wrap(_mm256_add_epi32(s1_30, s1_31));

    // ----- stage 3 -----
    // step1[0..4] pass-through.
    let s3_0 = s2_0;
    let s3_1 = s2_1;
    let s3_2 = s2_2;
    let s3_3 = s2_3;

    // step1[4..8] = cospi-4/12/20/28 butterflies on step2[4..8].
    let t1 = _mm256_sub_epi32(mul_const(s2_4, COSPI_28_64), mul_const(s2_7, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(s2_4, COSPI_4_64), mul_const(s2_7, COSPI_28_64));
    let s3_4 = wrap(drs(t1));
    let s3_7 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s2_5, COSPI_12_64), mul_const(s2_6, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(s2_5, COSPI_20_64), mul_const(s2_6, COSPI_12_64));
    let s3_5 = wrap(drs(t1));
    let s3_6 = wrap(drs(t2));

    // step1[8..16] = add/sub fan-out.
    let s3_8 = wrap(_mm256_add_epi32(s2_8, s2_9));
    let s3_9 = wrap(_mm256_sub_epi32(s2_8, s2_9));
    // step1[10] = wrap(-step2[10] + step2[11]) = wrap(step2[11] - step2[10])
    let s3_10 = wrap(_mm256_sub_epi32(s2_11, s2_10));
    let s3_11 = wrap(_mm256_add_epi32(s2_10, s2_11));
    let s3_12 = wrap(_mm256_add_epi32(s2_12, s2_13));
    let s3_13 = wrap(_mm256_sub_epi32(s2_12, s2_13));
    let s3_14 = wrap(_mm256_sub_epi32(s2_15, s2_14));
    let s3_15 = wrap(_mm256_add_epi32(s2_14, s2_15));

    // step1[16..32] = cospi-4/12/20/28 butterflies + pass-through.
    let s3_16 = s2_16;
    let s3_31 = s2_31;
    // step1[17] = drs(-s17*C4 + s30*C28) = drs(s30*C28 - s17*C4)
    // step1[30] = drs(s17*C28 + s30*C4)
    let t1 = _mm256_sub_epi32(mul_const(s2_30, COSPI_28_64), mul_const(s2_17, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(s2_17, COSPI_28_64), mul_const(s2_30, COSPI_4_64));
    let s3_17 = wrap(drs(t1));
    let s3_30 = wrap(drs(t2));
    // step1[18] = drs(-s18*C28 - s29*C4) = drs(-(s18*C28 + s29*C4))
    // step1[29] = drs(-s18*C4 + s29*C28) = drs(s29*C28 - s18*C4)
    let sum = _mm256_add_epi32(mul_const(s2_18, COSPI_28_64), mul_const(s2_29, COSPI_4_64));
    let t1 = _mm256_sub_epi32(zero, sum);
    let t2 = _mm256_sub_epi32(mul_const(s2_29, COSPI_28_64), mul_const(s2_18, COSPI_4_64));
    let s3_18 = wrap(drs(t1));
    let s3_29 = wrap(drs(t2));
    let s3_19 = s2_19;
    let s3_20 = s2_20;
    // step1[21] = drs(-s21*C20 + s26*C12) = drs(s26*C12 - s21*C20)
    // step1[26] = drs(s21*C12 + s26*C20)
    let t1 = _mm256_sub_epi32(mul_const(s2_26, COSPI_12_64), mul_const(s2_21, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(s2_21, COSPI_12_64), mul_const(s2_26, COSPI_20_64));
    let s3_21 = wrap(drs(t1));
    let s3_26 = wrap(drs(t2));
    // step1[22] = drs(-s22*C12 - s25*C20) = drs(-(s22*C12 + s25*C20))
    // step1[25] = drs(-s22*C20 + s25*C12) = drs(s25*C12 - s22*C20)
    let sum = _mm256_add_epi32(mul_const(s2_22, COSPI_12_64), mul_const(s2_25, COSPI_20_64));
    let t1 = _mm256_sub_epi32(zero, sum);
    let t2 = _mm256_sub_epi32(mul_const(s2_25, COSPI_12_64), mul_const(s2_22, COSPI_20_64));
    let s3_22 = wrap(drs(t1));
    let s3_25 = wrap(drs(t2));
    let s3_23 = s2_23;
    let s3_24 = s2_24;
    let s3_27 = s2_27;
    let s3_28 = s2_28;

    // ----- stage 4 -----
    // step2[0,1] = drs((step1[0] ± step1[1]) * C16)
    let t1 = mul_const(_mm256_add_epi32(s3_0, s3_1), COSPI_16_64);
    let t2 = mul_const(_mm256_sub_epi32(s3_0, s3_1), COSPI_16_64);
    let s4_0 = wrap(drs(t1));
    let s4_1 = wrap(drs(t2));
    // step2[2] = drs(step1[2] * C24 - step1[3] * C8)
    // step2[3] = drs(step1[2] * C8  + step1[3] * C24)
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
    // step2[9]  = drs(-step1[9]  * C8  + step1[14] * C24) = drs(step1[14]*C24 - step1[9]*C8)
    // step2[14] = drs( step1[9]  * C24 + step1[14] * C8)
    let t1 = _mm256_sub_epi32(mul_const(s3_14, COSPI_24_64), mul_const(s3_9, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s3_9, COSPI_24_64), mul_const(s3_14, COSPI_8_64));
    let s4_9 = wrap(drs(t1));
    let s4_14 = wrap(drs(t2));
    // step2[10] = drs(-step1[10] * C24 - step1[13] * C8) = drs(-(s10*C24 + s13*C8))
    // step2[13] = drs(-step1[10] * C8  + step1[13] * C24) = drs(s13*C24 - s10*C8)
    let sum = _mm256_add_epi32(mul_const(s3_10, COSPI_24_64), mul_const(s3_13, COSPI_8_64));
    let t1 = _mm256_sub_epi32(zero, sum);
    let t2 = _mm256_sub_epi32(mul_const(s3_13, COSPI_24_64), mul_const(s3_10, COSPI_8_64));
    let s4_10 = wrap(drs(t1));
    let s4_13 = wrap(drs(t2));
    let s4_11 = s3_11;
    let s4_12 = s3_12;

    // step2[16..32] = 4-wide add/sub fan-out (16↔19, 17↔18, 20↔23, 21↔22, ...).
    let s4_16 = wrap(_mm256_add_epi32(s3_16, s3_19));
    let s4_17 = wrap(_mm256_add_epi32(s3_17, s3_18));
    let s4_18 = wrap(_mm256_sub_epi32(s3_17, s3_18));
    let s4_19 = wrap(_mm256_sub_epi32(s3_16, s3_19));
    // step2[20] = wrap(-s20 + s23) = wrap(s23 - s20)
    let s4_20 = wrap(_mm256_sub_epi32(s3_23, s3_20));
    // step2[21] = wrap(-s21 + s22) = wrap(s22 - s21)
    let s4_21 = wrap(_mm256_sub_epi32(s3_22, s3_21));
    let s4_22 = wrap(_mm256_add_epi32(s3_21, s3_22));
    let s4_23 = wrap(_mm256_add_epi32(s3_20, s3_23));

    let s4_24 = wrap(_mm256_add_epi32(s3_24, s3_27));
    let s4_25 = wrap(_mm256_add_epi32(s3_25, s3_26));
    let s4_26 = wrap(_mm256_sub_epi32(s3_25, s3_26));
    let s4_27 = wrap(_mm256_sub_epi32(s3_24, s3_27));
    // step2[28] = wrap(-s28 + s31) = wrap(s31 - s28)
    let s4_28 = wrap(_mm256_sub_epi32(s3_31, s3_28));
    // step2[29] = wrap(-s29 + s30) = wrap(s30 - s29)
    let s4_29 = wrap(_mm256_sub_epi32(s3_30, s3_29));
    let s4_30 = wrap(_mm256_add_epi32(s3_29, s3_30));
    let s4_31 = wrap(_mm256_add_epi32(s3_28, s3_31));

    // ----- stage 5 -----
    let s5_0 = wrap(_mm256_add_epi32(s4_0, s4_3));
    let s5_1 = wrap(_mm256_add_epi32(s4_1, s4_2));
    let s5_2 = wrap(_mm256_sub_epi32(s4_1, s4_2));
    let s5_3 = wrap(_mm256_sub_epi32(s4_0, s4_3));
    let s5_4 = s4_4;
    // step1[5] = drs((step2[6] - step2[5]) * C16)
    // step1[6] = drs((step2[5] + step2[6]) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s4_6, s4_5), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s4_5, s4_6), COSPI_16_64);
    let s5_5 = wrap(drs(t1));
    let s5_6 = wrap(drs(t2));
    let s5_7 = s4_7;

    let s5_8 = wrap(_mm256_add_epi32(s4_8, s4_11));
    let s5_9 = wrap(_mm256_add_epi32(s4_9, s4_10));
    let s5_10 = wrap(_mm256_sub_epi32(s4_9, s4_10));
    let s5_11 = wrap(_mm256_sub_epi32(s4_8, s4_11));
    // step1[12] = wrap(-s12 + s15) = wrap(s15 - s12)
    let s5_12 = wrap(_mm256_sub_epi32(s4_15, s4_12));
    // step1[13] = wrap(-s13 + s14) = wrap(s14 - s13)
    let s5_13 = wrap(_mm256_sub_epi32(s4_14, s4_13));
    let s5_14 = wrap(_mm256_add_epi32(s4_13, s4_14));
    let s5_15 = wrap(_mm256_add_epi32(s4_12, s4_15));

    // step1[16..32] cospi-8/24 butterflies (with double-negation patterns at 20,21).
    let s5_16 = s4_16;
    let s5_17 = s4_17;
    // step1[18] = drs(-s18*C8 + s29*C24) = drs(s29*C24 - s18*C8)
    // step1[29] = drs( s18*C24 + s29*C8)
    let t1 = _mm256_sub_epi32(mul_const(s4_29, COSPI_24_64), mul_const(s4_18, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s4_18, COSPI_24_64), mul_const(s4_29, COSPI_8_64));
    let s5_18 = wrap(drs(t1));
    let s5_29 = wrap(drs(t2));
    // step1[19] = drs(-s19*C8 + s28*C24) = drs(s28*C24 - s19*C8)
    // step1[28] = drs( s19*C24 + s28*C8)
    let t1 = _mm256_sub_epi32(mul_const(s4_28, COSPI_24_64), mul_const(s4_19, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s4_19, COSPI_24_64), mul_const(s4_28, COSPI_8_64));
    let s5_19 = wrap(drs(t1));
    let s5_28 = wrap(drs(t2));
    // step1[20] = drs(-s20*C24 - s27*C8) = drs(-(s20*C24 + s27*C8))
    // step1[27] = drs(-s20*C8 + s27*C24) = drs(s27*C24 - s20*C8)
    let sum = _mm256_add_epi32(mul_const(s4_20, COSPI_24_64), mul_const(s4_27, COSPI_8_64));
    let t1 = _mm256_sub_epi32(zero, sum);
    let t2 = _mm256_sub_epi32(mul_const(s4_27, COSPI_24_64), mul_const(s4_20, COSPI_8_64));
    let s5_20 = wrap(drs(t1));
    let s5_27 = wrap(drs(t2));
    // step1[21] = drs(-s21*C24 - s26*C8) = drs(-(s21*C24 + s26*C8))
    // step1[26] = drs(-s21*C8 + s26*C24) = drs(s26*C24 - s21*C8)
    let sum = _mm256_add_epi32(mul_const(s4_21, COSPI_24_64), mul_const(s4_26, COSPI_8_64));
    let t1 = _mm256_sub_epi32(zero, sum);
    let t2 = _mm256_sub_epi32(mul_const(s4_26, COSPI_24_64), mul_const(s4_21, COSPI_8_64));
    let s5_21 = wrap(drs(t1));
    let s5_26 = wrap(drs(t2));
    let s5_22 = s4_22;
    let s5_23 = s4_23;
    let s5_24 = s4_24;
    let s5_25 = s4_25;
    let s5_30 = s4_30;
    let s5_31 = s4_31;

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
    // step2[10] = drs((-s10 + s13) * C16) = drs((s13 - s10) * C16)
    // step2[13] = drs(( s10 + s13) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s5_13, s5_10), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_10, s5_13), COSPI_16_64);
    let s6_10 = wrap(drs(t1));
    let s6_13 = wrap(drs(t2));
    // step2[11] = drs((-s11 + s12) * C16) = drs((s12 - s11) * C16)
    // step2[12] = drs(( s11 + s12) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s5_12, s5_11), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_11, s5_12), COSPI_16_64);
    let s6_11 = wrap(drs(t1));
    let s6_12 = wrap(drs(t2));
    let s6_14 = s5_14;
    let s6_15 = s5_15;

    // step2[16..32]: 8-wide fan-out with cross-add pattern.
    let s6_16 = wrap(_mm256_add_epi32(s5_16, s5_23));
    let s6_17 = wrap(_mm256_add_epi32(s5_17, s5_22));
    let s6_18 = wrap(_mm256_add_epi32(s5_18, s5_21));
    let s6_19 = wrap(_mm256_add_epi32(s5_19, s5_20));
    let s6_20 = wrap(_mm256_sub_epi32(s5_19, s5_20));
    let s6_21 = wrap(_mm256_sub_epi32(s5_18, s5_21));
    let s6_22 = wrap(_mm256_sub_epi32(s5_17, s5_22));
    let s6_23 = wrap(_mm256_sub_epi32(s5_16, s5_23));
    // step2[24] = wrap(-s24 + s31) = wrap(s31 - s24)
    let s6_24 = wrap(_mm256_sub_epi32(s5_31, s5_24));
    // step2[25] = wrap(-s25 + s30) = wrap(s30 - s25)
    let s6_25 = wrap(_mm256_sub_epi32(s5_30, s5_25));
    // step2[26] = wrap(-s26 + s29) = wrap(s29 - s26)
    let s6_26 = wrap(_mm256_sub_epi32(s5_29, s5_26));
    // step2[27] = wrap(-s27 + s28) = wrap(s28 - s27)
    let s6_27 = wrap(_mm256_sub_epi32(s5_28, s5_27));
    let s6_28 = wrap(_mm256_add_epi32(s5_27, s5_28));
    let s6_29 = wrap(_mm256_add_epi32(s5_26, s5_29));
    let s6_30 = wrap(_mm256_add_epi32(s5_25, s5_30));
    let s6_31 = wrap(_mm256_add_epi32(s5_24, s5_31));

    // ----- stage 7 -----
    let s7_0 = wrap(_mm256_add_epi32(s6_0, s6_15));
    let s7_1 = wrap(_mm256_add_epi32(s6_1, s6_14));
    let s7_2 = wrap(_mm256_add_epi32(s6_2, s6_13));
    let s7_3 = wrap(_mm256_add_epi32(s6_3, s6_12));
    let s7_4 = wrap(_mm256_add_epi32(s6_4, s6_11));
    let s7_5 = wrap(_mm256_add_epi32(s6_5, s6_10));
    let s7_6 = wrap(_mm256_add_epi32(s6_6, s6_9));
    let s7_7 = wrap(_mm256_add_epi32(s6_7, s6_8));
    let s7_8 = wrap(_mm256_sub_epi32(s6_7, s6_8));
    let s7_9 = wrap(_mm256_sub_epi32(s6_6, s6_9));
    let s7_10 = wrap(_mm256_sub_epi32(s6_5, s6_10));
    let s7_11 = wrap(_mm256_sub_epi32(s6_4, s6_11));
    let s7_12 = wrap(_mm256_sub_epi32(s6_3, s6_12));
    let s7_13 = wrap(_mm256_sub_epi32(s6_2, s6_13));
    let s7_14 = wrap(_mm256_sub_epi32(s6_1, s6_14));
    let s7_15 = wrap(_mm256_sub_epi32(s6_0, s6_15));

    let s7_16 = s6_16;
    let s7_17 = s6_17;
    let s7_18 = s6_18;
    let s7_19 = s6_19;
    // step1[20] = drs((-s20 + s27) * C16) = drs((s27 - s20) * C16)
    // step1[27] = drs(( s20 + s27) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s6_27, s6_20), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s6_20, s6_27), COSPI_16_64);
    let s7_20 = wrap(drs(t1));
    let s7_27 = wrap(drs(t2));
    // step1[21] = drs((-s21 + s26) * C16) = drs((s26 - s21) * C16)
    // step1[26] = drs(( s21 + s26) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s6_26, s6_21), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s6_21, s6_26), COSPI_16_64);
    let s7_21 = wrap(drs(t1));
    let s7_26 = wrap(drs(t2));
    // step1[22] = drs((-s22 + s25) * C16) = drs((s25 - s22) * C16)
    // step1[25] = drs(( s22 + s25) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s6_25, s6_22), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s6_22, s6_25), COSPI_16_64);
    let s7_22 = wrap(drs(t1));
    let s7_25 = wrap(drs(t2));
    // step1[23] = drs((-s23 + s24) * C16) = drs((s24 - s23) * C16)
    // step1[24] = drs(( s23 + s24) * C16)
    let t1 = mul_const(_mm256_sub_epi32(s6_24, s6_23), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s6_23, s6_24), COSPI_16_64);
    let s7_23 = wrap(drs(t1));
    let s7_24 = wrap(drs(t2));
    let s7_28 = s6_28;
    let s7_29 = s6_29;
    let s7_30 = s6_30;
    let s7_31 = s6_31;

    // ----- final stage: 32-wide cross add/sub. -----
    let o0 = wrap(_mm256_add_epi32(s7_0, s7_31));
    let o1 = wrap(_mm256_add_epi32(s7_1, s7_30));
    let o2 = wrap(_mm256_add_epi32(s7_2, s7_29));
    let o3 = wrap(_mm256_add_epi32(s7_3, s7_28));
    let o4 = wrap(_mm256_add_epi32(s7_4, s7_27));
    let o5 = wrap(_mm256_add_epi32(s7_5, s7_26));
    let o6 = wrap(_mm256_add_epi32(s7_6, s7_25));
    let o7 = wrap(_mm256_add_epi32(s7_7, s7_24));
    let o8 = wrap(_mm256_add_epi32(s7_8, s7_23));
    let o9 = wrap(_mm256_add_epi32(s7_9, s7_22));
    let o10 = wrap(_mm256_add_epi32(s7_10, s7_21));
    let o11 = wrap(_mm256_add_epi32(s7_11, s7_20));
    let o12 = wrap(_mm256_add_epi32(s7_12, s7_19));
    let o13 = wrap(_mm256_add_epi32(s7_13, s7_18));
    let o14 = wrap(_mm256_add_epi32(s7_14, s7_17));
    let o15 = wrap(_mm256_add_epi32(s7_15, s7_16));
    let o16 = wrap(_mm256_sub_epi32(s7_15, s7_16));
    let o17 = wrap(_mm256_sub_epi32(s7_14, s7_17));
    let o18 = wrap(_mm256_sub_epi32(s7_13, s7_18));
    let o19 = wrap(_mm256_sub_epi32(s7_12, s7_19));
    let o20 = wrap(_mm256_sub_epi32(s7_11, s7_20));
    let o21 = wrap(_mm256_sub_epi32(s7_10, s7_21));
    let o22 = wrap(_mm256_sub_epi32(s7_9, s7_22));
    let o23 = wrap(_mm256_sub_epi32(s7_8, s7_23));
    let o24 = wrap(_mm256_sub_epi32(s7_7, s7_24));
    let o25 = wrap(_mm256_sub_epi32(s7_6, s7_25));
    let o26 = wrap(_mm256_sub_epi32(s7_5, s7_26));
    let o27 = wrap(_mm256_sub_epi32(s7_4, s7_27));
    let o28 = wrap(_mm256_sub_epi32(s7_3, s7_28));
    let o29 = wrap(_mm256_sub_epi32(s7_2, s7_29));
    let o30 = wrap(_mm256_sub_epi32(s7_1, s7_30));
    let o31 = wrap(_mm256_sub_epi32(s7_0, s7_31));

    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19,
        o20, o21, o22, o23, o24, o25, o26, o27, o28, o29, o30, o31,
    ]
}

// ---------------------------------------------------------------------
// Pass helper: run idct32 on 8 rows in parallel, given the row data
// already in transposed (position-major) form. Splits the 32-wide
// transform into 4 quadrants of 8 inputs each (q0=positions 0..8,
// q1=8..16, q2=16..24, q3=24..32).
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_pass_8rows(
    q0: [__m256i; 8],
    q1: [__m256i; 8],
    q2: [__m256i; 8],
    q3: [__m256i; 8],
) -> ([__m256i; 8], [__m256i; 8], [__m256i; 8], [__m256i; 8]) {
    let v: [__m256i; 32] = [
        q0[0], q0[1], q0[2], q0[3], q0[4], q0[5], q0[6], q0[7], q1[0], q1[1], q1[2], q1[3], q1[4],
        q1[5], q1[6], q1[7], q2[0], q2[1], q2[2], q2[3], q2[4], q2[5], q2[6], q2[7], q3[0], q3[1],
        q3[2], q3[3], q3[4], q3[5], q3[6], q3[7],
    ];
    let out = idct32_avx2(v);
    let r0: [__m256i; 8] = [
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
    ];
    let r1: [__m256i; 8] = [
        out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15],
    ];
    let r2: [__m256i; 8] = [
        out[16], out[17], out[18], out[19], out[20], out[21], out[22], out[23],
    ];
    let r3: [__m256i; 8] = [
        out[24], out[25], out[26], out[27], out[28], out[29], out[30], out[31],
    ];
    (r0, r1, r2, r3)
}

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
/// **G4-γ full-1D** structure:
///
/// 1. Row pass over rows 0..8, 8..16, 16..24, 24..32: per group, load
///    8 rows × 4 ymm each, transpose each 8×8 sub-block so each ymm
///    holds one input position across 8 rows, call [`idct32_pass_8rows`],
///    then transpose back to row-major into a stack scratch buffer.
/// 2. Column pass over cols 0..8, 8..16, 16..24, 24..32: per group,
///    load 32 rows × 1 ymm (8 cols per row) — this is already in
///    position-major layout, no transpose needed — call [`idct32_avx2`],
///    stash outputs for the finisher.
/// 3. AVX2 add-clamp-pack finisher: each row's 32 outputs become a
///    saturated u8 row added to dst (two 16-byte halves).
///
/// All loads/stores from / to `dest` are 16-byte (`__m128i`), unaligned.
/// See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct32x32_add_avx2_inner(input: &[i32; 1024], dest: &mut [u8], stride: usize) {
    debug_assert!(
        dest.len() >= stride * 31 + 32,
        "dest too small for 32x32 add"
    );

    // Stack scratch for the intermediate (post-row-pass) 32×32 matrix
    // in row-major i32 layout. The column pass reads from this.
    let mut intermediate = [0i32; 1024];

    // ----- Row pass: process 8 rows at a time. -----
    for row_group in 0..4 {
        let row_base = row_group * 8;

        // Per-row early-exit: detect all-zero row (matches scalar
        // reference's libvpx zero_coeff fast skip). We OR-reduce across
        // the 4 ymm of each row; if every row's OR is zero, the entire
        // row group's intermediate output stays at the zero initialiser.
        let row_in_base = input.as_ptr().add(row_base * 32);
        let mut any_nonzero = false;
        for r in 0..8 {
            for k in 0..32 {
                if *row_in_base.add(r * 32 + k) != 0 {
                    any_nonzero = true;
                    break;
                }
            }
            if any_nonzero {
                break;
            }
        }
        if !any_nonzero {
            // All 8 rows are zero — skip the row pass for this group.
            // (intermediate is already zero-initialised in the relevant
            // slots.)
            continue;
        }

        // Load 8 input rows × 4 ymm each. Each ymm = 8 i32 (input
        // positions for one of 4 column quadrants).
        let in_ptr = input.as_ptr().add(row_base * 32) as *const __m256i;
        // r{R}_q{Q} = row R of the group, quadrant Q (Q ∈ 0..4).
        // Row R has 32 i32 → 4 ymm at offsets 4R, 4R+1, 4R+2, 4R+3.
        let r0_q0 = _mm256_loadu_si256(in_ptr.add(0));
        let r0_q1 = _mm256_loadu_si256(in_ptr.add(1));
        let r0_q2 = _mm256_loadu_si256(in_ptr.add(2));
        let r0_q3 = _mm256_loadu_si256(in_ptr.add(3));
        let r1_q0 = _mm256_loadu_si256(in_ptr.add(4));
        let r1_q1 = _mm256_loadu_si256(in_ptr.add(5));
        let r1_q2 = _mm256_loadu_si256(in_ptr.add(6));
        let r1_q3 = _mm256_loadu_si256(in_ptr.add(7));
        let r2_q0 = _mm256_loadu_si256(in_ptr.add(8));
        let r2_q1 = _mm256_loadu_si256(in_ptr.add(9));
        let r2_q2 = _mm256_loadu_si256(in_ptr.add(10));
        let r2_q3 = _mm256_loadu_si256(in_ptr.add(11));
        let r3_q0 = _mm256_loadu_si256(in_ptr.add(12));
        let r3_q1 = _mm256_loadu_si256(in_ptr.add(13));
        let r3_q2 = _mm256_loadu_si256(in_ptr.add(14));
        let r3_q3 = _mm256_loadu_si256(in_ptr.add(15));
        let r4_q0 = _mm256_loadu_si256(in_ptr.add(16));
        let r4_q1 = _mm256_loadu_si256(in_ptr.add(17));
        let r4_q2 = _mm256_loadu_si256(in_ptr.add(18));
        let r4_q3 = _mm256_loadu_si256(in_ptr.add(19));
        let r5_q0 = _mm256_loadu_si256(in_ptr.add(20));
        let r5_q1 = _mm256_loadu_si256(in_ptr.add(21));
        let r5_q2 = _mm256_loadu_si256(in_ptr.add(22));
        let r5_q3 = _mm256_loadu_si256(in_ptr.add(23));
        let r6_q0 = _mm256_loadu_si256(in_ptr.add(24));
        let r6_q1 = _mm256_loadu_si256(in_ptr.add(25));
        let r6_q2 = _mm256_loadu_si256(in_ptr.add(26));
        let r6_q3 = _mm256_loadu_si256(in_ptr.add(27));
        let r7_q0 = _mm256_loadu_si256(in_ptr.add(28));
        let r7_q1 = _mm256_loadu_si256(in_ptr.add(29));
        let r7_q2 = _mm256_loadu_si256(in_ptr.add(30));
        let r7_q3 = _mm256_loadu_si256(in_ptr.add(31));

        // Transpose each 8×8 quadrant so each ymm holds one input
        // position across 8 rows. q{Q}_cols[k] = input position (Q*8+k)
        // across the 8 rows of this group.
        let q0_cols = transpose8x8_i32(r0_q0, r1_q0, r2_q0, r3_q0, r4_q0, r5_q0, r6_q0, r7_q0);
        let q1_cols = transpose8x8_i32(r0_q1, r1_q1, r2_q1, r3_q1, r4_q1, r5_q1, r6_q1, r7_q1);
        let q2_cols = transpose8x8_i32(r0_q2, r1_q2, r2_q2, r3_q2, r4_q2, r5_q2, r6_q2, r7_q2);
        let q3_cols = transpose8x8_i32(r0_q3, r1_q3, r2_q3, r3_q3, r4_q3, r5_q3, r6_q3, r7_q3);

        // Apply idct32 over 8 rows in parallel.
        let (out_q0, out_q1, out_q2, out_q3) =
            idct32_pass_8rows(q0_cols, q1_cols, q2_cols, q3_cols);

        // Transpose each quadrant back to row-major. After this,
        // out_q{Q}_rows[r] = output positions (Q*8..Q*8+8) of
        // intermediate row r of this group.
        let q0_rows = transpose8x8_i32(
            out_q0[0], out_q0[1], out_q0[2], out_q0[3], out_q0[4], out_q0[5], out_q0[6], out_q0[7],
        );
        let q1_rows = transpose8x8_i32(
            out_q1[0], out_q1[1], out_q1[2], out_q1[3], out_q1[4], out_q1[5], out_q1[6], out_q1[7],
        );
        let q2_rows = transpose8x8_i32(
            out_q2[0], out_q2[1], out_q2[2], out_q2[3], out_q2[4], out_q2[5], out_q2[6], out_q2[7],
        );
        let q3_rows = transpose8x8_i32(
            out_q3[0], out_q3[1], out_q3[2], out_q3[3], out_q3[4], out_q3[5], out_q3[6], out_q3[7],
        );

        // Store into intermediate (row-major). For row r in 0..8 of
        // this group, store 4 ymm at offsets 4r, 4r+1, 4r+2, 4r+3.
        let out_ptr = intermediate.as_mut_ptr().add(row_base * 32) as *mut __m256i;
        for r in 0..8 {
            _mm256_storeu_si256(out_ptr.add(4 * r), q0_rows[r]);
            _mm256_storeu_si256(out_ptr.add(4 * r + 1), q1_rows[r]);
            _mm256_storeu_si256(out_ptr.add(4 * r + 2), q2_rows[r]);
            _mm256_storeu_si256(out_ptr.add(4 * r + 3), q3_rows[r]);
        }
    }

    // ----- Column pass: process 8 columns at a time. -----
    //
    // For each 8-column strip we load 32 rows × 8 i32 = 32 ymm. Each
    // ymm = one row of intermediate at columns `col_base..col_base+8`.
    // This IS already in position-major layout (each ymm = "input
    // position" k of the 1D transform, where the 32 inputs are the 32
    // rows of intermediate at the given column strip). So we can feed
    // straight into idct32_avx2 without any pre-transpose.
    //
    // Output: 32 ymm where out[k] = output row k across the 8 columns
    // being processed. We stash each strip's outputs into col_outputs
    // and the finisher gathers the 4 strips per row.
    let mut col_outputs: [__m256i; 128] = [_mm256_set1_epi32(0); 128];
    for col_group in 0..4 {
        let col_base = col_group * 8;

        let mut col_in: [__m256i; 32] = [_mm256_set1_epi32(0); 32];
        for (r, slot) in col_in.iter_mut().enumerate() {
            let p = intermediate.as_ptr().add(r * 32 + col_base) as *const __m256i;
            *slot = _mm256_loadu_si256(p);
        }

        let col_out = idct32_avx2(col_in);

        // Stash into col_outputs[col_group*32 .. col_group*32 + 32].
        for r in 0..32 {
            col_outputs[col_group * 32 + r] = col_out[r];
        }
    }

    // ----- Finisher: per row, gather 4 ymm strips, round, shift,
    // pack, add to dst, saturate, store. -----
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..32 {
        // 4 ymm: cols 0..8, 8..16, 16..24, 24..32 of row r.
        let v0 = col_outputs[r]; // cols 0..8
        let v1 = col_outputs[32 + r]; // cols 8..16
        let v2 = col_outputs[64 + r]; // cols 16..24
        let v3 = col_outputs[96 + r]; // cols 24..32

        let v0 = _mm256_srai_epi32(_mm256_add_epi32(v0, round), 6);
        let v1 = _mm256_srai_epi32(_mm256_add_epi32(v1, round), 6);
        let v2 = _mm256_srai_epi32(_mm256_add_epi32(v2, round), 6);
        let v3 = _mm256_srai_epi32(_mm256_add_epi32(v3, round), 6);

        // Saturate-pack i32 → i16, then permute(0xD8) to get natural
        // order in the low 128 bits of each ymm. See idct8x8_avx2.rs.
        let p0 = _mm256_packs_epi32(v0, v0);
        let p1 = _mm256_packs_epi32(v1, v1);
        let p2 = _mm256_packs_epi32(v2, v2);
        let p3 = _mm256_packs_epi32(v3, v3);
        let n0 = _mm256_permute4x64_epi64(p0, 0xD8);
        let n1 = _mm256_permute4x64_epi64(p1, 0xD8);
        let n2 = _mm256_permute4x64_epi64(p2, 0xD8);
        let n3 = _mm256_permute4x64_epi64(p3, 0xD8);
        let i16_a: __m128i = _mm256_castsi256_si128(n0); // cols 0..8
        let i16_b: __m128i = _mm256_castsi256_si128(n1); // cols 8..16
        let i16_c: __m128i = _mm256_castsi256_si128(n2); // cols 16..24
        let i16_d: __m128i = _mm256_castsi256_si128(n3); // cols 24..32

        // Load 32 u8 from dst row (unaligned, two 16-byte halves).
        // G4: loads from `dst` use _mm_loadu_si128 (unaligned)
        let dst_lo_ptr = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_hi_ptr = dest.as_mut_ptr().add(r * stride + 16) as *const __m128i;
        let dst_lo = _mm_loadu_si128(dst_lo_ptr);
        let dst_hi = _mm_loadu_si128(dst_hi_ptr);
        let dst_i16_a = _mm_unpacklo_epi8(dst_lo, zero); // cols 0..8
        let dst_i16_b = _mm_unpackhi_epi8(dst_lo, zero); // cols 8..16
        let dst_i16_c = _mm_unpacklo_epi8(dst_hi, zero); // cols 16..24
        let dst_i16_d = _mm_unpackhi_epi8(dst_hi, zero); // cols 24..32

        let sum_a = _mm_add_epi16(i16_a, dst_i16_a);
        let sum_b = _mm_add_epi16(i16_b, dst_i16_b);
        let sum_c = _mm_add_epi16(i16_c, dst_i16_c);
        let sum_d = _mm_add_epi16(i16_d, dst_i16_d);
        let sum_u8_lo = _mm_packus_epi16(sum_a, sum_b);
        let sum_u8_hi = _mm_packus_epi16(sum_c, sum_d);

        let dst_lo_mut = dest.as_mut_ptr().add(r * stride) as *mut __m128i;
        let dst_hi_mut = dest.as_mut_ptr().add(r * stride + 16) as *mut __m128i;
        _mm_storeu_si128(dst_lo_mut, sum_u8_lo);
        _mm_storeu_si128(dst_hi_mut, sum_u8_hi);
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

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct32x32_add`.
    //!
    //! **G4-γ refactor coverage** (mirrors the idct16x16 G4-γ test
    //! suite, F2 commit `7cc2fde`):
    //!
    //! - **`idct32x32_scalar_eq_avx2_proptest`** — 256-case proptest
    //!   with five stratified strategies (uniform, sparse, eob-bucket,
    //!   high-magnitude, pathological).
    //! - **`idct32x32_all_zero_input`** — zero coefficients leave dst
    //!   unchanged.
    //! - **`idct32x32_dc_only`** — single DC coefficient at position 0.
    //! - **`idct32x32_single_coefficient_impulses`** — sweep one
    //!   non-zero coefficient across all 1024 positions.
    //! - **`idct32x32_max_input`** — saturation stress.
    //! - **`idct32x32_misaligned_dst_pointers`** — per-skew arithmetic
    //!   regression guard.
    //! - **`idct32x32_dispatch_round_trip`** — forced-AVX2 vs
    //!   forced-scalar dispatch produces identical output.
    //! - **`idct32x32_full1d_pathological_directed`** — G4-γ directed
    //!   regression: row/col-dominant stress at every 8-row/8-col group
    //!   boundary plus high-magnitude alternating patterns.
    //! - **`idct32x32_transpose_sanity`** — standalone helper test for
    //!   the per-quadrant 8×8 transpose composition (catches transpose
    //!   bugs without the full butterfly).
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
    /// `idct32x32_34_add` / `_135_add` dispatcher boundaries; K=1024 is
    /// full density.
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
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(64).enumerate() {
            *slot = (i as i32) * 199 - 6300;
        }

        for skew in [0usize, 1, 3, 7, 15] {
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
    /// Tautology-guard: under forced-AVX2 caps the dispatch slot must
    /// NOT be the scalar function pointer.
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

    /// **G4-γ directed regression**: pathological row/col-dominant
    /// stress patterns at every 8-group boundary. Exercises:
    ///
    /// 1. Row 0 = ramp of high-magnitude positive values
    /// 2. Rows 8 / 16 / 24 = high-magnitude alternating sign (straddle
    ///    each row-group boundary — catches inter-group state leaks)
    /// 3. Column 0 = ramp (input set via column-0 coefficients)
    /// 4. Columns 8 / 16 / 24 = high-magnitude alternating sign
    /// 5. Single impulse at (31, 31) — odd-half stress at max position
    /// 6. Combined: row 0 ramp + row 8 alternating + impulse at (31,31)
    #[test]
    fn idct32x32_full1d_pathological_directed() {
        require_avx2!();

        // (1) Row 0 = ramp.
        let mut input = [0i32; 1024];
        for (c, slot) in input.iter_mut().take(32).enumerate() {
            *slot = (c as i32) * 1000 + 100;
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 1024];
            assert_match(&input, &dst_init);
        }

        // (2) Rows 8, 16, 24 = alternating sign (boundary tests).
        for &boundary_row in &[8usize, 16, 24] {
            let mut input = [0i32; 1024];
            for c in 0..32 {
                input[boundary_row * 32 + c] = if c & 1 == 0 { 28000 } else { -28000 };
            }
            for &dst_fill in &[0u8, 128, 255] {
                let dst_init = [dst_fill; 1024];
                assert_match(&input, &dst_init);
            }
        }

        // (3) Column 0 = ramp.
        let mut input = [0i32; 1024];
        for r in 0..32 {
            input[r * 32] = (r as i32) * 1000 + 100;
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 1024];
            assert_match(&input, &dst_init);
        }

        // (4) Columns 8, 16, 24 = alternating sign.
        for &boundary_col in &[8usize, 16, 24] {
            let mut input = [0i32; 1024];
            for r in 0..32 {
                input[r * 32 + boundary_col] = if r & 1 == 0 { 28000 } else { -28000 };
            }
            for &dst_fill in &[0u8, 128, 255] {
                let dst_init = [dst_fill; 1024];
                assert_match(&input, &dst_init);
            }
        }

        // (5) Single impulse at (31, 31) — odd-half stress.
        let mut input = [0i32; 1024];
        input[31 * 32 + 31] = 12345;
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 1024];
            assert_match(&input, &dst_init);
        }

        // (6) Combined.
        let mut input = [0i32; 1024];
        for c in 0..32 {
            input[c] = (c as i32) * 1000 + 100;
            input[8 * 32 + c] = if c & 1 == 0 { 28000 } else { -28000 };
        }
        input[31 * 32 + 31] = 12345;
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 1024];
            assert_match(&input, &dst_init);
        }
    }

    /// **G4-γ helper sanity**: standalone transpose unit test.
    ///
    /// Builds a 32×32 i32 matrix with `m[r][c] = r * 32 + c`, runs the
    /// per-quadrant 8×8 transpose composition the kernel uses, and
    /// confirms `m_transposed[r][c] == c * 32 + r`. The 32×32 transpose
    /// decomposes into a 4×4 grid of 8×8 sub-blocks: each sub-block is
    /// transposed in place, then sub-block (row_group, col_group) is
    /// stored at destination position (col_group, row_group).
    #[test]
    fn idct32x32_transpose_sanity() {
        require_avx2!();
        // SAFETY: gated on AVX2 detection.
        unsafe { idct32x32_transpose_sanity_inner() }
    }

    // G4: loads from `dst` use _mm_loadu_si128 (unaligned) — test-only helper does no `dst` I/O
    #[target_feature(enable = "avx2")]
    unsafe fn idct32x32_transpose_sanity_inner() {
        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

        // Build a 32×32 row-major matrix where m[r * 32 + c] = r*32 + c.
        let mut src = [0i32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                src[r * 32 + c] = (r * 32 + c) as i32;
            }
        }

        // Apply per-quadrant transpose: for each (row_group, col_group)
        // in 4×4, transpose the 8×8 sub-block and store at the swapped
        // (col_group, row_group) position.
        let mut dst = [0i32; 1024];
        for row_group in 0..4usize {
            for col_group in 0..4usize {
                let mut rs: [__m256i; 8] = [super::_mm256_set1_epi32(0); 8];
                for (r, slot) in rs.iter_mut().enumerate() {
                    let p = src.as_ptr().add((row_group * 8 + r) * 32 + col_group * 8)
                        as *const __m256i;
                    *slot = _mm256_loadu_si256(p);
                }
                let cols =
                    super::transpose8x8_i32(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6], rs[7]);
                for (k, &col) in cols.iter().enumerate() {
                    let p = dst
                        .as_mut_ptr()
                        .add((col_group * 8 + k) * 32 + row_group * 8)
                        as *mut __m256i;
                    _mm256_storeu_si256(p, col);
                }
            }
        }

        // Verify dst[r*32+c] == src[c*32+r] for all r,c.
        for r in 0..32 {
            for c in 0..32 {
                assert_eq!(
                    dst[r * 32 + c],
                    (c * 32 + r) as i32,
                    "transpose mismatch at (r={r}, c={c})"
                );
            }
        }
    }
}
