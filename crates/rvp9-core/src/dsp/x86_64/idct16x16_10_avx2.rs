//! 16×16 inverse-DCT + add — EOB ≤ 10 sparse-block fast path, x86_64
//! AVX2 implementation.
//!
//! **Wave G G1 — full-1D refactor of the F3 finisher-only EOB port.**
//! Both the row pass and the column pass are now fully vectorised in
//! AVX2 with 32-bit lanes, cloning the structure proven by the full-1D
//! parent kernel [`crate::dsp::x86_64::idct16x16_avx2`] (Wave F F2,
//! commit `7cc2fde`). Only the row count differs: the row pass runs the
//! first 8-row group ONLY (rows 0..8 of the input — rows 4..7 are
//! guaranteed zero by the caller contract). The second row group
//! (rows 8..16) is skipped via zero-fill — the intermediate scratch's
//! zero initialiser is bit-exact equivalent to running `idct16` over an
//! all-zero input.
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::transforms::inverse::idct16x16_10_add`].
//!
//! ## Caller contract
//!
//! Identical to the scalar reference: when `eob <= 10` the non-zero
//! coefficients are confined to the top-left 4×4 sub-block of the input
//! (raster order), so the row 1D pass only needs to run on the first 4
//! input rows; rows 4..16 of the input are assumed zero. Passing a
//! block whose true eob exceeds 10 silently drops coefficients in rows
//! 4..16 and diverges from the full kernel — callers are responsible
//! for the eob bucket gate.
//!
//! ## Bit-exactness vs scalar reference
//!
//! Same as the parent: every intermediate `wraplow` uses the modular
//! [`wrap`] helper (`srai(slli(x, 16), 16)`). The saturating
//! [`_mm256_packs_epi32`] is confined to the Stage 3 finisher's final
//! round/clip/pack stage where saturation is documented spec behaviour.
//!
//! The skip-and-zero-fill optimisation for the second row group is
//! bit-exact: `idct16` of an all-zero input is the all-zero output (see
//! the scalar `idct16x16_10_add` reference for the same pattern — it
//! does the row pass over rows 0..4 only and lets the zero initialiser
//! of `intermediate` carry rows 4..16 into the column pass). The first
//! row group still runs the full 8-wide AVX2 idct16 in parallel (rows
//! 4..7 contribute zero, producing zero output for the lower 4 lanes —
//! identical to the scalar truth).
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
    _mm256_srai_epi32, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_2_64, COSPI_4_64, COSPI_6_64, COSPI_8_64, COSPI_10_64, COSPI_12_64, COSPI_14_64,
    COSPI_16_64, COSPI_18_64, COSPI_20_64, COSPI_22_64, COSPI_24_64, COSPI_26_64, COSPI_28_64,
    COSPI_30_64,
};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers — duplicated verbatim from `idct16x16_avx2.rs`
// (Wave F F2 parent) to keep this kernel self-contained. All are
// `#[inline]` so the compiler folds them away.
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
/// 16 bits of each lane back to 32 bits — MODULAR i16 truncation, not
/// saturation. Critical: replacing with `_mm256_packs_epi32` (saturating)
/// diverges from the C reference on pathological inputs.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. Product fits in i32 for in-range operands.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m256i, c: i32) -> __m256i {
    _mm256_mullo_epi32(x, _mm256_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 8×8 i32 transpose — duplicated from `idct16x16_avx2::transpose8x8_i32`.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose8x8_i32(
    r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i,
    r4: __m256i, r5: __m256i, r6: __m256i, r7: __m256i,
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
// Vector 1D idct16 — duplicated verbatim from `idct16x16_avx2::idct16_avx2`.
// See parent module docstring for the 7-stage butterfly walkthrough.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct16_avx2(v: [__m256i; 16]) -> [__m256i; 16] {
    // ----- stage 1 -----
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

    // ----- stage 2 -----
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

    let s2_0 = s1_0;
    let s2_1 = s1_1;
    let s2_2 = s1_2;
    let s2_3 = s1_3;
    let s2_4 = s1_4;
    let s2_5 = s1_5;
    let s2_6 = s1_6;
    let s2_7 = s1_7;

    // ----- stage 3 -----
    let s3_0 = s2_0;
    let s3_1 = s2_1;
    let s3_2 = s2_2;
    let s3_3 = s2_3;

    let t1 = _mm256_sub_epi32(mul_const(s2_4, COSPI_28_64), mul_const(s2_7, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(s2_4, COSPI_4_64), mul_const(s2_7, COSPI_28_64));
    let s3_4 = wrap(drs(t1));
    let s3_7 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s2_5, COSPI_12_64), mul_const(s2_6, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(s2_5, COSPI_20_64), mul_const(s2_6, COSPI_12_64));
    let s3_5 = wrap(drs(t1));
    let s3_6 = wrap(drs(t2));

    let s3_8 = wrap(_mm256_add_epi32(s2_8, s2_9));
    let s3_9 = wrap(_mm256_sub_epi32(s2_8, s2_9));
    let s3_10 = wrap(_mm256_sub_epi32(s2_11, s2_10));
    let s3_11 = wrap(_mm256_add_epi32(s2_10, s2_11));
    let s3_12 = wrap(_mm256_add_epi32(s2_12, s2_13));
    let s3_13 = wrap(_mm256_sub_epi32(s2_12, s2_13));
    let s3_14 = wrap(_mm256_sub_epi32(s2_15, s2_14));
    let s3_15 = wrap(_mm256_add_epi32(s2_14, s2_15));

    // ----- stage 4 -----
    let t1 = mul_const(_mm256_add_epi32(s3_0, s3_1), COSPI_16_64);
    let t2 = mul_const(_mm256_sub_epi32(s3_0, s3_1), COSPI_16_64);
    let s4_0 = wrap(drs(t1));
    let s4_1 = wrap(drs(t2));

    let t1 = _mm256_sub_epi32(mul_const(s3_2, COSPI_24_64), mul_const(s3_3, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s3_2, COSPI_8_64), mul_const(s3_3, COSPI_24_64));
    let s4_2 = wrap(drs(t1));
    let s4_3 = wrap(drs(t2));

    let s4_4 = wrap(_mm256_add_epi32(s3_4, s3_5));
    let s4_5 = wrap(_mm256_sub_epi32(s3_4, s3_5));
    let s4_6 = wrap(_mm256_sub_epi32(s3_7, s3_6));
    let s4_7 = wrap(_mm256_add_epi32(s3_6, s3_7));

    let s4_8 = s3_8;
    let s4_15 = s3_15;
    let t1 = _mm256_sub_epi32(mul_const(s3_14, COSPI_24_64), mul_const(s3_9, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s3_9, COSPI_24_64), mul_const(s3_14, COSPI_8_64));
    let s4_9 = wrap(drs(t1));
    let s4_14 = wrap(drs(t2));
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
    let t1 = mul_const(_mm256_sub_epi32(s4_6, s4_5), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s4_5, s4_6), COSPI_16_64);
    let s5_5 = wrap(drs(t1));
    let s5_6 = wrap(drs(t2));
    let s5_7 = s4_7;

    let s5_8 = wrap(_mm256_add_epi32(s4_8, s4_11));
    let s5_9 = wrap(_mm256_add_epi32(s4_9, s4_10));
    let s5_10 = wrap(_mm256_sub_epi32(s4_9, s4_10));
    let s5_11 = wrap(_mm256_sub_epi32(s4_8, s4_11));
    let s5_12 = wrap(_mm256_sub_epi32(s4_15, s4_12));
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
    let t1 = mul_const(_mm256_sub_epi32(s5_13, s5_10), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_10, s5_13), COSPI_16_64);
    let s6_10 = wrap(drs(t1));
    let s6_13 = wrap(drs(t2));
    let t1 = mul_const(_mm256_sub_epi32(s5_12, s5_11), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s5_11, s5_12), COSPI_16_64);
    let s6_11 = wrap(drs(t1));
    let s6_12 = wrap(drs(t2));
    let s6_14 = s5_14;
    let s6_15 = s5_15;

    // ----- stage 7 -----
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

    [o0, o1, o2, o3, o4, o5, o6, o7,
     o8, o9, o10, o11, o12, o13, o14, o15]
}

// ---------------------------------------------------------------------
// Pass helper: run idct16 on 8 rows in parallel.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct16_pass_8rows(
    lo: [__m256i; 8],
    hi: [__m256i; 8],
) -> ([__m256i; 8], [__m256i; 8]) {
    let v: [__m256i; 16] = [
        lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
        hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7],
    ];
    let out = idct16_avx2(v);
    let out_lo: [__m256i; 8] = [out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]];
    let out_hi: [__m256i; 8] = [out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]];
    (out_lo, out_hi)
}

/// AVX2 implementation of the 16×16 inverse-DCT + residual add for the
/// `eob <= 10` sparse-block fast path.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct16x16_10_add_avx2`]) when `caps.avx2 == true`,
/// and caps are detected once at OnceLock init time.
///
/// `dest` must have at least `15 * stride + 16` bytes of valid storage.
///
/// # Implementation
///
/// **Wave G G1 full-1D** structure (clone of [`crate::dsp::x86_64::idct16x16_avx2`]
/// with row-count reduction):
///
/// 1. Row pass over the FIRST 8-row group only (rows 0..8): load 8
///    rows × 2 ymm each (lo/hi halves), transpose, parallel idct16,
///    transpose back, store to intermediate scratch. The caller
///    contract guarantees rows 4..7 of input are zero — the parallel
///    AVX2 idct16 produces zero outputs for those rows automatically.
/// 2. Skip the second row group (rows 8..16): `intermediate[8 * 16..]`
///    remains zero-initialised. Bit-exact equivalent of `idct16` of an
///    all-zero input (the scalar `idct16x16_10_add` uses the same
///    pattern).
/// 3. Column pass over all 16 cols (full, two 8-col strips).
/// 4. AVX2 add-clamp-pack finisher: identical to the parent kernel.
///
/// All loads/stores from / to `dest` are 16-byte (`__m128i`),
/// unaligned. See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct16x16_10_add_avx2_inner(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 15 + 16, "dest too small for 16x16 add");

    // Stack scratch for the intermediate (post-row-pass) 16×16 matrix
    // in row-major i32 layout. Zero-initialised so the skipped second
    // row group's intermediate rows (8..16) feed zero into the column
    // pass — bit-exact equivalent of running idct16 over zero input.
    let mut intermediate = [0i32; 256];

    // ----- Row pass: FIRST 8-row group only (rows 0..8). -----
    // Caller contract: input rows 4..16 are zero. We load rows 0..8 and
    // run the 8-wide AVX2 idct16; rows 4..7 contribute zero in their
    // lanes and produce zero output in those lanes — same truth as the
    // scalar reference, which iterates idct16 only over rows 0..4.
    {
        let in_ptr = input.as_ptr() as *const __m256i;
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

        let lo_cols = transpose8x8_i32(r0_lo, r1_lo, r2_lo, r3_lo, r4_lo, r5_lo, r6_lo, r7_lo);
        let hi_cols = transpose8x8_i32(r0_hi, r1_hi, r2_hi, r3_hi, r4_hi, r5_hi, r6_hi, r7_hi);

        let (out_lo, out_hi) = idct16_pass_8rows(lo_cols, hi_cols);

        let lo_rows = transpose8x8_i32(
            out_lo[0], out_lo[1], out_lo[2], out_lo[3],
            out_lo[4], out_lo[5], out_lo[6], out_lo[7],
        );
        let hi_rows = transpose8x8_i32(
            out_hi[0], out_hi[1], out_hi[2], out_hi[3],
            out_hi[4], out_hi[5], out_hi[6], out_hi[7],
        );

        let out_ptr = intermediate.as_mut_ptr() as *mut __m256i;
        for r in 0..8 {
            _mm256_storeu_si256(out_ptr.add(2 * r), lo_rows[r]);
            _mm256_storeu_si256(out_ptr.add(2 * r + 1), hi_rows[r]);
        }
    }

    // Second row group (rows 8..16) SKIPPED — `intermediate[8 * 16..]`
    // stays at zero per the caller contract.

    // ----- Column pass: process all 16 columns (two 8-column strips). -----
    let mut col_outputs: [__m256i; 32] = [_mm256_set1_epi32(0); 32];
    for col_group in 0..2 {
        let col_base = col_group * 8;

        let mut col_in: [__m256i; 16] = [_mm256_set1_epi32(0); 16];
        for r in 0..16 {
            let p = intermediate.as_ptr().add(r * 16 + col_base) as *const __m256i;
            col_in[r] = _mm256_loadu_si256(p);
        }

        let col_out = idct16_avx2(col_in);

        for r in 0..16 {
            col_outputs[col_group * 16 + r] = col_out[r];
        }
    }

    // ----- Finisher: per row, combine cols 0..8 and cols 8..16, round,
    // shift, pack, add to dst, saturate, store. -----
    let round = _mm256_set1_epi32(32);
    let zero = _mm_setzero_si128();
    for r in 0..16 {
        let lo_i32 = col_outputs[r];           // cols 0..8 of row r
        let hi_i32 = col_outputs[16 + r];      // cols 8..16 of row r

        let lo_i32 = _mm256_srai_epi32(_mm256_add_epi32(lo_i32, round), 6);
        let hi_i32 = _mm256_srai_epi32(_mm256_add_epi32(hi_i32, round), 6);

        let lo_packed = _mm256_packs_epi32(lo_i32, lo_i32);
        let hi_packed = _mm256_packs_epi32(hi_i32, hi_i32);
        let lo_natural = _mm256_permute4x64_epi64(lo_packed, 0xD8);
        let hi_natural = _mm256_permute4x64_epi64(hi_packed, 0xD8);
        let out_i16_lo: __m128i = _mm256_castsi256_si128(lo_natural);
        let out_i16_hi: __m128i = _mm256_castsi256_si128(hi_natural);

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

/// Safe wrapper for the AVX2 idct16x16_10 (EOB ≤ 10) inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct16x16_10_add`], so the
/// `SimdDispatch::idct16x16_10_add` slot can hold either.
///
/// **Caller contract:** same as the scalar reference — passing a block
/// whose true eob exceeds 10 silently diverges from the full kernel.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct16x16_10_add_avx2(input: &[i32; 256], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct16x16_10_avx2::tests` gates each call
    // on a runtime `is_x86_feature_detected!("avx2")` check, so calls
    // from tests are equally sound.
    unsafe { idct16x16_10_add_avx2_inner(input, dest, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct16x16_10_add`.
    //!
    //! Wave G G1 full-1D coverage (extends the original F3 finisher-only
    //! suite):
    //!
    //! - **`idct16x16_10_scalar_eq_avx2_proptest`** — 256-case proptest
    //!   with uniform / sparse / high-magnitude / pathological strategies
    //!   for the top-4-row coefficients.
    //! - **`idct16x16_10_all_zero_input`** — zero coefficients leave dst
    //!   unchanged.
    //! - **`idct16x16_10_dc_only`** — single DC coefficient at position 0
    //!   (full-1D K=1 sanity).
    //! - **`idct16x16_10_pathological_max_top_rows`** — saturation stress
    //!   with i16::MAX / i16::MIN coefficients in rows 0..4.
    //! - **`idct16x16_10_full1d_pathological_directed`** — Wave G G1
    //!   directed: [i16::MAX] / [i16::MIN] / alternating-sign in the
    //!   top-4 rows verifying bit-exactness through the full row+col
    //!   AVX2 path.
    //! - **`idct16x16_10_dispatch_round_trip`** — tautology guard.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs.

    use super::idct16x16_10_add_avx2;
    use crate::transforms::inverse::idct16x16_10_add;
    use proptest::prelude::*;

    /// Top-K-row coefficient count for this fast path. `idct16x16_10`
    /// is the EOB ≤ 10 bucket → top-left 4×4 sub-block, which spans
    /// the first 4 rows of the raster-order input array. Rows K..16
    /// MUST be zero in every proptest input.
    const TOP_ROWS: usize = 4;
    const TOP_COEFFS: usize = TOP_ROWS * 16;

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
        idct16x16_10_add_avx2(input, &mut dst, 16);
        dst
    }

    fn run_scalar(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        let mut dst = *dst_init;
        idct16x16_10_add(input, &mut dst, 16);
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
    // Strategies — every `[i32; 256]` MUST zero rows TOP_ROWS..16.
    // -----------------------------------------------------------------

    fn uniform_coeff() -> impl Strategy<Value = i32> {
        -32768i32..=32767
    }

    fn uniform_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(uniform_coeff(), TOP_COEFFS).prop_map(|v| {
            let mut out = [0i32; 256];
            out[..TOP_COEFFS].copy_from_slice(&v);
            out
        })
    }

    fn sparse_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec((0usize..TOP_COEFFS, uniform_coeff()), 1..=8).prop_map(|nz| {
            let mut out = [0i32; 256];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(
            (30000i32..=32767, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            TOP_COEFFS,
        )
        .prop_map(|v| {
            let mut out = [0i32; 256];
            out[..TOP_COEFFS].copy_from_slice(&v);
            out
        })
    }

    fn pathological_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop_oneof![
            Just({
                let mut out = [0i32; 256];
                for slot in out[..TOP_COEFFS].iter_mut() {
                    *slot = i16::MAX as i32;
                }
                out
            }),
            Just({
                let mut out = [0i32; 256];
                for slot in out[..TOP_COEFFS].iter_mut() {
                    *slot = i16::MIN as i32;
                }
                out
            }),
            Just({
                let mut out = [0i32; 256];
                for (i, slot) in out[..TOP_COEFFS].iter_mut().enumerate() {
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
        fn idct16x16_10_scalar_eq_avx2_proptest(
            input in input_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            for i in TOP_COEFFS..256 {
                prop_assert_eq!(input[i], 0, "test bug: row {} must be zero", i / 16);
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
    fn idct16x16_10_all_zero_input() {
        require_avx2!();
        let input = [0i32; 256];
        let dst_init = [42u8; 256];
        assert_match(&input, &dst_init);
    }

    /// DC-only K=1 sanity: single coefficient at position 0 exercising
    /// the full-1D path with sparse input.
    #[test]
    fn idct16x16_10_dc_only() {
        require_avx2!();
        let mut input = [0i32; 256];
        input[0] = 1234;
        let dst_init = [50u8; 256];
        assert_match(&input, &dst_init);
    }

    /// Directed pathological: `[i16::MAX; 4*16]` in rows 0..4, zero in
    /// rows 4..16.
    #[test]
    fn idct16x16_10_pathological_max_top_rows() {
        require_avx2!();
        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MAX as i32;
        }
        let dst_init = [0u8; 256];
        assert_match(&input, &dst_init);

        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MIN as i32;
        }
        let dst_init = [255u8; 256];
        assert_match(&input, &dst_init);
    }

    /// **Wave G G1 full-1D directed regression**: pathological
    /// row-dominant stress in the top-4 rows. Exercises the parallel
    /// AVX2 row pass (where the four lanes for rows 4..7 carry zero —
    /// must not corrupt the four active lanes), the transpose, and the
    /// full column pass.
    ///
    /// Each pattern uses `[i16::MAX]` / `[i16::MIN]` / alternating sign
    /// fills in the top-4 rows, verifying bit-exactness against the
    /// scalar reference for a representative set of dst initialisers.
    #[test]
    fn idct16x16_10_full1d_pathological_directed() {
        require_avx2!();

        // (1) i16::MAX across rows 0..4.
        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MAX as i32;
        }
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (2) i16::MIN across rows 0..4.
        let mut input = [0i32; 256];
        for slot in input[..TOP_COEFFS].iter_mut() {
            *slot = i16::MIN as i32;
        }
        for &dst_fill in &[0u8, 64, 128, 200, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (3) Alternating MAX/MIN across rows 0..4.
        let mut input = [0i32; 256];
        for (i, slot) in input[..TOP_COEFFS].iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (4) High-magnitude ramp on row 0 only — column-dominant.
        let mut input = [0i32; 256];
        for c in 0..16 {
            input[c] = (c as i32) * 2000 + 100;
        }
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }

        // (5) Single high-magnitude impulse at (3, 3) — within the
        //     top-left 4×4 sub-block — exercises both row 3 and col 3.
        let mut input = [0i32; 256];
        input[3 * 16 + 3] = 28000;
        for &dst_fill in &[0u8, 128, 255] {
            let dst_init = [dst_fill; 256];
            assert_match(&input, &dst_init);
        }
    }

    /// Tautology guard: forced-AVX2 dispatch slot must NOT be the
    /// scalar fn pointer, and the AVX2 wrapper must match the scalar
    /// reference on a non-trivial input.
    #[test]
    fn idct16x16_10_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let scalar_fp: fn(&[i32; 256], &mut [u8], usize) = idct16x16_10_add;
        assert_ne!(
            dispatch_avx2.idct16x16_10_add() as usize,
            scalar_fp as usize,
            "forced-AVX2 caps must install the AVX2 wrapper, not the scalar fn"
        );

        let mut input = [0i32; 256];
        for (i, slot) in input[..TOP_COEFFS].iter_mut().enumerate() {
            *slot = ((i as i32) * 29) % 4001 - 2000;
        }
        let dst_init = [97u8; 256];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct16x16_10_add())(&input, &mut dst_avx2, 16);
        (dispatch_scalar.idct16x16_10_add())(&input, &mut dst_scalar, 16);
        assert_eq!(dst_avx2, dst_scalar);
    }
}
