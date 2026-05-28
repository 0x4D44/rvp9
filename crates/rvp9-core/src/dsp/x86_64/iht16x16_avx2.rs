//! 16×16 hybrid inverse-transform + add — x86_64 AVX2 implementation.
//!
//! **Wave G G2 — full-1D refactor of F5 finisher-only.** Both the row
//! pass and the column pass are now fully vectorised in AVX2 with
//! 32-bit lanes for ALL four [`TxType`] variants. The previous Wave F
//! F5 pilot kept the scalar [`crate::transforms::inverse::idct16`] /
//! [`crate::transforms::inverse::iadst16`] calls and AVX2-accelerated
//! only the final `(out + 32) >> 6` + add + clip + pack stage. This
//! file mirrors the [`crate::dsp::x86_64::idct16x16_avx2`] full-1D
//! template (F2, `7cc2fde`) and the [`crate::dsp::x86_64::iht8x8_avx2`]
//! `tx_type` dispatch pattern (F7, `eddf458`), with a freshly
//! hand-derived `iadst16_avx2` for the ADST 16-point butterfly.
//!
//! Scalar reference: [`crate::transforms::inverse::iht16x16_256_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Every scalar operation in [`crate::transforms::inverse::idct16`] and
//! [`crate::transforms::inverse::iadst16`] is mirrored 1:1 in vector
//! form across 8 lanes (one row of the 8-row sub-block per lane). The
//! row pass runs twice (rows 0..8 and rows 8..16); the column pass runs
//! twice (cols 0..8 and cols 8..16).
//!
//! The critical correctness rule is the same as the rest of the AVX2
//! inverse-transform family:
//!
//! * `wraplow(x) = x as i16 as i32` is MODULAR i16 truncation, **not**
//!   saturation. The vector equivalent is `srai_epi32(slli_epi32(x,
//!   16), 16)` — sign-extend the low 16 bits. We MUST NOT use
//!   [`_mm256_packs_epi32`] (which saturates) for any intermediate
//!   wraplow: libvpx's SSE2 kernels take that shortcut and diverge from
//!   the C reference on pathological inputs. The
//!   `iht16x16_full1d_pathological_directed_*` fixtures lock that
//!   discipline in for all 4 tx_types.
//! * `dct_const_round_shift(x) = (x + 8192) >> 14` — [`_mm256_add_epi32`]
//!   + [`_mm256_srai_epi32`] by 14.
//! * Multiplies use [`_mm256_mullo_epi32`] (low 32 bits of the product).
//!
//! ## Operand-range safety
//!
//! Every intermediate is either a `wraplow` result (i16-range) or a raw
//! spec-bounded input coefficient (also i16). Cospi constants are
//! `< 2^15`. A single product `cospi * x` fits in i32 with peak
//! magnitude ≈ `2^15 * 2^15 = 2^30 ≈ 1.07e9`. The widest sum we form is
//! the iadst16 stage-1 4-product butterfly (`(c1*x0 + c31*x1) + (c17*x8
//! + c15*x9)`), worst-case ≈ `4 * 2^14 * 2^15 = 2^31 ≈ 2.14e9`. This
//! fits in `i32::MAX = 2.147e9` with a margin of ≈ 6 million — the
//! same envelope already proven by [`crate::dsp::x86_64::iht8x8_avx2`]
//! iadst8 stage 1. Therefore the scalar reference's `i64` intermediate
//! arithmetic can be folded to `i32` without divergence; libvpx's own
//! SSE2 iadst16 makes the same choice (`vpx_iadst16_8col_sse2`).
//!
//! ## TxType dispatch
//!
//! Per the [`crate::transforms::inverse`] `iht_16_dispatch` table —
//! libvpx struct order is `{cols, rows}`:
//!
//! | `tx_type`   | row pass        | col pass        |
//! |-------------|-----------------|-----------------|
//! | `DctDct`    | `idct16_avx2`   | `idct16_avx2`   |
//! | `AdstDct`   | `idct16_avx2`   | `iadst16_avx2`  |
//! | `DctAdst`   | `iadst16_avx2`  | `idct16_avx2`   |
//! | `AdstAdst`  | `iadst16_avx2`  | `iadst16_avx2`  |
//!
//! `DctDct` is **not** delegated to the dedicated `idct16x16_add_avx2`
//! kernel — we self-implement so the iht slot remains independently
//! testable and dispatch round-trip checks have a clean `assert_ne!`
//! against the scalar `iht16x16_256_add` reference.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadu_si128` (16 bytes per
//! row); coefficient-buffer reads via `_mm256_loadu_si256`. The grep
//! sentinel below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadu_si128, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_packs_epi32,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_slli_epi32, _mm256_srai_epi32, _mm256_storeu_si256, _mm256_sub_epi32,
    _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64,
};
use crate::transforms::types::TxType;

// ---------------------------------------------------------------------
// Tiny intrinsic helpers — duplicated verbatim from `idct16x16_avx2.rs`
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
/// 16 bits of each lane back to 32 bits — MODULAR i16 truncation,
/// NOT saturation. Replacing this with `_mm256_packs_epi32` would
/// diverge from the scalar reference on pathological inputs.
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
// 8×8 i32 transpose using a 3-stage unpack pyramid. Duplicate of the
// helper in `idct16x16_avx2.rs`.
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
// Vector 1D idct16. Mirrors `idct16` in `crate::transforms::inverse`
// line-by-line across 8 lanes; see `idct16x16_avx2.rs` for the full
// stage-by-stage commentary.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct16_avx2(v: [__m256i; 16]) -> [__m256i; 16] {
    // ----- stage 1: bit-reversed input gather. -----
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
    // step2[10] = wrap(drs(-(step1[10] * C24 + step1[13] * C8)))
    // step2[13] = wrap(drs( step1[13] * C24 - step1[10] * C8))
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
// Vector 1D iadst16 — hand-derived from the scalar reference at
// `crate::transforms::inverse::iadst16` (lines 1230-1396). Each
// `__m256i` lane holds one row's worth of input; the butterfly runs 8
// rows in parallel.
//
// 4-stage butterfly using cospi at frequencies 1/3/5/7/9/11/13/15/17/
// 19/21/23/25/27/29/31 (stage 1), 4/12/20/28 (stage 2), 8/24 (stage 3),
// and 16 (stage 4), followed by an output permutation with sign flips.
//
// Operand-range note (see module docstring): the stage-1 4-product
// butterfly `s{N} + s{N+8}` peaks at ≈ 2.14e9, fitting in i32 with
// ≈ 6 million margin. Every intermediate is `wrap`-truncated; the use
// of `_mm256_packs_epi32` is restricted to the Stage 3 finisher.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn iadst16_avx2(v: [__m256i; 16]) -> [__m256i; 16] {
    // libvpx remap: x0..x15 = in[15,0,13,2,11,4,9,6,7,8,5,10,3,12,1,14].
    let x0 = v[15];
    let x1 = v[0];
    let x2 = v[13];
    let x3 = v[2];
    let x4 = v[11];
    let x5 = v[4];
    let x6 = v[9];
    let x7 = v[6];
    let x8 = v[7];
    let x9 = v[8];
    let x10 = v[5];
    let x11 = v[10];
    let x12 = v[3];
    let x13 = v[12];
    let x14 = v[1];
    let x15 = v[14];

    // ---- Stage 1 ----
    // s0  = c1*x0  + c31*x1   s1  = c31*x0 - c1*x1
    // s2  = c5*x2  + c27*x3   s3  = c27*x2 - c5*x3
    // s4  = c9*x4  + c23*x5   s5  = c23*x4 - c9*x5
    // s6  = c13*x6 + c19*x7   s7  = c19*x6 - c13*x7
    // s8  = c17*x8 + c15*x9   s9  = c15*x8 - c17*x9
    // s10 = c21*x10+ c11*x11  s11 = c11*x10- c21*x11
    // s12 = c25*x12+ c7*x13   s13 = c7*x12 - c25*x13
    // s14 = c29*x14+ c3*x15   s15 = c3*x14 - c29*x15
    let s0 = _mm256_add_epi32(mul_const(x0, COSPI_1_64), mul_const(x1, COSPI_31_64));
    let s1 = _mm256_sub_epi32(mul_const(x0, COSPI_31_64), mul_const(x1, COSPI_1_64));
    let s2 = _mm256_add_epi32(mul_const(x2, COSPI_5_64), mul_const(x3, COSPI_27_64));
    let s3 = _mm256_sub_epi32(mul_const(x2, COSPI_27_64), mul_const(x3, COSPI_5_64));
    let s4 = _mm256_add_epi32(mul_const(x4, COSPI_9_64), mul_const(x5, COSPI_23_64));
    let s5 = _mm256_sub_epi32(mul_const(x4, COSPI_23_64), mul_const(x5, COSPI_9_64));
    let s6 = _mm256_add_epi32(mul_const(x6, COSPI_13_64), mul_const(x7, COSPI_19_64));
    let s7 = _mm256_sub_epi32(mul_const(x6, COSPI_19_64), mul_const(x7, COSPI_13_64));
    let s8 = _mm256_add_epi32(mul_const(x8, COSPI_17_64), mul_const(x9, COSPI_15_64));
    let s9 = _mm256_sub_epi32(mul_const(x8, COSPI_15_64), mul_const(x9, COSPI_17_64));
    let s10 = _mm256_add_epi32(mul_const(x10, COSPI_21_64), mul_const(x11, COSPI_11_64));
    let s11 = _mm256_sub_epi32(mul_const(x10, COSPI_11_64), mul_const(x11, COSPI_21_64));
    let s12 = _mm256_add_epi32(mul_const(x12, COSPI_25_64), mul_const(x13, COSPI_7_64));
    let s13 = _mm256_sub_epi32(mul_const(x12, COSPI_7_64), mul_const(x13, COSPI_25_64));
    let s14 = _mm256_add_epi32(mul_const(x14, COSPI_29_64), mul_const(x15, COSPI_3_64));
    let s15 = _mm256_sub_epi32(mul_const(x14, COSPI_3_64), mul_const(x15, COSPI_29_64));

    // x{0..15} = wraplow(drs(s{0..7} ± s{8..15}))
    let x0 = wrap(drs(_mm256_add_epi32(s0, s8)));
    let x1 = wrap(drs(_mm256_add_epi32(s1, s9)));
    let x2 = wrap(drs(_mm256_add_epi32(s2, s10)));
    let x3 = wrap(drs(_mm256_add_epi32(s3, s11)));
    let x4 = wrap(drs(_mm256_add_epi32(s4, s12)));
    let x5 = wrap(drs(_mm256_add_epi32(s5, s13)));
    let x6 = wrap(drs(_mm256_add_epi32(s6, s14)));
    let x7 = wrap(drs(_mm256_add_epi32(s7, s15)));
    let x8 = wrap(drs(_mm256_sub_epi32(s0, s8)));
    let x9 = wrap(drs(_mm256_sub_epi32(s1, s9)));
    let x10 = wrap(drs(_mm256_sub_epi32(s2, s10)));
    let x11 = wrap(drs(_mm256_sub_epi32(s3, s11)));
    let x12 = wrap(drs(_mm256_sub_epi32(s4, s12)));
    let x13 = wrap(drs(_mm256_sub_epi32(s5, s13)));
    let x14 = wrap(drs(_mm256_sub_epi32(s6, s14)));
    let x15 = wrap(drs(_mm256_sub_epi32(s7, s15)));

    // ---- Stage 2 ----
    // s0..s7 pass through (x0..x7 are already wrap-truncated).
    // s8  = c4*x8   + c28*x9    s9  = c28*x8  - c4*x9
    // s10 = c20*x10 + c12*x11   s11 = c12*x10 - c20*x11
    // s12 = -c28*x12 + c4*x13  → c4*x13 - c28*x12
    // s13 = c4*x12  + c28*x13
    // s14 = -c12*x14 + c20*x15 → c20*x15 - c12*x14
    // s15 = c20*x14 + c12*x15
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = x4;
    let s5 = x5;
    let s6 = x6;
    let s7 = x7;
    let s8 = _mm256_add_epi32(mul_const(x8, COSPI_4_64), mul_const(x9, COSPI_28_64));
    let s9 = _mm256_sub_epi32(mul_const(x8, COSPI_28_64), mul_const(x9, COSPI_4_64));
    let s10 = _mm256_add_epi32(mul_const(x10, COSPI_20_64), mul_const(x11, COSPI_12_64));
    let s11 = _mm256_sub_epi32(mul_const(x10, COSPI_12_64), mul_const(x11, COSPI_20_64));
    let s12 = _mm256_sub_epi32(mul_const(x13, COSPI_4_64), mul_const(x12, COSPI_28_64));
    let s13 = _mm256_add_epi32(mul_const(x12, COSPI_4_64), mul_const(x13, COSPI_28_64));
    let s14 = _mm256_sub_epi32(mul_const(x15, COSPI_20_64), mul_const(x14, COSPI_12_64));
    let s15 = _mm256_add_epi32(mul_const(x14, COSPI_20_64), mul_const(x15, COSPI_12_64));

    // x0..x7 = wraplow(s0..s3 ± s4..s7) — NO drs because s0..s7 are
    // already wrap-truncated i16-range values from stage 1.
    let x0 = wrap(_mm256_add_epi32(s0, s4));
    let x1 = wrap(_mm256_add_epi32(s1, s5));
    let x2 = wrap(_mm256_add_epi32(s2, s6));
    let x3 = wrap(_mm256_add_epi32(s3, s7));
    let x4 = wrap(_mm256_sub_epi32(s0, s4));
    let x5 = wrap(_mm256_sub_epi32(s1, s5));
    let x6 = wrap(_mm256_sub_epi32(s2, s6));
    let x7 = wrap(_mm256_sub_epi32(s3, s7));
    let x8 = wrap(drs(_mm256_add_epi32(s8, s12)));
    let x9 = wrap(drs(_mm256_add_epi32(s9, s13)));
    let x10 = wrap(drs(_mm256_add_epi32(s10, s14)));
    let x11 = wrap(drs(_mm256_add_epi32(s11, s15)));
    let x12 = wrap(drs(_mm256_sub_epi32(s8, s12)));
    let x13 = wrap(drs(_mm256_sub_epi32(s9, s13)));
    let x14 = wrap(drs(_mm256_sub_epi32(s10, s14)));
    let x15 = wrap(drs(_mm256_sub_epi32(s11, s15)));

    // ---- Stage 3 ----
    // s0..s3 pass through; s8..s11 pass through.
    // s4  = c8*x4  + c24*x5      s5  = c24*x4 - c8*x5
    // s6  = -c24*x6 + c8*x7  →  c8*x7 - c24*x6
    // s7  = c8*x6  + c24*x7
    // s12 = c8*x12 + c24*x13     s13 = c24*x12- c8*x13
    // s14 = -c24*x14+ c8*x15 →  c8*x15- c24*x14
    // s15 = c8*x14 + c24*x15
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = _mm256_add_epi32(mul_const(x4, COSPI_8_64), mul_const(x5, COSPI_24_64));
    let s5 = _mm256_sub_epi32(mul_const(x4, COSPI_24_64), mul_const(x5, COSPI_8_64));
    let s6 = _mm256_sub_epi32(mul_const(x7, COSPI_8_64), mul_const(x6, COSPI_24_64));
    let s7 = _mm256_add_epi32(mul_const(x6, COSPI_8_64), mul_const(x7, COSPI_24_64));
    let s8 = x8;
    let s9 = x9;
    let s10 = x10;
    let s11 = x11;
    let s12 = _mm256_add_epi32(mul_const(x12, COSPI_8_64), mul_const(x13, COSPI_24_64));
    let s13 = _mm256_sub_epi32(mul_const(x12, COSPI_24_64), mul_const(x13, COSPI_8_64));
    let s14 = _mm256_sub_epi32(mul_const(x15, COSPI_8_64), mul_const(x14, COSPI_24_64));
    let s15 = _mm256_add_epi32(mul_const(x14, COSPI_8_64), mul_const(x15, COSPI_24_64));

    // x0..x3 = wraplow(s0..s1 ± s2..s3)  (no drs — pass-through inputs)
    // x4..x7 = wraplow(drs(s4..s5 ± s6..s7))
    // x8..x11 = wraplow(s8..s9 ± s10..s11) (no drs — pass-through)
    // x12..x15 = wraplow(drs(s12..s13 ± s14..s15))
    let x0 = wrap(_mm256_add_epi32(s0, s2));
    let x1 = wrap(_mm256_add_epi32(s1, s3));
    let x2 = wrap(_mm256_sub_epi32(s0, s2));
    let x3 = wrap(_mm256_sub_epi32(s1, s3));
    let x4 = wrap(drs(_mm256_add_epi32(s4, s6)));
    let x5 = wrap(drs(_mm256_add_epi32(s5, s7)));
    let x6 = wrap(drs(_mm256_sub_epi32(s4, s6)));
    let x7 = wrap(drs(_mm256_sub_epi32(s5, s7)));
    let x8 = wrap(_mm256_add_epi32(s8, s10));
    let x9 = wrap(_mm256_add_epi32(s9, s11));
    let x10 = wrap(_mm256_sub_epi32(s8, s10));
    let x11 = wrap(_mm256_sub_epi32(s9, s11));
    let x12 = wrap(drs(_mm256_add_epi32(s12, s14)));
    let x13 = wrap(drs(_mm256_add_epi32(s13, s15)));
    let x14 = wrap(drs(_mm256_sub_epi32(s12, s14)));
    let x15 = wrap(drs(_mm256_sub_epi32(s13, s15)));

    // ---- Stage 4 ----
    // s2  = -c16*(x2 + x3)        s3  =  c16*(x2 - x3)
    // s6  =  c16*(x6 + x7)        s7  =  c16*(x7 - x6)
    // s10 =  c16*(x10 + x11)      s11 =  c16*(x11 - x10)
    // s14 = -c16*(x14 + x15)      s15 =  c16*(x14 - x15)
    let zero = _mm256_setzero_si256();
    let s2 = _mm256_sub_epi32(zero, mul_const(_mm256_add_epi32(x2, x3), COSPI_16_64));
    let s3 = mul_const(_mm256_sub_epi32(x2, x3), COSPI_16_64);
    let s6 = mul_const(_mm256_add_epi32(x6, x7), COSPI_16_64);
    let s7 = mul_const(_mm256_sub_epi32(x7, x6), COSPI_16_64);
    let s10 = mul_const(_mm256_add_epi32(x10, x11), COSPI_16_64);
    let s11 = mul_const(_mm256_sub_epi32(x11, x10), COSPI_16_64);
    let s14 = _mm256_sub_epi32(zero, mul_const(_mm256_add_epi32(x14, x15), COSPI_16_64));
    let s15 = mul_const(_mm256_sub_epi32(x14, x15), COSPI_16_64);

    let x2 = wrap(drs(s2));
    let x3 = wrap(drs(s3));
    let x6 = wrap(drs(s6));
    let x7 = wrap(drs(s7));
    let x10 = wrap(drs(s10));
    let x11 = wrap(drs(s11));
    let x14 = wrap(drs(s14));
    let x15 = wrap(drs(s15));

    // ---- Output permutation + sign flips ----
    // out = [ x0, -x8,  x12, -x4,  x6,  x14, x10,  x2,
    //         x3,  x11, x15,  x7,  x5, -x13, x9, -x1 ]
    let o0 = wrap(x0);
    let o1 = wrap(_mm256_sub_epi32(zero, x8));
    let o2 = wrap(x12);
    let o3 = wrap(_mm256_sub_epi32(zero, x4));
    let o4 = wrap(x6);
    let o5 = wrap(x14);
    let o6 = wrap(x10);
    let o7 = wrap(x2);
    let o8 = wrap(x3);
    let o9 = wrap(x11);
    let o10 = wrap(x15);
    let o11 = wrap(x7);
    let o12 = wrap(x5);
    let o13 = wrap(_mm256_sub_epi32(zero, x13));
    let o14 = wrap(x9);
    let o15 = wrap(_mm256_sub_epi32(zero, x1));

    [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ]
}

// ---------------------------------------------------------------------
// Pass helpers: dispatch the row / column 1D transform per `tx_type`.
// Mirrors the iht8x8 `run_2d` shape (F7).
// ---------------------------------------------------------------------

/// Run the appropriate row-pass kernel for `tx_type` on 8 rows in
/// parallel.
///
/// Input layout: 16 ymm of 8 lanes each, where `v[k]` holds input
/// position k across the 8 rows being transformed (one row per lane).
/// This is what comes out of `transpose8x8_i32` applied to each 8×8
/// quadrant of the row group's input.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn row_pass_16(v: [__m256i; 16], tx_type: TxType) -> [__m256i; 16] {
    // Per the iht_16_dispatch table, libvpx struct order is `{cols,
    // rows}`. So `AdstDct` = col=iadst16, row=idct16.
    match tx_type {
        TxType::DctDct | TxType::AdstDct => idct16_avx2(v),
        TxType::DctAdst | TxType::AdstAdst => iadst16_avx2(v),
    }
}

/// Run the appropriate column-pass kernel for `tx_type`.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn col_pass_16(v: [__m256i; 16], tx_type: TxType) -> [__m256i; 16] {
    match tx_type {
        TxType::DctDct | TxType::DctAdst => idct16_avx2(v),
        TxType::AdstDct | TxType::AdstAdst => iadst16_avx2(v),
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
/// Full-1D AVX2 (Wave G G2). Same structure as
/// [`crate::dsp::x86_64::idct16x16_avx2`] (F2) with the row / column
/// kernel picked per `tx_type`:
///
/// 1. Row pass over rows 0..8: load 8 rows × 2 ymm each (lo / hi
///    halves), transpose each 8×8 quadrant so each ymm holds one
///    input position across 8 rows, dispatch to `row_pass_16`, then
///    transpose back to row-major into a stack scratch buffer.
/// 2. Row pass over rows 8..16: same again.
/// 3. Column pass over cols 0..8: load 16 i32 per column-strip from
///    scratch, dispatch to `col_pass_16` (input already in
///    "per-position-per-lane" layout — no transpose needed since each
///    ymm = 8 cols of one row).
/// 4. Column pass over cols 8..16: same again.
/// 5. AVX2 add-clamp-pack finisher.
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
    debug_assert!(
        dest.len() >= stride * 15 + 16,
        "dest too small for 16x16 add"
    );

    // Stack scratch for the intermediate (post-row-pass) 16×16 matrix
    // in row-major i32 layout.
    let mut intermediate = [0i32; 256];

    // ----- Row pass: process 8 rows at a time. -----
    for row_group in 0..2 {
        let row_base = row_group * 8;

        // Load 8 input rows × 2 ymm each.
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
        // position across the 8 rows.
        let lo_cols = transpose8x8_i32(r0_lo, r1_lo, r2_lo, r3_lo, r4_lo, r5_lo, r6_lo, r7_lo);
        let hi_cols = transpose8x8_i32(r0_hi, r1_hi, r2_hi, r3_hi, r4_hi, r5_hi, r6_hi, r7_hi);

        // Apply 1D row transform over 8 rows in parallel.
        let v: [__m256i; 16] = [
            lo_cols[0], lo_cols[1], lo_cols[2], lo_cols[3], lo_cols[4], lo_cols[5], lo_cols[6],
            lo_cols[7], hi_cols[0], hi_cols[1], hi_cols[2], hi_cols[3], hi_cols[4], hi_cols[5],
            hi_cols[6], hi_cols[7],
        ];
        let out = row_pass_16(v, tx_type);
        let out_lo: [__m256i; 8] = [
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
        ];
        let out_hi: [__m256i; 8] = [
            out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15],
        ];

        // Transpose back to row-major.
        let lo_rows = transpose8x8_i32(
            out_lo[0], out_lo[1], out_lo[2], out_lo[3], out_lo[4], out_lo[5], out_lo[6], out_lo[7],
        );
        let hi_rows = transpose8x8_i32(
            out_hi[0], out_hi[1], out_hi[2], out_hi[3], out_hi[4], out_hi[5], out_hi[6], out_hi[7],
        );

        // Store into intermediate (row-major).
        let out_ptr = intermediate.as_mut_ptr().add(row_base * 16) as *mut __m256i;
        for r in 0..8 {
            _mm256_storeu_si256(out_ptr.add(2 * r), lo_rows[r]);
            _mm256_storeu_si256(out_ptr.add(2 * r + 1), hi_rows[r]);
        }
    }

    // ----- Column pass: process 8 columns at a time. -----
    //
    // For each 8-column strip, load 16 rows × 8 i32 directly from the
    // intermediate row-major buffer — each ymm IS the 8 i32 at row r,
    // cols `col_base..col_base+8`. That's exactly the layout
    // `idct16_avx2` / `iadst16_avx2` expect (ymm[k] = row-k = input
    // position k across 8 cols), so no transpose is needed.
    let mut col_outputs: [__m256i; 32] = [_mm256_set1_epi32(0); 32];
    for col_group in 0..2 {
        let col_base = col_group * 8;

        let mut col_in: [__m256i; 16] = [_mm256_set1_epi32(0); 16];
        for r in 0..16 {
            let p = intermediate.as_ptr().add(r * 16 + col_base) as *const __m256i;
            col_in[r] = _mm256_loadu_si256(p);
        }

        let col_out = col_pass_16(col_in, tx_type);

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

        // Saturate-pack i32 → i16 (per-128-bit-lane), then permute by
        // 64-bit qwords (control 0xD8 = [0, 2, 1, 3]) so the low 128
        // bits hold 8 i16 in natural order.
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
    //! Scalar-vs-AVX2 differential tests for `iht16x16_add` (Wave G G2
    //! full-1D).
    //!
    //! Coverage spans all 4 tx_types per the G2 acceptance criteria:
    //!
    //! - **`iht16x16_scalar_eq_avx2_proptest`** — 256-case proptest with
    //!   five stratified strategies (uniform, sparse, eob-bucket,
    //!   high-magnitude, pathological) × four tx_types.
    //! - **`iht16x16_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (all four tx_types).
    //! - **`iht16x16_dc_only`** — single DC coefficient at position 0
    //!   (all four tx_types).
    //! - **`iht16x16_single_coefficient_impulses`** — sweep one non-zero
    //!   coefficient across every position 0..256 with several
    //!   magnitudes (per tx_type).
    //! - **`iht16x16_max_input`** — saturation stress with i16::MAX /
    //!   i16::MIN coefficients (per tx_type).
    //! - **`iht16x16_pathological_per_tx_type`** — directed
    //!   `[i16::MAX, i16::MIN, ...]` alternating pattern (per tx_type).
    //! - **`iht16x16_misaligned_dst_pointers`** — per-skew arithmetic
    //!   regression guard (kernel uses `_mm_loadu_si128`, unaligned).
    //! - **`iht16x16_full1d_pathological_directed_{dctdct,adstdct,
    //!   dctadst,adstadst}`** — Wave G G2 per-tx_type directed
    //!   pathological fixtures (`[i16::MAX]`, `[i16::MIN]`, alternating
    //!   signs) guaranteed to run regardless of proptest distribution.
    //! - **`iadst16_avx2_standalone`** — scalar-vs-vector `iadst16`
    //!   sanity check on a single 16-vector input (proves the new 1D
    //!   function in isolation, before it's used inside the 2D kernel).
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
        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().take(32).enumerate() {
            *slot = (i as i32) * 199 - 3100;
        }

        for &tx_type in &ALL_TX_TYPES {
            for skew in [0usize, 1, 3, 7, 15] {
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

    // -----------------------------------------------------------------
    // Wave G G2 pathological directed fixtures — one per tx_type.
    // -----------------------------------------------------------------
    //
    // These tests guarantee that EVERY TxType variant is exercised on
    // each of the three pathological constant patterns even if the
    // proptest strategy distribution skews. They're the front-line
    // defence against any future "let me just use _mm256_packs_epi32
    // here for speed" regression in either idct16_avx2 or iadst16_avx2.

    fn pathological_inputs() -> [[i32; 256]; 3] {
        let mut alt = [0i32; 256];
        for (i, slot) in alt.iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        [[i16::MAX as i32; 256], [i16::MIN as i32; 256], alt]
    }

    fn run_pathological(tx_type: TxType) {
        require_avx2!();
        let dsts: [[u8; 256]; 3] = [[0u8; 256], [128u8; 256], [255u8; 256]];
        for input in pathological_inputs() {
            for dst_init in dsts {
                assert_match(&input, &dst_init, tx_type);
            }
        }
    }

    #[test]
    fn iht16x16_full1d_pathological_directed_dctdct() {
        run_pathological(TxType::DctDct);
    }

    #[test]
    fn iht16x16_full1d_pathological_directed_adstdct() {
        run_pathological(TxType::AdstDct);
    }

    #[test]
    fn iht16x16_full1d_pathological_directed_dctadst() {
        run_pathological(TxType::DctAdst);
    }

    #[test]
    fn iht16x16_full1d_pathological_directed_adstadst() {
        run_pathological(TxType::AdstAdst);
    }

    // -----------------------------------------------------------------
    // Wave G G2 standalone 1D sanity check: vector iadst16 vs scalar.
    // -----------------------------------------------------------------
    //
    // Proves the new `iadst16_avx2` 1D function matches the scalar
    // `iadst16` reference IN ISOLATION (before it's wrapped into the
    // 2D row/column pipeline). If this test fails, the bug is in
    // `iadst16_avx2` itself rather than the transpose / dispatch
    // plumbing; the 2D-only tests above don't distinguish those two
    // failure modes.
    //
    // We exercise 8 independent 16-vector inputs in parallel (one per
    // YMM lane) so the vector and scalar paths see identical data.

    /// Run the vector `iadst16_avx2` on 8 packed 16-vectors and compare
    /// each lane to the scalar `iadst16`. Returns on the first
    /// divergence with a diagnostic message.
    #[test]
    fn iadst16_avx2_standalone() {
        require_avx2!();
        // SAFETY: gated on AVX2 detection.
        unsafe { iadst16_avx2_standalone_inner() }
    }

    // G4: loads from `dst` use n/a (test helper — operates on stack scratch only)
    #[target_feature(enable = "avx2")]
    unsafe fn iadst16_avx2_standalone_inner() {
        use crate::transforms::inverse::iadst16;
        use core::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};

        // 8 distinct 16-vector inputs (one per YMM lane). Pick a mix
        // of magnitudes and signs that exercises the full butterfly:
        //  - lane 0: linear ramp
        //  - lane 1: alternating ±high-magnitude
        //  - lane 2: DC-only positive
        //  - lane 3: single high-frequency impulse
        //  - lane 4: all i16::MAX
        //  - lane 5: all i16::MIN
        //  - lane 6: alternating i16::MAX / i16::MIN
        //  - lane 7: random-looking arithmetic pattern
        let mut inputs: [[i32; 16]; 8] = [[0i32; 16]; 8];
        for c in 0..16 {
            inputs[0][c] = (c as i32) * 1000 - 7500;
        }
        for c in 0..16 {
            inputs[1][c] = if c & 1 == 0 { 28000 } else { -28000 };
        }
        inputs[2][0] = 1234;
        inputs[3][15] = 12345;
        inputs[4] = [i16::MAX as i32; 16];
        inputs[5] = [i16::MIN as i32; 16];
        for c in 0..16 {
            inputs[6][c] = if c & 1 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        for c in 0..16 {
            inputs[7][c] = (((c as i32) * 1997) % 4001) - 2000;
        }

        // Pack into a 16-position × 8-lane row-major buffer such that
        // `packed[k]` (one ymm) contains `inputs[0..8][k]` (input
        // position k across all 8 lanes/rows).
        let mut packed = [0i32; 16 * 8];
        for k in 0..16 {
            for lane in 0..8 {
                packed[k * 8 + lane] = inputs[lane][k];
            }
        }

        // Load into 16 ymm.
        let mut v: [super::__m256i; 16] = [super::_mm256_set1_epi32(0); 16];
        for (k, slot) in v.iter_mut().enumerate() {
            let p = packed.as_ptr().add(k * 8) as *const super::__m256i;
            *slot = _mm256_loadu_si256(p);
        }

        // Vector iadst16.
        let out_vec = super::iadst16_avx2(v);

        // Unpack back to per-lane row-major.
        let mut unpacked = [0i32; 16 * 8];
        for (k, reg) in out_vec.iter().enumerate() {
            let p = unpacked.as_mut_ptr().add(k * 8) as *mut super::__m256i;
            _mm256_storeu_si256(p, *reg);
        }

        // Scalar iadst16 per lane and compare.
        for lane in 0..8 {
            let mut scalar_out = [0i32; 16];
            iadst16(&inputs[lane], &mut scalar_out);
            for k in 0..16 {
                let v = unpacked[k * 8 + lane];
                assert_eq!(
                    scalar_out[k], v,
                    "iadst16_avx2 lane {lane} position {k} mismatch: \
                     scalar={} vector={}",
                    scalar_out[k], v
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

        // Tautology guard.
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
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch mismatch tx_type={tx_type:?}"
            );
        }
    }
}
