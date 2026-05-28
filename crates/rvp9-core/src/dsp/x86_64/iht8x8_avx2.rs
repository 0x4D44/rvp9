//! 8×8 hybrid inverse-transform + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor** (Wave F F7): both the row pass and the
//! column pass are now fully vectorised in AVX2 with 32-bit lanes for
//! ALL four [`TxType`] variants. The previous Wave E G4-β pilot kept
//! the scalar [`crate::transforms::inverse::idct8`] /
//! [`crate::transforms::inverse::iadst8`] calls and AVX2-accelerated
//! only the final `(out + 16) >> 5` + add + clip + pack stage. This
//! file now mirrors the `idct8x8_avx2.rs` full-1D pattern, with an
//! additional `iadst8_avx2` hand-derived from the scalar `iadst8`
//! butterfly.
//!
//! Scalar reference: [`crate::transforms::inverse::iht8x8_64_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Every scalar operation in [`crate::transforms::inverse::idct8`] and
//! [`crate::transforms::inverse::iadst8`] is mirrored 1:1 in vector
//! form across 8 lanes. The critical correctness rule is the same as
//! the dedicated [`crate::dsp::x86_64::idct8x8_avx2`] kernel:
//!
//! * `wraplow(x) = x as i16 as i32` is MODULAR i16 truncation, **not**
//!   saturation. The vector equivalent is `srai_epi32(slli_epi32(x,
//!   16), 16)` — sign-extend the low 16 bits. We MUST NOT use
//!   `_mm256_packs_epi32` (which saturates) for intermediate wraplows;
//!   libvpx's SSE2 kernel takes that shortcut and diverges from the C
//!   reference on out-of-range intermediates. Our proptest fixtures
//!   include `pathological_strategy` (`[i16::MAX; 64]`, `[i16::MIN;
//!   64]`, alternating-sign) for ALL four tx_types to lock that in.
//! * `dct_const_round_shift(x) = (x + 8192) >> 14` —
//!   [`_mm256_add_epi32`] + [`_mm256_srai_epi32`] by 14.
//! * Multiplies use [`_mm256_mullo_epi32`]; with i16-range operands
//!   and `< 2^15` constants, every product fits in i32, and 4-product
//!   sums in iadst8 stage 1 stay within `±2^31` with margin (worst
//!   case ≈ 1.5e9 vs i32::MAX ≈ 2.15e9).
//!
//! ## TxType dispatch
//!
//! Per the [`crate::transforms::inverse`] `iht_8_dispatch` table —
//! libvpx struct order is `{cols, rows}`:
//!
//! | `tx_type`   | row pass      | col pass      |
//! |-------------|---------------|---------------|
//! | `DctDct`    | `idct8_avx2`  | `idct8_avx2`  |
//! | `AdstDct`   | `idct8_avx2`  | `iadst8_avx2` |
//! | `DctAdst`   | `iadst8_avx2` | `idct8_avx2`  |
//! | `AdstAdst`  | `iadst8_avx2` | `iadst8_avx2` |
//!
//! `DctDct` is **not** delegated to the dedicated idct8x8 kernel — we
//! self-implement so the iht slot remains independently testable and
//! dispatch round-trip checks have a clean `assert_ne!` against the
//! scalar `iht8x8_64_add` reference.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dest` are unaligned via `_mm_loadl_epi64`; the input
//! coefficient buffer is read via `_mm256_loadu_si256` per row. The
//! grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storel_epi64, _mm_unpacklo_epi8, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_slli_epi32,
    _mm256_srai_epi32, _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_2_64, COSPI_4_64, COSPI_6_64, COSPI_8_64, COSPI_10_64, COSPI_12_64, COSPI_14_64,
    COSPI_16_64, COSPI_18_64, COSPI_20_64, COSPI_22_64, COSPI_24_64, COSPI_26_64, COSPI_28_64,
    COSPI_30_64,
};
use crate::transforms::types::TxType;

// ---------------------------------------------------------------------
// Tiny intrinsic helpers. Duplicated from `idct8x8_avx2.rs` — keeping
// them local makes the kernel self-contained and lets future tuning
// (e.g., FMA, AVX-512) diverge per kernel.
// ---------------------------------------------------------------------

/// `dct_const_round_shift(x) = (x + 8192) >> 14`, lane-parallel.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn drs(x: __m256i) -> __m256i {
    let round = _mm256_set1_epi32(1 << 13);
    _mm256_srai_epi32(_mm256_add_epi32(x, round), 14)
}

/// `wraplow(x) = x as i16 as i32`, lane-parallel. Sign-extends the low
/// 16 bits of each lane back to 32 bits — MODULAR i16 truncation,
/// NOT saturation.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. See `idct8x8_avx2.rs` for the operand-range argument that
/// keeps the i32 product exact.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m256i, c: i32) -> __m256i {
    _mm256_mullo_epi32(x, _mm256_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 8×8 i32 transpose using a 3-stage unpack pyramid. Duplicate of the
// helper in `idct8x8_avx2.rs`; see that file for the full lane-mixing
// commentary.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
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
// Vector 1D idct8 — identical to the implementation in
// `idct8x8_avx2.rs`. See that file for stage-by-stage commentary.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct8_avx2(v: [__m256i; 8]) -> [__m256i; 8] {
    let [v0, v1, v2, v3, v4, v5, v6, v7] = v;

    // stage 1
    let s1_0 = v0;
    let s1_2 = v4;
    let s1_1 = v2;
    let s1_3 = v6;
    let t1 = _mm256_sub_epi32(mul_const(v1, COSPI_28_64), mul_const(v7, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(v1, COSPI_4_64), mul_const(v7, COSPI_28_64));
    let s1_4 = wrap(drs(t1));
    let s1_7 = wrap(drs(t2));
    let t1 = _mm256_sub_epi32(mul_const(v5, COSPI_12_64), mul_const(v3, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(v5, COSPI_20_64), mul_const(v3, COSPI_12_64));
    let s1_5 = wrap(drs(t1));
    let s1_6 = wrap(drs(t2));

    // stage 2
    let t1 = mul_const(_mm256_add_epi32(s1_0, s1_2), COSPI_16_64);
    let t2 = mul_const(_mm256_sub_epi32(s1_0, s1_2), COSPI_16_64);
    let s2_0 = wrap(drs(t1));
    let s2_1 = wrap(drs(t2));
    let t1 = _mm256_sub_epi32(mul_const(s1_1, COSPI_24_64), mul_const(s1_3, COSPI_8_64));
    let t2 = _mm256_add_epi32(mul_const(s1_1, COSPI_8_64), mul_const(s1_3, COSPI_24_64));
    let s2_2 = wrap(drs(t1));
    let s2_3 = wrap(drs(t2));
    let s2_4 = wrap(_mm256_add_epi32(s1_4, s1_5));
    let s2_5 = wrap(_mm256_sub_epi32(s1_4, s1_5));
    let s2_6 = wrap(_mm256_sub_epi32(s1_7, s1_6));
    let s2_7 = wrap(_mm256_add_epi32(s1_6, s1_7));

    // stage 3
    let s3_0 = wrap(_mm256_add_epi32(s2_0, s2_3));
    let s3_1 = wrap(_mm256_add_epi32(s2_1, s2_2));
    let s3_2 = wrap(_mm256_sub_epi32(s2_1, s2_2));
    let s3_3 = wrap(_mm256_sub_epi32(s2_0, s2_3));
    let s3_4 = s2_4;
    let t1 = mul_const(_mm256_sub_epi32(s2_6, s2_5), COSPI_16_64);
    let t2 = mul_const(_mm256_add_epi32(s2_5, s2_6), COSPI_16_64);
    let s3_5 = wrap(drs(t1));
    let s3_6 = wrap(drs(t2));
    let s3_7 = s2_7;

    // stage 4 (final output)
    let o0 = wrap(_mm256_add_epi32(s3_0, s3_7));
    let o1 = wrap(_mm256_add_epi32(s3_1, s3_6));
    let o2 = wrap(_mm256_add_epi32(s3_2, s3_5));
    let o3 = wrap(_mm256_add_epi32(s3_3, s3_4));
    let o4 = wrap(_mm256_sub_epi32(s3_3, s3_4));
    let o5 = wrap(_mm256_sub_epi32(s3_2, s3_5));
    let o6 = wrap(_mm256_sub_epi32(s3_1, s3_6));
    let o7 = wrap(_mm256_sub_epi32(s3_0, s3_7));

    [o0, o1, o2, o3, o4, o5, o6, o7]
}

// ---------------------------------------------------------------------
// Vector 1D iadst8 — hand-derived from the scalar reference at
// `crate::transforms::inverse::iadst8`. Each `__m256i` lane holds one
// row's worth of input; the butterfly runs 8 rows in parallel.
//
// Operand-range safety: vector inputs are either raw spec-bounded
// coefficients (i16) or `wraplow` outputs (also i16-bounded). Cospi
// constants are < 2^15. Each `cospi*x` product therefore fits in i32
// with peak magnitude ≈ 5.34e8. A 4-product sum (e.g. stage 1
// `s0 + s4`) has worst-case ≈ 2.14e9, fitting in `i32` with margin
// against `i32::MAX = 2.147e9`.
//
// Stage layout matches the scalar reference 1:1; variable names use
// `sN_K` for "stage N, slot K" to ease cross-reference with
// `inverse.rs:1057`.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn iadst8_avx2(v: [__m256i; 8]) -> [__m256i; 8] {
    // libvpx remap: x0..x7 = in[7,0,5,2,3,4,1,6].
    let x0 = v[7];
    let x1 = v[0];
    let x2 = v[5];
    let x3 = v[2];
    let x4 = v[3];
    let x5 = v[4];
    let x6 = v[1];
    let x7 = v[6];

    // ---- Stage 1 ----
    // s0 = c2*x0 + c30*x1   s1 = c30*x0 - c2*x1
    // s2 = c10*x2 + c22*x3  s3 = c22*x2 - c10*x3
    // s4 = c18*x4 + c14*x5  s5 = c14*x4 - c18*x5
    // s6 = c26*x6 + c6*x7   s7 = c6*x6  - c26*x7
    let s0 = _mm256_add_epi32(mul_const(x0, COSPI_2_64), mul_const(x1, COSPI_30_64));
    let s1 = _mm256_sub_epi32(mul_const(x0, COSPI_30_64), mul_const(x1, COSPI_2_64));
    let s2 = _mm256_add_epi32(mul_const(x2, COSPI_10_64), mul_const(x3, COSPI_22_64));
    let s3 = _mm256_sub_epi32(mul_const(x2, COSPI_22_64), mul_const(x3, COSPI_10_64));
    let s4 = _mm256_add_epi32(mul_const(x4, COSPI_18_64), mul_const(x5, COSPI_14_64));
    let s5 = _mm256_sub_epi32(mul_const(x4, COSPI_14_64), mul_const(x5, COSPI_18_64));
    let s6 = _mm256_add_epi32(mul_const(x6, COSPI_26_64), mul_const(x7, COSPI_6_64));
    let s7 = _mm256_sub_epi32(mul_const(x6, COSPI_6_64), mul_const(x7, COSPI_26_64));

    // x{0..7} = wraplow(drs(s{0..3} ± s{4..7}))
    let x0 = wrap(drs(_mm256_add_epi32(s0, s4)));
    let x1 = wrap(drs(_mm256_add_epi32(s1, s5)));
    let x2 = wrap(drs(_mm256_add_epi32(s2, s6)));
    let x3 = wrap(drs(_mm256_add_epi32(s3, s7)));
    let x4 = wrap(drs(_mm256_sub_epi32(s0, s4)));
    let x5 = wrap(drs(_mm256_sub_epi32(s1, s5)));
    let x6 = wrap(drs(_mm256_sub_epi32(s2, s6)));
    let x7 = wrap(drs(_mm256_sub_epi32(s3, s7)));

    // ---- Stage 2 ----
    // s0..s3 pass through; s4..s7 mix x4..x7 with c8/c24.
    // s4 =  c8*x4 + c24*x5    s5 = c24*x4 - c8*x5
    // s6 = -c24*x6 + c8*x7    s7 =  c8*x6 + c24*x7
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = _mm256_add_epi32(mul_const(x4, COSPI_8_64), mul_const(x5, COSPI_24_64));
    let s5 = _mm256_sub_epi32(mul_const(x4, COSPI_24_64), mul_const(x5, COSPI_8_64));
    // s6 = -c24*x6 + c8*x7  →  c8*x7 - c24*x6
    let s6 = _mm256_sub_epi32(mul_const(x7, COSPI_8_64), mul_const(x6, COSPI_24_64));
    let s7 = _mm256_add_epi32(mul_const(x6, COSPI_8_64), mul_const(x7, COSPI_24_64));

    // x0..x3 = wraplow(s0..s3 ± s2..s3) — no drs because s0..s3 already
    // i16-range from the wraplow at end of stage 1.
    let x0 = wrap(_mm256_add_epi32(s0, s2));
    let x1 = wrap(_mm256_add_epi32(s1, s3));
    let x2 = wrap(_mm256_sub_epi32(s0, s2));
    let x3 = wrap(_mm256_sub_epi32(s1, s3));
    let x4 = wrap(drs(_mm256_add_epi32(s4, s6)));
    let x5 = wrap(drs(_mm256_add_epi32(s5, s7)));
    let x6 = wrap(drs(_mm256_sub_epi32(s4, s6)));
    let x7 = wrap(drs(_mm256_sub_epi32(s5, s7)));

    // ---- Stage 3 ----
    // s2 = c16*(x2 + x3)   s3 = c16*(x2 - x3)
    // s6 = c16*(x6 + x7)   s7 = c16*(x6 - x7)
    let s2 = mul_const(_mm256_add_epi32(x2, x3), COSPI_16_64);
    let s3 = mul_const(_mm256_sub_epi32(x2, x3), COSPI_16_64);
    let s6 = mul_const(_mm256_add_epi32(x6, x7), COSPI_16_64);
    let s7 = mul_const(_mm256_sub_epi32(x6, x7), COSPI_16_64);

    let x2 = wrap(drs(s2));
    let x3 = wrap(drs(s3));
    let x6 = wrap(drs(s6));
    let x7 = wrap(drs(s7));

    // ---- Output permutation + sign flips ----
    // out = [ x0, -x4,  x6, -x2,  x3, -x7,  x5, -x1 ]
    // Each `wraplow` applied; for the negations we compute `0 - xi` and
    // then `wrap`. Note: `wrap(x0)` is a no-op because stage-3 outputs
    // were already wrapped, but the scalar reference wraps again so we
    // preserve that for bit-exact parity.
    let zero = _mm256_setzero_si256();
    let o0 = wrap(x0);
    let o1 = wrap(_mm256_sub_epi32(zero, x4));
    let o2 = wrap(x6);
    let o3 = wrap(_mm256_sub_epi32(zero, x2));
    let o4 = wrap(x3);
    let o5 = wrap(_mm256_sub_epi32(zero, x7));
    let o6 = wrap(x5);
    let o7 = wrap(_mm256_sub_epi32(zero, x1));

    [o0, o1, o2, o3, o4, o5, o6, o7]
}

/// Run the row + column 1-D passes for `tx_type` and return 8 ymm
/// where each register holds one output row (8 columns in lanes).
///
/// The dispatch is a compile-time-friendly `match`: each arm directly
/// instantiates `row` and `col` so the inliner can specialise without
/// going through function pointers.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn run_2d(input: [__m256i; 8], tx_type: TxType) -> [__m256i; 8] {
    // Stage A: transpose so each ymm[k] holds input position k across
    // 8 rows. Allows the row-pass butterfly to act 8-way without any
    // in-lane shuffles.
    let cols = transpose8x8_i32(
        input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
    );

    // Stage B: row pass.
    // Stage C: transpose back to per-row layout.
    // Stage D: column pass.
    //
    // Per the iht_8_dispatch table, libvpx struct order is `{cols,
    // rows}` — i.e., the first field is the COLUMN xform. So in the
    // (row_xform, col_xform) pair (matching the existing helper),
    // `AdstDct` = row=idct8, col=iadst8.
    let row_out = match tx_type {
        TxType::DctDct | TxType::AdstDct => idct8_avx2(cols),
        TxType::DctAdst | TxType::AdstAdst => iadst8_avx2(cols),
    };

    let rows = transpose8x8_i32(
        row_out[0], row_out[1], row_out[2], row_out[3], row_out[4], row_out[5], row_out[6],
        row_out[7],
    );

    match tx_type {
        TxType::DctDct | TxType::DctAdst => idct8_avx2(rows),
        TxType::AdstDct | TxType::AdstAdst => iadst8_avx2(rows),
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
/// Full-1D AVX2 (G4-γ): both row and column passes use the vector
/// kernels [`idct8_avx2`] / [`iadst8_avx2`] picked per `tx_type`.
/// Transpose between passes uses a standard unpack pyramid
/// ([`transpose8x8_i32`]). The output add/clamp/pack finisher matches
/// the previous Wave E G4-β pilot.
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

    // ----- Load: 8 rows of 8 i32 each into ymm[0..7]. -----
    let in_ptr = input.as_ptr() as *const __m256i;
    let r0 = _mm256_loadu_si256(in_ptr.add(0));
    let r1 = _mm256_loadu_si256(in_ptr.add(1));
    let r2 = _mm256_loadu_si256(in_ptr.add(2));
    let r3 = _mm256_loadu_si256(in_ptr.add(3));
    let r4 = _mm256_loadu_si256(in_ptr.add(4));
    let r5 = _mm256_loadu_si256(in_ptr.add(5));
    let r6 = _mm256_loadu_si256(in_ptr.add(6));
    let r7 = _mm256_loadu_si256(in_ptr.add(7));

    let final_rows = run_2d([r0, r1, r2, r3, r4, r5, r6, r7], tx_type);

    // ----- Finisher: round +16, srai 5, pack i32→i16, add to dst,
    // saturate-pack i16→u8, store. Same shape as `idct8x8_avx2.rs`. ---
    let round = _mm256_set1_epi32(16);
    let zero = _mm_setzero_si128();
    for (r, row) in final_rows.iter().enumerate() {
        let out_i32 = _mm256_srai_epi32(_mm256_add_epi32(*row, round), 5);
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        let out_i16_lo = _mm256_castsi256_si128(packed_natural);

        // G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
        let dst_row_ptr = dest.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadl_epi64(dst_row_ptr);
        let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);
        let sum_i16 = _mm_add_epi16(out_i16_lo, dst_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
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

// Miri cannot interpret AVX2 intrinsics — see the parallel comment in
// `idct8x8_avx2.rs`.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `iht8x8_add` (G4-γ
    //! full-1D).
    //!
    //! Coverage spans all 4 tx_types per the Wave F F7 acceptance
    //! criteria:
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
    //! - **`iht8x8_scalar_eq_avx2_proptest`** — stratified proptest with
    //!   five strategies (uniform, sparse, EOB-bucket, high-magnitude,
    //!   pathological) × four tx_types. The `pathological_strategy`
    //!   includes `[i16::MAX; 64]`, `[i16::MIN; 64]`, and an
    //!   alternating-sign pattern — these are the inputs that prove our
    //!   modular-wraplow choice (vs libvpx's SSE2 saturating shortcut).
    //! - **`iht8x8_full1d_pathological_directed_*`** — one directed
    //!   regression per tx_type covering the same pathological patterns
    //!   outside proptest, guaranteeing they always run (proptest could
    //!   in principle reject them through other strategies).
    //! - **`iht8x8_full1d_row_dominant_directed`** /
    //!   **`iht8x8_full1d_column_dominant_directed`** — split-pass
    //!   stress for the new full-1D transpose-row-transpose-column
    //!   pipeline.
    //! - **`iht8x8_dispatch_round_trip`** — forced-AVX2 vs forced-scalar
    //!   dispatch produce identical output, with tautology guard
    //!   (forced-AVX2 slot fn pointer must NOT be the scalar fn).
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

    /// Pathological constant-pattern inputs — the critical fixture for
    /// the modular-wraplow vs saturating-pack rule. If the kernel ever
    /// regresses to `_mm256_packs_epi32` for intermediates, these
    /// patterns produce visible mismatches.
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
    // Directed fixtures (each spans all 4 tx_types unless tx-specific).
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

    // -----------------------------------------------------------------
    // Pathological directed fixtures — one per tx_type.
    // -----------------------------------------------------------------
    //
    // These tests guarantee that every TxType variant is exercised on
    // each of the three pathological constant patterns even if the
    // proptest strategy distribution skews. They're the front-line
    // defence against any future "let me just use _mm256_packs_epi32
    // here for speed" regression.

    fn pathological_inputs() -> [[i32; 64]; 3] {
        let mut alt = [0i32; 64];
        for (i, slot) in alt.iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        [[i16::MAX as i32; 64], [i16::MIN as i32; 64], alt]
    }

    fn run_pathological(tx_type: TxType) {
        require_avx2!();
        let dsts: [[u8; 64]; 3] = [[0u8; 64], [128u8; 64], [255u8; 64]];
        for input in pathological_inputs() {
            for dst_init in dsts {
                assert_match(&input, &dst_init, tx_type);
            }
        }
    }

    #[test]
    fn iht8x8_full1d_pathological_directed_dctdct() {
        run_pathological(TxType::DctDct);
    }

    #[test]
    fn iht8x8_full1d_pathological_directed_adstdct() {
        run_pathological(TxType::AdstDct);
    }

    #[test]
    fn iht8x8_full1d_pathological_directed_dctadst() {
        run_pathological(TxType::DctAdst);
    }

    #[test]
    fn iht8x8_full1d_pathological_directed_adstadst() {
        run_pathological(TxType::AdstAdst);
    }

    // -----------------------------------------------------------------
    // Split-pass stress (mirrors the idct8x8_avx2.rs row/column tests).
    // -----------------------------------------------------------------

    /// G4-γ directed regression: row-dominant stress. A bug in the
    /// transpose, in the row-pass butterfly, or in how we load /
    /// unload rows would corrupt the output even on "easy" inputs.
    #[test]
    fn iht8x8_full1d_row_dominant_directed() {
        require_avx2!();
        let mut input = [0i32; 64];
        for (c, slot) in input.iter_mut().take(8).enumerate() {
            *slot = (c as i32) * 1500 - 6000;
        }
        for (c, slot) in input[4 * 8..4 * 8 + 8].iter_mut().enumerate() {
            *slot = if c & 1 == 0 { 28000 } else { -28000 };
        }
        input[7 * 8 + 3] = 12345;

        for &dst_fill in &[0u8, 64, 128, 255] {
            let dst_init = [dst_fill; 64];
            for &tx_type in &ALL_TX_TYPES {
                assert_match(&input, &dst_init, tx_type);
            }
        }
    }

    /// G4-γ directed regression: column-dominant stress, mirror image
    /// of the row-dominant test.
    #[test]
    fn iht8x8_full1d_column_dominant_directed() {
        require_avx2!();
        let mut input = [0i32; 64];
        for r in 0..8 {
            input[r * 8] = (r as i32) * 1500 - 6000;
        }
        for r in 0..8 {
            input[r * 8 + 4] = if r & 1 == 0 { 28000 } else { -28000 };
        }
        input[5 * 8 + 2] = 12345;

        for &dst_fill in &[0u8, 64, 128, 255] {
            let dst_init = [dst_fill; 64];
            for &tx_type in &ALL_TX_TYPES {
                assert_match(&input, &dst_init, tx_type);
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
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch mismatch tx_type={tx_type:?}"
            );
        }
    }
}
