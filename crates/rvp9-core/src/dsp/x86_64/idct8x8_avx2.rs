//! 8×8 inverse-DCT + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor**: both the row pass and the column pass are
//! now fully vectorised in AVX2 with 32-bit lanes. The previous
//! "finisher-only" pilot kept the scalar [`crate::transforms::inverse::idct8`]
//! call site and only AVX2-accelerated the final round/clamp/pack stage.
//! That pattern is preserved for the smaller / hybrid kernels (idct4x4,
//! iht4x4, iht8x8, idct16x16, idct32x32) — see Wave F follow-up. This
//! file is the first kernel in the codebase to vectorise the 1-D
//! butterflies themselves.
//!
//! Convention compatibility:
//!
//! 1. The body uses `#[target_feature(enable = "avx2")]` on an
//!    `unsafe fn`. (`target_feature_1.1` is unstable on stable Rust
//!    today, so the function MUST be `unsafe`; see HLD V4 §G4
//!    RS3-M1.)
//! 2. A safe wrapper [`idct8x8_add_avx2`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant that makes the call sound — see
//!    [`crate::dsp::dispatch`] for the invariant.
//! 3. The dispatch slot
//!    [`crate::dsp::SimdDispatch::idct8x8_add`] points at the SAFE
//!    wrapper. Call sites stay safe Rust.
//!
//! ## Bit-exactness vs scalar reference
//!
//! VP9 conformance requires bit-exact match to libvpx's C reference.
//! The C reference [`crate::transforms::inverse::idct8x8_64_add`]
//! does the row + column 1-D idct8 in i32 with explicit `wraplow`
//! (`int16_t`-truncation) at every intermediate store, then a final
//! `(out + 16) >> 5` ROUND_POWER_OF_TWO with `clip_pixel_add`.
//!
//! ### How the vector form preserves bit-exactness
//!
//! Every scalar operation in [`crate::transforms::inverse::idct8`] is
//! mirrored 1:1 in vector form across 8 lanes (one lane per row of the
//! 8×8 block):
//!
//! * Multiplies use [`_mm256_mullo_epi32`]. Because the dequantised
//!   coefficients are spec-bounded to i16 range and the cospi
//!   constants are all `< 2^15`, every single product fits in `i32`.
//!   `_mm256_mullo_epi32` returns the low 32 bits of the product, which
//!   is exact for in-range operands. Subsequent add/sub of two such
//!   products fits comfortably in i32 (max abs ≈ 2 × 16k × 32k ≈ 1e9,
//!   < 2^31 = 2.15e9).
//! * `dct_const_round_shift` is `(x + 8192) >> 14` —
//!   [`_mm256_add_epi32`] + [`_mm256_srai_epi32`] by 14, lane-parallel.
//! * `wraplow` is `x as i16 as i32` (modular truncation, **not**
//!   saturation). The vector equivalent is `slli_epi32(_, 16)` then
//!   `srai_epi32(_, 16)` — sign-extend the low 16 bits, exactly
//!   reproducing the truncation. This is the critical correctness
//!   point: libvpx's SSE2 kernel uses `_mm_packs_epi32` (saturating)
//!   which **diverges from the C reference on out-of-range
//!   intermediates**. Our proptest fixtures exercise such inputs (the
//!   `pathological_strategy` includes `[i16::MAX; 64]`), so we MUST
//!   match the C truncation semantics, not libvpx's SSE2 saturation.
//!
//! After the column pass, results are in row-major layout (8 ymm,
//! each = one output row, lanes = 8 columns). The existing
//! round + add + clip + pack finisher runs unchanged.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` are unaligned via `_mm_loadl_epi64`; the input
//! coefficient buffer is read via `_mm256_loadu_si256` per row. The
//! grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_setzero_si128,
    _mm_storel_epi64, _mm_unpacklo_epi8, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_srai_epi32,
    _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_4_64, COSPI_8_64, COSPI_12_64, COSPI_16_64, COSPI_20_64, COSPI_24_64, COSPI_28_64,
};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers, each one-line. Kept `#[inline(always)]` so the
// compiler folds them away with no call overhead; this avoids littering
// the butterfly with `_mm256_*` noise.
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
/// 16 bits of each lane back to 32 bits — i.e., MODULAR i16 truncation,
/// not saturation.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m256i) -> __m256i {
    _mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. Implemented with `_mm256_mullo_epi32` against a
/// broadcast vector. Operand-range note: `x` is the result of a
/// previous `wraplow` (hence i16-range) **or** an unwrapped raw input
/// coefficient (also spec-bounded to i16). `c` is `< 2^15`. Product
/// fits in i32 with margin.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m256i, c: i32) -> __m256i {
    _mm256_mullo_epi32(x, _mm256_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 8×8 i32 transpose using a 3-stage unpack pyramid.
//
// Input rows R0..R7 each hold 8 i32 (one input row per ymm).
// Output columns C0..C7 each hold 8 i32 where C[k] = [R0[k], R1[k], ...,
// R7[k]] — i.e., column k of the original 8×8 matrix.
//
// Stage 1: pair adjacent rows via 32-bit unpack
// Stage 2: pair adjacent pair-results via 64-bit unpack
// Stage 3: swap 128-bit halves via permute2x128 to merge rows 0..3 with
//          rows 4..7 for each column
//
// Total: 8 unpacklo_epi32 + 8 (interleaved) + 8 unpacklo_epi64 + 8
// (interleaved) + 8 permute2x128 = 32 ops. Standard AVX2 transpose.
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
    // Stage 1: 32-bit unpack (pairs of rows).
    // unpacklo gives [r_i[0], r_j[0], r_i[1], r_j[1], r_i[4], r_j[4], r_i[5], r_j[5]]
    // unpackhi gives [r_i[2], r_j[2], r_i[3], r_j[3], r_i[6], r_j[6], r_i[7], r_j[7]]
    let a0 = _mm256_unpacklo_epi32(r0, r1);
    let a1 = _mm256_unpackhi_epi32(r0, r1);
    let a2 = _mm256_unpacklo_epi32(r2, r3);
    let a3 = _mm256_unpackhi_epi32(r2, r3);
    let a4 = _mm256_unpacklo_epi32(r4, r5);
    let a5 = _mm256_unpackhi_epi32(r4, r5);
    let a6 = _mm256_unpacklo_epi32(r6, r7);
    let a7 = _mm256_unpackhi_epi32(r6, r7);

    // Stage 2: 64-bit unpack (pairs of pair-results).
    // Each result is [col k for rows X, Y, Z, W (low 128), col k+4 for
    // rows X, Y, Z, W (high 128)] — still split across 128-bit lanes.
    let b0 = _mm256_unpacklo_epi64(a0, a2); // col 0, rows 0,1,2,3 (lo) / col 4 (hi)
    let b1 = _mm256_unpackhi_epi64(a0, a2); // col 1, rows 0..3 (lo) / col 5 (hi)
    let b2 = _mm256_unpacklo_epi64(a1, a3); // col 2 (lo) / col 6 (hi)
    let b3 = _mm256_unpackhi_epi64(a1, a3); // col 3 (lo) / col 7 (hi)
    let b4 = _mm256_unpacklo_epi64(a4, a6); // col 0, rows 4,5,6,7 (lo) / col 4 (hi)
    let b5 = _mm256_unpackhi_epi64(a4, a6); // col 1, rows 4..7 (lo) / col 5 (hi)
    let b6 = _mm256_unpacklo_epi64(a5, a7); // col 2, rows 4..7 (lo) / col 6 (hi)
    let b7 = _mm256_unpackhi_epi64(a5, a7); // col 3, rows 4..7 (lo) / col 7 (hi)

    // Stage 3: permute2x128 to assemble full columns.
    // imm 0x20 selects (src1.lo, src2.lo) → column k, rows 0..7.
    // imm 0x31 selects (src1.hi, src2.hi) → column k+4, rows 0..7.
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
// Vector 1D idct8. Each `__m256i` holds 8 i32 lanes corresponding to 8
// independent rows / columns being transformed in parallel. The
// arithmetic mirrors `idct8` in `crate::transforms::inverse` line-by-line.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct8_avx2(v: [__m256i; 8]) -> [__m256i; 8] {
    let [v0, v1, v2, v3, v4, v5, v6, v7] = v;

    // ----- stage 1 -----
    // step1[0] = input[0], step1[2] = input[4], step1[1] = input[2], step1[3] = input[6]
    let s1_0 = v0;
    let s1_2 = v4;
    let s1_1 = v2;
    let s1_3 = v6;
    // temp1 = v1*C28 - v7*C4 ; temp2 = v1*C4 + v7*C28
    let t1 = _mm256_sub_epi32(mul_const(v1, COSPI_28_64), mul_const(v7, COSPI_4_64));
    let t2 = _mm256_add_epi32(mul_const(v1, COSPI_4_64), mul_const(v7, COSPI_28_64));
    let s1_4 = wrap(drs(t1));
    let s1_7 = wrap(drs(t2));
    let t1 = _mm256_sub_epi32(mul_const(v5, COSPI_12_64), mul_const(v3, COSPI_20_64));
    let t2 = _mm256_add_epi32(mul_const(v5, COSPI_20_64), mul_const(v3, COSPI_12_64));
    let s1_5 = wrap(drs(t1));
    let s1_6 = wrap(drs(t2));

    // ----- stage 2 -----
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
    // step2[6] = wraplow(-step1[6] + step1[7]) = wraplow(step1[7] - step1[6])
    let s2_6 = wrap(_mm256_sub_epi32(s1_7, s1_6));
    let s2_7 = wrap(_mm256_add_epi32(s1_6, s1_7));

    // ----- stage 3 -----
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

    // ----- stage 4 (final output) -----
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

/// AVX2 implementation of the 8×8 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct8x8_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dst` must have at least `7 * stride + 8` bytes of valid storage.
///
/// # Implementation
///
/// Full-1D AVX2: both row and column passes use [`idct8_avx2`] (8-way
/// SIMD). Transpose between passes uses a standard unpack pyramid
/// ([`transpose8x8_i32`]). The output add/clamp/pack finisher matches
/// the original G4.1 pilot.
///
/// ```text
///   for r in 0..8: load row r as ymm[r]                ; vmovdqu ymm
///   ymm[k] = transpose(R0..R7)[k]    ; (no-op for row pass — see note)
///   ymm[k] = idct8_avx2(ymm[0..7])   ; row pass, 8-way
///   ymm[k] = transpose(ymm[0..7])    ; flip rows ↔ columns
///   ymm[k] = idct8_avx2(ymm[0..7])   ; column pass
///   (each ymm now = one output row)
///   for r in 0..8:
///     add round +16, srai 5, pack to i16
///     load 8 dst u8, zero-extend, add, saturate-pack u8
///     store
/// ```
///
/// Note: a "row pass" applied to 8 columns of inputs (each column = one
/// input position across 8 rows) is mathematically identical to
/// applying the row pass to 8 rows in parallel. Concretely: we load
/// rows as ymm[0..7], then **transpose first** so each ymm[k] holds
/// position k across 8 rows. The row pass operates on this layout
/// natively (each butterfly mixes ymm[i] with ymm[j] — different
/// registers, no in-lane shuffles needed). After the row pass, each
/// ymm[k] holds intermediate column k. A second transpose flips into
/// per-row layout for the column pass; the result of the column pass
/// is row-major (each ymm = one output row), ready for the finisher.
///
/// All loads/stores from / to `dst` are unaligned. See HLD V6 §G4
/// M6-6.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct8x8_add_avx2_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 7 + 8, "dst too small for 8x8 add");

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

    // ----- Stage A: transpose so each ymm[k] holds input position k
    // across 8 rows. -----
    let cols = transpose8x8_i32(r0, r1, r2, r3, r4, r5, r6, r7);

    // ----- Stage B: row pass. -----
    // After this, each ymm[k] = intermediate column k (8 rows in lanes).
    let row_out = idct8_avx2(cols);

    // ----- Stage C: transpose again to get per-row layout for the
    // column pass. After this, each ymm[r] = intermediate row r (8
    // columns in lanes). -----
    let rows = transpose8x8_i32(
        row_out[0], row_out[1], row_out[2], row_out[3], row_out[4], row_out[5], row_out[6],
        row_out[7],
    );

    // ----- Stage D: column pass.
    //
    // Reinterpretation: the column pass treats each COLUMN of the
    // intermediate as a separate 1D-idct8 input. Equivalently (since
    // we transposed), we now have rows in lanes and apply idct8
    // across the 8 ymm registers; each ymm holds one position of the
    // *column-pass input*, i.e., the value at fixed input-position k
    // varying over 8 rows. The vector idct8 logic is identical. After
    // this pass each ymm[k] holds one output row. -----
    let final_rows = idct8_avx2(rows);

    // ----- Stage E: round, add round constant +16, shift right by 5,
    // saturating-pack i32 → i16, add to dst, saturating-pack i16 → u8.
    // -----
    let round = _mm256_set1_epi32(16);
    let zero = _mm_setzero_si128();
    for (r, row) in final_rows.iter().enumerate() {
        let out_i32 = _mm256_srai_epi32(_mm256_add_epi32(*row, round), 5);
        // Saturate-pack i32 → i16. `_mm256_packs_epi32(x, x)` interleaves
        // per-128-bit-lane; `_mm256_permute4x64_epi64(_, 0xD8)` reorders
        // the 64-bit qwords so the low 128 bits contain 8 × i16 in
        // natural order.
        let packed_lanes = _mm256_packs_epi32(out_i32, out_i32);
        let packed_natural = _mm256_permute4x64_epi64(packed_lanes, 0xD8);
        let out_i16_lo = _mm256_castsi256_si128(packed_natural);

        // G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
        let dst_row_ptr = dst.as_mut_ptr().add(r * stride) as *const __m128i;
        let dst_u8 = _mm_loadl_epi64(dst_row_ptr);
        let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);
        let sum_i16 = _mm_add_epi16(out_i16_lo, dst_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
        let dst_row_ptr_mut = dst.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storel_epi64(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the AVX2 idct8x8 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct8x8_64_add`], so the
/// `SimdDispatch::idct8x8_add` slot can hold either.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly to do scalar-vs-AVX2
/// differential testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct8x8_add_avx2(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime (a CPU doesn't lose features mid-run).
    // The unit-test path in `idct8x8_avx2_tests` gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check, so calls from
    // tests are equally sound.
    unsafe { idct8x8_add_avx2_inner(input, dst, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct8x8_add`.
    //!
    //! Coverage strategy (per HLD V5 §G4 TQ4-M3 corrected EOB seam
    //! set for idct8x8 = {1, 2, 12, 13, 64}):
    //!
    //! - **`idct8x8_scalar_eq_avx2_proptest`** — proptest with five
    //!   stratified strategies (`prop_oneof!`):
    //!   1. `uniform_strategy` — random over the i32 coefficient
    //!      range (clamped to `[-32768, 32767]` per VP9 spec — the
    //!      dequantised coefficient bound from inv_txfm.c).
    //!   2. `sparse_strategy` — 1..=4 non-zero coefficients at random
    //!      positions.
    //!   3. `eob_bucket_strategy` — K ∈ {1, 2, 12, 13, 64} non-zero
    //!      coefficients to exercise dispatcher seams.
    //!   4. `high_magnitude_strategy` — coefficients near ±i16::MAX
    //!      (saturation paths).
    //!   5. `pathological_strategy` — all-equal / all-min / all-max /
    //!      alternating sign.
    //!
    //! - **`idct8x8_all_zero_input`** — DC-only fast path with
    //!   `input[0] = 0`.
    //!
    //! - **`idct8x8_single_coefficient_impulses`** — sweep a single
    //!   non-zero coefficient (`value = 100`) across every position
    //!   0..64 and confirm scalar / AVX2 agreement at each.
    //!
    //! - **`idct8x8_misaligned_dst_pointers`** — `dst` *backing
    //!   pointer* offset at byte offsets {0, 1, 3, 7, 15}. The kernel
    //!   uses only 8-byte loads (`_mm_loadl_epi64`), so this is a
    //!   per-offset arithmetic regression guard, NOT an alignment-fault
    //!   test (alignment exercise is deferred to G4.2 when 16/32-byte
    //!   loads land). Pointer arithmetic uses a `Box<[u8; 80]>`
    //!   over-allocation; sub-slicing then guarantees the requested
    //!   offset on the slice's `as_ptr`.
    //!
    //! - **`idct8x8_full1d_row_dominant_directed`** — G4-γ directed
    //!   regression: single-row impulse where the entire signal lives
    //!   in row 0 (no contribution from rows 1..7). Exercises the
    //!   "row pass produces zero for empty rows / non-zero for the
    //!   loaded row" property, which the integration tests may dilute.
    //!
    //! Every test is gated on `is_x86_feature_detected!("avx2")` and
    //! returns early on non-AVX2 CPUs (CI runner check).

    use super::idct8x8_add_avx2;
    use crate::transforms::inverse::idct8x8_64_add;
    use proptest::prelude::*;

    /// Skip-marker macro: bail out of a test on a non-AVX2 host.
    /// On the project's tier-1 CI/dev hosts AVX2 is always present;
    /// the early return is a belt-and-braces guard for older hardware.
    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Compute the AVX2 output for one set of coefficients + starting
    /// destination state. Returns the final dst bytes.
    ///
    /// Callers are expected to gate on `is_x86_feature_detected!("avx2")`
    /// via the `require_avx2!()` macro at the start of each test; the
    /// internal `assert!` here is defensive — if it ever fires it means
    /// a test was added that forgot the macro gate.
    fn run_avx2(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct8x8_add_avx2(input, &mut dst, 8);
        dst
    }

    /// Same shape via the scalar reference.
    fn run_scalar(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        let mut dst = *dst_init;
        idct8x8_64_add(input, &mut dst, 8);
        dst
    }

    /// Differential check.
    fn assert_match(input: &[i32; 64], dst_init: &[u8; 64]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------

    /// Uniform i32 coefficients clamped to `[-32768, 32767]`. VP9
    /// spec / libvpx bound the post-dequant coefficient range to
    /// i16. Even though our scalar reference takes `i32`, the bit-
    /// exact contract only applies within that range; out-of-range
    /// inputs are undefined behaviour for libvpx too.
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

    /// 1..=4 non-zero coefficients at random positions.
    fn sparse_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec((0usize..64, uniform_coeff()), 1..=4).prop_map(|nz| {
            let mut out = [0i32; 64];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// Exactly K non-zero coefficients in raster positions
    /// `0..K`, with K drawn from the idct8x8 dispatcher-seam set
    /// {1, 2, 12, 13, 64}. K=1 is the DC-only fast-path seam; K=12 /
    /// K=13 straddle the libvpx `idct8x8_12_add` / `_64_add`
    /// dispatcher boundary; K=64 is full density.
    ///
    /// We place the K non-zero coefficients at the lowest K raster
    /// positions rather than random ones because (a) it tests the
    /// dispatcher seam without needing an enormous rejection-sample
    /// loop to draw K unique positions from a 64-element domain
    /// (proptest's `hash_set(0..64, k)` rejects too aggressively when
    /// `k` approaches the domain size — empirically aborts at K=64),
    /// and (b) the random-position case is already covered by
    /// `sparse_strategy` and `uniform_strategy`.
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

    /// Coefficients near ±i16::MAX to stress the saturation path.
    fn high_magnitude_strategy() -> impl Strategy<Value = [i32; 64]> {
        // 30000..=32767 magnitude, random sign per coefficient.
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

    /// Pathological constant-pattern inputs.
    fn pathological_strategy() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            // all-equal at a random magnitude
            (-32768i32..=32767).prop_map(|v| [v; 64]),
            // all-min (i16::MIN)
            Just([i16::MIN as i32; 64]),
            // all-max (i16::MAX)
            Just([i16::MAX as i32; 64]),
            // alternating sign at i16::MAX magnitude
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

    /// Random u8 destination starting state — exercises the
    /// `dst + transform` saturation paths.
    fn dst_strategy() -> impl Strategy<Value = [u8; 64]> {
        prop::collection::vec(any::<u8>(), 64).prop_map(|v| {
            let mut out = [0u8; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    // -----------------------------------------------------------------
    // Proptest cases
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Stratified scalar-vs-AVX2 differential.
        #[test]
        fn idct8x8_scalar_eq_avx2_proptest(
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
    // Directed fixtures
    // -----------------------------------------------------------------

    /// All-zero coefficients: dst unchanged after round-trip (the
    /// transform output is the all-zero matrix, +16 >> 5 = 0).
    #[test]
    fn idct8x8_all_zero_input() {
        require_avx2!();
        let input = [0i32; 64];
        let dst_init = [42u8; 64];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep: for every position 0..64,
    /// set just that coefficient to a magnitude that exercises both
    /// positive and negative saturation, and confirm scalar / AVX2
    /// agree.
    #[test]
    fn idct8x8_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..64 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 64];
                input[pos] = val;
                let dst_init = [128u8; 64];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Dst-pointer-offset correctness regression guard.
    ///
    /// The kernel uses `_mm_loadl_epi64` (8-byte, alignment-free) so
    /// this test doesn't exercise alignment-fault paths — that's
    /// deferred to G4.2 when 16/32-byte loads land. This test catches
    /// per-offset arithmetic bugs (e.g., a stride miscompute that
    /// only fires at certain byte offsets).
    ///
    /// Mechanics: allocate an 80-byte buffer on the heap (via `Box`)
    /// and sub-slice at the requested byte offset, then differentially
    /// compare scalar vs AVX2 outputs at each skew.
    #[test]
    fn idct8x8_misaligned_dst_pointers() {
        require_avx2!();
        // Use a strong synthetic input — eob ≈ 16 with a mix of
        // magnitudes; same input for every skew, only `dst` ptr varies.
        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().take(16).enumerate() {
            *slot = (i as i32) * 137 - 800;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            // Heap allocate so the platform allocator picks the base
            // alignment; we then offset by `skew` bytes.
            let mut buf: Box<[u8; 80]> = Box::new([0u8; 80]);
            // Initialise the working region to a known pattern so any
            // misaligned out-of-range write is caught by the assert.
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(7);
            }
            // Snapshot the full buffer for the scalar reference run.
            let mut scalar_buf = *buf;

            // Run AVX2 on the offset sub-slice.
            idct8x8_add_avx2(&input, &mut buf[skew..skew + 64], 8);
            // Run scalar on the SAME offset of an independent buffer.
            idct8x8_64_add(&input, &mut scalar_buf[skew..skew + 64], 8);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}: \
                 per-offset arithmetic regression (kernel uses \
                 `_mm_loadl_epi64`, so this is not an alignment-fault \
                 test — see G4.2 follow-up for 16/32-byte loads)"
            );
        }
    }

    /// Property: dispatch through SimdDispatch picks AVX2 when
    /// caps.avx2 = true and scalar otherwise, and both end up with
    /// the same final dst.
    #[test]
    fn idct8x8_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        // Forced-AVX2 dispatch (matches host caps for our CI matrix).
        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        // Forced-scalar dispatch.
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let mut input = [0i32; 64];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 19) % 4001 - 2000;
        }
        let dst_init = [97u8; 64];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct8x8_add())(&input, &mut dst_avx2, 8);
        (dispatch_scalar.idct8x8_add())(&input, &mut dst_scalar, 8);
        assert_eq!(dst_avx2, dst_scalar);
    }

    /// **G4-γ directed regression**: row-dominant stress.
    ///
    /// The full-1D refactor splits the work across the row pass and
    /// the column pass with a transpose between. A bug in the
    /// transpose, in the row-pass butterfly, or in how we load /
    /// unload rows would corrupt the output even on "easy" inputs.
    /// This test plants two distinct row signatures (row 0 = ramp,
    /// row 4 = high-magnitude alternating sign, rest zero) and
    /// confirms scalar / AVX2 still agree. The signature is chosen so
    /// the row pass produces non-trivial intermediates that the
    /// column pass then mixes into every output row — making row /
    /// column / transpose bugs visible everywhere.
    #[test]
    fn idct8x8_full1d_row_dominant_directed() {
        require_avx2!();
        let mut input = [0i32; 64];
        // Row 0: ramp.
        for (c, slot) in input.iter_mut().take(8).enumerate() {
            *slot = (c as i32) * 1500 - 6000;
        }
        // Row 4: high-magnitude alternating sign.
        for (c, slot) in input[4 * 8..4 * 8 + 8].iter_mut().enumerate() {
            *slot = if c & 1 == 0 { 28000 } else { -28000 };
        }
        // Row 7: single impulse at column 3 (exercises the column-7
        // input position which the row pass routes through the
        // odd-coefficient half of the butterfly).
        input[7 * 8 + 3] = 12345;

        for &dst_fill in &[0u8, 64, 128, 255] {
            let dst_init = [dst_fill; 64];
            assert_match(&input, &dst_init);
        }
    }

    /// **G4-γ directed regression**: column-dominant stress, mirror
    /// image of the row-dominant test. Loads column 0, column 4, and
    /// a single (5, 2) coefficient; the row pass produces small
    /// intermediates across many rows, then the column pass amplifies
    /// each column independently. Different bug-pressure profile from
    /// the row-dominant directed test.
    #[test]
    fn idct8x8_full1d_column_dominant_directed() {
        require_avx2!();
        let mut input = [0i32; 64];
        // Column 0 of input: ramp.
        for r in 0..8 {
            input[r * 8] = (r as i32) * 1500 - 6000;
        }
        // Column 4: high-magnitude alternating sign.
        for r in 0..8 {
            input[r * 8 + 4] = if r & 1 == 0 { 28000 } else { -28000 };
        }
        // Single impulse at (5, 2).
        input[5 * 8 + 2] = 12345;

        for &dst_fill in &[0u8, 64, 128, 255] {
            let dst_init = [dst_fill; 64];
            assert_match(&input, &dst_init);
        }
    }
}
