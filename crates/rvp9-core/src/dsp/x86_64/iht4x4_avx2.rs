//! 4×4 hybrid inverse-transform + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor (Wave F F6)**: both the row pass and the
//! column pass are now fully vectorised in AVX2 with 32-bit lanes for
//! ALL four [`TxType`] variants. The previous Wave E G4-β pilot kept
//! the scalar [`crate::transforms::inverse::idct4`] /
//! [`crate::transforms::inverse::iadst4`] calls and AVX2-accelerated
//! only the final `(out + 8) >> 4` + add-clamp-pack stage. This file
//! now mirrors the [`crate::dsp::x86_64::idct4x4_avx2`] full-1D
//! pattern (4×4 xmm with 4 `i32` lanes per row) plus the
//! [`crate::dsp::x86_64::iht8x8_avx2`] iht dispatch pattern, with an
//! additional `iadst4_avx2` hand-derived from the scalar `iadst4`
//! butterfly.
//!
//! Scalar reference: [`crate::transforms::inverse::iht4x4_16_add`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Every scalar operation in [`crate::transforms::inverse::idct4`] and
//! [`crate::transforms::inverse::iadst4`] is mirrored 1:1 in vector
//! form across 4 lanes (one lane per row of the 4×4 block, computed in
//! parallel). The critical correctness rule:
//!
//! * `wraplow(x) = x as i16 as i32` is MODULAR i16 truncation, **not**
//!   saturation. The vector equivalent is `srai_epi32(slli_epi32(x,
//!   16), 16)` — sign-extend the low 16 bits. We MUST NOT use
//!   `_mm_packs_epi32` (which saturates) for intermediate wraplows;
//!   libvpx's SSE2 kernel takes that shortcut and diverges from the C
//!   reference on out-of-range intermediates. Our proptest fixtures
//!   include `pathological_strategy` (`[i16::MAX; 16]`, `[i16::MIN;
//!   16]`, alternating-sign) for ALL four tx_types to lock that in.
//! * `dct_const_round_shift(x) = (x + 8192) >> 14` —
//!   [`_mm_add_epi32`] + [`_mm_srai_epi32`] by 14.
//! * Multiplies use [`_mm_mullo_epi32`]; with i16-range operands and
//!   sinpi/cospi constants `< 2^15`, every product fits in i32. Note
//!   for `iadst4`: stage-1 three-product sums (e.g. `s0 + s3 + s5`)
//!   peak at ~1.0e9, and `output[3] = drs(s0 + s1 - s3)` peaks at
//!   ~2.43e9 which **does** exceed `i32::MAX = 2.147e9`. The i32 lanes
//!   wrap modulo 2^32; however, because the final `wraplow` extracts
//!   only the low 16 bits AFTER `drs`, and `>> 14` of an i32-wrap
//!   value differs from the i64 result by a multiple of `2^18`, the
//!   low 16 bits are preserved. Net: bit-exact equivalence holds. The
//!   `iht4x4_full1d_pathological_directed_*` fixtures pin this.
//!
//! ## TxType dispatch
//!
//! Per the [`crate::transforms::inverse`] `iht_4_dispatch` table —
//! libvpx struct order is `{cols, rows}`:
//!
//! | `tx_type`   | row pass      | col pass      |
//! |-------------|---------------|---------------|
//! | `DctDct`    | `idct4_avx2`  | `idct4_avx2`  |
//! | `AdstDct`   | `idct4_avx2`  | `iadst4_avx2` |
//! | `DctAdst`   | `iadst4_avx2` | `idct4_avx2`  |
//! | `AdstAdst`  | `iadst4_avx2` | `iadst4_avx2` |
//!
//! `DctDct` is **not** delegated to the dedicated
//! [`crate::dsp::x86_64::idct4x4_avx2`] kernel — we self-implement so
//! the iht slot remains independently testable and dispatch round-trip
//! checks have a clean `assert_ne!` against the scalar
//! `iht4x4_16_add` reference.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `dest` are 4-byte (one `i32` load per row via
//! `_mm_cvtsi32_si128`, sourcing memory via `read_unaligned`); the
//! input coefficient buffer is read via four `_mm_loadu_si128` ops
//! (one per row). All ops on `dest` are unaligned. The grep sentinel
//! below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_cvtsi128_si32, _mm_cvtsi32_si128,
    _mm_loadu_si128, _mm_mullo_epi32, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32,
    _mm_setzero_si128, _mm_slli_epi32, _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi32,
    _mm_unpackhi_epi32, _mm_unpacklo_epi32, _mm_unpacklo_epi8,
};

use crate::transforms::constants::{
    COSPI_8_64, COSPI_16_64, COSPI_24_64, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9,
};
use crate::transforms::types::TxType;

// ---------------------------------------------------------------------
// Tiny intrinsic helpers, each one-line. Kept `#[inline]` so the
// compiler folds them away with no call overhead. Mirror the
// `idct4x4_avx2.rs` helpers (`drs`, `wrap`, `mul_const`) — duplicated
// here so this kernel is self-contained.
// ---------------------------------------------------------------------

/// `dct_const_round_shift(x) = (x + 8192) >> 14`, lane-parallel.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn drs(x: __m128i) -> __m128i {
    let round = _mm_set1_epi32(1 << 13);
    _mm_srai_epi32(_mm_add_epi32(x, round), 14)
}

/// `wraplow(x) = x as i16 as i32`, lane-parallel. Sign-extends the low
/// 16 bits of each lane back to 32 bits — MODULAR i16 truncation,
/// NOT saturation. See module-level docstring for why this matters.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m128i) -> __m128i {
    _mm_srai_epi32(_mm_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. Implemented with `_mm_mullo_epi32` against a broadcast
/// vector. See module docstring for the operand-range argument.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m128i, c: i32) -> __m128i {
    _mm_mullo_epi32(x, _mm_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 4×4 i32 transpose using a 2-stage unpack pyramid. Duplicated from
// `idct4x4_avx2.rs` so this kernel is self-contained; see that file
// for the full lane-mixing commentary.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose4x4_i32(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
    use core::arch::x86_64::{_mm_unpackhi_epi64, _mm_unpacklo_epi64};
    // Stage 1: 32-bit unpack of (r0, r1) and (r2, r3).
    let a0 = _mm_unpacklo_epi32(r0, r1);
    let a1 = _mm_unpackhi_epi32(r0, r1);
    let a2 = _mm_unpacklo_epi32(r2, r3);
    let a3 = _mm_unpackhi_epi32(r2, r3);

    // Stage 2: 64-bit unpack to assemble full columns.
    let c0 = _mm_unpacklo_epi64(a0, a2);
    let c1 = _mm_unpackhi_epi64(a0, a2);
    let c2 = _mm_unpacklo_epi64(a1, a3);
    let c3 = _mm_unpackhi_epi64(a1, a3);

    [c0, c1, c2, c3]
}

// ---------------------------------------------------------------------
// Vector 1D idct4. Identical to the implementation in
// `idct4x4_avx2.rs::idct4_avx2`; see that file for stage-by-stage
// commentary. Mirrors scalar `crate::transforms::inverse::idct4`.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct4_avx2(v: [__m128i; 4]) -> [__m128i; 4] {
    let [v0, v1, v2, v3] = v;

    // ----- stage 1 -----
    let t1 = mul_const(_mm_add_epi32(v0, v2), COSPI_16_64);
    let t2 = mul_const(_mm_sub_epi32(v0, v2), COSPI_16_64);
    let step0 = wrap(drs(t1));
    let step1 = wrap(drs(t2));
    let t1 = _mm_sub_epi32(mul_const(v1, COSPI_24_64), mul_const(v3, COSPI_8_64));
    let t2 = _mm_add_epi32(mul_const(v1, COSPI_8_64), mul_const(v3, COSPI_24_64));
    let step2 = wrap(drs(t1));
    let step3 = wrap(drs(t2));

    // ----- stage 2 (final output) -----
    let o0 = wrap(_mm_add_epi32(step0, step3));
    let o1 = wrap(_mm_add_epi32(step1, step2));
    let o2 = wrap(_mm_sub_epi32(step1, step2));
    let o3 = wrap(_mm_sub_epi32(step0, step3));

    [o0, o1, o2, o3]
}

// ---------------------------------------------------------------------
// Vector 1D iadst4 — hand-derived from the scalar reference at
// `crate::transforms::inverse::iadst4`. Each `__m128i` lane holds one
// row's worth of input; the butterfly runs 4 rows in parallel.
//
// Scalar form (i64 arithmetic):
//   s0_a = SINPI_1_9 * x0    s1_a = SINPI_2_9 * x0
//   s2_a = SINPI_3_9 * x1
//   s3_a = SINPI_4_9 * x2    s4_a = SINPI_1_9 * x2
//   s5_a = SINPI_2_9 * x3    s6_a = SINPI_4_9 * x3
//   s7   = wraplow(x0 - x2 + x3)        // i32
//   s0   = s0_a + s3_a + s5_a            // i64
//   s1   = s1_a - s4_a - s6_a            // i64
//   s2   = SINPI_3_9 * s7                // i64
//   s3   = s2_a                          // i64
//   output[0] = wraplow(drs(s0 + s3))
//   output[1] = wraplow(drs(s1 + s3))
//   output[2] = wraplow(drs(s2))
//   output[3] = wraplow(drs(s0 + s1 - s3))
//
// Operand-range safety: even though `output[3]`'s pre-`drs` value can
// reach ~2.43e9 (overflowing i32::MAX ≈ 2.15e9), the final
// `wraplow(drs(...))` low-16-bit extraction is invariant under i32
// wrap. See module-level docstring for the full algebra.
//
// NOTE: we deliberately do NOT replicate the scalar all-zero
// early-out. With 4-way SIMD the cost of branchless arithmetic on
// all-zero lanes is one cycle, vs the cost of probing 4 lanes for
// zero. The proptest path includes all-zero inputs to verify the
// branchless form still produces zeroed output.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn iadst4_avx2(v: [__m128i; 4]) -> [__m128i; 4] {
    let x0 = v[0];
    let x1 = v[1];
    let x2 = v[2];
    let x3 = v[3];

    // s7 = wraplow(x0 - x2 + x3)
    let s7 = wrap(_mm_add_epi32(_mm_sub_epi32(x0, x2), x3));

    // Individual products (i32 lanes; see operand-range note).
    let s0_a = mul_const(x0, SINPI_1_9);
    let s1_a = mul_const(x0, SINPI_2_9);
    let s2_a = mul_const(x1, SINPI_3_9);
    let s3_a = mul_const(x2, SINPI_4_9);
    let s4_a = mul_const(x2, SINPI_1_9);
    let s5_a = mul_const(x3, SINPI_2_9);
    let s6_a = mul_const(x3, SINPI_4_9);

    // s0 = s0_a + s3_a + s5_a
    // s1 = s1_a - s4_a - s6_a
    // s2 = SINPI_3_9 * s7
    // s3 = s2_a
    let s0 = _mm_add_epi32(_mm_add_epi32(s0_a, s3_a), s5_a);
    let s1 = _mm_sub_epi32(_mm_sub_epi32(s1_a, s4_a), s6_a);
    let s2 = mul_const(s7, SINPI_3_9);
    let s3 = s2_a;

    // output[0] = wrap(drs(s0 + s3))
    // output[1] = wrap(drs(s1 + s3))
    // output[2] = wrap(drs(s2))
    // output[3] = wrap(drs(s0 + s1 - s3))
    let o0 = wrap(drs(_mm_add_epi32(s0, s3)));
    let o1 = wrap(drs(_mm_add_epi32(s1, s3)));
    let o2 = wrap(drs(s2));
    let o3 = wrap(drs(_mm_sub_epi32(_mm_add_epi32(s0, s1), s3)));

    [o0, o1, o2, o3]
}

/// Run the row + column 1-D passes for `tx_type` and return 4 xmm
/// where each register holds one output row (4 columns in lanes).
///
/// The dispatch is a compile-time-friendly `match`: each arm directly
/// instantiates `row` and `col` so the inliner can specialise without
/// going through function pointers.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn run_2d(input: [__m128i; 4], tx_type: TxType) -> [__m128i; 4] {
    // Stage A: transpose so each xmm[k] holds input position k across
    // 4 rows. Allows the row-pass butterfly to act 4-way without any
    // in-lane shuffles.
    let cols = transpose4x4_i32(input[0], input[1], input[2], input[3]);

    // Stage B: row pass.
    //
    // Per the iht_4_dispatch table, libvpx struct order is
    // `{cols, rows}` — i.e., the first field is the COLUMN xform. So in
    // the (row_xform, col_xform) pair (matching the scalar helper),
    // `AdstDct` = row=idct4, col=iadst4.
    let row_out = match tx_type {
        TxType::DctDct | TxType::AdstDct => idct4_avx2(cols),
        TxType::DctAdst | TxType::AdstAdst => iadst4_avx2(cols),
    };

    // Stage C: transpose back to per-row layout for the column pass.
    let rows = transpose4x4_i32(row_out[0], row_out[1], row_out[2], row_out[3]);

    // Stage D: column pass.
    match tx_type {
        TxType::DctDct | TxType::DctAdst => idct4_avx2(rows),
        TxType::AdstDct | TxType::AdstAdst => iadst4_avx2(rows),
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
/// Full-1D AVX2 (G4-γ): both row and column passes use the vector
/// kernels [`idct4_avx2`] / [`iadst4_avx2`] picked per `tx_type`.
/// Transpose between passes uses a standard unpack pyramid
/// ([`transpose4x4_i32`]). The output add/clamp/pack finisher matches
/// the previous Wave E G4-β pilot.
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

    // ----- Load: 4 rows of 4 i32 each into xmm[0..3]. -----
    let in_ptr = input.as_ptr() as *const __m128i;
    let r0 = _mm_loadu_si128(in_ptr.add(0));
    let r1 = _mm_loadu_si128(in_ptr.add(1));
    let r2 = _mm_loadu_si128(in_ptr.add(2));
    let r3 = _mm_loadu_si128(in_ptr.add(3));

    let final_rows = run_2d([r0, r1, r2, r3], tx_type);

    // ----- Finisher: round (+8), shift right by 4, saturating-pack
    // i32 → i16, add to dst, saturating-pack i16 → u8, store. Same
    // shape as `idct4x4_avx2.rs`. -----
    let round = _mm_set1_epi32(8);
    let zero = _mm_setzero_si128();

    let row_pack = |row: __m128i| -> __m128i {
        _mm_srai_epi32(_mm_add_epi32(row, round), 4)
    };

    // Pack rows (0, 1) and (2, 3) into one i16 register each. After
    // saturate-pack, lanes 0..3 = row a, lanes 4..7 = row b.
    let r01_i16 = _mm_packs_epi32(row_pack(final_rows[0]), row_pack(final_rows[1]));
    let r23_i16 = _mm_packs_epi32(row_pack(final_rows[2]), row_pack(final_rows[3]));

    // For each of the 4 rows, load 4 dst bytes, zero-extend to i16,
    // add the transform output, saturate-pack to u8, store.
    for r in 0..4 {
        // Pick the half of the packed-pair that holds this row.
        let row_i16: __m128i = match r {
            0 => r01_i16,
            // Row 1 lives in lanes 4..7 = bytes 8..15. Shift right by
            // 8 bytes to move it into lanes 0..3.
            1 => _mm_srli_si128::<8>(r01_i16),
            2 => r23_i16,
            _ => _mm_srli_si128::<8>(r23_i16),
        };

        // G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
        let dst_row_start = r * stride;
        // SAFETY: within the unsafe fn we read 4 bytes from
        // `dest[dst_row_start..dst_row_start + 4]`. The debug_assert
        // bounds `dest.len() >= 3*stride + 4`.
        let raw: i32 =
            core::ptr::read_unaligned(dest.as_ptr().add(dst_row_start) as *const i32);
        let dst_u8 = _mm_cvtsi32_si128(raw);
        // Zero-extend the 4 u8 lanes to i16.
        let dst_i16 = _mm_unpacklo_epi8(dst_u8, zero);

        // Add transform output and saturate-pack to u8. Only the low 4
        // lanes are meaningful — the high 4 are garbage and aren't
        // stored.
        let sum_i16 = _mm_add_epi16(row_i16, dst_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
        // Store the low 4 u8s back via _mm_cvtsi128_si32.
        let lo_i32 = _mm_cvtsi128_si32(sum_u8);
        core::ptr::write_unaligned(dest.as_mut_ptr().add(dst_row_start) as *mut i32, lo_i32);
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

// Miri cannot interpret AVX2 intrinsics — see the parallel comment in
// `idct4x4_avx2.rs`. Scalar reference paths (the `transforms::inverse`
// family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `iht4x4_add` (G4-γ
    //! full-1D refactor).
    //!
    //! Mirrors the F1 idct4x4 + F7 iht8x8 test suites. Coverage spans
    //! all 4 tx_types per the Wave F F6 acceptance criteria:
    //!
    //! - **`iht4x4_all_zero_input`** — zero coefficients leave dst
    //!   unchanged (all four tx_types). Verifies the branchless
    //!   `iadst4_avx2` (no scalar early-out) still zeroes output.
    //! - **`iht4x4_dc_only`** — single coefficient at position 0 (all
    //!   four tx_types).
    //! - **`iht4x4_single_coefficient_impulses`** — sweep one non-zero
    //!   coefficient across every position 0..16 (per tx_type).
    //! - **`iht4x4_max_input`** — i16::MAX / i16::MIN saturation
    //!   stress.
    //! - **`iht4x4_misaligned_dst_pointers`** — per-skew regression
    //!   guard (kernel uses `_mm_cvtsi32_si128`, unaligned).
    //! - **`iht4x4_scalar_eq_avx2_proptest`** — 256-case proptest with
    //!   five stratified strategies × four tx_types. The
    //!   `pathological_strategy` includes `[i16::MAX; 16]`,
    //!   `[i16::MIN; 16]`, and an alternating-sign pattern — these are
    //!   the inputs that prove our modular-wraplow choice (vs libvpx's
    //!   SSE2 saturating shortcut).
    //! - **`iht4x4_full1d_pathological_directed_{dctdct,adstdct,
    //!   dctadst,adstadst}`** — one directed regression per tx_type
    //!   covering the same pathological patterns outside proptest,
    //!   guaranteeing they always run.
    //! - **`iht4x4_dispatch_round_trip`** — forced-AVX2 vs
    //!   forced-scalar dispatch produce identical output, with
    //!   tautology guard (forced-AVX2 slot fn pointer must NOT be the
    //!   scalar fn).
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

    /// Pathological constant-pattern inputs — the critical fixture for
    /// the modular-wraplow vs saturating-pack rule. If the kernel ever
    /// regresses to `_mm_packs_epi32` for intermediates, these
    /// patterns produce visible mismatches.
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

    // -----------------------------------------------------------------
    // Pathological directed fixtures — one per tx_type.
    // -----------------------------------------------------------------
    //
    // These tests guarantee that every TxType variant is exercised on
    // each of the three pathological constant patterns even if the
    // proptest strategy distribution skews. They're the front-line
    // defence against any future "let me just use _mm_packs_epi32
    // here for speed" regression.

    fn pathological_inputs() -> [[i32; 16]; 3] {
        let mut alt = [0i32; 16];
        for (i, slot) in alt.iter_mut().enumerate() {
            *slot = if i % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            };
        }
        [
            [i16::MAX as i32; 16],
            [i16::MIN as i32; 16],
            alt,
        ]
    }

    fn run_pathological(tx_type: TxType) {
        require_avx2!();
        let dsts: [[u8; 16]; 3] = [[0u8; 16], [128u8; 16], [255u8; 16]];
        for input in pathological_inputs() {
            for dst_init in dsts {
                assert_match(&input, &dst_init, tx_type);
            }
        }
    }

    #[test]
    fn iht4x4_full1d_pathological_directed_dctdct() {
        run_pathological(TxType::DctDct);
    }

    #[test]
    fn iht4x4_full1d_pathological_directed_adstdct() {
        run_pathological(TxType::AdstDct);
    }

    #[test]
    fn iht4x4_full1d_pathological_directed_dctadst() {
        run_pathological(TxType::DctAdst);
    }

    #[test]
    fn iht4x4_full1d_pathological_directed_adstadst() {
        run_pathological(TxType::AdstAdst);
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
