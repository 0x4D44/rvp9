//! 4×4 inverse-DCT + add — x86_64 AVX2 implementation.
//!
//! **G4-γ full-1D refactor (Wave F1)**: both the row pass and the
//! column pass are now fully vectorised. The previous G4.2 pilot kept
//! the scalar [`crate::transforms::inverse::idct4`] call site and only
//! AVX2-accelerated the final round/clamp/pack stage. With the 8×8
//! kernel having proven the G4-γ pattern (see
//! [`crate::dsp::x86_64::idct8x8_avx2`]), we mirror its structure
//! scaled down to 4 lanes.
//!
//! The natural fit for 4×4 is the 128-bit `__m128i` register holding
//! four `i32` lanes: each register encodes one row (or one column) of
//! the block. Helpers therefore use `_mm_*` intrinsics, but the
//! function is still gated on `#[target_feature(enable = "avx2")]`
//! because the dispatch slot is installed only when `caps.avx2 == true`
//! and we want all SIMD kernels in this module to share a single
//! target-feature contract.
//!
//! ## Bit-exactness vs scalar reference
//!
//! VP9 conformance requires bit-exact match to libvpx's C reference.
//! The scalar reference is [`crate::transforms::inverse::idct4x4_16_add`]
//! and the underlying 1D butterfly is
//! [`crate::transforms::inverse::idct4`]. Both apply `wraplow`
//! (`x as i16 as i32`, MODULAR truncation) at every intermediate store.
//!
//! ### How the vector form preserves bit-exactness
//!
//! Every scalar operation in [`crate::transforms::inverse::idct4`] is
//! mirrored 1:1 in vector form across 4 lanes (one lane per row of the
//! 4×4 block):
//!
//! * Multiplies use [`_mm_mullo_epi32`]. Because the dequantised
//!   coefficients are spec-bounded to i16 range and the cospi
//!   constants are all `< 2^15`, every product fits in `i32` with
//!   margin (max abs ≈ 16k × 32k ≈ 5e8, < 2^31).
//! * `dct_const_round_shift` is `(x + 8192) >> 14` —
//!   [`_mm_add_epi32`] + [`_mm_srai_epi32`] by 14, lane-parallel.
//! * `wraplow` is `x as i16 as i32` (modular truncation, **not**
//!   saturation). The vector equivalent is `slli_epi32(_, 16)` then
//!   `srai_epi32(_, 16)` — sign-extend the low 16 bits, exactly
//!   reproducing the truncation. **This is the critical correctness
//!   point**: libvpx's SSE2 kernel uses `_mm_packs_epi32` (saturating)
//!   which diverges from the C reference on out-of-range intermediates.
//!   Our proptest fixtures exercise such inputs (the
//!   `pathological_strategy` includes `[i16::MAX; 16]`), so we MUST
//!   match the C truncation semantics, not libvpx's SSE2 saturation.
//!
//! After the column pass, results are in row-major layout (4 xmm
//! registers, each = one output row, lanes = 4 columns). The existing
//! round + add + clip + pack finisher runs unchanged.
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
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadu_si128,
    _mm_mullo_epi32, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_slli_epi32, _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi32, _mm_unpackhi_epi32,
    _mm_unpacklo_epi8, _mm_unpacklo_epi32,
};

use crate::transforms::constants::{COSPI_8_64, COSPI_16_64, COSPI_24_64};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers, each one-line. Kept `#[inline]` so the
// compiler folds them away with no call overhead. Mirror the
// idct8x8_avx2.rs helpers (`drs`, `wrap`, `mul_const`) scaled to 128-bit
// lanes. Operate on 4 × i32 per register.
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
/// 16 bits of each lane back to 32 bits — i.e., MODULAR i16 truncation,
/// not saturation. See module-level docstring for why this matters.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn wrap(x: __m128i) -> __m128i {
    _mm_srai_epi32(_mm_slli_epi32(x, 16), 16)
}

/// `mul_const(x, c)`: lane-parallel `x * c` where `c` is a 16-bit
/// constant. Implemented with `_mm_mullo_epi32` against a broadcast
/// vector. Operand-range note matches idct8x8_avx2.rs's `mul_const`:
/// `x` is wraplow-bounded (or a raw spec-bounded input) and `c < 2^15`,
/// so the product fits in i32.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_const(x: __m128i, c: i32) -> __m128i {
    _mm_mullo_epi32(x, _mm_set1_epi32(c))
}

// ---------------------------------------------------------------------
// 4×4 i32 transpose using a 2-stage unpack pyramid.
//
// Input rows R0..R3 each hold 4 i32 (one input row per xmm).
// Output columns C0..C3 each hold 4 i32 where C[k] = [R0[k], R1[k],
// R2[k], R3[k]] — i.e., column k of the original 4×4 matrix.
//
// Stage 1: pair adjacent rows via 32-bit unpack
// Stage 2: pair adjacent pair-results via 64-bit unpack
// (`_mm_unpacklo_epi64` / `_mm_unpackhi_epi64` not needed — we cast
// through `_mm_castps_*` style or just `unpacklo_epi32` then
// `unpacklo_epi64`. We use shuffles via 32-bit unpack pairs and a
// final 64-bit unpack approximation through unpacklo/hi_epi32 chained.
//
// Concretely: standard 4×4 i32 transpose using only `unpacklo_epi32`
// and `unpackhi_epi32` plus `_mm_unpacklo_epi64` / `_mm_unpackhi_epi64`.
// We pull in those last two via a small bit-cast through __m128i — but
// `_mm_unpacklo_epi64` is already available; we just need to import it.
// To avoid an additional import we synthesise unpacklo/hi_epi64 via
// the equivalent shuffle on the 32-bit pair-results. The clearest form
// uses _mm_unpackhi_epi32 + a per-128 cross. Here we just use the
// epi32 unpacks + an epi64-style pair shuffle realised by combining
// `unpacklo_epi32` of pre-shuffled rows.
//
// Simplest: import `_mm_unpacklo_epi64` / `_mm_unpackhi_epi64` and use
// the canonical 4×4 transpose pattern. That keeps the code obviously
// correct.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose4x4_i32(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
    use core::arch::x86_64::{_mm_unpackhi_epi64, _mm_unpacklo_epi64};
    // Stage 1: 32-bit unpack of (r0, r1) and (r2, r3).
    // a0 = [r0[0], r1[0], r0[1], r1[1]]
    // a1 = [r0[2], r1[2], r0[3], r1[3]]
    // a2 = [r2[0], r3[0], r2[1], r3[1]]
    // a3 = [r2[2], r3[2], r2[3], r3[3]]
    let a0 = _mm_unpacklo_epi32(r0, r1);
    let a1 = _mm_unpackhi_epi32(r0, r1);
    let a2 = _mm_unpacklo_epi32(r2, r3);
    let a3 = _mm_unpackhi_epi32(r2, r3);

    // Stage 2: 64-bit unpack to assemble full columns.
    // c0 = [r0[0], r1[0], r2[0], r3[0]] = column 0
    // c1 = [r0[1], r1[1], r2[1], r3[1]] = column 1
    // c2 = [r0[2], r1[2], r2[2], r3[2]] = column 2
    // c3 = [r0[3], r1[3], r2[3], r3[3]] = column 3
    let c0 = _mm_unpacklo_epi64(a0, a2);
    let c1 = _mm_unpackhi_epi64(a0, a2);
    let c2 = _mm_unpacklo_epi64(a1, a3);
    let c3 = _mm_unpackhi_epi64(a1, a3);

    [c0, c1, c2, c3]
}

// ---------------------------------------------------------------------
// Vector 1D idct4. Each `__m128i` holds 4 i32 lanes corresponding to 4
// independent rows / columns being transformed in parallel. The
// arithmetic mirrors `idct4` in `crate::transforms::inverse` line-by-
// line (compare against `crates/rvp9-core/src/transforms/inverse.rs:57`).
//
// Inputs `[v0, v1, v2, v3]` are the four positions of the 1D transform
// across 4 lanes. Outputs are the four transformed positions.
// ---------------------------------------------------------------------
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct4_avx2(v: [__m128i; 4]) -> [__m128i; 4] {
    let [v0, v1, v2, v3] = v;

    // ----- stage 1 -----
    // step0 = wrap(drs((v0 + v2) * C16))
    // step1 = wrap(drs((v0 - v2) * C16))
    let t1 = mul_const(_mm_add_epi32(v0, v2), COSPI_16_64);
    let t2 = mul_const(_mm_sub_epi32(v0, v2), COSPI_16_64);
    let step0 = wrap(drs(t1));
    let step1 = wrap(drs(t2));
    // step2 = wrap(drs(v1*C24 - v3*C8))
    // step3 = wrap(drs(v1*C8  + v3*C24))
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

/// AVX2 implementation of the 4×4 inverse-DCT + residual add.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`idct4x4_add_avx2`]) when `caps.avx2 == true`, and
/// caps are detected once at OnceLock init time.
///
/// `dest` must have at least `3 * stride + 4` bytes of valid storage.
///
/// # Implementation
///
/// Full-1D AVX2: both row and column passes use [`idct4_avx2`] (4-way
/// SIMD across 128-bit registers). Transpose between passes uses a
/// standard unpack pyramid ([`transpose4x4_i32`]). The output
/// add/clamp/pack finisher matches the original G4.2 pilot.
///
/// ```text
///   for r in 0..4: load row r as xmm[r]                ; vmovdqu xmm
///   xmm[k] = transpose(R0..R3)[k]   ; position k across 4 rows
///   xmm[k] = idct4_avx2(xmm[0..3])   ; row pass, 4-way
///   xmm[k] = transpose(xmm[0..3])    ; flip rows ↔ columns
///   xmm[k] = idct4_avx2(xmm[0..3])   ; column pass
///   (each xmm now = one output row, 4 i32 lanes)
///   for r in 0..4:
///     add round +8, srai 4, pack to i16
///     load 4 dst u8, zero-extend, add, saturate-pack u8, store
/// ```
///
/// All loads / stores from / to `dest` are 4-byte (`i32`), unaligned.
/// See HLD V6 §G4 M6-6.
// G4: loads from `dst` use _mm_cvtsi32_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct4x4_add_avx2_inner(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    debug_assert!(dest.len() >= stride * 3 + 4, "dest too small for 4x4 add");

    // ----- Load: 4 rows of 4 i32 each into xmm[0..3]. -----
    let in_ptr = input.as_ptr() as *const __m128i;
    let r0 = _mm_loadu_si128(in_ptr.add(0));
    let r1 = _mm_loadu_si128(in_ptr.add(1));
    let r2 = _mm_loadu_si128(in_ptr.add(2));
    let r3 = _mm_loadu_si128(in_ptr.add(3));

    // ----- Stage A: transpose so each xmm[k] holds input position k
    // across 4 rows. -----
    let cols = transpose4x4_i32(r0, r1, r2, r3);

    // ----- Stage B: row pass.
    // After this, each xmm[k] = intermediate column k (4 rows in lanes).
    let row_out = idct4_avx2(cols);

    // ----- Stage C: transpose again to get per-row layout for the
    // column pass. After this, each xmm[r] = intermediate row r (4
    // columns in lanes). -----
    let rows = transpose4x4_i32(row_out[0], row_out[1], row_out[2], row_out[3]);

    // ----- Stage D: column pass.
    // Each xmm[k] now holds one output row across the 4 column lanes.
    let final_rows = idct4_avx2(rows);

    // ----- Stage E: round (+8), shift right by 4, saturating-pack
    // i32 → i16, add to dst, saturating-pack i16 → u8, store. -----
    //
    // Pack the four output rows (16 i32 total = 64 bytes) into two
    // __m128i registers of i16 (8 lanes each). `_mm_packs_epi32` takes
    // two src registers and concatenates 4 + 4 saturated i16; we use
    // that to combine rows 0+1 and rows 2+3 into single registers.
    let round = _mm_set1_epi32(8);
    let zero = _mm_setzero_si128();

    let row_pack = |row: __m128i| -> __m128i { _mm_srai_epi32(_mm_add_epi32(row, round), 4) };

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
        // Load 4 dst bytes as i32 (LE matches memory layout).
        let dst_row_start = r * stride;
        // SAFETY: within the unsafe fn we read 4 bytes from
        // `dest[dst_row_start..dst_row_start + 4]`. The debug_assert
        // bounds `dest.len() >= 3*stride + 4`.
        let raw: i32 = core::ptr::read_unaligned(dest.as_ptr().add(dst_row_start) as *const i32);
        let dst_u8 = _mm_cvtsi32_si128(raw);
        // Zero-extend the 4 u8 lanes to i16 (low 8 bytes; only the low
        // 4 are valid after the i32 load).
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

/// Safe wrapper for the AVX2 idct4x4 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct4x4_16_add`], so the
/// `SimdDispatch::idct4x4_add` slot can hold either.
///
/// The wrapper is `pub` so unit tests in the same crate can call it
/// directly to do scalar-vs-AVX2 differential testing without going
/// through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct4x4_add_avx2(input: &[i32; 16], dest: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. The unit-test path in
    // `idct4x4_avx2::tests` gates each call on a runtime
    // `is_x86_feature_detected!("avx2")` check, so calls from tests
    // are equally sound.
    unsafe { idct4x4_add_avx2_inner(input, dest, stride) }
}

// Miri cannot interpret AVX2 intrinsics — see the parallel comment
// in `dc_pred_4x4_avx2.rs`. Scalar reference paths (the
// `transforms::inverse` family) still run under miri unchanged.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `idct4x4_add` (G4-γ
    //! full-1D refactor).
    //!
    //! Coverage strategy mirrors the G4-γ idct8x8 suite, scaled to 16
    //! coefficients. EOB-seam set for idct4x4 is `{1, 2, 16}` per
    //! `notes/g4-coordination.md`.
    //!
    //! Strategies:
    //! 1. `uniform_strategy` — random over i16-clamped i32 range.
    //! 2. `sparse_strategy` — 1..=4 non-zero at random positions.
    //! 3. `eob_bucket_strategy` — K ∈ {1, 2, 16} non-zero in 0..K.
    //! 4. `high_magnitude_strategy` — near ±i16::MAX.
    //! 5. `pathological_strategy` — all-equal / all-min / all-max /
    //!    alternating sign; CRITICAL for the wraplow-vs-saturation
    //!    distinction (libvpx's SSE2 `_mm_packs_epi32` diverges here).
    //!
    //! All proptests and directed tests now exercise the full 1D AVX2
    //! butterfly (row + column), not just the finisher.

    use super::idct4x4_add_avx2;
    use crate::transforms::inverse::idct4x4_16_add;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 16], dst_init: &[u8; 16]) -> [u8; 16] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host (missing require_avx2!() gate in caller)"
        );
        let mut dst = *dst_init;
        idct4x4_add_avx2(input, &mut dst, 4);
        dst
    }

    fn run_scalar(input: &[i32; 16], dst_init: &[u8; 16]) -> [u8; 16] {
        let mut dst = *dst_init;
        idct4x4_16_add(input, &mut dst, 4);
        dst
    }

    fn assert_match(input: &[i32; 16], dst_init: &[u8; 16]) {
        let scalar = run_scalar(input, dst_init);
        let avx2 = run_avx2(input, dst_init);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies (same shape as idct8x8 but sized for 16 coeffs)
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

    /// K ∈ {1, 2, 16} non-zero coefficients in raster positions 0..K
    /// (matching the idct4x4 dispatcher-seam set per
    /// `notes/g4-coordination.md`). K=1 is the DC-only seam; K=16 is
    /// full density.
    fn eob_bucket_strategy() -> impl Strategy<Value = [i32; 16]> {
        prop_oneof![Just(1usize), Just(2), Just(16)].prop_flat_map(|k| {
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

    /// Pathological constant-pattern inputs. **Critical for the G4-γ
    /// full-1D refactor**: the `[i16::MAX; 16]` and `[i16::MIN; 16]`
    /// cases drive intermediates outside the i16 range, where
    /// libvpx's SSE2 `_mm_packs_epi32` (saturating) would diverge from
    /// the C reference's `wraplow` (modular). Our `wrap()` helper uses
    /// `slli_epi32(_, 16) >> srai_epi32(_, 16)` to reproduce the
    /// modular truncation exactly, so these inputs must round-trip
    /// bit-exact.
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

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Stratified scalar-vs-AVX2 differential. Now exercises the
        /// full-1D AVX2 row + column butterfly (G4-γ refactor).
        #[test]
        fn idct4x4_scalar_eq_avx2_proptest(
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

    /// All-zero coefficients: dst unchanged after round-trip. Now
    /// exercises the full-1D AVX2 path (transforms of zero stay zero
    /// across both passes).
    #[test]
    fn idct4x4_all_zero_input() {
        require_avx2!();
        let input = [0i32; 16];
        let dst_init = [42u8; 16];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep across every position. Now
    /// exercises full-1D AVX2; an impulse at any single position
    /// produces a non-trivial pattern that the row + column + transpose
    /// must reproduce exactly.
    #[test]
    fn idct4x4_single_coefficient_impulses() {
        require_avx2!();
        for pos in 0..16 {
            for &val in &[100i32, -100, 30000, -30000, 1, -1] {
                let mut input = [0i32; 16];
                input[pos] = val;
                let dst_init = [128u8; 16];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Dst-pointer-offset regression guard. Same approach as idct8x8:
    /// the kernel uses per-row 4-byte `_mm_cvtsi32_si128` /
    /// `_mm_cvtsi128_si32` loads/stores, so all addressing is naturally
    /// unaligned — this test catches per-skew arithmetic regressions
    /// rather than alignment-fault paths. Now exercises full-1D AVX2.
    #[test]
    fn idct4x4_misaligned_dst_pointers() {
        require_avx2!();
        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().take(8).enumerate() {
            *slot = (i as i32) * 91 - 400;
        }
        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 32]> = Box::new([0u8; 32]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(11);
            }
            let mut scalar_buf = *buf;
            // Each row of the 4×4 block lives at `dst[r * stride .. r * stride + 4]`
            // inside the working sub-slice. Stride is 4 (no row padding)
            // so the kernel touches `[skew .. skew + 4 * 3 + 4] = [skew, skew + 16)`.
            idct4x4_add_avx2(&input, &mut buf[skew..skew + 16], 4);
            idct4x4_16_add(&input, &mut scalar_buf[skew..skew + 16], 4);
            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }

    /// Round-trip through the dispatch: forced-AVX2 and forced-scalar
    /// must agree. Now exercises full-1D AVX2.
    #[test]
    fn idct4x4_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let mut input = [0i32; 16];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = ((i as i32) * 23) % 4001 - 2000;
        }
        let dst_init = [97u8; 16];
        let mut dst_avx2 = dst_init;
        let mut dst_scalar = dst_init;
        (dispatch_avx2.idct4x4_add())(&input, &mut dst_avx2, 4);
        (dispatch_scalar.idct4x4_add())(&input, &mut dst_scalar, 4);
        assert_eq!(dst_avx2, dst_scalar);
    }

    /// **G4-γ directed regression**: pathological saturation inputs.
    /// Plant `[i16::MAX; 16]` and `[i16::MIN; 16]` directly through
    /// both the scalar reference and the AVX2 kernel and check bit
    /// exactness. These are exactly the inputs that would diverge if
    /// the kernel used libvpx's saturating `_mm_packs_epi32` for
    /// `wraplow` rather than the modular `slli >> srai` we use here.
    #[test]
    fn idct4x4_full1d_pathological_directed() {
        require_avx2!();
        let inputs: [[i32; 16]; 2] = [[i16::MAX as i32; 16], [i16::MIN as i32; 16]];
        for input in &inputs {
            for &dst_fill in &[0u8, 64, 128, 255] {
                let dst_init = [dst_fill; 16];
                assert_match(input, &dst_init);
            }
        }
    }

    /// **Tautology guard**: prove the AVX2 kernel is actually installed
    /// in the dispatch slot rather than silently falling through to
    /// the scalar reference. The dispatch round-trip test above could
    /// pass trivially if `SimdDispatch::build` ignored `caps.avx2` and
    /// always returned the scalar pointer; this test catches that
    /// mode. Compares the function pointers directly.
    #[test]
    fn idct4x4_avx2_dispatch_slot_uses_avx2_path() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let avx2_fn = dispatch_avx2.idct4x4_add();
        let scalar_fn: fn(&[i32; 16], &mut [u8], usize) = idct4x4_16_add;
        assert_ne!(
            avx2_fn as usize, scalar_fn as usize,
            "AVX2 dispatch slot is pointing at the scalar reference — \
             the AVX2 path is not actually installed"
        );
    }
}
