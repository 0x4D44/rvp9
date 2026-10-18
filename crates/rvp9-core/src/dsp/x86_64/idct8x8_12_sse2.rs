//! 8×8 inverse-DCT + add — x86_64 SSE2 implementation for the
//! `eob <= 12` (sparse top-left 4×4) bucket.
//!
//! This kernel is a line-by-line port of libvpx's
//! `vpx_idct8x8_12_add_sse2` (`third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:174-185`
//! and the inner kernel at `inv_txfm_sse2.h:285-341`). It is the first
//! SSE2-only kernel in this codebase; all prior SIMD work uses AVX2.
//!
//! ## eob<=12 contract
//!
//! When the caller knows the bitstream eob for an 8×8 transform block
//! is ≤ 12, the dequantised coefficients are confined to the top-left
//! 4×4 sub-block — i.e. rows 4..7 of the 8-row input are all zero, and
//! within rows 0..3 only the first 4 columns may be non-zero. The
//! kernel exploits this by loading only 4 i16 lanes per row for the
//! top 4 rows, running a 4×4 row transpose, a partial-rows row pass,
//! re-expanding to 8-row layout via `idct8x8_12_transpose_16bit_4x8`,
//! then running the full 8-row `idct8` column pass + finisher.
//!
//! Calling this with a non-sparse input is undefined behaviour
//! (matches libvpx semantics); the dispatch is the caller's
//! responsibility.
//!
//! ## Saturating-vs-modular semantic divergence
//!
//! Unlike the [`super::idct8x8_avx2`] kernel — which uses i32 lanes
//! end-to-end + modular `wraplow` (sign-extend low 16 bits) to
//! reproduce the C reference's `int16_t step[]` truncation **exactly**
//! including out-of-range intermediates — this SSE2 _12 kernel uses
//! the dense i16/`_mm_madd_epi16` pipeline from libvpx's SSE2 source.
//! `_mm_packs_epi32` (i32 → i16, saturating) clamps out-of-range
//! intermediates to `±i16::MAX` rather than truncating modularly.
//!
//! For spec-bounded bitstream inputs (post-dequant coefficients in
//! `[-32768, 32767]` and intermediates that stay within i16 range
//! through the 4-stage butterfly), the two semantics are identical.
//! The divergence only triggers for synthetic/adversarial inputs
//! where intermediate magnitudes exceed `i16::MAX` — i.e. the same
//! "in-bounds bitstream range" contract noted at
//! `super::idct8x8_avx2:42-58`, but accepted here rather than worked
//! around. This matches `vpxdec` behaviour exactly on real bitstreams.
//!
//! The differential proptest at the bottom of this file is bounded
//! accordingly (≤ i14 magnitudes); the AVX2-style "pathological"
//! strategies (`[i16::MAX; 64]`, alternating sign at i16 extremes)
//! are **deliberately omitted** because they trip the saturation
//! divergence and would fail spuriously.
//!
//! ## Convention compatibility
//!
//! 1. The body uses `#[target_feature(enable = "sse2")]` on an
//!    `unsafe fn`. (`target_feature_1.1` is unstable on stable Rust
//!    today, so the function MUST be `unsafe`; same constraint as the
//!    AVX2 kernels.)
//! 2. A safe wrapper [`idct8x8_add_sse2_12`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant — see [`crate::dsp::dispatch`].
//! 3. SSE2 is architecturally universal on x86_64 (every x86_64 CPU
//!    must implement it per the SysV ABI / MS x64 ABI), but tests
//!    and any future call sites still gate on
//!    `is_x86_feature_detected!("sse2")` belt-and-braces.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadl_epi64` (8-byte, unaligned). The
//! grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_loadl_epi64, _mm_loadu_si128, _mm_madd_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_srai_epi16, _mm_srai_epi32, _mm_storel_epi64, _mm_sub_epi16, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_4_64, COSPI_8_64, COSPI_12_64, COSPI_16_64, COSPI_20_64, COSPI_24_64, COSPI_28_64,
};

// ---------------------------------------------------------------------
// Tiny intrinsic helpers, each one-line and `#[inline]` so the compiler
// folds them away with no call overhead.
// ---------------------------------------------------------------------

/// `pair_set_epi16(a, b)` — pack `[a, b, a, b, a, b, a, b]` into an
/// `__m128i` of 8 i16 lanes (cospi pre-pack for `_mm_madd_epi16`).
///
/// Mirrors libvpx
/// `pair_set_epi16(a, b)` macro in
/// `third_party/libvpx/vpx_dsp/x86/txfm_common_sse2.h:17-19`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub(crate) unsafe fn pair_set_epi16(a: i16, b: i16) -> __m128i {
    // libvpx's macro is `_mm_set_epi16(b, a, b, a, b, a, b, a)`.
    // Equivalent to broadcasting a packed `(b << 16) | (a & 0xFFFF)` i32.
    let packed = ((b as u16 as u32) << 16) | (a as u16 as u32);
    _mm_set1_epi32(packed as i32)
}

/// `idct_madd_round_shift_sse2(in, cospi)` — lane-wise pair-multiply,
/// round-half-up by `DCT_CONST_ROUNDING = 8192`, shift right by
/// `DCT_CONST_BITS = 14`. Returns 4 × i32 lanes.
///
/// Mirrors libvpx `inv_txfm_sse2.h:65-69` (note: arg order matches
/// libvpx — `in` is the i16 inputs, `cospi` is the pair-set constants).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub(crate) unsafe fn idct_madd_round_shift_sse2(in_: __m128i, cospi: __m128i) -> __m128i {
    let t = _mm_madd_epi16(in_, cospi);
    _mm_srai_epi32(_mm_add_epi32(t, _mm_set1_epi32(1 << 13)), 14)
}

/// `idct_calc_wraplow_sse2(in0, in1, x)` — two `idct_madd_round_shift_sse2`
/// dot-products + saturating `_mm_packs_epi32` to 8 i16 lanes.
///
/// Mirrors libvpx `inv_txfm_sse2.h:72-78`. Note the saturating pack:
/// out-of-range i32 → clamped to `±i16::MAX`, **not** modular truncated.
/// This is the source of the saturation-semantics divergence vs the
/// scalar reference for adversarial inputs (see module docstring).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub(crate) unsafe fn idct_calc_wraplow_sse2(in0: __m128i, in1: __m128i, x: __m128i) -> __m128i {
    let t0 = idct_madd_round_shift_sse2(in0, x);
    let t1 = idct_madd_round_shift_sse2(in1, x);
    _mm_packs_epi32(t0, t1)
}

/// `butterfly(in0, in1, c0, c1) -> (out0, out1)` — two-output butterfly
/// via `idct_calc_wraplow_sse2` over `unpacklo` / `unpackhi` interleave
/// of `in0`/`in1` with `pair_set_epi16(c0, -c1)` and `pair_set_epi16(c1, c0)`.
///
/// Mirrors libvpx `inv_txfm_sse2.h:81-90`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub(crate) unsafe fn butterfly(in0: __m128i, in1: __m128i, c0: i16, c1: i16) -> (__m128i, __m128i) {
    let cst0 = pair_set_epi16(c0, -c1);
    let cst1 = pair_set_epi16(c1, c0);
    let lo = _mm_unpacklo_epi16(in0, in1);
    let hi = _mm_unpackhi_epi16(in0, in1);
    let out0 = idct_calc_wraplow_sse2(lo, hi, cst0);
    let out1 = idct_calc_wraplow_sse2(lo, hi, cst1);
    (out0, out1)
}

/// `transpose_16bit_4x4(in)` — 4×4 transpose of i16 lanes living in the
/// low half of 4 `__m128i`. Returns 2 `__m128i`, each holding 2
/// transposed rows.
///
/// Lane mapping (per libvpx `transpose_sse2.h:97-112` comments):
///
/// ```text
///   in[0]: 00 01 02 03  XX XX XX XX
///   in[1]: 10 11 12 13  XX XX XX XX
///   in[2]: 20 21 22 23  XX XX XX XX
///   in[3]: 30 31 32 33  XX XX XX XX
///   ->
///   out[0]: 00 10 20 30  01 11 21 31
///   out[1]: 02 12 22 32  03 13 23 33
/// ```
///
/// Implementation in-place: takes the 4-row input as `io[0..4]` and
/// overwrites `io[0]` / `io[1]` with the transposed rows. `io[2]` /
/// `io[3]` are clobbered (libvpx writes the result back to `out[0..2]`;
/// the caller in the _12 kernel uses `transpose_16bit_4x4(io, io)`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn transpose_16bit_4x4(io: &mut [__m128i; 8]) {
    let a0 = _mm_unpacklo_epi16(io[0], io[1]);
    let a1 = _mm_unpacklo_epi16(io[2], io[3]);
    io[0] = _mm_unpacklo_epi32(a0, a1);
    io[1] = _mm_unpackhi_epi32(a0, a1);
}

/// `load_input_data4(data)` — load 4 i32 coefficients and saturating-pack
/// to 4 i16 lanes in the low half of an `__m128i`.
///
/// Our coefficients are stored as i32 throughout the dequant pipeline.
/// In libvpx's non-HBD build this is `_mm_loadl_epi64` (already i16);
/// in the HBD build it's `_mm_load_si128 + _mm_packs_epi32`. We always
/// pack because our `tran_low_t` is i32. Spec-bounded inputs never
/// saturate (i16 range by VP9 §8.7.1).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn load_input_data4(data: *const i32) -> __m128i {
    let v = _mm_loadu_si128(data as *const __m128i);
    _mm_packs_epi32(v, _mm_setzero_si128())
}

/// Full 8-row 1-D inverse DCT, SSE2 i16-lane variant. Each `__m128i`
/// in `in` / `out` holds 8 i16 lanes corresponding to one row of the
/// 8×8 block; the 8 register-positions are the 8 independent inputs to
/// the 1-D idct8.
///
/// Line-by-line port of libvpx `inv_txfm_sse2.h:250-283`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct8_sse2(in_: &[__m128i; 8], out: &mut [__m128i; 8]) {
    // stage 1
    let (s1_4, s1_7) = butterfly(in_[1], in_[7], COSPI_28_64 as i16, COSPI_4_64 as i16);
    let (s1_5, s1_6) = butterfly(in_[5], in_[3], COSPI_12_64 as i16, COSPI_20_64 as i16);

    // stage 2 — note libvpx's butterfly call signature `&step2[1], &step2[0]`
    // (output order flipped) for the cospi_16,cospi_16 pair.
    let (s2_1, s2_0) = butterfly(in_[0], in_[4], COSPI_16_64 as i16, COSPI_16_64 as i16);
    let (s2_2, s2_3) = butterfly(in_[2], in_[6], COSPI_24_64 as i16, COSPI_8_64 as i16);

    let s2_4 = _mm_add_epi16(s1_4, s1_5);
    let s2_5 = _mm_sub_epi16(s1_4, s1_5);
    let s2_6 = _mm_sub_epi16(s1_7, s1_6);
    let s2_7 = _mm_add_epi16(s1_7, s1_6);

    // stage 3
    let st1_0 = _mm_add_epi16(s2_0, s2_3);
    let st1_1 = _mm_add_epi16(s2_1, s2_2);
    let st1_2 = _mm_sub_epi16(s2_1, s2_2);
    let st1_3 = _mm_sub_epi16(s2_0, s2_3);
    let (st1_5, st1_6) = butterfly(s2_6, s2_5, COSPI_16_64 as i16, COSPI_16_64 as i16);

    // stage 4
    out[0] = _mm_add_epi16(st1_0, s2_7);
    out[1] = _mm_add_epi16(st1_1, st1_6);
    out[2] = _mm_add_epi16(st1_2, st1_5);
    out[3] = _mm_add_epi16(st1_3, s2_4);
    out[4] = _mm_sub_epi16(st1_3, s2_4);
    out[5] = _mm_sub_epi16(st1_2, st1_5);
    out[6] = _mm_sub_epi16(st1_1, st1_6);
    out[7] = _mm_sub_epi16(st1_0, s2_7);
}

/// `idct8x8_12_transpose_16bit_4x8(tmp, io)` — re-layout 4 stage-4
/// outputs into 4-row 8-wide form for the column pass.
///
/// Line-by-line port of libvpx `inv_txfm_sse2.h:22-58`. Lane mapping
/// per libvpx comments (entries are row,column indices of the
/// post-stage-4 i16-lane layout):
///
/// ```text
///   in[0]: 30 31 32 33  00 01 02 03
///   in[1]: 20 21 22 23  10 11 12 13
///   in[2]: 40 41 42 43  70 71 72 73
///   in[3]: 50 51 52 53  60 61 62 63
///   ->
///   out[0]: 00 10 20 30  40 50 60 70
///   out[1]: 01 11 21 31  41 51 61 71
///   out[2]: 02 12 22 32  42 52 62 72
///   out[3]: 03 13 23 33  43 53 63 73
/// ```
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct8x8_12_transpose_16bit_4x8(in_: &[__m128i; 4], out: &mut [__m128i; 8]) {
    let tr0_0 = _mm_unpackhi_epi16(in_[0], in_[1]);
    let tr0_1 = _mm_unpacklo_epi16(in_[1], in_[0]);
    let tr0_2 = _mm_unpacklo_epi16(in_[2], in_[3]);
    let tr0_3 = _mm_unpackhi_epi16(in_[3], in_[2]);

    let tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
    let tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
    let tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
    let tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);

    out[0] = _mm_unpacklo_epi64(tr1_0, tr1_1);
    out[1] = _mm_unpackhi_epi64(tr1_0, tr1_1);
    out[2] = _mm_unpacklo_epi64(tr1_2, tr1_3);
    out[3] = _mm_unpackhi_epi64(tr1_2, tr1_3);
}

/// SSE2 implementation of the 8×8 inverse-DCT + residual add for the
/// `eob <= 12` sparse bucket.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2. Since SSE2 is
/// universal on x86_64 (mandated by both SysV AMD64 and MS x64 ABIs),
/// this is essentially free; the [`SimdDispatch::build`] constructor
/// guards via `caps.sse2 == true` for consistency with the other
/// SIMD slots.
///
/// `dst` must have at least `7 * stride + 8` bytes of valid storage.
///
/// **Sparseness contract**: only `input[0..32]` is read (the top
/// 4 rows × 8 cols of the 8×8 layout). Among those 32 entries, only
/// the top-left 4×4 sub-block is expected to be non-zero by the
/// libvpx eob<=12 contract, but the kernel reads all of
/// `input[0..32]` unconditionally. `input[32..64]` is ignored.
///
/// # Implementation
///
/// ```text
///   load row 0..3 (low 4 i16 each) via load_input_data4
///   transpose_16bit_4x4 — flip rows ↔ columns in the 4×4 corner
///   idct8x8_12_add_kernel partial row pass → 4 tmp regs
///   idct8x8_12_transpose_16bit_4x8 → 8-row 8-col i16 layout, rows 4..7 = zero
///   idct8 — full 8-row column pass on i16 lanes
///   write_buffer_8x8 — +16, >>5, 8-byte recon_and_store per row
/// ```
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct8x8_add_sse2_12_inner(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 7 + 8, "dst too small for 8x8 add");

    let zero = _mm_setzero_si128();
    let cp_16_16 = pair_set_epi16(COSPI_16_64 as i16, COSPI_16_64 as i16);
    let cp_16_n16 = pair_set_epi16(COSPI_16_64 as i16, -(COSPI_16_64 as i16));

    // Load the 4 sparse-bucket rows. Each register holds 4 i16 in the
    // low half + 0s in the high half.
    let mut io: [__m128i; 8] = [zero; 8];
    let in_ptr = input.as_ptr();
    io[0] = load_input_data4(in_ptr.add(0));
    io[1] = load_input_data4(in_ptr.add(8));
    io[2] = load_input_data4(in_ptr.add(16));
    io[3] = load_input_data4(in_ptr.add(24));

    // 4×4 transpose. After this:
    //   io[0]: 00 10 20 30  01 11 21 31
    //   io[1]: 02 12 22 32  03 13 23 33
    transpose_16bit_4x4(&mut io);

    // --- partial-rows row pass (libvpx idct8x8_12_add_kernel_sse2) ---

    let mut step1 = [zero; 8];
    let mut step2 = [zero; 8];
    let mut tmp = [zero; 4];

    // stage 1 — odd butterflies producing step1[4] / step1[5].
    {
        let cp_28_n4 = pair_set_epi16(COSPI_28_64 as i16, -(COSPI_4_64 as i16));
        let cp_4_28 = pair_set_epi16(COSPI_4_64 as i16, COSPI_28_64 as i16);
        let cp_n20_12 = pair_set_epi16(-(COSPI_20_64 as i16), COSPI_12_64 as i16);
        let cp_12_20 = pair_set_epi16(COSPI_12_64 as i16, COSPI_20_64 as i16);
        let lo_1 = _mm_unpackhi_epi16(io[0], zero);
        let lo_3 = _mm_unpackhi_epi16(io[1], zero);
        step1[4] = idct_calc_wraplow_sse2(cp_28_n4, cp_4_28, lo_1); // step1 4&7
        step1[5] = idct_calc_wraplow_sse2(cp_n20_12, cp_12_20, lo_3); // step1 5&6
    }

    // stage 2.
    {
        let cp_24_n8 = pair_set_epi16(COSPI_24_64 as i16, -(COSPI_8_64 as i16));
        let cp_8_24 = pair_set_epi16(COSPI_8_64 as i16, COSPI_24_64 as i16);
        let lo_0 = _mm_unpacklo_epi16(io[0], zero);
        let lo_2 = _mm_unpacklo_epi16(io[1], zero);
        let t = idct_madd_round_shift_sse2(cp_16_16, lo_0);
        step2[0] = _mm_packs_epi32(t, t); // step2 0&1
        step2[2] = idct_calc_wraplow_sse2(cp_8_24, cp_24_n8, lo_2); // step2 3&2
        step2[4] = _mm_add_epi16(step1[4], step1[5]); // step2 4&7
        step2[5] = _mm_sub_epi16(step1[4], step1[5]); // step2 5&6
        step2[6] = _mm_unpackhi_epi64(step2[5], zero); // step2 6
    }

    // stage 3.
    {
        let lo_65 = _mm_unpacklo_epi16(step2[6], step2[5]);
        tmp[0] = _mm_add_epi16(step2[0], step2[2]); // step1 0&1
        tmp[1] = _mm_sub_epi16(step2[0], step2[2]); // step1 3&2
        step1[2] = _mm_unpackhi_epi64(tmp[1], tmp[0]); // step1 2&1
        step1[3] = _mm_unpacklo_epi64(tmp[1], tmp[0]); // step1 3&0
        step1[5] = idct_calc_wraplow_sse2(cp_16_n16, cp_16_16, lo_65); // step1 5&6
    }

    // stage 4.
    tmp[0] = _mm_add_epi16(step1[3], step2[4]); // output 3&0
    tmp[1] = _mm_add_epi16(step1[2], step1[5]); // output 2&1
    tmp[2] = _mm_sub_epi16(step1[3], step2[4]); // output 4&7
    tmp[3] = _mm_sub_epi16(step1[2], step1[5]); // output 5&6

    // Expand 4 tmp lanes back to 8-row layout for the column pass.
    idct8x8_12_transpose_16bit_4x8(&tmp, &mut io);
    io[4] = zero;
    io[5] = zero;
    io[6] = zero;
    io[7] = zero;

    // Column pass — full 8-row idct8 on i16 lanes.
    let in_copy = io;
    idct8_sse2(&in_copy, &mut io);

    // --- write_buffer_8x8 finisher: (x + 16) >> 5, residual add. ---
    let final_rounding = _mm_set1_epi16(1 << 4);
    for (r, row) in io.iter().enumerate() {
        let mut t = _mm_add_epi16(*row, final_rounding);
        t = _mm_srai_epi16(t, 5);
        // recon_and_store: load 8 dst bytes, zero-extend to 8 × i16, add, sat-pack.
        let dst_row_ptr = dst.as_mut_ptr().add(r * stride) as *const __m128i;
        let d_u8 = _mm_loadl_epi64(dst_row_ptr);
        let d_i16 = _mm_unpacklo_epi8(d_u8, zero);
        let sum_i16 = _mm_add_epi16(t, d_i16);
        let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
        let dst_row_ptr_mut = dst.as_mut_ptr().add(r * stride) as *mut __m128i;
        _mm_storel_epi64(dst_row_ptr_mut, sum_u8);
    }
}

/// Safe wrapper for the SSE2 idct8x8 _12 inner kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct8x8_12_add`], so the
/// `SimdDispatch::idct8x8_12_add` slot can hold either.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly for scalar-vs-SSE2 differential
/// testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct8x8_add_sse2_12(input: &[i32; 64], dst: &mut [u8], stride: usize) {
    crate::dsp::validate_predictor_storage(dst.len(), stride, 8, 8)
        .expect("invalid inverse transform destination");
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is architecturally universal on
    // x86_64 (mandated by both SysV AMD64 and MS x64 ABIs), so the
    // gate is essentially free. The unit-test path gates each call
    // on a runtime `is_x86_feature_detected!("sse2")` check.
    unsafe { idct8x8_add_sse2_12_inner(input, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Per-helper unit tests + outer-kernel differential proptest for
    //! `idct8x8_add_sse2_12`.
    //!
    //! All proptest strategies bound coefficients to ≤ i14 magnitude
    //! so intermediate values stay within i16 range — the SSE2
    //! kernel's saturating `_mm_packs_epi32` then agrees bit-exactly
    //! with the scalar reference. See module docstring for the
    //! saturation-divergence rationale.

    use super::*;
    use crate::transforms::constants::{DCT_CONST_BITS, dct_const_round_shift, wraplow};
    use crate::transforms::inverse::{idct8, idct8x8_64_add};
    use core::arch::x86_64::_mm_storeu_si128;
    use proptest::prelude::*;

    macro_rules! require_sse2 {
        () => {
            if !is_x86_feature_detected!("sse2") {
                return;
            }
        };
    }

    /// Extract 8 i16 lanes from an `__m128i` via `_mm_storeu_si128`.
    fn extract_i16x8(v: __m128i) -> [i16; 8] {
        let mut out = [0i16; 8];
        // SAFETY: SSE2 is gated by the caller's `require_sse2!()`
        // macro; `out` is 8 × i16 = 16 bytes, matching `__m128i`.
        unsafe {
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);
        }
        out
    }

    /// Extract 4 i32 lanes from an `__m128i`.
    fn extract_i32x4(v: __m128i) -> [i32; 4] {
        let mut out = [0i32; 4];
        // SAFETY: SSE2 is gated by the caller's `require_sse2!()`
        // macro; `out` is 4 × i32 = 16 bytes, matching `__m128i`.
        unsafe {
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);
        }
        out
    }

    /// Pack 8 i16 lanes into an `__m128i`.
    fn pack_i16x8(lanes: [i16; 8]) -> __m128i {
        // SAFETY: SSE2 is gated by the caller's `require_sse2!()`
        // macro; `lanes` is 8 × i16 = 16 bytes.
        unsafe { _mm_loadu_si128(lanes.as_ptr() as *const __m128i) }
    }

    // -----------------------------------------------------------------
    // Sub-task 1: pair_set_epi16
    // -----------------------------------------------------------------

    #[test]
    fn pair_set_epi16_directed() {
        require_sse2!();
        let cases = [
            (0i16, 0i16),
            (1, 2),
            (-1, -2),
            (i16::MAX, i16::MIN),
            (COSPI_28_64 as i16, -(COSPI_4_64 as i16)),
            (COSPI_16_64 as i16, COSPI_16_64 as i16),
        ];
        for (a, b) in cases {
            // SAFETY: SSE2 gated by `require_sse2!()`.
            let v = unsafe { pair_set_epi16(a, b) };
            let lanes = extract_i16x8(v);
            assert_eq!(lanes, [a, b, a, b, a, b, a, b], "a={a}, b={b}");
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 2: idct_madd_round_shift_sse2
    // -----------------------------------------------------------------

    /// Scalar oracle for `_mm_madd_epi16(in, cospi) + 8192) >> 14`.
    /// `_mm_madd_epi16` produces lane k of the output as
    /// `in[2k] * cospi[2k] + in[2k+1] * cospi[2k+1]` (i32).
    fn scalar_madd_round_shift(in_: [i16; 8], cospi: [i16; 8]) -> [i32; 4] {
        let mut out = [0i32; 4];
        for k in 0..4 {
            let prod = (in_[2 * k] as i32) * (cospi[2 * k] as i32)
                + (in_[2 * k + 1] as i32) * (cospi[2 * k + 1] as i32);
            out[k] = (prod + (1 << (DCT_CONST_BITS - 1))) >> DCT_CONST_BITS;
        }
        out
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct_madd_round_shift_proptest(
            // i14-bounded so intermediate fits comfortably in i32 (no overflow).
            in_lanes in prop::array::uniform8(-16384i16..=16383i16),
            a in -16384i16..=16383i16,
            b in -16384i16..=16383i16,
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            // cospi is a pair-set [a, b, a, b, a, b, a, b].
            let cospi_lanes = [a, b, a, b, a, b, a, b];
            let in_v = pack_i16x8(in_lanes);
            let cospi_v = pack_i16x8(cospi_lanes);
            // SAFETY: SSE2 gated above.
            let result_v = unsafe { idct_madd_round_shift_sse2(in_v, cospi_v) };
            let result = extract_i32x4(result_v);
            let expected = scalar_madd_round_shift(in_lanes, cospi_lanes);
            prop_assert_eq!(result, expected);
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 3: idct_calc_wraplow_sse2
    // -----------------------------------------------------------------

    /// Scalar oracle: same as madd_round_shift but saturating-cast to i16.
    fn scalar_calc_wraplow(in0: [i16; 8], in1: [i16; 8], cospi: [i16; 8]) -> [i16; 8] {
        let t0 = scalar_madd_round_shift(in0, cospi);
        let t1 = scalar_madd_round_shift(in1, cospi);
        let sat = |v: i32| -> i16 { v.clamp(i16::MIN as i32, i16::MAX as i32) as i16 };
        [
            sat(t0[0]),
            sat(t0[1]),
            sat(t0[2]),
            sat(t0[3]),
            sat(t1[0]),
            sat(t1[1]),
            sat(t1[2]),
            sat(t1[3]),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct_calc_wraplow_proptest(
            in0_lanes in prop::array::uniform8(-16384i16..=16383i16),
            in1_lanes in prop::array::uniform8(-16384i16..=16383i16),
            a in -16384i16..=16383i16,
            b in -16384i16..=16383i16,
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let cospi_lanes = [a, b, a, b, a, b, a, b];
            let in0_v = pack_i16x8(in0_lanes);
            let in1_v = pack_i16x8(in1_lanes);
            let cospi_v = pack_i16x8(cospi_lanes);
            // SAFETY: SSE2 gated above.
            let result_v = unsafe { idct_calc_wraplow_sse2(in0_v, in1_v, cospi_v) };
            let result = extract_i16x8(result_v);
            let expected = scalar_calc_wraplow(in0_lanes, in1_lanes, cospi_lanes);
            prop_assert_eq!(result, expected);
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 4: butterfly
    // -----------------------------------------------------------------

    /// Scalar `butterfly(in0, in1, c0, c1) -> (out0, out1)` per libvpx.
    /// Each output lane k is `(in0[k] * c_a + in1[k] * c_b + 8192) >> 14`,
    /// saturating-cast to i16. For out0: c_a=c0, c_b=-c1. For out1: c_a=c1, c_b=c0.
    fn scalar_butterfly(in0: [i16; 8], in1: [i16; 8], c0: i16, c1: i16) -> ([i16; 8], [i16; 8]) {
        let sat = |v: i32| -> i16 { v.clamp(i16::MIN as i32, i16::MAX as i32) as i16 };
        let mut out0 = [0i16; 8];
        let mut out1 = [0i16; 8];
        for k in 0..8 {
            let p0 = (in0[k] as i32) * (c0 as i32) + (in1[k] as i32) * (-(c1 as i32));
            let p1 = (in0[k] as i32) * (c1 as i32) + (in1[k] as i32) * (c0 as i32);
            out0[k] = sat((p0 + (1 << 13)) >> 14);
            out1[k] = sat((p1 + (1 << 13)) >> 14);
        }
        (out0, out1)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn butterfly_proptest(
            in0_lanes in prop::array::uniform8(-16384i16..=16383i16),
            in1_lanes in prop::array::uniform8(-16384i16..=16383i16),
            c0 in 0i16..=16384,
            c1 in 0i16..=16384,
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let in0_v = pack_i16x8(in0_lanes);
            let in1_v = pack_i16x8(in1_lanes);
            // SAFETY: SSE2 gated above.
            let (out0_v, out1_v) = unsafe { butterfly(in0_v, in1_v, c0, c1) };
            let out0 = extract_i16x8(out0_v);
            let out1 = extract_i16x8(out1_v);
            let (e0, e1) = scalar_butterfly(in0_lanes, in1_lanes, c0, c1);
            prop_assert_eq!(out0, e0);
            prop_assert_eq!(out1, e1);
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 5: transpose_16bit_4x4
    // -----------------------------------------------------------------

    #[test]
    fn transpose_16bit_4x4_directed() {
        require_sse2!();
        // Input row r = [r*10+0, r*10+1, r*10+2, r*10+3, X, X, X, X]
        // for r in 0..4.
        let row_in =
            |r: i16| -> [i16; 8] { [r * 10, r * 10 + 1, r * 10 + 2, r * 10 + 3, -1, -2, -3, -4] };
        // SAFETY: _mm_setzero_si128 has no preconditions; unsafe required only because it is a compiler intrinsic.
        let mut io: [__m128i; 8] = [unsafe { _mm_setzero_si128() }; 8];
        io[0] = pack_i16x8(row_in(0));
        io[1] = pack_i16x8(row_in(1));
        io[2] = pack_i16x8(row_in(2));
        io[3] = pack_i16x8(row_in(3));
        // SAFETY: SSE2 gated by `require_sse2!()`.
        unsafe {
            transpose_16bit_4x4(&mut io);
        }
        let r0 = extract_i16x8(io[0]);
        let r1 = extract_i16x8(io[1]);
        // Expected per libvpx comments:
        //   out[0]: 00 10 20 30  01 11 21 31
        //   out[1]: 02 12 22 32  03 13 23 33
        assert_eq!(r0, [0, 10, 20, 30, 1, 11, 21, 31]);
        assert_eq!(r1, [2, 12, 22, 32, 3, 13, 23, 33]);
    }

    // -----------------------------------------------------------------
    // Sub-task 6: load_input_data4
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn load_input_data4_proptest(
            data in prop::array::uniform4(-32768i32..=32767),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            // SAFETY: SSE2 gated above.
            let v = unsafe { load_input_data4(data.as_ptr()) };
            let lanes = extract_i16x8(v);
            // Low 4 lanes = input cast to i16 (in-range, no saturation).
            // High 4 lanes = 0.
            prop_assert_eq!(lanes[0], data[0] as i16);
            prop_assert_eq!(lanes[1], data[1] as i16);
            prop_assert_eq!(lanes[2], data[2] as i16);
            prop_assert_eq!(lanes[3], data[3] as i16);
            prop_assert_eq!(lanes[4], 0);
            prop_assert_eq!(lanes[5], 0);
            prop_assert_eq!(lanes[6], 0);
            prop_assert_eq!(lanes[7], 0);
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 7: idct8_sse2 (full 8-row column-pass kernel)
    //
    // Strategy: pack 8 input "columns" (each a single i16-per-row
    // vector) into 8 __m128i; the scalar `idct8` from
    // transforms::inverse processes one 8-element column at a time.
    // We extract per-position columns from the SSE2 output and compare
    // to the scalar.
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct8_sse2_proptest(
            // Per-lane magnitude bounded to i12 (≈ ±2048): a full 8×8
            // worth of inputs at this magnitude keeps every 1-D idct8
            // butterfly intermediate inside i16 range, so the SSE2
            // saturating-pack stays bit-exact against the scalar
            // wraplow-truncating reference. Tighter than the
            // top-helpers' i14 bound because the inner kernel here
            // processes 8 inputs in parallel through 4 butterfly stages
            // (intermediate growth ≈ 8×) whereas a single
            // `idct_calc_wraplow_sse2` only does one madd-add.
            rows in prop::array::uniform8(prop::array::uniform8(-2048i16..=2047i16)),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            // Pack each row into one __m128i (8 i16 lanes per row).
            // After this, lane k of `in_v[r]` holds the value at
            // (row r, col k).
            let in_v: [__m128i; 8] = [
                pack_i16x8(rows[0]),
                pack_i16x8(rows[1]),
                pack_i16x8(rows[2]),
                pack_i16x8(rows[3]),
                pack_i16x8(rows[4]),
                pack_i16x8(rows[5]),
                pack_i16x8(rows[6]),
                pack_i16x8(rows[7]),
            ];
            // SAFETY: _mm_setzero_si128 has no preconditions; unsafe required only because it is a compiler intrinsic.
            let mut out_v: [__m128i; 8] = [unsafe { _mm_setzero_si128() }; 8];
            // SAFETY: SSE2 gated above.
            unsafe { idct8_sse2(&in_v, &mut out_v); }

            // Compare each of the 8 lane-positions (the 8 independent
            // 1-D idct8 instances).
            let out_lanes: [[i16; 8]; 8] = [
                extract_i16x8(out_v[0]),
                extract_i16x8(out_v[1]),
                extract_i16x8(out_v[2]),
                extract_i16x8(out_v[3]),
                extract_i16x8(out_v[4]),
                extract_i16x8(out_v[5]),
                extract_i16x8(out_v[6]),
                extract_i16x8(out_v[7]),
            ];
            for k in 0..8 {
                // Column k input = [rows[0][k], rows[1][k], ...].
                let scalar_in: [i32; 8] = [
                    rows[0][k] as i32, rows[1][k] as i32, rows[2][k] as i32,
                    rows[3][k] as i32, rows[4][k] as i32, rows[5][k] as i32,
                    rows[6][k] as i32, rows[7][k] as i32,
                ];
                let mut scalar_out = [0i32; 8];
                idct8(&scalar_in, &mut scalar_out);
                // Each row r of the SSE2 output, lane k = scalar_out[r]
                // (saturating-cast to i16 — within range here so it's
                // identical to a regular cast).
                for r in 0..8 {
                    let expected = scalar_out[r].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    prop_assert_eq!(
                        out_lanes[r][k], expected,
                        "row {} col {}: sse2={} scalar={}",
                        r, k, out_lanes[r][k], expected
                    );
                }
            }
        }
    }

    // Bind `dct_const_round_shift` / `wraplow` to silence unused-import
    // warnings on builds where they are not referenced by an active path.
    // The functions are kept available for any future scalar-oracle
    // helpers added below.
    #[allow(dead_code)]
    fn _silence_unused() {
        let _ = dct_const_round_shift;
        let _ = wraplow;
    }

    // -----------------------------------------------------------------
    // Sub-task 8: idct8x8_12_transpose_16bit_4x8
    // -----------------------------------------------------------------

    #[test]
    fn idct8x8_12_transpose_16bit_4x8_directed() {
        require_sse2!();
        // Build inputs per libvpx comments:
        //   in[0]: 30 31 32 33  00 01 02 03
        //   in[1]: 20 21 22 23  10 11 12 13
        //   in[2]: 40 41 42 43  70 71 72 73
        //   in[3]: 50 51 52 53  60 61 62 63
        //
        // We use the encoding `rc = r * 10 + c` for entry (row r, col c).
        let rc = |r: i16, c: i16| -> i16 { r * 10 + c };
        let in0 = [
            rc(3, 0),
            rc(3, 1),
            rc(3, 2),
            rc(3, 3),
            rc(0, 0),
            rc(0, 1),
            rc(0, 2),
            rc(0, 3),
        ];
        let in1 = [
            rc(2, 0),
            rc(2, 1),
            rc(2, 2),
            rc(2, 3),
            rc(1, 0),
            rc(1, 1),
            rc(1, 2),
            rc(1, 3),
        ];
        let in2 = [
            rc(4, 0),
            rc(4, 1),
            rc(4, 2),
            rc(4, 3),
            rc(7, 0),
            rc(7, 1),
            rc(7, 2),
            rc(7, 3),
        ];
        let in3 = [
            rc(5, 0),
            rc(5, 1),
            rc(5, 2),
            rc(5, 3),
            rc(6, 0),
            rc(6, 1),
            rc(6, 2),
            rc(6, 3),
        ];
        let in_v: [__m128i; 4] = [
            pack_i16x8(in0),
            pack_i16x8(in1),
            pack_i16x8(in2),
            pack_i16x8(in3),
        ];
        // SAFETY: _mm_setzero_si128 has no preconditions; unsafe required only because it is a compiler intrinsic.
        let mut out_v: [__m128i; 8] = [unsafe { _mm_setzero_si128() }; 8];
        // SAFETY: SSE2 gated by `require_sse2!()`.
        unsafe {
            idct8x8_12_transpose_16bit_4x8(&in_v, &mut out_v);
        }
        // Expected:
        //   out[0]: 00 10 20 30  40 50 60 70
        //   out[1]: 01 11 21 31  41 51 61 71
        //   out[2]: 02 12 22 32  42 52 62 72
        //   out[3]: 03 13 23 33  43 53 63 73
        let expected_col = |c: i16| -> [i16; 8] {
            [
                rc(0, c),
                rc(1, c),
                rc(2, c),
                rc(3, c),
                rc(4, c),
                rc(5, c),
                rc(6, c),
                rc(7, c),
            ]
        };
        assert_eq!(extract_i16x8(out_v[0]), expected_col(0));
        assert_eq!(extract_i16x8(out_v[1]), expected_col(1));
        assert_eq!(extract_i16x8(out_v[2]), expected_col(2));
        assert_eq!(extract_i16x8(out_v[3]), expected_col(3));
    }

    // -----------------------------------------------------------------
    // Sub-task 9: outer kernel differential against scalar idct8x8_64_add
    // (with rows 4..7 zeroed, since libvpx _12 ≡ _64 with rows 4..7 = 0).
    // -----------------------------------------------------------------

    /// Zero rows 4..7 (positions 32..64) of a 64-element coefficient
    /// array — the eob<=12 contract.
    fn force_top_left_4x4(mut input: [i32; 64]) -> [i32; 64] {
        for slot in &mut input[32..64] {
            *slot = 0;
        }
        // Further restrict to top-left 4×4: within rows 0..3, only
        // columns 0..3 may be non-zero. (The libvpx kernel reads cols
        // 0..3 of rows 0..3 only; cols 4..7 of rows 0..3 of the input
        // are not read.)
        for r in 0..4 {
            for c in 4..8 {
                input[r * 8 + c] = 0;
            }
        }
        input
    }

    fn run_sse2(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        assert!(is_x86_feature_detected!("sse2"));
        let mut dst = *dst_init;
        idct8x8_add_sse2_12(input, &mut dst, 8);
        dst
    }

    fn run_scalar(input: &[i32; 64], dst_init: &[u8; 64]) -> [u8; 64] {
        let mut dst = *dst_init;
        idct8x8_64_add(input, &mut dst, 8);
        dst
    }

    fn assert_match(input: &[i32; 64], dst_init: &[u8; 64]) {
        let input = force_top_left_4x4(*input);
        let scalar = run_scalar(&input, dst_init);
        let sse2 = run_sse2(&input, dst_init);
        assert_eq!(
            scalar, sse2,
            "SSE2 _12 mismatch:\ninput={input:?}\ndst_init={dst_init:?}\n\
             scalar={scalar:?}\nsse2={sse2:?}"
        );
    }

    // -----------------------------------------------------------------
    // Strategies (per plan §6)
    // -----------------------------------------------------------------

    /// `uniform_strategy_12` — i12 range, top-left 4×4 only.
    fn uniform_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(-2048i32..=2047, 16).prop_map(|v| {
            // 16 = 4×4 top-left non-zero slots.
            let mut out = [0i32; 64];
            for (i, val) in v.into_iter().enumerate() {
                let r = i / 4;
                let c = i % 4;
                out[r * 8 + c] = val;
            }
            out
        })
    }

    /// `sparse_strategy_12` — 1..=4 non-zero coefs in the 4×4 corner.
    /// Magnitudes at i12 to keep accumulated row/column-pass
    /// intermediates within i16 (no sat-pack divergence). With only 4
    /// coefs in play we could push to i13 safely, but i12 mirrors the
    /// plan's "production-realistic" range and aligns with
    /// `uniform_strategy_12`.
    fn sparse_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec((0usize..16, -2048i32..=2047), 1..=4).prop_map(|nz| {
            let mut out = [0i32; 64];
            for (idx, val) in nz {
                let r = idx / 4;
                let c = idx % 4;
                out[r * 8 + c] = val;
            }
            out
        })
    }

    /// `eob_bucket_strategy_12` — K ∈ {1, 2, 8, 12} non-zero coefs at
    /// low raster positions within the 4×4 corner. Magnitudes at i12
    /// so that K=12 (the dispatcher seam upper bound) does not
    /// accumulate past i16.
    fn eob_bucket_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![Just(1usize), Just(2), Just(8), Just(12)].prop_flat_map(|k| {
            prop::collection::vec(-2048i32..=2047, k).prop_map(move |values| {
                let mut out = [0i32; 64];
                // Place at the first k raster positions of the 4×4 corner.
                for (i, val) in values.into_iter().enumerate() {
                    let r = i / 4;
                    let c = i % 4;
                    out[r * 8 + c] = val;
                }
                out
            })
        })
    }

    /// `high_magnitude_strategy_12` — i14 magnitude (`[15000, 16384]`)
    /// at 1..=2 sparse positions within the 4×4 corner. Stresses the
    /// high-magnitude end of the bit-exact range; we cap the number of
    /// non-zero coefs because the SSE2 kernel's saturating-pack
    /// diverges from the wraplow-truncating scalar reference once
    /// stage intermediates accumulate past i16 (which happens quickly
    /// if you fill all 16 corner slots at i14).
    fn high_magnitude_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop::collection::vec(
            (
                0usize..16,
                (15000i32..=16384, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m }),
            ),
            1..=2,
        )
        .prop_map(|placements| {
            let mut out = [0i32; 64];
            for (idx, val) in placements {
                let r = idx / 4;
                let c = idx % 4;
                out[r * 8 + c] = val;
            }
            out
        })
    }

    fn input_strategy_12() -> impl Strategy<Value = [i32; 64]> {
        prop_oneof![
            uniform_strategy_12(),
            sparse_strategy_12(),
            eob_bucket_strategy_12(),
            high_magnitude_strategy_12(),
        ]
    }

    fn dst_strategy() -> impl Strategy<Value = [u8; 64]> {
        prop::collection::vec(any::<u8>(), 64).prop_map(|v| {
            let mut out = [0u8; 64];
            out.copy_from_slice(&v);
            out
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        /// Outer-kernel differential: scalar idct8x8_64_add(input with
        /// rows 4..7 and cols 4..7 of top rows zeroed) ≡ SSE2 _12 on
        /// the same input.
        #[test]
        fn idct8x8_12_scalar_eq_sse2_proptest(
            input in input_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let input = force_top_left_4x4(input);
            let scalar = run_scalar(&input, &dst_init);
            let sse2 = run_sse2(&input, &dst_init);
            prop_assert_eq!(scalar, sse2);
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures
    // -----------------------------------------------------------------

    #[test]
    fn idct8x8_12_all_zero_input() {
        require_sse2!();
        let input = [0i32; 64];
        let dst_init = [42u8; 64];
        assert_match(&input, &dst_init);
    }

    /// Single-coefficient impulse sweep: for every position 0..32 (the
    /// eob<=12 caller-contract domain — positions 32..63 are zero, and
    /// `force_top_left_4x4` further restricts to the 4×4 corner), set
    /// just that coefficient to a non-zero value and confirm scalar /
    /// SSE2 agreement.
    #[test]
    fn idct8x8_12_single_coefficient_impulses() {
        require_sse2!();
        for pos in 0..32 {
            for &val in &[100i32, -100, 16000, -16000, 1, -1] {
                let mut input = [0i32; 64];
                input[pos] = val;
                let dst_init = [128u8; 64];
                assert_match(&input, &dst_init);
            }
        }
    }

    /// Dst-pointer-offset correctness regression guard. Mirrors the
    /// AVX2 idct8x8 test of the same name (`super::idct8x8_avx2::tests::
    /// idct8x8_misaligned_dst_pointers`). The kernel uses
    /// `_mm_loadl_epi64` (8-byte, alignment-free) so this is a per-
    /// offset arithmetic regression guard, NOT an alignment-fault test.
    #[test]
    fn idct8x8_12_misaligned_dst_pointers() {
        require_sse2!();
        // Synthetic input — 12 non-zero coefs in the 4×4 top-left corner
        // (exactly the eob<=12 sparseness boundary).
        let mut input = [0i32; 64];
        for r in 0..4 {
            for c in 0..3 {
                input[r * 8 + c] = ((r as i32) * 4 + (c as i32)) * 137 - 800;
            }
        }
        // Pin to the strict eob<=12 layout.
        let input = force_top_left_4x4(input);

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 80]> = Box::new([0u8; 80]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(7);
            }
            let mut scalar_buf = *buf;

            idct8x8_add_sse2_12(&input, &mut buf[skew..skew + 64], 8);
            idct8x8_64_add(&input, &mut scalar_buf[skew..skew + 64], 8);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "SSE2/scalar mismatch at skew={skew}: per-offset \
                 arithmetic regression"
            );
        }
    }
}
