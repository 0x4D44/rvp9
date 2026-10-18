//! 32×32 inverse-DCT + add — x86_64 SSE2 i16-lane implementation
//! covering the dense `idct32x32_1024_add` slot when AVX2 is unavailable
//! (and as the preferred dense kernel even when AVX2 is available — i16
//! lanes pack 16 elements per xmm vs. 8 per ymm for the i32-lane AVX2
//! path, and benchmark-measured ~17% decode speedup carries forward
//! from the 16×16 flip).
//!
//! This kernel is a line-by-line port of libvpx's
//! `vpx_idct32x32_1024_add_sse2` (`third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:1163-1190`)
//! together with the inner 1-D engine `idct32_1024_8x32` (and its
//! `quarter_1/_2/_3_4` helpers) from `inv_txfm_sse2.c:1019-1161` and the
//! shared `idct32_8x32_quarter_2_stage_4_to_6` /
//! `idct32_8x32_quarter_3_4_stage_4_to_7` helpers from
//! `inv_txfm_sse2.h:599-697`. It is the third SSE2-only kernel in this
//! codebase (after `idct8x8_12_sse2` and `idct16x16_sse2_256`) — see
//! Track-T HLD `wrk_docs/Track_T/T4_i16_lane_IDCT_HLD.md` for the
//! porting plan.
//!
//! ## Saturating-vs-modular semantic divergence
//!
//! Like the [`super::idct16x16_sse2_256`] kernel, this implementation
//! uses the dense `_mm_madd_epi16` pipeline with `_mm_packs_epi32`
//! (saturating, clamps to `±i16::MAX`) at every wraplow site rather
//! than the modular `wraplow` (sign-extend low 16 bits) used by the
//! i32-lane AVX2 kernels and the scalar reference. The two policies
//! agree on every in-spec bitstream coefficient (and on the empirical
//! 201-vector conformance corpus), but diverge on synthetic adversarial
//! inputs whose intermediate magnitudes exceed `i16::MAX`.
//!
//! Per Track-T precedent, the differential proptest at the bottom of
//! this file is bounded to ≤ i12 magnitudes (`[-2048, 2047]`); the
//! AVX2-style "pathological" strategies (i16-extreme alternating-sign
//! patterns) are **deliberately omitted** because they trip the
//! saturation divergence and would fail spuriously.
//!
//! ## Convention compatibility
//!
//! 1. The body uses `#[target_feature(enable = "sse2")]` on
//!    `unsafe fn`. SSE2 is architecturally universal on x86_64 (per
//!    SysV AMD64 / MS x64 ABIs), but tests still gate on
//!    `is_x86_feature_detected!("sse2")` belt-and-braces.
//! 2. A safe wrapper [`idct32x32_add_sse2_1024`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant — see [`crate::dsp::dispatch`].
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadl_epi64` (8-byte, unaligned) via
//! `recon_and_store_8x1` (one 8-byte residual add per row × 32 rows × 4
//! strips). The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use crate::transforms::constants::{
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64,
};

#[cfg(target_arch = "x86_64")]
use super::idct8x8_12_sse2::{butterfly, idct_calc_wraplow_sse2, pair_set_epi16};

// ---------------------------------------------------------------------
// Tiny shared helpers (line-by-line ports from libvpx).
// ---------------------------------------------------------------------

/// `load_input_data8(data)` — load 8 i32 coefficients and saturating-pack
/// to 8 i16 lanes.
///
/// Same as `idct16x16_sse2_256::load_input_data8`. Local copy avoids
/// a cross-module `pub(crate)` re-export.
///
/// Mirrors libvpx `inv_txfm_sse2.h:111-119` (HBD branch).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2 (universal on x86_64; gated at
///   dispatch-build time via `caps.sse2`).
/// - `data..data+8` is a valid, readable 32-byte region of i32.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn load_input_data8(data: *const i32) -> __m128i {
    let in0 = _mm_loadu_si128(data as *const __m128i);
    let in1 = _mm_loadu_si128(data.add(4) as *const __m128i);
    _mm_packs_epi32(in0, in1)
}

/// `transpose_16bit_8x8(in, out)` — 8×8 transpose of i16 lanes.
///
/// Line-by-line port of libvpx `transpose_sse2.h:157-220`. Pure shuffle,
/// no arithmetic, so bit-exact for arbitrary i16 lane values.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn transpose_16bit_8x8(in_: &[__m128i], out: &mut [__m128i]) {
    // Stage 1 — 16-bit unpacks.
    let a0 = _mm_unpacklo_epi16(in_[0], in_[1]);
    let a1 = _mm_unpacklo_epi16(in_[2], in_[3]);
    let a2 = _mm_unpacklo_epi16(in_[4], in_[5]);
    let a3 = _mm_unpacklo_epi16(in_[6], in_[7]);
    let a4 = _mm_unpackhi_epi16(in_[0], in_[1]);
    let a5 = _mm_unpackhi_epi16(in_[2], in_[3]);
    let a6 = _mm_unpackhi_epi16(in_[4], in_[5]);
    let a7 = _mm_unpackhi_epi16(in_[6], in_[7]);

    // Stage 2 — 32-bit unpacks.
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpacklo_epi32(a2, a3);
    let b2 = _mm_unpacklo_epi32(a4, a5);
    let b3 = _mm_unpacklo_epi32(a6, a7);
    let b4 = _mm_unpackhi_epi32(a0, a1);
    let b5 = _mm_unpackhi_epi32(a2, a3);
    let b6 = _mm_unpackhi_epi32(a4, a5);
    let b7 = _mm_unpackhi_epi32(a6, a7);

    // Stage 3 — 64-bit unpacks → final transposed rows.
    out[0] = _mm_unpacklo_epi64(b0, b1);
    out[1] = _mm_unpackhi_epi64(b0, b1);
    out[2] = _mm_unpacklo_epi64(b4, b5);
    out[3] = _mm_unpackhi_epi64(b4, b5);
    out[4] = _mm_unpacklo_epi64(b2, b3);
    out[5] = _mm_unpackhi_epi64(b2, b3);
    out[6] = _mm_unpacklo_epi64(b6, b7);
    out[7] = _mm_unpackhi_epi64(b6, b7);
}

/// `load_transpose_16bit_8x8(input, stride, in_)` — load and transpose
/// an 8×8 i16 block, reading 8 rows of i32 coefficients at the given
/// stride.
///
/// Mirrors libvpx `inv_txfm_sse2.h:121-133`.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `input..input + 7*stride + 8` is a valid, readable region of i32.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn load_transpose_16bit_8x8(input: *const i32, stride: usize, in_: &mut [__m128i]) {
    in_[0] = load_input_data8(input);
    in_[1] = load_input_data8(input.add(stride));
    in_[2] = load_input_data8(input.add(2 * stride));
    in_[3] = load_input_data8(input.add(3 * stride));
    in_[4] = load_input_data8(input.add(4 * stride));
    in_[5] = load_input_data8(input.add(5 * stride));
    in_[6] = load_input_data8(input.add(6 * stride));
    in_[7] = load_input_data8(input.add(7 * stride));
    // libvpx calls `transpose_16bit_8x8(in, in)` — i.e. the 8×8 in-place
    // transpose pattern. Our helper takes `(&[..], &mut [..])` so we
    // copy through a tiny scratch array.
    let src: [__m128i; 8] = [
        in_[0], in_[1], in_[2], in_[3], in_[4], in_[5], in_[6], in_[7],
    ];
    transpose_16bit_8x8(&src, in_);
}

/// `recon_and_store_8x1(dest, in_)` — saturating residual add of 8 i16
/// lanes into 8 dst bytes, with the final +32 rounding / arith >>6 done
/// by the caller (see `store_buffer_8x32`).
///
/// Mirrors libvpx `inv_txfm_sse2.h:135-142` (`recon_and_store`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `dest..dest+8` is a valid, readable + writable u8 region.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn recon_and_store_8x1(dest: *mut u8, in_x: __m128i) {
    let zero = _mm_setzero_si128();
    let d_u8 = _mm_loadl_epi64(dest as *const __m128i);
    let d_i16 = _mm_unpacklo_epi8(d_u8, zero);
    let sum_i16 = _mm_add_epi16(in_x, d_i16);
    let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
    _mm_storel_epi64(dest as *mut __m128i, sum_u8);
}

/// `store_buffer_8x32(in_, dst, stride)` — final-rounding (+32, >>6) of
/// 32 rows × 8 i16 lanes followed by 8-byte residual adds. Mirrors
/// libvpx `inv_txfm_sse2.h:211-227` exactly (paired add + shift +
/// recon_and_store loop).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `dst..dst + 31*stride + 8` is a valid u8 region.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn store_buffer_8x32(in_: &mut [__m128i; 32], mut dst: *mut u8, stride: usize) {
    let final_rounding = _mm_set1_epi16(1 << 5);
    let mut j = 0;
    while j < 32 {
        in_[j] = _mm_adds_epi16(in_[j], final_rounding);
        in_[j + 1] = _mm_adds_epi16(in_[j + 1], final_rounding);

        in_[j] = _mm_srai_epi16(in_[j], 6);
        in_[j + 1] = _mm_srai_epi16(in_[j + 1], 6);

        recon_and_store_8x1(dst, in_[j]);
        dst = dst.add(stride);
        recon_and_store_8x1(dst, in_[j + 1]);
        dst = super::super::advance_idct_dst_row(dst, stride, j + 2 < 32);
        j += 2;
    }
}

/// `add_sub_butterfly(in_, out, size)` — paired add/sub butterfly used
/// at the final stage of the 1-D idct. Mirrors libvpx
/// `inv_txfm_sse2.h:238-248`.
///
/// For each `i` in `0..size/2`: `out[i] = in_[i] + in_[size-1-i]`,
/// `out[size-1-i] = in_[i] - in_[size-1-i]`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn add_sub_butterfly(in_: &[__m128i], out: &mut [__m128i], size: usize) {
    let half = size >> 1;
    let bound = size - 1;
    for i in 0..half {
        out[i] = _mm_add_epi16(in_[i], in_[bound - i]);
        out[bound - i] = _mm_sub_epi16(in_[i], in_[bound - i]);
    }
}

// ---------------------------------------------------------------------
// idct32_1024_8x32 quarter helpers (line-by-line ports from libvpx).
// ---------------------------------------------------------------------

/// `idct32_8x32_quarter_2_stage_4_to_6(step1, out)` — shared with the
/// 34/135 sparse paths. Mirrors libvpx `inv_txfm_sse2.h:599-630`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_8x32_quarter_2_stage_4_to_6(step1: &mut [__m128i; 16], out: &mut [__m128i]) {
    let mut step2: [__m128i; 16] = [_mm_setzero_si128(); 16];

    // stage 4
    step2[8] = step1[8];
    step2[15] = step1[15];
    let (s9, s14) = butterfly(step1[14], step1[9], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step2[9] = s9;
    step2[14] = s14;
    let (s10, s13) = butterfly(
        step1[13],
        step1[10],
        -(COSPI_8_64 as i16),
        COSPI_24_64 as i16,
    );
    step2[10] = s10;
    step2[13] = s13;
    step2[11] = step1[11];
    step2[12] = step1[12];

    // stage 5
    step1[8] = _mm_add_epi16(step2[8], step2[11]);
    step1[9] = _mm_add_epi16(step2[9], step2[10]);
    step1[10] = _mm_sub_epi16(step2[9], step2[10]);
    step1[11] = _mm_sub_epi16(step2[8], step2[11]);
    step1[12] = _mm_sub_epi16(step2[15], step2[12]);
    step1[13] = _mm_sub_epi16(step2[14], step2[13]);
    step1[14] = _mm_add_epi16(step2[14], step2[13]);
    step1[15] = _mm_add_epi16(step2[15], step2[12]);

    // stage 6
    out[8] = step1[8];
    out[9] = step1[9];
    let (o10, o13) = butterfly(step1[13], step1[10], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[10] = o10;
    out[13] = o13;
    let (o11, o12) = butterfly(step1[12], step1[11], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[11] = o11;
    out[12] = o12;
    out[14] = step1[14];
    out[15] = step1[15];
}

/// `idct32_8x32_quarter_3_4_stage_4_to_7(step1, out)` — mirrors libvpx
/// `inv_txfm_sse2.h:632-697`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_8x32_quarter_3_4_stage_4_to_7(step1: &mut [__m128i; 32], out: &mut [__m128i]) {
    let mut step2: [__m128i; 32] = [_mm_setzero_si128(); 32];

    // stage 4
    step2[16] = _mm_add_epi16(step1[16], step1[19]);
    step2[17] = _mm_add_epi16(step1[17], step1[18]);
    step2[18] = _mm_sub_epi16(step1[17], step1[18]);
    step2[19] = _mm_sub_epi16(step1[16], step1[19]);
    step2[20] = _mm_sub_epi16(step1[23], step1[20]);
    step2[21] = _mm_sub_epi16(step1[22], step1[21]);
    step2[22] = _mm_add_epi16(step1[22], step1[21]);
    step2[23] = _mm_add_epi16(step1[23], step1[20]);

    step2[24] = _mm_add_epi16(step1[24], step1[27]);
    step2[25] = _mm_add_epi16(step1[25], step1[26]);
    step2[26] = _mm_sub_epi16(step1[25], step1[26]);
    step2[27] = _mm_sub_epi16(step1[24], step1[27]);
    step2[28] = _mm_sub_epi16(step1[31], step1[28]);
    step2[29] = _mm_sub_epi16(step1[30], step1[29]);
    step2[30] = _mm_add_epi16(step1[29], step1[30]);
    step2[31] = _mm_add_epi16(step1[28], step1[31]);

    // stage 5
    step1[16] = step2[16];
    step1[17] = step2[17];
    let (s18, s29) = butterfly(step2[29], step2[18], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step1[18] = s18;
    step1[29] = s29;
    let (s19, s28) = butterfly(step2[28], step2[19], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step1[19] = s19;
    step1[28] = s28;
    let (s20, s27) = butterfly(
        step2[27],
        step2[20],
        -(COSPI_8_64 as i16),
        COSPI_24_64 as i16,
    );
    step1[20] = s20;
    step1[27] = s27;
    let (s21, s26) = butterfly(
        step2[26],
        step2[21],
        -(COSPI_8_64 as i16),
        COSPI_24_64 as i16,
    );
    step1[21] = s21;
    step1[26] = s26;
    step1[22] = step2[22];
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[25] = step2[25];
    step1[30] = step2[30];
    step1[31] = step2[31];

    // stage 6
    out[16] = _mm_add_epi16(step1[16], step1[23]);
    out[17] = _mm_add_epi16(step1[17], step1[22]);
    out[18] = _mm_add_epi16(step1[18], step1[21]);
    out[19] = _mm_add_epi16(step1[19], step1[20]);
    step2[20] = _mm_sub_epi16(step1[19], step1[20]);
    step2[21] = _mm_sub_epi16(step1[18], step1[21]);
    step2[22] = _mm_sub_epi16(step1[17], step1[22]);
    step2[23] = _mm_sub_epi16(step1[16], step1[23]);

    step2[24] = _mm_sub_epi16(step1[31], step1[24]);
    step2[25] = _mm_sub_epi16(step1[30], step1[25]);
    step2[26] = _mm_sub_epi16(step1[29], step1[26]);
    step2[27] = _mm_sub_epi16(step1[28], step1[27]);
    out[28] = _mm_add_epi16(step1[27], step1[28]);
    out[29] = _mm_add_epi16(step1[26], step1[29]);
    out[30] = _mm_add_epi16(step1[25], step1[30]);
    out[31] = _mm_add_epi16(step1[24], step1[31]);

    // stage 7
    let (o20, o27) = butterfly(step2[27], step2[20], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[20] = o20;
    out[27] = o27;
    let (o21, o26) = butterfly(step2[26], step2[21], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[21] = o21;
    out[26] = o26;
    let (o22, o25) = butterfly(step2[25], step2[22], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[22] = o22;
    out[25] = o25;
    let (o23, o24) = butterfly(step2[24], step2[23], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[23] = o23;
    out[24] = o24;
}

/// `idct32_1024_8x32_quarter_1(in_, out)` — even-index quarter of the
/// 1-D idct32. Mirrors libvpx `inv_txfm_sse2.c:1022-1056`.
///
/// Reads `in_[0, 4, 8, 12, 16, 20, 24, 28]`; writes `out[0..8]`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_1024_8x32_quarter_1(in_: &[__m128i; 32], out: &mut [__m128i]) {
    let mut step1: [__m128i; 8] = [_mm_setzero_si128(); 8];
    let mut step2: [__m128i; 8] = [_mm_setzero_si128(); 8];

    // stage 3
    let (s14, s17) = butterfly(in_[4], in_[28], COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[4] = s14;
    step1[7] = s17;
    let (s15, s16) = butterfly(in_[20], in_[12], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[5] = s15;
    step1[6] = s16;

    // stage 4 — note butterfly output order (s2_1, s2_0) for the
    // cospi_16,cospi_16 pair.
    let (s2_1, s2_0) = butterfly(in_[0], in_[16], COSPI_16_64 as i16, COSPI_16_64 as i16);
    step2[1] = s2_1;
    step2[0] = s2_0;
    let (s2_2, s2_3) = butterfly(in_[8], in_[24], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step2[2] = s2_2;
    step2[3] = s2_3;
    step2[4] = _mm_add_epi16(step1[4], step1[5]);
    step2[5] = _mm_sub_epi16(step1[4], step1[5]);
    step2[6] = _mm_sub_epi16(step1[7], step1[6]);
    step2[7] = _mm_add_epi16(step1[7], step1[6]);

    // stage 5
    step1[0] = _mm_add_epi16(step2[0], step2[3]);
    step1[1] = _mm_add_epi16(step2[1], step2[2]);
    step1[2] = _mm_sub_epi16(step2[1], step2[2]);
    step1[3] = _mm_sub_epi16(step2[0], step2[3]);
    step1[4] = step2[4];
    let (s5, s6) = butterfly(step2[6], step2[5], COSPI_16_64 as i16, COSPI_16_64 as i16);
    step1[5] = s5;
    step1[6] = s6;
    step1[7] = step2[7];

    // stage 6
    out[0] = _mm_add_epi16(step1[0], step1[7]);
    out[1] = _mm_add_epi16(step1[1], step1[6]);
    out[2] = _mm_add_epi16(step1[2], step1[5]);
    out[3] = _mm_add_epi16(step1[3], step1[4]);
    out[4] = _mm_sub_epi16(step1[3], step1[4]);
    out[5] = _mm_sub_epi16(step1[2], step1[5]);
    out[6] = _mm_sub_epi16(step1[1], step1[6]);
    out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

/// `idct32_1024_8x32_quarter_2(in_, out)` — odd-multiples-of-2 quarter.
/// Mirrors libvpx `inv_txfm_sse2.c:1061-1082`.
///
/// Reads `in_[2, 6, 10, 14, 18, 22, 26, 30]`; writes `out[8..16]`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_1024_8x32_quarter_2(in_: &[__m128i; 32], out: &mut [__m128i]) {
    let mut step1: [__m128i; 16] = [_mm_setzero_si128(); 16];
    let mut step2: [__m128i; 16] = [_mm_setzero_si128(); 16];

    // stage 2
    let (s8, s15) = butterfly(in_[2], in_[30], COSPI_30_64 as i16, COSPI_2_64 as i16);
    step2[8] = s8;
    step2[15] = s15;
    let (s9, s14) = butterfly(in_[18], in_[14], COSPI_14_64 as i16, COSPI_18_64 as i16);
    step2[9] = s9;
    step2[14] = s14;
    let (s10, s13) = butterfly(in_[10], in_[22], COSPI_22_64 as i16, COSPI_10_64 as i16);
    step2[10] = s10;
    step2[13] = s13;
    let (s11, s12) = butterfly(in_[26], in_[6], COSPI_6_64 as i16, COSPI_26_64 as i16);
    step2[11] = s11;
    step2[12] = s12;

    // stage 3
    step1[8] = _mm_add_epi16(step2[8], step2[9]);
    step1[9] = _mm_sub_epi16(step2[8], step2[9]);
    step1[10] = _mm_sub_epi16(step2[11], step2[10]);
    step1[11] = _mm_add_epi16(step2[11], step2[10]);
    step1[12] = _mm_add_epi16(step2[12], step2[13]);
    step1[13] = _mm_sub_epi16(step2[12], step2[13]);
    step1[14] = _mm_sub_epi16(step2[15], step2[14]);
    step1[15] = _mm_add_epi16(step2[15], step2[14]);

    idct32_8x32_quarter_2_stage_4_to_6(&mut step1, out);
}

/// `idct32_1024_8x32_quarter_1_2(in_, out)` — combines quarters 1 and 2
/// then runs the size-16 add/sub butterfly. Mirrors libvpx
/// `inv_txfm_sse2.c:1084-1091`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_1024_8x32_quarter_1_2(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let mut temp: [__m128i; 16] = [_mm_setzero_si128(); 16];
    idct32_1024_8x32_quarter_1(in_, &mut temp);
    idct32_1024_8x32_quarter_2(in_, &mut temp);
    // stage 7 — size-16 add/sub.
    add_sub_butterfly(&temp, &mut out[..16], 16);
}

/// `idct32_1024_8x32_quarter_3_4(in_, out)` — odd-indexed (1, 3, 5, 7,
/// 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31) inputs. Mirrors libvpx
/// `inv_txfm_sse2.c:1097-1151`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_1024_8x32_quarter_3_4(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let mut step1: [__m128i; 32] = [_mm_setzero_si128(); 32];
    let mut step2: [__m128i; 32] = [_mm_setzero_si128(); 32];

    // stage 1
    let (a16, a31) = butterfly(in_[1], in_[31], COSPI_31_64 as i16, COSPI_1_64 as i16);
    step1[16] = a16;
    step1[31] = a31;
    let (a17, a30) = butterfly(in_[17], in_[15], COSPI_15_64 as i16, COSPI_17_64 as i16);
    step1[17] = a17;
    step1[30] = a30;
    let (a18, a29) = butterfly(in_[9], in_[23], COSPI_23_64 as i16, COSPI_9_64 as i16);
    step1[18] = a18;
    step1[29] = a29;
    let (a19, a28) = butterfly(in_[25], in_[7], COSPI_7_64 as i16, COSPI_25_64 as i16);
    step1[19] = a19;
    step1[28] = a28;

    let (a20, a27) = butterfly(in_[5], in_[27], COSPI_27_64 as i16, COSPI_5_64 as i16);
    step1[20] = a20;
    step1[27] = a27;
    let (a21, a26) = butterfly(in_[21], in_[11], COSPI_11_64 as i16, COSPI_21_64 as i16);
    step1[21] = a21;
    step1[26] = a26;

    let (a22, a25) = butterfly(in_[13], in_[19], COSPI_19_64 as i16, COSPI_13_64 as i16);
    step1[22] = a22;
    step1[25] = a25;
    let (a23, a24) = butterfly(in_[29], in_[3], COSPI_3_64 as i16, COSPI_29_64 as i16);
    step1[23] = a23;
    step1[24] = a24;

    // stage 2
    step2[16] = _mm_add_epi16(step1[16], step1[17]);
    step2[17] = _mm_sub_epi16(step1[16], step1[17]);
    step2[18] = _mm_sub_epi16(step1[19], step1[18]);
    step2[19] = _mm_add_epi16(step1[19], step1[18]);
    step2[20] = _mm_add_epi16(step1[20], step1[21]);
    step2[21] = _mm_sub_epi16(step1[20], step1[21]);
    step2[22] = _mm_sub_epi16(step1[23], step1[22]);
    step2[23] = _mm_add_epi16(step1[23], step1[22]);

    step2[24] = _mm_add_epi16(step1[24], step1[25]);
    step2[25] = _mm_sub_epi16(step1[24], step1[25]);
    step2[26] = _mm_sub_epi16(step1[27], step1[26]);
    step2[27] = _mm_add_epi16(step1[27], step1[26]);
    step2[28] = _mm_add_epi16(step1[28], step1[29]);
    step2[29] = _mm_sub_epi16(step1[28], step1[29]);
    step2[30] = _mm_sub_epi16(step1[31], step1[30]);
    step2[31] = _mm_add_epi16(step1[31], step1[30]);

    // stage 3
    step1[16] = step2[16];
    step1[31] = step2[31];
    let (b17, b30) = butterfly(step2[30], step2[17], COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[17] = b17;
    step1[30] = b30;
    let (b18, b29) = butterfly(
        step2[29],
        step2[18],
        -(COSPI_4_64 as i16),
        COSPI_28_64 as i16,
    );
    step1[18] = b18;
    step1[29] = b29;
    step1[19] = step2[19];
    step1[20] = step2[20];
    let (b21, b26) = butterfly(step2[26], step2[21], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[21] = b21;
    step1[26] = b26;
    let (b22, b25) = butterfly(
        step2[25],
        step2[22],
        -(COSPI_20_64 as i16),
        COSPI_12_64 as i16,
    );
    step1[22] = b22;
    step1[25] = b25;
    step1[23] = step2[23];
    step1[24] = step2[24];
    step1[27] = step2[27];
    step1[28] = step2[28];

    idct32_8x32_quarter_3_4_stage_4_to_7(&mut step1, out);
}

/// `idct32_1024_8x32(in_, out)` — full 1-D 32-point inverse DCT,
/// processing 8 independent instances in parallel (one per i16 lane).
/// Mirrors libvpx `inv_txfm_sse2.c:1153-1161`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct32_1024_8x32(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let mut temp: [__m128i; 32] = [_mm_setzero_si128(); 32];

    idct32_1024_8x32_quarter_1_2(in_, &mut temp);
    idct32_1024_8x32_quarter_3_4(in_, &mut temp);
    // final stage — size-32 add/sub.
    let temp_copy = temp;
    add_sub_butterfly(&temp_copy, out, 32);
}

/// Test-only accessor for the private 1-D engine [`idct32_1024_8x32`],
/// used as the SSE2 oracle by the AVX2 sibling kernel's `idct32_1d_*`
/// equality proptest (`super::idct32x32_avx2_1024`). Does not alter any
/// kernel logic — a thin pass-through.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "sse2")]
pub unsafe fn idct32_1024_8x32_for_test(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    idct32_1024_8x32(in_, out)
}

// ---------------------------------------------------------------------
// Top-level kernel.
// ---------------------------------------------------------------------

/// Inner SSE2 implementation of the dense 32×32 inverse-DCT + residual
/// add. Two-pass `idct32_1024_8x32` (row pass + column pass) mirroring
/// libvpx `vpx_idct32x32_1024_add_sse2`.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2 (universal on x86_64).
/// - `dst` has at least `31 * stride + 32` bytes of valid storage.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct32x32_add_sse2_1024_inner(input: &[i32; 1024], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 31 + 32, "dst too small for 32x32 add");

    let zero = _mm_setzero_si128();
    // `col[strip][k]` — the row-pass output buffer. Per libvpx, the row
    // pass writes 4 strips of (32 cols × 32 rows-as-lanes) where each
    // strip processes 8 source rows. `col[strip]` holds 32 __m128i (one
    // per 32 output positions); each lane k corresponds to a source row
    // in the strip.
    let mut col: [[__m128i; 32]; 4] = [[zero; 32]; 4];
    let mut io: [__m128i; 32] = [zero; 32];

    // ---- Row pass: process 32 rows in 4 strips of 8 rows.
    //
    // For each strip: load + transpose 4× 8×8 i16 sub-blocks from the
    // 32-stride input (cols 0..8, 8..16, 16..24, 24..32). Then run the
    // 1-D 32-pt idct over all 32 register positions. The output is
    // saved into `col[strip][..]`.
    let mut input_ptr = input.as_ptr();
    for col_strip in &mut col {
        load_transpose_16bit_8x8(input_ptr, 32, &mut io[0..8]);
        load_transpose_16bit_8x8(input_ptr.add(8), 32, &mut io[8..16]);
        load_transpose_16bit_8x8(input_ptr.add(16), 32, &mut io[16..24]);
        load_transpose_16bit_8x8(input_ptr.add(24), 32, &mut io[24..32]);

        idct32_1024_8x32(&io, col_strip);

        // libvpx: `input += 32 << 3;` — advance to next 8-row strip.
        input_ptr = input_ptr.add(32 * 8);
    }

    // ---- Column pass: process 32 cols in 4 strips of 8 cols.
    //
    // For each strip i (i = 0, 8, 16, 24): transpose the 4 saved
    // sub-blocks (one row of 32 from col[0..3] each), 1-D idct32, then
    // store_buffer_8x32 (final +32 rounding, >>6, residual add).
    let mut dest_ptr = dst.as_mut_ptr();
    let mut i = 0usize;
    while i < 32 {
        // Transpose 32×8 block to 8×32 block — one 8×8 sub-block from
        // each of the 4 col[] strips at offset `i`.
        transpose_16bit_8x8(&col[0][i..i + 8], &mut io[0..8]);
        transpose_16bit_8x8(&col[1][i..i + 8], &mut io[8..16]);
        transpose_16bit_8x8(&col[2][i..i + 8], &mut io[16..24]);
        transpose_16bit_8x8(&col[3][i..i + 8], &mut io[24..32]);

        let io_copy = io;
        idct32_1024_8x32(&io_copy, &mut io);
        store_buffer_8x32(&mut io, dest_ptr, stride);

        dest_ptr = dest_ptr.add(8);
        i += 8;
    }
}

// ---------------------------------------------------------------------
// Higher-EOB fast-path kernels — i16-lane SSE2.
//
// Line-by-line ports of libvpx `vpx_idct32x32_34_add_sse2` and
// `vpx_idct32x32_135_add_sse2` (`inv_txfm_sse2.c:997-1217`) together
// with the EOB-34 quarter helpers `idct32_34_8x32_*` from
// `inv_txfm_sse2.c:890-994`. Same saturating-vs-modular semantics as
// the dense `_1024` kernel — see module docstring.
// ---------------------------------------------------------------------

/// `idct32_34_8x32_quarter_1(in_, out)` — even-input quarter of the
/// EOB-34 sparse 1-D 32-pt idct. Reads only `in_[0]` and `in_[4]`;
/// writes `out[0..8]`. Mirrors libvpx `inv_txfm_sse2.c:890-923`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_34_8x32_quarter_1(in_: &[__m128i; 32], out: &mut [__m128i]) {
    let zero = _mm_setzero_si128();
    let mut step1: [__m128i; 8] = [zero; 8];
    let mut step2: [__m128i; 8] = [zero; 8];

    // stage 3
    let (s4, s7) = butterfly(in_[4], zero, COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[4] = s4;
    step1[7] = s7;

    // stage 4
    step2[0] = butterfly_cospi16(in_[0]);
    step2[4] = step1[4];
    step2[5] = step1[4];
    step2[6] = step1[7];
    step2[7] = step1[7];

    // stage 5
    step1[0] = step2[0];
    step1[1] = step2[0];
    step1[2] = step2[0];
    step1[3] = step2[0];
    step1[4] = step2[4];
    let (s5, s6) = butterfly(step2[6], step2[5], COSPI_16_64 as i16, COSPI_16_64 as i16);
    step1[5] = s5;
    step1[6] = s6;
    step1[7] = step2[7];

    // stage 6
    out[0] = _mm_add_epi16(step1[0], step1[7]);
    out[1] = _mm_add_epi16(step1[1], step1[6]);
    out[2] = _mm_add_epi16(step1[2], step1[5]);
    out[3] = _mm_add_epi16(step1[3], step1[4]);
    out[4] = _mm_sub_epi16(step1[3], step1[4]);
    out[5] = _mm_sub_epi16(step1[2], step1[5]);
    out[6] = _mm_sub_epi16(step1[1], step1[6]);
    out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

/// `butterfly_cospi16(in)` — half-butterfly with cospi_16, reading only
/// the low 8 lanes (upper 8 zero). Mirrors libvpx
/// `inv_txfm_sse2.h:92-97`. Local copy avoids re-exporting from the
/// 16x16 SSE2 module.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn butterfly_cospi16(in_: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let cst = pair_set_epi16(COSPI_16_64 as i16, COSPI_16_64 as i16);
    let lo = _mm_unpacklo_epi16(in_, zero);
    let hi = _mm_unpackhi_epi16(in_, zero);
    idct_calc_wraplow_sse2(lo, hi, cst)
}

/// `idct32_34_8x32_quarter_2(in_, out)` — quarter-2 of the EOB-34
/// sparse 1-D. Reads `in_[2]` and `in_[6]`; updates `out[8..16]`.
/// Mirrors libvpx `inv_txfm_sse2.c:928-948`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_34_8x32_quarter_2(in_: &[__m128i; 32], out: &mut [__m128i]) {
    let zero = _mm_setzero_si128();
    let mut step1: [__m128i; 16] = [zero; 16];

    // stage 2
    let (s8, s15) = butterfly(in_[2], zero, COSPI_30_64 as i16, COSPI_2_64 as i16);
    let (s11, s12) = butterfly(zero, in_[6], COSPI_6_64 as i16, COSPI_26_64 as i16);

    // stage 3
    step1[8] = s8;
    step1[9] = s8;
    step1[14] = s15;
    step1[15] = s15;
    step1[10] = s11;
    step1[11] = s11;
    step1[12] = s12;
    step1[13] = s12;

    idct32_8x32_quarter_2_stage_4_to_6(&mut step1, out);
}

/// `idct32_34_8x32_quarter_1_2(in_, out)` — combine quarters 1+2, then
/// size-16 add/sub butterfly. Mirrors libvpx `inv_txfm_sse2.c:950-957`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_34_8x32_quarter_1_2(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let mut temp: [__m128i; 16] = [_mm_setzero_si128(); 16];
    idct32_34_8x32_quarter_1(in_, &mut temp);
    idct32_34_8x32_quarter_2(in_, &mut temp);
    add_sub_butterfly(&temp, &mut out[..16], 16);
}

/// `idct32_34_8x32_quarter_3_4(in_, out)` — odd-index quarter of the
/// EOB-34 sparse 1-D. Reads `in_[1, 3, 5, 7]`; writes `out[16..32]`.
/// Mirrors libvpx `inv_txfm_sse2.c:962-984`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct32_34_8x32_quarter_3_4(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let zero = _mm_setzero_si128();
    let mut step1: [__m128i; 32] = [zero; 32];

    // stage 1
    let (a16, a31) = butterfly(in_[1], zero, COSPI_31_64 as i16, COSPI_1_64 as i16);
    step1[16] = a16;
    step1[31] = a31;
    let (a19, a28) = butterfly(zero, in_[7], COSPI_7_64 as i16, COSPI_25_64 as i16);
    step1[19] = a19;
    step1[28] = a28;
    let (a20, a27) = butterfly(in_[5], zero, COSPI_27_64 as i16, COSPI_5_64 as i16);
    step1[20] = a20;
    step1[27] = a27;
    let (a23, a24) = butterfly(zero, in_[3], COSPI_3_64 as i16, COSPI_29_64 as i16);
    step1[23] = a23;
    step1[24] = a24;

    // stage 3 (libvpx skips stage 2 — sparse inputs make step2 == step1)
    let (b17, b30) = butterfly(step1[31], step1[16], COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[17] = b17;
    step1[30] = b30;
    let (b18, b29) = butterfly(
        step1[28],
        step1[19],
        -(COSPI_4_64 as i16),
        COSPI_28_64 as i16,
    );
    step1[18] = b18;
    step1[29] = b29;
    let (b21, b26) = butterfly(step1[27], step1[20], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[21] = b21;
    step1[26] = b26;
    let (b22, b25) = butterfly(
        step1[24],
        step1[23],
        -(COSPI_20_64 as i16),
        COSPI_12_64 as i16,
    );
    step1[22] = b22;
    step1[25] = b25;

    idct32_8x32_quarter_3_4_stage_4_to_7(&mut step1, out);
}

/// `idct32_34_8x32(in_, out)` — full 1-D 32-pt IDCT for the EOB-34
/// sparse contract. Mirrors libvpx `inv_txfm_sse2.c:986-994`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct32_34_8x32(in_: &[__m128i; 32], out: &mut [__m128i; 32]) {
    let mut temp: [__m128i; 32] = [_mm_setzero_si128(); 32];

    idct32_34_8x32_quarter_1_2(in_, &mut temp);
    idct32_34_8x32_quarter_3_4(in_, &mut temp);
    let temp_copy = temp;
    add_sub_butterfly(&temp_copy, out, 32);
}

/// Inner SSE2 implementation of the EOB ≤ 34 32×32 IDCT + residual
/// add. Mirrors libvpx `vpx_idct32x32_34_add_sse2`
/// (`inv_txfm_sse2.c:997-1017`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `dst` has at least `31 * stride + 32` bytes of valid storage.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct32x32_add_sse2_34_inner(input: &[i32; 1024], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 31 + 32, "dst too small for 32x32 add");

    let zero = _mm_setzero_si128();
    let mut io: [__m128i; 32] = [zero; 32];
    let mut col: [__m128i; 32] = [zero; 32];

    // Row pass: load only the top-left 8×8 (rows 0..8, cols 0..8) since
    // EOB ≤ 34 confines non-zero coefs to that sub-block. Positions
    // 8..32 of `io` stay zero from the initialiser.
    load_transpose_16bit_8x8(input.as_ptr(), 32, &mut io[..8]);
    idct32_34_8x32(&io, &mut col);

    // Column pass: 4 strips of 8 cols × 32 rows.
    let mut dest_ptr = dst.as_mut_ptr();
    let mut i = 0usize;
    while i < 32 {
        transpose_16bit_8x8(&col[i..i + 8], &mut io[..8]);
        // libvpx re-zeros nothing here — the prior column-pass iteration
        // leaves io[8..32] holding `idct32_34_8x32` outputs; but the
        // next call to `idct32_34_8x32` only reads in_[0, 2, 4, 6, 1, 3,
        // 5, 7] (sparse-quarter contract). So we need to zero io[8..32]
        // to match libvpx's `__m128i io[32]` initial-zero contract.
        //
        // Actually libvpx's source reuses `io` without re-zeroing —
        // see the loop body at `inv_txfm_sse2.c:1006-1013`. The trick is
        // that `idct32_34_8x32_sse2` only **reads** in_[0..8]; positions
        // 8..32 are never read. So we can match libvpx exactly: only
        // overwrite io[0..8] from the transpose, leave io[8..32] as-is.
        let io_copy = io;
        idct32_34_8x32(&io_copy, &mut io);
        store_buffer_8x32(&mut io, dest_ptr, stride);

        dest_ptr = dest_ptr.add(8);
        i += 8;
    }
}

/// Safe wrapper for the SSE2 i16-lane EOB ≤ 34 32×32 IDCT.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_sse2_34(coeffs: &[i32; 1024], dst: &mut [u8], stride: usize) {
    crate::dsp::validate_predictor_storage(dst.len(), stride, 32, 32)
        .expect("invalid inverse transform destination");
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is universal on x86_64.
    unsafe { idct32x32_add_sse2_34_inner(coeffs, dst, stride) }
}

/// Inner SSE2 implementation of the EOB ≤ 135 32×32 IDCT + residual
/// add. Mirrors libvpx `vpx_idct32x32_135_add_sse2`
/// (`inv_txfm_sse2.c:1192-1217`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `dst` has at least `31 * stride + 32` bytes of valid storage.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct32x32_add_sse2_135_inner(input: &[i32; 1024], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 31 + 32, "dst too small for 32x32 add");

    let zero = _mm_setzero_si128();
    let mut in_: [__m128i; 32] = [zero; 32];
    let mut out_: [__m128i; 32] = [zero; 32];
    let mut col: [[__m128i; 32]; 2] = [[zero; 32]; 2];

    // libvpx zeroes in_[16..32] explicitly — already zero from init.

    // Row pass: 2 strips × (8 rows × 16 cols / 2 sub-blocks). EOB ≤ 135
    // confines coefs to top-left 16x16, so rows 16..32 and cols 16..32
    // are all zero. We load 2× 8x8 sub-blocks per strip (cols 0..16).
    let mut input_ptr = input.as_ptr();
    for col_strip in &mut col {
        load_transpose_16bit_8x8(input_ptr, 32, &mut in_[..8]);
        load_transpose_16bit_8x8(input_ptr.add(8), 32, &mut in_[8..16]);
        idct32_1024_8x32(&in_, col_strip);
        input_ptr = input_ptr.add(32 * 8);
    }

    // Column pass: 4 strips of 8 cols × 32 rows.
    let mut dest_ptr = dst.as_mut_ptr();
    let mut i = 0usize;
    while i < 32 {
        transpose_16bit_8x8(&col[0][i..i + 8], &mut in_[..8]);
        transpose_16bit_8x8(&col[1][i..i + 8], &mut in_[8..16]);
        // in_[16..32] still zero (initialiser; the row-pass loop didn't
        // touch them since we only filled in_[..16]).
        idct32_1024_8x32(&in_, &mut out_);
        store_buffer_8x32(&mut out_, dest_ptr, stride);

        dest_ptr = dest_ptr.add(8);
        i += 8;
    }
}

/// Safe wrapper for the SSE2 i16-lane EOB ≤ 135 32×32 IDCT.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_sse2_135(coeffs: &[i32; 1024], dst: &mut [u8], stride: usize) {
    crate::dsp::validate_predictor_storage(dst.len(), stride, 32, 32)
        .expect("invalid inverse transform destination");
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is universal on x86_64.
    unsafe { idct32x32_add_sse2_135_inner(coeffs, dst, stride) }
}

/// Safe wrapper for the SSE2 i16-lane dense `idct32x32_1024_add` kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct32x32_1024_add`] and the existing
/// AVX2 wrapper [`super::idct32x32_add_avx2`], so the
/// `SimdDispatch::idct32x32_add` slot can hold any of them.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly for SSE2-vs-scalar differential
/// testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_sse2_1024(coeffs: &[i32; 1024], dst: &mut [u8], stride: usize) {
    crate::dsp::validate_predictor_storage(dst.len(), stride, 32, 32)
        .expect("invalid inverse transform destination");
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is architecturally universal on
    // x86_64 (mandated by both SysV AMD64 and MS x64 ABIs), so the
    // gate is essentially free. The unit-test path gates each call on
    // a runtime `is_x86_feature_detected!("sse2")` check.
    unsafe { idct32x32_add_sse2_1024_inner(coeffs, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Outer-kernel differential proptest for `idct32x32_add_sse2_1024`
    //! against the scalar reference, plus a small set of directed
    //! fixtures.
    //!
    //! Strategy envelopes are tighter than for 16×16 because the
    //! 32-point 1-D IDCT has more accumulation depth (8 stages with up
    //! to 32-way sums), so even moderate full-density i12 inputs can
    //! push intermediates outside i16 and trip the SSE2 saturating
    //! `_mm_packs_epi32`. We use:
    //!
    //! - **sparse i12** — up to 16 non-zero coefs; trivially safe
    //!   because few terms accumulate.
    //! - **dense i8** — full 1024 positions at ≤ ±127; analysis of the
    //!   libvpx flowgraph gives a row-pass output bound ≈
    //!   `32 * i8 ≈ ±4096 < i16::MAX`, and the column pass starts from
    //!   a bounded i16 (after the row-pass saturating pack) so the
    //!   final ranges also stay inside i16.
    //! - **EOB-34 / EOB-135 at i12** — first K raster positions only;
    //!   the empty tail means total accumulation stays below the
    //!   saturation boundary in practice.
    //!
    //! The pathological "i16 extremes" strategy from
    //! `idct32x32_avx2::tests` is **deliberately omitted** because it
    //! trips the saturation divergence (see module docstring). Real-
    //! world bit-exactness is covered by the 201-vector conformance
    //! corpus, not the synthetic proptests.
    //!
    //! Sub-tests:
    //! 1. Bit-exact directed fixtures (all-zero, DC-only, small EOB).
    //! 2. Outer differential vs `idct32x32_1024_add` scalar over 4
    //!    strategies × 32 cases each (modest case count: each call
    //!    runs a full 32×32 inverse DCT, much heavier than 16×16).
    //! 3. Misaligned dst pointer regression guard.

    use super::*;
    use crate::transforms::inverse::idct32x32_1024_add;
    use proptest::prelude::*;

    macro_rules! require_sse2 {
        () => {
            if !is_x86_feature_detected!("sse2") {
                return;
            }
        };
    }

    fn run_sse2(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        assert!(is_x86_feature_detected!("sse2"));
        let mut dst = *dst_init;
        idct32x32_add_sse2_1024(input, &mut dst, 32);
        dst
    }

    fn run_scalar_oracle(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        let mut dst = *dst_init;
        idct32x32_1024_add(input, &mut dst, 32);
        dst
    }

    // -----------------------------------------------------------------
    // Sub-task 1: directed fixtures (bit-exact vs scalar reference).
    // -----------------------------------------------------------------

    #[test]
    fn idct32x32_sse2_all_zero_input() {
        require_sse2!();
        let input = [0i32; 1024];
        let dst_init = [42u8; 1024];
        let sse2 = run_sse2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(sse2, scalar);
        // All-zero input ⇒ rounding (+32 >> 6) == 0, so dst unchanged.
        assert_eq!(sse2, dst_init);
    }

    /// DC-only impulse — only `input[0]` non-zero.
    #[test]
    fn idct32x32_sse2_dc_impulse() {
        require_sse2!();
        for &val in &[100i32, -100, 2000, -2000, 1, -1] {
            let mut input = [0i32; 1024];
            input[0] = val;
            let dst_init = [128u8; 1024];
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            assert_eq!(sse2, scalar, "val={val}");
        }
    }

    /// Small EOB region — first 34 raster positions non-zero (covers
    /// idct32x32_34 fast-path density).
    #[test]
    fn idct32x32_sse2_eob_34() {
        require_sse2!();
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(34).enumerate() {
            *slot = ((i as i32) * 37) - 600;
        }
        let dst_init = [64u8; 1024];
        let sse2 = run_sse2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(sse2, scalar);
    }

    /// Medium EOB region — first 135 raster positions non-zero
    /// (covers idct32x32_135 fast-path density).
    #[test]
    fn idct32x32_sse2_eob_135() {
        require_sse2!();
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(135).enumerate() {
            *slot = ((i as i32) * 19) - 1200;
        }
        let dst_init = [80u8; 1024];
        let sse2 = run_sse2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(sse2, scalar);
    }

    // -----------------------------------------------------------------
    // Sub-task 2: outer differential against the scalar reference.
    // -----------------------------------------------------------------

    /// `dense_8` — full-density i8 coefficients in every position.
    ///
    /// The dense envelope is tighter than for 16×16 (i12) because the
    /// 32-point 1-D IDCT has 8 stages and up to 32-way summation; even
    /// moderate full-density i12 inputs push intermediates outside
    /// i16, tripping the saturating `_mm_packs_epi32`. i8 keeps a
    /// comfortable safety margin (`32 * 127 ≈ ±4096 < i16::MAX`).
    fn dense_strategy_8() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-128i32..=127, 1024).prop_map(|v| {
            let mut out = [0i32; 1024];
            out.copy_from_slice(&v);
            out
        })
    }

    /// `sparse_12` — 1..=16 non-zero coefs at arbitrary raster positions.
    fn sparse_strategy_12() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec((0usize..1024, -2048i32..=2047), 1..=16).prop_map(|nz| {
            let mut out = [0i32; 1024];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// `eob_bucket_34` — exactly 34 non-zero coefs at the first 34
    /// raster positions; mirrors libvpx's idct32x32_34_add dispatcher
    /// boundary.
    fn eob_bucket_34_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-2048i32..=2047, 34).prop_map(|values| {
            let mut out = [0i32; 1024];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
    }

    /// `eob_bucket_135` — exactly 135 non-zero coefs at the first 135
    /// raster positions, bounded to i10 magnitude; mirrors libvpx's
    /// idct32x32_135_add dispatcher boundary. The i10 envelope (vs.
    /// i12 elsewhere) keeps row-pass accumulation comfortably inside
    /// i16 across all 135 positions.
    fn eob_bucket_135_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-512i32..=511, 135).prop_map(|values| {
            let mut out = [0i32; 1024];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
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
            cases: 32, max_shrink_iters: 256, .. ProptestConfig::default()
        })]

        #[test]
        fn idct32x32_sse2_eq_scalar_dense_8(
            input in dense_strategy_8(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }

        #[test]
        fn idct32x32_sse2_eq_scalar_sparse_12(
            input in sparse_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }

        #[test]
        fn idct32x32_sse2_eq_scalar_eob_34(
            input in eob_bucket_34_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }

        #[test]
        fn idct32x32_sse2_eq_scalar_eob_135(
            input in eob_bucket_135_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, scalar);
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 3: misaligned dst pointer regression guard.
    // -----------------------------------------------------------------

    /// The kernel uses `_mm_loadl_epi64` / `_mm_storel_epi64` for `dst`
    /// I/O (8-byte, alignment-free), so this is a per-offset arithmetic
    /// regression guard, NOT an alignment-fault test.
    #[test]
    fn idct32x32_sse2_misaligned_dst_pointers() {
        require_sse2!();
        // Synthetic input — moderate-magnitude across the eob-bucket
        // region (first 135 raster positions). Bounded to a small
        // magnitude (-200..+400) so row-pass accumulation stays inside
        // i16 across all 135 positions — saturating-pack divergence
        // is documented at the module level but unwanted here.
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(135).enumerate() {
            *slot = ((i as i32) * 5) - 200;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 1040]> = Box::new([0u8; 1040]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;

            idct32x32_add_sse2_1024(&input, &mut buf[skew..skew + 1024], 32);
            idct32x32_1024_add(&input, &mut scalar_buf[skew..skew + 1024], 32);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "SSE2/scalar mismatch at skew={skew}: per-offset arithmetic regression"
            );
        }
    }
}
