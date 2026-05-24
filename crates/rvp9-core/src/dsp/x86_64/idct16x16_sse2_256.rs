//! 16×16 inverse-DCT + add — x86_64 SSE2 i16-lane implementation
//! covering the dense path (the `idct16x16_256_add` slot when AVX2 is
//! unavailable).
//!
//! This kernel is a line-by-line port of libvpx's
//! `vpx_idct16x16_256_add_sse2` (`third_party/libvpx/vpx_dsp/x86/inv_txfm_sse2.c:403-431`)
//! along with the inner 1-D core `idct16_8col` from
//! `inv_txfm_sse2.h:343-427`. It is the second SSE2-only kernel in this
//! codebase (after `idct8x8_12_sse2`) and the first dense-path SSE2 i16
//! kernel — see Track-T HLD `wrk_docs/Track_T/T4_i16_lane_IDCT_HLD.md`
//! for the porting plan.
//!
//! ## Saturating-vs-modular semantic divergence
//!
//! Like the [`super::idct8x8_12_sse2`] kernel, this implementation uses
//! the dense `_mm_madd_epi16` pipeline with `_mm_packs_epi32`
//! (saturating, clamps to `±i16::MAX`) at every wraplow site rather
//! than the modular `wraplow` (sign-extend low 16 bits) used by the
//! i32-lane AVX2 kernels and the scalar reference. The two policies
//! agree on every in-spec bitstream coefficient (and on the empirical
//! 201-vector conformance corpus), but diverge on synthetic adversarial
//! inputs whose intermediate magnitudes exceed `i16::MAX`.
//!
//! Per Task A precedent, the differential proptest at the bottom of
//! this file is bounded to ≤ i12 magnitudes (`[-2048, 2047]`); the
//! AVX2-style "pathological" strategies (i16-extreme alternating-sign
//! patterns) are **deliberately omitted** because they trip the
//! saturation divergence and would fail spuriously. See HLD §6 for the
//! validation strategy.
//!
//! ## Convention compatibility
//!
//! 1. The body uses `#[target_feature(enable = "sse2")]` on
//!    `unsafe fn`. SSE2 is architecturally universal on x86_64 (per
//!    SysV AMD64 / MS x64 ABIs), but tests still gate on
//!    `is_x86_feature_detected!("sse2")` belt-and-braces.
//! 2. A safe wrapper [`idct16x16_add_sse2_256`] performs a single
//!    `unsafe { ... }` call into the inner. The SAFETY comment names
//!    the OnceLock + caps invariant — see [`crate::dsp::dispatch`].
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadl_epi64` (8-byte, unaligned) via
//! `write_buffer_8x1` (one 8-byte residual add per row × 16 rows × 2
//! strips). The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8,
};

use crate::transforms::constants::{
    COSPI_2_64, COSPI_4_64, COSPI_6_64, COSPI_8_64, COSPI_10_64, COSPI_12_64, COSPI_14_64,
    COSPI_16_64, COSPI_18_64, COSPI_20_64, COSPI_22_64, COSPI_24_64, COSPI_26_64, COSPI_28_64,
    COSPI_30_64,
};

#[cfg(target_arch = "x86_64")]
use super::idct8x8_12_sse2::butterfly;

// ---------------------------------------------------------------------
// New helpers (line-by-line ports from libvpx).
// ---------------------------------------------------------------------

/// `load_input_data8(data)` — load 8 i32 coefficients and saturating-pack
/// to 8 i16 lanes.
///
/// Our coefficients are stored as i32 throughout the dequant pipeline
/// (matching libvpx's `CONFIG_VP9_HIGHBITDEPTH` build path, even though
/// we are 8-bit). For spec-bounded bitstream inputs (i16 range per VP9
/// §8.7.1) the saturating-pack never clips.
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
/// Three-stage unpack pyramid (16/32/64 bit). Line-by-line port of
/// libvpx `transpose_sse2.h:157-220`. Pure shuffle, no arithmetic, so
/// bit-exact for arbitrary i16 lane values.
///
/// Lane mapping (per libvpx comments):
///
/// ```text
///   in[0]: 00 01 02 03  04 05 06 07
///   in[1]: 10 11 12 13  14 15 16 17
///   ...
///   in[7]: 70 71 72 73  74 75 76 77
///   ->
///   out[0]: 00 10 20 30  40 50 60 70
///   out[1]: 01 11 21 31  41 51 61 71
///   ...
///   out[7]: 07 17 27 37  47 57 67 77
/// ```
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2 (universal on
/// x86_64; gated at dispatch-build time via `caps.sse2`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn transpose_16bit_8x8(in_: &[__m128i; 8], out: &mut [__m128i; 8]) {
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

/// `idct16_8col(in, out)` — 1-D 16-point inverse DCT, processing 8
/// independent instances in parallel (one per i16 lane).
///
/// Line-by-line port of libvpx `inv_txfm_sse2.h:343-427`. Built from
/// `butterfly` calls (reused from Task A) plus `_mm_add_epi16` /
/// `_mm_sub_epi16` sum/difference combinations.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports SSE2 (universal on
/// x86_64; gated at dispatch-build time via `caps.sse2`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct16_8col(in_: &[__m128i; 16], out: &mut [__m128i; 16]) {
    let mut step1: [__m128i; 16] = [_mm_setzero_si128(); 16];
    let mut step2: [__m128i; 16] = [_mm_setzero_si128(); 16];

    // stage 2
    let (s2_8, s2_15) = butterfly(in_[1], in_[15], COSPI_30_64 as i16, COSPI_2_64 as i16);
    let (s2_9, s2_14) = butterfly(in_[9], in_[7], COSPI_14_64 as i16, COSPI_18_64 as i16);
    let (s2_10, s2_13) = butterfly(in_[5], in_[11], COSPI_22_64 as i16, COSPI_10_64 as i16);
    let (s2_11, s2_12) = butterfly(in_[13], in_[3], COSPI_6_64 as i16, COSPI_26_64 as i16);
    step2[8] = s2_8;
    step2[9] = s2_9;
    step2[10] = s2_10;
    step2[11] = s2_11;
    step2[12] = s2_12;
    step2[13] = s2_13;
    step2[14] = s2_14;
    step2[15] = s2_15;

    // stage 3
    let (s1_4, s1_7) = butterfly(in_[2], in_[14], COSPI_28_64 as i16, COSPI_4_64 as i16);
    let (s1_5, s1_6) = butterfly(in_[10], in_[6], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[4] = s1_4;
    step1[5] = s1_5;
    step1[6] = s1_6;
    step1[7] = s1_7;
    step1[8] = _mm_add_epi16(step2[8], step2[9]);
    step1[9] = _mm_sub_epi16(step2[8], step2[9]);
    step1[10] = _mm_sub_epi16(step2[11], step2[10]);
    step1[11] = _mm_add_epi16(step2[10], step2[11]);
    step1[12] = _mm_add_epi16(step2[12], step2[13]);
    step1[13] = _mm_sub_epi16(step2[12], step2[13]);
    step1[14] = _mm_sub_epi16(step2[15], step2[14]);
    step1[15] = _mm_add_epi16(step2[14], step2[15]);

    // stage 4 — note libvpx's butterfly call signature `&step2[1], &step2[0]`
    // (output order flipped) for the cospi_16,cospi_16 pair; same flip for
    // the `step1[10], step1[13]` butterfly with -cospi_8, -cospi_24.
    let (s2_1, s2_0) = butterfly(in_[0], in_[8], COSPI_16_64 as i16, COSPI_16_64 as i16);
    let (s2_2, s2_3) = butterfly(in_[4], in_[12], COSPI_24_64 as i16, COSPI_8_64 as i16);
    let (s2_9_new, s2_14_new) = butterfly(
        step1[14],
        step1[9],
        COSPI_24_64 as i16,
        COSPI_8_64 as i16,
    );
    let (s2_13_new, s2_10_new) = butterfly(
        step1[10],
        step1[13],
        -(COSPI_8_64 as i16),
        -(COSPI_24_64 as i16),
    );
    step2[0] = s2_0;
    step2[1] = s2_1;
    step2[2] = s2_2;
    step2[3] = s2_3;
    step2[5] = _mm_sub_epi16(step1[4], step1[5]);
    step1[4] = _mm_add_epi16(step1[4], step1[5]);
    step2[6] = _mm_sub_epi16(step1[7], step1[6]);
    step1[7] = _mm_add_epi16(step1[6], step1[7]);
    step2[8] = step1[8];
    step2[9] = s2_9_new;
    step2[10] = s2_10_new;
    step2[11] = step1[11];
    step2[12] = step1[12];
    step2[13] = s2_13_new;
    step2[14] = s2_14_new;
    step2[15] = step1[15];

    // stage 5
    step1[0] = _mm_add_epi16(step2[0], step2[3]);
    step1[1] = _mm_add_epi16(step2[1], step2[2]);
    step1[2] = _mm_sub_epi16(step2[1], step2[2]);
    step1[3] = _mm_sub_epi16(step2[0], step2[3]);
    let (s1_5_new, s1_6_new) = butterfly(
        step2[6],
        step2[5],
        COSPI_16_64 as i16,
        COSPI_16_64 as i16,
    );
    step1[5] = s1_5_new;
    step1[6] = s1_6_new;
    step1[8] = _mm_add_epi16(step2[8], step2[11]);
    step1[9] = _mm_add_epi16(step2[9], step2[10]);
    step1[10] = _mm_sub_epi16(step2[9], step2[10]);
    step1[11] = _mm_sub_epi16(step2[8], step2[11]);
    step1[12] = _mm_sub_epi16(step2[15], step2[12]);
    step1[13] = _mm_sub_epi16(step2[14], step2[13]);
    step1[14] = _mm_add_epi16(step2[14], step2[13]);
    step1[15] = _mm_add_epi16(step2[15], step2[12]);

    // stage 6
    step2[0] = _mm_add_epi16(step1[0], step1[7]);
    step2[1] = _mm_add_epi16(step1[1], step1[6]);
    step2[2] = _mm_add_epi16(step1[2], step1[5]);
    step2[3] = _mm_add_epi16(step1[3], step1[4]);
    step2[4] = _mm_sub_epi16(step1[3], step1[4]);
    step2[5] = _mm_sub_epi16(step1[2], step1[5]);
    step2[6] = _mm_sub_epi16(step1[1], step1[6]);
    step2[7] = _mm_sub_epi16(step1[0], step1[7]);
    let (s2_10_b, s2_13_b) = butterfly(
        step1[13],
        step1[10],
        COSPI_16_64 as i16,
        COSPI_16_64 as i16,
    );
    let (s2_11_b, s2_12_b) = butterfly(
        step1[12],
        step1[11],
        COSPI_16_64 as i16,
        COSPI_16_64 as i16,
    );
    step2[10] = s2_10_b;
    step2[11] = s2_11_b;
    step2[12] = s2_12_b;
    step2[13] = s2_13_b;

    // stage 7
    out[0] = _mm_add_epi16(step2[0], step1[15]);
    out[1] = _mm_add_epi16(step2[1], step1[14]);
    out[2] = _mm_add_epi16(step2[2], step2[13]);
    out[3] = _mm_add_epi16(step2[3], step2[12]);
    out[4] = _mm_add_epi16(step2[4], step2[11]);
    out[5] = _mm_add_epi16(step2[5], step2[10]);
    out[6] = _mm_add_epi16(step2[6], step1[9]);
    out[7] = _mm_add_epi16(step2[7], step1[8]);
    out[8] = _mm_sub_epi16(step2[7], step1[8]);
    out[9] = _mm_sub_epi16(step2[6], step1[9]);
    out[10] = _mm_sub_epi16(step2[5], step2[10]);
    out[11] = _mm_sub_epi16(step2[4], step2[11]);
    out[12] = _mm_sub_epi16(step2[3], step2[12]);
    out[13] = _mm_sub_epi16(step2[2], step2[13]);
    out[14] = _mm_sub_epi16(step2[1], step1[14]);
    out[15] = _mm_sub_epi16(step2[0], step1[15]);
}

/// `idct16_load8x8(input, in)` — load 8 rows × 8 i16 lanes from the
/// i32 coefficient buffer.
///
/// Mirrors libvpx `inv_txfm_sse2.c:391-401`. The 8 rows are read at
/// stride 16 (the 16×16 input layout); each `load_input_data8` packs
/// 8 i32 → 8 i16.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2.
/// - `input..input+7*16+8` is a valid, readable region of i32.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn idct16_load8x8(input: *const i32, in_: &mut [__m128i]) {
    // Row stride = 16 (the 16×16 input layout). The `N * 16` pattern
    // mirrors libvpx `inv_txfm_sse2.c:393-400` verbatim.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    {
        in_[0] = load_input_data8(input.add(0 * 16));
        in_[1] = load_input_data8(input.add(1 * 16));
        in_[2] = load_input_data8(input.add(2 * 16));
        in_[3] = load_input_data8(input.add(3 * 16));
        in_[4] = load_input_data8(input.add(4 * 16));
        in_[5] = load_input_data8(input.add(5 * 16));
        in_[6] = load_input_data8(input.add(6 * 16));
        in_[7] = load_input_data8(input.add(7 * 16));
    }
}

/// `write_buffer_8x1(dest, in)` — final-rounding (+32, >>6) + 8-byte
/// residual add for one row of the 16-col output strip.
///
/// Mirrors libvpx `inv_txfm_sse2.h:229-235` + the `recon_and_store`
/// helper at lines 135-142. Uses `_mm_adds_epi16` (saturating add) for
/// the +32 rounding — matches libvpx semantics, and saturation here is
/// benign because the rounding constant is tiny.
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
unsafe fn write_buffer_8x1(dest: *mut u8, in_x: __m128i) {
    let final_rounding = _mm_set1_epi16(1 << 5);
    let mut out = _mm_adds_epi16(in_x, final_rounding);
    out = _mm_srai_epi16(out, 6);

    let zero = _mm_setzero_si128();
    let d_u8 = _mm_loadl_epi64(dest as *const __m128i);
    let d_i16 = _mm_unpacklo_epi8(d_u8, zero);
    let sum_i16 = _mm_add_epi16(out, d_i16);
    let sum_u8 = _mm_packus_epi16(sum_i16, sum_i16);
    _mm_storel_epi64(dest as *mut __m128i, sum_u8);
}

/// Inner SSE2 implementation of the dense 16×16 inverse-DCT + residual
/// add. Two-pass `idct16_8col` (row pass + column pass) mirroring
/// libvpx `vpx_idct16x16_256_add_sse2`.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports SSE2 (universal on x86_64).
/// - `dst` has at least `15 * stride + 16` bytes of valid storage.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn idct16x16_add_sse2_256_inner(input: &[i32; 256], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 15 + 16, "dst too small for 16x16 add");

    let zero = _mm_setzero_si128();
    let mut l: [__m128i; 16] = [zero; 16];
    let mut r: [__m128i; 16] = [zero; 16];

    // ---- Row pass: process 16 rows in two 8-row strips (left + right).
    //
    // For each strip, the 16 input columns are loaded as two 8-col
    // halves (idct16_load8x8 on `input` and `input + 8`), each
    // transposed 8×8 in place, then a single `idct16_8col` runs over
    // the full 16-row register file (rows 0..7 = left half, rows 8..15
    // = right half — the lane positions correspond to the 8 cols of
    // the active 8-row strip).
    let mut input_ptr = input.as_ptr();
    for strip in 0..2 {
        let buf: &mut [__m128i; 16] = if strip == 0 { &mut l } else { &mut r };

        // Load left 8 cols (rows 0..7 of input strip) → buf[0..8].
        idct16_load8x8(input_ptr, &mut buf[..8]);
        // Transpose in place: write back into buf[0..8].
        {
            let src: [__m128i; 8] = [
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ];
            let mut tmp: [__m128i; 8] = [zero; 8];
            transpose_16bit_8x8(&src, &mut tmp);
            buf[..8].copy_from_slice(&tmp);
        }

        // Load right 8 cols (cols 8..15 of the same rows) → buf[8..16].
        idct16_load8x8(input_ptr.add(8), &mut buf[8..]);
        {
            let src: [__m128i; 8] = [
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ];
            let mut tmp: [__m128i; 8] = [zero; 8];
            transpose_16bit_8x8(&src, &mut tmp);
            buf[8..].copy_from_slice(&tmp);
        }

        // 1-D row pass — operates on the full 16-register set.
        let in_copy = *buf;
        idct16_8col(&in_copy, buf);

        input_ptr = input_ptr.add(128); // advance to next 8-row strip
    }

    // ---- Column pass: 2 strips of 8 cols, 16 rows each.
    //
    // For each strip, take the 8 rows of `l` and the 8 rows of `r` at
    // positions [strip*8 .. strip*8 + 8], transpose each 8×8 block,
    // then run idct16_8col to produce the final 16 output rows. Each
    // output row is then written via `write_buffer_8x1` (8 bytes per
    // row × 16 rows = 128 bytes per strip).
    let mut dest_ptr = dst.as_mut_ptr();
    for strip in 0..2 {
        let row_off = strip * 8;
        let mut out: [__m128i; 16] = [zero; 16];

        // Top half — first 8 rows of l[strip*8 .. strip*8 + 8] → out[0..8].
        let l_src: [__m128i; 8] = [
            l[row_off],
            l[row_off + 1],
            l[row_off + 2],
            l[row_off + 3],
            l[row_off + 4],
            l[row_off + 5],
            l[row_off + 6],
            l[row_off + 7],
        ];
        let mut tmp: [__m128i; 8] = [zero; 8];
        transpose_16bit_8x8(&l_src, &mut tmp);
        out[..8].copy_from_slice(&tmp);

        // Bottom half — first 8 rows of r[strip*8 .. strip*8 + 8] → out[8..16].
        let r_src: [__m128i; 8] = [
            r[row_off],
            r[row_off + 1],
            r[row_off + 2],
            r[row_off + 3],
            r[row_off + 4],
            r[row_off + 5],
            r[row_off + 6],
            r[row_off + 7],
        ];
        transpose_16bit_8x8(&r_src, &mut tmp);
        out[8..].copy_from_slice(&tmp);

        // 1-D column pass.
        let in_copy = out;
        idct16_8col(&in_copy, &mut out);

        // Write 16 output rows × 8 cols per strip.
        for (j, row) in out.iter().enumerate() {
            write_buffer_8x1(dest_ptr.add(j * stride), *row);
        }

        dest_ptr = dest_ptr.add(8);
    }
}

/// Safe wrapper for the SSE2 i16-lane dense `idct16x16_256_add` kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct16x16_256_add`] and the existing
/// AVX2 wrapper [`super::idct16x16_avx2::idct16x16_add_avx2`], so the
/// `SimdDispatch::idct16x16_add` slot can hold any of them.
///
/// The wrapper is `pub` (rather than `pub(crate)`) so unit tests in
/// the same crate can call it directly for SSE2-vs-AVX2 differential
/// testing without going through the dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct16x16_add_sse2_256(input: &[i32; 256], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.sse2 == true`. SSE2 is architecturally universal on
    // x86_64 (mandated by both SysV AMD64 and MS x64 ABIs), so the
    // gate is essentially free. The unit-test path gates each call on
    // a runtime `is_x86_feature_detected!("sse2")` check.
    unsafe { idct16x16_add_sse2_256_inner(input, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Per-helper unit tests + outer-kernel differential proptest for
    //! `idct16x16_add_sse2_256`.
    //!
    //! All differential proptest strategies bound coefficients to ≤ i12
    //! magnitude so that every 1-D `idct16` butterfly intermediate
    //! stays within i16 range — the SSE2 kernel's saturating
    //! `_mm_packs_epi32` then agrees bit-exactly with the modular
    //! `wraplow`-truncating scalar / i32-lane AVX2 reference. The
    //! pathological "i16 extremes" strategy from
    //! `idct16x16_avx2::tests` is deliberately omitted because it
    //! trips the saturation divergence (see module docstring).
    //!
    //! Sub-tests:
    //! 1. `load_input_data8` — i12 envelope (so saturating-pack equals
    //!    cast).
    //! 2. `transpose_16bit_8x8` — arbitrary i16 lanes (pure shuffle).
    //! 3. `idct16_8col` — i12 envelope; scalar oracle =
    //!    `crate::transforms::inverse::idct16` per column.
    //! 4. `write_buffer_8x1` — i12 envelope; scalar oracle = saturating
    //!    `((x + 32) >> 6) + dst`, clamped to u8.
    //! 5. Outer differential vs `idct16x16_add_avx2` — 4 strategies ×
    //!    256 cases each.

    use super::*;
    use crate::dsp::x86_64::idct16x16_add_avx2;
    use crate::transforms::inverse::idct16;
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
        // SAFETY: SSE2 is universal on x86_64 and gated by the caller's
        // `require_sse2!()` macro; `out` is 8 × i16 = 16 bytes,
        // matching the `__m128i` width.
        unsafe {
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v);
        }
        out
    }

    /// Pack 8 i16 lanes into an `__m128i`.
    fn pack_i16x8(lanes: [i16; 8]) -> __m128i {
        // SAFETY: SSE2 gated by caller's `require_sse2!()`; `lanes`
        // is 8 × i16 = 16 bytes.
        unsafe { _mm_loadu_si128(lanes.as_ptr() as *const __m128i) }
    }

    // -----------------------------------------------------------------
    // Sub-task 1: load_input_data8
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn load_input_data8_proptest(
            // i12 envelope — far below the saturating-pack threshold,
            // so the result equals a straight i32 → i16 cast.
            data in prop::array::uniform8(-2048i32..=2047),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            // SAFETY: SSE2 gated above; `data.as_ptr()` is valid for
            // 8 × i32 = 32 bytes (matches the two _mm_loadu_si128 reads).
            let v = unsafe { load_input_data8(data.as_ptr()) };
            let lanes = extract_i16x8(v);
            // Oracle: in-envelope saturating-clamp + cast == direct cast.
            for k in 0..8 {
                let expected = data[k].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                prop_assert_eq!(lanes[k], expected);
            }
        }

        /// Confirms saturating behaviour on out-of-envelope inputs (the
        /// HBD pack-saturate semantics that the in-envelope proptest
        /// above can't exercise).
        #[test]
        fn load_input_data8_saturates_proptest(
            data in prop::array::uniform8(i32::MIN..=i32::MAX),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            // SAFETY: SSE2 gated above.
            let v = unsafe { load_input_data8(data.as_ptr()) };
            let lanes = extract_i16x8(v);
            for k in 0..8 {
                let expected = data[k].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                prop_assert_eq!(lanes[k], expected);
            }
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 2: transpose_16bit_8x8
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn transpose_16bit_8x8_proptest(
            // Arbitrary i16 — pure shuffle is bit-exact for any value.
            rows in prop::array::uniform8(prop::array::uniform8(i16::MIN..=i16::MAX)),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
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
            let mut out_v: [__m128i; 8] = [unsafe { _mm_setzero_si128() }; 8];
            // SAFETY: SSE2 gated above.
            unsafe { transpose_16bit_8x8(&in_v, &mut out_v); }
            // Oracle: simple 8×8 matrix transpose.
            for r in 0..8 {
                let lanes = extract_i16x8(out_v[r]);
                for c in 0..8 {
                    prop_assert_eq!(
                        lanes[c], rows[c][r],
                        "row {} col {}: got={} expected={}",
                        r, c, lanes[c], rows[c][r]
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 3: idct16_8col
    //
    // Strategy: pack 16 rows × 8 i16 lanes into the input register
    // file. Each lane k corresponds to one independent 1-D idct16
    // instance. Compare lane-by-lane against the scalar `idct16`
    // applied to the per-column input column.
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct16_8col_proptest(
            // i12 envelope keeps every butterfly intermediate inside
            // i16 across all 6 stages, matching the saturating-pack ==
            // wraplow boundary.
            rows in prop::array::uniform16(prop::array::uniform8(-2048i16..=2047i16)),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let in_v: [__m128i; 16] = [
                pack_i16x8(rows[0]),  pack_i16x8(rows[1]),
                pack_i16x8(rows[2]),  pack_i16x8(rows[3]),
                pack_i16x8(rows[4]),  pack_i16x8(rows[5]),
                pack_i16x8(rows[6]),  pack_i16x8(rows[7]),
                pack_i16x8(rows[8]),  pack_i16x8(rows[9]),
                pack_i16x8(rows[10]), pack_i16x8(rows[11]),
                pack_i16x8(rows[12]), pack_i16x8(rows[13]),
                pack_i16x8(rows[14]), pack_i16x8(rows[15]),
            ];
            let mut out_v: [__m128i; 16] = [unsafe { _mm_setzero_si128() }; 16];
            // SAFETY: SSE2 gated above.
            unsafe { idct16_8col(&in_v, &mut out_v); }

            // Per-column scalar oracle.
            for k in 0..8 {
                let scalar_in: [i32; 16] = [
                    rows[0][k] as i32,  rows[1][k] as i32,
                    rows[2][k] as i32,  rows[3][k] as i32,
                    rows[4][k] as i32,  rows[5][k] as i32,
                    rows[6][k] as i32,  rows[7][k] as i32,
                    rows[8][k] as i32,  rows[9][k] as i32,
                    rows[10][k] as i32, rows[11][k] as i32,
                    rows[12][k] as i32, rows[13][k] as i32,
                    rows[14][k] as i32, rows[15][k] as i32,
                ];
                let mut scalar_out = [0i32; 16];
                idct16(&scalar_in, &mut scalar_out);
                for r in 0..16 {
                    let lanes = extract_i16x8(out_v[r]);
                    // In-envelope ⇒ scalar fits in i16, saturating-cast
                    // == direct cast.
                    let expected =
                        scalar_out[r].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    prop_assert_eq!(
                        lanes[k], expected,
                        "row {} col {}: sse2={} scalar={}",
                        r, k, lanes[k], expected
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 4: write_buffer_8x1
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn write_buffer_8x1_proptest(
            // i12 envelope — `_mm_adds_epi16(x, 32)` only saturates for
            // x ≥ i16::MAX - 32, which an i12 input cannot reach.
            in_lanes in prop::array::uniform8(-2048i16..=2047i16),
            dst_bytes in prop::array::uniform8(0u8..=255),
        ) {
            if !is_x86_feature_detected!("sse2") {
                return Ok(());
            }
            let in_v = pack_i16x8(in_lanes);
            let mut dst = dst_bytes;
            // SAFETY: SSE2 gated above; `dst.as_mut_ptr()` is valid for
            // 8 bytes (matches the _mm_loadl_epi64 / _mm_storel_epi64
            // pair inside write_buffer_8x1).
            unsafe { write_buffer_8x1(dst.as_mut_ptr(), in_v); }
            // Oracle: ((x + 32) >> 6) (arithmetic shift, saturating
            // add — both fold to plain (x + 32) / 64 with rounding-half-
            // away-from-zero towards -inf), then add dst u8 (sign-
            // extended to i16 via zero-extend), then saturating-pack
            // back to u8.
            for k in 0..8 {
                let stage = ((in_lanes[k] as i32) + 32) >> 6; // arithmetic >> 6
                let sum = stage + (dst_bytes[k] as i32);
                let expected = sum.clamp(0, 255) as u8;
                prop_assert_eq!(
                    dst[k], expected,
                    "lane {}: in={} dst_in={} got={} expected={}",
                    k, in_lanes[k], dst_bytes[k], dst[k], expected
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 5: outer differential against idct16x16_add_avx2
    //             (the i32-lane oracle, bit-exact within envelope).
    // -----------------------------------------------------------------

    fn run_sse2(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        assert!(is_x86_feature_detected!("sse2"));
        let mut dst = *dst_init;
        idct16x16_add_sse2_256(input, &mut dst, 16);
        dst
    }

    fn run_avx2_oracle(input: &[i32; 256], dst_init: &[u8; 256]) -> [u8; 256] {
        assert!(is_x86_feature_detected!("avx2"));
        let mut dst = *dst_init;
        idct16x16_add_avx2(input, &mut dst, 16);
        dst
    }

    /// `uniform_12` — full-density i12 coefficients in every position.
    fn uniform_strategy_12() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(-2048i32..=2047, 256).prop_map(|v| {
            let mut out = [0i32; 256];
            out.copy_from_slice(&v);
            out
        })
    }

    /// `sparse_12` — 1..=8 non-zero coefs at arbitrary raster positions
    /// (the libvpx _10/_38 sparse-bucket density region).
    fn sparse_strategy_12() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec((0usize..256, -2048i32..=2047), 1..=8).prop_map(|nz| {
            let mut out = [0i32; 256];
            for (pos, val) in nz {
                out[pos] = val;
            }
            out
        })
    }

    /// `eob_bucket_38` — exactly 38 non-zero coefs at the first 38
    /// raster positions; straddles libvpx's idct16x16_38_add dispatcher
    /// boundary in the eob-bucket sense.
    fn eob_bucket_38_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(-2048i32..=2047, 38).prop_map(|values| {
            let mut out = [0i32; 256];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
    }

    /// `eob_bucket_dense` — full 256-coef density at i12 magnitude
    /// (mirrors `uniform_strategy_12` but is kept as a separate
    /// strategy entry to align with the HLD §6.2 strategy list).
    fn eob_bucket_dense_strategy() -> impl Strategy<Value = [i32; 256]> {
        prop::collection::vec(-2048i32..=2047, 256).prop_map(|values| {
            let mut out = [0i32; 256];
            out.copy_from_slice(&values);
            out
        })
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
            cases: 256, max_shrink_iters: 1024, .. ProptestConfig::default()
        })]

        #[test]
        fn idct16x16_sse2_eq_avx2_uniform_12(
            input in uniform_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") || !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let avx2 = run_avx2_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, avx2);
        }

        #[test]
        fn idct16x16_sse2_eq_avx2_sparse_12(
            input in sparse_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") || !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let avx2 = run_avx2_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, avx2);
        }

        #[test]
        fn idct16x16_sse2_eq_avx2_eob_38(
            input in eob_bucket_38_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") || !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let avx2 = run_avx2_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, avx2);
        }

        #[test]
        fn idct16x16_sse2_eq_avx2_eob_dense(
            input in eob_bucket_dense_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("sse2") || !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let sse2 = run_sse2(&input, &dst_init);
            let avx2 = run_avx2_oracle(&input, &dst_init);
            prop_assert_eq!(sse2, avx2);
        }
    }

    // -----------------------------------------------------------------
    // Directed fixtures.
    // -----------------------------------------------------------------

    #[test]
    fn idct16x16_sse2_all_zero_input() {
        require_sse2!();
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [0i32; 256];
        let dst_init = [42u8; 256];
        let sse2 = run_sse2(&input, &dst_init);
        let avx2 = run_avx2_oracle(&input, &dst_init);
        assert_eq!(sse2, avx2);
        // All-zero input ⇒ rounding (+32 >> 6) == 0, so dst unchanged.
        assert_eq!(sse2, dst_init);
    }

    /// DC-only impulse — only `input[0]` non-zero.
    #[test]
    fn idct16x16_sse2_dc_impulse() {
        require_sse2!();
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &val in &[100i32, -100, 2000, -2000, 1, -1] {
            let mut input = [0i32; 256];
            input[0] = val;
            let dst_init = [128u8; 256];
            let sse2 = run_sse2(&input, &dst_init);
            let avx2 = run_avx2_oracle(&input, &dst_init);
            assert_eq!(sse2, avx2, "val={val}");
        }
    }

    /// Dst-pointer-offset correctness regression guard. Mirrors
    /// `idct8x8_12_misaligned_dst_pointers` from the Task A test
    /// suite. The kernel uses `_mm_loadl_epi64` / `_mm_storel_epi64`
    /// for `dst` I/O (8-byte, alignment-free), so this is a per-offset
    /// arithmetic regression guard, NOT an alignment-fault test.
    #[test]
    fn idct16x16_sse2_misaligned_dst_pointers() {
        require_sse2!();
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Synthetic input — moderate-magnitude across the eob-bucket
        // region (first 38 raster positions).
        let mut input = [0i32; 256];
        for (i, slot) in input.iter_mut().take(38).enumerate() {
            *slot = ((i as i32) * 47) - 800;
        }

        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 272]> = Box::new([0u8; 272]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut avx2_buf = *buf;

            idct16x16_add_sse2_256(&input, &mut buf[skew..skew + 256], 16);
            idct16x16_add_avx2(&input, &mut avx2_buf[skew..skew + 256], 16);

            assert_eq!(
                &buf[..],
                &avx2_buf[..],
                "SSE2/AVX2 mismatch at skew={skew}: per-offset arithmetic regression"
            );
        }
    }
}
