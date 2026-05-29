//! 32×32 inverse-DCT + add — x86_64 AVX2 i16-lane implementation
//! covering the dense `idct32x32_1024_add` slot when AVX2 is available.
//!
//! This kernel is a line-by-line port of libvpx's
//! `vpx_idct32x32_1024_add_avx2` (`third_party/libvpx/vpx_dsp/x86/inv_txfm_avx2.c:574-601`)
//! together with the inner 1-D engine `idct32_1024_16x32` (and its
//! `quarter_1/_2/_3_4` helpers) and the shared leaf helpers
//! `butterfly16` (`:63`), `add_sub_butterfly_avx2` (`:297`),
//! `recon_and_store16` (`:159`), `store_buffer_16x32` (`:179`) and the
//! `transpose2_8x8_avx2` / `transpose_16bit_16x16_avx2` transposes
//! (`:197`, `:237`).
//!
//! It is the 256-bit i16-lane sibling of the SSE2 i16-lane kernel
//! [`super::idct32x32_sse2_1024`]: same flowgraph, same cospi
//! constants, same `(x+8192)>>14` round-shift and same saturating
//! `_mm256_packs_epi32` wraplow — just widened from 8 i16 lanes per
//! xmm to 16 i16 lanes per ymm. The row/column passes therefore run
//! 2 strips of 16 (rather than 4 strips of 8), halving the
//! instruction count / transpose count / intermediate traffic.
//!
//! ## Saturating-vs-modular semantic divergence
//!
//! Identical to the SSE2 sibling: every `wraplow` site uses the dense
//! `_mm256_madd_epi16` pipeline with saturating `_mm256_packs_epi32`
//! (clamps to `±i16::MAX`) rather than the modular `wraplow`
//! (sign-extend low 16 bits) used by the i32-lane AVX2 kernels and the
//! scalar reference. The two policies agree on every in-spec bitstream
//! coefficient (and the 201-vector conformance corpus), but diverge on
//! synthetic adversarial inputs whose intermediates exceed `i16::MAX`.
//!
//! The differential proptests at the bottom of this file are bounded
//! to the same envelopes as the SSE2 sibling (`dense_8`, `sparse_12`,
//! `eob_34`, `eob_135` at ≤ i12); the i16-extreme strategies are
//! deliberately omitted. An additional SSE2-vs-AVX2 equality proptest
//! feeds arbitrary i16 lanes (no saturation bound) directly into both
//! 1-D engines and asserts they agree exactly.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Loads from `dst` use `_mm_loadu_si128` (16-byte, unaligned) via
//! `recon_and_store16`. The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm_storeu_si128, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_adds_epi16, _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_madd_epi16, _mm256_packs_epi32, _mm256_packus_epi16,
    _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set_epi16,
    _mm256_setzero_si256, _mm256_srai_epi16, _mm256_srai_epi32, _mm256_sub_epi16,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_unpacklo_epi8,
};

use crate::transforms::constants::{
    COSPI_1_64, COSPI_2_64, COSPI_3_64, COSPI_4_64, COSPI_5_64, COSPI_6_64, COSPI_7_64, COSPI_8_64,
    COSPI_9_64, COSPI_10_64, COSPI_11_64, COSPI_12_64, COSPI_13_64, COSPI_14_64, COSPI_15_64,
    COSPI_16_64, COSPI_17_64, COSPI_18_64, COSPI_19_64, COSPI_20_64, COSPI_21_64, COSPI_22_64,
    COSPI_23_64, COSPI_24_64, COSPI_25_64, COSPI_26_64, COSPI_27_64, COSPI_28_64, COSPI_29_64,
    COSPI_30_64, COSPI_31_64,
};

// ---------------------------------------------------------------------
// Tiny shared helpers (line-by-line ports from libvpx).
// ---------------------------------------------------------------------

/// `pair256_set_epi16(a, b)` — broadcast the interleaved (a, b) pair
/// across all 16 i16 lanes of a ymm: `[a, b, a, b, ...]`. Mirrors
/// libvpx's `PAIR256_SET_EPI16` macro (`inv_txfm_avx2.c:16-20`). Note
/// `_mm256_set_epi16` takes the highest lane first, so the lane-0 (a)
/// argument is supplied last.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pair256_set_epi16(a: i16, b: i16) -> __m256i {
    _mm256_set_epi16(
        b, a, b, a, b, a, b, a, b, a, b, a, b, a, b, a,
    )
}

/// `dct_round_shift(x) = (x + 8192) >> 14`, lane-parallel over 8 i32.
/// Mirrors libvpx `dct_round_shift_avx2` (`inv_txfm_avx2.c:44-47`):
/// `DCT_CONST_ROUNDING = 1 << 13`, `DCT_CONST_BITS = 14`.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dct_round_shift_avx2(x: __m256i) -> __m256i {
    let t = _mm256_add_epi32(x, _mm256_set1_epi32(1 << 13));
    _mm256_srai_epi32(t, 14)
}

/// `idct_calc_wraplow_avx2(in0, in1, x)` — two `_mm256_madd_epi16`
/// dot-products + round-shift + saturating `_mm256_packs_epi32` to 16
/// i16 lanes. Mirrors libvpx `idct_calc_wraplow_avx2`
/// (`inv_txfm_avx2.c:49-60`). The pack is SATURATING — same policy as
/// the SSE2 sibling's `idct_calc_wraplow_sse2`.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct_calc_wraplow_avx2(in0: __m256i, in1: __m256i, x: __m256i) -> __m256i {
    let t0 = dct_round_shift_avx2(_mm256_madd_epi16(in0, x));
    let t1 = dct_round_shift_avx2(_mm256_madd_epi16(in1, x));
    _mm256_packs_epi32(t0, t1)
}

/// `butterfly16(in0, in1, c0, c1) -> (out0, out1)` — two-output
/// butterfly via `idct_calc_wraplow_avx2` over the `unpacklo` /
/// `unpackhi` interleave of `in0`/`in1` with `pair256_set_epi16(c0, -c1)`
/// and `pair256_set_epi16(c1, c0)`.
///
/// Mirrors libvpx `butterfly16` (`inv_txfm_avx2.c:63-71`); the i16-lane
/// twin of the SSE2 `butterfly` (`idct8x8_12_sse2.rs:145`).
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn butterfly16(in0: __m256i, in1: __m256i, c0: i16, c1: i16) -> (__m256i, __m256i) {
    let cst0 = pair256_set_epi16(c0, -c1);
    let cst1 = pair256_set_epi16(c1, c0);
    let lo = _mm256_unpacklo_epi16(in0, in1);
    let hi = _mm256_unpackhi_epi16(in0, in1);
    let out0 = idct_calc_wraplow_avx2(lo, hi, cst0);
    let out1 = idct_calc_wraplow_avx2(lo, hi, cst1);
    (out0, out1)
}

/// `recon_and_store16(dest, in_x)` — saturating residual add of 16 i16
/// lanes into 16 dst bytes (final +32 rounding / arith >>6 done by the
/// caller). Mirrors libvpx `recon_and_store16` (`inv_txfm_avx2.c:159-169`).
///
/// The `_mm256_permute4x64_epi64(_, 0xd8)` (`:162`) reorders the 4
/// 64-bit lanes `[0,1,2,3] -> [0,2,1,3]` so that the subsequent
/// `_mm256_unpacklo_epi8` widens all 16 dst bytes (not just the low 8
/// of each 128-bit half) to i16 in the correct lane order.
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports AVX2.
/// - `dest..dest+16` is a valid, readable + writable u8 region.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn recon_and_store16(dest: *mut u8, in_x: __m256i) {
    let zero = _mm256_setzero_si256();
    let d128 = _mm_loadu_si128(dest as *const __m128i);
    let mut d0 = _mm256_castsi128_si256(d128);
    d0 = _mm256_permute4x64_epi64(d0, 0xd8);
    d0 = _mm256_unpacklo_epi8(d0, zero);
    d0 = _mm256_add_epi16(in_x, d0);
    let hi = _mm256_castsi128_si256(_mm256_extracti128_si256(d0, 1));
    d0 = _mm256_packus_epi16(d0, hi);
    _mm_storeu_si128(dest as *mut __m128i, _mm256_castsi256_si128(d0));
}

/// `store_buffer_16x32(in_, dst, stride)` — final-rounding (+32, >>6) of
/// 32 rows × 16 i16 lanes followed by 16-byte residual adds. Mirrors
/// libvpx `store_buffer_16x32` (`inv_txfm_avx2.c:179-195`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports AVX2.
/// - `dst..dst + 31*stride + 16` is a valid u8 region.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn store_buffer_16x32(in_: &mut [__m256i; 32], mut dst: *mut u8, stride: usize) {
    let final_rounding = _mm256_set1_epi16(1 << 5);
    let mut j = 0;
    while j < 32 {
        in_[j] = _mm256_adds_epi16(in_[j], final_rounding);
        in_[j + 1] = _mm256_adds_epi16(in_[j + 1], final_rounding);

        in_[j] = _mm256_srai_epi16(in_[j], 6);
        in_[j + 1] = _mm256_srai_epi16(in_[j + 1], 6);

        recon_and_store16(dst, in_[j]);
        dst = dst.add(stride);
        recon_and_store16(dst, in_[j + 1]);
        dst = dst.add(stride);
        j += 2;
    }
}

/// `add_sub_butterfly_avx2(in_, out, size)` — paired add/sub butterfly
/// used at the final stage of the 1-D idct. Mirrors libvpx
/// `add_sub_butterfly_avx2` (`inv_txfm_avx2.c:297-306`).
///
/// For each `i` in `0..size/2`: `out[i] = in_[i] + in_[size-1-i]`,
/// `out[size-1-i] = in_[i] - in_[size-1-i]`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn add_sub_butterfly_avx2(in_: &[__m256i], out: &mut [__m256i], size: usize) {
    let half = size >> 1;
    let bound = size - 1;
    for i in 0..half {
        out[i] = _mm256_add_epi16(in_[i], in_[bound - i]);
        out[bound - i] = _mm256_sub_epi16(in_[i], in_[bound - i]);
    }
}

// ---------------------------------------------------------------------
// Transposes (pure lane shuffles — bit-exact for arbitrary i16 lanes).
// ---------------------------------------------------------------------

/// `transpose2_8x8_avx2(in, out)` — transpose two stacked 8×8 i16
/// blocks (the two 128-bit halves of each ymm carry independent 8×8
/// blocks). Mirrors libvpx `transpose2_8x8_avx2`
/// (`inv_txfm_avx2.c:197-235`).
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose2_8x8_avx2(in_: &[__m256i], out: &mut [__m256i]) {
    let mut t: [__m256i; 16] = [_mm256_setzero_si256(); 16];
    let mut u: [__m256i; 16] = [_mm256_setzero_si256(); 16];

    // (1st, 2nd) ==> (lo, hi)
    for i in 0..4 {
        t[2 * i] = _mm256_unpacklo_epi16(in_[2 * i], in_[2 * i + 1]);
        t[2 * i + 1] = _mm256_unpackhi_epi16(in_[2 * i], in_[2 * i + 1]);
    }

    for i in 0..2 {
        u[i] = _mm256_unpacklo_epi32(t[i], t[i + 2]);
        u[i + 2] = _mm256_unpackhi_epi32(t[i], t[i + 2]);

        u[i + 4] = _mm256_unpacklo_epi32(t[i + 4], t[i + 6]);
        u[i + 6] = _mm256_unpackhi_epi32(t[i + 4], t[i + 6]);
    }

    for i in 0..2 {
        out[2 * i] = _mm256_unpacklo_epi64(u[2 * i], u[2 * i + 4]);
        out[2 * i + 1] = _mm256_unpackhi_epi64(u[2 * i], u[2 * i + 4]);

        out[2 * i + 4] = _mm256_unpacklo_epi64(u[2 * i + 1], u[2 * i + 5]);
        out[2 * i + 5] = _mm256_unpackhi_epi64(u[2 * i + 1], u[2 * i + 5]);
    }
}

/// `build_transpose_t(src)` — the `LOADL`/`LOADR` interleave stage of
/// `transpose_16bit_16x16_avx2`, returning the intermediate `t[16]`.
///
/// libvpx's `LOADL`/`LOADR` macros perform aligned 128-bit loads from
/// the local `in[]` register array reinterpreted as 32 `__m128i`:
/// `&in[idx]` as `*__m128i` reads the low half of row `idx`, `&in[idx]+1`
/// reads the high half. We reconstruct that with
/// `_mm256_castsi256_si128` (low half) / `_mm256_extracti128_si256(_, 1)`
/// (high half). Every read happens up front, so `src` may alias the
/// caller's eventual output (in-place row pass) bit-exactly.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2 and `src.len() >= 16`.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn build_transpose_t(src: &[__m256i]) -> [__m256i; 16] {
    let mut t: [__m256i; 16] = [_mm256_setzero_si256(); 16];
    for idx in 0..8 {
        // LOADL(idx): t[idx] = [ low(row idx) | low(row idx+8) ]
        let la = _mm256_castsi256_si128(src[idx]);
        let lb = _mm256_castsi256_si128(src[idx + 8]);
        t[idx] = _mm256_inserti128_si256(_mm256_castsi128_si256(la), lb, 1);
        // LOADR(idx): t[8+idx] = [ high(row idx) | high(row idx+8) ]
        let ha = _mm256_extracti128_si256(src[idx], 1);
        let hb = _mm256_extracti128_si256(src[idx + 8], 1);
        t[8 + idx] = _mm256_inserti128_si256(_mm256_castsi128_si256(ha), hb, 1);
    }
    t
}

/// `transpose_16bit_16x16_avx2(io)` — in-place transpose of a 16×16 i16
/// block (each `io[i]` holds a 16-wide row). Mirrors libvpx's in-place
/// `transpose_16bit_16x16_avx2(in, in)` call site (the row pass).
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn transpose_16bit_16x16_avx2(io: &mut [__m256i]) {
    let t = build_transpose_t(io);
    let (top, bottom) = io.split_at_mut(8);
    transpose2_8x8_avx2(&t[0..8], top);
    transpose2_8x8_avx2(&t[8..16], bottom);
}

/// `transpose_16bit_16x16_avx2_to(src, dst)` — transpose a 16×16 i16
/// block from `src` into a DISTINCT `dst` (the column pass:
/// `transpose_16bit_16x16_avx2(l + i, out)`). `src` and `dst` must not
/// overlap.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2, `src.len() >= 16`,
/// `dst.len() >= 16`, and that `src`/`dst` do not overlap.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn transpose_16bit_16x16_avx2_to(src: &[__m256i], dst: &mut [__m256i]) {
    let t = build_transpose_t(src);
    let (top, bottom) = dst.split_at_mut(8);
    transpose2_8x8_avx2(&t[0..8], top);
    transpose2_8x8_avx2(&t[8..16], bottom);
}

// ---------------------------------------------------------------------
// idct32_1024_16x32 quarter helpers (line-by-line ports from libvpx).
// ---------------------------------------------------------------------

/// `idct32_1024_16x32_quarter_1(in, out)` — even-index quarter. Reads
/// `in[0, 4, 8, 12, 16, 20, 24, 28]`; writes `out[0..8]`. Mirrors
/// libvpx `inv_txfm_avx2.c:311-345`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_1024_16x32_quarter_1(in_: &[__m256i; 32], out: &mut [__m256i]) {
    let zero = _mm256_setzero_si256();
    let mut step1: [__m256i; 8] = [zero; 8];
    let mut step2: [__m256i; 8] = [zero; 8];

    // stage 3
    let (s4, s7) = butterfly16(in_[4], in_[28], COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[4] = s4;
    step1[7] = s7;
    let (s5, s6) = butterfly16(in_[20], in_[12], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[5] = s5;
    step1[6] = s6;

    // stage 4 — note butterfly output order (s2_1, s2_0) for the
    // cospi_16,cospi_16 pair.
    let (s2_1, s2_0) = butterfly16(in_[0], in_[16], COSPI_16_64 as i16, COSPI_16_64 as i16);
    step2[1] = s2_1;
    step2[0] = s2_0;
    let (s2_2, s2_3) = butterfly16(in_[8], in_[24], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step2[2] = s2_2;
    step2[3] = s2_3;
    step2[4] = _mm256_add_epi16(step1[4], step1[5]);
    step2[5] = _mm256_sub_epi16(step1[4], step1[5]);
    step2[6] = _mm256_sub_epi16(step1[7], step1[6]);
    step2[7] = _mm256_add_epi16(step1[7], step1[6]);

    // stage 5
    step1[0] = _mm256_add_epi16(step2[0], step2[3]);
    step1[1] = _mm256_add_epi16(step2[1], step2[2]);
    step1[2] = _mm256_sub_epi16(step2[1], step2[2]);
    step1[3] = _mm256_sub_epi16(step2[0], step2[3]);
    step1[4] = step2[4];
    let (s5b, s6b) = butterfly16(step2[6], step2[5], COSPI_16_64 as i16, COSPI_16_64 as i16);
    step1[5] = s5b;
    step1[6] = s6b;
    step1[7] = step2[7];

    // stage 6
    out[0] = _mm256_add_epi16(step1[0], step1[7]);
    out[1] = _mm256_add_epi16(step1[1], step1[6]);
    out[2] = _mm256_add_epi16(step1[2], step1[5]);
    out[3] = _mm256_add_epi16(step1[3], step1[4]);
    out[4] = _mm256_sub_epi16(step1[3], step1[4]);
    out[5] = _mm256_sub_epi16(step1[2], step1[5]);
    out[6] = _mm256_sub_epi16(step1[1], step1[6]);
    out[7] = _mm256_sub_epi16(step1[0], step1[7]);
}

/// `idct32_16x32_quarter_2_stage_4_to_6(step1, out)` — shared stage 4-6
/// of quarter 2. Mirrors libvpx `inv_txfm_avx2.c:347-380`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_16x32_quarter_2_stage_4_to_6(step1: &mut [__m256i; 16], out: &mut [__m256i]) {
    let mut step2: [__m256i; 16] = [_mm256_setzero_si256(); 16];

    // stage 4
    step2[8] = step1[8];
    step2[15] = step1[15];
    let (s9, s14) = butterfly16(step1[14], step1[9], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step2[9] = s9;
    step2[14] = s14;
    let (s10, s13) = butterfly16(
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
    step1[8] = _mm256_add_epi16(step2[8], step2[11]);
    step1[9] = _mm256_add_epi16(step2[9], step2[10]);
    step1[10] = _mm256_sub_epi16(step2[9], step2[10]);
    step1[11] = _mm256_sub_epi16(step2[8], step2[11]);
    step1[12] = _mm256_sub_epi16(step2[15], step2[12]);
    step1[13] = _mm256_sub_epi16(step2[14], step2[13]);
    step1[14] = _mm256_add_epi16(step2[14], step2[13]);
    step1[15] = _mm256_add_epi16(step2[15], step2[12]);

    // stage 6
    out[8] = step1[8];
    out[9] = step1[9];
    let (o10, o13) = butterfly16(step1[13], step1[10], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[10] = o10;
    out[13] = o13;
    let (o11, o12) = butterfly16(step1[12], step1[11], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[11] = o11;
    out[12] = o12;
    out[14] = step1[14];
    out[15] = step1[15];
}

/// `idct32_1024_16x32_quarter_2(in, out)` — odd-multiples-of-2 quarter.
/// Reads `in[2, 6, 10, 14, 18, 22, 26, 30]`; writes `out[8..16]`.
/// Mirrors libvpx `inv_txfm_avx2.c:385-405`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_1024_16x32_quarter_2(in_: &[__m256i; 32], out: &mut [__m256i]) {
    let zero = _mm256_setzero_si256();
    let mut step1: [__m256i; 16] = [zero; 16];
    let mut step2: [__m256i; 16] = [zero; 16];

    // stage 2
    let (s8, s15) = butterfly16(in_[2], in_[30], COSPI_30_64 as i16, COSPI_2_64 as i16);
    step2[8] = s8;
    step2[15] = s15;
    let (s9, s14) = butterfly16(in_[18], in_[14], COSPI_14_64 as i16, COSPI_18_64 as i16);
    step2[9] = s9;
    step2[14] = s14;
    let (s10, s13) = butterfly16(in_[10], in_[22], COSPI_22_64 as i16, COSPI_10_64 as i16);
    step2[10] = s10;
    step2[13] = s13;
    let (s11, s12) = butterfly16(in_[26], in_[6], COSPI_6_64 as i16, COSPI_26_64 as i16);
    step2[11] = s11;
    step2[12] = s12;

    // stage 3
    step1[8] = _mm256_add_epi16(step2[8], step2[9]);
    step1[9] = _mm256_sub_epi16(step2[8], step2[9]);
    step1[10] = _mm256_sub_epi16(step2[11], step2[10]);
    step1[11] = _mm256_add_epi16(step2[11], step2[10]);
    step1[12] = _mm256_add_epi16(step2[12], step2[13]);
    step1[13] = _mm256_sub_epi16(step2[12], step2[13]);
    step1[14] = _mm256_sub_epi16(step2[15], step2[14]);
    step1[15] = _mm256_add_epi16(step2[15], step2[14]);

    idct32_16x32_quarter_2_stage_4_to_6(&mut step1, out);
}

/// `idct32_1024_16x32_quarter_1_2(in, out)` — combine quarters 1+2 then
/// size-16 add/sub. Mirrors libvpx `inv_txfm_avx2.c:478-492`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_1024_16x32_quarter_1_2(in_: &[__m256i; 32], out: &mut [__m256i; 32]) {
    let mut temp: [__m256i; 16] = [_mm256_setzero_si256(); 16];
    idct32_1024_16x32_quarter_1(in_, &mut temp);
    idct32_1024_16x32_quarter_2(in_, &mut temp);
    // stage 7 — size-16 add/sub.
    add_sub_butterfly_avx2(&temp, &mut out[..16], 16);
}

/// `idct32_16x32_quarter_3_4_stage_4_to_7(step1, out)` — shared stage
/// 4-7 of the odd quarter. Mirrors libvpx `inv_txfm_avx2.c:407-476`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_16x32_quarter_3_4_stage_4_to_7(step1: &mut [__m256i; 32], out: &mut [__m256i]) {
    let mut step2: [__m256i; 32] = [_mm256_setzero_si256(); 32];

    // stage 4
    step2[16] = _mm256_add_epi16(step1[16], step1[19]);
    step2[17] = _mm256_add_epi16(step1[17], step1[18]);
    step2[18] = _mm256_sub_epi16(step1[17], step1[18]);
    step2[19] = _mm256_sub_epi16(step1[16], step1[19]);
    step2[20] = _mm256_sub_epi16(step1[23], step1[20]);
    step2[21] = _mm256_sub_epi16(step1[22], step1[21]);
    step2[22] = _mm256_add_epi16(step1[22], step1[21]);
    step2[23] = _mm256_add_epi16(step1[23], step1[20]);

    step2[24] = _mm256_add_epi16(step1[24], step1[27]);
    step2[25] = _mm256_add_epi16(step1[25], step1[26]);
    step2[26] = _mm256_sub_epi16(step1[25], step1[26]);
    step2[27] = _mm256_sub_epi16(step1[24], step1[27]);
    step2[28] = _mm256_sub_epi16(step1[31], step1[28]);
    step2[29] = _mm256_sub_epi16(step1[30], step1[29]);
    step2[30] = _mm256_add_epi16(step1[29], step1[30]);
    step2[31] = _mm256_add_epi16(step1[28], step1[31]);

    // stage 5
    step1[16] = step2[16];
    step1[17] = step2[17];
    let (s18, s29) = butterfly16(step2[29], step2[18], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step1[18] = s18;
    step1[29] = s29;
    let (s19, s28) = butterfly16(step2[28], step2[19], COSPI_24_64 as i16, COSPI_8_64 as i16);
    step1[19] = s19;
    step1[28] = s28;
    let (s20, s27) = butterfly16(
        step2[27],
        step2[20],
        -(COSPI_8_64 as i16),
        COSPI_24_64 as i16,
    );
    step1[20] = s20;
    step1[27] = s27;
    let (s21, s26) = butterfly16(
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
    out[16] = _mm256_add_epi16(step1[16], step1[23]);
    out[17] = _mm256_add_epi16(step1[17], step1[22]);
    out[18] = _mm256_add_epi16(step1[18], step1[21]);
    out[19] = _mm256_add_epi16(step1[19], step1[20]);
    step2[20] = _mm256_sub_epi16(step1[19], step1[20]);
    step2[21] = _mm256_sub_epi16(step1[18], step1[21]);
    step2[22] = _mm256_sub_epi16(step1[17], step1[22]);
    step2[23] = _mm256_sub_epi16(step1[16], step1[23]);

    step2[24] = _mm256_sub_epi16(step1[31], step1[24]);
    step2[25] = _mm256_sub_epi16(step1[30], step1[25]);
    step2[26] = _mm256_sub_epi16(step1[29], step1[26]);
    step2[27] = _mm256_sub_epi16(step1[28], step1[27]);
    out[28] = _mm256_add_epi16(step1[27], step1[28]);
    out[29] = _mm256_add_epi16(step1[26], step1[29]);
    out[30] = _mm256_add_epi16(step1[25], step1[30]);
    out[31] = _mm256_add_epi16(step1[24], step1[31]);

    // stage 7
    let (o20, o27) = butterfly16(step2[27], step2[20], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[20] = o20;
    out[27] = o27;
    let (o21, o26) = butterfly16(step2[26], step2[21], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[21] = o21;
    out[26] = o26;
    let (o22, o25) = butterfly16(step2[25], step2[22], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[22] = o22;
    out[25] = o25;
    let (o23, o24) = butterfly16(step2[24], step2[23], COSPI_16_64 as i16, COSPI_16_64 as i16);
    out[23] = o23;
    out[24] = o24;
}

/// `idct32_1024_16x32_quarter_3_4(in, out)` — odd-indexed quarter.
/// Reads the 16 odd indices; writes `out[16..32]`. Mirrors libvpx
/// `inv_txfm_avx2.c:498-551`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn idct32_1024_16x32_quarter_3_4(in_: &[__m256i; 32], out: &mut [__m256i; 32]) {
    let zero = _mm256_setzero_si256();
    let mut step1: [__m256i; 32] = [zero; 32];
    let mut step2: [__m256i; 32] = [zero; 32];

    // stage 1
    let (a16, a31) = butterfly16(in_[1], in_[31], COSPI_31_64 as i16, COSPI_1_64 as i16);
    step1[16] = a16;
    step1[31] = a31;
    let (a17, a30) = butterfly16(in_[17], in_[15], COSPI_15_64 as i16, COSPI_17_64 as i16);
    step1[17] = a17;
    step1[30] = a30;
    let (a18, a29) = butterfly16(in_[9], in_[23], COSPI_23_64 as i16, COSPI_9_64 as i16);
    step1[18] = a18;
    step1[29] = a29;
    let (a19, a28) = butterfly16(in_[25], in_[7], COSPI_7_64 as i16, COSPI_25_64 as i16);
    step1[19] = a19;
    step1[28] = a28;

    let (a20, a27) = butterfly16(in_[5], in_[27], COSPI_27_64 as i16, COSPI_5_64 as i16);
    step1[20] = a20;
    step1[27] = a27;
    let (a21, a26) = butterfly16(in_[21], in_[11], COSPI_11_64 as i16, COSPI_21_64 as i16);
    step1[21] = a21;
    step1[26] = a26;

    let (a22, a25) = butterfly16(in_[13], in_[19], COSPI_19_64 as i16, COSPI_13_64 as i16);
    step1[22] = a22;
    step1[25] = a25;
    let (a23, a24) = butterfly16(in_[29], in_[3], COSPI_3_64 as i16, COSPI_29_64 as i16);
    step1[23] = a23;
    step1[24] = a24;

    // stage 2
    step2[16] = _mm256_add_epi16(step1[16], step1[17]);
    step2[17] = _mm256_sub_epi16(step1[16], step1[17]);
    step2[18] = _mm256_sub_epi16(step1[19], step1[18]);
    step2[19] = _mm256_add_epi16(step1[19], step1[18]);
    step2[20] = _mm256_add_epi16(step1[20], step1[21]);
    step2[21] = _mm256_sub_epi16(step1[20], step1[21]);
    step2[22] = _mm256_sub_epi16(step1[23], step1[22]);
    step2[23] = _mm256_add_epi16(step1[23], step1[22]);

    step2[24] = _mm256_add_epi16(step1[24], step1[25]);
    step2[25] = _mm256_sub_epi16(step1[24], step1[25]);
    step2[26] = _mm256_sub_epi16(step1[27], step1[26]);
    step2[27] = _mm256_add_epi16(step1[27], step1[26]);
    step2[28] = _mm256_add_epi16(step1[28], step1[29]);
    step2[29] = _mm256_sub_epi16(step1[28], step1[29]);
    step2[30] = _mm256_sub_epi16(step1[31], step1[30]);
    step2[31] = _mm256_add_epi16(step1[31], step1[30]);

    // stage 3
    step1[16] = step2[16];
    step1[31] = step2[31];
    let (b17, b30) = butterfly16(step2[30], step2[17], COSPI_28_64 as i16, COSPI_4_64 as i16);
    step1[17] = b17;
    step1[30] = b30;
    let (b18, b29) = butterfly16(
        step2[29],
        step2[18],
        -(COSPI_4_64 as i16),
        COSPI_28_64 as i16,
    );
    step1[18] = b18;
    step1[29] = b29;
    step1[19] = step2[19];
    step1[20] = step2[20];
    let (b21, b26) = butterfly16(step2[26], step2[21], COSPI_12_64 as i16, COSPI_20_64 as i16);
    step1[21] = b21;
    step1[26] = b26;
    let (b22, b25) = butterfly16(
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

    idct32_16x32_quarter_3_4_stage_4_to_7(&mut step1, out);
}

/// `idct32_1024_16x32(io)` — full 1-D 32-point inverse DCT, processing
/// 16 independent instances in parallel (one per i16 lane), IN PLACE.
/// Mirrors libvpx `inv_txfm_avx2.c:553-572`, which is always called
/// in-place (`idct32_1024_16x32(in, in)`).
///
/// The two quarter helpers read `io` and write the local `temp`; the
/// final size-32 add/sub reads `temp` (fully populated) and overwrites
/// `io`. Since `temp` is a distinct local, the in-place rewrite of `io`
/// is bit-exact — no input value is clobbered before it is consumed.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn idct32_1024_16x32(io: &mut [__m256i; 32]) {
    let mut temp: [__m256i; 32] = [_mm256_setzero_si256(); 32];

    idct32_1024_16x32_quarter_1_2(io, &mut temp);
    idct32_1024_16x32_quarter_3_4(io, &mut temp);
    // final stage — size-32 add/sub. Reads `temp` (local), writes `io`.
    add_sub_butterfly_avx2(&temp, io, 32);
}

// ---------------------------------------------------------------------
// Load helper — 16×16 i16 block from the i32 coefficient buffer.
// ---------------------------------------------------------------------

/// `idct_load16x16(input, in, stride)` — load a 16×16 block of i16
/// values from the i32 coefficient buffer. Each output row packs cols
/// 0..16 into one ymm (low 128 = cols 0..8, high 128 = cols 8..16),
/// saturating-packing the i32 coefficients down to i16. Mirrors libvpx
/// `idct_load16x16` (`inv_txfm_avx2.c:22-42`, HBD branch — our
/// coefficients are i32 like libvpx's `tran_low_t` under HBD).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports AVX2.
/// - `input..input + 15*stride + 16` is a valid, readable region of i32.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn idct_load16x16(input: *const i32, in_: &mut [__m256i], stride: usize) {
    use core::arch::x86_64::_mm_packs_epi32;
    for (i, slot) in in_.iter_mut().take(16).enumerate() {
        let base = input.add(i * stride);
        let in0 = _mm_loadu_si128(base as *const __m128i);
        let in1 = _mm_loadu_si128(base.add(4) as *const __m128i);
        let in2 = _mm_loadu_si128(base.add(8) as *const __m128i);
        let in3 = _mm_loadu_si128(base.add(12) as *const __m128i);
        let ls = _mm_packs_epi32(in0, in1);
        let rs = _mm_packs_epi32(in2, in3);
        *slot = _mm256_inserti128_si256(_mm256_castsi128_si256(ls), rs, 1);
    }
}

// ---------------------------------------------------------------------
// Top-level kernel.
// ---------------------------------------------------------------------

/// Inner AVX2 implementation of the dense 32×32 inverse-DCT + residual
/// add. Two-pass `idct32_1024_16x32` (row pass + column pass) mirroring
/// libvpx `vpx_idct32x32_1024_add_avx2` (`inv_txfm_avx2.c:574-601`).
///
/// # Safety
///
/// Caller must guarantee:
/// - The running CPU supports AVX2.
/// - `dst` has at least `31 * stride + 32` bytes of valid storage.
// G4: loads from `dst` use _mm_loadu_si128 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn idct32x32_add_avx2_1024_inner(input: &[i32; 1024], dst: &mut [u8], stride: usize) {
    debug_assert!(dst.len() >= stride * 31 + 32, "dst too small for 32x32 add");

    let zero = _mm256_setzero_si256();
    // `l` / `r` hold the two row-pass output strips (left 16 cols' worth
    // of source rows processed as 16-wide instances, then right 16).
    let mut l: [__m256i; 32] = [zero; 32];
    let mut r: [__m256i; 32] = [zero; 32];

    // ---- Row pass: 2 iterations (i = 0 → l, i = 1 → r). Each loads the
    // left 16×16 and right 16×16 sub-block of the next 16 source rows,
    // transposes each in place, then runs the 1-D 32-pt idct over all 32
    // register positions in-place.
    let mut input_ptr = input.as_ptr();
    for iter in 0..2 {
        let dest: &mut [__m256i; 32] = if iter == 0 { &mut l } else { &mut r };

        // in = dest[0..16] (left 16 cols), dest[16..32] (right 16 cols).
        idct_load16x16(input_ptr, &mut dest[0..16], 32);
        transpose_16bit_16x16_avx2(&mut dest[0..16]);
        idct_load16x16(input_ptr.add(16), &mut dest[16..32], 32);
        transpose_16bit_16x16_avx2(&mut dest[16..32]);

        idct32_1024_16x32(dest);

        // libvpx: `input += 32 << 4;` — advance 16 rows of stride-32.
        input_ptr = input_ptr.add(32 * 16);
    }

    // ---- Column pass: 2 iterations (i = 0, 16). Transpose a 16×16
    // sub-block from each of `l` / `r` at offset `i` into `out`, run the
    // 1-D idct32, then store_buffer_16x32 (final +32 rounding, >>6, add).
    let mut dest_ptr = dst.as_mut_ptr();
    let mut out: [__m256i; 32] = [zero; 32];
    let mut i = 0usize;
    while i < 32 {
        transpose_16bit_16x16_avx2_to(&l[i..i + 16], &mut out[0..16]);
        transpose_16bit_16x16_avx2_to(&r[i..i + 16], &mut out[16..32]);

        idct32_1024_16x32(&mut out);
        store_buffer_16x32(&mut out, dest_ptr, stride);

        dest_ptr = dest_ptr.add(16);
        i += 16;
    }
}

/// Safe wrapper for the AVX2 i16-lane dense `idct32x32_1024_add` kernel.
///
/// Pointer-compatible with the scalar reference
/// [`crate::transforms::inverse::idct32x32_1024_add`] and the SSE2
/// wrapper [`super::idct32x32_add_sse2_1024`], so the
/// `SimdDispatch::idct32x32_add` slot can hold any of them.
///
/// `pub` (rather than `pub(crate)`) so unit tests in the same crate can
/// call it directly for differential testing without going through the
/// dispatch.
#[cfg(target_arch = "x86_64")]
pub fn idct32x32_add_avx2_1024(coeffs: &[i32; 1024], dst: &mut [u8], stride: usize) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. The unit-test path gates each call on a
    // runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { idct32x32_add_avx2_1024_inner(coeffs, dst, stride) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Differential / equality tests for `idct32x32_add_avx2_1024`.
    //!
    //! 1. Leaf-helper isolation tests (transpose round-trip, recon add).
    //! 2. SSE2-vs-AVX2 1-D-engine equality over arbitrary i16 lanes.
    //! 3. Outer differential vs the scalar reference (bounded envelopes,
    //!    same as the SSE2 sibling — saturation divergence excluded).
    //! 4. Outer equality vs the SSE2 kernel (any input — both saturate).
    //! 5. Directed fixtures + misaligned-dst guard.

    use super::*;
    use crate::dsp::x86_64::idct32x32_sse2_1024::idct32x32_add_sse2_1024;
    use crate::transforms::inverse::idct32x32_1024_add;
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_setzero_si128, _mm_storeu_si128, _mm256_extracti128_si256,
        _mm256_setzero_si256,
    };
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
        assert!(is_x86_feature_detected!("avx2"));
        let mut dst = *dst_init;
        idct32x32_add_avx2_1024(input, &mut dst, 32);
        dst
    }

    fn run_sse2(input: &[i32; 1024], dst_init: &[u8; 1024]) -> [u8; 1024] {
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
    // Sub-task 1: leaf-helper isolation tests.
    // -----------------------------------------------------------------

    /// Extract the 16 i16 lanes of a ymm into an array (low 128 first,
    /// then high 128) for comparison.
    unsafe fn ymm_to_i16x16(v: __m256i) -> [i16; 16] {
        let mut out = [0i16; 16];
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, lo);
        _mm_storeu_si128(out.as_mut_ptr().add(8) as *mut __m128i, hi);
        out
    }

    unsafe fn i16x16_to_ymm(vals: &[i16; 16]) -> __m256i {
        let lo = _mm_loadu_si128(vals.as_ptr() as *const __m128i);
        let hi = _mm_loadu_si128(vals.as_ptr().add(8) as *const __m128i);
        _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
    }

    /// `transpose_16bit_16x16_avx2` must transpose a 16×16 i16 matrix.
    /// We build a matrix where row r col c = r*16 + c (mod i16) and
    /// verify out[c][r] == in[r][c].
    #[test]
    fn transpose_16x16_roundtrip() {
        require_avx2!();
        unsafe {
            let mut in_: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            let mut expect: [[i16; 16]; 16] = [[0; 16]; 16];
            for (r, slot) in in_.iter_mut().enumerate() {
                let mut row = [0i16; 16];
                for (c, cell) in row.iter_mut().enumerate() {
                    let v = (r as i16) * 16 + c as i16;
                    *cell = v;
                    expect[r][c] = v;
                }
                *slot = i16x16_to_ymm(&row);
            }
            let mut out: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            transpose_16bit_16x16_avx2_to(&in_, &mut out);
            for (r, &orow) in out.iter().enumerate() {
                let got = ymm_to_i16x16(orow);
                for c in 0..16 {
                    // transposed: out[r][c] == in[c][r] == expect[c][r]
                    assert_eq!(
                        got[c], expect[c][r],
                        "transpose mismatch at out[{r}][{c}]"
                    );
                }
            }
        }
    }

    /// Double-transpose must be the identity (composition with the
    /// known-good `transpose2_8x8_avx2` lane logic).
    #[test]
    fn transpose_16x16_double_is_identity() {
        require_avx2!();
        unsafe {
            let mut in_: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            for (r, slot) in in_.iter_mut().enumerate() {
                let mut row = [0i16; 16];
                for (c, cell) in row.iter_mut().enumerate() {
                    *cell = ((r * 37 + c * 7) as i16).wrapping_mul(11) - 3000;
                }
                *slot = i16x16_to_ymm(&row);
            }
            let mut t: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            let mut tt: [__m256i; 16] = [_mm256_setzero_si256(); 16];
            transpose_16bit_16x16_avx2_to(&in_, &mut t);
            transpose_16bit_16x16_avx2_to(&t, &mut tt);
            for r in 0..16 {
                assert_eq!(
                    ymm_to_i16x16(in_[r]),
                    ymm_to_i16x16(tt[r]),
                    "double-transpose not identity at row {r}"
                );
            }
        }
    }

    /// `recon_and_store16` must add 16 i16 residuals to 16 dst bytes
    /// with u8 saturation, in the correct lane order.
    #[test]
    fn recon_and_store16_add_clip() {
        require_avx2!();
        unsafe {
            let mut dst = [0u8; 16];
            for (i, b) in dst.iter_mut().enumerate() {
                *b = (i as u8) * 15 + 5;
            }
            let residual: [i16; 16] = [
                10, -10, 200, -200, 50, 127, -300, 1, 0, 3, -3, 100, -100, 250, -5, 7,
            ];
            let expect: [u8; 16] = core::array::from_fn(|i| {
                (dst[i] as i32 + residual[i] as i32).clamp(0, 255) as u8
            });
            let r = i16x16_to_ymm(&residual);
            recon_and_store16(dst.as_mut_ptr(), r);
            assert_eq!(dst, expect, "recon_and_store16 lane/clip mismatch");
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 2: SSE2-vs-AVX2 1-D-engine equality (arbitrary i16).
    //
    // Feed identical i16 lanes into the SSE2 `idct32_1024_8x32` (run
    // twice — once for the low 8 lanes, once for the high 8 of each
    // ymm) and the AVX2 `idct32_1024_16x32` (once over all 16), then
    // assert the 16 output lanes agree. No saturation bound needed: both
    // engines saturate identically.
    // -----------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 48, max_shrink_iters: 256, .. ProptestConfig::default()
        })]

        #[test]
        fn idct32_1d_avx2_eq_sse2(
            // 32 positions × 16 lanes of arbitrary i16.
            lanes in prop::collection::vec(
                prop::collection::vec(any::<i16>(), 16),
                32,
            ),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            // SAFETY: avx2 gated above; sse2 universal on x86_64.
            unsafe {
                use crate::dsp::x86_64::idct32x32_sse2_1024::idct32_1024_8x32_for_test;

                // Build AVX2 in: 16 lanes per position.
                let mut avx_in: [__m256i; 32] = [_mm256_setzero_si256(); 32];
                let mut lo_in: [__m128i; 32] = [_mm_setzero_si128(); 32];
                let mut hi_in: [__m128i; 32] = [_mm_setzero_si128(); 32];
                for pos in 0..32 {
                    let mut arr = [0i16; 16];
                    arr.copy_from_slice(&lanes[pos]);
                    avx_in[pos] = i16x16_to_ymm(&arr);
                    let mut lo = [0i16; 8];
                    let mut hi = [0i16; 8];
                    lo.copy_from_slice(&arr[0..8]);
                    hi.copy_from_slice(&arr[8..16]);
                    lo_in[pos] = _mm_loadu_si128(lo.as_ptr() as *const __m128i);
                    hi_in[pos] = _mm_loadu_si128(hi.as_ptr() as *const __m128i);
                }

                let mut avx_out: [__m256i; 32] = avx_in;
                idct32_1024_16x32(&mut avx_out);

                let mut lo_out: [__m128i; 32] = [_mm_setzero_si128(); 32];
                let mut hi_out: [__m128i; 32] = [_mm_setzero_si128(); 32];
                idct32_1024_8x32_for_test(&lo_in, &mut lo_out);
                idct32_1024_8x32_for_test(&hi_in, &mut hi_out);

                for pos in 0..32 {
                    let got = ymm_to_i16x16(avx_out[pos]);
                    let mut want_lo = [0i16; 8];
                    let mut want_hi = [0i16; 8];
                    _mm_storeu_si128(want_lo.as_mut_ptr() as *mut __m128i, lo_out[pos]);
                    _mm_storeu_si128(want_hi.as_mut_ptr() as *mut __m128i, hi_out[pos]);
                    for lane in 0..8 {
                        prop_assert_eq!(got[lane], want_lo[lane], "pos {} lo lane {}", pos, lane);
                        prop_assert_eq!(got[lane + 8], want_hi[lane], "pos {} hi lane {}", pos, lane);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Sub-task 3: directed fixtures (bit-exact vs scalar reference).
    // -----------------------------------------------------------------

    #[test]
    fn idct32x32_avx2_all_zero_input() {
        require_avx2!();
        let input = [0i32; 1024];
        let dst_init = [42u8; 1024];
        let avx2 = run_avx2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(avx2, scalar);
        assert_eq!(avx2, dst_init);
    }

    #[test]
    fn idct32x32_avx2_dc_impulse() {
        require_avx2!();
        for &val in &[100i32, -100, 2000, -2000, 1, -1] {
            let mut input = [0i32; 1024];
            input[0] = val;
            let dst_init = [128u8; 1024];
            let avx2 = run_avx2(&input, &dst_init);
            let scalar = run_scalar_oracle(&input, &dst_init);
            assert_eq!(avx2, scalar, "val={val}");
        }
    }

    #[test]
    fn idct32x32_avx2_eob_34() {
        require_avx2!();
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(34).enumerate() {
            *slot = ((i as i32) * 37) - 600;
        }
        let dst_init = [64u8; 1024];
        let avx2 = run_avx2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(avx2, scalar);
    }

    #[test]
    fn idct32x32_avx2_eob_135() {
        require_avx2!();
        let mut input = [0i32; 1024];
        for (i, slot) in input.iter_mut().take(135).enumerate() {
            *slot = ((i as i32) * 19) - 1200;
        }
        let dst_init = [80u8; 1024];
        let avx2 = run_avx2(&input, &dst_init);
        let scalar = run_scalar_oracle(&input, &dst_init);
        assert_eq!(avx2, scalar);
    }

    // -----------------------------------------------------------------
    // Sub-task 4 + 5: outer differential proptests.
    // -----------------------------------------------------------------

    /// `dense_8` — full-density i8 coefficients (same bounded envelope
    /// as the SSE2 sibling — keeps row-pass accumulation inside i16).
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

    /// `eob_bucket_34` — exactly 34 non-zero coefs (first 34 positions).
    fn eob_bucket_34_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-2048i32..=2047, 34).prop_map(|values| {
            let mut out = [0i32; 1024];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
    }

    /// `eob_bucket_135` — exactly 135 i10-bounded coefs (first 135).
    fn eob_bucket_135_strategy() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-512i32..=511, 135).prop_map(|values| {
            let mut out = [0i32; 1024];
            for (i, val) in values.into_iter().enumerate() {
                out[i] = val;
            }
            out
        })
    }

    /// `dense_i12` — full-density i12; used ONLY for the AVX2-vs-SSE2
    /// equality test (both saturate, so any input is legal).
    fn dense_strategy_12() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(-2048i32..=2047, 1024).prop_map(|v| {
            let mut out = [0i32; 1024];
            out.copy_from_slice(&v);
            out
        })
    }

    /// `dense_i16_full` — full-density coefficients spanning the entire
    /// `i16::MIN..=i16::MAX` range. Used ONLY for the AVX2-vs-SSE2 driver
    /// equality test: both kernels share the same flowgraph and saturate
    /// (`packs_epi32`) identically, so any i16 input is legal here. This is
    /// NOT valid against the scalar oracle, which is modular not saturating.
    fn dense_strategy_i16_full() -> impl Strategy<Value = [i32; 1024]> {
        prop::collection::vec(any::<i16>(), 1024).prop_map(|v| {
            let mut out = [0i32; 1024];
            for (slot, &val) in out.iter_mut().zip(v.iter()) {
                *slot = val as i32;
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

        // ---- vs scalar (bounded envelopes, saturation excluded) ----

        #[test]
        fn idct32x32_avx2_eq_scalar_dense_8(
            input in dense_strategy_8(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar_oracle(&input, &dst_init));
        }

        #[test]
        fn idct32x32_avx2_eq_scalar_sparse_12(
            input in sparse_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar_oracle(&input, &dst_init));
        }

        #[test]
        fn idct32x32_avx2_eq_scalar_eob_34(
            input in eob_bucket_34_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar_oracle(&input, &dst_init));
        }

        #[test]
        fn idct32x32_avx2_eq_scalar_eob_135(
            input in eob_bucket_135_strategy(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_scalar_oracle(&input, &dst_init));
        }

        // ---- vs SSE2 kernel (any input — both kernels saturate) ----

        #[test]
        fn idct32x32_avx2_eq_sse2_dense_12(
            input in dense_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_sse2(&input, &dst_init));
        }

        // Full-magnitude i16 driver equality: the SSE2 i16-lane kernel is
        // the saturating oracle (the scalar reference diverges here), and
        // both kernels share the same saturating flowgraph, so any i16
        // coefficient is legal. Widens the `_dense_12` envelope to the full
        // `i16::MIN..=i16::MAX` range to exercise the saturating wraplow on
        // the driver, not just the isolated 1-D engine.
        #[test]
        fn idct32x32_avx2_eq_sse2_dense_i16_full(
            input in dense_strategy_i16_full(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_sse2(&input, &dst_init));
        }

        #[test]
        fn idct32x32_avx2_eq_sse2_sparse_12(
            input in sparse_strategy_12(),
            dst_init in dst_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            prop_assert_eq!(run_avx2(&input, &dst_init), run_sse2(&input, &dst_init));
        }
    }

    // -----------------------------------------------------------------
    // Misaligned dst pointer regression guard.
    // -----------------------------------------------------------------

    #[test]
    fn idct32x32_avx2_misaligned_dst_pointers() {
        require_avx2!();
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

            idct32x32_add_avx2_1024(&input, &mut buf[skew..skew + 1024], 32);
            idct32x32_1024_add(&input, &mut scalar_buf[skew..skew + 1024], 32);

            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}: per-offset arithmetic regression"
            );
        }
    }
}
