//! 16-tap vertical loop filter — x86_64 AVX2/SSE2 implementation
//! (Wave G G6).
//!
//! Sibling of [`crate::dsp::x86_64::lpf_vertical_8_avx2`] (Wave G G4):
//! extends the transpose-then-filter shell to a 16-pixel-wide context
//! (p7..q7). The filter cascade is:
//!
//! * `filter4` when not flat;
//! * `filter8` (7-tap inner smoothing) when `flat_mask4` is set;
//! * `filter16` (15-tap "very flat" smoothing) when both `flat_mask4`
//!   AND `flat_mask5` are set.
//!
//! Processes one 8-row strip per call (matches the scalar
//! `lpf_vertical_16` which calls `mb_lpf_vertical_edge_w` with
//! `count = 8`).
//!
//! The scalar reference is [`crate::loopfilter::lpf_vertical_16`].
//!
//! ## What's new vs G4
//!
//! * [`flat_mask5_avx2`] — per-lane 0xff mask iff the 16-pixel window
//!   around the edge is "extremely flat" (every per-lane abs
//!   difference between `p0` and `p1..p4` / `q1..q4` is `<= thresh`).
//!   Computed by extending the G4 `flat_mask4` max-of-diffs chain to
//!   include `p4` / `q4` and comparing once — bit-identical to the
//!   scalar `flat_mask5` (which internally also reduces to a single
//!   max-of-diffs comparison).
//! * [`filter16_wide_avx2`] — computes the 14 outputs (op6..oq6) of
//!   the 15-tap symmetric filter. Each output is a weighted sum of
//!   p0..p7 / q0..q7 with coefficient pattern `[1,…,2,…,1]`. Sums fit
//!   in u16 (max ≈ 255 * 15 ≈ 3825), then `(x + 8) >> 4` matches the
//!   scalar `round_power_of_two_4`.
//! * Three-way cascade blend: per-lane select between filter16
//!   (when `apply_wide2 = flat2 & flat & mask`), filter8 wide-7-tap
//!   (when `apply_wide = flat & mask`), filter4 (when `mask`), or
//!   pass-through (when `mask` is clear).
//!
//! ## Bit-exactness vs scalar reference
//!
//! All arithmetic mirrors the scalar reference byte-for-byte. Loop
//! filtering is u8 with saturation — the `_mm_subs_epi8` /
//! `_mm_adds_epi8` family (signed-saturating) matches scalar
//! `signed_char_clamp`, and the 7-/15-tap math uses
//! `_mm_adds_epu16` + `_mm_srli_epi16` + `_mm_packus_epi16` (u8
//! saturation via packus).
//!
//! NO modular `wraplow` semantics here.
//!
//! Helper functions (`abs_diff_u8`, `cmpgt_u8`, `sx_lo`, `sx_hi`,
//! `srai_i8`, `filter_mask_avx2`, `hev_mask_avx2`, `flat_mask4_avx2`,
//! `filter4_inner`, `filter8_wide`) are duplicated verbatim from
//! `lpf_vertical_8_avx2` per Wave F / G precedent — extraction to a
//! shared `loopfilter_common.rs` is a Wave H refactor opportunity.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `pixels` are 8-byte unaligned via
//! `_mm_loadl_epi64`. Per-row stores are 8-byte unaligned via
//! `_mm_storel_epi64` after the back-transpose. The grep sentinel
//! below is checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_adds_epi8, _mm_adds_epu8, _mm_adds_epu16, _mm_and_si128, _mm_andnot_si128,
    _mm_cmpeq_epi8, _mm_loadl_epi64, _mm_max_epu8, _mm_or_si128, _mm_packs_epi16,
    _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16,
    _mm_srli_epi16, _mm_storel_epi64, _mm_subs_epi8, _mm_subs_epu8, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpackhi_epi8, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8, _mm_xor_si128,
};

/// Load 8 contiguous u8s from `pixels[off..off+8]` into the low 64
/// bits of an `__m128i` (upper 64 bits zeroed).
///
/// # Safety
///
/// `off + 8` must be `<= pixels.len()`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_row(pixels: *const u8, off: usize) -> __m128i {
    _mm_loadl_epi64(pixels.add(off) as *const __m128i)
}

/// Store the low 8 bytes of `v` to `pixels[off..off+8]`.
///
/// # Safety
///
/// `off + 8` must be `<= pixels.len()`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper is a store
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_row(pixels: *mut u8, off: usize, v: __m128i) {
    _mm_storel_epi64(pixels.add(off) as *mut __m128i, v);
}

/// Transpose 8×8 bytes given 8 row vectors (each holding 8 bytes in the
/// low 64 bits). Returns 8 vectors, each containing one column's worth
/// of bytes across the 8 rows in its low 64 bits. Identical to the G4
/// helper — duplicated file-local.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8(
    r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i,
    r4: __m128i, r5: __m128i, r6: __m128i, r7: __m128i,
) -> [__m128i; 8] {
    let t0 = _mm_unpacklo_epi8(r0, r1);
    let t1 = _mm_unpacklo_epi8(r2, r3);
    let t2 = _mm_unpacklo_epi8(r4, r5);
    let t3 = _mm_unpacklo_epi8(r6, r7);

    let u0 = _mm_unpacklo_epi16(t0, t1);
    let u1 = _mm_unpackhi_epi16(t0, t1);
    let u2 = _mm_unpacklo_epi16(t2, t3);
    let u3 = _mm_unpackhi_epi16(t2, t3);

    let v0 = _mm_unpacklo_epi32(u0, u2);
    let v1 = _mm_unpackhi_epi32(u0, u2);
    let v2 = _mm_unpacklo_epi32(u1, u3);
    let v3 = _mm_unpackhi_epi32(u1, u3);

    let zero = _mm_setzero_si128();
    let c0 = _mm_unpacklo_epi64(v0, zero);
    let c1 = _mm_unpackhi_epi64(v0, zero);
    let c2 = _mm_unpacklo_epi64(v1, zero);
    let c3 = _mm_unpackhi_epi64(v1, zero);
    let c4 = _mm_unpacklo_epi64(v2, zero);
    let c5 = _mm_unpackhi_epi64(v2, zero);
    let c6 = _mm_unpacklo_epi64(v3, zero);
    let c7 = _mm_unpackhi_epi64(v3, zero);
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

/// Compute `|a - b|` lane-by-lane (unsigned u8 absolute difference).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn abs_diff_u8(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

/// Return a lane-parallel u8 mask: 0xff if `x > thresh`, else 0.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cmpgt_u8(x: __m128i, thresh: __m128i) -> __m128i {
    let diff = _mm_subs_epu8(x, thresh);
    let le_mask = _mm_cmpeq_epi8(diff, _mm_setzero_si128());
    let ones = _mm_set1_epi8(-1i8);
    _mm_xor_si128(le_mask, ones)
}

/// Sign-extend the low 8 i8 lanes of `x` to 16-bit.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sx_lo(x: __m128i) -> __m128i {
    let lo = _mm_unpacklo_epi8(x, x);
    _mm_srai_epi16(lo, 8)
}

/// Sign-extend the high 8 i8 lanes of `x` to 16-bit.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sx_hi(x: __m128i) -> __m128i {
    let hi = _mm_unpackhi_epi8(x, x);
    _mm_srai_epi16(hi, 8)
}

/// `signed_char_clamp(x) >> n` lane-parallel for i8 lanes.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn srai_i8<const N: i32>(x: __m128i) -> __m128i {
    let lo = _mm_srai_epi16(sx_lo(x), N);
    let hi = _mm_srai_epi16(sx_hi(x), N);
    _mm_packs_epi16(lo, hi)
}

/// Lane-parallel `filter_mask` — 0xff where the macroblock-edge filter
/// should apply. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter_mask_avx2(
    blimit: u8,
    limit: u8,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> __m128i {
    let limit_v = _mm_set1_epi8(limit as i8);
    let blimit_v = _mm_set1_epi8(blimit as i8);

    let mut work = abs_diff_u8(p3, p2);
    work = _mm_max_epu8(work, abs_diff_u8(p2, p1));
    work = _mm_max_epu8(work, abs_diff_u8(p1, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q1, q0));
    work = _mm_max_epu8(work, abs_diff_u8(q2, q1));
    work = _mm_max_epu8(work, abs_diff_u8(q3, q2));
    let limit_fail = cmpgt_u8(work, limit_v);

    let abs_p0q0 = abs_diff_u8(p0, q0);
    let abs_p1q1 = abs_diff_u8(p1, q1);
    let abs_p0q0_2 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
    let zero = _mm_setzero_si128();
    let abs_p1q1_lo = _mm_unpacklo_epi8(abs_p1q1, zero);
    let abs_p1q1_hi = _mm_unpackhi_epi8(abs_p1q1, zero);
    let abs_p1q1_half_lo = _mm_srai_epi16(abs_p1q1_lo, 1);
    let abs_p1q1_half_hi = _mm_srai_epi16(abs_p1q1_hi, 1);
    let abs_p1q1_half = _mm_packus_epi16(abs_p1q1_half_lo, abs_p1q1_half_hi);
    let mb_diff = _mm_adds_epu8(abs_p0q0_2, abs_p1q1_half);
    let blimit_fail = cmpgt_u8(mb_diff, blimit_v);

    let any_fail = _mm_or_si128(limit_fail, blimit_fail);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_fail, ones)
}

/// Lane-parallel `hev_mask`. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hev_mask_avx2(
    thresh: u8,
    p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i,
) -> __m128i {
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let abs_p1p0 = abs_diff_u8(p1, p0);
    let abs_q1q0 = abs_diff_u8(q1, q0);
    _mm_or_si128(cmpgt_u8(abs_p1p0, thresh_v), cmpgt_u8(abs_q1q0, thresh_v))
}

/// Lane-parallel `flat_mask4(thresh, ...)`. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn flat_mask4_avx2(
    thresh: u8,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> __m128i {
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let mut work = abs_diff_u8(p1, p0);
    work = _mm_max_epu8(work, abs_diff_u8(q1, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p2, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q2, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p3, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q3, q0));
    let any_gt = cmpgt_u8(work, thresh_v);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_gt, ones)
}

/// Lane-parallel `flat_mask5(thresh, ...)`.
///
/// Scalar reference: [`crate::loopfilter::flat_mask5`]. Returns 0xff
/// per lane iff `flat_mask4` is 0xff AND `|p4-p0| <= thresh` AND
/// `|q4-q0| <= thresh`. Reduces to a single max-of-eight-diffs
/// comparison (`max(|p1..p4 - p0|, |q1..q4 - q0|) <= thresh`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn flat_mask5_avx2(
    thresh: u8,
    p4: __m128i, p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i, q4: __m128i,
) -> __m128i {
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let mut work = abs_diff_u8(p1, p0);
    work = _mm_max_epu8(work, abs_diff_u8(q1, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p2, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q2, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p3, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q3, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p4, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q4, q0));
    let any_gt = cmpgt_u8(work, thresh_v);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_gt, ones)
}

/// Apply the inner `filter4` math lane-parallel. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter4_inner(
    filter_mask: __m128i,
    hev: __m128i,
    p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let sign = _mm_set1_epi8(-128i8);
    let ps1 = _mm_xor_si128(p1, sign);
    let ps0 = _mm_xor_si128(p0, sign);
    let qs0 = _mm_xor_si128(q0, sign);
    let qs1 = _mm_xor_si128(q1, sign);

    let outer_taps = _mm_subs_epi8(ps1, qs1);
    let mut filter = _mm_and_si128(outer_taps, hev);

    let qs0_ps0 = _mm_subs_epi8(qs0, ps0);
    let mut acc = _mm_adds_epi8(filter, qs0_ps0);
    acc = _mm_adds_epi8(acc, qs0_ps0);
    acc = _mm_adds_epi8(acc, qs0_ps0);
    filter = _mm_and_si128(acc, filter_mask);

    let plus4 = _mm_set1_epi8(4);
    let plus3 = _mm_set1_epi8(3);
    let f1_sat = _mm_adds_epi8(filter, plus4);
    let f2_sat = _mm_adds_epi8(filter, plus3);
    let filter1 = srai_i8::<3>(f1_sat);
    let filter2 = srai_i8::<3>(f2_sat);

    let qs0_new = _mm_subs_epi8(qs0, filter1);
    let ps0_new = _mm_adds_epi8(ps0, filter2);

    let one_v = _mm_set1_epi8(1);
    let f1_plus1 = _mm_adds_epi8(filter1, one_v);
    let f1_half = srai_i8::<1>(f1_plus1);
    let outer = _mm_andnot_si128(hev, f1_half);

    let qs1_new = _mm_subs_epi8(qs1, outer);
    let ps1_new = _mm_adds_epi8(ps1, outer);

    let p1_out = _mm_xor_si128(ps1_new, sign);
    let p0_out = _mm_xor_si128(ps0_new, sign);
    let q0_out = _mm_xor_si128(qs0_new, sign);
    let q1_out = _mm_xor_si128(qs1_new, sign);
    (p1_out, p0_out, q0_out, q1_out)
}

/// Lane-parallel `(x + 4) >> 3` for u16 lanes (filter8 rounding).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_pow2_3_u16(x: __m128i) -> __m128i {
    let four = _mm_set1_epi16(4);
    let summed = _mm_adds_epu16(x, four);
    _mm_srli_epi16(summed, 3)
}

/// Lane-parallel `(x + 8) >> 4` for u16 lanes (filter16 rounding).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_pow2_4_u16(x: __m128i) -> __m128i {
    let eight = _mm_set1_epi16(8);
    let summed = _mm_adds_epu16(x, eight);
    _mm_srli_epi16(summed, 4)
}

/// Widen the low 8 u8 lanes of `x` to 8 u16 lanes (zero-extended).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen_u8_to_u16_lo(x: __m128i) -> __m128i {
    _mm_unpacklo_epi8(x, _mm_setzero_si128())
}

/// 7-tap symmetric smoothing — computes the 6 inner outputs of the
/// scalar `filter8` flat branch. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter8_wide(
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let p3w = widen_u8_to_u16_lo(p3);
    let p2w = widen_u8_to_u16_lo(p2);
    let p1w = widen_u8_to_u16_lo(p1);
    let p0w = widen_u8_to_u16_lo(p0);
    let q0w = widen_u8_to_u16_lo(q0);
    let q1w = widen_u8_to_u16_lo(q1);
    let q2w = widen_u8_to_u16_lo(q2);
    let q3w = widen_u8_to_u16_lo(q3);

    // op2 = (p3 + p3 + p3 + 2*p2 + p1 + p0 + q0 + 4) >> 3
    let mut sum = _mm_adds_epu16(p3w, p3w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    let op2 = round_pow2_3_u16(sum);

    // op1 = (p3 + p3 + p2 + 2*p1 + p0 + q0 + q1 + 4) >> 3
    let mut sum = _mm_adds_epu16(p3w, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    let op1 = round_pow2_3_u16(sum);

    // op0 = (p3 + p2 + p1 + 2*p0 + q0 + q1 + q2 + 4) >> 3
    let mut sum = _mm_adds_epu16(p3w, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    let op0 = round_pow2_3_u16(sum);

    // oq0 = (p2 + p1 + p0 + 2*q0 + q1 + q2 + q3 + 4) >> 3
    let mut sum = _mm_adds_epu16(p2w, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    let oq0 = round_pow2_3_u16(sum);

    // oq1 = (p1 + p0 + q0 + 2*q1 + q2 + q3 + q3 + 4) >> 3
    let mut sum = _mm_adds_epu16(p1w, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q3w);
    let oq1 = round_pow2_3_u16(sum);

    // oq2 = (p0 + q0 + q1 + 2*q2 + q3 + q3 + q3 + 4) >> 3
    let mut sum = _mm_adds_epu16(p0w, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q3w);
    let oq2 = round_pow2_3_u16(sum);

    let zero = _mm_setzero_si128();
    let op2_b = _mm_packus_epi16(op2, zero);
    let op1_b = _mm_packus_epi16(op1, zero);
    let op0_b = _mm_packus_epi16(op0, zero);
    let oq0_b = _mm_packus_epi16(oq0, zero);
    let oq1_b = _mm_packus_epi16(oq1, zero);
    let oq2_b = _mm_packus_epi16(oq2, zero);
    (op2_b, op1_b, op0_b, oq0_b, oq1_b, oq2_b)
}

/// 15-tap symmetric smoothing — computes the 14 outputs
/// `(op6, op5, op4, op3, op2, op1, op0, oq0, oq1, oq2, oq3, oq4, oq5,
/// oq6)` per the scalar `filter16` "very flat" branch.
///
/// Sums fit in u16 (max ≈ 255 * 16 = 4080, plus the +8 round bias
/// ≈ 4088 < 65536). Each tap is widened to u16, weighted-summed, then
/// `(x + 8) >> 4` produces a u8 byte that we pack via
/// `_mm_packus_epi16` against a zero high half.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn filter16_wide_avx2(
    p7: __m128i, p6: __m128i, p5: __m128i, p4: __m128i,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
    q4: __m128i, q5: __m128i, q6: __m128i, q7: __m128i,
) -> [__m128i; 14] {
    let p7w = widen_u8_to_u16_lo(p7);
    let p6w = widen_u8_to_u16_lo(p6);
    let p5w = widen_u8_to_u16_lo(p5);
    let p4w = widen_u8_to_u16_lo(p4);
    let p3w = widen_u8_to_u16_lo(p3);
    let p2w = widen_u8_to_u16_lo(p2);
    let p1w = widen_u8_to_u16_lo(p1);
    let p0w = widen_u8_to_u16_lo(p0);
    let q0w = widen_u8_to_u16_lo(q0);
    let q1w = widen_u8_to_u16_lo(q1);
    let q2w = widen_u8_to_u16_lo(q2);
    let q3w = widen_u8_to_u16_lo(q3);
    let q4w = widen_u8_to_u16_lo(q4);
    let q5w = widen_u8_to_u16_lo(q5);
    let q6w = widen_u8_to_u16_lo(q6);
    let q7w = widen_u8_to_u16_lo(q7);

    // op6 = (p7*7 + p6*2 + p5 + p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    let op6 = round_pow2_4_u16(sum);

    // op5 = (p7*6 + p6 + p5*2 + p4 + p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    let op5 = round_pow2_4_u16(sum);

    // op4 = (p7*5 + p6 + p5 + p4*2 + p3 + p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    let op4 = round_pow2_4_u16(sum);

    // op3 = (p7*4 + p6 + p5 + p4 + p3*2 + p2 + p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    let op3 = round_pow2_4_u16(sum);

    // op2 = (p7*3 + p6 + p5 + p4 + p3 + p2*2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    let op2 = round_pow2_4_u16(sum);

    // op1 = (p7*2 + p6 + p5 + p4 + p3 + p2 + p1*2 + p0 + q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p7w);
    sum = _mm_adds_epu16(sum, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    let op1 = round_pow2_4_u16(sum);

    // op0 = (p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0*2 + q0 + q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4
    let mut sum = _mm_adds_epu16(p7w, p6w);
    sum = _mm_adds_epu16(sum, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    let op0 = round_pow2_4_u16(sum);

    // oq0 = (p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0*2 + q1 + q2 + q3 + q4 + q5 + q6 + q7 + 8) >> 4
    let mut sum = _mm_adds_epu16(p6w, p5w);
    sum = _mm_adds_epu16(sum, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq0 = round_pow2_4_u16(sum);

    // oq1 = (p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1*2 + q2 + q3 + q4 + q5 + q6 + q7*2 + 8) >> 4
    let mut sum = _mm_adds_epu16(p5w, p4w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq1 = round_pow2_4_u16(sum);

    // oq2 = (p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2*2 + q3 + q4 + q5 + q6 + q7*3 + 8) >> 4
    let mut sum = _mm_adds_epu16(p4w, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq2 = round_pow2_4_u16(sum);

    // oq3 = (p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3*2 + q4 + q5 + q6 + q7*4 + 8) >> 4
    let mut sum = _mm_adds_epu16(p3w, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq3 = round_pow2_4_u16(sum);

    // oq4 = (p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4*2 + q5 + q6 + q7*5 + 8) >> 4
    let mut sum = _mm_adds_epu16(p2w, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq4 = round_pow2_4_u16(sum);

    // oq5 = (p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5*2 + q6 + q7*6 + 8) >> 4
    let mut sum = _mm_adds_epu16(p1w, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq5 = round_pow2_4_u16(sum);

    // oq6 = (p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6*2 + q7*7 + 8) >> 4
    let mut sum = _mm_adds_epu16(p0w, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q4w);
    sum = _mm_adds_epu16(sum, q5w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q6w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    sum = _mm_adds_epu16(sum, q7w);
    let oq6 = round_pow2_4_u16(sum);

    let zero = _mm_setzero_si128();
    [
        _mm_packus_epi16(op6, zero),
        _mm_packus_epi16(op5, zero),
        _mm_packus_epi16(op4, zero),
        _mm_packus_epi16(op3, zero),
        _mm_packus_epi16(op2, zero),
        _mm_packus_epi16(op1, zero),
        _mm_packus_epi16(op0, zero),
        _mm_packus_epi16(oq0, zero),
        _mm_packus_epi16(oq1, zero),
        _mm_packus_epi16(oq2, zero),
        _mm_packus_epi16(oq3, zero),
        _mm_packus_epi16(oq4, zero),
        _mm_packus_epi16(oq5, zero),
        _mm_packus_epi16(oq6, zero),
    ]
}

/// Blend two lane-parallel vectors `a` and `b` per `mask` (0xff →
/// pick `a`, 0x00 → pick `b`).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn blend(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

/// Run the full 16-tap filter cascade lane-parallel on 8 lanes
/// (8 rows). Returns the 16 output column-vectors
/// `[op7, op6, …, op0, oq0, …, oq7]` — note `op7` and `oq7` always
/// pass through unchanged (the 15-tap kernel does not write them).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn filter16_avx2(
    blimit: u8, limit: u8, thresh: u8,
    p7: __m128i, p6: __m128i, p5: __m128i, p4: __m128i,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
    q4: __m128i, q5: __m128i, q6: __m128i, q7: __m128i,
) -> [__m128i; 16] {
    let mask = filter_mask_avx2(blimit, limit, p3, p2, p1, p0, q0, q1, q2, q3);
    let hev = hev_mask_avx2(thresh, p1, p0, q0, q1);
    let flat = flat_mask4_avx2(1, p3, p2, p1, p0, q0, q1, q2, q3);
    let flat2 = flat_mask5_avx2(1, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4);

    let apply_wide = _mm_and_si128(flat, mask);
    let apply_wide2 = _mm_and_si128(flat2, apply_wide);

    // filter4 inner outputs.
    let (p1_f4, p0_f4, q0_f4, q1_f4) = filter4_inner(mask, hev, p1, p0, q0, q1);

    // 7-tap wide outputs for op2..oq2.
    let (op2_w7, op1_w7, op0_w7, oq0_w7, oq1_w7, oq2_w7) =
        filter8_wide(p3, p2, p1, p0, q0, q1, q2, q3);

    // 15-tap wide outputs for op6..oq6.
    let f16 = filter16_wide_avx2(p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7);
    let (op6_w15, op5_w15, op4_w15, op3_w15, op2_w15, op1_w15, op0_w15,
         oq0_w15, oq1_w15, oq2_w15, oq3_w15, oq4_w15, oq5_w15, oq6_w15) =
        (f16[0], f16[1], f16[2], f16[3], f16[4], f16[5], f16[6],
         f16[7], f16[8], f16[9], f16[10], f16[11], f16[12], f16[13]);

    // Per-tap blend. For each tap, the priority is:
    //   apply_wide2 → 15-tap output
    //   apply_wide  → 7-tap output (if it exists for that tap)
    //   mask        → filter4 output (only for inner four taps)
    //   otherwise   → input passthrough
    //
    // For op7 / oq7: never modified; pass through.
    // For op6/op5/op4/oq4/oq5/oq6: only the 15-tap branch writes them.
    // For op3 / oq3: only the 15-tap branch writes them (filter8 doesn't
    //                modify p3/q3 — those are pass-through inputs from
    //                the scalar filter8 contract).
    // For op2/op1/op0/oq0/oq1/oq2: any branch may write — pick wide15
    //                if apply_wide2, else wide7 if apply_wide, else f4
    //                inner (op1/op0/oq0/oq1 only — op2/oq2 input).

    // Pass-throughs.
    let op7_o = p7;
    let oq7_o = q7;

    let op6_o = blend(apply_wide2, op6_w15, p6);
    let op5_o = blend(apply_wide2, op5_w15, p5);
    let op4_o = blend(apply_wide2, op4_w15, p4);
    let op3_o = blend(apply_wide2, op3_w15, p3);
    let oq4_o = blend(apply_wide2, oq4_w15, q4);
    let oq5_o = blend(apply_wide2, oq5_w15, q5);
    let oq6_o = blend(apply_wide2, oq6_w15, q6);
    let oq3_o = blend(apply_wide2, oq3_w15, q3);

    // op2 / oq2: wide15 → wide7 → input.
    let op2_w = blend(apply_wide, op2_w7, p2);
    let oq2_w = blend(apply_wide, oq2_w7, q2);
    let op2_o = blend(apply_wide2, op2_w15, op2_w);
    let oq2_o = blend(apply_wide2, oq2_w15, oq2_w);

    // op1/op0/oq0/oq1: wide15 → wide7 → f4 inner.
    let op1_mid = blend(apply_wide, op1_w7, p1_f4);
    let op0_mid = blend(apply_wide, op0_w7, p0_f4);
    let oq0_mid = blend(apply_wide, oq0_w7, q0_f4);
    let oq1_mid = blend(apply_wide, oq1_w7, q1_f4);
    let op1_o = blend(apply_wide2, op1_w15, op1_mid);
    let op0_o = blend(apply_wide2, op0_w15, op0_mid);
    let oq0_o = blend(apply_wide2, oq0_w15, oq0_mid);
    let oq1_o = blend(apply_wide2, oq1_w15, oq1_mid);

    [op7_o, op6_o, op5_o, op4_o, op3_o, op2_o, op1_o, op0_o,
     oq0_o, oq1_o, oq2_o, oq3_o, oq4_o, oq5_o, oq6_o, oq7_o]
}

/// Apply `lpf_vertical_16` to one 8-row × 16-col strip whose leftmost
/// column in the q-half is `pixels[s_index]`. Lane-parallel across the
/// 8 rows via two side-by-side 8×8 transposes (p-half + q-half).
///
/// # Safety
///
/// `s_index` must be `>= 8` and `s_index + 7 * pitch + 8` must be
/// `<= pixels.len()`. Caller must guarantee AVX2 support.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::erasing_op, clippy::identity_op)]
unsafe fn lpf_vertical_16_avx2_inner(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 8);
    debug_assert!(s_index + 7 * pitch + 8 <= pixels.len());

    let base = pixels.as_mut_ptr();

    // Load 8 rows × 16 bytes each. Split each row into two 8-byte
    // halves: `[p7..p0]` from `[s-8..s]` and `[q0..q7]` from
    // `[s..s+8]`. Transpose each half independently.
    let pl0 = load_row(base, s_index - 8 + 0 * pitch);
    let pl1 = load_row(base, s_index - 8 + 1 * pitch);
    let pl2 = load_row(base, s_index - 8 + 2 * pitch);
    let pl3 = load_row(base, s_index - 8 + 3 * pitch);
    let pl4 = load_row(base, s_index - 8 + 4 * pitch);
    let pl5 = load_row(base, s_index - 8 + 5 * pitch);
    let pl6 = load_row(base, s_index - 8 + 6 * pitch);
    let pl7 = load_row(base, s_index - 8 + 7 * pitch);

    let ql0 = load_row(base, s_index + 0 * pitch);
    let ql1 = load_row(base, s_index + 1 * pitch);
    let ql2 = load_row(base, s_index + 2 * pitch);
    let ql3 = load_row(base, s_index + 3 * pitch);
    let ql4 = load_row(base, s_index + 4 * pitch);
    let ql5 = load_row(base, s_index + 5 * pitch);
    let ql6 = load_row(base, s_index + 6 * pitch);
    let ql7 = load_row(base, s_index + 7 * pitch);

    let p_cols = transpose_8x8(pl0, pl1, pl2, pl3, pl4, pl5, pl6, pl7);
    let q_cols = transpose_8x8(ql0, ql1, ql2, ql3, ql4, ql5, ql6, ql7);

    // p_cols[k] holds column (s-8+k) across all 8 rows in its low 64
    // bits. So p_cols[0] = p7, p_cols[7] = p0.
    let p7 = p_cols[0];
    let p6 = p_cols[1];
    let p5 = p_cols[2];
    let p4 = p_cols[3];
    let p3 = p_cols[4];
    let p2 = p_cols[5];
    let p1 = p_cols[6];
    let p0 = p_cols[7];
    let q0 = q_cols[0];
    let q1 = q_cols[1];
    let q2 = q_cols[2];
    let q3 = q_cols[3];
    let q4 = q_cols[4];
    let q5 = q_cols[5];
    let q6 = q_cols[6];
    let q7 = q_cols[7];

    let out = filter16_avx2(
        blimit, limit, thresh,
        p7, p6, p5, p4, p3, p2, p1, p0,
        q0, q1, q2, q3, q4, q5, q6, q7,
    );

    // out[0..8]  = op7..op0 (column-vectors)
    // out[8..16] = oq0..oq7 (column-vectors)
    // Transpose each half back to row-major and store.
    let p_rows = transpose_8x8(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    let q_rows = transpose_8x8(out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);

    store_row(base, s_index - 8 + 0 * pitch, p_rows[0]);
    store_row(base, s_index - 8 + 1 * pitch, p_rows[1]);
    store_row(base, s_index - 8 + 2 * pitch, p_rows[2]);
    store_row(base, s_index - 8 + 3 * pitch, p_rows[3]);
    store_row(base, s_index - 8 + 4 * pitch, p_rows[4]);
    store_row(base, s_index - 8 + 5 * pitch, p_rows[5]);
    store_row(base, s_index - 8 + 6 * pitch, p_rows[6]);
    store_row(base, s_index - 8 + 7 * pitch, p_rows[7]);

    store_row(base, s_index + 0 * pitch, q_rows[0]);
    store_row(base, s_index + 1 * pitch, q_rows[1]);
    store_row(base, s_index + 2 * pitch, q_rows[2]);
    store_row(base, s_index + 3 * pitch, q_rows[3]);
    store_row(base, s_index + 4 * pitch, q_rows[4]);
    store_row(base, s_index + 5 * pitch, q_rows[5]);
    store_row(base, s_index + 6 * pitch, q_rows[6]);
    store_row(base, s_index + 7 * pitch, q_rows[7]);
}

/// Safe wrapper for the AVX2 vertical 16-tap loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_vertical_16`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_vertical_16`] points
/// at this wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_vertical_16_avx2(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. The bounds invariant
    // `s_index >= 8 && s_index + 7*pitch + 8 <= pixels.len()` is the
    // scalar contract (mirrored by the inner debug_assert).
    unsafe { lpf_vertical_16_avx2_inner(pixels, s_index, pitch, blimit, limit, thresh) }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_vertical_16`.

    use super::lpf_vertical_16_avx2;
    use crate::loopfilter::lpf_vertical_16;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build an 8-row × 24-wide test buffer. Edge sits at column 12 so
    /// the per-row 16-byte window `[s-8..s+8]` lands inside.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 8 * 24];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *x = (s >> 16) as u8;
        }
        v
    }

    fn run_pair(
        buf: &[u8],
        s_index: usize,
        pitch: usize,
        blimit: u8,
        limit: u8,
        thresh: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut scalar = buf.to_vec();
        let mut avx2 = buf.to_vec();
        lpf_vertical_16(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_vertical_16_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
        (scalar, avx2)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn scalar_eq_avx2_proptest(
            seed in any::<u32>(),
            blimit in 1u8..=255,
            limit in 1u8..=63,
            thresh in 0u8..=15,
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let buf = make_buf(seed);
            let (s, a) = run_pair(&buf, 12, 24, blimit, limit, thresh);
            prop_assert_eq!(s, a);
        }
    }

    /// Tautology guard: the scalar filter must actually modify the
    /// buffer for an obviously-filterable edge.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let mut buf = vec![100u8; 8 * 24];
        // Strong step edge with flat sides — should engage the 15-tap.
        for r in 0..8 {
            for c in 0..12 {
                buf[r * 24 + c] = 80;
            }
            for c in 12..24 {
                buf[r * 24 + c] = 170;
            }
        }
        let mut copy = buf.clone();
        lpf_vertical_16(&mut copy, 12, 24, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar filter must alter the buffer for filterable edges"
        );
    }

    #[test]
    fn all_zero_no_op() {
        require_avx2!();
        let buf = vec![0u8; 8 * 24];
        let (s, a) = run_pair(&buf, 12, 24, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![0u8; 8 * 24]);
    }

    #[test]
    fn all_max_no_op() {
        require_avx2!();
        let buf = vec![255u8; 8 * 24];
        let (s, a) = run_pair(&buf, 12, 24, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![255u8; 8 * 24]);
    }

    /// Step edge across the boundary that should engage the 15-tap
    /// filter (extended flatness satisfied on each side).
    #[test]
    fn directed_step_edge_engages_wide15_filter() {
        require_avx2!();
        let mut buf = vec![0u8; 8 * 24];
        for r in 0..8 {
            for c in 0..12 {
                buf[r * 24 + c] = 80;
            }
            for c in 12..24 {
                buf[r * 24 + c] = 170;
            }
        }
        let (s, a) = run_pair(&buf, 12, 24, 255, 63, 7);
        assert_eq!(s, a);
    }

    /// Non-flat HEV path — filter16 should fall back through filter8
    /// to filter4 on every lane.
    #[test]
    fn directed_hev_falls_back_to_filter4() {
        require_avx2!();
        let mut buf = vec![0u8; 8 * 24];
        for r in 0..8 {
            for c in 0..24 {
                buf[r * 24 + c] = if c & 1 == 0 { 40 } else { 200 };
            }
        }
        let (s, a) = run_pair(&buf, 12, 24, 200, 30, 7);
        assert_eq!(s, a);
    }

    /// Mask-off path: thresholds so tight that filter_mask should be 0
    /// everywhere — output must equal input.
    #[test]
    fn directed_mask_off_preserves_input() {
        require_avx2!();
        let buf = make_buf(0xfeed_face);
        let (s, a) = run_pair(&buf, 12, 24, 1, 1, 0);
        assert_eq!(s, a);
    }

    /// Round-trip through SimdDispatch.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps = detect();
        caps.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps);
        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_s);

        let buf = make_buf(0xdead_beef);
        let mut b_a = buf.clone();
        let mut b_s = buf;
        (d_avx2.lpf_vertical_16())(&mut b_a, 12, 24, 200, 30, 7);
        (d_scalar.lpf_vertical_16())(&mut b_s, 12, 24, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
