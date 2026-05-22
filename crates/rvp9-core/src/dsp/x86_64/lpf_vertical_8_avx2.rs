//! 8-tap vertical loop filter — x86_64 AVX2/SSE2 implementation
//! (Wave G G4).
//!
//! Sibling of [`crate::dsp::x86_64::lpf_vertical_4_avx2`] (Wave F G F10):
//! same transpose-then-filter shell, but the inner filter is the wider
//! `filter8` (7-tap symmetric smoothing for flat areas + per-lane
//! fallback to `filter4` for non-flat areas) gated by [`flat_mask4`].
//! Eight pixels per row (p3..q3) are modified instead of four.
//!
//! The scalar reference is [`crate::loopfilter::lpf_vertical_8`].
//!
//! ## What's new vs F10
//!
//! * [`flat_mask4_avx2`] — per-lane 0xff mask iff the 8 pixels around
//!   the edge are "flat enough" (all per-lane abs differences `<= 1`).
//!   When set for a lane, that lane gets the wider 7-tap smoothing;
//!   when clear, filter4 fallback.
//! * [`filter8_avx2`] — computes BOTH the filter4 outputs and the
//!   wider 7-tap outputs per lane, blends with `flat & mask`. The
//!   7-tap math is u8-saturated arithmetic — widened to u16 (sums fit
//!   in u16 trivially: max is 255 * 7 ≈ 1785), `(x + 4) >> 3`, packed
//!   back to u8 via `_mm_packus_epi16`.
//!
//! ## Bit-exactness vs scalar reference
//!
//! All arithmetic mirrors the scalar reference byte-for-byte. Loop
//! filtering is u8 with saturation — the `_mm_subs_epi8` /
//! `_mm_adds_epi8` family (signed-saturating) matches scalar
//! `signed_char_clamp`, and the 7-tap math uses `_mm_adds_epu16` +
//! `_mm_srli_epi16` + `_mm_packus_epi16` (u8 saturation via packus).
//!
//! NO modular `wraplow` semantics here.
//!
//! Helper functions (`abs_diff_u8`, `cmpgt_u8`, `sx_lo`, `sx_hi`,
//! `srai_i8`, `filter_mask_avx2`, `hev_mask_avx2`, `filter4_avx2`)
//! are duplicated verbatim from `lpf_vertical_4_avx2` per Wave F
//! precedent — extraction to a shared `loopfilter_common.rs` is a
//! Wave H refactor opportunity.
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
/// of bytes across the 8 rows in its low 64 bits.
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

/// `signed_char_clamp(x) >> n` lane-parallel for i8 lanes, returning
/// i8 lanes. `n` must be in `0..=15`.
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
/// should apply, 0 otherwise. Mirrors scalar `filter_mask`.
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

/// Lane-parallel `hev_mask` — 0xff where either inner-tap difference
/// exceeds `thresh`. Mirrors scalar `hev_mask`.
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

/// Lane-parallel `flat_mask4(1, ...)` — 0xff where every per-lane
/// abs difference between `p0` and each of `p1, p2, p3, q1, q2, q3`
/// (and the symmetric set on the q side) is `<= 1`. The scalar
/// reference is called with `thresh = 1` from `lpf_*_8`.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn flat_mask4_avx2(
    thresh: u8,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> __m128i {
    // scalar: mask = -1 if any of these > thresh; flat = NOT(mask).
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let mut work = abs_diff_u8(p1, p0);
    work = _mm_max_epu8(work, abs_diff_u8(q1, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p2, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q2, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p3, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q3, q0));
    // flat = NOT(any > thresh) = (max <= thresh).
    let any_gt = cmpgt_u8(work, thresh_v);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_gt, ones)
}

/// Apply the inner `filter4` math lane-parallel. Returns
/// `(p1_out, p0_out, q0_out, q1_out)` in signed-centered domain
/// **unchanged** (still XORed with 0x80 — caller flips back). Mirrors
/// the F10 / G3 inner sequence.
///
/// `filter_mask` and `hev` are the masks computed by
/// [`filter_mask_avx2`] / [`hev_mask_avx2`].
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

/// Lane-parallel `(x + 4) >> 3` for u16 lanes. Sum inputs are
/// guaranteed `<= 255 * 7 + 255 = 2040 < 65536`, so u16 arithmetic
/// cannot overflow.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_pow2_3_u16(x: __m128i) -> __m128i {
    let four = _mm_set1_epi16(4);
    let summed = _mm_adds_epu16(x, four);
    _mm_srli_epi16(summed, 3)
}

/// Widen the low 8 u8 lanes of `x` to 8 u16 lanes (zero-extended).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen_u8_to_u16_lo(x: __m128i) -> __m128i {
    _mm_unpacklo_epi8(x, _mm_setzero_si128())
}

/// 7-tap symmetric smoothing — computes the 6 inner outputs
/// `(op2, op1, op0, oq0, oq1, oq2)` per the scalar `filter8` flat
/// branch. Output is u8 lanes in the low 64 bits of each vector.
///
/// Sums fit in u16 (max ≈ 255 * 9 = 2295). We widen each tap to u16,
/// compute the 6 weighted sums, then `(x + 4) >> 3`, then pack u16
/// back to u8 via `_mm_packus_epi16` against a zero high half.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter8_wide(
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // Widen each tap to u16. Only the low 8 lanes carry data per the
    // 8-row layout.
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

/// Lane-parallel `filter8`: returns the six modified middle taps
/// `(op2, op1, op0, oq0, oq1, oq2)`. The outer taps `op3` / `oq3` are
/// unchanged (caller writes back the originals).
///
/// Per scalar contract:
/// * lanes with `flat & filter_mask` set → wide 7-tap output
/// * lanes with `filter_mask` set but `flat` clear → filter4 fallback
///   (only the inner four taps op1/op0/oq0/oq1 change; op2 / oq2
///   pass through unchanged)
/// * lanes with `filter_mask` clear → all taps unchanged (filter4
///   internally ANDs with `filter_mask`, so its outputs equal the
///   inputs)
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn filter8_avx2(
    filter_mask: __m128i,
    hev: __m128i,
    flat: __m128i,
    p3: __m128i, p2: __m128i, p1: __m128i, p0: __m128i,
    q0: __m128i, q1: __m128i, q2: __m128i, q3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    // Apply-wide mask: lanes where flat AND mask are both 0xff.
    let apply_wide = _mm_and_si128(flat, filter_mask);

    // filter4 outputs for inner four taps.
    let (p1_f4, p0_f4, q0_f4, q1_f4) = filter4_inner(filter_mask, hev, p1, p0, q0, q1);

    // 7-tap wider outputs.
    let (op2_w, op1_w, op0_w, oq0_w, oq1_w, oq2_w) =
        filter8_wide(p3, p2, p1, p0, q0, q1, q2, q3);

    // Blend per lane:
    //   wide lanes -> 7-tap; else -> filter4 (for inner four),
    //   pass-through p2/q2 unchanged (filter4 fallback doesn't
    //   touch them).
    let op2 = _mm_or_si128(
        _mm_and_si128(apply_wide, op2_w),
        _mm_andnot_si128(apply_wide, p2),
    );
    let op1 = _mm_or_si128(
        _mm_and_si128(apply_wide, op1_w),
        _mm_andnot_si128(apply_wide, p1_f4),
    );
    let op0 = _mm_or_si128(
        _mm_and_si128(apply_wide, op0_w),
        _mm_andnot_si128(apply_wide, p0_f4),
    );
    let oq0 = _mm_or_si128(
        _mm_and_si128(apply_wide, oq0_w),
        _mm_andnot_si128(apply_wide, q0_f4),
    );
    let oq1 = _mm_or_si128(
        _mm_and_si128(apply_wide, oq1_w),
        _mm_andnot_si128(apply_wide, q1_f4),
    );
    let oq2 = _mm_or_si128(
        _mm_and_si128(apply_wide, oq2_w),
        _mm_andnot_si128(apply_wide, q2),
    );
    (op2, op1, op0, oq0, oq1, oq2)
}

/// Apply `lpf_vertical_8` to one 8-row × 8-col strip whose leftmost
/// column in the q-half is `pixels[s_index]`. Lane-parallel across the
/// 8 rows via the same transpose pattern as F10.
///
/// # Safety
///
/// `s_index` must be `>= 4` and `s_index + 7 * pitch + 4` must be
/// `<= pixels.len()`. Caller must guarantee AVX2 support.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::erasing_op, clippy::identity_op)]
unsafe fn lpf_vertical_8_avx2_inner(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 4);
    debug_assert!(s_index + 7 * pitch + 4 <= pixels.len());

    let base = pixels.as_mut_ptr();
    // Load 8 rows × 8 bytes each, starting at `s_index - 4` so bytes
    // are `[p3, p2, p1, p0, q0, q1, q2, q3]` per row.
    let r0 = load_row(base, s_index - 4 + 0 * pitch);
    let r1 = load_row(base, s_index - 4 + 1 * pitch);
    let r2 = load_row(base, s_index - 4 + 2 * pitch);
    let r3 = load_row(base, s_index - 4 + 3 * pitch);
    let r4 = load_row(base, s_index - 4 + 4 * pitch);
    let r5 = load_row(base, s_index - 4 + 5 * pitch);
    let r6 = load_row(base, s_index - 4 + 6 * pitch);
    let r7 = load_row(base, s_index - 4 + 7 * pitch);

    // Transpose so each vector holds one column across all 8 rows.
    let cols = transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);
    let p3 = cols[0];
    let p2 = cols[1];
    let p1 = cols[2];
    let p0 = cols[3];
    let q0 = cols[4];
    let q1 = cols[5];
    let q2 = cols[6];
    let q3 = cols[7];

    let mask = filter_mask_avx2(blimit, limit, p3, p2, p1, p0, q0, q1, q2, q3);
    let hev = hev_mask_avx2(thresh, p1, p0, q0, q1);
    let flat = flat_mask4_avx2(1, p3, p2, p1, p0, q0, q1, q2, q3);

    let (op2, op1, op0, oq0, oq1, oq2) =
        filter8_avx2(mask, hev, flat, p3, p2, p1, p0, q0, q1, q2, q3);

    // The 8 output columns per row are:
    //   [p3 (unchanged), op2, op1, op0, oq0, oq1, oq2, q3 (unchanged)]
    // Transpose back to row-major 8-byte rows and store.
    let cols_out: [__m128i; 8] = [p3, op2, op1, op0, oq0, oq1, oq2, q3];

    // Each `cols_out[i]` holds 8 column-i bytes (one per row) in its
    // low 64 bits. We need to rebuild 8 vectors where row j holds the
    // 8 bytes [c0[j], c1[j], ..., c7[j]]. The 8x8 transpose is its
    // own inverse — apply the same transpose helper.
    let rows = transpose_8x8(
        cols_out[0], cols_out[1], cols_out[2], cols_out[3],
        cols_out[4], cols_out[5], cols_out[6], cols_out[7],
    );

    // Store each row's 8 bytes at the original load offset.
    store_row(base, s_index - 4 + 0 * pitch, rows[0]);
    store_row(base, s_index - 4 + 1 * pitch, rows[1]);
    store_row(base, s_index - 4 + 2 * pitch, rows[2]);
    store_row(base, s_index - 4 + 3 * pitch, rows[3]);
    store_row(base, s_index - 4 + 4 * pitch, rows[4]);
    store_row(base, s_index - 4 + 5 * pitch, rows[5]);
    store_row(base, s_index - 4 + 6 * pitch, rows[6]);
    store_row(base, s_index - 4 + 7 * pitch, rows[7]);
}

/// Safe wrapper for the AVX2 vertical 8-tap loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_vertical_8`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_vertical_8`] points at
/// this wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_vertical_8_avx2(
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
    // `s_index >= 4 && s_index + 7*pitch + 4 <= pixels.len()` is the
    // scalar contract (mirrored by the inner debug_assert).
    unsafe { lpf_vertical_8_avx2_inner(pixels, s_index, pitch, blimit, limit, thresh) }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_vertical_8`.

    use super::lpf_vertical_8_avx2;
    use crate::loopfilter::lpf_vertical_8;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build an 8-row × 16-wide test buffer.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 8 * 16];
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
        lpf_vertical_8(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_vertical_8_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
        (scalar, avx2)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Differential proptest over random byte fills and reasonable
        /// thresholds.
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
            let (s, a) = run_pair(&buf, 8, 16, blimit, limit, thresh);
            prop_assert_eq!(s, a);
        }
    }

    /// Tautology guard: the scalar filter must actually modify the
    /// buffer for an obviously-filterable edge.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let mut buf = vec![100u8; 8 * 16];
        for r in 0..8 {
            buf[r * 16 + 4] = 98;
            buf[r * 16 + 5] = 99;
            buf[r * 16 + 6] = 100;
            buf[r * 16 + 7] = 101;
            buf[r * 16 + 8] = 110;
            buf[r * 16 + 9] = 109;
            buf[r * 16 + 10] = 108;
            buf[r * 16 + 11] = 107;
        }
        let mut copy = buf.clone();
        lpf_vertical_8(&mut copy, 8, 16, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar filter must alter the buffer for filterable edges"
        );
    }

    #[test]
    fn all_zero_no_op() {
        require_avx2!();
        let buf = vec![0u8; 8 * 16];
        let (s, a) = run_pair(&buf, 8, 16, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![0u8; 8 * 16]);
    }

    #[test]
    fn all_max_no_op() {
        require_avx2!();
        let buf = vec![255u8; 8 * 16];
        let (s, a) = run_pair(&buf, 8, 16, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![255u8; 8 * 16]);
    }

    /// Step edge across the boundary that should engage the wider
    /// filter (flat-mask satisfied on each side).
    #[test]
    fn directed_step_edge_engages_wide_filter() {
        require_avx2!();
        let mut buf = vec![0u8; 8 * 16];
        for r in 0..8 {
            for c in 0..8 {
                buf[r * 16 + c] = 80;
                buf[r * 16 + 8 + c] = 170;
            }
        }
        let (s, a) = run_pair(&buf, 8, 16, 255, 63, 7);
        assert_eq!(s, a);
    }

    /// Non-flat HEV path — filter8 should fall back to filter4 on
    /// every lane (high-variance taps prevent the wider smoothing).
    #[test]
    fn directed_hev_falls_back_to_filter4() {
        require_avx2!();
        let mut buf = vec![0u8; 8 * 16];
        for r in 0..8 {
            buf[r * 16 + 4] = 40;
            buf[r * 16 + 5] = 200;
            buf[r * 16 + 6] = 40;
            buf[r * 16 + 7] = 200;
            buf[r * 16 + 8] = 50;
            buf[r * 16 + 9] = 210;
            buf[r * 16 + 10] = 50;
            buf[r * 16 + 11] = 210;
        }
        let (s, a) = run_pair(&buf, 8, 16, 200, 30, 7);
        assert_eq!(s, a);
    }

    /// Mask-off path: thresholds so tight that filter_mask should be 0
    /// everywhere — output must equal input.
    #[test]
    fn directed_mask_off_preserves_input() {
        require_avx2!();
        let buf = make_buf(0xfeed_face);
        let (s, a) = run_pair(&buf, 8, 16, 1, 1, 0);
        assert_eq!(s, a);
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// dispatches must agree.
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
        (d_avx2.lpf_vertical_8())(&mut b_a, 8, 16, 200, 30, 7);
        (d_scalar.lpf_vertical_8())(&mut b_s, 8, 16, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
