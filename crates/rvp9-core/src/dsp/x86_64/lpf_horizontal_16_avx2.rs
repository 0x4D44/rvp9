//! 16-tap horizontal loop filter — x86_64 AVX2/SSE2 implementation
//! (Wave G G6).
//!
//! Sibling of [`crate::dsp::x86_64::lpf_horizontal_8_avx2`] (Wave G G5)
//! and [`crate::dsp::x86_64::lpf_vertical_16_avx2`] (Wave G G6): same
//! 15-tap filter cascade (filter16 → filter8 → filter4) lane-parallel
//! across 8 columns, but the rows-are-contiguous layout means NO
//! transpose — each tap is a single `_mm_loadl_epi64` of one row's 8
//! bytes.
//!
//! Processes one 8-column strip per call (matches the scalar
//! `lpf_horizontal_16` which calls `mb_lpf_horizontal_edge_w` with
//! `count = 1`).
//!
//! The scalar reference is [`crate::loopfilter::lpf_horizontal_16`].
//!
//! ## Bit-exactness vs scalar reference
//!
//! Identical to the G6 vertical sibling — see that file's module
//! docstring. All helper functions (`abs_diff_u8`, `cmpgt_u8`,
//! `sx_lo`, `sx_hi`, `srai_i8`, `filter_mask_avx2`, `hev_mask_avx2`,
//! `flat_mask4_avx2`, `flat_mask5_avx2`, `filter4_inner`,
//! `filter8_wide`, `filter16_wide_avx2`, `blend`, `filter16_avx2`)
//! are duplicated verbatim from `lpf_vertical_16_avx2` per Wave F / G
//! precedent — extraction to a shared `loopfilter_common.rs` is a
//! Wave H refactor opportunity.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `pixels` are 8-byte unaligned via
//! `_mm_loadl_epi64`. Per-row stores are 8-byte unaligned via
//! `_mm_storel_epi64`. The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_adds_epi8, _mm_adds_epu8, _mm_adds_epu16, _mm_and_si128,
    _mm_andnot_si128, _mm_cmpeq_epi8, _mm_cmpgt_epi16, _mm_loadl_epi64, _mm_max_epu8, _mm_or_si128,
    _mm_packs_epi16, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_setzero_si128,
    _mm_srai_epi16, _mm_srli_epi16, _mm_storel_epi64, _mm_subs_epi8, _mm_subs_epu8,
    _mm_unpackhi_epi8, _mm_unpacklo_epi8, _mm_xor_si128,
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

/// Lane-parallel `filter_mask`. Identical to G6 vertical sibling.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter_mask_avx2(blimit: u8, limit: u8, taps: [__m128i; 8]) -> __m128i {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = taps;
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
    let blimit_fail = if blimit == u8::MAX {
        // Saturation preserves comparisons below 255, but a true metric
        // above 255 must still reject the edge when blimit itself is 255.
        // The widened metric is at most 2 * 255 + 255 / 2 = 637.
        let abs_p0q0_lo = _mm_unpacklo_epi8(abs_p0q0, zero);
        let abs_p0q0_hi = _mm_unpackhi_epi8(abs_p0q0, zero);
        let mb_diff_lo = _mm_add_epi16(_mm_add_epi16(abs_p0q0_lo, abs_p0q0_lo), abs_p1q1_half_lo);
        let mb_diff_hi = _mm_add_epi16(_mm_add_epi16(abs_p0q0_hi, abs_p0q0_hi), abs_p1q1_half_hi);
        let blimit_wide = _mm_set1_epi16(i16::from(blimit));
        _mm_packs_epi16(
            _mm_cmpgt_epi16(mb_diff_lo, blimit_wide),
            _mm_cmpgt_epi16(mb_diff_hi, blimit_wide),
        )
    } else {
        cmpgt_u8(mb_diff, blimit_v)
    };

    let any_fail = _mm_or_si128(limit_fail, blimit_fail);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_fail, ones)
}

/// Lane-parallel `hev_mask`. Identical.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hev_mask_avx2(thresh: u8, p1: __m128i, p0: __m128i, q0: __m128i, q1: __m128i) -> __m128i {
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let abs_p1p0 = abs_diff_u8(p1, p0);
    let abs_q1q0 = abs_diff_u8(q1, q0);
    _mm_or_si128(cmpgt_u8(abs_p1p0, thresh_v), cmpgt_u8(abs_q1q0, thresh_v))
}

/// Lane-parallel `flat_mask4`. Identical.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn flat_mask4_avx2(thresh: u8, taps: [__m128i; 8]) -> __m128i {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = taps;
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

/// Lane-parallel `flat_mask5`. See G6 vertical sibling for derivation.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn flat_mask5_avx2(
    thresh: u8,
    p7: __m128i,
    p6: __m128i,
    p5: __m128i,
    p4: __m128i,
    p0: __m128i,
    q0: __m128i,
    q4: __m128i,
    q5: __m128i,
    q6: __m128i,
    q7: __m128i,
) -> __m128i {
    // Wave H H4.1 — parameter names corrected to match the scalar
    // `flat_mask5` formal-param convention (which is misleading in
    // libvpx — formal `p4..p1` actually receive outer pixels at
    // `pixels[s-8..s-5]`). Same fix as Wave H H7 applied to the
    // sibling `lpf_vertical_16_avx2` (commit b9d4cdb).
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let mut work = abs_diff_u8(p4, p0);
    work = _mm_max_epu8(work, abs_diff_u8(q4, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p5, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q5, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p6, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q6, q0));
    work = _mm_max_epu8(work, abs_diff_u8(p7, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q7, q0));
    let any_gt = cmpgt_u8(work, thresh_v);
    let ones = _mm_set1_epi8(-1i8);
    _mm_andnot_si128(any_gt, ones)
}

/// Lane-parallel `filter4` inner. Identical to G4.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter4_inner(
    filter_mask: __m128i,
    hev: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
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

/// `(x + 4) >> 3` for u16 lanes (filter8 rounding).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_pow2_3_u16(x: __m128i) -> __m128i {
    let four = _mm_set1_epi16(4);
    let summed = _mm_adds_epu16(x, four);
    _mm_srli_epi16(summed, 3)
}

/// `(x + 8) >> 4` for u16 lanes (filter16 rounding).
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

/// 7-tap symmetric smoothing — identical to G5.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn filter8_wide(
    taps: [__m128i; 8],
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = taps;
    let p3w = widen_u8_to_u16_lo(p3);
    let p2w = widen_u8_to_u16_lo(p2);
    let p1w = widen_u8_to_u16_lo(p1);
    let p0w = widen_u8_to_u16_lo(p0);
    let q0w = widen_u8_to_u16_lo(q0);
    let q1w = widen_u8_to_u16_lo(q1);
    let q2w = widen_u8_to_u16_lo(q2);
    let q3w = widen_u8_to_u16_lo(q3);

    let mut sum = _mm_adds_epu16(p3w, p3w);
    sum = _mm_adds_epu16(sum, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    let op2 = round_pow2_3_u16(sum);

    let mut sum = _mm_adds_epu16(p3w, p3w);
    sum = _mm_adds_epu16(sum, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    let op1 = round_pow2_3_u16(sum);

    let mut sum = _mm_adds_epu16(p3w, p2w);
    sum = _mm_adds_epu16(sum, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    let op0 = round_pow2_3_u16(sum);

    let mut sum = _mm_adds_epu16(p2w, p1w);
    sum = _mm_adds_epu16(sum, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    let oq0 = round_pow2_3_u16(sum);

    let mut sum = _mm_adds_epu16(p1w, p0w);
    sum = _mm_adds_epu16(sum, q0w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q1w);
    sum = _mm_adds_epu16(sum, q2w);
    sum = _mm_adds_epu16(sum, q3w);
    sum = _mm_adds_epu16(sum, q3w);
    let oq1 = round_pow2_3_u16(sum);

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

/// 15-tap symmetric smoothing — identical to G6 vertical sibling.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn filter16_wide_avx2(
    p7: __m128i,
    p6: __m128i,
    p5: __m128i,
    p4: __m128i,
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    q4: __m128i,
    q5: __m128i,
    q6: __m128i,
    q7: __m128i,
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

    // op6
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

    // op5
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

    // op4
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

    // op3
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

    // op2
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

    // op1
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

    // op0
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

    // oq0
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

    // oq1
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

    // oq2
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

    // oq3
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

    // oq4
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

    // oq5
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

    // oq6
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

/// Blend two vectors per byte-mask.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn blend(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

/// Apply `lpf_horizontal_16` to one 8-column strip whose top-left
/// pixel in the q-half is `pixels[s_index]`. No transpose — each tap
/// is a contiguous 8-byte row.
///
/// # Safety
///
/// `s_index` must be `>= 8 * pitch`, and
/// `s_index + 7 * pitch + 8` must be `<= pixels.len()`. Caller must
/// guarantee AVX2 support.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::erasing_op, clippy::identity_op)]
unsafe fn lpf_horizontal_16_avx2_inner(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 8 * pitch);
    debug_assert!(s_index + 7 * pitch + 8 <= pixels.len());

    let base = pixels.as_mut_ptr();
    // Each tap is one CONTIGUOUS 8-byte row offset by k*pitch from the
    // edge. No transpose.
    let p7 = load_row(base, s_index - 8 * pitch);
    let p6 = load_row(base, s_index - 7 * pitch);
    let p5 = load_row(base, s_index - 6 * pitch);
    let p4 = load_row(base, s_index - 5 * pitch);
    let p3 = load_row(base, s_index - 4 * pitch);
    let p2 = load_row(base, s_index - 3 * pitch);
    let p1 = load_row(base, s_index - 2 * pitch);
    let p0 = load_row(base, s_index - 1 * pitch);
    let q0 = load_row(base, s_index + 0 * pitch);
    let q1 = load_row(base, s_index + 1 * pitch);
    let q2 = load_row(base, s_index + 2 * pitch);
    let q3 = load_row(base, s_index + 3 * pitch);
    let q4 = load_row(base, s_index + 4 * pitch);
    let q5 = load_row(base, s_index + 5 * pitch);
    let q6 = load_row(base, s_index + 6 * pitch);
    let q7 = load_row(base, s_index + 7 * pitch);

    let taps = [p3, p2, p1, p0, q0, q1, q2, q3];
    let mask = filter_mask_avx2(blimit, limit, taps);
    let hev = hev_mask_avx2(thresh, p1, p0, q0, q1);
    let flat = flat_mask4_avx2(1, taps);
    // Wave H H4.1 — call site passes OUTER pixels (p7..p4, q4..q7) to
    // match the scalar `flat_mask5` semantics; H7 fixed the parallel
    // bug in the vertical sibling at b9d4cdb.
    let flat2 = flat_mask5_avx2(1, p7, p6, p5, p4, p0, q0, q4, q5, q6, q7);

    let apply_wide = _mm_and_si128(flat, mask);
    let apply_wide2 = _mm_and_si128(flat2, apply_wide);

    let (p1_f4, p0_f4, q0_f4, q1_f4) = filter4_inner(mask, hev, p1, p0, q0, q1);
    let (op2_w7, op1_w7, op0_w7, oq0_w7, oq1_w7, oq2_w7) = filter8_wide(taps);
    let f16 = filter16_wide_avx2(
        p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7,
    );
    let (
        op6_w15,
        op5_w15,
        op4_w15,
        op3_w15,
        op2_w15,
        op1_w15,
        op0_w15,
        oq0_w15,
        oq1_w15,
        oq2_w15,
        oq3_w15,
        oq4_w15,
        oq5_w15,
        oq6_w15,
    ) = (
        f16[0], f16[1], f16[2], f16[3], f16[4], f16[5], f16[6], f16[7], f16[8], f16[9], f16[10],
        f16[11], f16[12], f16[13],
    );

    // Blend (see vertical sibling for cascade structure).
    let op6_o = blend(apply_wide2, op6_w15, p6);
    let op5_o = blend(apply_wide2, op5_w15, p5);
    let op4_o = blend(apply_wide2, op4_w15, p4);
    let op3_o = blend(apply_wide2, op3_w15, p3);
    let oq3_o = blend(apply_wide2, oq3_w15, q3);
    let oq4_o = blend(apply_wide2, oq4_w15, q4);
    let oq5_o = blend(apply_wide2, oq5_w15, q5);
    let oq6_o = blend(apply_wide2, oq6_w15, q6);

    let op2_w = blend(apply_wide, op2_w7, p2);
    let oq2_w = blend(apply_wide, oq2_w7, q2);
    let op2_o = blend(apply_wide2, op2_w15, op2_w);
    let oq2_o = blend(apply_wide2, oq2_w15, oq2_w);

    let op1_mid = blend(apply_wide, op1_w7, p1_f4);
    let op0_mid = blend(apply_wide, op0_w7, p0_f4);
    let oq0_mid = blend(apply_wide, oq0_w7, q0_f4);
    let oq1_mid = blend(apply_wide, oq1_w7, q1_f4);
    let op1_o = blend(apply_wide2, op1_w15, op1_mid);
    let op0_o = blend(apply_wide2, op0_w15, op0_mid);
    let oq0_o = blend(apply_wide2, oq0_w15, oq0_mid);
    let oq1_o = blend(apply_wide2, oq1_w15, oq1_mid);

    // p7 / q7 pass through unchanged — no store.
    store_row(base, s_index - 7 * pitch, op6_o);
    store_row(base, s_index - 6 * pitch, op5_o);
    store_row(base, s_index - 5 * pitch, op4_o);
    store_row(base, s_index - 4 * pitch, op3_o);
    store_row(base, s_index - 3 * pitch, op2_o);
    store_row(base, s_index - 2 * pitch, op1_o);
    store_row(base, s_index - 1 * pitch, op0_o);
    store_row(base, s_index + 0 * pitch, oq0_o);
    store_row(base, s_index + 1 * pitch, oq1_o);
    store_row(base, s_index + 2 * pitch, oq2_o);
    store_row(base, s_index + 3 * pitch, oq3_o);
    store_row(base, s_index + 4 * pitch, oq4_o);
    store_row(base, s_index + 5 * pitch, oq5_o);
    store_row(base, s_index + 6 * pitch, oq6_o);
}

/// Safe wrapper for the AVX2 horizontal 16-tap loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_horizontal_16`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_horizontal_16`] points
/// at this wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_horizontal_16_avx2(
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
    // `s_index >= 8*pitch && s_index + 7*pitch + 8 <= pixels.len()`
    // is the scalar contract (mirrored by the inner debug_assert).
    unsafe { lpf_horizontal_16_avx2_inner(pixels, s_index, pitch, blimit, limit, thresh) }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_horizontal_16`.

    use super::lpf_horizontal_16_avx2;
    use crate::loopfilter::lpf_horizontal_16;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build a 24-row × 8-wide test buffer. Edge sits between row 11
    /// and row 12 at `s_index = 12 * pitch`.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 24 * 8];
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
        lpf_horizontal_16(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_horizontal_16_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
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
            let (s, a) = run_pair(&buf, 12 * 8, 8, blimit, limit, thresh);
            prop_assert_eq!(s, a);
        }
    }

    #[test]
    fn blimit_255_saturation_matches_scalar() {
        require_avx2!();

        // (p0, p1, q0, q1, exact macroblock-edge metric). Different
        // columns exercise both sides of the 255 boundary together.
        let lanes: [(u8, u8, u8, u8, u16); 8] = [
            (127, 1, 0, 0, 254),
            (127, 2, 0, 0, 255),
            (128, 1, 0, 0, 256),
            (246, 246, 3, 3, 607),
            (80, 80, 170, 170, 225),
            (80, 80, 182, 182, 255),
            (0, 0, 255, 255, 637),
            (0, 0, 128, 1, 256),
        ];
        for pitch in [8, 19] {
            let mut buf = vec![77; 24 * pitch];
            for (col, &(p0, p1, q0, q1, _)) in lanes.iter().enumerate() {
                for row in 0..24 {
                    buf[row * pitch + col] = match row {
                        0..=10 => p1,
                        11 => p0,
                        12 => q0,
                        _ => q1,
                    };
                }
            }
            for blimit in [193, 254, 255] {
                let (scalar, avx2) = run_pair(&buf, 12 * pitch, pitch, blimit, 255, 255);
                for (col, &(p0, p1, q0, q1, metric)) in lanes.iter().enumerate() {
                    let exact_metric =
                        2 * u16::from(p0.abs_diff(q0)) + u16::from(p1.abs_diff(q1)) / 2;
                    assert_eq!(exact_metric, metric);
                    let changed =
                        (0..24).any(|row| scalar[row * pitch + col] != buf[row * pitch + col]);
                    assert_eq!(
                        changed,
                        metric <= u16::from(blimit),
                        "scalar control: pitch={pitch} col={col} blimit={blimit}"
                    );
                }
                assert_eq!(scalar, avx2, "pitch={pitch} blimit={blimit}");
            }
        }
    }

    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let pitch = 8;
        let mut buf = vec![100u8; 24 * pitch];
        for r in 0..12 {
            for c in 0..pitch {
                buf[r * pitch + c] = 80;
            }
        }
        for r in 12..24 {
            for c in 0..pitch {
                buf[r * pitch + c] = 170;
            }
        }
        let mut copy = buf.clone();
        lpf_horizontal_16(&mut copy, 12 * pitch, pitch, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar filter must alter the buffer for filterable edges"
        );
    }

    #[test]
    fn all_zero_no_op() {
        require_avx2!();
        let buf = vec![0u8; 24 * 8];
        let (s, a) = run_pair(&buf, 12 * 8, 8, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![0u8; 24 * 8]);
    }

    #[test]
    fn all_max_no_op() {
        require_avx2!();
        let buf = vec![255u8; 24 * 8];
        let (s, a) = run_pair(&buf, 12 * 8, 8, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![255u8; 24 * 8]);
    }

    #[test]
    fn directed_step_edge_engages_wide15_filter() {
        require_avx2!();
        let pitch = 8;
        let mut buf = vec![0u8; 24 * pitch];
        for r in 0..12 {
            for c in 0..pitch {
                buf[r * pitch + c] = 80;
            }
        }
        for r in 12..24 {
            for c in 0..pitch {
                buf[r * pitch + c] = 170;
            }
        }
        let (s, a) = run_pair(&buf, 12 * pitch, pitch, 255, 63, 7);
        assert_eq!(s, a);
    }

    #[test]
    fn directed_hev_falls_back_to_filter4() {
        require_avx2!();
        let pitch = 8;
        let mut buf = vec![0u8; 24 * pitch];
        for r in 0..24 {
            for c in 0..pitch {
                buf[r * pitch + c] = if r & 1 == 0 { 40 } else { 200 };
            }
        }
        let (s, a) = run_pair(&buf, 12 * pitch, pitch, 200, 30, 7);
        assert_eq!(s, a);
    }

    #[test]
    fn directed_mask_off_preserves_input() {
        require_avx2!();
        let buf = make_buf(0xfeed_face);
        let (s, a) = run_pair(&buf, 12 * 8, 8, 1, 1, 0);
        assert_eq!(s, a);
    }

    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps = detect();
        caps.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps);
        let mut caps_s = detect();
        caps_s.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_s);

        let buf = make_buf(0xdead_beef);
        let mut b_a = buf.clone();
        let mut b_s = buf;
        (d_avx2.lpf_horizontal_16())(&mut b_a, 12 * 8, 8, 200, 30, 7);
        (d_scalar.lpf_horizontal_16())(&mut b_s, 12 * 8, 8, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
