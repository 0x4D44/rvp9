//! 4-tap vertical loop filter — x86_64 AVX2/SSE2 implementation (Wave F G F10).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::loopfilter::lpf_vertical_4`].
//!
//! ## Background
//!
//! `lpf_vertical_4` processes 8 rows of an 8-pixel-wide horizontal
//! window straddling a vertical edge. Per row, the kernel reads
//! `[p3, p2, p1, p0 | q0, q1, q2, q3]` (8 contiguous bytes), evaluates
//! the filter mask and high-edge-variance mask, and conditionally
//! updates the inner four bytes `[p1, p0, q0, q1]`.
//!
//! Naive per-row SIMD wastes most of the 16-byte register: only the
//! low 8 bytes carry data and only the inner 4 are written. The
//! libvpx approach (this implementation) is to **transpose 8 rows of
//! 8 bytes into 8 vectors-of-8-bytes (one vector per column)** so the
//! filter logic runs lane-parallel across all 8 rows in one pass.
//! After the filter, the four inner-column vectors are transposed
//! back into row-major form and stored.
//!
//! ## Bit-exactness vs scalar reference
//!
//! All arithmetic mirrors the scalar reference byte-for-byte:
//!
//! * `_mm_subs_epu8` / `_mm_adds_epu8` — saturating u8 arithmetic for
//!   the absolute-difference computations in `filter_mask` / `hev_mask`.
//! * `_mm_xor_si128` against a broadcast `0x80` — the same sign-flip
//!   trick the scalar uses (`(u8 ^ 0x80) as i8 as i32`) to convert
//!   between `[0, 255]` and `[-128, 127]`.
//! * The filter1 / filter2 / outer-tap arithmetic uses `_mm_subs_epi8`
//!   / `_mm_adds_epi8` (signed-saturating, matching scalar
//!   `signed_char_clamp`) for the i8-domain math, plus an
//!   i16-expand + `_mm_srai_epi16` + `_mm_packs_epi16` to do the
//!   signed arithmetic shift (SSE2 has no `_mm_srai_epi8`).
//!
//! NO modular `wraplow` semantics here — loop filtering is u8 with
//! saturation. We use the `_mm_subs_epi8` / `_mm_adds_epi8` family
//! deliberately (signed saturation matching the scalar
//! `signed_char_clamp` helper).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `pixels` are 8-byte unaligned via
//! `_mm_loadl_epi64`. Per-row stores are 4-byte unaligned via
//! `_mm_cvtsi128_si32` + `write_unaligned`. The grep sentinel below is
//! checked by `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8,
    _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_max_epu8, _mm_or_si128, _mm_packs_epi16,
    _mm_packus_epi16, _mm_set1_epi8, _mm_setzero_si128, _mm_shuffle_epi32, _mm_srai_epi16,
    _mm_subs_epi8, _mm_subs_epu8, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm_xor_si128,
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

/// Transpose 8×8 bytes given 8 row vectors (each holding 8 bytes in the
/// low 64 bits). Returns 8 vectors, each containing one column's worth
/// of bytes across the 8 rows in its low 64 bits.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose_8x8(
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
    r4: __m128i,
    r5: __m128i,
    r6: __m128i,
    r7: __m128i,
) -> [__m128i; 8] {
    // First stage: interleave pairs of rows as bytes.
    let t0 = _mm_unpacklo_epi8(r0, r1);
    let t1 = _mm_unpacklo_epi8(r2, r3);
    let t2 = _mm_unpacklo_epi8(r4, r5);
    let t3 = _mm_unpacklo_epi8(r6, r7);

    // Second stage: interleave 16-bit lanes.
    let u0 = _mm_unpacklo_epi16(t0, t1); // cols 0..3 for rows 0..3
    let u1 = _mm_unpackhi_epi16(t0, t1); // cols 4..7 for rows 0..3
    let u2 = _mm_unpacklo_epi16(t2, t3); // cols 0..3 for rows 4..7
    let u3 = _mm_unpackhi_epi16(t2, t3); // cols 4..7 for rows 4..7

    // Third stage: interleave 32-bit lanes — produces 16-byte vectors
    // where the low 8 bytes are one column (rows 0..7) and the high 8
    // bytes are the next column (rows 0..7).
    let v0 = _mm_unpacklo_epi32(u0, u2); // col 0 + col 1
    let v1 = _mm_unpackhi_epi32(u0, u2); // col 2 + col 3
    let v2 = _mm_unpacklo_epi32(u1, u3); // col 4 + col 5
    let v3 = _mm_unpackhi_epi32(u1, u3); // col 6 + col 7

    // Split each 16-byte register into two 8-byte halves — each half
    // is one column.
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
///
/// Implementation: `_mm_subs_epu8(x, thresh)` is 0 iff `x <= thresh`.
/// `_mm_cmpeq_epi8(_, 0)` gives 0xff iff the result was zero —
/// i.e. 0xff iff `x <= thresh`. XOR with 0xff to flip into the
/// "greater than" mask.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cmpgt_u8(x: __m128i, thresh: __m128i) -> __m128i {
    let diff = _mm_subs_epu8(x, thresh);
    let le_mask = _mm_cmpeq_epi8(diff, _mm_setzero_si128()); // 0xff iff x <= thresh
    let ones = _mm_set1_epi8(-1i8);
    _mm_xor_si128(le_mask, ones) // 0xff iff x > thresh
}

/// Sign-extend the low 8 i8 lanes of `x` to 16-bit (low 8 i16 lanes
/// of result).
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — helper does no `dst` I/O
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sx_lo(x: __m128i) -> __m128i {
    // unpacklo_epi8(x, x) duplicates each byte across both bytes of a
    // 16-bit lane, then arithmetic shift right by 8 preserves the
    // sign bit.
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

/// Apply `lpf_vertical_4` to one 8x8 strip at offset `s_index` into
/// `pixels`. Lane-parallel across 8 rows.
///
/// # Safety
///
/// `s_index` must be ≥ 4 and `s_index + 7 * pitch + 4` must be ≤
/// `pixels.len()`. Caller must guarantee AVX2 support.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::erasing_op, clippy::identity_op)]
unsafe fn lpf_vertical_4_avx2_inner(
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

    // ---- filter_mask ----
    let limit_v = _mm_set1_epi8(limit as i8);
    let blimit_v = _mm_set1_epi8(blimit as i8);

    let mut work = abs_diff_u8(p3, p2);
    work = _mm_max_epu8(work, abs_diff_u8(p2, p1));
    work = _mm_max_epu8(work, abs_diff_u8(p1, p0));
    work = _mm_max_epu8(work, abs_diff_u8(q1, q0));
    work = _mm_max_epu8(work, abs_diff_u8(q2, q1));
    work = _mm_max_epu8(work, abs_diff_u8(q3, q2));
    let limit_fail = cmpgt_u8(work, limit_v);

    // mb_diff = sat_u8(|p0-q0| * 2 + |p1-q1| / 2)
    let abs_p0q0 = abs_diff_u8(p0, q0);
    let abs_p1q1 = abs_diff_u8(p1, q1);
    let abs_p0q0_2 = _mm_adds_epu8(abs_p0q0, abs_p0q0);
    // |p1-q1| / 2 — expand to u16, shift, pack back.
    let zero = _mm_setzero_si128();
    let abs_p1q1_lo = _mm_unpacklo_epi8(abs_p1q1, zero);
    let abs_p1q1_hi = _mm_unpackhi_epi8(abs_p1q1, zero);
    let abs_p1q1_half_lo = _mm_srai_epi16(abs_p1q1_lo, 1);
    let abs_p1q1_half_hi = _mm_srai_epi16(abs_p1q1_hi, 1);
    let abs_p1q1_half = _mm_packus_epi16(abs_p1q1_half_lo, abs_p1q1_half_hi);
    let mb_diff = _mm_adds_epu8(abs_p0q0_2, abs_p1q1_half);
    let blimit_fail = cmpgt_u8(mb_diff, blimit_v);

    // filter_mask = NOT (limit_fail OR blimit_fail). 0xff = apply.
    let any_fail = _mm_or_si128(limit_fail, blimit_fail);
    let ones = _mm_set1_epi8(-1i8);
    let filter_mask = _mm_andnot_si128(any_fail, ones);

    // ---- hev_mask ----
    let thresh_v = _mm_set1_epi8(thresh as i8);
    let abs_p1p0 = abs_diff_u8(p1, p0);
    let abs_q1q0 = abs_diff_u8(q1, q0);
    let hev = _mm_or_si128(cmpgt_u8(abs_p1p0, thresh_v), cmpgt_u8(abs_q1q0, thresh_v));

    // ---- filter4 ----
    let sign = _mm_set1_epi8(-128i8); // 0x80
    let ps1 = _mm_xor_si128(p1, sign);
    let ps0 = _mm_xor_si128(p0, sign);
    let qs0 = _mm_xor_si128(q0, sign);
    let qs1 = _mm_xor_si128(q1, sign);

    // filter = clamp(ps1 - qs1) & hev
    let outer_taps = _mm_subs_epi8(ps1, qs1);
    let mut filter = _mm_and_si128(outer_taps, hev);

    // filter = clamp(filter + 3 * (qs0 - ps0)) & mask
    let qs0_ps0 = _mm_subs_epi8(qs0, ps0);
    let mut acc = _mm_adds_epi8(filter, qs0_ps0);
    acc = _mm_adds_epi8(acc, qs0_ps0);
    acc = _mm_adds_epi8(acc, qs0_ps0);
    filter = _mm_and_si128(acc, filter_mask);

    // filter1 = clamp(filter + 4) >> 3 (arithmetic).
    let plus4 = _mm_set1_epi8(4);
    let plus3 = _mm_set1_epi8(3);
    let f1_sat = _mm_adds_epi8(filter, plus4);
    let f2_sat = _mm_adds_epi8(filter, plus3);
    let filter1 = srai_i8::<3>(f1_sat);
    let filter2 = srai_i8::<3>(f2_sat);

    let qs0_new = _mm_subs_epi8(qs0, filter1);
    let ps0_new = _mm_adds_epi8(ps0, filter2);

    // Outer-tap adjustment: outer = ((filter1 + 1) >> 1) & ~hev
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

    // ---- Transpose 4×8 back to 8 rows × 4 bytes ----
    // Each of p1_out/p0_out/q0_out/q1_out holds 8 bytes in its low
    // 64 bits, one byte per row.
    let pp = _mm_unpacklo_epi8(p1_out, p0_out); // [p1[0], p0[0], p1[1], p0[1], ...]
    let qq = _mm_unpacklo_epi8(q0_out, q1_out); // [q0[0], q1[0], q0[1], q1[1], ...]
    // Interleave 16-bit lanes -> 32-bit row-major quartets.
    let rows_lo = _mm_unpacklo_epi16(pp, qq); // rows 0..3, each 4 bytes
    let rows_hi = _mm_unpackhi_epi16(pp, qq); // rows 4..7, each 4 bytes

    let row0 = _mm_cvtsi128_si32(rows_lo);
    let row1 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b01>(rows_lo));
    let row2 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b10>(rows_lo));
    let row3 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b11>(rows_lo));
    let row4 = _mm_cvtsi128_si32(rows_hi);
    let row5 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b01>(rows_hi));
    let row6 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b10>(rows_hi));
    let row7 = _mm_cvtsi128_si32(_mm_shuffle_epi32::<0b11>(rows_hi));

    let rows = [row0, row1, row2, row3, row4, row5, row6, row7];
    for (r, &val) in rows.iter().enumerate() {
        // G4: loads from `dst` use _mm_cvtsi128_si32 / write_unaligned (unaligned)
        core::ptr::write_unaligned(base.add(s_index + r * pitch - 2) as *mut i32, val);
    }
}

/// Safe wrapper for the AVX2 vertical 4-tap loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_vertical_4`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_vertical_4`] points at
/// this wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_vertical_4_avx2(
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
    // for the process lifetime. The bounds invariant `s_index >= 4 &&
    // s_index + 7*pitch + 4 <= pixels.len()` is the scalar contract
    // (mirrored by the inner debug_assert).
    unsafe { lpf_vertical_4_avx2_inner(pixels, s_index, pitch, blimit, limit, thresh) }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_vertical_4`.

    use super::lpf_vertical_4_avx2;
    use crate::loopfilter::lpf_vertical_4;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build an 8-row × 16-wide test buffer. The vertical edge sits
    /// at column 8 so each row's 8-byte window `[s-4..s+4]` lands
    /// entirely in valid memory.
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
        lpf_vertical_4(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_vertical_4_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
        (scalar, avx2)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Differential proptest over random byte fills and reasonable
        /// thresholds. The buffer is 8 rows × 16 cols at pitch 16 with
        /// the edge at column 8 so all 8 bytes of the per-row window
        /// land in valid memory.
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
    /// buffer for an obviously-filterable edge — proves the test
    /// harness isn't trivially returning the input.
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
        lpf_vertical_4(&mut copy, 8, 16, 255, 63, 0);
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

    /// Step edge across the boundary that should engage the filter on
    /// every row.
    #[test]
    fn directed_step_edge_engages_filter() {
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

    /// High-edge-variance path stress.
    #[test]
    fn directed_hev_path() {
        require_avx2!();
        let mut buf = vec![0u8; 8 * 16];
        for r in 0..8 {
            buf[r * 16 + 6] = 40;
            buf[r * 16 + 7] = 200;
            buf[r * 16 + 8] = 50;
            buf[r * 16 + 9] = 210;
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
        // blimit = 1, limit = 1 — nearly anything fails.
        let (s, a) = run_pair(&buf, 8, 16, 1, 1, 0);
        assert_eq!(s, a);
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// dispatches must agree.
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
        (d_avx2.lpf_vertical_4())(&mut b_a, 8, 16, 200, 30, 7);
        (d_scalar.lpf_vertical_4())(&mut b_s, 8, 16, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
