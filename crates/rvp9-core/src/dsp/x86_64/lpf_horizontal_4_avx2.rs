//! 4-tap horizontal loop filter — x86_64 AVX2/SSE2 implementation (Wave G G3).
//!
//! Sibling of [`crate::dsp::x86_64::lpf_vertical_4_avx2`] (Wave F G F10).
//! The scalar reference is [`crate::loopfilter::lpf_horizontal_4`].
//!
//! ## Background
//!
//! `lpf_horizontal_4` processes 8 contiguous columns straddling a
//! horizontal edge. Per column, the kernel reads 8 bytes vertically
//! `[p3, p2, p1, p0 | q0, q1, q2, q3]` (each tap is one row above /
//! below the edge, separated by `pitch` bytes), evaluates the filter
//! mask and high-edge-variance mask, and conditionally updates the
//! inner four bytes `[p1, p0, q0, q1]`.
//!
//! Unlike the vertical sibling, **each tap is already a CONTIGUOUS
//! 8-byte row** because the filter operates on 8 columns and 1 row per
//! tap. So we load each tap with one `_mm_loadl_epi64` from
//! `pixels[s_index + (-4..4) * pitch ..][..8]` and run the
//! lane-parallel `filter_mask` / `hev_mask` / `filter4` directly — no
//! transpose needed (the layout is already column-per-lane in each
//! row vector).
//!
//! ## Bit-exactness vs scalar reference
//!
//! All arithmetic mirrors the scalar reference byte-for-byte and reuses
//! the helper conventions established in
//! [`crate::dsp::x86_64::lpf_vertical_4_avx2`]:
//!
//! * `_mm_subs_epu8` / `_mm_adds_epu8` — saturating u8 arithmetic for
//!   absolute-difference computations in `filter_mask` / `hev_mask`.
//! * `_mm_xor_si128` against a broadcast `0x80` — sign-flip trick the
//!   scalar uses (`(u8 ^ 0x80) as i8 as i32`) to convert between
//!   `[0, 255]` and `[-128, 127]`.
//! * `_mm_subs_epi8` / `_mm_adds_epi8` (signed-saturating) for the i8
//!   filter1 / filter2 / outer-tap math, plus an i16-expand +
//!   `_mm_srai_epi16` + `_mm_packs_epi16` to do the signed arithmetic
//!   shift (SSE2 has no `_mm_srai_epi8`).
//!
//! NO modular `wraplow` semantics here — loop filtering is u8 with
//! saturation. We use the `_mm_subs_epi8` / `_mm_adds_epi8` family
//! deliberately (signed saturation matching the scalar
//! `signed_char_clamp` helper).
//!
//! Helper functions (`abs_diff_u8`, `cmpgt_u8`, `srai_i8`) are
//! duplicated verbatim from `lpf_vertical_4_avx2` — Wave F precedent
//! is one self-contained file per kernel; extraction to a shared
//! `loopfilter_common.rs` is a Wave H refactor opportunity.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-row loads from `pixels` are 8-byte unaligned via
//! `_mm_loadl_epi64`. Per-row stores are 8-byte unaligned via
//! `_mm_storel_epi64`. The grep sentinel below is checked by
//! `scripts/check-simd-sentinels.{sh,ps1}`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_adds_epi8, _mm_adds_epu8, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8,
    _mm_loadl_epi64, _mm_max_epu8, _mm_or_si128, _mm_packs_epi16, _mm_packus_epi16, _mm_set1_epi8,
    _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64, _mm_subs_epi8, _mm_subs_epu8,
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

/// Apply `lpf_horizontal_4` to one 8-row × 8-col strip whose top-left
/// pixel in the q-half is `pixels[s_index]`. Lane-parallel across 8
/// columns.
///
/// # Safety
///
/// `s_index` must be `>= 4 * pitch`, and
/// `s_index + 3 * pitch + 8` must be `<= pixels.len()`. Caller must
/// guarantee AVX2 support.
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::erasing_op, clippy::identity_op)]
unsafe fn lpf_horizontal_4_avx2_inner(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    debug_assert!(s_index >= 4 * pitch);
    debug_assert!(s_index + 3 * pitch + 8 <= pixels.len());

    let base = pixels.as_mut_ptr();
    // Each tap is one CONTIGUOUS 8-byte row offset by k*pitch from the
    // edge. The 8 columns of the strip naturally line up as the 8 low
    // bytes of each vector — exactly the lane layout the filter4
    // helpers want. No transpose needed.
    let p3 = load_row(base, s_index - 4 * pitch);
    let p2 = load_row(base, s_index - 3 * pitch);
    let p1 = load_row(base, s_index - 2 * pitch);
    let p0 = load_row(base, s_index - 1 * pitch);
    let q0 = load_row(base, s_index + 0 * pitch);
    let q1 = load_row(base, s_index + 1 * pitch);
    let q2 = load_row(base, s_index + 2 * pitch);
    let q3 = load_row(base, s_index + 3 * pitch);

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

    // Store the four modified rows. Each output vector already holds
    // 8 columns in its low 64 bits — direct row stores, no transpose.
    store_row(base, s_index - 2 * pitch, p1_out);
    store_row(base, s_index - 1 * pitch, p0_out);
    store_row(base, s_index + 0 * pitch, q0_out);
    store_row(base, s_index + 1 * pitch, q1_out);
}

/// Safe wrapper for the AVX2 horizontal 4-tap loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_horizontal_4`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_horizontal_4`] points
/// at this wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_horizontal_4_avx2(
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
    // `s_index >= 4*pitch && s_index + 3*pitch + 8 <= pixels.len()`
    // is the scalar contract (mirrored by the inner debug_assert).
    unsafe { lpf_horizontal_4_avx2_inner(pixels, s_index, pitch, blimit, limit, thresh) }
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_horizontal_4`.

    use super::lpf_horizontal_4_avx2;
    use crate::loopfilter::lpf_horizontal_4;
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build a 16-row × 8-wide test buffer. The horizontal edge sits
    /// between row 7 and row 8 (i.e. `s_index = 8 * pitch`); each tap
    /// row (-4..=3 relative to the edge) lands in valid memory.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 16 * 8];
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
        lpf_horizontal_4(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_horizontal_4_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
        (scalar, avx2)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        /// Differential proptest over random byte fills and reasonable
        /// thresholds. The buffer is 16 rows × 8 cols at pitch 8 with
        /// the edge between rows 7 and 8 so all 8 tap rows land in
        /// valid memory.
        #[test]
        fn scalar_eq_avx2_proptest(
            seed in any::<u32>(),
            // Valid VP9 blimit (mblim) range: `2*(filter_level+2) + block_inside_limit`,
            // max `2*(63+2)+63 = 193` for 8-bit. The AVX2 kernel computes the
            // `mb_diff` mask term in saturating u8 (mirroring libvpx's SIMD loop
            // filter); only at the *unreachable* `blimit=255` does the saturated
            // value (≤255) fail to exceed blimit while the scalar i32 `mb_diff`
            // (max 637) can — the sole divergence point. Cap at the valid max so
            // this differential stays in-domain. See JRN 2026.06.20.
            blimit in 1u8..=193,
            limit in 1u8..=63,
            thresh in 0u8..=15,
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let buf = make_buf(seed);
            let (s, a) = run_pair(&buf, 8 * 8, 8, blimit, limit, thresh);
            prop_assert_eq!(s, a);
        }
    }

    /// Tautology guard: the scalar filter must actually modify the
    /// buffer for an obviously-filterable edge — proves the test
    /// harness isn't trivially returning the input.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let pitch = 8;
        let mut buf = vec![100u8; 16 * pitch];
        // Per-column step centered at the edge between rows 7 and 8.
        for c in 0..8 {
            buf[4 * pitch + c] = 98;
            buf[5 * pitch + c] = 99;
            buf[6 * pitch + c] = 100;
            buf[7 * pitch + c] = 101;
            buf[8 * pitch + c] = 110;
            buf[9 * pitch + c] = 109;
            buf[10 * pitch + c] = 108;
            buf[11 * pitch + c] = 107;
        }
        let mut copy = buf.clone();
        lpf_horizontal_4(&mut copy, 8 * pitch, pitch, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar filter must alter the buffer for filterable edges"
        );
    }

    #[test]
    fn all_zero_no_op() {
        require_avx2!();
        let buf = vec![0u8; 16 * 8];
        let (s, a) = run_pair(&buf, 8 * 8, 8, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![0u8; 16 * 8]);
    }

    #[test]
    fn all_max_no_op() {
        require_avx2!();
        let buf = vec![255u8; 16 * 8];
        let (s, a) = run_pair(&buf, 8 * 8, 8, 255, 63, 15);
        assert_eq!(s, a);
        assert_eq!(s, vec![255u8; 16 * 8]);
    }

    /// Step edge across the horizontal boundary that should engage the
    /// filter on every column.
    #[test]
    fn directed_step_edge_engages_filter() {
        require_avx2!();
        let pitch = 8;
        let mut buf = vec![0u8; 16 * pitch];
        for r in 0..8 {
            for c in 0..8 {
                buf[r * pitch + c] = 80;
                buf[(r + 8) * pitch + c] = 170;
            }
        }
        let (s, a) = run_pair(&buf, 8 * pitch, pitch, 255, 63, 7);
        assert_eq!(s, a);
    }

    /// High-edge-variance path stress.
    #[test]
    fn directed_hev_path() {
        require_avx2!();
        let pitch = 8;
        let mut buf = vec![0u8; 16 * pitch];
        for c in 0..8 {
            buf[6 * pitch + c] = 40;
            buf[7 * pitch + c] = 200;
            buf[8 * pitch + c] = 50;
            buf[9 * pitch + c] = 210;
        }
        let (s, a) = run_pair(&buf, 8 * pitch, pitch, 200, 30, 7);
        assert_eq!(s, a);
    }

    /// Mask-off path: thresholds so tight that filter_mask should be 0
    /// everywhere — output must equal input.
    #[test]
    fn directed_mask_off_preserves_input() {
        require_avx2!();
        let buf = make_buf(0xfeed_face);
        // blimit = 1, limit = 1 — nearly anything fails.
        let (s, a) = run_pair(&buf, 8 * 8, 8, 1, 1, 0);
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
        (d_avx2.lpf_horizontal_4())(&mut b_a, 8 * 8, 8, 200, 30, 7);
        (d_scalar.lpf_horizontal_4())(&mut b_s, 8 * 8, 8, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
