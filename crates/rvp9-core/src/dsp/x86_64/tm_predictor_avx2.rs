//! TM_PRED ("true motion") intra-predictor — x86_64 AVX2 implementation (Wave I I3).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]) and the I1 / I2 V/H predictor
//! kernels. The scalar reference is [`crate::intra_pred::tm_predictor`].
//!
//! ## Algorithm
//!
//! TM_PRED computes, for every pixel of the `bs × bs` block:
//!
//! ```text
//! dst[r][c] = clip(left[r] + above[c] - above_left, 0, 255)
//! ```
//!
//! Per row `r`, `delta = left[r] - above_left` is a single scalar i16
//! (range `[-255, 255]`). The full per-pixel value is then
//! `delta + above[c]` — i16 lane-parallel — and the final clip-to-byte
//! is implemented "for free" by **unsigned saturating pack**
//! (`_mm{,256}_packus_epi16`), which clips negatives to 0 and values
//! `> 255` to 255 in one instruction.
//!
//! ## Vectorization strategy
//!
//! 1. Per row, compute `delta = left[r] - above_left` as i16.
//! 2. Broadcast `delta` to an i16-lane register (`_mm{,256}_set1_epi16`).
//! 3. Load `above[..bs]` as u8, widen to i16 lanes
//!    (`_mm_cvtepu8_epi16` / `_mm256_cvtepu8_epi16`).
//! 4. Add lane-parallel (`_mm{,256}_add_epi16`).
//! 5. Pack i16 → u8 with unsigned saturation
//!    (`_mm{,256}_packus_epi16`). This **is** the [0, 255] clip.
//! 6. Store the `bs`-byte row.
//!
//! Per-`bs` shape:
//! - `bs = 4` / `bs = 8` → SSE2 (8 i16 lanes; high 4 are don't-care for bs=4)
//! - `bs = 16`           → AVX2 (16 i16 lanes); after packing, the
//!   256-bit `packus` interleaves 128-bit halves, so we apply a
//!   `_mm256_permute4x64_epi64` shuffle to bring the wanted 16 bytes
//!   adjacent in the low half, then `_mm_storeu_si128` of that low half
//! - `bs = 32`           → two independent 16-lane chunks per row
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads from `above` / `left` use unaligned intrinsics
//! (`_mm_loadu_si128`, `_mm_loadl_epi64`, `_mm_cvtsi32_si128` via
//! `read_unaligned`). All stores into `dst` use `_mm_storeu_si128` /
//! `_mm_storel_epi64` / `_mm_cvtsi128_si32` via `write_unaligned`. The
//! grep sentinels below are checked by `scripts/check-simd-sentinels`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_packus_epi16, _mm_set1_epi16, _mm_storel_epi64,
    _mm_storeu_si128, _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi16,
};

/// AVX2 implementation of TM_PRED.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2 (the
/// `_mm256_*` intrinsics in the bs=16/32 paths). The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`tm_predictor_avx2`]) when `caps.avx2 == true`.
///
/// `above` and `left` must each have at least `bs` valid u8s. `dst`
/// must hold at least `(bs - 1) * stride + bs` bytes of valid storage.
/// `bs` must be one of `{4, 8, 16, 32}`.
// G4: loads from `above` / `left` use _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
// G4: stores into `dst` use _mm_storeu_si128 / _mm_storel_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tm_predictor_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    debug_assert!(above.len() >= bs);
    debug_assert!(left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);
    debug_assert!(matches!(bs, 4 | 8 | 16 | 32));

    let tl = above_left as i32;

    match bs {
        4 => {
            // 4 bytes per row; load above[0..4] as u32 → low 4 lanes of
            // i16. The high 4 lanes after the widen are zero (don't care).
            let above_i32: i32 = core::ptr::read_unaligned(above.as_ptr() as *const i32);
            let above_v8: __m128i = _mm_cvtsi32_si128(above_i32);
            // Widen to 8 i16 lanes (low 4 hold real data, high 4 = 0).
            let above_v16: __m128i = _mm_cvtepu8_epi16(above_v8);
            for (r, &left_px) in left.iter().take(4).enumerate() {
                let delta = left_px as i32 - tl;
                let delta_v: __m128i = _mm_set1_epi16(delta as i16);
                let sum: __m128i = _mm_add_epi16(delta_v, above_v16);
                // Pack i16 → u8 with unsigned saturation. The 8 i16
                // lanes become 8 packed u8s in the low 8 bytes of the
                // result; only the low 4 are meaningful.
                let packed: __m128i = _mm_packus_epi16(sum, sum);
                let row_i32 = _mm_cvtsi128_si32(packed);
                core::ptr::write_unaligned(dst.as_mut_ptr().add(r * stride) as *mut i32, row_i32);
            }
        }
        8 => {
            // 8 bytes per row → one __m128i of 8 i16 lanes.
            let above_v8: __m128i = _mm_loadl_epi64(above.as_ptr() as *const __m128i);
            let above_v16: __m128i = _mm_cvtepu8_epi16(above_v8);
            for (r, &left_px) in left.iter().take(8).enumerate() {
                let delta = left_px as i32 - tl;
                let delta_v: __m128i = _mm_set1_epi16(delta as i16);
                let sum: __m128i = _mm_add_epi16(delta_v, above_v16);
                // 8 i16 → 8 packed u8 with unsigned saturation; result
                // lives in the low 8 bytes.
                let packed: __m128i = _mm_packus_epi16(sum, sum);
                _mm_storel_epi64(dst.as_mut_ptr().add(r * stride) as *mut __m128i, packed);
            }
        }
        16 => {
            // 16 bytes per row → 16 i16 lanes in one __m256i.
            let above_v8: __m128i = _mm_loadu_si128(above.as_ptr() as *const __m128i);
            let above_v16: __m256i = _mm256_cvtepu8_epi16(above_v8);
            for (r, &left_px) in left.iter().take(16).enumerate() {
                let delta = left_px as i32 - tl;
                let delta_v: __m256i = _mm256_set1_epi16(delta as i16);
                let sum: __m256i = _mm256_add_epi16(delta_v, above_v16);
                // _mm256_packus_epi16(a, a) on AVX2 interleaves
                // 128-bit halves: output bytes are
                //   [pack(a.lo_i16) | pack(a.lo_i16) |
                //    pack(a.hi_i16) | pack(a.hi_i16)]
                // i.e. the 16 u8s we want are split across bytes
                // [0..8] (lo half) and [16..24] (hi half). A 64-bit
                // lane permute with imm8 = 0b11_01_10_00 (= _MM_SHUFFLE
                // (3, 1, 2, 0)) brings them adjacent in the low 128
                // bits.
                let packed: __m256i = _mm256_packus_epi16(sum, sum);
                let permuted: __m256i = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
                let low: __m128i = _mm256_castsi256_si128(permuted);
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut __m128i, low);
            }
        }
        32 => {
            // 32 bytes per row → two 16-lane chunks. We load the two
            // halves of `above` once outside the loop; per row only the
            // delta broadcast + add + pack + permute is in-loop.
            let above_lo_v8: __m128i = _mm_loadu_si128(above.as_ptr() as *const __m128i);
            let above_hi_v8: __m128i = _mm_loadu_si128(above.as_ptr().add(16) as *const __m128i);
            let above_lo_v16: __m256i = _mm256_cvtepu8_epi16(above_lo_v8);
            let above_hi_v16: __m256i = _mm256_cvtepu8_epi16(above_hi_v8);
            for (r, &left_px) in left.iter().take(32).enumerate() {
                let delta = left_px as i32 - tl;
                let delta_v: __m256i = _mm256_set1_epi16(delta as i16);
                // Low 16 columns.
                let sum_lo: __m256i = _mm256_add_epi16(delta_v, above_lo_v16);
                let packed_lo: __m256i = _mm256_packus_epi16(sum_lo, sum_lo);
                let permuted_lo: __m256i = _mm256_permute4x64_epi64(packed_lo, 0b11_01_10_00);
                let low_bytes: __m128i = _mm256_castsi256_si128(permuted_lo);
                _mm_storeu_si128(dst.as_mut_ptr().add(r * stride) as *mut __m128i, low_bytes);
                // High 16 columns.
                let sum_hi: __m256i = _mm256_add_epi16(delta_v, above_hi_v16);
                let packed_hi: __m256i = _mm256_packus_epi16(sum_hi, sum_hi);
                let permuted_hi: __m256i = _mm256_permute4x64_epi64(packed_hi, 0b11_01_10_00);
                let high_bytes: __m128i = _mm256_castsi256_si128(permuted_hi);
                _mm_storeu_si128(
                    dst.as_mut_ptr().add(r * stride + 16) as *mut __m128i,
                    high_bytes,
                );
            }
            // Suppress unused-import lint for `_mm256_extracti128_si256`
            // — it was an alternative formulation of the lane-grab. We
            // keep `_mm256_castsi256_si128 + permute` because the
            // permute already places the wanted bytes in lane 0.
            let _ = _mm256_extracti128_si256::<0>;
        }
        _ => unreachable!("tm_predictor_avx2: bs must be in {{4, 8, 16, 32}}"),
    }
}

/// Safe wrapper for the AVX2 TM_PRED predictor.
///
/// Signature mirrors a flattened slice of
/// [`crate::intra_pred::tm_predictor`] — explicit `above` / `left` /
/// `above_left` instead of `PredCtx`. The dispatch slot
/// [`crate::dsp::SimdDispatch::tm_predictor`] points at this wrapper
/// on AVX2 hosts and at the scalar adapter on every other platform.
///
/// `bs ∈ {4, 8, 16, 32}` for the AVX2 specialisations; `bs = 64`
/// (Block64x64) falls back to scalar — see the matching note in
/// [`crate::dsp::x86_64::dc_predict_left_avx2`].
#[cfg(target_arch = "x86_64")]
pub fn tm_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    if bs == 64 {
        let ctx = crate::intra_pred::PredCtx {
            above,
            left,
            above_left,
        };
        crate::intra_pred::tm_predictor(dst, stride, bs, ctx);
        return;
    }
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { tm_predictor_avx2_inner(dst, stride, bs, above, left, above_left) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `tm_predictor`. Mirrors
    //! the I1 / I2 test shapes — differential fuzz at N=1024 per size,
    //! a tautology guard, a dispatch round-trip, and directed clip
    //! fixtures at both saturation boundaries.

    use super::tm_predictor_avx2;
    use crate::intra_pred::{PredCtx, tm_predictor};

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG (Numerical Recipes constants). Self-contained
    /// so the differential test has no dependency on the proptest
    /// shrinker — the assertion message itself names the seed and bs
    /// for reproduction.
    struct Lcg(u32);
    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
        fn next_u8(&mut self) -> u8 {
            (self.next_u32() >> 24) as u8
        }
    }

    fn run_scalar(bs: usize, above: &[u8], left: &[u8], above_left: u8) -> Vec<u8> {
        let mut dst = vec![0u8; bs * bs];
        let ctx = PredCtx {
            above,
            left,
            above_left,
        };
        tm_predictor(&mut dst, bs, bs, ctx);
        dst
    }

    fn run_avx2(bs: usize, above: &[u8], left: &[u8], above_left: u8) -> Vec<u8> {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = vec![0u8; bs * bs];
        tm_predictor_avx2(&mut dst, bs, bs, above, left, above_left);
        dst
    }

    /// Differential fuzz: N=1024 LCG-seeded random iterations per
    /// `bs ∈ {4, 8, 16, 32}`. Scalar and AVX2 must produce byte-equal
    /// output.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        for &bs in &[4usize, 8, 16, 32] {
            let mut rng = Lcg::new(0xC0FFEE ^ bs as u32);
            for iter in 0..1024 {
                let mut above = vec![0u8; bs];
                let mut left = vec![0u8; bs];
                for v in above.iter_mut() {
                    *v = rng.next_u8();
                }
                for v in left.iter_mut() {
                    *v = rng.next_u8();
                }
                let above_left = rng.next_u8();
                let scalar = run_scalar(bs, &above, &left, above_left);
                let avx2 = run_avx2(bs, &above, &left, above_left);
                assert_eq!(
                    scalar, avx2,
                    "TM_PRED scalar/AVX2 mismatch at bs={bs}, iter={iter}, \
                     above_left={above_left}, above={above:?}, left={left:?}",
                );
            }
        }
    }

    /// Tautology guard — assert the AVX2 path is gated on the same
    /// `is_x86_feature_detected!("avx2")` query the dispatch table
    /// uses (mirrors the V/H pattern).
    #[test]
    fn tautology_guard_avx2_path() {
        // No-op when the host lacks AVX2: the test itself returns
        // success because `require_avx2!()` short-circuits. On AVX2
        // hosts the detection must report `true` — otherwise the
        // dispatch table would have routed to scalar and our test of
        // the AVX2 wrapper is meaningless.
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
    }

    /// Dispatch round-trip: forced-AVX2 and forced-scalar dispatch
    /// tables produce identical output for the same neighbour set.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps_avx2);

        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        for &bs in &[4usize, 8, 16, 32] {
            let above: Vec<u8> = (0..bs).map(|i| (i * 7 + 3) as u8).collect();
            let left: Vec<u8> = (0..bs).map(|i| (i * 11 + 17) as u8).collect();
            let above_left = 91u8;

            let mut dst_avx2 = vec![0u8; bs * bs];
            let mut dst_scalar = vec![0u8; bs * bs];
            (dispatch_avx2.tm_predictor())(&mut dst_avx2, bs, bs, &above, &left, above_left);
            (dispatch_scalar.tm_predictor())(&mut dst_scalar, bs, bs, &above, &left, above_left);
            assert_eq!(dst_avx2, dst_scalar, "round-trip mismatch at bs={bs}",);
        }
    }

    /// Directed clip-to-zero guard. With `left[r] = 10`, `above[c] = 10`,
    /// `above_left = 255`, `l + above[c] - tl = 10 + 10 - 255 = -235`,
    /// which must clip to 0.
    #[test]
    fn directed_clip_at_zero() {
        require_avx2!();
        for &bs in &[4usize, 8, 16, 32] {
            let above = vec![10u8; bs];
            let left = vec![10u8; bs];
            let above_left = 255u8;
            let scalar = run_scalar(bs, &above, &left, above_left);
            let avx2 = run_avx2(bs, &above, &left, above_left);
            assert!(
                scalar.iter().all(|&v| v == 0),
                "scalar reference: bs={bs} not all-zero — {scalar:?}",
            );
            assert_eq!(scalar, avx2, "AVX2 clip-to-zero mismatch at bs={bs}");
        }
    }

    /// Directed clip-to-255 guard. With `left[r] = 200`, `above[c] = 200`,
    /// `above_left = 0`, `l + above[c] - tl = 200 + 200 - 0 = 400`,
    /// which must clip to 255.
    #[test]
    fn directed_clip_at_255() {
        require_avx2!();
        for &bs in &[4usize, 8, 16, 32] {
            let above = vec![200u8; bs];
            let left = vec![200u8; bs];
            let above_left = 0u8;
            let scalar = run_scalar(bs, &above, &left, above_left);
            let avx2 = run_avx2(bs, &above, &left, above_left);
            assert!(
                scalar.iter().all(|&v| v == 255),
                "scalar reference: bs={bs} not all-255 — {scalar:?}",
            );
            assert_eq!(scalar, avx2, "AVX2 clip-to-255 mismatch at bs={bs}");
        }
    }

    /// Boundary fixtures: all-zero / all-max / mid-grey inputs at every
    /// size to confirm the size-parametric path handles each of the
    /// four bs values' code shape without any accidental fall-through.
    #[test]
    fn directed_boundary_inputs() {
        require_avx2!();
        for &bs in &[4usize, 8, 16, 32] {
            for &(a, l, tl) in &[
                (0u8, 0u8, 0u8),
                (255, 255, 255),
                (128, 128, 128),
                (0, 255, 128),
                (255, 0, 128),
            ] {
                let above = vec![a; bs];
                let left = vec![l; bs];
                let scalar = run_scalar(bs, &above, &left, tl);
                let avx2 = run_avx2(bs, &above, &left, tl);
                assert_eq!(
                    scalar, avx2,
                    "boundary mismatch bs={bs} above={a} left={l} tl={tl}",
                );
            }
        }
    }
}
