//! 4×4 DC intra-predictor — x86_64 AVX2 implementation (G4.3).
//!
//! Mirrors the safe-wrapper convention established by the G4.1 pilot
//! ([`crate::dsp::x86_64::idct8x8_avx2`]). The scalar reference is
//! [`crate::intra_pred::dc_predictor`] with `bs = 4`.
//!
//! ## Algorithm
//!
//! DC_PRED averages the 4 `above` neighbour pixels and the 4 `left`
//! neighbour pixels, rounds to nearest with halves up, then fills the
//! entire 4×4 block with that constant byte. The libvpx scalar (and
//! our Rust port) compute:
//!
//! ```text
//! sum  = Σ above[0..4] + Σ left[0..4]
//! dc   = (sum + 4) / 8     // half-up rounding
//! ```
//!
//! For 4×4 the SIMD speedup is modest (we're shifting 16 bytes total)
//! but the slot satisfies the G4.3 intra-pred family exit criterion
//! and the kernel doubles as a worked example of integer-reduction
//! patterns for the larger DC_PRED sizes (8×8 / 16×16 / 32×32) that
//! land in subsequent G4.x phases.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Per-neighbour loads use `_mm_cvtsi32_si128` (4-byte) sourced via
//! `core::ptr::read_unaligned` — no alignment assumption on the
//! `above` / `left` slices. Per-row stores into `dst` use
//! `_mm_cvtsi128_si32` + `write_unaligned`. The grep sentinel below
//! is checked by `scripts/check-simd-sentinels.sh` / `.ps1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_cvtsi128_si32, _mm_cvtsi32_si128, _mm_sad_epu8, _mm_set1_epi8, _mm_setzero_si128,
};

/// AVX2 implementation of the 4×4 DC intra-predictor.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`dc_predictor_4x4_avx2`]) when `caps.avx2 == true`.
///
/// `above` and `left` must each have at least 4 valid u8s. `dst` must
/// have at least `3 * stride + 4` bytes of valid storage.
///
/// # Implementation
///
/// Loads 4 bytes from each of `above` and `left`, packs both into the
/// low 8 bytes of an `__m128i`, then uses `_mm_sad_epu8` to sum the
/// 8 lanes against a zero register — this gives a 16-bit sum in the
/// low 16 bits of the result with one instruction. We then add 4
/// (round-up half), shift right by 3 (= /8), and broadcast the byte
/// across a 32-bit lane for the 4 per-row stores.
///
/// Although named `_avx2`, the intrinsics used here are SSE2 (the
/// safe wrapper still requires `caps.avx2 == true` for consistency
/// with the rest of the G4 dispatch — we'd swap to a 128-bit AVX
/// register if we tightened the caps gating). Per HLD V6 §G4 M6-6
/// the unaligned-load discipline applies regardless.
// TODO(G4-tiering): Caps doesn't model SSE2/SSE4.1 separately today.
// When a finer-grained caps struct lands, retag this slot from
// caps.avx2 → caps.sse2 so older hosts run the same kernel. See
// `notes/g4-coordination.md` § "Per-slot caps tiering follow-ups".
// G4: loads from `dst` use _mm_cvtsi128_si32 / read_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dc_predictor_4x4_avx2_inner(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]) {
    debug_assert!(above.len() >= 4);
    debug_assert!(left.len() >= 4);
    debug_assert!(dst.len() >= stride * 3 + 4, "dst too small for 4x4 block");

    // (1) Load 4 bytes from `above` and `left` as i32 (unaligned).
    let above_i32: i32 = core::ptr::read_unaligned(above.as_ptr() as *const i32);
    let left_i32: i32 = core::ptr::read_unaligned(left.as_ptr() as *const i32);
    // (2) Place them in the low and next-up bytes of an __m128i. We
    // shift the `left` value up by 4 bytes by inserting at i32-lane 1.
    // The simplest path: insert both into the same i64 lane of a
    // 128-bit register via setr_epi32-equivalent compositing.
    // Equivalent C: `__m128i v = _mm_set_epi32(0, 0, left_i32, above_i32);`
    let v_above = _mm_cvtsi32_si128(above_i32);
    let v_left = _mm_cvtsi32_si128(left_i32);
    // Compose: low 4 bytes = above, next 4 = left, high 8 = zero.
    let v_packed = core::arch::x86_64::_mm_unpacklo_epi32(v_above, v_left);
    // (3) Sum-of-absolute-differences vs zero is just the sum of bytes
    // (since all bytes are non-negative). SAD result is a 16-bit sum
    // in the low 16 bits of each 64-bit lane; we only care about the
    // low lane (the high lane sums zero bytes and is therefore 0).
    let zero = _mm_setzero_si128();
    let sum_v = _mm_sad_epu8(v_packed, zero);
    // (4) Extract the 16-bit sum, add 4 for half-up rounding, divide
    // by 8 (the count = 2 * bs = 8). All arithmetic stays in scalar
    // i32 from here; the vector form gains nothing past this point.
    let sum = _mm_cvtsi128_si32(sum_v) as u32;
    let dc: u8 = ((sum + 4) >> 3) as u8;
    // (5) Broadcast `dc` across an __m128i and store the low 4 bytes
    // into each of the 4 rows.
    let fill_v: __m128i = _mm_set1_epi8(dc as i8);
    let fill_i32 = _mm_cvtsi128_si32(fill_v);
    for r in 0..4 {
        let row_start = r * stride;
        // G4: loads from `dst` use _mm_cvtsi128_si32 / write_unaligned (unaligned)
        core::ptr::write_unaligned(dst.as_mut_ptr().add(row_start) as *mut i32, fill_i32);
    }
}

/// Safe wrapper for the AVX2 4×4 DC predictor.
///
/// Signature mirrors a specialised slice of
/// [`crate::intra_pred::dc_predictor`] (with `bs = 4` baked in and a
/// flat `above` / `left` arg pair instead of `PredCtx`). The dispatch
/// slot [`crate::dsp::SimdDispatch::dc_predictor_4x4`] points at this
/// wrapper on AVX2 hosts and at the scalar adapter on every other
/// platform.
#[cfg(target_arch = "x86_64")]
pub fn dc_predictor_4x4_avx2(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock
    // init time (see `crate::dsp::dispatch::init`) and are monotonic
    // for the process lifetime. Test paths in this module's tests
    // gate each call on a runtime `is_x86_feature_detected!("avx2")`
    // check.
    unsafe { dc_predictor_4x4_avx2_inner(dst, stride, above, left) }
}

// Miri cannot interpret AVX2 intrinsics (`_mm256_*` etc.), so this
// `tests` module is excluded from `cargo +nightly miri test`. The
// AVX2 kernel itself is x86_64-only, and the scalar reference is
// already covered by `crate::intra_pred::tests` which DOES run
// under miri. See `scripts/run_miri.ps1` for the full miri lane.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `dc_predictor_4x4`.
    //!
    //! Strategy set: a single proptest over uniform random neighbours
    //! (the AVX2 finisher fully consumes any 4+4-byte combination), plus
    //! a handful of directed `#[test]` fixtures for the boundary seams
    //! — all-0 / all-255 / alternating — and a sweep of the half-up
    //! rounding bucket around `sum ∈ {8..=12}`. A separate
    //! `dc_predictor_4x4_rounding_seams` test sweeps every reachable
    //! `(above_sum, left_sum)` pair in `[0, 20]²`.

    use super::dc_predictor_4x4_avx2;
    use crate::intra_pred::{dc_predictor, PredCtx};
    use proptest::prelude::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn run_avx2(above: &[u8; 4], left: &[u8; 4]) -> [u8; 16] {
        assert!(
            is_x86_feature_detected!("avx2"),
            "run_avx2 called on non-AVX2 host"
        );
        let mut dst = [0u8; 16];
        dc_predictor_4x4_avx2(&mut dst, 4, above, left);
        dst
    }

    fn run_scalar(above: &[u8; 4], left: &[u8; 4]) -> [u8; 16] {
        let mut dst = [0u8; 16];
        let ctx = PredCtx {
            above,
            left,
            above_left: 0, // unused by DC_PRED
        };
        dc_predictor(&mut dst, 4, 4, ctx);
        dst
    }

    fn assert_match(above: &[u8; 4], left: &[u8; 4]) {
        let scalar = run_scalar(above, left);
        let avx2 = run_avx2(above, left);
        assert_eq!(
            scalar, avx2,
            "AVX2 mismatch:\nabove={above:?} left={left:?}\n\
             scalar={scalar:?}\navx2={avx2:?}"
        );
    }

    fn neighbour_strategy() -> impl Strategy<Value = ([u8; 4], [u8; 4])> {
        let v = prop::collection::vec(any::<u8>(), 4).prop_map(|x| {
            let mut out = [0u8; 4];
            out.copy_from_slice(&x);
            out
        });
        (v.clone(), v)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 1024,
            .. ProptestConfig::default()
        })]

        #[test]
        fn dc_predictor_4x4_scalar_eq_avx2_proptest(
            (above, left) in neighbour_strategy(),
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let scalar = run_scalar(&above, &left);
            let avx2 = run_avx2(&above, &left);
            prop_assert_eq!(scalar, avx2);
        }
    }

    // Directed pathological fixtures exercising boundary rounding /
    // saturation seams: all-0, all-255, alternating, and half-up
    // rounding seams around sum ∈ {8, 9, 10, 11, 12}. Previously these
    // ran as a `prop_oneof![ Just(...); 10]` proptest with `cases =
    // 256`, which re-executed the same 10 fixed inputs ~25× each.
    // Reviewer minor (G4 rollout review): convert to one `#[test]` per
    // input to make intent explicit and avoid duplicate work.

    #[test]
    fn dc_predictor_4x4_pathological_all_zero_zero() {
        require_avx2!();
        assert_match(&[0u8; 4], &[0u8; 4]);
    }

    #[test]
    fn dc_predictor_4x4_pathological_all_max_max() {
        require_avx2!();
        assert_match(&[255u8; 4], &[255u8; 4]);
    }

    #[test]
    fn dc_predictor_4x4_pathological_zero_max() {
        require_avx2!();
        assert_match(&[0u8; 4], &[255u8; 4]);
    }

    #[test]
    fn dc_predictor_4x4_pathological_max_zero() {
        require_avx2!();
        assert_match(&[255u8; 4], &[0u8; 4]);
    }

    #[test]
    fn dc_predictor_4x4_pathological_alternating() {
        require_avx2!();
        assert_match(&[0, 255, 0, 255], &[255, 0, 255, 0]);
    }

    /// sum = 8, dc = (8 + 4) / 8 = 1.
    #[test]
    fn dc_predictor_4x4_pathological_round_sum_8() {
        require_avx2!();
        assert_match(&[1, 1, 1, 1], &[1, 1, 1, 1]);
    }

    /// sum = 9, dc = (9 + 4) / 8 = 1.
    #[test]
    fn dc_predictor_4x4_pathological_round_sum_9() {
        require_avx2!();
        assert_match(&[1, 1, 1, 1], &[1, 1, 1, 2]);
    }

    /// sum = 10, dc = (10 + 4) / 8 = 1.
    #[test]
    fn dc_predictor_4x4_pathological_round_sum_10() {
        require_avx2!();
        assert_match(&[1, 1, 1, 1], &[1, 1, 2, 2]);
    }

    /// sum = 11, dc = (11 + 4) / 8 = 1.
    #[test]
    fn dc_predictor_4x4_pathological_round_sum_11() {
        require_avx2!();
        assert_match(&[1, 1, 1, 1], &[1, 2, 2, 2]);
    }

    /// sum = 12, dc = (12 + 4) / 8 = 2 — the first sum that rounds up
    /// into the next dc bucket.
    #[test]
    fn dc_predictor_4x4_pathological_round_sum_12() {
        require_avx2!();
        assert_match(&[1, 1, 1, 1], &[2, 2, 2, 2]);
    }

    /// Directed fixture: all-zero neighbours produces an all-zero block.
    #[test]
    fn dc_predictor_4x4_all_zero() {
        require_avx2!();
        let above = [0u8; 4];
        let left = [0u8; 4];
        assert_match(&above, &left);
    }

    /// Directed fixture: all-255 neighbours produces an all-255 block.
    #[test]
    fn dc_predictor_4x4_all_max() {
        require_avx2!();
        let above = [255u8; 4];
        let left = [255u8; 4];
        assert_match(&above, &left);
    }

    /// Directed fixture: every (sum mod 8) seam — for sum = 0..=32 we
    /// hit each rounding bucket boundary at least twice.
    #[test]
    fn dc_predictor_4x4_rounding_seams() {
        require_avx2!();
        for above_sum in 0..=20u32 {
            for left_sum in 0..=20u32 {
                // Build neighbour arrays whose byte-sum equals the chosen
                // target sum. We use ascending fill: above = [s/4; 4] is
                // too coarse, so pick a four-byte split that sums to
                // `above_sum`.
                let a0 = (above_sum.min(255)) as u8;
                let above = [
                    a0,
                    (above_sum.saturating_sub(a0 as u32).min(255)) as u8,
                    0,
                    0,
                ];
                let real_above_sum = above.iter().map(|&x| x as u32).sum::<u32>();
                if real_above_sum != above_sum {
                    continue; // pick another iteration; not all sums are reachable with this split
                }
                let l0 = (left_sum.min(255)) as u8;
                let left = [
                    l0,
                    (left_sum.saturating_sub(l0 as u32).min(255)) as u8,
                    0,
                    0,
                ];
                let real_left_sum = left.iter().map(|&x| x as u32).sum::<u32>();
                if real_left_sum != left_sum {
                    continue;
                }
                assert_match(&above, &left);
            }
        }
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// must agree.
    #[test]
    fn dc_predictor_4x4_dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::caps::detect;
        use crate::dsp::SimdDispatch;

        let mut caps = detect();
        caps.avx2 = true;
        let dispatch_avx2 = SimdDispatch::build(caps);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let dispatch_scalar = SimdDispatch::build(caps_scalar);

        let above = [10u8, 50, 100, 200];
        let left = [200u8, 100, 50, 10];
        let mut dst_avx2 = [0u8; 16];
        let mut dst_scalar = [0u8; 16];
        (dispatch_avx2.dc_predictor_4x4())(&mut dst_avx2, 4, &above, &left);
        (dispatch_scalar.dc_predictor_4x4())(&mut dst_scalar, 4, &above, &left);
        assert_eq!(dst_avx2, dst_scalar);
    }

    /// Misaligned `dst` regression guard — same approach as idct4x4_avx2.
    /// All loads/stores on `dst` use 4-byte unaligned ops, so this
    /// checks per-skew arithmetic correctness rather than alignment-
    /// fault paths.
    #[test]
    fn dc_predictor_4x4_misaligned_dst_pointers() {
        require_avx2!();
        let above = [10u8, 20, 30, 40];
        let left = [50u8, 60, 70, 80];
        for skew in [0usize, 1, 3, 7, 15] {
            let mut buf: Box<[u8; 32]> = Box::new([0u8; 32]);
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (i as u8).wrapping_mul(13);
            }
            let mut scalar_buf = *buf;
            dc_predictor_4x4_avx2(&mut buf[skew..skew + 16], 4, &above, &left);
            let ctx = PredCtx {
                above: &above,
                left: &left,
                above_left: 0,
            };
            dc_predictor(&mut scalar_buf[skew..skew + 16], 4, 4, ctx);
            assert_eq!(
                &buf[..],
                &scalar_buf[..],
                "AVX2/scalar mismatch at skew={skew}"
            );
        }
    }
}
