//! 4-tap vertical loop filter, dual variant — x86_64 AVX2 (Wave H H2).
//!
//! Scalar reference: [`crate::loopfilter::lpf_vertical_4_dual`]
//! (`vpx_lpf_vertical_4_dual_c`, `vpx_dsp/loopfilter.c:154-160`).
//!
//! The "dual" wraps two adjacent invocations of the single-variant
//! kernel on adjacent 8-row × 8-col groups (16 rows total) with
//! independent threshold sets — i.e. the scalar literally does:
//!
//! ```ignore
//! lpf_vertical_4(pixels, s_index,             pitch, b0, l0, t0);
//! lpf_vertical_4(pixels, s_index + 8 * pitch, pitch, b1, l1, t1);
//! ```
//!
//! The two halves operate on disjoint row ranges and disjoint
//! threshold sets — there is no inter-half data flow — so the
//! simplest correct AVX2 implementation chains two calls to the
//! single-variant AVX2 wrapper [`lpf_vertical_4_avx2`]. The single
//! wrapper already encapsulates the F10 transpose-8×8 + filter4
//! pipeline; a more sophisticated "process both halves in a single
//! AVX2 (256-bit / 16-row) register" rewrite is a future
//! optimisation and would have to re-verify bit-exactness against the
//! scalar.
//!
//! Bit-exactness with the scalar dual follows directly from
//! bit-exactness of the single AVX2 kernel against the scalar single
//! (verified at Wave F G F10 close — see
//! [`crate::dsp::x86_64::lpf_vertical_4_avx2`] proptests).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! This wrapper performs NO direct memory access; both halves' loads
//! and stores happen inside `lpf_vertical_4_avx2_inner` which already
//! uses `_mm_loadl_epi64` and `write_unaligned` per F10. Sentinel:
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — delegated to single inner

use super::lpf_vertical_4_avx2::lpf_vertical_4_avx2;

/// Safe wrapper for the AVX2 vertical 4-tap dual loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_vertical_4_dual`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_vertical_4_dual`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// otherwise.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub fn lpf_vertical_4_dual_avx2(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    let second_index = super::lpf_bounds::validate(pixels.len(), s_index, pitch, 4, true, 16);

    // Two-singles approach — mirrors the scalar reference structure
    // exactly. The single AVX2 wrapper carries its own SAFETY note
    // (caps invariant via dispatch OnceLock + scalar bounds contract).
    lpf_vertical_4_avx2(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_vertical_4_avx2(pixels, second_index, pitch, blimit1, limit1, thresh1);
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_vertical_4_dual`.

    use super::lpf_vertical_4_dual_avx2;
    use crate::loopfilter::lpf_vertical_4_dual;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build a 16-row × 16-wide test buffer. The vertical edge sits
    /// at column 8 so each row's 8-byte window `[s-4..s+4]` lands
    /// entirely in valid memory across all 16 rows.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 16 * 16];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *x = (s >> 16) as u8;
        }
        v
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pair(
        buf: &[u8],
        s_index: usize,
        pitch: usize,
        blimit0: u8,
        limit0: u8,
        thresh0: u8,
        blimit1: u8,
        limit1: u8,
        thresh1: u8,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut scalar = buf.to_vec();
        let mut avx2 = buf.to_vec();
        lpf_vertical_4_dual(
            &mut scalar,
            s_index,
            pitch,
            blimit0,
            limit0,
            thresh0,
            blimit1,
            limit1,
            thresh1,
        );
        lpf_vertical_4_dual_avx2(
            &mut avx2, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
        );
        (scalar, avx2)
    }

    /// Deterministic-RNG differential stress: 1024 iterations of
    /// scalar-dual vs AVX2-dual across random buffers and independent
    /// threshold sets. Mirrors the H2 TDD safety net.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut s: u32 = 0xC0DE_BABE;
        let next = |s: &mut u32| -> u32 {
            *s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *s
        };
        for i in 0..1024 {
            let buf = make_buf(next(&mut s));
            let blimit0 = (next(&mut s) % 255 + 1) as u8;
            let limit0 = (next(&mut s) % 63 + 1) as u8;
            let thresh0 = (next(&mut s) % 16) as u8;
            let blimit1 = (next(&mut s) % 255 + 1) as u8;
            let limit1 = (next(&mut s) % 63 + 1) as u8;
            let thresh1 = (next(&mut s) % 16) as u8;
            let (so, ao) = run_pair(
                &buf, 8, 16, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
            );
            assert_eq!(so, ao, "iter {}: scalar vs AVX2 mismatch", i);
        }
    }

    /// Tautology guard: the scalar dual must actually modify the
    /// buffer for an obviously-filterable edge spanning both halves.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let mut buf = vec![100u8; 16 * 16];
        for r in 0..16 {
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
        lpf_vertical_4_dual(&mut copy, 8, 16, 255, 63, 0, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar dual must alter the buffer for filterable edges"
        );
    }

    /// Asymmetric threshold sets: lower half hot, upper half masked
    /// off — round-trip through both kernels must agree.
    #[test]
    fn asymmetric_threshold_sets() {
        require_avx2!();
        let buf = make_buf(0xFEED_FACE);
        let (s, a) = run_pair(&buf, 8, 16, 255, 63, 7, 1, 1, 0);
        assert_eq!(s, a);
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// dispatches must agree on a deterministic input.
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

        let buf = make_buf(0xDEAD_BEEF);
        let mut b_a = buf.clone();
        let mut b_s = buf;
        (d_avx2.lpf_vertical_4_dual())(&mut b_a, 8, 16, 200, 30, 7, 220, 20, 5);
        (d_scalar.lpf_vertical_4_dual())(&mut b_s, 8, 16, 200, 30, 7, 220, 20, 5);
        assert_eq!(b_a, b_s);
    }
}
