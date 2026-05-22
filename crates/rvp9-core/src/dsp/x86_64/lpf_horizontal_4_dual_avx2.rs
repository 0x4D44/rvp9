//! 4-tap horizontal loop filter, dual variant — x86_64 AVX2 (Wave H H2).
//!
//! Scalar reference: [`crate::loopfilter::lpf_horizontal_4_dual`]
//! (`vpx_lpf_horizontal_4_dual_c`, `vpx_dsp/loopfilter.c:189-195`).
//!
//! The "dual" wraps two adjacent invocations of the single-variant
//! kernel on adjacent 8-column groups (16 columns total) with
//! independent threshold sets — i.e. the scalar literally does:
//!
//! ```ignore
//! lpf_horizontal_4(pixels, s_index,     pitch, b0, l0, t0);
//! lpf_horizontal_4(pixels, s_index + 8, pitch, b1, l1, t1);
//! ```
//!
//! The two halves operate on disjoint column ranges and disjoint
//! threshold sets — there is no inter-half data flow — so the
//! simplest correct AVX2 implementation chains two calls to the
//! single-variant AVX2 wrapper [`lpf_horizontal_4_avx2`]. The single
//! wrapper already encapsulates the G3 row-load + filter4 pipeline.
//! A more sophisticated "process all 16 columns in a single 256-bit
//! AVX2 register" rewrite is a future optimisation that would have
//! to re-verify bit-exactness against the scalar.
//!
//! Bit-exactness with the scalar dual follows directly from
//! bit-exactness of the single AVX2 kernel against the scalar single
//! (verified at Wave G G3 close — see
//! [`crate::dsp::x86_64::lpf_horizontal_4_avx2`] proptests).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! This wrapper performs NO direct memory access; both halves' loads
//! and stores happen inside `lpf_horizontal_4_avx2_inner` which
//! already uses `_mm_loadl_epi64` and `_mm_storel_epi64` per G3.
//! Sentinel:
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — delegated to single inner

use super::lpf_horizontal_4_avx2::lpf_horizontal_4_avx2;

/// Safe wrapper for the AVX2 horizontal 4-tap dual loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_horizontal_4_dual`]. The
/// dispatch slot [`crate::dsp::SimdDispatch::lpf_horizontal_4_dual`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// otherwise.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub fn lpf_horizontal_4_dual_avx2(
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
    // Two-singles approach — mirrors the scalar reference structure
    // exactly. The single AVX2 wrapper carries its own SAFETY note
    // (caps invariant via dispatch OnceLock + scalar bounds contract).
    lpf_horizontal_4_avx2(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_horizontal_4_avx2(pixels, s_index + 8, pitch, blimit1, limit1, thresh1);
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_horizontal_4_dual`.

    use super::lpf_horizontal_4_dual_avx2;
    use crate::loopfilter::lpf_horizontal_4_dual;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build a 16-row × 16-wide test buffer at pitch 16. The
    /// horizontal edge sits between row 7 and row 8 (`s_index = 8 *
    /// pitch`); each tap row (-4..=3 relative to the edge) lands in
    /// valid memory, and both column halves (cols 0..=7 and 8..=15)
    /// fit within the row.
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
        lpf_horizontal_4_dual(
            &mut scalar, s_index, pitch, blimit0, limit0, thresh0, blimit1, limit1, thresh1,
        );
        lpf_horizontal_4_dual_avx2(
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
        let mut s: u32 = 0xA11C_E0F0;
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
                &buf,
                8 * 16,
                16,
                blimit0,
                limit0,
                thresh0,
                blimit1,
                limit1,
                thresh1,
            );
            assert_eq!(so, ao, "iter {}: scalar vs AVX2 mismatch", i);
        }
    }

    /// Tautology guard: scalar dual must alter the buffer for an
    /// obviously-filterable horizontal edge across both halves.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let pitch = 16;
        let mut buf = vec![100u8; 16 * pitch];
        for c in 0..16 {
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
        lpf_horizontal_4_dual(&mut copy, 8 * pitch, pitch, 255, 63, 0, 255, 63, 0);
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
        let (s, a) = run_pair(&buf, 8 * 16, 16, 255, 63, 7, 1, 1, 0);
        assert_eq!(s, a);
    }

    /// Round-trip through SimdDispatch: forced-AVX2 and forced-scalar
    /// dispatches must agree on a deterministic input.
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

        let buf = make_buf(0xDEAD_BEEF);
        let mut b_a = buf.clone();
        let mut b_s = buf;
        (d_avx2.lpf_horizontal_4_dual())(&mut b_a, 8 * 16, 16, 200, 30, 7, 220, 20, 5);
        (d_scalar.lpf_horizontal_4_dual())(&mut b_s, 8 * 16, 16, 200, 30, 7, 220, 20, 5);
        assert_eq!(b_a, b_s);
    }
}
