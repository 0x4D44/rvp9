//! 16-tap horizontal loop filter, dual variant — x86_64 AVX2 (Wave H H4).
//!
//! Scalar reference: [`crate::loopfilter::lpf_horizontal_16_dual`]
//! (`vpx_lpf_horizontal_16_dual_c`, `vpx_dsp/loopfilter.c:323-326`).
//!
//! Unlike the 4/8-tap duals, libvpx's 16-tap dual shares a **single
//! threshold set** across both halves (6-arg signature). The scalar
//! implementation calls the shared inner driver `mb_lpf_horizontal_edge_w`
//! with `count = 2`, which sweeps `2 * 8 = 16` adjacent columns starting
//! at `s_index` — each column processed independently with the same
//! `(blimit, limit, thresh)` triple.
//!
//! Because the 16-column sweep is column-wise independent, calling the
//! single-variant kernel (`count = 1` → 8 columns) **twice** — once at
//! `s_index` and once at `s_index + 8` with the **same** threshold
//! triple — covers the identical column range with identical
//! semantics:
//!
//! ```ignore
//! lpf_horizontal_16(pixels, s_index,     pitch, b, l, t);
//! lpf_horizontal_16(pixels, s_index + 8, pitch, b, l, t);
//! ```
//!
//! There is no inter-half data flow (each column's 16-pixel vertical
//! window touches only that column), so two-singles delegation is
//! bit-exact with the scalar dual. Mirrors the H2/H3 strategy.
//!
//! A future optimisation could process all 16 columns in a single 256-bit
//! AVX2 register, but it would have to re-verify bit-exactness — and
//! the single AVX2 kernel already amortises the per-column setup well.
//!
//! Bit-exactness with the scalar dual follows directly from
//! bit-exactness of the single AVX2 kernel against the scalar single
//! (verified at Wave G G6 close + Wave H H7 fix — see
//! [`crate::dsp::x86_64::lpf_horizontal_16_avx2`] proptests).
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! This wrapper performs NO direct memory access; both halves' loads
//! and stores happen inside `lpf_horizontal_16_avx2_inner`. Sentinel:
// G4: loads from `dst` use _mm_loadl_epi64 (unaligned) — delegated to single inner

use super::lpf_horizontal_16_avx2::lpf_horizontal_16_avx2;

/// Safe wrapper for the AVX2 horizontal 16-tap dual loop filter.
///
/// Signature mirrors [`crate::loopfilter::lpf_horizontal_16_dual`]
/// (6-arg, shared threshold set). The dispatch slot
/// [`crate::dsp::SimdDispatch::lpf_horizontal_16_dual`] points at this
/// wrapper on AVX2 hosts and at the scalar reference otherwise.
#[cfg(target_arch = "x86_64")]
pub fn lpf_horizontal_16_dual_avx2(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    let second_index = super::lpf_bounds::validate(pixels.len(), s_index, pitch, 8, false, 16);

    // Two-singles approach — both halves share the same threshold set
    // (libvpx convention). The single AVX2 wrapper carries its own
    // SAFETY note (caps invariant via dispatch OnceLock + scalar bounds
    // contract).
    lpf_horizontal_16_avx2(pixels, s_index, pitch, blimit, limit, thresh);
    lpf_horizontal_16_avx2(pixels, second_index, pitch, blimit, limit, thresh);
}

// Miri cannot interpret AVX2 intrinsics, so this tests module is
// excluded from `cargo +nightly miri test`. The scalar reference is
// covered by `crate::loopfilter` tests which DO run under miri.
#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `lpf_horizontal_16_dual`.

    use super::lpf_horizontal_16_dual_avx2;
    use crate::loopfilter::lpf_horizontal_16_dual;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Build a 32-row × 32-wide test buffer. The horizontal edge sits
    /// between row 15 and row 16 (`s_index = 16 * pitch`); the 16-tap
    /// filter touches rows `[s-8*pitch .. s+8*pitch)` so both vertical
    /// tap stacks fit. The 16-column sweep (cols 0..=15) fits within
    /// the row.
    fn make_buf(seed: u32) -> Vec<u8> {
        let mut v = vec![0u8; 32 * 32];
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
        lpf_horizontal_16_dual(&mut scalar, s_index, pitch, blimit, limit, thresh);
        lpf_horizontal_16_dual_avx2(&mut avx2, s_index, pitch, blimit, limit, thresh);
        (scalar, avx2)
    }

    /// Deterministic-RNG differential stress: 1024 iterations of
    /// scalar-dual vs AVX2-dual across random buffers and shared
    /// threshold triples. Mirrors the H2/H3 TDD safety net.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut s: u32 = 0xA110_CAFE;
        let next = |s: &mut u32| -> u32 {
            *s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *s
        };
        let pitch = 32;
        for i in 0..1024 {
            let buf = make_buf(next(&mut s));
            let blimit = (next(&mut s) % 255 + 1) as u8;
            let limit = (next(&mut s) % 63 + 1) as u8;
            let thresh = (next(&mut s) % 16) as u8;
            let (so, ao) = run_pair(&buf, 16 * pitch, pitch, blimit, limit, thresh);
            assert_eq!(so, ao, "iter {}: scalar vs AVX2 mismatch", i);
        }
    }

    /// Tautology guard: scalar dual must alter the buffer for an
    /// obviously-filterable horizontal edge across all 16 columns.
    #[test]
    fn tautology_guard_scalar_modifies_buffer() {
        let pitch = 32;
        let mut buf = vec![100u8; 32 * pitch];
        // Wide flat region with a small step right at the edge —
        // engages filter16 over all 16 swept columns.
        for r in 0..32 {
            for c in 0..16 {
                buf[r * pitch + c] = if r < 16 { 99 } else { 101 };
            }
        }
        let mut copy = buf.clone();
        lpf_horizontal_16_dual(&mut copy, 16 * pitch, pitch, 255, 63, 0);
        assert_ne!(
            buf, copy,
            "scalar dual must alter the buffer for filterable edges"
        );
    }

    /// Flat-region stress: a perfectly flat 32x32 patch with a tiny
    /// step at the edge engages the wider 15-tap filter (flat_mask4 &
    /// flat_mask5 both become 0xff) so we exercise the smoothing branch
    /// across all 16 columns.
    #[test]
    fn directed_flat_engages_wide_filter() {
        require_avx2!();
        let pitch = 32;
        let mut buf = vec![128u8; 32 * pitch];
        for r in 0..32 {
            for c in 0..pitch {
                let v = if r < 16 { 127 } else { 130 };
                buf[r * pitch + c] = v;
            }
        }
        let (s, a) = run_pair(&buf, 16 * pitch, pitch, 255, 63, 7);
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
        (d_avx2.lpf_horizontal_16_dual())(&mut b_a, 16 * 32, 32, 200, 30, 7);
        (d_scalar.lpf_horizontal_16_dual())(&mut b_s, 16 * 32, 32, 200, 30, 7);
        assert_eq!(b_a, b_s);
    }
}
