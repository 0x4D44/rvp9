//! Full 2-D separable 8-tap sub-pel convolution — x86_64 AVX2
//! (Wave J J5).
//!
//! Fires when an inter block has a non-zero sub-pel offset in *both*
//! dimensions. Mirrors libvpx `vpx_convolve8_c`
//! (`vpx_dsp/vpx_convolve.c:158-190`) and the scalar reference
//! [`crate::inter::filter::convolve8`]: a horizontal pass filters
//! `h + 7` source rows into a `u8` intermediate buffer, then a vertical
//! pass filters that buffer into `dst`.
//!
//! ## Composition
//!
//! Rather than fuse the two passes, this kernel reuses the already
//! bit-exact AVX2 single-axis kernels:
//!
//! 1. [`convolve8_horiz_avx2`] over `h + 7` rows → intermediate
//!    (stride 64, clipped to `u8`, exactly as the scalar intermediate).
//! 2. [`convolve8_vert_avx2`] over the intermediate → `dst`.
//!
//! The intermediate's row 0 is already three rows above the destination
//! origin (the caller pre-shifts `src` up by three rows for the
//! horizontal pass), so the vertical pass consumes it as-is — matching
//! the scalar driver's hand-off.
//!
//! ## Buffer sizing
//!
//! The intermediate is `[u8; 64 * 72]`. The horizontal pass writes
//! `intermediate_height = h + 7 ≤ 71` rows at stride 64 (max write
//! index `(h + 6) * 64 + 64 = 71 * 64 = 4544`); the vertical pass reads
//! `(h - 1 + 7) * 64 + w ≤ 4544` bytes. Both fit within `64 * 72 =
//! 4608`.
//!
//! ## Contract
//!
//! Both sub-pel filter rows have taps fitting i8 (the full-pel row is
//! routed elsewhere) — inherited from the single-axis kernels.

#[cfg(target_arch = "x86_64")]
use super::{convolve8_horiz_avx2, convolve8_vert_avx2};

/// Stride of the intermediate buffer (libvpx uses 64; any value ≥ w
/// works, 64 keeps rows cache-line aligned and matches the scalar).
#[cfg(target_arch = "x86_64")]
const INTER_STRIDE: usize = 64;

#[cfg(target_arch = "x86_64")]
std::thread_local! {
    /// Reusable horizontal-pass intermediate. Zeroed once at thread
    /// init, then reused across every `convolve8_avx2` call — this
    /// eliminates the per-call 4608-byte memset that dominated the
    /// small-block (8×8/16×16) 2-D convolve cost (the kernel is the #1
    /// inter-MC per-pixel cost). Stale bytes in the unused columns
    /// `w..INTER_STRIDE` are HARMLESS: the horizontal pass writes
    /// columns `0..w` of every row it produces, the vertical pass's
    /// stored output columns `0..w` depend ONLY on those freshly-written
    /// columns, and the wider SIMD column loads that touch `w..` are
    /// computed-then-discarded. So output stays byte-identical to the
    /// zeroed-scratch form (validated by the scalar-differential tests).
    /// The buffer is always initialised (zeroed at construction) so no
    /// uninitialised byte is ever read.
    static CONV8_INTERMEDIATE: std::cell::RefCell<[u8; INTER_STRIDE * 72]> =
        const { std::cell::RefCell::new([0u8; INTER_STRIDE * 72]) };
}

/// AVX2 implementation of the full 2-D separable 8-tap convolution.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve8`] points at
/// this fn on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve8`) elsewhere.
///
/// `src` must satisfy the scalar `convolve8` footprint
/// (`(h - 1 + 7) * src_stride + w + 7` bytes, the 2-D filter window);
/// `dst` at least `(h - 1) * dst_stride + w`. `w` and `h` must be in
/// `1..=64`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)] // mirrors libvpx's argument list 1:1
pub fn convolve8_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    filter_y: &[i16; 8],
    w: usize,
    h: usize,
) {
    assert!(
        matches!(w, 4 | 8 | 16 | 32 | 64),
        "unsupported convolution width"
    );
    assert!((1..=64).contains(&h), "unsupported convolution height");

    let intermediate_height = h + 7;

    // Reuse the per-thread intermediate (no per-call memset — see the
    // CONV8_INTERMEDIATE docstring for the byte-exactness argument).
    CONV8_INTERMEDIATE.with_borrow_mut(|intermediate| {
        // Horizontal pass: produce `h + 7` rows. `src` is pre-shifted up
        // by three rows by the caller, so intermediate row 0 corresponds
        // to three rows above the destination origin.
        convolve8_horiz_avx2(
            src,
            src_stride,
            intermediate,
            INTER_STRIDE,
            filter_x,
            w,
            intermediate_height,
        );

        // Vertical pass: intermediate row 0 is already the first of the
        // eight taps for output row 0 (it sits three rows above the dst
        // origin), so it is handed over as-is — no extra shift.
        convolve8_vert_avx2(intermediate, INTER_STRIDE, dst, dst_stride, filter_y, w, h);
    });
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve8_avx2`.
    use super::convolve8_avx2;
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL, convolve8,
    };

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    fn lcg_next(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    /// Differential vs scalar across the VP9 (w, h) matrix with random
    /// (filter_x, filter_y) sub-pel row pairs drawn from all four
    /// kernel tables. The full-pel row 0 is excluded for both axes
    /// (routed to `convolve_copy` / single-axis kernels upstream).
    #[test]
    fn differential_2d() {
        require_avx2!();
        let shapes: &[(usize, usize)] = &[
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 64),
        ];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0xC0FF_EE00;
        for &(w, h) in shapes {
            for _iter in 0..256 {
                // Pick random (kernel, row) for each axis, rows 1..16.
                let kx = &kernels[(lcg_next(&mut state) as usize) % kernels.len()];
                let ky = &kernels[(lcg_next(&mut state) as usize) % kernels.len()];
                let rx = 1 + (lcg_next(&mut state) as usize) % 15;
                let ry = 1 + (lcg_next(&mut state) as usize) % 15;
                let filter_x = &kx[rx];
                let filter_y = &ky[ry];

                let src_stride = w + 7 + (lcg_next(&mut state) as usize & 0x7);
                let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                let src_len = (h - 1 + 7) * src_stride + w + 7;
                let dst_len = (h - 1) * dst_stride + w;
                let mut src = vec![0u8; src_len];
                for s in src.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let mut dst_scalar = vec![0u8; dst_len];
                let mut dst_avx2 = vec![0u8; dst_len];
                convolve8(
                    &src,
                    src_stride,
                    &mut dst_scalar,
                    dst_stride,
                    filter_x,
                    filter_y,
                    w,
                    h,
                );
                convolve8_avx2(
                    &src,
                    src_stride,
                    &mut dst_avx2,
                    dst_stride,
                    filter_x,
                    filter_y,
                    w,
                    h,
                );
                assert_eq!(
                    dst_scalar, dst_avx2,
                    "mismatch w={w} h={h} fx={filter_x:?} fy={filter_y:?}"
                );
            }
        }
    }

    /// Round-trip through SimdDispatch with forced-AVX2 / forced-scalar.
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);

        let fx = &EIGHTTAP_KERNEL[6];
        let fy = &EIGHTTAP_SHARP_KERNEL[10];
        let mut state: u32 = 0x1357_9BDF;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let stride = w + 7;
            let src_len = (h - 1 + 7) * stride + w + 7;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = vec![0u8; h * w];
            let mut dst_scalar = vec![0u8; h * w];
            (d_avx2.convolve8())(&src, stride, &mut dst_avx2, w, fx, fy, w, h);
            (d_scalar.convolve8())(&src, stride, &mut dst_scalar, w, fx, fy, w, h);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch w={w} h={h}"
            );
        }
    }
}
