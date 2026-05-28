//! Integer-pel block copy — x86_64 AVX2 implementation (Wave J J1).
//!
//! Fires when an inter-predicted block has zero sub-pel offset in both
//! dimensions: the predictor reduces to copying `w × h` bytes from
//! `src` to `dst` row-by-row. Mirrors libvpx `vpx_convolve_copy_c`
//! (`vpx_convolve.c:206-223`). The scalar reference is
//! [`crate::inter::filter::convolve_copy`].
//!
//! ## Size dispatch
//!
//! VP9 inter blocks span `w ∈ {4, 8, 16, 32, 64}` with the same set of
//! heights. A single safe wrapper takes the runtime `(w, h)` pair and
//! dispatches on `w` to one of five hand-unrolled per-row loops:
//!
//! | `w` | load                                    | store                                    |
//! |-----|-----------------------------------------|------------------------------------------|
//! |  4  | `read_unaligned::<u32>` (scalar)        | `write_unaligned::<u32>` (scalar)        |
//! |  8  | `_mm_loadl_epi64` (64-bit)              | `_mm_storel_epi64` (64-bit)              |
//! | 16  | `_mm_loadu_si128` (128-bit)             | `_mm_storeu_si128` (128-bit)             |
//! | 32  | `_mm256_loadu_si256` (256-bit)          | `_mm256_storeu_si256` (256-bit)          |
//! | 64  | two `_mm256_loadu_si256` per row        | two `_mm256_storeu_si256` per row        |
//!
//! Any out-of-set width falls through to the scalar reference — the
//! `debug_assert!` catches it in test builds; release builds remain
//! correct via the scalar fallback.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Source strides into the reference frame are arbitrary (frame
//! geometry, border-padding, sub-block offsets). Reads from `src` are
//! never guaranteed 16- or 32-byte-aligned, so every SIMD load is the
//! `loadu_*` unaligned form. Stores into `dst` follow the same rule —
//! destination strides into the prediction raster are likewise
//! arbitrary.
//!
//! See `crates/rvp9-core/src/dsp/x86_64/mod.rs:14-26` for the
//! workspace-wide sentinel-comment convention.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    _mm_loadl_epi64, _mm_loadu_si128, _mm_storel_epi64, _mm_storeu_si128, _mm256_loadu_si256,
    _mm256_storeu_si256,
};

/// AVX2 implementation of integer-pel block copy.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve_copy_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1) * src_stride + w` bytes of valid
/// storage; `dst` must hold at least `(h - 1) * dst_stride + w` bytes.
/// `w` must be one of `{4, 8, 16, 32, 64}` for the SIMD fast path; any
/// other width is delegated to the scalar reference.
// G4: loads from `src` use _mm256_loadu_si256 / _mm_loadu_si128 / _mm_loadl_epi64 / read_unaligned (unaligned)
// G4: stores to `dst` use _mm256_storeu_si256 / _mm_storeu_si128 / _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve_copy_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    debug_assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    debug_assert!(src.len() >= h.saturating_sub(1) * src_stride + w);
    debug_assert!(dst.len() >= h.saturating_sub(1) * dst_stride + w);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    match w {
        4 => {
            for r in 0..h {
                let v = core::ptr::read_unaligned(src_ptr.add(r * src_stride) as *const u32);
                core::ptr::write_unaligned(dst_ptr.add(r * dst_stride) as *mut u32, v);
            }
        }
        8 => {
            for r in 0..h {
                let v = _mm_loadl_epi64(src_ptr.add(r * src_stride) as *const _);
                _mm_storel_epi64(dst_ptr.add(r * dst_stride) as *mut _, v);
            }
        }
        16 => {
            for r in 0..h {
                let v = _mm_loadu_si128(src_ptr.add(r * src_stride) as *const _);
                _mm_storeu_si128(dst_ptr.add(r * dst_stride) as *mut _, v);
            }
        }
        32 => {
            for r in 0..h {
                let v = _mm256_loadu_si256(src_ptr.add(r * src_stride) as *const _);
                _mm256_storeu_si256(dst_ptr.add(r * dst_stride) as *mut _, v);
            }
        }
        64 => {
            for r in 0..h {
                let src_row = src_ptr.add(r * src_stride);
                let dst_row = dst_ptr.add(r * dst_stride);
                let lo = _mm256_loadu_si256(src_row as *const _);
                let hi = _mm256_loadu_si256(src_row.add(32) as *const _);
                _mm256_storeu_si256(dst_row as *mut _, lo);
                _mm256_storeu_si256(dst_row.add(32) as *mut _, hi);
            }
        }
        _ => {
            // Unexpected width — fall back to scalar for correctness.
            // `debug_assert!` above catches this in test builds.
            crate::inter::filter::convolve_copy(src, src_stride, dst, dst_stride, w, h);
        }
    }
}

/// Safe wrapper for the AVX2 integer-pel block copy.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve_copy`] points
/// at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve_copy`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve_copy_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // time (see `crate::dsp::dispatch::init`) and are monotonic for the
    // process lifetime. Test paths in this module gate each call on a
    // runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { convolve_copy_avx2_inner(src, src_stride, dst, dst_stride, w, h) }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::convolve_copy_avx2;
    use crate::inter::filter::convolve_copy as scalar_convolve_copy;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG: `state = state * 1103515245 + 12345`. Pulls
    /// 8 bits per step from the high half of the 32-bit state. Used
    /// instead of an external RNG dependency.
    struct Lcg32 {
        state: u32,
    }

    impl Lcg32 {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }
        fn next_byte(&mut self) -> u8 {
            self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (self.state >> 16) as u8
        }
        fn next_usize(&mut self, modulus: usize) -> usize {
            self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((self.state >> 8) as usize) % modulus
        }
    }

    /// All VP9 inter shapes (w, h) ∈ {4,8,16,32,64}² that fire in the
    /// integer-pel copy path. Hand-picked subset chosen to keep the
    /// 1024-iter sweep brisk while exercising every width branch and
    /// both square and rectangular heights.
    const SHAPES: &[(usize, usize)] = &[
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

    /// Differential AVX2 vs scalar across all shapes, 1024 iters each.
    /// `src_stride` and `dst_stride` are randomly chosen ≥ `w` (with a
    /// 0..8 jitter on top of `w`) to exercise non-tight strides — this
    /// is the realistic call pattern from `inter_predictor`, where
    /// `src` indexes into a reference frame and `dst` indexes into a
    /// prediction raster.
    #[test]
    fn differential_1024_iters() {
        require_avx2!();
        let mut rng = Lcg32::new(0xC0FF_EE42);
        for &(w, h) in SHAPES {
            for _iter in 0..1024 {
                let src_stride = w + rng.next_usize(8);
                let dst_stride = w + rng.next_usize(8);
                let src_len = (h - 1) * src_stride + w;
                let dst_len = (h - 1) * dst_stride + w;
                let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
                let mut dst_scalar = vec![0u8; dst_len];
                let mut dst_avx2 = vec![0u8; dst_len];
                // Pre-fill the destination with non-zero noise so any
                // skipped row would diverge from the scalar reference.
                for b in dst_scalar.iter_mut() {
                    *b = rng.next_byte();
                }
                dst_avx2.copy_from_slice(&dst_scalar);

                scalar_convolve_copy(&src, src_stride, &mut dst_scalar, dst_stride, w, h);
                convolve_copy_avx2(&src, src_stride, &mut dst_avx2, dst_stride, w, h);

                assert_eq!(dst_scalar, dst_avx2, "AVX2/scalar mismatch at w={w}, h={h}");
            }
        }
    }

    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(
            is_x86_feature_detected!("avx2"),
            "tautology guard: AVX2 detection must report true on this host"
        );
        // Sanity: the kernel actually copies bytes (not zero-fills).
        // Use a 64-wide block to exercise the dual-256 store path.
        let w = 64usize;
        let h = 4usize;
        let src_stride = 80;
        let dst_stride = 96;
        let src: Vec<u8> = (0..(h - 1) * src_stride + w)
            .map(|i| (i as u32 * 31 + 7) as u8)
            .collect();
        let mut dst = vec![0xAAu8; (h - 1) * dst_stride + w];
        convolve_copy_avx2(&src, src_stride, &mut dst, dst_stride, w, h);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    dst[y * dst_stride + x],
                    src[y * src_stride + x],
                    "tautology guard: byte mismatch at (y={y}, x={x})"
                );
            }
        }
    }

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

        let mut rng = Lcg32::new(0xDEAD_BEEF);
        for &(w, h) in SHAPES {
            let src_stride = w + 3;
            let dst_stride = w + 5;
            let src_len = (h - 1) * src_stride + w;
            let dst_len = (h - 1) * dst_stride + w;
            let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
            let mut dst_a = vec![0u8; dst_len];
            let mut dst_s = vec![0u8; dst_len];
            (dispatch_avx2.convolve_copy())(&src, src_stride, &mut dst_a, dst_stride, w, h);
            (dispatch_scalar.convolve_copy())(&src, src_stride, &mut dst_s, dst_stride, w, h);
            assert_eq!(dst_a, dst_s, "dispatch round-trip mismatch at w={w}, h={h}");
        }
    }
}
