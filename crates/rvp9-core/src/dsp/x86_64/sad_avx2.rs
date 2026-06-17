//! Sum-of-absolute-differences (SAD) — x86_64 AVX2 implementation (P1-B).
//!
//! The full-pel diamond search (`crate::mv_search::full_pixel`) and the
//! integer-projection motion estimator (`crate::mv_search::int_pro_me`)
//! compute SAD as their inner-loop distortion metric. The scalar
//! reference is the `(w, h)`-generic `crate::dsp::distortion::sad`
//! (and its 13 per-size `vpx_sad{W}x{H}` shims).
//!
//! ## Byte-exactness with scalar (by construction)
//!
//! SAD is the integer reduction `Σ|a − b|` over `w × h` byte pairs. The
//! maximum value is `64 × 64 × 255 = 1 044 480 < 2³²`, so it never
//! overflows a `u32` accumulator. The AVX2 horizontal sum of
//! `_mm256_sad_epu8` / `_mm_sad_epu8` byte-absolute-difference reductions
//! therefore equals the scalar `Σ|a − b|` **exactly** for any input —
//! addition is associative over the integers and no intermediate is lost.
//! The `differential_*` tests in this module pin this per size.
//!
//! ## Size dispatch
//!
//! VP9 motion-search blocks span `w ∈ {4, 8, 16, 32, 64}` with heights in
//! the same set. SAD is width- and height-generic, so a single
//! `(w, h)`-generic inner kernel dispatches on `w`:
//!
//! | `w` | per-row reduction                                     |
//! |-----|-------------------------------------------------------|
//! |  4  | `read_unaligned::<u32>` into xmm + `_mm_sad_epu8`      |
//! |  8  | `_mm_loadl_epi64` (64-bit) + `_mm_sad_epu8`           |
//! | 16  | `_mm_loadu_si128` (128-bit) + `_mm_sad_epu8`         |
//! | 32  | `_mm256_loadu_si256` (256-bit) + `_mm256_sad_epu8`   |
//! | 64  | two `_mm256_loadu_si256` per row + `_mm256_sad_epu8` |
//!
//! Mirrors libvpx `vpx_dsp/x86/sad_avx2.c` (`sad32xh_avx2` / `sad64xh_avx2`,
//! 256-bit) and the `_mm_sad_epu8` shape of `vpx_dsp/x86/sad_sse2.asm`
//! (`SAD4`/`SAD8`/`SAD16`, 128-bit) for the ≤16-wide sizes. Any out-of-set
//! width falls back to the scalar reference — the `debug_assert!` catches
//! it in test builds; release builds remain correct via the fallback.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Source / reference strides into the frame and prediction rasters are
//! arbitrary (frame geometry, border-padding, sub-block offsets), so every
//! SIMD load is the `loadu_*` unaligned form. See
//! `crates/rvp9-core/src/dsp/x86_64/mod.rs:14-26` for the workspace-wide
//! sentinel-comment convention.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_sad_epu8, _mm_setzero_si128, _mm_unpackhi_epi64, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_sad_epu8, _mm256_setzero_si256,
    _mm256_srli_si256,
};

/// Horizontally sum the four 64-bit SAD partials packed in a 256-bit
/// accumulator down to a single `u32`. Mirrors the reduction tail of
/// libvpx `sad32xh_avx2` / `sad64xh_avx2`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(sum: __m256i) -> u32 {
    // The four 64-bit lanes (only the low 16 bits of each are populated by
    // `_mm256_sad_epu8`) are folded: srli by 8 bytes brings lane 1→0 and
    // 3→2, then the 128-bit halves are added.
    let sum = _mm256_add_epi32(sum, _mm256_srli_si256(sum, 8));
    let hi = _mm256_extracti128_si256(sum, 1);
    let s128 = _mm_add_epi32(_mm256_castsi256_si128(sum), hi);
    _mm_cvtsi128_si32(s128) as u32
}

/// Horizontally sum the two 64-bit SAD partials packed in a 128-bit
/// accumulator (from `_mm_sad_epu8`) down to a single `u32`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2 (the dispatch slot
/// only installs this on AVX2 hosts; SSE2 alone would also suffice for the
/// 128-bit ops).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum128(sum: __m128i) -> u32 {
    // `_mm_sad_epu8` deposits the SAD of the low 8 bytes in lane 0 and the
    // SAD of the high 8 bytes in lane 1 (64-bit lanes). Fold lane 1 → 0.
    let s = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
    _mm_cvtsi128_si32(s) as u32
}

/// `(w, h)`-generic AVX2 SAD core.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. `src` must hold at
/// least `(h - 1) * src_stride + w` bytes; `reference` at least
/// `(h - 1) * ref_stride + w`. `w` must be one of `{4, 8, 16, 32, 64}` for
/// the SIMD fast path; any other width is delegated to the scalar
/// reference.
// G4: loads from `src` / `ref` use _mm256_loadu_si256 / _mm_loadu_si128 / _mm_loadl_epi64 / read_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_avx2_inner(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    debug_assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    debug_assert!(src.len() >= h.saturating_sub(1) * src_stride + w);
    debug_assert!(reference.len() >= h.saturating_sub(1) * ref_stride + w);

    let src_ptr = src.as_ptr();
    let ref_ptr = reference.as_ptr();

    match w {
        4 => {
            let mut acc = _mm_setzero_si128();
            for r in 0..h {
                let s = core::ptr::read_unaligned(src_ptr.add(r * src_stride) as *const u32);
                let rf = core::ptr::read_unaligned(ref_ptr.add(r * ref_stride) as *const u32);
                // Place the 4 bytes in the low lane of an xmm; the upper 12
                // bytes are zero on both sides so they contribute 0 to SAD.
                let sv = core::mem::transmute::<[u32; 4], __m128i>([s, 0, 0, 0]);
                let rv = core::mem::transmute::<[u32; 4], __m128i>([rf, 0, 0, 0]);
                acc = _mm_add_epi32(acc, _mm_sad_epu8(sv, rv));
            }
            hsum128(acc)
        }
        8 => {
            let mut acc = _mm_setzero_si128();
            for r in 0..h {
                let sv = _mm_loadl_epi64(src_ptr.add(r * src_stride) as *const __m128i);
                let rv = _mm_loadl_epi64(ref_ptr.add(r * ref_stride) as *const __m128i);
                acc = _mm_add_epi32(acc, _mm_sad_epu8(sv, rv));
            }
            hsum128(acc)
        }
        16 => {
            let mut acc = _mm_setzero_si128();
            for r in 0..h {
                let sv = _mm_loadu_si128(src_ptr.add(r * src_stride) as *const __m128i);
                let rv = _mm_loadu_si128(ref_ptr.add(r * ref_stride) as *const __m128i);
                acc = _mm_add_epi32(acc, _mm_sad_epu8(sv, rv));
            }
            hsum128(acc)
        }
        32 => {
            let mut acc = _mm256_setzero_si256();
            for r in 0..h {
                let sv = _mm256_loadu_si256(src_ptr.add(r * src_stride) as *const __m256i);
                let rv = _mm256_loadu_si256(ref_ptr.add(r * ref_stride) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_sad_epu8(sv, rv));
            }
            hsum256(acc)
        }
        64 => {
            let mut acc = _mm256_setzero_si256();
            for r in 0..h {
                let s_row = src_ptr.add(r * src_stride);
                let r_row = ref_ptr.add(r * ref_stride);
                let sv0 = _mm256_loadu_si256(s_row as *const __m256i);
                let sv1 = _mm256_loadu_si256(s_row.add(32) as *const __m256i);
                let rv0 = _mm256_loadu_si256(r_row as *const __m256i);
                let rv1 = _mm256_loadu_si256(r_row.add(32) as *const __m256i);
                let d = _mm256_add_epi32(_mm256_sad_epu8(sv0, rv0), _mm256_sad_epu8(sv1, rv1));
                acc = _mm256_add_epi32(acc, d);
            }
            hsum256(acc)
        }
        _ => {
            // Unexpected width — fall back to scalar for correctness.
            crate::dsp::distortion::sad(src, src_stride, reference, ref_stride, w, h)
        }
    }
}

/// Generate the 13 per-size AVX2 SAD safe wrappers. Each mirrors the
/// scalar `vpx_sad{W}x{H}` shim signature exactly so it can be installed
/// directly into the `SimdDispatch` per-size SAD table.
#[cfg(target_arch = "x86_64")]
macro_rules! sad_avx2_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("AVX2 SAD for a ", stringify!($w), "×", stringify!($h),
                                        " block. Byte-exact with `crate::dsp::distortion::vpx_sad",
                                        stringify!($w), "x", stringify!($h), "`.")]
        ///
        /// The dispatch slot points here on AVX2 hosts and at the scalar
        /// reference elsewhere.
        pub fn $name(src: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize) -> u32 {
            // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
            // when `caps.avx2 == true`. Caps are detected once at OnceLock
            // init time (see `crate::dsp::dispatch::init`) and are monotonic
            // for the process lifetime. Test paths gate each call on a
            // runtime `is_x86_feature_detected!("avx2")` check.
            unsafe { sad_avx2_inner(src, src_stride, reference, ref_stride, $w, $h) }
        }
    };
}

#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad4x4_avx2, 4, 4);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad4x8_avx2, 4, 8);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad8x4_avx2, 8, 4);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad8x8_avx2, 8, 8);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad8x16_avx2, 8, 16);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad16x8_avx2, 16, 8);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad16x16_avx2, 16, 16);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad16x32_avx2, 16, 32);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad32x16_avx2, 32, 16);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad32x32_avx2, 32, 32);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad32x64_avx2, 32, 64);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad64x32_avx2, 64, 32);
#[cfg(target_arch = "x86_64")]
sad_avx2_fn!(vpx_sad64x64_avx2, 64, 64);

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::*;
    use crate::dsp::distortion::sad as scalar_sad;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG: `state = state * 1103515245 + 12345`. Pulls 8
    /// bits per step from the high half of the 32-bit state. Mirrors the
    /// convolve tests' RNG so the differential suites share one pattern.
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

    /// All 13 VP9 motion-search shapes `(w, h)` and the matching AVX2 fn.
    type SadFn = fn(&[u8], usize, &[u8], usize) -> u32;
    const SHAPES: &[(usize, usize, SadFn)] = &[
        (4, 4, vpx_sad4x4_avx2),
        (4, 8, vpx_sad4x8_avx2),
        (8, 4, vpx_sad8x4_avx2),
        (8, 8, vpx_sad8x8_avx2),
        (8, 16, vpx_sad8x16_avx2),
        (16, 8, vpx_sad16x8_avx2),
        (16, 16, vpx_sad16x16_avx2),
        (16, 32, vpx_sad16x32_avx2),
        (32, 16, vpx_sad32x16_avx2),
        (32, 32, vpx_sad32x32_avx2),
        (32, 64, vpx_sad32x64_avx2),
        (64, 32, vpx_sad64x32_avx2),
        (64, 64, vpx_sad64x64_avx2),
    ];

    /// Differential AVX2 vs scalar over all 13 shapes, 512 random iters
    /// each, with non-tight strides (a 0..16 jitter on top of `w`) — the
    /// realistic call pattern from the diamond search, where `src` and
    /// `ref` index into frame / prediction rasters.
    #[test]
    fn differential_random_all_sizes() {
        require_avx2!();
        let mut rng = Lcg32::new(0x5AD0_F00D);
        for &(w, h, avx2) in SHAPES {
            for _ in 0..512 {
                let src_stride = w + rng.next_usize(16);
                let ref_stride = w + rng.next_usize(16);
                let src_len = (h - 1) * src_stride + w;
                let ref_len = (h - 1) * ref_stride + w;
                let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
                let rf: Vec<u8> = (0..ref_len).map(|_| rng.next_byte()).collect();
                let s = scalar_sad(&src, src_stride, &rf, ref_stride, w, h);
                let a = avx2(&src, src_stride, &rf, ref_stride);
                assert_eq!(s, a, "AVX2/scalar SAD mismatch at w={w}, h={h}");
            }
        }
    }

    /// Edge content: all-zero, all-max (255), and the max-diff corner
    /// (src all-0 vs ref all-255 → SAD = w*h*255, the u32-accumulator
    /// overflow corner). Asserts exact equality for every size.
    #[test]
    fn differential_edge_content() {
        require_avx2!();
        for &(w, h, avx2) in SHAPES {
            // Tight and loose strides both exercised.
            for &(ss, rs) in &[(w, w), (w + 7, w + 3)] {
                let src_len = (h - 1) * ss + w;
                let ref_len = (h - 1) * rs + w;
                let cases: &[(u8, u8)] = &[(0, 0), (255, 255), (0, 255), (255, 0), (128, 17)];
                for &(sval, rval) in cases {
                    let src = vec![sval; src_len];
                    let rf = vec![rval; ref_len];
                    let s = scalar_sad(&src, ss, &rf, rs, w, h);
                    let a = avx2(&src, ss, &rf, rs);
                    assert_eq!(
                        s, a,
                        "AVX2/scalar SAD mismatch at w={w}, h={h}, src={sval}, ref={rval}"
                    );
                }
                // Max-diff corner must equal exactly w*h*255.
                let src = vec![0u8; src_len];
                let rf = vec![255u8; ref_len];
                assert_eq!(
                    avx2(&src, ss, &rf, rs),
                    (w * h * 255) as u32,
                    "AVX2 SAD max-diff corner wrong at w={w}, h={h}"
                );
            }
        }
    }

    /// Forced-caps dispatch round-trip: the per-size SAD slot resolved from
    /// `build(avx2:true)` must equal the slot from `build(avx2:false)` for
    /// every size (mirrors `convolve_copy_avx2::dispatch_round_trip`).
    #[test]
    fn dispatch_round_trip() {
        require_avx2!();
        use crate::block::BlockSize;
        use crate::dsp::SimdDispatch;
        use crate::dsp::caps::detect;

        let mut caps_avx2 = detect();
        caps_avx2.avx2 = true;
        let d_avx2 = SimdDispatch::build(caps_avx2);
        let mut caps_scalar = detect();
        caps_scalar.avx2 = false;
        let d_scalar = SimdDispatch::build(caps_scalar);

        const SIZES: &[BlockSize] = &[
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ];
        let mut rng = Lcg32::new(0xD15A_7C00);
        for &bs in SIZES {
            let w = usize::from(bs.width());
            let h = usize::from(bs.height());
            let ss = w + 5;
            let rs = w + 2;
            let src: Vec<u8> = (0..(h - 1) * ss + w).map(|_| rng.next_byte()).collect();
            let rf: Vec<u8> = (0..(h - 1) * rs + w).map(|_| rng.next_byte()).collect();
            let a = (d_avx2.sad(bs))(&src, ss, &rf, rs);
            let s = (d_scalar.sad(bs))(&src, ss, &rf, rs);
            assert_eq!(a, s, "dispatch round-trip SAD mismatch at {bs:?}");
        }
    }
}
