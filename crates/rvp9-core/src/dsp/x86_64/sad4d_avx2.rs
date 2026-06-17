//! 4-reference Sum-of-absolute-differences (SAD×4) — x86_64 AVX2 (P1-B / PR3).
//!
//! The full-pel diamond search's `all_in` fast path
//! (`crate::mv_search::full_pixel::diamond_search_sad`) evaluates **exactly
//! four axis-aligned candidate sites per step** — one source block against
//! four reference offsets that share a single reference stride. This is
//! libvpx's `vpx_sad{W}x{H}x4d` contract (`vpx_dsp/x86/sad4d_avx2.c`): load
//! each source row once and accumulate four running SADs in parallel,
//! amortizing the source-block load across all four candidates.
//!
//! ## Byte-exactness with `4 × scalar sad()` (by construction)
//!
//! Each of the four outputs is the integer reduction `Σ|src − ref_k|` over
//! `w × h` byte pairs — exactly the value the scalar
//! `crate::dsp::distortion::sad` returns for that `(src, ref_k)` pair. The
//! per-candidate accumulator never exceeds `64 × 64 × 255 = 1 044 480 < 2³²`,
//! so no `u32` lane overflows. Addition is associative over the integers and
//! `_mm{,256}_sad_epu8` computes the same byte-absolute-difference sum
//! regardless of lane grouping, so each AVX2 lane equals its scalar
//! counterpart **exactly** for any input. The `differential_*` tests in this
//! module pin `out[k] == scalar_sad(src, ref_k)` per size.
//!
//! ## Size dispatch
//!
//! The per-row reduction matches the single-SAD kernel
//! (`crate::dsp::x86_64::sad_avx2`) exactly, applied four times (once per
//! reference) with a shared source load:
//!
//! | `w` | per-row source load + four `sad_epu8`                  |
//! |-----|--------------------------------------------------------|
//! |  4  | `read_unaligned::<u32>` into xmm + `_mm_sad_epu8`       |
//! |  8  | `_mm_loadl_epi64` (64-bit) + `_mm_sad_epu8`            |
//! | 16  | `_mm_loadu_si128` (128-bit) + `_mm_sad_epu8`          |
//! | 32  | `_mm256_loadu_si256` (256-bit) + `_mm256_sad_epu8`    |
//! | 64  | two `_mm256_loadu_si256` per row + `_mm256_sad_epu8`  |
//!
//! Mirrors libvpx `sad32xhx4d_avx2` / `sad64xhx4d_avx2` (256-bit) and the
//! `_mm_sad_epu8` shape of `vpx_sad{4,8,16}x{H}x4d_sse2` for the ≤16-wide
//! sizes. Any out-of-set width falls back to four scalar `sad()` calls — the
//! `debug_assert!` catches it in test builds; release builds remain correct.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Source / reference strides into the frame and prediction rasters are
//! arbitrary, so every SIMD load is the `loadu_*` unaligned form (libvpx's
//! `sad4d_avx2.c` uses an *aligned* `_mm256_load_si256` on the source, which
//! requires a 32-byte-aligned `src`; we use `loadu` on both sides because the
//! diamond search hands us arbitrarily-aligned sub-block slices). See
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
/// accumulator down to a single `u32`. Identical in form to
/// `crate::dsp::x86_64::sad_avx2`'s `hsum256` (kept module-local so the
/// 4-ref kernel is self-contained).
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(sum: __m256i) -> u32 {
    let sum = _mm256_add_epi32(sum, _mm256_srli_si256(sum, 8));
    let hi = _mm256_extracti128_si256(sum, 1);
    let s128 = _mm_add_epi32(_mm256_castsi256_si128(sum), hi);
    _mm_cvtsi128_si32(s128) as u32
}

/// Horizontally sum the two 64-bit SAD partials packed in a 128-bit
/// accumulator (from `_mm_sad_epu8`) down to a single `u32`. Identical in
/// form to `crate::dsp::x86_64::sad_avx2`'s `hsum128`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum128(sum: __m128i) -> u32 {
    let s = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
    _mm_cvtsi128_si32(s) as u32
}

/// `(w, h)`-generic AVX2 4-reference SAD core. Computes `Σ|src − ref_k|`
/// for `k ∈ 0..4` and writes the four results into `out`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. `src` must hold at
/// least `(h - 1) * src_stride + w` bytes; each `refs[k]` at least
/// `(h - 1) * ref_stride + w`. `w` must be one of `{4, 8, 16, 32, 64}` for
/// the SIMD fast path; any other width is delegated to the scalar reference.
// G4: loads from `src` / `refs[k]` use _mm256_loadu_si256 / _mm_loadu_si128 / _mm_loadl_epi64 / read_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad4d_avx2_inner(
    src: &[u8],
    src_stride: usize,
    refs: &[&[u8]; 4],
    ref_stride: usize,
    w: usize,
    h: usize,
    out: &mut [u32; 4],
) {
    debug_assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    debug_assert!(src.len() >= h.saturating_sub(1) * src_stride + w);

    let src_ptr = src.as_ptr();
    let rp = [
        refs[0].as_ptr(),
        refs[1].as_ptr(),
        refs[2].as_ptr(),
        refs[3].as_ptr(),
    ];

    match w {
        4 => {
            let mut acc = [_mm_setzero_si128(); 4];
            for r in 0..h {
                let s = core::ptr::read_unaligned(src_ptr.add(r * src_stride) as *const u32);
                // Upper 12 bytes are zero on both sides → contribute 0 to SAD.
                let sv = core::mem::transmute::<[u32; 4], __m128i>([s, 0, 0, 0]);
                for k in 0..4 {
                    let rf = core::ptr::read_unaligned(rp[k].add(r * ref_stride) as *const u32);
                    let rv = core::mem::transmute::<[u32; 4], __m128i>([rf, 0, 0, 0]);
                    acc[k] = _mm_add_epi32(acc[k], _mm_sad_epu8(sv, rv));
                }
            }
            for k in 0..4 {
                out[k] = hsum128(acc[k]);
            }
        }
        8 => {
            let mut acc = [_mm_setzero_si128(); 4];
            for r in 0..h {
                let sv = _mm_loadl_epi64(src_ptr.add(r * src_stride) as *const __m128i);
                for k in 0..4 {
                    let rv = _mm_loadl_epi64(rp[k].add(r * ref_stride) as *const __m128i);
                    acc[k] = _mm_add_epi32(acc[k], _mm_sad_epu8(sv, rv));
                }
            }
            for k in 0..4 {
                out[k] = hsum128(acc[k]);
            }
        }
        16 => {
            let mut acc = [_mm_setzero_si128(); 4];
            for r in 0..h {
                let sv = _mm_loadu_si128(src_ptr.add(r * src_stride) as *const __m128i);
                for k in 0..4 {
                    let rv = _mm_loadu_si128(rp[k].add(r * ref_stride) as *const __m128i);
                    acc[k] = _mm_add_epi32(acc[k], _mm_sad_epu8(sv, rv));
                }
            }
            for k in 0..4 {
                out[k] = hsum128(acc[k]);
            }
        }
        32 => {
            let mut acc = [_mm256_setzero_si256(); 4];
            for r in 0..h {
                let sv = _mm256_loadu_si256(src_ptr.add(r * src_stride) as *const __m256i);
                for k in 0..4 {
                    let rv = _mm256_loadu_si256(rp[k].add(r * ref_stride) as *const __m256i);
                    acc[k] = _mm256_add_epi32(acc[k], _mm256_sad_epu8(sv, rv));
                }
            }
            for k in 0..4 {
                out[k] = hsum256(acc[k]);
            }
        }
        64 => {
            let mut acc = [_mm256_setzero_si256(); 4];
            for r in 0..h {
                let s_row = src_ptr.add(r * src_stride);
                let sv0 = _mm256_loadu_si256(s_row as *const __m256i);
                let sv1 = _mm256_loadu_si256(s_row.add(32) as *const __m256i);
                for k in 0..4 {
                    let r_row = rp[k].add(r * ref_stride);
                    let rv0 = _mm256_loadu_si256(r_row as *const __m256i);
                    let rv1 = _mm256_loadu_si256(r_row.add(32) as *const __m256i);
                    let d = _mm256_add_epi32(_mm256_sad_epu8(sv0, rv0), _mm256_sad_epu8(sv1, rv1));
                    acc[k] = _mm256_add_epi32(acc[k], d);
                }
            }
            for k in 0..4 {
                out[k] = hsum256(acc[k]);
            }
        }
        _ => {
            // Unexpected width — fall back to four scalar SADs for correctness.
            for k in 0..4 {
                out[k] = crate::dsp::distortion::sad(src, src_stride, refs[k], ref_stride, w, h);
            }
        }
    }
}

/// Generate the 13 per-size AVX2 4-reference SAD safe wrappers. Each writes
/// `out[k] = Σ|src − refs[k]|` (byte-exact with `crate::dsp::distortion::sad`
/// called per reference), suitable for installation into the `SimdDispatch`
/// per-size `sad4d` table.
#[cfg(target_arch = "x86_64")]
macro_rules! sad4d_avx2_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("AVX2 4-reference SAD for a ", stringify!($w), "×",
                                        stringify!($h), " block. Each lane is byte-exact with ",
                                        "`crate::dsp::distortion::sad` for that reference offset.")]
        ///
        /// The `sad4d` dispatch slot points here on AVX2 hosts and at the
        /// scalar 4×-`sad` reference elsewhere.
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            refs: &[&[u8]; 4],
            ref_stride: usize,
            out: &mut [u32; 4],
        ) {
            // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
            // when `caps.avx2 == true`. Caps are detected once at OnceLock
            // init time and are monotonic for the process lifetime. Test
            // paths gate each call on a runtime `is_x86_feature_detected!`
            // check.
            unsafe { sad4d_avx2_inner(src, src_stride, refs, ref_stride, $w, $h, out) }
        }
    };
}

#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad4x4x4d_avx2, 4, 4);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad4x8x4d_avx2, 4, 8);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad8x4x4d_avx2, 8, 4);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad8x8x4d_avx2, 8, 8);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad8x16x4d_avx2, 8, 16);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad16x8x4d_avx2, 16, 8);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad16x16x4d_avx2, 16, 16);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad16x32x4d_avx2, 16, 32);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad32x16x4d_avx2, 32, 16);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad32x32x4d_avx2, 32, 32);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad32x64x4d_avx2, 32, 64);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad64x32x4d_avx2, 64, 32);
#[cfg(target_arch = "x86_64")]
sad4d_avx2_fn!(vpx_sad64x64x4d_avx2, 64, 64);

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

    /// Deterministic LCG mirroring the single-SAD differential suite's RNG.
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

    type Sad4dFn = fn(&[u8], usize, &[&[u8]; 4], usize, &mut [u32; 4]);
    const SHAPES: &[(usize, usize, Sad4dFn)] = &[
        (4, 4, vpx_sad4x4x4d_avx2),
        (4, 8, vpx_sad4x8x4d_avx2),
        (8, 4, vpx_sad8x4x4d_avx2),
        (8, 8, vpx_sad8x8x4d_avx2),
        (8, 16, vpx_sad8x16x4d_avx2),
        (16, 8, vpx_sad16x8x4d_avx2),
        (16, 16, vpx_sad16x16x4d_avx2),
        (16, 32, vpx_sad16x32x4d_avx2),
        (32, 16, vpx_sad32x16x4d_avx2),
        (32, 32, vpx_sad32x32x4d_avx2),
        (32, 64, vpx_sad32x64x4d_avx2),
        (64, 32, vpx_sad64x32x4d_avx2),
        (64, 64, vpx_sad64x64x4d_avx2),
    ];

    /// Differential: each AVX2 `sad4d` lane equals the independent scalar
    /// `sad()` for that reference, over all 13 shapes, 512 random iters each,
    /// with non-tight strides and four *distinct* reference buffers (the
    /// realistic diamond-search pattern: one src, four ref offsets sharing a
    /// stride).
    #[test]
    fn differential_random_all_sizes() {
        require_avx2!();
        let mut rng = Lcg32::new(0x4D04_5AD4);
        for &(w, h, avx2) in SHAPES {
            for _ in 0..512 {
                let src_stride = w + rng.next_usize(16);
                let ref_stride = w + rng.next_usize(16);
                let src_len = (h - 1) * src_stride + w;
                let ref_len = (h - 1) * ref_stride + w;
                let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
                let refs_owned: Vec<Vec<u8>> = (0..4)
                    .map(|_| (0..ref_len).map(|_| rng.next_byte()).collect())
                    .collect();
                let refs: [&[u8]; 4] = [
                    &refs_owned[0],
                    &refs_owned[1],
                    &refs_owned[2],
                    &refs_owned[3],
                ];

                let mut out = [0u32; 4];
                avx2(&src, src_stride, &refs, ref_stride, &mut out);

                for k in 0..4 {
                    let want = scalar_sad(&src, src_stride, refs[k], ref_stride, w, h);
                    assert_eq!(
                        out[k], want,
                        "AVX2 sad4d lane {k} != scalar sad at w={w}, h={h}"
                    );
                }
            }
        }
    }

    /// All four references aliasing windows of ONE buffer at distinct offsets
    /// sharing a single stride — the exact diamond `all_in` shape (one
    /// reference plane, four axis-aligned candidate origins). Asserts each
    /// lane matches the scalar SAD of that window.
    #[test]
    fn differential_shared_plane_offsets() {
        require_avx2!();
        let mut rng = Lcg32::new(0x0FF5_E70F);
        for &(w, h, avx2) in SHAPES {
            for _ in 0..256 {
                let stride = w + 8 + rng.next_usize(24);
                // A plane large enough to hold four offset windows.
                let plane_rows = h + 4;
                let plane: Vec<u8> = (0..plane_rows * stride).map(|_| rng.next_byte()).collect();
                let src: Vec<u8> = (0..(h - 1) * stride + w).map(|_| rng.next_byte()).collect();

                // Four axis-aligned origins (mimicking ss_mv offsets): the
                // four windows start at byte offsets 0, +1 row, +1 col, +2.
                let offs = [0usize, stride, 1, stride + 1];
                let refs: [&[u8]; 4] = [
                    &plane[offs[0]..],
                    &plane[offs[1]..],
                    &plane[offs[2]..],
                    &plane[offs[3]..],
                ];

                let mut out = [0u32; 4];
                avx2(&src, stride, &refs, stride, &mut out);
                for k in 0..4 {
                    let want = scalar_sad(&src, stride, refs[k], stride, w, h);
                    assert_eq!(
                        out[k], want,
                        "AVX2 sad4d shared-plane lane {k} != scalar at w={w}, h={h}"
                    );
                }
            }
        }
    }

    /// Edge content: all-zero, all-max, and the max-diff overflow corner per
    /// lane (src all-0 vs ref all-255 → lane SAD = w*h*255). Distinct content
    /// per reference exercises the parallel accumulators independently.
    #[test]
    fn differential_edge_content() {
        require_avx2!();
        for &(w, h, avx2) in SHAPES {
            for &(ss, rs) in &[(w, w), (w + 7, w + 3)] {
                let src_len = (h - 1) * ss + w;
                let ref_len = (h - 1) * rs + w;
                // Four references each filled with a distinct constant so the
                // four lanes carry different SADs (catches lane cross-talk).
                let rvals = [0u8, 255u8, 128u8, 17u8];
                let svals = [255u8, 0u8, 17u8, 128u8];
                for &sval in &svals {
                    let src = vec![sval; src_len];
                    let bufs: Vec<Vec<u8>> = rvals.iter().map(|&rv| vec![rv; ref_len]).collect();
                    let refs: [&[u8]; 4] = [&bufs[0], &bufs[1], &bufs[2], &bufs[3]];
                    let mut out = [0u32; 4];
                    avx2(&src, ss, &refs, rs, &mut out);
                    for k in 0..4 {
                        let want = scalar_sad(&src, ss, refs[k], rs, w, h);
                        assert_eq!(
                            out[k], want,
                            "AVX2 sad4d edge lane {k} != scalar at w={w}, h={h}, s={sval}, r={}",
                            rvals[k]
                        );
                    }
                }
                // Max-diff overflow corner on lane 0 (src 0 vs ref 255).
                let src = vec![0u8; src_len];
                let bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![255u8; ref_len]).collect();
                let refs: [&[u8]; 4] = [&bufs[0], &bufs[1], &bufs[2], &bufs[3]];
                let mut out = [0u32; 4];
                avx2(&src, ss, &refs, rs, &mut out);
                for k in 0..4 {
                    assert_eq!(
                        out[k],
                        (w * h * 255) as u32,
                        "AVX2 sad4d max-diff corner wrong lane {k} at w={w}, h={h}"
                    );
                }
            }
        }
    }

    /// Forced-caps dispatch round-trip: the per-size `sad4d` slot from
    /// `build(avx2:true)` must agree with `build(avx2:false)` for every size
    /// (mirrors `sad_avx2::dispatch_round_trip`).
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
        let mut rng = Lcg32::new(0xD15A_4D40);
        for &bs in SIZES {
            let w = usize::from(bs.width());
            let h = usize::from(bs.height());
            let ss = w + 5;
            let rs = w + 2;
            let src: Vec<u8> = (0..(h - 1) * ss + w).map(|_| rng.next_byte()).collect();
            let bufs: Vec<Vec<u8>> = (0..4)
                .map(|_| (0..(h - 1) * rs + w).map(|_| rng.next_byte()).collect())
                .collect();
            let refs: [&[u8]; 4] = [&bufs[0], &bufs[1], &bufs[2], &bufs[3]];
            let mut a = [0u32; 4];
            let mut s = [0u32; 4];
            (d_avx2.sad4d(bs))(&src, ss, &refs, rs, &mut a);
            (d_scalar.sad4d(bs))(&src, ss, &refs, rs, &mut s);
            assert_eq!(a, s, "dispatch round-trip sad4d mismatch at {bs:?}");
        }
    }
}
