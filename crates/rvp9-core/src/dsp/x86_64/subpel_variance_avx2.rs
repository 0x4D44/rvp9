//! Sub-pixel variance — x86_64 AVX2 implementation (PR4).
//!
//! The sub-pel MV refinement (`crate::mv_search::sub_pixel`) measures the
//! distortion of a candidate predictor sampled at a fractional `(xoffset,
//! yoffset)` position. The scalar reference is
//! `crate::dsp::distortion::sub_pixel_variance` (and its 13 per-size
//! `vpx_sub_pixel_variance{W}x{H}` shims).
//!
//! ## Two exactness surfaces (both pinned by the differential tests)
//!
//! Sub-pixel variance is a bilinear interpolation *followed by* a variance
//! reduction, and each is its own exactness surface:
//!
//! 1. **The interpolated `u8` temp must be byte-identical.** The AVX2
//!    bilinear first pass (horizontal, `u8 → u16`) and second pass
//!    (vertical, `u16 → u8`) must produce the same rounded buffer as the
//!    scalar `round_power_of_two(·, FILTER_BITS=7)` path. Both taps sum to
//!    `128`, so every intermediate `v = a·f0 + b·f1 ∈ [0, 32 640]` fits a
//!    signed `i16`; `(v + 64) >> 7` is computed in `i16` lanes, exactly
//!    matching the scalar `(v + 64) >> 7`. The `differential_temp_*` test
//!    pins this BEFORE the variance.
//! 2. **The variance over the temp must be byte-identical.** Once the temp
//!    is byte-identical, the variance over it is identical. We reuse the
//!    proven block `(sum, sse)` reduction
//!    (`crate::dsp::x86_64::variance_avx2::var_sum_sse_avx2`) and apply the
//!    SAME scalar finisher `crate::dsp::distortion::variance` uses
//!    (`var = (sse − sum²/N).max(0)`, i64 form) — so surface (b) is
//!    arithmetically identical to the scalar reference, not a re-derivation.
//!
//! Mirrors libvpx `vpx_dsp/x86/subpel_variance_avx2`-family kernels (bilinear
//! first/second pass then variance), specialised to the byte-widening form so
//! the temp and the reduction are both exact.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Source strides are arbitrary; every SIMD load is the `loadu_*` unaligned
//! form. See `crates/rvp9-core/src/dsp/x86_64/mod.rs:14-26`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi16, _mm256_cvtepu8_epi16, _mm256_loadu_si256,
    _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_srai_epi16, _mm256_storeu_si256,
};

/// Number of fractional bits in the bilinear filter taps (FILTER_BITS).
#[cfg(target_arch = "x86_64")]
const FILTER_BITS: i32 = 7;

#[cfg(target_arch = "x86_64")]
const MAX_SUBPEL_WIDTH: usize = 64;
#[cfg(target_arch = "x86_64")]
const MAX_SUBPEL_HEIGHT: usize = 64;

#[cfg(target_arch = "x86_64")]
struct SubPixelVarianceScratch {
    fdata: [i16; (MAX_SUBPEL_HEIGHT + 1) * MAX_SUBPEL_WIDTH],
    temp: [u8; MAX_SUBPEL_HEIGHT * MAX_SUBPEL_WIDTH],
}

#[cfg(target_arch = "x86_64")]
struct BilinearBuffers<'a> {
    fdata: &'a mut [i16],
    temp: &'a mut [u8],
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Reused by the AVX2 sub-pixel-variance wrappers on one search worker.
    /// Each call overwrites both complete active prefixes before reading them.
    static SUBPEL_VARIANCE_SCRATCH: std::cell::RefCell<SubPixelVarianceScratch> =
        const { std::cell::RefCell::new(SubPixelVarianceScratch {
            fdata: [0; (MAX_SUBPEL_HEIGHT + 1) * MAX_SUBPEL_WIDTH],
            temp: [0; MAX_SUBPEL_HEIGHT * MAX_SUBPEL_WIDTH],
        }) };
}

/// 2-tap bilinear filters — verbatim copy of
/// `crate::dsp::distortion::BILINEAR_FILTERS` (private there).
#[cfg(target_arch = "x86_64")]
const BILINEAR_FILTERS: [[u16; 2]; 8] = [
    [128, 0],
    [112, 16],
    [96, 32],
    [80, 48],
    [64, 64],
    [48, 80],
    [32, 96],
    [16, 112],
];

/// Bilinear-filter a contiguous run of `n` widened `i16` pairs:
/// `out[k] = ((a[k]·f0 + b[k]·f1) + 64) >> 7`, all in `i16` lanes. The inputs
/// are already widened from `u8`; the result is the rounded `i16` (which fits
/// `u8`, being a rounded convex combination of two bytes).
///
/// # Safety
///
/// Caller must guarantee AVX2 support and that `a`/`b` hold ≥ 16 valid `i16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bil_round(a: __m256i, b: __m256i, f0: __m256i, f1: __m256i, round: __m256i) -> __m256i {
    // a·f0 + b·f1 (each ≤ 255·128 = 32 640, sum ≤ 32 640, fits i16), + 64,
    // then arithmetic >> 7. mullo keeps the low 16 bits (exact here).
    let prod_a = _mm256_mullo_epi16(a, f0);
    let prod_b = _mm256_mullo_epi16(b, f1);
    let v = _mm256_add_epi16(prod_a, prod_b);
    let v = _mm256_add_epi16(v, round);
    _mm256_srai_epi16::<FILTER_BITS>(v)
}

/// AVX2 bilinear two-pass interpolation of `src` at `(xoffset, yoffset)` into
/// a byte-identical `h × w` `u8` temp buffer. This is exactly the scalar
/// `var_filter_block2d_bil_*` two-pass output BEFORE the variance.
///
/// Writes `temp[i*w + j]` for `i ∈ 0..h`, `j ∈ 0..w`.
///
/// # Safety
///
/// Caller must guarantee AVX2 support. `src` must cover `(h+1)` rows ×
/// `(w+1)` cols at `src_stride` (the bilinear footprint). `w` must be a
/// multiple of 16 OR ≤ 16 (the kernel processes the row in 16-wide chunks
/// and a single ≤16 partial chunk). `temp.len()` must be `≥ h * w`.
// G4: loads from `src` use _mm256_loadu_si256 (unaligned, via cvtepu8 from xmm)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bilinear_to_u8_temp(
    src: &[u8],
    src_stride: usize,
    xoffset: usize,
    yoffset: usize,
    w: usize,
    h: usize,
    buffers: BilinearBuffers<'_>,
) {
    let BilinearBuffers { fdata, temp } = buffers;
    debug_assert!(xoffset < 8 && yoffset < 8);
    debug_assert!(fdata.len() >= (h + 1) * w);
    debug_assert!(temp.len() >= h * w);

    let xf = BILINEAR_FILTERS[xoffset];
    let yf = BILINEAR_FILTERS[yoffset];
    let xf0 = _mm256_set1_epi16(xf[0] as i16);
    let xf1 = _mm256_set1_epi16(xf[1] as i16);
    let yf0 = _mm256_set1_epi16(yf[0] as i16);
    let yf1 = _mm256_set1_epi16(yf[1] as i16);
    let round = _mm256_set1_epi16(1 << (FILTER_BITS - 1));

    // First pass: horizontal bilinear → (h+1) × w i16 buffer (`fdata`).
    // Each output row holds w i16 values in [0, 255].
    for i in 0..(h + 1) {
        let row_base = i * src_stride;
        let mut j = 0usize;
        while j < w {
            let count = (w - j).min(16);
            // Bounded loads: copy the `count` left-neighbours and the `count`
            // right-neighbours (offset +1) into zero-padded 16-lane buffers.
            // This stays in-bounds for any `w`/stride (no 16-byte over-read at
            // a partial chunk or the last row).
            let a = _mm256_cvtepu8_epi16(load_bytes_bounded(src, row_base + j, count));
            let b = _mm256_cvtepu8_epi16(load_bytes_bounded(src, row_base + j + 1, count));
            let out = bil_round(a, b, xf0, xf1, round);
            store_i16_chunk(out, &mut fdata[i * w + j..], count);
            j += 16;
        }
    }

    // Second pass: vertical bilinear over fdata → h × w u8 temp.
    for i in 0..h {
        let top = i * w;
        let bot = (i + 1) * w;
        let mut j = 0usize;
        while j < w {
            let a = load_i16_chunk(&fdata[top + j..], w - j);
            let b = load_i16_chunk(&fdata[bot + j..], w - j);
            let out = bil_round(a, b, yf0, yf1, round);
            // out holds i16 in [0,255]; narrow to u8 and store.
            store_u8_chunk(out, &mut temp[i * w + j..], w - j);
            j += 16;
        }
    }
}

/// Copy `count` (≤ 16) bytes from `src[off..off+count]` into the low lanes of
/// a zero-padded xmm, so `cvtepu8_epi16` widens exactly those bytes (the
/// zero-padded high lanes give diff/products that the masked store discards).
/// Bounds-checked against the `src` slice, so it stays in-bounds for any
/// `w`/stride — no 16-byte over-read at a partial chunk or the last row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_bytes_bounded(src: &[u8], off: usize, count: usize) -> __m128i {
    debug_assert!(count <= 16);
    let mut buf = [0u8; 16];
    buf[..count].copy_from_slice(&src[off..off + count]);
    core::arch::x86_64::_mm_loadu_si128(buf.as_ptr() as *const __m128i)
}

/// Load the first `count` (≤ 16) `i16` from `s` into a 256-bit vector. For a
/// full chunk (`count >= 16`) this is a direct 16-wide load; for a partial tail
/// (`count < 16`) the `count` elements are copied into a zero-padded buffer
/// first, so the load stays in-bounds when `w` is not a multiple of 16. Lanes
/// past `count` read as zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_i16_chunk(s: &[i16], count: usize) -> __m256i {
    if count >= 16 {
        _mm256_loadu_si256(s.as_ptr() as *const __m256i)
    } else {
        let mut buf = [0i16; 16];
        buf[..count].copy_from_slice(&s[..count]);
        _mm256_loadu_si256(buf.as_ptr() as *const __m256i)
    }
}

/// Store the low `min(count, 16)` `i16` lanes of `v` into `dst`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn store_i16_chunk(v: __m256i, dst: &mut [i16], count: usize) {
    let n = count.min(16);
    if n == 16 {
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, v);
    } else {
        let mut buf = [0i16; 16];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, v);
        dst[..n].copy_from_slice(&buf[..n]);
    }
}

/// Narrow the low `min(count, 16)` `i16` lanes of `v` (each in `[0, 255]`) to
/// `u8` and store into `dst`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn store_u8_chunk(v: __m256i, dst: &mut [u8], count: usize) {
    let n = count.min(16);
    let mut buf = [0i16; 16];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, v);
    for k in 0..n {
        dst[k] = buf[k] as u8;
    }
}

/// AVX2 sub-pixel variance. Byte-exact with
/// `crate::dsp::distortion::sub_pixel_variance`:
///
/// 1. interpolate `src` at `(xoffset, yoffset)` into a byte-identical `u8`
///    temp via [`bilinear_to_u8_temp`], then
/// 2. compute variance against `reference` using the SAME `(sum, sse)`
///    reduction + scalar finisher the scalar reference uses.
///
/// Returns `(var, sse)`. `w` is one of the supported VP9 widths.
///
/// The dispatch slot points here on AVX2 hosts and at the scalar reference
/// elsewhere.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub fn sub_pixel_variance_avx2(
    src: &[u8],
    src_stride: usize,
    reference: &[u8],
    ref_stride: usize,
    xoffset: usize,
    yoffset: usize,
    w: usize,
    h: usize,
) -> (u32, u32) {
    debug_assert!(w <= MAX_SUBPEL_WIDTH && h <= MAX_SUBPEL_HEIGHT);
    SUBPEL_VARIANCE_SCRATCH.with_borrow_mut(|scratch| {
        let SubPixelVarianceScratch { fdata, temp } = scratch;
        let fdata = &mut fdata[..(h + 1) * w];
        let temp = &mut temp[..h * w];
        // SAFETY: installed only on AVX2 hosts by `SimdDispatch::build`
        // (caps.avx2). The fixed scratch covers every supported VP9 block.
        unsafe {
            bilinear_to_u8_temp(
                src,
                src_stride,
                xoffset,
                yoffset,
                w,
                h,
                BilinearBuffers { fdata, temp },
            );
        }
        // Surface (b): variance of the byte-identical temp vs `reference`. We
        // use the proven AVX2 (sum, sse) reduction and apply the EXACT scalar
        // finisher (i64 sub + max(0)).
        let (sum, sse) =
            super::variance_avx2::var_sum_sse_avx2(temp, w, reference, ref_stride, w, h);
        let n = (w * h) as i64;
        let sum_sq = sum * sum;
        let var = ((sse as i64) - sum_sq / n).max(0) as u32;
        (var, sse as u32)
    })
}

/// Generate the 11 per-size AVX2 sub-pixel-variance safe wrappers (the 5
/// square + 6 rectangular sizes the sub-pel search uses). Each mirrors the
/// scalar `vpx_sub_pixel_variance{W}x{H}` signature exactly.
#[cfg(target_arch = "x86_64")]
macro_rules! subpel_var_avx2_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("AVX2 sub-pixel variance for a ", stringify!($w), "×",
                                                stringify!($h), " block. Byte-exact with ",
                                                "`crate::dsp::distortion::vpx_sub_pixel_variance",
                                                stringify!($w), "x", stringify!($h), "`.")]
        pub fn $name(
            src: &[u8],
            src_stride: usize,
            xoffset: usize,
            yoffset: usize,
            reference: &[u8],
            ref_stride: usize,
        ) -> (u32, u32) {
            sub_pixel_variance_avx2(
                src, src_stride, reference, ref_stride, xoffset, yoffset, $w, $h,
            )
        }
    };
}

#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance4x4_avx2, 4, 4);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance8x8_avx2, 8, 8);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance16x16_avx2, 16, 16);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance32x32_avx2, 32, 32);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance64x64_avx2, 64, 64);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance16x8_avx2, 16, 8);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance8x16_avx2, 8, 16);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance32x16_avx2, 32, 16);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance16x32_avx2, 16, 32);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance64x32_avx2, 64, 32);
#[cfg(target_arch = "x86_64")]
subpel_var_avx2_fn!(vpx_sub_pixel_variance32x64_avx2, 32, 64);

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::*;
    use crate::dsp::distortion::sub_pixel_variance as scalar_svf;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

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

    /// Scalar bilinear temp reference — replicates the scalar two-pass
    /// `var_filter_block2d_bil_*` exactly to pin surface (a).
    fn scalar_bilinear_temp(
        src: &[u8],
        src_stride: usize,
        xoffset: usize,
        yoffset: usize,
        w: usize,
        h: usize,
    ) -> Vec<u8> {
        const BF: [[i32; 2]; 8] = [
            [128, 0],
            [112, 16],
            [96, 32],
            [80, 48],
            [64, 64],
            [48, 80],
            [32, 96],
            [16, 112],
        ];
        let rpo = |v: i32| (v + 64) >> 7;
        let xf = BF[xoffset];
        let yf = BF[yoffset];
        // First pass.
        let mut fdata = vec![0u16; (h + 1) * w];
        for i in 0..(h + 1) {
            for j in 0..w {
                let idx = i * src_stride + j;
                let a = src[idx] as i32;
                let b = src[idx + 1] as i32;
                fdata[i * w + j] = rpo(a * xf[0] + b * xf[1]) as u16;
            }
        }
        // Second pass.
        let mut temp = vec![0u8; h * w];
        for i in 0..h {
            for j in 0..w {
                let a = fdata[i * w + j] as i32;
                let b = fdata[(i + 1) * w + j] as i32;
                temp[i * w + j] = rpo(a * yf[0] + b * yf[1]) as u8;
            }
        }
        temp
    }

    const SHAPES: &[(usize, usize)] = &[
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (16, 8),
        (8, 16),
        (32, 16),
        (16, 32),
        (64, 32),
        (32, 64),
    ];

    /// Surface (a): the AVX2 interpolated `u8` temp is byte-identical to the
    /// scalar `round_power_of_two(·, 7)` two-pass temp, for every
    /// (xoffset, yoffset) ∈ 0..8 × 0..8 and every shape.
    #[test]
    fn differential_temp_byte_identical() {
        require_avx2!();
        let mut rng = Lcg32::new(0x7E_3902);
        for &(w, h) in SHAPES {
            let src_stride = w + 1 + rng.next_usize(16);
            // Footprint: (h+1) rows × (w+1) cols.
            let src_len = (h + 1) * src_stride + (w + 1) + 16;
            let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
            for yoff in 0..8 {
                for xoff in 0..8 {
                    let scalar = scalar_bilinear_temp(&src, src_stride, xoff, yoff, w, h);
                    let mut avx2 = vec![0u8; h * w];
                    let mut fdata = vec![0i16; (h + 1) * w];
                    // SAFETY: guarded by the `require_avx2!()` check at the top of
                    // this test, which returns early unless the running CPU
                    // supports AVX2.
                    unsafe {
                        bilinear_to_u8_temp(
                            &src,
                            src_stride,
                            xoff,
                            yoff,
                            w,
                            h,
                            BilinearBuffers {
                                fdata: &mut fdata,
                                temp: &mut avx2,
                            },
                        );
                    }
                    assert_eq!(
                        scalar, avx2,
                        "interpolated temp mismatch at w={w}, h={h}, xoff={xoff}, yoff={yoff}"
                    );
                }
            }
        }
    }

    /// Surface (b): the final `(var, sse)` equals the scalar
    /// `sub_pixel_variance` for every (xoffset, yoffset) and shape.
    #[test]
    fn differential_var_sse() {
        require_avx2!();
        let mut rng = Lcg32::new(0xA1_1CE5);
        for &(w, h) in SHAPES {
            let src_stride = w + 1 + rng.next_usize(16);
            let ref_stride = w + rng.next_usize(16);
            let src_len = (h + 1) * src_stride + (w + 1) + 16;
            let ref_len = (h - 1) * ref_stride + w;
            let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
            let reference: Vec<u8> = (0..ref_len).map(|_| rng.next_byte()).collect();
            for yoff in 0..8 {
                for xoff in 0..8 {
                    let s = scalar_svf(&src, src_stride, &reference, ref_stride, xoff, yoff, w, h);
                    let a = sub_pixel_variance_avx2(
                        &src, src_stride, &reference, ref_stride, xoff, yoff, w, h,
                    );
                    assert_eq!(
                        s, a,
                        "(var,sse) mismatch at w={w}, h={h}, xoff={xoff}, yoff={yoff}"
                    );
                }
            }
        }
    }

    /// Edge content: identity offset (0,0) must equal integer-pel variance;
    /// all-zero / all-max corners.
    #[test]
    fn differential_edge_content() {
        require_avx2!();
        for &(w, h) in SHAPES {
            let src_stride = w + 1;
            let ref_stride = w;
            let src_len = (h + 1) * src_stride + (w + 1) + 16;
            let ref_len = (h - 1) * ref_stride + w;
            let cases: &[(u8, u8)] = &[(0, 0), (255, 255), (0, 255), (255, 0), (128, 17)];
            for &(sval, rval) in cases {
                let src = vec![sval; src_len];
                let reference = vec![rval; ref_len];
                for &(xoff, yoff) in &[(0, 0), (4, 4), (1, 7), (7, 1), (3, 5)] {
                    let s = scalar_svf(&src, src_stride, &reference, ref_stride, xoff, yoff, w, h);
                    let a = sub_pixel_variance_avx2(
                        &src, src_stride, &reference, ref_stride, xoff, yoff, w, h,
                    );
                    assert_eq!(
                        s, a,
                        "edge mismatch w={w} h={h} s={sval} r={rval} ({xoff},{yoff})"
                    );
                }
            }
        }
    }

    #[test]
    fn production_wrapper_reuses_thread_local_scratch() {
        require_avx2!();
        let w = MAX_SUBPEL_WIDTH;
        let h = MAX_SUBPEL_HEIGHT;
        let src_stride = w + 1;
        let ref_stride = w;
        let src = vec![73u8; (h + 1) * src_stride];
        let reference = vec![191u8; h * ref_stride];
        let (fdata_ptr, temp_ptr) = SUBPEL_VARIANCE_SCRATCH.with_borrow_mut(|scratch| {
            scratch.fdata.fill(i16::MIN);
            scratch.temp.fill(u8::MAX);
            (scratch.fdata.as_ptr(), scratch.temp.as_ptr())
        });

        let actual = sub_pixel_variance_avx2(&src, src_stride, &reference, ref_stride, 3, 5, w, h);
        let expected = scalar_svf(&src, src_stride, &reference, ref_stride, 3, 5, w, h);
        assert_eq!(actual, expected);
        SUBPEL_VARIANCE_SCRATCH.with_borrow(|scratch| {
            assert_eq!(scratch.fdata.as_ptr(), fdata_ptr);
            assert_eq!(scratch.temp.as_ptr(), temp_ptr);
            assert_ne!(scratch.fdata[0], i16::MIN);
            assert_ne!(scratch.temp[0], u8::MAX);
        });
    }

    /// Forced-caps dispatch round-trip for the subpel_variance slot family.
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

        const SIZES: &[(BlockSize, usize, usize)] = &[
            (BlockSize::Block4x4, 4, 4),
            (BlockSize::Block8x8, 8, 8),
            (BlockSize::Block16x16, 16, 16),
            (BlockSize::Block32x32, 32, 32),
            (BlockSize::Block64x64, 64, 64),
            (BlockSize::Block16x8, 16, 8),
            (BlockSize::Block8x16, 8, 16),
            (BlockSize::Block32x16, 32, 16),
            (BlockSize::Block16x32, 16, 32),
            (BlockSize::Block64x32, 64, 32),
            (BlockSize::Block32x64, 32, 64),
        ];
        let mut rng = Lcg32::new(0x5B5B_5B5B);
        for &(bs, w, h) in SIZES {
            let src_stride = w + 1 + 3;
            let ref_stride = w + 2;
            let src: Vec<u8> = (0..(h + 1) * src_stride + (w + 1) + 16)
                .map(|_| rng.next_byte())
                .collect();
            let reference: Vec<u8> = (0..(h - 1) * ref_stride + w)
                .map(|_| rng.next_byte())
                .collect();
            let avx2_fn = d_avx2
                .subpel_variance(bs)
                .expect("supported size has a kernel");
            let scalar_fn = d_scalar
                .subpel_variance(bs)
                .expect("supported size has a kernel");
            for &(xoff, yoff) in &[(0, 0), (3, 5), (7, 7)] {
                let a = avx2_fn(&src, src_stride, xoff, yoff, &reference, ref_stride);
                let s = scalar_fn(&src, src_stride, xoff, yoff, &reference, ref_stride);
                assert_eq!(
                    a, s,
                    "dispatch round-trip subpel mismatch at {bs:?} ({xoff},{yoff})"
                );
            }
        }
    }
}
