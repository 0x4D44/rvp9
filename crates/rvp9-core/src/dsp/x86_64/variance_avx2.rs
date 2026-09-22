//! Variance `(sum, sse)` reductions — x86_64 AVX2 implementation (PR4).
//!
//! The inter RD model (`crate::mode_decision::model_rd`) needs two integer
//! reductions over the `src − pred` residual of a block:
//!
//! - a **block-level** `(sum, sse)` for `compute_variance`
//!   (`model_rd.rs`), where `sum = Σ(src − pred)` and `sse = Σ(src − pred)²`;
//! - a **per-8×8-cell** `(sum, sse)` for `compute_variance_8x8`
//!   (`model_rd.rs`), the per-leaf hot loop for blocks ≥ 32×32.
//!
//! ## Mandate — reductions only, finishers stay scalar
//!
//! These kernels return ONLY the integer-exact `(sum, sse)` reductions.
//! The three distinct scalar finishers in `model_rd.rs`
//! (`compute_variance`'s `saturating_sub`, `compute_variance_8x8`'s abs-diff
//! `>>6`, and `aggregate_to_tx_size`'s abs-diff `>>tx_pixels_log2`) consume
//! these reductions UNCHANGED. Byte-identity reduces to "the reductions are
//! integer-exact vs the scalar accumulation" — which holds by construction
//! (see below) and is pinned by the `differential_*` tests in this module.
//!
//! ## Byte-exactness with scalar (by construction)
//!
//! Per-pixel `diff = src − pred ∈ [−255, 255]`. The reductions are:
//!
//! - `sse = Σ diff²`. `diff²` peaks at `255² = 65 025`; over a 64×64 block
//!   the total peaks at `4096 × 65 025 ≈ 2.66 × 10⁸ < 2³²`, so `sse` fits a
//!   `u32` (and the scalar reference accumulates it as `u32` / `u64`). The
//!   AVX2 path forms `diff` as `i16`, squares-and-pairwise-adds via
//!   `_mm256_madd_epi16(diff, diff)` (exact `i32` per pair, no overflow:
//!   max pair `2 × 65 025 < 2³¹`), then widens partial sums to `i64`/`u64`
//!   before they could overflow `i32`. Integer addition is associative, so
//!   the horizontal sum equals the scalar `Σ diff²` exactly.
//! - `sum = Σ diff`. `|sum|` peaks at `4096 × 255 ≈ 1.04 M < 2³¹`. The AVX2
//!   path widens each `diff` to `i32` via `_mm256_madd_epi16(diff, ones16)`
//!   (each `i16` lane multiplied by 1 and pairwise-added → exact `i32`),
//!   accumulates in `i32` lanes (no overflow), and horizontally sums to
//!   `i64`. Exact vs the scalar `Σ diff`, including sign.
//!
//! ## Size dispatch
//!
//! The block-level kernel is `(w, h)`-generic over `w ∈ {4, 8, 16, 32, 64}`
//! (heights in the same set), loading one row at a time and widening bytes to
//! `i16` lanes. The per-cell kernel is fixed 8×8. Mirrors libvpx
//! `vpx_dsp/x86/variance_avx2.c` (`variance_avx2` / `vpx_get8x8var`'s madd
//! reduction shape), specialised to the byte-widening `madd` form so the
//! sum/sse are exact `i32`/`u32` partials.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! Source / prediction strides are arbitrary (frame geometry, sub-block
//! offsets), so every SIMD load is the `loadu_*` unaligned form. See
//! `crates/rvp9-core/src/dsp/x86_64/mod.rs:14-26` for the workspace-wide
//! sentinel-comment convention.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadl_epi64, _mm_loadu_si128,
    _mm_unpackhi_epi64, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
    _mm256_sub_epi16,
};

/// Horizontally sum the eight `i32` lanes of a 256-bit accumulator to one
/// `i32`. The partial sums never overflow `i32` (sse partials ≤ 2.66e8, sum
/// partials |·| ≤ 1.04e6), so a plain `i32` fold is exact.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_epi32(v: __m256i) -> i32 {
    // Fold lanes [4..8) into [0..4), then [2..4) into [0..2), then [1] into [0].
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s)); // lanes 2,3 -> 0,1
    let s = _mm_add_epi32(s, _mm_srli_si128_4(s)); // lane 1 -> 0
    _mm_cvtsi128_si32(s)
}

/// `s >> 4 bytes` within an xmm (bring lane 1 into lane 0). Wraps
/// `_mm_srli_si128::<4>` so the const generic stays at the call site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn _mm_srli_si128_4(s: __m128i) -> __m128i {
    core::arch::x86_64::_mm_srli_si128::<4>(s)
}

/// Reduce one row's `(sum_acc, sse_acc)` over `w` byte pairs, given the
/// already-widened `i16` diff vector and the `ones16` multiplier. Folds
/// `madd(diff, ones)` into `sum_acc` and `madd(diff, diff)` into `sse_acc`.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_diff(
    diff: __m256i,
    ones16: __m256i,
    sum_acc: &mut __m256i,
    sse_acc: &mut __m256i,
) {
    // sum += Σ diff  (each i16 * 1, pairwise-added into i32 lanes)
    *sum_acc = _mm256_add_epi32(*sum_acc, _mm256_madd_epi16(diff, ones16));
    // sse += Σ diff² (each i16², pairwise-added into i32 lanes)
    *sse_acc = _mm256_add_epi32(*sse_acc, _mm256_madd_epi16(diff, diff));
}

/// Load and widen one 16-byte row of `src − pred`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_diff_16(src: *const u8, pred: *const u8) -> __m256i {
    let sw = _mm256_cvtepu8_epi16(_mm_loadu_si128(src.cast()));
    let pw = _mm256_cvtepu8_epi16(_mm_loadu_si128(pred.cast()));
    _mm256_sub_epi16(sw, pw)
}

/// `(w, h)`-generic AVX2 `(sum, sse)` reduction over `src − pred`.
///
/// Returns `(sum, sse)` with `sum = Σ(src − pred)` (signed) and
/// `sse = Σ(src − pred)²`, integer-exact vs the scalar accumulation.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. `src` must hold at
/// least `(h − 1) * src_stride + w` bytes; `pred` at least
/// `(h − 1) * pred_stride + w`. `w` must be one of `{4, 8, 16, 32, 64}`;
/// any other width is delegated to the scalar reference.
// G4: loads from `src` / `pred` use _mm256_loadu_si256 / _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn var_sum_sse_inner(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> (i64, u64) {
    debug_assert!(matches!(w, 4 | 8 | 16 | 32 | 64));
    debug_assert!(src.len() >= h.saturating_sub(1) * src_stride + w);
    debug_assert!(pred.len() >= h.saturating_sub(1) * pred_stride + w);

    let src_ptr = src.as_ptr();
    let pred_ptr = pred.as_ptr();
    let ones16 = _mm256_set1_epi16(1);

    let mut sum_acc = _mm256_setzero_si256();
    let mut sse_acc = _mm256_setzero_si256();

    // Process the row in 16-byte chunks (each widened to a full 256-bit i16
    // vector). w ∈ {4, 8, 16, 32, 64} → {partial, partial, 1, 2, 4} chunks.
    let chunks = w / 16;
    let tail = w % 16; // 4 or 8 for w < 16; 0 otherwise.

    for r in 0..h {
        let s_row = src_ptr.add(r * src_stride);
        let p_row = pred_ptr.add(r * pred_stride);
        for c in 0..chunks {
            let s = _mm_loadu_si128(s_row.add(c * 16) as *const __m128i);
            let p = _mm_loadu_si128(p_row.add(c * 16) as *const __m128i);
            let sw = _mm256_cvtepu8_epi16(s);
            let pw = _mm256_cvtepu8_epi16(p);
            let diff = _mm256_sub_epi16(sw, pw);
            accumulate_diff(diff, ones16, &mut sum_acc, &mut sse_acc);
        }
        if tail != 0 {
            // Load `tail` (4 or 8) bytes into the low lane of an xmm; the
            // upper bytes are loaded too but masked to diff=0 by reusing the
            // same partial-width load on both sides. We zero-fill the unused
            // high bytes so they contribute 0 to both sum and sse.
            let (s, p) = if tail == 8 {
                (
                    _mm_loadl_epi64(s_row.add(chunks * 16) as *const __m128i),
                    _mm_loadl_epi64(p_row.add(chunks * 16) as *const __m128i),
                )
            } else {
                // tail == 4: read 4 bytes each into the low lane.
                let sv = core::ptr::read_unaligned(s_row.add(chunks * 16) as *const u32);
                let pv = core::ptr::read_unaligned(p_row.add(chunks * 16) as *const u32);
                (
                    core::mem::transmute::<[u32; 4], __m128i>([sv, 0, 0, 0]),
                    core::mem::transmute::<[u32; 4], __m128i>([pv, 0, 0, 0]),
                )
            };
            // Widen only the relevant bytes; the zeroed high bytes give
            // diff = 0 there, contributing nothing to sum/sse.
            let sw = _mm256_cvtepu8_epi16(s);
            let pw = _mm256_cvtepu8_epi16(p);
            let diff = _mm256_sub_epi16(sw, pw);
            accumulate_diff(diff, ones16, &mut sum_acc, &mut sse_acc);
        }
    }

    let sum = hsum256_epi32(sum_acc) as i64;
    let sse = hsum256_epi32(sse_acc) as u32 as u64;
    (sum, sse)
}

/// Four-way 16×16 AVX2 reduction with independent dependency chains.
///
/// The generic kernel is intentionally retained for every other shape. The
/// 16×16 shape is hot enough to justify keeping four `(sum, sse)` pairs live:
/// each pair covers four rows, so the final fold remains well within the
/// exact i32 ranges documented above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn var_sum_sse_16x16_inner(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i64, u64) {
    debug_assert!(src.len() >= 15 * src_stride + 16);
    debug_assert!(pred.len() >= 15 * pred_stride + 16);

    let src_ptr = src.as_ptr();
    let pred_ptr = pred.as_ptr();
    let ones16 = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let mut sum0 = zero;
    let mut sum1 = zero;
    let mut sum2 = zero;
    let mut sum3 = zero;
    let mut sse0 = zero;
    let mut sse1 = zero;
    let mut sse2 = zero;
    let mut sse3 = zero;

    // Interleave four independent chains. This leaves the multiply/add
    // latency of one chain available while the next row is being widened.
    for r in 0..4 {
        let row0 = r * src_stride;
        let row1 = (r + 4) * src_stride;
        let row2 = (r + 8) * src_stride;
        let row3 = (r + 12) * src_stride;
        let pred_row0 = r * pred_stride;
        let pred_row1 = (r + 4) * pred_stride;
        let pred_row2 = (r + 8) * pred_stride;
        let pred_row3 = (r + 12) * pred_stride;

        let diff0 = load_diff_16(src_ptr.add(row0), pred_ptr.add(pred_row0));
        accumulate_diff(diff0, ones16, &mut sum0, &mut sse0);
        let diff1 = load_diff_16(src_ptr.add(row1), pred_ptr.add(pred_row1));
        accumulate_diff(diff1, ones16, &mut sum1, &mut sse1);
        let diff2 = load_diff_16(src_ptr.add(row2), pred_ptr.add(pred_row2));
        accumulate_diff(diff2, ones16, &mut sum2, &mut sse2);
        let diff3 = load_diff_16(src_ptr.add(row3), pred_ptr.add(pred_row3));
        accumulate_diff(diff3, ones16, &mut sum3, &mut sse3);
    }

    let sum_acc = _mm256_add_epi32(_mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));
    let sse_acc = _mm256_add_epi32(_mm256_add_epi32(sse0, sse1), _mm256_add_epi32(sse2, sse3));
    let sum = hsum256_epi32(sum_acc) as i64;
    let sse = hsum256_epi32(sse_acc) as u32 as u64;
    (sum, sse)
}

/// AVX2 block-level `(sum, sse)` reduction over `src − pred`. Byte-exact with
/// the scalar accumulation in `crate::mode_decision::model_rd::compute_variance`
/// (which then applies its own scalar `saturating_sub` finisher).
///
/// The dispatch slot points here on AVX2 hosts and at the scalar reference
/// elsewhere. `w` is one of the 13 VP9 block widths.
#[cfg(target_arch = "x86_64")]
pub fn var_sum_sse_avx2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> (i64, u64) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper when
    // `caps.avx2 == true`. Caps are detected once at OnceLock init time
    // (see `crate::dsp::dispatch::init`) and are monotonic for the process
    // lifetime.
    if w == 16 && h == 16 {
        // SAFETY: The same AVX2 dispatch invariant as the generic path; the
        // specialized kernel has the identical slice bounds contract.
        unsafe { var_sum_sse_16x16_inner(src, src_stride, pred, pred_stride) }
    } else {
        // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper when
        // `caps.avx2 == true`.
        unsafe { var_sum_sse_inner(src, src_stride, pred, pred_stride, w, h) }
    }
}

/// Generate the 13 per-size AVX2 block `(sum, sse)` safe wrappers. Each
/// mirrors the per-size scalar `crate::dsp::distortion::var_sum_sse_{W}x{H}`
/// shim signature exactly so it can be installed directly into the
/// `SimdDispatch::variance` per-size table.
#[cfg(target_arch = "x86_64")]
macro_rules! var_sum_sse_avx2_fn {
    ($name:ident, $w:expr, $h:expr) => {
        #[doc = concat!("AVX2 (sum, sse) reduction for a ", stringify!($w), "×",
                                stringify!($h), " block. Byte-exact with ",
                                "`crate::dsp::distortion::var_sum_sse_", stringify!($w), "x",
                                stringify!($h), "`.")]
        pub fn $name(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> (i64, u64) {
            // SAFETY: installed only on AVX2 hosts by `SimdDispatch::build`.
            unsafe { var_sum_sse_inner(src, src_stride, pred, pred_stride, $w, $h) }
        }
    };
}

#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_4x4_avx2, 4, 4);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_4x8_avx2, 4, 8);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_8x4_avx2, 8, 4);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_8x8_avx2, 8, 8);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_8x16_avx2, 8, 16);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_16x8_avx2, 16, 8);
/// AVX2 `(sum, sse)` reduction for a 16×16 block using four independent
/// accumulator pairs. Byte-exact with the scalar reduction.
#[cfg(target_arch = "x86_64")]
pub fn var_sum_sse_16x16_avx2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i64, u64) {
    // SAFETY: installed only on AVX2 hosts by `SimdDispatch::build`.
    unsafe { var_sum_sse_16x16_inner(src, src_stride, pred, pred_stride) }
}
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_16x32_avx2, 16, 32);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_32x16_avx2, 32, 16);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_32x32_avx2, 32, 32);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_32x64_avx2, 32, 64);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_64x32_avx2, 64, 32);
#[cfg(target_arch = "x86_64")]
var_sum_sse_avx2_fn!(var_sum_sse_64x64_avx2, 64, 64);

/// AVX2 per-8×8-cell `(sum, sse)` reduction over `src − pred`. Byte-exact with
/// the per-cell scalar accumulation in
/// `crate::mode_decision::model_rd::compute_variance_8x8` (which then applies
/// its own scalar abs-diff `>>6` finisher per cell).
///
/// Returns `(sum, sse)` for one 8×8 cell. `sum ∈ [−16320, 16320]` (fits
/// `i32`) and `sse ∈ [0, 4 161 600]` (fits `u32`), matching the scalar
/// `cell_sum: i32` / `cell_sse: u32` types.
#[cfg(target_arch = "x86_64")]
pub fn get_var_sse_8x8_avx2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) -> (i32, u32) {
    // SAFETY: see `var_sum_sse_avx2` — installed only on AVX2 hosts.
    let (sum, sse) = unsafe { var_sum_sse_inner(src, src_stride, pred, pred_stride, 8, 8) };
    (sum as i32, sse as u32)
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    use super::*;

    macro_rules! require_avx2 {
        () => {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
        };
    }

    /// Deterministic LCG mirroring the SAD / convolve test RNG.
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

    /// Scalar reference reduction: exactly the accumulation
    /// `compute_variance` performs (i64 sum, u64 sse), before any finisher.
    fn scalar_sum_sse(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> (i64, u64) {
        let mut sum: i64 = 0;
        let mut sum_sq: u64 = 0;
        for row in 0..h {
            let s_off = row * src_stride;
            let p_off = row * pred_stride;
            for col in 0..w {
                let diff = src[s_off + col] as i32 - pred[p_off + col] as i32;
                sum += diff as i64;
                sum_sq += (diff * diff) as u64;
            }
        }
        (sum, sum_sq)
    }

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

    /// Differential AVX2 vs scalar `(sum, sse)` over all 13 shapes, with
    /// non-tight strides (0..16 jitter) — the realistic call pattern from the
    /// RD model where `src`/`pred` index into rasters.
    #[test]
    fn differential_random_all_sizes() {
        require_avx2!();
        let mut rng = Lcg32::new(0x5EED_BEEF);
        for &(w, h) in SHAPES {
            for _ in 0..512 {
                let src_stride = w + rng.next_usize(16);
                let pred_stride = w + rng.next_usize(16);
                let src_len = (h - 1) * src_stride + w;
                let pred_len = (h - 1) * pred_stride + w;
                let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
                let pred: Vec<u8> = (0..pred_len).map(|_| rng.next_byte()).collect();
                let s = scalar_sum_sse(&src, src_stride, &pred, pred_stride, w, h);
                let a = var_sum_sse_avx2(&src, src_stride, &pred, pred_stride, w, h);
                assert_eq!(s, a, "AVX2/scalar (sum,sse) mismatch at w={w}, h={h}");
            }
        }
    }

    /// Signed-sum sign correctness + the sse overflow corner: src all-0 vs
    /// pred all-255 (sum = −w*h*255, sse = w*h*255²) and the reverse
    /// (sum = +w*h*255). Asserts exact equality including the sum's sign.
    #[test]
    fn differential_edge_content() {
        require_avx2!();
        for &(w, h) in SHAPES {
            for &(ss, ps) in &[(w, w), (w + 7, w + 3)] {
                let src_len = (h - 1) * ss + w;
                let pred_len = (h - 1) * ps + w;
                let n = (w * h) as i64;
                let cases: &[(u8, u8, i64, u64)] = &[
                    (0, 0, 0, 0),
                    (255, 255, 0, 0),
                    (0, 255, -255 * n, (255u64 * 255) * n as u64),
                    (255, 0, 255 * n, (255u64 * 255) * n as u64),
                    (128, 17, 111 * n, (111u64 * 111) * n as u64),
                ];
                for &(sval, pval, exp_sum, exp_sse) in cases {
                    let src = vec![sval; src_len];
                    let pred = vec![pval; pred_len];
                    let s = scalar_sum_sse(&src, ss, &pred, ps, w, h);
                    let a = var_sum_sse_avx2(&src, ss, &pred, ps, w, h);
                    assert_eq!(
                        s, a,
                        "AVX2/scalar mismatch at w={w}, h={h}, s={sval}, p={pval}"
                    );
                    assert_eq!(
                        a,
                        (exp_sum, exp_sse),
                        "AVX2 (sum,sse) wrong vs hand-computed at w={w}, h={h}"
                    );
                }
            }
        }
    }

    /// The per-8×8-cell kernel must equal the scalar 8×8 accumulation
    /// (i32 sum, u32 sse) — feeds finisher #2.
    #[test]
    fn differential_8x8_cell() {
        require_avx2!();
        let mut rng = Lcg32::new(0xC0FF_EE42);
        for _ in 0..2048 {
            let src_stride = 8 + rng.next_usize(16);
            let pred_stride = 8 + rng.next_usize(16);
            let src_len = 7 * src_stride + 8;
            let pred_len = 7 * pred_stride + 8;
            let src: Vec<u8> = (0..src_len).map(|_| rng.next_byte()).collect();
            let pred: Vec<u8> = (0..pred_len).map(|_| rng.next_byte()).collect();
            let (ssum, ssse) = scalar_sum_sse(&src, src_stride, &pred, pred_stride, 8, 8);
            let (asum, asse) = get_var_sse_8x8_avx2(&src, src_stride, &pred, pred_stride);
            assert_eq!(
                (ssum as i32, ssse as u32),
                (asum, asse),
                "8x8 cell mismatch"
            );
        }
    }

    /// Forced-caps dispatch round-trip for the variance slot family: the
    /// per-size reduction resolved from `build(avx2:true)` must equal the
    /// `build(avx2:false)` slot for every size (mirrors the SAD round-trip).
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
        let mut rng = Lcg32::new(0xD15A_5EED);
        for &bs in SIZES {
            let w = usize::from(bs.width());
            let h = usize::from(bs.height());
            let ss = w + 5;
            let ps = w + 2;
            let src: Vec<u8> = (0..(h - 1) * ss + w).map(|_| rng.next_byte()).collect();
            let pred: Vec<u8> = (0..(h - 1) * ps + w).map(|_| rng.next_byte()).collect();
            let a = (d_avx2.variance(bs))(&src, ss, &pred, ps);
            let s = (d_scalar.variance(bs))(&src, ss, &pred, ps);
            assert_eq!(a, s, "dispatch round-trip variance mismatch at {bs:?}");
        }

        // Per-8×8-cell slot round-trip.
        let mut rng = Lcg32::new(0x8888_0808);
        for _ in 0..64 {
            let ss = 8 + rng.next_usize(8);
            let ps = 8 + rng.next_usize(8);
            let src: Vec<u8> = (0..7 * ss + 8).map(|_| rng.next_byte()).collect();
            let pred: Vec<u8> = (0..7 * ps + 8).map(|_| rng.next_byte()).collect();
            let a = (d_avx2.get_var_sse_8x8)(&src, ss, &pred, ps);
            let s = (d_scalar.get_var_sse_8x8)(&src, ss, &pred, ps);
            assert_eq!(a, s, "dispatch round-trip get_var_sse_8x8 mismatch");
        }
    }
}
