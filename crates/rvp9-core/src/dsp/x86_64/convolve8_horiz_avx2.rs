//! 8-tap horizontal sub-pel convolution — x86_64 AVX2 implementation
//! (Wave J J3).
//!
//! The hot inner kernel of motion compensation: for every output pixel
//! `dst[y][x]`, convolve eight horizontally-adjacent source samples
//! with the resolved sub-pel filter row:
//!
//! ```text
//! dst[y][x] = clip255( ( Σ_{k=0..8} src[y][x+k] * filter_x[k] + 64 ) >> 7 )
//! ```
//!
//! Mirrors libvpx `convolve_horiz` (`vpx_dsp/vpx_convolve.c:22-42`) with
//! `x_step_q4 == SUBPEL_SHIFTS`. The scalar reference is
//! [`crate::inter::filter::convolve8_horiz`]; callers pre-shift `src`
//! left by three columns so `src[y*src_stride + x]` is the first of the
//! eight taps for output `(y, x)`.
//!
//! ## Algorithm (libvpx `vpx_filter_block1d*_h8_avx2` math, ported)
//!
//! Two rows are processed per outer iteration via 256-bit lanes. A
//! 256-bit register holds row `y` in lane 0 (low 128 bits) and row
//! `y+1` in lane 1 (high 128 bits); shuffling the four tap-pair masks
//! within each 128-bit lane gathers the eight tap operands, then
//! `_mm256_maddubs_epi16` does the per-lane multiply-add. The four
//! i16 partial sums are regrouped `(x0+x2)+(x1+x3)` with the `+64`
//! rounding bias folded in early and a saturating final add (the exact
//! libvpx order that keeps every intermediate inside i16 for all VP9
//! filter rows), then `srai 7` and `_mm256_packus_epi16` (whose
//! unsigned saturation performs the `[0, 255]` clip).
//!
//! Eight output columns are produced per inner step (low 128 bits of
//! the packed register = row `y`, high 128 bits = row `y+1`), so for a
//! 16-wide block the inner loop runs twice. Compared to the previous
//! 128-bit-only single-row form, this halves both the row iterations
//! and the per-row compute (each `_mm256_maddubs_epi16` does the work
//! of two `_mm_maddubs_epi16`).
//!
//! Odd-`h` tail and `w == 4` fall back to a single-row 128-bit path
//! that reuses the same shuffle masks and reduction order.
//!
//! ## Source over-read avoidance
//!
//! A 16-byte load at the right edge of any row reads up to one byte
//! past `row_base + w + 7`. For every row except the **last**, those
//! bytes land at the start of the next row — still inside the caller
//! buffer (the contract guarantees `(h-1)*src_stride + w + 7` valid
//! bytes). The last row is the only one that can fall past the tight
//! scalar footprint, and there only by one byte — which feeds output
//! lanes `≥ w` (discarded). To honor the *exact* scalar contract
//! (safe drop-in for any caller), the last row's `w + 7` source span
//! is copied into a zeroed `[u8; 80]` stack scratch and the SIMD loads
//! for that single row read from the scratch. All other rows are
//! loaded directly from `src`. The scratch copy is one `≤71`-byte
//! `copy_from_slice` per call, not per row.
//!
//! ## Aligned-load discipline (HLD V6 §G4 M6-6)
//!
//! All loads and stores use unaligned intrinsics. The 256-bit pair
//! loop and the non-last-row tail load directly from `src` via
//! `_mm_loadu_si128` / `_mm256_inserti128_si256`; the last row (and
//! `w == 4`, where the pair loop is bypassed and every row but the
//! last reads from `src` directly while the last reads from scratch)
//! uses `_mm_loadu_si128` against a zeroed `[u8; 80]` stack scratch.
//! Stores to `dst` are `_mm_storel_epi64` for 8-byte writes and
//! `core::ptr::write_unaligned` for the 4-byte (`w == 4`) tail. See
//! `crates/rvp9-core/src/dsp/x86_64/mod.rs`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_adds_epi16, _mm_cvtsi128_si32, _mm_loadu_si128,
    _mm_maddubs_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_setr_epi8, _mm_shuffle_epi8,
    _mm_srai_epi16, _mm_storel_epi64, _mm256_add_epi16, _mm256_adds_epi16, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_inserti128_si256,
    _mm256_maddubs_epi16, _mm256_packus_epi16, _mm256_set1_epi16, _mm256_shuffle_epi8,
    _mm256_srai_epi16,
};

/// Pack two adjacent filter taps into the replicated i8-pair form
/// consumed by `_mm_maddubs_epi16`'s signed operand. `lo` lands in the
/// low byte of every 16-bit lane, `hi` in the high byte.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair_128(lo: i16, hi: i16) -> __m128i {
    let packed = i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]);
    _mm_set1_epi16(packed)
}

/// 256-bit broadcast of a tap pair. Both 128-bit lanes hold the same
/// replicated `(lo, hi)` i8 pair so the same filter applies to row `y`
/// (low lane) and row `y+1` (high lane).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn tap_pair_256(lo: i16, hi: i16) -> __m256i {
    let packed = i16::from_le_bytes([lo as i8 as u8, hi as i8 as u8]);
    _mm256_set1_epi16(packed)
}

/// Convolve eight output columns from a 128-bit source register.
/// `s` holds `src[x0 .. x0+16)`; returns an `__m128i` whose low eight
/// 16-bit lanes are the rounded/shifted (pre-clip) results for output
/// columns `x0 .. x0+8`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_8(
    s: __m128i,
    f01: __m128i,
    f23: __m128i,
    f45: __m128i,
    f67: __m128i,
) -> __m128i {
    let m01 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let m23 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);
    let m45 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12);
    let m67 = _mm_setr_epi8(6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14);

    let x0 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m01), f01);
    let x1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m23), f23);
    let x2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m45), f45);
    let x3 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m67), f67);

    // Regroup (x0+x2)+(x1+x3) — the only order that keeps every
    // intermediate inside i16 for all VP9 filters (libvpx note).
    let k_64 = _mm_set1_epi16(1 << 6);
    let mut sum1 = _mm_add_epi16(x0, x2);
    let sum2 = _mm_add_epi16(x1, x3);
    sum1 = _mm_add_epi16(sum1, k_64);
    sum1 = _mm_adds_epi16(sum1, sum2);
    _mm_srai_epi16(sum1, 7)
}

/// Convolve eight output columns across **two rows** in one 256-bit
/// pass. `s` holds row `y`'s `src[x0 .. x0+16)` in the low lane and
/// row `y+1`'s `src[x0 .. x0+16)` in the high lane. Returns an
/// `__m256i` whose low/high 128-bit halves each contain the eight
/// rounded/shifted (pre-clip) i16 results for the corresponding row.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_8x2(
    s: __m256i,
    f01: __m256i,
    f23: __m256i,
    f45: __m256i,
    f67: __m256i,
) -> __m256i {
    // Shuffle masks replicated across both 128-bit lanes — the same
    // four tap-pair gathers used by `convolve8_8`, broadcast.
    let m01 = bcast_mask(_mm_setr_epi8(
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    ));
    let m23 = bcast_mask(_mm_setr_epi8(
        2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    ));
    let m45 = bcast_mask(_mm_setr_epi8(
        4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
    ));
    let m67 = bcast_mask(_mm_setr_epi8(
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
    ));

    let x0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m01), f01);
    let x1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m23), f23);
    let x2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m45), f45);
    let x3 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m67), f67);

    let k_64 = _mm256_set1_epi16(1 << 6);
    let mut sum1 = _mm256_add_epi16(x0, x2);
    let sum2 = _mm256_add_epi16(x1, x3);
    sum1 = _mm256_add_epi16(sum1, k_64);
    sum1 = _mm256_adds_epi16(sum1, sum2);
    _mm256_srai_epi16(sum1, 7)
}

/// Broadcast a 16-byte shuffle mask across both 128-bit lanes of a
/// `__m256i`. `_mm256_shuffle_epi8` indexes within each 128-bit lane
/// independently, so the same byte pattern in both lanes produces the
/// per-row gather we need.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bcast_mask(m: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(m), m)
}

/// Process one row's 8 output columns from a 128-bit source register
/// and store them. Handles the `w == 4` narrow tail via a 32-bit
/// write.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)] // mirrors the convolve8_8 arg list
unsafe fn store_row_8(
    dst_row: *mut u8,
    x: usize,
    remaining: usize,
    s: __m128i,
    f01: __m128i,
    f23: __m128i,
    f45: __m128i,
    f67: __m128i,
) {
    let res16 = convolve8_8(s, f01, f23, f45, f67);
    let res8 = _mm_packus_epi16(res16, res16);
    if remaining >= 8 {
        _mm_storel_epi64(dst_row.add(x) as *mut _, res8);
    } else {
        // Narrow tail (only w == 4 reaches here). Store the valid low
        // bytes via a 32-bit write.
        debug_assert_eq!(remaining, 4);
        let four = _mm_cvtsi128_si32(res8) as u32;
        core::ptr::write_unaligned(dst_row.add(x) as *mut u32, four);
    }
}

/// Run the scalar-equivalent SIMD path for one row, reading from
/// `scratch_ptr` (a per-row zero-padded copy of the source footprint)
/// and writing 8 output columns at a time to `dst_row`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn process_row(
    scratch_ptr: *const u8,
    dst_row: *mut u8,
    w: usize,
    f01: __m128i,
    f23: __m128i,
    f45: __m128i,
    f67: __m128i,
) {
    let mut x = 0usize;
    while x < w {
        let s = _mm_loadu_si128(scratch_ptr.add(x) as *const _);
        let remaining = w - x;
        store_row_8(dst_row, x, remaining, s, f01, f23, f45, f67);
        x += 8;
    }
}

/// AVX2 implementation of the 8-tap horizontal sub-pel convolution.
///
/// # Safety
///
/// Caller must guarantee the running CPU supports AVX2. The
/// `SimdDispatch::build` constructor only installs this fn (via the
/// safe wrapper [`convolve8_horiz_avx2`]) when `caps.avx2 == true`.
///
/// `src` must hold at least `(h - 1) * src_stride + w + 7` valid bytes
/// (the scalar `convolve8_horiz` footprint); `dst` at least
/// `(h - 1) * dst_stride + w`. `w` and `h` must be in `1..=64`.
// G4: loads from the source scratch use _mm_loadu_si128 / _mm256_inserti128_si256 (unaligned)
// G4: stores to `dst` use _mm_storel_epi64 / write_unaligned (unaligned)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convolve8_horiz_avx2_inner(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    debug_assert!((1..=64).contains(&w));
    // The inner column loop steps `x += 8`. `w == 4` is the only
    // narrow case the tail handles (via the 4-byte store branch of
    // `store_row_8`); all wider widths must be multiples of 8 so the
    // step lands cleanly on `x == w`. VP9 block widths are
    // {4, 8, 16, 32, 64}, which satisfies this.
    debug_assert!(
        w == 4 || w == 8 || w == 16 || w == 32 || w == 64,
        "convolve8_horiz_avx2: unsupported width {w}"
    );
    // `h` spans 1..=64 for a direct convolve8_horiz call, but the 2-D
    // convolve8 (J5) reuses this kernel for the horizontal pass over
    // `h + 7 ≤ 71` intermediate rows — so the upper bound is 71.
    debug_assert!((1..=71).contains(&h));
    debug_assert!(src.len() >= (h - 1) * src_stride + w + 7);
    debug_assert!(dst.len() >= (h - 1) * dst_stride + w);
    // Contract: only sub-pel filter rows reach this kernel. The
    // `maddubs_epi16` signed operand requires every tap to fit in i8
    // (|tap| ≤ 127). The sole VP9 tap that violates this is the
    // full-pel centre tap 128 (filter row 0) — that case never reaches
    // here because `inter_predictor` routes zero sub-pel offsets to
    // `convolve_copy`. See module doc.
    debug_assert!(
        filter_x.iter().all(|&t| (-128..=127).contains(&t)),
        "convolve8_horiz_avx2: tap out of i8 range (full-pel row must use convolve_copy)"
    );

    // 128-bit tap pairs (used by the last-row scratch path).
    let f01 = tap_pair_128(filter_x[0], filter_x[1]);
    let f23 = tap_pair_128(filter_x[2], filter_x[3]);
    let f45 = tap_pair_128(filter_x[4], filter_x[5]);
    let f67 = tap_pair_128(filter_x[6], filter_x[7]);
    // 256-bit broadcasted tap pairs (used by the 2-row main loop).
    let f01_256 = tap_pair_256(filter_x[0], filter_x[1]);
    let f23_256 = tap_pair_256(filter_x[2], filter_x[3]);
    let f45_256 = tap_pair_256(filter_x[4], filter_x[5]);
    let f67_256 = tap_pair_256(filter_x[6], filter_x[7]);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // 2-row main loop. **Precondition: `w >= 8`.** The inner column
    // step advances 8 outputs per iter, and each step issues a 16-byte
    // load at column `x`. The right-edge load (`x = w - 8`) reads
    // through `row_base + w + 7`, which the caller contract covers for
    // any row `r ≤ h - 1`. For `w == 4` the inner loop would only run
    // at `x = 0` and the 16-byte load would read through `row_base +
    // 15`, which is 4 bytes past the scalar footprint and can OOB on
    // tight strides (e.g. `src_stride == 4`). Gating on `w >= 8` skips
    // the pair loop entirely for `w == 4`; that case is handled by the
    // single-row 128-bit tail (`process_row` + `store_row_8`), which
    // already supports `w == 4` via `_mm_cvtsi128_si32` and uses the
    // last-row scratch path for any row whose tail read would escape.
    //
    // Both rows of a pair must be safe to read directly (no scratch):
    // the pair `(y, y+1)` is fully safe iff `y + 1 < h - 1`, i.e.,
    // `y + 2 < h`. The pair containing the very last row falls to the
    // tail so its scratch-staged load is used.
    let mut y = 0usize;
    while y + 2 < h && w >= 8 {
        let row_a = src_ptr.add(y * src_stride);
        let row_b = src_ptr.add((y + 1) * src_stride);
        let dst_a = dst_ptr.add(y * dst_stride);
        let dst_b = dst_ptr.add((y + 1) * dst_stride);

        let mut x = 0usize;
        while x < w {
            // Assemble a 256-bit src register: row `y` in low lane,
            // row `y+1` in high lane.
            let s_a = _mm_loadu_si128(row_a.add(x) as *const _);
            let s_b = _mm_loadu_si128(row_b.add(x) as *const _);
            let s = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s_a), s_b);

            let res16 = convolve8_8x2(s, f01_256, f23_256, f45_256, f67_256);
            // Pack 16-bit → 8-bit. `_mm256_packus_epi16(a, a)` packs
            // each 128-bit lane independently, so the low lane's 8
            // useful bytes land at positions 0..8 of the low lane and
            // the high lane's at positions 0..8 of the high lane.
            let res8 = _mm256_packus_epi16(res16, res16);
            let lo = _mm256_castsi256_si128(res8);
            let hi = _mm256_extracti128_si256::<1>(res8);

            let remaining = w - x;
            if remaining >= 8 {
                _mm_storel_epi64(dst_a.add(x) as *mut _, lo);
                _mm_storel_epi64(dst_b.add(x) as *mut _, hi);
            } else {
                debug_assert_eq!(remaining, 4);
                let four_a = _mm_cvtsi128_si32(lo) as u32;
                let four_b = _mm_cvtsi128_si32(hi) as u32;
                core::ptr::write_unaligned(dst_a.add(x) as *mut u32, four_a);
                core::ptr::write_unaligned(dst_b.add(x) as *mut u32, four_b);
            }
            x += 8;
        }

        y += 2;
    }

    // Tail: 1, 2, or 3 remaining rows, the last of which is the last
    // row of the buffer and may not have an OOB-safe byte after it.
    // Process each via the 1-row 128-bit path. For all but the last
    // row, read `src` directly; for the last row, stage into a zeroed
    // scratch first so the 16-byte load never escapes the contract.
    while y < h {
        let dst_row = dst_ptr.add(y * dst_stride);
        if y < h - 1 {
            let row = src_ptr.add(y * src_stride);
            process_row(row, dst_row, w, f01, f23, f45, f67);
        } else {
            // Last row — copy footprint into a zeroed scratch.
            let mut scratch = [0u8; 80];
            let base = y * src_stride;
            scratch[..w + 7].copy_from_slice(&src[base..base + w + 7]);
            process_row(scratch.as_ptr(), dst_row, w, f01, f23, f45, f67);
        }
        y += 1;
    }
}

/// Safe wrapper for the AVX2 8-tap horizontal sub-pel convolution.
///
/// The dispatch slot [`crate::dsp::SimdDispatch::convolve8_horiz`]
/// points at this wrapper on AVX2 hosts and at the scalar reference
/// (`crate::inter::filter::convolve8_horiz`) elsewhere.
#[cfg(target_arch = "x86_64")]
pub fn convolve8_horiz_avx2(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    filter_x: &[i16; 8],
    w: usize,
    h: usize,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // time (see `crate::dsp::dispatch::init`) and are monotonic for the
    // process lifetime. Test paths gate every call on a runtime
    // `is_x86_feature_detected!("avx2")` check via `require_avx2!()`.
    unsafe {
        convolve8_horiz_avx2_inner(src, src_stride, dst, dst_stride, filter_x, w, h);
    }
}

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `convolve8_horiz_avx2`.
    use super::convolve8_horiz_avx2;
    use crate::inter::filter::{
        BILINEAR_KERNEL, EIGHTTAP_KERNEL, EIGHTTAP_SHARP_KERNEL, EIGHTTAP_SMOOTH_KERNEL,
        convolve8_horiz,
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

    /// Differential vs scalar across the VP9 (w, h) matrix, every
    /// sub-pel filter row, and all four kernel tables. Row 0 (the
    /// full-pel passthrough, centre tap 128) is skipped: it never
    /// reaches this kernel in production (`inter_predictor` routes
    /// zero sub-pel offsets to `convolve_copy`) and its 128 tap does
    /// not fit the `maddubs` i8 operand. The cartesian product over
    /// the remaining 15 rows × 4 kernels × shapes already exceeds 1024
    /// distinct configurations per shape.
    #[test]
    fn differential_all_filters() {
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
        let mut state: u32 = 0x1234_5678;
        for &(w, h) in shapes {
            for kernel in kernels {
                for filter in kernel.iter().skip(1) {
                    for _iter in 0..16 {
                        let src_stride = w + 7 + (lcg_next(&mut state) as usize & 0x7);
                        let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                        let src_len = (h - 1) * src_stride + w + 7;
                        let dst_len = (h - 1) * dst_stride + w;
                        let mut src = vec![0u8; src_len];
                        for s in src.iter_mut() {
                            *s = (lcg_next(&mut state) & 0xFF) as u8;
                        }
                        let mut dst_scalar = vec![0u8; dst_len];
                        let mut dst_avx2 = vec![0u8; dst_len];
                        convolve8_horiz(
                            &src,
                            src_stride,
                            &mut dst_scalar,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        convolve8_horiz_avx2(
                            &src,
                            src_stride,
                            &mut dst_avx2,
                            dst_stride,
                            filter,
                            w,
                            h,
                        );
                        assert_eq!(
                            dst_scalar, dst_avx2,
                            "mismatch w={w} h={h} filter={filter:?}"
                        );
                    }
                }
            }
        }
    }

    /// Shapes the head-to-head harness doesn't hit but production
    /// callers can: odd heights (forces the 1-row tail), narrow w=8
    /// pair count of 2, and the 32-wide square. Pinned random seed so
    /// failures bisect cleanly.
    ///
    /// Includes **tight-stride** cases (`src_stride == w`, no slack)
    /// to guard against pair-loop over-reads. Pre-fix, the
    /// `(w=4, h=6, src_stride=4)` case would overrun the contract by
    /// one byte at row 3 (where the pair loop's 16-byte direct load
    /// reads `row_base + 15`, past the contract's
    /// `(h-1)*4 + 11 = 31`-byte footprint). Post-fix the pair loop
    /// is gated on `w >= 8` so the case falls to the tail and stays
    /// in-bounds.
    #[test]
    fn differential_odd_and_narrow_shapes() {
        require_avx2!();
        // (w, h, src_stride). `src_stride == 0` means
        // "default randomized stride with slack" (preserve the
        // original mix of cases); `src_stride > 0` pins it to that
        // exact tight value.
        //
        // Shapes exercised:
        //   - (8, 4) / (8, 5) — smallest width that uses the pair loop
        //     (odd h hits both pair and 1-row tail)
        //   - (16, 15) — odd h, pair-loop predominant
        //   - (32, 32) / (32, 33) — square / odd, multiple inner iters
        //   - (4, 1) / (8, 1) / (16, 1) — h == 1 (zero pairs, tail-only)
        //   - (4, 6, 4) — **B1 regression guard**: pre-fix this
        //     overran the contract by 1 byte (would crash under MIRI,
        //     OOB on a tight allocation)
        //   - (4, 2, 4) — minimal tight-stride w==4 with h >= 2
        //   - (8, 5, 8) — tight stride at the next-narrower width
        //   - (16, 8, 16) — tight stride at w==16 (exercises pair
        //     loop's right-edge load, which is safe by contract)
        let shapes: &[(usize, usize, usize)] = &[
            (8, 4, 0),
            (8, 5, 0),
            (16, 15, 0),
            (32, 32, 0),
            (32, 33, 0),
            (4, 1, 0),
            (8, 1, 0),
            (16, 1, 0),
            // Tight-stride cases (no slack):
            (4, 6, 4),
            (4, 2, 4),
            (8, 5, 8),
            (16, 8, 16),
        ];
        let kernels = [
            &EIGHTTAP_KERNEL,
            &EIGHTTAP_SMOOTH_KERNEL,
            &EIGHTTAP_SHARP_KERNEL,
            &BILINEAR_KERNEL,
        ];
        let mut state: u32 = 0xDEAD_BEEF;
        for &(w, h, fixed_stride) in shapes {
            for kernel in kernels {
                // Pick a few sub-pel rows (1, 4, 8, 12) — covers
                // every-other-tap-zero, mid-range, half-pel symmetric,
                // and asymmetric-high.
                for &r in &[1usize, 4, 8, 12] {
                    let filter = &kernel[r];
                    let src_stride = if fixed_stride > 0 {
                        fixed_stride
                    } else {
                        w + 7 + (lcg_next(&mut state) as usize & 0x7)
                    };
                    let dst_stride = w + (lcg_next(&mut state) as usize & 0x7);
                    // Allocate exactly the scalar contract footprint —
                    // **no slack** — so any pair-loop over-read past
                    // `w + 7` on a non-last row trips a Vec OOB
                    // (under MIRI) or reads adjacent heap (under
                    // release). With B1's `w >= 8` guard, this stays
                    // in-bounds for every shape above.
                    let src_len = (h - 1) * src_stride + w + 7;
                    let dst_len = (h - 1) * dst_stride + w;
                    let mut src = vec![0u8; src_len];
                    for s in src.iter_mut() {
                        *s = (lcg_next(&mut state) & 0xFF) as u8;
                    }
                    let mut dst_scalar = vec![0u8; dst_len];
                    let mut dst_avx2 = vec![0u8; dst_len];
                    convolve8_horiz(&src, src_stride, &mut dst_scalar, dst_stride, filter, w, h);
                    convolve8_horiz_avx2(&src, src_stride, &mut dst_avx2, dst_stride, filter, w, h);
                    assert_eq!(
                        dst_scalar, dst_avx2,
                        "mismatch w={w} h={h} src_stride={src_stride} r={r} filter={filter:?}"
                    );
                }
            }
        }
    }

    /// Tautology guard — the AVX2 path must actually be exercised on the
    /// test host, and a known sub-pel filter must produce a non-trivial
    /// (non-passthrough) result.
    #[test]
    fn tautology_guard_avx2_path() {
        require_avx2!();
        assert!(is_x86_feature_detected!("avx2"));
        // EIGHTTAP row 8 is the half-pel filter — symmetric, definitely
        // not the identity. Feed a ramp and check output differs from a
        // plain copy of the centre tap.
        let filter = &EIGHTTAP_KERNEL[8];
        let w = 8;
        let h = 1;
        let src: Vec<u8> = (0..(w + 7)).map(|i| (i * 13 + 7) as u8).collect();
        let mut dst = vec![0u8; w];
        convolve8_horiz_avx2(&src, w + 7, &mut dst, w, filter, w, h);
        // Output must not equal src[3..3+w] (the integer-pel passthrough).
        assert_ne!(
            &dst[..],
            &src[3..3 + w],
            "half-pel filter must blur the ramp"
        );
    }

    /// Round-trip through SimdDispatch with forced-AVX2 and
    /// forced-scalar caps — identical output on a deterministic input.
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

        let filter = &EIGHTTAP_SHARP_KERNEL[5];
        let mut state: u32 = 0xABCD_1234;
        for &(w, h) in &[(4usize, 4usize), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let stride = w + 7;
            let src_len = (h - 1) * stride + w + 7;
            let mut src = vec![0u8; src_len];
            for s in src.iter_mut() {
                *s = (lcg_next(&mut state) & 0xFF) as u8;
            }
            let mut dst_avx2 = vec![0u8; h * w];
            let mut dst_scalar = vec![0u8; h * w];
            (d_avx2.convolve8_horiz())(&src, stride, &mut dst_avx2, w, filter, w, h);
            (d_scalar.convolve8_horiz())(&src, stride, &mut dst_scalar, w, filter, w, h);
            assert_eq!(
                dst_avx2, dst_scalar,
                "dispatch round-trip mismatch w={w} h={h}"
            );
        }
    }
}
