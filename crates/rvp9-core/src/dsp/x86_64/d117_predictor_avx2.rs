//! D117_PRED directional intra-predictor — x86_64 AVX2 (Wave K K5).
//!
//! Scalar reference: [`crate::intra_pred::d117_predictor`] (generic
//! for all `bs ∈ {4,8,16,32}` — no 4×4 special). Bit-exact against
//! the scalar reference (differential test is the oracle).
//!
//! ## Algorithm (uses `above`, `left`, `above_left`)
//!
//! Three seed edges are filtered, then a diagonal recurrence fills the
//! interior:
//! - **Row 0** = `AVG2` over `A = [al, above[0..bs]]`:
//!   `dst[0][c] = AVG2(A[c], A[c+1])`.
//! - **Row 1** = `AVG3` over `B = [left[0], al, above[0..bs]]`:
//!   `dst[1][c] = AVG3(B[c], B[c+1], B[c+2])`.
//! - **Column 0, rows 2..bs** = `AVG3` over `C = [al, left[0..bs-1]]`:
//!   `dst[r][0] = AVG3(C[r-2], C[r-1], C[r])`.
//! - **Interior**: `dst[r][c] = dst[r-2][c-1]` for `r ≥ 2, c ≥ 1`.
//!
//! The recurrence is exactly "row r (cols 1..bs) = row r-2 (cols
//! 0..bs-1)" — a right-shift-by-one copy with `col0[r]` as the new
//! head. It is pure data movement (no arithmetic), so it stays a
//! `copy_within` (bit-identical to scalar, autovectorized by the
//! compiler); the SIMD win is in the three AVG edge arrays.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg2_window, avg3_window, store_row};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_insert_epi8, _mm_loadu_si128, _mm_slli_si128, _mm_storeu_si128,
    _mm256_alignr_epi8, _mm256_insert_epi8, _mm256_loadu_si256, _mm256_permute2x128_si256,
    _mm256_storeu_si256,
};

/// AVX2 D117 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `above.len() >= bs`, `left.len() >= bs`;
/// `dst.len() >= (bs-1)*stride + bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d117_predictor_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    al: u8,
) {
    debug_assert!(above.len() >= bs && left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);

    // Edge seed arrays (zero-padded so wide SIMD loads stay in-bounds).
    // A = [al, above[0..bs]]            → row 0 via AVG2
    // B = [left[0], al, above[0..bs]]   → row 1 via AVG3
    // C = [al, left[0..bs-1]]           → col 0 via AVG3
    let mut sa = [0u8; 66];
    let mut sb = [0u8; 66];
    let mut sc = [0u8; 66];
    sa[0] = al;
    sa[1..1 + bs].copy_from_slice(&above[..bs]);
    sb[0] = left[0];
    sb[1] = al;
    sb[2..2 + bs].copy_from_slice(&above[..bs]);
    sc[0] = al;
    sc[1..bs].copy_from_slice(&left[..bs - 1]);

    let mut row0 = [0u8; 64];
    let mut row1 = [0u8; 64];
    let mut col0 = [0u8; 64];
    avg2_window(&sa, &mut row0); // row0[c] = AVG2(A[c], A[c+1])
    avg3_window(&sb, &mut row1); // row1[c] = AVG3(B[c], B[c+1], B[c+2])
    avg3_window(&sc, &mut col0); // col0[j] = AVG3(C[j], C[j+1], C[j+2]) (row 2+j)

    store_row(&mut dst[0..], &row0, bs);
    store_row(&mut dst[stride..], &row1, bs);

    // L5: SIMD shift-copy recurrence. Row r (r≥2) cols 1..bs = row r-2
    // cols 0..bs-1 (shift right by one byte) with `col0[r-2]` as the new
    // head — an r-2 dependency, so two interleaved chains (even rows seed
    // from row0, odd from row1). Each step is shift-by-1 + head insert,
    // folding the per-row col0 strided write into the same pass. No
    // read-back from `dst`; bounded `store_row` write. Bit-exact with the
    // copy_within form.
    if bs <= 16 {
        let mut tmp = [0u8; 16];
        let mut cur_even = _mm_loadu_si128(row0.as_ptr() as *const __m128i);
        let mut cur_odd = _mm_loadu_si128(row1.as_ptr() as *const __m128i);
        for r in 2..bs {
            let base = if r & 1 == 0 { cur_even } else { cur_odd };
            let mut cur = _mm_slli_si128(base, 1);
            cur = _mm_insert_epi8(cur, i32::from(col0[r - 2]), 0);
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, cur);
            store_row(&mut dst[r * stride..], &tmp, bs);
            if r & 1 == 0 {
                cur_even = cur;
            } else {
                cur_odd = cur;
            }
        }
    } else {
        // bs == 32: full-width 256-bit shift-left-by-1 via
        // `alignr(v, permute2x128(v,v,0x08), 15)` (carries the cross-lane
        // byte that `slli_si256` would drop).
        let mut tmp = [0u8; 32];
        let mut cur_even = _mm256_loadu_si256(row0.as_ptr() as *const __m256i);
        let mut cur_odd = _mm256_loadu_si256(row1.as_ptr() as *const __m256i);
        for r in 2..bs {
            let base = if r & 1 == 0 { cur_even } else { cur_odd };
            let mut cur = _mm256_alignr_epi8(base, _mm256_permute2x128_si256(base, base, 0x08), 15);
            cur = _mm256_insert_epi8(cur, col0[r - 2] as i8, 0);
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, cur);
            store_row(&mut dst[r * stride..], &tmp, bs);
            if r & 1 == 0 {
                cur_even = cur;
            } else {
                cur_odd = cur;
            }
        }
    }
}

/// Safe wrapper for the AVX2 D117 predictor. Signature mirrors the
/// flattened scalar [`crate::intra_pred::d117_predictor`] (`PredCtx`
/// reduced to `above`, `left`, `above_left`).
pub fn d117_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len().min(left.len()))
        .unwrap_or_else(|reason| panic!("d117_predictor_avx2: {reason}"));
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and monotonic for the process lifetime. Tests gate every call on
    // a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d117_predictor_avx2_inner(dst, stride, bs, above, left, above_left) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d117_predictor_avx2`.
    use super::d117_predictor_avx2;
    use crate::intra_pred::{PredCtx, d117_predictor};

    #[test]
    #[should_panic(expected = "edge slice too short")]
    fn rejects_short_edge_before_avx2_inner() {
        let mut dst = [0u8; 16];
        let above = [0u8; 4];
        let left = [0u8; 3];
        d117_predictor_avx2(&mut dst, 4, 4, &above, &left, 0);
    }

    #[test]
    #[should_panic(expected = "unsupported block size")]
    fn rejects_invalid_block_size_before_avx2_inner() {
        d117_predictor_avx2(&mut [], 4, 3, &[], &[], 0);
    }

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

    #[test]
    fn differential_all_sizes() {
        require_avx2!();
        let mut state: u32 = 0x117_0000;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..512 {
                let mut above = vec![0u8; bs];
                let mut left = vec![0u8; bs];
                for s in above.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                for s in left.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let al = (lcg_next(&mut state) & 0xFF) as u8;
                let stride = bs + (lcg_next(&mut state) as usize & 0x7);
                let dst_len = (bs - 1) * stride + bs;
                let mut d_scalar = vec![0u8; dst_len];
                let mut d_avx2 = vec![0u8; dst_len];
                let ctx = PredCtx {
                    above: &above,
                    left: &left,
                    above_left: al,
                };
                d117_predictor(&mut d_scalar, stride, bs, ctx);
                d117_predictor_avx2(&mut d_avx2, stride, bs, &above, &left, al);
                assert_eq!(
                    d_scalar, d_avx2,
                    "mismatch bs={bs} above={above:?} left={left:?} al={al}"
                );
            }
        }
    }

    #[test]
    fn constant_input() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let above = vec![91u8; bs];
            let left = vec![91u8; bs];
            let mut d = vec![0u8; bs * bs];
            d117_predictor_avx2(&mut d, bs, bs, &above, &left, 91);
            assert!(d.iter().all(|&v| v == 91), "bs={bs} not constant");
        }
    }
}
