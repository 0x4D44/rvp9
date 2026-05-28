//! D153_PRED directional intra-predictor — x86_64 AVX2 (Wave K K6).
//!
//! Scalar reference: [`crate::intra_pred::d153_predictor`] (generic
//! for all `bs ∈ {4,8,16,32}` — no 4×4 special). Bit-exact against
//! the scalar reference (differential test is the oracle).
//!
//! ## Algorithm (uses `above`, `left`, `above_left`)
//!
//! Mirror of D117 but shifted by two columns. Three seed edges:
//! - **Column 0** (all rows) = `AVG2` over `L0 = [al, left[0..bs]]`:
//!   `dst[r][0] = AVG2(L0[r], L0[r+1])`.
//! - **Column 1** (all rows) = `AVG3` over
//!   `L1 = [above[0], al, left[0..bs]]`:
//!   `dst[r][1] = AVG3(L1[r], L1[r+1], L1[r+2])`. (Uses AVG3's
//!   symmetry in its first/third operands to fold the scalar's
//!   `AVG3(left[0], al, above[0])` top cell into this array form.)
//! - **Row 0, cols 2..bs** = `AVG3` over `Atop = [al, above[0..bs-1]]`:
//!   `dst[0][c] = AVG3(Atop[c-2], Atop[c-1], Atop[c])`.
//! - **Interior**: `dst[r][c] = dst[r-1][c-2]` for `r ≥ 1, c ≥ 2` —
//!   "row r cols 2..bs = row r-1 cols 0..bs-2", a right-shift-by-two
//!   copy with `col0[r]`/`col1[r]` as the two new heads. Pure data
//!   movement, kept as bit-exact `copy_within`; SIMD win is the AVG
//!   edge arrays.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg2_window, avg3_window, store_row};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_insert_epi8, _mm_loadu_si128, _mm_slli_si128, _mm_storeu_si128,
    _mm256_alignr_epi8, _mm256_insert_epi8, _mm256_loadu_si256, _mm256_permute2x128_si256,
    _mm256_storeu_si256,
};

/// AVX2 D153 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `above.len() >= bs`, `left.len() >= bs`;
/// `dst.len() >= (bs-1)*stride + bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d153_predictor_avx2_inner(
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
    // L0   = [al, left[0..bs]]            → col 0 via AVG2
    // L1   = [above[0], al, left[0..bs]]  → col 1 via AVG3
    // Atop = [al, above[0..bs-1]]         → row 0 tail via AVG3
    let mut sl0 = [0u8; 66];
    let mut sl1 = [0u8; 66];
    let mut sat = [0u8; 66];
    sl0[0] = al;
    sl0[1..1 + bs].copy_from_slice(&left[..bs]);
    sl1[0] = above[0];
    sl1[1] = al;
    sl1[2..2 + bs].copy_from_slice(&left[..bs]);
    sat[0] = al;
    sat[1..bs].copy_from_slice(&above[..bs - 1]);

    let mut col0 = [0u8; 64];
    let mut col1 = [0u8; 64];
    let mut rtop = [0u8; 64];
    avg2_window(&sl0, &mut col0); // col0[r] = AVG2(L0[r], L0[r+1])
    avg3_window(&sl1, &mut col1); // col1[r] = AVG3(L1[r], L1[r+1], L1[r+2])
    avg3_window(&sat, &mut rtop); // rtop[j] = AVG3(Atop[j], Atop[j+1], Atop[j+2]); col 2+j of row 0

    if bs <= 16 {
        // L5: SIMD shift-copy recurrence in a single 128-bit lane. Each
        // row is the previous row shifted right by two bytes, with
        // `col0[r]`/`col1[r]` inserted as the two new heads. No read-back
        // from `dst` (avoids over-read); bounded `store_row` write (avoids
        // over-store). Folds the per-row col0/col1 strided byte writes
        // into the same pass. Bit-exact with the copy_within form.
        let mut tmp = [0u8; 16];
        let r0 = _mm_loadu_si128(rtop.as_ptr() as *const __m128i);
        let mut cur = _mm_slli_si128(r0, 2);
        cur = _mm_insert_epi8(cur, i32::from(col0[0]), 0);
        cur = _mm_insert_epi8(cur, i32::from(col1[0]), 1);
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, cur);
        store_row(dst, &tmp, bs);
        for r in 1..bs {
            cur = _mm_slli_si128(cur, 2);
            cur = _mm_insert_epi8(cur, i32::from(col0[r]), 0);
            cur = _mm_insert_epi8(cur, i32::from(col1[r]), 1);
            _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, cur);
            store_row(&mut dst[r * stride..], &tmp, bs);
        }
    } else {
        // bs == 32: same recurrence with a full-width 256-bit byte shift.
        // `slli_si256` only shifts within each 128-bit lane, so the 2-byte
        // carry across the lane boundary is done with
        // `alignr(v, permute2x128(v,v,0x08), 14)` (the standard full-width
        // left-shift-by-2 idiom). Heads inserted via `insert_epi8`.
        let mut tmp = [0u8; 32];
        let r0 = _mm256_loadu_si256(rtop.as_ptr() as *const __m256i);
        let mut cur = _mm256_alignr_epi8(r0, _mm256_permute2x128_si256(r0, r0, 0x08), 14);
        cur = _mm256_insert_epi8(cur, col0[0] as i8, 0);
        cur = _mm256_insert_epi8(cur, col1[0] as i8, 1);
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, cur);
        store_row(dst, &tmp, bs);
        for r in 1..bs {
            cur = _mm256_alignr_epi8(cur, _mm256_permute2x128_si256(cur, cur, 0x08), 14);
            cur = _mm256_insert_epi8(cur, col0[r] as i8, 0);
            cur = _mm256_insert_epi8(cur, col1[r] as i8, 1);
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, cur);
            store_row(&mut dst[r * stride..], &tmp, bs);
        }
    }
}

/// Safe wrapper for the AVX2 D153 predictor. Signature mirrors the
/// flattened scalar [`crate::intra_pred::d153_predictor`] (`PredCtx`
/// reduced to `above`, `left`, `above_left`).
pub fn d153_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and monotonic for the process lifetime. Tests gate every call on
    // a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d153_predictor_avx2_inner(dst, stride, bs, above, left, above_left) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d153_predictor_avx2`.
    use super::d153_predictor_avx2;
    use crate::intra_pred::{PredCtx, d153_predictor};

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
        let mut state: u32 = 0x153_0000;
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
                d153_predictor(&mut d_scalar, stride, bs, ctx);
                d153_predictor_avx2(&mut d_avx2, stride, bs, &above, &left, al);
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
            let above = vec![44u8; bs];
            let left = vec![44u8; bs];
            let mut d = vec![0u8; bs * bs];
            d153_predictor_avx2(&mut d, bs, bs, &above, &left, 44);
            assert!(d.iter().all(|&v| v == 44), "bs={bs} not constant");
        }
    }
}
