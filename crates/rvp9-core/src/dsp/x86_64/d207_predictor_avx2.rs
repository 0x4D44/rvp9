//! D207_PRED directional intra-predictor — x86_64 AVX2 (Wave K K2).
//!
//! Scalar reference: [`crate::intra_pred::d207_predictor`] (generic)
//! and [`crate::intra_pred::d207_predictor_4x4`]. Unlike D45/D63, the
//! D207 generic construction reproduces the 4×4 specialization
//! **exactly** (the scalar's own comment notes the two coincide), so
//! this AVX2 kernel uses one uniform path for every `bs ∈ {4,8,16,32}`.
//!
//! ## Algorithm (uses only `left`)
//!
//! The scalar builds column 0 = `AVG2` along `left`, column 1 = `AVG3`
//! along `left`, then fills cols ≥ 2 via the recurrence
//! `dst[r][c] = dst[r+1][c-2]`. Unfolding the recurrence:
//! - even col `c=2j`: `dst[r][c] = col0[r+j]`
//! - odd  col `c=2j+1`: `dst[r][c] = col1[r+j]`
//!
//! So **row r is the byte-interleave** of windows `col0[r..]` and
//! `col1[r..]`: `dst[r] = [col0[r], col1[r], col0[r+1], col1[r+1], …]`.
//!
//! Boundary trick: with `lext[i] = left[i]` for `i < bs` and
//! `lext[i] = left[bs-1]` for `i ≥ bs` (last-sample replication), the
//! scalar's boundary cases collapse exactly:
//! - `col0[i] = AVG2(lext[i], lext[i+1])` for all `i` (at `i=bs-1`,
//!   `AVG2(left[bs-1], left[bs-1]) = left[bs-1]`, matching scalar).
//! - `col1[i] = AVG3(lext[i], lext[i+1], lext[i+2])` for all `i`.
//!
//! So the kernel: build `lext`, compute `col0`/`col1` with SIMD
//! `AVG2`/`AVG3`, then store each row via a `punpcklbw` interleave.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg2_window, avg3_window};

/// Interleave-store: `dst[0..bs] = [c0[0], c1[0], c0[1], c1[1], …]`,
/// reading `bs/2` bytes from each of `c0`, `c1`.
///
/// # Safety
/// Requires AVX2. `c0.len() >= bs/2 + 8`, `c1.len() >= bs/2 + 8`
/// (we issue 128-bit loads), `dst.len() >= bs`, `bs ∈ {4,8,16,32}`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn interleave_row(dst: &mut [u8], c0: &[u8], c1: &[u8], bs: usize) {
    use core::arch::x86_64::{
        _mm_cvtsi128_si32, _mm_loadu_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_unpacklo_epi8,
    };
    // G4: loads from `c0`/`c1` use _mm_loadu_si128 (unaligned)
    let a = _mm_loadu_si128(c0.as_ptr() as *const _);
    let b = _mm_loadu_si128(c1.as_ptr() as *const _);
    let lo = _mm_unpacklo_epi8(a, b); // [c0[0],c1[0],c0[1],c1[1],…] (16 bytes)
    match bs {
        4 => {
            let v = _mm_cvtsi128_si32(lo) as u32;
            // G4: stores to `dst` use write_unaligned::<u32> (unaligned)
            core::ptr::write_unaligned(dst.as_mut_ptr() as *mut u32, v);
        }
        8 => {
            // G4: stores to `dst` use _mm_storel_epi64 (unaligned)
            _mm_storel_epi64(dst.as_mut_ptr() as *mut _, lo);
        }
        16 => {
            // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
            _mm_storeu_si128(dst.as_mut_ptr() as *mut _, lo);
        }
        _ => {
            // bs == 32: low 16 cols from c0/c1[..8], high 16 from [8..].
            // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
            _mm_storeu_si128(dst.as_mut_ptr() as *mut _, lo);
            let a2 = _mm_loadu_si128(c0.as_ptr().add(8) as *const _);
            let b2 = _mm_loadu_si128(c1.as_ptr().add(8) as *const _);
            let hi = _mm_unpacklo_epi8(a2, b2);
            // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
            _mm_storeu_si128(dst.as_mut_ptr().add(16) as *mut _, hi);
        }
    }
}

/// AVX2 D207 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `left.len() >= bs`; `dst.len() >= (bs-1)*stride +
/// bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d207_predictor_avx2_inner(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    debug_assert!(left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);

    // lext: left with the last sample replicated out to 2*bs+2 ≤ 66.
    let mut lext = [0u8; 66];
    lext[..bs].copy_from_slice(&left[..bs]);
    let last = left[bs - 1];
    for slot in lext.iter_mut().take(2 * bs + 2).skip(bs) {
        *slot = last;
    }

    // col0[i] = AVG2(lext[i], lext[i+1]); col1[i] = AVG3(lext[i..i+3]).
    // We need indices up to (bs-1) + (bs/2 - 1) < 2*bs; build 2*bs lanes
    // (one 256-bit pass per 32 lanes — two passes only for bs == 32).
    let mut col0 = [0u8; 64];
    let mut col1 = [0u8; 64];
    let mut off = 0;
    while off < 2 * bs {
        avg2_window(&lext[off..], &mut col0[off..]);
        avg3_window(&lext[off..], &mut col1[off..]);
        off += 32;
    }

    for r in 0..bs {
        interleave_row(&mut dst[r * stride..], &col0[r..], &col1[r..], bs);
    }
}

/// Safe wrapper for the AVX2 D207 predictor (`left`-only). Mirrors the
/// flattened scalar [`crate::intra_pred::d207_predictor`].
pub fn d207_predictor_avx2(dst: &mut [u8], stride: usize, bs: usize, left: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and monotonic for the process lifetime. Tests gate every call on
    // a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d207_predictor_avx2_inner(dst, stride, bs, left) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d207_predictor_avx2`.
    use super::d207_predictor_avx2;
    use crate::intra_pred::{d207_predictor, d207_predictor_4x4};

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
        let mut state: u32 = 0x207_0000;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..512 {
                let mut left = vec![0u8; bs];
                for s in left.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let stride = bs + (lcg_next(&mut state) as usize & 0x7);
                let dst_len = (bs - 1) * stride + bs;
                let mut d_scalar = vec![0u8; dst_len];
                let mut d_avx2 = vec![0u8; dst_len];
                if bs == 4 {
                    d207_predictor_4x4(&mut d_scalar, stride, &left);
                } else {
                    d207_predictor(&mut d_scalar, stride, bs, &left);
                }
                d207_predictor_avx2(&mut d_avx2, stride, bs, &left);
                assert_eq!(d_scalar, d_avx2, "mismatch bs={bs} left={left:?}");
            }
        }
    }

    #[test]
    fn constant_input() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let left = vec![77u8; bs];
            let mut d = vec![0u8; bs * bs];
            d207_predictor_avx2(&mut d, bs, bs, &left);
            assert!(d.iter().all(|&v| v == 77), "bs={bs} not constant");
        }
    }
}
