//! Shared AVX2 primitives for the directional intra-predictors
//! (Wave K). The six VP9 directional modes (D45, D63, D117, D135,
//! D153, D207) all reduce to the same two edge-filter operations,
//! `AVG3` and `AVG2`, applied along the `above` row / `left` column /
//! a combined border, followed by index-based window/shift fills.
//!
//! ## The AVG3 / AVG2 lane primitives
//!
//! The scalar reference (`crate::intra_pred`) defines:
//! - `avg3(a,b,c) = (a + 2b + c + 2) >> 2`
//! - `avg2(a,b)   = (a + b + 1) >> 1`
//!
//! `AVG2` is exactly `_mm_avg_epu8` / `_mm256_avg_epu8`
//! (`(a+b+1)>>1`). `AVG3` uses libvpx's "trick from pascal"
//! (`vpx_dsp/x86/intrapred_ssse3.asm:165-177`), which computes the
//! exact 3-tap rounded average with two `pavgb` + a parity
//! correction that cancels the double-rounding bias:
//!
//! ```text
//! r = avg(x, z)            ; (x+z+1)>>1
//! r -= (x ^ z) & 1         ; → floor((x+z)/2)
//! r = avg(r, y)            ; (floor((x+z)/2) + y + 1) >> 1
//! ```
//!
//! This equals `(x + 2y + z + 2) >> 2` for all byte inputs (verified
//! by case analysis on the parity of `x+z`), so the AVX2 kernels are
//! bit-exact against the scalar `avg3`.
//!
//! ## Bit-exactness strategy
//!
//! Wave K kernels copy their `above` / `left` edges into a
//! zero-padded stack scratch (so wide unaligned SIMD loads never
//! read past the caller's slice), compute the AVG3/AVG2 edge arrays
//! with these primitives, then perform the **same index-based fills
//! as the scalar reference** (pure data movement, no arithmetic).
//! Only the averaging arithmetic moves to SIMD; the structural fills
//! are identical, so the output is bit-exact by construction.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_avg_epu8, _mm256_set1_epi8, _mm256_sub_epi8,
    _mm256_xor_si256,
};

/// `AVG3(x, y, z) = (x + 2y + z + 2) >> 2`, per 8-bit lane, 256-bit.
///
/// # Safety
/// Requires AVX2.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avg3_epu8_256(x: __m256i, y: __m256i, z: __m256i) -> __m256i {
    let r = _mm256_avg_epu8(x, z);
    let corr = _mm256_and_si256(_mm256_xor_si256(x, z), _mm256_set1_epi8(1));
    let r = _mm256_sub_epi8(r, corr);
    _mm256_avg_epu8(r, y)
}

/// Compute `out[k] = AVG3(in[k], in[k+1], in[k+2])` for `k in 0..n`,
/// where `n <= 32`. `src` must hold at least `n + 2` valid bytes
/// (callers pass a zero-padded `[u8; 64]` scratch). Writes exactly the
/// first `n` bytes of `out` (which must be `>= n` long).
///
/// Single 256-bit pass — produces up to 32 lanes; the caller uses the
/// first `n`. Reading `src[2..34]` stays in-bounds for a 64-byte
/// scratch.
///
/// # Safety
/// Requires AVX2. `src.len() >= 34` and `out.len() >= 32`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avg3_window(src: &[u8], out: &mut [u8]) {
    use core::arch::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};
    debug_assert!(src.len() >= 34);
    debug_assert!(out.len() >= 32);
    // G4: loads from `src` use _mm256_loadu_si256 (unaligned)
    let x = _mm256_loadu_si256(src.as_ptr() as *const _);
    let y = _mm256_loadu_si256(src.as_ptr().add(1) as *const _);
    let z = _mm256_loadu_si256(src.as_ptr().add(2) as *const _);
    let f = avg3_epu8_256(x, y, z);
    // G4: stores to `out` use _mm256_storeu_si256 (unaligned)
    _mm256_storeu_si256(out.as_mut_ptr() as *mut _, f);
}

/// Store the `bs`-byte window `src[0..bs]` to `dst[0..bs]` using the
/// widest unaligned store that fits `bs ∈ {4, 8, 16, 32}`.
///
/// # Safety
/// Requires AVX2. `src.len() >= bs`, `dst.len() >= bs`,
/// `bs ∈ {4, 8, 16, 32}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn store_row(dst: &mut [u8], src: &[u8], bs: usize) {
    use core::arch::x86_64::{
        _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadl_epi64, _mm_loadu_si128,
        _mm_storel_epi64, _mm_storeu_si128,
    };
    debug_assert!(src.len() >= bs && dst.len() >= bs);
    match bs {
        4 => {
            // G4: loads from `src` use read_unaligned::<u32> (unaligned)
            let v = core::ptr::read_unaligned(src.as_ptr() as *const u32);
            // G4: stores to `dst` use write_unaligned::<u32> (unaligned)
            core::ptr::write_unaligned(dst.as_mut_ptr() as *mut u32, v);
        }
        8 => {
            // G4: loads from `src` use _mm_loadl_epi64 (unaligned)
            let v = _mm_loadl_epi64(src.as_ptr() as *const _);
            // G4: stores to `dst` use _mm_storel_epi64 (unaligned)
            _mm_storel_epi64(dst.as_mut_ptr() as *mut _, v);
        }
        16 => {
            // G4: loads from `src` use _mm_loadu_si128 (unaligned)
            let v = _mm_loadu_si128(src.as_ptr() as *const _);
            // G4: stores to `dst` use _mm_storeu_si128 (unaligned)
            _mm_storeu_si128(dst.as_mut_ptr() as *mut _, v);
        }
        _ => {
            // bs == 32
            // G4: loads from `src` use _mm256_loadu_si256 (unaligned)
            let v = _mm256_loadu_si256(src.as_ptr() as *const _);
            // G4: stores to `dst` use _mm256_storeu_si256 (unaligned)
            _mm256_storeu_si256(dst.as_mut_ptr() as *mut _, v);
        }
    }
}
