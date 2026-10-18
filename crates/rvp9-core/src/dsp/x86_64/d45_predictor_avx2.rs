//! D45_PRED directional intra-predictor — x86_64 AVX2 (Wave K K1).
//!
//! Scalar reference: [`crate::intra_pred::d45_predictor`] (generic,
//! `bs ∈ {8,16,32}`) and [`crate::intra_pred::d45_predictor_4x4`]
//! (the libvpx `intra_pred_no_4x4` specialization). This AVX2 kernel
//! reproduces **both**, dispatching on `bs == 4`, and is bit-exact
//! against the scalar reference (the differential test is the oracle).
//!
//! ## Generic algorithm (bs ∈ {8,16,32})
//!
//! The scalar computes a filtered row `F[x] = AVG3(above[x],
//! above[x+1], above[x+2])` for `x in 0..bs-1`, sets `dst[0][bs-1] =
//! above[bs-1]` (the "above-right" pad), and each subsequent row is
//! row 0 shifted left by one with the trailing cells re-padded.
//!
//! Equivalently: build an array `g[0..2*bs]` where `g[0..bs-1] = F`
//! and `g[bs-1..2*bs] = above[bs-1]`, then **row r = g[r..r+bs]** — a
//! sliding window. (Derivation: scalar row r col c is `F[r+c]` when
//! `r+c < bs-1`, else `above[bs-1]`; `g[r+c]` yields exactly that.)
//!
//! So the kernel is: one SIMD AVG3 pass to build `F`, a pad fill, then
//! `bs` windowed stores.
//!
//! ## 4×4 specialization
//!
//! The scalar 4×4 uses the genuine diagonal `above[4..8]` rather than
//! padding with `above[3]`. Define `D[k] = AVG3(above[k..k+3])`; then
//! row r = `D[r..r+4]` for r in 0..4, **except** `dst[3][3] =
//! above[7]` (the scalar's explicit corner). One SIMD AVG3 pass over a
//! zero-padded copy of `above[0..8]` builds `D`, then four windowed
//! `u32` stores plus the corner fixup.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg3_window, store_row};

/// AVX2 D45 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `above.len() >= 2*bs`; `dst.len() >= (bs-1)*stride +
/// bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d45_predictor_avx2_inner(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= 2 * bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);

    if bs == 4 {
        // 4×4 specialization. Zero-padded copy so the offset-2 SIMD
        // load stays in-bounds (above has exactly 8 valid bytes).
        let mut sa = [0u8; 64];
        sa[..8].copy_from_slice(&above[..8]);
        let mut d = [0u8; 32];
        avg3_window(&sa, &mut d); // d[k] = AVG3(above[k..k+3])
        for r in 0..4 {
            store_row(&mut dst[r * stride..], &d[r..], 4);
        }
        dst[3 * stride + 3] = above[7];
        return;
    }

    // Generic bs ∈ {8,16,32}. Build g[0..2*bs].
    let mut sa = [0u8; 64];
    sa[..2 * bs].copy_from_slice(&above[..2 * bs]);
    let mut g = [0u8; 64];
    avg3_window(&sa, &mut g); // g[0..32] = AVG3 lanes; g[0..bs-1] is F
    // Pad g[bs-1..2*bs] with above-right (overwrites the unused F lanes).
    let above_right = above[bs - 1];
    for slot in g.iter_mut().take(2 * bs).skip(bs - 1) {
        *slot = above_right;
    }
    // Sliding-window stores: row r = g[r..r+bs].
    for r in 0..bs {
        store_row(&mut dst[r * stride..], &g[r..], bs);
    }
}

/// Safe wrapper for the AVX2 D45 predictor. Signature mirrors the
/// flattened scalar [`crate::intra_pred::d45_predictor`] /
/// `d45_predictor_4x4` (`above`-only). The `SimdDispatch` slot points
/// here on AVX2 hosts and at the scalar adapter elsewhere.
pub fn d45_predictor_avx2(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len())
        .unwrap_or_else(|reason| panic!("d45_predictor_avx2: {reason}"));
    let extended_edge_len = bs
        .checked_mul(2)
        .expect("d45_predictor_avx2: extended edge length overflow");
    assert!(
        above.len() >= extended_edge_len,
        "d45_predictor_avx2: extended above edge too short"
    );
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and are monotonic for the process lifetime. Tests gate every call
    // on a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d45_predictor_avx2_inner(dst, stride, bs, above) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d45_predictor_avx2`.
    use super::d45_predictor_avx2;
    use crate::intra_pred::{d45_predictor, d45_predictor_4x4};

    #[test]
    #[should_panic(expected = "extended above edge too short")]
    fn rejects_short_extended_edge_before_avx2_inner() {
        let mut dst = [0u8; 16];
        let above = [0u8; 4];
        d45_predictor_avx2(&mut dst, 4, 4, &above);
    }

    #[test]
    #[should_panic(expected = "unsupported block size")]
    fn rejects_invalid_block_size_before_avx2_inner() {
        d45_predictor_avx2(&mut [], 4, 3, &[]);
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

    /// Differential vs scalar across all four sizes with randomized
    /// `above` rows (2*bs bytes), random dst stride. Bit-exact.
    #[test]
    fn differential_all_sizes() {
        require_avx2!();
        let mut state: u32 = 0xD45_0000;
        for bs in [4usize, 8, 16, 32] {
            for _iter in 0..512 {
                let mut above = vec![0u8; 2 * bs];
                for s in above.iter_mut() {
                    *s = (lcg_next(&mut state) & 0xFF) as u8;
                }
                let stride = bs + (lcg_next(&mut state) as usize & 0x7);
                let dst_len = (bs - 1) * stride + bs;
                let mut d_scalar = vec![0u8; dst_len];
                let mut d_avx2 = vec![0u8; dst_len];
                if bs == 4 {
                    d45_predictor_4x4(&mut d_scalar, stride, &above);
                } else {
                    d45_predictor(&mut d_scalar, stride, bs, &above);
                }
                d45_predictor_avx2(&mut d_avx2, stride, bs, &above);
                assert_eq!(d_scalar, d_avx2, "mismatch bs={bs} above={above:?}");
            }
        }
    }

    /// Constant input → constant AVG3/AVG2 output (sanity / tautology
    /// guard that the AVX2 path is actually exercised on this host).
    #[test]
    fn constant_input() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let above = vec![123u8; 2 * bs];
            let mut d = vec![0u8; bs * bs];
            d45_predictor_avx2(&mut d, bs, bs, &above);
            assert!(d.iter().all(|&v| v == 123), "bs={bs} not constant");
        }
    }
}
