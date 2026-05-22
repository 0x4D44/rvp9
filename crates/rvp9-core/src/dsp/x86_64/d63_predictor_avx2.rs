//! D63_PRED directional intra-predictor — x86_64 AVX2 (Wave K K3).
//!
//! Scalar reference: [`crate::intra_pred::d63_predictor`] (generic,
//! `bs ∈ {8,16,32}`) and [`crate::intra_pred::d63_predictor_4x4`].
//! Bit-exact against the scalar reference (differential test is the
//! oracle).
//!
//! ## Algorithm (uses only `above`)
//!
//! Two seed rows are filtered from `above`:
//! - `R0[c] = AVG2(above[c], above[c+1])`  (row 0)
//! - `R1[c] = AVG3(above[c], above[c+1], above[c+2])` (row 1)
//!
//! The scalar then fills rows ≥ 2 by shifted copies: even row `r`
//! copies `R0` from offset `r/2`, odd row `r` copies `R1` from offset
//! `(r-1)/2`, padding the right with `above[bs-1]`.
//!
//! Equivalently (generic): rows 0/1 are the full `R0[0..bs]` /
//! `R1[0..bs]`; for rows ≥ 2 use padded edge arrays `R0e`/`R1e` where
//! index `≥ bs-1` is `above[bs-1]`, and even row `r = R0e[r/2 ..]`,
//! odd row `r = R1e[(r-1)/2 ..]`. (The pad starts at `bs-1`, not `bs`,
//! because the scalar's right-fill begins one cell earlier than row
//! 0's last genuine AVG2 cell.)
//!
//! ## 4×4 specialization
//!
//! The scalar 4×4 (libvpx `intra_pred_no_4x4`) does **not** pad — it
//! uses the genuine extended diagonal: rows are `R0[0..4]`, `R1[0..4]`,
//! `R0[1..5]`, `R1[1..5]`. So bs == 4 takes a distinct path.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg2_window, avg3_window, store_row};

/// AVX2 D63 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `above.len() >= 2*bs`; `dst.len() >= (bs-1)*stride +
/// bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d63_predictor_avx2_inner(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    debug_assert!(above.len() >= 2 * bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);

    // Zero-padded copy of `above` so wide SIMD loads stay in-bounds.
    let mut sa = [0u8; 66];
    sa[..2 * bs].copy_from_slice(&above[..2 * bs]);

    // Build R0 (AVG2) and R1 (AVG3) edge arrays. Need indices up to
    // ~3*bs/2 < 2*bs, so build 2*bs lanes (two 256-bit passes for
    // bs == 32, one otherwise).
    let mut r0 = [0u8; 64];
    let mut r1 = [0u8; 64];
    let mut off = 0;
    while off < 2 * bs {
        avg2_window(&sa[off..], &mut r0[off..]);
        avg3_window(&sa[off..], &mut r1[off..]);
        off += 32;
    }

    if bs == 4 {
        // No padding: genuine extended diagonal.
        store_row(&mut dst[0..], &r0[0..], 4); // row 0 = R0[0..4]
        store_row(&mut dst[stride..], &r1[0..], 4); // row 1 = R1[0..4]
        store_row(&mut dst[2 * stride..], &r0[1..], 4); // row 2 = R0[1..5]
        store_row(&mut dst[3 * stride..], &r1[1..], 4); // row 3 = R1[1..5]
        return;
    }

    // Generic: rows 0 and 1 are the full (un-padded) seed rows.
    store_row(&mut dst[0..], &r0[0..], bs);
    store_row(&mut dst[stride..], &r1[0..], bs);

    // Convert r0/r1 into the padded edge arrays R0e/R1e in place:
    // index ≥ bs-1 becomes above[bs-1].
    let pad = above[bs - 1];
    for slot in r0.iter_mut().take(2 * bs).skip(bs - 1) {
        *slot = pad;
    }
    for slot in r1.iter_mut().take(2 * bs).skip(bs - 1) {
        *slot = pad;
    }

    // Rows ≥ 2: even → R0e[r/2..], odd → R1e[(r-1)/2..].
    for r in 2..bs {
        if r & 1 == 0 {
            store_row(&mut dst[r * stride..], &r0[r / 2..], bs);
        } else {
            store_row(&mut dst[r * stride..], &r1[(r - 1) / 2..], bs);
        }
    }
}

/// Safe wrapper for the AVX2 D63 predictor (`above`-only). Mirrors the
/// flattened scalar [`crate::intra_pred::d63_predictor`].
pub fn d63_predictor_avx2(dst: &mut [u8], stride: usize, bs: usize, above: &[u8]) {
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and monotonic for the process lifetime. Tests gate every call on
    // a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d63_predictor_avx2_inner(dst, stride, bs, above) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d63_predictor_avx2`.
    use super::d63_predictor_avx2;
    use crate::intra_pred::{d63_predictor, d63_predictor_4x4};

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
        let mut state: u32 = 0x63_0000;
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
                    d63_predictor_4x4(&mut d_scalar, stride, &above);
                } else {
                    d63_predictor(&mut d_scalar, stride, bs, &above);
                }
                d63_predictor_avx2(&mut d_avx2, stride, bs, &above);
                assert_eq!(d_scalar, d_avx2, "mismatch bs={bs} above={above:?}");
            }
        }
    }

    #[test]
    fn constant_input() {
        require_avx2!();
        for bs in [4usize, 8, 16, 32] {
            let above = vec![200u8; 2 * bs];
            let mut d = vec![0u8; bs * bs];
            d63_predictor_avx2(&mut d, bs, bs, &above);
            assert!(d.iter().all(|&v| v == 200), "bs={bs} not constant");
        }
    }
}
