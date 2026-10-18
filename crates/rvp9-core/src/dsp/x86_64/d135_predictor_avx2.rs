//! D135_PRED directional intra-predictor — x86_64 AVX2 (Wave K K4).
//!
//! Scalar reference: [`crate::intra_pred::d135_predictor`] (generic
//! for all `bs ∈ {4,8,16,32}` — no 4×4 special). Bit-exact against
//! the scalar reference (differential test is the oracle).
//!
//! ## Algorithm (uses `above`, `left`, `above_left`)
//!
//! Builds a length-`2*bs-1` `border` array running bottom-left →
//! `above_left` → top-right, all via `AVG3`; each output row is a
//! `bs`-byte window into `border`, shifting one cell left per row down:
//! `row i = border[bs-1-i .. bs-1-i+bs]`.
//!
//! The border has three regions:
//! - `border[0..bs-2]` — `AVG3` along `left`, indexed **bottom-up**:
//!   `border[i] = AVG3(left[bs-3-i], left[bs-2-i], left[bs-1-i])`.
//!   Using AVG3's symmetry in its first/third operands, this equals
//!   `AVG3` over a **reversed-left** array `Lr[k] = left[bs-1-k]`:
//!   `border[k] = AVG3(Lr[k], Lr[k+1], Lr[k+2])`.
//! - `border[bs-2..=bs]` — three junction cells around `above_left`
//!   (computed with scalar `avg3`, 3 bytes).
//! - `border[bs+1..]` — `AVG3` along `above`, ascending:
//!   `border[bs+1+i] = AVG3(above[i], above[i+1], above[i+2])`.
//!
//! So two SIMD `AVG3` passes (reversed-left, forward-above) build the
//! bulk; the window stores are pure data movement.

#![cfg(target_arch = "x86_64")]

use super::intra_directional_common_avx2::{avg3_window, store_row};

/// Scalar `AVG3` for the three junction cells (matches
/// [`crate::intra_pred`]'s `avg3` exactly).
#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u32 + 2 * b as u32 + c as u32 + 2) >> 2) as u8
}

/// AVX2 D135 predictor inner kernel.
///
/// # Safety
/// Requires AVX2. `above.len() >= bs`, `left.len() >= bs`;
/// `dst.len() >= (bs-1)*stride + bs`; `bs ∈ {4,8,16,32}`.
#[target_feature(enable = "avx2")]
unsafe fn d135_predictor_avx2_inner(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    al: u8,
) {
    debug_assert!(above.len() >= bs && left.len() >= bs);
    debug_assert!(dst.len() >= (bs - 1) * stride + bs);

    // Reversed-left scratch: Lr[k] = left[bs-1-k]; above scratch.
    let mut s_lr = [0u8; 66];
    let mut s_ab = [0u8; 66];
    for k in 0..bs {
        s_lr[k] = left[bs - 1 - k];
    }
    s_ab[..bs].copy_from_slice(&above[..bs]);

    let mut tmp_a = [0u8; 64];
    let mut tmp_b = [0u8; 64];
    avg3_window(&s_lr, &mut tmp_a); // tmp_a[k] = AVG3(Lr[k..k+3])
    avg3_window(&s_ab, &mut tmp_b); // tmp_b[i] = AVG3(above[i..i+3])

    // Assemble the border (length 2*bs-1).
    let mut border = [0u8; 64];
    border[..bs - 2].copy_from_slice(&tmp_a[..bs - 2]);
    border[bs - 2] = avg3(al, left[0], left[1]);
    border[bs - 1] = avg3(left[0], al, above[0]);
    border[bs] = avg3(al, above[0], above[1]);
    border[bs + 1..2 * bs - 1].copy_from_slice(&tmp_b[..bs - 2]);

    // Each row is a bs-window into border, starting at bs-1-i.
    for i in 0..bs {
        let start = bs - 1 - i;
        store_row(&mut dst[i * stride..], &border[start..], bs);
    }
}

/// Safe wrapper for the AVX2 D135 predictor. Signature mirrors the
/// flattened scalar [`crate::intra_pred::d135_predictor`] (`PredCtx`
/// reduced to `above`, `left`, `above_left`).
pub fn d135_predictor_avx2(
    dst: &mut [u8],
    stride: usize,
    bs: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    crate::dsp::validate_intra_predictor_layout(dst.len(), stride, bs, above.len().min(left.len()))
        .unwrap_or_else(|reason| panic!("d135_predictor_avx2: {reason}"));
    // SAFETY: `SimdDispatch::build(caps)` only installs this wrapper
    // when `caps.avx2 == true`. Caps are detected once at OnceLock init
    // and monotonic for the process lifetime. Tests gate every call on
    // a runtime `is_x86_feature_detected!("avx2")` check.
    unsafe { d135_predictor_avx2_inner(dst, stride, bs, above, left, above_left) }
}

#[cfg(all(test, not(miri)))]
mod tests {
    //! Scalar-vs-AVX2 differential tests for `d135_predictor_avx2`.
    use super::d135_predictor_avx2;
    use crate::intra_pred::{PredCtx, d135_predictor};

    #[test]
    #[should_panic(expected = "edge slice too short")]
    fn rejects_short_edge_before_avx2_inner() {
        let mut dst = [0u8; 16];
        let above = [0u8; 4];
        let left = [0u8; 3];
        d135_predictor_avx2(&mut dst, 4, 4, &above, &left, 0);
    }

    #[test]
    #[should_panic(expected = "unsupported block size")]
    fn rejects_invalid_block_size_before_avx2_inner() {
        d135_predictor_avx2(&mut [], 4, 3, &[], &[], 0);
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
        let mut state: u32 = 0x135_0000;
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
                d135_predictor(&mut d_scalar, stride, bs, ctx);
                d135_predictor_avx2(&mut d_avx2, stride, bs, &above, &left, al);
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
            let above = vec![160u8; bs];
            let left = vec![160u8; bs];
            let mut d = vec![0u8; bs * bs];
            d135_predictor_avx2(&mut d, bs, bs, &above, &left, 160);
            assert!(d.iter().all(|&v| v == 160), "bs={bs} not constant");
        }
    }
}
