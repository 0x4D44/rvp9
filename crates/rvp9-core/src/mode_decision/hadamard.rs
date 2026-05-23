//! Scalar Walsh-Hadamard transform for rate-model SATD in the picker.
//!
//! Faithful port of libvpx's `vpx_hadamard_8x8_c` and
//! `vpx_hadamard_16x16_c` from `vpx_dsp/avg.c:199-286` (v1.14.1,
//! commit `12f3a2ac6`). Used by the inlined block_yrd accumulator
//! in `mode_decision::inter::reconstruct_via_round_trip` to produce
//! a rate-proxy SATD on Hadamard-domain magnitudes (closes DV-F1-1).
//!
//! ## Why a separate module
//!
//! - `crates/rvp9-core/src/transforms/` is hands-off (other team's
//!   WiP per CLAUDE.md). The Hadamard kernel doesn't belong there
//!   architecturally either — it's a rate-model proxy, not a coding
//!   transform.
//! - Putting it here in `mode_decision/` puts it next to its only
//!   caller (`inter.rs::reconstruct_via_round_trip`) and keeps the
//!   `transforms/` module hermetic.
//!
//! ## Scalar-only
//!
//! Libvpx has SIMD ports (SSE2, AVX2, NEON) for these. Scalar
//! reference only here — the Hadamard transform is ~3-4× cheaper
//! than the DCT it complements; SIMD acceleration is an F.4.x
//! follow-up if realtime budget pressure increases.

/// Walsh-Hadamard 1-D transform of 8 i16 inputs, stored at
/// `src[i * src_stride]` for `i ∈ [0, 8)`. Output `coeff[0..8]` is
/// the 8-point transform.
///
/// Mirrors libvpx `hadamard_col8` (`vpx_dsp/avg.c:199-227`). The
/// output ordering matches libvpx's (the final transpose is
/// intentionally skipped for SATD purposes — coefficient ORDER
/// doesn't matter for sum-of-absolute-values).
#[inline]
fn hadamard_col8(src: &[i16], src_stride: usize, coeff: &mut [i16; 8]) {
    let b0 = src[0 * src_stride] + src[1 * src_stride];
    let b1 = src[0 * src_stride] - src[1 * src_stride];
    let b2 = src[2 * src_stride] + src[3 * src_stride];
    let b3 = src[2 * src_stride] - src[3 * src_stride];
    let b4 = src[4 * src_stride] + src[5 * src_stride];
    let b5 = src[4 * src_stride] - src[5 * src_stride];
    let b6 = src[6 * src_stride] + src[7 * src_stride];
    let b7 = src[6 * src_stride] - src[7 * src_stride];

    let c0 = b0 + b2;
    let c1 = b1 + b3;
    let c2 = b0 - b2;
    let c3 = b1 - b3;
    let c4 = b4 + b6;
    let c5 = b5 + b7;
    let c6 = b4 - b6;
    let c7 = b5 - b7;

    coeff[0] = c0 + c4;
    coeff[7] = c1 + c5;
    coeff[3] = c2 + c6;
    coeff[4] = c3 + c7;
    coeff[2] = c0 - c4;
    coeff[6] = c1 - c5;
    coeff[1] = c2 - c6;
    coeff[5] = c3 - c7;
}

/// 2-D Walsh-Hadamard transform of an 8×8 block.
///
/// `src` is the residual buffer (8 rows × `src_stride` columns).
/// `coeff` is the 64-element output. Mirrors libvpx
/// `vpx_hadamard_8x8_c` (`vpx_dsp/avg.c:231-254`).
///
/// The transform is two passes of `hadamard_col8` — first columns,
/// then rows of the column-pass output.
pub fn hadamard_8x8(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    debug_assert!(coeff.len() >= 64, "coeff must be at least 64 elements");
    let mut buffer = [0i16; 64];

    // First pass: columns of src.
    for idx in 0..8 {
        let mut tmp = [0i16; 8];
        hadamard_col8(&src[idx..], src_stride, &mut tmp);
        for (i, &v) in tmp.iter().enumerate() {
            buffer[idx * 8 + i] = v;
        }
    }

    // Second pass: columns of buffer (which were rows of the first
    // pass output). Output buffer2 row-major.
    let mut buffer2 = [0i16; 64];
    for idx in 0..8 {
        let mut tmp = [0i16; 8];
        hadamard_col8(&buffer[idx..], 8, &mut tmp);
        for (i, &v) in tmp.iter().enumerate() {
            buffer2[idx + i * 8] = v;
        }
    }

    coeff[..64].copy_from_slice(&buffer2);
}

/// 2-D Walsh-Hadamard transform of a 16×16 block.
///
/// `src` is the residual buffer (16 rows × `src_stride` columns).
/// `coeff` is the 256-element output. Mirrors libvpx
/// `vpx_hadamard_16x16_c` (`vpx_dsp/avg.c:257-286`).
///
/// The 16×16 transform is computed as 4× 8×8 Hadamard sub-transforms
/// followed by a single 2×2-Hadamard combination pass across the
/// four sub-block top-left coefficients.
pub fn hadamard_16x16(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    debug_assert!(coeff.len() >= 256, "coeff must be at least 256 elements");
    // Four 8×8 sub-blocks. Layout follows libvpx:
    //   sub_block_idx | (row_off, col_off)
    //   0             | (0, 0)
    //   1             | (0, 8)
    //   2             | (8, 0)
    //   3             | (8, 8)
    for sb in 0..4 {
        let row_off = (sb / 2) * 8;
        let col_off = (sb & 1) * 8;
        let src_ptr = &src[row_off * src_stride + col_off..];
        let coeff_off = sb * 64;
        hadamard_8x8(src_ptr, src_stride, &mut coeff[coeff_off..coeff_off + 64]);
    }

    // 2×2 Hadamard combination across the four sub-blocks' co-located
    // coefficients. libvpx `vpx_dsp/avg.c:268-285`:
    //   for idx in 0..64:
    //     a0 = coeff[idx]; a1 = coeff[idx + 64];
    //     a2 = coeff[idx + 128]; a3 = coeff[idx + 192];
    //     b0 = (a0 + a1) >> 1; b1 = (a0 - a1) >> 1;
    //     b2 = (a2 + a3) >> 1; b3 = (a2 - a3) >> 1;
    //     coeff[idx] = b0 + b2;
    //     coeff[idx + 64] = b1 + b3;
    //     coeff[idx + 128] = b0 - b2;
    //     coeff[idx + 192] = b1 - b3;
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];

        let b0 = (a0 + a1) >> 1;
        let b1 = (a0 - a1) >> 1;
        let b2 = (a2 + a3) >> 1;
        let b3 = (a2 - a3) >> 1;

        coeff[idx] = b0 + b2;
        coeff[idx + 64] = b1 + b3;
        coeff[idx + 128] = b0 - b2;
        coeff[idx + 192] = b1 - b3;
    }
}

/// 2-D Walsh-Hadamard transform of a 32×32 block.
///
/// Mirrors libvpx `vpx_hadamard_32x32_c` (`vpx_dsp/avg.c:288-`).
/// 4× 16×16 sub-transforms + 2×2 cross-combination. The output is
/// stored as 4×256 sub-blocks; the 2×2 combination is on the
/// top-left coefficients of each.
pub fn hadamard_32x32(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    debug_assert!(coeff.len() >= 1024, "coeff must be at least 1024 elements");
    for sb in 0..4 {
        let row_off = (sb / 2) * 16;
        let col_off = (sb & 1) * 16;
        let src_ptr = &src[row_off * src_stride + col_off..];
        let coeff_off = sb * 256;
        hadamard_16x16(src_ptr, src_stride, &mut coeff[coeff_off..coeff_off + 256]);
    }

    // 2×2 combination across the four 16×16 sub-blocks' top-left coefs.
    for idx in 0..256 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 256];
        let a2 = coeff[idx + 512];
        let a3 = coeff[idx + 768];

        let b0 = (a0 + a1) >> 2;
        let b1 = (a0 - a1) >> 2;
        let b2 = (a2 + a3) >> 2;
        let b3 = (a2 - a3) >> 2;

        coeff[idx] = b0 + b2;
        coeff[idx + 256] = b1 + b3;
        coeff[idx + 512] = b0 - b2;
        coeff[idx + 768] = b1 - b3;
    }
}

/// Sum of absolute values over an i16 slice — the SATD a Hadamard-
/// based rate proxy applies after the transform.
///
/// Mirrors libvpx `vpx_satd_c` (`vpx_dsp/avg.c:334-341`) but takes
/// `i16` instead of `tran_low_t` (rvp9's quant/DCT pipeline uses
/// `i16` for qcoeffs natively).
#[must_use]
pub fn hadamard_satd(coeff: &[i16]) -> u32 {
    let mut sum: u32 = 0;
    for &c in coeff {
        sum = sum.saturating_add(i32::from(c).unsigned_abs());
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_8x8_all_zero_input_yields_all_zero() {
        let src = [0i16; 64];
        let mut coeff = [0i16; 64];
        hadamard_8x8(&src, 8, &mut coeff);
        assert!(coeff.iter().all(|&v| v == 0));
    }

    #[test]
    fn hadamard_8x8_dc_only_concentrates_in_dc() {
        // Constant input → all 64 samples equal to a constant. The
        // 2-D Hadamard transform of a DC field is a single DC peak
        // at coeff[0] = N * input (N = 64 for 8x8), with all other
        // bins = 0 (after the 2-D Hadamard's energy concentration).
        let dc: i16 = 5;
        let src = [dc; 64];
        let mut coeff = [0i16; 64];
        hadamard_8x8(&src, 8, &mut coeff);
        // Exactly one bin should be non-zero; its magnitude is
        // 64 * dc = 320.
        let nonzero: Vec<(usize, i16)> = coeff
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
            .collect();
        assert_eq!(nonzero.len(), 1, "DC input should produce single bin");
        assert_eq!(nonzero[0].1, 64 * dc);
    }

    #[test]
    fn hadamard_8x8_preserves_parseval_modulo_factor_64() {
        // Walsh-Hadamard transform satisfies sum_i x_i^2 * N = sum_i y_i^2
        // for orthogonal Hadamard with normalization H * H^T = N*I,
        // where here N = 64 (8x8 transform without sqrt(N) factor).
        // Use small random-ish input.
        let src: [i16; 64] = std::array::from_fn(|i| {
            ((i as i16) % 11) - 5
        });
        let input_energy: i64 = src
            .iter()
            .map(|&v| i64::from(v) * i64::from(v))
            .sum();
        let mut coeff = [0i16; 64];
        hadamard_8x8(&src, 8, &mut coeff);
        let coeff_energy: i64 = coeff
            .iter()
            .map(|&v| i64::from(v) * i64::from(v))
            .sum();
        assert_eq!(
            coeff_energy, 64 * input_energy,
            "Parseval: coeff energy = 64 * input energy for 8x8 WHT"
        );
    }

    #[test]
    fn hadamard_8x8_impulse_at_origin_is_constant_output() {
        // Single non-zero pixel at (0,0): each output coefficient is
        // ±1 * source[0][0] = ±5 (with signs from the 2-D Hadamard
        // basis function values). All 64 outputs have magnitude 5.
        let mut src = [0i16; 64];
        src[0] = 5;
        let mut coeff = [0i16; 64];
        hadamard_8x8(&src, 8, &mut coeff);
        for &c in coeff.iter() {
            assert!(
                c.abs() == 5,
                "impulse input should produce all ±5 outputs, got {c}"
            );
        }
    }

    #[test]
    fn hadamard_16x16_dc_concentrates() {
        let dc: i16 = 3;
        let src = [dc; 256];
        let mut coeff = [0i16; 256];
        hadamard_16x16(&src, 16, &mut coeff);
        // DC bin magnitude = 256 * dc / 2 (the 16×16 has an extra
        // >> 1 in the 2x2 combination pass), but since the
        // sub-blocks have N=64 inside each and there are 4 of them
        // with sums that get >> 1, the math is: each sub-block's
        // DC = 64 * dc = 192; then 2x2 combine: (192+192)>>1 +
        // (192+192)>>1 = 192 + 192 = 384. So DC bin = 384.
        assert_eq!(coeff[0], 4 * 64 * i16::from(dc) / 2);
    }

    #[test]
    fn hadamard_satd_zero_is_zero() {
        let coeff = [0i16; 64];
        assert_eq!(hadamard_satd(&coeff), 0);
    }

    #[test]
    fn hadamard_satd_handles_negatives() {
        let coeff: [i16; 4] = [3, -7, 2, -1];
        assert_eq!(hadamard_satd(&coeff), 13);
    }

    #[test]
    fn hadamard_16x16_zero_input() {
        let src = [0i16; 256];
        let mut coeff = [0i16; 256];
        hadamard_16x16(&src, 16, &mut coeff);
        assert!(coeff.iter().all(|&v| v == 0));
    }

    #[test]
    fn hadamard_32x32_zero_input() {
        let src = [0i16; 1024];
        let mut coeff = [0i16; 1024];
        hadamard_32x32(&src, 32, &mut coeff);
        assert!(coeff.iter().all(|&v| v == 0));
    }
}
