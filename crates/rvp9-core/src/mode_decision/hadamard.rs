//! Walsh-Hadamard transform for rate-model SATD in the picker.
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
//! The scalar implementation is the platform-independent oracle.
//! AArch64 dispatches to an integer-exact NEON port of libvpx's
//! `vpx_hadamard_{8x8,16x16,32x32}_neon` kernels.

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[path = "hadamard_neon.rs"]
mod neon;

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "hadamard_sse2.rs"]
mod sse2;

/// Walsh-Hadamard 1-D transform of 8 i16 inputs, stored at
/// `src[i * src_stride]` for `i ∈ [0, 8)`. Output `coeff[0..8]` is
/// the 8-point transform.
///
/// Mirrors libvpx `hadamard_col8` (`vpx_dsp/avg.c:199-227`). The
/// output ordering matches libvpx's (the final transpose is
/// intentionally skipped for SATD purposes — coefficient ORDER
/// doesn't matter for sum-of-absolute-values).
#[inline]
#[cfg(any(test, miri, not(target_arch = "aarch64")))]
fn hadamard_col8(src: &[i16], src_stride: usize, coeff: &mut [i16; 8]) {
    let b0 = src[0] + src[src_stride];
    let b1 = src[0] - src[src_stride];
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
#[cfg(any(test, miri, not(target_arch = "aarch64")))]
fn hadamard_8x8_scalar(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
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
#[cfg(any(test, miri, not(target_arch = "aarch64")))]
fn hadamard_16x16_scalar(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
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
        hadamard_8x8_scalar(src_ptr, src_stride, &mut coeff[coeff_off..coeff_off + 64]);
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

        let b0 = ((i32::from(a0) + i32::from(a1)) >> 1) as i16;
        let b1 = ((i32::from(a0) - i32::from(a1)) >> 1) as i16;
        let b2 = ((i32::from(a2) + i32::from(a3)) >> 1) as i16;
        let b3 = ((i32::from(a2) - i32::from(a3)) >> 1) as i16;

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
#[cfg(any(test, miri, not(target_arch = "aarch64")))]
fn hadamard_32x32_scalar(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    debug_assert!(coeff.len() >= 1024, "coeff must be at least 1024 elements");
    for sb in 0..4 {
        let row_off = (sb / 2) * 16;
        let col_off = (sb & 1) * 16;
        let src_ptr = &src[row_off * src_stride + col_off..];
        let coeff_off = sb * 256;
        hadamard_16x16_scalar(src_ptr, src_stride, &mut coeff[coeff_off..coeff_off + 256]);
    }

    // 2×2 combination across the four 16×16 sub-blocks' top-left coefs.
    for idx in 0..256 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 256];
        let a2 = coeff[idx + 512];
        let a3 = coeff[idx + 768];

        let b0 = ((i32::from(a0) + i32::from(a1)) >> 2) as i16;
        let b1 = ((i32::from(a0) - i32::from(a1)) >> 2) as i16;
        let b2 = ((i32::from(a2) + i32::from(a3)) >> 2) as i16;
        let b3 = ((i32::from(a2) - i32::from(a3)) >> 2) as i16;

        coeff[idx] = b0 + b2;
        coeff[idx + 256] = b1 + b3;
        coeff[idx + 512] = b0 - b2;
        coeff[idx + 768] = b1 - b3;
    }
}

/// 2-D Walsh-Hadamard transform of an 8×8 block.
pub fn hadamard_8x8(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        neon::hadamard_8x8(src, src_stride, coeff);
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if crate::dsp::get_ref().caps.sse2 {
            sse2::hadamard_8x8(src, src_stride, coeff);
        } else {
            hadamard_8x8_scalar(src, src_stride, coeff);
        }
    }
    #[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")), miri))]
    {
        hadamard_8x8_scalar(src, src_stride, coeff);
    }
}

/// 2-D Walsh-Hadamard transform of a 16×16 block.
pub fn hadamard_16x16(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        neon::hadamard_16x16(src, src_stride, coeff);
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if crate::dsp::get_ref().caps.sse2 {
            sse2::hadamard_16x16(src, src_stride, coeff);
        } else {
            hadamard_16x16_scalar(src, src_stride, coeff);
        }
    }
    #[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")), miri))]
    {
        hadamard_16x16_scalar(src, src_stride, coeff);
    }
}

/// Fused SIMD `src - pred` and 8×8 Walsh-Hadamard transform.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
pub fn hadamard_8x8_subtract(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    coeff: &mut [i16],
) {
    #[cfg(target_arch = "aarch64")]
    neon::hadamard_8x8_subtract(src, src_stride, pred, pred_stride, coeff);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::get_ref().caps.sse2 {
        sse2::hadamard_8x8_subtract(src, src_stride, pred, pred_stride, coeff);
    } else {
        hadamard_subtract_scalar(
            src,
            src_stride,
            pred,
            pred_stride,
            coeff,
            8,
            hadamard_8x8_scalar,
        );
    }
}

/// Fused SIMD `src - pred` and 16×16 Walsh-Hadamard transform.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
pub fn hadamard_16x16_subtract(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    coeff: &mut [i16],
) {
    #[cfg(target_arch = "aarch64")]
    neon::hadamard_16x16_subtract(src, src_stride, pred, pred_stride, coeff);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::get_ref().caps.sse2 {
        sse2::hadamard_16x16_subtract(src, src_stride, pred, pred_stride, coeff);
    } else {
        hadamard_subtract_scalar(
            src,
            src_stride,
            pred,
            pred_stride,
            coeff,
            16,
            hadamard_16x16_scalar,
        );
    }
}

/// Fused SIMD `src - pred` and 32×32 Walsh-Hadamard transform.
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
pub fn hadamard_32x32_subtract(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    coeff: &mut [i16],
) {
    #[cfg(target_arch = "aarch64")]
    neon::hadamard_32x32_subtract(src, src_stride, pred, pred_stride, coeff);
    #[cfg(target_arch = "x86_64")]
    if crate::dsp::get_ref().caps.sse2 {
        sse2::hadamard_32x32_subtract(src, src_stride, pred, pred_stride, coeff);
    } else {
        hadamard_subtract_scalar(
            src,
            src_stride,
            pred,
            pred_stride,
            coeff,
            32,
            hadamard_32x32_scalar,
        );
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
fn hadamard_subtract_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    coeff: &mut [i16],
    side: usize,
    transform: fn(&[i16], usize, &mut [i16]),
) {
    sse2::validate_subtract(src, src_stride, pred, pred_stride, coeff, side);
    let mut residual = [0i16; 32 * 32];
    for row in 0..side {
        for col in 0..side {
            residual[row * side + col] =
                i16::from(src[row * src_stride + col]) - i16::from(pred[row * pred_stride + col]);
        }
    }
    transform(&residual[..side * side], side, coeff);
}

/// 2-D Walsh-Hadamard transform of a 32×32 block.
pub fn hadamard_32x32(src: &[i16], src_stride: usize, coeff: &mut [i16]) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        neon::hadamard_32x32(src, src_stride, coeff);
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if crate::dsp::get_ref().caps.sse2 {
            sse2::hadamard_32x32(src, src_stride, coeff);
        } else {
            hadamard_32x32_scalar(src, src_stride, coeff);
        }
    }
    #[cfg(any(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")), miri))]
    {
        hadamard_32x32_scalar(src, src_stride, coeff);
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

#[cfg(all(test, target_arch = "x86_64", not(miri)))]
pub(crate) fn exercise_force_scalar_paths() {
    for side in [8usize, 16, 32] {
        let residual: Vec<i16> = (0..side * side)
            .map(|index| ((index * 37 + 13) % 255) as i16 - 127)
            .collect();
        let source: Vec<u8> = (0..side * side)
            .map(|index| ((index * 43 + 19) & 0xff) as u8)
            .collect();
        let prediction: Vec<u8> = (0..side * side)
            .map(|index| ((index * 17 + 3) & 0xff) as u8)
            .collect();
        let mut coeff = vec![0i16; side * side];

        match side {
            8 => hadamard_8x8(&residual, side, &mut coeff),
            16 => hadamard_16x16(&residual, side, &mut coeff),
            32 => hadamard_32x32(&residual, side, &mut coeff),
            _ => unreachable!(),
        }
        assert!(coeff.iter().any(|&value| value != 0));

        match side {
            8 => hadamard_8x8_subtract(&source, side, &prediction, side, &mut coeff),
            16 => hadamard_16x16_subtract(&source, side, &prediction, side, &mut coeff),
            32 => hadamard_32x32_subtract(&source, side, &prediction, side, &mut coeff),
            _ => unreachable!(),
        }
        assert!(coeff.iter().any(|&value| value != 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    struct Lcg32(u32);

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    impl Lcg32 {
        fn residual(&mut self) -> i16 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 16) % 511) as i16 - 255
        }
    }

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
        let src: [i16; 64] = std::array::from_fn(|i| ((i as i16) % 11) - 5);
        let input_energy: i64 = src.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        let mut coeff = [0i16; 64];
        hadamard_8x8(&src, 8, &mut coeff);
        let coeff_energy: i64 = coeff.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
        assert_eq!(
            coeff_energy,
            64 * input_energy,
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
        assert_eq!(coeff[0], 4 * 64 * dc / 2);
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

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    #[test]
    fn simd_matches_scalar_for_random_residuals_and_stride_jitter() {
        let mut rng = Lcg32(0x4841_4441);
        #[cfg(target_arch = "aarch64")]
        let simd = [
            neon::hadamard_8x8 as fn(&[i16], usize, &mut [i16]),
            neon::hadamard_16x16,
            neon::hadamard_32x32,
        ];
        #[cfg(target_arch = "x86_64")]
        let simd = [
            sse2::hadamard_8x8 as fn(&[i16], usize, &mut [i16]),
            sse2::hadamard_16x16,
            sse2::hadamard_32x32,
        ];
        for (size, scalar, simd) in [
            (
                8usize,
                hadamard_8x8_scalar as fn(&[i16], usize, &mut [i16]),
                simd[0],
            ),
            (16, hadamard_16x16_scalar, simd[1]),
            (32, hadamard_32x32_scalar, simd[2]),
        ] {
            for jitter in 0..9 {
                let stride = size + jitter;
                let mut src = vec![0i16; (size - 1) * stride + size];
                for value in &mut src {
                    *value = rng.residual();
                }
                let mut expected = vec![0i16; size * size];
                let mut actual = vec![0i16; size * size];
                scalar(&src, stride, &mut expected);
                simd(&src, stride, &mut actual);
                assert_eq!(
                    actual, expected,
                    "SIMD mismatch for {size}x{size}, stride jitter {jitter}"
                );
            }
        }
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    #[test]
    fn fused_subtract_matches_staged_hadamard_for_all_sizes_and_strides() {
        type Transform = fn(&[i16], usize, &mut [i16]);
        type FusedTransform = fn(&[u8], usize, &[u8], usize, &mut [i16]);

        for (size, staged, fused) in [
            (
                8usize,
                hadamard_8x8 as Transform,
                hadamard_8x8_subtract as FusedTransform,
            ),
            (16, hadamard_16x16, hadamard_16x16_subtract),
            (32, hadamard_32x32, hadamard_32x32_subtract),
        ] {
            for jitter in [0usize, 1, 5] {
                let src_stride = size + jitter;
                let pred_stride = size + 5 - jitter;
                let mut src = vec![0u8; (size - 1) * src_stride + size];
                let mut pred = vec![0u8; (size - 1) * pred_stride + size];
                for row in 0..size {
                    for col in 0..size {
                        src[row * src_stride + col] =
                            ((row * 73 + col * 151 + jitter * 29) & 255) as u8;
                        pred[row * pred_stride + col] =
                            ((row * 197 + col * 41 + jitter * 113 + 255) & 255) as u8;
                    }
                }
                src[0] = 0;
                pred[0] = 255;
                src[(size - 1) * src_stride + size - 1] = 255;
                pred[(size - 1) * pred_stride + size - 1] = 0;

                let mut residual = vec![0i16; size * size];
                for row in 0..size {
                    for col in 0..size {
                        residual[row * size + col] = i16::from(src[row * src_stride + col])
                            - i16::from(pred[row * pred_stride + col]);
                    }
                }
                let mut expected = vec![0i16; size * size];
                let mut actual = vec![0i16; size * size];
                staged(&residual, size, &mut expected);
                fused(&src, src_stride, &pred, pred_stride, &mut actual);
                assert_eq!(
                    actual, expected,
                    "fused subtract mismatch for {size}x{size}, stride jitter {jitter}"
                );
            }
        }
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), not(miri)))]
    #[test]
    fn simd_matches_scalar_for_residual_extremes() {
        for size in [8usize, 16, 32] {
            let stride = size + 5;
            for (low, high) in [(0, 0), (-255, 255), (255, -255)] {
                let src: Vec<i16> = (0..((size - 1) * stride + size))
                    .map(|idx| {
                        if (idx / stride + idx % stride) & 1 == 0 {
                            low
                        } else {
                            high
                        }
                    })
                    .collect();
                let mut expected = vec![0i16; size * size];
                let mut actual = vec![0i16; size * size];
                match size {
                    8 => {
                        hadamard_8x8_scalar(&src, stride, &mut expected);
                        #[cfg(target_arch = "aarch64")]
                        neon::hadamard_8x8(&src, stride, &mut actual);
                        #[cfg(target_arch = "x86_64")]
                        sse2::hadamard_8x8(&src, stride, &mut actual);
                    }
                    16 => {
                        hadamard_16x16_scalar(&src, stride, &mut expected);
                        #[cfg(target_arch = "aarch64")]
                        neon::hadamard_16x16(&src, stride, &mut actual);
                        #[cfg(target_arch = "x86_64")]
                        sse2::hadamard_16x16(&src, stride, &mut actual);
                    }
                    32 => {
                        hadamard_32x32_scalar(&src, stride, &mut expected);
                        #[cfg(target_arch = "aarch64")]
                        neon::hadamard_32x32(&src, stride, &mut actual);
                        #[cfg(target_arch = "x86_64")]
                        sse2::hadamard_32x32(&src, stride, &mut actual);
                    }
                    _ => unreachable!(),
                }
                assert_eq!(actual, expected, "SIMD mismatch for {size}x{size}");
            }
        }
    }
}
