//! NEON fast-path VP9 forward quantization.
//!
//! This mirrors libvpx `vp9/encoder/arm/neon/vp9_quantize_neon.c` while
//! accepting rvp9's `i32` transform coefficients. Coefficients are processed
//! in raster order; the inverse scan supplies the exact one-based EOB
//! position.

use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_8(
    coeff: *const i32,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    iscan: *const i16,
    round_lo: int32x4_t,
    round_hi: int32x4_t,
    quant_lo: int32x4_t,
    quant_hi: int32x4_t,
    dequant: int16x8_t,
    eob_max: &mut int16x8_t,
) {
    let coeff_lo = vld1q_s32(coeff);
    let coeff_hi = vld1q_s32(coeff.add(4));
    let sign_lo = vshrq_n_s32::<31>(coeff_lo);
    let sign_hi = vshrq_n_s32::<31>(coeff_hi);
    let max = vdupq_n_s32(i16::MAX as i32);

    let rounded_lo = vminq_s32(vaddq_s32(vabsq_s32(coeff_lo), round_lo), max);
    let rounded_hi = vminq_s32(vaddq_s32(vabsq_s32(coeff_hi), round_hi), max);
    let abs_q_lo = vshrq_n_s32::<16>(vmulq_s32(rounded_lo, quant_lo));
    let abs_q_hi = vshrq_n_s32::<16>(vmulq_s32(rounded_hi, quant_hi));
    let signed_lo = vsubq_s32(veorq_s32(abs_q_lo, sign_lo), sign_lo);
    let signed_hi = vsubq_s32(veorq_s32(abs_q_hi, sign_hi), sign_hi);
    let q = vcombine_s16(vmovn_s32(signed_lo), vmovn_s32(signed_hi));
    vst1q_s16(qcoeff, q);

    let dq_lo = vmull_s16(vget_low_s16(q), vget_low_s16(dequant));
    let dq_hi = vmull_s16(vget_high_s16(q), vget_high_s16(dequant));
    vst1q_s16(dqcoeff, vcombine_s16(vmovn_s32(dq_lo), vmovn_s32(dq_hi)));

    let nonzero = vmvnq_u16(vceqq_s16(q, vdupq_n_s16(0)));
    let eob = vbslq_s16(nonzero, vld1q_s16(iscan), vdupq_n_s16(0));
    *eob_max = vmaxq_s16(*eob_max, eob);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_neon_inner(
    coeff: *const i32,
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: &[i16; 2],
    iscan: *const i16,
) -> u16 {
    let round_dc_ac = vld1q_s32(
        [
            i32::from(round[0]),
            i32::from(round[1]),
            i32::from(round[1]),
            i32::from(round[1]),
        ]
        .as_ptr(),
    );
    let round_ac = vdupq_n_s32(i32::from(round[1]));
    let quant_dc_ac = vld1q_s32(
        [
            i32::from(quant[0]),
            i32::from(quant[1]),
            i32::from(quant[1]),
            i32::from(quant[1]),
        ]
        .as_ptr(),
    );
    let quant_ac = vdupq_n_s32(i32::from(quant[1]));
    let dequant_dc_ac = vld1q_s16(
        [
            dequant[0], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1],
            dequant[1],
        ]
        .as_ptr(),
    );
    let dequant_ac = vdupq_n_s16(dequant[1]);
    let mut eob_max = vdupq_n_s16(0);

    quantize_8(
        coeff,
        qcoeff,
        dqcoeff,
        iscan,
        round_dc_ac,
        round_ac,
        quant_dc_ac,
        quant_ac,
        dequant_dc_ac,
        &mut eob_max,
    );
    let mut i = 8;
    while i < n_coeffs {
        quantize_8(
            coeff.add(i),
            qcoeff.add(i),
            dqcoeff.add(i),
            iscan.add(i),
            round_ac,
            round_ac,
            quant_ac,
            quant_ac,
            dequant_ac,
            &mut eob_max,
        );
        i += 8;
    }

    vmaxvq_s16(eob_max) as u16
}

/// Quantize 16, 64, or 256 coefficients with the realtime fast quantizer.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_fp_neon(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> u16 {
    assert!(matches!(n_coeffs, 16 | 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    assert!(qcoeff.len() >= n_coeffs);
    assert!(dqcoeff.len() >= n_coeffs);
    assert!(iscan.len() >= n_coeffs);

    if skip_block {
        qcoeff[..n_coeffs].fill(0);
        dqcoeff[..n_coeffs].fill(0);
        return 0;
    }

    // SAFETY: NEON is mandatory on the supported AArch64 targets. The slice
    // assertions prove each 8-lane load and store stays in bounds. Codec
    // transform coefficients are 16-bit-bounded; widening them to i32 keeps
    // every add and multiply within i32 before the exact narrowing step.
    unsafe {
        vp9_quantize_fp_neon_inner(
            coeff.as_ptr(),
            n_coeffs,
            round,
            quant,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant,
            iscan.as_ptr(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::forward::{QuantizerDeltas, init_plane_quantizers, vp9_quantize_fp};
    use crate::transforms::scans::scan_for;
    use crate::transforms::{TxSize, TxType};

    struct Lcg32(u32);

    impl Lcg32 {
        fn coeff(&mut self) -> i32 {
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            i32::from(self.0 as i16)
        }
    }

    #[test]
    fn differential_all_realtime_sizes_scans_and_qindices() {
        let mut rng = Lcg32(0x51A7_C0DE);
        for tx_size in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16] {
            let n = tx_size.n_coeffs();
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ] {
                let order = scan_for(tx_size, tx_type);
                for q_index in [0, 1, 20, 80, 160, 255] {
                    let qp = init_plane_quantizers(q_index, QuantizerDeltas::Y);
                    let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
                    let mut scalar_q = vec![7; n];
                    let mut scalar_dq = vec![9; n];
                    let mut neon_q = vec![11; n];
                    let mut neon_dq = vec![13; n];
                    let scalar_eob = vp9_quantize_fp(
                        &coeff,
                        n,
                        false,
                        &qp.round_fp,
                        &qp.quant_fp,
                        &mut scalar_q,
                        &mut scalar_dq,
                        &qp.dequant,
                        order.scan,
                    );
                    let neon_eob = vp9_quantize_fp_neon(
                        &coeff,
                        n,
                        false,
                        &qp.round_fp,
                        &qp.quant_fp,
                        &mut neon_q,
                        &mut neon_dq,
                        &qp.dequant,
                        order.iscan,
                    );
                    assert_eq!(neon_eob, scalar_eob, "{tx_size:?}/{tx_type:?}/q{q_index}");
                    assert_eq!(neon_q, scalar_q, "{tx_size:?}/{tx_type:?}/q{q_index}");
                    assert_eq!(neon_dq, scalar_dq, "{tx_size:?}/{tx_type:?}/q{q_index}");
                }
            }
        }
    }

    #[test]
    fn differential_sparse_eob_positions() {
        let qp = init_plane_quantizers(80, QuantizerDeltas::Y);
        for tx_size in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16] {
            let n = tx_size.n_coeffs();
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ] {
                let order = scan_for(tx_size, tx_type);
                for scan_position in [None, Some(0), Some(1), Some(n / 2), Some(n - 1)] {
                    let mut coeff = vec![0; n];
                    if let Some(position) = scan_position {
                        coeff[order.scan[position] as usize] =
                            if position & 1 == 0 { 32_767 } else { -32_768 };
                    }
                    let mut scalar_q = vec![0; n];
                    let mut scalar_dq = vec![0; n];
                    let mut neon_q = vec![0; n];
                    let mut neon_dq = vec![0; n];
                    let scalar_eob = vp9_quantize_fp(
                        &coeff,
                        n,
                        false,
                        &qp.round_fp,
                        &qp.quant_fp,
                        &mut scalar_q,
                        &mut scalar_dq,
                        &qp.dequant,
                        order.scan,
                    );
                    let neon_eob = vp9_quantize_fp_neon(
                        &coeff,
                        n,
                        false,
                        &qp.round_fp,
                        &qp.quant_fp,
                        &mut neon_q,
                        &mut neon_dq,
                        &qp.dequant,
                        order.iscan,
                    );
                    assert_eq!(
                        (neon_eob, neon_q, neon_dq),
                        (scalar_eob, scalar_q, scalar_dq),
                        "{tx_size:?}/{tx_type:?}/{scan_position:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn skip_block_zeroes_outputs() {
        let order = scan_for(TxSize::Tx8x8, TxType::DctDct);
        let mut qcoeff = [1; 64];
        let mut dqcoeff = [2; 64];
        assert_eq!(
            vp9_quantize_fp_neon(
                &[99; 64],
                64,
                true,
                &[1, 2],
                &[3, 4],
                &mut qcoeff,
                &mut dqcoeff,
                &[5, 6],
                order.iscan,
            ),
            0
        );
        assert_eq!(qcoeff, [0; 64]);
        assert_eq!(dqcoeff, [0; 64]);
    }
}
