//! AVX2 output quantizer for the realtime Hadamard/DCT fast path.
//!
//! The scalar [`crate::quant::forward::vp9_quantize_fp`] walks the scan order
//! so it can update EOB as it quantizes. Standard VP9 scan orders are a
//! permutation whose first entry is raster coefficient zero, so the
//! quantization arithmetic itself can run in raster order and a final reverse
//! scan computes the same EOB. Raster order makes both the coefficient reads
//! and the qcoeff/dqcoeff writes contiguous, which avoids a gather/scatter
//! pair for every eight coefficients.
//!
//! This kernel is only selected for the standard `ScanOrder` values used by
//! the encoder. Callers retain the scalar path for any non-standard order.

use core::arch::x86_64::{
    _mm_and_si128, _mm_packus_epi32, _mm_set1_epi32, _mm_storeu_si128, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_max_epi32,
    _mm256_min_epi32, _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_slli_epi32, _mm256_srai_epi32,
    _mm256_sub_epi32, _mm256_xor_si256,
};

const INT16_MIN_I32: i32 = i16::MIN as i32;
const INT16_MAX_I32: i32 = i16::MAX as i32;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_i16x8(values: core::arch::x86_64::__m256i, dst: *mut i16) {
    let mask = _mm_set1_epi32(0xffff);
    let lo = _mm_and_si128(_mm256_castsi256_si128(values), mask);
    let hi = _mm_and_si128(_mm256_extracti128_si256(values, 1), mask);
    let packed = _mm_packus_epi32(lo, hi);
    _mm_storeu_si128(dst.cast(), packed);
}

#[inline]
fn quantize_one(coeff: i32, round: i16, quant: i16, dequant: i16) -> (i16, i16, bool) {
    let coeff_sign = coeff >> 31;
    let abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    let tmp = (abs_coeff + i32::from(round)).clamp(INT16_MIN_I32, INT16_MAX_I32);
    let tmp = (tmp * i32::from(quant)) >> 16;
    let signed = (tmp ^ coeff_sign) - coeff_sign;
    let qcoeff = signed as i16;
    let dqcoeff = (i32::from(qcoeff) * i32::from(dequant)) as i16;
    (qcoeff, dqcoeff, tmp != 0)
}

/// Quantize a standard 4×4, 8×8, or 16×16 VP9 block with AVX2.
///
/// # Safety
///
/// The caller must guarantee AVX2 support. `scan[0]` must be raster
/// coefficient zero and `scan[..n_coeffs]` must be a complete permutation of
/// the raster coefficients. The slices must have at least `n_coeffs` entries.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_avx2_inner(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    scan: &[i16],
) -> u16 {
    debug_assert!(matches!(n_coeffs, 16 | 64 | 256));
    debug_assert!(coeff.len() >= n_coeffs);
    debug_assert!(qcoeff.len() >= n_coeffs);
    debug_assert!(dqcoeff.len() >= n_coeffs);
    debug_assert!(scan.len() >= n_coeffs);
    debug_assert_eq!(scan[0], 0);

    if skip_block {
        qcoeff[..n_coeffs].fill(0);
        dqcoeff[..n_coeffs].fill(0);
        return 0;
    }

    let (dc_q, dc_dq, _) = quantize_one(coeff[0], round[0], quant[0], dequant[0]);
    qcoeff[0] = dc_q;
    dqcoeff[0] = dc_dq;

    let round_ac = _mm256_set1_epi32(i32::from(round[1]));
    let quant_ac = _mm256_set1_epi32(i32::from(quant[1]));
    let dequant_ac = _mm256_set1_epi32(i32::from(dequant[1]));
    let vector_end = 1 + ((n_coeffs - 1) / 8) * 8;
    for offset in (1..vector_end).step_by(8) {
        // G4: loads from `coeff` use _mm256_loadu_si256 (unaligned)
        let values = _mm256_loadu_si256(coeff.as_ptr().add(offset).cast());
        let sign = _mm256_srai_epi32(values, 31);
        let abs = _mm256_sub_epi32(_mm256_xor_si256(values, sign), sign);
        let rounded = _mm256_add_epi32(abs, round_ac);
        let rounded = _mm256_max_epi32(
            _mm256_set1_epi32(INT16_MIN_I32),
            _mm256_min_epi32(_mm256_set1_epi32(INT16_MAX_I32), rounded),
        );
        let magnitude = _mm256_srai_epi32(_mm256_mullo_epi32(rounded, quant_ac), 16);
        let signed = _mm256_sub_epi32(_mm256_xor_si256(magnitude, sign), sign);
        let q_i16_sign_extended = _mm256_srai_epi32(_mm256_slli_epi32(signed, 16), 16);
        let dq = _mm256_mullo_epi32(q_i16_sign_extended, dequant_ac);
        store_i16x8(signed, qcoeff.as_mut_ptr().add(offset));
        store_i16x8(dq, dqcoeff.as_mut_ptr().add(offset));
    }
    for offset in vector_end..n_coeffs {
        let (q, dq, _) = quantize_one(coeff[offset], round[1], quant[1], dequant[1]);
        qcoeff[offset] = q;
        dqcoeff[offset] = dq;
    }

    // `tmp != 0` is equivalent to qcoeff != 0 for valid PlaneQuantizers.
    // The reverse scan reproduces the scalar loop's last-nonzero position.
    let mut eob = 0usize;
    for i in (0..n_coeffs).rev() {
        if qcoeff[scan[i] as usize] != 0 {
            eob = i + 1;
            break;
        }
    }
    eob as u16
}

/// Quantize a standard 4×4, 8×8, or 16×16 VP9 block with AVX2.
///
/// The caller must have selected this function through the process-wide
/// dispatch table, which guarantees AVX2 support. The safe wrapper follows
/// the same dispatch invariant as the other x86 SIMD kernels.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_fp_avx2(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    scan: &[i16],
) -> u16 {
    assert!(matches!(n_coeffs, 16 | 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    assert!(qcoeff.len() >= n_coeffs);
    assert!(dqcoeff.len() >= n_coeffs);
    assert!(scan.len() >= n_coeffs);
    assert_eq!(scan[0], 0);
    // SAFETY: Callers reach this wrapper only after the immutable dispatch
    // table reports AVX2 support; the assertions above prove all slices and
    // the standard scan precondition required by the AVX2 inner kernel.
    unsafe {
        vp9_quantize_fp_avx2_inner(
            coeff, n_coeffs, skip_block, round, quant, qcoeff, dqcoeff, dequant, scan,
        )
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::quant::forward::{QuantizerDeltas, init_plane_quantizers, vp9_quantize_fp};
    use crate::transforms::{TxSize, TxType, scan_for};

    fn require_avx2() {
        assert!(is_x86_feature_detected!("avx2"));
    }

    #[test]
    fn scalar_equivalence_random_coefficients() {
        require_avx2();
        let mut coeff = vec![0i32; 256];
        for (index, value) in coeff.iter_mut().enumerate() {
            *value = ((index as i32 * 7919) % 65_000) - 32_000;
        }
        for (tx_size, n) in [
            (TxSize::Tx4x4, 16),
            (TxSize::Tx8x8, 64),
            (TxSize::Tx16x16, 256),
        ] {
            let scan = scan_for(tx_size, TxType::DctDct);
            let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
            let mut expected_q = vec![99i16; n];
            let mut expected_dq = vec![99i16; n];
            let mut actual_q = vec![99i16; n];
            let mut actual_dq = vec![99i16; n];
            let expected_eob = vp9_quantize_fp(
                &coeff[..n],
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut expected_q,
                &mut expected_dq,
                &qp.dequant,
                scan.scan(),
            );
            let actual_eob = vp9_quantize_fp_avx2(
                &coeff[..n],
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut actual_q,
                &mut actual_dq,
                &qp.dequant,
                scan.scan(),
            );
            assert_eq!(actual_eob, expected_eob, "{tx_size:?} eob");
            assert_eq!(actual_q, expected_q, "{tx_size:?} qcoeff");
            assert_eq!(actual_dq, expected_dq, "{tx_size:?} dqcoeff");
        }
    }

    #[test]
    fn skip_block_clears_reused_outputs() {
        require_avx2();
        let coeff = [123i32; 16];
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
        let mut qcoeff = [99i16; 16];
        let mut dqcoeff = [99i16; 16];
        let eob = vp9_quantize_fp_avx2(
            &coeff,
            16,
            true,
            &qp.round_fp,
            &qp.quant_fp,
            &mut qcoeff,
            &mut dqcoeff,
            &qp.dequant,
            scan.scan(),
        );
        assert_eq!(eob, 0);
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }
}
