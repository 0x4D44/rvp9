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
    _mm256_andnot_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_max_epi16, _mm256_max_epi32, _mm256_min_epi32, _mm256_mullo_epi32,
    _mm256_or_si256, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srai_epi32,
    _mm256_storeu_si256, _mm256_sub_epi32, _mm256_testz_si256, _mm256_xor_si256,
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

#[inline(always)]
fn skip_zero_chunk(testz_result: i32) -> bool {
    testz_result != 0
}

#[inline(always)]
fn can_return_zero_eob(provisional_ac_all_zero: bool, dc_q: i16) -> bool {
    // Raster lane zero was quantized with AC parameters before scalar DC
    // replacement. Its provisional value can only suppress this shortcut.
    provisional_ac_all_zero && dc_q == 0
}

/// Reduce raster-order quantized coefficients to the one-past-last scan
/// position using the inverse scan table.  The vector path is worthwhile for
/// 8×8 and 16×16 blocks; the 4×4 case remains with the scalar reverse scan so
/// its fixed setup cost does not dominate the small block.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn eob_from_iscan(qcoeff: &[i16], n_coeffs: usize, iscan: &[i16]) -> u16 {
    debug_assert!(n_coeffs >= 64);
    debug_assert!(qcoeff.len() >= n_coeffs);
    debug_assert!(iscan.len() >= n_coeffs);

    let zero = _mm256_setzero_si256();
    // Standard iscan entries are one-based positions in [1, n_coeffs], and
    // this helper only serves the supported 8x8/16x16 sizes (64 or 256).
    let mut max_position = zero;
    for offset in (0..n_coeffs).step_by(16) {
        let values = _mm256_loadu_si256(qcoeff.as_ptr().add(offset).cast());
        // A zero chunk contributes no scan position; avoid its iscan load and
        // max reduction. `testz` returns one only when every lane is zero.
        if skip_zero_chunk(_mm256_testz_si256(values, values)) {
            continue;
        }
        let positions = _mm256_loadu_si256(iscan.as_ptr().add(offset).cast());
        let zero_mask = _mm256_cmpeq_epi16(values, zero);
        let nonzero_positions = _mm256_andnot_si256(zero_mask, positions);
        max_position = _mm256_max_epi16(max_position, nonzero_positions);
    }

    let mut lanes = [0i16; 16];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), max_position);
    lanes.into_iter().max().unwrap_or(0) as u16
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
unsafe fn vp9_quantize_fp_avx2_inner<const TRACK_ZERO_EOB: bool>(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    scan: &[i16],
    iscan: Option<&[i16]>,
) -> u16 {
    debug_assert!(matches!(n_coeffs, 16 | 64 | 256));
    debug_assert!(coeff.len() >= n_coeffs);
    debug_assert!(qcoeff.len() >= n_coeffs);
    debug_assert!(dqcoeff.len() >= n_coeffs);
    debug_assert!(scan.len() >= n_coeffs);
    if let Some(iscan) = iscan {
        debug_assert!(iscan.len() >= n_coeffs);
    }
    debug_assert_eq!(scan[0], 0);

    if skip_block {
        qcoeff[..n_coeffs].fill(0);
        dqcoeff[..n_coeffs].fill(0);
        return 0;
    }

    let (dc_q, dc_dq, _) = quantize_one(coeff[0], round[0], quant[0], dequant[0]);

    let round_ac = _mm256_set1_epi32(i32::from(round[1]));
    let quant_ac = _mm256_set1_epi32(i32::from(quant[1]));
    let dequant_ac = _mm256_set1_epi32(i32::from(dequant[1]));
    // All supported block sizes are multiples of eight. Starting at raster
    // zero removes the seven-element scalar AC tail; the lane-0 AC result is
    // overwritten with the separately-quantized DC value below.
    let vector_end = n_coeffs & !7;
    let mut provisional_ac_nonzero = _mm256_setzero_si256();
    for offset in (0..vector_end).step_by(8) {
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
        if TRACK_ZERO_EOB {
            // These are the exact i16 values written below, sign-extended to
            // i32. Keep one OR across the production TX8/TX16 chunks; the
            // no-iscan and TX4 instantiations compile this work out.
            provisional_ac_nonzero = _mm256_or_si256(provisional_ac_nonzero, q_i16_sign_extended);
        }
        let dq = _mm256_mullo_epi32(q_i16_sign_extended, dequant_ac);
        store_i16x8(signed, qcoeff.as_mut_ptr().add(offset));
        store_i16x8(dq, dqcoeff.as_mut_ptr().add(offset));
    }
    for offset in vector_end..n_coeffs {
        let (q, dq, _) = quantize_one(coeff[offset], round[1], quant[1], dequant[1]);
        qcoeff[offset] = q;
        dqcoeff[offset] = dq;
    }

    qcoeff[0] = dc_q;
    dqcoeff[0] = dc_dq;

    if TRACK_ZERO_EOB
        && can_return_zero_eob(
            _mm256_testz_si256(provisional_ac_nonzero, provisional_ac_nonzero) != 0,
            dc_q,
        )
    {
        return 0;
    }

    if let Some(iscan) = iscan
        && n_coeffs >= 64
    {
        // Dense blocks are common and the scalar reverse scan normally
        // finds their EOB in its first few positions.  Probe that tail
        // before paying the fixed cost of the full raster reduction.
        for i in (n_coeffs.saturating_sub(16)..n_coeffs).rev() {
            if qcoeff[scan[i] as usize] != 0 {
                return (i + 1) as u16;
            }
        }
        return eob_from_iscan(qcoeff, n_coeffs, iscan);
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
pub(crate) fn vp9_quantize_fp_avx2(
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
        vp9_quantize_fp_avx2_inner::<false>(
            coeff, n_coeffs, skip_block, round, quant, qcoeff, dqcoeff, dequant, scan, None,
        )
    }
}

/// Internal production wrapper that uses the complete scan order's inverse
/// table to vectorize the EOB reduction after raster-order quantization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vp9_quantize_fp_avx2_with_order(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    scan: &[i16],
    iscan: &[i16],
) -> u16 {
    assert!(matches!(n_coeffs, 16 | 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    assert!(qcoeff.len() >= n_coeffs);
    assert!(dqcoeff.len() >= n_coeffs);
    assert!(scan.len() >= n_coeffs);
    assert!(iscan.len() >= n_coeffs);
    assert_eq!(scan[0], 0);
    // SAFETY: Callers reach this wrapper through the AVX2 dispatch table; the
    // assertions above prove all slice and standard-scan preconditions.
    unsafe {
        if n_coeffs >= 64 {
            vp9_quantize_fp_avx2_inner::<true>(
                coeff,
                n_coeffs,
                skip_block,
                round,
                quant,
                qcoeff,
                dqcoeff,
                dequant,
                scan,
                Some(iscan),
            )
        } else {
            vp9_quantize_fp_avx2_inner::<false>(
                coeff,
                n_coeffs,
                skip_block,
                round,
                quant,
                qcoeff,
                dqcoeff,
                dequant,
                scan,
                Some(iscan),
            )
        }
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
    fn zero_eob_shortcut_requires_zero_dc_and_zero_vector() {
        assert!(can_return_zero_eob(true, 0));
        assert!(!can_return_zero_eob(true, 1));
        assert!(!can_return_zero_eob(false, 0));
    }

    #[test]
    fn zero_eob_shortcut_matches_scalar_with_late_nonzero_and_reused_outputs() {
        require_avx2();
        for (tx_size, n) in [
            (TxSize::Tx4x4, 16),
            (TxSize::Tx8x8, 64),
            (TxSize::Tx16x16, 256),
        ] {
            for tx_type in [TxType::DctDct, TxType::DctAdst, TxType::AdstDct] {
                let order = scan_for(tx_size, tx_type);
                for case in 0..5 {
                    let mut coeff = vec![0i32; n];
                    let mut quant = [16_000i16, 30_000i16];
                    let skip_block = case == 4;
                    match case {
                        1 => coeff[n - 1] = 32_000, // Final eight-lane chunk.
                        2 => coeff[0] = 32_000,     // DC only.
                        3 => {
                            coeff[0] = 32_000;
                            quant[0] = 0; // Provisional AC lane 0 is nonzero; real DC is zero.
                        }
                        4 => coeff[n - 1] = 32_000,
                        _ => {} // All coefficients quantize to zero.
                    }
                    let round = [0i16, 0i16];
                    let dequant = [31i16, 57i16];
                    let mut expected_q = vec![99i16; n];
                    let mut expected_dq = vec![99i16; n];
                    let expected_eob = vp9_quantize_fp(
                        &coeff,
                        n,
                        skip_block,
                        &round,
                        &quant,
                        &mut expected_q,
                        &mut expected_dq,
                        &dequant,
                        order.scan(),
                    );
                    let mut actual_q = vec![99i16; n];
                    let mut actual_dq = vec![99i16; n];
                    let actual_eob = vp9_quantize_fp_avx2_with_order(
                        &coeff,
                        n,
                        skip_block,
                        &round,
                        &quant,
                        &mut actual_q,
                        &mut actual_dq,
                        &dequant,
                        order.scan(),
                        order.iscan(),
                    );
                    assert_eq!(
                        actual_eob, expected_eob,
                        "{tx_size:?} {tx_type:?} case {case}"
                    );
                    assert_eq!(actual_q, expected_q, "{tx_size:?} {tx_type:?} case {case}");
                    assert_eq!(
                        actual_dq, expected_dq,
                        "{tx_size:?} {tx_type:?} case {case}"
                    );
                    actual_q.fill(99);
                    actual_dq.fill(99);
                    let direct_eob = vp9_quantize_fp_avx2(
                        &coeff,
                        n,
                        skip_block,
                        &round,
                        &quant,
                        &mut actual_q,
                        &mut actual_dq,
                        &dequant,
                        order.scan(),
                    );
                    assert_eq!(direct_eob, expected_eob, "direct {tx_size:?} case {case}");
                    assert_eq!(actual_q, expected_q, "direct {tx_size:?} case {case}");
                    assert_eq!(actual_dq, expected_dq, "direct {tx_size:?} case {case}");
                    if case == 0 || case >= 3 {
                        assert_eq!(expected_eob, 0, "{tx_size:?} {tx_type:?} case {case}");
                    } else {
                        assert_ne!(expected_eob, 0, "{tx_size:?} {tx_type:?} case {case}");
                    }
                }
            }
        }
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
            let actual_eob = vp9_quantize_fp_avx2_with_order(
                &coeff[..n],
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut actual_q,
                &mut actual_dq,
                &qp.dequant,
                scan.scan(),
                scan.iscan(),
            );
            assert_eq!(actual_eob, expected_eob, "{tx_size:?} eob");
            assert_eq!(actual_q, expected_q, "{tx_size:?} qcoeff");
            assert_eq!(actual_dq, expected_dq, "{tx_size:?} dqcoeff");
        }
    }

    #[test]
    fn iscan_eob_reduction_handles_sparse_boundaries() {
        require_avx2();
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        for (tx_size, n) in [(TxSize::Tx8x8, 64), (TxSize::Tx16x16, 256)] {
            for tx_type in [TxType::DctDct, TxType::DctAdst, TxType::AdstDct] {
                let order = scan_for(tx_size, tx_type);
                let mut coeff = vec![0i32; n];
                for &raster in [0usize, n / 2, n - 1].iter() {
                    coeff[raster] = 32_000;
                }
                let mut expected_q = vec![0i16; n];
                let mut expected_dq = vec![0i16; n];
                let mut actual_q = vec![0i16; n];
                let mut actual_dq = vec![0i16; n];
                let expected_eob = vp9_quantize_fp(
                    &coeff,
                    n,
                    false,
                    &qp.round_fp,
                    &qp.quant_fp,
                    &mut expected_q,
                    &mut expected_dq,
                    &qp.dequant,
                    order.scan(),
                );
                let actual_eob = vp9_quantize_fp_avx2_with_order(
                    &coeff,
                    n,
                    false,
                    &qp.round_fp,
                    &qp.quant_fp,
                    &mut actual_q,
                    &mut actual_dq,
                    &qp.dequant,
                    order.scan(),
                    order.iscan(),
                );
                assert_eq!(actual_eob, expected_eob, "{tx_size:?} {tx_type:?} eob");
                assert_eq!(actual_q, expected_q, "{tx_size:?} {tx_type:?} qcoeff");
                assert_eq!(actual_dq, expected_dq, "{tx_size:?} {tx_type:?} dqcoeff");
            }
        }
    }

    #[test]
    fn iscan_eob_vector_fallback_is_exercised() {
        require_avx2();
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        for (tx_size, n) in [(TxSize::Tx8x8, 64), (TxSize::Tx16x16, 256)] {
            let order = scan_for(tx_size, TxType::DctDct);
            let mut coeff = vec![0i32; n];
            for &scan_position in [0usize, n / 2].iter() {
                coeff[order.scan()[scan_position] as usize] = 32_000;
            }
            let mut expected_q = vec![0i16; n];
            let mut expected_dq = vec![0i16; n];
            let mut actual_q = vec![0i16; n];
            let mut actual_dq = vec![0i16; n];
            let expected_eob = vp9_quantize_fp(
                &coeff,
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut expected_q,
                &mut expected_dq,
                &qp.dequant,
                order.scan(),
            );
            let actual_eob = vp9_quantize_fp_avx2_with_order(
                &coeff,
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut actual_q,
                &mut actual_dq,
                &qp.dequant,
                order.scan(),
                order.iscan(),
            );
            assert_eq!(expected_eob, (n / 2 + 1) as u16, "{tx_size:?} oracle eob");
            assert_eq!(actual_eob, expected_eob, "{tx_size:?} eob");
            assert_eq!(actual_q, expected_q, "{tx_size:?} qcoeff");
            assert_eq!(actual_dq, expected_dq, "{tx_size:?} dqcoeff");
        }
    }

    #[test]
    fn iscan_zero_chunk_skip_matches_scalar_for_sparse_multichunk_blocks() {
        assert!(skip_zero_chunk(1));
        assert!(!skip_zero_chunk(0));

        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        for (tx_size, n) in [(TxSize::Tx8x8, 64), (TxSize::Tx16x16, 256)] {
            let order = scan_for(tx_size, TxType::DctDct);
            let tail = &order.scan()[n - 16..n];
            let mut coeff = vec![0i32; n];
            for (index, chunk) in [0, n / 32].into_iter().enumerate() {
                let raster = (chunk * 16..(chunk + 1) * 16)
                    .find(|&raster| !tail.contains(&(raster as i16)))
                    .expect("selected raster chunk has a position before the tail probe");
                coeff[raster] = if index == 0 { 32_000 } else { -32_000 };
            }

            let mut expected_q = vec![0i16; n];
            let mut expected_dq = vec![0i16; n];
            let expected_eob = vp9_quantize_fp(
                &coeff,
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut expected_q,
                &mut expected_dq,
                &qp.dequant,
                order.scan(),
            );
            assert_eq!(
                expected_q
                    .chunks_exact(16)
                    .filter(|chunk| chunk.iter().any(|&value| value != 0))
                    .count(),
                2,
                "{tx_size:?} nonzero chunks"
            );
            assert!(expected_eob <= (n - 16) as u16, "{tx_size:?} tail probe");

            // Hosts without AVX2 still verify the fixture and skip decision;
            // run the full differential wherever the hardware supports it.
            if !is_x86_feature_detected!("avx2") {
                continue;
            }
            let mut actual_q = vec![0i16; n];
            let mut actual_dq = vec![0i16; n];
            let actual_eob = vp9_quantize_fp_avx2_with_order(
                &coeff,
                n,
                false,
                &qp.round_fp,
                &qp.quant_fp,
                &mut actual_q,
                &mut actual_dq,
                &qp.dequant,
                order.scan(),
                order.iscan(),
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
