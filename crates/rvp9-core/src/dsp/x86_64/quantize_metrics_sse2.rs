//! SSE2 quantizer metrics for the realtime Hadamard picker.
//!
//! The arithmetic mirrors the pinned low-bit-depth `vp9_quantize_fp` and
//! `vp9_quantize_fp_32x32` kernels, but retains only the nonzero, SATD, and
//! coefficient-error outputs consumed by mode decision.

use core::arch::x86_64::*;

/// Fused outputs needed by coefficient-domain Hadamard ranking.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HadamardQuantMetrics {
    /// Whether any quantized coefficient is nonzero.
    pub nonzero: bool,
    /// Sum of absolute quantized coefficients.
    pub satd: u32,
    /// Sum of squared `(coeff - dqcoeff)` differences.
    pub error: i64,
}

/// Fused outputs needed by model-distortion Hadamard ranking.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HadamardQuantRateMetrics {
    /// Whether any quantized coefficient is nonzero.
    pub nonzero: bool,
    /// Sum of absolute quantized coefficients.
    pub satd: u32,
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn abs_saturating_i16(values: __m128i) -> __m128i {
    _mm_max_epi16(values, _mm_subs_epi16(_mm_setzero_si128(), values))
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn restore_sign(magnitude: __m128i, source: __m128i) -> __m128i {
    let sign = _mm_srai_epi16(source, 15);
    _mm_sub_epi16(_mm_xor_si128(magnitude, sign), sign)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn quantize_fp_magnitude(coeff: __m128i, round: __m128i, quant: __m128i) -> __m128i {
    let abs = abs_saturating_i16(coeff);
    let rounded = _mm_adds_epi16(abs, round);
    _mm_mulhi_epu16(rounded, quant)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn quantize_fp_32_magnitude(
    coeff: __m128i,
    round: __m128i,
    quant: __m128i,
    gate: __m128i,
) -> __m128i {
    let abs = abs_saturating_i16(coeff);
    let keep = _mm_cmpgt_epi16(abs, _mm_sub_epi16(gate, _mm_set1_epi16(1)));
    let rounded = _mm_adds_epi16(abs, round);
    let high = _mm_mulhi_epu16(rounded, quant);
    let low = _mm_mullo_epi16(rounded, quant);
    let magnitude = _mm_or_si128(_mm_slli_epi16(high, 1), _mm_srli_epi16(low, 15));
    _mm_and_si128(magnitude, keep)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn dc_ac(values: &[i16; 2]) -> __m128i {
    _mm_setr_epi16(
        values[0], values[1], values[1], values[1], values[1], values[1], values[1], values[1],
    )
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn difference_fits_i16(coeff: __m128i, dq: __m128i, diff: __m128i) -> bool {
    _mm_movemask_epi8(_mm_cmpeq_epi16(diff, _mm_subs_epi16(coeff, dq))) == 0xffff
}

struct MetricAccumulator {
    satd: __m128i,
    error_even_lo: __m128i,
    error_odd_lo: __m128i,
    error_even_hi: __m128i,
    error_odd_hi: __m128i,
}

impl MetricAccumulator {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn new() -> Self {
        let zero = _mm_setzero_si128();
        Self {
            satd: zero,
            error_even_lo: zero,
            error_odd_lo: zero,
            error_even_hi: zero,
            error_odd_hi: zero,
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add(&mut self, coeff: __m128i, dq: __m128i, abs_q: __m128i) {
        let zero = _mm_setzero_si128();
        self.satd = _mm_add_epi32(
            self.satd,
            _mm_add_epi32(
                _mm_unpacklo_epi16(abs_q, zero),
                _mm_unpackhi_epi16(abs_q, zero),
            ),
        );

        let narrow_diff = _mm_sub_epi16(coeff, dq);
        if difference_fits_i16(coeff, dq, narrow_diff) {
            let pair_squares = _mm_madd_epi16(narrow_diff, narrow_diff);
            // A pair of i16::MIN values squares to 0x80000000, so widen as u32.
            self.error_even_lo =
                _mm_add_epi64(self.error_even_lo, _mm_unpacklo_epi32(pair_squares, zero));
            self.error_odd_lo =
                _mm_add_epi64(self.error_odd_lo, _mm_unpackhi_epi32(pair_squares, zero));
            return;
        }

        let coeff_sign = _mm_srai_epi16(coeff, 15);
        let dq_sign = _mm_srai_epi16(dq, 15);
        let diff_lo = _mm_sub_epi32(
            _mm_unpacklo_epi16(coeff, coeff_sign),
            _mm_unpacklo_epi16(dq, dq_sign),
        );
        let diff_hi = _mm_sub_epi32(
            _mm_unpackhi_epi16(coeff, coeff_sign),
            _mm_unpackhi_epi16(dq, dq_sign),
        );
        let abs_diff_lo = {
            let sign = _mm_srai_epi32(diff_lo, 31);
            _mm_sub_epi32(_mm_xor_si128(diff_lo, sign), sign)
        };
        let abs_diff_hi = {
            let sign = _mm_srai_epi32(diff_hi, 31);
            _mm_sub_epi32(_mm_xor_si128(diff_hi, sign), sign)
        };
        self.error_even_lo =
            _mm_add_epi64(self.error_even_lo, _mm_mul_epu32(abs_diff_lo, abs_diff_lo));
        self.error_odd_lo = _mm_add_epi64(
            self.error_odd_lo,
            _mm_mul_epu32(
                _mm_srli_si128(abs_diff_lo, 4),
                _mm_srli_si128(abs_diff_lo, 4),
            ),
        );
        self.error_even_hi =
            _mm_add_epi64(self.error_even_hi, _mm_mul_epu32(abs_diff_hi, abs_diff_hi));
        self.error_odd_hi = _mm_add_epi64(
            self.error_odd_hi,
            _mm_mul_epu32(
                _mm_srli_si128(abs_diff_hi, 4),
                _mm_srli_si128(abs_diff_hi, 4),
            ),
        );
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn finish(self) -> HadamardQuantMetrics {
        let mut satd_lanes = [0u32; 4];
        let error = _mm_add_epi64(
            _mm_add_epi64(self.error_even_lo, self.error_odd_lo),
            _mm_add_epi64(self.error_even_hi, self.error_odd_hi),
        );
        let mut error_lanes = [0u64; 2];
        // SAFETY: Both destinations provide a complete unaligned 128-bit store.
        unsafe {
            _mm_storeu_si128(satd_lanes.as_mut_ptr().cast(), self.satd);
            _mm_storeu_si128(error_lanes.as_mut_ptr().cast(), error);
        }
        let satd = satd_lanes.into_iter().sum();
        HadamardQuantMetrics {
            nonzero: satd != 0,
            satd,
            error: error_lanes.into_iter().sum::<u64>() as i64,
        }
    }
}

#[target_feature(enable = "sse2")]
unsafe fn metrics_inner(
    coeff: *const i16,
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantMetrics {
    let mut accumulator = MetricAccumulator::new();
    let mut offset = 0;
    while offset < n_coeffs {
        // SAFETY: The wrapper proves each eight-coefficient load is in bounds.
        let values = unsafe { _mm_loadu_si128(coeff.add(offset).cast()) };
        let (round_values, quant_values, dequant_values) = if offset == 0 {
            (dc_ac(round), dc_ac(quant), dc_ac(dequant))
        } else {
            (
                _mm_set1_epi16(round[1]),
                _mm_set1_epi16(quant[1]),
                _mm_set1_epi16(dequant[1]),
            )
        };
        let magnitude = quantize_fp_magnitude(values, round_values, quant_values);
        let q = restore_sign(magnitude, values);
        let dq = _mm_mullo_epi16(q, dequant_values);
        accumulator.add(values, dq, magnitude);
        offset += 8;
    }
    accumulator.finish()
}

#[target_feature(enable = "sse2")]
unsafe fn rate_metrics_inner(
    coeff: *const i16,
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
) -> HadamardQuantRateMetrics {
    let zero = _mm_setzero_si128();
    let mut satd_lanes = zero;
    let mut offset = 0;
    while offset < n_coeffs {
        // SAFETY: The wrapper proves each eight-coefficient load is in bounds.
        let values = unsafe { _mm_loadu_si128(coeff.add(offset).cast()) };
        let (round_values, quant_values) = if offset == 0 {
            (dc_ac(round), dc_ac(quant))
        } else {
            (_mm_set1_epi16(round[1]), _mm_set1_epi16(quant[1]))
        };
        let magnitude = quantize_fp_magnitude(values, round_values, quant_values);
        satd_lanes = _mm_add_epi32(
            satd_lanes,
            _mm_add_epi32(
                _mm_unpacklo_epi16(magnitude, zero),
                _mm_unpackhi_epi16(magnitude, zero),
            ),
        );
        offset += 8;
    }
    let mut lanes = [0u32; 4];
    // SAFETY: `lanes` provides a complete unaligned 128-bit store.
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), satd_lanes) };
    let satd = lanes.into_iter().sum();
    HadamardQuantRateMetrics {
        nonzero: satd != 0,
        satd,
    }
}

#[target_feature(enable = "sse2")]
unsafe fn metrics_32_inner(
    coeff: *const i16,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantMetrics {
    let round_dc_ac = [((round[0] + 1) >> 1), ((round[1] + 1) >> 1)];
    let gate_dc_ac = [dequant[0] >> 2, dequant[1] >> 2];
    let mut satd = 0u32;
    let mut error = 0i64;
    for offset in (0..1024).step_by(8) {
        // SAFETY: The wrapper proves each eight-coefficient load is in bounds.
        let values = unsafe { _mm_loadu_si128(coeff.add(offset).cast()) };
        let (round_values, quant_values, gate_values) = if offset == 0 {
            (dc_ac(&round_dc_ac), dc_ac(quant), dc_ac(&gate_dc_ac))
        } else {
            (
                _mm_set1_epi16(round_dc_ac[1]),
                _mm_set1_epi16(quant[1]),
                _mm_set1_epi16(gate_dc_ac[1]),
            )
        };
        let magnitude = quantize_fp_32_magnitude(values, round_values, quant_values, gate_values);
        let q = restore_sign(magnitude, values);
        let mut source_lanes = [0i16; 8];
        let mut q_lanes = [0i16; 8];
        let mut magnitude_lanes = [0u16; 8];
        // SAFETY: Each destination provides a complete unaligned 128-bit store.
        unsafe {
            _mm_storeu_si128(source_lanes.as_mut_ptr().cast(), values);
            _mm_storeu_si128(q_lanes.as_mut_ptr().cast(), q);
            _mm_storeu_si128(magnitude_lanes.as_mut_ptr().cast(), magnitude);
        }
        for lane in 0..8 {
            let band = usize::from(offset + lane != 0);
            let dq = ((i32::from(q_lanes[lane]) * i32::from(dequant[band])) / 2) as i16;
            let diff = i32::from(source_lanes[lane]) - i32::from(dq);
            satd += u32::from(magnitude_lanes[lane]);
            error += i64::from(diff) * i64::from(diff);
        }
    }
    HadamardQuantMetrics {
        nonzero: satd != 0,
        satd,
        error,
    }
}

#[target_feature(enable = "sse2")]
unsafe fn rate_metrics_32_inner(
    coeff: *const i16,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantRateMetrics {
    let round_dc_ac = [((round[0] + 1) >> 1), ((round[1] + 1) >> 1)];
    let gate_dc_ac = [dequant[0] >> 2, dequant[1] >> 2];
    let zero = _mm_setzero_si128();
    let mut satd_lanes = zero;
    for offset in (0..1024).step_by(8) {
        // SAFETY: The wrapper proves each eight-coefficient load is in bounds.
        let values = unsafe { _mm_loadu_si128(coeff.add(offset).cast()) };
        let (round_values, quant_values, gate_values) = if offset == 0 {
            (dc_ac(&round_dc_ac), dc_ac(quant), dc_ac(&gate_dc_ac))
        } else {
            (
                _mm_set1_epi16(round_dc_ac[1]),
                _mm_set1_epi16(quant[1]),
                _mm_set1_epi16(gate_dc_ac[1]),
            )
        };
        let magnitude = quantize_fp_32_magnitude(values, round_values, quant_values, gate_values);
        satd_lanes = _mm_add_epi32(
            satd_lanes,
            _mm_add_epi32(
                _mm_unpacklo_epi16(magnitude, zero),
                _mm_unpackhi_epi16(magnitude, zero),
            ),
        );
    }
    let mut lanes = [0u32; 4];
    // SAFETY: `lanes` provides a complete unaligned 128-bit store.
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), satd_lanes) };
    let satd = lanes.into_iter().sum();
    HadamardQuantRateMetrics {
        nonzero: satd != 0,
        satd,
    }
}

/// Quantize 64 or 256 Hadamard coefficients and return fused RD metrics.
pub fn vp9_quantize_fp_i16_metrics_sse2(
    coeff: &[i16],
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantMetrics {
    assert!(matches!(n_coeffs, 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    // SAFETY: x86-64 guarantees SSE2; assertions prove every vector load.
    unsafe { metrics_inner(coeff.as_ptr(), n_coeffs, round, quant, dequant) }
}

/// Quantize 64 or 256 Hadamard coefficients and return rate-only metrics.
pub fn vp9_quantize_fp_i16_rate_metrics_sse2(
    coeff: &[i16],
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
) -> HadamardQuantRateMetrics {
    assert!(matches!(n_coeffs, 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    // SAFETY: x86-64 guarantees SSE2; assertions prove every vector load.
    unsafe { rate_metrics_inner(coeff.as_ptr(), n_coeffs, round, quant) }
}

/// Quantize 1024 Hadamard coefficients with 32x32 scaling and return RD metrics.
pub fn vp9_quantize_fp_32x32_i16_metrics_sse2(
    coeff: &[i16],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantMetrics {
    assert!(coeff.len() >= 1024);
    // SAFETY: x86-64 guarantees SSE2; the assertion proves every vector load.
    unsafe { metrics_32_inner(coeff.as_ptr(), round, quant, dequant) }
}

/// Quantize 1024 Hadamard coefficients with 32x32 scaling and return rate metrics.
pub fn vp9_quantize_fp_32x32_i16_rate_metrics_sse2(
    coeff: &[i16],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
) -> HadamardQuantRateMetrics {
    assert!(coeff.len() >= 1024);
    // SAFETY: x86-64 guarantees SSE2; the assertion proves every vector load.
    unsafe { rate_metrics_32_inner(coeff.as_ptr(), round, quant, dequant) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode_decision::block_yrd::{block_error_fp_i32, satd_i16};
    use crate::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, vp9_quantize_fp_32x32_with_order,
        vp9_quantize_fp_with_order,
    };
    use crate::transforms::{TxSize, TxType, scan_for};

    #[test]
    fn pairwise_error_matches_scalar_and_falls_back_on_overflow() {
        /// (coeff, dqcoeff, |qcoeff|, whether coeff - dqcoeff fits in i16)
        type PairwiseCase = ([i16; 8], [i16; 8], [i16; 8], bool);
        let cases: [PairwiseCase; 3] = [
            (
                [15, -25, 1000, -1000, 321, -321, i16::MAX, i16::MIN],
                [5, -5, 500, -500, 1, -1, i16::MAX - 1, i16::MIN + 1],
                [1, 2, 3, 4, 5, 6, 7, 8],
                true,
            ),
            ([i16::MIN, i16::MIN, 0, 1, 2, 3, 4, 5], [0; 8], [0; 8], true),
            (
                [i16::MAX, i16::MIN, 1, -2, 3, -4, 5, -6],
                [-1, i16::MAX, 0, 0, 0, 0, 0, 0],
                [9, 0, 0, 1, 0, 2, 0, 3],
                false,
            ),
        ];
        let mut expected_error = 0i64;
        let mut expected_satd = 0u32;
        // SAFETY: x86-64 guarantees SSE2, and each array provides eight i16 lanes.
        let actual = unsafe {
            let mut accumulator = MetricAccumulator::new();
            for (coeff, dq, abs_q, expected_narrow) in cases {
                let coeff_v = _mm_loadu_si128(coeff.as_ptr().cast());
                let dq_v = _mm_loadu_si128(dq.as_ptr().cast());
                let abs_q_v = _mm_loadu_si128(abs_q.as_ptr().cast());
                let diff = _mm_sub_epi16(coeff_v, dq_v);
                assert_eq!(difference_fits_i16(coeff_v, dq_v, diff), expected_narrow);
                accumulator.add(coeff_v, dq_v, abs_q_v);
                expected_error += coeff
                    .into_iter()
                    .zip(dq)
                    .map(|(c, d)| {
                        let difference = i64::from(c) - i64::from(d);
                        difference * difference
                    })
                    .sum::<i64>();
                expected_satd += abs_q.into_iter().map(|q| q as u32).sum::<u32>();
            }
            accumulator.finish()
        };
        assert_eq!(actual.error, expected_error);
        assert_eq!(actual.satd, expected_satd);
        assert!(actual.nonzero);
    }

    fn scalar_metrics(coeff: &[i16], tx_size: TxSize, q_index: i32) -> HadamardQuantMetrics {
        let quantizer = init_plane_quantizers(q_index, QuantizerDeltas::Y);
        let n = tx_size.n_coeffs();
        let wide: Vec<i32> = coeff[..n].iter().copied().map(i32::from).collect();
        let mut qcoeff = vec![0i16; n];
        let mut dqcoeff = vec![0i16; n];
        let order = scan_for(tx_size, TxType::DctDct);
        if tx_size == TxSize::Tx32x32 {
            let _ = vp9_quantize_fp_32x32_with_order(
                &wide,
                false,
                &quantizer.round_fp,
                &quantizer.quant_fp,
                &mut qcoeff,
                &mut dqcoeff,
                &quantizer.dequant,
                &order,
            );
        } else {
            let _ = vp9_quantize_fp_with_order(
                &wide,
                n,
                false,
                &quantizer.round_fp,
                &quantizer.quant_fp,
                &mut qcoeff,
                &mut dqcoeff,
                &quantizer.dequant,
                &order,
            );
        }
        let satd = satd_i16(&qcoeff);
        HadamardQuantMetrics {
            nonzero: satd != 0,
            satd,
            error: block_error_fp_i32(&wide, &dqcoeff),
        }
    }

    #[test]
    fn sse2_matches_scalar_across_sizes_quantizers_and_extremes() {
        let mut state = 0x5155_414eu32;
        for tx_size in [TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            let n = tx_size.n_coeffs();
            for q_index in [0, 1, 31, 63, 127, 191, 255] {
                let quantizer = init_plane_quantizers(q_index, QuantizerDeltas::Y);
                let mut coeff = vec![0i16; n];
                for value in &mut coeff {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *value = ((state >> 16) % 32_641) as i16 - 16_320;
                }
                coeff[0] = -16_320;
                coeff[n - 1] = 16_320;
                let expected = scalar_metrics(&coeff, tx_size, q_index);
                let actual = if tx_size == TxSize::Tx32x32 {
                    vp9_quantize_fp_32x32_i16_metrics_sse2(
                        &coeff,
                        &quantizer.round_fp,
                        &quantizer.quant_fp,
                        &quantizer.dequant,
                    )
                } else {
                    vp9_quantize_fp_i16_metrics_sse2(
                        &coeff,
                        n,
                        &quantizer.round_fp,
                        &quantizer.quant_fp,
                        &quantizer.dequant,
                    )
                };
                assert_eq!(actual, expected, "mismatch at {tx_size:?}, q={q_index}");

                let rate = if tx_size == TxSize::Tx32x32 {
                    vp9_quantize_fp_32x32_i16_rate_metrics_sse2(
                        &coeff,
                        &quantizer.round_fp,
                        &quantizer.quant_fp,
                        &quantizer.dequant,
                    )
                } else {
                    vp9_quantize_fp_i16_rate_metrics_sse2(
                        &coeff,
                        n,
                        &quantizer.round_fp,
                        &quantizer.quant_fp,
                    )
                };
                assert_eq!(rate.nonzero, expected.nonzero);
                assert_eq!(rate.satd, expected.satd);
            }
        }
    }
}
