//! NEON fast-path VP9 forward quantization.
//!
//! This mirrors libvpx `vp9/encoder/arm/neon/vp9_quantize_neon.c` while
//! accepting rvp9's `i32` transform coefficients. Coefficients are processed
//! in raster order; the inverse scan supplies the exact one-based EOB
//! position.

use core::arch::aarch64::*;

/// Fused outputs needed by the realtime Hadamard rate-distortion model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HadamardQuantMetrics {
    /// One-based final nonzero scan position.
    pub eob: u16,
    /// Sum of absolute quantized coefficients.
    pub satd: u32,
    /// Sum of squared `(coeff - dqcoeff)` differences.
    pub error: i64,
}

/// Fused outputs needed by the normal pixel-domain realtime Hadamard model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HadamardQuantRateMetrics {
    /// One-based final nonzero scan position.
    pub eob: u16,
    /// Sum of absolute quantized coefficients.
    pub satd: u32,
}

struct HadamardMetricAccumulator {
    eob_max: int16x8_t,
    satd: uint32x4_t,
    error_0: int64x2_t,
    error_1: int64x2_t,
    error_2: int64x2_t,
    error_3: int64x2_t,
}

struct HadamardRateAccumulator {
    eob_max: int16x8_t,
    satd: uint32x4_t,
}

impl HadamardRateAccumulator {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn new() -> Self {
        Self {
            eob_max: vdupq_n_s16(0),
            satd: vdupq_n_u32(0),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add(&mut self, q: int16x8_t, iscan: *const i16) {
        self.satd = vpadalq_u16(self.satd, vreinterpretq_u16_s16(vabsq_s16(q)));

        let nonzero = vmvnq_u16(vceqq_s16(q, vdupq_n_s16(0)));
        let eob = vbslq_s16(nonzero, vld1q_s16(iscan), vdupq_n_s16(0));
        self.eob_max = vmaxq_s16(self.eob_max, eob);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn finish(self) -> HadamardQuantRateMetrics {
        HadamardQuantRateMetrics {
            eob: vmaxvq_s16(self.eob_max) as u16,
            satd: vaddvq_u32(self.satd),
        }
    }
}

impl HadamardMetricAccumulator {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn new() -> Self {
        Self {
            eob_max: vdupq_n_s16(0),
            satd: vdupq_n_u32(0),
            error_0: vdupq_n_s64(0),
            error_1: vdupq_n_s64(0),
            error_2: vdupq_n_s64(0),
            error_3: vdupq_n_s64(0),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add(
        &mut self,
        coeff_lo: int32x4_t,
        coeff_hi: int32x4_t,
        q: int16x8_t,
        dq: int16x8_t,
        iscan: *const i16,
    ) {
        self.satd = vpadalq_u16(self.satd, vreinterpretq_u16_s16(vabsq_s16(q)));

        let diff_lo = vsubq_s32(coeff_lo, vmovl_s16(vget_low_s16(dq)));
        let diff_hi = vsubq_s32(coeff_hi, vmovl_s16(vget_high_s16(dq)));
        self.error_0 = vmlal_s32(self.error_0, vget_low_s32(diff_lo), vget_low_s32(diff_lo));
        self.error_1 = vmlal_s32(self.error_1, vget_high_s32(diff_lo), vget_high_s32(diff_lo));
        self.error_2 = vmlal_s32(self.error_2, vget_low_s32(diff_hi), vget_low_s32(diff_hi));
        self.error_3 = vmlal_s32(self.error_3, vget_high_s32(diff_hi), vget_high_s32(diff_hi));

        let nonzero = vmvnq_u16(vceqq_s16(q, vdupq_n_s16(0)));
        let eob = vbslq_s16(nonzero, vld1q_s16(iscan), vdupq_n_s16(0));
        self.eob_max = vmaxq_s16(self.eob_max, eob);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn finish(self) -> HadamardQuantMetrics {
        HadamardQuantMetrics {
            eob: vmaxvq_s16(self.eob_max) as u16,
            satd: vaddvq_u32(self.satd),
            error: vaddvq_s64(vaddq_s64(
                vaddq_s64(self.error_0, self.error_1),
                vaddq_s64(self.error_2, self.error_3),
            )),
        }
    }
}

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

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_i16_8(
    coeff: *const i16,
    iscan: *const i16,
    round: int16x8_t,
    quant: int16x8_t,
    dequant: int16x8_t,
    metrics: &mut HadamardMetricAccumulator,
) {
    let coeff = vld1q_s16(coeff);
    let sign = vshrq_n_s16::<15>(coeff);
    let rounded = vqaddq_s16(vqabsq_s16(coeff), round);
    // vqdmulh computes floor(rounded * quant / 2^15). Halving that
    // non-negative result exactly matches floor(rounded * quant / 2^16).
    let abs_q = vshrq_n_s16::<1>(vqdmulhq_s16(rounded, quant));
    let q = vsubq_s16(veorq_s16(abs_q, sign), sign);

    let coeff_lo = vmovl_s16(vget_low_s16(coeff));
    let coeff_hi = vmovl_s16(vget_high_s16(coeff));

    // The scalar result casts `q * dequant` back to i16, so only the low
    // product half is observable. Multiply all eight lanes directly.
    let dq = vmulq_s16(q, dequant);
    metrics.add(coeff_lo, coeff_hi, q, dq, iscan);
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_i16_8_rate(
    coeff: *const i16,
    iscan: *const i16,
    round: int16x8_t,
    quant: int16x8_t,
    metrics: &mut HadamardRateAccumulator,
) {
    let coeff = vld1q_s16(coeff);
    let sign = vshrq_n_s16::<15>(coeff);
    let rounded = vqaddq_s16(vqabsq_s16(coeff), round);
    let abs_q = vshrq_n_s16::<1>(vqdmulhq_s16(rounded, quant));
    let q = vsubq_s16(veorq_s16(abs_q, sign), sign);
    metrics.add(q, iscan);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_i16_metrics_neon_inner(
    coeff: *const i16,
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: *const i16,
) -> HadamardQuantMetrics {
    let dc_ac_i16 = |values: &[i16; 2]| {
        vld1q_s16(
            [
                values[0], values[1], values[1], values[1], values[1], values[1], values[1],
                values[1],
            ]
            .as_ptr(),
        )
    };
    let ac_i16 = |value: i16| vdupq_n_s16(value);
    let dequant_dc_ac = vld1q_s16(
        [
            dequant[0], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1],
            dequant[1],
        ]
        .as_ptr(),
    );
    let dequant_ac = vdupq_n_s16(dequant[1]);
    let mut metrics = HadamardMetricAccumulator::new();

    quantize_i16_8(
        coeff,
        iscan,
        dc_ac_i16(round),
        dc_ac_i16(quant),
        dequant_dc_ac,
        &mut metrics,
    );
    let round_ac = ac_i16(round[1]);
    let quant_ac = ac_i16(quant[1]);
    let mut i = 8;
    while i < n_coeffs {
        quantize_i16_8(
            coeff.add(i),
            iscan.add(i),
            round_ac,
            quant_ac,
            dequant_ac,
            &mut metrics,
        );
        i += 8;
    }

    metrics.finish()
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_i16_rate_metrics_neon_inner(
    coeff: *const i16,
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    iscan: *const i16,
) -> HadamardQuantRateMetrics {
    let dc_ac_i16 = |values: &[i16; 2]| {
        vld1q_s16(
            [
                values[0], values[1], values[1], values[1], values[1], values[1], values[1],
                values[1],
            ]
            .as_ptr(),
        )
    };
    let ac_i16 = |value: i16| vdupq_n_s16(value);
    let mut metrics = HadamardRateAccumulator::new();

    quantize_i16_8_rate(
        coeff,
        iscan,
        dc_ac_i16(round),
        dc_ac_i16(quant),
        &mut metrics,
    );
    let round_ac = ac_i16(round[1]);
    let quant_ac = ac_i16(quant[1]);
    let mut i = 8;
    while i < n_coeffs {
        quantize_i16_8_rate(coeff.add(i), iscan.add(i), round_ac, quant_ac, &mut metrics);
        i += 8;
    }

    metrics.finish()
}

/// Quantize 64 or 256 Hadamard coefficients and return fused RD metrics.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_fp_i16_metrics_neon(
    coeff: &[i16],
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> HadamardQuantMetrics {
    assert!(matches!(n_coeffs, 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    assert!(iscan.len() >= n_coeffs);

    // SAFETY: NEON is mandatory on AArch64. The slice assertions prove
    // every eight-lane load; coefficients widen to i32 before
    // absolute value, preserving the i16::MIN behavior of the i32 oracle.
    unsafe {
        vp9_quantize_fp_i16_metrics_neon_inner(
            coeff.as_ptr(),
            n_coeffs,
            round,
            quant,
            dequant,
            iscan.as_ptr(),
        )
    }
}

/// Quantize 64 or 256 Hadamard coefficients and return EOB/rate metrics.
pub fn vp9_quantize_fp_i16_rate_metrics_neon(
    coeff: &[i16],
    n_coeffs: usize,
    round: &[i16; 2],
    quant: &[i16; 2],
    iscan: &[i16],
) -> HadamardQuantRateMetrics {
    assert!(matches!(n_coeffs, 64 | 256));
    assert!(coeff.len() >= n_coeffs);
    assert!(iscan.len() >= n_coeffs);

    // SAFETY: NEON is mandatory on AArch64. The slice assertions prove
    // every eight-lane load; coefficients widen to i32 before absolute
    // value, preserving the i16::MIN behavior of the full-metrics oracle.
    unsafe {
        vp9_quantize_fp_i16_rate_metrics_neon_inner(
            coeff.as_ptr(),
            n_coeffs,
            round,
            quant,
            iscan.as_ptr(),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_fp_32x32_8(
    coeff: *const i32,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    iscan: *const i16,
    round_lo: int32x4_t,
    round_hi: int32x4_t,
    quant_lo: int32x4_t,
    quant_hi: int32x4_t,
    dequant_lo: int32x4_t,
    dequant_hi: int32x4_t,
    gate_lo: int32x4_t,
    gate_hi: int32x4_t,
    eob_max: &mut int16x8_t,
) {
    let coeff_lo = vld1q_s32(coeff);
    let coeff_hi = vld1q_s32(coeff.add(4));
    let sign_lo = vshrq_n_s32::<31>(coeff_lo);
    let sign_hi = vshrq_n_s32::<31>(coeff_hi);
    let abs_lo = vabsq_s32(coeff_lo);
    let abs_hi = vabsq_s32(coeff_hi);
    let keep_lo = vcgeq_s32(abs_lo, gate_lo);
    let keep_hi = vcgeq_s32(abs_hi, gate_hi);
    let max = vdupq_n_s32(i16::MAX as i32);

    let rounded_lo = vminq_s32(vaddq_s32(abs_lo, round_lo), max);
    let rounded_hi = vminq_s32(vaddq_s32(abs_hi, round_hi), max);
    let abs_q_lo = vshrq_n_s32::<15>(vmulq_s32(rounded_lo, quant_lo));
    let abs_q_hi = vshrq_n_s32::<15>(vmulq_s32(rounded_hi, quant_hi));
    let signed_lo = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_lo, sign_lo), sign_lo),
        vreinterpretq_s32_u32(keep_lo),
    );
    let signed_hi = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_hi, sign_hi), sign_hi),
        vreinterpretq_s32_u32(keep_hi),
    );
    let q = vcombine_s16(vmovn_s32(signed_lo), vmovn_s32(signed_hi));
    vst1q_s16(qcoeff, q);

    let dq_lo = vmulq_s32(signed_lo, dequant_lo);
    let dq_hi = vmulq_s32(signed_hi, dequant_hi);
    let negative_lo = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_lo)));
    let negative_hi = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_hi)));
    let dq_lo = vshrq_n_s32::<1>(vaddq_s32(dq_lo, negative_lo));
    let dq_hi = vshrq_n_s32::<1>(vaddq_s32(dq_hi, negative_hi));
    vst1q_s16(dqcoeff, vcombine_s16(vmovn_s32(dq_lo), vmovn_s32(dq_hi)));

    let nonzero = vmvnq_u16(vceqq_s16(q, vdupq_n_s16(0)));
    let eob = vbslq_s16(nonzero, vld1q_s16(iscan), vdupq_n_s16(0));
    *eob_max = vmaxq_s16(*eob_max, eob);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_32x32_neon_inner(
    coeff: *const i32,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: &[i16; 2],
    iscan: *const i16,
) -> u16 {
    let dc_ac = |values: &[i16; 2]| {
        vld1q_s32(
            [
                i32::from(values[0]),
                i32::from(values[1]),
                i32::from(values[1]),
                i32::from(values[1]),
            ]
            .as_ptr(),
        )
    };
    let ac = |value: i16| vdupq_n_s32(i32::from(value));
    let round_dc_ac = vshrq_n_s32::<1>(vaddq_s32(dc_ac(round), vdupq_n_s32(1)));
    let round_ac = vdupq_n_s32((i32::from(round[1]) + 1) >> 1);
    let quant_dc_ac = dc_ac(quant);
    let quant_ac = ac(quant[1]);
    let dequant_dc_ac = dc_ac(dequant);
    let dequant_ac = ac(dequant[1]);
    let gate_dc_ac = vshrq_n_s32::<2>(dequant_dc_ac);
    let gate_ac = vdupq_n_s32(i32::from(dequant[1]) >> 2);
    let mut eob_max = vdupq_n_s16(0);

    quantize_fp_32x32_8(
        coeff,
        qcoeff,
        dqcoeff,
        iscan,
        round_dc_ac,
        round_ac,
        quant_dc_ac,
        quant_ac,
        dequant_dc_ac,
        dequant_ac,
        gate_dc_ac,
        gate_ac,
        &mut eob_max,
    );
    let mut i = 8;
    while i < 1024 {
        quantize_fp_32x32_8(
            coeff.add(i),
            qcoeff.add(i),
            dqcoeff.add(i),
            iscan.add(i),
            round_ac,
            round_ac,
            quant_ac,
            quant_ac,
            dequant_ac,
            dequant_ac,
            gate_ac,
            gate_ac,
            &mut eob_max,
        );
        i += 8;
    }

    vmaxvq_s16(eob_max) as u16
}

/// Quantize a 32x32 transform with the realtime fast quantizer.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_fp_32x32_neon(
    coeff: &[i32],
    skip_block: bool,
    round: &[i16; 2],
    quant: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> u16 {
    const N_COEFFS: usize = 1024;
    assert!(coeff.len() >= N_COEFFS);
    assert!(qcoeff.len() >= N_COEFFS);
    assert!(dqcoeff.len() >= N_COEFFS);
    assert!(iscan.len() >= N_COEFFS);

    if skip_block {
        qcoeff[..N_COEFFS].fill(0);
        dqcoeff[..N_COEFFS].fill(0);
        return 0;
    }

    // SAFETY: NEON is mandatory on AArch64. The slice assertions prove
    // every 8-lane load and store. Valid 8-bit transform coefficients are
    // i16-bounded, so all widened arithmetic remains within i32.
    unsafe {
        vp9_quantize_fp_32x32_neon_inner(
            coeff.as_ptr(),
            round,
            quant,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant,
            iscan.as_ptr(),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_fp_32x32_i16_8(
    coeff: *const i16,
    iscan: *const i16,
    round: int16x8_t,
    quant: int16x8_t,
    dequant: int16x8_t,
    gate: int16x8_t,
    metrics: &mut HadamardMetricAccumulator,
) {
    let coeff = vld1q_s16(coeff);
    let sign = vshrq_n_s16::<15>(coeff);
    let abs = vqabsq_s16(coeff);
    let keep = vcgeq_s16(abs, gate);
    let rounded = vqaddq_s16(abs, round);
    // The 32x32 quantizer's >> 15 scale is exactly the non-negative
    // saturating doubling-high product for the bounded i16 operands.
    let abs_q = vqdmulhq_s16(rounded, quant);
    let q = vandq_s16(
        vsubq_s16(veorq_s16(abs_q, sign), sign),
        vreinterpretq_s16_u16(keep),
    );

    let coeff_lo = vmovl_s16(vget_low_s16(coeff));
    let coeff_hi = vmovl_s16(vget_high_s16(coeff));
    let dq_lo = vmull_s16(vget_low_s16(q), vget_low_s16(dequant));
    let dq_hi = vmull_s16(vget_high_s16(q), vget_high_s16(dequant));
    let negative_lo = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_lo)));
    let negative_hi = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_hi)));
    let dq_lo = vshrq_n_s32::<1>(vaddq_s32(dq_lo, negative_lo));
    let dq_hi = vshrq_n_s32::<1>(vaddq_s32(dq_hi, negative_hi));
    let dq = vcombine_s16(vmovn_s32(dq_lo), vmovn_s32(dq_hi));
    metrics.add(coeff_lo, coeff_hi, q, dq, iscan);
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_fp_32x32_i16_8_rate(
    coeff: *const i16,
    iscan: *const i16,
    round: int16x8_t,
    quant: int16x8_t,
    gate: int16x8_t,
    metrics: &mut HadamardRateAccumulator,
) {
    let coeff = vld1q_s16(coeff);
    let sign = vshrq_n_s16::<15>(coeff);
    let abs = vqabsq_s16(coeff);
    let keep = vcgeq_s16(abs, gate);
    let rounded = vqaddq_s16(abs, round);
    let abs_q = vqdmulhq_s16(rounded, quant);
    let q = vandq_s16(
        vsubq_s16(veorq_s16(abs_q, sign), sign),
        vreinterpretq_s16_u16(keep),
    );
    metrics.add(q, iscan);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_32x32_i16_metrics_neon_inner(
    coeff: *const i16,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: *const i16,
) -> HadamardQuantMetrics {
    let dc_ac = |values: &[i16; 2]| {
        vld1q_s16(
            [
                values[0], values[1], values[1], values[1], values[1], values[1], values[1],
                values[1],
            ]
            .as_ptr(),
        )
    };
    let ac = |value: i16| vdupq_n_s16(value);
    let round_dc_ac = vshrq_n_s16::<1>(vaddq_s16(dc_ac(round), vdupq_n_s16(1)));
    let round_ac = vdupq_n_s16((round[1] + 1) >> 1);
    let quant_dc_ac = dc_ac(quant);
    let quant_ac = ac(quant[1]);
    let dequant_dc_ac = dc_ac(dequant);
    let dequant_ac = ac(dequant[1]);
    let gate_dc_ac = vshrq_n_s16::<2>(dequant_dc_ac);
    let gate_ac = vdupq_n_s16(dequant[1] >> 2);
    let mut metrics = HadamardMetricAccumulator::new();

    quantize_fp_32x32_i16_8(
        coeff,
        iscan,
        round_dc_ac,
        quant_dc_ac,
        dequant_dc_ac,
        gate_dc_ac,
        &mut metrics,
    );
    let mut i = 8;
    while i < 1024 {
        quantize_fp_32x32_i16_8(
            coeff.add(i),
            iscan.add(i),
            round_ac,
            quant_ac,
            dequant_ac,
            gate_ac,
            &mut metrics,
        );
        i += 8;
    }

    metrics.finish()
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_fp_32x32_i16_rate_metrics_neon_inner(
    coeff: *const i16,
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: *const i16,
) -> HadamardQuantRateMetrics {
    let dc_ac = |values: &[i16; 2]| {
        vld1q_s16(
            [
                values[0], values[1], values[1], values[1], values[1], values[1], values[1],
                values[1],
            ]
            .as_ptr(),
        )
    };
    let ac = |value: i16| vdupq_n_s16(value);
    let round_dc_ac = vshrq_n_s16::<1>(vaddq_s16(dc_ac(round), vdupq_n_s16(1)));
    let round_ac = vdupq_n_s16((round[1] + 1) >> 1);
    let quant_dc_ac = dc_ac(quant);
    let quant_ac = ac(quant[1]);
    let gate_dc_ac = vshrq_n_s16::<2>(dc_ac(dequant));
    let gate_ac = vdupq_n_s16(dequant[1] >> 2);
    let mut metrics = HadamardRateAccumulator::new();

    quantize_fp_32x32_i16_8_rate(
        coeff,
        iscan,
        round_dc_ac,
        quant_dc_ac,
        gate_dc_ac,
        &mut metrics,
    );
    let mut i = 8;
    while i < 1024 {
        quantize_fp_32x32_i16_8_rate(
            coeff.add(i),
            iscan.add(i),
            round_ac,
            quant_ac,
            gate_ac,
            &mut metrics,
        );
        i += 8;
    }

    metrics.finish()
}

/// Quantize 32x32 Hadamard coefficients and return fused RD metrics.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_fp_32x32_i16_metrics_neon(
    coeff: &[i16],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> HadamardQuantMetrics {
    const N_COEFFS: usize = 1024;
    assert!(coeff.len() >= N_COEFFS);
    assert!(iscan.len() >= N_COEFFS);

    // SAFETY: see `vp9_quantize_fp_i16_metrics_neon`; saturating i16 absolute
    // value and rounding preserve the scalar 32x32 coefficient bound.
    unsafe {
        vp9_quantize_fp_32x32_i16_metrics_neon_inner(
            coeff.as_ptr(),
            round,
            quant,
            dequant,
            iscan.as_ptr(),
        )
    }
}

/// Quantize 32x32 Hadamard coefficients and return EOB/rate metrics.
pub fn vp9_quantize_fp_32x32_i16_rate_metrics_neon(
    coeff: &[i16],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> HadamardQuantRateMetrics {
    const N_COEFFS: usize = 1024;
    assert!(coeff.len() >= N_COEFFS);
    assert!(iscan.len() >= N_COEFFS);

    // SAFETY: see `vp9_quantize_fp_i16_rate_metrics_neon`; the 32x32
    // arithmetic uses the same bounded quantization and threshold gates as
    // the full-metrics oracle.
    unsafe {
        vp9_quantize_fp_32x32_i16_rate_metrics_neon_inner(
            coeff.as_ptr(),
            round,
            quant,
            dequant,
            iscan.as_ptr(),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_b_8(
    coeff: *const i32,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    iscan: *const i16,
    zbin_lo: int32x4_t,
    zbin_hi: int32x4_t,
    round_lo: int32x4_t,
    round_hi: int32x4_t,
    quant_lo: int32x4_t,
    quant_hi: int32x4_t,
    quant_shift_lo: int32x4_t,
    quant_shift_hi: int32x4_t,
    dequant: int16x8_t,
    eob_max: &mut int16x8_t,
) {
    let coeff_lo = vld1q_s32(coeff);
    let coeff_hi = vld1q_s32(coeff.add(4));
    let sign_lo = vshrq_n_s32::<31>(coeff_lo);
    let sign_hi = vshrq_n_s32::<31>(coeff_hi);
    let abs_lo = vabsq_s32(coeff_lo);
    let abs_hi = vabsq_s32(coeff_hi);
    let keep_lo = vcgeq_s32(abs_lo, zbin_lo);
    let keep_hi = vcgeq_s32(abs_hi, zbin_hi);
    let max = vdupq_n_s32(i16::MAX as i32);

    let rounded_lo = vminq_s32(vaddq_s32(abs_lo, round_lo), max);
    let rounded_hi = vminq_s32(vaddq_s32(abs_hi, round_hi), max);
    let stage_lo = vaddq_s32(
        vshrq_n_s32::<16>(vmulq_s32(rounded_lo, quant_lo)),
        rounded_lo,
    );
    let stage_hi = vaddq_s32(
        vshrq_n_s32::<16>(vmulq_s32(rounded_hi, quant_hi)),
        rounded_hi,
    );
    let abs_q_lo = vshrq_n_s32::<16>(vmulq_s32(stage_lo, quant_shift_lo));
    let abs_q_hi = vshrq_n_s32::<16>(vmulq_s32(stage_hi, quant_shift_hi));
    let signed_lo = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_lo, sign_lo), sign_lo),
        vreinterpretq_s32_u32(keep_lo),
    );
    let signed_hi = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_hi, sign_hi), sign_hi),
        vreinterpretq_s32_u32(keep_hi),
    );
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
unsafe fn vp9_quantize_b_neon_inner(
    coeff: *const i32,
    n_coeffs: usize,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: &[i16; 2],
    iscan: *const i16,
) -> u16 {
    let dc_ac = |values: &[i16; 2]| {
        vld1q_s32(
            [
                i32::from(values[0]),
                i32::from(values[1]),
                i32::from(values[1]),
                i32::from(values[1]),
            ]
            .as_ptr(),
        )
    };
    let ac = |values: &[i16; 2]| vdupq_n_s32(i32::from(values[1]));
    let dequant_dc_ac = vld1q_s16(
        [
            dequant[0], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1],
            dequant[1],
        ]
        .as_ptr(),
    );
    let dequant_ac = vdupq_n_s16(dequant[1]);
    let mut eob_max = vdupq_n_s16(0);

    quantize_b_8(
        coeff,
        qcoeff,
        dqcoeff,
        iscan,
        dc_ac(zbin),
        ac(zbin),
        dc_ac(round),
        ac(round),
        dc_ac(quant),
        ac(quant),
        dc_ac(quant_shift),
        ac(quant_shift),
        dequant_dc_ac,
        &mut eob_max,
    );
    let zbin_ac = ac(zbin);
    let round_ac = ac(round);
    let quant_ac = ac(quant);
    let quant_shift_ac = ac(quant_shift);
    let mut i = 8;
    while i < n_coeffs {
        quantize_b_8(
            coeff.add(i),
            qcoeff.add(i),
            dqcoeff.add(i),
            iscan.add(i),
            zbin_ac,
            zbin_ac,
            round_ac,
            round_ac,
            quant_ac,
            quant_ac,
            quant_shift_ac,
            quant_shift_ac,
            dequant_ac,
            &mut eob_max,
        );
        i += 8;
    }

    vmaxvq_s16(eob_max) as u16
}

/// Quantize 16, 64, or 256 coefficients with the zbin quantizer.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_b_neon(
    coeff: &[i32],
    n_coeffs: usize,
    skip_block: bool,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
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

    // SAFETY: see `vp9_quantize_fp_neon`. This kernel has the same bounded
    // coefficient loads and output stores; all intermediate products match
    // the scalar i32 quantizer and remain within i32 for valid VP9 input.
    unsafe {
        vp9_quantize_b_neon_inner(
            coeff.as_ptr(),
            n_coeffs,
            zbin,
            round,
            quant,
            quant_shift,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant,
            iscan.as_ptr(),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn quantize_b_32x32_8(
    coeff: *const i32,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    iscan: *const i16,
    zbin_lo: int32x4_t,
    zbin_hi: int32x4_t,
    round_lo: int32x4_t,
    round_hi: int32x4_t,
    quant_lo: int32x4_t,
    quant_hi: int32x4_t,
    quant_shift_lo: int32x4_t,
    quant_shift_hi: int32x4_t,
    dequant: int16x8_t,
    eob_max: &mut int16x8_t,
) {
    let coeff_lo = vld1q_s32(coeff);
    let coeff_hi = vld1q_s32(coeff.add(4));
    let sign_lo = vshrq_n_s32::<31>(coeff_lo);
    let sign_hi = vshrq_n_s32::<31>(coeff_hi);
    let abs_lo = vabsq_s32(coeff_lo);
    let abs_hi = vabsq_s32(coeff_hi);
    let keep_lo = vcgeq_s32(abs_lo, zbin_lo);
    let keep_hi = vcgeq_s32(abs_hi, zbin_hi);
    let max = vdupq_n_s32(i16::MAX as i32);

    let rounded_lo = vminq_s32(vaddq_s32(abs_lo, round_lo), max);
    let rounded_hi = vminq_s32(vaddq_s32(abs_hi, round_hi), max);
    let stage_lo = vaddq_s32(
        vshrq_n_s32::<16>(vmulq_s32(rounded_lo, quant_lo)),
        rounded_lo,
    );
    let stage_hi = vaddq_s32(
        vshrq_n_s32::<16>(vmulq_s32(rounded_hi, quant_hi)),
        rounded_hi,
    );
    let abs_q_lo = vshrq_n_s32::<15>(vmulq_s32(stage_lo, quant_shift_lo));
    let abs_q_hi = vshrq_n_s32::<15>(vmulq_s32(stage_hi, quant_shift_hi));
    let signed_lo = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_lo, sign_lo), sign_lo),
        vreinterpretq_s32_u32(keep_lo),
    );
    let signed_hi = vandq_s32(
        vsubq_s32(veorq_s32(abs_q_hi, sign_hi), sign_hi),
        vreinterpretq_s32_u32(keep_hi),
    );
    let q = vcombine_s16(vmovn_s32(signed_lo), vmovn_s32(signed_hi));
    vst1q_s16(qcoeff, q);

    let mut dq_lo = vmull_s16(vget_low_s16(q), vget_low_s16(dequant));
    let mut dq_hi = vmull_s16(vget_high_s16(q), vget_high_s16(dequant));
    let negative_lo = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_lo)));
    let negative_hi = vreinterpretq_s32_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(dq_hi)));
    dq_lo = vshrq_n_s32::<1>(vaddq_s32(dq_lo, negative_lo));
    dq_hi = vshrq_n_s32::<1>(vaddq_s32(dq_hi, negative_hi));
    vst1q_s16(dqcoeff, vcombine_s16(vqmovn_s32(dq_lo), vqmovn_s32(dq_hi)));

    let nonzero = vmvnq_u16(vceqq_s16(q, vdupq_n_s16(0)));
    let eob = vbslq_s16(nonzero, vld1q_s16(iscan), vdupq_n_s16(0));
    *eob_max = vmaxq_s16(*eob_max, eob);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vp9_quantize_b_32x32_neon_inner(
    coeff: *const i32,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: &[i16; 2],
    iscan: *const i16,
) -> u16 {
    let dc_ac = |values: &[i16; 2], halve: bool| {
        let dc = i32::from(values[0]);
        let ac = i32::from(values[1]);
        let adjusted = |value: i32| if halve { (value + 1) >> 1 } else { value };
        vld1q_s32([adjusted(dc), adjusted(ac), adjusted(ac), adjusted(ac)].as_ptr())
    };
    let ac = |values: &[i16; 2], halve: bool| {
        let value = i32::from(values[1]);
        vdupq_n_s32(if halve { (value + 1) >> 1 } else { value })
    };
    let dequant_dc_ac = vld1q_s16(
        [
            dequant[0], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1], dequant[1],
            dequant[1],
        ]
        .as_ptr(),
    );
    let dequant_ac = vdupq_n_s16(dequant[1]);
    let mut eob_max = vdupq_n_s16(0);

    quantize_b_32x32_8(
        coeff,
        qcoeff,
        dqcoeff,
        iscan,
        dc_ac(zbin, true),
        ac(zbin, true),
        dc_ac(round, true),
        ac(round, true),
        dc_ac(quant, false),
        ac(quant, false),
        dc_ac(quant_shift, false),
        ac(quant_shift, false),
        dequant_dc_ac,
        &mut eob_max,
    );
    let zbin_ac = ac(zbin, true);
    let round_ac = ac(round, true);
    let quant_ac = ac(quant, false);
    let quant_shift_ac = ac(quant_shift, false);
    let mut i = 8;
    while i < 1024 {
        quantize_b_32x32_8(
            coeff.add(i),
            qcoeff.add(i),
            dqcoeff.add(i),
            iscan.add(i),
            zbin_ac,
            zbin_ac,
            round_ac,
            round_ac,
            quant_ac,
            quant_ac,
            quant_shift_ac,
            quant_shift_ac,
            dequant_ac,
            &mut eob_max,
        );
        i += 8;
    }

    vmaxvq_s16(eob_max) as u16
}

/// Quantize 1024 coefficients with the 32x32 zbin quantizer.
#[allow(clippy::too_many_arguments)]
pub fn vp9_quantize_b_32x32_neon(
    coeff: &[i32],
    skip_block: bool,
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 2],
    iscan: &[i16],
) -> u16 {
    const N_COEFFS: usize = 1024;
    assert!(coeff.len() >= N_COEFFS);
    assert!(qcoeff.len() >= N_COEFFS);
    assert!(dqcoeff.len() >= N_COEFFS);
    assert!(iscan.len() >= N_COEFFS);

    if skip_block {
        qcoeff[..N_COEFFS].fill(0);
        dqcoeff[..N_COEFFS].fill(0);
        return 0;
    }

    // SAFETY: all pointers cover 1024 elements. The widened arithmetic matches
    // the scalar quantizer, and the final narrowing saturates dqcoeff exactly as
    // the scalar implementation does.
    unsafe {
        vp9_quantize_b_32x32_neon_inner(
            coeff.as_ptr(),
            zbin,
            round,
            quant,
            quant_shift,
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
    use crate::mode_decision::block_yrd::{block_error_fp_i32, satd_i16};
    use crate::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, vp9_quantize_b, vp9_quantize_b_32x32,
        vp9_quantize_fp, vp9_quantize_fp_32x32,
    };
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

    #[test]
    fn differential_fast_32x32_all_qindices_thresholds_and_eob() {
        let order = scan_for(TxSize::Tx32x32, TxType::DctDct);
        let mut rng = Lcg32(0x32F0_C0DE);
        for q_index in [0, 1, 20, 80, 160, 255] {
            let qp = init_plane_quantizers(q_index, QuantizerDeltas::Y);
            let mut inputs = Vec::new();
            inputs.push((0..1024).map(|_| rng.coeff()).collect::<Vec<_>>());
            for scan_position in [None, Some(0), Some(1), Some(511), Some(1023)] {
                for delta in [-1, 0, 1] {
                    let mut coeff = vec![0; 1024];
                    if let Some(position) = scan_position {
                        let band = usize::from(position != 0);
                        coeff[order.scan[position] as usize] =
                            (i32::from(qp.dequant[band]) >> 2) + delta;
                    }
                    inputs.push(coeff);
                }
            }

            for (case, coeff) in inputs.into_iter().enumerate() {
                let mut scalar_q = vec![7; 1024];
                let mut scalar_dq = vec![9; 1024];
                let mut neon_q = vec![11; 1024];
                let mut neon_dq = vec![13; 1024];
                let scalar_eob = vp9_quantize_fp_32x32(
                    &coeff,
                    false,
                    &qp.round_fp,
                    &qp.quant_fp,
                    &mut scalar_q,
                    &mut scalar_dq,
                    &qp.dequant,
                    order.scan,
                );
                let neon_eob = vp9_quantize_fp_32x32_neon(
                    &coeff,
                    false,
                    &qp.round_fp,
                    &qp.quant_fp,
                    &mut neon_q,
                    &mut neon_dq,
                    &qp.dequant,
                    order.iscan,
                );
                assert_eq!(neon_eob, scalar_eob, "q{q_index}/case{case}");
                assert_eq!(neon_q, scalar_q, "q{q_index}/case{case}");
                assert_eq!(neon_dq, scalar_dq, "q{q_index}/case{case}");
            }
        }

        let mut qcoeff = [1; 1024];
        let mut dqcoeff = [2; 1024];
        assert_eq!(
            vp9_quantize_fp_32x32_neon(
                &[99; 1024],
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
        assert_eq!(qcoeff, [0; 1024]);
        assert_eq!(dqcoeff, [0; 1024]);
    }

    #[test]
    fn narrowed_hadamard_inputs_match_widened_fast_quantizers() {
        let mut rng = Lcg32(0x1616_3232);
        for tx_size in [TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32] {
            let n = tx_size.n_coeffs();
            let order = scan_for(tx_size, TxType::DctDct);
            for q_index in [0, 1, 20, 80, 160, 255] {
                let qp = init_plane_quantizers(q_index, QuantizerDeltas::Y);
                let mut random: Vec<i16> = (0..n).map(|_| rng.coeff() as i16).collect();
                random[0] = i16::MIN;
                random[n / 2] = i16::MAX;
                let mut dc_only = vec![0; n];
                dc_only[0] = i16::MAX;
                let mut final_scan_only = vec![0; n];
                let final_scan = i16::try_from(n).expect("VP9 transform size fits i16");
                let final_scan_index = order
                    .iscan
                    .iter()
                    .position(|&scan_position| scan_position == final_scan)
                    .expect("inverse scan includes the final position");
                final_scan_only[final_scan_index] = i16::MIN;

                for (case, coeff_i16) in [
                    random,
                    vec![0; n],
                    dc_only,
                    final_scan_only,
                    vec![i16::MIN; n],
                    vec![i16::MAX; n],
                ]
                .into_iter()
                .enumerate()
                {
                    let label = format!("{tx_size:?}/q{q_index}/case{case}");
                    let coeff_i32: Vec<i32> =
                        coeff_i16.iter().map(|&value| i32::from(value)).collect();
                    let mut wide_q = vec![7; n];
                    let mut wide_dq = vec![9; n];

                    let wide_eob = if tx_size == TxSize::Tx32x32 {
                        vp9_quantize_fp_32x32_neon(
                            &coeff_i32,
                            false,
                            &qp.round_fp,
                            &qp.quant_fp,
                            &mut wide_q,
                            &mut wide_dq,
                            &qp.dequant,
                            order.iscan,
                        )
                    } else {
                        vp9_quantize_fp_neon(
                            &coeff_i32,
                            n,
                            false,
                            &qp.round_fp,
                            &qp.quant_fp,
                            &mut wide_q,
                            &mut wide_dq,
                            &qp.dequant,
                            order.iscan,
                        )
                    };
                    let metrics = if tx_size == TxSize::Tx32x32 {
                        vp9_quantize_fp_32x32_i16_metrics_neon(
                            &coeff_i16,
                            &qp.round_fp,
                            &qp.quant_fp,
                            &qp.dequant,
                            order.iscan,
                        )
                    } else {
                        vp9_quantize_fp_i16_metrics_neon(
                            &coeff_i16,
                            n,
                            &qp.round_fp,
                            &qp.quant_fp,
                            &qp.dequant,
                            order.iscan,
                        )
                    };
                    let rate_metrics = if tx_size == TxSize::Tx32x32 {
                        vp9_quantize_fp_32x32_i16_rate_metrics_neon(
                            &coeff_i16,
                            &qp.round_fp,
                            &qp.quant_fp,
                            &qp.dequant,
                            order.iscan,
                        )
                    } else {
                        vp9_quantize_fp_i16_rate_metrics_neon(
                            &coeff_i16,
                            n,
                            &qp.round_fp,
                            &qp.quant_fp,
                            order.iscan,
                        )
                    };

                    assert_eq!(metrics.eob, wide_eob, "{label}");
                    assert_eq!(metrics.satd, satd_i16(&wide_q), "{label}");
                    assert_eq!(
                        metrics.error,
                        block_error_fp_i32(&coeff_i32, &wide_dq),
                        "{label}"
                    );
                    assert_eq!(rate_metrics.eob, metrics.eob, "{label}/rate-eob");
                    assert_eq!(rate_metrics.satd, metrics.satd, "{label}/rate-satd");
                }
            }
        }
    }

    #[test]
    fn differential_zbin_quantizer_all_sizes_scans_and_qindices() {
        let mut rng = Lcg32(0x2B1A_C0DE);
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
                    let scalar_eob = vp9_quantize_b(
                        &coeff,
                        n,
                        false,
                        &qp.zbin,
                        &qp.round,
                        &qp.quant,
                        &qp.quant_shift,
                        &mut scalar_q,
                        &mut scalar_dq,
                        &qp.dequant,
                        order.scan,
                    );
                    let neon_eob = vp9_quantize_b_neon(
                        &coeff,
                        n,
                        false,
                        &qp.zbin,
                        &qp.round,
                        &qp.quant,
                        &qp.quant_shift,
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
    fn differential_zbin_thresholds_and_sparse_eob() {
        for tx_size in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16] {
            let n = tx_size.n_coeffs();
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
            ] {
                let order = scan_for(tx_size, tx_type);
                for q_index in [1, 80, 255] {
                    let qp = init_plane_quantizers(q_index, QuantizerDeltas::Y);
                    for scan_position in [None, Some(0), Some(1), Some(n / 2), Some(n - 1)] {
                        for delta in [-1, 0, 1] {
                            let mut coeff = vec![0; n];
                            if let Some(position) = scan_position {
                                let band = usize::from(position != 0);
                                coeff[order.scan[position] as usize] =
                                    i32::from(qp.zbin[band]) + delta;
                            }
                            let mut scalar_q = vec![0; n];
                            let mut scalar_dq = vec![0; n];
                            let mut neon_q = vec![0; n];
                            let mut neon_dq = vec![0; n];
                            let scalar_eob = vp9_quantize_b(
                                &coeff,
                                n,
                                false,
                                &qp.zbin,
                                &qp.round,
                                &qp.quant,
                                &qp.quant_shift,
                                &mut scalar_q,
                                &mut scalar_dq,
                                &qp.dequant,
                                order.scan,
                            );
                            let neon_eob = vp9_quantize_b_neon(
                                &coeff,
                                n,
                                false,
                                &qp.zbin,
                                &qp.round,
                                &qp.quant,
                                &qp.quant_shift,
                                &mut neon_q,
                                &mut neon_dq,
                                &qp.dequant,
                                order.iscan,
                            );
                            assert_eq!(
                                (neon_eob, neon_q, neon_dq),
                                (scalar_eob, scalar_q, scalar_dq),
                                "{tx_size:?}/{tx_type:?}/q{q_index}/{scan_position:?}/{delta}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn differential_32x32_zbin_all_qindices_thresholds_and_eob() {
        let order = scan_for(TxSize::Tx32x32, TxType::DctDct);
        let mut rng = Lcg32(0x32B0_C0DE);
        for q_index in [0, 1, 20, 80, 160, 255] {
            let qp = init_plane_quantizers(q_index, QuantizerDeltas::Y);
            let mut inputs = Vec::new();
            inputs.push((0..1024).map(|_| rng.coeff()).collect::<Vec<_>>());
            for scan_position in [None, Some(0), Some(1), Some(511), Some(1023)] {
                for sign in [-1, 1] {
                    for delta in [-1, 0, 1] {
                        let mut coeff = vec![0; 1024];
                        if let Some(position) = scan_position {
                            let band = usize::from(position != 0);
                            let threshold = (i32::from(qp.zbin[band]) + 1) >> 1;
                            coeff[order.scan[position] as usize] = sign * (threshold + delta);
                        }
                        inputs.push(coeff);
                    }
                }
            }

            for (case, coeff) in inputs.into_iter().enumerate() {
                let mut scalar_q = vec![7; 1024];
                let mut scalar_dq = vec![9; 1024];
                let mut neon_q = vec![11; 1024];
                let mut neon_dq = vec![13; 1024];
                let scalar_eob = vp9_quantize_b_32x32(
                    &coeff,
                    false,
                    &qp.zbin,
                    &qp.round,
                    &qp.quant,
                    &qp.quant_shift,
                    &mut scalar_q,
                    &mut scalar_dq,
                    &qp.dequant,
                    order.scan,
                );
                let neon_eob = vp9_quantize_b_32x32_neon(
                    &coeff,
                    false,
                    &qp.zbin,
                    &qp.round,
                    &qp.quant,
                    &qp.quant_shift,
                    &mut neon_q,
                    &mut neon_dq,
                    &qp.dequant,
                    order.iscan,
                );
                assert_eq!(neon_eob, scalar_eob, "q{q_index}/case{case}");
                assert_eq!(neon_q, scalar_q, "q{q_index}/case{case}");
                assert_eq!(neon_dq, scalar_dq, "q{q_index}/case{case}");
            }
        }
    }

    #[test]
    fn zbin_32x32_skip_block_zeroes_outputs() {
        let order = scan_for(TxSize::Tx32x32, TxType::DctDct);
        let mut qcoeff = [1; 1024];
        let mut dqcoeff = [2; 1024];
        assert_eq!(
            vp9_quantize_b_32x32_neon(
                &[99; 1024],
                true,
                &[1, 2],
                &[3, 4],
                &[5, 6],
                &[7, 8],
                &mut qcoeff,
                &mut dqcoeff,
                &[9, 10],
                order.iscan,
            ),
            0
        );
        assert_eq!(qcoeff, [0; 1024]);
        assert_eq!(dqcoeff, [0; 1024]);
    }
}
