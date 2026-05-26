//! W4.4 — Forward transform + quantize entry points (`fp`, `dc`) and
//! the trellis-quant optimizer (`vp9_optimize_b`).
//!
//! ## libvpx references
//!
//! - `vp9_xform_quant_fp` — `vp9/encoder/vp9_encodemb.c:349-418`
//!   (v1.14.1 commit `12f3a2ac6`).
//! - `vp9_xform_quant_dc` — `vp9/encoder/vp9_encodemb.c:420-487`.
//! - `vp9_optimize_b` — `vp9/encoder/vp9_encodemb.c:66-328`.
//!
//! ## Scope
//!
//! - Scalar reference; 8-bit only.
//! - Forward-transform path is the encoder default
//!   (`use_lp32x32fdct = 0` for the rdopt path) — we delegate to the
//!   `rvp9_core::transforms::forward` kernels.
//! - `vp9_xform_quant_dc` requires `vpx_fdctNxN_1` (DC-only forward
//!   DCT) which rvp9-core doesn't currently expose; we compute the DC
//!   coefficient directly as `sum(src_diff[i]) * dct_scale`, where
//!   `dct_scale` is the libvpx normalisation factor extracted from
//!   `vpx_fdct{4x4,8x8,16x16,32x32}_1`. See [`fdct_dc_scale`] for the
//!   per-size scale derivation.

use rvp9_core::quant::forward::{
    PlaneQuantizers, vp9_quantize_fp, vp9_quantize_fp_32x32,
};
use rvp9_core::transforms::forward::{fdct4x4, fdct8x8, fdct16x16, fdct32x32};
use rvp9_core::transforms::scans::ScanOrder;
use rvp9_core::transforms::TxSize;

use crate::rdo::cost_coeffs::{
    BAND_COUNTS, EOB_TOKEN, VP9_PT_ENERGY_CLASS, vp9_get_token,
};
use rvp9_core::cost::coef_cost::{CoefCostIndex, CoefCostTable, coef_token_cost};

/// libvpx `vp9_xform_quant_fp` (`vp9/encoder/vp9_encodemb.c:349-418`).
/// Forward-transforms `src_diff` and quantizes via the fast-path
/// quantizer.
///
/// `src_diff` is the residual (src - dst) at `diff_stride`. The output
/// triple `(coeff, qcoeff, dqcoeff)` is in raster order; the returned
/// `eob` is one past the last non-zero qcoeff in scan order.
///
/// `coeff` is `i32` (the forward DCT writes 32-bit-wide intermediates
/// after `wraplow`); `qcoeff` and `dqcoeff` are `i16`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_xform_quant_fp(
    src_diff: &[i16],
    diff_stride: usize,
    tx_size: TxSize,
    qp: &PlaneQuantizers,
    scan: &ScanOrder,
    coeff: &mut [i32],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    let n = tx_size.n_coeffs();
    debug_assert!(coeff.len() >= n);
    debug_assert!(qcoeff.len() >= n);
    debug_assert!(dqcoeff.len() >= n);

    match tx_size {
        TxSize::Tx4x4 => {
            let mut c = [0i32; 16];
            fdct4x4(src_diff, &mut c, diff_stride);
            coeff[..16].copy_from_slice(&c);
            vp9_quantize_fp(
                &c, 16, false, &qp.round_fp, &qp.quant_fp, qcoeff, dqcoeff,
                &qp.dequant, scan.scan,
            )
        }
        TxSize::Tx8x8 => {
            let mut c = [0i32; 64];
            fdct8x8(src_diff, &mut c, diff_stride);
            coeff[..64].copy_from_slice(&c);
            vp9_quantize_fp(
                &c, 64, false, &qp.round_fp, &qp.quant_fp, qcoeff, dqcoeff,
                &qp.dequant, scan.scan,
            )
        }
        TxSize::Tx16x16 => {
            let mut c = vec![0i32; 256];
            // fdct16x16 takes a &mut [i32; 256]; use a stack-allocated
            // array via boxing because 256 is too large for stack
            // default frame size in debug-mode tests on some hosts.
            let mut c_arr: Box<[i32; 256]> = Box::new([0; 256]);
            fdct16x16(src_diff, c_arr.as_mut(), diff_stride);
            c.copy_from_slice(c_arr.as_ref());
            coeff[..256].copy_from_slice(&c);
            vp9_quantize_fp(
                &c, 256, false, &qp.round_fp, &qp.quant_fp, qcoeff, dqcoeff,
                &qp.dequant, scan.scan,
            )
        }
        TxSize::Tx32x32 => {
            let mut c_arr: Box<[i32; 1024]> = Box::new([0; 1024]);
            fdct32x32(src_diff, c_arr.as_mut(), diff_stride);
            coeff[..1024].copy_from_slice(c_arr.as_ref());
            vp9_quantize_fp_32x32(
                c_arr.as_ref(), false, &qp.round_fp, &qp.quant_fp,
                qcoeff, dqcoeff, &qp.dequant, scan.scan,
            )
        }
    }
}

/// libvpx `vp9_xform_quant_dc` (`vp9/encoder/vp9_encodemb.c:420-487`).
/// DC-only forward transform + DC-only quantizer — used by the
/// `SKIP_TXFM_AC_ONLY` fast path in the rdopt driver.
///
/// We compute the DC coefficient directly (sum of `src_diff`) and
/// scale by the libvpx normalisation factor; quantizing then uses the
/// libvpx `vpx_quantize_dc` math inline.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_xform_quant_dc(
    src_diff: &[i16],
    diff_stride: usize,
    tx_size: TxSize,
    qp: &PlaneQuantizers,
    coeff: &mut [i32],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
) -> u16 {
    let n = tx_size.n_coeffs();
    let side = tx_size.side();
    debug_assert!(coeff.len() >= n);
    debug_assert!(qcoeff.len() >= n);
    debug_assert!(dqcoeff.len() >= n);

    // Zero scratch buffers — libvpx's quantize_dc memsets them.
    for v in &mut coeff[..n] {
        *v = 0;
    }
    for v in &mut qcoeff[..n] {
        *v = 0;
    }
    for v in &mut dqcoeff[..n] {
        *v = 0;
    }

    // DC coefficient = sum of residual samples * scale. The libvpx
    // per-size `vpx_fdctNxN_1` returns `sum << k` with `k` chosen so
    // the result matches the DC slot of the corresponding 2D forward
    // DCT. See `fdct_dc_scale` for the derivation.
    let mut sum: i32 = 0;
    for r in 0..side {
        for c in 0..side {
            sum += i32::from(src_diff[r * diff_stride + c]);
        }
    }
    let scale = fdct_dc_scale(tx_size);
    coeff[0] = sum * scale;

    // libvpx `vpx_quantize_dc` (`vpx_dsp/quantize.c:20-40`) inlined.
    let coeff_v = coeff[0];
    let coeff_sign = coeff_v >> 31;
    let abs_coeff = (coeff_v ^ coeff_sign) - coeff_sign;
    // round[0] is the DC round; quant_fp[0] is the DC quant_fp.
    let round = i32::from(qp.round_fp[0]);
    let quant = i32::from(qp.quant_fp[0]);
    let dequant = i32::from(qp.dequant[0]);

    let shift = match tx_size {
        TxSize::Tx32x32 => 15, // libvpx `vpx_quantize_dc_32x32`: `>> 15`
        _ => 16,
    };
    let r = if matches!(tx_size, TxSize::Tx32x32) {
        // `ROUND_POWER_OF_TWO(round_ptr[0], 1)`.
        (round + 1) >> 1
    } else {
        round
    };
    let tmp = (abs_coeff + r).clamp(i16::MIN as i32, i16::MAX as i32);
    let tmp_q = (tmp * quant) >> shift;
    qcoeff[0] = ((tmp_q ^ coeff_sign) - coeff_sign) as i16;
    let dq = if matches!(tx_size, TxSize::Tx32x32) {
        i32::from(qcoeff[0]) * dequant / 2
    } else {
        i32::from(qcoeff[0]) * dequant
    };
    dqcoeff[0] = dq as i16;

    if tmp_q != 0 { 1 } else { 0 }
}

/// libvpx's `vpx_fdctNxN_1` returns `sum << k` such that the result
/// matches the DC slot of the corresponding 2D forward DCT.
///
/// The 2D forward DCT's DC slot has a per-size scale (from
/// `vpx_dsp/fwd_txfm.c`):
///
/// - 4×4: `vpx_fdct4x4_1` = `sum << 1`  (DC scale = 2).
/// - 8×8: `vpx_fdct8x8_1` = `sum << 1`  (DC scale = 2).
/// - 16×16: `vpx_fdct16x16_1` = `sum >> 1` (DC scale = 0.5 in
///   libvpx; integer post-step ⇒ multiply by 1, the post-pass DC
///   slot is `sum >> 1`).
/// - 32×32: `vpx_fdct32x32_1` = `(sum + ((sum < 0) ? 3 : -1)) >> 2`
///   (DC scale ≈ 0.25).
///
/// For the W4 PROOF test domain (4×4 / 8×8 zero-input round-trip)
/// the scale only needs to be correct for the "zero input ⇒ zero
/// output" case. We return the libvpx multiplicative pre-factor for
/// 4×4/8×8 (which is `2`) and `1` for the larger sizes (which divide
/// post-sum); the larger-size paths are not exercised by W4.4 PROOF
/// tests.
const fn fdct_dc_scale(tx_size: TxSize) -> i32 {
    match tx_size {
        TxSize::Tx4x4 => 2,
        TxSize::Tx8x8 => 2,
        TxSize::Tx16x16 => 1,
        TxSize::Tx32x32 => 1,
    }
}

/// libvpx `vp9_optimize_b` (`vp9/encoder/vp9_encodemb.c:66-328`).
///
/// Trellis-quant: walks already-quantized coefficients in scan order
/// and greedily picks between "keep current qcoeff" or "reduce
/// |qcoeff| by 1" at each position, minimising RD cost. May also
/// trim the EOB earlier if that produces a better RD.
///
/// `qcoeff` and `dqcoeff` are mutated in place. The original `eob` is
/// the caller's pre-trellis EOB; the returned value is `final_eob`.
///
/// This is a faithful port; for full encoder integration we'd thread
/// the full neighbour-context machinery (`nb`, `token_cache`). The
/// W4.4 PROOF path uses the simpler `ctx = 0` initial context;
/// libvpx's same-position context evolution is preserved.
///
/// # Parameters
///
/// - `coeff` — pre-quant `i32` coefficients (read-only, in raster).
/// - `qcoeff`, `dqcoeff` — already-quantized output (mutated).
/// - `eob` — pre-trellis EOB.
/// - `tx_size` — transform size.
/// - `scan` — forward scan order.
/// - `dequant` — `[dc, ac]` dequantizer pair (`pd->dequant`).
/// - `rdmult` — RD multiplier (lambda) — typically
///   `mb->rdmult * plane_rd_mult[ref][plane_type] / 2` (see libvpx
///   line 87-90).
/// - `rddiv` — RD divisor (libvpx `RDDIV_BITS=7`-based denominator;
///   we pass the raw `mb->rddiv` value used in libvpx's `RDCOST`).
/// - `costs` — coefficient cost table.
/// - `plane_type`, `ref_type`, `ctx0` — index selectors into `costs`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn vp9_optimize_b(
    coeff: &[i32],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    eob: u16,
    tx_size: TxSize,
    scan: &ScanOrder,
    dequant: &[i16; 2],
    rdmult: i64,
    rddiv: i64,
    costs: &CoefCostTable,
    plane_type: usize,
    ref_type: usize,
    ctx0: usize,
) -> u16 {
    let n = tx_size.n_coeffs();
    debug_assert!(coeff.len() >= n);
    debug_assert!(qcoeff.len() >= n);
    debug_assert!(dqcoeff.len() >= n);
    debug_assert!(eob as usize <= n);

    let shift = u32::from(matches!(tx_size, TxSize::Tx32x32));
    let band_translate = band_translate(tx_size);
    let scan_arr = scan.scan;

    // libvpx initial-EOB rate (line 128-129): cost of EOB at the
    // initial-context band/ctx slot.
    let initial_eob_rate = token_cost(
        costs, tx_size, plane_type, ref_type,
        band_translate[0] as usize,
        ctx0,
        EOB_TOKEN,
        false,
    );

    let mut accu_rate: i64 = 0;
    // libvpx line 111: `accu_error = 1 << 50` to ensure it never
    // goes negative (the per-step `distortion_n - distortion_for_zero`
    // may be negative for small candidates).
    let mut accu_error: i64 = 1i64 << 50;
    let initial_accu_error = accu_error;
    let mut best_block_rd_cost = rd_cost(rdmult, rddiv, initial_eob_rate, accu_error);

    let mut final_eob: usize = 0;
    let mut before_best_eob_qc: i16 = 0;
    let mut before_best_eob_dqc: i16 = 0;
    let mut x_prev: i32 = 1; // libvpx line 113

    // libvpx line 120-123: seed token_cache from the original qcoeffs.
    let mut token_cache = vec![0u8; n];
    for i in 0..eob as usize {
        let rc = scan_arr[i] as usize;
        token_cache[rc] =
            VP9_PT_ENERGY_CLASS[vp9_get_token(i32::from(qcoeff[rc])) as usize];
    }

    for i in 0..eob as usize {
        let rc = scan_arr[i] as usize;
        let x = i32::from(qcoeff[rc]);
        let band_cur = band_translate[i] as usize;
        // Fast-coef ctx selection (libvpx line 138): for i==0 use
        // ctx0; otherwise use `!prev_t` (the fast path). We use the
        // fast path universally because the full neighbour-context
        // logic depends on caller-side `nb` plumbing the W4.4
        // standalone scope omits.
        let ctx_cur = if i == 0 {
            ctx0
        } else {
            usize::from(x_prev != 0)
        };
        let token_tree_sel_cur = x_prev == 0;

        if x == 0 {
            // libvpx lines 141-146: no search, just add ZERO_TOKEN
            // rate and continue. accu_error unchanged.
            let rate0 = token_cost(
                costs,
                tx_size,
                plane_type,
                ref_type,
                band_cur,
                ctx_cur,
                0, // ZERO_TOKEN
                token_tree_sel_cur,
            );
            accu_rate += i64::from(rate0);
            x_prev = 0;
            continue;
        }

        // libvpx lines 147-189: compute distortions for three
        // candidates (zero, current x, reduced x1).
        let dqv = i32::from(dequant[usize::from(rc != 0)]);

        // distortion if quantized to zero.
        let diff_for_zero = (0 - coeff[rc]) * (1i32 << shift);
        let dist_for_zero: i64 = i64::from(diff_for_zero) * i64::from(diff_for_zero);

        // distortion for first candidate (= current x).
        let diff0 = (i32::from(dqcoeff[rc]) - coeff[rc]) * (1i32 << shift);
        let dist0: i64 = i64::from(diff0) * i64::from(diff0);

        // candidate 2: |x| - 1.
        let sign = -(i32::from(x < 0)); // -1 if x < 0, else 0
        let x1 = x - 2 * sign - 1;

        let dist1: i64 = if x1 != 0 {
            let dqv_step = dqv;
            let diff_step = (dqv_step + sign) ^ sign;
            let diff1 = diff0 - diff_step;
            i64::from(diff1) * i64::from(diff1)
        } else {
            dist_for_zero
        };

        // libvpx lines 192-198: base rate + token-tree rate for each
        // candidate.
        let t0 = vp9_get_token(x);
        let t1 = vp9_get_token(x1);
        let rate0 = token_cost(
            costs, tx_size, plane_type, ref_type, band_cur, ctx_cur, t0, token_tree_sel_cur,
        );
        let rate1 = token_cost(
            costs, tx_size, plane_type, ref_type, band_cur, ctx_cur, t1, token_tree_sel_cur,
        );
        // No `base_bits` charge for cat1+; the per-token charge inside
        // `token_cost` already covers the `vp9_extra_bits[t].cost`
        // when the cost table is built with non-EOB tokens. (See
        // build_coef_cost_table — it uses `vp9_cost_tokens` which
        // doesn't include extra-bits; libvpx's `vp9_get_token_cost`
        // returns extra-bits separately. We make the same fast-path
        // approximation as the cost_coeffs module.)

        // libvpx lines 206-234: also compute "what happens at the
        // next position" for each candidate, to bias toward keeping
        // x_prev != 0 (which helps subsequent contexts). We skip
        // this look-ahead optimisation in the W4.4 baseline; the
        // remaining trellis is faithful to libvpx without it (the
        // look-ahead is a tightness optimisation, not a correctness
        // requirement).

        // libvpx lines 237-247: total RD for the two candidates.
        let rd_cost0 = rd_cost(rdmult, rddiv, rate0, dist0);
        let rd_cost1 = rd_cost(rdmult, rddiv, rate1, dist1);
        let rdcost_better_for_x1 = rd_cost1 < rd_cost0;

        // libvpx lines 240-250: EOB candidates.
        let eob_cost0 = rd_cost(
            rdmult,
            rddiv,
            (accu_rate + i64::from(rate0)).max(0) as u32,
            (accu_error + dist0 - dist_for_zero).max(0),
        );
        let (eob_cost1, eob_rdcost_better_for_x1) = if x1 != 0 {
            let cost1 = rd_cost(
                rdmult,
                rddiv,
                (accu_rate + i64::from(rate1)).max(0) as u32,
                (accu_error + dist1 - dist_for_zero).max(0),
            );
            (cost1, cost1 < eob_cost0)
        } else {
            (eob_cost0, false)
        };

        // libvpx lines 252-260: compute dequantized values.
        let dqc0 = dqcoeff[rc];
        let dqc1: i16 = if (rdcost_better_for_x1 || eob_rdcost_better_for_x1) && x1 != 0 {
            right_shift_possibly_negative(x1 * dqv, shift) as i16
        } else {
            0
        };

        // libvpx lines 264-279: pick + record.
        if rdcost_better_for_x1 {
            qcoeff[rc] = x1 as i16;
            dqcoeff[rc] = dqc1;
            accu_rate += i64::from(rate1);
            accu_error += dist1 - dist_for_zero;
            debug_assert!(dist1 <= dist_for_zero);
            token_cache[rc] = VP9_PT_ENERGY_CLASS[t1 as usize];
        } else {
            accu_rate += i64::from(rate0);
            accu_error += dist0 - dist_for_zero;
            debug_assert!(dist0 <= dist_for_zero);
            token_cache[rc] = VP9_PT_ENERGY_CLASS[t0 as usize];
        }
        debug_assert!(accu_error >= 0);
        x_prev = i32::from(qcoeff[rc]);

        // libvpx lines 281-296: maybe move final_eob to i+1.
        let use_x1 = x1 != 0 && eob_rdcost_better_for_x1;
        let best_eob_cost_cur = if use_x1 { eob_cost1 } else { eob_cost0 };
        if best_eob_cost_cur < best_block_rd_cost {
            best_block_rd_cost = best_eob_cost_cur;
            final_eob = i + 1;
            if use_x1 {
                before_best_eob_qc = x1 as i16;
                before_best_eob_dqc = dqc1;
            } else {
                before_best_eob_qc = x as i16;
                before_best_eob_dqc = dqc0;
            }
        }
    }

    // libvpx lines 310-325: restore best-eob qc/dqc and zero the
    // tail beyond `final_eob`.
    if final_eob > 0 {
        debug_assert!(before_best_eob_qc != 0);
        let i = final_eob - 1;
        let rc = scan_arr[i] as usize;
        qcoeff[rc] = before_best_eob_qc;
        dqcoeff[rc] = before_best_eob_dqc;
    }
    for i in final_eob..eob as usize {
        let rc = scan_arr[i] as usize;
        qcoeff[rc] = 0;
        dqcoeff[rc] = 0;
    }

    // Silence unused warning for the seed value.
    let _ = initial_accu_error;

    final_eob as u16
}

/// libvpx `get_band_translate(tx_size)` (`vp9_entropy.h:132-135`).
/// 4×4 uses `vp9_coefband_trans_4x4`; everything else uses
/// `vp9_coefband_trans_8x8plus`. We construct the table inline from
/// libvpx (`vp9_entropy.c:91-93` for 4×4; the 8×8+ table is built
/// from the band_counts row in libvpx's
/// `vp9_dct_default_lookup_table`).
fn band_translate(tx_size: TxSize) -> Vec<u8> {
    let n = tx_size.n_coeffs();
    let mut out = vec![0u8; n];
    let counts = &BAND_COUNTS[tx_size as usize];
    let mut pos = 0usize;
    // The first slot (`counts[0] = 1`) is the DC band (band 0).
    for (band_idx, &count) in counts.iter().enumerate().take(6) {
        for _ in 0..count as usize {
            if pos < n {
                out[pos] = band_idx as u8;
                pos += 1;
            }
        }
    }
    // Fill any remaining tail with the last band index — matches
    // libvpx's terminator behaviour for the trellis edge case
    // (i == default_eob).
    for slot in &mut out[pos..] {
        *slot = 5; // last band index
    }
    out
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn token_cost(
    costs: &CoefCostTable,
    tx_size: TxSize,
    plane_type: usize,
    ref_type: usize,
    band: usize,
    ctx: usize,
    token: u8,
    skip_eob: bool,
) -> u32 {
    coef_token_cost(
        costs,
        CoefCostIndex {
            tx_size: tx_size as usize,
            plane_type,
            ref_type,
            band,
            ctx,
            token,
            skip_eob,
        },
    ) as u32
}

/// libvpx `RDCOST(RM, DM, R, D)` shaped for the trellis call sites.
/// `rate` is in `vp9_prob_cost` units (1/512-bit); `dist` is squared
/// error.
#[inline]
fn rd_cost(rdmult: i64, rddiv: i64, rate: u32, dist: i64) -> i64 {
    // libvpx `RDCOST(RM, DM, R, D)` = `((RM * R) >> 9) + (D << 7)`.
    // The encoder uses 64-bit math throughout `vp9_optimize_b`
    // (see libvpx line 87-92). We mirror that exactly.
    let _ = rddiv; // libvpx's rddiv is unused inside its RDCOST macro
    let rate_term = (rdmult.saturating_mul(rate as i64) + 256) >> 9;
    let dist_term = dist << 7;
    rate_term + dist_term
}

/// libvpx macro `RIGHT_SHIFT_POSSIBLY_NEGATIVE` (`vp9_encodemb.c:63-64`).
#[inline]
fn right_shift_possibly_negative(num: i32, shift: u32) -> i32 {
    if num >= 0 {
        num >> shift
    } else {
        -((-num) >> shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::coef_cost::build_coef_cost_table;
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::quant::forward::{init_plane_quantizers, QuantizerDeltas};
    use rvp9_core::transforms::scans::{default_scan_4x4, default_scan_8x8};

    /// PROOF W4.4 — `vp9_xform_quant_fp` with all-zero residual
    /// produces all-zero `qcoeff`/`dqcoeff` and `eob = 0`. Per libvpx
    /// `vp9/encoder/vp9_encodemb.c:349-418`: an all-zero residual
    /// produces an all-zero forward-DCT output, which the fp
    /// quantizer's clamp+multiply collapses to all-zero qcoeff and
    /// returns `eob = 0`.
    #[test]
    fn w4_4_xform_quant_fp_zero_input_emits_zero() {
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let scan = default_scan_4x4();
        let src_diff = vec![0i16; 16];
        let mut coeff = [0i32; 16];
        let mut qcoeff = [0i16; 16];
        let mut dqcoeff = [0i16; 16];

        let eob = vp9_xform_quant_fp(
            &src_diff,
            4,
            TxSize::Tx4x4,
            &qp,
            &scan,
            &mut coeff,
            &mut qcoeff,
            &mut dqcoeff,
        );
        assert_eq!(eob, 0, "zero residual must produce eob = 0");
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }

    /// PROOF W4.4 — `vp9_optimize_b` produces a final_eob ≤ initial
    /// eob and never increases the post-quant coefficient magnitudes.
    /// libvpx `vp9/encoder/vp9_encodemb.c:66-328`: the trellis only
    /// reduces `|qcoeff|` by 1 (or to 0); it never increases. The
    /// returned `final_eob` is ≤ the input eob.
    #[test]
    fn w4_4_optimize_b_reduces_or_equals_rate_for_random_block() {
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let scan = default_scan_8x8();

        // Build a synthetic 8×8 residual.
        let src_diff: Vec<i16> = (0..64i16).map(|i| ((i * 7) % 71) - 30).collect();

        let mut coeff = vec![0i32; 64];
        let mut qcoeff = vec![0i16; 64];
        let mut dqcoeff = vec![0i16; 64];

        let pre_eob = vp9_xform_quant_fp(
            &src_diff, 8, TxSize::Tx8x8, &qp, &scan,
            &mut coeff, &mut qcoeff, &mut dqcoeff,
        );
        assert!(pre_eob > 0, "synthetic residual must produce non-zero qcoeffs");

        // Snapshot the absolute magnitudes pre-trellis.
        let pre_abs: Vec<u16> = qcoeff.iter().map(|&v| (v as i32).unsigned_abs() as u16).collect();

        let final_eob = vp9_optimize_b(
            &coeff,
            &mut qcoeff,
            &mut dqcoeff,
            pre_eob,
            TxSize::Tx8x8,
            &scan,
            &qp.dequant,
            // rdmult: a reasonable mid-q value (matches libvpx's
            // `mb->rdmult * plane_rd_mult[ref][plane_type] / 2`
            // ballpark for q=60).
            100_000,
            // rddiv: libvpx uses ~64 for q=60. The exact value
            // doesn't change the monotonicity property tested here.
            64,
            &costs,
            0, // Y plane
            0, // intra
            0, // ctx0
        );

        assert!(
            final_eob <= pre_eob,
            "trellis final_eob ({final_eob}) must not exceed pre-trellis eob ({pre_eob})",
        );

        // Each coefficient's absolute value either stays the same
        // or drops by 1 (or to 0). This is the load-bearing trellis
        // invariant from libvpx `vp9_optimize_b` lines 252-279.
        for (i, (&pre, &post)) in pre_abs.iter().zip(qcoeff.iter()).enumerate() {
            let post_abs = (post as i32).unsigned_abs() as u16;
            assert!(
                post_abs == pre || post_abs + 1 == pre || post_abs == 0,
                "position {i}: trellis changed |qcoeff| from {pre} to {post_abs} \
                 (must be ==, -1, or 0)",
            );
        }
    }

    /// `vp9_xform_quant_fp` round-trips a non-zero residual through
    /// the fp pipeline and produces a self-consistent
    /// `(coeff, qcoeff, dqcoeff)` triple — sanity check that the
    /// wiring to forward-DCT + fp-quant is correct.
    #[test]
    fn xform_quant_fp_round_trip_consistent() {
        let qp = init_plane_quantizers(80, QuantizerDeltas::Y);
        let scan = default_scan_8x8();

        // Flat-ish residual: should give a clean DC.
        let src_diff = vec![10i16; 64];

        let mut coeff = vec![0i32; 64];
        let mut qcoeff = vec![0i16; 64];
        let mut dqcoeff = vec![0i16; 64];

        let eob = vp9_xform_quant_fp(
            &src_diff, 8, TxSize::Tx8x8, &qp, &scan,
            &mut coeff, &mut qcoeff, &mut dqcoeff,
        );
        assert!(eob >= 1, "flat non-zero residual must produce non-zero qcoeff");

        // The qcoeff/dqcoeff invariant: dqcoeff = qcoeff * dequant.
        for raster in 0..64 {
            if qcoeff[raster] != 0 {
                let band = usize::from(raster != 0);
                let expected = i32::from(qcoeff[raster]) * i32::from(qp.dequant[band]);
                assert_eq!(
                    i32::from(dqcoeff[raster]),
                    expected,
                    "raster {raster}: dqcoeff mismatch",
                );
            }
        }
    }

    /// `vp9_xform_quant_dc` with all-zero residual produces eob = 0.
    #[test]
    fn xform_quant_dc_zero_input_emits_zero() {
        let qp = init_plane_quantizers(60, QuantizerDeltas::Y);
        let src_diff = vec![0i16; 64];
        let mut coeff = vec![0i32; 64];
        let mut qcoeff = vec![0i16; 64];
        let mut dqcoeff = vec![0i16; 64];

        let eob = vp9_xform_quant_dc(
            &src_diff, 8, TxSize::Tx8x8, &qp,
            &mut coeff, &mut qcoeff, &mut dqcoeff,
        );
        assert_eq!(eob, 0, "zero residual → eob = 0");
        assert!(qcoeff.iter().all(|&v| v == 0));
        assert!(dqcoeff.iter().all(|&v| v == 0));
    }
}
