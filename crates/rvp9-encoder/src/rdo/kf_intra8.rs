//! Exact realtime keyframe luma RD for one 8×8 leaf.
//!
//! Pinned libvpx routes keyframe blocks smaller than 16×16 through
//! `vp9_rd_pick_intra_mode_sb`, even when `use_nonrd_pick_mode` is enabled.
//! For an 8×8 leaf, `rd_pick_intra_sby_mode` evaluates intra candidates with
//! one 8×8 transform, full coefficient pricing, reconstructed-pixel SSE, and
//! the keyframe mode cost. This module mirrors that path.

use rvp9_core::cost::coef_cost::CoefCostTable;
use rvp9_core::intra_pred::{DcAvailability, IntraMode, NUM_INTRA_MODES, PredCtx, intra_predict};
use rvp9_core::quant::forward::{PlaneQuantizers, scan_for_forward, vp9_quantize_b_with_order};
use rvp9_core::transforms::TxSize;
use rvp9_core::transforms::forward::fht8x8;
use rvp9_core::transforms::scans::scan_for;

use crate::rdo::cost_coeffs::cost_coeffs_with_contexts_cached;

const SIDE: usize = 8;

#[inline]
fn rdcost(rdmult: i64, rate: u32, dist: i64) -> i64 {
    let rate_term = (rdmult.saturating_mul(i64::from(rate)) + 256) >> 9;
    let dist_term = dist << 7;
    rate_term.saturating_add(dist_term)
}

fn conditional_skip(mode: IntraMode, best_mode: IntraMode) -> bool {
    match mode {
        IntraMode::D117 => !matches!(best_mode, IntraMode::V | IntraMode::D135),
        IntraMode::D63 => !matches!(best_mode, IntraMode::V | IntraMode::D45),
        IntraMode::D207 => !matches!(best_mode, IntraMode::H | IntraMode::D45),
        IntraMode::D153 => !matches!(best_mode, IntraMode::H | IntraMode::D135),
        _ => false,
    }
}

/// Inputs for [`pick_intra8x8_y_mode`].
pub struct PickIntra8x8Params<'a> {
    /// Source luma samples, row-major.
    pub src: &'a [u8; SIDE * SIDE],
    /// Reconstructed samples above the leaf, including eight above-right
    /// samples required by directional predictors.
    pub above: &'a [u8; 2 * SIDE],
    /// Reconstructed samples immediately left of the leaf.
    pub left: &'a [u8; SIDE],
    /// Reconstructed top-left sample, or the VP9 unavailable fill.
    pub above_left: u8,
    /// Whether an above neighbour is available.
    pub have_above: bool,
    /// Whether a left neighbour is available.
    pub have_left: bool,
    /// Keyframe mode-code costs for the causal `(above_mode, left_mode)` row.
    pub bmode_costs: &'a [i32; NUM_INTRA_MODES],
    /// Luma quantizer tables for this leaf.
    pub quant: &'a PlaneQuantizers,
    /// Coefficient token costs from the current frame context.
    pub coef_costs: &'a CoefCostTable,
    /// Combined 8×8 entropy context from the two above and two left 4×4
    /// coefficient contexts.
    pub coeff_ctx: usize,
    /// Keyframe RD multiplier.
    pub rdmult: i64,
    /// Current partition-search RD cap.
    pub best_rd: i64,
}

/// Winning 8×8 keyframe luma decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PickIntra8x8Result {
    /// Selected intra prediction mode.
    pub mode: IntraMode,
    /// Mode plus coefficient rate in 1/512-bit units.
    pub rate: u32,
    /// Coefficient-only rate.
    pub rate_y: u32,
    /// Reconstructed-pixel SSE multiplied by 16.
    pub dist: i64,
    /// Combined RD cost.
    pub rd: i64,
    /// Whether the winning transform has no non-zero coefficients.
    pub skippable: bool,
    /// Winning quantized coefficients in raster order.
    pub qcoeff: [i16; SIDE * SIDE],
    /// Winning end-of-block position in the mode's scan order.
    pub eob: u16,
    /// Winning reconstructed pixels, row-major.
    pub recon: [u8; SIDE * SIDE],
}

/// Mirror libvpx `rd_pick_intra_sby_mode` for a keyframe `BLOCK_8X8` leaf.
#[must_use]
pub fn pick_intra8x8_y_mode(params: &PickIntra8x8Params<'_>) -> Option<PickIntra8x8Result> {
    pick_intra8x8_y_mode_through(params, IntraMode::Tm)
}

/// Pick an exact 8x8 keyframe luma mode, stopping after `max_mode` in VP9
/// enum order.
///
/// CPU8 temporal SVC uses this boundary to retain full coefficient and pixel
/// RD scoring while avoiding the seven secondary directional candidates.
#[must_use]
pub fn pick_intra8x8_y_mode_through(
    params: &PickIntra8x8Params<'_>,
    max_mode: IntraMode,
) -> Option<PickIntra8x8Result> {
    let dispatch = rvp9_core::dsp::get();
    let mut best_rd = params.best_rd;
    let mut best: Option<PickIntra8x8Result> = None;
    let mut token_cache = [0u8; SIDE * SIDE];

    for raw_mode in 0u8..=max_mode as u8 {
        let mode = IntraMode::from_u8(raw_mode).expect("0..10 valid");
        let best_mode = best.as_ref().map_or(IntraMode::Dc, |winner| winner.mode);
        if conditional_skip(mode, best_mode) {
            continue;
        }
        if best.as_ref().is_some_and(|winner| winner.skippable) {
            break;
        }

        let tx_type = mode.tx_type();
        let mut recon = [0u8; SIDE * SIDE];
        intra_predict(
            mode,
            &mut recon,
            SIDE,
            SIDE,
            PredCtx {
                above: params.above,
                left: params.left,
                above_left: params.above_left,
            },
            DcAvailability {
                above_available: params.have_above,
                left_available: params.have_left,
            },
            &dispatch,
        );
        let mut residual = [0i16; SIDE * SIDE];
        for (diff, (&src, &pred)) in residual.iter_mut().zip(params.src.iter().zip(&recon)) {
            *diff = i16::from(src) - i16::from(pred);
        }

        let mut coeff = [0i32; SIDE * SIDE];
        fht8x8(&residual, &mut coeff, SIDE, tx_type);
        let scan = scan_for(TxSize::Tx8x8, tx_type);
        let fwd_scan = scan_for_forward(&scan);
        let mut qcoeff = [0i16; SIDE * SIDE];
        let mut dqcoeff = [0i16; SIDE * SIDE];
        let eob = vp9_quantize_b_with_order(
            &coeff,
            SIDE * SIDE,
            false,
            &params.quant.zbin,
            &params.quant.round,
            &params.quant.quant,
            &params.quant.quant_shift,
            &mut qcoeff,
            &mut dqcoeff,
            &[params.quant.dequant[0], params.quant.dequant[1]],
            &scan,
        );

        let rate_y = cost_coeffs_with_contexts_cached(
            &qcoeff,
            fwd_scan,
            scan.neighbors(),
            eob,
            TxSize::Tx8x8,
            0,
            0,
            params.coeff_ctx,
            params.coef_costs,
            &mut token_cache,
        );

        if eob > 0 {
            let dqcoeff: [i32; SIDE * SIDE] = dqcoeff.map(i32::from);
            crate::enc_inverse_transform_add(
                TxSize::Tx8x8,
                tx_type,
                &dqcoeff,
                eob as usize,
                &mut recon,
                SIDE,
            );
        }

        let dist = params
            .src
            .iter()
            .zip(&recon)
            .map(|(&src, &rec)| {
                let error = i64::from(src) - i64::from(rec);
                error * error
            })
            .sum::<i64>()
            * 16;
        let rate = rate_y.saturating_add(params.bmode_costs[mode as usize].max(0) as u32);
        let this_rd = rdcost(params.rdmult, rate, dist);

        if this_rd < best_rd {
            best_rd = this_rd;
            best = Some(PickIntra8x8Result {
                mode,
                rate,
                rate_y,
                dist,
                rd: this_rd,
                skippable: eob == 0,
                qcoeff,
                eob,
                recon,
            });
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::{coef_cost::build_coef_cost_table, intra_y_mode_costs_kf_table};
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    use crate::rdo::sub8_kf_intra::{PickIntra4x4Params, Sub4x4Edges, pick_intra4x4_block};

    #[test]
    fn ducks_first_larger_leaf_matches_pinned_pixel_rd() {
        let src = [
            176, 174, 176, 188, 192, 196, 187, 166, 185, 175, 160, 161, 179, 178, 163, 161, 167,
            131, 117, 109, 119, 148, 154, 161, 148, 139, 137, 142, 146, 153, 171, 170, 174, 181,
            176, 177, 172, 159, 148, 143, 140, 143, 141, 145, 141, 141, 145, 154, 117, 127, 130,
            141, 147, 168, 177, 181, 176, 177, 178, 184, 185, 186, 192, 194,
        ];
        let above = [127; 16];
        let left = [183, 188, 181, 167, 179, 142, 105, 164];
        let costs = intra_y_mode_costs_kf_table();
        let quant = init_plane_quantizers(119, QuantizerDeltas::Y);
        let coef_costs = build_coef_cost_table(&FrameContext::new_default());
        let params = PickIntra8x8Params {
            src: &src,
            above: &above,
            left: &left,
            above_left: 127,
            have_above: false,
            have_left: true,
            bmode_costs: &costs[IntraMode::Dc as usize][IntraMode::Dc as usize],
            quant: &quant,
            coef_costs: &coef_costs,
            coeff_ctx: 1,
            rdmult: 65_430,
            best_rd: i64::MAX,
        };
        let result = pick_intra8x8_y_mode(&params).expect("unbounded 8x8 pick must select a mode");
        let capped = pick_intra8x8_y_mode_through(&params, IntraMode::H)
            .expect("bounded 8x8 pick must select a mode");
        assert_eq!(capped, result);

        assert_eq!(result.mode, IntraMode::H);
        assert_eq!(result.rate_y, 38_564);
        assert_eq!(result.rate, 40_149);
        assert_eq!(result.dist, 25_792);
        assert_eq!(result.rd, 8_432_136);
        assert!(!result.skippable);
        assert_eq!(
            result.recon,
            [
                181, 178, 182, 189, 192, 188, 179, 171, 181, 169, 166, 168, 171, 169, 163, 158,
                163, 132, 117, 119, 132, 147, 158, 164, 158, 144, 138, 142, 150, 159, 164, 167,
                180, 181, 181, 178, 170, 159, 148, 141, 141, 140, 140, 141, 141, 142, 142, 142,
                109, 118, 127, 139, 152, 164, 175, 181, 168, 174, 179, 182, 184, 187, 190, 192,
            ]
        );

        // The reconstructed right edge and H mode are the causal inputs to
        // the immediately following 4x4 leaf at pixel (56, 0). Pin that
        // hand-off too: it was the first live mismatch before the 8x8 RD
        // path was ported.
        let next_src = [
            148, 142, 137, 135, 162, 165, 168, 170, 183, 188, 197, 196, 158, 150, 156, 148,
        ];
        let next = pick_intra4x4_block(&PickIntra4x4Params {
            blk_idx: 0,
            src: &next_src,
            edges: Sub4x4Edges {
                above: [127; 8],
                left: [
                    result.recon[7],
                    result.recon[15],
                    result.recon[23],
                    result.recon[31],
                ],
                above_left: 127,
                above_available: false,
                left_available: true,
            },
            bmode_costs: &costs[IntraMode::Dc as usize][IntraMode::H as usize],
            quant: &quant,
            coef_costs: &coef_costs,
            coeff_ctx: 1,
            rdmult: 65_430,
            rd_thresh: i64::MAX,
        })
        .expect("causal 4x4 follower must select a mode");
        assert_eq!(next.winner.mode, IntraMode::H);
        assert_eq!(next.winner.rate_y, 9_914);
        assert_eq!(next.winner.rate, 10_655);
        assert_eq!(next.winner.dist, 6_029);
        assert_eq!(next.winner.rd, 2_133_346);
        assert_eq!(
            [next.recon[3], next.recon[7], next.recon[11], next.recon[15]],
            [134, 168, 203, 155]
        );
    }

    #[test]
    fn ducks_standalone_tx8_extends_top_right_for_d63() {
        let src = [
            187, 188, 189, 188, 183, 180, 183, 176, 198, 196, 192, 190, 182, 180, 182, 181, 176,
            177, 178, 179, 181, 184, 185, 191, 162, 176, 185, 190, 196, 195, 202, 208, 185, 194,
            197, 194, 186, 184, 179, 179, 151, 148, 149, 150, 141, 143, 137, 132, 155, 153, 157,
            161, 162, 163, 165, 163, 193, 199, 199, 192, 200, 201, 203, 205,
        ];
        // A standalone 8x8 leaf has one Tx8. No transform to its right has
        // been reconstructed, so VP9 repeats the last causal top pixel.
        let above = [
            181, 181, 181, 181, 184, 186, 188, 189, 189, 189, 189, 189, 189, 189, 189, 189,
        ];
        let left = [187, 197, 158, 142, 195, 140, 137, 187];
        let costs = intra_y_mode_costs_kf_table();
        let quant = init_plane_quantizers(119, QuantizerDeltas::Y);
        let coef_costs = build_coef_cost_table(&FrameContext::new_default());
        let params = PickIntra8x8Params {
            src: &src,
            above: &above,
            left: &left,
            above_left: 181,
            have_above: true,
            have_left: true,
            bmode_costs: &costs[IntraMode::H as usize][IntraMode::H as usize],
            quant: &quant,
            coef_costs: &coef_costs,
            coeff_ctx: 2,
            rdmult: 65_430,
            best_rd: i64::MAX,
        };
        let result = pick_intra8x8_y_mode(&params).expect("unbounded 8x8 pick must select a mode");
        let capped = pick_intra8x8_y_mode_through(&params, IntraMode::H)
            .expect("bounded 8x8 pick must select a mode");
        assert_eq!(capped.mode, IntraMode::V);
        assert_eq!(capped.rate_y, 31_924);
        assert_eq!(capped.rate, 34_460);
        assert_eq!(capped.dist, 16_064);
        assert_eq!(capped.rd, 6_459_938);
        assert!(!capped.skippable);

        assert_eq!(result.mode, IntraMode::D63);
        assert_eq!(result.rate_y, 31_315);
        assert_eq!(result.rate, 34_834);
        assert_eq!(result.dist, 15_072);
        assert_eq!(result.rd, 6_380_756);
        assert!(!result.skippable);
        assert_eq!(
            result.recon,
            [
                187, 186, 185, 185, 185, 185, 185, 185, 194, 192, 190, 188, 186, 184, 182, 181,
                173, 173, 176, 179, 183, 186, 187, 188, 172, 175, 181, 188, 196, 202, 206, 208,
                185, 187, 187, 187, 187, 186, 184, 183, 150, 150, 149, 147, 144, 140, 137, 136,
                148, 151, 154, 157, 159, 161, 162, 162, 195, 197, 199, 201, 202, 203, 203, 204,
            ]
        );
    }
}
