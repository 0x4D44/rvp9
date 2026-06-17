//! W4.10b — sub-8 inter picker (**real** transform-domain RD).
//!
//! ## libvpx reference
//!
//! Ports the per-sub-block RD core of `rd_pick_best_sub8x8_mode`
//! (`vp9/encoder/vp9_rdopt.c:2077-2428`, v1.14.1 commit `12f3a2ac6`)
//! and its inner helper `encode_inter_mb_segment`
//! (`vp9/encoder/vp9_rdopt.c:1608-1734`).
//!
//! For an 8×8 parent the partition driver descends into 4 independent
//! 4×4 sub-blocks. For each `(sub_block, mode, ref)` candidate libvpx:
//!
//! 1. builds the **real** inter prediction for the 4×4 sub-block
//!    (`vp9_build_inter_predictor` with the 8-tap kernel,
//!    `vp9_rdopt.c:1660`),
//! 2. forms the residual `src - pred` (`vpx_subtract_block`, line 1684),
//! 3. forward-transforms each 4×4 tile (`x->fwd_txfm4x4` = `vpx_fdct4x4`,
//!    line 1706) and quantizes (`vpx_quantize_b`, line 1714),
//! 4. accumulates transform-domain distortion
//!    (`vp9_block_error(coeff, dqcoeff, 16)`, line 1716) and `sse`
//!    (the `ssz` out-param) and the coefficient rate (`cost_coeffs`,
//!    line 1719),
//! 5. composes the per-sub-block `brdcost = RDCOST(rdmult, byrate,
//!    bdist >> 2)` (line 1733) and then folds in the MV/mode-signalling
//!    rate (`brdcost += RDCOST(rdmult, brate, 0)`, line 2346).
//!
//! The per-sub-block winner is the lowest-`brdcost` mode (line 2357);
//! the 4 sub-block winners are aggregated into the segment RD
//! (lines 2380-2400).
//!
//! ## What changed (anti-cheat: argmin → real RD)
//!
//! The re-audit pass 2 flagged this module as a cheat: it accepted
//! pre-computed `(rate, dist, rd)` candidate tuples and returned the
//! argmin, so the load-bearing inner loop (predict / subtract / fdct /
//! quant / cost_coeffs / block_error) lived in the caller and was never
//! tested. This module now computes RD for real inside the picker,
//! mirroring `encode_inter_mb_segment`.
//!
//! ## Scope (recorded deviations)
//!
//! - **Only `Block8x8` parent → 4×4 sub-blocks.** libvpx's sub-8 picker
//!   also handles `Block8x4 / Block4x8` parents (4×8 / 8×4 sub-blocks);
//!   in the standalone scope the parent is fixed to `Block8x8` (the only
//!   one the W4.10a partition driver enters this picker for) and the
//!   sub-blocks are always 4×4 (one transform tile each).
//! - **Per-sub-block candidate MVs are pre-resolved.** The MV-search
//!   dispatch (`vp9_rdopt.c:2168-2278`, `joint_motion_search` /
//!   `cpi->fn_ptr[bsize].xxx`) is NOT ported; the caller supplies the
//!   candidate `(mode, ref, mv)` set. The picker computes the **real**
//!   prediction and RD for each candidate MV — the cheat that was
//!   removed was the pre-computed *RD*, not the pre-resolved MV.
//! - **MV/mode-signalling rate is supplied by the caller** as
//!   `mv_mode_rate` (libvpx `set_and_cost_bmi_mvs` `brate`, line 2280).
//!   The coefficient rate (`byrate`) and the distortion are computed
//!   for real.
//! - **No filter cache** (libvpx `filter_idx > 0` reuse, lines
//!   2301-2336). The picker evaluates every candidate.
//! - **No `coeff_ctx` neighbour threading** — uses `coeff_ctx = 0`
//!   (start-of-block). The full plumbing lands with the trellis.
//! - **8-bit, scalar, single + compound.**

use rvp9_core::bitstream::frame::InterpFilter;
use rvp9_core::block::BlockSize;
use rvp9_core::cost::coef_cost::CoefCostTable;
use rvp9_core::inter::filter::{
    FILTER_TAPS, convolve_copy, convolve8, convolve8_horiz, convolve8_vert, kernel_for,
};
use rvp9_core::mode_info::{InterMode, Mv, RefFrame};
use rvp9_core::mv_search::full_pixel::PlaneRef;
use rvp9_core::quant::forward::{PlaneQuantizers, scan_for_forward};
use rvp9_core::transforms::forward::fdct4x4;
use rvp9_core::transforms::scans::scan_for;
use rvp9_core::transforms::{TxSize, TxType};

use crate::rdo::block_error::vp9_block_error;
use crate::rdo::cost_coeffs::cost_coeffs;

/// Number of 4×4 sub-blocks inside an 8×8 parent. Mirrors libvpx
/// `num_4x4_blocks_high_lookup[BLOCK_8X8] *
/// num_4x4_blocks_wide_lookup[BLOCK_8X8] = 4`.
pub const NUM_SUB_BLOCKS: usize = 4;

/// Side of a 4×4 sub-block.
const SUB: usize = 4;

/// libvpx `RDCOST(RM, DM, R, D) = ((RM * R + 256) >> 9) + (D << 7)`
/// (`vp9_rd.h:29`, shared-codebase convention; see
/// `block_yrd::libvpx_rdcost`).
#[inline]
fn rdcost(rdmult: i64, rate: u32, dist: i64) -> i64 {
    let rate_term = (rdmult.saturating_mul(i64::from(rate)) + 256) >> 9;
    let dist_term = dist << 7;
    rate_term.saturating_add(dist_term)
}

/// A single reference for a 4×4 sub-block inter prediction: the plane
/// to read from plus the MV applied to it. For single-ref candidates
/// there is one of these; for compound there are two (averaged).
#[derive(Copy, Clone, Debug)]
pub struct SubRef<'a> {
    /// Reference frame this MV reads from.
    pub ref_frame: RefFrame,
    /// Reference Y plane (pre-translated to the chosen ref frame).
    pub plane: PlaneRef<'a>,
    /// Candidate MV (1/8-pel Q3 units).
    pub mv: Mv,
}

/// Per-sub-block inter candidate. Carries everything the picker needs
/// to compute the **real** RD: the source view, the reference(s) + MV,
/// the interp filter, and the caller-supplied MV/mode-signalling rate.
///
/// Mirrors one `(blk_idx, this_mode, ref)` iteration of libvpx's
/// per-sub-block mode loop (`vp9_rdopt.c:2150-2360`).
pub struct SubBlockCandidate<'a> {
    /// Sub-block index (`0..NUM_SUB_BLOCKS`). libvpx `block = idy*2+idx`.
    pub blk_idx: u8,
    /// Inter mode being evaluated.
    pub mode: InterMode,
    /// Source Y plane.
    pub src_plane: PlaneRef<'a>,
    /// Top-left of the 4×4 sub-block in source plane samples `(col,row)`.
    pub src_offset: (usize, usize),
    /// Reference(s) + MV: one entry for single-ref, two for compound.
    pub refs: &'a [SubRef<'a>],
    /// Interp filter (8-tap kernel applied to the prediction).
    pub interp_filter: InterpFilter,
    /// MV + mode signalling rate, in `vp9_prob_cost` (1/512-bit) units.
    /// libvpx `set_and_cost_bmi_mvs` `brate` (line 2280). The
    /// coefficient rate is computed for real and added on top.
    pub mv_mode_rate: u32,
}

/// Per-sub-block winner. The picker collects one per sub-block.
#[derive(Copy, Clone, Debug)]
pub struct SubBlockWinner {
    /// Sub-block index.
    pub blk_idx: u8,
    /// Winning inter mode.
    pub mode: InterMode,
    /// Winning first ref-frame.
    pub ref_frame: RefFrame,
    /// Winning second ref-frame (`NoRefFrame` for single-ref).
    pub ref_frame1: RefFrame,
    /// Winning first MV.
    pub mv: Mv,
    /// Winning second MV (`ZERO` for single-ref).
    pub mv1: Mv,
    /// Winning total rate (`byrate` coef bits + `brate` MV/mode bits).
    pub rate: u32,
    /// Winning distortion (`bdist`, transform-domain `block_error >> 2`).
    pub dist: i64,
    /// Winning pre-coding `sse` (`bsse`, `ssz >> 2`).
    pub sse: i64,
    /// Winning RD (`brdcost`).
    pub rd: i64,
}

/// Real per-sub-block RD for one `(mode, ref(s), mv, filter)` candidate.
///
/// Mirrors `encode_inter_mb_segment` (`vp9_rdopt.c:1608-1734`) for the
/// single 4×4 tile case, plus the MV/mode-rate fold-in from
/// `rd_pick_best_sub8x8_mode` (line 2346-2349).
///
/// Returns `(byrate, brate, bdist, bsse, brdcost)`:
/// - `byrate` — coefficient bits (`cost_coeffs`),
/// - `brate` — `byrate + mv_mode_rate`,
/// - `bdist` — `block_error >> 2`,
/// - `bsse` — `ssz >> 2`,
/// - `brdcost` — `RDCOST(byrate, bdist) + RDCOST(mv_mode_rate, 0)`.
#[must_use]
fn encode_inter_sub_block(
    cand: &SubBlockCandidate<'_>,
    quant: &PlaneQuantizers,
    coef_costs: &CoefCostTable,
    rdmult: i64,
) -> (u32, u32, i64, i64, i64) {
    // 1. Build the real 4×4 inter prediction. For compound (two refs)
    //    average the two predictions (libvpx `vp9_build_inter_predictor`
    //    with `ref` accumulating into `dst` via the ROUND2 average for
    //    the second ref — we mirror the final averaged result).
    let mut pred = [0u16; SUB * SUB]; // accumulate at u16 for the average
    let n_refs = cand.refs.len().max(1);
    for sub_ref in cand.refs {
        let mut one = [0u8; SUB * SUB];
        build_inter_pred_4x4(
            &sub_ref.plane,
            cand.src_offset.0,
            cand.src_offset.1,
            sub_ref.mv,
            &mut one,
            cand.interp_filter,
        );
        for (acc, &p) in pred.iter_mut().zip(one.iter()) {
            *acc += u16::from(p);
        }
    }
    let mut pred8 = [0u8; SUB * SUB];
    for (out, &acc) in pred8.iter_mut().zip(pred.iter()) {
        // ROUND_POWER_OF_TWO(acc, log2(n_refs)) for n_refs ∈ {1, 2}.
        *out = match n_refs {
            1 => acc as u8,
            _ => ((acc + (n_refs as u16 >> 1)) / n_refs as u16) as u8,
        };
    }

    // 2. Residual = src - pred (libvpx `vpx_subtract_block`).
    let src = &cand.src_plane;
    let (sx, sy) = cand.src_offset;
    let mut src_diff = [0i16; SUB * SUB];
    for r in 0..SUB {
        for c in 0..SUB {
            let s = i16::from(src.data[(sy + r) * src.stride + (sx + c)]);
            let p = i16::from(pred8[r * SUB + c]);
            src_diff[r * SUB + c] = s - p;
        }
    }

    // 3. Forward DCT (the encoder's `fwd_txfm4x4` for the inter path is
    //    `vpx_fdct4x4` — DCT only, not hybrid) + quantize (vp9_quantize_b,
    //    which is what `vpx_quantize_b` maps to). We use the DC/DCT scan.
    let mut coeff = [0i32; 16];
    fdct4x4(&src_diff, &mut coeff, SUB);

    let scan = scan_for(TxSize::Tx4x4, TxType::DctDct);
    let fwd_scan = scan_for_forward(&scan);
    let mut qcoeff = [0i16; 16];
    let mut dqcoeff = [0i16; 16];
    let dequant_pair = [quant.dequant[0], quant.dequant[1]];
    let eob = rvp9_core::quant::forward::vp9_quantize_b(
        &coeff,
        16,
        false,
        &quant.zbin,
        &quant.round,
        &quant.quant,
        &quant.quant_shift,
        &mut qcoeff,
        &mut dqcoeff,
        &dequant_pair,
        fwd_scan,
    );

    // 4. Transform-domain distortion + sse (libvpx lines 1716-1718).
    let (error, ssz) = vp9_block_error(&coeff, &dqcoeff);

    // 5. Coefficient rate (libvpx line 1719).
    let byrate = cost_coeffs(
        &qcoeff,
        fwd_scan,
        eob,
        TxSize::Tx4x4,
        0, // plane_type: Y
        0, // ref_type
        0, // coeff_ctx (start-of-block, see scope note)
        coef_costs,
    );

    // libvpx lines 1729-1733: distortion / sse shifted by 2; the
    // per-sub-block label RD is RDCOST(rdmult, byrate, bdist).
    let bdist = error >> 2;
    let bsse = ssz >> 2;

    // libvpx lines 2346-2349: the brdcost folds in the MV/mode rate
    // (`brdcost += RDCOST(rdmult, brate, 0)`) and the total brate sums
    // the coef bits and the MV/mode bits.
    let brdcost =
        rdcost(rdmult, byrate, bdist).saturating_add(rdcost(rdmult, cand.mv_mode_rate, 0));
    let brate = byrate.saturating_add(cand.mv_mode_rate);

    (byrate, brate, bdist, bsse, brdcost)
}

/// Driver input for [`pick_inter_mode_sub8x8`].
pub struct PickSub8x8Params<'a> {
    /// Parent bsize. Fixed to `Block8x8` in the standalone scope.
    pub parent_bsize: BlockSize,
    /// Candidate set, flat-indexed. The picker walks every entry,
    /// computes its **real** RD, groups by `blk_idx`, and picks the
    /// lowest-RD candidate per sub-block.
    pub candidates: &'a [SubBlockCandidate<'a>],
    /// Y-plane quantizers.
    pub quant: &'a PlaneQuantizers,
    /// Coefficient-bit cost table.
    pub coef_costs: &'a CoefCostTable,
    /// `rdmult` (lambda).
    pub rdmult: i64,
    /// `best_rd_so_far` cap (libvpx line 2406). The picker returns
    /// `None` if the aggregate segment RD exceeds this.
    pub best_rd_so_far: i64,
}

/// Output of [`pick_inter_mode_sub8x8`]. Mirrors libvpx `bsi->{r,d,sse,
/// segment_rd}` (lines 2397-2401).
#[derive(Clone, Debug)]
pub struct PickSub8x8Result {
    /// Per-sub-block winners. Always `NUM_SUB_BLOCKS = 4` entries.
    pub sub_winners: [SubBlockWinner; NUM_SUB_BLOCKS],
    /// Aggregated rate (`bsi->r`).
    pub total_rate: u32,
    /// Aggregated distortion (`bsi->d`).
    pub total_dist: i64,
    /// Aggregated pre-coding sse (`bsi->sse`).
    pub total_sse: i64,
    /// Aggregated segment RD (`bsi->segment_rd`).
    pub total_rd: i64,
}

/// libvpx `rd_pick_best_sub8x8_mode` per-sub-block RD core
/// (`vp9_rdopt.c:2077-2428`). Computes the **real** RD for every
/// `(sub_block, mode, ref)` candidate, picks the lowest-RD candidate
/// per sub-block, and aggregates. Returns `None` if any sub-block has
/// no candidate or the aggregate RD exceeds `best_rd_so_far`.
#[must_use]
pub fn pick_inter_mode_sub8x8(params: &PickSub8x8Params<'_>) -> Option<PickSub8x8Result> {
    assert!(
        matches!(params.parent_bsize, BlockSize::Block8x8),
        "W4.10b standalone scope: parent_bsize must be Block8x8 (got {:?})",
        params.parent_bsize
    );

    // libvpx lines 2150-2360: for each sub-block, evaluate every
    // candidate mode for real, keep the lowest brdcost.
    let mut sub_winners: [Option<SubBlockWinner>; NUM_SUB_BLOCKS] = [None; NUM_SUB_BLOCKS];
    for cand in params.candidates {
        let idx = cand.blk_idx as usize;
        if idx >= NUM_SUB_BLOCKS {
            continue;
        }
        let (_byrate, brate, bdist, bsse, brdcost) =
            encode_inter_sub_block(cand, params.quant, params.coef_costs, params.rdmult);

        let (ref0, ref1, mv0, mv1) = unpack_refs(cand.refs);
        let winner = SubBlockWinner {
            blk_idx: cand.blk_idx,
            mode: cand.mode,
            ref_frame: ref0,
            ref_frame1: ref1,
            mv: mv0,
            mv1,
            rate: brate,
            dist: bdist,
            sse: bsse,
            rd: brdcost,
        };
        // libvpx line 2357: if (brdcost < best_rd) { record }.
        let replace = match sub_winners[idx] {
            None => true,
            Some(prev) => brdcost < prev.rd,
        };
        if replace {
            sub_winners[idx] = Some(winner);
        }
    }

    // libvpx line 2363-2370: any sub-block with no valid candidate → the
    // segment is invalid (returns INT64_MAX).
    let unwrapped: [SubBlockWinner; NUM_SUB_BLOCKS] = {
        let mut out = [SubBlockWinner {
            blk_idx: 0,
            mode: InterMode::ZeroMv,
            ref_frame: RefFrame::NoRefFrame,
            ref_frame1: RefFrame::NoRefFrame,
            mv: Mv::ZERO,
            mv1: Mv::ZERO,
            rate: 0,
            dist: 0,
            sse: 0,
            rd: i64::MAX,
        }; NUM_SUB_BLOCKS];
        for (i, w) in sub_winners.iter().enumerate() {
            out[i] = (*w)?;
        }
        out
    };

    // libvpx lines 2380-2400: aggregate.
    let mut total_rate: u64 = 0;
    let mut total_dist: i64 = 0;
    let mut total_sse: i64 = 0;
    let mut total_rd: i64 = 0;
    for w in &unwrapped {
        total_rate = total_rate.saturating_add(u64::from(w.rate));
        total_dist = total_dist.saturating_add(w.dist);
        total_sse = total_sse.saturating_add(w.sse);
        total_rd = total_rd.saturating_add(w.rd);
    }

    // libvpx line 2406: cap against best_rd_so_far.
    if total_rd >= params.best_rd_so_far {
        return None;
    }

    Some(PickSub8x8Result {
        sub_winners: unwrapped,
        total_rate: u32::try_from(total_rate).unwrap_or(u32::MAX),
        total_dist,
        total_sse,
        total_rd,
    })
}

/// Unpack a candidate's reference list into `(ref0, ref1, mv0, mv1)`.
fn unpack_refs(refs: &[SubRef<'_>]) -> (RefFrame, RefFrame, Mv, Mv) {
    let ref0 = refs.first().map_or(RefFrame::NoRefFrame, |r| r.ref_frame);
    let mv0 = refs.first().map_or(Mv::ZERO, |r| r.mv);
    let ref1 = refs.get(1).map_or(RefFrame::NoRefFrame, |r| r.ref_frame);
    let mv1 = refs.get(1).map_or(Mv::ZERO, |r| r.mv);
    (ref0, ref1, mv0, mv1)
}

/// Build a 4×4 inter prediction at `src_offset + (mv >> 3)` in the
/// reference plane with sub-pel interpolation. Mirror of
/// `handle_inter::build_inter_pred` specialised to a 4×4 tile.
/// libvpx `inter_predictor` (`vp9_reconinter.h:23-32`).
fn build_inter_pred_4x4(
    ref_plane: &PlaneRef<'_>,
    src_x: usize,
    src_y: usize,
    mv: Mv,
    dst: &mut [u8; SUB * SUB],
    filter: InterpFilter,
) {
    let int_row = i32::from(mv.row) >> 3;
    let int_col = i32::from(mv.col) >> 3;
    let subpel_y_q4 = (i32::from(mv.row) & 7) << 1;
    let subpel_x_q4 = (i32::from(mv.col) & 7) << 1;

    let ref_x = src_x as isize + int_col as isize;
    let ref_y = src_y as isize + int_row as isize;

    let x_nonzero = subpel_x_q4 != 0;
    let y_nonzero = subpel_y_q4 != 0;

    let pad_x = if x_nonzero {
        (FILTER_TAPS / 2 - 1) as isize
    } else {
        0
    };
    let pad_y = if y_nonzero {
        (FILTER_TAPS / 2 - 1) as isize
    } else {
        0
    };
    let win_x = ref_x - pad_x;
    let win_y = ref_y - pad_y;
    debug_assert!(
        win_x >= 0 && win_y >= 0,
        "sub8 inter pred read window out of bounds: ({win_x}, {win_y})",
    );

    let src_start = (win_y as usize) * ref_plane.stride + (win_x as usize);
    let src_slice = &ref_plane.data[src_start..];

    let kernel = kernel_for(filter);
    let filter_x = &kernel[subpel_x_q4 as usize];
    let filter_y = &kernel[subpel_y_q4 as usize];

    match (x_nonzero, y_nonzero) {
        (false, false) => convolve_copy(src_slice, ref_plane.stride, dst, SUB, SUB, SUB),
        (true, false) => convolve8_horiz(src_slice, ref_plane.stride, dst, SUB, filter_x, SUB, SUB),
        (false, true) => convolve8_vert(src_slice, ref_plane.stride, dst, SUB, filter_y, SUB, SUB),
        (true, true) => convolve8(
            src_slice,
            ref_plane.stride,
            dst,
            SUB,
            filter_x,
            filter_y,
            SUB,
            SUB,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::cost::coef_cost::build_coef_cost_table;
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    /// Deterministic textured pattern (so the residual is non-trivial).
    fn texture(x: usize, y: usize) -> u8 {
        let v = (x.wrapping_mul(7) ^ y.wrapping_mul(13)).wrapping_add(x * y / 3);
        (40 + (v % 160)) as u8
    }

    /// A textured source + a reference that is the source shifted by
    /// `(shift_row, shift_col)` luma pixels, with an 8-pixel border so a
    /// small MV + 8-tap halo stays in-bounds.
    struct Harness {
        src: Vec<u8>,
        refp: Vec<u8>,
        stride: usize,
        offset: (usize, usize),
    }

    fn build_harness(shift_row: i32, shift_col: i32) -> Harness {
        let pad = 8usize;
        let stride = SUB + 2 * pad;
        let height = SUB + 2 * pad;
        let offset = (pad, pad);
        let mut src = vec![0u8; stride * height];
        let mut refp = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..stride {
                src[y * stride + x] = texture(x, y);
                let sx = (x as i32 + shift_col).clamp(0, stride as i32 - 1) as usize;
                let sy = (y as i32 + shift_row).clamp(0, height as i32 - 1) as usize;
                refp[y * stride + x] = texture(sx, sy);
            }
        }
        Harness {
            src,
            refp,
            stride,
            offset,
        }
    }

    fn src_plane(h: &Harness) -> PlaneRef<'_> {
        PlaneRef {
            data: &h.src,
            stride: h.stride,
            width: h.stride,
            height: h.src.len() / h.stride,
        }
    }
    fn ref_plane(h: &Harness) -> PlaneRef<'_> {
        PlaneRef {
            data: &h.refp,
            stride: h.stride,
            width: h.stride,
            height: h.refp.len() / h.stride,
        }
    }

    /// PROOF W4.10b (DIFFERENTIAL ORACLE — matched MV beats mismatched
    /// MV on REAL transform-domain RD) — libvpx
    /// `encode_inter_mb_segment` (`vp9_rdopt.c:1608-1734`). The
    /// reference is the source shifted by (1, 2) px, so the true motion
    /// for the 4×4 sub-block is mv = (-8, -16) in Q3: that prediction
    /// recovers the source exactly (zero residual → zero distortion).
    /// ZEROMV reads the unshifted reference → a large residual. The
    /// matched NEWMV must produce strictly lower real RD. A
    /// fake/argmin picker that never built the prediction could not
    /// rank these.
    #[test]
    fn w4_10b_matched_mv_beats_zeromv_on_real_motion() {
        let shift_row = 1;
        let shift_col = 2;
        let h = build_harness(shift_row, shift_col);
        let sp = src_plane(&h);
        let rp = ref_plane(&h);
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let true_mv = Mv {
            row: (-shift_row * 8) as i16,
            col: (-shift_col * 8) as i16,
        };
        let zero_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: Mv::ZERO,
        }];
        let new_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: true_mv,
        }];

        let zero_cand = SubBlockCandidate {
            blk_idx: 0,
            mode: InterMode::ZeroMv,
            src_plane: sp,
            src_offset: h.offset,
            refs: &zero_refs,
            interp_filter: InterpFilter::EightTap,
            mv_mode_rate: 0, // ZEROMV pays no MV bits
        };
        let new_cand = SubBlockCandidate {
            blk_idx: 0,
            mode: InterMode::NewMv,
            src_plane: sp,
            src_offset: h.offset,
            refs: &new_refs,
            interp_filter: InterpFilter::EightTap,
            mv_mode_rate: 120, // NEWMV pays an MV-rate penalty
        };

        let (_zr, _zbr, zdist, _zsse, zrd) = encode_inter_sub_block(&zero_cand, &quant, &costs, 80);
        let (_nr, _nbr, ndist, _nsse, nrd) = encode_inter_sub_block(&new_cand, &quant, &costs, 80);

        // The matched MV recovers the source: zero residual → zero dist.
        assert_eq!(
            ndist, 0,
            "matched NEWMV recovers the source exactly → zero distortion",
        );
        assert!(
            zdist > 0,
            "mismatched ZEROMV mispredicts → non-zero distortion (got {zdist})",
        );
        // And the matched MV wins on real RD even paying the MV-rate.
        assert!(
            nrd < zrd,
            "matched NEWMV real RD {nrd} must beat ZEROMV real RD {zrd} \
             (anti-cheat: fake RD can't tell these apart)",
        );
    }

    /// PROOF W4.10b (per-sub-block argmin over REAL RD) — libvpx
    /// `vp9_rdopt.c:2357` keeps the lowest-`brdcost` mode per
    /// sub-block. We give sub-block 0 two candidates: a matched-MV
    /// NEWMV (exact prediction) and a mismatched ZEROMV. The picker must
    /// elect NEWMV purely because its REAL RD is lower — not because a
    /// pre-supplied `rd` field said so.
    #[test]
    fn w4_10b_picker_elects_lower_real_rd_per_subblock() {
        let shift_row = 1;
        let shift_col = 2;
        let h = build_harness(shift_row, shift_col);
        let sp = src_plane(&h);
        let rp = ref_plane(&h);
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let true_mv = Mv {
            row: (-shift_row * 8) as i16,
            col: (-shift_col * 8) as i16,
        };
        let zero_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: Mv::ZERO,
        }];
        let new_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: true_mv,
        }];

        // Every sub-block gets the same two candidates (matched +
        // mismatched). The matched NEWMV should win all four.
        let mut cands = Vec::new();
        for blk in 0..NUM_SUB_BLOCKS as u8 {
            cands.push(SubBlockCandidate {
                blk_idx: blk,
                mode: InterMode::ZeroMv,
                src_plane: sp,
                src_offset: h.offset,
                refs: &zero_refs,
                interp_filter: InterpFilter::EightTap,
                mv_mode_rate: 0,
            });
            cands.push(SubBlockCandidate {
                blk_idx: blk,
                mode: InterMode::NewMv,
                src_plane: sp,
                src_offset: h.offset,
                refs: &new_refs,
                interp_filter: InterpFilter::EightTap,
                mv_mode_rate: 120,
            });
        }
        let params = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &cands,
            quant: &quant,
            coef_costs: &costs,
            rdmult: 80,
            best_rd_so_far: i64::MAX,
        };
        let r = pick_inter_mode_sub8x8(&params).expect("must pick a winner");
        for (i, w) in r.sub_winners.iter().enumerate() {
            assert_eq!(w.blk_idx as usize, i, "winner[{i}] blk_idx mismatch");
            assert_eq!(
                w.mode,
                InterMode::NewMv,
                "sub-block {i} must elect the matched NEWMV (lower real RD)",
            );
            assert_eq!(
                w.dist, 0,
                "matched NEWMV → zero distortion for sub-block {i}"
            );
        }
        // The whole segment recovers the source → zero aggregate dist.
        assert_eq!(r.total_dist, 0);
    }

    /// PROOF W4.10b (compound averages two predictions, real RD) —
    /// libvpx `encode_inter_mb_segment` accumulates predictions for
    /// `ref < 1 + is_compound` (`vp9_rdopt.c:1635`). With two references
    /// that BOTH recover the source at the true MV, the averaged
    /// compound prediction also recovers it (zero distortion). We assert
    /// the compound path runs (two refs averaged) and produces a finite
    /// real RD with zero distortion.
    #[test]
    fn w4_10b_compound_averages_two_real_predictions() {
        let shift_row = 1;
        let shift_col = 2;
        let h = build_harness(shift_row, shift_col);
        let sp = src_plane(&h);
        let rp = ref_plane(&h);
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let true_mv = Mv {
            row: (-shift_row * 8) as i16,
            col: (-shift_col * 8) as i16,
        };
        // Two refs, both the same shifted plane at the true MV → the
        // average equals the (exact) prediction → zero residual.
        let refs = [
            SubRef {
                ref_frame: RefFrame::LastFrame,
                plane: rp,
                mv: true_mv,
            },
            SubRef {
                ref_frame: RefFrame::AltrefFrame,
                plane: rp,
                mv: true_mv,
            },
        ];
        let cand = SubBlockCandidate {
            blk_idx: 0,
            mode: InterMode::NewMv,
            src_plane: sp,
            src_offset: h.offset,
            refs: &refs,
            interp_filter: InterpFilter::EightTap,
            mv_mode_rate: 200,
        };
        let (_byr, _br, dist, _sse, rd) = encode_inter_sub_block(&cand, &quant, &costs, 80);
        assert_eq!(
            dist, 0,
            "compound average of two exact predictions → zero distortion",
        );
        assert!(rd < i64::MAX / 2, "compound RD must be finite (got {rd})");
        // The RD must reflect the MV/mode rate (200) since dist == 0.
        assert!(rd > 0, "compound RD with non-zero MV rate must be positive");
    }

    /// `best_rd_so_far` cap fires: if aggregate segment RD ≥ cap, the
    /// picker returns None. libvpx `vp9_rdopt.c:2406`.
    #[test]
    fn w4_10b_picker_respects_best_rd_so_far_cap() {
        let h = build_harness(0, 0); // identical ref → ZEROMV exact
        let sp = src_plane(&h);
        let rp = ref_plane(&h);
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let quant = init_plane_quantizers(96, QuantizerDeltas::Y);

        let zero_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: Mv::ZERO,
        }];
        let mut cands = Vec::new();
        for blk in 0..NUM_SUB_BLOCKS as u8 {
            cands.push(SubBlockCandidate {
                blk_idx: blk,
                mode: InterMode::ZeroMv,
                src_plane: sp,
                src_offset: h.offset,
                refs: &zero_refs,
                interp_filter: InterpFilter::EightTap,
                // Large MV/mode rate so the aggregate RD is well above 1.
                mv_mode_rate: 5_000,
            });
        }
        let tight = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &cands,
            quant: &quant,
            coef_costs: &costs,
            rdmult: 80,
            best_rd_so_far: 1, // impossibly tight
        };
        assert!(pick_inter_mode_sub8x8(&tight).is_none());

        // Same input with a generous cap succeeds and (identical ref)
        // yields zero distortion.
        let ok = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &cands,
            quant: &quant,
            coef_costs: &costs,
            rdmult: 80,
            best_rd_so_far: i64::MAX,
        };
        let r = pick_inter_mode_sub8x8(&ok).expect("RD below cap");
        assert_eq!(r.total_dist, 0, "identical ref + ZEROMV → zero distortion");
    }

    /// Missing-sub-block guard: if one sub-block has no candidate, the
    /// picker returns None. libvpx `vp9_rdopt.c:2363-2370` marks the
    /// segment invalid.
    #[test]
    fn w4_10b_picker_returns_none_when_sub_block_missing() {
        let h = build_harness(0, 0);
        let sp = src_plane(&h);
        let rp = ref_plane(&h);
        let fc = FrameContext::new_default();
        let costs = build_coef_cost_table(&fc);
        let quant = init_plane_quantizers(96, QuantizerDeltas::Y);
        let zero_refs = [SubRef {
            ref_frame: RefFrame::LastFrame,
            plane: rp,
            mv: Mv::ZERO,
        }];

        // Only sub-blocks 0, 2, 3 have candidates (1 missing).
        let cands: Vec<SubBlockCandidate> = [0u8, 2, 3]
            .iter()
            .map(|&blk| SubBlockCandidate {
                blk_idx: blk,
                mode: InterMode::ZeroMv,
                src_plane: sp,
                src_offset: h.offset,
                refs: &zero_refs,
                interp_filter: InterpFilter::EightTap,
                mv_mode_rate: 0,
            })
            .collect();
        let params = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &cands,
            quant: &quant,
            coef_costs: &costs,
            rdmult: 80,
            best_rd_so_far: i64::MAX,
        };
        assert!(pick_inter_mode_sub8x8(&params).is_none());
    }
}
