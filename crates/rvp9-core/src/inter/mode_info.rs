//! Per-inter-block mode-info reader (`read_inter_block_mode_info`).
//!
//! Mirrors libvpx `read_inter_block_mode_info`
//! (`vp9/decoder/vp9_decodemv.c:691-786`) — the function called by
//! `read_inter_frame_mode_info` after `is_inter_block` returns `true`.
//! It populates a [`ModeInfo`]'s
//! [`BlockMode::Inter`](crate::mode_info::BlockMode::Inter) variant
//! with:
//!
//! - `ref_frame[0]`, `ref_frame[1]` (single or compound, via
//!   [`read_ref_frames`]).
//! - `interp_filter` (from the frame header, or per-block via
//!   [`read_switchable_interp_filter`] when the header's
//!   `interp_filter == Switchable`).
//! - `inter_mode` (NEARESTMV / NEARMV / ZEROMV / NEWMV, via
//!   [`crate::mode_info::read_inter_mode`]).
//! - `mv[0]`, `mv[1]` (the resolved per-ref MVs, including [`read_mv`]
//!   for NEWMV and reference-list lookup for the others).
//!
//! ## Sub-8 inter blocks
//!
//! For `Block4x4` / `Block4x8` / `Block8x4`, libvpx reads `inter_mode` + MV per-bmi (per 4×4 sub-block).
//! The top-level `mi.mode` is set to the last sub-block's mode, and `mi.mv` is the last sub-block's MV pair.
//! This module mirrors that pattern (see the sub-8 branch of [`read_inter_block_mode_info`]).
//! The sub-8 handling threads `append_sub8x8_mvs_for_idx` for NEAREST/NEAR candidates and shares one
//! `find_mv_refs` per-ref result for NEW (matching libvpx `got_mv_refs_for_new`).
//!
//! ## Scope (Stage 3 Task D.1)
//!
//! This file lands the read path; the consuming reconstruction
//! driver (D.2 — per-inter-block predict + reconstruct) wires the
//! resulting `ModeInfo` to [`crate::inter::scale::inter_predictor`]
//! and the rest of the inter pipeline.
//!
//! Segmentation features (`SEG_LVL_SKIP`, `SEG_LVL_REF_FRAME`) are
//! out of scope for the Stage 3 v1 floor — the keyframe path
//! (K2 `decode_block`) similarly ignores them. The hooks are
//! reserved on [`InterModeReadParams`] for a later pass.

use crate::bitstream::{InterpFilter, ReferenceMode};
use crate::block::BlockSize;
use crate::entropy::{
    BoolReader, BoolWriter, FrameContext, NmvCounts, COMP_INTER_CONTEXTS, INTER_MODE_CONTEXTS,
    INTRA_INTER_CONTEXTS, REF_CONTEXTS, SWITCHABLE_FILTERS, SWITCHABLE_FILTER_CONTEXTS,
};
use crate::inter::mv::{read_mv, write_mv, write_tree};
use crate::inter::mv_ref::{
    append_sub8x8_mvs_for_idx, find_mv_refs, lower_mv_precision, BlockBounds, NeighbourLookup,
    Position, PrevFrameMvRef,
};
use crate::mode_info::{
    read_inter_mode, BlockMode, BmiInfo, InterMode, ModeInfo, Mv, RefFrame, INTER_MODE_TREE,
};

/// libvpx `vp9_switchable_interp_tree` (`vp9_entropymode.c:337-338`).
/// 4-entry flat tree decoding one of the 3 switchable filters from 2
/// boolean probabilities (`switchable_interp_prob[ctx]`):
///
/// ```text
///       node 0
///      /       \
///  EIGHTTAP     node 2
///              /       \
///   EIGHTTAP_SMOOTH   EIGHTTAP_SHARP
/// ```
#[rustfmt::skip]
pub static SWITCHABLE_INTERP_TREE: [i8; 2 * SWITCHABLE_FILTERS - 2] = [
    -(InterpFilter::EightTap as i8), 2,
    -(InterpFilter::EightTapSmooth as i8), -(InterpFilter::EightTapSharp as i8),
];

/// libvpx `mode_2_counter[MB_MODE_COUNT]`
/// (`vp9_mvref_common.h:47-62`). Per `PREDICTION_MODE`, the
/// per-neighbour contribution to the `context_counter` summed in
/// `get_mode_context` (= 9 for intra, 0 for NEAREST/NEAR, 3 for ZERO,
/// 1 for NEW). The intra block's `BlockMode::Intra` always
/// contributes 9; the inter modes contribute per [`InterMode::idx`].
const MODE_2_COUNTER_INTER: [i32; 4] = [
    /* NEARESTMV */ 0,
    /* NEARMV    */ 0,
    /* ZEROMV    */ 3,
    /* NEWMV     */ 1,
];
const MODE_2_COUNTER_INTRA: i32 = 9;

/// libvpx `counter_to_context[19]` (`vp9_mvref_common.h:67-87`).
/// Maps the summed-context-counter (0..=18) into the
/// [`INTER_MODE_CONTEXTS`] index used to fetch
/// `fc.inter_mode_probs[ctx]`. Values 5/7/8/11/13..=17 are
/// `INVALID_CASE` — libvpx asserts they're unreachable; we map them
/// to ctx 0 (BOTH_PREDICTED) defensively.
#[rustfmt::skip]
const COUNTER_TO_CONTEXT: [u8; 19] = [
    2, // BOTH_PREDICTED
    3, // NEW_PLUS_NON_INTRA
    4, // BOTH_NEW
    1, // ZERO_PLUS_PREDICTED
    3, // NEW_PLUS_NON_INTRA
    0, // INVALID_CASE (mapped to 0)
    0, // BOTH_ZERO
    0, // INVALID_CASE
    0, // INVALID_CASE
    5, // INTRA_PLUS_NON_INTRA
    5, // INTRA_PLUS_NON_INTRA
    0, // INVALID_CASE
    5, // INTRA_PLUS_NON_INTRA
    0, // INVALID_CASE
    0, // INVALID_CASE
    0, // INVALID_CASE
    0, // INVALID_CASE
    0, // INVALID_CASE
    6, // BOTH_INTRA
];

/// Compute the inter-mode context for a block at the given
/// `(mi_row, mi_col)` (tile-relative, mi-units). Mirrors libvpx
/// `get_mode_context` (`vp9_decodemv.c:670-689`).
///
/// Scans the first 2 entries of the block-size-specific neighbour
/// table ([`crate::inter::mv_ref::MV_REF_BLOCKS`]) and sums each
/// candidate's `mode_2_counter` contribution. The summed counter is
/// then mapped through [`COUNTER_TO_CONTEXT`].
#[must_use]
pub fn get_mode_context<N: NeighbourLookup>(
    block_size: BlockSize,
    neighbours: &N,
) -> usize {
    let mv_ref_search = &crate::inter::mv_ref::MV_REF_BLOCKS[block_size.idx()];
    let mut counter: i32 = 0;
    for &mv_ref in mv_ref_search.iter().take(2) {
        if let Some(candidate) = neighbours.get(mv_ref) {
            counter += match candidate.mode {
                BlockMode::Intra { .. } => MODE_2_COUNTER_INTRA,
                BlockMode::Inter { mode, .. } => MODE_2_COUNTER_INTER[mode.idx()],
            };
        }
    }
    // Clamp to the valid range — `counter` is always 0..=18 by
    // construction, but defensively floor at 0 / cap at 18 to match
    // libvpx's `counter_to_context` array bounds.
    let idx = counter.clamp(0, 18) as usize;
    COUNTER_TO_CONTEXT[idx] as usize
}

/// libvpx `vp9_get_pred_context_switchable_interp`
/// (`vp9_pred_common.h:69-88`). 0..=3 context derived from the
/// (above, left) neighbours' `interp_filter`.
#[must_use]
pub fn get_pred_context_switchable_interp(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
) -> usize {
    let neighbour_filter = |mi: Option<&ModeInfo>| -> u8 {
        match mi {
            None => SWITCHABLE_FILTERS as u8, // libvpx's "no neighbour" sentinel
            Some(m) => match m.mode {
                BlockMode::Inter { interp_filter, .. } => interp_filter as u8,
                // Intra neighbours have `interp_filter ==
                // SWITCHABLE_FILTERS` in libvpx
                // (`read_intra_block_mode_info` :383). Mirror that.
                BlockMode::Intra { .. } => SWITCHABLE_FILTERS as u8,
            },
        }
    };
    let left_type = neighbour_filter(left);
    let above_type = neighbour_filter(above);
    let sentinel = SWITCHABLE_FILTERS as u8;
    let ctx = if left_type == above_type {
        left_type
    } else if left_type == sentinel {
        above_type
    } else if above_type == sentinel {
        left_type
    } else {
        sentinel
    };
    // Final context is in 0..=SWITCHABLE_FILTER_CONTEXTS-1 = 0..=3.
    // Both `left_type == above_type == sentinel` and the "mismatch"
    // path produce `sentinel = 3`, which is the same ctx as the
    // legitimate `EightTapSharp = 2 + 1`? — Note libvpx's
    // SWITCHABLE_FILTERS = 3 sentinel collides with neither real
    // filter (0, 1, 2), and the function returns `SWITCHABLE_FILTERS
    // = 3` as a fourth context value. So our mapping is correct.
    debug_assert!((ctx as usize) < SWITCHABLE_FILTER_CONTEXTS);
    ctx as usize
}

/// Read the per-block switchable-interpolation filter. Mirrors libvpx
/// `read_switchable_interp_filter` (`vp9_decodemv.c:343-352`). Only
/// called when the frame's `interp_filter == Switchable`.
///
/// `mode_counts`, when `Some`, accumulates
/// `switchable_interp[ctx][filter]` for backward adaptation. Mirrors
/// libvpx `++counts->switchable_interp[ctx][type]`
/// (`vp9_decodemv.c:350`).
#[must_use]
pub fn read_switchable_interp_filter(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    mode_counts: Option<&mut crate::entropy::adapt::FrameCounts>,
) -> InterpFilter {
    let ctx = get_pred_context_switchable_interp(above, left);
    let probs = &fc.switchable_interp_prob[ctx];
    let raw = reader.read_tree(&SWITCHABLE_INTERP_TREE, probs);
    let filter = match raw {
        0 => InterpFilter::EightTap,
        1 => InterpFilter::EightTapSmooth,
        2 => InterpFilter::EightTapSharp,
        // Unreachable: SWITCHABLE_INTERP_TREE only emits leaves 0..=2.
        _ => unreachable!("switchable interp tree produced {raw}"),
    };
    if let Some(c) = mode_counts {
        c.switchable_interp[ctx][raw as usize] += 1;
    }
    filter
}

/// libvpx `has_second_ref` (`vp9_blockd.h:107-109`): `true` iff the
/// inter block's `ref_frame[1] > INTRA_FRAME`.
#[inline]
fn has_second_ref(mi: Option<&ModeInfo>) -> bool {
    match mi.map(|m| &m.mode) {
        Some(BlockMode::Inter { ref_frames, .. }) => ref_frames[1].is_inter(),
        _ => false,
    }
}

/// libvpx `is_inter_block` for an `Option<&ModeInfo>`.
#[inline]
fn is_inter_block(mi: Option<&ModeInfo>) -> bool {
    matches!(mi.map(|m| &m.mode), Some(BlockMode::Inter { .. }))
}

/// Slot 0 ref-frame of a candidate neighbour. Mirrors libvpx
/// `above_mi->ref_frame[0]`. Intra / missing → `IntraFrame`.
#[inline]
fn ref0(mi: Option<&ModeInfo>) -> RefFrame {
    match mi.map(|m| &m.mode) {
        Some(BlockMode::Inter { ref_frames, .. }) => ref_frames[0],
        _ => RefFrame::IntraFrame,
    }
}

/// Slot 1 ref-frame of a candidate neighbour. Mirrors libvpx
/// `above_mi->ref_frame[1]`. Intra / missing → `NoRefFrame`.
#[inline]
fn ref1(mi: Option<&ModeInfo>) -> RefFrame {
    match mi.map(|m| &m.mode) {
        Some(BlockMode::Inter { ref_frames, .. }) => ref_frames[1],
        _ => RefFrame::NoRefFrame,
    }
}

/// libvpx `vp9_get_reference_mode_context`
/// (`vp9_pred_common.c:42-82`). 0..=4 context for the
/// `compound_vs_single` boolean read.
#[must_use]
pub fn get_reference_mode_context(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    comp_fixed_ref: RefFrame,
) -> usize {
    let has_above = above.is_some();
    let has_left = left.is_some();
    if has_above && has_left {
        let a_has_second = has_second_ref(above);
        let l_has_second = has_second_ref(left);
        if !a_has_second && !l_has_second {
            // both single-ref: ctx ∈ {0, 1}.
            let a_match = ref0(above) == comp_fixed_ref;
            let l_match = ref0(left) == comp_fixed_ref;
            usize::from(a_match ^ l_match)
        } else if !a_has_second {
            2 + usize::from(
                ref0(above) == comp_fixed_ref || !is_inter_block(above),
            )
        } else if !l_has_second {
            2 + usize::from(
                ref0(left) == comp_fixed_ref || !is_inter_block(left),
            )
        } else {
            4
        }
    } else if has_above || has_left {
        let edge = if has_above { above } else { left };
        if !has_second_ref(edge) {
            usize::from(ref0(edge) == comp_fixed_ref)
        } else {
            3
        }
    } else {
        1
    }
}

/// libvpx `vp9_get_pred_context_comp_ref_p`
/// (`vp9_pred_common.c:85-165`). 0..=4 context for the compound
/// "variable ref" bit.
#[must_use]
pub fn get_pred_context_comp_ref_p(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    sign_bias: &[bool; 4],
    comp_fixed_ref: RefFrame,
    comp_var_ref: [RefFrame; 2],
) -> usize {
    let above_in_image = above.is_some();
    let left_in_image = left.is_some();
    let fix_ref_idx = sign_bias[comp_fixed_ref as usize] as usize;
    let var_ref_idx = 1 - fix_ref_idx;
    let var1 = comp_var_ref[1];

    let var_ref_of = |mi: Option<&ModeInfo>| -> RefFrame {
        match mi.map(|m| &m.mode) {
            Some(BlockMode::Inter { ref_frames, .. }) => ref_frames[var_ref_idx],
            _ => RefFrame::IntraFrame,
        }
    };

    if above_in_image && left_in_image {
        let above_intra = !is_inter_block(above);
        let left_intra = !is_inter_block(left);
        if above_intra && left_intra {
            2
        } else if above_intra || left_intra {
            let edge = if above_intra { left } else { above };
            if !has_second_ref(edge) {
                1 + 2 * usize::from(ref0(edge) != var1)
            } else {
                1 + 2 * usize::from(var_ref_of(edge) != var1)
            }
        } else {
            // inter/inter
            let l_sg = !has_second_ref(left);
            let a_sg = !has_second_ref(above);
            let vrfa = if a_sg { ref0(above) } else { var_ref_of(above) };
            let vrfl = if l_sg { ref0(left) } else { var_ref_of(left) };
            if vrfa == vrfl && var1 == vrfa {
                0
            } else if l_sg && a_sg {
                if (vrfa == comp_fixed_ref && vrfl == comp_var_ref[0])
                    || (vrfl == comp_fixed_ref && vrfa == comp_var_ref[0])
                {
                    4
                } else if vrfa == vrfl {
                    3
                } else {
                    1
                }
            } else if l_sg || a_sg {
                let vrfc = if l_sg { vrfa } else { vrfl };
                let rfs = if a_sg { vrfa } else { vrfl };
                if vrfc == var1 && rfs != var1 {
                    1
                } else if rfs == var1 && vrfc != var1 {
                    2
                } else {
                    4
                }
            } else if vrfa == vrfl {
                4
            } else {
                2
            }
        }
    } else if above_in_image || left_in_image {
        let edge = if above_in_image { above } else { left };
        if !is_inter_block(edge) {
            2
        } else if has_second_ref(edge) {
            4 * usize::from(var_ref_of(edge) != var1)
        } else {
            3 * usize::from(ref0(edge) != var1)
        }
    } else {
        2
    }
}

/// libvpx `vp9_get_pred_context_single_ref_p1`
/// (`vp9_pred_common.c:167-231`). 0..=4 context for the single-ref
/// first selection bit (`bit0` of single-ref decode).
#[must_use]
pub fn get_pred_context_single_ref_p1(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
) -> usize {
    let has_above = above.is_some();
    let has_left = left.is_some();
    if has_above && has_left {
        let above_intra = !is_inter_block(above);
        let left_intra = !is_inter_block(left);
        if above_intra && left_intra {
            2
        } else if above_intra || left_intra {
            let edge = if above_intra { left } else { above };
            if !has_second_ref(edge) {
                4 * usize::from(ref0(edge) == RefFrame::LastFrame)
            } else {
                1 + usize::from(
                    ref0(edge) == RefFrame::LastFrame
                        || ref1(edge) == RefFrame::LastFrame,
                )
            }
        } else {
            let a_has_second = has_second_ref(above);
            let l_has_second = has_second_ref(left);
            let above0 = ref0(above);
            let above1 = ref1(above);
            let left0 = ref0(left);
            let left1 = ref1(left);
            if a_has_second && l_has_second {
                1 + usize::from(
                    above0 == RefFrame::LastFrame
                        || above1 == RefFrame::LastFrame
                        || left0 == RefFrame::LastFrame
                        || left1 == RefFrame::LastFrame,
                )
            } else if a_has_second || l_has_second {
                let rfs = if !a_has_second { above0 } else { left0 };
                let crf1 = if a_has_second { above0 } else { left0 };
                let crf2 = if a_has_second { above1 } else { left1 };
                if rfs == RefFrame::LastFrame {
                    3 + usize::from(
                        crf1 == RefFrame::LastFrame || crf2 == RefFrame::LastFrame,
                    )
                } else {
                    usize::from(
                        crf1 == RefFrame::LastFrame || crf2 == RefFrame::LastFrame,
                    )
                }
            } else {
                2 * usize::from(above0 == RefFrame::LastFrame)
                    + 2 * usize::from(left0 == RefFrame::LastFrame)
            }
        }
    } else if has_above || has_left {
        let edge = if has_above { above } else { left };
        if !is_inter_block(edge) {
            2
        } else if !has_second_ref(edge) {
            4 * usize::from(ref0(edge) == RefFrame::LastFrame)
        } else {
            1 + usize::from(
                ref0(edge) == RefFrame::LastFrame || ref1(edge) == RefFrame::LastFrame,
            )
        }
    } else {
        2
    }
}

/// libvpx `vp9_get_pred_context_single_ref_p2`
/// (`vp9_pred_common.c:233-316`). 0..=4 context for the single-ref
/// second selection bit (`bit1` — GOLDEN vs ALTREF).
#[must_use]
pub fn get_pred_context_single_ref_p2(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
) -> usize {
    let has_above = above.is_some();
    let has_left = left.is_some();
    if has_above && has_left {
        let above_intra = !is_inter_block(above);
        let left_intra = !is_inter_block(left);
        if above_intra && left_intra {
            2
        } else if above_intra || left_intra {
            let edge = if above_intra { left } else { above };
            if !has_second_ref(edge) {
                if ref0(edge) == RefFrame::LastFrame {
                    3
                } else {
                    4 * usize::from(ref0(edge) == RefFrame::GoldenFrame)
                }
            } else {
                1 + 2 * usize::from(
                    ref0(edge) == RefFrame::GoldenFrame
                        || ref1(edge) == RefFrame::GoldenFrame,
                )
            }
        } else {
            let a_has_second = has_second_ref(above);
            let l_has_second = has_second_ref(left);
            let above0 = ref0(above);
            let above1 = ref1(above);
            let left0 = ref0(left);
            let left1 = ref1(left);
            if a_has_second && l_has_second {
                if above0 == left0 && above1 == left1 {
                    3 * usize::from(
                        above0 == RefFrame::GoldenFrame
                            || above1 == RefFrame::GoldenFrame
                            || left0 == RefFrame::GoldenFrame
                            || left1 == RefFrame::GoldenFrame,
                    )
                } else {
                    2
                }
            } else if a_has_second || l_has_second {
                let rfs = if !a_has_second { above0 } else { left0 };
                let crf1 = if a_has_second { above0 } else { left0 };
                let crf2 = if a_has_second { above1 } else { left1 };
                if rfs == RefFrame::GoldenFrame {
                    3 + usize::from(
                        crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame,
                    )
                } else if rfs == RefFrame::AltrefFrame {
                    usize::from(
                        crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame,
                    )
                } else {
                    1 + 2 * usize::from(
                        crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame,
                    )
                }
            } else if above0 == RefFrame::LastFrame && left0 == RefFrame::LastFrame {
                3
            } else if above0 == RefFrame::LastFrame || left0 == RefFrame::LastFrame {
                let edge0 = if above0 == RefFrame::LastFrame { left0 } else { above0 };
                4 * usize::from(edge0 == RefFrame::GoldenFrame)
            } else {
                2 * usize::from(above0 == RefFrame::GoldenFrame)
                    + 2 * usize::from(left0 == RefFrame::GoldenFrame)
            }
        }
    } else if has_above || has_left {
        let edge = if has_above { above } else { left };
        if !is_inter_block(edge)
            || (ref0(edge) == RefFrame::LastFrame && !has_second_ref(edge))
        {
            2
        } else if !has_second_ref(edge) {
            4 * usize::from(ref0(edge) == RefFrame::GoldenFrame)
        } else {
            3 * usize::from(
                ref0(edge) == RefFrame::GoldenFrame || ref1(edge) == RefFrame::GoldenFrame,
            )
        }
    } else {
        2
    }
}

/// Decode `ref_frame[0..2]` for an inter block. Mirrors libvpx
/// `read_ref_frames` (`vp9_decodemv.c:303-341`). Skips the
/// `SEG_LVL_REF_FRAME` segmentation override (out of scope for v1).
///
/// Returns `(ref0, ref1, is_compound)`. For single-ref blocks `ref1
/// == NoRefFrame`.
///
/// Parameters:
/// - `reader`: bool reader positioned at the ref-frame bits.
/// - `fc`: frame context (probability bank).
/// - `params`: frame-level inter parameters (reference_mode +
///   comp_fixed_ref / comp_var_ref).
/// - `above` / `left`: the block's spatial neighbours, for context
///   computation.
pub fn read_ref_frames(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    mut mode_counts: Option<&mut crate::entropy::adapt::FrameCounts>,
) -> (RefFrame, RefFrame, bool) {
    // First: read or infer the block reference mode (Single vs
    // Compound). `Select` triggers a read; the other two are
    // frame-fixed.
    let is_compound = match params.reference_mode {
        ReferenceMode::Select => {
            let ctx = get_reference_mode_context(above, left, params.comp_fixed_ref);
            debug_assert!(ctx < COMP_INTER_CONTEXTS);
            let bit = reader.read(fc.comp_inter_prob[ctx]);
            if let Some(c) = mode_counts.as_deref_mut() {
                c.comp_inter[ctx][bit as usize] += 1;
            }
            bit != 0
        }
        ReferenceMode::Compound => true,
        ReferenceMode::Single => false,
    };

    if is_compound {
        {
            let idx = params.sign_bias[params.comp_fixed_ref as usize] as usize;
            let ctx = get_pred_context_comp_ref_p(
                above,
                left,
                &params.sign_bias,
                params.comp_fixed_ref,
                params.comp_var_ref,
            );
            debug_assert!(ctx < REF_CONTEXTS);
            let bit = reader.read(fc.comp_ref_prob[ctx]);
            if let Some(c) = mode_counts {
                c.comp_ref[ctx][bit as usize] += 1;
            }
            let var = params.comp_var_ref[bit as usize];
            let mut ref_frames = [RefFrame::NoRefFrame; 2];
            ref_frames[idx] = params.comp_fixed_ref;
            ref_frames[1 - idx] = var;
            (ref_frames[0], ref_frames[1], true)
        }
    } else {
        {
            let ctx0 = get_pred_context_single_ref_p1(above, left);
            debug_assert!(ctx0 < REF_CONTEXTS);
            let bit0 = reader.read(fc.single_ref_prob[ctx0][0]);
            if let Some(c) = mode_counts.as_deref_mut() {
                c.single_ref[ctx0][0][bit0 as usize] += 1;
            }
            let r0 = if bit0 != 0 {
                let ctx1 = get_pred_context_single_ref_p2(above, left);
                debug_assert!(ctx1 < REF_CONTEXTS);
                let bit1 = reader.read(fc.single_ref_prob[ctx1][1]);
                if let Some(c) = mode_counts {
                    c.single_ref[ctx1][1][bit1 as usize] += 1;
                }
                if bit1 != 0 {
                    RefFrame::AltrefFrame
                } else {
                    RefFrame::GoldenFrame
                }
            } else {
                RefFrame::LastFrame
            };
            (r0, RefFrame::NoRefFrame, false)
        }
    }
}

// (helpers above; reader continues below)

/// libvpx `get_intra_inter_context` (`vp9_pred_common.h:97-111`). 0..=3
/// context for the per-block `is_inter` boolean. Exposed alongside
/// the inter-mode reader for callers that pre-compute it before
/// reaching here.
#[must_use]
pub fn get_intra_inter_context(
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
) -> usize {
    let has_above = above.is_some();
    let has_left = left.is_some();
    let ctx = if has_above && has_left {
        let above_intra = !is_inter_block(above);
        let left_intra = !is_inter_block(left);
        if left_intra && above_intra {
            3
        } else {
            usize::from(left_intra || above_intra)
        }
    } else if has_above || has_left {
        let edge = if has_above { above } else { left };
        2 * usize::from(!is_inter_block(edge))
    } else {
        0
    };
    debug_assert!(ctx < INTRA_INTER_CONTEXTS);
    ctx
}

/// Aggregated frame-level inter parameters needed by
/// [`read_inter_block_mode_info`]. Mirrors the libvpx fields the
/// per-block reader reaches into:
///
/// - `allow_high_precision_mv` — frame-header bit.
/// - `interp_filter` — frame-header field; per-block reads happen
///   only when this is `Switchable`.
/// - `reference_mode` — compressed-header field (single / compound /
///   select).
/// - `comp_fixed_ref` / `comp_var_ref` — derived from
///   `ref_frame_sign_bias` via libvpx `vp9_setup_compound_reference_mode`
///   (`vp9_pred_common.c:24-40`).
/// - `sign_bias` — 4-slot sign-bias table
///   (`[INTRA, LAST, GOLDEN, ALTREF]`) used by both the ref-frame
///   context fns and `find_mv_refs`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InterModeReadParams {
    /// `cm->allow_high_precision_mv` — 1/8-pel MV residual gate.
    pub allow_high_precision_mv: bool,
    /// `cm->interp_filter` — frame-level filter selector.
    pub interp_filter: InterpFilter,
    /// `cm->reference_mode` — block ref-mode selector.
    pub reference_mode: ReferenceMode,
    /// `cm->comp_fixed_ref` from
    /// [`setup_compound_reference_mode`].
    pub comp_fixed_ref: RefFrame,
    /// `cm->comp_var_ref[0..2]` — variable-ref candidates for
    /// compound mode.
    pub comp_var_ref: [RefFrame; 2],
    /// 4-slot sign-bias `[INTRA, LAST, GOLDEN, ALTREF]`
    /// (see [`crate::inter::mv_ref::sign_bias_from_header`]).
    pub sign_bias: [bool; 4],
}

/// libvpx `vp9_setup_compound_reference_mode`
/// (`vp9_pred_common.c:24-40`). Derive `(comp_fixed_ref, comp_var_ref[0],
/// comp_var_ref[1])` from the per-ref sign-bias table.
///
/// Returns `(fixed, var)`. Callers typically wrap this once per frame
/// (when `reference_mode != Single`) to populate the
/// [`InterModeReadParams`] fixed/var slots.
#[must_use]
pub fn setup_compound_reference_mode(
    sign_bias: &[bool; 4],
) -> (RefFrame, [RefFrame; 2]) {
    let last = sign_bias[RefFrame::LastFrame as usize];
    let golden = sign_bias[RefFrame::GoldenFrame as usize];
    let altref = sign_bias[RefFrame::AltrefFrame as usize];
    if last == golden {
        (
            RefFrame::AltrefFrame,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
        )
    } else if last == altref {
        (
            RefFrame::GoldenFrame,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
        )
    } else {
        (
            RefFrame::LastFrame,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
        )
    }
}

/// Decode the per-inter-block mode info. Mirrors libvpx
/// `read_inter_block_mode_info` (`vp9_decodemv.c:691-786`).
///
/// Populates `out`'s `mode` (`BlockMode::Inter`) with the resolved
/// `(inter_mode, ref_frames, mvs, interp_filter)`, and writes the
/// per-bmi sub-block info for sub-8 partitions.
///
/// Inputs:
/// - `reader`: positioned just past the per-block `is_inter` bit
///   (i.e. at the ref-frame bits).
/// - `fc`: frame context for probability lookups.
/// - `counts`: optional MV-count sink for backward adaptation. The
///   counts for `inter_mode`/`comp_inter`/`comp_ref`/`single_ref`/
///   `switchable_interp` are NOT yet plumbed (no top-level
///   `FrameCounts` in v1); only `read_mv`'s `NmvCounts` sink is
///   wired today, matching the existing B.1 contract.
/// - `params`: frame-level inter parameters.
/// - `neighbours`: 2-D neighbour lookup (decoder threads a
///   [`crate::inter::mv_ref::NeighbourLookup`] adapter).
/// - `block_size`: the inter block's sb_type.
/// - `bounds`: per-block edge bounds (for MV clamping in
///   [`find_mv_refs`]).
/// - `prev_mv`: optional co-located previous-frame MV ref (v1 stub —
///   pass `None` until the prev-frame MV grid is wired).
/// - `out`: target [`ModeInfo`]; only `mode` / `bmi` are touched.
///   The caller must have filled in `block_size` / `tx_size` /
///   `segment_id` / `skip` already.
#[allow(clippy::too_many_arguments)]
pub fn read_inter_block_mode_info<N: NeighbourLookup>(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    counts: Option<&mut NmvCounts>,
    mut mode_counts: Option<&mut crate::entropy::adapt::FrameCounts>,
    params: &InterModeReadParams,
    neighbours: &N,
    block_size: BlockSize,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    out: &mut ModeInfo,
) {
    let above = neighbours.get(Position { row: -1, col: 0 });
    let left = neighbours.get(Position { row: 0, col: -1 });

    // 1. Reference frames.
    let (ref0, ref1, is_compound) =
        read_ref_frames(reader, fc, params, above, left, mode_counts.as_deref_mut());

    // 2. Parent inter mode + interpolation filter. For sub-8
    //    libvpx reads `b_mode` per-bmi later; the parent `mi->mode`
    //    is assigned to the last sub-block's mode after the loop.
    let inter_mode_ctx = get_mode_context(block_size, neighbours);
    debug_assert!(inter_mode_ctx < INTER_MODE_CONTEXTS);
    let parent_mode = if (block_size as u8) >= (BlockSize::Block8x8 as u8) {
        read_inter_mode(
            reader,
            &fc.inter_mode_probs[inter_mode_ctx],
            inter_mode_ctx,
            mode_counts.as_deref_mut(),
        )
    } else {
        InterMode::NearestMv // placeholder — overwritten per-bmi below
    };

    let interp_filter = if params.interp_filter == InterpFilter::Switchable {
        read_switchable_interp_filter(reader, fc, above, left, mode_counts.as_deref_mut())
    } else {
        params.interp_filter
    };

    // 3. Resolve MVs. Two paths: parent (≥8×8) vs sub-8.
    let (final_mode, final_mvs, bmi) = if (block_size as u8) >= (BlockSize::Block8x8 as u8) {
        let mvs = resolve_parent_mvs(
            reader,
            fc,
            counts,
            params,
            neighbours,
            block_size,
            bounds,
            prev_mv,
            parent_mode,
            [ref0, ref1],
            is_compound,
        );
        (parent_mode, mvs, [BmiInfo::default(); 4])
    } else {
        let (mode, mvs, bmi) = resolve_sub8_bmi(
            reader,
            fc,
            counts,
            mode_counts,
            params,
            neighbours,
            block_size,
            bounds,
            prev_mv,
            inter_mode_ctx,
            [ref0, ref1],
            is_compound,
        );
        (mode, mvs, bmi)
    };

    out.mode = BlockMode::Inter {
        mode: final_mode,
        ref_frames: [ref0, ref1],
        mvs: final_mvs,
        interp_filter,
    };
    out.bmi = bmi;
}

/// Errors returned by [`write_inter_block_mode_info`].
///
/// Covers both the parent inter-block path (block_size ≥
/// `Block8x8`, landed in S4.16b) and the sub-8 inter-block path
/// (`Block4x4` / `Block4x8` / `Block8x4`, landed in S4.16e — mirrors
/// `resolve_sub8_bmi` on the decoder side).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InterModeWriteError {
    /// The supplied `ModeInfo::mode` was [`BlockMode::Intra`]. The
    /// inter-block writer must only be called on inter blocks; the
    /// caller's mode driver is expected to dispatch intra blocks to
    /// the intra writer.
    NotInter,
    /// A sub-8 sub-block's `bmi[j].as_mode` did not map to a valid
    /// [`InterMode`] via [`InterMode::from_idx`]. The caller is
    /// expected to populate `as_mode = sub_mode.idx() as u8` for
    /// each of the 4 sub-blocks of a sub-8 inter block.
    Sub8InvalidSubMode {
        /// The 4×4 sub-block index (`0..=3`).
        j: usize,
        /// The raw `as_mode` byte that failed to decode.
        raw: u8,
    },
    /// The parent's `mi.mode` (inter mode discriminant) did not
    /// match the last sub-block's `bmi[3].as_mode` for a sub-8 inter
    /// block. Decoder convention is that the parent inherits the
    /// last sub-block's mode (libvpx `mi->mode = b_mode` on the last
    /// loop iteration); the writer enforces the same invariant on
    /// its caller.
    Sub8ParentModeMismatch {
        /// Parent's inter-mode index.
        parent: u8,
        /// `bmi[3].as_mode` (last sub-block's mode index).
        last_sub: u8,
    },
}

impl core::fmt::Display for InterModeWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotInter => f.write_str(
                "write_inter_block_mode_info: expected ModeInfo::mode = BlockMode::Inter, got intra",
            ),
            Self::Sub8InvalidSubMode { j, raw } => write!(
                f,
                "write_inter_block_mode_info: sub-8 bmi[{j}].as_mode = {raw} is not a valid InterMode index"
            ),
            Self::Sub8ParentModeMismatch { parent, last_sub } => write!(
                f,
                "write_inter_block_mode_info: sub-8 parent mode index {parent} != bmi[3].as_mode {last_sub}"
            ),
        }
    }
}

impl std::error::Error for InterModeWriteError {}

/// Encode the per-inter-block mode info. Mirrors libvpx
/// `vp9/encoder/vp9_bitstream.c:pack_inter_mode_mvs` for both the
/// parent-block (block_size ≥ `Block8x8`) and sub-8 (`Block4x4` /
/// `Block4x8` / `Block8x4`) paths.
///
/// This is the inverse of [`read_inter_block_mode_info`] — round-tripping
/// `(fc, params, neighbours, block_size, bounds, prev_mv, mi)` through
/// `write_inter_block_mode_info` and then `read_inter_block_mode_info`
/// must recover the same `(mode, ref_frames, mvs, interp_filter)`.
///
/// Stages, in the exact wire order the decoder consumes them:
/// 1. Reference frames — single or compound, via the
///    [`read_ref_frames`] inverse. The `is_compound` decision is
///    derived from `mi.ref_frames[1]` (compound iff
///    `ref_frames[1].is_inter()`); the `Select` reference-mode emits
///    one boolean against `comp_inter_prob[ctx]`, otherwise no
///    `comp_inter` bit is emitted.
/// 2. Parent inter mode — [`INTER_MODE_TREE`] via
///    `fc.inter_mode_probs[ctx]`, where
///    `ctx = get_mode_context(block_size, neighbours)`. **Skipped
///    for sub-8 blocks** (each 4×4 sub-block emits its own mode in
///    step 4).
/// 3. Switchable interpolation filter — only emitted when
///    `params.interp_filter == InterpFilter::Switchable`. The
///    [`SWITCHABLE_INTERP_TREE`] is walked against
///    `fc.switchable_interp_prob[ctx]`. Emitted once for the parent
///    block in both the ≥8×8 and sub-8 paths.
/// 4. MV / sub-block payload — for ≥8×8 blocks, the [`InterMode::NewMv`]
///    branch emits the per-ref MV residual via [`write_mv`]
///    (NEAREST / NEAR / ZERO are MV-bit-free — the decoder
///    reconstructs them from the ref-MV list). For sub-8 blocks,
///    each 4×4 sub-block emits its own inter-mode (via
///    [`INTER_MODE_TREE`]) followed by any NEWMV residuals; see
///    [`write_sub8_bmi`].
///
/// ## Sub-8 caller contract
///
/// For `Block4x4` / `Block4x8` / `Block8x4`, the caller must
/// populate `mi.bmi[j].as_mode = sub_mode.idx() as u8` and
/// `mi.bmi[j].as_mv` for each of the 4 sub-blocks. The parent's
/// `mi.mode` must match `mi.bmi[3].as_mode` (libvpx's
/// `mi->mode = b_mode` at end-of-loop convention). The 2×2
/// replication rules are also asserted on the input:
///
/// - `Block4x8` (num_4x4_h = 2): `bmi[2] == bmi[0]` and
///   `bmi[3] == bmi[1]`.
/// - `Block8x4` (num_4x4_w = 2): `bmi[1] == bmi[0]` and
///   `bmi[3] == bmi[2]`.
/// - `Block4x4`: all 4 entries are independent.
///
/// Caller violations of the per-sub-block invariants surface as
/// [`InterModeWriteError::Sub8InvalidSubMode`] /
/// [`InterModeWriteError::Sub8ParentModeMismatch`]; the replication
/// rules and per-mode MV invariants are enforced via `debug_assert!`.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_block_mode_info<N: NeighbourLookup>(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    neighbours: &N,
    block_size: BlockSize,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    mi: &ModeInfo,
) -> Result<(), InterModeWriteError> {
    let (mode, ref_frames, mvs, interp_filter) = match mi.mode {
        BlockMode::Inter {
            mode,
            ref_frames,
            mvs,
            interp_filter,
        } => (mode, ref_frames, mvs, interp_filter),
        BlockMode::Intra { .. } => return Err(InterModeWriteError::NotInter),
    };
    let is_sub8 = (block_size as u8) < (BlockSize::Block8x8 as u8);

    let above = neighbours.get(Position { row: -1, col: 0 });
    let left = neighbours.get(Position { row: 0, col: -1 });

    // 1. Reference frames — inverse of `read_ref_frames`.
    let is_compound = ref_frames[1].is_inter();
    match params.reference_mode {
        ReferenceMode::Select => {
            let ctx = get_reference_mode_context(above, left, params.comp_fixed_ref);
            debug_assert!(ctx < COMP_INTER_CONTEXTS);
            bw.write(u8::from(is_compound), fc.comp_inter_prob[ctx]);
        }
        ReferenceMode::Compound => debug_assert!(is_compound),
        ReferenceMode::Single => debug_assert!(!is_compound),
    }

    if is_compound {
        // Decoder: `ref_frames[idx] = comp_fixed_ref;
        //           ref_frames[1-idx] = comp_var_ref[bit]`.
        // Inverse: bit = (ref_frames[1-idx] == comp_var_ref[1]).
        let idx = params.sign_bias[params.comp_fixed_ref as usize] as usize;
        debug_assert_eq!(ref_frames[idx], params.comp_fixed_ref);
        let var_slot = ref_frames[1 - idx];
        let bit: u8 = if var_slot == params.comp_var_ref[1] {
            1
        } else {
            debug_assert_eq!(var_slot, params.comp_var_ref[0]);
            0
        };
        let ctx = get_pred_context_comp_ref_p(
            above,
            left,
            &params.sign_bias,
            params.comp_fixed_ref,
            params.comp_var_ref,
        );
        debug_assert!(ctx < REF_CONTEXTS);
        bw.write(bit, fc.comp_ref_prob[ctx]);
    } else {
        // Decoder: bit0=0 → LAST; bit0=1, bit1=0 → GOLDEN;
        //          bit0=1, bit1=1 → ALTREF.
        let ctx0 = get_pred_context_single_ref_p1(above, left);
        debug_assert!(ctx0 < REF_CONTEXTS);
        match ref_frames[0] {
            RefFrame::LastFrame => bw.write(0, fc.single_ref_prob[ctx0][0]),
            RefFrame::GoldenFrame => {
                bw.write(1, fc.single_ref_prob[ctx0][0]);
                let ctx1 = get_pred_context_single_ref_p2(above, left);
                debug_assert!(ctx1 < REF_CONTEXTS);
                bw.write(0, fc.single_ref_prob[ctx1][1]);
            }
            RefFrame::AltrefFrame => {
                bw.write(1, fc.single_ref_prob[ctx0][0]);
                let ctx1 = get_pred_context_single_ref_p2(above, left);
                debug_assert!(ctx1 < REF_CONTEXTS);
                bw.write(1, fc.single_ref_prob[ctx1][1]);
            }
            // Single-ref block must reference a real ref frame.
            _ => debug_assert!(
                false,
                "write_inter_block_mode_info: single-ref block has invalid ref_frames[0] = {:?}",
                ref_frames[0]
            ),
        }
    }

    // 2. Parent inter mode — INTER_MODE_TREE inverse. The tree
    //    leaves are `InterMode::idx()` (NEAREST=0, NEAR=1, ZERO=2,
    //    NEW=3). Skipped for sub-8 blocks: the decoder reads one
    //    `b_mode` per sub-block instead (see `write_sub8_bmi`).
    let inter_mode_ctx = get_mode_context(block_size, neighbours);
    debug_assert!(inter_mode_ctx < INTER_MODE_CONTEXTS);
    if !is_sub8 {
        write_tree(
            bw,
            &INTER_MODE_TREE,
            &fc.inter_mode_probs[inter_mode_ctx],
            mode.idx() as i8,
        );
    }

    // 3. Switchable interpolation filter — only when the frame header
    //    says `Switchable`. SWITCHABLE_INTERP_TREE leaves are the raw
    //    `InterpFilter` discriminants (EightTap=0, EightTapSmooth=1,
    //    EightTapSharp=2). Emitted once for the parent in both paths.
    if params.interp_filter == InterpFilter::Switchable {
        let ctx = get_pred_context_switchable_interp(above, left);
        write_tree(
            bw,
            &SWITCHABLE_INTERP_TREE,
            &fc.switchable_interp_prob[ctx],
            interp_filter as i8,
        );
    }

    // 4. MV / sub-block payload.
    if is_sub8 {
        return write_sub8_bmi(
            bw,
            fc,
            params,
            neighbours,
            block_size,
            bounds,
            prev_mv,
            ref_frames,
            is_compound,
            inter_mode_ctx,
            mi,
        );
    }

    // Parent-block (≥8×8) MV residuals — only for NEWMV.
    // NEAREST/NEAR/ZERO emit no MV bits; the decoder reconstructs
    // them from the ref-MV list.
    if matches!(mode, InterMode::NewMv) {
        let num_refs = 1 + usize::from(is_compound);
        for i in 0..num_refs {
            // Replicate the decoder's `best_ref_mv` computation
            // (`resolve_parent_mvs`): find_mv_refs → mvs[count-1] →
            // lower_mv_precision.
            let list = find_mv_refs(
                InterMode::NewMv,
                block_size,
                ref_frames[i],
                neighbours,
                &params.sign_bias,
                bounds,
                prev_mv,
            );
            let idx = list.count.saturating_sub(1).min(list.mvs.len() - 1);
            let best_ref_mv =
                lower_mv_precision(list.mvs[idx], params.allow_high_precision_mv);
            write_mv(
                bw,
                mvs[i],
                best_ref_mv,
                params.allow_high_precision_mv,
                &fc.nmvc,
            );
        }
    }

    Ok(())
}

/// Encode the per-bmi inter mode + MV(s) for sub-8 inter blocks
/// (`Block4x4` / `Block4x8` / `Block8x4`). Mirror of the decoder's
/// `resolve_sub8_bmi` (see this module) — emits the wire-format
/// counterpart of libvpx `vp9_decodemv.c:723-770`.
///
/// Wire order (per active sub-block, in row-major
/// `idy ∈ {0, num_4x4_h}, idx ∈ {0, num_4x4_w}` order):
///
/// 1. The sub-block's inter mode via [`INTER_MODE_TREE`] against
///    `fc.inter_mode_probs[inter_mode_ctx]` (same context as the
///    parent — the decoder uses the parent-block-level context for
///    every sub-block).
/// 2. For NEWMV: per active ref, the MV residual via [`write_mv`].
///    The reference MV is `lower_mv_precision(find_mv_refs.mvs[0])`,
///    computed **once per ref** and cached across sub-blocks
///    (libvpx `got_mv_refs_for_new`).
/// 3. For NEAREST / NEAR: nothing on the wire — the decoder
///    reconstructs the MV via `append_sub8x8_mvs_for_idx`. We
///    `debug_assert!` that the caller-supplied `bmi[j].as_mv`
///    matches that reconstruction.
/// 4. For ZERO: nothing on the wire — we `debug_assert!` that
///    `bmi[j].as_mv == [Mv::ZERO; 2]`.
///
/// The 2×2 replication is enforced as an input invariant (the
/// decoder copies sub-block `j` into `j+num_4x4_w` /
/// `j+num_4x4_h*2`; the writer requires the caller to have
/// pre-replicated identical entries at those slots).
#[allow(clippy::too_many_arguments)]
fn write_sub8_bmi<N: NeighbourLookup>(
    bw: &mut BoolWriter<'_>,
    fc: &FrameContext,
    params: &InterModeReadParams,
    neighbours: &N,
    block_size: BlockSize,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    ref_frames: [RefFrame; 2],
    is_compound: bool,
    inter_mode_ctx: usize,
    mi: &ModeInfo,
) -> Result<(), InterModeWriteError> {
    // libvpx num_4x4_w/h for the sub-8 sizes; see `resolve_sub8_bmi`.
    let (num_4x4_w, num_4x4_h) = match block_size {
        BlockSize::Block4x4 => (1usize, 1usize),
        BlockSize::Block4x8 => (1usize, 2usize),
        BlockSize::Block8x4 => (2usize, 1usize),
        // Non-sub-8 callers are routed past us by the parent path.
        _ => (1, 1),
    };

    // Parent-mode invariant: libvpx assigns `mi->mode = b_mode` on
    // the last loop iteration, so the parent's mode mirrors
    // `bmi[3].as_mode`. Surface a structured error rather than a
    // silent assertion so encoders catch caller bugs early.
    let parent_idx = match mi.mode {
        BlockMode::Inter { mode, .. } => mode.idx() as u8,
        BlockMode::Intra { .. } => return Err(InterModeWriteError::NotInter),
    };
    let last_sub = mi.bmi[3].as_mode;
    if parent_idx != last_sub {
        return Err(InterModeWriteError::Sub8ParentModeMismatch {
            parent: parent_idx,
            last_sub,
        });
    }

    // libvpx `got_mv_refs_for_new` — find_mv_refs result for NEWMV is
    // shared across all sub-blocks.
    let mut best_ref_mvs_for_new = [Mv::ZERO; 2];
    let mut got_mv_refs_for_new = false;
    let num_refs = 1 + usize::from(is_compound);

    let mut idy = 0usize;
    while idy < 2 {
        let mut idx = 0usize;
        while idx < 2 {
            let j = idy * 2 + idx;
            let raw = mi.bmi[j].as_mode;
            let b_mode = InterMode::from_idx(raw)
                .ok_or(InterModeWriteError::Sub8InvalidSubMode { j, raw })?;

            // 1. Emit the sub-block's inter mode.
            write_tree(
                bw,
                &INTER_MODE_TREE,
                &fc.inter_mode_probs[inter_mode_ctx],
                b_mode.idx() as i8,
            );

            match b_mode {
                InterMode::NewMv => {
                    if !got_mv_refs_for_new {
                        for r in 0..num_refs {
                            let list = find_mv_refs(
                                InterMode::NewMv,
                                block_size,
                                ref_frames[r],
                                neighbours,
                                &params.sign_bias,
                                bounds,
                                prev_mv,
                            );
                            best_ref_mvs_for_new[r] = lower_mv_precision(
                                list.mvs[0],
                                params.allow_high_precision_mv,
                            );
                        }
                        got_mv_refs_for_new = true;
                    }
                    // 2. Emit the per-ref MV residual.
                    for (r, &ref_mv) in
                        best_ref_mvs_for_new.iter().enumerate().take(num_refs)
                    {
                        write_mv(
                            bw,
                            mi.bmi[j].as_mv[r],
                            ref_mv,
                            params.allow_high_precision_mv,
                            &fc.nmvc,
                        );
                    }
                }
                InterMode::NearestMv | InterMode::NearMv => {
                    // 3. No MV bits emitted; assert the caller-supplied
                    //    MV matches `append_sub8x8_mvs_for_idx` for each
                    //    active ref. We use a synthetic parent
                    //    `ModeInfo` matching the decoder's call site
                    //    (block_size + b_mode + ref_frames + the
                    //    accumulated bmi so far).
                    for r in 0..num_refs {
                        let synthetic = ModeInfo {
                            block_size,
                            tx_size: crate::transforms::TxSize::Tx4x4,
                            skip: false,
                            segment_id: 0,
                            seg_id_predicted: false,
                            mode: BlockMode::Inter {
                                mode: b_mode,
                                ref_frames,
                                mvs: [Mv::ZERO; 2],
                                interp_filter: InterpFilter::EightTap,
                            },
                            bmi: mi.bmi,
                        };
                        let expected = append_sub8x8_mvs_for_idx(
                            b_mode,
                            &synthetic,
                            block_size,
                            ref_frames[r],
                            r,
                            j,
                            neighbours,
                            &params.sign_bias,
                            bounds,
                            prev_mv,
                        );
                        debug_assert_eq!(
                            mi.bmi[j].as_mv[r], expected,
                            "sub-8 bmi[{j}].as_mv[{r}] must match append_sub8x8_mvs_for_idx",
                        );
                    }
                }
                InterMode::ZeroMv => {
                    // 4. No MV bits; assert the caller supplied
                    //    `[Mv::ZERO; 2]`.
                    debug_assert_eq!(
                        mi.bmi[j].as_mv,
                        [Mv::ZERO; 2],
                        "sub-8 ZEROMV bmi[{j}].as_mv must be [Mv::ZERO; 2]",
                    );
                }
            }

            // Replication invariant — the decoder copies sub-block
            // `j` into the un-iterated 4×4 cells; require the caller
            // to have done the same so the writer's view matches.
            if num_4x4_h == 2 {
                debug_assert_eq!(
                    mi.bmi[j + 2], mi.bmi[j],
                    "sub-8 replication: bmi[{}] must mirror bmi[{}]",
                    j + 2,
                    j,
                );
            }
            if num_4x4_w == 2 {
                debug_assert_eq!(
                    mi.bmi[j + 1], mi.bmi[j],
                    "sub-8 replication: bmi[{}] must mirror bmi[{}]",
                    j + 1,
                    j,
                );
            }

            idx += num_4x4_w;
        }
        idy += num_4x4_h;
    }
    Ok(())
}

/// Resolve `mv[0]` (+ `mv[1]` for compound) for a parent inter block
/// (`block_size >= Block8x8`). Mirrors libvpx
/// `vp9_decodemv.c:771-785`.
///
/// For non-ZEROMV the function calls `find_mv_refs` per active ref
/// and picks `mv_list[refmv_count - 1]` (which is `mvs[0]` for
/// NEAREST/NEW and `mvs[1]` for NEAR). For NEWMV it then calls
/// `read_mv` against that ref-mv. For ZEROMV the result is `(0, 0)`.
#[allow(clippy::too_many_arguments)]
fn resolve_parent_mvs<N: NeighbourLookup>(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    mut counts: Option<&mut NmvCounts>,
    params: &InterModeReadParams,
    neighbours: &N,
    block_size: BlockSize,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    mode: InterMode,
    ref_frames: [RefFrame; 2],
    is_compound: bool,
) -> [Mv; 2] {
    let mut best_ref_mvs = [Mv::ZERO; 2];
    if !matches!(mode, InterMode::ZeroMv) {
        let num_refs = 1 + usize::from(is_compound);
        for i in 0..num_refs {
            let list = find_mv_refs(
                mode,
                block_size,
                ref_frames[i],
                neighbours,
                &params.sign_bias,
                bounds,
                prev_mv,
            );
            // libvpx: `tmp_mvs[refmv_count - 1]`. `count` is
            // pre-clamped to 1 (or 2 for NEARMV) by `finish()` in
            // mv_ref.rs.
            let idx = list.count.saturating_sub(1).min(list.mvs.len() - 1);
            best_ref_mvs[i] = lower_mv_precision(list.mvs[idx], params.allow_high_precision_mv);
        }
    }
    let mut mvs = [Mv::ZERO; 2];
    let num_refs = 1 + usize::from(is_compound);
    match mode {
        InterMode::NewMv => {
            for i in 0..num_refs {
                mvs[i] = read_mv(
                    reader,
                    best_ref_mvs[i],
                    params.allow_high_precision_mv,
                    &fc.nmvc,
                    counts.as_deref_mut(),
                );
            }
        }
        InterMode::NearestMv | InterMode::NearMv => {
            mvs[..num_refs].copy_from_slice(&best_ref_mvs[..num_refs]);
        }
        InterMode::ZeroMv => {
            // mvs already zeroed.
        }
    }
    mvs
}

/// Resolve the per-bmi inter mode + MV(s) for sub-8 inter blocks
/// (`Block4x4` / `Block4x8` / `Block8x4`). Mirrors libvpx
/// `vp9_decodemv.c:723-770` — the sub-8 loop inside
/// `read_inter_block_mode_info`.
///
/// Returns `(parent_mode, parent_mvs, bmi)` where `parent_mode` is
/// the *last* sub-block's mode (libvpx `mi->mode = b_mode`),
/// `parent_mvs` is the last sub-block's MV pair (libvpx
/// `copy_mv_pair(mi->mv, mi->bmi[3].as_mv)`), and `bmi[0..4]` are
/// the per-sub-block MVs.
#[allow(clippy::too_many_arguments)]
fn resolve_sub8_bmi<N: NeighbourLookup>(
    reader: &mut BoolReader<'_>,
    fc: &FrameContext,
    mut counts: Option<&mut NmvCounts>,
    mut mode_counts: Option<&mut crate::entropy::adapt::FrameCounts>,
    params: &InterModeReadParams,
    neighbours: &N,
    block_size: BlockSize,
    bounds: &BlockBounds,
    prev_mv: Option<PrevFrameMvRef>,
    inter_mode_ctx: usize,
    ref_frames: [RefFrame; 2],
    is_compound: bool,
) -> (InterMode, [Mv; 2], [BmiInfo; 4]) {
    // libvpx `num_4x4_w = 1 << xd->bmode_blocks_wl`,
    // `num_4x4_h = 1 << xd->bmode_blocks_hl`. For BLOCK_4X4 both are 1;
    // BLOCK_4X8 is (1, 2); BLOCK_8X4 is (2, 1). The loop iterates
    // `idy ∈ {0, num_4x4_h}` and `idx ∈ {0, num_4x4_w}` over a 2x2
    // grid, replicating each decoded sub into the un-iterated 4×4 cells.
    let (num_4x4_w, num_4x4_h) = match block_size {
        BlockSize::Block4x4 => (1usize, 1usize),
        BlockSize::Block4x8 => (1usize, 2usize),
        BlockSize::Block8x4 => (2usize, 1usize),
        _ => (1, 1), // unreachable
    };

    let mut bmi = [BmiInfo::default(); 4];
    let mut last_mode = InterMode::NearestMv;

    // libvpx: `got_mv_refs_for_new` — for NEWMV the find_mv_refs
    // result is shared across all sub-blocks (since the parent's
    // ref_frames are fixed and `find_mv_refs` is parent-block).
    let mut best_ref_mvs_for_new = [Mv::ZERO; 2];
    let mut got_mv_refs_for_new = false;

    let num_refs = 1 + usize::from(is_compound);

    let mut idy = 0usize;
    while idy < 2 {
        let mut idx = 0usize;
        while idx < 2 {
            let j = idy * 2 + idx;
            let b_mode = read_inter_mode(
                reader,
                &fc.inter_mode_probs[inter_mode_ctx],
                inter_mode_ctx,
                mode_counts.as_deref_mut(),
            );

            let mut best_sub8x8 = [Mv::ZERO; 2];

            match b_mode {
                InterMode::NearestMv | InterMode::NearMv => {
                    for r in 0..num_refs {
                        best_sub8x8[r] = append_sub8x8_mvs_for_idx(
                            b_mode,
                            &ModeInfo {
                                block_size,
                                tx_size: crate::transforms::TxSize::Tx4x4,
                                skip: false,
                                segment_id: 0,
                                seg_id_predicted: false,
                                mode: BlockMode::Inter {
                                    mode: b_mode,
                                    ref_frames,
                                    mvs: [Mv::ZERO; 2],
                                    interp_filter: InterpFilter::EightTap,
                                },
                                bmi,
                            },
                            block_size,
                            ref_frames[r],
                            r,
                            j,
                            neighbours,
                            &params.sign_bias,
                            bounds,
                            prev_mv,
                        );
                    }
                }
                InterMode::NewMv => {
                    if !got_mv_refs_for_new {
                        for r in 0..num_refs {
                            let list = find_mv_refs(
                                InterMode::NewMv,
                                block_size,
                                ref_frames[r],
                                neighbours,
                                &params.sign_bias,
                                bounds,
                                prev_mv,
                            );
                            best_ref_mvs_for_new[r] = lower_mv_precision(
                                list.mvs[0],
                                params.allow_high_precision_mv,
                            );
                        }
                        got_mv_refs_for_new = true;
                    }
                }
                InterMode::ZeroMv => {
                    // best_sub8x8 already zero.
                }
            }

            // Decode the per-bmi MV(s).
            let mut as_mv = [Mv::ZERO; 2];
            match b_mode {
                InterMode::NewMv => {
                    for r in 0..num_refs {
                        as_mv[r] = read_mv(
                            reader,
                            best_ref_mvs_for_new[r],
                            params.allow_high_precision_mv,
                            &fc.nmvc,
                            counts.as_deref_mut(),
                        );
                    }
                }
                InterMode::NearestMv | InterMode::NearMv => {
                    as_mv[..num_refs].copy_from_slice(&best_sub8x8[..num_refs]);
                }
                InterMode::ZeroMv => {
                    // already zero.
                }
            }
            bmi[j].as_mv = as_mv;
            // Replicate into the un-iterated sub-blocks (libvpx
            // `if (num_4x4_h == 2) mi->bmi[j + 2] = mi->bmi[j];` etc.)
            if num_4x4_h == 2 {
                bmi[j + 2] = bmi[j];
            }
            if num_4x4_w == 2 {
                bmi[j + 1] = bmi[j];
            }
            last_mode = b_mode;

            idx += num_4x4_w;
        }
        idy += num_4x4_h;
    }
    // libvpx: `copy_mv_pair(mi->mv, mi->bmi[3].as_mv)` — the
    // parent's MV is the last (bottom-right) sub-block's MV.
    let parent_mvs = bmi[3].as_mv;
    (last_mode, parent_mvs, bmi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockSize;
    use crate::entropy::{BoolWriter, FrameContext};
    use crate::inter::mv_ref::{sign_bias_from_header, NeighbourLookup};
    use crate::intra_pred::IntraMode;
    use crate::mode_info::BlockMode;
    use crate::transforms::TxSize;

    /// A list-backed `NeighbourLookup` (mirror of the helper in
    /// `mv_ref.rs::tests`).
    struct ListNeighbours<'a> {
        entries: Vec<(Position, &'a ModeInfo)>,
    }
    impl<'a> ListNeighbours<'a> {
        fn new() -> Self {
            Self { entries: Vec::new() }
        }
        fn with(mut self, pos: Position, mi: &'a ModeInfo) -> Self {
            self.entries.push((pos, mi));
            self
        }
    }
    impl NeighbourLookup for ListNeighbours<'_> {
        fn get(&self, pos: Position) -> Option<&ModeInfo> {
            self.entries
                .iter()
                .find_map(|(p, mi)| if *p == pos { Some(*mi) } else { None })
        }
    }

    fn loose_bounds() -> BlockBounds {
        BlockBounds {
            left: -10_000,
            right: 10_000,
            top: -10_000,
            bottom: 10_000,
        }
    }

    /// Default inter params: SINGLE ref mode, no sign-bias, no HP,
    /// default 8-tap filter.
    fn default_params() -> InterModeReadParams {
        let sign_bias = sign_bias_from_header(&[false, false, false]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        InterModeReadParams {
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            reference_mode: ReferenceMode::Single,
            comp_fixed_ref: fixed,
            comp_var_ref: var,
            sign_bias,
        }
    }

    /// Skeleton inter [`ModeInfo`] (will be overwritten by the
    /// reader; only `block_size` matters for tests that read it back).
    fn skeleton_mi(block_size: BlockSize) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        }
    }

    /// Inter neighbour for tests (LAST ref, given MV).
    fn inter_neighbour(block_size: BlockSize, mv: Mv) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [mv, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        }
    }

    /// `setup_compound_reference_mode`: LAST and GOLDEN share sign →
    /// fixed = ALTREF.
    #[test]
    fn setup_compound_reference_mode_table() {
        let sb = [false, false, false, true]; // LAST/GOLDEN=0, ALTREF=1.
        let (fixed, var) = setup_compound_reference_mode(&sb);
        assert_eq!(fixed, RefFrame::AltrefFrame);
        assert_eq!(var, [RefFrame::LastFrame, RefFrame::GoldenFrame]);

        // LAST and ALTREF share sign → fixed = GOLDEN.
        let sb = [false, false, true, false];
        let (fixed, var) = setup_compound_reference_mode(&sb);
        assert_eq!(fixed, RefFrame::GoldenFrame);
        assert_eq!(var, [RefFrame::LastFrame, RefFrame::AltrefFrame]);

        // GOLDEN and ALTREF share sign → fixed = LAST.
        let sb = [false, false, true, true];
        let (fixed, var) = setup_compound_reference_mode(&sb);
        assert_eq!(fixed, RefFrame::LastFrame);
        assert_eq!(var, [RefFrame::GoldenFrame, RefFrame::AltrefFrame]);
    }

    /// Encode the bit sequence for the given `InterMode` (parent
    /// 8×8+) against the inter-mode probs at p=128. Tree paths:
    /// - ZEROMV    = `0`
    /// - NEARESTMV = `1 0`
    /// - NEARMV    = `1 1 0`
    /// - NEWMV     = `1 1 1`
    fn write_inter_mode_bits(w: &mut BoolWriter<'_>, m: InterMode, probs: &[u8; 3]) {
        match m {
            InterMode::ZeroMv => w.write(0, probs[0]),
            InterMode::NearestMv => {
                w.write(1, probs[0]);
                w.write(0, probs[1]);
            }
            InterMode::NearMv => {
                w.write(1, probs[0]);
                w.write(1, probs[1]);
                w.write(0, probs[2]);
            }
            InterMode::NewMv => {
                w.write(1, probs[0]);
                w.write(1, probs[1]);
                w.write(1, probs[2]);
            }
        }
    }

    /// Single-ref encoding: `LAST` is bit0=0; `GOLDEN`/`ALTREF` need
    /// bit0=1, bit1=0/1.
    fn write_single_ref_bits(
        w: &mut BoolWriter<'_>,
        target: RefFrame,
        probs0: u8,
        probs1: u8,
    ) {
        match target {
            RefFrame::LastFrame => w.write(0, probs0),
            RefFrame::GoldenFrame => {
                w.write(1, probs0);
                w.write(0, probs1);
            }
            RefFrame::AltrefFrame => {
                w.write(1, probs0);
                w.write(1, probs1);
            }
            _ => panic!("invalid single-ref target {target:?}"),
        }
    }

    /// Pad the writer with zero bits so the reader's lookahead is
    /// always satisfied, then close it. Returns the byte count the
    /// caller should truncate the underlying buffer to.
    fn finalise(w: BoolWriter<'_>) -> usize {
        let mut w = w;
        for _ in 0..16 {
            w.write_bit(0);
        }
        w.stop_encode().expect("stop_encode")
    }

    /// **ZEROMV non-compound**: skips `read_mv`, MV = (0,0).
    ///
    /// Build a stream that decodes to a LAST single ref + ZEROMV at a
    /// frame-edge block (no neighbours → all contexts = libvpx
    /// frame-edge defaults).
    #[test]
    fn zeromv_non_compound_no_mv_read() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref read: ctx0 = single_ref_p1 ctx = 2 (no edges).
        let p_single = fc.single_ref_prob[2][0];
        write_single_ref_bits(&mut w, RefFrame::LastFrame, p_single, 0);
        // Inter mode: ctx = 0 (BOTH_PREDICTED from no neighbours), use ZEROMV.
        let p_inter = &fc.inter_mode_probs[0];
        write_inter_mode_bits(&mut w, InterMode::ZeroMv, p_inter);
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );

        match mi.mode {
            BlockMode::Inter { mode, ref_frames, mvs, .. } => {
                assert_eq!(mode, InterMode::ZeroMv);
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                assert_eq!(mvs, [Mv::ZERO, Mv::ZERO]);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// **NEARESTMV non-compound**: pulls `mv_list[0]`. With a single
    /// above LAST neighbour with MV = (10, -8) (even — so HP
    /// lowering doesn't change anything), the result is exactly that
    /// MV.
    #[test]
    fn nearestmv_non_compound_uses_list0() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nbmv = Mv { row: 10, col: -8 };
        let above = inter_neighbour(bsize, nbmv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref (one neighbour, also LAST) → bit0=0 path.
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), None);
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        // Inter mode ctx: one inter neighbour with mode=NewMv → counter
        // = 1 → ctx = NEW_PLUS_NON_INTRA = 3.
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(
            &mut w,
            InterMode::NearestMv,
            &fc.inter_mode_probs[inter_ctx],
        );
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, mvs, .. } => {
                assert_eq!(mode, InterMode::NearestMv);
                assert_eq!(mvs[0], nbmv, "nearest = neighbour MV");
                assert_eq!(mvs[1], Mv::ZERO);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// **NEARMV**: with two distinct LAST neighbours (above + left)
    /// the find_mv_refs list has count=2; the decoder picks
    /// `mvs[count-1]` = `mvs[1]` = the second (left) MV.
    #[test]
    fn nearmv_non_compound_picks_list_back() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let above_mv = Mv { row: 4, col: -2 };
        let left_mv = Mv { row: -8, col: 12 };
        let above = inter_neighbour(bsize, above_mv);
        let left = inter_neighbour(bsize, left_mv);
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref: two LAST neighbours → both inter, neither intra,
        // ctx0 = 2*1 + 2*1 = 4 (per pred_common p1). Then bit0 = 0
        // (LAST).
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), Some(&left));
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        // Inter mode ctx via get_mode_context.
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(&mut w, InterMode::NearMv, &fc.inter_mode_probs[inter_ctx]);
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, mvs, .. } => {
                assert_eq!(mode, InterMode::NearMv);
                assert_eq!(mvs[0], left_mv, "near = list.mvs[1] = left neighbour");
                assert_eq!(mvs[1], Mv::ZERO);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// **NEWMV non-compound**: calls `read_mv` once with `ref_mv =
    /// mv_pred(list)`. Encode a small MV residual on top of an above
    /// LAST neighbour's MV (= ref_mv).
    #[test]
    fn newmv_non_compound_reads_mv_once() {
        use crate::inter::mv::{get_mv_joint, MvClass, MvJoint};
        use crate::inter::mv_ref::sign_bias_from_header;
        let fc = FrameContext::new_default();
        let mut params = default_params();
        let sign_bias = sign_bias_from_header(&[false, false, false]);
        params.sign_bias = sign_bias;

        let bsize = BlockSize::Block16x16;
        let ref_mv = Mv { row: 8, col: 0 }; // even → HP lowering is no-op.
        let above = inter_neighbour(bsize, ref_mv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();

        // The MV diff we encode (and expect to be added to ref_mv).
        // Use a class-0 small magnitude so the encoded bits are
        // simple. value=4 → z=3 → class 0, d=0, f=1, e=1 → mag=4.
        // Use it as the row component; col=0.
        let diff = Mv { row: 4, col: 0 };

        let mut buf = vec![0u8; 128];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref: LAST.
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), None);
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        // Inter mode = NEWMV.
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(&mut w, InterMode::NewMv, &fc.inter_mode_probs[inter_ctx]);

        // MV joint (HzVnz: row nonzero, col zero).
        let joint = get_mv_joint(diff);
        assert_eq!(joint, MvJoint::HzVnz);
        // Encode joint tree: HzVnz path = `1 1 0`.
        let jp = &fc.nmvc.joints;
        w.write(1, jp[0]);
        w.write(1, jp[1]);
        w.write(0, jp[2]);

        // Encode row component:
        //   sign=0, class=0, d=0, fp=1 (path `1 0`), hp=1 — and
        //   `allow_hp=false` so HP bit is NOT emitted; libvpx
        //   decoder hard-codes hp=1.
        let comp = &fc.nmvc.comps[0];
        // Sign.
        w.write(0, comp.sign);
        // Class tree (class 0 → leaf `0` path): walk MV_CLASS_TREE.
        // MV_CLASS_TREE[0] = -CLASS_0 = 0, so path is just bit=0.
        use crate::inter::mv::MV_CLASS_TREE;
        // Find path to class 0 — leaf `0` is at the very first
        // node. write 0 to enter that leaf.
        w.write(0, comp.classes[0]);
        // d bit (class0 → one bit against class0[0]).
        let _ = MvClass::Class0;
        w.write(0, comp.class0[0]); // d=0
                                    // fp tree at class0_fp[0] (`d=0`), target f=1 (`1 0`).
        w.write(1, comp.class0_fp[0][0]);
        w.write(0, comp.class0_fp[0][1]);
        // No HP bit because allow_hp = false.

        // Use `MV_CLASS_TREE` so the import isn't dead.
        let _ = MV_CLASS_TREE.len();

        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, mvs, .. } => {
                assert_eq!(mode, InterMode::NewMv);
                // result = ref_mv (above's MV) + diff.
                assert_eq!(mvs[0], Mv { row: ref_mv.row + diff.row, col: 0 });
                assert_eq!(mvs[1], Mv::ZERO);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// `read_switchable_interp_filter`: at zero neighbours the
    /// context is `SWITCHABLE_FILTERS = 3` and the read picks one of
    /// the three filters from the tree.
    #[test]
    fn switchable_interp_filter_read() {
        let fc = FrameContext::new_default();
        let nb_above: Option<&ModeInfo> = None;
        let nb_left: Option<&ModeInfo> = None;
        let ctx = get_pred_context_switchable_interp(nb_above, nb_left);
        assert_eq!(ctx, SWITCHABLE_FILTERS); // sentinel collapse → 3.

        // Encode the path to EIGHTTAP_SMOOTH (`1 0` in
        // SWITCHABLE_INTERP_TREE).
        let probs = &fc.switchable_interp_prob[ctx];
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, probs[0]);
        w.write(0, probs[1]);
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let got = read_switchable_interp_filter(&mut reader, &fc, nb_above, nb_left, None);
        assert_eq!(got, InterpFilter::EightTapSmooth);
    }

    /// `get_mode_context`: zero neighbours → counter = 0 →
    /// BOTH_PREDICTED = ctx 2.
    #[test]
    fn mode_context_no_neighbours() {
        let nb = ListNeighbours::new();
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 2);
    }

    /// `get_mode_context`: one inter neighbour with mode=NewMv →
    /// counter += 1 → NEW_PLUS_NON_INTRA = ctx 3.
    #[test]
    fn mode_context_one_inter_new_neighbour() {
        let above = inter_neighbour(BlockSize::Block16x16, Mv { row: 0, col: 0 });
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 3);
    }

    /// `get_mode_context`: two intra neighbours → counter = 18 →
    /// BOTH_INTRA = ctx 6.
    #[test]
    fn mode_context_two_intra_neighbours() {
        let intra = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &intra)
            .with(Position { row: 0, col: -1 }, &intra);
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 6);
    }

    /// **Compound NEWMV**: calls `read_mv` TWICE, once per ref. We
    /// verify by constructing a stream where the per-ref reads result
    /// in distinct MVs.
    ///
    /// Frame is in compound mode (reference_mode = Compound — bypasses
    /// the comp_inter_p read entirely). Both refs need to be in the
    /// "comp_var_ref" path; we set sign_bias such that the fixed ref
    /// slot lands in `[0]` and var bit selects var_ref[1] = GOLDEN.
    ///
    /// For simplicity use ref_mv = (0,0) for both refs and decode a
    /// non-zero diff on each.
    #[test]
    fn compound_newmv_reads_mv_twice() {
        use crate::inter::mv::{get_mv_joint, MvJoint};

        let fc = FrameContext::new_default();
        // sign_bias: LAST=0, GOLDEN=0, ALTREF=1 → comp_fixed = ALT,
        // comp_var = [LAST, GOLDEN].
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;

        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();

        // comp_fixed_ref = ALTREF; sign_bias[ALTREF] = 1 → idx = 1.
        // So ref_frames[1] = ALTREF, ref_frames[0] = var[bit].
        // Pick bit = 0 → var_ref[0] = LAST. Result: [LAST, ALTREF].

        let mut buf = vec![0u8; 256];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Compound ref: ctx from get_pred_context_comp_ref_p (no
        // neighbours → 2). Write bit=0.
        let comp_ctx = get_pred_context_comp_ref_p(
            None,
            None,
            &params.sign_bias,
            params.comp_fixed_ref,
            params.comp_var_ref,
        );
        w.write(0, fc.comp_ref_prob[comp_ctx]);
        // Inter mode = NEWMV.
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(&mut w, InterMode::NewMv, &fc.inter_mode_probs[inter_ctx]);

        // Two MV reads. Use the same simple `(row=4, col=0)` diff.
        // For NEWMV, the find_mv_refs result for each ref is empty
        // (no neighbours), so best_ref_mvs = (0, 0) for both.
        // Joint = HzVnz = `1 1 0`.
        let jp = &fc.nmvc.joints;
        let comp = &fc.nmvc.comps[0];

        for _ in 0..2 {
            // Joint HzVnz.
            assert_eq!(get_mv_joint(Mv { row: 4, col: 0 }), MvJoint::HzVnz);
            w.write(1, jp[0]);
            w.write(1, jp[1]);
            w.write(0, jp[2]);
            // Row component: sign=0, class=0, d=0, f=1 (`1 0`).
            w.write(0, comp.sign);
            w.write(0, comp.classes[0]);
            w.write(0, comp.class0[0]);
            w.write(1, comp.class0_fp[0][0]);
            w.write(0, comp.class0_fp[0][1]);
        }
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, ref_frames, mvs, .. } => {
                assert_eq!(mode, InterMode::NewMv);
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::AltrefFrame]);
                // Each MV is the encoded diff (0+4, 0+0).
                assert_eq!(mvs[0], Mv { row: 4, col: 0 });
                assert_eq!(mvs[1], Mv { row: 4, col: 0 });
            }
            _ => panic!("expected Inter"),
        }
    }

    /// **Sub-8 (`Block4x4`) NEARESTMV**: 4 sub-blocks each get
    /// resolved per-bmi. The reader emits 4 inter-mode reads (one
    /// per sub-block) and NO MV bits (NEAREST never reads MV). Each
    /// sub-block's MV is the parent's resolved nearest MV (or the
    /// previously-decoded peer per `append_sub8x8_mvs_for_idx`).
    ///
    /// Setup: a single above LAST neighbour with bmi[2].as_mv[0] = a
    /// known MV (the libvpx `get_sub_block_mv` pick for the j=0 sub).
    /// All sub-blocks inherit MVs through the sub-8 dispatch.
    #[test]
    fn sub8_block4x4_nearestmv_each_sub_decoded() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block4x4;
        let mut above = inter_neighbour(bsize, Mv::ZERO);
        above.bmi[2].as_mv[0] = Mv { row: 6, col: -6 }; // even mv
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();

        let mut buf = vec![0u8; 128];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref: LAST → bit0=0 path.
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), None);
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        // For Block4x4, the parent inter-mode read is SKIPPED in the
        // libvpx implementation (the `if (bsize >= BLOCK_8X8)` gate).
        // Then the sub-8 path reads inter_mode 4 times (j=0..3).
        let inter_ctx = get_mode_context(bsize, &nb);
        let p = &fc.inter_mode_probs[inter_ctx];
        for _ in 0..4 {
            write_inter_mode_bits(&mut w, InterMode::NearestMv, p);
        }
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, mvs, .. } => {
                assert_eq!(mode, InterMode::NearestMv);
                // libvpx: parent mv = bmi[3].as_mv. For NEARESTMV
                // j=3, `append_sub8x8_mvs_for_idx` returns bmi[2].as_mv
                // (the j=2 peer). j=2's MV came from j=1's peer
                // (bmi[0]) which itself derived from
                // `dec_find_mv_refs(block=0)` → above.bmi[2].as_mv[0].
                // So bmi[3] == bmi[0] == { 6, -6 }.
                assert_eq!(mvs[0], Mv { row: 6, col: -6 });
                assert_eq!(mvs[1], Mv::ZERO);
                // bmi[0..4] all populated with the propagated MV.
                assert_eq!(mi.bmi[0].as_mv[0], Mv { row: 6, col: -6 });
                assert_eq!(mi.bmi[3].as_mv[0], Mv { row: 6, col: -6 });
            }
            _ => panic!("expected Inter"),
        }
    }

    /// **Sign-bias propagation**: with sign_bias[GOLDEN]=true vs
    /// block's LAST (sign_bias[LAST]=false), the GOLDEN neighbour's
    /// MV is negated by `find_mv_refs`'s second-walk. We verify by
    /// reading a NEARESTMV against a GOLDEN neighbour and checking
    /// the MV is negated.
    #[test]
    fn sign_bias_propagation_negates_mv() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, true, false]); // GOLDEN flipped.

        let bsize = BlockSize::Block16x16;
        // Above neighbour: GOLDEN ref, MV = (10, -8).
        let above = ModeInfo {
            block_size: bsize,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
                mvs: [Mv { row: 10, col: -8 }, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref: target LAST. Above is GOLDEN, so ctx0 path: above
        // is inter, !has_second_ref, ref_frame[0] != LAST → ctx = 0
        // (per pred_common_p1 single-edge path: `!is_inter ? 2 :
        // !has_second_ref(edge) ? 4 * (ref_frame[0] == LAST) :
        // ... ` → 0).
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), None);
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        // Inter mode = NEARESTMV.
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(
            &mut w,
            InterMode::NearestMv,
            &fc.inter_mode_probs[inter_ctx],
        );
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            None,
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        match mi.mode {
            BlockMode::Inter { mode, mvs, .. } => {
                assert_eq!(mode, InterMode::NearestMv);
                // The GOLDEN neighbour's MV is negated (sign-bias
                // mismatch with this block's LAST ref).
                assert_eq!(mvs[0], Mv { row: -10, col: 8 });
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Stage 3 Task E.1.c.2/.3: `read_inter_block_mode_info` populates
    /// `mode_counts.single_ref[ctx0][0][bit0]` /
    /// `mode_counts.inter_mode[inter_ctx][mode_idx]` /
    /// `mode_counts.switchable_interp[interp_ctx][interp]` when fed a
    /// non-`None` sink. Verifies the wireup at the higher-level
    /// function entry point, mirroring the leaf-level test in
    /// `crate::mode_info::tests::leaf_decoders_populate_frame_counts`.
    #[test]
    fn read_inter_block_mode_info_populates_mode_counts() {
        use crate::entropy::adapt::FrameCounts;

        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nbmv = Mv { row: 10, col: -8 };
        let above = inter_neighbour(bsize, nbmv);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &above);
        let bounds = loose_bounds();

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        // Single-ref → bit0=0 path (LAST).
        let ctx0 = get_pred_context_single_ref_p1(Some(&above), None);
        write_single_ref_bits(&mut w, RefFrame::LastFrame, fc.single_ref_prob[ctx0][0], 0);
        let inter_ctx = get_mode_context(bsize, &nb);
        write_inter_mode_bits(
            &mut w,
            InterMode::NearestMv,
            &fc.inter_mode_probs[inter_ctx],
        );
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut mi = skeleton_mi(bsize);
        let mut mode_counts = FrameCounts::default();
        read_inter_block_mode_info(
            &mut reader,
            &fc,
            None,
            Some(&mut mode_counts),
            &params,
            &nb,
            bsize,
            &bounds,
            None,
            &mut mi,
        );
        // Single-ref path: ctx0 sees bit0=0 (LastFrame).
        assert_eq!(mode_counts.single_ref[ctx0][0][0], 1);
        // Inter mode: ctx=`inter_ctx`, mode=NearestMv (idx 0).
        assert_eq!(mode_counts.inter_mode[inter_ctx][InterMode::NearestMv.idx()], 1);
        // No comp_inter / comp_ref / switchable_interp reads happen
        // here (reference_mode=Single, interp_filter≠Switchable).
        assert_eq!(mode_counts.comp_inter, FrameCounts::default().comp_inter);
        assert_eq!(mode_counts.comp_ref, FrameCounts::default().comp_ref);
        assert_eq!(
            mode_counts.switchable_interp,
            FrameCounts::default().switchable_interp
        );
    }

    // ----- S4.16b: write_inter_block_mode_info round-trip tests ----

    /// Make a built inter [`ModeInfo`] for round-trip tests. The
    /// `block_size` is mirrored into the skeleton fields; only
    /// `mode` matters for the writer's wire output.
    fn build_inter_mi(
        block_size: BlockSize,
        mode: InterMode,
        ref_frames: [RefFrame; 2],
        mvs: [Mv; 2],
        interp_filter: InterpFilter,
    ) -> ModeInfo {
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode,
                ref_frames,
                mvs,
                interp_filter,
            },
            bmi: Default::default(),
        }
    }

    /// Encode `mi` via [`write_inter_block_mode_info`], then decode it
    /// back via [`read_inter_block_mode_info`], asserting the recovered
    /// `BlockMode::Inter` payload matches the input. Returns the
    /// recovered `ModeInfo` for any extra assertions.
    fn round_trip(
        fc: &FrameContext,
        params: &InterModeReadParams,
        neighbours: &impl NeighbourLookup,
        block_size: BlockSize,
        bounds: &BlockBounds,
        prev_mv: Option<PrevFrameMvRef>,
        mi: &ModeInfo,
    ) -> ModeInfo {
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w, fc, params, neighbours, block_size, bounds, prev_mv, mi,
            )
            .expect("writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init reader");
        let mut got = skeleton_mi(block_size);
        read_inter_block_mode_info(
            &mut reader,
            fc,
            None,
            None,
            params,
            neighbours,
            block_size,
            bounds,
            prev_mv,
            &mut got,
        );
        match (&mi.mode, &got.mode) {
            (
                BlockMode::Inter {
                    mode: m_in,
                    ref_frames: rf_in,
                    mvs: mvs_in,
                    interp_filter: if_in,
                },
                BlockMode::Inter {
                    mode: m_out,
                    ref_frames: rf_out,
                    mvs: mvs_out,
                    interp_filter: if_out,
                },
            ) => {
                assert_eq!(m_in, m_out, "mode round-trip");
                assert_eq!(rf_in, rf_out, "ref_frames round-trip");
                assert_eq!(mvs_in, mvs_out, "mvs round-trip");
                assert_eq!(if_in, if_out, "interp_filter round-trip");
            }
            _ => panic!("expected Inter after round trip"),
        }
        got
    }

    /// Single-ref LASTFRAME, NEARESTMV, EightTap, no neighbours.
    #[test]
    fn s4_16b_round_trip_single_ref_nearest_mv() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block8x8;
        // NEAREST with no neighbours: find_mv_refs returns count=0 →
        // best_ref_mv = (0,0); decoder copies that into mvs[0].
        let mi = build_inter_mi(
            bsize,
            InterMode::NearestMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Single-ref GOLDENFRAME, ZEROMV.
    #[test]
    fn s4_16b_round_trip_single_ref_zero_mv() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Single-ref LASTFRAME, NEWMV with mvs[0] = (12, -4), allow_hp=true.
    /// With no neighbours, find_mv_refs returns count=0 →
    /// best_ref_mv = (0,0); the encoded diff is the MV itself.
    #[test]
    fn s4_16b_round_trip_single_ref_new_mv() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;
        let mi = build_inter_mi(
            bsize,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv { row: 12, col: -4 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Compound mode, ZEROMV, ref_frames=[LASTFRAME, ALTREFFRAME].
    /// Set sign-bias so comp_fixed_ref = ALTREF, comp_var_ref =
    /// [LAST, GOLDEN]; idx = sign_bias[ALTREF] = 1, so the bit must
    /// be 0 (selecting var_ref[0] = LAST in slot 0).
    #[test]
    fn s4_16b_round_trip_compound_zero_mv() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Switchable interp filter, single-ref ZEROMV, filter =
    /// EightTapSmooth. Verifies the per-block switchable filter is
    /// round-tripped via SWITCHABLE_INTERP_TREE.
    #[test]
    fn s4_16b_round_trip_switchable_interp() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block8x8;
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO, Mv::ZERO],
            InterpFilter::EightTapSmooth,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Intra mode info → writer rejects with `NotInter`.
    #[test]
    fn s4_16b_rejects_intra() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;
        // skeleton_mi yields BlockMode::Intra.
        let mi = skeleton_mi(bsize);
        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        let err = write_inter_block_mode_info(
            &mut w, &fc, &params, &nb, bsize, &bounds, None, &mi,
        )
        .expect_err("expected NotInter");
        assert_eq!(err, InterModeWriteError::NotInter);
    }

    // ----- S4.16e: sub-8 inter block round-trip tests ----

    /// Build a sub-8 inter [`ModeInfo`] from a per-sub-block
    /// `(mode, [mv0, mv1])` array. Caller provides 4 entries in
    /// row-major order; this helper applies the libvpx 2×2
    /// replication rules (so a `Block4x8` caller passes 2 distinct
    /// entries at `[0]` / `[1]`, etc., and gets `[2] = [0]`,
    /// `[3] = [1]`). The parent's `mi.mode` is set to `entries[3].0`
    /// (matching `mi->mode = b_mode` at end-of-loop in libvpx).
    fn build_sub8_inter_mi(
        block_size: BlockSize,
        ref_frames: [RefFrame; 2],
        interp_filter: InterpFilter,
        entries: [(InterMode, [Mv; 2]); 4],
    ) -> ModeInfo {
        let mut bmi = [BmiInfo::default(); 4];
        for (j, (mode, mvs)) in entries.iter().enumerate() {
            bmi[j].as_mode = mode.idx() as u8;
            bmi[j].as_mv = *mvs;
        }
        // Apply the replication contract — the writer asserts on it.
        match block_size {
            BlockSize::Block4x8 => {
                bmi[2] = bmi[0];
                bmi[3] = bmi[1];
            }
            BlockSize::Block8x4 => {
                bmi[1] = bmi[0];
                bmi[3] = bmi[2];
            }
            _ => {}
        }
        let parent_mode = InterMode::from_idx(bmi[3].as_mode).expect("valid parent mode");
        let parent_mvs = bmi[3].as_mv;
        ModeInfo {
            block_size,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: parent_mode,
                ref_frames,
                mvs: parent_mvs,
                interp_filter,
            },
            bmi,
        }
    }

    /// Sub-8 round-trip: encode then decode, asserting parent
    /// `(mode, ref_frames, mvs, interp_filter)` and `bmi[j].as_mv`
    /// for `j ∈ 0..4` match. The decoder doesn't populate
    /// `bmi[j].as_mode` on the inter path, so we don't compare it.
    fn round_trip_sub8(
        fc: &FrameContext,
        params: &InterModeReadParams,
        neighbours: &impl NeighbourLookup,
        block_size: BlockSize,
        bounds: &BlockBounds,
        prev_mv: Option<PrevFrameMvRef>,
        mi: &ModeInfo,
    ) -> ModeInfo {
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w, fc, params, neighbours, block_size, bounds, prev_mv, mi,
            )
            .expect("sub-8 writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init reader");
        let mut got = skeleton_mi(block_size);
        read_inter_block_mode_info(
            &mut reader,
            fc,
            None,
            None,
            params,
            neighbours,
            block_size,
            bounds,
            prev_mv,
            &mut got,
        );
        match (&mi.mode, &got.mode) {
            (
                BlockMode::Inter {
                    mode: m_in,
                    ref_frames: rf_in,
                    mvs: mvs_in,
                    interp_filter: if_in,
                },
                BlockMode::Inter {
                    mode: m_out,
                    ref_frames: rf_out,
                    mvs: mvs_out,
                    interp_filter: if_out,
                },
            ) => {
                assert_eq!(m_in, m_out, "parent mode round-trip");
                assert_eq!(rf_in, rf_out, "ref_frames round-trip");
                assert_eq!(mvs_in, mvs_out, "parent mvs round-trip");
                assert_eq!(if_in, if_out, "interp_filter round-trip");
            }
            _ => panic!("expected Inter after sub-8 round trip"),
        }
        for j in 0..4 {
            assert_eq!(
                mi.bmi[j].as_mv, got.bmi[j].as_mv,
                "bmi[{j}].as_mv round-trip",
            );
        }
        got
    }

    /// Block4x4, all 4 sub-blocks ZEROMV. No MV bits on the wire;
    /// only the inter-mode tree leaves.
    #[test]
    fn s4_16e_round_trip_sub8_block4x4_all_zero() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block4x4, all 4 sub-blocks NEARESTMV with no neighbours
    /// (`append_sub8x8_mvs_for_idx` → `(0, 0)`).
    #[test]
    fn s4_16e_round_trip_sub8_block4x4_all_nearest() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        let entries = [
            (InterMode::NearestMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NearestMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NearestMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NearestMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block4x4, 4 distinct NEWMV MVs per sub-block. No neighbours
    /// so the cached `best_ref_mv_for_new` is `(0, 0)` for the
    /// single LAST ref; each sub-block's MV is encoded as the diff
    /// against that.
    #[test]
    fn s4_16e_round_trip_sub8_block4x4_new_mv() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        let entries = [
            (InterMode::NewMv, [Mv { row: 4, col: 0 }, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: -2, col: 6 }, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: 8, col: -4 }, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: -6, col: 10 }, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block4x8 (num_4x4_h = 2): only `bmi[0]` and `bmi[1]` are
    /// active; `bmi[2]` mirrors `bmi[0]` and `bmi[3]` mirrors
    /// `bmi[1]` (per libvpx's `num_4x4_h == 2` replication).
    #[test]
    fn s4_16e_round_trip_sub8_block4x8_zero() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x8;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block8x4 (num_4x4_w = 2): active sub-blocks are `bmi[0]` and
    /// `bmi[2]`; `bmi[1]` mirrors `bmi[0]` and `bmi[3]` mirrors
    /// `bmi[2]`.
    #[test]
    fn s4_16e_round_trip_sub8_block8x4_zero() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block8x4;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block4x4 with mixed modes: ZERO / NEAREST / NEW / NEW. The
    /// last sub-block's mode (NEW) becomes the parent's `mode`. The
    /// NEWMV cache is populated once on the first NEWMV (sub-block
    /// `j=2`) and reused for `j=3`.
    #[test]
    fn s4_16e_round_trip_sub8_block4x4_mixed_modes() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NearestMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: 4, col: -2 }, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: -8, col: 6 }, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Compound sub-8: Block4x4, all 4 sub-blocks ZEROMV with two
    /// active refs (LAST + ALTREF). Confirms both ref slots survive
    /// round-trip through the sub-8 path.
    #[test]
    fn s4_16e_round_trip_sub8_compound_zero() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// `mi.mode != bmi[3].as_mode` → writer rejects with
    /// `Sub8ParentModeMismatch`.
    #[test]
    fn s4_16e_rejects_parent_mode_mismatch() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        // bmi[3] is ZEROMV (idx=2) but parent says NEWMV (idx=3).
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
        ];
        let mut mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        // Force the mismatch.
        if let BlockMode::Inter { ref mut mode, .. } = mi.mode {
            *mode = InterMode::NewMv;
        }
        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        let err = write_inter_block_mode_info(
            &mut w, &fc, &params, &nb, bsize, &bounds, None, &mi,
        )
        .expect_err("expected Sub8ParentModeMismatch");
        assert_eq!(
            err,
            InterModeWriteError::Sub8ParentModeMismatch {
                parent: InterMode::NewMv.idx() as u8,
                last_sub: InterMode::ZeroMv.idx() as u8,
            },
        );
    }
}
