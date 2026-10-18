//! Per-inter-block mode-info reader (`read_inter_block_mode_info`).
//!
//! Mirrors libvpx `read_inter_block_mode_info`
//! (`vp9/decoder/vp9_decodemv.c:691-786`) — the function called by
//! `read_inter_frame_mode_info` after `is_inter_block` returns `true`.
//! It populates a [`ModeInfo`]'s
//! [`BlockMode::Inter`] variant
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
//! The per-segment `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` features are
//! consumed via the [`InterModeReadParams::seg_ref_frame`] /
//! [`InterModeReadParams::seg_skip`] hooks: when active they force the
//! reference frame / ZEROMV and suppress the corresponding bit reads
//! (mirroring libvpx `read_ref_frames` / `read_inter_block_mode_info`).

use crate::bitstream::{InterpFilter, ReferenceMode};
use crate::block::BlockSize;
use crate::entropy::bitacct::{BitCategory, set_category};
use crate::entropy::{
    BoolReader, BoolWriter, COMP_INTER_CONTEXTS, FrameContext, FrameCounts, INTER_MODE_CONTEXTS,
    INTRA_INTER_CONTEXTS, NmvCounts, REF_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
use crate::inter::mv::{count_mv, read_mv, write_mv, write_tree};
use crate::inter::mv_ref::{
    BlockBounds, NeighbourLookup, Position, PrevFrameMvRef, append_sub8x8_mvs_for_idx,
    find_mv_refs, lower_mv_precision,
};
use crate::mode_info::{
    BlockMode, BmiInfo, INTER_MODE_TREE, InterMode, ModeInfo, Mv, RefFrame, read_inter_mode,
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
    /* NEARESTMV */ 0, /* NEARMV    */ 0, /* ZEROMV    */ 3, /* NEWMV     */ 1,
];
const MODE_2_COUNTER_INTRA: i32 = 9;

#[inline]
pub(crate) fn mode_context_counter(mode_info: &ModeInfo) -> i32 {
    match mode_info.mode {
        BlockMode::Intra { .. } => MODE_2_COUNTER_INTRA,
        BlockMode::Inter { mode, .. } => MODE_2_COUNTER_INTER[mode.idx()],
    }
}

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

#[inline]
pub(crate) fn mode_context_from_counter(counter: i32) -> usize {
    COUNTER_TO_CONTEXT[counter.clamp(0, 18) as usize] as usize
}

/// Compute the inter-mode context for a block at the given
/// `(mi_row, mi_col)` (tile-relative, mi-units). Mirrors libvpx
/// `get_mode_context` (`vp9_decodemv.c:670-689`).
///
/// Scans the first 2 entries of the block-size-specific neighbour
/// table ([`crate::inter::mv_ref::MV_REF_BLOCKS`]) and sums each
/// candidate's `mode_2_counter` contribution. The summed counter is
/// then mapped through `COUNTER_TO_CONTEXT`.
#[must_use]
pub fn get_mode_context<N: NeighbourLookup>(block_size: BlockSize, neighbours: &N) -> usize {
    let mv_ref_search = &crate::inter::mv_ref::MV_REF_BLOCKS[block_size.idx()];
    let mut counter: i32 = 0;
    for &mv_ref in mv_ref_search.iter().take(2) {
        if let Some(candidate) = neighbours.get(mv_ref) {
            counter += mode_context_counter(candidate);
        }
    }
    // Clamp to the valid range — `counter` is always 0..=18 by
    // construction, but defensively floor at 0 / cap at 18 to match
    // libvpx's `counter_to_context` array bounds.
    mode_context_from_counter(counter)
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

/// Choose the realtime picker's neighbour interpolation-filter reference.
///
/// Mirrors pinned libvpx `vp9_pickmode.c:1908-1917`: prefer the above
/// neighbour when it is inter, otherwise use an inter left neighbour, and
/// fall back to [`InterpFilter::Switchable`] when neither neighbour is inter.
/// The caller uses this reference to decide whether it can reuse a concrete
/// neighbour filter or needs to search the realtime filter set.
#[must_use]
pub fn get_pred_filter_ref(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> InterpFilter {
    let inter_filter = |mi: Option<&ModeInfo>| match mi.map(|m| m.mode) {
        Some(BlockMode::Inter { interp_filter, .. }) => Some(interp_filter),
        _ => None,
    };
    inter_filter(above)
        .or_else(|| inter_filter(left))
        .unwrap_or(InterpFilter::Switchable)
}

/// libvpx `vp9_get_skip_context` (`vp9_pred_common.h:55-61`).
///
/// ```c
/// static INLINE int vp9_get_skip_context(const MACROBLOCKD *xd) {
///   const MODE_INFO *const above_mi = xd->above_mi;
///   const MODE_INFO *const left_mi  = xd->left_mi;
///   const int above_skip = (above_mi != NULL) ? above_mi->skip : 0;
///   const int left_skip  = (left_mi  != NULL) ? left_mi->skip  : 0;
///   return above_skip + left_skip;
/// }
/// ```
///
/// Returns `above.skip + left.skip` as an index into the 3-entry
/// `skip_probs[SKIP_CONTEXTS]` table. Missing neighbours contribute 0.
///
/// Used by the picker's skip-flag tie-break
/// (`vp9_pickmode.c:2355-2362`) to pick `cm->fc->skip_probs[ctx]` —
/// see [`crate::mode_decision::inter::pick_inter_mode`].
#[must_use]
pub fn get_pred_context_skip(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
    let above_skip = above.is_some_and(|mi| mi.skip) as usize;
    let left_skip = left.is_some_and(|mi| mi.skip) as usize;
    above_skip + left_skip
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
            2 + usize::from(ref0(above) == comp_fixed_ref || !is_inter_block(above))
        } else if !l_has_second {
            2 + usize::from(ref0(left) == comp_fixed_ref || !is_inter_block(left))
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
pub fn get_pred_context_single_ref_p1(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
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
                    ref0(edge) == RefFrame::LastFrame || ref1(edge) == RefFrame::LastFrame,
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
                    3 + usize::from(crf1 == RefFrame::LastFrame || crf2 == RefFrame::LastFrame)
                } else {
                    usize::from(crf1 == RefFrame::LastFrame || crf2 == RefFrame::LastFrame)
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
            1 + usize::from(ref0(edge) == RefFrame::LastFrame || ref1(edge) == RefFrame::LastFrame)
        }
    } else {
        2
    }
}

/// libvpx `vp9_get_pred_context_single_ref_p2`
/// (`vp9_pred_common.c:233-316`). 0..=4 context for the single-ref
/// second selection bit (`bit1` — GOLDEN vs ALTREF).
#[must_use]
pub fn get_pred_context_single_ref_p2(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
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
                    ref0(edge) == RefFrame::GoldenFrame || ref1(edge) == RefFrame::GoldenFrame,
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
                    3 + usize::from(crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame)
                } else if rfs == RefFrame::AltrefFrame {
                    usize::from(crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame)
                } else {
                    1 + 2 * usize::from(
                        crf1 == RefFrame::GoldenFrame || crf2 == RefFrame::GoldenFrame,
                    )
                }
            } else if above0 == RefFrame::LastFrame && left0 == RefFrame::LastFrame {
                3
            } else if above0 == RefFrame::LastFrame || left0 == RefFrame::LastFrame {
                let edge0 = if above0 == RefFrame::LastFrame {
                    left0
                } else {
                    above0
                };
                4 * usize::from(edge0 == RefFrame::GoldenFrame)
            } else {
                2 * usize::from(above0 == RefFrame::GoldenFrame)
                    + 2 * usize::from(left0 == RefFrame::GoldenFrame)
            }
        }
    } else if has_above || has_left {
        let edge = if has_above { above } else { left };
        if !is_inter_block(edge) || (ref0(edge) == RefFrame::LastFrame && !has_second_ref(edge)) {
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
/// `read_ref_frames` (`vp9_decodemv.c:303-341`), including the
/// `SEG_LVL_REF_FRAME` segmentation override: when
/// `params.seg_ref_frame` is `Some`, `ref_frame[0]` is forced to that
/// segdata value, `ref_frame[1] = NoRefFrame` (single-ref), and NO
/// comp/single ref bits are read.
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
    // SEG_LVL_REF_FRAME force (libvpx `vp9_decodemv.c:303-309`):
    // ref_frame[0] = segdata, ref_frame[1] = NO_REF_FRAME (single
    // ref), and the comp/single ref bits are NOT read.
    if let Some(forced) = params.seg_ref_frame {
        return (forced, RefFrame::NoRefFrame, false);
    }
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
pub fn get_intra_inter_context(above: Option<&ModeInfo>, left: Option<&ModeInfo>) -> usize {
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
    /// `SEG_LVL_REF_FRAME` segdata for this block's segment (`None`
    /// when inactive). When `Some`, `ref_frame[0]` is forced and the
    /// comp/single ref bits are NOT read (libvpx `read_ref_frames`,
    /// `vp9_decodemv.c:303-309`).
    pub seg_ref_frame: Option<RefFrame>,
    /// `SEG_LVL_SKIP` active for this block's segment. Forces ZEROMV
    /// and suppresses the inter-mode read (libvpx
    /// `read_inter_block_mode_info`, `vp9_decodemv.c:703-717`).
    pub seg_skip: bool,
}

/// libvpx `vp9_setup_compound_reference_mode`
/// (`vp9_pred_common.c:24-40`). Derive `(comp_fixed_ref, comp_var_ref[0],
/// comp_var_ref[1])` from the per-ref sign-bias table.
///
/// Returns `(fixed, var)`. Callers typically wrap this once per frame
/// (when `reference_mode != Single`) to populate the
/// [`InterModeReadParams`] fixed/var slots.
#[must_use]
pub fn setup_compound_reference_mode(sign_bias: &[bool; 4]) -> (RefFrame, [RefFrame; 2]) {
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
    //
    //    SEG_LVL_SKIP force (libvpx `vp9_decodemv.c:703-717`): when
    //    active, `mi->mode = ZEROMV` and the inter-mode bits are NOT
    //    read. The sub-8 case is a stream error in libvpx
    //    (`vp9_decodemv.c:709-712`) and is rejected by the decoder
    //    before reaching this reader.
    let inter_mode_ctx = get_mode_context(block_size, neighbours);
    debug_assert!(inter_mode_ctx < INTER_MODE_CONTEXTS);
    let parent_mode = if params.seg_skip {
        InterMode::ZeroMv // forced, no inter-mode bits read
    } else if (block_size as u8) >= (BlockSize::Block8x8 as u8) {
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

/// Accumulate the non-coefficient counts for one inter mode-info payload.
///
/// `inter_mode_ctx` and `best_ref_mvs` are explicit because realtime mode
/// decision already derives them before committing a leaf. Passing those
/// values lets it retain exact header counts without repeating the spatial
/// MV-reference walk. Sub-8 mode and MV counts intentionally remain omitted,
/// matching the existing writer-side count behavior.
#[allow(clippy::too_many_arguments)]
pub fn count_inter_block_mode_info(
    params: &InterModeReadParams,
    block_size: BlockSize,
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
    inter_mode_ctx: usize,
    best_ref_mvs: [Mv; 2],
    mi: &ModeInfo,
    counts: &mut FrameCounts,
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
    let is_compound = ref_frames[1].is_inter();

    match params.reference_mode {
        ReferenceMode::Select => {
            let ctx = get_reference_mode_context(above, left, params.comp_fixed_ref);
            debug_assert!(ctx < COMP_INTER_CONTEXTS);
            counts.comp_inter[ctx][usize::from(is_compound)] += 1;
        }
        ReferenceMode::Compound => debug_assert!(is_compound),
        ReferenceMode::Single => debug_assert!(!is_compound),
    }

    if is_compound {
        let idx = params.sign_bias[params.comp_fixed_ref as usize] as usize;
        debug_assert_eq!(ref_frames[idx], params.comp_fixed_ref);
        let var_slot = ref_frames[1 - idx];
        let bit = u8::from(var_slot == params.comp_var_ref[1]);
        debug_assert!(bit == 1 || var_slot == params.comp_var_ref[0]);
        let ctx = get_pred_context_comp_ref_p(
            above,
            left,
            &params.sign_bias,
            params.comp_fixed_ref,
            params.comp_var_ref,
        );
        debug_assert!(ctx < REF_CONTEXTS);
        counts.comp_ref[ctx][bit as usize] += 1;
    } else {
        let ctx0 = get_pred_context_single_ref_p1(above, left);
        debug_assert!(ctx0 < REF_CONTEXTS);
        match ref_frames[0] {
            RefFrame::LastFrame => counts.single_ref[ctx0][0][0] += 1,
            RefFrame::GoldenFrame | RefFrame::AltrefFrame => {
                counts.single_ref[ctx0][0][1] += 1;
                let ctx1 = get_pred_context_single_ref_p2(above, left);
                debug_assert!(ctx1 < REF_CONTEXTS);
                let bit = usize::from(ref_frames[0] == RefFrame::AltrefFrame);
                counts.single_ref[ctx1][1][bit] += 1;
            }
            _ => debug_assert!(
                false,
                "count_inter_block_mode_info: invalid single reference {:?}",
                ref_frames[0]
            ),
        }
    }

    debug_assert!(inter_mode_ctx < INTER_MODE_CONTEXTS);
    if !is_sub8 {
        counts.inter_mode[inter_mode_ctx][mode.idx()] += 1;
    }

    if params.interp_filter == InterpFilter::Switchable {
        let ctx = get_pred_context_switchable_interp(above, left);
        counts.switchable_interp[ctx][interp_filter as usize] += 1;
    }

    if !is_sub8 && matches!(mode, InterMode::NewMv) {
        let num_refs = 1 + usize::from(is_compound);
        for i in 0..num_refs {
            count_mv(
                mvs[i],
                best_ref_mvs[i],
                params.allow_high_precision_mv,
                &mut counts.mv,
            );
        }
    }

    Ok(())
}

/// Encode one inter block's reference, mode, filter, and motion-vector syntax.
///
/// This is the inverse of [`read_inter_block_mode_info`]. Parent blocks emit
/// one inter mode and any `NEWMV` residuals; sub-8 blocks delegate their
/// per-4x4 payload to `write_sub8_bmi`. When `counts` is present, the shared
/// [`count_inter_block_mode_info`] helper retains the exact adaptation counts.
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
    mut counts: Option<&mut FrameCounts>,
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
            RefFrame::LastFrame => {
                bw.write(0, fc.single_ref_prob[ctx0][0]);
            }
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
        if let Some(c) = counts.as_deref_mut() {
            count_inter_block_mode_info(
                params,
                block_size,
                above,
                left,
                inter_mode_ctx,
                [Mv::ZERO; 2],
                mi,
                c,
            )?;
        }
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
    let mut best_ref_mvs = [Mv::ZERO; 2];
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
            let best_ref_mv = lower_mv_precision(list.mvs[idx], params.allow_high_precision_mv);
            best_ref_mvs[i] = best_ref_mv;
            set_category(BitCategory::MvDelta);
            write_mv(
                bw,
                mvs[i],
                best_ref_mv,
                params.allow_high_precision_mv,
                &fc.nmvc,
                None,
            );
            set_category(BitCategory::ModeRef);
        }
    }

    if let Some(c) = counts {
        count_inter_block_mode_info(
            params,
            block_size,
            above,
            left,
            inter_mode_ctx,
            best_ref_mvs,
            mi,
            c,
        )?;
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
                            best_ref_mvs_for_new[r] =
                                lower_mv_precision(list.mvs[0], params.allow_high_precision_mv);
                        }
                        got_mv_refs_for_new = true;
                    }
                    // 2. Emit the per-ref MV residual.
                    set_category(BitCategory::MvDelta);
                    for (r, &ref_mv) in best_ref_mvs_for_new.iter().enumerate().take(num_refs) {
                        write_mv(
                            bw,
                            mi.bmi[j].as_mv[r],
                            ref_mv,
                            params.allow_high_precision_mv,
                            &fc.nmvc,
                            None, // P0b: sub-8 MV counts omitted (rare path)
                        );
                    }
                    set_category(BitCategory::ModeRef);
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
                    mi.bmi[j + 2],
                    mi.bmi[j],
                    "sub-8 replication: bmi[{}] must mirror bmi[{}]",
                    j + 2,
                    j,
                );
            }
            if num_4x4_w == 2 {
                debug_assert_eq!(
                    mi.bmi[j + 1],
                    mi.bmi[j],
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
                            best_ref_mvs_for_new[r] =
                                lower_mv_precision(list.mvs[0], params.allow_high_precision_mv);
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
    use crate::inter::mv_ref::{NeighbourLookup, sign_bias_from_header};
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
            Self {
                entries: Vec::new(),
            }
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
            seg_ref_frame: None,
            seg_skip: false,
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
    fn write_single_ref_bits(w: &mut BoolWriter<'_>, target: RefFrame, probs0: u8, probs1: u8) {
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
            BlockMode::Inter {
                mode,
                ref_frames,
                mvs,
                ..
            } => {
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
        use crate::inter::mv::{MvClass, MvJoint, get_mv_joint};
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
                assert_eq!(
                    mvs[0],
                    Mv {
                        row: ref_mv.row + diff.row,
                        col: 0
                    }
                );
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
        use crate::inter::mv::{MvJoint, get_mv_joint};

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
            BlockMode::Inter {
                mode,
                ref_frames,
                mvs,
                ..
            } => {
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
        assert_eq!(
            mode_counts.inter_mode[inter_ctx][InterMode::NearestMv.idx()],
            1
        );
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
                &mut w, fc, params, neighbours, block_size, bounds, prev_mv, mi, None,
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
        let err =
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
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
                &mut w, fc, params, neighbours, block_size, bounds, prev_mv, mi, None,
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
        let err =
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect_err("expected Sub8ParentModeMismatch");
        assert_eq!(
            err,
            InterModeWriteError::Sub8ParentModeMismatch {
                parent: InterMode::NewMv.idx() as u8,
                last_sub: InterMode::ZeroMv.idx() as u8,
            },
        );
    }

    // ---------- Stage 5 — branch-coverage tests -------------------------
    //
    // The tests below target the missed-branch dimensions in
    // `read_inter_block_mode_info` / `write_inter_block_mode_info` and the
    // surrounding `get_pred_context_*` helpers. Each test exercises a
    // specific dimension (compound reference, reference-mode Select,
    // switchable interp filter context, `is_inter` context, sub-8 mixed
    // modes, etc.) via the `round_trip` / `round_trip_sub8` oracle plus
    // a dimension-specific assertion that proves the path was actually
    // hit (not just any inter block).

    /// Build an inter `ModeInfo` for tests that need a fully-specified
    /// neighbour with `interp_filter`. Matches `inter_neighbour` but
    /// lets the caller pick the mode / filter / ref-frame combo.
    fn inter_neighbour_full(
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

    fn intra_neighbour(block_size: BlockSize) -> ModeInfo {
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

    // ----- Compound reference: all 3 fixed pairs (LAST+GOLDEN,
    //       LAST+ALTREF, GOLDEN+ALTREF) round-trip via `Compound` mode

    /// Compound `[LAST, GOLDEN]` with sign_bias all-zero → fixed =
    /// ALTREF, var = [LAST, GOLDEN]. idx = sign_bias[ALTREF] = 0, so
    /// `ref_frames[0] = ALTREF` ... but we want [LAST, GOLDEN].
    /// Choose sign-bias such that the fixed slot lands in `[1]` and
    /// the var bit picks `var_ref[1] = GOLDEN` in slot 0.
    ///
    /// Easiest: with `sign_bias[ALTREF] = false` (fixed = ALTREF, idx
    /// = 0), `ref_frames[0] = ALTREF`. Not what we want.
    ///
    /// Use sign_bias = `[LAST=false, GOLDEN=false, ALTREF=true]` →
    /// fixed=ALTREF, var=[LAST, GOLDEN]; idx=1 so `ref_frames[1] =
    /// ALTREF`, `ref_frames[0] = var[bit]`. Picking `bit=1` →
    /// `var_ref[1] = GOLDEN`, so `ref_frames = [GOLDEN, ALTREF]`.
    /// That's not [LAST, GOLDEN] either. To actually get
    /// [LAST, GOLDEN] we need fixed = neither LAST nor GOLDEN →
    /// fixed = ALTREF AND ref_frames slot for it should be position 0
    /// → idx = 0 → sign_bias[ALTREF] = 0. Then ref_frames[1] is var
    /// (LAST or GOLDEN). Setting bit = 1 picks GOLDEN: ref_frames =
    /// [ALTREF, GOLDEN] — order ALTREF/GOLDEN. Choosing bit=0 →
    /// [ALTREF, LAST].
    ///
    /// So the natural compound pair is whichever the libvpx-derived
    /// layout produces; just round-trip a known compound pair and
    /// confirm both refs survive.
    #[test]
    fn compound_lastframe_altrefframe_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        // sign_bias = [false, false, true] (header LAST/GOLDEN/ALTREF).
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        // fixed = ALTREF, var = [LAST, GOLDEN].
        assert_eq!(fixed, RefFrame::AltrefFrame);
        assert_eq!(var, [RefFrame::LastFrame, RefFrame::GoldenFrame]);
        // sign_bias[ALTREF] = true → idx = 1 → ref_frames[1] = ALTREF.
        // Pick the variable ref so it lands in slot 0.
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::AltrefFrame]);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Compound `[GOLDEN, ALTREF]` via sign_bias = `[false, true,
    /// false]` (LAST/ALTREF share sign) → fixed = GOLDEN.
    #[test]
    fn compound_goldenframe_altrefframe_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, true, false]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        // LAST=0, GOLDEN=1, ALTREF=0 → LAST and ALTREF share → fixed=GOLDEN.
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        assert_eq!(fixed, RefFrame::GoldenFrame);
        assert_eq!(var, [RefFrame::LastFrame, RefFrame::AltrefFrame]);
        // sign_bias[GOLDEN] = true → idx = 1 → ref_frames[1] = GOLDEN.
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames, [RefFrame::AltrefFrame, RefFrame::GoldenFrame]);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Compound `[LAST, GOLDEN]` via sign_bias = `[true, false, false]`
    /// (GOLDEN/ALTREF share sign) → fixed = LAST.
    #[test]
    fn compound_lastframe_goldenframe_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[true, false, false]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        assert_eq!(fixed, RefFrame::LastFrame);
        assert_eq!(var, [RefFrame::GoldenFrame, RefFrame::AltrefFrame]);
        // sign_bias[LAST] = true → idx = 1 → ref_frames[1] = LAST.
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::LastFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames, [RefFrame::GoldenFrame, RefFrame::LastFrame]);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Compound NEWMV with `[LAST, ALTREF]` and per-ref MVs. Verifies
    /// `read_mv` runs for each compound slot.
    #[test]
    fn compound_newmv_per_ref_mvs_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mvs = [Mv { row: 8, col: -4 }, Mv { row: -12, col: 6 }];
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            mvs,
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { mvs: got_mvs, .. } => {
                assert_eq!(got_mvs, mvs, "both compound MVs survive");
            }
            _ => panic!("expected Inter"),
        }
    }

    // ----- Reference-mode SELECT: compound bit + each context arm

    /// Reference-mode `Select` with a single-ref output. The encoder
    /// emits the `comp_inter` bit = 0, then the single-ref bits.
    /// `comp_inter_ctx = 1` (no neighbours → libvpx `else` arm at
    /// `vp9_pred_common.c:80`).
    #[test]
    fn reference_mode_select_single_no_neighbours_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.reference_mode = ReferenceMode::Select;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                // Single-ref: slot 1 = NoRefFrame.
                assert_eq!(ref_frames[1], RefFrame::NoRefFrame);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Reference-mode `Select` with a compound output. The encoder
    /// emits `comp_inter` bit = 1 then the compound bit.
    #[test]
    fn reference_mode_select_compound_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Select;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert!(ref_frames[1].is_inter(), "compound: slot 1 is a real ref");
            }
            _ => panic!("expected Inter"),
        }
    }

    /// `get_reference_mode_context` boundary: above + left both
    /// single-ref, both match `comp_fixed_ref` → ctx = 0 (XOR = 0).
    /// Verify via a Select-mode round-trip with the matching
    /// neighbours.
    #[test]
    fn reference_mode_select_ctx_both_match_fixed() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Select;
        // fixed = ALTREF; build two single-ref ALTREF neighbours.
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = above; // same shape
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let ctx = get_reference_mode_context(Some(&above), Some(&left), fixed);
        assert_eq!(ctx, 0, "both neighbours single, both match fixed → XOR=0");
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
    }

    /// `get_reference_mode_context`: above single-ref matches fixed,
    /// left does NOT → ctx = 1 (XOR = 1).
    #[test]
    fn reference_mode_select_ctx_xor_mismatch() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Select;
        // above = ALTREF (= fixed), left = LAST (≠ fixed).
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let ctx = get_reference_mode_context(Some(&above), Some(&left), fixed);
        assert_eq!(ctx, 1);
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
    }

    /// `get_reference_mode_context` ctx = 4 path: both neighbours are
    /// compound. Verify the context number AND a round-trip succeeds.
    #[test]
    fn reference_mode_select_ctx_both_compound() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Select;
        // Both neighbours compound [LAST, ALTREF].
        let nb_mi = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &nb_mi)
            .with(Position { row: 0, col: -1 }, &nb_mi);
        let ctx = get_reference_mode_context(Some(&nb_mi), Some(&nb_mi), fixed);
        assert_eq!(ctx, 4);
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
    }

    /// `get_reference_mode_context` ctx = 3 path: a single edge with
    /// a compound neighbour.
    #[test]
    fn reference_mode_select_ctx_single_edge_compound() {
        let nb_mi = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx_above = get_reference_mode_context(Some(&nb_mi), None, RefFrame::AltrefFrame);
        assert_eq!(ctx_above, 3, "single edge compound → ctx 3");
        let ctx_left = get_reference_mode_context(None, Some(&nb_mi), RefFrame::AltrefFrame);
        assert_eq!(ctx_left, 3);
    }

    /// `get_reference_mode_context` boundary: one neighbour single +
    /// other compound → ctx 2 + (ref0 == comp_fixed_ref || intra)
    /// path.
    #[test]
    fn reference_mode_select_ctx_single_plus_compound() {
        // above = single LAST, left = compound [LAST, ALTREF]; fixed
        // = ALTREF. above is single, !a_has_second; left has second
        // → match `else if !a_has_second` branch → 2 + (ref0(above)
        // == ALTREF || !is_inter(above)) → above is inter LAST → 2 + 0
        // = 2.
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_reference_mode_context(Some(&above), Some(&left), RefFrame::AltrefFrame);
        assert_eq!(ctx, 2, "single+compound, above_single ref0≠fixed → ctx 2");
    }

    // ----- Switchable interpolation filter: 3 contexts × 3 filters

    /// Switchable interp with both neighbours present using EightTap
    /// filter → ctx = 0 (libvpx: left==above==0).
    #[test]
    fn switchable_interp_ctx0_round_trip_eighttap() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let nb_mi = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &nb_mi)
            .with(Position { row: 0, col: -1 }, &nb_mi);
        let ctx = get_pred_context_switchable_interp(Some(&nb_mi), Some(&nb_mi));
        assert_eq!(ctx, 0, "both EightTap → ctx 0");
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
    }

    /// Switchable interp: both neighbours EightTapSmooth → ctx = 1.
    #[test]
    fn switchable_interp_ctx1_smooth_neighbours() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let nb_mi = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &nb_mi)
            .with(Position { row: 0, col: -1 }, &nb_mi);
        let ctx = get_pred_context_switchable_interp(Some(&nb_mi), Some(&nb_mi));
        assert_eq!(ctx, InterpFilter::EightTapSmooth as usize);
        let bounds = loose_bounds();
        // Round-trip with the same filter as neighbours to keep the
        // round-trip simple.
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { interp_filter, .. } => {
                assert_eq!(interp_filter, InterpFilter::EightTapSmooth);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Switchable interp: both neighbours EightTapSharp → ctx = 2.
    #[test]
    fn switchable_interp_ctx2_sharp_neighbours() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let nb_mi = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &nb_mi)
            .with(Position { row: 0, col: -1 }, &nb_mi);
        let ctx = get_pred_context_switchable_interp(Some(&nb_mi), Some(&nb_mi));
        assert_eq!(ctx, InterpFilter::EightTapSharp as usize);
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { interp_filter, .. } => {
                assert_eq!(interp_filter, InterpFilter::EightTapSharp);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Switchable interp: neighbours have mismatched filters → ctx =
    /// SWITCHABLE_FILTERS = 3 (sentinel).
    #[test]
    fn switchable_interp_ctx3_mismatched_neighbours() {
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        let ctx = get_pred_context_switchable_interp(Some(&above), Some(&left));
        assert_eq!(ctx, SWITCHABLE_FILTERS);
    }

    /// Switchable interp: above is intra → uses left's filter (intra
    /// neighbour collapses to sentinel; the other axis wins).
    #[test]
    fn switchable_interp_intra_above_uses_left() {
        let above = intra_neighbour(BlockSize::Block16x16);
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
        );
        let ctx = get_pred_context_switchable_interp(Some(&above), Some(&left));
        // above_type = SWITCHABLE_FILTERS (sentinel; intra). left_type
        // = EightTapSmooth = 1. left != above → above is sentinel →
        // ctx = left = 1.
        assert_eq!(ctx, InterpFilter::EightTapSmooth as usize);
    }

    /// All-3 switchable filters round-trip via Switchable mode + no
    /// neighbours (ctx = 3). Exercises each `SWITCHABLE_INTERP_TREE`
    /// arm at the same context.
    #[test]
    fn switchable_interp_all_three_filters_no_neighbours_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        for filter in [
            InterpFilter::EightTap,
            InterpFilter::EightTapSmooth,
            InterpFilter::EightTapSharp,
        ] {
            let mi = build_inter_mi(
                BlockSize::Block16x16,
                InterMode::ZeroMv,
                [RefFrame::LastFrame, RefFrame::NoRefFrame],
                [Mv::ZERO; 2],
                filter,
            );
            let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
            match got.mode {
                BlockMode::Inter { interp_filter, .. } => assert_eq!(interp_filter, filter),
                _ => panic!("expected Inter for filter {filter:?}"),
            }
        }
    }

    // ----- get_intra_inter_context: each of 4 contexts

    #[test]
    fn intra_inter_context_no_neighbours() {
        assert_eq!(get_intra_inter_context(None, None), 0);
    }

    #[test]
    fn intra_inter_context_single_inter_edge() {
        let above = inter_neighbour(BlockSize::Block16x16, Mv::ZERO);
        // single edge (above), not intra → ctx = 2 * 0 = 0.
        assert_eq!(get_intra_inter_context(Some(&above), None), 0);
    }

    #[test]
    fn intra_inter_context_single_intra_edge() {
        let above = intra_neighbour(BlockSize::Block16x16);
        // single edge, intra → ctx = 2 * 1 = 2.
        assert_eq!(get_intra_inter_context(Some(&above), None), 2);
    }

    #[test]
    fn intra_inter_context_both_inter() {
        let n = inter_neighbour(BlockSize::Block16x16, Mv::ZERO);
        // both present and inter → ctx = 0 (left_intra || above_intra = false).
        assert_eq!(get_intra_inter_context(Some(&n), Some(&n)), 0);
    }

    #[test]
    fn intra_inter_context_one_intra_one_inter() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let inter = inter_neighbour(BlockSize::Block16x16, Mv::ZERO);
        // one intra → ctx = 1.
        assert_eq!(get_intra_inter_context(Some(&intra), Some(&inter)), 1);
        assert_eq!(get_intra_inter_context(Some(&inter), Some(&intra)), 1);
    }

    #[test]
    fn intra_inter_context_both_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        assert_eq!(get_intra_inter_context(Some(&intra), Some(&intra)), 3);
    }

    // ----- get_pred_context_single_ref_p1: contexts 0..=4

    /// Both intra neighbours → ctx = 2.
    #[test]
    fn single_ref_p1_both_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx = get_pred_context_single_ref_p1(Some(&intra), Some(&intra));
        assert_eq!(ctx, 2);
    }

    /// One intra + one inter LAST (single-ref) edge → ctx = 4 *
    /// (ref0(edge) == LAST) = 4.
    #[test]
    fn single_ref_p1_intra_plus_inter_last() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&intra), Some(&inter));
        assert_eq!(ctx, 4);
    }

    /// One intra + one inter GOLDEN (single-ref) edge → ctx = 0.
    #[test]
    fn single_ref_p1_intra_plus_inter_golden() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&intra), Some(&inter));
        assert_eq!(ctx, 0);
    }

    /// One intra + one inter compound w/ LAST in either slot → ctx =
    /// 1 + (has LAST) = 2.
    #[test]
    fn single_ref_p1_intra_plus_inter_compound_with_last() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&intra), Some(&inter));
        assert_eq!(ctx, 2);
    }

    /// Both inter single-ref, both LAST → ctx = 2*1 + 2*1 = 4.
    #[test]
    fn single_ref_p1_both_inter_single_both_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&inter), Some(&inter));
        assert_eq!(ctx, 4);
    }

    /// Both inter, one LAST, one GOLDEN → ctx = 2*1 + 2*0 = 2.
    #[test]
    fn single_ref_p1_both_inter_one_last_one_golden() {
        let last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&last), Some(&golden));
        assert_eq!(ctx, 2);
    }

    /// Both inter compound, has LAST somewhere → ctx = 1 + 1 = 2.
    #[test]
    fn single_ref_p1_both_compound_with_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&inter), Some(&inter));
        assert_eq!(ctx, 2);
    }

    /// Both inter compound, no LAST → ctx = 1 + 0 = 1.
    #[test]
    fn single_ref_p1_both_compound_no_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&inter), Some(&inter));
        assert_eq!(ctx, 1);
    }

    /// One compound + one single: rfs (single) == LAST → ctx = 3 + 1
    /// (crf1/crf2 has LAST).
    #[test]
    fn single_ref_p1_compound_plus_single_both_last_paths() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single_last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // above = single LAST → !a_has_second; left = compound. rfs =
        // above0 = LAST; crf1=left0=LAST, crf2=left1=ALTREF. rfs == LAST
        // → 3 + (crf1==LAST || crf2==LAST) = 3 + 1 = 4.
        let ctx = get_pred_context_single_ref_p1(Some(&single_last), Some(&compound));
        assert_eq!(ctx, 4);
    }

    /// One compound + one single: rfs (single) ≠ LAST → ctx =
    /// (crf1/crf2 has LAST).
    #[test]
    fn single_ref_p1_compound_plus_single_not_last() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single_golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // above = single GOLDEN → rfs = GOLDEN (≠ LAST). crf1/2 =
        // left0/1 = GOLDEN/ALTREF → neither is LAST → ctx = 0.
        let ctx = get_pred_context_single_ref_p1(Some(&single_golden), Some(&compound));
        assert_eq!(ctx, 0);
    }

    /// Single edge: intra → ctx = 2.
    #[test]
    fn single_ref_p1_single_edge_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx = get_pred_context_single_ref_p1(Some(&intra), None);
        assert_eq!(ctx, 2);
    }

    /// Single edge: inter single LAST → ctx = 4.
    #[test]
    fn single_ref_p1_single_edge_inter_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(None, Some(&inter));
        assert_eq!(ctx, 4);
    }

    /// Single edge: inter compound w/ LAST → ctx = 1 + 1 = 2.
    #[test]
    fn single_ref_p1_single_edge_compound_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&inter), None);
        assert_eq!(ctx, 2);
    }

    /// Single edge: inter compound, no LAST → ctx = 1.
    #[test]
    fn single_ref_p1_single_edge_compound_no_last() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&inter), None);
        assert_eq!(ctx, 1);
    }

    // ----- get_pred_context_single_ref_p2: various arms

    /// Both intra → ctx = 2.
    #[test]
    fn single_ref_p2_both_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx = get_pred_context_single_ref_p2(Some(&intra), Some(&intra));
        assert_eq!(ctx, 2);
    }

    /// Intra + inter single LAST → ctx = 3.
    #[test]
    fn single_ref_p2_intra_plus_single_last() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&intra), Some(&last));
        assert_eq!(ctx, 3);
    }

    /// Intra + inter single GOLDEN → ctx = 4.
    #[test]
    fn single_ref_p2_intra_plus_single_golden() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&intra), Some(&golden));
        assert_eq!(ctx, 4);
    }

    /// Intra + inter single ALTREF → ctx = 0.
    #[test]
    fn single_ref_p2_intra_plus_single_altref() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let altref = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&intra), Some(&altref));
        assert_eq!(ctx, 0);
    }

    /// Intra + compound w/ GOLDEN → ctx = 1 + 2*1 = 3.
    #[test]
    fn single_ref_p2_intra_plus_compound_with_golden() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&intra), Some(&compound));
        assert_eq!(ctx, 3);
    }

    /// Both compound w/ matching ref pairs and a GOLDEN → ctx = 3.
    #[test]
    fn single_ref_p2_both_compound_matching_golden() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&inter), Some(&inter));
        assert_eq!(ctx, 3);
    }

    /// Both compound with mismatched ref pairs → ctx = 2.
    #[test]
    fn single_ref_p2_both_compound_mismatched() {
        let a = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let b = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&a), Some(&b));
        assert_eq!(ctx, 2);
    }

    /// Compound + single GOLDEN: rfs (single) = GOLDEN → ctx = 3 + 0
    /// (compound has neither GOLDEN/ALTREF → check). compound = [LAST,
    /// ALTREF] → crf1=LAST, crf2=ALTREF → neither GOLDEN → ctx = 3 + 0
    /// = 3.
    #[test]
    fn single_ref_p2_compound_plus_single_golden() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&single), Some(&compound));
        assert_eq!(ctx, 3);
    }

    /// Single edge: inter single LAST → ctx = 2.
    #[test]
    fn single_ref_p2_single_edge_inter_last() {
        let last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&last), None);
        assert_eq!(ctx, 2);
    }

    /// Single edge: inter single GOLDEN → ctx = 4.
    #[test]
    fn single_ref_p2_single_edge_inter_golden() {
        let golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&golden), None);
        assert_eq!(ctx, 4);
    }

    /// Single edge: inter compound with GOLDEN → ctx = 3.
    #[test]
    fn single_ref_p2_single_edge_compound_golden() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&inter), None);
        assert_eq!(ctx, 3);
    }

    /// Single edge: inter compound, no GOLDEN → ctx = 0.
    #[test]
    fn single_ref_p2_single_edge_compound_no_golden() {
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&inter), None);
        assert_eq!(ctx, 0);
    }

    // ----- get_pred_context_comp_ref_p: various arms

    /// Both intra → ctx = 2.
    #[test]
    fn comp_ref_p_both_intra() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx = get_pred_context_comp_ref_p(Some(&intra), Some(&intra), &sign_bias, fixed, var);
        assert_eq!(ctx, 2);
    }

    /// One intra + one compound: single-ref branch.
    #[test]
    fn comp_ref_p_intra_plus_compound() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let intra = intra_neighbour(BlockSize::Block16x16);
        // compound [LAST, ALTREF]; var = [LAST, GOLDEN]; var[1] =
        // GOLDEN. For edge=compound: var_ref_of(edge) takes index 1
        // (since sign_bias[ALTREF]=true → fix_ref_idx=1, var_ref_idx=0).
        // var_ref_of(edge) = ref_frames[0] = LAST. var1 = GOLDEN.
        // 1 + 2*(LAST != GOLDEN) = 1 + 2 = 3.
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx =
            get_pred_context_comp_ref_p(Some(&intra), Some(&compound), &sign_bias, fixed, var);
        assert_eq!(ctx, 3);
    }

    /// Single edge with intra → ctx = 2 (intra branch).
    #[test]
    fn comp_ref_p_single_intra_edge() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx_above = get_pred_context_comp_ref_p(Some(&intra), None, &sign_bias, fixed, var);
        assert_eq!(ctx_above, 2);
    }

    /// Single edge compound (has_second_ref true) → ctx = 4 * (var_ref
    /// != var1).
    #[test]
    fn comp_ref_p_single_compound_edge_mismatch() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var = [LAST, GOLDEN]; var1 = GOLDEN.
        // edge compound [LAST, ALTREF]; var_ref_of = ref_frames[0] =
        // LAST ≠ GOLDEN → ctx = 4 * 1 = 4.
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&inter), None, &sign_bias, fixed, var);
        assert_eq!(ctx, 4);
    }

    /// Single edge compound match → ctx = 0.
    #[test]
    fn comp_ref_p_single_compound_edge_match() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var = [LAST, GOLDEN]; var1 = GOLDEN. Edge compound: pick
        // ref slot 0 = GOLDEN → var_ref_of = GOLDEN == var1 → ctx = 0.
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&inter), None, &sign_bias, fixed, var);
        assert_eq!(ctx, 0);
    }

    /// Single edge single-ref: 3 * (ref0 != var1).
    #[test]
    fn comp_ref_p_single_singleref_edge_no_match() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var1 = GOLDEN. single ref = LAST → 3 * 1 = 3.
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&inter), None, &sign_bias, fixed, var);
        assert_eq!(ctx, 3);
    }

    /// Single edge single-ref: 3 * 0 = 0 when ref == var1.
    #[test]
    fn comp_ref_p_single_singleref_edge_match() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var1 = GOLDEN; single ref = GOLDEN → 0.
        let inter = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&inter), None, &sign_bias, fixed, var);
        assert_eq!(ctx, 0);
    }

    /// `get_reference_mode_context`: above compound + left single
    /// → hits the `!l_has_second` arm (lines 282-283).
    #[test]
    fn reference_mode_ctx_above_compound_left_single() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, _var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // a_has_second=true, l_has_second=false → `!l_has_second`
        // branch: 2 + (ref0(left) == fixed || !is_inter(left)).
        // left is inter LAST; fixed = ALTREF; LAST != ALTREF →
        // !is_inter false → 2 + 0 = 2.
        let ctx = get_reference_mode_context(Some(&above), Some(&left), fixed);
        assert_eq!(ctx, 2);
    }

    /// `get_reference_mode_context`: above compound + left single
    /// matching fixed → ctx = 3 (2 + 1).
    #[test]
    fn reference_mode_ctx_above_compound_left_single_match() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, _var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // ref0(left) = ALTREF == fixed → 2 + 1 = 3.
        let ctx = get_reference_mode_context(Some(&above), Some(&left), fixed);
        assert_eq!(ctx, 3);
    }

    /// `comp_ref_p`: one intra + one single neighbour → exercises
    /// the `1 + 2*(ref0(edge) != var1)` branch (line 332).
    #[test]
    fn comp_ref_p_intra_plus_single_inter() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var1 = GOLDEN. single neighbour ref0 = LAST → not GOLDEN
        // → 1 + 2 = 3.
        let intra = intra_neighbour(BlockSize::Block16x16);
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&intra), Some(&single), &sign_bias, fixed, var);
        assert_eq!(ctx, 3);
    }

    /// `comp_ref_p`: l_sg || a_sg (one single one compound), neither
    /// vrfc nor rfs equals var1 → ctx = 4 (line 362).
    /// Setup: above = single ALTREF (vrfa=ALTREF), left = compound
    /// [LAST, ALTREF] → var_ref_of(left) = LAST. var1=GOLDEN.
    /// l_sg = false (compound), a_sg = true. vrfc = vrfl (since
    /// !l_sg) = LAST. rfs = vrfa (since a_sg) = ALTREF.
    /// vrfc==var1? LAST==GOLDEN: no. rfs==var1? ALTREF==GOLDEN: no.
    /// → ctx = 4.
    #[test]
    fn comp_ref_p_compound_plus_single_neither_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        assert_eq!(ctx, 4);
    }

    /// `single_ref_p1`: intra + compound w/ LAST in slot 1 (only) →
    /// 1 + (ref0==LAST || ref1==LAST) where ref0 is not LAST and
    /// ref1 is LAST → 1 + 1 = 2 (exercises ref1 check at line 406).
    #[test]
    fn single_ref_p1_intra_plus_compound_last_only_in_slot1() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::LastFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p1(Some(&intra), Some(&compound));
        assert_eq!(ctx, 2);
    }

    /// `comp_ref_p` no neighbours → ctx = 2.
    #[test]
    fn comp_ref_p_no_neighbours() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let ctx = get_pred_context_comp_ref_p(None, None, &sign_bias, fixed, var);
        assert_eq!(ctx, 2);
    }

    // ----- get_mode_context: each value 0..=6 via neighbour shapes

    /// Counter = 0 → ctx 2.
    #[test]
    fn mode_context_no_neighbours_both_predicted() {
        let nb = ListNeighbours::new();
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 2);
    }

    /// Counter = 3 (one ZERO) → ctx 1.
    #[test]
    fn mode_context_one_zero_neighbour_ctx1() {
        let zero_n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &zero_n);
        // counter = 0 (single neighbour, ZERO contributes 3) → wait,
        // ZEROMV contributes 3 per table → counter = 3 → ctx 1.
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 1);
    }

    /// Counter = 6 (two ZEROs) → ctx 0 (BOTH_ZERO).
    #[test]
    fn mode_context_two_zero_neighbours_ctx0() {
        let zero_n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &zero_n)
            .with(Position { row: 0, col: -1 }, &zero_n);
        // counter = 6 → ctx 0.
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 0);
    }

    /// Counter = 9 (one intra) → ctx 5 (INTRA_PLUS_NON_INTRA).
    #[test]
    fn mode_context_one_intra_neighbour_ctx5() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let nb = ListNeighbours::new().with(Position { row: -1, col: 0 }, &intra);
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 5);
    }

    /// Counter = 2 (two NEW) → ctx 4 (BOTH_NEW).
    #[test]
    fn mode_context_two_new_neighbours_ctx4() {
        let new_n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &new_n)
            .with(Position { row: 0, col: -1 }, &new_n);
        // 1 + 1 = 2 → ctx 4.
        assert_eq!(get_mode_context(BlockSize::Block16x16, &nb), 4);
    }

    // ----- Round-trip for each parent block size (8x8..=64x64) with
    //       NEWMV — exercises `resolve_parent_mvs` for each size

    fn parent_newmv_round_trip(bsize: BlockSize) {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv { row: 4, col: -2 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { mode, .. } => assert_eq!(mode, InterMode::NewMv),
            _ => panic!("expected Inter"),
        }
    }

    #[test]
    fn parent_newmv_block8x8() {
        parent_newmv_round_trip(BlockSize::Block8x8);
    }
    #[test]
    fn parent_newmv_block8x16() {
        parent_newmv_round_trip(BlockSize::Block8x16);
    }
    #[test]
    fn parent_newmv_block16x8() {
        parent_newmv_round_trip(BlockSize::Block16x8);
    }
    #[test]
    fn parent_newmv_block16x32() {
        parent_newmv_round_trip(BlockSize::Block16x32);
    }
    #[test]
    fn parent_newmv_block32x16() {
        parent_newmv_round_trip(BlockSize::Block32x16);
    }
    #[test]
    fn parent_newmv_block32x32() {
        parent_newmv_round_trip(BlockSize::Block32x32);
    }
    #[test]
    fn parent_newmv_block32x64() {
        parent_newmv_round_trip(BlockSize::Block32x64);
    }
    #[test]
    fn parent_newmv_block64x32() {
        parent_newmv_round_trip(BlockSize::Block64x32);
    }
    #[test]
    fn parent_newmv_block64x64() {
        parent_newmv_round_trip(BlockSize::Block64x64);
    }

    /// NEARMV at 16x16 with two distinct neighbours: list count = 2;
    /// the decoder picks mvs[1] = left.
    #[test]
    fn parent_nearmv_with_two_distinct_neighbours_round_trip() {
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
        let mi = build_inter_mi(
            bsize,
            InterMode::NearMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [left_mv, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { mvs, .. } => {
                assert_eq!(mvs[0], left_mv, "NEAR = list[1] = left");
            }
            _ => panic!("expected Inter"),
        }
    }

    // ----- Sub-8 round trips: NEW with neighbours; mixed sizes

    /// Block4x8 with mixed modes — top-half ZERO, bottom-half NEW.
    /// The replication invariant means bmi[2] = bmi[0] (also ZERO)
    /// and bmi[3] = bmi[1] (NEW with the new MV).
    #[test]
    fn sub8_block4x8_mixed_zero_new_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x8;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: 6, col: -4 }, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: 6, col: -4 }, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Block8x4 with mixed modes.
    #[test]
    fn sub8_block8x4_mixed_zero_new_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.allow_high_precision_mv = true;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block8x4;
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::ZeroMv, [Mv::ZERO, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: -4, col: 8 }, Mv::ZERO]),
            (InterMode::NewMv, [Mv { row: -4, col: 8 }, Mv::ZERO]),
        ];
        let mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    /// Sub-8 round-trip with switchable interpolation filter (sharp).
    #[test]
    fn sub8_block4x4_switchable_sharp_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
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
            InterpFilter::EightTapSharp,
            entries,
        );
        let got = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { interp_filter, .. } => {
                assert_eq!(interp_filter, InterpFilter::EightTapSharp);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Sub-8 compound NEWMV: each sub-block ZeroMv with two ref slots
    /// populated. Verify both compound refs round-trip via sub-8.
    #[test]
    fn sub8_block4x8_compound_zero_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
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
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let got = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::AltrefFrame]);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Sub-8 round-trip with `Block8x4` compound.
    #[test]
    fn sub8_block8x4_compound_zero_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
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
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            InterpFilter::EightTap,
            entries,
        );
        let _ = round_trip_sub8(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    // ----- Single-ref GOLDEN / ALTREF round-trips at parent level

    /// Single-ref GOLDEN at 16x16, NEWMV. Exercises the `bit0=1,
    /// bit1=0` single-ref path.
    #[test]
    fn parent_newmv_single_golden_round_trip() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::NewMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv { row: 6, col: -2 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames[0], RefFrame::GoldenFrame);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Single-ref ALTREF at 16x16, NEWMV. Exercises the `bit0=1,
    /// bit1=1` single-ref path.
    #[test]
    fn parent_newmv_single_altref_round_trip() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            BlockSize::Block16x16,
            InterMode::NewMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv { row: -6, col: 2 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let got = round_trip(&fc, &params, &nb, BlockSize::Block16x16, &bounds, None, &mi);
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames[0], RefFrame::AltrefFrame);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// Single-ref GOLDEN with NEAR — exercises NEAR list builder with
    /// the non-LAST single ref.
    #[test]
    fn parent_nearmv_single_golden_round_trip() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let above = inter_neighbour_full(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv { row: 4, col: 0 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv { row: -2, col: 6 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::NearMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            // list[1] = left.
            [Mv { row: -2, col: 6 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    // ----- mode_counts accumulation: compound, single-ref non-LAST,
    //       switchable interp filter, ref-mode select

    /// `read_inter_block_mode_info` with mode_counts populated, in
    /// compound mode → exercises the `c.comp_ref` write path.
    #[test]
    fn read_inter_block_mode_info_populates_comp_ref_count() {
        use crate::entropy::adapt::FrameCounts;
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;

        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();

        // Encode a known compound block via the writer.
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect("writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut got = skeleton_mi(bsize);
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
            &mut got,
        );
        // The comp_ref counter at the relevant ctx must have been
        // incremented exactly once.
        let total_comp_ref: usize = mode_counts
            .comp_ref
            .iter()
            .map(|p| p[0] + p[1])
            .sum::<u32>() as usize;
        assert_eq!(total_comp_ref, 1, "comp_ref incremented exactly once");
    }

    /// Single-ref GOLDEN with mode_counts populated → exercises
    /// `c.single_ref[ctx1][1][bit1=0]` increment.
    #[test]
    fn read_inter_block_mode_info_populates_single_ref_p2_count() {
        use crate::entropy::adapt::FrameCounts;
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect("writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut got = skeleton_mi(bsize);
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
            &mut got,
        );
        // single_ref[ctx0][0] (p1) and single_ref[ctx1][1] (p2)
        // both incremented. We don't depend on the specific ctx but
        // we assert the slot-1 (p2) accumulator was touched.
        let total_p2: u32 = mode_counts
            .single_ref
            .iter()
            .map(|c| c[1][0] + c[1][1])
            .sum();
        assert_eq!(total_p2, 1, "single_ref p2 (bit1) incremented");
        // Bit1 = 0 (GOLDEN), so the [1][0] slot should be 1.
        let total_bit1_zero: u32 = mode_counts.single_ref.iter().map(|c| c[1][0]).sum();
        assert_eq!(total_bit1_zero, 1, "bit1=0 for GOLDEN");
        // The decoded GOLDEN ref must round-trip.
        match got.mode {
            BlockMode::Inter { ref_frames, .. } => {
                assert_eq!(ref_frames[0], RefFrame::GoldenFrame);
            }
            _ => panic!("expected Inter"),
        }
    }

    /// `read_inter_block_mode_info` with reference_mode=Select +
    /// counts populated → exercises `comp_inter` count.
    #[test]
    fn read_inter_block_mode_info_populates_comp_inter_count() {
        use crate::entropy::adapt::FrameCounts;
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.reference_mode = ReferenceMode::Select;

        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );

        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect("writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut got = skeleton_mi(bsize);
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
            &mut got,
        );
        let total_comp_inter: u32 = mode_counts.comp_inter.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total_comp_inter, 1, "comp_inter counted once");
    }

    /// `read_inter_block_mode_info` with switchable interp +
    /// counts populated → exercises `switchable_interp` count.
    #[test]
    fn read_inter_block_mode_info_populates_switchable_interp_count() {
        use crate::entropy::adapt::FrameCounts;
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;

        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        let mut buf = vec![0u8; 256];
        let size = {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect("writer ok");
            finalise(w)
        };
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let mut got = skeleton_mi(bsize);
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
            &mut got,
        );
        let total: u32 = mode_counts
            .switchable_interp
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 1);
    }

    // ----- Writer-side count plumbing — exercises the `counts` arg
    //       in write_inter_block_mode_info (parallel to reader)

    /// Writer accumulates `single_ref` count for GOLDEN target.
    #[test]
    fn write_inter_block_mode_info_populates_single_ref_p2_count() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        let total_p1: u32 = counts.single_ref.iter().map(|c| c[0][0] + c[0][1]).sum();
        let total_p2: u32 = counts.single_ref.iter().map(|c| c[1][0] + c[1][1]).sum();
        assert_eq!(total_p1, 1, "single_ref p1 written once");
        assert_eq!(total_p2, 1, "single_ref p2 written once (GOLDEN target)");
    }

    /// Writer accumulates `single_ref` count for ALTREF target —
    /// p2 bit = 1.
    #[test]
    fn write_inter_block_mode_info_populates_single_ref_altref_count() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        // For ALTREF the bit1 = 1 → single_ref[ctx1][1][1] += 1.
        let total_bit1_one: u32 = counts.single_ref.iter().map(|c| c[1][1]).sum();
        assert_eq!(total_bit1_one, 1, "ALTREF: p2 bit=1");
    }

    /// Writer compound path: accumulates `comp_ref` count.
    #[test]
    fn write_inter_block_mode_info_populates_comp_ref_count() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        let total: u32 = counts.comp_ref.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total, 1, "comp_ref counted once");
    }

    /// Writer Select path: accumulates `comp_inter` count.
    #[test]
    fn write_inter_block_mode_info_populates_comp_inter_count() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.reference_mode = ReferenceMode::Select;
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        let total: u32 = counts.comp_inter.iter().map(|c| c[0] + c[1]).sum();
        assert_eq!(total, 1);
    }

    /// Writer switchable interp: accumulates `switchable_interp`.
    #[test]
    fn write_inter_block_mode_info_populates_switchable_interp_count() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.interp_filter = InterpFilter::Switchable;
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        let total: u32 = counts
            .switchable_interp
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 1);
    }

    /// Writer NEWMV: accumulates `inter_mode` count for the NEW
    /// leaf.
    #[test]
    fn write_inter_block_mode_info_populates_inter_mode_count_new() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let bsize = BlockSize::Block16x16;
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let mi = build_inter_mi(
            bsize,
            InterMode::NewMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv { row: 4, col: 0 }, Mv::ZERO],
            InterpFilter::EightTap,
        );
        let mut buf = vec![0u8; 256];
        let mut counts = FrameCounts::default();
        {
            let mut w = BoolWriter::start_encode(&mut buf);
            write_inter_block_mode_info(
                &mut w,
                &fc,
                &params,
                &nb,
                bsize,
                &bounds,
                None,
                &mi,
                Some(&mut counts),
            )
            .expect("writer ok");
            let _ = finalise(w);
        }
        let total: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c.iter().sum::<u32>())
            .sum();
        assert_eq!(total, 1);
        // NEW = idx 3 → the [_][3] slot incremented.
        let total_new: u32 = counts
            .inter_mode
            .iter()
            .map(|c| c[InterMode::NewMv.idx()])
            .sum();
        assert_eq!(total_new, 1, "NEWMV slot incremented");
    }

    // ----- get_pred_context_switchable_interp: the "above_type ==
    //       sentinel" branch (left has filter, above is sentinel)

    /// Switchable interp ctx: left has filter, above is sentinel
    /// (None) → ctx = left's filter.
    #[test]
    fn switchable_interp_ctx_above_sentinel_left_real() {
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        // above = None (sentinel), left = EightTapSharp.
        let ctx = get_pred_context_switchable_interp(None, Some(&left));
        assert_eq!(ctx, InterpFilter::EightTapSharp as usize);
    }

    /// Switchable interp ctx: above = intra (sentinel), left = real.
    #[test]
    fn switchable_interp_ctx_above_intra_left_real() {
        let above = intra_neighbour(BlockSize::Block16x16);
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSharp,
        );
        let ctx = get_pred_context_switchable_interp(Some(&above), Some(&left));
        assert_eq!(ctx, InterpFilter::EightTapSharp as usize);
    }

    /// The realtime filter reference prefers an inter above neighbour even
    /// when the inter left neighbour chose a different filter.
    #[test]
    fn pred_filter_ref_prefers_inter_above_then_left() {
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTapSmooth,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        assert_eq!(
            get_pred_filter_ref(Some(&above), Some(&left)),
            InterpFilter::EightTapSmooth,
        );
        assert_eq!(
            get_pred_filter_ref(Some(&intra_neighbour(BlockSize::Block16x16)), Some(&left)),
            InterpFilter::EightTap,
        );
        assert_eq!(get_pred_filter_ref(None, None), InterpFilter::Switchable,);
    }

    // ----- get_reference_mode_context: edge has_second_ref vs not

    /// `get_reference_mode_context` single-edge intra → ctx = 1
    /// (no compound, `!is_inter_block` → has_second_ref returns
    /// false for intra → reads `ref0(edge) == comp_fixed_ref` where
    /// ref0 of intra = IntraFrame).
    #[test]
    fn reference_mode_ctx_single_edge_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        // For intra edge: has_second_ref = false; ref0(edge) = IntraFrame.
        // IntraFrame != ALTREF → ctx = 0.
        let ctx = get_reference_mode_context(Some(&intra), None, RefFrame::AltrefFrame);
        assert_eq!(ctx, 0);
    }

    /// `get_reference_mode_context` no neighbours → ctx = 1.
    #[test]
    fn reference_mode_ctx_no_neighbours() {
        let ctx = get_reference_mode_context(None, None, RefFrame::AltrefFrame);
        assert_eq!(ctx, 1);
    }

    // ----- get_pred_context_comp_ref_p: inter+inter compound branch
    //       arms (lines 343-367 in the production code)

    /// `comp_ref_p`: both single-ref, both equal & equal to var1 → ctx 0.
    /// vrfa = vrfl = ref0(above/left); var1 = comp_var_ref[1].
    /// Setup: sign_bias = `[false, false, true]` → fixed=ALTREF,
    /// var=[LAST, GOLDEN], var1=GOLDEN. Both neighbours single-ref
    /// GOLDEN → vrfa==vrfl==GOLDEN==var1 → ctx = 0.
    #[test]
    fn comp_ref_p_both_inter_single_both_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&n), Some(&n), &sign_bias, fixed, var);
        assert_eq!(ctx, 0);
    }

    /// `comp_ref_p`: both single, vrfa == comp_fixed_ref AND vrfl ==
    /// comp_var_ref[0] → ctx = 4 (first cross-match branch).
    /// Setup: fixed=ALTREF, var=[LAST, GOLDEN]. above=ALTREF,
    /// left=LAST → vrfa=ALTREF=fixed, vrfl=LAST=var[0] → ctx = 4.
    #[test]
    fn comp_ref_p_both_inter_single_fixed_var0_cross() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        assert_eq!(ctx, 4);
    }

    /// `comp_ref_p`: both single, vrfa == vrfl but != var1 → ctx = 3.
    /// Setup: both LAST → vrfa=vrfl=LAST. var1=GOLDEN.
    /// Not the cross-match branch (LAST==var[0] but also need
    /// vrfl==fixed). LAST ≠ ALTREF, so we fall through to vrfa==vrfl.
    #[test]
    fn comp_ref_p_both_inter_single_same_not_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // vrfa=vrfl=LAST, var1=GOLDEN, comp_fixed_ref=ALTREF,
        // comp_var_ref[0]=LAST → vrfa==comp_fixed_ref? ALTREF==LAST?
        // No. vrfl==comp_fixed_ref? No. Not the cross branch. vrfa
        // == vrfl true → ctx = 3.
        let ctx = get_pred_context_comp_ref_p(Some(&n), Some(&n), &sign_bias, fixed, var);
        assert_eq!(ctx, 3);
    }

    /// `comp_ref_p`: both single, vrfa != vrfl, neither cross-match
    /// → ctx = 1.
    /// Setup: above=LAST, left=GOLDEN. var=[LAST,GOLDEN], fixed=ALTREF.
    /// vrfa=LAST, vrfl=GOLDEN. vrfa==var1? var1=GOLDEN, LAST≠GOLDEN.
    /// vrfa==vrfl? No. Cross: LAST=fixed? No. GOLDEN=fixed? No.
    /// vrfa==vrfl? No → ctx = 1.
    #[test]
    fn comp_ref_p_both_inter_single_distinct_no_match() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        assert_eq!(ctx, 1);
    }

    /// `comp_ref_p`: one compound + one single, vrfc==var1, rfs!=var1
    /// → ctx = 1.
    /// Setup: above = single LAST, left = compound [LAST,GOLDEN].
    /// var=[LAST,GOLDEN], var1=GOLDEN, fixed=ALTREF. var_ref_idx = 0
    /// (sign_bias[ALTREF]=true → fix_ref_idx=1 → var_ref_idx=0).
    /// l_sg = !has_second(left) = false (left is compound).
    /// a_sg = !has_second(above) = true.
    /// vrfa = ref0(above) = LAST. vrfl = var_ref_of(left) =
    /// ref_frames[0] = LAST.
    /// vrfa == vrfl: LAST==LAST true; var1=GOLDEN ≠ vrfa → not arm 0.
    /// l_sg && a_sg: false (only one is single).
    /// l_sg || a_sg true (a_sg). vrfc: l_sg=false → vrfc = vrfl =
    /// LAST. rfs: a_sg=true → rfs = vrfa = LAST.
    /// vrfc == var1? LAST == GOLDEN? No. rfs == var1? No → else → 4.
    /// So this would be ctx 4. Let's get ctx 1 differently.
    ///
    /// For vrfc == var1 = GOLDEN, rfs != var1: need single edge ref0
    /// != GOLDEN AND compound edge var_ref_of == GOLDEN.
    /// Set above = single LAST (rfs=LAST), left = compound
    /// [GOLDEN, ALTREF] → var_ref_of(left) = ref_frames[0] = GOLDEN.
    #[test]
    fn comp_ref_p_compound_plus_single_vrfc_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // var1 = GOLDEN.
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        // a_sg = true (above single), vrfa = LAST. l_sg = false
        // (left compound). vrfl = ref_frames[var_ref_idx=0] = GOLDEN.
        // vrfa != vrfl. l_sg || a_sg true.
        // vrfc = vrfl = GOLDEN. rfs = vrfa = LAST.
        // vrfc == var1 = GOLDEN: YES. rfs != var1: YES. → ctx = 1.
        assert_eq!(ctx, 1);
    }

    /// `comp_ref_p`: one compound + one single, rfs==var1, vrfc!=var1
    /// → ctx = 2.
    /// Setup: above single GOLDEN (rfs=GOLDEN=var1), left compound
    /// [LAST, ALTREF] → var_ref_of(left)=LAST (≠var1).
    #[test]
    fn comp_ref_p_compound_plus_single_rfs_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        assert_eq!(ctx, 2);
    }

    /// `comp_ref_p`: both compound, vrfa==vrfl != var1 → ctx = 4.
    #[test]
    fn comp_ref_p_both_compound_equal_not_var1() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        // Both compound [LAST, ALTREF] → var_ref_of = LAST.
        // var1 = GOLDEN. LAST != GOLDEN.
        let n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&n), Some(&n), &sign_bias, fixed, var);
        assert_eq!(ctx, 4);
    }

    /// `comp_ref_p`: both compound, vrfa != vrfl → ctx = 2.
    #[test]
    fn comp_ref_p_both_compound_distinct() {
        let sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&sign_bias);
        let above = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_comp_ref_p(Some(&above), Some(&left), &sign_bias, fixed, var);
        assert_eq!(ctx, 2);
    }

    // ----- get_pred_context_single_ref_p2: inter+inter detailed arms

    /// `single_ref_p2`: both inter compound, same ref pair, has
    /// GOLDEN → ctx = 3.
    /// (Already covered above; this complements with no GOLDEN.)
    #[test]
    fn single_ref_p2_both_compound_same_no_golden_zero() {
        let n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // Same pair, no GOLDEN → 3 * 0 = 0.
        let ctx = get_pred_context_single_ref_p2(Some(&n), Some(&n));
        assert_eq!(ctx, 0);
    }

    /// `single_ref_p2`: one compound + one single GOLDEN where rfs
    /// = GOLDEN → ctx = 3.
    #[test]
    fn single_ref_p2_compound_single_rfs_golden_no_match() {
        // compound = [LAST, ALTREF] (no GOLDEN); single GOLDEN.
        // rfs = GOLDEN → 3 + (crf1 or crf2 == GOLDEN). LAST/ALTREF
        // → 0 → ctx = 3.
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // single above + compound left.
        let ctx = get_pred_context_single_ref_p2(Some(&single), Some(&compound));
        assert_eq!(ctx, 3);
    }

    /// `single_ref_p2`: one compound + one single ALTREF where rfs
    /// = ALTREF, crf has GOLDEN → ctx = 1.
    #[test]
    fn single_ref_p2_compound_single_rfs_altref_crf_golden() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        // single above + compound left. rfs = above0 = ALTREF.
        // crf1=left0=LAST, crf2=left1=GOLDEN. ALTREF branch: ctx = (crf
        // has GOLDEN) = 1.
        let ctx = get_pred_context_single_ref_p2(Some(&single), Some(&compound));
        assert_eq!(ctx, 1);
    }

    /// `single_ref_p2`: one compound + one single LAST where rfs =
    /// LAST (neither GOLDEN nor ALTREF) and crf doesn't have GOLDEN
    /// → ctx = 1 (1 + 2*0).
    #[test]
    fn single_ref_p2_compound_single_rfs_last_no_golden() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&single), Some(&compound));
        // rfs = LAST (neither GOLDEN nor ALTREF). 1 + 2*0 = 1.
        assert_eq!(ctx, 1);
    }

    /// `single_ref_p2`: one compound + one single LAST where crf has
    /// GOLDEN → ctx = 1 + 2 = 3.
    #[test]
    fn single_ref_p2_compound_single_rfs_last_crf_golden() {
        let compound = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::GoldenFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let single = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&single), Some(&compound));
        assert_eq!(ctx, 3);
    }

    /// `single_ref_p2`: both inter single, both LAST → ctx = 3.
    #[test]
    fn single_ref_p2_both_inter_single_both_last() {
        let n = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&n), Some(&n));
        assert_eq!(ctx, 3);
    }

    /// `single_ref_p2`: both inter single, one LAST one GOLDEN →
    /// ctx = 4.
    #[test]
    fn single_ref_p2_both_inter_single_one_last_one_golden() {
        let last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&last), Some(&golden));
        assert_eq!(ctx, 4);
    }

    /// `single_ref_p2`: both inter single, one LAST one ALTREF →
    /// ctx = 0 (4 * (ALTREF == GOLDEN) = 0).
    #[test]
    fn single_ref_p2_both_inter_single_one_last_one_altref() {
        let last = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let altref = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&last), Some(&altref));
        assert_eq!(ctx, 0);
    }

    /// `single_ref_p2`: both inter single, no LAST, one GOLDEN +
    /// one ALTREF → ctx = 2 * 1 + 2 * 0 = 2.
    #[test]
    fn single_ref_p2_both_inter_single_no_last_one_golden() {
        let golden = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::GoldenFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let altref = inter_neighbour_full(
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            [RefFrame::AltrefFrame, RefFrame::NoRefFrame],
            [Mv::ZERO; 2],
            InterpFilter::EightTap,
        );
        let ctx = get_pred_context_single_ref_p2(Some(&golden), Some(&altref));
        assert_eq!(ctx, 2);
    }

    /// `single_ref_p2`: single edge intra → ctx = 2.
    #[test]
    fn single_ref_p2_single_edge_intra() {
        let intra = intra_neighbour(BlockSize::Block16x16);
        let ctx = get_pred_context_single_ref_p2(Some(&intra), None);
        assert_eq!(ctx, 2);
    }

    /// `single_ref_p2`: no neighbours → ctx = 2.
    #[test]
    fn single_ref_p2_no_neighbours() {
        let ctx = get_pred_context_single_ref_p2(None, None);
        assert_eq!(ctx, 2);
    }

    /// `single_ref_p1`: no neighbours → ctx = 2.
    #[test]
    fn single_ref_p1_no_neighbours() {
        let ctx = get_pred_context_single_ref_p1(None, None);
        assert_eq!(ctx, 2);
    }

    // ----- InterModeWriteError Display impl

    /// `InterModeWriteError::NotInter` Display formatting.
    #[test]
    fn inter_mode_write_error_display_not_inter() {
        let e = InterModeWriteError::NotInter;
        let s = format!("{e}");
        assert!(s.contains("expected"), "actual: {s}");
        assert!(s.contains("Inter"), "actual: {s}");
    }

    /// `InterModeWriteError::Sub8InvalidSubMode` Display formatting.
    #[test]
    fn inter_mode_write_error_display_sub8_invalid_sub_mode() {
        let e = InterModeWriteError::Sub8InvalidSubMode { j: 1, raw: 42 };
        let s = format!("{e}");
        assert!(s.contains("bmi[1]"), "actual: {s}");
        assert!(s.contains("42"), "actual: {s}");
    }

    /// `InterModeWriteError::Sub8ParentModeMismatch` Display
    /// formatting.
    #[test]
    fn inter_mode_write_error_display_parent_mode_mismatch() {
        let e = InterModeWriteError::Sub8ParentModeMismatch {
            parent: 3,
            last_sub: 0,
        };
        let s = format!("{e}");
        assert!(s.contains("parent mode"), "actual: {s}");
        assert!(s.contains('3') && s.contains('0'), "actual: {s}");
    }

    // ----- write_sub8_bmi error path: invalid sub-mode index

    /// `write_inter_block_mode_info` for a sub-8 block whose bmi[j]
    /// has an invalid `as_mode` raw byte → `Sub8InvalidSubMode`.
    #[test]
    fn write_sub8_rejects_invalid_sub_mode_index() {
        let fc = FrameContext::new_default();
        let params = default_params();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block4x4;
        // Build a valid sub-8 cell and corrupt bmi[0].as_mode to an
        // invalid index (4 — InterMode has only 0..=3).
        let entries = [
            (InterMode::ZeroMv, [Mv::ZERO; 2]),
            (InterMode::ZeroMv, [Mv::ZERO; 2]),
            (InterMode::ZeroMv, [Mv::ZERO; 2]),
            (InterMode::ZeroMv, [Mv::ZERO; 2]),
        ];
        let mut mi = build_sub8_inter_mi(
            bsize,
            [RefFrame::LastFrame, RefFrame::NoRefFrame],
            InterpFilter::EightTap,
            entries,
        );
        mi.bmi[0].as_mode = 99;
        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        let err =
            write_inter_block_mode_info(&mut w, &fc, &params, &nb, bsize, &bounds, None, &mi, None)
                .expect_err("expected Sub8InvalidSubMode");
        assert!(
            matches!(
                err,
                InterModeWriteError::Sub8InvalidSubMode { j: 0, raw: 99 }
            ),
            "actual: {err:?}",
        );
    }

    /// Compound NEAR at 16x16: exercises NEAR list builder for both
    /// compound slots (per-ref `find_mv_refs` runs twice).
    #[test]
    fn parent_compound_nearmv_round_trip() {
        let fc = FrameContext::new_default();
        let mut params = default_params();
        params.sign_bias = sign_bias_from_header(&[false, false, true]);
        let (fixed, var) = setup_compound_reference_mode(&params.sign_bias);
        params.comp_fixed_ref = fixed;
        params.comp_var_ref = var;
        params.reference_mode = ReferenceMode::Compound;
        let bsize = BlockSize::Block16x16;
        let above = inter_neighbour_full(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv { row: 4, col: 0 }, Mv { row: -8, col: 0 }],
            InterpFilter::EightTap,
        );
        let left = inter_neighbour_full(
            bsize,
            InterMode::ZeroMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv { row: -4, col: 2 }, Mv { row: 8, col: -2 }],
            InterpFilter::EightTap,
        );
        let nb = ListNeighbours::new()
            .with(Position { row: -1, col: 0 }, &above)
            .with(Position { row: 0, col: -1 }, &left);
        let bounds = loose_bounds();
        // For NEAR with two distinct neighbours: list[1] = left's MV
        // for each ref slot.
        let mi = build_inter_mi(
            bsize,
            InterMode::NearMv,
            [RefFrame::LastFrame, RefFrame::AltrefFrame],
            [Mv { row: -4, col: 2 }, Mv { row: 8, col: -2 }],
            InterpFilter::EightTap,
        );
        let _ = round_trip(&fc, &params, &nb, bsize, &bounds, None, &mi);
    }

    // -----------------------------------------------------------------
    // G-C: SEG_LVL_REF_FRAME / SEG_LVL_SKIP forcing oracle (§4.4).
    //
    // The "zero bits consumed" assertions (reader register unchanged)
    // are THE oracle for bool-reader sync: a forced branch that
    // accidentally read a bit would advance `cursor_state()`.
    // -----------------------------------------------------------------

    /// A valid (decodable) bool-coder buffer of zero-bits. Used by the
    /// "zero bits consumed" oracle tests: `BoolReader::new` requires a
    /// proper marker bit, so a raw `0xFF` buffer would be rejected.
    fn valid_padding_buf() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let mut w = BoolWriter::start_encode(&mut buf);
        for _ in 0..64 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("stop_encode");
        buf.truncate(size);
        buf
    }

    /// Inter params with `SEG_LVL_REF_FRAME` active, forcing `ref0` to
    /// the given frame.
    fn forced_ref_params(forced: RefFrame) -> InterModeReadParams {
        InterModeReadParams {
            seg_ref_frame: Some(forced),
            ..default_params()
        }
    }

    /// **read_ref_frames force**: `SEG_LVL_REF_FRAME` active ⇒
    /// `(ref0 = segdata, ref1 = NoRefFrame, is_compound = false)` and
    /// ZERO comp/ref bits consumed. Mirrors libvpx `read_ref_frames`
    /// (`vp9_decodemv.c:303-309`).
    #[test]
    fn read_ref_frames_seg_force_consumes_zero_bits() {
        let fc = FrameContext::new_default();
        // A VALID bool-coder buffer (proper marker bit + padding). If
        // any ref bit WERE read it would advance the reader; we assert
        // the reader register is byte-identical before/after, so the
        // buffer's contents past the marker are irrelevant.
        let buf = valid_padding_buf();

        for forced in [
            RefFrame::LastFrame,
            RefFrame::GoldenFrame,
            RefFrame::AltrefFrame,
        ] {
            let params = forced_ref_params(forced);
            let mut reader = BoolReader::new(&buf).expect("init");
            let before = reader.cursor_state();
            let (ref0, ref1, is_compound) =
                read_ref_frames(&mut reader, &fc, &params, None, None, None);
            let after = reader.cursor_state();

            assert_eq!(ref0, forced, "ref0 must be the forced segdata");
            assert_eq!(ref1, RefFrame::NoRefFrame, "single-ref forced");
            assert!(!is_compound, "forced ref is always single");
            assert_eq!(
                before, after,
                "SEG_LVL_REF_FRAME force must consume ZERO bits (reader register unchanged)",
            );
        }
    }

    /// **SKIP ZEROMV force**: `SEG_LVL_SKIP` active, `bsize >= 8x8` ⇒
    /// `mode == ZeroMv`, `ref_frames = (forced, NONE)`, MVs zero, and
    /// NO inter-mode / comp / ref bits consumed. The whole mode-info
    /// read advances the reader by ZERO because both the ref read
    /// (REF_FRAME force) and the inter-mode read (SKIP force) are
    /// suppressed, and ZEROMV consumes no MV bits. With a non-switchable
    /// interp filter no filter bit is read either. Mirrors libvpx
    /// `read_inter_block_mode_info` (`vp9_decodemv.c:703-717`).
    #[test]
    fn seg_skip_forces_zeromv_and_consumes_zero_bits() {
        let fc = FrameContext::new_default();
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;
        // Both features active: REF_FRAME forces LAST, SKIP forces ZEROMV.
        let params = InterModeReadParams {
            seg_ref_frame: Some(RefFrame::LastFrame),
            seg_skip: true,
            ..default_params()
        };
        let buf = valid_padding_buf();
        let mut reader = BoolReader::new(&buf).expect("init");
        let before = reader.cursor_state();
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
        let after = reader.cursor_state();

        match mi.mode {
            BlockMode::Inter {
                mode,
                ref_frames,
                mvs,
                ..
            } => {
                assert_eq!(mode, InterMode::ZeroMv, "SEG_LVL_SKIP forces ZEROMV");
                assert_eq!(
                    ref_frames,
                    [RefFrame::LastFrame, RefFrame::NoRefFrame],
                    "REF_FRAME force pins ref0 and single-refs ref1",
                );
                assert_eq!(mvs, [Mv::ZERO, Mv::ZERO], "ZEROMV ⇒ zero MVs");
            }
            _ => panic!("expected Inter"),
        }
        assert_eq!(
            before, after,
            "forced REF_FRAME + SKIP (ZEROMV) must consume ZERO bits",
        );
    }

    /// **Inactive-feature regression guard**: with both features OFF,
    /// `read_inter_block_mode_info` still reads the ref + inter-mode
    /// bits from the stream (it ADVANCES the reader) and decodes the
    /// hand-built single-ref-LAST + ZEROMV exactly. This guards that
    /// the new gates don't perturb the common (non-forced) path.
    #[test]
    fn features_inactive_common_path_still_reads_bits() {
        let fc = FrameContext::new_default();
        let params = default_params(); // seg_ref_frame: None, seg_skip: false
        let nb = ListNeighbours::new();
        let bounds = loose_bounds();
        let bsize = BlockSize::Block16x16;

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        let p_single = fc.single_ref_prob[2][0];
        write_single_ref_bits(&mut w, RefFrame::LastFrame, p_single, 0);
        write_inter_mode_bits(&mut w, InterMode::ZeroMv, &fc.inter_mode_probs[0]);
        let size = finalise(w);
        buf.truncate(size);

        let mut reader = BoolReader::new(&buf).expect("init");
        let before = reader.cursor_state();
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
        let after = reader.cursor_state();

        match mi.mode {
            BlockMode::Inter {
                mode, ref_frames, ..
            } => {
                assert_eq!(mode, InterMode::ZeroMv);
                assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
            }
            _ => panic!("expected Inter"),
        }
        assert_ne!(
            before, after,
            "inactive features ⇒ common path MUST read bits (reader advances)",
        );
    }
}
