//! Per-8×8-block descriptor — the [`ModeInfo`] struct + the
//! [`InterMode`], [`RefFrame`], [`Mv`], and [`BlockMode`] types it
//! aggregates.
//!
//! Mirrors libvpx `MODE_INFO` (`vp9/common/vp9_blockd.h:74-97`) plus
//! the supporting `PREDICTION_MODE` / `MV_REFERENCE_FRAME` /
//! `int_mv` / `b_mode_info` definitions.
//!
//! ## Design note: split inter / intra modes
//!
//! libvpx unifies intra and inter modes under a single 4-bit
//! `PREDICTION_MODE` (`0..=13`). Rvp9 separates them via the
//! [`BlockMode`] enum:
//!
//! - [`BlockMode::Intra`] carries `y_mode` / `uv_mode` as
//!   [`IntraMode`] (the existing [`crate::intra_pred::IntraMode`]).
//! - [`BlockMode::Inter`] carries [`InterMode`] + `ref_frames` +
//!   motion vectors + interpolation filter.
//!
//! The two cases are mutually exclusive (a block's `ref_frame[0] ==
//! INTRA_FRAME` means intra; anything else is inter). The enum
//! discriminant makes that exclusion type-safe and lets the
//! per-block decoder pattern-match on `BlockMode` rather than
//! threading a separate `is_inter` boolean.

use crate::bitstream::{
    InterpFilter, MAX_SEGMENTS, PREDICTION_PROBS, SEG_TREE_PROBS, tx_mode_to_biggest_tx_size,
};
use crate::block::{BlockSize, PARTITION_TYPES, PartitionType};
use crate::entropy::adapt::FrameCounts;
use crate::entropy::{
    BoolReader, BoolWriter, COMP_INTER_CONTEXTS, DefaultTxProbs, INTRA_INTER_CONTEXTS,
    REF_CONTEXTS, SKIP_CONTEXTS,
};
use crate::intra_pred::IntraMode;
use crate::transforms::{TxMode, TxSize};

/// libvpx `vp9_segment_tree` (`vp9_seg_common.c:58-60`). A
/// 14-entry balanced binary tree over the 8 segment IDs. Indexed
/// by 7 probabilities in `seg.tree_probs[0..7]`.
#[rustfmt::skip]
pub static SEGMENT_TREE: [i8; 2 * MAX_SEGMENTS - 2] = [
    2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7,
];

/// Decode the per-block segment ID by walking [`SEGMENT_TREE`].
/// Mirrors libvpx `read_segment_id` (`vp9_decodemv.c:60-62`).
///
/// Returns a value in `0..MAX_SEGMENTS = 0..=7`. Caller is the
/// per-block segmentation-map decode path; the result indexes into
/// the feature_data table that drives per-segment quantizer + LF
/// deltas.
#[inline]
#[must_use]
pub fn read_segment_id(r: &mut BoolReader<'_>, tree_probs: &[u8; SEG_TREE_PROBS]) -> u8 {
    r.read_tree(&SEGMENT_TREE, tree_probs) as u8
}

/// Inter-frame segmentation-map params: the four-state knob libvpx
/// extracts from `cm->seg` (`vp9_seg_common.h:42-65`) at the per-block
/// segment-id decode site.
///
/// Mirrors `read_inter_segment_id` (`vp9_decodemv.c:146-177`):
/// - `enabled == false` ⇒ segment_id is always 0; bitstream consumes
///   no bits.
/// - `update_map == false` ⇒ inherit `predicted_segment_id` from the
///   previous frame's seg map (zero before any frame).
/// - `update_map == true && temporal_update == false` ⇒ tree decode
///   via `tree_probs`.
/// - `update_map == true && temporal_update == true` ⇒ read a single
///   pred-flag bit against `pred_probs[ctx]`; on true inherit the
///   predicted ID, otherwise tree decode.
#[derive(Copy, Clone, Debug, Default)]
pub struct SegmentationReadParams {
    /// `seg.enabled`.
    pub enabled: bool,
    /// `seg.update_map`.
    pub update_map: bool,
    /// `seg.temporal_update`.
    pub temporal_update: bool,
    /// `seg.tree_probs`.
    pub tree_probs: [u8; SEG_TREE_PROBS],
    /// `seg.pred_probs`.
    pub pred_probs: [u8; PREDICTION_PROBS],
}

/// Decode the per-block segment ID for an inter frame. Mirrors
/// libvpx `read_inter_segment_id` (`vp9_decodemv.c:146-177`):
///
/// 1. `predicted_segment_id` = min over the previous-frame seg-map
///    cells that this block covers. (Matches libvpx
///    `dec_get_segment_id` at `vp9_decodemv.c:91-102`.)
/// 2. If `!seg.update_map`: pure inheritance — return
///    `predicted_segment_id` (no bits consumed).
/// 3. If `seg.update_map && seg.temporal_update`: read one bit against
///    `pred_probs[seg_id_pred_ctx]`. On true return
///    `predicted_segment_id`; on false fall through to the tree
///    decode. The caller is responsible for stashing the read flag in
///    `mi.seg_id_predicted` (returned as the second tuple element).
/// 4. If `seg.update_map && !seg.temporal_update`: tree decode via
///    `tree_probs` (no `seg_id_predicted` bit consumed; caller should
///    leave `mi.seg_id_predicted == false`).
///
/// Returns `(segment_id, seg_id_predicted)`. `seg_id_predicted` is the
/// value to store in `mi.seg_id_predicted` (the seg-id-predicted bit
/// participates in the next block's `seg_id_pred_ctx`).
///
/// `prev_segment_ids` is the previous frame's seg map at frame stride
/// `mi_stride` (= frame mi_cols). Pass an empty slice if no prev map
/// is available (matches libvpx's `cm->last_frame_seg_map == NULL`
/// path — `predicted_segment_id` collapses to 0). `mi_row` / `mi_col`
/// are absolute, in 8×8 mi units; `x_mis` / `y_mis` are the block's
/// covered cell counts.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)] // mirrors libvpx's `read_inter_segment_id` 1:1.
pub fn read_inter_segment_id(
    r: &mut BoolReader<'_>,
    seg: &SegmentationReadParams,
    prev_segment_ids: &[u8],
    mi_stride: usize,
    mi_row: usize,
    mi_col: usize,
    x_mis: usize,
    y_mis: usize,
    seg_id_pred_ctx: usize,
) -> (u8, bool) {
    if !seg.enabled {
        return (0, false);
    }

    // 1. Predicted segment_id = min over covered prev-frame cells.
    //    Mirrors libvpx `dec_get_segment_id`.
    let predicted_segment_id = if prev_segment_ids.is_empty() {
        0
    } else {
        let mut acc: u8 = (MAX_SEGMENTS as u8) - 1;
        let base = mi_row * mi_stride + mi_col;
        for y in 0..y_mis {
            let row_off = base + y * mi_stride;
            for x in 0..x_mis {
                let v = prev_segment_ids[row_off + x];
                if v < acc {
                    acc = v;
                }
            }
        }
        acc
    };

    if !seg.update_map {
        // Pure inheritance — no bits consumed.
        return (predicted_segment_id, false);
    }

    if seg.temporal_update {
        let pred_prob = seg.pred_probs[seg_id_pred_ctx];
        let pred_flag = r.read(pred_prob) != 0;
        if pred_flag {
            (predicted_segment_id, true)
        } else {
            (read_segment_id(r, &seg.tree_probs), false)
        }
    } else {
        (read_segment_id(r, &seg.tree_probs), false)
    }
}

/// Decode the per-block `skip` flag. Mirrors libvpx `read_skip`
/// (`vp9_decodemv.c:226-234`).
///
/// `ctx` is the 0..=2 skip-context index from the above + left
/// neighbours' skip flags (computed at the per-block level by the
/// caller). `probs = &fc.skip_probs`.
///
/// `counts`, when `Some`, accumulates a `skip[ctx][skip_bit]` tally
/// for backward adaptation
/// ([`adapt_mode_probs`](crate::entropy::adapt::adapt_mode_probs)).
/// Mirrors libvpx `++counts->skip[ctx][skip]` (`vp9_decodemv.c:187`).
#[inline]
#[must_use]
pub fn read_skip(
    r: &mut BoolReader<'_>,
    probs: &[u8; SKIP_CONTEXTS],
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> bool {
    let bit = r.read(probs[ctx]);
    if let Some(c) = counts {
        c.skip[ctx][bit as usize] += 1;
    }
    bit != 0
}

/// Decode the per-block intra/inter classification bit. Mirrors
/// libvpx `read_is_inter_block` (`vp9_decodemv.c:546-554`).
///
/// Returns `true` for an inter block. `ctx` is the 0..=3
/// intra-inter context computed from neighbours' is_inter flags.
///
/// `counts`, when `Some`, accumulates `intra_inter[ctx][is_inter]`
/// for backward adaptation. Mirrors libvpx
/// `++counts->intra_inter[ctx][is_inter]` (`vp9_decodemv.c:444`).
#[inline]
#[must_use]
pub fn read_is_inter(
    r: &mut BoolReader<'_>,
    probs: &[u8; INTRA_INTER_CONTEXTS],
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> bool {
    let bit = r.read(probs[ctx]);
    if let Some(c) = counts {
        c.intra_inter[ctx][bit as usize] += 1;
    }
    bit != 0
}

/// Decode the per-block compound-vs-single reference selection.
/// Mirrors libvpx `read_block_reference_mode` (`vp9_decodemv.c:457-464`).
///
/// Only called when the frame's `reference_mode == Select`.
/// Returns `true` for compound, `false` for single. `ctx` is the
/// 0..=4 compound-inter context computed from neighbour
/// references' time-direction agreement.
///
/// `counts`, when `Some`, accumulates `comp_inter[ctx][is_compound]`
/// for backward adaptation. Mirrors libvpx
/// `++counts->comp_inter[ctx][mode]` (`vp9_decodemv.c:295`).
#[inline]
#[must_use]
pub fn read_is_compound(
    r: &mut BoolReader<'_>,
    probs: &[u8; COMP_INTER_CONTEXTS],
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> bool {
    let bit = r.read(probs[ctx]);
    if let Some(c) = counts {
        c.comp_inter[ctx][bit as usize] += 1;
    }
    bit != 0
}

/// Decode one bit of the single-ref selection. Mirrors libvpx
/// `read_ref_frames` single-ref path (`vp9_decodemv.c:497-516`).
///
/// VP9 single-ref selection is two sequential boolean choices:
/// - First bit (`single_ref_prob[ctx][0]`): "is this NOT
///   LAST_FRAME?"
/// - If yes, second bit (`single_ref_prob[ctx][1]`): "is this
///   ALTREF_FRAME?" (otherwise GOLDEN_FRAME).
///
/// This helper reads one such bit at a specific
/// `single_ref_prob[ctx][slot]` slot. `slot` is `0` or `1`.
///
/// `counts`, when `Some`, accumulates
/// `single_ref[ctx][slot][bit]` for backward adaptation. Mirrors
/// libvpx `++counts->single_ref[ctx0][0][bit0]` /
/// `++counts->single_ref[ctx1][1][bit1]` (`vp9_decodemv.c:326,330`).
#[inline]
#[must_use]
pub fn read_single_ref_bit(
    r: &mut BoolReader<'_>,
    probs: &[[u8; 2]; REF_CONTEXTS],
    ctx: usize,
    slot: usize,
    counts: Option<&mut FrameCounts>,
) -> bool {
    let bit = r.read(probs[ctx][slot]);
    if let Some(c) = counts {
        c.single_ref[ctx][slot][bit as usize] += 1;
    }
    bit != 0
}

/// libvpx `vp9_partition_tree` (`vp9_entropymode.c:262-264`):
///
/// ```text
///         node 0
///        /       \
///    NONE         node 2
///                /      \
///            HORZ       node 4
///                       /     \
///                    VERT    SPLIT
/// ```
///
/// Flat representation: positive entries are next-node indices,
/// negative entries `-leaf` are `PartitionType` terminals.
#[rustfmt::skip]
pub static PARTITION_TREE: [i8; 2 * PARTITION_TYPES - 2] = [
    -(PartitionType::None as i8),
    2,
    -(PartitionType::Horz as i8),
    4,
    -(PartitionType::Vert as i8),
    -(PartitionType::Split as i8),
];

/// libvpx `vp9_intra_mode_tree` (`vp9_entropymode.c:245-255`).
/// 18-entry flat tree that decodes one of the 10 [`IntraMode`]
/// values from 9 boolean probabilities. The associated `probs`
/// slice has shape `[u8; INTRA_MODES - 1] = [u8; 9]`.
#[rustfmt::skip]
pub static INTRA_MODE_TREE: [i8; 2 * 10 - 2] = [
    // 0 = DC_NODE
    -(IntraMode::Dc as i8), 2,
    // 1 = TM_NODE
    -(IntraMode::Tm as i8), 4,
    // 2 = V_NODE
    -(IntraMode::V as i8), 6,
    // 3 = COM_NODE
    8, 12,
    // 4 = H_NODE
    -(IntraMode::H as i8), 10,
    // 5 = D135_NODE
    -(IntraMode::D135 as i8), -(IntraMode::D117 as i8),
    // 6 = D45_NODE
    -(IntraMode::D45 as i8), 14,
    // 7 = D63_NODE
    -(IntraMode::D63 as i8), 16,
    // 8 = D153_NODE
    -(IntraMode::D153 as i8), -(IntraMode::D207 as i8),
];

/// libvpx `vp9_inter_mode_tree` (`vp9_entropymode.c:257-260`).
/// 6-entry flat tree that decodes one of the 4 [`InterMode`]
/// indices (0..=3). The leaves are `INTER_OFFSET(mode) = mode -
/// NEARESTMV`, so the tree returns a value in `0..=3` that maps to
/// [`InterMode`] via [`InterMode::from_idx`].
#[rustfmt::skip]
pub static INTER_MODE_TREE: [i8; 2 * NUM_INTER_MODES - 2] = [
    // INTER_OFFSET(ZEROMV) = 2
    -(InterMode::ZeroMv.idx() as i8), 2,
    // INTER_OFFSET(NEARESTMV) = 0
    -(InterMode::NearestMv.idx() as i8), 4,
    // INTER_OFFSET(NEARMV) = 1
    -(InterMode::NearMv.idx() as i8),
    // INTER_OFFSET(NEWMV) = 3
    -(InterMode::NewMv.idx() as i8),
];

/// Decode an intra-prediction mode by walking [`INTRA_MODE_TREE`]
/// against the given 9-entry probability slot. Used for
/// non-keyframe Y/UV modes (`fc.y_mode_prob[block_size_group]`
/// or `fc.uv_mode_prob[y_mode]`) and for keyframe modes
/// (`vp9_kf_y_mode_prob` / `vp9_kf_uv_mode_prob`).
///
/// **Note:** this primitive returns only the decoded mode; the caller
/// is responsible for incrementing the appropriate count slot
/// (`y_mode[size_group][mode]` or `uv_mode[y_mode][uv_mode]`). The
/// count is not threaded through here because the same primitive
/// serves both Y and UV reads with different count axes, and the
/// keyframe path (which uses `vp9_kf_*_prob` and is NOT adapted)
/// would otherwise need to suppress the count. See
/// [`read_intra_mode_y`] and [`read_intra_mode_uv`] for count-aware
/// variants.
#[inline]
#[must_use]
pub fn read_intra_mode(r: &mut BoolReader<'_>, probs: &[u8; 10 - 1]) -> IntraMode {
    let raw = r.read_tree(&INTRA_MODE_TREE, probs);
    IntraMode::from_u8(raw as u8).expect("intra mode tree returns valid IntraMode")
}

/// Decode the per-block intra Y mode for the non-keyframe path.
/// Mirrors libvpx `read_intra_mode_y` (`vp9_decodemv.c:30-37`).
///
/// `probs = &fc.y_mode_prob[size_group]` (9-entry slot per the
/// libvpx `size_group_lookup` mapping; sub-8 blocks use slot 0).
/// `size_group` is the 0..=3 lookup index, used to address the
/// per-frame count slot `y_mode[size_group][mode]`.
#[inline]
#[must_use]
pub fn read_intra_mode_y(
    r: &mut BoolReader<'_>,
    probs: &[u8; 10 - 1],
    size_group: usize,
    counts: Option<&mut FrameCounts>,
) -> IntraMode {
    let mode = read_intra_mode(r, probs);
    if let Some(c) = counts {
        c.y_mode[size_group][mode as usize] += 1;
    }
    mode
}

/// Decode the per-block intra UV mode for the non-keyframe path.
/// Mirrors libvpx `read_intra_mode_uv` (`vp9_decodemv.c:40-47`).
///
/// `probs = &fc.uv_mode_prob[y_mode]` (9-entry slot per Y mode).
/// `y_mode` is the parent Y mode value, used to address the
/// per-frame count slot `uv_mode[y_mode][uv_mode]`.
#[inline]
#[must_use]
pub fn read_intra_mode_uv(
    r: &mut BoolReader<'_>,
    probs: &[u8; 10 - 1],
    y_mode: IntraMode,
    counts: Option<&mut FrameCounts>,
) -> IntraMode {
    let mode = read_intra_mode(r, probs);
    if let Some(c) = counts {
        c.uv_mode[y_mode as usize][mode as usize] += 1;
    }
    mode
}

/// Decode an inter-prediction mode by walking [`INTER_MODE_TREE`]
/// against the given 3-entry probability slot
/// (`fc.inter_mode_probs[ctx]`).
///
/// `counts`, when `Some`, accumulates `inter_mode[ctx][mode_idx]`
/// for backward adaptation. Mirrors libvpx
/// `++counts->inter_mode[ctx][INTER_OFFSET(mode)]`
/// (`vp9_decodemv.c:55`). `mode_idx` is the libvpx INTER_OFFSET
/// (0..=3, matching [`InterMode::idx`]).
#[inline]
#[must_use]
pub fn read_inter_mode(
    r: &mut BoolReader<'_>,
    probs: &[u8; NUM_INTER_MODES - 1],
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> InterMode {
    let raw = r.read_tree(&INTER_MODE_TREE, probs);
    let mode = InterMode::from_idx(raw as u8).expect("inter mode tree returns valid InterMode");
    if let Some(c) = counts {
        c.inter_mode[ctx][mode.idx()] += 1;
    }
    mode
}

/// Decode one [`PartitionType`] from the boolean reader given a
/// 3-entry context probability slot. Mirrors libvpx
/// `read_partition` (`vp9_decodeframe.c:1147-1168`) — the tree-decode
/// part. The caller provides the right context's probability slot
/// by computing the 16-way `partition_plane_context`.
///
/// `probs` is `&fc.partition_prob[ctx]` (3 probabilities), OR
/// `&vp9_kf_partition_probs[ctx]` on keyframes (libvpx
/// `set_partition_probs`).
///
/// For block sizes < 8×8 (where `Block8x8` is the smallest
/// partitionable parent), libvpx forces `PARTITION_NONE` and
/// skips the tree decode entirely. The caller handles that case.
///
/// `counts`, when `Some`, accumulates `partition[ctx][partition_type]`
/// for backward adaptation. Mirrors libvpx
/// `++counts->partition[ctx][p]` (`vp9_decodeframe.c:1165`).
#[inline]
#[must_use]
pub fn read_partition(
    r: &mut BoolReader<'_>,
    probs: &[u8; PARTITION_TYPES - 1],
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> PartitionType {
    let raw = r.read_tree(&PARTITION_TREE, probs);
    // Safety: PARTITION_TREE only contains negations of valid
    // PartitionType discriminants, so `from_u8` always succeeds.
    let p = PartitionType::from_u8(raw as u8).expect("partition tree returns valid PartitionType");
    if let Some(c) = counts {
        c.partition[ctx][p as usize] += 1;
    }
    p
}

/// Compute the 0..=1 `tx_size` context from the above + left
/// [`ModeInfo`] neighbours. Mirrors libvpx `get_tx_size_context`
/// (`vp9_pred_common.h:156-171`).
///
///
/// The libvpx algorithm:
/// - `above_ctx` = above neighbour's `tx_size` if the neighbour
///   exists and is not skipped; else `max_tx_size`.
/// - `left_ctx`  = same, for the left neighbour.
/// - If either neighbour is missing, that side mirrors the other.
/// - Result: `(above_ctx + left_ctx) > max_tx_size`.
///
/// This implementation accepts `Option<(TxSize, bool)>` for each
/// neighbour (`Some((tx_size, skip))` if the block exists; `None`
/// at the frame edge). The two-`Option<>`s are kept independent so
/// the caller doesn't have to fabricate sentinels.
///
/// Note: only meaningful when `max_tx_size >= TX_8X8` — sub-8 blocks
/// never reach the `read_selected_tx_size` path so don't need a
/// context.
///
/// **Downstream wiring:** the per-block decoder must thread the
/// actual above/left [`ModeInfo`] neighbours into this function.
/// The neighbour grid lives in
/// `rvp9_decoder::tile_ctx::TileDecodeCtx` (Wave 2 D / Stage 3
/// Task 8.2.6). The first real caller will be the per-block
/// decode loop landed by Wave 3 K (Stage 3 Task 8.3 — decoder
/// integration). The mapping is:
///
/// ```ignore
/// let above = ctx.above(mi_col).map(|m| (m.tx_size, m.skip));
/// let left  = ctx.left(mi_row_in_sb).map(|m| (m.tx_size, m.skip));
/// let tx_ctx = get_tx_size_context(above, left, max_tx_size);
/// ```
// TODO(overnight-2026-05-17): callers — wire via TileDecodeCtx
// when Wave 3 K (per-block decode loop) lands.
#[inline]
#[must_use]
pub fn get_tx_size_context(
    above: Option<(TxSize, bool)>,
    left: Option<(TxSize, bool)>,
    max_tx_size: TxSize,
) -> usize {
    let neighbour_ctx = |n: Option<(TxSize, bool)>| -> Option<u8> {
        n.map(|(tx, skip)| if skip { max_tx_size as u8 } else { tx as u8 })
    };
    let above_n = neighbour_ctx(above);
    let left_n = neighbour_ctx(left);
    // If a side is missing, it mirrors the other (libvpx
    // `if (!has_left) left_ctx = above_ctx;` and converse).
    let (above_ctx, left_ctx) = match (above_n, left_n) {
        (Some(a), Some(l)) => (a, l),
        (Some(a), None) => (a, a),
        (None, Some(l)) => (l, l),
        // Both missing: libvpx's MACROBLOCKD has the dummy zeroed
        // border row/col, so `above_mi` / `left_mi` are non-null
        // at the very top-left only when the frame has them. With
        // truly no neighbours, both sides collapse to `max_tx_size`
        // (the "skipped" default) — and the test reduces to
        // `(2 * max_tx_size) > max_tx_size`, always `true` for
        // `max_tx_size > 0`.
        (None, None) => (max_tx_size as u8, max_tx_size as u8),
    };
    usize::from(u32::from(above_ctx) + u32::from(left_ctx) > u32::from(max_tx_size as u8))
}

/// Look up the tx-size probability slice for a given
/// `(max_tx_size, ctx)` pair. Mirrors libvpx `get_tx_probs`
/// (`vp9_pred_common.h:173-181`). Returns `None` if
/// `max_tx_size == TX_4X4` (libvpx asserts on that case — sub-8
/// blocks never reach the selected-tx-size path).
#[inline]
#[must_use]
fn get_tx_probs(
    tx_probs: &DefaultTxProbs,
    max_tx_size: TxSize,
    ctx: usize,
) -> Option<&[u8]> {
    match max_tx_size {
        TxSize::Tx8x8 => Some(&tx_probs.p8x8[ctx]),
        TxSize::Tx16x16 => Some(&tx_probs.p16x16[ctx]),
        TxSize::Tx32x32 => Some(&tx_probs.p32x32[ctx]),
        TxSize::Tx4x4 => None,
    }
}

/// Read a per-block transform size from the boolean reader. Mirrors
/// libvpx `read_tx_size` + `read_selected_tx_size`
/// (`vp9_decodemv.c:64-89`).
///
/// Parameters:
/// - `tx_probs`: the frame's `fc->tx_probs` bank.
/// - `tx_mode`:  the frame-level tx-mode (from the compressed
///   header).
/// - `max_tx_size`: the block's `max_txsize_lookup[sb_type]`
///   (computed by the caller via [`crate::block::max_txsize`]).
/// - `allow_select`: libvpx's `allow_select` flag. For intra
///   blocks libvpx hardcodes `1`. For inter blocks it's `!skip ||
///   !inter_block` — i.e. only intra-coded inter blocks (in
///   "skip" mode, they're effectively intra-residual). The
///   caller computes this.
/// - `ctx`: the 0..=1 tx-size context from
///   [`get_tx_size_context`].
/// - `counts`: when `Some`, the read-selected branch accumulates
///   `tx.p{8x8,16x16,32x32}[ctx][selected]` for backward
///   adaptation. Mirrors libvpx
///   `++get_tx_counts(max_tx_size, ctx, &counts->tx)[tx_size]`
///   (`vp9_decodemv.c:76`). The non-select branch does not adapt
///   tx probs — libvpx skips the increment.
///
/// Algorithm:
/// - If `allow_select && tx_mode == TxModeSelect && max_tx_size >=
///   Tx8x8`, walk the tx-size tree against `tx_probs[max_tx_size]
///   [ctx]`:
///     - Read bit 0: if 1, tx_size becomes 1.
///     - If `tx_size != Tx4x4 && max_tx_size >= Tx16x16`, read bit
///       1; if 1, tx_size += 1.
///     - If `tx_size != Tx8x8 && max_tx_size >= Tx32x32`, read bit
///       2; if 1, tx_size += 1.
/// - Otherwise, return `min(max_tx_size,
///   tx_mode_to_biggest_tx_size(tx_mode))`.
///
/// # Panics
///
/// Panics in debug builds if the algorithm tries to read selected
/// tx_size with `max_tx_size == Tx4x4`, which is libvpx's `assert(0
/// && "Invalid max_tx_size")` case (the caller must gate this on
/// `bsize >= BLOCK_8X8` ↔ `max_tx_size >= Tx8x8`).
#[inline]
#[must_use]
pub fn read_tx_size(
    r: &mut BoolReader<'_>,
    tx_probs: &DefaultTxProbs,
    tx_mode: TxMode,
    max_tx_size: TxSize,
    allow_select: bool,
    ctx: usize,
    counts: Option<&mut FrameCounts>,
) -> TxSize {
    if allow_select && tx_mode == TxMode::TxModeSelect && max_tx_size >= TxSize::Tx8x8 {
        let probs = get_tx_probs(tx_probs, max_tx_size, ctx)
            .expect("read_selected_tx_size requires max_tx_size >= Tx8x8");
        let mut raw: u8 = r.read(probs[0]);
        if raw != TxSize::Tx4x4 as u8 && max_tx_size >= TxSize::Tx16x16 {
            raw += r.read(probs[1]);
            if raw != TxSize::Tx8x8 as u8 && max_tx_size >= TxSize::Tx32x32 {
                raw += r.read(probs[2]);
            }
        }
        let tx = match raw {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            3 => TxSize::Tx32x32,
            // Unreachable: read_selected_tx_size accumulates at most 3
            // single-bit increments starting from 0.
            _ => unreachable!("tx_size tree walk produced {raw}"),
        };
        if let Some(c) = counts {
            let idx = tx as usize;
            match max_tx_size {
                TxSize::Tx8x8 => c.tx.p8x8[ctx][idx] += 1,
                TxSize::Tx16x16 => c.tx.p16x16[ctx][idx] += 1,
                TxSize::Tx32x32 => c.tx.p32x32[ctx][idx] += 1,
                // Unreachable: the outer gate excludes Tx4x4.
                TxSize::Tx4x4 => unreachable!("read_selected_tx_size gate excludes Tx4x4"),
            }
        }
        tx
    } else {
        // tx_mode-driven cap. Libvpx does NOT increment tx_counts in
        // this branch.
        let _ = counts;
        let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
        let raw = core::cmp::min(max_tx_size as u8, cap);
        match raw {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            3 => TxSize::Tx32x32,
            // Unreachable: both inputs are TxSize-derived (0..=3).
            _ => unreachable!("min of TxSize values out of range: {raw}"),
        }
    }
}

/// Write a per-block transform size to the boolean writer. Encoder
/// mirror of [`read_tx_size`].
///
/// Walks the same selected-tx-size tree the reader walks against
/// `tx_probs[max_tx_size][ctx]`, emitting the bit-sequence that
/// `read_tx_size` will decode back to `tx_size`:
///
/// | tx_size  | bits at max=Tx8x8 | max=Tx16x16  | max=Tx32x32   |
/// |----------|-------------------|--------------|---------------|
/// | Tx4x4    | `0`               | `0`          | `0`           |
/// | Tx8x8    | `1`               | `1, 0`       | `1, 0`        |
/// | Tx16x16  | —                 | `1, 1`       | `1, 1, 0`     |
/// | Tx32x32  | —                 | —            | `1, 1, 1`     |
///
/// In the non-select path (`!allow_select || tx_mode !=
/// TxModeSelect`), no bits are emitted: the decoder derives
/// `tx_size = min(max_tx_size, tx_mode_to_biggest_tx_size(tx_mode))`
/// directly. This function `debug_assert!`s that the caller-supplied
/// `tx_size` matches that derived value — surfacing encoder mode-
/// decision bugs early without bloating release builds.
///
/// Parameters mirror [`read_tx_size`] 1:1 (minus the `counts` slot,
/// which is reader-only state for backward adaptation).
///
/// # Panics
///
/// In debug builds:
/// - if the algorithm tries to emit a selected tx_size with
///   `max_tx_size == Tx4x4` (the gate that drops to the non-select
///   path),
/// - if `tx_size > max_tx_size` in either path (the value is
///   un-encodable),
/// - if the non-select path receives a `tx_size` that doesn't match
///   the `min(max_tx_size, cap)` the decoder will derive (encoder
///   mode-decision inconsistency).
#[inline]
pub fn write_tx_size(
    bw: &mut BoolWriter<'_>,
    tx_probs: &DefaultTxProbs,
    tx_mode: TxMode,
    max_tx_size: TxSize,
    allow_select: bool,
    ctx: usize,
    tx_size: TxSize,
) {
    debug_assert!(
        (tx_size as u8) <= (max_tx_size as u8),
        "write_tx_size: tx_size {tx_size:?} exceeds max_tx_size {max_tx_size:?}"
    );
    if allow_select && tx_mode == TxMode::TxModeSelect && max_tx_size >= TxSize::Tx8x8 {
        let probs = get_tx_probs(tx_probs, max_tx_size, ctx)
            .expect("write_selected_tx_size requires max_tx_size >= Tx8x8");
        // Tree walk mirrors `read_tx_size`. We emit bits in the same
        // order the reader reads them — first bit branches at Tx4x4,
        // second at Tx8x8 (only when max >= Tx16x16), third at Tx16x16
        // (only when max >= Tx32x32).
        match tx_size {
            TxSize::Tx4x4 => {
                bw.write(0, probs[0]);
            }
            TxSize::Tx8x8 => {
                bw.write(1, probs[0]);
                if max_tx_size >= TxSize::Tx16x16 {
                    bw.write(0, probs[1]);
                }
            }
            TxSize::Tx16x16 => {
                // max must be >= Tx16x16 for this leaf to be reachable;
                // the assert at the top of the function enforces it.
                bw.write(1, probs[0]);
                bw.write(1, probs[1]);
                if max_tx_size >= TxSize::Tx32x32 {
                    bw.write(0, probs[2]);
                }
            }
            TxSize::Tx32x32 => {
                debug_assert_eq!(
                    max_tx_size,
                    TxSize::Tx32x32,
                    "Tx32x32 is only reachable when max_tx_size == Tx32x32"
                );
                bw.write(1, probs[0]);
                bw.write(1, probs[1]);
                bw.write(1, probs[2]);
            }
        }
    } else {
        // Non-select path emits no bits; debug-assert the caller's
        // `tx_size` matches the decoder-derived value.
        let cap = tx_mode_to_biggest_tx_size(tx_mode) as u8;
        let expected = core::cmp::min(max_tx_size as u8, cap);
        debug_assert_eq!(
            tx_size as u8, expected,
            "write_tx_size non-select path: tx_size {tx_size:?} disagrees with min(max_tx_size={max_tx_size:?}, tx_mode_cap={cap})"
        );
        let _ = (bw, tx_probs, ctx);
    }
}

/// VP9 inter-prediction mode. Mirrors libvpx's
/// `NEARESTMV..=NEWMV` (`vp9_enums.h:120-123`); numeric values
/// match libvpx (`10..=13`) so libvpx-pinned tables (e.g.
/// `vp9_inter_mode_tree`) can index by `as u8 - 10`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum InterMode {
    /// `NEARESTMV` — copy the nearest decoded MV (zero residual).
    NearestMv = 10,
    /// `NEARMV` — copy the second-nearest decoded MV.
    NearMv = 11,
    /// `ZEROMV` — motion vector is `(0, 0)`.
    ZeroMv = 12,
    /// `NEWMV` — explicit MV difference encoded in the bitstream.
    NewMv = 13,
}

/// Number of inter modes (= `INTER_MODES` in libvpx).
pub const NUM_INTER_MODES: usize = 4;

impl InterMode {
    /// Zero-based index `[0..=3]` used to index per-context
    /// probability tables (which omit the leading `NEARESTMV` =
    /// libvpx convention).
    #[inline]
    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize - InterMode::NearestMv as usize
    }

    /// Reverse of [`idx`](Self::idx): build from a `0..=3` index.
    /// Returns `None` for out-of-range input.
    #[inline]
    #[must_use]
    pub const fn from_idx(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Self::NearestMv),
            1 => Some(Self::NearMv),
            2 => Some(Self::ZeroMv),
            3 => Some(Self::NewMv),
            _ => None,
        }
    }
}

/// Motion-vector reference frame. Mirrors libvpx
/// `MV_REFERENCE_FRAME` (`vp9_blockd.h:57-65`).
///
/// Numeric values match libvpx exactly. `NoRefFrame` covers libvpx
/// `NO_REF_FRAME = -1`; the three real reference frames are
/// `LastFrame / GoldenFrame / AltrefFrame` (1, 2, 3). Intra-frame
/// blocks use `IntraFrame = 0`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(i8)]
pub enum RefFrame {
    /// `NO_REF_FRAME = -1` — unused slot (e.g. single-ref block's
    /// `ref_frames[1]`).
    NoRefFrame = -1,
    /// `INTRA_FRAME = 0` — intra-coded block.
    IntraFrame = 0,
    /// `LAST_FRAME = 1` — most-recently-decoded frame.
    LastFrame = 1,
    /// `GOLDEN_FRAME = 2` — periodically-refreshed reference.
    GoldenFrame = 2,
    /// `ALTREF_FRAME = 3` — alternate-reference frame.
    AltrefFrame = 3,
}

impl RefFrame {
    /// Build a `RefFrame` from a raw signed byte. Returns `None`
    /// for values outside the legal `-1..=3` range.
    #[inline]
    #[must_use]
    pub const fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Self::NoRefFrame),
            0 => Some(Self::IntraFrame),
            1 => Some(Self::LastFrame),
            2 => Some(Self::GoldenFrame),
            3 => Some(Self::AltrefFrame),
            _ => None,
        }
    }

    /// `true` for any reference frame that isn't `IntraFrame` or
    /// `NoRefFrame` (i.e. a real inter reference).
    #[inline]
    #[must_use]
    pub const fn is_inter(self) -> bool {
        matches!(self, Self::LastFrame | Self::GoldenFrame | Self::AltrefFrame)
    }
}

/// Motion vector at 1/8-pel precision (`Q3` fixed-point). Mirrors
/// libvpx `MV` struct (`vp9_mv.h:24-27`): `row` and `col` are
/// signed offsets in 1/8-pel units.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct Mv {
    /// Vertical component (positive = downward).
    pub row: i16,
    /// Horizontal component (positive = rightward).
    pub col: i16,
}

impl Mv {
    /// The zero MV (`{0, 0}`). Returned by `ZEROMV` blocks.
    pub const ZERO: Self = Self { row: 0, col: 0 };
}

/// Per-4×4 sub-block information for sub-8×8 partitions. Mirrors
/// libvpx `b_mode_info` (`vp9_blockd.h:48-51`).
///
/// Sub-8 (`Block4x4 / Block4x8 / Block8x4`) blocks store their
/// per-4×4 prediction info here; the parent [`ModeInfo`]'s `mode`
/// field is then not used for Y prediction (libvpx uses
/// `mi->bmi[i].as_mode` instead — see `vp9_blockd.h:99-101`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct BmiInfo {
    /// Per-sub-block intra mode (for keyframes / intra blocks) OR
    /// the MV for the sub-block (for inter blocks). The decoder
    /// distinguishes via the parent block's [`BlockMode`].
    pub as_mode: u8,
    /// First / second MV — only used by inter sub-8 blocks.
    pub as_mv: [Mv; 2],
}

/// Block-level prediction mode discriminant. Mirrors the conceptual
/// is-intra-or-inter split that libvpx encodes via
/// `ref_frame[0] == INTRA_FRAME`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockMode {
    /// Intra block: separate Y and UV prediction modes (the
    /// chroma mode is conditional on the luma mode in the
    /// bitstream).
    Intra {
        /// Luma prediction mode.
        y_mode: IntraMode,
        /// Chroma prediction mode.
        uv_mode: IntraMode,
    },
    /// Inter block: prediction mode + references + motion vectors.
    Inter {
        /// Inter-prediction mode (NEARESTMV / NEARMV / ZEROMV /
        /// NEWMV).
        mode: InterMode,
        /// 2 reference frames; `ref_frames[1] = NoRefFrame` for
        /// single-reference blocks.
        ref_frames: [RefFrame; 2],
        /// Motion vectors corresponding to `ref_frames`.
        mvs: [Mv; 2],
        /// Sub-pel interpolation filter (only meaningful when the
        /// frame's `interp_filter == Switchable`).
        interp_filter: InterpFilter,
    },
}

impl BlockMode {
    /// `true` iff this is an intra block.
    #[inline]
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self, Self::Intra { .. })
    }

    /// `true` iff this is an inter block.
    #[inline]
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        matches!(self, Self::Inter { .. })
    }
}

/// Per-8×8-block descriptor produced by the per-block decoder
/// (Task 8.2+). Mirrors libvpx `MODE_INFO`
/// (`vp9_blockd.h:74-97`).
///
/// One `ModeInfo` covers an aligned 8×8 luma region; sub-8
/// partitions still get a single `ModeInfo` per parent 8×8, with
/// the per-4×4 sub-block details in `bmi[4]`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModeInfo {
    /// libvpx `sb_type` — the partition-leaf block size that this
    /// `ModeInfo` represents.
    pub block_size: BlockSize,
    /// Luma transform size for this block.
    pub tx_size: TxSize,
    /// `skip` flag — block has no encoded residual.
    pub skip: bool,
    /// Segmentation index (`0..=MAX_SEGMENTS = 8`).
    pub segment_id: u8,
    /// `seg_id_predicted` — only meaningful when the segmentation
    /// header has `temporal_update == true`.
    pub seg_id_predicted: bool,
    /// Block-level prediction mode (carries the intra/inter split).
    pub mode: BlockMode,
    /// Per-4×4 sub-block info for sub-8 partitions
    /// (`Block4x4 / Block4x8 / Block8x4`). Untouched for ≥8×8
    /// partitions (libvpx leaves it zeroed in that case).
    pub bmi: [BmiInfo; 4],
}

impl ModeInfo {
    /// Convenience: `true` iff this is an intra block.
    #[inline]
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        self.mode.is_intra()
    }

    /// Convenience: `true` iff this is an inter block.
    #[inline]
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        self.mode.is_inter()
    }

    /// libvpx `get_y_mode` (`vp9_blockd.h:99-101`). For sub-8
    /// blocks, returns the per-`bmi[block]` intra mode (parsed
    /// as a `u8` and converted back to [`IntraMode`]). For ≥8×8
    /// intra blocks, returns the parent's `y_mode`. For inter
    /// blocks, returns `None`.
    #[inline]
    #[must_use]
    pub fn y_mode_for_subblock(&self, block: usize) -> Option<IntraMode> {
        match &self.mode {
            BlockMode::Intra { y_mode, .. } => {
                if matches!(
                    self.block_size,
                    BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
                ) {
                    IntraMode::from_u8(self.bmi[block].as_mode)
                } else {
                    Some(*y_mode)
                }
            }
            BlockMode::Inter { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `InterMode` numeric values match libvpx.
    #[test]
    fn inter_mode_values() {
        assert_eq!(InterMode::NearestMv as u8, 10);
        assert_eq!(InterMode::NearMv as u8, 11);
        assert_eq!(InterMode::ZeroMv as u8, 12);
        assert_eq!(InterMode::NewMv as u8, 13);
        assert_eq!(NUM_INTER_MODES, 4);
    }

    /// `InterMode::idx` is the zero-based offset libvpx uses for
    /// `[INTER_MODES - 1]` prob tables.
    #[test]
    fn inter_mode_idx() {
        assert_eq!(InterMode::NearestMv.idx(), 0);
        assert_eq!(InterMode::NearMv.idx(), 1);
        assert_eq!(InterMode::ZeroMv.idx(), 2);
        assert_eq!(InterMode::NewMv.idx(), 3);
        // Round-trip.
        for v in 0..=3u8 {
            let m = InterMode::from_idx(v).expect("legal");
            assert_eq!(m.idx() as u8, v);
        }
        assert!(InterMode::from_idx(4).is_none());
    }

    /// `RefFrame` numeric values match libvpx exactly.
    #[test]
    fn ref_frame_values() {
        assert_eq!(RefFrame::NoRefFrame as i8, -1);
        assert_eq!(RefFrame::IntraFrame as i8, 0);
        assert_eq!(RefFrame::LastFrame as i8, 1);
        assert_eq!(RefFrame::GoldenFrame as i8, 2);
        assert_eq!(RefFrame::AltrefFrame as i8, 3);
    }

    /// `RefFrame::from_i8` round-trips legal values and rejects
    /// out-of-range.
    #[test]
    fn ref_frame_from_i8() {
        for v in -1..=3i8 {
            let r = RefFrame::from_i8(v).expect("legal");
            assert_eq!(r as i8, v);
        }
        assert!(RefFrame::from_i8(4).is_none());
        assert!(RefFrame::from_i8(-2).is_none());
    }

    /// `RefFrame::is_inter` returns true only for real inter refs.
    #[test]
    fn ref_frame_is_inter() {
        assert!(!RefFrame::NoRefFrame.is_inter());
        assert!(!RefFrame::IntraFrame.is_inter());
        assert!(RefFrame::LastFrame.is_inter());
        assert!(RefFrame::GoldenFrame.is_inter());
        assert!(RefFrame::AltrefFrame.is_inter());
    }

    /// `Mv::ZERO` is `{0, 0}`.
    #[test]
    fn mv_zero() {
        assert_eq!(Mv::ZERO, Mv { row: 0, col: 0 });
    }

    /// `BlockMode::is_intra` / `is_inter` discriminator.
    #[test]
    fn block_mode_is_intra_is_inter() {
        let intra = BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::V,
        };
        assert!(intra.is_intra());
        assert!(!intra.is_inter());

        let inter = BlockMode::Inter {
            mode: InterMode::ZeroMv,
            ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
            mvs: [Mv::ZERO, Mv::ZERO],
            interp_filter: InterpFilter::EightTap,
        };
        assert!(!inter.is_intra());
        assert!(inter.is_inter());
    }

    /// `ModeInfo::y_mode_for_subblock`: ≥8×8 intra blocks return
    /// the parent's y_mode for any sub-block index; sub-8 intra
    /// blocks return the per-bmi mode.
    #[test]
    fn y_mode_for_subblock_paths() {
        // ≥8×8 intra: returns parent y_mode.
        let mi_big_intra = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::D45,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        for block in 0..4 {
            assert_eq!(
                mi_big_intra.y_mode_for_subblock(block),
                Some(IntraMode::D45)
            );
        }

        // Sub-8 intra: per-bmi mode.
        let mut bmi: [BmiInfo; 4] = Default::default();
        bmi[0].as_mode = IntraMode::H as u8;
        bmi[1].as_mode = IntraMode::V as u8;
        bmi[2].as_mode = IntraMode::Tm as u8;
        bmi[3].as_mode = IntraMode::Dc as u8;
        let mi_sub8_intra = ModeInfo {
            block_size: BlockSize::Block4x4,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc, // ignored at sub-8
                uv_mode: IntraMode::Dc,
            },
            bmi,
        };
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(0), Some(IntraMode::H));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(1), Some(IntraMode::V));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(2), Some(IntraMode::Tm));
        assert_eq!(mi_sub8_intra.y_mode_for_subblock(3), Some(IntraMode::Dc));

        // Inter: always None.
        let mi_inter = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx8x8,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Inter {
                mode: InterMode::NewMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv { row: 8, col: -4 }, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        for block in 0..4 {
            assert_eq!(mi_inter.y_mode_for_subblock(block), None);
        }
    }

    /// `PARTITION_TREE` is the libvpx-shaped 6-entry table.
    /// Hand-verified against `vp9_entropymode.c:262-264`.
    #[test]
    fn partition_tree_shape() {
        assert_eq!(PARTITION_TREE.len(), 6);
        // node 0: [-NONE, 2]
        assert_eq!(PARTITION_TREE[0], -(PartitionType::None as i8));
        assert_eq!(PARTITION_TREE[1], 2);
        // node 2: [-HORZ, 4]
        assert_eq!(PARTITION_TREE[2], -(PartitionType::Horz as i8));
        assert_eq!(PARTITION_TREE[3], 4);
        // node 4: [-VERT, -SPLIT]
        assert_eq!(PARTITION_TREE[4], -(PartitionType::Vert as i8));
        assert_eq!(PARTITION_TREE[5], -(PartitionType::Split as i8));
    }

    /// `SEGMENT_TREE` shape: 14 entries, structured as a balanced
    /// binary tree over 8 segment values.
    #[test]
    fn segment_tree_shape() {
        assert_eq!(SEGMENT_TREE.len(), 14);
        // Internal nodes
        assert_eq!(SEGMENT_TREE[0], 2);
        assert_eq!(SEGMENT_TREE[1], 4);
        assert_eq!(SEGMENT_TREE[2], 6);
        assert_eq!(SEGMENT_TREE[3], 8);
        assert_eq!(SEGMENT_TREE[4], 10);
        assert_eq!(SEGMENT_TREE[5], 12);
        // Leaves (node 6: [0, -1], node 8: [-2, -3], etc.)
        assert_eq!(SEGMENT_TREE[6], 0);
        assert_eq!(SEGMENT_TREE[7], -1);
        assert_eq!(SEGMENT_TREE[13], -7);
    }

    /// `read_segment_id` round-trips every segment value 0..=7 at
    /// p=128.
    #[test]
    fn read_segment_id_round_trip() {
        use crate::entropy::BoolWriter;

        let probs = [128u8; SEG_TREE_PROBS];

        for target in 0u8..8 {
            // Encode by walking the tree to the target leaf.
            let mut buf = vec![0u8; 64];
            let mut w = BoolWriter::start_encode(&mut buf);
            let mut i: i32 = 0;
            loop {
                // Determine which child (left=tree[i], right=tree[i+1])
                // contains the target leaf via DFS.
                fn subtree_contains(tree: &[i8], node: i8, target: u8) -> bool {
                    if node <= 0 {
                        (-node) as u8 == target
                    } else {
                        let i = node as usize;
                        subtree_contains(tree, tree[i], target)
                            || subtree_contains(tree, tree[i + 1], target)
                    }
                }
                let left = SEGMENT_TREE[i as usize];
                let right = SEGMENT_TREE[i as usize + 1];
                let bit = if subtree_contains(&SEGMENT_TREE, left, target) { 0 } else { 1 };
                w.write(bit, probs[(i >> 1) as usize]);
                let next = if bit == 0 { left } else { right };
                if next <= 0 {
                    break;
                }
                i = next as i32;
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_segment_id(&mut r, &probs);
            assert_eq!(got, target, "round-trip segment_id={target}");
        }
    }

    /// `read_skip` / `read_is_inter` / `read_is_compound` /
    /// `read_single_ref_bit` are all 1-bit boolean reads. Verify
    /// both possible values round-trip cleanly.
    #[test]
    fn one_bit_leaf_decoders_round_trip() {
        use crate::entropy::BoolWriter;

        fn build(bit: u8, prob: u8) -> Vec<u8> {
            let mut buf = vec![0u8; 32];
            let mut w = BoolWriter::start_encode(&mut buf);
            w.write(bit, prob);
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);
            buf
        }

        // read_skip.
        let skip_probs = [200u8; SKIP_CONTEXTS];
        let buf = build(0, skip_probs[1]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(!read_skip(&mut r, &skip_probs, 1, None));
        let buf = build(1, skip_probs[2]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_skip(&mut r, &skip_probs, 2, None));

        // read_is_inter.
        let ii_probs = [128u8; INTRA_INTER_CONTEXTS];
        let buf = build(0, ii_probs[0]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(!read_is_inter(&mut r, &ii_probs, 0, None));
        let buf = build(1, ii_probs[3]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_is_inter(&mut r, &ii_probs, 3, None));

        // read_is_compound.
        let ci_probs = [100u8; COMP_INTER_CONTEXTS];
        let buf = build(0, ci_probs[2]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(!read_is_compound(&mut r, &ci_probs, 2, None));
        let buf = build(1, ci_probs[4]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_is_compound(&mut r, &ci_probs, 4, None));

        // read_single_ref_bit.
        let sr_probs = [[120u8, 80u8]; REF_CONTEXTS];
        let buf = build(0, sr_probs[1][0]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(!read_single_ref_bit(&mut r, &sr_probs, 1, 0, None));
        let buf = build(1, sr_probs[3][1]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_single_ref_bit(&mut r, &sr_probs, 3, 1, None));
    }

    /// `INTRA_MODE_TREE` shape matches libvpx
    /// (`vp9_entropymode.c:245-255`).
    #[test]
    fn intra_mode_tree_shape() {
        assert_eq!(INTRA_MODE_TREE.len(), 18);
        // node 0 = DC_NODE
        assert_eq!(INTRA_MODE_TREE[0], -(IntraMode::Dc as i8));
        assert_eq!(INTRA_MODE_TREE[1], 2);
        // node 1 = TM_NODE
        assert_eq!(INTRA_MODE_TREE[2], -(IntraMode::Tm as i8));
        assert_eq!(INTRA_MODE_TREE[3], 4);
        // node 5 = D135_NODE (D135 left, D117 right)
        assert_eq!(INTRA_MODE_TREE[10], -(IntraMode::D135 as i8));
        assert_eq!(INTRA_MODE_TREE[11], -(IntraMode::D117 as i8));
        // node 8 = D153_NODE (D153 left, D207 right)
        assert_eq!(INTRA_MODE_TREE[16], -(IntraMode::D153 as i8));
        assert_eq!(INTRA_MODE_TREE[17], -(IntraMode::D207 as i8));
    }

    /// `read_intra_mode` round-trips every IntraMode at p=128.
    /// Verified by encoding the canonical bit path for each mode
    /// and reading it back.
    #[test]
    fn read_intra_mode_round_trip() {
        use crate::entropy::BoolWriter;

        let probs = [128u8; 9];

        // Build a buffer that drives the tree to the target mode
        // by replaying the read_tree algorithm in the writer.
        fn encode_for_mode(target: IntraMode, probs: &[u8; 9]) -> Vec<u8> {
            let mut buf = vec![0u8; 64];
            let mut w = BoolWriter::start_encode(&mut buf);
            let mut i: usize = 0;
            loop {
                let prob = probs[i >> 1];
                let left = INTRA_MODE_TREE[i];
                let right = INTRA_MODE_TREE[i + 1];
                // Choose bit = 0 if left subtree reaches target;
                // else bit = 1.
                let bit = if subtree_contains(left, target) { 0 } else { 1 };
                w.write(bit, prob);
                let next = if bit == 0 { left } else { right };
                if next <= 0 {
                    break;
                }
                i = next as usize;
            }
            // Pad.
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);
            buf
        }

        // Recursive subtree-contains check.
        fn subtree_contains(node: i8, target: IntraMode) -> bool {
            if node <= 0 {
                (-node) as u8 == target as u8
            } else {
                let i = node as usize;
                subtree_contains(INTRA_MODE_TREE[i], target)
                    || subtree_contains(INTRA_MODE_TREE[i + 1], target)
            }
        }

        for target in [
            IntraMode::Dc,
            IntraMode::V,
            IntraMode::H,
            IntraMode::D45,
            IntraMode::D135,
            IntraMode::D117,
            IntraMode::D153,
            IntraMode::D207,
            IntraMode::D63,
            IntraMode::Tm,
        ] {
            let buf = encode_for_mode(target, &probs);
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_intra_mode(&mut r, &probs);
            assert_eq!(got, target, "round-trip {target:?}");
        }
    }

    /// `INTER_MODE_TREE` shape matches libvpx
    /// (`vp9_entropymode.c:257-260`). Leaf values are
    /// `INTER_OFFSET(mode)` (0..=3).
    #[test]
    fn inter_mode_tree_shape() {
        assert_eq!(INTER_MODE_TREE.len(), 6);
        // node 0: [-ZEROMV(idx=2), 2]
        assert_eq!(INTER_MODE_TREE[0], -(InterMode::ZeroMv.idx() as i8));
        assert_eq!(INTER_MODE_TREE[1], 2);
        // node 2: [-NEARESTMV(idx=0), 4]
        assert_eq!(INTER_MODE_TREE[2], -(InterMode::NearestMv.idx() as i8));
        assert_eq!(INTER_MODE_TREE[3], 4);
        // node 4: [-NEARMV(idx=1), -NEWMV(idx=3)]
        assert_eq!(INTER_MODE_TREE[4], -(InterMode::NearMv.idx() as i8));
        assert_eq!(INTER_MODE_TREE[5], -(InterMode::NewMv.idx() as i8));
    }

    /// `read_inter_mode` round-trips every InterMode at p=128.
    /// Tree paths:
    /// - ZEROMV   = `0`
    /// - NEARESTMV = `1 0`
    /// - NEARMV   = `1 1 0`
    /// - NEWMV    = `1 1 1`
    #[test]
    fn read_inter_mode_round_trip() {
        use crate::entropy::BoolWriter;

        let probs = [128u8; 3];

        let cases = [
            (InterMode::ZeroMv, vec![0u8]),
            (InterMode::NearestMv, vec![1, 0]),
            (InterMode::NearMv, vec![1, 1, 0]),
            (InterMode::NewMv, vec![1, 1, 1]),
        ];

        for (expected, bits) in cases {
            let mut buf = vec![0u8; 64];
            let mut w = BoolWriter::start_encode(&mut buf);
            for (i, bit) in bits.iter().enumerate() {
                w.write(*bit, probs[i]);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_inter_mode(&mut r, &probs, 0, None);
            assert_eq!(got, expected, "round-trip {expected:?}");
        }
    }

    /// `read_partition` round-trips every PartitionType for an
    /// arbitrary context. The boolean tree encodes:
    /// - PARTITION_NONE  via bit sequence `0`
    /// - PARTITION_HORZ  via `1 0`
    /// - PARTITION_VERT  via `1 1 0`
    /// - PARTITION_SPLIT via `1 1 1`
    #[test]
    fn read_partition_round_trip() {
        use crate::entropy::BoolWriter;

        // We need a specific probs layout: use p=128 (max-entropy)
        // for every node so the encoder bits encode directly.
        let probs: [u8; 3] = [128, 128, 128];

        // Build, for each PartitionType, a buffer with the
        // appropriate bit sequence; verify the decoder returns
        // the expected value.
        let cases = [
            (PartitionType::None, vec![(0u8, probs[0])]),
            (PartitionType::Horz, vec![(1, probs[0]), (0, probs[1])]),
            (PartitionType::Vert, vec![(1, probs[0]), (1, probs[1]), (0, probs[2])]),
            (PartitionType::Split, vec![(1, probs[0]), (1, probs[1]), (1, probs[2])]),
        ];

        for (expected, bits) in cases {
            let mut buf = vec![0u8; 64];
            let mut w = BoolWriter::start_encode(&mut buf);
            for (bit, prob) in &bits {
                w.write(*bit, *prob);
            }
            // Pad.
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_partition(&mut r, &probs, 0, None);
            assert_eq!(got, expected, "round-trip {expected:?}");
        }
    }

    /// Stage 3 Task E.1.c.2 / .3: leaf decoders increment the
    /// correct [`FrameCounts`] slot when fed a non-`None` sink.
    /// Verifies plumbing for: read_skip, read_is_inter, read_is_compound,
    /// read_single_ref_bit, read_partition, read_intra_mode_y,
    /// read_intra_mode_uv, read_inter_mode, read_tx_size (selected path).
    #[test]
    fn leaf_decoders_populate_frame_counts() {
        use crate::entropy::BoolWriter;
        use crate::entropy::adapt::FrameCounts;

        fn build_one_bit(bit: u8, prob: u8) -> Vec<u8> {
            let mut buf = vec![0u8; 32];
            let mut w = BoolWriter::start_encode(&mut buf);
            w.write(bit, prob);
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);
            buf
        }

        // read_skip: ctx=1, bit=1 → counts.skip[1][1] += 1.
        let mut counts = FrameCounts::default();
        let skip_probs = [200u8; SKIP_CONTEXTS];
        let buf = build_one_bit(1, skip_probs[1]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_skip(&mut r, &skip_probs, 1, Some(&mut counts)));
        assert_eq!(counts.skip[1][1], 1);
        assert_eq!(counts.skip[1][0], 0);

        // read_is_inter: ctx=2, bit=0 → counts.intra_inter[2][0] += 1.
        let mut counts = FrameCounts::default();
        let ii_probs = [128u8; INTRA_INTER_CONTEXTS];
        let buf = build_one_bit(0, ii_probs[2]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(!read_is_inter(&mut r, &ii_probs, 2, Some(&mut counts)));
        assert_eq!(counts.intra_inter[2][0], 1);

        // read_is_compound: ctx=3, bit=1 → counts.comp_inter[3][1] += 1.
        let mut counts = FrameCounts::default();
        let ci_probs = [100u8; COMP_INTER_CONTEXTS];
        let buf = build_one_bit(1, ci_probs[3]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_is_compound(&mut r, &ci_probs, 3, Some(&mut counts)));
        assert_eq!(counts.comp_inter[3][1], 1);

        // read_single_ref_bit: ctx=4, slot=1, bit=1 →
        // counts.single_ref[4][1][1] += 1.
        let mut counts = FrameCounts::default();
        let sr_probs = [[120u8, 80u8]; REF_CONTEXTS];
        let buf = build_one_bit(1, sr_probs[4][1]);
        let mut r = BoolReader::new(&buf).expect("init");
        assert!(read_single_ref_bit(&mut r, &sr_probs, 4, 1, Some(&mut counts)));
        assert_eq!(counts.single_ref[4][1][1], 1);

        // read_partition: ctx=7, encoded as PartitionType::Horz (`1 0`).
        let mut counts = FrameCounts::default();
        let probs = [128u8, 128u8, 128u8];
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, probs[0]);
        w.write(0, probs[1]);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let p = read_partition(&mut r, &probs, 7, Some(&mut counts));
        assert_eq!(p, PartitionType::Horz);
        assert_eq!(counts.partition[7][PartitionType::Horz as usize], 1);

        // read_intra_mode_y / read_intra_mode_uv: encode the
        // canonical bit path for IntraMode::Dc (left-most leaf:
        // bit 0 at node 0).
        let mut counts = FrameCounts::default();
        let probs = [128u8; 9];
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(0, probs[0]);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let m = read_intra_mode_y(&mut r, &probs, /* size_group = */ 2, Some(&mut counts));
        assert_eq!(m, IntraMode::Dc);
        assert_eq!(counts.y_mode[2][IntraMode::Dc as usize], 1);

        let mut counts = FrameCounts::default();
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(0, probs[0]);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let m =
            read_intra_mode_uv(&mut r, &probs, /* y_mode = */ IntraMode::H, Some(&mut counts));
        assert_eq!(m, IntraMode::Dc);
        assert_eq!(
            counts.uv_mode[IntraMode::H as usize][IntraMode::Dc as usize],
            1
        );

        // read_inter_mode: encode InterMode::ZeroMv (single bit 0 at
        // node 0 of INTER_MODE_TREE).
        let mut counts = FrameCounts::default();
        let probs = [128u8; NUM_INTER_MODES - 1];
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(0, probs[0]);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let m = read_inter_mode(&mut r, &probs, /* ctx = */ 5, Some(&mut counts));
        assert_eq!(m, InterMode::ZeroMv);
        assert_eq!(counts.inter_mode[5][InterMode::ZeroMv.idx()], 1);

        // read_tx_size (selected path, max=Tx16x16): encode 1 0 →
        // Tx8x8 → counts.tx.p16x16[ctx][Tx8x8 as usize] += 1.
        let fc = crate::entropy::FrameContext::new_default();
        let mut counts = FrameCounts::default();
        let p = &fc.tx_probs.p16x16[1];
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, p[0]);
        w.write(0, p[1]);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let got = read_tx_size(
            &mut r,
            &fc.tx_probs,
            TxMode::TxModeSelect,
            TxSize::Tx16x16,
            true,
            1,
            Some(&mut counts),
        );
        assert_eq!(got, TxSize::Tx8x8);
        assert_eq!(counts.tx.p16x16[1][TxSize::Tx8x8 as usize], 1);
    }

    /// `ModeInfo::is_intra` / `is_inter` delegates to `BlockMode`.
    #[test]
    fn mode_info_is_intra_is_inter() {
        let intra_mi = ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx8x8,
            skip: true,
            segment_id: 3,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        assert!(intra_mi.is_intra());
        assert!(!intra_mi.is_inter());

        let inter_mi = ModeInfo {
            block_size: BlockSize::Block16x16,
            tx_size: TxSize::Tx16x16,
            skip: false,
            segment_id: 0,
            seg_id_predicted: true,
            mode: BlockMode::Inter {
                mode: InterMode::NearestMv,
                ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                mvs: [Mv::ZERO, Mv::ZERO],
                interp_filter: InterpFilter::EightTap,
            },
            bmi: Default::default(),
        };
        assert!(!inter_mi.is_intra());
        assert!(inter_mi.is_inter());
    }

    /// `get_tx_size_context`: with both above + left present and
    /// not skipped, the result is `(above.tx_size + left.tx_size) >
    /// max_tx_size`.
    #[test]
    fn get_tx_size_context_both_present() {
        // max = Tx32x32 (3). Sum = 0 + 0 = 0 < 3 → ctx 0.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx4x4, false)),
                Some((TxSize::Tx4x4, false)),
                TxSize::Tx32x32,
            ),
            0
        );
        // Sum = 2 + 2 = 4 > 3 → ctx 1.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx16x16, false)),
                Some((TxSize::Tx16x16, false)),
                TxSize::Tx32x32,
            ),
            1
        );
        // Sum = 1 + 2 = 3 == max → NOT greater → ctx 0.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx8x8, false)),
                Some((TxSize::Tx16x16, false)),
                TxSize::Tx32x32,
            ),
            0
        );
        // Sum = 2 + 2 = 4 > max=2 → ctx 1.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx16x16, false)),
                Some((TxSize::Tx16x16, false)),
                TxSize::Tx16x16,
            ),
            1
        );
    }

    /// `get_tx_size_context`: skipped neighbours contribute
    /// `max_tx_size` instead of their actual tx_size.
    #[test]
    fn get_tx_size_context_skipped_neighbour() {
        // Above is Tx4x4 but skipped → treated as max_tx_size=3.
        // Left is Tx4x4 unskipped → contributes 0. Sum = 3 + 0 = 3
        // = max → ctx 0.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx4x4, true)),
                Some((TxSize::Tx4x4, false)),
                TxSize::Tx32x32,
            ),
            0
        );
        // Both skipped → both contribute max=3. Sum=6 > 3 → ctx 1.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx4x4, true)),
                Some((TxSize::Tx4x4, true)),
                TxSize::Tx32x32,
            ),
            1
        );
    }

    /// `get_tx_size_context`: a missing neighbour mirrors the
    /// present one.
    #[test]
    fn get_tx_size_context_missing_neighbour() {
        // Left missing → mirrors above. Sum = above + above = 2*0
        // = 0 < 3 → ctx 0.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx4x4, false)),
                None,
                TxSize::Tx32x32,
            ),
            0
        );
        // Left missing, above = Tx32x32. Sum = 3 + 3 = 6 > 3 → ctx 1.
        assert_eq!(
            get_tx_size_context(
                Some((TxSize::Tx32x32, false)),
                None,
                TxSize::Tx32x32,
            ),
            1
        );
        // Above missing → mirrors left.
        assert_eq!(
            get_tx_size_context(
                None,
                Some((TxSize::Tx32x32, false)),
                TxSize::Tx32x32,
            ),
            1
        );
    }

    /// `get_tx_size_context`: both neighbours missing → both
    /// collapse to `max_tx_size`. Result is `2*max > max`, always
    /// `1` for `max_tx_size > 0`.
    #[test]
    fn get_tx_size_context_no_neighbours() {
        assert_eq!(
            get_tx_size_context(None, None, TxSize::Tx32x32),
            1
        );
        assert_eq!(
            get_tx_size_context(None, None, TxSize::Tx16x16),
            1
        );
        assert_eq!(
            get_tx_size_context(None, None, TxSize::Tx8x8),
            1
        );
    }

    /// `read_tx_size` non-select path: returns `min(max_tx_size,
    /// tx_mode_to_biggest)` without consuming any bool reader bits.
    #[test]
    fn read_tx_size_capped_path() {
        let fc = crate::entropy::FrameContext::new_default();
        // We need a BoolReader but the capped path consumes 0 bits.
        // Build a tiny dummy stream.
        let buf = [0u8; 4];
        let cases = [
            // (tx_mode, max_tx_size, expected)
            (TxMode::OnlyTx4x4, TxSize::Tx32x32, TxSize::Tx4x4),
            (TxMode::OnlyTx4x4, TxSize::Tx4x4, TxSize::Tx4x4),
            (TxMode::AllowTx8x8, TxSize::Tx32x32, TxSize::Tx8x8),
            (TxMode::AllowTx8x8, TxSize::Tx4x4, TxSize::Tx4x4),
            (TxMode::AllowTx16x16, TxSize::Tx8x8, TxSize::Tx8x8),
            (TxMode::AllowTx16x16, TxSize::Tx32x32, TxSize::Tx16x16),
            (TxMode::AllowTx32x32, TxSize::Tx16x16, TxSize::Tx16x16),
            (TxMode::AllowTx32x32, TxSize::Tx32x32, TxSize::Tx32x32),
            // TxModeSelect but max_tx_size < Tx8x8 → still capped.
            (TxMode::TxModeSelect, TxSize::Tx4x4, TxSize::Tx4x4),
        ];
        for (tx_mode, max_tx, expected) in cases {
            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_tx_size(&mut r, &fc.tx_probs, tx_mode, max_tx, true, 0, None);
            assert_eq!(got, expected, "tx_mode={tx_mode:?} max={max_tx:?}");
        }
    }

    /// `read_tx_size`: `allow_select == false` forces the capped
    /// path even when `tx_mode == TxModeSelect`.
    #[test]
    fn read_tx_size_allow_select_false_caps() {
        let fc = crate::entropy::FrameContext::new_default();
        let buf = [0u8; 4];
        let mut r = BoolReader::new(&buf).expect("init");
        // Capped: min(Tx32x32, biggest(TxModeSelect)=3) = Tx32x32.
        let got = read_tx_size(
            &mut r,
            &fc.tx_probs,
            TxMode::TxModeSelect,
            TxSize::Tx32x32,
            false,
            0,
            None,
        );
        assert_eq!(got, TxSize::Tx32x32);
    }

    /// `read_tx_size` selected path: at `max_tx_size = Tx8x8`,
    /// a single bit chooses Tx4x4 (bit=0) or Tx8x8 (bit=1) against
    /// `tx_probs.p8x8[ctx][0]`.
    #[test]
    fn read_tx_size_selected_tx8x8_round_trip() {
        use crate::entropy::BoolWriter;

        let fc = crate::entropy::FrameContext::new_default();

        for (bit, expected) in [(0u8, TxSize::Tx4x4), (1u8, TxSize::Tx8x8)] {
            for ctx in 0..2 {
                let mut buf = vec![0u8; 32];
                let mut w = BoolWriter::start_encode(&mut buf);
                w.write(bit, fc.tx_probs.p8x8[ctx][0]);
                for _ in 0..16 {
                    w.write_bit(0);
                }
                let size = w.stop_encode().expect("ok");
                buf.truncate(size);

                let mut r = BoolReader::new(&buf).expect("init");
                let got = read_tx_size(
                    &mut r,
                    &fc.tx_probs,
                    TxMode::TxModeSelect,
                    TxSize::Tx8x8,
                    true,
                    ctx,
                    None,
                );
                assert_eq!(got, expected, "bit={bit} ctx={ctx}");
            }
        }
    }

    /// `read_tx_size` selected path at `max_tx_size = Tx16x16`:
    /// 2-bit tree.
    /// - bit0=0 → Tx4x4 (1 bit consumed).
    /// - bit0=1, bit1=0 → Tx8x8 (2 bits consumed).
    /// - bit0=1, bit1=1 → Tx16x16 (2 bits consumed).
    #[test]
    fn read_tx_size_selected_tx16x16_round_trip() {
        use crate::entropy::BoolWriter;

        let fc = crate::entropy::FrameContext::new_default();
        let ctx = 1usize;
        let p = &fc.tx_probs.p16x16[ctx];

        let cases: &[(&[(u8, u8)], TxSize)] = &[
            (&[(0, p[0])], TxSize::Tx4x4),
            (&[(1, p[0]), (0, p[1])], TxSize::Tx8x8),
            (&[(1, p[0]), (1, p[1])], TxSize::Tx16x16),
        ];

        for (bits, expected) in cases {
            let mut buf = vec![0u8; 32];
            let mut w = BoolWriter::start_encode(&mut buf);
            for (bit, prob) in *bits {
                w.write(*bit, *prob);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_tx_size(
                &mut r,
                &fc.tx_probs,
                TxMode::TxModeSelect,
                TxSize::Tx16x16,
                true,
                ctx,
                None,
            );
            assert_eq!(got, *expected, "bits={bits:?}");
        }
    }

    /// `read_tx_size` selected path at `max_tx_size = Tx32x32`:
    /// full 3-bit tree.
    /// - bit0=0 → Tx4x4 (1 bit).
    /// - bit0=1, bit1=0 → Tx8x8 (2 bits).
    /// - bit0=1, bit1=1, bit2=0 → Tx16x16 (3 bits).
    /// - bit0=1, bit1=1, bit2=1 → Tx32x32 (3 bits).
    #[test]
    fn read_tx_size_selected_tx32x32_round_trip() {
        use crate::entropy::BoolWriter;

        let fc = crate::entropy::FrameContext::new_default();
        let ctx = 0usize;
        let p = &fc.tx_probs.p32x32[ctx];

        let cases: &[(&[(u8, u8)], TxSize)] = &[
            (&[(0, p[0])], TxSize::Tx4x4),
            (&[(1, p[0]), (0, p[1])], TxSize::Tx8x8),
            (&[(1, p[0]), (1, p[1]), (0, p[2])], TxSize::Tx16x16),
            (&[(1, p[0]), (1, p[1]), (1, p[2])], TxSize::Tx32x32),
        ];

        for (bits, expected) in cases {
            let mut buf = vec![0u8; 32];
            let mut w = BoolWriter::start_encode(&mut buf);
            for (bit, prob) in *bits {
                w.write(*bit, *prob);
            }
            for _ in 0..16 {
                w.write_bit(0);
            }
            let size = w.stop_encode().expect("ok");
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("init");
            let got = read_tx_size(
                &mut r,
                &fc.tx_probs,
                TxMode::TxModeSelect,
                TxSize::Tx32x32,
                true,
                ctx,
                None,
            );
            assert_eq!(got, *expected, "bits={bits:?}");
        }
    }

    /// `read_tx_size` selected path exercises both contexts. We
    /// only need to verify the prob slot picked differs by `ctx`;
    /// using different per-context bits forces the test to fail
    /// if the function used the wrong row.
    #[test]
    fn read_tx_size_selected_uses_correct_context() {
        use crate::entropy::BoolWriter;

        // Build a custom tx_probs where ctx 0 forces Tx4x4 (prob
        // high → near-deterministic 0-bit), ctx 1 forces Tx8x8
        // (prob low → near-deterministic 1-bit).
        // Using p=1 (almost always 0) and p=255 (almost always 1)
        // makes raw-bit encoding deterministic.
        let custom_tx = DefaultTxProbs {
            p32x32: [[128; 3]; 2],
            p16x16: [[128; 2]; 2],
            p8x8: [[255], [1]], // ctx 0: prob=255 → P(bit=0)≈1; ctx 1: prob=1 → P(bit=0)≈0.
        };

        // ctx 0: encode bit=0 at p=255.
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(0, 255);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let got =
            read_tx_size(&mut r, &custom_tx, TxMode::TxModeSelect, TxSize::Tx8x8, true, 0, None);
        assert_eq!(got, TxSize::Tx4x4, "ctx=0 should pick Tx4x4");

        // ctx 1: encode bit=1 at p=1.
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, 1);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        let mut r = BoolReader::new(&buf).expect("init");
        let got =
            read_tx_size(&mut r, &custom_tx, TxMode::TxModeSelect, TxSize::Tx8x8, true, 1, None);
        assert_eq!(got, TxSize::Tx8x8, "ctx=1 should pick Tx8x8");
    }

    /// `write_tx_size` → `read_tx_size` round-trip at every legal
    /// (max_tx_size, tx_size, ctx) triple in the selected-tx path.
    /// Covers all 9 reachable leaves (`Tx4x4/Tx8x8 @ max=Tx8x8`,
    /// `+Tx16x16 @ max=Tx16x16`, `+Tx32x32 @ max=Tx32x32`) × 2 ctx
    /// values.
    #[test]
    fn write_tx_size_round_trips_selected_path_all_combinations() {
        let fc = crate::entropy::FrameContext::new_default();

        let cases: &[(TxSize, &[TxSize])] = &[
            (TxSize::Tx8x8, &[TxSize::Tx4x4, TxSize::Tx8x8]),
            (TxSize::Tx16x16, &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16]),
            (
                TxSize::Tx32x32,
                &[TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16, TxSize::Tx32x32],
            ),
        ];

        for (max_tx, allowed_tx) in cases {
            for &tx_size in *allowed_tx {
                for ctx in 0..2 {
                    let mut buf = vec![0u8; 32];
                    let size = {
                        let mut bw = BoolWriter::start_encode(&mut buf);
                        write_tx_size(
                            &mut bw,
                            &fc.tx_probs,
                            TxMode::TxModeSelect,
                            *max_tx,
                            true,
                            ctx,
                            tx_size,
                        );
                        // Pad so the BoolReader can flush its window.
                        for _ in 0..16 {
                            bw.write_bit(0);
                        }
                        bw.stop_encode().expect("stop_encode")
                    };
                    buf.truncate(size);

                    let mut r = BoolReader::new(&buf).expect("BoolReader::new");
                    let got = read_tx_size(
                        &mut r,
                        &fc.tx_probs,
                        TxMode::TxModeSelect,
                        *max_tx,
                        true,
                        ctx,
                        None,
                    );
                    assert_eq!(
                        got, tx_size,
                        "round-trip failed for max={max_tx:?} tx_size={tx_size:?} ctx={ctx}"
                    );
                }
            }
        }
    }

    /// Non-select path: `write_tx_size` must emit zero bits. We
    /// verify by writing a sentinel `1` immediately after; if any
    /// bits were emitted before it, the sentinel would decode as
    /// `0`.
    #[test]
    fn write_tx_size_non_select_path_emits_no_bits() {
        let fc = crate::entropy::FrameContext::new_default();

        // Exercise both the non-select tx_mode and the
        // `allow_select == false` gate. Tx32x32 cap → derived tx_size
        // is min(Tx32x32, Tx32x32) = Tx32x32.
        let non_select_cases: &[(TxMode, bool, TxSize, TxSize)] = &[
            (TxMode::AllowTx32x32, true, TxSize::Tx32x32, TxSize::Tx32x32),
            // allow_select=false forces the cap path even with TxModeSelect.
            (TxMode::TxModeSelect, false, TxSize::Tx32x32, TxSize::Tx32x32),
            // AllowTx16x16 caps tx_size at Tx16x16 regardless of max.
            (TxMode::AllowTx16x16, true, TxSize::Tx32x32, TxSize::Tx16x16),
            // OnlyTx4x4 caps to Tx4x4.
            (TxMode::OnlyTx4x4, true, TxSize::Tx8x8, TxSize::Tx4x4),
        ];

        for (tx_mode, allow_select, max_tx, expected_tx_size) in non_select_cases {
            let mut buf = vec![0u8; 32];
            let size = {
                let mut bw = BoolWriter::start_encode(&mut buf);
                write_tx_size(
                    &mut bw,
                    &fc.tx_probs,
                    *tx_mode,
                    *max_tx,
                    *allow_select,
                    0,
                    *expected_tx_size,
                );
                // Sentinel `1` at p=128; if write_tx_size emitted bits
                // they'd shift the cursor and the sentinel would
                // decode as `0`.
                bw.write(1, 128);
                for _ in 0..16 {
                    bw.write_bit(0);
                }
                bw.stop_encode().expect("stop_encode")
            };
            buf.truncate(size);

            let mut r = BoolReader::new(&buf).expect("BoolReader::new");
            let sentinel = r.read(128);
            assert_eq!(
                sentinel, 1,
                "write_tx_size emitted bits in non-select path: tx_mode={tx_mode:?} allow_select={allow_select} max_tx={max_tx:?}"
            );
            // Also verify that read_tx_size returns the expected
            // derived value with no further reads (sentinel was the
            // first/only read).
            let got =
                read_tx_size(&mut r, &fc.tx_probs, *tx_mode, *max_tx, *allow_select, 0, None);
            assert_eq!(
                got, *expected_tx_size,
                "decoder derivation disagrees with writer expectation"
            );
        }
    }

    /// Helpers shared by the `read_inter_segment_id` tests. Encodes a
    /// segment_id leaf via the libvpx tree walk using
    /// `tree_probs = [128; 7]`; appends trailing zero bits so the
    /// bool-reader has padding.
    #[cfg(test)]
    fn encode_segment_id_leaf(target: u8, tree_probs: &[u8; SEG_TREE_PROBS]) -> Vec<u8> {
        use crate::entropy::BoolWriter;

        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        let mut i: i32 = 0;
        fn subtree_contains(tree: &[i8], node: i8, target: u8) -> bool {
            if node <= 0 {
                (-node) as u8 == target
            } else {
                let idx = node as usize;
                subtree_contains(tree, tree[idx], target)
                    || subtree_contains(tree, tree[idx + 1], target)
            }
        }
        loop {
            let left = SEGMENT_TREE[i as usize];
            let right = SEGMENT_TREE[i as usize + 1];
            let bit = if subtree_contains(&SEGMENT_TREE, left, target) { 0 } else { 1 };
            w.write(bit, tree_probs[(i >> 1) as usize]);
            let next = if bit == 0 { left } else { right };
            if next <= 0 {
                break;
            }
            i = next as i32;
        }
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);
        buf
    }

    /// D.10 — `read_inter_segment_id` with `update_map == false`
    /// inherits the predicted segment_id from `prev_segment_ids` and
    /// consumes ZERO bits from the bool-reader. The cursor must remain
    /// untouched (verified by reading a sentinel bit afterwards).
    #[test]
    fn read_inter_segment_id_pure_inheritance() {
        use crate::entropy::BoolWriter;

        let seg = SegmentationReadParams {
            enabled: true,
            update_map: false,
            temporal_update: false,
            tree_probs: [128; SEG_TREE_PROBS],
            pred_probs: [128; PREDICTION_PROBS],
        };

        // Sentinel bit-1 at p=128 follows. If `read_inter_segment_id`
        // pulled any bits, the sentinel would decode as 0 instead.
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, 128);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);

        // Prev seg map: 4×4 mi grid, the 2×2 block at (0,0) reads as
        // min(3,5,5,5) == 3.
        let mi_stride = 4;
        let prev: [u8; 16] = [3, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, seg_pred) =
            read_inter_segment_id(&mut r, &seg, &prev, mi_stride, 0, 0, 2, 2, 0);
        assert_eq!(got_id, 3, "min over 2×2 prev cells = 3");
        assert!(!seg_pred, "no seg_id_predicted bit on the inheritance path");
        // Sentinel still bit-1 ⇒ bool-coder cursor is intact.
        assert_eq!(r.read(128), 1, "no bool-coder bits consumed");
    }

    /// D.10 — `update_map == true`, `temporal_update == true`,
    /// pred_flag bit reads as 1 ⇒ inherit `predicted_segment_id`.
    #[test]
    fn read_inter_segment_id_temporal_update_pred_flag_true() {
        use crate::entropy::BoolWriter;

        let seg = SegmentationReadParams {
            enabled: true,
            update_map: true,
            temporal_update: true,
            tree_probs: [128; SEG_TREE_PROBS],
            pred_probs: [200, 100, 50],
        };

        // Encode pred_flag = 1 against `pred_probs[ctx=1] = 100`.
        let mut buf = vec![0u8; 32];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, 100);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);

        let mi_stride = 2;
        let prev: [u8; 4] = [5, 5, 5, 5];

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, seg_pred) =
            read_inter_segment_id(&mut r, &seg, &prev, mi_stride, 0, 0, 2, 2, /* ctx = */ 1);
        assert_eq!(got_id, 5, "inherited segment_id");
        assert!(seg_pred, "seg_id_predicted bit fires for pred_flag = 1");
    }

    /// D.10 — `update_map == true`, `temporal_update == true`,
    /// pred_flag bit reads as 0 ⇒ tree decode. Verify it picks up the
    /// encoded leaf and the seg_id_predicted bit is `false`.
    #[test]
    fn read_inter_segment_id_temporal_update_pred_flag_false() {
        use crate::entropy::BoolWriter;

        let tree_probs = [128u8; SEG_TREE_PROBS];
        // Pre-build the tree-decode payload for segment_id = 4 and
        // prepend a pred_flag = 0 bit against pred_probs[2] = 50.
        let mut buf = vec![0u8; 64];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(0, 50); // pred_flag = 0
        // Tree walk for target = 4. Reuse the same algorithm as
        // `encode_segment_id_leaf` inline.
        fn subtree_contains(tree: &[i8], node: i8, target: u8) -> bool {
            if node <= 0 {
                (-node) as u8 == target
            } else {
                let idx = node as usize;
                subtree_contains(tree, tree[idx], target)
                    || subtree_contains(tree, tree[idx + 1], target)
            }
        }
        let target: u8 = 4;
        let mut i: i32 = 0;
        loop {
            let left = SEGMENT_TREE[i as usize];
            let right = SEGMENT_TREE[i as usize + 1];
            let bit = if subtree_contains(&SEGMENT_TREE, left, target) { 0 } else { 1 };
            w.write(bit, tree_probs[(i >> 1) as usize]);
            let next = if bit == 0 { left } else { right };
            if next <= 0 {
                break;
            }
            i = next as i32;
        }
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);

        let seg = SegmentationReadParams {
            enabled: true,
            update_map: true,
            temporal_update: true,
            tree_probs,
            pred_probs: [200, 100, 50],
        };
        // Prev cells deliberately != target so we'd detect a wrongly-
        // taken inheritance branch.
        let prev: [u8; 4] = [7, 7, 7, 7];

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, seg_pred) =
            read_inter_segment_id(&mut r, &seg, &prev, 2, 0, 0, 2, 2, /* ctx = */ 2);
        assert_eq!(got_id, target, "tree-decoded segment_id");
        assert!(!seg_pred, "seg_id_predicted bit stays false on tree-decode arm");
    }

    /// D.10 — `update_map == true`, `temporal_update == false`:
    /// straight tree decode, prev_segment_ids ignored.
    #[test]
    fn read_inter_segment_id_update_map_no_temporal() {
        let tree_probs = [128u8; SEG_TREE_PROBS];
        let buf = encode_segment_id_leaf(2, &tree_probs);

        let seg = SegmentationReadParams {
            enabled: true,
            update_map: true,
            temporal_update: false,
            tree_probs,
            pred_probs: [200; PREDICTION_PROBS],
        };
        let prev: [u8; 4] = [7, 7, 7, 7];

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, seg_pred) =
            read_inter_segment_id(&mut r, &seg, &prev, 2, 0, 0, 2, 2, 0);
        assert_eq!(got_id, 2);
        assert!(!seg_pred);
    }

    /// D.10 — inheritance over a 2x2 block must pick the MIN cell, not
    /// the top-left cell. Mirrors libvpx `dec_get_segment_id`.
    #[test]
    fn read_inter_segment_id_inheritance_takes_min_across_block() {
        let seg = SegmentationReadParams {
            enabled: true,
            update_map: false,
            temporal_update: false,
            tree_probs: [128; SEG_TREE_PROBS],
            pred_probs: [128; PREDICTION_PROBS],
        };

        // 4x4 mi grid. Block at (1,1) of size 2x2 covers cells
        // {(1,1)=7, (1,2)=3, (2,1)=4, (2,2)=5} ⇒ min = 3.
        let mi_stride = 4;
        let mut prev = [0u8; 16];
        prev[mi_stride + 1] = 7;
        prev[mi_stride + 2] = 3;
        prev[2 * mi_stride + 1] = 4;
        prev[2 * mi_stride + 2] = 5;

        // No bool-coder reads happen on this path; use a 1-byte dummy
        // buffer. We need a valid bool-reader initial state, so write
        // a single bit first.
        use crate::entropy::BoolWriter;
        let mut buf = vec![0u8; 16];
        let mut w = BoolWriter::start_encode(&mut buf);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, _) = read_inter_segment_id(&mut r, &seg, &prev, mi_stride, 1, 1, 2, 2, 0);
        assert_eq!(got_id, 3, "MIN over the covered block, not just top-left");
    }

    /// D.10 — segmentation disabled ⇒ segment_id is always 0,
    /// regardless of prev_segment_ids, `update_map`, or
    /// `temporal_update`. Mirrors libvpx `if (!seg->enabled) return 0`.
    #[test]
    fn read_inter_segment_id_disabled_returns_zero() {
        use crate::entropy::BoolWriter;

        let seg = SegmentationReadParams {
            enabled: false,
            update_map: true,
            temporal_update: true,
            tree_probs: [128; SEG_TREE_PROBS],
            pred_probs: [128; PREDICTION_PROBS],
        };

        let prev: [u8; 4] = [7, 7, 7, 7];
        let mut buf = vec![0u8; 16];
        let mut w = BoolWriter::start_encode(&mut buf);
        w.write(1, 200);
        for _ in 0..16 {
            w.write_bit(0);
        }
        let size = w.stop_encode().expect("ok");
        buf.truncate(size);

        let mut r = BoolReader::new(&buf).expect("init");
        let (got_id, seg_pred) =
            read_inter_segment_id(&mut r, &seg, &prev, 2, 0, 0, 2, 2, 0);
        assert_eq!(got_id, 0);
        assert!(!seg_pred);
        // Bool-reader still pristine — verifies no bits were consumed.
        assert_eq!(r.read(200), 1);
    }
}
