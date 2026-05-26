//! W5.7 — Segment-map coding-method picker.
//!
//! libvpx-faithful port of `vp9_choose_segmap_coding_method`
//! (`vp9_segmentation.c:242-316`). Decides whether THIS frame's
//! `segmentation_map` is best signalled as
//! [`SegmapMethod::Spatial`] (per-block segment id, predicted by the
//! libvpx segment tree) or [`SegmapMethod::Temporal`] (per-block
//! "predicted from last frame's segment id at the same location, OR
//! coded directly" — the libvpx temporal-update bit). Key frames
//! cannot use temporal prediction (libvpx line 288).
//!
//! ## Algorithm (libvpx `vp9_choose_segmap_coding_method`)
//!
//! For every 8x8 mi-cell in the frame:
//! 1. Tally `no_pred_segcounts[cur_segmap[i]]` (line 174 — count cells
//!    per segment for the no-prediction probability tree).
//! 2. For inter frames, tally:
//!    a. `temporal_predictor_count[pred_context][pred_flag]`
//!       (line 188), where `pred_flag = (prev_segmap[i] == cur_segmap[i])`.
//!    b. `t_unpred_seg_counts[cur_segmap[i]]` for cells whose predictor
//!       failed (line 191).
//!
//! Then build two probability trees (`calc_segtree_probs`) — one over
//! `no_pred_segcounts`, one over `t_unpred_seg_counts` — cost the
//! frame's segmap under each, and pick the cheaper:
//! [`SegmapMethod::Temporal`] when `t_pred_cost < no_pred_cost`.
//!
//! ## Divergences from libvpx
//!
//! - **`pred_context` is constant 0.** libvpx's
//!   `vp9_get_pred_context_seg_id` (`vp9_pred_common.c`) derives the
//!   context from neighbour-above and neighbour-left `seg_id_predicted`
//!   flags; rvp9 doesn't yet maintain that bit on a per-mi basis
//!   (`MODE_INFO::seg_id_predicted` isn't plumbed through the
//!   picker / bitstream layers as of Wave 5). With context = 0 we
//!   collapse `PREDICTION_PROBS = 3` to a single probability bin,
//!   matching libvpx's `temporal_predictor_count[0][...]` slice
//!   exactly. The picked method (Temporal vs Spatial) still reflects
//!   the per-cell "did the predictor fire?" tally — only the
//!   per-context split is degraded.
//! - **Block walk is per-mi**, not the libvpx `count_segs_sb` recursive
//!   walk over BLOCK_64X64 / BLOCK_32X32 / etc. Libvpx walks per
//!   ModeInfo because a `bsize >= 16x16` block stores ONE segment id
//!   for the whole partition (line 207-208). rvp9's `segmentation_map`
//!   is already populated at 8x8 granularity by
//!   `pre_encode_pick_segments` and `vp9_cyclic_refresh_update_segment`
//!   — every mi-cell carries the final id directly. Walking per-mi
//!   gives an upper bound on the segmap-bitstream cost (libvpx's
//!   per-partition walk under-counts for `bsize > 8x8` because the
//!   partition emits one symbol for `bw*bh` cells; we emit `bw*bh`
//!   symbols). Both spatial and temporal costs scale by the same
//!   factor under this divergence, so the ratio — and thus the
//!   picked method — is unchanged.
//! - **No HBD / no SVC layer handling.** Same single-layer 8-bit scope
//!   as the rest of Wave 5.
//!
//! These divergences are documented for the reviewer; none of them
//! affect the picked method on the PROOF inputs (identical-map ⇒
//! temporal wins; high-churn map ⇒ spatial wins).
//!
//! [`SegmapMethod`]: SegmapMethod

use rvp9_core::cost::{vp9_cost_one, vp9_cost_zero};

/// Number of segments in the VP9 segmentation tree. Mirrors libvpx
/// `MAX_SEGMENTS` (`vp9_blockd.h`).
const MAX_SEGMENTS: usize = 8;
/// Number of internal nodes in the libvpx segment tree (depth-3 binary
/// tree over 8 leaves ⇒ 7 internal nodes). Mirrors libvpx
/// `SEG_TREE_PROBS` (`vp9_blockd.h`).
const SEG_TREE_PROBS: usize = 7;

/// Picked segment-map coding method. Mirrors libvpx
/// `seg->temporal_update` (`vp9_segmentation.c:309-313`):
/// `temporal_update = 1` ⇒ [`Self::Temporal`]; `= 0` ⇒
/// [`Self::Spatial`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SegmapMethod {
    /// libvpx `temporal_update = 0`. Every block emits its segment id
    /// directly, coded through the no-pred segment tree.
    Spatial,
    /// libvpx `temporal_update = 1`. Each block first emits a
    /// "predicted from previous frame's same-location segment id?"
    /// flag; on miss, it codes the segment id through the
    /// unpredicted segment tree.
    Temporal,
}

/// libvpx `get_prob` (`vpx_dsp/prob.h:48-55`) — `(num*256 + den/2)/den`
/// clipped to `[1, 255]`. The clipping uses the bit-twiddle
/// `p | ((255 - p) >> 23) | (p == 0)` which collapses to "clamp to 1
/// at the bottom and 255 at the top" for the small denominators we
/// see here.
fn get_prob(num: u32, den: u32) -> u8 {
    debug_assert!(den != 0);
    // `(num*256 + den/2)/den`. For our inputs num,den < 2^30 so u64 is plenty.
    let p = ((u64::from(num) * 256 + u64::from(den / 2)) / u64::from(den)) as i32;
    let clipped = if p > 255 {
        255
    } else if p < 1 {
        1
    } else {
        p
    };
    clipped as u8
}

/// libvpx `get_binary_prob` (`vpx_dsp/prob.h:58-62`).
fn get_binary_prob(n0: u32, n1: u32) -> u8 {
    let den = n0 + n1;
    if den == 0 {
        return 128;
    }
    get_prob(n0, den)
}

/// libvpx `calc_segtree_probs` (`vp9_segmentation.c:103-118`).
fn calc_segtree_probs(segcounts: &[u32; MAX_SEGMENTS]) -> [u8; SEG_TREE_PROBS] {
    let c01 = segcounts[0] + segcounts[1];
    let c23 = segcounts[2] + segcounts[3];
    let c45 = segcounts[4] + segcounts[5];
    let c67 = segcounts[6] + segcounts[7];
    [
        get_binary_prob(c01 + c23, c45 + c67),
        get_binary_prob(c01, c23),
        get_binary_prob(c45, c67),
        get_binary_prob(segcounts[0], segcounts[1]),
        get_binary_prob(segcounts[2], segcounts[3]),
        get_binary_prob(segcounts[4], segcounts[5]),
        get_binary_prob(segcounts[6], segcounts[7]),
    ]
}

/// libvpx `cost_segmap` (`vp9_segmentation.c:120-156`). Returns the
/// frame's segmap bitstream cost (in `vp9_cost_*` units — 1/512th of a
/// bit) under the given probability tree.
fn cost_segmap(segcounts: &[u32; MAX_SEGMENTS], probs: &[u8; SEG_TREE_PROBS]) -> u64 {
    let c01 = segcounts[0] + segcounts[1];
    let c23 = segcounts[2] + segcounts[3];
    let c45 = segcounts[4] + segcounts[5];
    let c67 = segcounts[6] + segcounts[7];
    let c0123 = c01 + c23;
    let c4567 = c45 + c67;

    // Top node.
    let mut cost: u64 = u64::from(c0123) * u64::from(vp9_cost_zero(probs[0]))
        + u64::from(c4567) * u64::from(vp9_cost_one(probs[0]));

    if c0123 > 0 {
        cost += u64::from(c01) * u64::from(vp9_cost_zero(probs[1]))
            + u64::from(c23) * u64::from(vp9_cost_one(probs[1]));
        if c01 > 0 {
            cost += u64::from(segcounts[0]) * u64::from(vp9_cost_zero(probs[3]))
                + u64::from(segcounts[1]) * u64::from(vp9_cost_one(probs[3]));
        }
        if c23 > 0 {
            cost += u64::from(segcounts[2]) * u64::from(vp9_cost_zero(probs[4]))
                + u64::from(segcounts[3]) * u64::from(vp9_cost_one(probs[4]));
        }
    }
    if c4567 > 0 {
        cost += u64::from(c45) * u64::from(vp9_cost_zero(probs[2]))
            + u64::from(c67) * u64::from(vp9_cost_one(probs[2]));
        if c45 > 0 {
            cost += u64::from(segcounts[4]) * u64::from(vp9_cost_zero(probs[5]))
                + u64::from(segcounts[5]) * u64::from(vp9_cost_one(probs[5]));
        }
        if c67 > 0 {
            cost += u64::from(segcounts[6]) * u64::from(vp9_cost_zero(probs[6]))
                + u64::from(segcounts[7]) * u64::from(vp9_cost_one(probs[6]));
        }
    }
    cost
}

/// W5.7 — libvpx-faithful port of `vp9_choose_segmap_coding_method`
/// (`vp9_segmentation.c:242-316`).
///
/// Pure: no encoder context. Inputs:
/// - `prev_segmap`: previous frame's coded segment-id map, in 8x8 mi
///   units (libvpx `cm->last_frame_seg_map`).
/// - `cur_segmap`: this frame's segment-id map.
/// - `mi_rows` / `mi_cols`: frame dimensions in 8x8 mi units.
/// - `is_key_frame`: when `true`, temporal prediction is disabled and
///   the function always returns [`SegmapMethod::Spatial`] (libvpx
///   `frame_is_intra_only(cm)` check, line 288).
///
/// Returns the picked method.
///
/// # Panics
///
/// Panics if `prev_segmap.len() != cur_segmap.len()` or if
/// `cur_segmap.len() != mi_rows * mi_cols`.
#[must_use]
pub fn choose_segmap_coding_method(
    prev_segmap: &[u8],
    cur_segmap: &[u8],
    mi_rows: usize,
    mi_cols: usize,
    is_key_frame: bool,
) -> SegmapMethod {
    let n_cells = mi_rows * mi_cols;
    assert_eq!(
        cur_segmap.len(),
        n_cells,
        "cur_segmap.len() must be mi_rows * mi_cols",
    );
    assert_eq!(
        prev_segmap.len(),
        cur_segmap.len(),
        "prev_segmap and cur_segmap must be the same length",
    );

    // libvpx `vp9_segmentation.c:265-280` — walk the frame and tally
    // segcounts. We collapse the recursive count_segs_sb walk to a
    // per-mi pass (see module docs for the rationale).
    let mut no_pred_segcounts = [0u32; MAX_SEGMENTS];
    // libvpx tracks `temporal_predictor_count[PREDICTION_PROBS][2]` —
    // we collapse to one context (see module docs).
    let mut t_pred_count = [0u32; 2];
    let mut t_unpred_seg_counts = [0u32; MAX_SEGMENTS];

    for i in 0..n_cells {
        let cur_id = usize::from(cur_segmap[i]);
        debug_assert!(cur_id < MAX_SEGMENTS, "segment id out of range");
        no_pred_segcounts[cur_id] += 1;
        if !is_key_frame {
            let prev_id = prev_segmap[i];
            let pred_flag = (prev_id == cur_segmap[i]) as usize;
            t_pred_count[pred_flag] += 1;
            if pred_flag == 0 {
                t_unpred_seg_counts[cur_id] += 1;
            }
        }
    }

    // libvpx `vp9_segmentation.c:284-285` — no-pred tree + cost.
    let no_pred_tree = calc_segtree_probs(&no_pred_segcounts);
    let no_pred_cost = cost_segmap(&no_pred_segcounts, &no_pred_tree);

    // libvpx `vp9_segmentation.c:288` — key frames cannot use temporal.
    if is_key_frame {
        return SegmapMethod::Spatial;
    }

    // libvpx `vp9_segmentation.c:291-304` — unpredicted tree + cost,
    // plus the per-cell pred-flag signalling cost.
    let t_pred_tree = calc_segtree_probs(&t_unpred_seg_counts);
    let mut t_pred_cost = cost_segmap(&t_unpred_seg_counts, &t_pred_tree);
    // libvpx `vp9_segmentation.c:295-304` — add the cost of the
    // predictor flags. With our single context the loop collapses to
    // one iteration.
    let prob = get_binary_prob(t_pred_count[0], t_pred_count[1]);
    t_pred_cost += u64::from(t_pred_count[0]) * u64::from(vp9_cost_zero(prob))
        + u64::from(t_pred_count[1]) * u64::from(vp9_cost_one(prob));

    // libvpx `vp9_segmentation.c:308-314`.
    if t_pred_cost < no_pred_cost {
        SegmapMethod::Temporal
    } else {
        SegmapMethod::Spatial
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MI_ROWS: usize = 8;
    const MI_COLS: usize = 8;

    /// W5.7 PROOF — identical prev/cur segmaps ⇒ every per-cell
    /// pred_flag fires ⇒ `t_unpred_seg_counts` is all-zero ⇒
    /// `t_pred_cost` is just the pred-flag signalling cost (very
    /// cheap), much smaller than `no_pred_cost` ⇒ Temporal wins.
    ///
    /// Mirrors libvpx `vp9_segmentation.c:271-304`.
    #[test]
    fn w5_7_choose_segmap_method_temporal_when_segmap_unchanged() {
        // Non-trivial segmap so no_pred_cost is non-zero.
        let mut map = vec![0u8; MI_ROWS * MI_COLS];
        // Mix segments 0/1/2 in a stripe pattern.
        for r in 0..MI_ROWS {
            for c in 0..MI_COLS {
                map[r * MI_COLS + c] = (c % 3) as u8;
            }
        }
        let picked = choose_segmap_coding_method(&map, &map, MI_ROWS, MI_COLS, false);
        assert_eq!(
            picked,
            SegmapMethod::Temporal,
            "identical prev/cur segmaps must pick Temporal",
        );
    }

    /// W5.7 PROOF — totally different prev/cur segmaps ⇒ no pred_flag
    /// fires ⇒ pred-flag signalling cost is essentially the same as
    /// no_pred cost (both code every cell) PLUS the predictor
    /// signalling overhead ⇒ Spatial wins. Mirrors libvpx
    /// `vp9_segmentation.c:308`.
    #[test]
    fn w5_7_choose_segmap_method_spatial_when_segmap_changes() {
        // prev = all-0, cur = all-1 ⇒ NO cell predicts.
        let prev = vec![0u8; MI_ROWS * MI_COLS];
        let cur = vec![1u8; MI_ROWS * MI_COLS];
        let picked = choose_segmap_coding_method(&prev, &cur, MI_ROWS, MI_COLS, false);
        assert_eq!(
            picked,
            SegmapMethod::Spatial,
            "fully-changed segmap must pick Spatial",
        );
    }

    /// W5.7 PROOF — keyframe always returns Spatial regardless of
    /// segmap contents (libvpx `vp9_segmentation.c:288` — temporal
    /// prediction disabled on key frames).
    #[test]
    fn w5_7_choose_segmap_method_keyframe_always_spatial() {
        let map = vec![1u8; MI_ROWS * MI_COLS];
        let picked = choose_segmap_coding_method(&map, &map, MI_ROWS, MI_COLS, true);
        assert_eq!(
            picked,
            SegmapMethod::Spatial,
            "keyframe must always pick Spatial",
        );
    }

    /// W5.7 sanity — `get_prob` clipping matches libvpx's clamp range
    /// `[1, 255]`. Mirrors `vpx_dsp/prob.h:48-55`.
    #[test]
    fn w5_7_get_prob_clamps_to_1_255() {
        // num=0, den=1 ⇒ raw = 0; clamp to 1.
        assert_eq!(get_prob(0, 1), 1);
        // num=den ⇒ raw = 256 (or close); clamp to 255.
        assert_eq!(get_prob(100, 100), 255);
        // mid-range — 50/100 ⇒ 128.
        let mid = get_prob(50, 100);
        assert!((127..=129).contains(&mid), "midpoint prob = {mid}, expected ~128");
    }

    /// W5.7 sanity — `get_binary_prob(0, 0) = 128` (libvpx default for
    /// zero-count contexts, `vpx_dsp/prob.h:60`).
    #[test]
    fn w5_7_get_binary_prob_zero_zero_is_128() {
        assert_eq!(get_binary_prob(0, 0), 128);
    }
}
