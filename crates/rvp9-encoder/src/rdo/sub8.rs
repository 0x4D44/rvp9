//! W4.10b — `vp9_rd_pick_inter_mode_sub8x8` skeleton.
//!
//! ## libvpx reference
//!
//! Port of `vp9_rd_pick_inter_mode_sub8x8` —
//! `vp9/encoder/vp9_rdopt.c:4285-5060` (v1.14.1 commit `12f3a2ac6`).
//! The sub-8 RDO picker is the leaf of the partition tree when the
//! 8×8 superblock decides to subdivide further into 4×4 / 4×8 / 8×4
//! sub-blocks.
//!
//! ## libvpx control flow (the load-bearing skeleton)
//!
//! 1. **Per-ref candidate setup** (libvpx lines 4349-4359): For every
//!    enabled ref-frame, initialise `frame_mv[NEWMV][ref] = INVALID_MV`
//!    and `frame_mv[ZEROMV][ref] = 0`. NEAREST / NEAR come from the
//!    `setup_buffer_inter` call.
//! 2. **Per-`ref_index` iteration** (libvpx lines 4361-4900): Walks
//!    `vp9_ref_order` (the same `MAX_REFS = 4` table as the bsize ≥ 8
//!    picker, plus a few extra compound-ref entries). For each
//!    `(ref_frame, second_ref_frame)` pair:
//!    - Run `rd_pick_best_sub8x8_mode` (libvpx line 4646) to walk the
//!      4 sub-block × 4-mode cross product and pick a per-sub-block
//!      winner.
//!    - Accumulate per-block RD across the 4 sub-blocks.
//!    - Compare against `best_rd_so_far` (libvpx line 4900); track the
//!      winner.
//! 3. **Post-pick rewrite + intra fallback** (libvpx lines 4980-5060):
//!    Same NEWMV → NEAREST/NEAR/ZERO rewrite as the bsize ≥ 8 picker.
//!
//! ## Scope (recorded deviations)
//!
//! - **Single-ref only.** Compound modes are out of scope; the
//!   skeleton iterates `{LAST, GOLDEN, ALTREF}` only.
//! - **No `rd_pick_best_sub8x8_mode` port.** The standalone module
//!   accepts the caller's per-sub-block RD candidates as input
//!   (mirroring the output of the libvpx inner-loop walker). The
//!   actual `bestsme` / `cpi->fn_ptr[bsize].xxx` MV-search dispatch is
//!   left for follow-up — for the standalone scope we treat the
//!   per-sub-block candidate set as pre-computed.
//! - **No `seg_ref_active` / `ref_frame_sign_bias` plumbing.**
//! - **No HBD / SIMD.**
//! - **No `pc_tree` reconstruction caching.** The picker returns
//!   per-sub-block winners; the caller replays the reconstruction.
//! - **Only `Block8x8` is supported as the parent.** libvpx's sub-8
//!   picker handles `Block8x8 / Block8x4 / Block4x8 / Block4x4`
//!   parent bsizes; in the standalone scope the parent is fixed to
//!   `Block8x8` (the only one the partition driver enters this picker
//!   for) and the sub-blocks are always 4×4.

use rvp9_core::block::BlockSize;
use rvp9_core::mode_info::{InterMode, Mv, RefFrame};

/// Number of 4×4 sub-blocks inside an 8×8 parent. Mirrors libvpx
/// `num_4x4_blocks_high_lookup[BLOCK_8X8] *
/// num_4x4_blocks_wide_lookup[BLOCK_8X8] = 4`.
pub const NUM_SUB_BLOCKS: usize = 4;

/// Per-sub-block candidate RD pair. The caller supplies one of these
/// per `(sub_block, mode, ref_frame)` triple; the picker walks the
/// cross product and picks the per-sub-block winner.
///
/// Mirrors the per-sub-block accumulator inside libvpx's
/// `rd_pick_best_sub8x8_mode` inner loop (`vp9_rdopt.c:3290-3437`):
/// for every `(blk_idx, mode, ref)` it computes `bsi->bsi[ref]
/// .modes[blk_idx]` + `bsi->bsi[ref].rd` and tracks the best.
#[derive(Copy, Clone, Debug)]
pub struct SubBlockCandidate {
    /// Sub-block index (`0..NUM_SUB_BLOCKS`). Mirrors libvpx
    /// `blk_idx`.
    pub blk_idx: u8,
    /// Inter mode being evaluated.
    pub mode: InterMode,
    /// Ref-frame for this candidate.
    pub ref_frame: RefFrame,
    /// MV for this candidate (1/8-pel units). Zero for ZEROMV.
    pub mv: Mv,
    /// Rate in `vp9_prob_cost` 1/512-bit units (mode bits + MV bits +
    /// model rate). Mirrors libvpx `rate` accumulator at line 3382.
    pub rate: u32,
    /// Distortion (SSE-domain). Mirrors libvpx `distortion` at line
    /// 3383.
    pub dist: i64,
    /// Full RD: `rate * rdmult + dist`. Caller pre-computes; the
    /// picker doesn't recompute. Mirrors libvpx `rd` at line 3389.
    pub rd: i64,
}

/// Per-sub-block winner. The picker collects these for every sub-block.
#[derive(Copy, Clone, Debug)]
pub struct SubBlockWinner {
    /// Sub-block index.
    pub blk_idx: u8,
    /// Winning inter mode.
    pub mode: InterMode,
    /// Winning ref-frame.
    pub ref_frame: RefFrame,
    /// Winning MV.
    pub mv: Mv,
    /// Winning rate.
    pub rate: u32,
    /// Winning distortion.
    pub dist: i64,
    /// Winning RD.
    pub rd: i64,
}

/// Driver input for [`pick_inter_mode_sub8x8`].
pub struct PickSub8x8Params<'a> {
    /// Parent bsize. Fixed to `Block8x8` in the standalone-scope
    /// implementation (libvpx allows `Block8x4 / Block4x8 / Block4x4`
    /// too, but only `Block8x8` reaches the sub-8 picker through the
    /// W4.10a partition driver).
    pub parent_bsize: BlockSize,
    /// Candidate set, flat-indexed. The picker walks every entry,
    /// groups by `blk_idx`, and picks the lowest-RD candidate per
    /// sub-block.
    pub candidates: &'a [SubBlockCandidate],
    /// `best_rd_so_far` cap (libvpx line 4301). The picker returns
    /// `None` if the total RD exceeds this.
    pub best_rd_so_far: i64,
}

/// Output of [`pick_inter_mode_sub8x8`]. Mirrors libvpx
/// `bsi->bsi[best_ref_index]` plus the aggregated RD (lines 5045-5060).
#[derive(Clone, Debug)]
pub struct PickSub8x8Result {
    /// Per-sub-block winners. Always `NUM_SUB_BLOCKS = 4` entries.
    pub sub_winners: [SubBlockWinner; NUM_SUB_BLOCKS],
    /// Aggregated rate (sum of per-sub-block rates).
    pub total_rate: u32,
    /// Aggregated distortion.
    pub total_dist: i64,
    /// Aggregated RD.
    pub total_rd: i64,
}

/// libvpx `vp9_rd_pick_inter_mode_sub8x8` skeleton
/// (`vp9_rdopt.c:4285-5060`).
///
/// Picks the best inter mode independently for each of the 4 sub-4×4
/// blocks inside the parent 8×8, then aggregates. Returns `None` if
/// the aggregate RD exceeds `best_rd_so_far` (libvpx line 4900).
///
/// ## Algorithm (standalone-scope)
///
/// 1. For each `blk_idx ∈ 0..4`, scan `candidates` filtering on
///    matching `blk_idx`, track the lowest-RD entry as the
///    sub-block winner.
/// 2. If any sub-block has zero candidates, return `None` (libvpx
///    would treat this as "no valid mode" and bail).
/// 3. Sum per-sub-block winners → aggregate RD.
/// 4. If aggregate RD ≥ `best_rd_so_far`, return `None`.
/// 5. Otherwise return the aggregated result.
#[must_use]
pub fn pick_inter_mode_sub8x8(params: &PickSub8x8Params<'_>) -> Option<PickSub8x8Result> {
    // libvpx assertion: only Block8x8 parents flow through the W4.10a
    // partition driver into the sub-8 picker.
    assert!(
        matches!(params.parent_bsize, BlockSize::Block8x8),
        "W4.10b standalone scope: parent_bsize must be Block8x8 (got {:?})",
        params.parent_bsize
    );

    // libvpx lines 3290-3437: rd_pick_best_sub8x8_mode tracks the
    // lowest-RD candidate per `(blk_idx, ref_frame)` combination, then
    // picks across refs. The standalone-scope flatten merges the two
    // loops: pick the lowest-RD candidate per `blk_idx` outright.
    let mut sub_winners: [Option<SubBlockWinner>; NUM_SUB_BLOCKS] = [None; NUM_SUB_BLOCKS];
    for cand in params.candidates {
        let idx = cand.blk_idx as usize;
        if idx >= NUM_SUB_BLOCKS {
            continue;
        }
        let replace = match sub_winners[idx] {
            None => true,
            Some(prev) => cand.rd < prev.rd,
        };
        if replace {
            sub_winners[idx] = Some(SubBlockWinner {
                blk_idx: cand.blk_idx,
                mode: cand.mode,
                ref_frame: cand.ref_frame,
                mv: cand.mv,
                rate: cand.rate,
                dist: cand.dist,
                rd: cand.rd,
            });
        }
    }

    // libvpx line 4900: if any sub-block has no valid candidate,
    // bail out (no valid mode for this ref-index).
    let unwrapped: [SubBlockWinner; NUM_SUB_BLOCKS] = {
        let mut out: [SubBlockWinner; NUM_SUB_BLOCKS] = [SubBlockWinner {
            blk_idx: 0,
            mode: InterMode::ZeroMv,
            ref_frame: RefFrame::IntraFrame,
            mv: Mv::ZERO,
            rate: 0,
            dist: 0,
            rd: i64::MAX,
        }; NUM_SUB_BLOCKS];
        for (i, w) in sub_winners.iter().enumerate() {
            out[i] = (*w)?;
        }
        out
    };

    // libvpx lines 4880-4900: aggregate the 4 sub-block RDs.
    let mut total_rate: u64 = 0;
    let mut total_dist: i64 = 0;
    let mut total_rd: i64 = 0;
    for w in &unwrapped {
        total_rate = total_rate.saturating_add(u64::from(w.rate));
        total_dist = total_dist.saturating_add(w.dist);
        total_rd = total_rd.saturating_add(w.rd);
    }

    // libvpx line 4900: cap against best_rd_so_far.
    if total_rd >= params.best_rd_so_far {
        return None;
    }

    Some(PickSub8x8Result {
        sub_winners: unwrapped,
        total_rate: u32::try_from(total_rate).unwrap_or(u32::MAX),
        total_dist,
        total_rd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cand(
        blk_idx: u8,
        mode: InterMode,
        ref_frame: RefFrame,
        rd: i64,
    ) -> SubBlockCandidate {
        SubBlockCandidate {
            blk_idx,
            mode,
            ref_frame,
            mv: Mv::ZERO,
            rate: 500,
            dist: 10_000,
            rd,
        }
    }

    /// PROOF W4.10b — sub-8 picker returns a winner on sane 8×8 input.
    ///
    /// libvpx `vp9_rdopt.c:4285-5060`: a sane (4 sub-blocks, ≥1
    /// candidate per sub-block, total RD < best_rd_so_far) input must
    /// produce a `PickSub8x8Result` with 4 per-sub-block winners and a
    /// finite aggregate RD. We exercise the per-sub-block winner
    /// selection by giving each sub-block a clear lowest-RD entry.
    #[test]
    fn w4_10b_sub8_pick_returns_a_winner_on_8x8_input() {
        let candidates = vec![
            // Sub-block 0: NEAREST@LAST wins (lowest RD).
            make_cand(0, InterMode::ZeroMv, RefFrame::LastFrame, 200_000),
            make_cand(0, InterMode::NearestMv, RefFrame::LastFrame, 50_000),
            make_cand(0, InterMode::NearMv, RefFrame::GoldenFrame, 150_000),
            // Sub-block 1: NEW@LAST wins.
            make_cand(1, InterMode::ZeroMv, RefFrame::LastFrame, 180_000),
            make_cand(1, InterMode::NewMv, RefFrame::LastFrame, 40_000),
            // Sub-block 2: ZERO@GOLDEN wins.
            make_cand(2, InterMode::ZeroMv, RefFrame::GoldenFrame, 30_000),
            make_cand(2, InterMode::NearestMv, RefFrame::LastFrame, 100_000),
            // Sub-block 3: NEAREST@GOLDEN wins.
            make_cand(3, InterMode::NearestMv, RefFrame::GoldenFrame, 60_000),
            make_cand(3, InterMode::ZeroMv, RefFrame::AltrefFrame, 120_000),
        ];
        let params = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd_so_far: i64::MAX,
        };
        let result = pick_inter_mode_sub8x8(&params).expect("must pick a winner");

        // libvpx invariant: 4 sub-block winners, each at its blk_idx.
        for (i, w) in result.sub_winners.iter().enumerate() {
            assert_eq!(w.blk_idx as usize, i, "winner[{i}] blk_idx mismatch");
        }

        // Each sub-block picked its lowest-RD candidate.
        assert_eq!(result.sub_winners[0].mode, InterMode::NearestMv);
        assert_eq!(result.sub_winners[0].rd, 50_000);
        assert_eq!(result.sub_winners[1].mode, InterMode::NewMv);
        assert_eq!(result.sub_winners[1].rd, 40_000);
        assert_eq!(result.sub_winners[2].mode, InterMode::ZeroMv);
        assert_eq!(result.sub_winners[2].ref_frame, RefFrame::GoldenFrame);
        assert_eq!(result.sub_winners[2].rd, 30_000);
        assert_eq!(result.sub_winners[3].mode, InterMode::NearestMv);
        assert_eq!(result.sub_winners[3].rd, 60_000);

        // Aggregate RD = 50k + 40k + 30k + 60k = 180k.
        assert_eq!(result.total_rd, 180_000);
        // Aggregate rate = 4 × 500 = 2000.
        assert_eq!(result.total_rate, 2000);
        // Aggregate dist = 4 × 10_000 = 40_000.
        assert_eq!(result.total_dist, 40_000);
    }

    /// `best_rd_so_far` cap fires: if aggregate RD ≥ cap, return None.
    /// libvpx `vp9_rdopt.c:4900`: `if (this_rd >= best_rd) continue`.
    #[test]
    fn sub8_picker_respects_best_rd_so_far_cap() {
        let candidates = vec![
            make_cand(0, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
            make_cand(1, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
            make_cand(2, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
            make_cand(3, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
        ];
        // Aggregate RD = 400_000; set cap below.
        let params = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd_so_far: 300_000,
        };
        assert!(pick_inter_mode_sub8x8(&params).is_none());

        // Cap above aggregate — picker succeeds.
        let params_ok = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd_so_far: 500_000,
        };
        let r = pick_inter_mode_sub8x8(&params_ok).expect("RD below cap");
        assert_eq!(r.total_rd, 400_000);
    }

    /// Missing-sub-block guard: if one sub-block has no candidate,
    /// the picker returns None. libvpx treats this as "no valid mode"
    /// and continues to the next ref_index.
    #[test]
    fn sub8_picker_returns_none_when_sub_block_missing() {
        let candidates = vec![
            make_cand(0, InterMode::ZeroMv, RefFrame::LastFrame, 50_000),
            // sub-block 1 missing
            make_cand(2, InterMode::ZeroMv, RefFrame::LastFrame, 50_000),
            make_cand(3, InterMode::ZeroMv, RefFrame::LastFrame, 50_000),
        ];
        let params = PickSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd_so_far: i64::MAX,
        };
        assert!(pick_inter_mode_sub8x8(&params).is_none());
    }
}
