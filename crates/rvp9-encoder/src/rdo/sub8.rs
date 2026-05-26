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

/// W7.9 — Sub-8 inter **compound** picker extension.
///
/// ## libvpx reference
///
/// Ports the compound-ref half of `rd_pick_best_sub8x8_mode`
/// (`vp9/encoder/vp9_rdopt.c:2077-2428`, v1.14.1 `12f3a2ac6`). The
/// per-sub-block inner loop is identical to the single-ref path
/// (lines 2150-2360) but with `has_second_rf == 1` toggling:
///
/// - the per-ref MV setup loop (`for (ref = 0; ref < 1 + has_second_rf;
///   ++ref)`) at libvpx lines 2141, 2284, 2306, 2315
/// - the dual MV-bound check at libvpx line 2297-2299
/// - the `joint_motion_search` gate at libvpx lines 2259-2278
/// - per-sub-block `set_and_cost_bmi_mvs` writes both `as_mv[0]` and
///   `as_mv[1]` at libvpx lines 2284-2293
///
/// The outer driver `vp9_rd_pick_inter_mode_sub8x8`
/// (`vp9_rdopt.c:4285-5060`) walks `vp9_ref_order` which includes
/// compound entries (`ref_index 4..MAX_REFS_COMP` in libvpx). When
/// compound is disallowed for the frame (`!allow_comp_inter_inter`,
/// libvpx `vp9_encodeframe.c:6281-6294`), only single-ref entries
/// are walked, so the compound picker is bypassed entirely.
///
/// ## Scope (recorded deviations)
///
/// - **Single+compound only.** Both winners (single-ref and
///   compound-ref) are picked here; the picker returns whichever is
///   lower-RD. The caller chooses based on `compound_allowed`.
/// - **No joint search.** libvpx's `joint_motion_search`
///   (`vp9_rdopt.c:2259-2278`) is a multi-iteration refinement
///   between the two reference MVs; out of scope for the standalone
///   picker. Candidates are pre-computed.
/// - **No filter cache.** libvpx's `filter_idx > 0` cache at lines
///   2301-2336 reuses a previous filter's RD when the MVs match; the
///   standalone picker walks all candidates.
/// - **No HBD / SIMD.**
pub mod compound {
    use super::{NUM_SUB_BLOCKS, PickSub8x8Result, SubBlockWinner};
    use rvp9_core::block::BlockSize;
    use rvp9_core::mode_info::{InterMode, Mv, RefFrame};

    /// Per-sub-block, per-`(mode, ref_pair)` compound candidate.
    /// Mirrors libvpx `bsi->rdstat[block][mode_idx]` at
    /// `vp9_rdopt.c:2280-2360` with `has_second_rf == 1` — both
    /// `mvs[0]` and `mvs[1]` are populated, and the per-candidate RD
    /// folds in the compound MV-rate from
    /// `set_and_cost_bmi_mvs` (libvpx line 2280-2282).
    #[derive(Copy, Clone, Debug)]
    pub struct CompoundSubBlockCandidate {
        /// Sub-block index (`0..NUM_SUB_BLOCKS = 4`).
        pub blk_idx: u8,
        /// Inter mode (compound supports the same `NEARESTMV..=NEWMV`
        /// set as single-ref).
        pub mode: InterMode,
        /// First reference frame (libvpx `mi->ref_frame[0]`).
        pub ref0: RefFrame,
        /// Second reference frame (libvpx `mi->ref_frame[1]`).
        pub ref1: RefFrame,
        /// First MV (libvpx `mvs[0]`).
        pub mv0: Mv,
        /// Second MV (libvpx `mvs[1]`).
        pub mv1: Mv,
        /// Aggregate rate (mode bits + dual MV bits + token rate).
        pub rate: u32,
        /// Distortion (SSE-domain).
        pub dist: i64,
        /// Full RD: `RDCOST(rdmult, rddiv, rate, distortion)`. Caller
        /// pre-computes.
        pub rd: i64,
    }

    /// Per-sub-block compound winner. Mirrors the compound-arm of
    /// libvpx `bsi->rdstat[block][best_mode]`.
    #[derive(Copy, Clone, Debug)]
    pub struct CompoundSubBlockWinner {
        /// Sub-block index.
        pub blk_idx: u8,
        /// Winning inter mode.
        pub mode: InterMode,
        /// Winning ref pair.
        pub ref0: RefFrame,
        /// Winning ref pair.
        pub ref1: RefFrame,
        /// Winning MVs.
        pub mv0: Mv,
        /// Winning MVs.
        pub mv1: Mv,
        /// Winning rate.
        pub rate: u32,
        /// Winning distortion.
        pub dist: i64,
        /// Winning RD.
        pub rd: i64,
    }

    /// Aggregated compound result. Mirrors the compound-arm of the
    /// final accumulators at libvpx `vp9_rdopt.c:2380-2400`
    /// (`br += brate`, `bd += bdist`, `this_segment_rd += brdcost`).
    #[derive(Clone, Debug)]
    pub struct PickSub8x8CompoundResult {
        /// Per-sub-block winners (always 4 entries).
        pub sub_winners: [CompoundSubBlockWinner; NUM_SUB_BLOCKS],
        /// Aggregate rate (`bsi->r`, libvpx line 2397).
        pub total_rate: u32,
        /// Aggregate distortion (`bsi->d`, libvpx line 2398).
        pub total_dist: i64,
        /// Aggregate RD (`bsi->segment_rd`, libvpx line 2400).
        pub total_rd: i64,
    }

    /// Either-arm winner returned by [`pick_inter_mode_sub8x8_compound`].
    ///
    /// libvpx returns whichever ref_index produced the lowest segment
    /// RD; for our purposes we surface both arms so the caller can see
    /// which picker won and unpack the per-sub-block winners.
    #[derive(Clone, Debug)]
    pub enum CompoundOrSingleWinner {
        /// Compound arm (`has_second_rf == 1`) won.
        Compound(PickSub8x8CompoundResult),
        /// Single-ref arm (`has_second_rf == 0`) won. Carries the
        /// W4.10b single-ref result so callers don't need to merge.
        Single(PickSub8x8Result),
    }

    /// Driver input for [`pick_inter_mode_sub8x8_compound`].
    pub struct PickSub8x8CompoundParams<'a> {
        /// Parent bsize. Fixed to `Block8x8` (libvpx's only entry path).
        pub parent_bsize: BlockSize,
        /// Compound-ref candidates (used iff `compound_allowed`).
        pub compound_candidates: &'a [CompoundSubBlockCandidate],
        /// Single-ref winners — the pre-picked output of
        /// [`super::pick_inter_mode_sub8x8`]. Mirrors libvpx's
        /// `single_filter` cache at `vp9_rdopt.c:4530-4540`.
        pub single_result: Option<PickSub8x8Result>,
        /// Compound-allowed gate (libvpx's per-frame
        /// `allow_comp_inter_inter` from `vp9_encodeframe.c:6281-6294`).
        /// When `false`, only `single_result` is considered.
        pub compound_allowed: bool,
        /// `best_rd_so_far` cap (libvpx line 2406:
        /// `if (bsi->segment_rd > best_rd_so_far) return INT64_MAX`).
        pub best_rd_so_far: i64,
    }

    /// libvpx compound sub-8 picker (`vp9_rdopt.c:2077-2428`,
    /// `has_second_rf == 1` arm) — picks the best compound mode
    /// per-sub-block, aggregates, and compares against the single-ref
    /// winner.
    ///
    /// ## Algorithm (standalone-scope)
    ///
    /// 1. If `!compound_allowed`, return the single-ref winner (or
    ///    `None` if that's also empty). Mirrors libvpx's
    ///    `vp9_ref_order` walk which simply omits compound entries
    ///    when the frame-level gate is off.
    /// 2. Otherwise:
    ///    a. For each sub-block, pick the lowest-RD compound candidate.
    ///       If any sub-block has no compound candidate, the compound
    ///       arm is invalid (libvpx returns `INT64_MAX` for that
    ///       ref_index, line 2369).
    ///    b. Aggregate. If aggregate ≥ `best_rd_so_far`, the compound
    ///       arm is invalid (libvpx line 2406).
    ///    c. Compare against single-ref winner. Return the lower-RD.
    #[must_use]
    pub fn pick_inter_mode_sub8x8_compound(
        params: &PickSub8x8CompoundParams<'_>,
    ) -> Option<CompoundOrSingleWinner> {
        // libvpx invariant: only Block8x8 reaches the sub-8 picker.
        assert!(
            matches!(params.parent_bsize, BlockSize::Block8x8),
            "W7.9 compound: parent_bsize must be Block8x8 (got {:?})",
            params.parent_bsize
        );

        // Bail out path 1: compound disallowed at frame level
        // (libvpx `vp9_encodeframe.c:6281-6294` gate). Return the
        // single-ref winner directly.
        if !params.compound_allowed {
            return params
                .single_result
                .as_ref()
                .map(|r| CompoundOrSingleWinner::Single(r.clone()));
        }

        // Bail out path 2: no compound candidates → fall back to
        // single-ref (same as if compound were disallowed).
        if params.compound_candidates.is_empty() {
            return params
                .single_result
                .as_ref()
                .map(|r| CompoundOrSingleWinner::Single(r.clone()));
        }

        // libvpx lines 2141-2360: per-sub-block compound mode walk.
        // We collapse the inner per-mode loop (libvpx 2150-2360) into
        // a single "min by RD" pass over the caller's candidate list.
        let mut sub_winners: [Option<CompoundSubBlockWinner>; NUM_SUB_BLOCKS] =
            [None; NUM_SUB_BLOCKS];
        for cand in params.compound_candidates {
            let idx = cand.blk_idx as usize;
            if idx >= NUM_SUB_BLOCKS {
                continue;
            }
            // libvpx line 2357: if (brdcost < best_rd) best_rd = brdcost;
            let replace = match sub_winners[idx] {
                None => true,
                Some(prev) => cand.rd < prev.rd,
            };
            if replace {
                sub_winners[idx] = Some(CompoundSubBlockWinner {
                    blk_idx: cand.blk_idx,
                    mode: cand.mode,
                    ref0: cand.ref0,
                    ref1: cand.ref1,
                    mv0: cand.mv0,
                    mv1: cand.mv1,
                    rate: cand.rate,
                    dist: cand.dist,
                    rd: cand.rd,
                });
            }
        }

        // libvpx lines 2363-2370: if any sub-block missing → compound
        // arm invalid for this ref_index (returns INT64_MAX).
        let compound_result: Option<PickSub8x8CompoundResult> = (|| {
            let mut unwrapped = [CompoundSubBlockWinner {
                blk_idx: 0,
                mode: InterMode::ZeroMv,
                ref0: RefFrame::NoRefFrame,
                ref1: RefFrame::NoRefFrame,
                mv0: Mv::ZERO,
                mv1: Mv::ZERO,
                rate: 0,
                dist: 0,
                rd: i64::MAX,
            }; NUM_SUB_BLOCKS];
            for (i, w) in sub_winners.iter().enumerate() {
                unwrapped[i] = (*w)?;
            }
            // libvpx lines 2380-2400: accumulate.
            let mut total_rate: u64 = 0;
            let mut total_dist: i64 = 0;
            let mut total_rd: i64 = 0;
            for w in &unwrapped {
                total_rate = total_rate.saturating_add(u64::from(w.rate));
                total_dist = total_dist.saturating_add(w.dist);
                total_rd = total_rd.saturating_add(w.rd);
            }
            // libvpx line 2406: if (bsi->segment_rd > best_rd_so_far)
            // return INT64_MAX.
            if total_rd >= params.best_rd_so_far {
                return None;
            }
            Some(PickSub8x8CompoundResult {
                sub_winners: unwrapped,
                total_rate: u32::try_from(total_rate).unwrap_or(u32::MAX),
                total_dist,
                total_rd,
            })
        })();

        // libvpx `vp9_rd_pick_inter_mode_sub8x8` outer driver
        // (`vp9_rdopt.c:4900`) compares `this_rd` against
        // `best_rd_so_far` and tracks the overall winner across all
        // ref_indices. We do the equivalent: pick whichever arm
        // (compound vs single) has lower aggregate RD.
        match (compound_result, params.single_result.as_ref()) {
            (Some(comp), Some(sing)) => {
                if comp.total_rd < sing.total_rd {
                    Some(CompoundOrSingleWinner::Compound(comp))
                } else {
                    Some(CompoundOrSingleWinner::Single(sing.clone()))
                }
            }
            (Some(comp), None) => Some(CompoundOrSingleWinner::Compound(comp)),
            (None, Some(sing)) => Some(CompoundOrSingleWinner::Single(sing.clone())),
            (None, None) => None,
        }
    }

    /// Helper: extract aggregate RD from either arm. Test convenience
    /// + future caller convenience for `best_rd_so_far` propagation.
    #[must_use]
    pub fn winner_total_rd(w: &CompoundOrSingleWinner) -> i64 {
        match w {
            CompoundOrSingleWinner::Compound(r) => r.total_rd,
            CompoundOrSingleWinner::Single(r) => r.total_rd,
        }
    }

    /// Helper: was the compound arm the winner?
    #[must_use]
    pub fn winner_is_compound(w: &CompoundOrSingleWinner) -> bool {
        matches!(w, CompoundOrSingleWinner::Compound(_))
    }

    #[cfg(test)]
    mod tests {
        use super::super::{
            PickSub8x8Params, SubBlockCandidate, pick_inter_mode_sub8x8,
        };
        use super::*;

        fn make_single_cand(
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

        fn make_compound_cand(
            blk_idx: u8,
            mode: InterMode,
            rd: i64,
        ) -> CompoundSubBlockCandidate {
            CompoundSubBlockCandidate {
                blk_idx,
                mode,
                ref0: RefFrame::LastFrame,
                ref1: RefFrame::AltrefFrame,
                mv0: Mv::ZERO,
                mv1: Mv::ZERO,
                rate: 700,
                dist: 9_000,
                rd,
            }
        }

        fn single_result_400k() -> PickSub8x8Result {
            // 4 sub-blocks @ 100k RD each, total 400k.
            let cands = [
                make_single_cand(0, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
                make_single_cand(1, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
                make_single_cand(2, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
                make_single_cand(3, InterMode::ZeroMv, RefFrame::LastFrame, 100_000),
            ];
            pick_inter_mode_sub8x8(&PickSub8x8Params {
                parent_bsize: BlockSize::Block8x8,
                candidates: &cands,
                best_rd_so_far: i64::MAX,
            })
            .expect("single-ref winner exists")
        }

        /// PROOF W7.9 (sub-8 inter compound returns a winner) —
        /// libvpx `vp9_rdopt.c:2077-2428`
        /// (`rd_pick_best_sub8x8_mode`, compound arm): with all 4
        /// sub-blocks having a compound candidate below the
        /// `best_rd_so_far` cap, and the compound aggregate beating
        /// the single-ref winner, the picker returns the compound
        /// arm.
        #[test]
        fn w7_9_sub8_inter_compound_returns_winner() {
            // 4 sub-blocks @ 20k RD each → total 80k (< single's 400k).
            let cands = vec![
                make_compound_cand(0, InterMode::NewMv, 20_000),
                make_compound_cand(0, InterMode::NearestMv, 30_000), // distractor
                make_compound_cand(1, InterMode::NewMv, 20_000),
                make_compound_cand(2, InterMode::NewMv, 20_000),
                make_compound_cand(3, InterMode::NewMv, 20_000),
            ];
            let params = PickSub8x8CompoundParams {
                parent_bsize: BlockSize::Block8x8,
                compound_candidates: &cands,
                single_result: Some(single_result_400k()),
                compound_allowed: true,
                best_rd_so_far: i64::MAX,
            };
            let w = pick_inter_mode_sub8x8_compound(&params).expect("picker returns a winner");
            assert!(winner_is_compound(&w), "compound should win (80k < 400k)");
            assert_eq!(winner_total_rd(&w), 80_000);

            // Each sub-block picked its compound candidate.
            if let CompoundOrSingleWinner::Compound(c) = w {
                for (i, sw) in c.sub_winners.iter().enumerate() {
                    assert_eq!(sw.blk_idx as usize, i);
                    assert_eq!(sw.mode, InterMode::NewMv);
                    assert_eq!(sw.ref0, RefFrame::LastFrame);
                    assert_eq!(sw.ref1, RefFrame::AltrefFrame);
                    assert_eq!(sw.rd, 20_000);
                }
                assert_eq!(c.total_rd, 80_000);
                assert_eq!(c.total_rate, 4 * 700);
                assert_eq!(c.total_dist, 4 * 9_000);
            }
        }

        /// PROOF W7.9 (compound disallowed → single-ref result) —
        /// libvpx's per-frame gate `allow_comp_inter_inter`
        /// (`vp9_encodeframe.c:6281-6294`): when compound is
        /// disallowed for the frame, `vp9_ref_order` walks only
        /// single-ref entries, so the compound arm never runs. The
        /// picker must surface the single-ref winner.
        #[test]
        fn w7_9_sub8_inter_compound_falls_back_to_single_when_compound_disallowed() {
            let cands = vec![
                // Compound candidates that WOULD win if allowed.
                make_compound_cand(0, InterMode::NewMv, 1),
                make_compound_cand(1, InterMode::NewMv, 1),
                make_compound_cand(2, InterMode::NewMv, 1),
                make_compound_cand(3, InterMode::NewMv, 1),
            ];
            let single = single_result_400k();
            let params = PickSub8x8CompoundParams {
                parent_bsize: BlockSize::Block8x8,
                compound_candidates: &cands,
                single_result: Some(single.clone()),
                // Gate OFF — compound must be ignored.
                compound_allowed: false,
                best_rd_so_far: i64::MAX,
            };
            let w = pick_inter_mode_sub8x8_compound(&params).expect("returns single-ref");
            assert!(
                !winner_is_compound(&w),
                "single-ref must win when compound disallowed"
            );
            assert_eq!(winner_total_rd(&w), 400_000);
            if let CompoundOrSingleWinner::Single(s) = w {
                assert_eq!(s.total_rd, single.total_rd);
            }
        }

        /// Compound arm picks the lower-RD candidate per sub-block,
        /// matching libvpx's `if (brdcost < best_rd) best_rd = brdcost`
        /// at `vp9_rdopt.c:2357`.
        #[test]
        fn sub8_compound_min_rd_per_subblock() {
            let cands = vec![
                make_compound_cand(0, InterMode::NewMv, 50_000),
                make_compound_cand(0, InterMode::NearestMv, 30_000), // winner sub 0
                make_compound_cand(0, InterMode::NearMv, 40_000),
                make_compound_cand(1, InterMode::NewMv, 25_000), // winner sub 1
                make_compound_cand(1, InterMode::NearestMv, 60_000),
                make_compound_cand(2, InterMode::ZeroMv, 35_000), // winner sub 2
                make_compound_cand(3, InterMode::NewMv, 45_000), // winner sub 3
            ];
            let params = PickSub8x8CompoundParams {
                parent_bsize: BlockSize::Block8x8,
                compound_candidates: &cands,
                single_result: None,
                compound_allowed: true,
                best_rd_so_far: i64::MAX,
            };
            let w = pick_inter_mode_sub8x8_compound(&params).expect("picker returns a winner");
            assert!(winner_is_compound(&w));
            if let CompoundOrSingleWinner::Compound(c) = w {
                assert_eq!(c.sub_winners[0].mode, InterMode::NearestMv);
                assert_eq!(c.sub_winners[0].rd, 30_000);
                assert_eq!(c.sub_winners[1].mode, InterMode::NewMv);
                assert_eq!(c.sub_winners[1].rd, 25_000);
                assert_eq!(c.sub_winners[2].mode, InterMode::ZeroMv);
                assert_eq!(c.sub_winners[2].rd, 35_000);
                assert_eq!(c.sub_winners[3].mode, InterMode::NewMv);
                assert_eq!(c.sub_winners[3].rd, 45_000);
                assert_eq!(c.total_rd, 30_000 + 25_000 + 35_000 + 45_000);
            }
        }

        /// libvpx `vp9_rdopt.c:2369` — if any sub-block has no valid
        /// compound candidate, the compound arm is invalid. With no
        /// single-ref fallback, the picker returns `None`.
        #[test]
        fn sub8_compound_returns_none_when_subblock_missing_and_no_single() {
            let cands = vec![
                make_compound_cand(0, InterMode::NewMv, 10_000),
                // sub-block 1 missing
                make_compound_cand(2, InterMode::NewMv, 10_000),
                make_compound_cand(3, InterMode::NewMv, 10_000),
            ];
            let params = PickSub8x8CompoundParams {
                parent_bsize: BlockSize::Block8x8,
                compound_candidates: &cands,
                single_result: None,
                compound_allowed: true,
                best_rd_so_far: i64::MAX,
            };
            assert!(pick_inter_mode_sub8x8_compound(&params).is_none());
        }

        /// libvpx `vp9_rdopt.c:2406` — `best_rd_so_far` cap fires on
        /// the aggregate. If compound aggregate ≥ cap, the compound
        /// arm is invalid; falls back to single-ref.
        #[test]
        fn sub8_compound_respects_best_rd_so_far() {
            let cands = vec![
                make_compound_cand(0, InterMode::NewMv, 100_000),
                make_compound_cand(1, InterMode::NewMv, 100_000),
                make_compound_cand(2, InterMode::NewMv, 100_000),
                make_compound_cand(3, InterMode::NewMv, 100_000),
            ];
            // Compound aggregate = 400_000. Cap below → compound rejected.
            let params = PickSub8x8CompoundParams {
                parent_bsize: BlockSize::Block8x8,
                compound_candidates: &cands,
                single_result: Some(single_result_400k()),
                compound_allowed: true,
                best_rd_so_far: 350_000,
            };
            let w = pick_inter_mode_sub8x8_compound(&params).expect("falls back to single");
            assert!(
                !winner_is_compound(&w),
                "compound rejected by cap (400k ≥ 350k), single (400k) wins"
            );
        }
    }
}
