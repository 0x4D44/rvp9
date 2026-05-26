//! W7.9 — Sub-8 KF intra picker.
//!
//! ## libvpx reference
//!
//! Ports the per-4×4 KF intra picker stack:
//!
//! - `rd_pick_intra4x4block` — `vp9/encoder/vp9_rdopt.c:1061-1297`
//!   (v1.14.1 commit `12f3a2ac6`). Picks the best of the 10 intra
//!   prediction modes for a single 4×4 sub-block, gated by
//!   `cpi->sf.intra_y_mode_mask[TX_4X4]` and short-circuited on
//!   `RDCOST(ratey, distortion) >= best_rd`.
//! - `rd_pick_intra_sub_8x8_y_mode` — `vp9/encoder/vp9_rdopt.c:1299-1360`.
//!   Walks the 4 sub-blocks of an 8×8 KF block (parent `BLOCK_8X8`,
//!   loop bounds `idy/idx ∈ {0,1}` step `num_4x4_blocks_{high,wide} = 1`)
//!   and aggregates the per-sub-block winners.
//!
//! ## Scope (recorded deviations)
//!
//! - **8-bit only.** The `CONFIG_VP9_HIGHBITDEPTH` branch at libvpx
//!   `vp9_rdopt.c:1082-1199` is skipped (D-01 RESOLVED, no HBD in rvp9).
//! - **Scalar reference.** No SIMD dispatch; the candidate set is
//!   supplied as pre-computed per-`(blk_idx, mode)` RD pairs (mirroring
//!   the inner-loop output of libvpx lines 1218-1287).
//! - **Standalone.** No encoder integration; the picker exposes a
//!   pure-function interface (input candidates → winner) and does not
//!   touch any frame / coder state.
//! - **No reconstruction caching.** libvpx caches the winning predicted
//!   sub-block into `best_dst` (lines 1283-1294) for the parent's
//!   subsequent passes; we delegate reconstruction to the caller.
//! - **No `intra_y_mode_mask` / `FLAG_SKIP_INTRA_DIRMISMATCH` speed
//!   feature.** The mask gate at libvpx line 1207 and the
//!   `conditional_skipintra` gate at libvpx line 1211 are speed-feature
//!   optimisations; the standalone picker walks every candidate the
//!   caller supplies, matching the `sf` = default = "all modes" path.
//! - **Cost mode.** The per-mode rate (libvpx `bmode_costs[mode]` at
//!   line 1205, the per-`(above, left)` table from
//!   `cpi->y_mode_costs[A][L]` at libvpx `vp9_rdopt.c:1329`) is folded
//!   into the candidate's `rate` field by the caller — the picker only
//!   sees the final RD it needs to compare.
//!
//! ## Why this picker matters
//!
//! At KF (and for any 8×8 block whose RDO picker selected sub-8 luma
//! prediction), libvpx descends into per-4×4 mode decision. Without
//! this leaf, the W4.10b sub-8 partition driver has no inter-vs-intra
//! arbitration for KF blocks below 8×8 — the picker would silently
//! force intra at the parent 8×8 level even when per-4×4 would BD
//! materially better.

use rvp9_core::block::BlockSize;
use rvp9_core::intra_pred::IntraMode;

/// Number of 4×4 sub-blocks inside an 8×8 parent. Mirrors libvpx
/// `num_4x4_blocks_high_lookup[BLOCK_8X8] *
/// num_4x4_blocks_wide_lookup[BLOCK_8X8] = 4`.
pub const NUM_SUB_BLOCKS: usize = 4;

/// Per-sub-block, per-mode candidate. Mirrors the inner-loop
/// accumulator inside libvpx `rd_pick_intra4x4block`
/// (`vp9_rdopt.c:1201-1287`): for every `(blk_idx, mode)` it builds a
/// `(rate, distortion, rd)` triple where:
/// - `rate = bmode_costs[mode] + ratey`            (libvpx lines 1205, 1272)
/// - `distortion` = per-`(idy, idx)` sum-of-block-error >> 2  (line 1261)
/// - `rd = RDCOST(rdmult, rddiv, rate, distortion)`           (line 1273)
#[derive(Copy, Clone, Debug)]
pub struct KfIntraCandidate {
    /// Sub-block index (`0..NUM_SUB_BLOCKS`). Mirrors libvpx
    /// `i = idy * 2 + idx` at `vp9_rdopt.c:1324`.
    pub blk_idx: u8,
    /// Intra prediction mode being evaluated. libvpx walks
    /// `DC_PRED..=TM_PRED` (line 1201).
    pub mode: IntraMode,
    /// Aggregate rate: `bmode_costs[mode]` + coefficient ratey
    /// (libvpx `rate += ratey` at line 1272). Units: 1/512-bit
    /// (vp9_prob_cost units).
    pub rate: u32,
    /// Token-only rate (coefficient bits, no mode-bits component).
    /// Mirrors libvpx `ratey` at line 1203, stored separately so the
    /// parent picker (`rd_pick_intra_sub_8x8_y_mode`) can return both
    /// totals to its caller (`*rate_y` at `vp9_rdopt.c:1355`).
    pub rate_y: u32,
    /// Distortion (SSE >> 2, libvpx line 1261).
    pub dist: i64,
    /// Full RD: `RDCOST(rdmult, rddiv, rate, distortion)`. The picker
    /// does NOT recompute — caller pre-computes once.
    pub rd: i64,
}

/// Per-sub-block winner. Mirrors `(best_mode, *bestrate, *bestratey,
/// *bestdistortion, best_rd)` at `vp9_rdopt.c:1174-1179`.
#[derive(Copy, Clone, Debug)]
pub struct KfIntraSubWinner {
    /// Sub-block index.
    pub blk_idx: u8,
    /// Winning intra mode (libvpx `*best_mode`).
    pub mode: IntraMode,
    /// Winning aggregate rate.
    pub rate: u32,
    /// Winning token-only rate.
    pub rate_y: u32,
    /// Winning distortion.
    pub dist: i64,
    /// Winning RD.
    pub rd: i64,
}

/// Driver input for [`pick_intra4x4_block`]. Per-sub-block scope: the
/// caller filters candidates to a single `blk_idx` before calling.
pub struct PickIntra4x4Params<'a> {
    /// Sub-block index (only candidates with matching `blk_idx` are
    /// considered; others are ignored).
    pub blk_idx: u8,
    /// All candidates the caller pre-computed for this sub-block.
    pub candidates: &'a [KfIntraCandidate],
    /// `rd_thresh` cap (libvpx line 1066, `rd_thresh` parameter). The
    /// picker returns `None` if no candidate beats the cap.
    pub rd_thresh: i64,
}

/// libvpx `rd_pick_intra4x4block` (`vp9_rdopt.c:1061-1297`) — picks the
/// best 4×4 intra mode for a single sub-block.
///
/// ## Algorithm (standalone-scope)
///
/// 1. Filter `params.candidates` to entries with matching `blk_idx`.
/// 2. Track the lowest-RD candidate as `best`.
/// 3. If the lowest RD ≥ `rd_thresh`, return `None` (mirrors libvpx's
///    `best_rd = rd_thresh` initialisation at line 1069: any RD that
///    fails to beat the cap is rejected).
#[must_use]
pub fn pick_intra4x4_block(params: &PickIntra4x4Params<'_>) -> Option<KfIntraSubWinner> {
    // libvpx line 1069: best_rd = rd_thresh (cap initialiser).
    let mut best: Option<KfIntraSubWinner> = None;
    for cand in params.candidates {
        if cand.blk_idx != params.blk_idx {
            continue;
        }
        // libvpx line 1174: if (this_rd < best_rd) { ... best_rd = this_rd; }
        let replace = match best {
            None => cand.rd < params.rd_thresh,
            Some(prev) => cand.rd < prev.rd,
        };
        if replace {
            best = Some(KfIntraSubWinner {
                blk_idx: cand.blk_idx,
                mode: cand.mode,
                rate: cand.rate,
                rate_y: cand.rate_y,
                dist: cand.dist,
                rd: cand.rd,
            });
        }
    }
    best
}

/// Driver input for [`pick_intra_sub_8x8_y_mode`].
pub struct PickIntraSub8x8Params<'a> {
    /// Parent bsize. Must be `Block8x8` (libvpx asserts via
    /// `xd->mi[0]->sb_type` at `vp9_rdopt.c:1308`; the only entry
    /// point is the 8×8 KF intra path).
    pub parent_bsize: BlockSize,
    /// Flat candidate set across all 4 sub-blocks.
    pub candidates: &'a [KfIntraCandidate],
    /// `best_rd` cap (libvpx line 1302). The picker returns `None` if
    /// the aggregate exceeds this.
    pub best_rd: i64,
}

/// Aggregated result of [`pick_intra_sub_8x8_y_mode`]. Mirrors libvpx
/// outputs `*rate / *rate_y / *distortion` at `vp9_rdopt.c:1354-1357`
/// plus the per-sub-block winners and the final `RDCOST`.
#[derive(Clone, Debug)]
pub struct PickIntraSub8x8Result {
    /// Per-sub-block winners (always 4 entries).
    pub sub_winners: [KfIntraSubWinner; NUM_SUB_BLOCKS],
    /// `*rate` (aggregate rate, libvpx line 1354).
    pub total_rate: u32,
    /// `*rate_y` (aggregate token-only rate, libvpx line 1355).
    pub total_rate_y: u32,
    /// `*distortion` (aggregate distortion, libvpx line 1356).
    pub total_distortion: i64,
    /// Final `RDCOST` (libvpx return value, line 1359).
    pub total_rd: i64,
}

/// libvpx `rd_pick_intra_sub_8x8_y_mode` (`vp9_rdopt.c:1299-1360`) —
/// picks the best intra mode independently for each of the 4 4×4
/// sub-blocks inside the parent 8×8, then aggregates.
///
/// libvpx returns `INT64_MAX` on cap exceedance; we return `None`.
///
/// ## Algorithm
///
/// 1. For each `blk_idx ∈ 0..4`:
///    - Call [`pick_intra4x4_block`] with the running cap
///      `best_rd - total_rd` (libvpx line 1335).
///    - If no candidate beats the cap → return `None` (libvpx line 1337
///      returns `INT64_MAX`).
///    - Accumulate.
///    - If aggregate ≥ cap → return `None` (libvpx line 1350).
/// 2. Return the aggregated result.
#[must_use]
pub fn pick_intra_sub_8x8_y_mode(
    params: &PickIntraSub8x8Params<'_>,
) -> Option<PickIntraSub8x8Result> {
    // libvpx `vp9_rdopt.c:1308` asserts parent is BLOCK_8X8 (the only
    // size that reaches this picker). Match libvpx's hard invariant.
    assert!(
        matches!(params.parent_bsize, BlockSize::Block8x8),
        "W7.9 standalone scope: parent_bsize must be Block8x8 (got {:?})",
        params.parent_bsize
    );

    let mut sub_winners: [Option<KfIntraSubWinner>; NUM_SUB_BLOCKS] = [None; NUM_SUB_BLOCKS];
    let mut total_rate: u64 = 0;
    let mut total_rate_y: u64 = 0;
    let mut total_distortion: i64 = 0;
    let mut total_rd: i64 = 0;

    // libvpx `vp9_rdopt.c:1319-1320`: walks `idy/idx ∈ {0,1}` with
    // step `num_4x4_blocks_{high,wide} = 1` (for BLOCK_8X8). Each
    // iteration handles one 4×4 sub-block (`i = idy * 2 + idx`).
    for blk_idx in 0..NUM_SUB_BLOCKS {
        let sub_cap = params.best_rd.saturating_sub(total_rd);
        let sub_params = PickIntra4x4Params {
            blk_idx: blk_idx as u8,
            candidates: params.candidates,
            rd_thresh: sub_cap,
        };
        // libvpx line 1332-1337: rd_pick_intra4x4block result tested
        // against `best_rd - total_rd`; INT64_MAX on miss.
        let winner = pick_intra4x4_block(&sub_params)?;
        if winner.rd >= sub_cap {
            // Mirrors libvpx line 1337: `if (this_rd >= best_rd -
            // total_rd) return INT64_MAX;`
            return None;
        }

        // libvpx lines 1339-1342: accumulate.
        total_rd = total_rd.saturating_add(winner.rd);
        total_rate = total_rate.saturating_add(u64::from(winner.rate));
        total_rate_y = total_rate_y.saturating_add(u64::from(winner.rate_y));
        total_distortion = total_distortion.saturating_add(winner.dist);
        sub_winners[blk_idx] = Some(winner);

        // libvpx line 1350: `if (total_rd >= best_rd) return INT64_MAX;`
        if total_rd >= params.best_rd {
            return None;
        }
    }

    // Unwrap winners — by construction every slot is filled (every
    // `?` above returned early on a miss).
    let unwrapped: [KfIntraSubWinner; NUM_SUB_BLOCKS] = {
        let mut out = [KfIntraSubWinner {
            blk_idx: 0,
            mode: IntraMode::Dc,
            rate: 0,
            rate_y: 0,
            dist: 0,
            rd: i64::MAX,
        }; NUM_SUB_BLOCKS];
        for (i, w) in sub_winners.iter().enumerate() {
            out[i] = w.expect("sub_winners[i] filled by loop above");
        }
        out
    };

    Some(PickIntraSub8x8Result {
        sub_winners: unwrapped,
        total_rate: u32::try_from(total_rate).unwrap_or(u32::MAX),
        total_rate_y: u32::try_from(total_rate_y).unwrap_or(u32::MAX),
        total_distortion,
        total_rd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cand(blk_idx: u8, mode: IntraMode, rd: i64) -> KfIntraCandidate {
        KfIntraCandidate {
            blk_idx,
            mode,
            rate: 800,
            rate_y: 200,
            dist: 10_000,
            rd,
        }
    }

    /// PROOF W7.9 (KF intra 4×4 picks a Y mode) — libvpx
    /// `vp9_rdopt.c:1061-1297` (`rd_pick_intra4x4block`): given a
    /// candidate set covering several intra modes for sub-block 0,
    /// the picker must return one of them (the lowest-RD entry).
    #[test]
    fn w7_9_kf_intra_4x4_picks_a_y_mode() {
        let candidates = vec![
            make_cand(0, IntraMode::Dc, 90_000),
            make_cand(0, IntraMode::V, 50_000), // winner
            make_cand(0, IntraMode::H, 80_000),
            make_cand(0, IntraMode::Tm, 70_000),
            // Distractors for other sub-blocks (must be ignored).
            make_cand(1, IntraMode::Dc, 10_000),
            make_cand(2, IntraMode::V, 20_000),
        ];
        let params = PickIntra4x4Params {
            blk_idx: 0,
            candidates: &candidates,
            rd_thresh: i64::MAX,
        };
        let w = pick_intra4x4_block(&params).expect("picker returns a winner");
        assert_eq!(w.blk_idx, 0);
        assert_eq!(w.mode, IntraMode::V);
        assert_eq!(w.rd, 50_000);
    }

    /// PROOF W7.9 (sub-8×8 KF intra iterates all 4 sub-blocks) —
    /// libvpx `vp9_rdopt.c:1299-1360` (`rd_pick_intra_sub_8x8_y_mode`):
    /// for an 8×8 KF block, the picker walks `idy/idx ∈ {0,1}` giving
    /// 4 sub-blocks at `i = idy*2 + idx ∈ {0,1,2,3}`, picks the best
    /// per-sub mode, and aggregates rate/distortion/RD across all 4.
    #[test]
    fn w7_9_kf_intra_sub_8x8_iterates_all_4_subblocks() {
        let candidates = vec![
            // sub 0: H wins at 50_000
            make_cand(0, IntraMode::Dc, 100_000),
            make_cand(0, IntraMode::H, 50_000),
            // sub 1: V wins at 40_000
            make_cand(1, IntraMode::Dc, 90_000),
            make_cand(1, IntraMode::V, 40_000),
            // sub 2: TM wins at 30_000
            make_cand(2, IntraMode::Dc, 80_000),
            make_cand(2, IntraMode::Tm, 30_000),
            // sub 3: DC wins at 25_000
            make_cand(3, IntraMode::Dc, 25_000),
            make_cand(3, IntraMode::H, 95_000),
        ];
        let params = PickIntraSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd: i64::MAX,
        };
        let r = pick_intra_sub_8x8_y_mode(&params).expect("must pick a winner");

        // Every sub-block has a winner at its own index.
        for (i, w) in r.sub_winners.iter().enumerate() {
            assert_eq!(w.blk_idx as usize, i, "winner[{i}] blk_idx mismatch");
        }
        assert_eq!(r.sub_winners[0].mode, IntraMode::H);
        assert_eq!(r.sub_winners[1].mode, IntraMode::V);
        assert_eq!(r.sub_winners[2].mode, IntraMode::Tm);
        assert_eq!(r.sub_winners[3].mode, IntraMode::Dc);

        // Aggregates: rate = 4 * 800 = 3_200; rate_y = 4 * 200 = 800;
        // dist = 4 * 10_000 = 40_000; rd = 50k + 40k + 30k + 25k = 145k.
        assert_eq!(r.total_rate, 3_200);
        assert_eq!(r.total_rate_y, 800);
        assert_eq!(r.total_distortion, 40_000);
        assert_eq!(r.total_rd, 145_000);
    }

    /// libvpx `vp9_rdopt.c:1350` cap fires: aggregate RD ≥ cap → bail.
    #[test]
    fn w7_9_kf_intra_sub_8x8_respects_best_rd_cap() {
        let candidates = vec![
            make_cand(0, IntraMode::Dc, 100_000),
            make_cand(1, IntraMode::Dc, 100_000),
            make_cand(2, IntraMode::Dc, 100_000),
            make_cand(3, IntraMode::Dc, 100_000),
        ];
        // Aggregate would be 400_000; cap below.
        let params = PickIntraSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd: 300_000,
        };
        assert!(pick_intra_sub_8x8_y_mode(&params).is_none());
    }

    /// PROOF W7.9 anti-G3 (per-mode discrimination) — re-audit
    /// pass 2 flagged `w7_9_kf_intra_4x4_picks_a_y_mode` as an
    /// argmin tautology (the picker is defined as "return the
    /// lowest-RD candidate", so the original test only verifies the
    /// definition).
    ///
    /// This stronger PROOF runs the picker once per intra mode with
    /// that mode as the unique RD minimum, and asserts the winner's
    /// mode field tracks the perturbation. If the inner-loop iteration
    /// were stripped (e.g. `return candidates[0]`), the winner would
    /// stay pinned at `IntraMode::Dc` regardless of input.
    #[test]
    fn w7_9_kf_intra_4x4_picker_tracks_per_mode_perturbations() {
        let modes = [
            IntraMode::Dc,
            IntraMode::V,
            IntraMode::H,
            IntraMode::Tm,
            IntraMode::D45,
            IntraMode::D135,
            IntraMode::D117,
            IntraMode::D153,
            IntraMode::D207,
            IntraMode::D63,
        ];
        for &target in &modes {
            let mut cands = Vec::with_capacity(modes.len());
            for &m in &modes {
                // Baseline RD = 100_000; target mode gets 10_000 (must win).
                let rd = if m == target { 10_000 } else { 100_000 };
                cands.push(make_cand(0, m, rd));
            }
            let params = PickIntra4x4Params {
                blk_idx: 0,
                candidates: &cands,
                rd_thresh: i64::MAX,
            };
            let w = pick_intra4x4_block(&params).expect("winner");
            assert_eq!(
                w.mode, target,
                "perturbing mode {target:?} to the unique min must elect it (got {:?})",
                w.mode,
            );
            assert_eq!(w.rd, 10_000, "winner.rd tracks the perturbation");
        }
    }

    /// PROOF W7.9 anti-G3 (sub-block independence) — re-audit pass 2
    /// flagged `w7_9_kf_intra_sub_8x8_iterates_all_4_subblocks` as
    /// tautological. The original asserted "each sub-block's argmin
    /// is returned at its own index", but didn't prove the iteration
    /// loop processes every slot — a stripped implementation that
    /// hardcoded `sub_winners[i] = candidates[i]` would still pass.
    ///
    /// This stronger PROOF takes the baseline scenario (every sub-block
    /// has multiple candidates) and perturbs ONE sub-block's winner at
    /// a time — verifying that ONLY that slot's chosen mode shifts.
    /// A stripped per-slot loop (or a "return the first cand per slot"
    /// strip) would not respond to the perturbation.
    #[test]
    fn w7_9_kf_intra_sub_8x8_each_subblock_independently_affects_its_winner() {
        // Baseline winners: sub 0 → V, 1 → H, 2 → Tm, 3 → Dc.
        let baseline = vec![
            make_cand(0, IntraMode::Dc, 90_000),
            make_cand(0, IntraMode::V, 50_000),
            make_cand(1, IntraMode::Dc, 90_000),
            make_cand(1, IntraMode::H, 40_000),
            make_cand(2, IntraMode::Dc, 80_000),
            make_cand(2, IntraMode::Tm, 30_000),
            make_cand(3, IntraMode::Dc, 25_000),
            make_cand(3, IntraMode::H, 95_000),
        ];
        let params = PickIntraSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &baseline,
            best_rd: i64::MAX,
        };
        let base = pick_intra_sub_8x8_y_mode(&params).expect("baseline winner");
        let base_modes = [
            base.sub_winners[0].mode,
            base.sub_winners[1].mode,
            base.sub_winners[2].mode,
            base.sub_winners[3].mode,
        ];
        assert_eq!(base_modes, [IntraMode::V, IntraMode::H, IntraMode::Tm, IntraMode::Dc]);

        // Perturb each sub-block in turn: drop the *other* candidate's
        // RD below the baseline winner, and verify only THAT slot's
        // mode flips. Sub-blocks not perturbed must keep the baseline
        // winner.
        for target_blk in 0..NUM_SUB_BLOCKS as u8 {
            let mut perturbed: Vec<KfIntraCandidate> = baseline.clone();
            // For target_blk, override the DC candidate to win.
            for c in perturbed.iter_mut() {
                if c.blk_idx == target_blk && c.mode == IntraMode::Dc {
                    c.rd = 1_000; // becomes the new min for this sub.
                }
            }
            let p = PickIntraSub8x8Params {
                parent_bsize: BlockSize::Block8x8,
                candidates: &perturbed,
                best_rd: i64::MAX,
            };
            let r = pick_intra_sub_8x8_y_mode(&p).expect("perturbed winner");
            for slot in 0..NUM_SUB_BLOCKS {
                let got = r.sub_winners[slot].mode;
                let expected = if slot as u8 == target_blk {
                    IntraMode::Dc
                } else {
                    base_modes[slot]
                };
                assert_eq!(
                    got, expected,
                    "perturbing sub {target_blk}: slot {slot} expected {expected:?}, got {got:?} \
                     (perturbing one slot must not affect the others)",
                );
            }
        }
    }

    /// libvpx `vp9_rdopt.c:1337`: per-sub miss (no candidate beats
    /// the running cap) → outer picker returns INT64_MAX → we return
    /// None.
    #[test]
    fn w7_9_kf_intra_sub_8x8_returns_none_when_subblock_missing() {
        let candidates = vec![
            make_cand(0, IntraMode::Dc, 10_000),
            // sub-block 1 missing entirely
            make_cand(2, IntraMode::Dc, 10_000),
            make_cand(3, IntraMode::Dc, 10_000),
        ];
        let params = PickIntraSub8x8Params {
            parent_bsize: BlockSize::Block8x8,
            candidates: &candidates,
            best_rd: i64::MAX,
        };
        assert!(pick_intra_sub_8x8_y_mode(&params).is_none());
    }
}
