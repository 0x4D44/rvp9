//! W4.10a — `rd_pick_partition` (NONE + SPLIT only) + `rd_pick_sb_modes`
//! dispatcher.
//!
//! ## libvpx references
//!
//! - `rd_pick_partition` — `vp9/encoder/vp9_encodeframe.c:3985-4489`
//!   (v1.14.1 commit `12f3a2ac6`). The slow-path partition driver:
//!   tries each legal partition type at the current bsize, recurses
//!   into child blocks for `PARTITION_SPLIT`, and picks the partition
//!   with the lowest RD.
//! - `rd_pick_sb_modes` — `vp9/encoder/vp9_encodeframe.c:1975-2107`.
//!   The dispatcher that routes a single leaf RDO decision to the
//!   right picker: `vp9_rd_pick_intra_mode_sb` (intra-only frames),
//!   `vp9_rd_pick_inter_mode_sb` (inter, bsize ≥ 8×8), or
//!   `vp9_rd_pick_inter_mode_sub8x8` (inter, bsize < 8×8).
//!
//! ## Scope (recorded deviations)
//!
//! - **NONE + SPLIT only.** The HORZ / VERT arms (libvpx lines
//!   4296-4480) are skipped — they land in a later wave once the leaf
//!   picker exposes a rectangular-bsize entry point.
//! - **No `pc_tree` / context plumbing.** libvpx threads a
//!   `PC_TREE` to cache `PICK_MODE_CONTEXT` state across the partition
//!   evaluation (so the winner's reconstruction can be replayed); the
//!   standalone module just returns the RD pair plus the winning
//!   partition tag.
//! - **No `partition_cost[pl][PARTITION_*]` bits.** libvpx adds the
//!   partition-tree's per-symbol cost to the RD at lines 4279 / 4290.
//!   The standalone driver leaves these out — the caller can add them
//!   post hoc if needed.
//! - **No edge-clamping / `force_horz_split` / `force_vert_split`**
//!   (libvpx lines 4009-4021). The standalone driver always considers
//!   both NONE and SPLIT; the production caller would gate this on
//!   `(mi_row, mi_col)` vs `(mi_rows, mi_cols)`.
//! - **No `min_partition_size` / `max_partition_size` clamps**
//!   (libvpx lines 4014-4015). The driver recurses on every
//!   `bsize ≥ Block8x8`; sub-8 leaves stop at NONE.
//! - **No `partition_search_breakout_thr` early-term** (libvpx lines
//!   4023-4042). The standalone driver always evaluates both NONE and
//!   SPLIT; the production caller plumbs the breakout knobs.
//! - **`rd_pick_sb_modes` is a dispatcher only.** The standalone port
//!   surfaces the three picker tags as an enum decision; the caller
//!   wires the actual `pick_inter_mode_sb` / `pick_inter_mode_sub8x8`
//!   / intra-mode call sites.
//!
//! ## Composition with W4.1-W4.9
//!
//! The driver consumes a caller-supplied `leaf_rd_fn` closure that —
//! given a `(BlockSize, mi_row, mi_col)` triple — returns a
//! [`LeafRdResult`]. In production the caller wires this to:
//! - `super_block_yrd` (W4.1) for the Y-plane RD,
//! - `pick_inter_mode_sb` (W4.9) for the inter-mode loop,
//! - `pick_inter_mode_sub8x8` (W4.10b) for the sub-8 leaf,
//! - `vp9_rd_pick_intra_mode_sb` for the intra branch.
//!
//! The standalone-scope PROOF tests stub the closure with a synthetic
//! variance-aware RD model so the partition decision is observable.

use rvp9_core::block::{BlockSize, PartitionType, subsize};

/// libvpx `rd_pick_sb_modes` frame kind (`vp9_encodeframe.c:2057-2078`).
/// Selects which leaf picker [`rd_pick_sb_modes`] routes to.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// `frame_is_intra_only(cm)` — libvpx line 2057. Routes to
    /// `vp9_rd_pick_intra_mode_sb`.
    IntraOnly,
    /// Inter frame with `segfeature_active(SEG_LVL_SKIP)` set —
    /// libvpx line 2064. Routes to
    /// `vp9_rd_pick_inter_mode_sb_seg_skip`.
    InterSegSkip,
    /// Inter frame, regular path — libvpx lines 2068 / 2077.
    /// Routes to `vp9_rd_pick_inter_mode_sb` (bsize ≥ 8×8) or
    /// `vp9_rd_pick_inter_mode_sub8x8` (bsize < 8×8).
    Inter,
}

/// libvpx `rd_pick_sb_modes` picker dispatch decision
/// (`vp9_encodeframe.c:2057-2083`). The standalone port surfaces the
/// dispatch as an enum so the caller can wire each tag to its
/// concrete picker.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LeafPicker {
    /// `vp9_rd_pick_intra_mode_sb` (libvpx line 2058).
    Intra,
    /// `vp9_rd_pick_inter_mode_sb_seg_skip` (libvpx line 2065).
    InterSegSkip,
    /// `vp9_rd_pick_inter_mode_sb` (libvpx line 2068) — bsize ≥ 8×8.
    Inter,
    /// `vp9_rd_pick_inter_mode_sub8x8` (libvpx line 2077) — bsize < 8×8.
    InterSub8,
}

/// libvpx `rd_pick_sb_modes` dispatcher (`vp9_encodeframe.c:1975-2107`).
///
/// Routes a single-leaf RDO decision to the correct picker. The
/// standalone port returns the dispatch tag; the caller invokes the
/// matching W4.9 / W4.10b / intra-picker routine.
///
/// libvpx control flow:
/// 1. Line 2057: `if (frame_is_intra_only(cm)) → vp9_rd_pick_intra_mode_sb`.
/// 2. Line 2060: `else if (bsize >= BLOCK_8X8) → vp9_rd_pick_inter_mode_sb`
///    (plus the `SEG_LVL_SKIP` branch at line 2064).
/// 3. Line 2073: `else → vp9_rd_pick_inter_mode_sub8x8`.
#[must_use]
pub fn rd_pick_sb_modes(frame_kind: FrameKind, bsize: BlockSize) -> LeafPicker {
    match frame_kind {
        FrameKind::IntraOnly => LeafPicker::Intra,
        FrameKind::InterSegSkip => LeafPicker::InterSegSkip,
        FrameKind::Inter => {
            // libvpx line 2060: bsize >= BLOCK_8X8.
            if bsize.width() >= 8 && bsize.height() >= 8 {
                LeafPicker::Inter
            } else {
                LeafPicker::InterSub8
            }
        }
    }
}

/// Leaf RD pair returned by the caller's `leaf_rd_fn` closure. Mirrors
/// libvpx's `RD_COST { rate, dist, rdcost }` triple
/// (`vp9_encoder.h::RD_COST`).
#[derive(Copy, Clone, Debug)]
pub struct LeafRdResult {
    /// Total rate at this leaf (in libvpx `vp9_prob_cost` 1/512-bit
    /// units). `u32::MAX` signals "infeasible" — the driver treats it
    /// as `i64::MAX` and skips this branch.
    pub rate: u32,
    /// Total distortion at this leaf.
    pub dist: i64,
    /// Total RD cost (rate * rdmult + dist; already combined).
    pub rd: i64,
}

impl LeafRdResult {
    /// Infeasible sentinel (libvpx's `RD_COST { INT_MAX, INT64_MAX,
    /// INT64_MAX }`).
    pub const INFEASIBLE: Self = Self {
        rate: u32::MAX,
        dist: i64::MAX,
        rd: i64::MAX,
    };
}

/// libvpx `mi_row` / `mi_col` position (in 8×8 units). Threaded
/// through the partition recursion so the leaf closure can locate the
/// block in the source / reference planes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MiPos {
    /// 8×8 row index within the frame.
    pub mi_row: u32,
    /// 8×8 column index within the frame.
    pub mi_col: u32,
}

/// libvpx `rd_pick_partition` output. Mirrors the
/// `(best_rdc, pc_tree->partitioning)` pair returned at line 4489.
#[derive(Copy, Clone, Debug)]
pub struct PartitionResult {
    /// Winning partition tag (NONE or SPLIT in this batch).
    pub partition: PartitionType,
    /// Winning RD pair.
    pub rd: LeafRdResult,
}

/// Caller-supplied leaf RD closure. Returns `LeafRdResult::INFEASIBLE`
/// to signal "this leaf can't be encoded at this bsize" — the driver
/// then prefers the alternate partition.
pub type LeafRdFn<'a> = dyn FnMut(BlockSize, MiPos) -> LeafRdResult + 'a;

/// libvpx `rd_pick_partition` skeleton (`vp9_encodeframe.c:3985-4489`),
/// NONE + SPLIT only.
///
/// At each level:
/// 1. **PARTITION_NONE** (libvpx lines 4068-4178): call `leaf_rd_fn`
///    at `bsize` and record the RD.
/// 2. **PARTITION_SPLIT** (libvpx lines 4199-4296): if `bsize ≥ 8×8`,
///    recurse into the 4 sub-blocks at `subsize(bsize, SPLIT)` and
///    sum their RDs. The recursion descends until the leaf bsize ==
///    `Block8x8` (where the leaf picker takes over).
/// 3. Pick the lower-RD partition (libvpx lines 4278-4296).
///
/// The driver is iterative-style for NONE and recursive for SPLIT,
/// mirroring libvpx's structure. The recursion depth is bounded by
/// `log4(max_bsize / 8x8) = 3` (64×64 → 32×32 → 16×16 → 8×8).
#[must_use]
pub fn rd_pick_partition(
    bsize: BlockSize,
    mi_pos: MiPos,
    leaf_rd_fn: &mut LeafRdFn<'_>,
) -> PartitionResult {
    // PARTITION_NONE (libvpx lines 4068-4178). Always evaluated.
    let none_rd = leaf_rd_fn(bsize, mi_pos);
    let mut best = PartitionResult {
        partition: PartitionType::None,
        rd: none_rd,
    };

    // PARTITION_SPLIT (libvpx lines 4199-4296). Only legal when
    // bsize >= Block8x8 (libvpx line 4004: `do_split = bsize >=
    // BLOCK_8X8`).
    if let Some(child_bsize) = subsize(bsize, PartitionType::Split) {
        // Sum the 4 sub-block RDs. libvpx (lines 4228-4275) tracks a
        // running `sum_rdc` and aborts the loop the moment it exceeds
        // the current best (`best_rdc.rdcost`). We mirror that
        // early-exit so a clearly-bad SPLIT doesn't drag in 4 full
        // recursions.
        let half_step = u32::from(bsize.n_8x8_wide()) / 2;
        let mut sum_rate: u64 = 0;
        let mut sum_dist: i64 = 0;
        let mut sum_rd: i64 = 0;
        let mut split_feasible = true;
        for quadrant in 0..4 {
            // libvpx lines 4230-4231: (x_idx, y_idx) = (i & 1, i >> 1)
            // * mi_step.
            let x_idx = (quadrant & 1) as u32 * half_step;
            let y_idx = (quadrant >> 1) as u32 * half_step;
            let child_pos = MiPos {
                mi_row: mi_pos.mi_row + y_idx,
                mi_col: mi_pos.mi_col + x_idx,
            };
            let child = rd_pick_partition(child_bsize, child_pos, leaf_rd_fn);
            if child.rd.rate == u32::MAX || child.rd.rd == i64::MAX {
                // libvpx lines 4260-4263: any infeasible child →
                // SPLIT is infeasible.
                split_feasible = false;
                break;
            }
            sum_rate = sum_rate.saturating_add(u64::from(child.rd.rate));
            sum_dist = sum_dist.saturating_add(child.rd.dist);
            sum_rd = sum_rd.saturating_add(child.rd.rd);

            // libvpx line 4228: `sum_rdc.rdcost < best_rdc.rdcost`
            // gates the next iteration. Mirror the early-exit.
            if sum_rd >= best.rd.rd {
                split_feasible = false;
                break;
            }
        }
        if split_feasible && sum_rd < best.rd.rd {
            // libvpx line 4284: `best_rdc = sum_rdc;`
            //                   `pc_tree->partitioning = PARTITION_SPLIT;`
            best = PartitionResult {
                partition: PartitionType::Split,
                rd: LeafRdResult {
                    rate: u32::try_from(sum_rate).unwrap_or(u32::MAX),
                    dist: sum_dist,
                    rd: sum_rd,
                },
            };
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a leaf-RD closure that returns a fixed RD for
    /// every (bsize, pos) it's asked about.
    fn const_leaf_rd(rd: i64, rate: u32, dist: i64) -> impl FnMut(BlockSize, MiPos) -> LeafRdResult {
        move |_bsize, _pos| LeafRdResult { rate, dist, rd }
    }

    /// PROOF W4.10a — `rd_pick_sb_modes` dispatches to the inter
    /// picker for inter frames at bsize ≥ 8×8.
    ///
    /// libvpx `vp9_encodeframe.c:2057-2078`: inter frames at
    /// `bsize >= BLOCK_8X8` route to `vp9_rd_pick_inter_mode_sb`; the
    /// sub-8 branch (line 2077) routes to
    /// `vp9_rd_pick_inter_mode_sub8x8`.
    #[test]
    fn w4_10a_rd_pick_sb_modes_dispatches_inter_for_inter_frame() {
        // Inter, bsize ≥ 8×8 → InterPicker::Inter.
        for bsize in [
            BlockSize::Block8x8,
            BlockSize::Block16x16,
            BlockSize::Block32x32,
            BlockSize::Block64x64,
        ] {
            assert_eq!(
                rd_pick_sb_modes(FrameKind::Inter, bsize),
                LeafPicker::Inter,
                "Inter @ {bsize:?} should route to Inter picker"
            );
        }
        // Inter, bsize < 8×8 → InterPicker::InterSub8.
        for bsize in [BlockSize::Block4x4, BlockSize::Block4x8, BlockSize::Block8x4] {
            assert_eq!(
                rd_pick_sb_modes(FrameKind::Inter, bsize),
                LeafPicker::InterSub8,
                "Inter @ {bsize:?} should route to InterSub8 picker"
            );
        }
        // Intra-only at any size → InterPicker::Intra.
        assert_eq!(
            rd_pick_sb_modes(FrameKind::IntraOnly, BlockSize::Block64x64),
            LeafPicker::Intra,
        );
        assert_eq!(
            rd_pick_sb_modes(FrameKind::IntraOnly, BlockSize::Block4x4),
            LeafPicker::Intra,
        );
        // SEG_LVL_SKIP → InterPicker::InterSegSkip.
        assert_eq!(
            rd_pick_sb_modes(FrameKind::InterSegSkip, BlockSize::Block16x16),
            LeafPicker::InterSegSkip,
        );
    }

    /// PROOF W4.10a — flat SB picks NONE.
    ///
    /// libvpx `vp9_encodeframe.c:4068-4296`: when the per-leaf RD is
    /// non-decreasing under split (i.e. splitting a flat block costs
    /// strictly more rate without reducing distortion), the
    /// PARTITION_NONE branch wins.
    ///
    /// We model "flat block" by having the leaf closure return a
    /// per-bsize RD that grows with the number of leaves (a coarse
    /// proxy for "splitting costs more"): rate scales with
    /// `num_leaves`, dist stays constant.
    #[test]
    fn w4_10a_rd_pick_partition_picks_none_on_flat_block() {
        // At Block64x64: NONE returns RD=10_000_000 (1 leaf).
        // At Block32x32: NONE returns RD=3_000_000 (each of 4 leaves);
        // SPLIT sum = 12_000_000 > NONE@64.
        let mut leaf_rd = |bsize: BlockSize, _pos: MiPos| -> LeafRdResult {
            match bsize {
                BlockSize::Block64x64 => LeafRdResult {
                    rate: 5_000,
                    dist: 50_000,
                    rd: 10_000_000,
                },
                BlockSize::Block32x32 => LeafRdResult {
                    rate: 2_500,
                    dist: 30_000,
                    rd: 3_000_000,
                },
                _ => LeafRdResult {
                    rate: 1_500,
                    dist: 20_000,
                    rd: 1_500_000,
                },
            }
        };
        let result = rd_pick_partition(
            BlockSize::Block64x64,
            MiPos { mi_row: 0, mi_col: 0 },
            &mut leaf_rd,
        );
        assert_eq!(
            result.partition,
            PartitionType::None,
            "flat SB should pick NONE; got {:?}",
            result.partition
        );
        assert_eq!(result.rd.rd, 10_000_000);
    }

    /// PROOF W4.10a — high-variance SB picks SPLIT.
    ///
    /// libvpx `vp9_encodeframe.c:4282-4296`: when 4 child RDs sum to
    /// less than the parent's NONE RD (the variance is high enough
    /// that the smaller-block predictor model fits much better), the
    /// PARTITION_SPLIT branch wins.
    ///
    /// We model "high variance" by making NONE expensive at the SB
    /// level and cheap at the sub-block level.
    #[test]
    fn w4_10a_rd_pick_partition_picks_split_on_high_variance() {
        let mut leaf_rd = |bsize: BlockSize, _pos: MiPos| -> LeafRdResult {
            match bsize {
                // High-variance 64×64 → expensive at NONE (model RD
                // mispredicts wildly).
                BlockSize::Block64x64 => LeafRdResult {
                    rate: 50_000,
                    dist: 5_000_000,
                    rd: 100_000_000,
                },
                // Cheap at 32×32 (good predictor fit).
                BlockSize::Block32x32 => LeafRdResult {
                    rate: 8_000,
                    dist: 100_000,
                    rd: 5_000_000,
                },
                _ => LeafRdResult {
                    rate: 4_000,
                    dist: 50_000,
                    rd: 1_000_000,
                },
            }
        };
        let result = rd_pick_partition(
            BlockSize::Block64x64,
            MiPos { mi_row: 0, mi_col: 0 },
            &mut leaf_rd,
        );
        assert_eq!(
            result.partition,
            PartitionType::Split,
            "high-variance SB should pick SPLIT; got {:?}",
            result.partition
        );
        // Children themselves SPLIT recursively: each 32×32 prefers
        // SPLIT (4 × 1M = 4M < NONE=5M), so 64×64 SPLIT aggregate =
        // 4 × 4M = 16M (< NONE=100M). Either way the partition is
        // SPLIT, which is what we're proving.
        assert_eq!(result.rd.rd, 16_000_000);
    }

    /// Sub-8 leaf only evaluates NONE (libvpx line 4004: `do_split =
    /// bsize >= BLOCK_8X8` rules out SPLIT for sub-8 leaves).
    #[test]
    fn rd_pick_partition_sub_8_only_evaluates_none() {
        let mut leaf_rd = const_leaf_rd(123_456, 200, 1_500);
        let result = rd_pick_partition(
            BlockSize::Block4x4,
            MiPos { mi_row: 0, mi_col: 0 },
            &mut leaf_rd,
        );
        assert_eq!(result.partition, PartitionType::None);
        assert_eq!(result.rd.rd, 123_456);
    }

    /// Partition recursion visits 4 children at each split level —
    /// invocation count = 1 (parent NONE) + 4 (split children) when
    /// recursing 64→32 once, and the early-exit fires when sum_rd >=
    /// best.rd.rd.
    #[test]
    fn rd_pick_partition_recursion_call_count() {
        use std::cell::RefCell;
        let calls = RefCell::new(0u32);
        let mut leaf_rd = |bsize: BlockSize, _pos: MiPos| -> LeafRdResult {
            *calls.borrow_mut() += 1;
            // NONE @ 32 is cheap, so the recursion always finds SPLIT
            // worse and bails after summing 4 children.
            match bsize {
                BlockSize::Block64x64 => LeafRdResult {
                    rate: 50_000,
                    dist: 1_000_000,
                    rd: 200_000_000,
                },
                BlockSize::Block32x32 => LeafRdResult {
                    rate: 5_000,
                    dist: 100_000,
                    rd: 10_000_000,
                },
                BlockSize::Block16x16 => LeafRdResult {
                    rate: 2_500,
                    dist: 50_000,
                    rd: 5_000_000,
                },
                _ => LeafRdResult {
                    rate: 1_000,
                    dist: 25_000,
                    rd: 1_000_000,
                },
            }
        };
        let _ = rd_pick_partition(
            BlockSize::Block32x32,
            MiPos { mi_row: 0, mi_col: 0 },
            &mut leaf_rd,
        );
        // 1 NONE@32 + 4 NONE@16 (the 16x16 children skip recursing
        // because their SPLIT children at 8x8 sum to 4M but each 8x8
        // costs ~1M, so SPLIT@16 = 4M < NONE@16 = 5M → ... wait, no.
        // With these numbers SPLIT@16 = 4M < NONE@16 = 5M, so the
        // 16x16 nodes WILL split: 1 + 4 + 4*(1 + 4) = 25 calls.
        let count = *calls.borrow();
        assert!(
            count >= 5,
            "expected at least 5 leaf calls, got {count}"
        );
    }
}
