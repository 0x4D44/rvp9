//! Port of `vp9_mv_pred` (`vp9_rd.c:588-639`).
//!
//! Evaluates SAD at each MV-ref candidate and returns the best one as
//! the NEWMV diamond-search seed. When the encoder has multiple MV
//! candidates (nearest, near, pred_mv), starting the diamond at the
//! one with the lowest SAD dramatically improves NEWMV search quality
//! — the diamond is a local search, so a better starting point means
//! a better result.
//!
//! The function takes candidates in **1/8-pel** units (as produced by
//! `find_mv_refs`), rounds each to full-pel for SAD evaluation using
//! the same biased-rounding formula as libvpx, and returns the winner
//! in 1/8-pel units (suitable for direct use as `start_mv` in the
//! diamond search).

use crate::block::BlockSize;
use crate::dsp::dispatch::SadFn;
use crate::mode_info::Mv;
use crate::mv_search::full_pixel::{PlaneRef, raw_sad_at_mv_with_fn};

/// Round a 1/8-pel MV component to full-pel using libvpx's biased
/// rounding: `(val + 3 + (val >= 0)) >> 3`. This rounds toward
/// the nearest full-pel position (positive bias for positive values,
/// floor+3 bias for negative values).
#[inline]
fn q3_to_full_pel(val: i16) -> i16 {
    let bias = if val >= 0 { 4 } else { 3 };
    (val.wrapping_add(bias)) >> 3
}

/// Evaluate SAD at each MV-ref candidate and return the best seed for
/// the NEWMV diamond search.
///
/// Port of `vp9_mv_pred` (`vp9_rd.c:588-639`). The returned MV is in
/// **1/8-pel** units (the full-pel-rounded position, scaled back to Q3)
/// so it can be passed directly as `start_mv` to the diamond search.
///
/// # Arguments
///
/// * `src_plane` / `src_offset` — source block location.
/// * `ref_plane` — reference frame plane.
/// * `mv_candidates` — slice of MV candidates in 1/8-pel units
///   (typically `[nearest, near]` from `find_mv_refs`).
/// * `block_size` — block dimensions for the SAD kernel.
///
/// # Returns
///
/// `(best_mv, best_sad)` — the candidate MV (1/8-pel) with the lowest
/// SAD, and that SAD value. Returns `(Mv::ZERO, u32::MAX)` if no
/// candidate could be evaluated (all OOB).
#[must_use]
pub fn mv_pred_sad_seed(
    src_plane: &PlaneRef<'_>,
    ref_plane: &PlaneRef<'_>,
    src_offset: (usize, usize),
    mv_candidates: &[Mv],
    block_size: BlockSize,
) -> (Mv, u32) {
    let mut best_mv = Mv::ZERO;
    let mut best_sad = u32::MAX;
    let mut zero_seen = false;
    let sad_fn: SadFn = crate::dsp::get_ref().sad(block_size);

    for (i, cand) in mv_candidates.iter().enumerate() {
        // Skip INT16_MAX sentinel values (libvpx convention for
        // "no candidate").
        if cand.row == i16::MAX || cand.col == i16::MAX {
            continue;
        }

        // Round 1/8-pel to full-pel.
        let fp_row = q3_to_full_pel(cand.row);
        let fp_col = q3_to_full_pel(cand.col);

        // Skip duplicate zero-MV evaluations (libvpx lines 621-622).
        if fp_row == 0 && fp_col == 0 {
            if zero_seen {
                continue;
            }
            zero_seen = true;
        }

        // Skip if candidate #1 equals candidate #0 (near == nearest),
        // mirroring libvpx line 616.
        if i == 1 && mv_candidates.len() > 1 {
            let prev_fp_row = q3_to_full_pel(mv_candidates[0].row);
            let prev_fp_col = q3_to_full_pel(mv_candidates[0].col);
            if fp_row == prev_fp_row && fp_col == prev_fp_col {
                continue;
            }
        }

        // Evaluate SAD. `raw_sad_at_mv` returns `None` if the
        // reference block falls outside the plane — skip that
        // candidate.
        if let Some(sad) = raw_sad_at_mv_with_fn(
            block_size, sad_fn, src_plane, src_offset, ref_plane, fp_row, fp_col,
        ) && sad < best_sad
        {
            best_sad = sad;
            best_mv = Mv {
                row: fp_row * 8,
                col: fp_col * 8,
            };
        }
    }

    (best_mv, best_sad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a flat plane filled with `val`.
    fn make_plane(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    /// When the source and reference are identical, SAD at (0,0)
    /// should be zero and win trivially.
    #[test]
    fn identical_planes_zero_mv_wins() {
        let w = 32;
        let h = 32;
        let data = make_plane(w, h, 128);
        let plane = PlaneRef {
            data: &data,
            stride: w,
            width: w,
            height: h,
        };
        let candidates = [
            Mv { row: 0, col: 0 },
            Mv { row: 8, col: 8 }, // 1 full-pel down-right
        ];
        let (best, sad) =
            mv_pred_sad_seed(&plane, &plane, (0, 0), &candidates, BlockSize::Block8x8);
        assert_eq!(sad, 0);
        assert_eq!(best, Mv::ZERO);
    }

    /// When no candidates are evaluable (all OOB), returns the
    /// default `(ZERO, MAX)`.
    #[test]
    fn all_oob_returns_default() {
        let w = 8;
        let h = 8;
        let data = make_plane(w, h, 100);
        let plane = PlaneRef {
            data: &data,
            stride: w,
            width: w,
            height: h,
        };
        // Candidates that point far outside the 8x8 plane.
        let candidates = [Mv { row: 800, col: 0 }, Mv { row: 0, col: 800 }];
        let (best, sad) =
            mv_pred_sad_seed(&plane, &plane, (0, 0), &candidates, BlockSize::Block8x8);
        assert_eq!(best, Mv::ZERO);
        assert_eq!(sad, u32::MAX);
    }

    /// Sentinel values (i16::MAX) are skipped.
    #[test]
    fn sentinel_values_skipped() {
        let w = 16;
        let h = 16;
        let data = make_plane(w, h, 42);
        let plane = PlaneRef {
            data: &data,
            stride: w,
            width: w,
            height: h,
        };
        let candidates = [
            Mv {
                row: i16::MAX,
                col: i16::MAX,
            },
            Mv { row: 0, col: 0 },
        ];
        let (best, sad) =
            mv_pred_sad_seed(&plane, &plane, (0, 0), &candidates, BlockSize::Block8x8);
        assert_eq!(sad, 0);
        assert_eq!(best, Mv::ZERO);
    }

    /// The candidate with lower SAD wins.
    #[test]
    fn lower_sad_candidate_wins() {
        // Source: 16x16 plane filled with 100.
        let w = 32;
        let h = 32;
        let src_data = make_plane(w, h, 100);
        let src = PlaneRef {
            data: &src_data,
            stride: w,
            width: w,
            height: h,
        };

        // Reference: mostly filled with 200 (high SAD at most positions),
        // but with an 8x8 block at (8, 8) filled with 100 (zero SAD there).
        let mut ref_data = make_plane(w, h, 200);
        for r in 8..16 {
            for c in 8..16 {
                ref_data[r * w + c] = 100;
            }
        }
        let refp = PlaneRef {
            data: &ref_data,
            stride: w,
            width: w,
            height: h,
        };

        // Candidate 0: (0,0) → high SAD.
        // Candidate 1: (8,8) in full-pel → (64,64) in Q3.
        let candidates = [
            Mv { row: 0, col: 0 },
            Mv { row: 64, col: 64 }, // 8 full-pel → 64 Q3
        ];
        let (best, sad) = mv_pred_sad_seed(&src, &refp, (0, 0), &candidates, BlockSize::Block8x8);
        assert_eq!(sad, 0);
        assert_eq!(best, Mv { row: 64, col: 64 });
    }

    /// Q3-to-full-pel rounding matches libvpx formula.
    #[test]
    fn q3_rounding() {
        // Positive: (val + 4) >> 3
        assert_eq!(q3_to_full_pel(0), 0);
        assert_eq!(q3_to_full_pel(8), 1); // exact full-pel
        assert_eq!(q3_to_full_pel(4), 1); // 0.5 → rounds up
        assert_eq!(q3_to_full_pel(3), 0); // 0.375 → rounds down
        assert_eq!(q3_to_full_pel(7), 1); // 0.875 → rounds up
        assert_eq!(q3_to_full_pel(16), 2);

        // Negative: (val + 3) >> 3 — arithmetic shift rounds toward -inf.
        assert_eq!(q3_to_full_pel(-8), -1); // -1.0 → -1
        assert_eq!(q3_to_full_pel(-4), -1); // -0.5 → -1 (not toward zero)
        assert_eq!(q3_to_full_pel(-5), -1); // -0.625 → -1
        assert_eq!(q3_to_full_pel(-6), -1); // -0.75 → -1
        assert_eq!(q3_to_full_pel(-16), -2); // -2.0 → -2
        assert_eq!(q3_to_full_pel(-1), 0); // -0.125 → 0 ((-1+3)>>3 = 0)
        assert_eq!(q3_to_full_pel(-3), 0); // -0.375 → 0 ((-3+3)>>3 = 0)
    }

    /// Empty candidate list returns default.
    #[test]
    fn empty_candidates() {
        let w = 16;
        let h = 16;
        let data = make_plane(w, h, 50);
        let plane = PlaneRef {
            data: &data,
            stride: w,
            width: w,
            height: h,
        };
        let (best, sad) = mv_pred_sad_seed(&plane, &plane, (0, 0), &[], BlockSize::Block8x8);
        assert_eq!(best, Mv::ZERO);
        assert_eq!(sad, u32::MAX);
    }
}
