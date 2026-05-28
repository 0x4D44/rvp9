//! MV search-range bounds + clamp helpers.
//!
//! Mirrors libvpx `MvLimits` (`vp9/encoder/vp9_block.h:58-63`) and the
//! `vp9_set_mv_search_range` / `vp9_set_subpel_mv_search_range` helpers
//! (`vp9/encoder/vp9_mcomp.c:32-67`).

/// Maximum number of step-search levels (`vp9_mcomp.h:26`).
pub const MAX_MVSEARCH_STEPS: u32 = 11;

/// Maximum first step in full pixels (`vp9_mcomp.h:31`). `= 1024`.
pub const MAX_FIRST_STEP: i32 = 1 << (MAX_MVSEARCH_STEPS - 1);

/// Maximum full-pel MV magnitude (`vp9_mcomp.h:29`). `= 1023`.
pub const MAX_FULL_PEL_VAL: i32 = MAX_FIRST_STEP - 1;

/// Lowest encodable 1/8-pel MV component value
/// (`vp9_entropymv.h:80`). `= -16384`.
pub const MV_LOW: i32 = -(1 << 14);

/// Highest encodable 1/8-pel MV component value (`vp9_entropymv.h:79`).
/// `= 16383`.
pub const MV_UPP: i32 = (1 << 14) - 1;

/// Full-pel-unit MV search bounds. Mirrors libvpx `MvLimits`
/// (`vp9_block.h:58-63`); fields are `i32` to match the libvpx C
/// struct layout exactly. For sub-pel search, the same struct is
/// reused in 1/8-pel units (`vp9_set_subpel_mv_search_range`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MvLimits {
    /// Lowest legal column (inclusive).
    pub col_min: i32,
    /// Highest legal column (inclusive).
    pub col_max: i32,
    /// Lowest legal row (inclusive).
    pub row_min: i32,
    /// Highest legal row (inclusive).
    pub row_max: i32,
}

impl Default for MvLimits {
    /// Wide-open limits: `[-MAX_FULL_PEL_VAL, MAX_FULL_PEL_VAL]` on
    /// both axes. The encoder usually narrows this per-block via
    /// [`set_mv_search_range`].
    fn default() -> Self {
        Self {
            col_min: -MAX_FULL_PEL_VAL,
            col_max: MAX_FULL_PEL_VAL,
            row_min: -MAX_FULL_PEL_VAL,
            row_max: MAX_FULL_PEL_VAL,
        }
    }
}

impl MvLimits {
    /// Construct explicit limits.
    #[inline]
    #[must_use]
    pub const fn new(col_min: i32, col_max: i32, row_min: i32, row_max: i32) -> Self {
        Self {
            col_min,
            col_max,
            row_min,
            row_max,
        }
    }
}

/// `vp9_set_mv_search_range` (`vp9_mcomp.c:32-49`). Narrows the
/// caller-provided `mv_limits` (full-pel units) to the intersection
/// of:
///
/// - the existing window (typically the per-block UMV window), and
/// - a `MAX_FULL_PEL_VAL`-radius ball around the reference MV.
///
/// `ref_mv` is in 1/8-pel units; this function converts to full-pel
/// with the libvpx rounding (`(v >> 3) + (v & 7 != 0 ? 1 : 0)` for
/// the lower bound, plain `v >> 3` for the upper bound).
pub fn set_mv_search_range(mv_limits: &mut MvLimits, ref_mv_row: i16, ref_mv_col: i16) {
    let row = i32::from(ref_mv_row);
    let col = i32::from(ref_mv_col);

    let bump_col = if (col & 7) != 0 { 1 } else { 0 };
    let bump_row = if (row & 7) != 0 { 1 } else { 0 };

    let mut col_min = (col >> 3) - MAX_FULL_PEL_VAL + bump_col;
    let mut row_min = (row >> 3) - MAX_FULL_PEL_VAL + bump_row;
    let mut col_max = (col >> 3) + MAX_FULL_PEL_VAL;
    let mut row_max = (row >> 3) + MAX_FULL_PEL_VAL;

    col_min = col_min.max((MV_LOW >> 3) + 1);
    row_min = row_min.max((MV_LOW >> 3) + 1);
    col_max = col_max.min((MV_UPP >> 3) - 1);
    row_max = row_max.min((MV_UPP >> 3) - 1);

    // Intersect with the existing window.
    if mv_limits.col_min < col_min {
        mv_limits.col_min = col_min;
    }
    if mv_limits.col_max > col_max {
        mv_limits.col_max = col_max;
    }
    if mv_limits.row_min < row_min {
        mv_limits.row_min = row_min;
    }
    if mv_limits.row_max > row_max {
        mv_limits.row_max = row_max;
    }
}

/// `vp9_set_subpel_mv_search_range` (`vp9_mcomp.c:51-67`). Derives a
/// sub-pel (1/8-pel-units) window from the full-pel `umv_window` and
/// the reference MV.
pub fn set_subpel_mv_search_range(
    subpel: &mut MvLimits,
    umv_window: &MvLimits,
    ref_mv_row: i16,
    ref_mv_col: i16,
) {
    let row = i32::from(ref_mv_row);
    let col = i32::from(ref_mv_col);

    subpel.col_min = (umv_window.col_min * 8).max(col - MAX_FULL_PEL_VAL * 8);
    subpel.col_max = (umv_window.col_max * 8).min(col + MAX_FULL_PEL_VAL * 8);
    subpel.row_min = (umv_window.row_min * 8).max(row - MAX_FULL_PEL_VAL * 8);
    subpel.row_max = (umv_window.row_max * 8).min(row + MAX_FULL_PEL_VAL * 8);

    subpel.col_min = subpel.col_min.max(MV_LOW + 1);
    subpel.col_max = subpel.col_max.min(MV_UPP - 1);
    subpel.row_min = subpel.row_min.max(MV_LOW + 1);
    subpel.row_max = subpel.row_max.min(MV_UPP - 1);
}

/// `is_mv_in` (`vp9_mcomp.c:938-941`). Returns `true` when both
/// components of `(row, col)` lie inside `mv_limits` (inclusive on
/// both ends — matches the C source `<= max`/`>= min` checks).
#[inline]
#[must_use]
pub fn is_mv_in(mv_limits: &MvLimits, row: i32, col: i32) -> bool {
    col >= mv_limits.col_min
        && col <= mv_limits.col_max
        && row >= mv_limits.row_min
        && row <= mv_limits.row_max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_wide_open() {
        let l = MvLimits::default();
        assert_eq!(l.col_min, -MAX_FULL_PEL_VAL);
        assert_eq!(l.col_max, MAX_FULL_PEL_VAL);
        assert_eq!(l.row_min, -MAX_FULL_PEL_VAL);
        assert_eq!(l.row_max, MAX_FULL_PEL_VAL);
    }

    #[test]
    fn is_mv_in_boundary_inclusive() {
        let l = MvLimits::new(-10, 10, -5, 5);
        assert!(is_mv_in(&l, 0, 0));
        assert!(is_mv_in(&l, 5, 10));
        assert!(is_mv_in(&l, -5, -10));
        // Outside on each axis.
        assert!(!is_mv_in(&l, 6, 0));
        assert!(!is_mv_in(&l, -6, 0));
        assert!(!is_mv_in(&l, 0, 11));
        assert!(!is_mv_in(&l, 0, -11));
    }

    #[test]
    fn set_mv_search_range_narrows() {
        // Start with the wide-open limits: ±MAX_FULL_PEL_VAL = ±1023.
        // libvpx's `set_mv_search_range` INTERSECTS the existing
        // window with the ref-MV-centred ball: it only narrows, never
        // widens. For `col=-8` (full-pel = -1):
        //   candidate col_min = -1 - 1023 = -1024 (looser than -1023 → unchanged)
        //   candidate col_max = -1 + 1023 = 1022 (tighter than 1023 → narrows)
        let mut l = MvLimits::default();
        set_mv_search_range(&mut l, 8, -8);
        assert_eq!(l.col_min, -MAX_FULL_PEL_VAL);
        assert_eq!(l.col_max, -1 + MAX_FULL_PEL_VAL); // 1022
        assert_eq!(l.row_min, 1 - MAX_FULL_PEL_VAL); // -1022
        assert_eq!(l.row_max, MAX_FULL_PEL_VAL);
    }

    #[test]
    fn set_mv_search_range_bump_for_nonzero_phase() {
        // Non-zero fractional bits in `col` shift the candidate
        // `col_min` up by 1 (the libvpx ceil-on-min behaviour).
        // col = 9 (full-pel = 1, &7 = 1 → bump):
        //   candidate col_min = 1 - 1023 + 1 = -1021 (tighter → narrows)
        //   candidate col_max = 1 + 1023 = 1024  (looser → unchanged)
        let mut l = MvLimits::default();
        set_mv_search_range(&mut l, 0, 9);
        assert_eq!(l.col_min, 1 - MAX_FULL_PEL_VAL + 1); // -1021
        assert_eq!(l.col_max, MAX_FULL_PEL_VAL);
        assert_eq!(l.row_min, -MAX_FULL_PEL_VAL);
        assert_eq!(l.row_max, MAX_FULL_PEL_VAL);
    }

    #[test]
    fn set_subpel_mv_search_range_derives_1_8_pel_window() {
        let umv = MvLimits::new(-100, 100, -50, 50);
        let mut subpel = MvLimits::default();
        set_subpel_mv_search_range(&mut subpel, &umv, 0, 0);
        // ref_mv = (0, 0), umv * 8 vs ref ± MAX_FULL_PEL_VAL * 8.
        // col_min = max(-100*8, 0 - 1023*8) = max(-800, -8184) = -800.
        assert_eq!(subpel.col_min, -800);
        assert_eq!(subpel.col_max, 800);
        assert_eq!(subpel.row_min, -400);
        assert_eq!(subpel.row_max, 400);
    }

    #[test]
    fn set_subpel_mv_search_range_clamps_to_encodable_range() {
        // Build a UMV window that maps to values outside [MV_LOW+1, MV_UPP-1].
        let umv = MvLimits::new(i32::MIN / 16, i32::MAX / 16, i32::MIN / 16, i32::MAX / 16);
        let mut subpel = MvLimits::default();
        set_subpel_mv_search_range(&mut subpel, &umv, 0, 0);
        assert!(subpel.col_min > MV_LOW);
        assert!(subpel.col_max < MV_UPP);
        assert!(subpel.row_min > MV_LOW);
        assert!(subpel.row_max < MV_UPP);
    }
}
