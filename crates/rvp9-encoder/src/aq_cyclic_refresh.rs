//! Cyclic-refresh AQ — libvpx-inspired port of
//! `vp9/encoder/vp9_aq_cyclicrefresh.c`.
//!
//! Refreshes ~1/N of the super-blocks per frame at a lower Q. Cycles
//! through all SBs over N frames so every block gets re-quantized at
//! finer Q periodically — keeps quality even when the rate-control
//! base Q is high.
//!
//! ## Algorithm (one frame loop)
//!
//! 1. **`pre_encode_pick_segments`** — walk the SB grid from
//!    `sb_cursor` forward (wrapping), picking SBs whose every cell has
//!    `map[i] >= config.min_refresh_age` to mark as segment 1 (boost).
//!    Stops once we've picked the target fraction of cells.
//! 2. **Caller encodes** with the picked segmentation_map. SBs in
//!    segment 1 get an `alt_q` delta (negative ⇒ finer Q ⇒ higher
//!    quality).
//! 3. **`post_encode_update`** — for cells that were actually coded as
//!    segment 1 AND not skipped, reset map age to 0 (just refreshed).
//!    For all other cells, increment age (saturating at 255).
//!
//! ## Divergences from libvpx
//!
//! Libvpx encodes "eligibility" with `map[i] == 0` and "recently
//! refreshed" as `map[i] < 0` (counter ticking up toward 0). Plus a
//! `last_coded_q_map` to track per-cell quality history. We use a
//! simpler "age in cycles since last refresh" semantic — eligibility
//! is `age >= min_refresh_age`. The libvpx feedback loops
//! (rate-correction, scene-change detection, `consec_zero_mv`,
//! `noise_estimate`, BOOST2 segment for big zero-MV blocks) are NOT
//! ported in v1; we only emit segment 1 (BOOST1 in libvpx
//! nomenclature) with a fixed config Q-delta.
//!
//! ## v1 scope (this module)
//!
//! Pure state machine: cell-age tracking + cursor-driven SB picking.
//! The actual segmentation-map plumbing into [`SegmentationParams`]
//! (the `feature_data[seg][SEG_LVL_ALT_Q]` table) and the frame-loop
//! call into `pre_encode_pick_segments`/`post_encode_update` is wired
//! by **AQ.3** (frame-loop integration).
//!
//! [`SegmentationParams`]: rvp9_core::bitstream::sub_parsers::SegmentationParams

/// VP9 super-block side in 8x8 mi units (64/8 = 8).
const MI_PER_SB: usize = 8;

/// Boost segment id used by cyclic refresh (libvpx
/// `CR_SEGMENT_ID_BOOST1 = 1`).
pub const CR_SEGMENT_ID_BOOST: u8 = 1;

/// Configuration knobs for cyclic refresh.
///
/// All fields are read once per frame; mutating between frames is
/// allowed (the cursor + map state survive).
#[derive(Debug, Clone, Copy)]
pub struct CyclicRefreshConfig {
    /// Fraction of 8x8 cells to refresh per frame, in 1/256ths
    /// (libvpx convention from `vp9_segmentation_probs.c` weighting).
    /// E.g. `25` ≈ 10%, `64` = 25%. Libvpx default ~10% =
    /// `target ≈ 25/256`; see `vp9_aq_cyclicrefresh.c:506`
    /// (`cr->percent_refresh = 10`).
    pub target_refresh_fraction_256: u32,
    /// Q-delta applied to refreshed (segment 1) SBs via
    /// `SegmentationParams.feature_data[1][SEG_LVL_ALT_Q]`. Negative ⇒
    /// finer quantization ⇒ higher quality. Libvpx-typical −5..−15
    /// depending on `rate_ratio_qdelta` and base Q
    /// (`vp9_aq_cyclicrefresh.c:656`).
    pub segment1_q_delta: i16,
    /// Minimum age (in post-encode cycles) before a cell is eligible
    /// for refresh. Libvpx default is implicitly 0 (any cell with
    /// `map[i] == 0` is eligible), but the `time_for_refresh` mechanism
    /// (`vp9_aq_cyclicrefresh.c:509` — set to 0 in default config,
    /// raised under specific conditions) plays a similar role. Using
    /// a positive value here ensures every cell waits at least N
    /// cycles between refreshes — avoids hammering the same cells.
    pub min_refresh_age: u8,
}

impl Default for CyclicRefreshConfig {
    fn default() -> Self {
        Self {
            // 25/256 ≈ 9.77% — close to libvpx's `percent_refresh = 10`.
            target_refresh_fraction_256: 25,
            // libvpx-typical at mid base-Q under `rate_ratio_qdelta = 2.0`.
            segment1_q_delta: -8,
            // Wait 4 post-encode cycles before re-refreshing a cell.
            min_refresh_age: 4,
        }
    }
}

/// Cyclic-refresh state. One per encoder instance.
///
/// Construct via [`CyclicRefresh::new`]; drive per frame via:
///
/// 1. [`CyclicRefresh::pre_encode_pick_segments`] — pick which cells
///    get segment 1 for this frame's segmentation map.
/// 2. The frame loop encodes using that segmentation map.
/// 3. [`CyclicRefresh::post_encode_update`] — feed the coded segment
///    ids + skip flags back to age the map.
///
/// The map / cursor survive across frames — they're the state-machine
/// memory.
#[derive(Debug, Clone)]
pub struct CyclicRefresh {
    /// Per-mi-cell map: age in post-encode cycles since the cell was
    /// last refreshed. 0 = just refreshed (or never refreshed since
    /// init), 255 = aged-out cap (saturating).
    pub map: Vec<u8>,
    /// SB cursor (raster order over the SB grid). Walk resumes here
    /// on the next `pre_encode_pick_segments`.
    pub sb_cursor: usize,
    /// Frame height in 8x8 mi units.
    pub mi_rows: usize,
    /// Frame width in 8x8 mi units.
    pub mi_cols: usize,
    /// Configuration.
    pub config: CyclicRefreshConfig,
}

impl CyclicRefresh {
    /// Allocate fresh cyclic-refresh state for an `mi_rows × mi_cols`
    /// frame. All map cells start at age 0 (just-refreshed) — i.e. no
    /// cell is eligible for refresh until at least `min_refresh_age`
    /// post-encode calls have aged them. Mirrors libvpx
    /// `vp9_cyclic_refresh_alloc`'s zero-init.
    #[must_use]
    pub fn new(mi_rows: usize, mi_cols: usize, config: CyclicRefreshConfig) -> Self {
        Self {
            map: vec![0u8; mi_rows * mi_cols],
            sb_cursor: 0,
            mi_rows,
            mi_cols,
            config,
        }
    }

    /// Number of super-block columns in the frame (ceiling).
    #[must_use]
    pub fn sb_cols(&self) -> usize {
        self.mi_cols.div_ceil(MI_PER_SB)
    }

    /// Number of super-block rows in the frame (ceiling).
    #[must_use]
    pub fn sb_rows(&self) -> usize {
        self.mi_rows.div_ceil(MI_PER_SB)
    }

    /// Total super-blocks in the frame.
    #[must_use]
    pub fn sbs_in_frame(&self) -> usize {
        self.sb_cols() * self.sb_rows()
    }

    /// Target number of 8x8 cells to refresh this frame, derived from
    /// `target_refresh_fraction_256` × frame area / 256.
    #[must_use]
    pub fn target_refresh_cells(&self) -> usize {
        let total = self.mi_rows * self.mi_cols;
        (total * (self.config.target_refresh_fraction_256 as usize)) / 256
    }

    /// Pick the segmentation map for the next frame.
    ///
    /// Walks the SB grid starting at `sb_cursor`. For each SB:
    /// - if **every** 8x8 cell inside the SB has aged to at least
    ///   `min_refresh_age`, mark all cells in the SB as
    ///   [`CR_SEGMENT_ID_BOOST`] (segment 1);
    /// - otherwise leave them as segment 0 (baseline).
    ///
    /// Stops once we've marked at least `target_refresh_cells` cells
    /// OR walked the whole frame back to the starting cursor. Returns
    /// `(segmentation_map of length mi_rows * mi_cols, n_refreshed_mi_cells)`.
    ///
    /// Mirrors libvpx `cyclic_refresh_update_map` (the loop body's
    /// "set segment 1 if at least half the cells are eligible" rule;
    /// we tighten this to "all cells eligible" because we don't have
    /// the libvpx noise / consec_zero_mv signals to qualify partial
    /// eligibility).
    ///
    /// **Note:** this method does NOT mutate `sb_cursor` — that
    /// happens after the frame is coded, in [`Self::post_encode_update`].
    /// This lets the caller call `pre_encode_pick_segments` multiple
    /// times (e.g. for rate-control bit estimation) without
    /// double-advancing the cursor.
    #[must_use]
    pub fn pre_encode_pick_segments(&self) -> (Vec<u8>, usize) {
        let n_cells = self.mi_rows * self.mi_cols;
        let mut seg_map = vec![0u8; n_cells];
        let mut n_refreshed: usize = 0;

        let target = self.target_refresh_cells();
        let sbs = self.sbs_in_frame();
        if sbs == 0 || target == 0 {
            return (seg_map, 0);
        }

        let sb_cols = self.sb_cols();
        // Walk SBs starting at sb_cursor; stop after a full lap or
        // once we've hit the target.
        for step in 0..sbs {
            let sb_idx = (self.sb_cursor + step) % sbs;
            let sb_row = sb_idx / sb_cols;
            let sb_col = sb_idx - sb_row * sb_cols;
            let mi_row0 = sb_row * MI_PER_SB;
            let mi_col0 = sb_col * MI_PER_SB;
            let mi_row_end = (mi_row0 + MI_PER_SB).min(self.mi_rows);
            let mi_col_end = (mi_col0 + MI_PER_SB).min(self.mi_cols);

            // Eligibility: every cell in this SB has aged enough.
            let mut eligible = true;
            'sb_check: for r in mi_row0..mi_row_end {
                for c in mi_col0..mi_col_end {
                    if self.map[r * self.mi_cols + c] < self.config.min_refresh_age {
                        eligible = false;
                        break 'sb_check;
                    }
                }
            }

            if eligible {
                let mut sb_cells: usize = 0;
                for r in mi_row0..mi_row_end {
                    for c in mi_col0..mi_col_end {
                        seg_map[r * self.mi_cols + c] = CR_SEGMENT_ID_BOOST;
                        sb_cells += 1;
                    }
                }
                n_refreshed += sb_cells;
                if n_refreshed >= target {
                    break;
                }
            }
        }

        (seg_map, n_refreshed)
    }

    /// Post-encode map update.
    ///
    /// For each 8x8 cell:
    /// - If it was coded as [`CR_SEGMENT_ID_BOOST`] **and** `skip ==
    ///   false` ⇒ it was genuinely re-quantized at finer Q; reset
    ///   `map[i] = 0`.
    /// - Otherwise (baseline OR boosted-but-skipped) ⇒ increment
    ///   `map[i]` (saturating at 255). Aging baseline cells eventually
    ///   pushes them past `min_refresh_age` so a future
    ///   `pre_encode_pick_segments` will pick their SB.
    ///
    /// Also advances `sb_cursor` past the SBs that were refreshed so
    /// the next frame starts after the most-recent boundary —
    /// implementing the "cycle through frame" mechanic.
    ///
    /// The skip-exclusion mirrors libvpx
    /// `vp9_cyclic_refresh_update_sb_postencode` (lines 243–252 in
    /// `vp9_aq_cyclicrefresh.c`): inter-skip blocks were not actually
    /// coded at the boosted Q, so refreshing the per-cell quality
    /// history would be incorrect. Libvpx encodes this by NOT
    /// touching `last_coded_q_map`; we encode it by treating the cell
    /// as baseline for age-reset purposes.
    ///
    /// # Panics
    ///
    /// Panics if `coded_segments.len() != mi_rows * mi_cols` or
    /// `coded_skip.len() != mi_rows * mi_cols`.
    pub fn post_encode_update(&mut self, coded_segments: &[u8], coded_skip: &[bool]) {
        let n_cells = self.mi_rows * self.mi_cols;
        assert_eq!(
            coded_segments.len(),
            n_cells,
            "coded_segments must be mi_rows * mi_cols",
        );
        assert_eq!(
            coded_skip.len(),
            n_cells,
            "coded_skip must be mi_rows * mi_cols",
        );

        let mut last_refreshed_sb: Option<usize> = None;
        let sb_cols = self.sb_cols();

        for i in 0..n_cells {
            let actually_refreshed = coded_segments[i] == CR_SEGMENT_ID_BOOST && !coded_skip[i];
            if actually_refreshed {
                self.map[i] = 0;
                let mi_row = i / self.mi_cols;
                let mi_col = i - mi_row * self.mi_cols;
                let sb_row = mi_row / MI_PER_SB;
                let sb_col = mi_col / MI_PER_SB;
                let sb_idx = sb_row * sb_cols + sb_col;
                // Track the highest-cursor-distance refreshed SB so we
                // can advance the cursor past it. Use the wrap-aware
                // distance from current cursor.
                last_refreshed_sb = Some(match last_refreshed_sb {
                    None => sb_idx,
                    Some(prev) => {
                        let sbs = self.sbs_in_frame();
                        let dist_new = (sb_idx + sbs - self.sb_cursor) % sbs;
                        let dist_prev = (prev + sbs - self.sb_cursor) % sbs;
                        if dist_new > dist_prev { sb_idx } else { prev }
                    }
                });
            } else {
                self.map[i] = self.map[i].saturating_add(1);
            }
        }

        // Advance the cursor to one past the furthest refreshed SB so
        // the next frame picks up where we left off. If nothing was
        // refreshed, leave the cursor in place (rare — typically means
        // no cells were eligible).
        if let Some(sb_idx) = last_refreshed_sb {
            let sbs = self.sbs_in_frame();
            self.sb_cursor = (sb_idx + 1) % sbs;
        }
    }

    /// Helper for AQ.3 wiring: returns the per-segment Q-delta table
    /// suitable for [`SegmentationParams`]`.feature_data[seg][0]`
    /// (slot 0 == `SEG_LVL_ALT_Q`). Slots are `None` for segments
    /// that don't have a Q-delta; segment 1 ([`CR_SEGMENT_ID_BOOST`])
    /// carries the configured `segment1_q_delta`.
    ///
    /// [`SegmentationParams`]: rvp9_core::bitstream::sub_parsers::SegmentationParams
    #[must_use]
    pub fn segment_q_deltas(&self) -> [Option<i16>; 8] {
        let mut out = [None; 8];
        out[CR_SEGMENT_ID_BOOST as usize] = Some(self.config.segment1_q_delta);
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 8x8 SB frame = 64x64 mi cells; convenient size for verifying
    /// per-SB walk + cursor wrap.
    const MI_ROWS: usize = 16;
    const MI_COLS: usize = 16;

    fn make_cr(config: CyclicRefreshConfig) -> CyclicRefresh {
        CyclicRefresh::new(MI_ROWS, MI_COLS, config)
    }

    #[test]
    fn cyclic_refresh_initial_state_marks_no_refresh() {
        // Default config has min_refresh_age = 4. Fresh state has all
        // map entries at age 0, so nothing is eligible. The picked
        // segmentation map must be all zeros.
        let cr = make_cr(CyclicRefreshConfig::default());
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments();
        assert_eq!(seg_map.len(), MI_ROWS * MI_COLS);
        assert!(
            seg_map.iter().all(|&s| s == 0),
            "fresh-state seg_map must be all zeros (no cell has aged yet)",
        );
        assert_eq!(n_refreshed, 0);
    }

    #[test]
    fn cyclic_refresh_after_aging_eligible_sbs_get_marked() {
        // Manually age the top-left SB's cells (rows 0..8, cols 0..8)
        // to min_refresh_age. pre_encode should pick them.
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64, // 25% → 64 cells
            segment1_q_delta: -8,
            min_refresh_age: 4,
        });
        for r in 0..8 {
            for c in 0..8 {
                cr.map[r * MI_COLS + c] = 4;
            }
        }
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments();
        // Top-left SB (8x8 = 64 cells) should be all segment 1.
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    seg_map[r * MI_COLS + c],
                    CR_SEGMENT_ID_BOOST,
                    "top-left SB at ({r},{c}) should be boosted",
                );
            }
        }
        // Other SBs (cells outside rows<8 AND cols<8) stay at 0.
        for r in 0..MI_ROWS {
            for c in 0..MI_COLS {
                if r >= 8 || c >= 8 {
                    assert_eq!(
                        seg_map[r * MI_COLS + c],
                        0,
                        "non-aged SB cell at ({r},{c}) should stay baseline",
                    );
                }
            }
        }
        assert_eq!(n_refreshed, 64);
    }

    #[test]
    fn cyclic_refresh_cycle_completes_in_n_frames() {
        // 25% per frame; min_refresh_age = 0 so every cell is eligible
        // from the start. Over 4 frames every SB should get refreshed
        // exactly once (4 × 25% = 100% coverage).
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64, // 25%
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        let mut ever_refreshed = vec![false; n_cells];

        for _frame in 0..4 {
            let (seg_map, _n_refreshed) = cr.pre_encode_pick_segments();
            for (i, &s) in seg_map.iter().enumerate() {
                if s == CR_SEGMENT_ID_BOOST {
                    ever_refreshed[i] = true;
                }
            }
            // post_encode with coded == picked and no skips.
            let coded_skip = vec![false; n_cells];
            cr.post_encode_update(&seg_map, &coded_skip);
        }

        assert!(
            ever_refreshed.iter().all(|&r| r),
            "every cell must be refreshed at least once within 4 frames \
             at 25% refresh rate; missed: {} cells",
            ever_refreshed.iter().filter(|&&r| !r).count(),
        );
    }

    #[test]
    fn cyclic_refresh_cycle_completes_at_multiple_fractions() {
        // Oracle: 1/F frames should fully cover at fraction F/256.
        // Test 25% (4 frames), 50% (2 frames), 12.5% (8 frames).
        for (fraction_256, n_frames) in [(64u32, 4usize), (128, 2), (32, 8)] {
            let mut cr = make_cr(CyclicRefreshConfig {
                target_refresh_fraction_256: fraction_256,
                segment1_q_delta: -8,
                min_refresh_age: 0,
            });
            let n_cells = MI_ROWS * MI_COLS;
            let mut ever_refreshed = vec![false; n_cells];

            for _ in 0..n_frames {
                let (seg_map, _) = cr.pre_encode_pick_segments();
                for (i, &s) in seg_map.iter().enumerate() {
                    if s == CR_SEGMENT_ID_BOOST {
                        ever_refreshed[i] = true;
                    }
                }
                let coded_skip = vec![false; n_cells];
                cr.post_encode_update(&seg_map, &coded_skip);
            }

            assert!(
                ever_refreshed.iter().all(|&r| r),
                "fraction={fraction_256}/256 over {n_frames} frames must cover all cells",
            );
        }
    }

    #[test]
    fn cyclic_refresh_postencode_resets_refreshed_to_zero_and_ages_baseline() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        // Pre-age all cells to 7 so we can see the increment-cap behaviour.
        for v in cr.map.iter_mut() {
            *v = 7;
        }
        // Pretend the top-left SB was coded with segment 1; everything
        // else baseline. No skips.
        let mut coded_segments = vec![0u8; n_cells];
        for r in 0..8 {
            for c in 0..8 {
                coded_segments[r * MI_COLS + c] = CR_SEGMENT_ID_BOOST;
            }
        }
        let coded_skip = vec![false; n_cells];
        cr.post_encode_update(&coded_segments, &coded_skip);

        for r in 0..MI_ROWS {
            for c in 0..MI_COLS {
                let in_refreshed_sb = r < 8 && c < 8;
                let expected = if in_refreshed_sb { 0 } else { 8 };
                assert_eq!(
                    cr.map[r * MI_COLS + c],
                    expected,
                    "cell ({r},{c}): in_refreshed_sb={in_refreshed_sb}",
                );
            }
        }
    }

    #[test]
    fn cyclic_refresh_skipped_blocks_excluded_from_refresh_tracking() {
        // A cell coded as segment 1 BUT skip=true should NOT have its
        // age reset — matches libvpx's "inter-skip blocks weren't
        // actually coded at the boosted Q" behaviour.
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        for v in cr.map.iter_mut() {
            *v = 3;
        }
        let mut coded_segments = vec![0u8; n_cells];
        let mut coded_skip = vec![false; n_cells];
        // SB at (0,0): all segment 1, all skip → must NOT reset.
        // SB at (0,1) [cols 8..16, rows 0..8]: all segment 1, no skip → DOES reset.
        for r in 0..8 {
            for c in 0..16 {
                coded_segments[r * MI_COLS + c] = CR_SEGMENT_ID_BOOST;
            }
            for c in 0..8 {
                coded_skip[r * MI_COLS + c] = true;
            }
        }
        cr.post_encode_update(&coded_segments, &coded_skip);

        // Skipped boost cells: age incremented to 4 (not reset).
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    cr.map[r * MI_COLS + c],
                    4,
                    "skipped boost cell at ({r},{c}) must age, not reset",
                );
            }
        }
        // Non-skipped boost cells: age reset to 0.
        for r in 0..8 {
            for c in 8..16 {
                assert_eq!(
                    cr.map[r * MI_COLS + c],
                    0,
                    "non-skipped boost cell at ({r},{c}) must reset to 0",
                );
            }
        }
        // Baseline cells: aged from 3 to 4.
        for r in 8..MI_ROWS {
            for c in 0..MI_COLS {
                assert_eq!(
                    cr.map[r * MI_COLS + c],
                    4,
                    "baseline cell at ({r},{c}) must age 3 → 4",
                );
            }
        }
    }

    #[test]
    fn cyclic_refresh_segment_q_deltas_returns_correct_table() {
        let cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 25,
            segment1_q_delta: -10,
            min_refresh_age: 4,
        });
        let table = cr.segment_q_deltas();
        assert_eq!(table[1], Some(-10), "segment 1 must carry the q-delta");
        for (seg, slot) in table.iter().enumerate() {
            if seg != 1 {
                assert_eq!(*slot, None, "segment {seg} must be None");
            }
        }
    }

    #[test]
    fn cyclic_refresh_sb_cursor_wraps_around_frame() {
        // 2x2 SB grid = 4 SBs total. Refresh fraction is set to 1 SB
        // per frame (64/256 = 25% = 1 of 4 SBs).
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        let sbs = cr.sbs_in_frame();
        assert_eq!(sbs, 4, "16x16 mi frame should yield 4 SBs");

        // Walk 5 frames; cursor should hit each of the 4 SB-start
        // indices over the cycle and wrap back.
        let mut cursors_seen = std::collections::HashSet::new();
        cursors_seen.insert(cr.sb_cursor);
        for _ in 0..5 {
            let (seg_map, _) = cr.pre_encode_pick_segments();
            let coded_skip = vec![false; n_cells];
            cr.post_encode_update(&seg_map, &coded_skip);
            cursors_seen.insert(cr.sb_cursor);
        }
        // We expect all 4 cursor positions to have been visited
        // (0, 1, 2, 3 — modulo sbs).
        assert_eq!(
            cursors_seen.len(),
            4,
            "cursor must visit every SB starting position over a full \
             cycle; saw {cursors_seen:?}",
        );
        // Cursor must be in range.
        assert!(cr.sb_cursor < sbs, "sb_cursor must stay < sbs_in_frame");
    }

    #[test]
    fn cyclic_refresh_handles_zero_target_fraction() {
        let cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 0,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments();
        assert!(seg_map.iter().all(|&s| s == 0));
        assert_eq!(n_refreshed, 0);
    }
}
