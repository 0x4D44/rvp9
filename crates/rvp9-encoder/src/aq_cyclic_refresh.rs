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
//!    Stops once we've picked the target fraction of cells, then advances
//!    the cursor past the last SB inspected.
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
//! refreshed" as `map[i] < 0` (counter ticking up toward 0). We use a
//! simpler "age in cycles since last refresh" semantic — eligibility
//! is `age >= min_refresh_age`. AQ-night-7 ported the following
//! libvpx feedback signals:
//!
//! - **`apply_cyclic_refresh` gate** ([`CyclicRefresh::apply_cyclic_refresh`])
//!   — frame-level skip. AQ is suppressed on keyframes, on frames
//!   where `avg_frame_qindex_inter < 20`, and on
//!   "stuck at MAXQ end" frames (`> 233 && frames_since_key > 20`).
//!   Mirrors `vp9_aq_cyclicrefresh.c:487-505` for the camera-content
//!   non-SVC operating point.
//! - **`last_coded_q_map`** — per-mi-cell q-history. Seeded to MAXQ,
//!   updated post-encode with libvpx's skip-aware MIN-clamp logic
//!   (`vp9_aq_cyclicrefresh.c:224-254`). Eligibility in
//!   [`CyclicRefresh::pre_encode_pick_segments`] gates on `> qindex_thresh`,
//!   matching `vp9_aq_cyclicrefresh.c:437`.
//! - **"≥ half eligible" gate** — SB is painted segment 1 iff at
//!   least half its cells are eligible under the libvpx `sum_map >=
//!   xmis * ymis / 2` rule (`vp9_aq_cyclicrefresh.c:449`).
//!
//! Still **NOT ported in v1**: BOOST2 segment (`candidate_refresh_aq`
//! demotes big zero-MV inter blocks to a more aggressive delta-q;
//! needs the inter picker's per-leaf rate/dist + motion thresholds,
//! which the rvp9 non-RD picker doesn't surface), `consec_zero_mv`
//! map, `noise_estimate`, scene-change SAD detection, screen-content
//! flat-static block exclusion, ROI gates.
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

use rvp9_core::quant::tables::AC_QLOOKUP;

/// VP9 super-block side in 8x8 mi units (64/8 = 8).
const MI_PER_SB: usize = 8;

/// Boost segment id used by cyclic refresh (libvpx
/// `CR_SEGMENT_ID_BOOST1 = 1`).
pub const CR_SEGMENT_ID_BOOST: u8 = 1;

/// libvpx `CR_SEGMENT_ID_BASE` (`vp9_aq_cyclicrefresh.h`).
pub const CR_SEGMENT_ID_BASE: u8 = 0;

/// libvpx alias `CR_SEGMENT_ID_BOOST1` (`vp9_aq_cyclicrefresh.h`). Same
/// value as [`CR_SEGMENT_ID_BOOST`] (= 1); kept as an alias for code
/// that mirrors libvpx naming.
pub const CR_SEGMENT_ID_BOOST1: u8 = CR_SEGMENT_ID_BOOST;

/// libvpx `CR_SEGMENT_ID_BOOST2` (`vp9_aq_cyclicrefresh.h`). More
/// aggressive Q-delta segment; assigned by `candidate_refresh_aq` to
/// big zero-MV inter blocks (`vp9_aq_cyclicrefresh.c:80-84`).
pub const CR_SEGMENT_ID_BOOST2: u8 = 2;

/// libvpx `CR_MAX_RATE_TARGET_RATIO`
/// (`vp9_aq_cyclicrefresh.h:30` — `#define CR_MAX_RATE_TARGET_RATIO 4.0`).
/// Used to clamp the BOOST2 rate factor in
/// [`compute_boost2_rate_factor`].
pub const CR_MAX_RATE_TARGET_RATIO: f64 = 4.0;

/// W5.5 — `FRAME_TYPE` mirror used by [`vp9_compute_qdelta_by_rate`].
///
/// libvpx's `FRAME_TYPE` is `{ KEY_FRAME = 0, INTER_FRAME = 1 }`
/// (`vp9_common.h`). We mirror only the distinction that
/// `vp9_rc_bits_per_mb` cares about: the enumerator constant
/// (2_700_000 for KEY vs 1_800_000 for INTER —
/// `vp9_ratectrl.c:188`).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FrameKind {
    /// libvpx `KEY_FRAME`.
    Key,
    /// libvpx `INTER_FRAME`.
    Inter,
}

/// libvpx `vp9_convert_qindex_to_q` for 8-bit
/// (`vp9_ratectrl.c:159-172`, non-HBD branch — line 170):
///
/// ```c
/// return vp9_ac_quant(qindex, 0, bit_depth) / 4.0;
/// ```
///
/// `vp9_ac_quant(qindex, 0, VPX_BITS_8) = AC_QLOOKUP[qindex]`.
fn convert_qindex_to_q(qindex: u8) -> f64 {
    f64::from(AC_QLOOKUP[qindex as usize]) / 4.0
}

/// libvpx `vp9_rc_bits_per_mb` (`vp9_ratectrl.c:185-196`) with
/// `correction_factor = 1.0` (the call site in
/// `vp9_compute_qdelta_by_rate` always passes 1.0):
///
/// ```c
/// const double q = vp9_convert_qindex_to_q(qindex, bit_depth);
/// int enumerator = frame_type == KEY_FRAME ? 2700000 : 1800000;
/// enumerator += (int)(enumerator * q) >> 12;
/// return (int)(enumerator * correction_factor / q);
/// ```
///
/// Note libvpx uses int-truncation on the `enumerator * q >> 12`
/// shift; we replicate by casting to `i64` before the shift. The
/// final `/q` is float division then cast to int — we round toward
/// zero to match `(int)` cast semantics on positive values.
fn rc_bits_per_mb_unit_corr(frame_kind: FrameKind, qindex: u8) -> i32 {
    let q = convert_qindex_to_q(qindex);
    let mut enumerator: i64 = match frame_kind {
        FrameKind::Key => 2_700_000,
        FrameKind::Inter => 1_800_000,
    };
    // `(int)(enumerator * q) >> 12`. libvpx multiplies an `int` by a
    // `double` then casts to `int` (truncate toward zero), then
    // arithmetic-shifts right 12. With `enumerator <= 2.7e6` and
    // `q <= AC_QLOOKUP[255]/4 = 457`, the product fits easily in i64.
    let prod_trunc: i64 = (enumerator as f64 * q) as i64;
    enumerator += prod_trunc >> 12;
    // Final divide & cast-to-int (truncation toward zero).
    (enumerator as f64 / q) as i32
}

/// libvpx `MAXQ`. Used both as the seed value for `last_coded_q_map`
/// (`vp9_aq_cyclicrefresh.c::vp9_cyclic_refresh_alloc` line 48-49)
/// and as the clamp upper bound for stored per-cell q values
/// (`vp9_aq_cyclicrefresh.c::vp9_cyclic_refresh_update_sb_postencode`
/// line 246).
const MAXQ: u8 = 255;

/// libvpx `apply_cyclic_refresh` gate: minimum
/// `avg_frame_qindex[INTER_FRAME]` below which AQ is suppressed for
/// this frame. Mirrors `vp9_aq_cyclicrefresh.c:487-488`:
///
/// ```c
/// int qp_thresh = VPXMIN((cpi->oxcf.content == VP9E_CONTENT_SCREEN)
///                            ? 35 : 20,
///                        rc->best_quality << 1);
/// ```
///
/// At very low average Q the encoder is already spending bits freely;
/// AQ overhead is pure cost. We hard-code the camera-content branch
/// (we have no screen-content mode).
const QP_THRESH_CAMERA: i32 = 20;

/// libvpx `apply_cyclic_refresh` gate: maximum
/// `avg_frame_qindex[INTER_FRAME]` above which AQ is suppressed once
/// the encoder has walked `> 20` frames past the last keyframe.
/// Mirrors `vp9_aq_cyclicrefresh.c:489`: `qp_max_thresh = 117 * MAXQ >> 7`
/// = `117 * 255 >> 7 = 233`.
const QP_MAX_THRESH: i32 = (117 * MAXQ as i32) >> 7;

/// libvpx `apply_cyclic_refresh` frames-since-key window after which
/// the high-Q suppression engages (`vp9_aq_cyclicrefresh.c:500`).
const QP_MAX_THRESH_FRAMES_SINCE_KEY: u32 = 20;

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
    /// last refreshed. 0 = just refreshed, 255 = aged-out cap
    /// (saturating). Fresh cells start at `min_refresh_age` so their
    /// zero-initialized libvpx counterparts are immediately eligible.
    pub map: Vec<u8>,
    /// Per-mi-cell map of the q-index this cell was last coded at.
    /// Mirrors libvpx `cr->last_coded_q_map` (`vp9_aq_cyclicrefresh.c`
    /// lines 42-49). Seeded to `MAXQ` so every cell is initially seen
    /// as "coarsely coded" and therefore eligible for refresh. Updated
    /// in [`Self::post_encode_update`] using libvpx's
    /// `vp9_cyclic_refresh_update_sb_postencode` logic (skip-aware
    /// MIN clamp).
    pub last_coded_q_map: Vec<u8>,
    /// SB cursor (raster order over the SB grid). Walk resumes here
    /// on the next `pre_encode_pick_segments`.
    pub sb_cursor: usize,
    /// Number of 8x8 cells coded as BOOST1 in the previous frame's
    /// final segmentation map. Counts the segment id regardless of
    /// whether the cell skipped, matching libvpx's postencode count.
    pub actual_num_seg1_blocks: usize,
    /// Number of 8x8 cells coded as BOOST2 in the previous frame's
    /// final segmentation map. Counts the segment id regardless of
    /// whether the cell skipped, matching libvpx's postencode count.
    pub actual_num_seg2_blocks: usize,
    /// Frame height in 8x8 mi units.
    pub mi_rows: usize,
    /// Frame width in 8x8 mi units.
    pub mi_cols: usize,
    /// Configuration.
    pub config: CyclicRefreshConfig,
}

impl CyclicRefresh {
    /// Allocate fresh cyclic-refresh state for an `mi_rows × mi_cols`
    /// frame. All map cells start eligible, mirroring libvpx's
    /// zero-initialized `map[i] == 0` eligibility. In rvp9's age-based
    /// encoding, the equivalent value is `min_refresh_age`.
    #[must_use]
    pub fn new(mi_rows: usize, mi_cols: usize, config: CyclicRefreshConfig) -> Self {
        Self {
            map: vec![config.min_refresh_age; mi_rows * mi_cols],
            // libvpx seeds `last_coded_q_map` to `MAXQ`
            // (`vp9_aq_cyclicrefresh.c:48-49`) so every cell starts
            // marked "coded at coarsest q" and is therefore eligible
            // for refresh under the
            // `last_coded_q_map[i] > qindex_thresh` gate.
            last_coded_q_map: vec![MAXQ; mi_rows * mi_cols],
            sb_cursor: 0,
            actual_num_seg1_blocks: 0,
            actual_num_seg2_blocks: 0,
            mi_rows,
            mi_cols,
            config,
        }
    }

    /// libvpx `apply_cyclic_refresh` frame-level gate.
    ///
    /// Decides whether to apply cyclic refresh THIS frame based on the
    /// rate controller's `avg_frame_qindex[INTER_FRAME]` and the
    /// frame-since-key counter. Mirrors the early-out branch in
    /// `vp9_aq_cyclicrefresh.c::vp9_cyclic_refresh_update_parameters`
    /// (lines 487-505), restricted to the camera-content / single-
    /// layer / non-lossless / non-SVC operating point — the only one
    /// rvp9 ships in v1.
    ///
    /// Returns `false` (suppress AQ this frame) when:
    /// - `is_keyframe == true`. Libvpx tests `frame_is_intra_only(cm)`
    ///   — for our scope (no `show_existing_frame`, no intra-only inter
    ///   frames) this collapses to "keyframe".
    /// - `avg_frame_qindex_inter < QP_THRESH_CAMERA` (`= 20`). The
    ///   encoder is already running fine; AQ overhead is pure cost.
    /// - `avg_frame_qindex_inter > QP_MAX_THRESH` (`= 233`) AND
    ///   `frames_since_key > 20`. The encoder is saturating at the
    ///   coarsest end; AQ can't improve a frame already at MAXQ.
    ///
    /// The `best_quality` parameter mirrors libvpx
    /// `rc->best_quality` and tightens `qp_thresh` to
    /// `min(QP_THRESH_CAMERA, best_quality << 1)`.
    ///
    /// **NOT ported (libvpx-faithful divergences):** no `cpi->use_svc`
    /// SVC layer test (single-layer only), no `is_lossless_requested`
    /// gate (lossless out of v1 scope), no
    /// `avg_frame_low_motion < 20 && frames_since_key > 40` low-motion
    /// gate (would need a motion-content estimator we don't have),
    /// no `cpi->roi.enabled && cpi->roi.skip[...]` ROI gate (ROI not
    /// implemented).
    #[must_use]
    pub fn apply_cyclic_refresh(
        &self,
        is_keyframe: bool,
        avg_frame_qindex_inter: u8,
        best_quality: i32,
        frames_since_key: u32,
    ) -> bool {
        if is_keyframe {
            return false;
        }
        let qp_thresh = QP_THRESH_CAMERA.min(best_quality << 1);
        let avg_q = i32::from(avg_frame_qindex_inter);
        if avg_q < qp_thresh {
            return false;
        }
        if avg_q > QP_MAX_THRESH && frames_since_key > QP_MAX_THRESH_FRAMES_SINCE_KEY {
            return false;
        }
        true
    }

    /// libvpx eligibility threshold for `last_coded_q_map`.
    ///
    /// In libvpx (`vp9_aq_cyclicrefresh.c:388-391`) this is
    /// `vp9_get_qindex(seg, CR_SEGMENT_ID_BOOST1, base_qindex)`, which
    /// resolves to `base_qindex + qindex_delta[1]` under SEGMENT_DELTADATA.
    /// We compute the same: a cell is "stale enough to refresh" iff its
    /// `last_coded_q_map[i]` was coarser than the q it would get under
    /// segment 1 of THIS frame.
    fn qindex_thresh_for_eligibility(&self, base_qindex: u8) -> i32 {
        let base = i32::from(base_qindex);
        (base + i32::from(self.config.segment1_q_delta)).clamp(0, MAXQ as i32)
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

    /// Compute libvpx's pre-Q cyclic-refresh segment weight.
    ///
    /// The sum is shifted before conversion to `f64`, preserving the
    /// integer order in `vp9_cyclic_refresh_update_parameters`
    /// (`vp9_aq_cyclicrefresh.c:571-577`). The target fraction is used
    /// when the previous frame's actual refresh fraction is more than
    /// `8/7` of the target.
    #[must_use]
    pub fn weight_segment(
        target_cells: usize,
        actual_num_seg1_blocks: usize,
        actual_num_seg2_blocks: usize,
        total_cells: usize,
    ) -> f64 {
        if total_cells == 0 {
            return 0.0;
        }
        let target_weight = target_cells as f64 / total_cells as f64;
        let weight = ((target_cells + actual_num_seg1_blocks + actual_num_seg2_blocks) >> 1) as f64
            / total_cells as f64;
        if target_weight < 7.0 * weight / 8.0 {
            target_weight
        } else {
            weight
        }
    }

    /// Pick the segmentation map for the next frame.
    ///
    /// Walks the SB grid starting at `sb_cursor`. For each SB the
    /// 8x8 cell eligibility is computed as in libvpx
    /// `cyclic_refresh_update_map` (lines 428-449):
    ///
    /// 1. A cell with `map[i] >= min_refresh_age` (libvpx tests
    ///    `cr->map[i] == 0`; in our age-based encoding the equivalent
    ///    is "aged at least the configured minimum") **and** with its
    ///    `last_coded_q_map[i]` coarser than the segment-1 refresh
    ///    threshold contributes 1 to `sum_map`.
    /// 2. If `sum_map >= xmis * ymis / 2` (at least HALF the cells
    ///    in the SB are eligible), the entire SB is painted
    ///    [`CR_SEGMENT_ID_BOOST`].
    ///
    /// The "≥ half" rule replaces the previous "ALL cells eligible"
    /// gate — bug-for-bug fix per libvpx `vp9_aq_cyclicrefresh.c:449`.
    ///
    /// The `base_qindex` parameter feeds
    /// `qindex_thresh_for_eligibility`; pass the q-index that
    /// will be used for this frame so the eligibility filter is
    /// computed against the correct boost segment q. Libvpx threads
    /// `cm->base_qindex` through `cyclic_refresh_update_map`
    /// (`vp9_aq_cyclicrefresh.c:391`).
    ///
    /// Stops once we've marked at least `target_refresh_cells` cells
    /// OR walked the whole frame back to the starting cursor. Returns
    /// `(segmentation_map of length mi_rows * mi_cols, n_refreshed_mi_cells)`.
    ///
    /// Advances `sb_cursor` past the last SB inspected, matching
    /// libvpx `cyclic_refresh_update_map`: cursor progress depends on
    /// the map-selection walk, not whether the selected blocks later
    /// skip their residual transforms.
    #[must_use]
    pub fn pre_encode_pick_segments(&mut self, base_qindex: u8) -> (Vec<u8>, usize) {
        let n_cells = self.mi_rows * self.mi_cols;
        let mut seg_map = vec![0u8; n_cells];
        let mut n_refreshed: usize = 0;

        let target = self.target_refresh_cells();
        let sbs = self.sbs_in_frame();
        if sbs == 0 || target == 0 {
            return (seg_map, 0);
        }

        let qindex_thresh = self.qindex_thresh_for_eligibility(base_qindex);
        let sb_cols = self.sb_cols();
        let mut next_cursor = self.sb_cursor;
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
            let xmis = mi_col_end - mi_col0;
            let ymis = mi_row_end - mi_row0;
            let sb_cells = xmis * ymis;

            // Eligibility: per-cell — libvpx
            // `vp9_aq_cyclicrefresh.c:434-441`. A cell is eligible iff
            // age >= min_refresh_age AND last_coded_q_map[i] > qindex_thresh.
            // (libvpx uses `cr->map[i] == 0` as the age gate; in our
            // age-based encoding that's age >= min_refresh_age.)
            let mut sum_map: usize = 0;
            for r in mi_row0..mi_row_end {
                for c in mi_col0..mi_col_end {
                    let i = r * self.mi_cols + c;
                    let age_ok = self.map[i] >= self.config.min_refresh_age;
                    let q_ok = i32::from(self.last_coded_q_map[i]) > qindex_thresh;
                    if age_ok && q_ok {
                        sum_map += 1;
                    }
                }
            }

            // libvpx "≥ half cells eligible" rule
            // (`vp9_aq_cyclicrefresh.c:449`).
            if sum_map * 2 >= sb_cells {
                for r in mi_row0..mi_row_end {
                    for c in mi_col0..mi_col_end {
                        seg_map[r * self.mi_cols + c] = CR_SEGMENT_ID_BOOST;
                    }
                }
                n_refreshed += sb_cells;
            }

            // libvpx increments `i` after every inspected SB and
            // commits it to `cr->sb_index` when the walk stops.
            next_cursor = (sb_idx + 1) % sbs;
            if n_refreshed >= target {
                break;
            }
        }

        self.sb_cursor = next_cursor;
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
    /// Also updates `last_coded_q_map[i]` per libvpx
    /// `vp9_cyclic_refresh_update_sb_postencode` (lines 224-254 in
    /// `vp9_aq_cyclicrefresh.c`):
    /// - Non-skipped (or intra) cells coded under a refresh segment:
    ///   write the effective q (`base_qindex + qindex_delta[seg]`).
    /// - Inter-skipped cells coded under a refresh segment: take the
    ///   MIN of the previous value and the effective q. (Skipped cells
    ///   weren't actually re-quantized; libvpx documents this as the
    ///   conservative "we can't know what q this cell ended up at,
    ///   so don't push it higher than what was recorded".)
    /// - Baseline cells coded without skip: write `base_qindex`.
    ///
    /// The previous frame's final BOOST1/BOOST2 segment counts are reset
    /// and recomputed from `coded_segments`, independently of `coded_skip`,
    /// for the next frame's pre-Q rate-control estimate.
    ///
    /// **rvp9 simplification:** rvp9 doesn't track per-cell intra/inter
    /// nor a real skip flag for baseline cells, so the inter-skip
    /// branch is keyed only on the `coded_skip` flag — which the
    /// frame loop populates from `picked_mode == ZEROMV && skip ==
    /// true`. Empirically equivalent within the v1 picker shape.
    ///
    /// # Panics
    ///
    /// Panics if `coded_segments.len() != mi_rows * mi_cols` or
    /// `coded_skip.len() != mi_rows * mi_cols`.
    pub fn post_encode_update(
        &mut self,
        coded_segments: &[u8],
        coded_skip: &[bool],
        base_qindex: u8,
    ) {
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

        let base = i32::from(base_qindex);
        self.actual_num_seg1_blocks = 0;
        self.actual_num_seg2_blocks = 0;

        for i in 0..n_cells {
            let seg = coded_segments[i];
            let skip = coded_skip[i];
            let actually_refreshed = seg == CR_SEGMENT_ID_BOOST && !skip;

            match seg {
                CR_SEGMENT_ID_BOOST1 => self.actual_num_seg1_blocks += 1,
                CR_SEGMENT_ID_BOOST2 => self.actual_num_seg2_blocks += 1,
                _ => {}
            }

            // `last_coded_q_map` update — mirrors libvpx
            // `vp9_cyclic_refresh_update_sb_postencode`. We use
            // `seg == CR_SEGMENT_ID_BOOST` as the boosted-segment test;
            // segment 0 = base.
            let seg_delta = if seg == CR_SEGMENT_ID_BOOST {
                i32::from(self.config.segment1_q_delta)
            } else {
                0
            };
            let effective_q = (base + seg_delta).clamp(0, MAXQ as i32) as u8;
            if !skip {
                self.last_coded_q_map[i] = effective_q;
            } else if seg == CR_SEGMENT_ID_BOOST {
                // Inter-skip on boost segment: take MIN with previous
                // value (`vp9_aq_cyclicrefresh.c:247-251`).
                self.last_coded_q_map[i] = self.last_coded_q_map[i].min(effective_q);
            } else {
                // Inter-skip on base segment: libvpx leaves
                // last_coded_q_map UNCHANGED (the `else if` branch is
                // not taken). This preserves stale-q history for
                // skipped baseline cells — correct because they
                // weren't re-coded at the new q.
            }

            // Age map update.
            if actually_refreshed {
                self.map[i] = 0;
            } else {
                self.map[i] = self.map[i].saturating_add(1);
            }
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

    /// W5.6 — libvpx `vp9_cyclic_refresh_update_segment`
    /// (`vp9_aq_cyclicrefresh.c:160-222`).
    ///
    /// Post-pick walk over a SB's 8x8 mi cells, deciding whether each
    /// cell should be promoted to a refresh segment and updating both
    /// the cyclic-refresh `map` and the output `segmentation_map`.
    ///
    /// Inputs mirror libvpx (per-block, ie. per `bsize` chunk of cells):
    /// - `mi_row` / `mi_col`: top-left mi position of the picker block
    ///   (`vp9_aq_cyclicrefresh.c:161`).
    /// - `bw_mi` / `bh_mi`: block size in 8x8 mi units —
    ///   `num_8x8_blocks_wide_lookup[bsize]` /
    ///   `num_8x8_blocks_high_lookup[bsize]` (lines 166-167).
    /// - `rate` / `dist`: picker output —
    ///   thresholded against `cr->thresh_rate_sb` /
    ///   `cr->thresh_dist_sb` (lines 75-83).
    /// - `mv_row_abs` / `mv_col_abs`: absolute MV components (any) —
    ///   thresholded against `cr->motion_thresh` (line 76-77).
    /// - `is_inter`: `is_inter_block(mi)`.
    /// - `is_skin`: from W2.5 pre-analysis skin map. libvpx computes
    ///   this on-the-fly inside `update_segment` (lines 177-184); we
    ///   accept the pre-computed flag.
    /// - `prior_segment_id`: `mi->segment_id` as set by the picker
    ///   (libvpx's `cyclic_refresh_segment_id_boosted(mi->segment_id)`
    ///   check, line 192).
    /// - `skip`: picker's skip flag.
    /// - `use_nonrd_pick_mode`: `cpi->sf.use_nonrd_pick_mode` (line 191).
    /// - `is_vbr_golden`: `cpi->oxcf.rc_mode == VPX_VBR &&
    ///   mi->ref_frame[0] == GOLDEN_FRAME` (line 186 — VBR-GOLDEN
    ///   demotion).
    /// - `bsize_is_lt_16x16`: skin promotion gated to
    ///   `bsize <= BLOCK_16X16` (line 178).
    /// - `bsize_is_ge_16x16`: BOOST2 gate (line 80 — `bsize >= BLOCK_16X16`).
    /// - `use_skin_detection`: `cpi->use_skin_detection` (line 179).
    /// - `thresh_rate_sb` / `thresh_dist_sb` / `motion_thresh` /
    ///   `rate_boost_fac`: per-frame thresholds from
    ///   `vp9_aq_cyclicrefresh.c:653-655` (we accept them as args
    ///   rather than depending on the full CR struct's threshold-table
    ///   plumbing, which isn't ported yet).
    ///
    /// Mutates `self.map` and writes the final segment id of every
    /// cell in the block to `segmentation_map`.
    ///
    /// # Panics
    ///
    /// Panics if `segmentation_map.len() != mi_rows * mi_cols`.
    #[allow(clippy::too_many_arguments)]
    pub fn vp9_cyclic_refresh_update_segment(
        &mut self,
        segmentation_map: &mut [u8],
        mi_row: usize,
        mi_col: usize,
        bw_mi: usize,
        bh_mi: usize,
        rate: i64,
        dist: i64,
        mv_row_abs: i32,
        mv_col_abs: i32,
        is_inter: bool,
        is_zero_mv: bool,
        is_skin: bool,
        prior_segment_id: u8,
        skip: bool,
        use_nonrd_pick_mode: bool,
        is_vbr_golden: bool,
        bsize_is_ge_16x16: bool,
        bsize_is_le_16x16: bool,
        use_skin_detection: bool,
        thresh_rate_sb: i64,
        thresh_dist_sb: i64,
        motion_thresh: i32,
        rate_boost_fac: i32,
    ) {
        assert_eq!(
            segmentation_map.len(),
            self.mi_rows * self.mi_cols,
            "segmentation_map must be mi_rows * mi_cols",
        );
        // libvpx `vp9_aq_cyclicrefresh.c:168-170`:
        //   const int xmis = VPXMIN(cm->mi_cols - mi_col, bw);
        //   const int ymis = VPXMIN(cm->mi_rows - mi_row, bh);
        let xmis = (self.mi_cols.saturating_sub(mi_col)).min(bw_mi);
        let ymis = (self.mi_rows.saturating_sub(mi_row)).min(bh_mi);
        if xmis == 0 || ymis == 0 {
            return;
        }
        let block_index = mi_row * self.mi_cols + mi_col;

        // libvpx `candidate_refresh_aq` (`vp9_aq_cyclicrefresh.c:67-87`).
        // Returns the candidate segment id (BASE / BOOST1 / BOOST2).
        let mut refresh_this_block: u8 = if dist > thresh_dist_sb
            && (mv_row_abs > motion_thresh || mv_col_abs > motion_thresh || !is_inter)
        {
            CR_SEGMENT_ID_BASE
        } else if bsize_is_ge_16x16
            && rate < thresh_rate_sb
            && is_inter
            && is_zero_mv
            && rate_boost_fac > 10
        {
            CR_SEGMENT_ID_BOOST2
        } else {
            CR_SEGMENT_ID_BOOST1
        };

        // libvpx `vp9_aq_cyclicrefresh.c:177-184` — skin promotion to
        // BOOST1 (only when not already a refresh candidate, bsize ≤ 16x16,
        // and skin detection enabled).
        if refresh_this_block == CR_SEGMENT_ID_BASE
            && bsize_is_le_16x16
            && use_skin_detection
            && is_skin
        {
            refresh_this_block = CR_SEGMENT_ID_BOOST1;
        }

        // libvpx `vp9_aq_cyclicrefresh.c:186-187` — VBR-GOLDEN demotion.
        if is_vbr_golden {
            refresh_this_block = CR_SEGMENT_ID_BASE;
        }

        // libvpx `vp9_aq_cyclicrefresh.c:189-196` — segment id reset for
        // the non-RD pick path. If the picker labelled it a refresh
        // segment, replace with the `candidate_refresh_aq` decision;
        // additionally, skipped cells fall back to BASE.
        let mut final_segment_id = prior_segment_id;
        if use_nonrd_pick_mode
            && (prior_segment_id == CR_SEGMENT_ID_BOOST1
                || prior_segment_id == CR_SEGMENT_ID_BOOST2)
        {
            final_segment_id = refresh_this_block;
            if skip {
                final_segment_id = CR_SEGMENT_ID_BASE;
            }
        }

        // libvpx `vp9_aq_cyclicrefresh.c:198-212` — cyclic-refresh map
        // update value. In libvpx the encoding is:
        //   - boosted segment → map = -time_for_refresh (recently-refreshed)
        //   - candidate but `map == 1` (cleanup pending) → map = 0
        //   - else → map = 1
        // rvp9 uses an "age in cycles" encoding (see
        // [`CyclicRefresh::post_encode_update`]). Translating:
        //   - boosted segment → age reset to 0 (fresh).
        //   - candidate, previously at age 1 → set to 0 (about to refresh).
        //   - otherwise → age = 1 (not a candidate; let post-encode age it).
        let boosted =
            final_segment_id == CR_SEGMENT_ID_BOOST1 || final_segment_id == CR_SEGMENT_ID_BOOST2;
        let new_map_value: u8 = if boosted
            || (refresh_this_block != CR_SEGMENT_ID_BASE && self.map[block_index] == 1)
        {
            0
        } else if refresh_this_block != CR_SEGMENT_ID_BASE {
            // candidate but not yet at age==1 — leave the existing age.
            self.map[block_index]
        } else {
            1
        };

        // libvpx `vp9_aq_cyclicrefresh.c:214-221` — splat new_map_value
        // and final_segment_id over the xmis × ymis cells.
        for y in 0..ymis {
            for x in 0..xmis {
                let map_offset = block_index + y * self.mi_cols + x;
                self.map[map_offset] = new_map_value;
                segmentation_map[map_offset] = final_segment_id;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// W5.5 — Q-delta computation
// ---------------------------------------------------------------------------

/// W5.5 — libvpx `vp9_compute_qdelta_by_rate`
/// (`vp9_ratectrl.c:2562-2584`).
///
/// Computes the Q-index delta that produces a target bits/MB equal to
/// `rate_target_ratio * bits_per_mb(q)`. A `rate_target_ratio > 1`
/// targets more bits ⇒ finer Q ⇒ negative delta. A
/// `rate_target_ratio < 1` targets fewer bits ⇒ coarser Q ⇒ positive
/// delta. Hard-coded to 8-bit `bit_depth`; HBD is out of scope.
///
/// The libvpx loop walks `i = best_quality .. worst_quality` and stops
/// at the first `i` whose `bits_per_mb(i) <= target_bits_per_mb`. We
/// mirror that walk faithfully — including the libvpx behaviour where
/// the loop never matches and falls through to
/// `target_index = rc->worst_quality`.
///
/// `best_quality` / `worst_quality` mirror libvpx's
/// `rc->best_quality` / `rc->worst_quality`. Libvpx-typical values are
/// `0` and `MAXQ = 255` (`vp9_ratectrl.c::vp9_rc_init` line 408-409).
/// Pass `0` and `255` for the standard range.
#[must_use]
pub fn vp9_compute_qdelta_by_rate(
    qindex: u8,
    frame_kind: FrameKind,
    rate_target_ratio: f64,
    best_quality: u8,
    worst_quality: u8,
) -> i16 {
    // libvpx `vp9_ratectrl.c:2569-2570` — base bits/MB at the given
    // qindex, with `correction_factor = 1.0`.
    let base_bits_per_mb = rc_bits_per_mb_unit_corr(frame_kind, qindex);
    // libvpx `vp9_ratectrl.c:2572-2573`.
    let target_bits_per_mb = (rate_target_ratio * f64::from(base_bits_per_mb)) as i32;

    // libvpx `vp9_ratectrl.c:2575-2581` — scan from best_q toward
    // worst_q for the first q whose bits/mb falls at or below target.
    // Default fallthrough: worst_quality.
    let mut target_index: i32 = i32::from(worst_quality);
    let lo = i32::from(best_quality);
    let hi = i32::from(worst_quality);
    for i in lo..hi {
        if rc_bits_per_mb_unit_corr(frame_kind, i as u8) <= target_bits_per_mb {
            target_index = i;
            break;
        }
    }
    (target_index - i32::from(qindex)) as i16
}

/// W5.5 — libvpx `compute_deltaq` (`vp9_aq_cyclicrefresh.c:89-99`).
///
/// Wraps [`vp9_compute_qdelta_by_rate`] with the CR-specific clamp:
/// `(-deltaq) <= max_qdelta_perc * q / 100`. libvpx default
/// `max_qdelta_perc = 60` (`vp9_aq_cyclicrefresh.c:508`).
///
/// `frame_kind` — libvpx uses `cpi->common.frame_type` here; for the
/// rvp9 single-layer CBR camera path this is always
/// [`FrameKind::Inter`] when CR is active (CR is gated off on
/// keyframes by [`CyclicRefresh::apply_cyclic_refresh`]). Callers
/// SHOULD pass [`FrameKind::Inter`].
#[must_use]
pub fn compute_deltaq(
    qindex: u8,
    rate_factor: f64,
    max_qdelta_perc: i32,
    frame_kind: FrameKind,
    best_quality: u8,
    worst_quality: u8,
) -> i16 {
    let raw =
        vp9_compute_qdelta_by_rate(qindex, frame_kind, rate_factor, best_quality, worst_quality);
    let q = i32::from(qindex);
    let clamp_neg = max_qdelta_perc * q / 100;
    // libvpx `vp9_aq_cyclicrefresh.c:95-97`:
    //   if ((-deltaq) > cr->max_qdelta_perc * q / 100)
    //     deltaq = -cr->max_qdelta_perc * q / 100;
    if i32::from(-raw) > clamp_neg {
        return -(clamp_neg as i16);
    }
    raw
}

/// W5.5 — BOOST2 rate factor mirror of libvpx
/// `vp9_aq_cyclicrefresh.c:667-670`:
///
/// ```c
/// qindex_delta = compute_deltaq(
///     cpi, cm->base_qindex,
///     VPXMIN(CR_MAX_RATE_TARGET_RATIO,
///            0.1 * cr->rate_boost_fac * cr->rate_ratio_qdelta));
/// ```
///
/// Returns the rate factor to pass to [`compute_deltaq`] for segment
/// BOOST2. `rate_boost_fac` libvpx-typical = 15 (line 511);
/// `rate_ratio_qdelta` libvpx-typical = 2.0 (line 520).
#[must_use]
pub fn compute_boost2_rate_factor(rate_boost_fac: i32, rate_ratio_qdelta: f64) -> f64 {
    let raw = 0.1 * f64::from(rate_boost_fac) * rate_ratio_qdelta;
    raw.min(CR_MAX_RATE_TARGET_RATIO)
}

/// W5.5 — three-segment `qindex_delta[3]` ramp mirroring libvpx
/// `vp9_aq_cyclicrefresh.c:656-672`:
///
/// ```c
/// qindex_delta = compute_deltaq(cpi, cm->base_qindex, cr->rate_ratio_qdelta);
/// cr->qindex_delta[1] = qindex_delta;
/// ...
/// qindex_delta = compute_deltaq(
///     cpi, cm->base_qindex,
///     VPXMIN(CR_MAX_RATE_TARGET_RATIO,
///            0.1 * cr->rate_boost_fac * cr->rate_ratio_qdelta));
/// cr->qindex_delta[2] = qindex_delta;
/// ```
///
/// Segment 0 (BASE) is always 0 — libvpx never writes
/// `qindex_delta[0]` (it stays zero-initialised from
/// `vp9_cyclic_refresh_alloc`).
///
/// Returns `[seg0_delta, seg1_delta, seg2_delta]`. Under the libvpx
/// defaults all three should be `<= 0` (BASE = 0; BOOST1, BOOST2
/// negative). The ramp is monotone-non-increasing (BOOST2 ≤ BOOST1 ≤
/// 0) under the default `rate_boost_fac = 15 > 10`, because the
/// BOOST2 rate factor (`0.1 * 15 * rate_ratio_qdelta = 1.5 *
/// rate_ratio_qdelta`) ≥ `rate_ratio_qdelta` for any
/// `rate_ratio_qdelta ≥ 0`, so it targets fewer bits ⇒ at least as
/// negative a delta.
#[must_use]
pub fn qindex_delta_ramp(
    qindex: u8,
    rate_ratio_qdelta: f64,
    rate_boost_fac: i32,
    max_qdelta_perc: i32,
    frame_kind: FrameKind,
    best_quality: u8,
    worst_quality: u8,
) -> [i16; 3] {
    let seg0 = 0i16;
    let seg1 = compute_deltaq(
        qindex,
        rate_ratio_qdelta,
        max_qdelta_perc,
        frame_kind,
        best_quality,
        worst_quality,
    );
    let boost2_factor = compute_boost2_rate_factor(rate_boost_fac, rate_ratio_qdelta);
    let seg2 = compute_deltaq(
        qindex,
        boost2_factor,
        max_qdelta_perc,
        frame_kind,
        best_quality,
        worst_quality,
    );
    [seg0, seg1, seg2]
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
    fn cyclic_refresh_initial_state_is_immediately_eligible() {
        // Libvpx's zero-initialized refresh map is eligible immediately.
        // Rvp9's age encoding must seed fresh cells at min_refresh_age
        // rather than delaying the first refresh cycle.
        let mut cr = make_cr(CyclicRefreshConfig::default());
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments(100);
        assert_eq!(seg_map.len(), MI_ROWS * MI_COLS);
        assert!(
            seg_map.contains(&CR_SEGMENT_ID_BOOST),
            "fresh-state seg_map must refresh cells immediately",
        );
        assert!(n_refreshed >= cr.target_refresh_cells());
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
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments(100);
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
            let (seg_map, _n_refreshed) = cr.pre_encode_pick_segments(100);
            for (i, &s) in seg_map.iter().enumerate() {
                if s == CR_SEGMENT_ID_BOOST {
                    ever_refreshed[i] = true;
                }
            }
            // post_encode with coded == picked and no skips.
            let coded_skip = vec![false; n_cells];
            cr.post_encode_update(&seg_map, &coded_skip, 100);
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
                let (seg_map, _) = cr.pre_encode_pick_segments(100);
                for (i, &s) in seg_map.iter().enumerate() {
                    if s == CR_SEGMENT_ID_BOOST {
                        ever_refreshed[i] = true;
                    }
                }
                let coded_skip = vec![false; n_cells];
                cr.post_encode_update(&seg_map, &coded_skip, 100);
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
        cr.post_encode_update(&coded_segments, &coded_skip, 100);

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
        cr.post_encode_update(&coded_segments, &coded_skip, 100);

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
    fn cyclic_refresh_postencode_recounts_actual_segments_including_skips() {
        let mut cr = make_cr(CyclicRefreshConfig::default());
        let n_cells = MI_ROWS * MI_COLS;
        let mut coded_segments = vec![CR_SEGMENT_ID_BASE; n_cells];
        let mut coded_skip = vec![false; n_cells];

        coded_segments[0] = CR_SEGMENT_ID_BOOST1;
        coded_skip[0] = true;
        coded_segments[1] = CR_SEGMENT_ID_BOOST1;
        coded_segments[2] = CR_SEGMENT_ID_BOOST2;
        coded_skip[2] = true;
        coded_segments[3] = CR_SEGMENT_ID_BOOST2;

        cr.post_encode_update(&coded_segments, &coded_skip, 100);
        assert_eq!(cr.actual_num_seg1_blocks, 2);
        assert_eq!(cr.actual_num_seg2_blocks, 2);

        coded_segments.fill(CR_SEGMENT_ID_BASE);
        coded_skip.fill(false);
        cr.post_encode_update(&coded_segments, &coded_skip, 100);
        assert_eq!(cr.actual_num_seg1_blocks, 0);
        assert_eq!(cr.actual_num_seg2_blocks, 0);
    }

    #[test]
    fn cyclic_refresh_weight_segment_preserves_integer_shift_and_clamp() {
        // 25 + 10 + 6 = 41; libvpx shifts first, so 41 >> 1 = 20.
        assert_eq!(CyclicRefresh::weight_segment(25, 10, 6, 100), 0.20);

        // 25 + 20 + 20 = 65; 65 >> 1 = 32, then the 7/8 clamp
        // selects the target weight (0.25) over 0.32.
        assert_eq!(CyclicRefresh::weight_segment(25, 20, 20, 100), 0.25);
    }

    #[test]
    fn cyclic_refresh_cursor_advances_when_selected_block_skips() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;

        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments(100);
        assert_eq!(n_refreshed, 64);
        assert_eq!(
            cr.sb_cursor, 1,
            "map-selection walk must advance past the selected SB",
        );

        let coded_skip = vec![true; n_cells];
        cr.post_encode_update(&seg_map, &coded_skip, 100);
        assert_eq!(
            cr.sb_cursor, 1,
            "postencode skip feedback must not rewind the selection cursor",
        );

        let (next_seg_map, _) = cr.pre_encode_pick_segments(100);
        assert_eq!(
            next_seg_map[MI_PER_SB], CR_SEGMENT_ID_BOOST,
            "the next frame must select the next SB even when the prior SB skipped",
        );
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
            let (seg_map, _) = cr.pre_encode_pick_segments(100);
            let coded_skip = vec![false; n_cells];
            cr.post_encode_update(&seg_map, &coded_skip, 100);
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
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 0,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let (seg_map, n_refreshed) = cr.pre_encode_pick_segments(100);
        assert!(seg_map.iter().all(|&s| s == 0));
        assert_eq!(n_refreshed, 0);
    }

    /// AQ-night-7 — `apply_cyclic_refresh` frame-level gate.
    ///
    /// Mirrors libvpx `vp9_aq_cyclicrefresh.c:487-505`:
    /// - KF always suppresses.
    /// - Low avg-q (< 20) suppresses.
    /// - High avg-q (> 233) suppresses iff `frames_since_key > 20`.
    /// - Mid-range avg-q passes.
    #[test]
    fn apply_cyclic_refresh_gates_match_libvpx() {
        let cr = make_cr(CyclicRefreshConfig::default());
        // Use a default-like best_quality so qp_thresh = min(20, 2*best).
        // best_q = 32 gives qp_thresh = min(20, 64) = 20 (libvpx default).
        let best_q = 32;
        // Keyframe → always off, regardless of q.
        assert!(!cr.apply_cyclic_refresh(true, 100, best_q, 50));
        // Low avg_q → off (10 < qp_thresh = 20).
        assert!(!cr.apply_cyclic_refresh(false, 10, best_q, 100));
        // High avg_q but freshly post-KF → still on.
        assert!(cr.apply_cyclic_refresh(false, 240, best_q, 5));
        // High avg_q and frames_since_key > 20 → off.
        assert!(!cr.apply_cyclic_refresh(false, 240, best_q, 25));
        // Mid-range → on.
        assert!(cr.apply_cyclic_refresh(false, 128, best_q, 50));
        // best_quality squeeze: when best_quality << 1 < QP_THRESH_CAMERA,
        // qp_thresh is tightened. E.g. best_q = 5 ⇒ qp_thresh = 10,
        // avg_q = 12 passes (where the default qp_thresh = 20 would have
        // rejected).
        assert!(cr.apply_cyclic_refresh(false, 12, 5, 50));
    }

    /// AQ-night-7 — `last_coded_q_map` is seeded to MAXQ.
    #[test]
    fn cyclic_refresh_last_coded_q_map_seeded_maxq() {
        let cr = make_cr(CyclicRefreshConfig::default());
        assert_eq!(cr.last_coded_q_map.len(), MI_ROWS * MI_COLS);
        assert!(
            cr.last_coded_q_map.iter().all(|&q| q == MAXQ),
            "last_coded_q_map must seed to MAXQ for libvpx parity",
        );
    }

    /// AQ-night-7 — `last_coded_q_map` update under libvpx's skip-aware rules.
    #[test]
    fn cyclic_refresh_last_coded_q_map_update_libvpx_faithful() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        // Cell layout:
        //   col 0: base seg + skip=false → write base_qindex = 100
        //   col 1: base seg + skip=true  → preserve previous (MAXQ)
        //   col 2: boost seg + skip=false → write base + delta = 92
        //   col 3: boost seg + skip=true  → min(MAXQ, 92) = 92
        let mut coded_segments = vec![0u8; n_cells];
        let mut coded_skip = vec![false; n_cells];
        for r in 0..MI_ROWS {
            // col 1: base + skip
            coded_skip[r * MI_COLS + 1] = true;
            // col 2: boost + no skip
            coded_segments[r * MI_COLS + 2] = CR_SEGMENT_ID_BOOST;
            // col 3: boost + skip
            coded_segments[r * MI_COLS + 3] = CR_SEGMENT_ID_BOOST;
            coded_skip[r * MI_COLS + 3] = true;
        }
        cr.post_encode_update(&coded_segments, &coded_skip, 100);

        for r in 0..MI_ROWS {
            assert_eq!(cr.last_coded_q_map[r * MI_COLS], 100, "col 0 = base_q");
            assert_eq!(
                cr.last_coded_q_map[r * MI_COLS + 1],
                MAXQ,
                "col 1 base+skip preserves"
            );
            assert_eq!(
                cr.last_coded_q_map[r * MI_COLS + 2],
                92,
                "col 2 boost = base+delta"
            );
            assert_eq!(
                cr.last_coded_q_map[r * MI_COLS + 3],
                92,
                "col 3 boost+skip = min(255, 92)"
            );
        }
    }

    /// AQ-night-7 — eligibility uses the `last_coded_q_map > qindex_thresh` filter.
    /// Pre-seeding `last_coded_q_map` BELOW the threshold should suppress refresh
    /// even when age is sufficient.
    #[test]
    fn cyclic_refresh_eligibility_respects_last_coded_q_map() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64, // 25%
            segment1_q_delta: -8,
            min_refresh_age: 4,
        });
        // Age all cells past the gate.
        for v in cr.map.iter_mut() {
            *v = 4;
        }
        // Mark the top-left SB as "recently coded at fine q (50)" — below
        // the qindex_thresh = base + delta = 100 - 8 = 92. → NOT eligible.
        for r in 0..8 {
            for c in 0..8 {
                cr.last_coded_q_map[r * MI_COLS + c] = 50;
            }
        }
        let (seg_map, _) = cr.pre_encode_pick_segments(100);
        // Top-left SB: NOT refreshed (last_coded_q < qindex_thresh).
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    seg_map[r * MI_COLS + c],
                    0,
                    "top-left SB cells coded at q=50 < 92 must NOT be refreshed",
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // W5.5 / W5.6 PROOF tests
    // -----------------------------------------------------------------

    /// W5.5 PROOF — `vp9_compute_qdelta_by_rate` returns a negative
    /// delta when `rate_target_ratio > 1.0` (target more bits ⇒ finer
    /// Q ⇒ lower qindex ⇒ negative delta).
    ///
    /// Mirrors libvpx `vp9_ratectrl.c:2562-2584` — for any q above
    /// `best_quality`, doubling the bit budget must yield a smaller q.
    #[test]
    fn w5_5_compute_qdelta_by_rate_returns_negative_for_high_rate_factor() {
        // Mid-range qindex; libvpx-standard best/worst.
        let q = 128u8;
        // rate_factor = 2.0 — like libvpx's default `rate_ratio_qdelta`
        // (`vp9_aq_cyclicrefresh.c:520`).
        let dq = vp9_compute_qdelta_by_rate(q, FrameKind::Inter, 2.0, 0, 255);
        assert!(
            dq < 0,
            "rate_factor > 1 must produce negative qdelta, got {dq}"
        );
        // rate_factor = 0.5 — fewer bits ⇒ coarser Q ⇒ positive delta.
        let dq_neg = vp9_compute_qdelta_by_rate(q, FrameKind::Inter, 0.5, 0, 255);
        assert!(
            dq_neg > 0,
            "rate_factor < 1 must produce positive qdelta, got {dq_neg}",
        );
    }

    /// W5.5 PROOF — the three-segment Q-delta ramp is monotone
    /// non-increasing: seg0 (=0) ≥ seg1 ≥ seg2 (all ≤ 0). Mirrors
    /// libvpx `vp9_aq_cyclicrefresh.c:656-672` under the default
    /// `rate_boost_fac = 15`, `rate_ratio_qdelta = 2.0`.
    #[test]
    fn w5_5_qindex_delta_ramp_three_segments_monotone() {
        let q = 128u8;
        let ramp = qindex_delta_ramp(q, 2.0, 15, 60, FrameKind::Inter, 0, 255);
        assert_eq!(ramp[0], 0, "segment 0 (BASE) must always be 0");
        assert!(
            ramp[1] <= 0,
            "segment 1 must be ≤ 0 at rate_factor=2.0, got {}",
            ramp[1]
        );
        assert!(ramp[2] <= 0, "segment 2 must be ≤ 0, got {}", ramp[2]);
        assert!(
            ramp[2] <= ramp[1],
            "BOOST2 must be at least as aggressive as BOOST1: \
             seg1={} seg2={} (libvpx rate_boost_fac=15>10 ⇒ \
             boost2 rate factor = 0.1*15*2.0 = 3.0 > 2.0 ⇒ finer q)",
            ramp[1],
            ramp[2],
        );
        assert!(
            ramp[1] <= ramp[0],
            "BOOST1 must be ≤ BASE: seg0={} seg1={}",
            ramp[0],
            ramp[1],
        );
    }

    /// W5.5 PROOF — `compute_deltaq` clamps via `max_qdelta_perc`
    /// (libvpx `vp9_aq_cyclicrefresh.c:95-97`). At high `q`, a huge
    /// rate factor would otherwise produce an overshoot; the clamp
    /// caps `-deltaq` at `max_qdelta_perc * q / 100`.
    #[test]
    fn w5_5_compute_deltaq_respects_max_qdelta_perc_clamp() {
        let q = 200u8;
        // Tight clamp: max_qdelta_perc = 10 ⇒ -deltaq ≤ 10*200/100 = 20.
        let dq = compute_deltaq(q, 4.0, 10, FrameKind::Inter, 0, 255);
        assert!(
            (-dq) <= 20,
            "compute_deltaq must respect max_qdelta_perc clamp: dq={dq} > 20",
        );
        // Loose clamp: should be unclamped (or at the natural limit).
        let dq_loose = compute_deltaq(q, 4.0, 60, FrameKind::Inter, 0, 255);
        assert!(
            (-dq_loose) <= 60 * 200 / 100,
            "loose clamp dq={dq_loose} should still respect 60% bound",
        );
        // Loose clamp must be at least as aggressive as the tight one.
        assert!(
            dq_loose <= dq,
            "loose clamp ({dq_loose}) must be ≤ tight clamp ({dq})",
        );
    }

    /// W5.5 PROOF — at `qindex = 0` the BPM is finite and the delta is
    /// 0 (best-quality already; no further reduction possible). Sanity-
    /// check the LUT edge.
    #[test]
    fn w5_5_compute_qdelta_by_rate_handles_best_quality_edge() {
        // q=0 ⇒ AC_QLOOKUP[0]=4, convert_qindex_to_q=1.0; well-defined.
        // best_quality = worst_quality = 0 → loop range is empty,
        // target_index falls through to worst_quality = 0.
        let dq = vp9_compute_qdelta_by_rate(0, FrameKind::Inter, 2.0, 0, 0);
        assert_eq!(dq, 0, "at best=worst=0 the loop never runs, target=worst=0");
    }

    /// W5.6 PROOF — `vp9_cyclic_refresh_update_segment` promotes a
    /// big-zero-MV inter block (with `rate < thresh_rate_sb`) to
    /// BOOST2. Mirrors libvpx `vp9_aq_cyclicrefresh.c:80-84`.
    ///
    /// "High noise" in the test name corresponds to libvpx's intent:
    /// big static blocks (zero MV, low rate per block — the picker is
    /// finding them cheap to inter-code) are exactly the candidates
    /// that benefit most from a finer Q to suppress noise.
    #[test]
    fn w5_6_update_segment_promotes_to_boost2_on_high_noise() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        let mut segmap = vec![0u8; n_cells];

        // Big (16x16 = 2x2 mi) zero-MV inter block at (0,0). The
        // picker labelled it as BOOST1 (segment 1). rate=1 below
        // thresh_rate_sb=1000; dist=0 below thresh_dist_sb=1000;
        // rate_boost_fac=15 > 10. → must promote to BOOST2.
        cr.vp9_cyclic_refresh_update_segment(
            &mut segmap,
            /*mi_row*/ 0,
            /*mi_col*/ 0,
            /*bw_mi*/ 2,
            /*bh_mi*/ 2,
            /*rate*/ 1,
            /*dist*/ 0,
            /*mv_row_abs*/ 0,
            /*mv_col_abs*/ 0,
            /*is_inter*/ true,
            /*is_zero_mv*/ true,
            /*is_skin*/ false,
            /*prior_segment_id*/ CR_SEGMENT_ID_BOOST1,
            /*skip*/ false,
            /*use_nonrd_pick_mode*/ true,
            /*is_vbr_golden*/ false,
            /*bsize_is_ge_16x16*/ true,
            /*bsize_is_le_16x16*/ true,
            /*use_skin_detection*/ false,
            /*thresh_rate_sb*/ 1000,
            /*thresh_dist_sb*/ 1000,
            /*motion_thresh*/ 4,
            /*rate_boost_fac*/ 15,
        );
        // All 2x2 cells of the block must be CR_SEGMENT_ID_BOOST2.
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(
                    segmap[y * MI_COLS + x],
                    CR_SEGMENT_ID_BOOST2,
                    "cell ({y},{x}) must be promoted to BOOST2",
                );
            }
        }
    }

    /// W5.6 PROOF — VBR-GOLDEN demotion: if `is_vbr_golden`, the block
    /// is forced back to BASE (libvpx `vp9_aq_cyclicrefresh.c:186-187`).
    #[test]
    fn w5_6_update_segment_demotes_vbr_golden_to_base() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        let mut segmap = vec![0u8; n_cells];
        cr.vp9_cyclic_refresh_update_segment(
            &mut segmap,
            0,
            0,
            2,
            2,
            1,
            0,
            0,
            0,
            true,
            true,
            false,
            CR_SEGMENT_ID_BOOST1,
            false,
            true,
            /*is_vbr_golden*/ true,
            true,
            true,
            false,
            1000,
            1000,
            4,
            15,
        );
        // VBR-GOLDEN: even the BOOST1 picker label is demoted to BASE.
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(
                    segmap[y * MI_COLS + x],
                    CR_SEGMENT_ID_BASE,
                    "VBR-GOLDEN cell ({y},{x}) must demote to BASE",
                );
            }
        }
    }

    /// W5.6 PROOF — skin promotion: a block that isn't otherwise a
    /// refresh candidate (high dist + intra) gets promoted to BOOST1
    /// when `is_skin && bsize ≤ 16x16 && use_skin_detection`. Mirrors
    /// libvpx `vp9_aq_cyclicrefresh.c:177-184`.
    #[test]
    fn w5_6_update_segment_skin_promotes_to_boost1() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 0,
        });
        let n_cells = MI_ROWS * MI_COLS;
        let mut segmap = vec![0u8; n_cells];
        // High dist + intra → candidate_refresh_aq returns BASE.
        // Skin override should re-promote to BOOST1.
        cr.vp9_cyclic_refresh_update_segment(
            &mut segmap,
            0,
            0,
            2,
            2,
            10_000,
            10_000,
            0,
            0,
            /*is_inter*/ false,
            /*is_zero_mv*/ false,
            /*is_skin*/ true,
            /*prior_segment_id*/ CR_SEGMENT_ID_BOOST1,
            false,
            true,
            false,
            true,
            /*bsize_is_le_16x16*/ true,
            /*use_skin_detection*/ true,
            100,
            100,
            4,
            15,
        );
        // Even though candidate_refresh_aq would have returned BASE
        // (intra + high dist), skin promotion lifts to BOOST1.
        assert_eq!(
            segmap[0], CR_SEGMENT_ID_BOOST1,
            "skin block must be promoted to BOOST1",
        );
    }

    /// AQ-night-7 — half-eligibility rule: SB refreshes when >= half cells
    /// are eligible, not all.
    #[test]
    fn cyclic_refresh_half_eligibility_rule() {
        let mut cr = make_cr(CyclicRefreshConfig {
            target_refresh_fraction_256: 64,
            segment1_q_delta: -8,
            min_refresh_age: 4,
        });
        // Top-left SB: 32 of 64 cells aged enough (=half exactly).
        // Layout: rows 0..4 all aged, rows 4..8 not aged.
        for r in 0..4 {
            for c in 0..8 {
                cr.map[r * MI_COLS + c] = 4;
            }
        }
        // last_coded_q_map seeded to MAXQ everywhere → q_ok always true.
        let (seg_map, _) = cr.pre_encode_pick_segments(100);
        // The "≥ half" rule should still trip (32 >= 64/2 = 32).
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(
                    seg_map[r * MI_COLS + c],
                    CR_SEGMENT_ID_BOOST,
                    "SB with exactly half cells eligible should be refreshed",
                );
            }
        }
    }
}
