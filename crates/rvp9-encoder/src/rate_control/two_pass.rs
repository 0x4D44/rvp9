//! Two-pass VBR rate control.
//!
//! Bounded port of the per-sequence two-pass VBR path in libvpx
//! `vp9/encoder/vp9_firstpass.c` + `vp9/encoder/vp9_ratectrl.c`.
//!
//! ## Architecture (per-SEQUENCE two-pass)
//!
//! libvpx's two-pass runs the whole sequence once in a stats-only
//! "first pass" that writes a [`FirstPassStats`] record per frame
//! (`vp9_first_pass`, `vp9_firstpass.c:1392`; the per-frame
//! `FIRSTPASS_STATS` struct is `vp9_firstpass_stats.h:20-48` — we model
//! the `coded_error` / `intra_error` / `pcnt_inter` subset). A second
//! pass then reads the stats, distributes the sequence bit budget across
//! frames by their relative coded-error "score"
//! (`calculate_norm_frame_score` / `gf_group->bit_allocation`,
//! `vp9_firstpass.c:279`, `:2334-2440`), and re-encodes emitting the
//! real bitstream, with the per-frame Q regulated to hit each frame's
//! allocated target (`get_twopass_worst_quality`,
//! `vp9_firstpass.c:1557-1626`; `rc_pick_q_and_bounds_two_pass`,
//! `vp9_ratectrl.c:1427`).
//!
//! ## What this module implements (bounded scope)
//!
//! - [`FirstPassStats`] — the per-frame stats record (libvpx subset:
//!   `coded_error`, `intra_error`, `pcnt_inter`).
//! - [`TwoPassState`] — the second-pass driver state: the full stats
//!   buffer, a frame cursor, the precomputed per-frame **bit
//!   allocation** (a real, stats-driven distribution of the sequence
//!   budget), and the total-error normaliser.
//! - [`pick_q_and_bounds`] — the 2-pass `pick_q` arm. For each frame it
//!   reads the precomputed per-frame target (allocated from the
//!   first-pass error distribution, NOT the flat 1-pass GOP-amortised
//!   target) and runs [`super::cbr::regulate_q`] within an
//!   error-driven active-Q range. The per-frame target — and therefore
//!   the picked Q — is a function of the first-pass stats, so a 2-pass
//!   encode demonstrably differs from a 1-pass encode of the same input.
//!
//! ## DOCUMENTED DIVERGENCES (deferred follow-ups)
//!
//! The full libvpx two-pass is ~800-1200 LOC. This is a bounded, real
//! two-pass that lands the core "first-pass stats drive pass-2 bit
//! allocation" loop. Intentionally NOT ported here:
//!
//! - **ARF placement / GF-group structure** (`define_gf_group` +
//!   `define_gf_group_structure`, `vp9_firstpass.c:2257-2302, 2692`).
//!   We allocate per-frame, not per-GF-group with ARF boost. The
//!   `GfGroup` (`crate::gop`) already carries the 1-pass schedule; the
//!   2-pass ARF-boost reshuffle is the documented next step.
//! - **KF group detection** (`find_next_key_frame`,
//!   `vp9_firstpass.c:3247`). We honour the configured `gop_length` KF
//!   cadence (the encoder's existing schedule), not a stats-driven
//!   adaptive KF placement.
//! - **The first-pass motion search** (`vp9_first_pass` runs a real
//!   per-MB motion search to gather `MVr`/`MVc`/`pcnt_motion`). Our
//!   first pass gathers `coded_error` from the actual coded
//!   reconstruction and a DC-intra `intra_error` proxy + a
//!   motion-effectiveness `pcnt_inter` proxy — three genuine,
//!   content-driven stats sufficient to drive the allocation. The full
//!   motion-stats set (`MVrv`, `mv_in_out_count`, ...) is deferred.
//! - **`bpm_factor` / noise / inactive-zone adjustments**
//!   (`get_twopass_worst_quality`, `vp9_firstpass.c:1568-1620`).

use super::cbr::{FRAME_OVERHEAD_BITS, get_rate_correction_factor, regulate_q, rtc_minq_approx};
use super::{FrameKind, RateControl};

/// Per-frame first-pass statistics.
///
/// Models the `coded_error` / `intra_error` / `pcnt_inter` subset of
/// libvpx's `FIRSTPASS_STATS` (`vp9_firstpass_stats.h:20-48`). These are
/// the three stats the bounded two-pass allocation consumes:
///
/// - `coded_error` — sum-of-squared-error of the actual coded
///   reconstruction vs the source (libvpx accumulates this per-MB in
///   `vp9_first_pass`, `vp9_firstpass.c:1392`). The dominant signal: a
///   frame that costs more residual energy to code gets a larger share
///   of the sequence bit budget.
/// - `intra_error` — SSE of a DC (frame-mean) intra prediction vs the
///   source. A cheap, content-driven proxy for libvpx's per-MB
///   best-intra error; used to derive `pcnt_inter` and as an alternate
///   complexity floor.
/// - `pcnt_inter` — fraction of the frame "explained" by inter
///   prediction, derived from how far `coded_error` beats `intra_error`
///   (motion-compensation effectiveness). `0.0` on keyframes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FirstPassStats {
    /// `true` if this frame was coded as a keyframe in the first pass.
    pub is_key: bool,
    /// SSE of coded reconstruction vs source (libvpx `coded_error`).
    pub coded_error: f64,
    /// SSE of a DC-intra prediction vs source (libvpx `intra_error`).
    pub intra_error: f64,
    /// Inter-prediction effectiveness in `[0, 1]` (libvpx `pcnt_inter`).
    pub pcnt_inter: f64,
}

impl FirstPassStats {
    /// Per-frame "complexity score" driving the bit allocation. Mirrors
    /// the spirit of libvpx `calculate_norm_frame_score`
    /// (`vp9_firstpass.c:279`): a frame's score is its coded-error
    /// magnitude (the harder-to-code the frame, the more bits it earns).
    /// Floored at `1.0` so a degenerate all-zero-error frame still earns
    /// a non-zero share (libvpx's `DOUBLE_DIVIDE_CHECK` guards the same
    /// way).
    #[must_use]
    pub fn score(&self) -> f64 {
        self.coded_error.max(1.0)
    }
}

/// Second-pass driver state.
///
/// Holds the complete first-pass stats buffer, a frame cursor, and the
/// precomputed per-frame bit allocation. Constructed once from the
/// gathered first-pass stats + the sequence bit budget; consumed
/// frame-by-frame by [`pick_q_and_bounds`] via the cursor.
#[derive(Clone, Debug)]
pub struct TwoPassState {
    /// One [`FirstPassStats`] per display frame, in encode order.
    stats: Vec<FirstPassStats>,
    /// Precomputed per-frame target bits. `bit_allocation[i]` is the
    /// share of the sequence budget assigned to frame `i`, proportional
    /// to that frame's [`FirstPassStats::score`]. Same length as
    /// `stats`. This is the genuinely-stats-driven quantity — it differs
    /// from the 1-pass flat GOP-amortised target.
    bit_allocation: Vec<u32>,
    /// Cursor into `stats` / `bit_allocation` — the next frame's index.
    cursor: usize,
}

impl TwoPassState {
    /// Build the second-pass state from gathered first-pass `stats` and
    /// the sequence bit budget.
    ///
    /// The budget is distributed across frames proportional to each
    /// frame's coded-error score (libvpx `gf_group->bit_allocation`,
    /// `vp9_firstpass.c:2334-2440`, distributes the group budget by the
    /// per-frame normalised score). Keyframes additionally receive a
    /// boost (libvpx allocates the KF a large share via `kf_bits`,
    /// `vp9_firstpass.c:3425`); we apply a fixed KF score multiplier so
    /// the keyframe earns proportionally more of the budget, matching
    /// the libvpx "spend more on the intra anchor" behaviour.
    ///
    /// `avg_frame_bandwidth` is the per-frame average bit budget
    /// (`target_bps / framerate`); the sequence budget is
    /// `avg_frame_bandwidth * n_frames`.
    #[must_use]
    pub fn new(stats: Vec<FirstPassStats>, avg_frame_bandwidth: u32) -> Self {
        let n = stats.len();
        if n == 0 {
            return Self {
                stats,
                bit_allocation: Vec::new(),
                cursor: 0,
            };
        }

        // libvpx keyframe boost: the intra anchor earns a larger share
        // of the budget. We weight the KF score by KF_SCORE_BOOST so a
        // KF with the same coded-error as an inter frame still gets a
        // bigger allocation (mirrors `kf_bits` >> per-inter allocation in
        // `vp9_firstpass.c:3409-3425`).
        const KF_SCORE_BOOST: f64 = 8.0;

        let weighted: Vec<f64> = stats
            .iter()
            .map(|s| {
                let base = s.score();
                if s.is_key {
                    base * KF_SCORE_BOOST
                } else {
                    base
                }
            })
            .collect();

        let total_score: f64 = weighted.iter().sum::<f64>().max(1.0);

        // Sequence budget = avg_frame_bandwidth * n_frames.
        let seq_budget: f64 = f64::from(avg_frame_bandwidth) * n as f64;

        // Per-frame target = budget * (frame_score / total_score),
        // floored at FRAME_OVERHEAD_BITS so no frame is starved below
        // the header cost.
        let bit_allocation: Vec<u32> = weighted
            .iter()
            .map(|w| {
                let share = seq_budget * (w / total_score);
                let clamped = share.max(f64::from(FRAME_OVERHEAD_BITS));
                if clamped >= f64::from(u32::MAX) {
                    u32::MAX
                } else {
                    clamped as u32
                }
            })
            .collect();

        Self {
            stats,
            bit_allocation,
            cursor: 0,
        }
    }

    /// Number of frames the first pass recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    /// `true` if no first-pass stats were gathered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// The current frame's precomputed bit allocation, or `None` if the
    /// cursor has run past the recorded stats (a robustness fallback —
    /// pass 2 should encode exactly the frames pass 1 saw).
    fn current_target(&self) -> Option<u32> {
        self.bit_allocation.get(self.cursor).copied()
    }

    /// Advance the cursor after a frame has been picked. Called once per
    /// frame by [`RateControl::advance_frame`] when in two-pass mode.
    pub(super) fn advance(&mut self) {
        self.cursor = self.cursor.saturating_add(1);
    }

    /// Read-only access to the per-frame bit allocation (test / probe).
    #[must_use]
    pub fn bit_allocation(&self) -> &[u32] {
        &self.bit_allocation
    }
}

/// Pick the Q-index and `(bottom, top)` bounds for the next frame under
/// two-pass VBR.
///
/// Mirrors `rc_pick_q_and_bounds_two_pass` (`vp9_ratectrl.c:1427`) for
/// the bounded, non-ARF path:
///
/// 1. The per-frame target is the **first-pass-driven allocation**
///    ([`TwoPassState::current_target`]) — NOT the 1-pass flat target.
///    This is the load-bearing difference: the target (hence the picked
///    Q) is a function of the first-pass `coded_error` distribution.
/// 2. The active-best floor mirrors the 2-pass inter path
///    (`inter_minq[active_worst]`, `vp9_ratectrl.c:1492`) via the CBR
///    `rtc_minq_approx`; keyframes float to `best_quality`.
/// 3. The active-worst ceiling is the controller's `worst_quality`
///    (libvpx seeds `active_worst_quality` from
///    `get_twopass_worst_quality`; we use the controller bound, the
///    regulate_q step does the rate-vs-Q work).
/// 4. `regulate_q` picks the Q hitting the per-frame target within the
///    active range (`vp9_firstpass.c:1612-1619`).
pub(super) fn pick_q_and_bounds(rc: &mut RateControl, frame_kind: FrameKind) -> (u8, u8, u8) {
    // 1. Per-frame target bits from the first-pass allocation. If the
    //    cursor has run off the end (pass-2 frame count mismatch), fall
    //    back to the average per-frame bandwidth.
    let target_bits = rc
        .two_pass
        .as_ref()
        .and_then(TwoPassState::current_target)
        .unwrap_or(rc.avg_frame_bandwidth)
        .max(FRAME_OVERHEAD_BITS);
    rc.this_frame_target = target_bits;

    // 2/3. Active Q range. Keyframes get the full [best, worst] range
    //    with the best-quality floor; inter frames floor the active-best
    //    at the rtc_minq approximation of the worst-quality ceiling
    //    (libvpx 2-pass inter: `active_best = inter_minq[active_worst]`,
    //    `vp9_ratectrl.c:1492`).
    let active_worst = rc.worst_quality;
    let active_best = match frame_kind {
        FrameKind::Key => rc.best_quality,
        FrameKind::Inter => rtc_minq_approx(rc, active_worst),
    };
    let active_best = active_best.clamp(rc.best_quality, rc.worst_quality);
    let active_worst = active_worst.clamp(active_best, rc.worst_quality);

    // 4. Regulate Q within the active range.
    let correction_factor = get_rate_correction_factor(rc, frame_kind);
    let multiplier = rc.config().bits_per_mb_multiplier;
    let q = regulate_q(
        frame_kind,
        target_bits,
        active_best,
        active_worst,
        rc.mbs,
        correction_factor,
        multiplier,
    );
    let q = q.clamp(active_best, active_worst);

    rc.active_min_qindex = active_best as u8;
    rc.active_max_qindex = active_worst as u8;
    rc.last_picked_qindex = q as u8;

    (q as u8, active_best as u8, active_worst as u8)
}

#[cfg(test)]
mod tests {
    use super::super::RcMode;
    use super::super::config::RateControlConfig;
    use super::*;

    fn stat(is_key: bool, coded_error: f64) -> FirstPassStats {
        FirstPassStats {
            is_key,
            coded_error,
            intra_error: coded_error * 2.0,
            pcnt_inter: if is_key { 0.0 } else { 0.5 },
        }
    }

    #[test]
    fn allocation_proportional_to_coded_error() {
        // Two inter frames: one with 4x the coded error gets ~4x the
        // bits (proportional allocation).
        let stats = vec![stat(false, 100.0), stat(false, 400.0)];
        let tp = TwoPassState::new(stats, 10_000);
        let alloc = tp.bit_allocation();
        assert_eq!(alloc.len(), 2);
        // Frame 1 (4x error) should get ~4x the allocation of frame 0.
        let ratio = f64::from(alloc[1]) / f64::from(alloc[0]);
        assert!(
            (3.5..=4.5).contains(&ratio),
            "high-error frame must earn ~4x: alloc={alloc:?}, ratio={ratio}"
        );
    }

    #[test]
    fn keyframe_earns_boosted_share() {
        // A KF with the SAME coded error as an inter frame must earn a
        // larger allocation (the KF_SCORE_BOOST multiplier).
        let stats = vec![stat(true, 100.0), stat(false, 100.0)];
        let tp = TwoPassState::new(stats, 10_000);
        let alloc = tp.bit_allocation();
        assert!(
            alloc[0] > alloc[1],
            "KF must earn more than an equal-error inter: alloc={alloc:?}"
        );
    }

    #[test]
    fn empty_stats_yields_empty_allocation() {
        let tp = TwoPassState::new(Vec::new(), 10_000);
        assert!(tp.is_empty());
        assert_eq!(tp.len(), 0);
        assert!(tp.bit_allocation().is_empty());
    }

    #[test]
    fn cursor_advances_through_frames() {
        let stats = vec![stat(true, 100.0), stat(false, 200.0)];
        let mut tp = TwoPassState::new(stats, 10_000);
        assert_eq!(tp.cursor, 0);
        let t0 = tp.current_target();
        tp.advance();
        assert_eq!(tp.cursor, 1);
        let t1 = tp.current_target();
        assert_ne!(t0, t1, "different frames have different targets");
        tp.advance();
        assert_eq!(tp.current_target(), None, "past-end yields None");
    }

    fn two_pass_config() -> RateControlConfig {
        RateControlConfig {
            rc_mode: RcMode::VbrTwoPass,
            ..RateControlConfig::default()
        }
    }

    #[test]
    fn pick_q_returns_in_range_triple() {
        // The 2-pass arm returns a valid (q, lo, hi) triple inside the
        // controller's [best, worst] budget.
        let mut rc = RateControl::new(two_pass_config());
        rc.attach_two_pass(TwoPassState::new(vec![stat(false, 100.0)], 10_000));
        let (q, lo, hi) = rc.pick_q_and_bounds(FrameKind::Inter);
        assert!(lo <= q && q <= hi, "q={q} must lie in [{lo},{hi}]");
        assert!(i32::from(lo) >= rc.config().min_qindex as i32);
        assert!(i32::from(hi) <= rc.config().max_qindex as i32);
    }

    #[test]
    fn pick_q_differs_when_allocation_differs() {
        // Two-frame sequence where frame 0 has tiny error and frame 1
        // has huge error. Frame 1's larger target → lower Q than frame
        // 0's smaller target, proving the stats drive the per-frame Q.
        let stats = vec![stat(false, 10.0), stat(false, 1_000_000.0)];
        let mut rc = RateControl::new(two_pass_config());
        rc.attach_two_pass(TwoPassState::new(stats, 50_000));
        let (q0, _, _) = rc.pick_q_and_bounds(FrameKind::Inter);
        rc.postencode_update(rc.this_frame_target, FrameKind::Inter);
        rc.advance_frame();
        let (q1, _, _) = rc.pick_q_and_bounds(FrameKind::Inter);
        assert!(
            q1 < q0,
            "high-error frame (more bits) must pick lower Q than \
             low-error frame: q0={q0}, q1={q1}"
        );
    }
}
