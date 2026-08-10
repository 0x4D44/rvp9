//! Post-encode state update.
//!
//! Port of `vp9_rc_postencode_update` (`vp9_ratectrl.c:1845-1995`),
//! restricted to the fields actually used by the 1-pass CBR pick logic.

use super::cbr::{
    self, MAX_BPB_FACTOR, MIN_BPB_FACTOR, RateFactorLevel, estimate_bits_at_q_with_model,
    get_rate_correction_factor, set_rate_correction_factor,
};
use super::{CyclicRefreshRateModel, FrameKind, RateControl, RcMode};

/// Port of `vp9_rc_update_rate_correction_factors`
/// (`vp9_ratectrl.c:737-817`).
///
/// Closes the loop between *predicted* bits-at-Q and *actual* bits-at-Q
/// by multiplicatively walking the per-level correction factor:
///
/// ```text
/// projected = estimate_bits_at_q(frame_type, picked_q, MBs, current_factor)
/// correction_factor_pct = 100 * actual_bits / projected   // libvpx int math
/// adjustment_limit = first_time ? 1.0
///                                : 0.25 + 0.5 * min(1, |log10(0.01 * cf)|)
/// if cf > 102: factor *= (100 + (cf - 100) * adj) / 100
/// elif cf < 99: factor *= (100 - (100 - cf) * adj) / 100
/// clamp to [MIN_BPB_FACTOR, MAX_BPB_FACTOR]
/// ```
///
/// For CBR, the libvpx oscillation history (`q_1_frame` / `q_2_frame` /
/// `rc_1_frame` / `rc_2_frame`, lines 781-795) is updated here and consumed
/// by `cbr::adjust_q_cbr` to clamp resonating Q picks. The aggressive
/// GF-overshoot rescue path (`vp9_encodedframe_overshoot`) remains out of
/// scope; one-pass VBR does not use the CBR resonance history.
fn update_cbr_resonance_history(rc: &mut RateControl, correction_pct: i64) {
    rc.q_2_frame = rc.q_1_frame;
    rc.q_1_frame = rc.last_picked_qindex;
    rc.rc_2_frame = rc.rc_1_frame;
    rc.rc_1_frame = if correction_pct > 110 {
        -1
    } else if correction_pct < 90 {
        1
    } else {
        0
    };

    // A massive overshoot immediately after an undershoot should not look
    // like a normal alternation to adjust_q_cbr.
    if rc.rc_1_frame == -1 && rc.rc_2_frame == 1 && correction_pct > 1000 {
        rc.rc_2_frame = 0;
    }
}

fn update_rate_correction_factors(
    rc: &mut RateControl,
    actual_bits: u32,
    frame_kind: FrameKind,
    model: Option<CyclicRefreshRateModel>,
) {
    // 1. Recover the factor in force when this frame's Q was picked.
    let current_factor = get_rate_correction_factor(rc, frame_kind);

    // 2. Re-project the frame size at the picked Q under that factor.
    //    libvpx uses `cm->base_qindex` (== the Q the encoder actually
    //    handed to the entropy coder); we record that in
    //    `rc.last_picked_qindex` in `cbr::pick_q_and_bounds`.
    let picked_q = i32::from(rc.last_picked_qindex);
    let multiplier = rc.config().bits_per_mb_multiplier;
    let projected = estimate_bits_at_q_with_model(
        frame_kind,
        picked_q,
        rc.mbs,
        current_factor,
        multiplier,
        model,
    );

    // 3. Compute the integer correction-factor percentage. libvpx uses a
    //    neutral 100% sample when the estimate is only frame overhead; CBR
    //    still consumes that sample for its resonance history.
    let projected_above_overhead = projected > i64::from(cbr::FRAME_OVERHEAD_BITS);
    let correction_pct: i64 = if projected_above_overhead {
        (100_i64 * i64::from(actual_bits)) / projected
    } else {
        100
    };

    // The Q-resonance history is a CBR-only input to the regulator. Keep
    // VBR and the other non-CBR paths unchanged, including their former
    // no-op overhead-only correction update.
    if matches!(rc.config().rc_mode, RcMode::Cbr) {
        update_cbr_resonance_history(rc, correction_pct);
    }

    // 4. Damping. First frame at this rate-factor level: full
    //    correction. Subsequent: damp by the log-magnitude of the
    //    miss (libvpx `vp9_ratectrl.c:770-779`).
    // For non-CBR modes, preserve the previous early-return behaviour when
    // the estimate contains only frame overhead. CBR must continue so the
    // neutral history sample reaches the damped update below.
    if !projected_above_overhead && !matches!(rc.config().rc_mode, RcMode::Cbr) {
        return;
    }

    let level = RateFactorLevel::for_frame(frame_kind);
    // In pinned libvpx one-pass VBR, `get_rate_correction_factor()` still
    // selects independent KF / inter / GF factor values. The damping gate is
    // different: `vp9_rc_update_rate_correction_factors()` indexes
    // `damped_adjustment[]` through the otherwise-unused two-pass GF-group
    // level, which remains `INTER_NORMAL` for every one-pass frame. The first
    // keyframe therefore arms damping for the following inter and GOLDEN
    // updates instead of letting each factor slot make one undamped jump.
    let damping_level = if matches!(rc.config().rc_mode, RcMode::Vbr) {
        RateFactorLevel::InterNormal
    } else {
        level
    };
    let slot = damping_level.as_index();
    let adjustment_limit: f64 = if !rc.damped_adjustment[slot] {
        rc.damped_adjustment[slot] = true;
        1.0
    } else {
        let pct_f = correction_pct as f64;
        let abs_log = (0.01 * pct_f).abs().log10().abs();
        0.25 + 0.5 * abs_log.min(1.0)
    };

    // 5. Multiplicatively walk the per-level factor. libvpx's gates
    //    are 102 / 99 (so a small dead-band on either side is
    //    ignored). We replicate the int-math truncation exactly so
    //    behaviour matches the reference at the bit level.
    let mut new_factor = current_factor;
    if correction_pct > 102 {
        let damped = (100_i64 + (((correction_pct - 100) as f64) * adjustment_limit) as i64) as f64;
        new_factor = (new_factor * damped) / 100.0;
        if new_factor > MAX_BPB_FACTOR {
            new_factor = MAX_BPB_FACTOR;
        }
    } else if correction_pct < 99 {
        let damped = (100_i64 - (((100 - correction_pct) as f64) * adjustment_limit) as i64) as f64;
        new_factor = (new_factor * damped) / 100.0;
        if new_factor < MIN_BPB_FACTOR {
            new_factor = MIN_BPB_FACTOR;
        }
    }
    // else: inside the [99, 102] dead-band — leave factor unchanged.

    set_rate_correction_factor(rc, frame_kind, new_factor);
}

/// Update buffer + Q history given the frame just encoded.
///
/// libvpx flow (`vp9_ratectrl.c:1845-1975`):
///
/// 1. `projected_frame_size = bytes_used << 3` — we receive bits
///    already.
/// 2. `update_rate_correction_factors` — port at
///    [`update_rate_correction_factors`].
/// 3. Update `last_q[type]`, `avg_frame_qindex[type]` via
///    `(3 * prev + new) >> 2`.
/// 4. `update_buffer_level_postencode`:
///    - `bits_off_target += avg_frame_bandwidth` (the preencode
///      half — libvpx does this in a separate function but we fold
///      it in here so the caller has a single touchpoint).
///    - `bits_off_target -= encoded_frame_size`.
///    - Clamp to `[-max_buf, +max_buf]`.
///    - `buffer_level = bits_off_target`.
/// 5. Update `total_actual_bits` / `total_target_bits`.
/// 6. If KF: `frames_since_key = 0`; else `frames_since_key++`.
/// 7. `frames_to_key--`.
///
/// Steps 6 + 7 live in [`RateControl::advance_frame`] so the caller
/// keeps full control.
///
/// [`RateControl::advance_frame`]: super::RateControl::advance_frame
pub(super) fn postencode_update(rc: &mut RateControl, actual_bits: u32, frame_kind: FrameKind) {
    postencode_update_with_model(rc, actual_bits, frame_kind, None);
}

pub(super) fn postencode_update_with_model(
    rc: &mut RateControl,
    actual_bits: u32,
    frame_kind: FrameKind,
    model: Option<CyclicRefreshRateModel>,
) {
    postencode_update_impl(rc, actual_bits, frame_kind, model, false);
}

pub(super) fn postencode_update_svc_with_model(
    rc: &mut RateControl,
    actual_bits: u32,
    frame_kind: FrameKind,
    model: Option<CyclicRefreshRateModel>,
) {
    postencode_update_impl(rc, actual_bits, frame_kind, model, true);
}

fn postencode_update_impl(
    rc: &mut RateControl,
    actual_bits: u32,
    frame_kind: FrameKind,
    model: Option<CyclicRefreshRateModel>,
    preencode_credit_applied: bool,
) {
    // -- Adapt the per-level rate-correction factor BEFORE we touch
    //    last_q / avg_frame_qindex (libvpx ordering — line 1859 then
    //    line 1864). The damped update reads `last_picked_qindex`,
    //    not the EWMA history, so the order matters only for symmetry
    //    with libvpx.
    update_rate_correction_factors(rc, actual_bits, frame_kind, model);

    // -- Update last_q / avg_frame_qindex --
    // libvpx `vp9_rc_postencode_update:1850` records the *actual coded*
    // frame Q (`cm->base_qindex`) into `last_q[]` and the ambient EWMA
    // `avg_frame_qindex[] = (3*avg + qindex) >> 2`. The frame loop codes
    // at `last_picked_qindex` (the `regulate_q` result, clamped to
    // `[active_best, active_worst]`), so that — NOT `active_min_qindex`
    // (the active_best floor) — is the value to record.
    //
    // E8 FIX: this previously recorded `active_min_qindex`. Because the
    // inter `active_best` is `rtc_minq_approx(ambient) ≈ 0.75·ambient`,
    // feeding it back as the next-frame ambient drove a geometric decay
    // (ambient 255→30 over 31 frames on foreman). The collapsing ambient
    // pulled the buffer-ramp `active_worst` floor down with it (the ramp
    // interpolates UP *from ambient*), so Q was pinned ~190-211 instead
    // of rising toward 255 as the buffer drained — the inter overshoot.
    // Recording the real coded Q keeps the ambient where libvpx keeps it.
    let qindex = rc.last_picked_qindex;
    match frame_kind {
        FrameKind::Key => {
            rc.last_q_keyframe = qindex;
            // libvpx `ROUND_POWER_OF_TWO(..., 2)` adds the half-step
            // before shifting, so this is rounded rather than truncated.
            let new_avg = (3 * u32::from(rc.avg_frame_qindex_key) + u32::from(qindex) + 2) >> 2;
            rc.avg_frame_qindex_key = new_avg as u8;
        }
        FrameKind::Inter => {
            rc.last_q_inter = qindex;
            let new_avg = (3 * u32::from(rc.avg_frame_qindex_inter) + u32::from(qindex) + 2) >> 2;
            rc.avg_frame_qindex_inter = new_avg as u8;
        }
        // Pinned libvpx excludes boosted GOLDEN/ALTREF refreshes from normal
        // inter Q history. Feeding them back contaminates both the ambient
        // inter bound and the next GOLDEN bound.
        FrameKind::Golden => {}
    }

    // -- Buffer update --
    // Ordinary rvp9 folds libvpx's preencode credit into this update. SVC
    // applies the cumulative credit to the active and all higher layers
    // before Q selection, so its postencode path only debits actual bits.
    let delta = if preencode_credit_applied {
        -i64::from(actual_bits)
    } else {
        i64::from(rc.avg_frame_bandwidth) - i64::from(actual_bits)
    };
    rc.bits_off_target = rc.bits_off_target.saturating_add(delta);
    // Clamp upper bound at max_buffer_size (libvpx).
    if rc.bits_off_target > rc.maximum_buffer_size {
        rc.bits_off_target = rc.maximum_buffer_size;
    }
    // DOCUMENTED DIVERGENCE: clamp lower bound at `-maximum_buffer_size`.
    // libvpx only clamps the lower bound for VP9E_CONTENT_SCREEN; for
    // normal content it lets the buffer go arbitrarily negative because
    // the frame-drop logic will engage. We don't have drop logic in
    // v1, so we cap the deficit to keep the buffer math bounded.
    if rc.bits_off_target < -rc.maximum_buffer_size {
        rc.bits_off_target = -rc.maximum_buffer_size;
    }
    rc.buffer_level = rc.bits_off_target;

    // -- Totals --
    rc.total_actual_bits = rc.total_actual_bits.saturating_add(i64::from(actual_bits));
    // libvpx: `total_target_bits += show_frame ? avg_frame_bandwidth : 0;`
    // We have no show-frame distinction at this layer, so we always
    // accumulate.
    rc.total_target_bits = rc
        .total_target_bits
        .saturating_add(i64::from(rc.avg_frame_bandwidth));

    // -- KF reset of frames_since_key --
    // libvpx does this at the bottom of postencode_update
    // (`vp9_ratectrl.c:1971`). Putting it here keeps the encoder's
    // GOP arithmetic correct without the caller having to do the
    // bookkeeping.
    if frame_kind.is_key() {
        rc.frames_since_key = 0;
        rc.frames_to_key = rc.config().gop_length.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::super::cbr::RateFactorLevel;
    use super::super::config::RateControlConfig;
    use super::super::{FrameKind, RateControl, RcMode};
    use super::{
        postencode_update_svc_with_model, update_cbr_resonance_history,
        update_rate_correction_factors,
    };

    fn fresh_rc() -> RateControl {
        let mut rc = RateControl::new(RateControlConfig::default());
        // Run one KF pick so this_frame_target / active_min are
        // populated.
        let _ = rc.pick_q_and_bounds(FrameKind::Key);
        rc
    }

    #[test]
    fn rate_control_postencode_balanced() {
        // Test #6 — encode exactly avg_frame_bandwidth → buffer
        // returns to where it was.
        let mut rc = fresh_rc();
        let before = rc.buffer_level;
        let bits = rc.avg_frame_bandwidth;
        rc.postencode_update(bits, FrameKind::Key);
        let after = rc.buffer_level;
        assert_eq!(
            after, before,
            "balanced spend should not move buffer (before={before}, after={after})"
        );
    }

    #[test]
    fn rate_control_postencode_over_budget() {
        // Test #7 — 2x avg → buffer falls by ~avg_frame_bandwidth.
        let mut rc = fresh_rc();
        let before = rc.buffer_level;
        let spend = rc.avg_frame_bandwidth * 2;
        rc.postencode_update(spend, FrameKind::Inter);
        let after = rc.buffer_level;
        let expected_delta = -i64::from(rc.avg_frame_bandwidth);
        assert_eq!(
            after - before,
            expected_delta,
            "2x avg spend: expected buffer to drop by avg_frame_bandwidth, \
             got {} (before={before}, after={after})",
            after - before
        );
    }

    #[test]
    fn rate_control_postencode_under_budget() {
        // Test #8 — 0.5x avg → buffer rises by 0.5x avg.
        let mut rc = fresh_rc();
        let before = rc.buffer_level;
        let spend = rc.avg_frame_bandwidth / 2;
        rc.postencode_update(spend, FrameKind::Inter);
        let after = rc.buffer_level;
        let expected_delta = i64::from(rc.avg_frame_bandwidth) - i64::from(spend);
        assert_eq!(
            after - before,
            expected_delta,
            "0.5x avg spend: expected buffer to rise by half avg, \
             got {} (before={before}, after={after})",
            after - before
        );
    }

    #[test]
    fn svc_preencode_credit_is_not_applied_twice() {
        let mut rc = fresh_rc();
        rc.bits_off_target = 0;
        rc.buffer_level = 0;
        let avg = rc.avg_frame_bandwidth;
        let actual = avg / 2;

        rc.svc_preencode_buffer_update();
        postencode_update_svc_with_model(&mut rc, actual, FrameKind::Inter, None);

        assert_eq!(rc.bits_off_target, i64::from(avg - actual));
        assert_eq!(rc.buffer_level, rc.bits_off_target);
    }

    #[test]
    fn rate_control_buffer_clamped_to_floor() {
        // Test #9 — repeated over-budget eventually clamps the deficit
        // at -max_buffer_size.
        let mut rc = fresh_rc();
        let huge = rc.avg_frame_bandwidth * 100;
        for _ in 0..200 {
            rc.postencode_update(huge, FrameKind::Inter);
        }
        assert_eq!(
            rc.buffer_level, -rc.maximum_buffer_size,
            "expected lower clamp at -max_buffer_size"
        );
    }

    #[test]
    fn rate_control_buffer_clamped_to_max() {
        // Test #10 — repeated under-budget clamps at maximum_buffer_size.
        let mut rc = fresh_rc();
        // Spend 1 bit/frame → near-empty spend.
        for _ in 0..1000 {
            rc.postencode_update(1, FrameKind::Inter);
        }
        assert_eq!(
            rc.buffer_level, rc.maximum_buffer_size,
            "expected upper clamp at +maximum_buffer_size"
        );
    }

    #[test]
    fn rate_control_advance_frame_decrements_frames_to_key() {
        // Test #12 — advance_frame reduces frames_to_key by 1.
        let mut rc = fresh_rc();
        rc.frames_to_key = 10;
        rc.advance_frame();
        assert_eq!(rc.frames_to_key, 9);
        rc.advance_frame();
        assert_eq!(rc.frames_to_key, 8);
    }

    #[test]
    fn rate_control_postencode_kf_resets_frames_since_key() {
        let mut rc = fresh_rc();
        // Pretend we're 15 frames into a GOP.
        rc.frames_since_key = 15;
        let bits = rc.avg_frame_bandwidth;
        rc.postencode_update(bits, FrameKind::Key);
        assert_eq!(rc.frames_since_key, 0);
        assert_eq!(rc.frames_to_key, rc.config().gop_length - 1);
    }

    #[test]
    fn golden_postencode_preserves_normal_inter_q_history() {
        let mut rc = fresh_rc();
        rc.last_q_inter = 121;
        rc.avg_frame_qindex_inter = 117;
        rc.last_picked_qindex = 64;

        rc.postencode_update(rc.avg_frame_bandwidth, FrameKind::Golden);

        assert_eq!(rc.last_q_inter, 121);
        assert_eq!(rc.avg_frame_qindex_inter, 117);
    }

    #[test]
    fn inter_q_history_uses_libvpx_rounding() {
        let mut rc = fresh_rc();
        rc.avg_frame_qindex_inter = 100;
        rc.last_picked_qindex = 103;

        rc.postencode_update(rc.avg_frame_bandwidth, FrameKind::Inter);

        assert_eq!(rc.last_q_inter, 103);
        assert_eq!(rc.avg_frame_qindex_inter, 101);
    }

    #[test]
    fn one_pass_vbr_feedback_shares_the_inter_normal_damping_gate() {
        for kind in [FrameKind::Key, FrameKind::Inter, FrameKind::Golden] {
            let cfg = RateControlConfig {
                rc_mode: RcMode::Vbr,
                ..RateControlConfig::default()
            };
            let mut rc = RateControl::new(cfg);
            let _ = rc.pick_q_and_bounds(kind);
            rc.postencode_update(rc.avg_frame_bandwidth, kind);

            assert!(
                rc.damped_adjustment[RateFactorLevel::InterNormal.as_index()],
                "{kind:?} must arm libvpx's shared one-pass VBR damping gate"
            );
            for level in [RateFactorLevel::GfArfStd, RateFactorLevel::KfStd] {
                assert!(
                    !rc.damped_adjustment[level.as_index()],
                    "{kind:?} must not arm its correction-factor slot as a damping gate"
                );
            }
        }
    }

    #[test]
    fn cbr_feedback_keeps_per_frame_kind_damping_slots() {
        let mut rc = RateControl::new(RateControlConfig::default());
        let _ = rc.pick_q_and_bounds(FrameKind::Key);
        rc.postencode_update(rc.avg_frame_bandwidth, FrameKind::Key);

        assert!(rc.damped_adjustment[RateFactorLevel::KfStd.as_index()]);
        assert!(!rc.damped_adjustment[RateFactorLevel::InterNormal.as_index()]);
    }

    #[test]
    fn cbr_resonance_history_shifts_and_classifies_corrections() {
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.q_1_frame = 11;
        rc.q_2_frame = 7;
        rc.rc_1_frame = 1;
        rc.rc_2_frame = -1;
        rc.last_picked_qindex = 33;

        update_cbr_resonance_history(&mut rc, 111);
        assert_eq!((rc.q_2_frame, rc.q_1_frame), (11, 33));
        assert_eq!((rc.rc_2_frame, rc.rc_1_frame), (1, -1));

        rc.last_picked_qindex = 44;
        update_cbr_resonance_history(&mut rc, 89);
        assert_eq!((rc.q_2_frame, rc.q_1_frame), (33, 44));
        assert_eq!((rc.rc_2_frame, rc.rc_1_frame), (-1, 1));

        rc.last_picked_qindex = 55;
        update_cbr_resonance_history(&mut rc, 100);
        assert_eq!((rc.q_2_frame, rc.q_1_frame), (44, 55));
        assert_eq!((rc.rc_2_frame, rc.rc_1_frame), (1, 0));
    }

    #[test]
    fn cbr_resonance_history_resets_massive_overshoot_alternation() {
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.q_1_frame = 80;
        rc.q_2_frame = 40;
        rc.rc_1_frame = 1;
        rc.rc_2_frame = 0;
        rc.last_picked_qindex = 100;

        update_cbr_resonance_history(&mut rc, 1001);

        assert_eq!(rc.rc_1_frame, -1);
        assert_eq!(rc.rc_2_frame, 0);
    }

    #[test]
    fn cbr_overhead_projection_advances_neutral_resonance_history() {
        let config = RateControlConfig {
            frame_width: 16,
            frame_height: 16,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(config);
        rc.q_1_frame = 41;
        rc.q_2_frame = 17;
        rc.rc_1_frame = -1;
        rc.rc_2_frame = 1;
        rc.last_picked_qindex = 255;

        update_rate_correction_factors(&mut rc, u32::MAX, FrameKind::Inter, None);

        assert_eq!((rc.q_2_frame, rc.q_1_frame), (41, 255));
        assert_eq!((rc.rc_2_frame, rc.rc_1_frame), (-1, 0));
    }

    // Closed-loop convergence oracle (RC.1).
    //
    // Drives the rate controller through a synthetic encoder whose
    // response curve is `bits_per_mb(picked_q, correction_factor=1.0)`
    // scaled by a `content_factor`. At a fixed correction factor of
    // 1.0 the rvp9 controller's prediction is 1/content_factor of
    // reality and the actual bitrate saturates accordingly (the
    // documented 5-40x divergence-ledger failure mode). A faithful
    // port of `vp9_rc_update_rate_correction_factors` must walk that
    // factor to match `content_factor` over time. The oracle gates
    // the average bits/frame at ±20 % of target across three orders
    // of magnitude.

    /// "Truth" content response curve used by
    /// [`simulate_closed_loop`].
    ///
    /// Models a content scene that costs `content_factor` times what
    /// the controller's untuned `bits_per_mb` formula predicts at
    /// `correction_factor = 1.0`. The rate-correction-factor feedback
    /// loop's whole job is to walk the per-level factor to this same
    /// `content_factor` over time, so the controller's predictions
    /// match reality. If the port is correct, the loop converges
    /// regardless of `content_factor` (within `[MIN_BPB_FACTOR,
    /// MAX_BPB_FACTOR]`).
    fn synth_bits_at_q(kind: FrameKind, q: i32, mbs: u32, content_factor: f64) -> u32 {
        use super::super::cbr::bits_per_mb;
        // Use 1.0 inside `bits_per_mb` and apply `content_factor`
        // ourselves so the synthesised value doesn't get clamped by
        // the [MIN_BPB_FACTOR, MAX_BPB_FACTOR] gate (content_factor
        // is allowed to be slightly outside that — that's the test).
        let bpm = f64::from(bits_per_mb(kind, q, 1.0)) * content_factor;
        let bits = (bpm as i64).saturating_mul(i64::from(mbs)) >> 9;
        bits.clamp(0, i64::from(u32::MAX)) as u32
    }

    fn simulate_closed_loop(
        target_bps: u32,
        frame_w: u32,
        frame_h: u32,
        frames: u32,
        content_factor: f64,
    ) -> (u64, u64) {
        use super::super::config::RateControlConfig;

        let cfg = RateControlConfig {
            target_bitrate_bps: target_bps,
            frame_width: frame_w,
            frame_height: frame_h,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        let mbs = rc.config().mbs();

        // Warm-up: 1 keyframe + half the requested run as a settling
        // period. We do NOT count any warm-up bits toward the budget
        // gate — libvpx-equivalent harnesses report mean inter
        // bitrate after the rate-correction factor has converged and
        // the buffer has reached its steady-state level. The
        // controller still observes the warm-up so the per-level
        // factor and the ambient Q EWMA both walk to their
        // equilibrium points.
        let warmup_frames = frames / 2;
        let (q, _, _) = rc.pick_q_and_bounds(FrameKind::Key);
        let actual = synth_bits_at_q(FrameKind::Key, i32::from(q), mbs, content_factor);
        rc.postencode_update(actual, FrameKind::Key);
        rc.advance_frame();

        // This oracle exercises the rate-correction-factor feedback
        // loop (the predict→actual→adjust cycle), NOT buffer-drain
        // dynamics. With the libvpx-faithful first-frame keyframe
        // target (`starting_buffer_level / 2`), the synthetic KF —
        // scaled by `content_factor` and with its KF_STD correction
        // factor not yet converged after a single sample — drains the
        // leaky-bucket buffer well below optimal. v1 has no frame-drop
        // path (`vp9_rc_drop_frame`, documented divergence) to refill
        // it, so the inter `calc_pframe_target_size` under-full-buffer
        // adjustment would otherwise shave inter targets indefinitely
        // and mask the loop's convergence. Reset the buffer to its
        // starting level so the measured inter steady-state reflects
        // the correction-factor loop, the property under test.
        rc.buffer_level = rc.starting_buffer_level;
        rc.bits_off_target = rc.starting_buffer_level;

        for _ in 0..warmup_frames {
            let (q, _, _) = rc.pick_q_and_bounds(FrameKind::Inter);
            let actual = synth_bits_at_q(FrameKind::Inter, i32::from(q), mbs, content_factor);
            rc.postencode_update(actual, FrameKind::Inter);
            rc.advance_frame();
        }

        let mut total_bits: u64 = 0;
        for _ in 0..frames {
            let (q, _, _) = rc.pick_q_and_bounds(FrameKind::Inter);
            let actual = synth_bits_at_q(FrameKind::Inter, i32::from(q), mbs, content_factor);
            rc.postencode_update(actual, FrameKind::Inter);
            rc.advance_frame();
            total_bits += u64::from(actual);
        }

        let avg_actual = total_bits / u64::from(frames);
        let avg_target = u64::from(rc.avg_frame_bandwidth);
        (avg_actual, avg_target)
    }

    // Convergence oracle resolutions are matched to the operating
    // bitrate so the test exercises the rate-correction-factor
    // feedback loop, not the controller's worst-Q saturation floor.
    // At 720p with `bits_per_mb(q=255, factor=1.0)` the encoder cannot
    // produce fewer than ~30,800 bits/frame (= 924 kbps @ 30 fps), so
    // sub-megabit 720p targets are unachievable in our model even
    // with a perfect controller — that's libvpx's documented
    // "needs frame-drop" regime (`vp9_rc_drop_frame`), which we
    // explicitly don't port. The chosen resolutions match WebRTC's
    // standard bitrate ladders (240p @ 250 kbps, 360p @ 1 Mbps, 720p
    // @ 4 Mbps), keeping every test inside the controller's
    // discrimination range.
    //
    // The content_factor = 3.0 below models the documented
    // observation that rvp9 actual bitrate runs 5-40x below target
    // before the loop is closed (divergence ledger entry "rvp9
    // rate-controller bitrate saturation"). At content_factor = 1.0
    // the controller's predictions match the content model exactly
    // and the feedback loop has nothing to compensate for; 3.0 forces
    // the loop to walk the per-level factor away from its initial
    // 1.0 to a steady-state ≈ 3.0.

    #[test]
    fn rate_control_converges_at_multiple_bitrates_250kbps() {
        // 250 kbps / 30 fps / 320x240 — low end of the WebRTC ladder.
        let (avg_actual, avg_target) = simulate_closed_loop(250_000, 320, 240, 120, 3.0);
        let diff = avg_actual as i64 - avg_target as i64;
        let pct = diff.unsigned_abs() * 100 / avg_target;
        assert!(
            pct <= 20,
            "250 kbps @ 240p: avg_actual={avg_actual}, avg_target={avg_target}, \
             diff={diff} ({pct}% — expected ≤ 20%)"
        );
    }

    #[test]
    fn rate_control_converges_at_multiple_bitrates_1mbps() {
        // 1 Mbps / 30 fps / 640x360 — mid-bitrate ladder rung.
        let (avg_actual, avg_target) = simulate_closed_loop(1_000_000, 640, 360, 120, 3.0);
        let diff = avg_actual as i64 - avg_target as i64;
        let pct = diff.unsigned_abs() * 100 / avg_target;
        assert!(
            pct <= 20,
            "1 Mbps @ 360p: avg_actual={avg_actual}, avg_target={avg_target}, \
             diff={diff} ({pct}% — expected ≤ 20%)"
        );
    }

    #[test]
    fn rate_control_converges_at_multiple_bitrates_4mbps() {
        // 4 Mbps / 30 fps / 1280x720 — high-bitrate ladder rung.
        let (avg_actual, avg_target) = simulate_closed_loop(4_000_000, 1280, 720, 120, 3.0);
        let diff = avg_actual as i64 - avg_target as i64;
        let pct = diff.unsigned_abs() * 100 / avg_target;
        assert!(
            pct <= 20,
            "4 Mbps @ 720p: avg_actual={avg_actual}, avg_target={avg_target}, \
             diff={diff} ({pct}% — expected ≤ 20%)"
        );
    }

    #[test]
    #[ignore]
    #[allow(clippy::print_stdout)]
    fn rate_control_closed_loop_trace() {
        // Diagnostic — run with `cargo test -- --ignored
        // rate_control_closed_loop_trace --nocapture` to see the
        // convergence numbers for each of the three convergence-
        // oracle bitrates.
        for (label, target_bps, w, h) in [
            ("250 kbps @ 240p", 250_000_u32, 320_u32, 240_u32),
            ("1 Mbps @ 360p", 1_000_000, 640, 360),
            ("4 Mbps @ 720p", 4_000_000, 1280, 720),
        ] {
            let (avg_actual, avg_target) = simulate_closed_loop(target_bps, w, h, 120, 3.0);
            let diff = avg_actual as i64 - avg_target as i64;
            let pct = (diff.unsigned_abs() as f64) * 100.0 / avg_target as f64;
            let sign = if diff >= 0 { '+' } else { '-' };
            println!(
                "{label}: avg_actual={avg_actual} avg_target={avg_target} \
                 diff={sign}{} ({sign}{pct:.1}%)",
                diff.unsigned_abs()
            );
        }
    }

    #[test]
    fn rate_control_simulation_30_frames() {
        // Test #11 — the headline oracle. Simulate a 30-frame
        // sequence (1 keyframe + 29 inter frames) and verify the
        // average *inter*-frame bits/frame tracks `avg_frame_bandwidth`
        // within 10%.
        //
        // We mimic an encoder that obeys the picked Q: at each Q the
        // encoder hits its target (±5% jitter), and we feed that back
        // in. That makes the simulation a closed loop between the
        // pick_q step and the postencode_update step.
        //
        // The keyframe is EXCLUDED from the convergence average. Per
        // the libvpx-faithful `vp9_calc_iframe_target_size_one_pass_cbr`
        // first-frame rule, the keyframe target is
        // `starting_buffer_level / 2` — many multiples of the inter
        // target. Over a full GOP that one-time intra cost amortizes,
        // but it is NOT part of the inter steady-state the rate-
        // correction-factor loop tracks, so a 30-frame KF+inter average
        // is dominated by the KF and is not a meaningful convergence
        // signal. (Before the E6 keyframe-boost fix, the KF target was
        // erroneously clamped to ~4x the inter target, which is why the
        // old KF-inclusive average happened to land near `avg`.)
        let mut rc = RateControl::new(RateControlConfig::default());

        let mut inter_bits: u64 = 0;
        let mut inter_frames: u64 = 0;
        let total_frames = 30u32;

        for i in 0..total_frames {
            let kind = if i == 0 {
                FrameKind::Key
            } else {
                FrameKind::Inter
            };
            let (q, _lo, _hi) = rc.pick_q_and_bounds(kind);

            // Encoder hits its target with a small ±5% perturbation.
            let target = rc.this_frame_target;
            let jitter_pct = if i % 2 == 0 { 95 } else { 105 };
            let actual_bits = (u64::from(target) * jitter_pct / 100) as u32;

            rc.postencode_update(actual_bits, kind);
            rc.advance_frame();

            // After the (large, libvpx-faithful) keyframe, restore the
            // buffer to its starting level. The KF spends
            // `starting_buffer_level / 2` and v1 has no frame-drop path
            // (documented divergence) to refill the leaky bucket, so the
            // under-full-buffer adjustment in `calc_pframe_target_size`
            // would otherwise shave every inter target below `avg` and
            // mask the property under test: that the controller targets
            // `avg_frame_bandwidth` for inter frames given a healthy
            // buffer.
            if kind == FrameKind::Key {
                rc.buffer_level = rc.starting_buffer_level;
                rc.bits_off_target = rc.starting_buffer_level;
            }

            if kind == FrameKind::Inter {
                inter_bits += u64::from(actual_bits);
                inter_frames += 1;
            }

            let _ = q; // Q drives the encoder; we only need the bit count.
        }

        let avg_actual = inter_bits / inter_frames;
        let avg_target = u64::from(rc.avg_frame_bandwidth);
        let diff_pct = if avg_actual >= avg_target {
            (avg_actual - avg_target) * 100 / avg_target
        } else {
            (avg_target - avg_actual) * 100 / avg_target
        };
        assert!(
            diff_pct <= 10,
            "29-inter-frame avg bits/frame ({avg_actual}) deviated {diff_pct}% from \
             target ({avg_target}); should be within 10%"
        );
    }
}
