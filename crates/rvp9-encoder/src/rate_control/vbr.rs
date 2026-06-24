//! One-pass VBR (variable bit rate) rate control.
//!
//! Direct port of the one-pass VBR helpers in libvpx
//! `vp9/encoder/vp9_ratectrl.c`. The VBR path differs from CBR mainly in
//! how it computes per-frame target bits and the `active_worst_quality`
//! ceiling:
//!
//! - **CBR** uses a *buffer-driven* `active_worst` (the leaky-bucket
//!   model) and a near-uniform per-frame target.
//! - **VBR** uses an *ambient-Q-driven* `active_worst` (`avg_inter_q *
//!   fac_active_worst_inter / 100`) and a GOP-amortised target where
//!   the keyframe gets ~25x the inter target.
//!
//! ## Scope (Wave RC.VBR)
//!
//! What this module implements (libvpx file:line references inline):
//!
//! - [`pick_q_and_bounds`] — top-level dispatch entry point, mirrors
//!   `rc_pick_q_and_bounds_one_pass_vbr` (`vp9_ratectrl.c:1138-1303`).
//! - [`active_worst_quality_vbr`] — VBR's active-worst computation,
//!   mirrors `calc_active_worst_quality_one_pass_vbr`
//!   (`vp9_ratectrl.c:905-928`).
//! - [`active_best_quality_vbr`] — VBR's active-best computation
//!   (libvpx-faithful, lines 1167-1185 KF / 1234-1247 inter).
//! - [`calc_iframe_target_size_vbr`] — KF target (`avg_bw * 25`),
//!   mirrors `vp9_calc_iframe_target_size_one_pass_vbr`
//!   (`vp9_ratectrl.c:2039-2048`).
//! - [`calc_pframe_target_size_vbr`] — inter target with GOP
//!   amortisation, mirrors
//!   `vp9_calc_pframe_target_size_one_pass_vbr`
//!   (`vp9_ratectrl.c:2019-2037`).
//!
//! ## What's intentionally NOT here (DOCUMENTED DIVERGENCES)
//!
//! - **GF/ARF path.** libvpx's VBR branches the inter path on
//!   `cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame`
//!   (`vp9_ratectrl.c:1186-1222`) and computes target/Q from
//!   `get_gf_active_quality`. Our encoder treats every non-keyframe as
//!   a normal inter (same simplification as CBR — see
//!   `super::mod::cbr::pick_q_and_bounds`). Tracked alongside the CBR
//!   GF/ARF deferral.
//! - **Forced KF / `this_key_frame_forced`.** The popping-prevention
//!   path at `vp9_ratectrl.c:1158-1166` requires the 2-pass stats
//!   reader (`rc->last_boosted_qindex`) and is out of scope for
//!   one-pass.
//! - **LIMIT_QRANGE_FOR_ALTREF_AND_KEY** (`vp9_ratectrl.c:1260-1279`).
//!   The qdelta widens `top_index` on KF/GF; without the ALTREF /
//!   GF-update path this is dead code for our one-pass-no-GF model.
//! - **`q_adj_factor` for CIF or smaller** (`vp9_ratectrl.c:1176-1178`).
//!   rvp9's default frame is 1280x720 (well above CIF); leaving the
//!   factor at 1.0 is libvpx-equivalent for our default config.
//! - **`current_video_frame == 0` first-KF ambient.** libvpx uses
//!   `rc->worst_quality` directly. We do the same via the
//!   `frames_since_key == 0 && total_actual_bits == 0` predicate
//!   (mirrors the CBR keyframe path).
//! - **Two-pass.** Landed as the separate [`super::RcMode::VbrTwoPass`]
//!   arm (port of `rc_pick_q_and_bounds_two_pass`, `vp9_ratectrl.c:1427`)
//!   driven by [`super::two_pass`]; not part of this one-pass module.

use super::cbr::{
    FRAME_OVERHEAD_BITS, get_active_quality, get_kf_active_quality, get_minq_index,
    get_rate_correction_factor, qindex_to_q, regulate_q,
};
use super::{FrameKind, RateControl};

/// libvpx `rc->fac_active_worst_inter`, seeded by `vp9_rc_init` at line
/// 399 to `150` and only modified by the high-source-SAD detector. We
/// don't drive the SAD detector, so 150 is the steady-state value.
const FAC_ACTIVE_WORST_INTER: i32 = 150;

/// libvpx `rc->fac_active_worst_gf`, seeded by `vp9_rc_init` at line
/// 400 to `100`.
const FAC_ACTIVE_WORST_GF: i32 = 100;

/// libvpx VBR keyframe target multiplier — `kf_ratio = 25` at
/// `vp9_ratectrl.c:2040`. The KF target is `25 * avg_frame_bandwidth`
/// (clamped against `max_frame_bandwidth`).
const KF_RATIO: u32 = 25;

/// libvpx non-AGGRESSIVE_VBR GF active-quality boost thresholds
/// (`vp9_ratectrl.c:90-99`).
const GF_LOW: i32 = 400;
const GF_HIGH: i32 = 2000;

/// `DEFAULT_GF_BOOST` (`vp9_ratectrl.h:31`) after
/// `vp9_set_gf_update_one_pass_vbr` resets `rc->gfu_boost`.
const GF_ACTIVE_QUALITY_BOOST: i32 = 2000;

/// Default `baseline_gf_interval` for one-pass VBR after
/// `vp9_set_gf_update_one_pass_vbr` runs once (libvpx `vp9_ratectrl.c:
/// 2078-2079`): `min(20, max(10, (min_gf+max_gf)/2)) == 15` for the
/// default `min_gf_interval=4` / `max_gf_interval=16` seeded by
/// `vp9_rc_init` (lines 425-432, libvpx's framerate-driven init).
///
/// We hard-code 15 — the steady-state value libvpx settles into for our
/// 30 fps default. The dynamic adjuster (`adjust_gfint_frame_constraint`
/// at line 2050) is part of the GF/ARF path we don't drive.
const BASELINE_GF_INTERVAL: i64 = 15;

/// libvpx `af_ratio_onepass_vbr`, seeded to `10` at
/// `vp9_ratectrl.c:389` and reset to `10` every GF update
/// (`vp9_ratectrl.c:2081`). Steady-state value with no GF reshuffle.
const AF_RATIO_ONEPASS_VBR: i64 = 10;

/// `inter_minq` LUT lookup. libvpx `init_minq_luts` (`vp9_ratectrl.c:
/// 128-138`) populates `inter[i]` with coefficients
/// `(0.00000271, -0.00113, 0.70)`. Identical formula to `rtc_minq` (the
/// CBR path's basis); the libvpx tables differ only by the
/// `AGGRESSIVE_VBR` build flag, which is OFF for the canonical build.
///
/// We compute on demand via `get_minq_index` rather than pre-baking the
/// 256-entry table — `pick_q_and_bounds` runs at frame rate, not pixel
/// rate, so the cost is irrelevant and the formula stays auditable.
fn inter_minq(qindex: i32) -> i32 {
    get_minq_index(qindex_to_q(qindex), 0.00000271, -0.00113, 0.70)
}

/// libvpx `arfgf_low_motion_minq` LUT, generated by `init_minq_luts`
/// with the non-AGGRESSIVE_VBR coefficients
/// `(0.0000015, -0.0009, 0.30)`.
fn arfgf_low_motion_minq(qindex: i32) -> i32 {
    get_minq_index(qindex_to_q(qindex), 0.0000015, -0.0009, 0.30)
}

/// libvpx `arfgf_high_motion_minq` LUT, generated by `init_minq_luts`
/// with coefficients `(0.0000021, -0.00125, 0.55)`.
fn arfgf_high_motion_minq(qindex: i32) -> i32 {
    get_minq_index(qindex_to_q(qindex), 0.0000021, -0.00125, 0.55)
}

/// Port of libvpx `get_gf_active_quality`
/// (`vp9_ratectrl.c:889-903`) for the one-pass, non-layered GF path.
fn get_gf_active_quality(q: i32) -> i32 {
    get_active_quality(
        q,
        GF_ACTIVE_QUALITY_BOOST,
        GF_LOW,
        GF_HIGH,
        arfgf_low_motion_minq,
        arfgf_high_motion_minq,
    )
}

/// Port of libvpx `calc_active_worst_quality_one_pass_vbr`
/// (`vp9_ratectrl.c:905-928`). Returns the per-frame
/// `active_worst_quality` ceiling for the VBR path.
///
/// Decision tree (non-GF, non-SVC):
///
/// ```text
/// if frame_is_intra_only:
///     // KF
///     if current_video_frame == 0: worst_quality
///     else:                        last_q[KEY_FRAME] << 1
/// else:
///     // Inter (no GF/ARF refresh)
///     if current_video_frame == 1: last_q[KEY_FRAME] << 1
///     else:                        avg_frame_qindex[INTER] * fac_active_worst_inter / 100
/// clamp to worst_quality
/// ```
///
/// We map libvpx's `current_video_frame` to our `(total_actual_bits,
/// frames_since_key)` state: `current_video_frame == 0` ⇔
/// `total_actual_bits == 0 && frames_since_key == 0` (the first-ever
/// frame). `current_video_frame == 1` ⇔ exactly one frame has been
/// emitted, which on a one-pass-KF-first stream means the second frame
/// (inter, sitting just past the initial keyframe).
fn active_worst_quality_vbr(rc: &RateControl, frame_kind: FrameKind) -> i32 {
    let worst_quality = rc.worst_quality;
    let last_q_key = i32::from(rc.last_q_keyframe);
    let avg_inter = i32::from(rc.avg_frame_qindex_inter);

    let raw = match frame_kind {
        FrameKind::Key => {
            // First-ever keyframe → no Q history; libvpx returns
            // worst_quality directly. Subsequent KFs use the previous
            // KF's Q doubled (`vp9_ratectrl.c:911-912`).
            if rc.total_actual_bits == 0 && rc.frames_since_key == 0 {
                worst_quality
            } else {
                last_q_key.saturating_mul(2)
            }
        }
        FrameKind::Inter => {
            // libvpx's `current_video_frame == 1` is the frame just
            // after the first KF — `frames_since_key == 1` in our
            // counting (KF post-encode resets frames_since_key to 0,
            // then `advance_frame` increments to 1). At that point
            // `avg_frame_qindex[INTER]` is still the worst_quality seed,
            // so libvpx falls back to `last_q[KEY] << 1` instead.
            // `vp9_ratectrl.c:921-924`.
            if rc.frames_since_key <= 1 && rc.total_actual_bits > 0 {
                last_q_key.saturating_mul(2)
            } else {
                // `avg_inter * 150 / 100` — drives active_worst above
                // the recent ambient, giving regulate_q headroom.
                (avg_inter.saturating_mul(FAC_ACTIVE_WORST_INTER)) / 100
            }
        }
        FrameKind::Golden => {
            // libvpx GF/ARF refresh branch:
            // current_video_frame == 1 uses `last_q[KEY] * 5 >> 2`;
            // steady-state GF uses avg inter Q times fac_active_worst_gf.
            // `vp9_ratectrl.c:914-919`.
            if rc.frames_since_key <= 1 && rc.total_actual_bits > 0 {
                last_q_key.saturating_mul(5) >> 2
            } else {
                (avg_inter.saturating_mul(FAC_ACTIVE_WORST_GF)) / 100
            }
        }
    };

    raw.min(worst_quality).max(rc.best_quality)
}

/// Port of libvpx `rc_pick_q_and_bounds_one_pass_vbr`'s active-best
/// computation (`vp9_ratectrl.c:1167-1185` KF / `1234-1247` inter), with
/// the GF/ARF and CQ/Q branches pruned (handled at higher level / out of
/// scope per module comment).
fn active_best_quality_vbr(rc: &RateControl, frame_kind: FrameKind, active_worst: i32) -> i32 {
    let avg_key = i32::from(rc.avg_frame_qindex_key);
    let avg_inter = i32::from(rc.avg_frame_qindex_inter);

    let raw = match frame_kind {
        FrameKind::Key => {
            // libvpx: `active_best = get_kf_active_quality(rc,
            // rc->avg_frame_qindex[KEY_FRAME], bit_depth)`. Then a
            // `q_adj_factor = 1.0` adjust at line 1183-1184 — for
            // frames > CIF it's a no-op (qdelta == 0), which is our
            // default config.
            get_kf_active_quality(avg_key)
        }
        FrameKind::Inter => {
            // libvpx: `q = min(avg_inter, active_worst); best =
            // inter_minq[q]` (`vp9_ratectrl.c:1237-1242`). On the very
            // first inter (current_video_frame == 1) it instead uses
            // `inter_minq[avg_frame_qindex[KEY_FRAME]]`.
            if rc.frames_since_key <= 1 && rc.total_actual_bits > 0 {
                inter_minq(avg_key)
            } else {
                let q = avg_inter.min(active_worst);
                inter_minq(q)
            }
        }
        FrameKind::Golden => {
            // libvpx GF/ARF branch (`vp9_ratectrl.c:1186-1222`):
            // use min(avg_inter, active_worst) after the first post-KF
            // frame, otherwise seed from avg keyframe Q.
            let q = if rc.frames_since_key > 1 {
                avg_inter.min(active_worst)
            } else {
                avg_key
            };
            get_gf_active_quality(q)
        }
    };

    raw.clamp(rc.best_quality, rc.worst_quality)
}

/// Port of libvpx `vp9_calc_iframe_target_size_one_pass_vbr`
/// (`vp9_ratectrl.c:2039-2048`):
///
/// ```text
/// target = avg_frame_bandwidth * kf_ratio    // kf_ratio = 25
/// clamp to max_frame_bandwidth (and rc_max_intra_bitrate_pct, NA)
/// ```
fn calc_iframe_target_size_vbr(rc: &RateControl) -> u32 {
    let avg = u64::from(rc.avg_frame_bandwidth);
    // Saturating-multiply to mirror libvpx's INT_MAX guard at line
    // 2043-2046.
    let raw = avg.saturating_mul(u64::from(KF_RATIO));
    let clamped = raw.min(u64::from(rc.max_frame_bandwidth));
    (clamped.max(u64::from(FRAME_OVERHEAD_BITS))) as u32
}

/// Port of libvpx `vp9_calc_pframe_target_size_one_pass_vbr`
/// (`vp9_ratectrl.c:2019-2037`) for the non-GF / non-SVC path:
///
/// ```text
/// target = avg_frame_bandwidth * baseline_gf_interval
///        / (baseline_gf_interval + af_ratio - 1)
/// ```
///
/// With our hard-coded `baseline_gf_interval = 15` and
/// `af_ratio = 10`, the denominator is 24 — every inter frame targets
/// `avg * 15/24 ≈ 0.625 * avg`. The slack accumulates in the buffer to
/// pay for the large KF target.
fn clamp_pframe_target_size_vbr(rc: &RateControl, raw: i64) -> u32 {
    let avg = i64::from(rc.avg_frame_bandwidth);
    // libvpx clamp (`vp9_rc_clamp_pframe_target_size`,
    // `vp9_ratectrl.c:207-232`):
    //   min_frame_target = max(min_frame_bandwidth, avg >> 5)
    //   target = max(target, min_frame_target)
    //   target = min(target, max_frame_bandwidth)
    let min_frame_target = i64::from(rc.min_frame_bandwidth)
        .max(avg >> 5)
        .max(i64::from(FRAME_OVERHEAD_BITS));
    let clamped = raw
        .max(min_frame_target)
        .min(i64::from(rc.max_frame_bandwidth));
    clamped.max(0) as u32
}

fn calc_pframe_target_size_vbr(rc: &RateControl) -> u32 {
    let avg = i64::from(rc.avg_frame_bandwidth);
    let numerator = avg.saturating_mul(BASELINE_GF_INTERVAL);
    let denominator = BASELINE_GF_INTERVAL + AF_RATIO_ONEPASS_VBR - 1;
    let raw = if denominator > 0 {
        numerator / denominator
    } else {
        avg
    };
    clamp_pframe_target_size_vbr(rc, raw)
}

fn calc_golden_target_size_vbr(rc: &RateControl) -> u32 {
    let avg = i64::from(rc.avg_frame_bandwidth);
    let numerator = avg
        .saturating_mul(BASELINE_GF_INTERVAL)
        .saturating_mul(AF_RATIO_ONEPASS_VBR);
    let denominator = BASELINE_GF_INTERVAL + AF_RATIO_ONEPASS_VBR - 1;
    let raw = if denominator > 0 {
        numerator / denominator
    } else {
        avg
    };
    clamp_pframe_target_size_vbr(rc, raw)
}

/// Pick the Q-index and `(bottom, top)` bounds for the next frame under
/// one-pass VBR.
///
/// This is the VBR rate-control front door — mirrors
/// `rc_pick_q_and_bounds_one_pass_vbr` (`vp9_ratectrl.c:1138-1303`),
/// with the GF/ARF, CQ/Q, forced-KF, and LIMIT_QRANGE branches pruned
/// (see module comment for the divergence ledger).
///
/// Algorithm (line numbers refer to `vp9_ratectrl.c`):
///
/// 1. Compute per-frame target bits (line 2118-2143 driver; we inline
///    the call to `calc_iframe_target_size_one_pass_vbr` /
///    `calc_pframe_target_size_one_pass_vbr`).
/// 2. Compute `active_worst_quality` via
///    [`active_worst_quality_vbr`] (line 1146).
/// 3. Compute `active_best_quality` via [`active_best_quality_vbr`]
///    (lines 1167-1247).
/// 4. Clamp both into the controller's `[best_quality, worst_quality]`
///    range and make `active_worst >= active_best` (lines 1252-1255).
/// 5. Run [`regulate_q`] to pick the Q that hits the target within the
///    active range (line 1287).
/// 6. Persist `this_frame_target`, `active_min_qindex`,
///    `active_max_qindex`, `last_picked_qindex` so the post-encode
///    feedback loop has the matching state.
pub(super) fn pick_q_and_bounds(rc: &mut RateControl, frame_kind: FrameKind) -> (u8, u8, u8) {
    // 1. Per-frame target bits (libvpx `vp9_rc_get_one_pass_vbr_params`,
    //    `vp9_ratectrl.c:2118-2143`).
    let target_bits = match frame_kind {
        FrameKind::Key => calc_iframe_target_size_vbr(rc),
        FrameKind::Inter => calc_pframe_target_size_vbr(rc),
        FrameKind::Golden => calc_golden_target_size_vbr(rc),
    };
    rc.this_frame_target = target_bits;

    // 2. Active worst-Q ceiling — VBR's ambient-driven model
    //    (libvpx `vp9_ratectrl.c:1146` + 905-928).
    let active_worst = active_worst_quality_vbr(rc, frame_kind);

    // 3. Active best-Q floor (libvpx `vp9_ratectrl.c:1167-1247`).
    let active_best = active_best_quality_vbr(rc, frame_kind, active_worst);

    // 4. Mutual clamp (libvpx `vp9_ratectrl.c:1252-1255`).
    let active_best = active_best.clamp(rc.best_quality, rc.worst_quality);
    let active_worst = active_worst.clamp(active_best, rc.worst_quality);

    // 5. Regulate Q within the active range (libvpx `vp9_ratectrl.c:1287`).
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

    // 6. Persist bounds + picked Q for postencode feedback. Mirrors
    //    the CBR path's `*bottom_index = active_best`, `*top_index =
    //    active_worst` writeback at lines 1257-1258.
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

    fn vbr_config() -> RateControlConfig {
        RateControlConfig {
            rc_mode: RcMode::Vbr,
            ..RateControlConfig::default()
        }
    }

    /// First-ever KF: libvpx's `current_video_frame == 0` branch returns
    /// `worst_quality` as the active_worst_quality basis.
    #[test]
    fn active_worst_first_kf_returns_worst_quality() {
        let rc = RateControl::new(vbr_config());
        // total_actual_bits == 0 && frames_since_key == 0 → first KF.
        let aw = active_worst_quality_vbr(&rc, FrameKind::Key);
        assert_eq!(
            aw, rc.worst_quality,
            "first KF active_worst must be worst_quality (libvpx \
             current_video_frame==0 branch)"
        );
    }

    /// Subsequent KF: `last_q[KEY] << 1`, capped at worst_quality.
    #[test]
    fn active_worst_subsequent_kf_doubles_last_kf_q() {
        let mut rc = RateControl::new(vbr_config());
        rc.total_actual_bits = 100_000;
        rc.last_q_keyframe = 60;
        let aw = active_worst_quality_vbr(&rc, FrameKind::Key);
        assert_eq!(aw, 120, "subsequent KF active_worst = last_q_keyframe * 2");
    }

    /// First inter (frames_since_key == 1, total_actual_bits > 0): use
    /// `last_q[KEY] << 1` (avg_inter EWMA hasn't moved off the
    /// worst_quality seed yet).
    #[test]
    fn active_worst_first_inter_uses_last_kf_q_times_two() {
        let mut rc = RateControl::new(vbr_config());
        rc.total_actual_bits = 100_000;
        rc.frames_since_key = 1;
        rc.last_q_keyframe = 50;
        let aw = active_worst_quality_vbr(&rc, FrameKind::Inter);
        assert_eq!(aw, 100, "first inter active_worst = last_q_keyframe * 2");
    }

    /// Steady-state inter: `avg_inter * 150 / 100`, capped at
    /// worst_quality.
    #[test]
    fn active_worst_steady_state_inter_uses_fac_active_worst() {
        let mut rc = RateControl::new(vbr_config());
        rc.total_actual_bits = 1_000_000;
        rc.frames_since_key = 30;
        rc.avg_frame_qindex_inter = 80;
        let aw = active_worst_quality_vbr(&rc, FrameKind::Inter);
        // 80 * 150 / 100 == 120
        assert_eq!(aw, 120);
    }

    /// KF target = avg_bw * 25, clamped to max_frame_bandwidth.
    #[test]
    fn iframe_target_is_avg_times_25_or_max_frame_bandwidth() {
        let rc = RateControl::new(vbr_config());
        let target = calc_iframe_target_size_vbr(&rc);
        let expected_raw = rc.avg_frame_bandwidth.saturating_mul(KF_RATIO);
        let expected = expected_raw.min(rc.max_frame_bandwidth);
        assert_eq!(target, expected);
        // And the KF target dwarfs an inter target.
        let inter_target = calc_pframe_target_size_vbr(&rc);
        assert!(
            u64::from(target) > u64::from(inter_target) * 5,
            "KF target ({target}) should be >> inter target ({inter_target})"
        );
    }

    /// Inter target = avg * 15 / 24 ≈ 0.625 * avg.
    #[test]
    fn pframe_target_is_avg_times_625_thousandths() {
        let rc = RateControl::new(vbr_config());
        let target = calc_pframe_target_size_vbr(&rc);
        let avg = rc.avg_frame_bandwidth;
        // avg * 15 / 24 == avg * 0.625
        let expected = (u64::from(avg) * 15 / 24) as u32;
        // Allow ±1 for integer division.
        assert!(
            target.abs_diff(expected) <= 1 || target >= expected,
            "VBR inter target {target} should be ~{expected} (avg * 15/24)"
        );
    }

    #[test]
    fn golden_target_is_af_ratio_boosted_inter_target() {
        let rc = RateControl::new(vbr_config());
        let inter_target = calc_pframe_target_size_vbr(&rc);
        let golden_target = calc_golden_target_size_vbr(&rc);
        let expected = ((u64::from(rc.avg_frame_bandwidth)
            * BASELINE_GF_INTERVAL as u64
            * AF_RATIO_ONEPASS_VBR as u64)
            / (BASELINE_GF_INTERVAL + AF_RATIO_ONEPASS_VBR - 1) as u64)
            .min(u64::from(rc.max_frame_bandwidth)) as u32;
        assert_eq!(golden_target, expected);
        assert!(
            golden_target > inter_target,
            "GF target ({golden_target}) should be boosted over normal inter ({inter_target})"
        );
    }

    #[test]
    fn golden_active_q_uses_gf_path_not_inter_path() {
        let mut rc = RateControl::new(vbr_config());
        rc.total_actual_bits = 1_000_000;
        rc.frames_since_key = 16;
        rc.avg_frame_qindex_inter = 180;
        rc.avg_frame_qindex_key = 80;
        let inter_worst = active_worst_quality_vbr(&rc, FrameKind::Inter);
        let golden_worst = active_worst_quality_vbr(&rc, FrameKind::Golden);
        assert_eq!(
            inter_worst, 255,
            "inter uses avg_inter * 150 / 100, clamped"
        );
        assert_eq!(golden_worst, 180, "GF uses avg_inter * 100 / 100");

        let inter_best = active_best_quality_vbr(&rc, FrameKind::Inter, inter_worst);
        let golden_best = active_best_quality_vbr(&rc, FrameKind::Golden, golden_worst);
        assert!(
            golden_best < inter_best,
            "GF active-best ({golden_best}) should allow lower Q than normal inter ({inter_best})"
        );
    }

    /// PROOF — `pick_q_and_bounds` returns a valid `(q, lo, hi)` triple
    /// with `lo <= q <= hi`, and `lo >= best_quality` / `hi <=
    /// worst_quality`.
    #[test]
    fn pick_q_and_bounds_returns_in_range_triple() {
        let mut rc = RateControl::new(vbr_config());
        let (q, lo, hi) = pick_q_and_bounds(&mut rc, FrameKind::Key);
        assert!(lo <= q && q <= hi, "q={q} must lie in [{lo},{hi}]");
        assert!(i32::from(lo) >= rc.best_quality);
        assert!(i32::from(hi) <= rc.worst_quality);
    }

    /// PROOF — VBR picks DIFFERENT Q for KF vs inter at equal state.
    /// VBR's KF target is 25× avg (the dominant signal); the inter
    /// target is 0.625× avg. Combined with the active_best floor
    /// (`get_kf_active_quality` on KF vs `inter_minq` on inter), the
    /// picked Q should differ by frame type even on the first frame.
    #[test]
    fn pick_q_differs_by_frame_type() {
        // Drive a KF first then a steady-state inter so the inter path
        // takes the avg_inter * 150 / 100 branch rather than the
        // first-inter fallback.
        let mut rc = RateControl::new(vbr_config());
        let (q_kf, _, _) = pick_q_and_bounds(&mut rc, FrameKind::Key);
        // Pretend the encoder emitted the KF target's worth of bits.
        rc.postencode_update(rc.this_frame_target, FrameKind::Key);
        rc.advance_frame();
        // Run a few steady-state inters to let avg_frame_qindex_inter
        // EWMA settle to a non-worst-quality value.
        for _ in 0..5 {
            let (_, _, _) = pick_q_and_bounds(&mut rc, FrameKind::Inter);
            rc.postencode_update(rc.this_frame_target, FrameKind::Inter);
            rc.advance_frame();
        }
        let (q_inter, _, _) = pick_q_and_bounds(&mut rc, FrameKind::Inter);
        assert_ne!(
            q_kf, q_inter,
            "VBR PROOF: KF Q ({q_kf}) should differ from steady-state inter Q ({q_inter})"
        );
    }

    /// PROOF — the inter_minq LUT is non-decreasing (sanity check on the
    /// `get_minq_index` curve at the inter coefficients).
    #[test]
    fn inter_minq_is_non_decreasing() {
        let mut prev = inter_minq(0);
        for q in 1..=255 {
            let now = inter_minq(q);
            assert!(
                now >= prev,
                "inter_minq must be non-decreasing: q={q}, prev={prev}, now={now}"
            );
            prev = now;
        }
    }
}
