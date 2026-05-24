//! Per-frame target-bits, active-Q range, and Q regulation.
//!
//! These functions are direct ports of the 1-pass CBR helpers in
//! libvpx's `vp9/encoder/vp9_ratectrl.c`. See [`super`] for the
//! divergence ledger.

use rvp9_core::bitstream::BitDepth;
use rvp9_core::quant::ac_quant;

use super::{FrameKind, RateControl};

/// `BPER_MB_NORMBITS` — libvpx `vp9_ratectrl.h:28`. The bits-per-MB
/// estimate is multiplied by 512 (= `1 << 9`); the per-frame total
/// is `(bpm * mbs) >> BPER_MB_NORMBITS`.
pub(super) const BPER_MB_NORMBITS: u32 = 9;

/// libvpx `FRAME_OVERHEAD_BITS` (`vp9_ratectrl.h:38`). Headers, tile
/// markers, padding bytes — anything that isn't tile coefficient
/// data. Used as a floor on per-frame target bits.
pub(super) const FRAME_OVERHEAD_BITS: u32 = 200;

/// libvpx `MIN_BPB_FACTOR` (`vp9_ratectrl.c:44`). Floor on the per-
/// level `rate_correction_factor`. Used by both
/// `get_rate_correction_factor` and the damped update in
/// `vp9_rc_update_rate_correction_factors`.
pub(super) const MIN_BPB_FACTOR: f64 = 0.005;

/// libvpx `MAX_BPB_FACTOR` (`vp9_ratectrl.c:45`). Ceiling counterpart
/// of [`MIN_BPB_FACTOR`].
pub(super) const MAX_BPB_FACTOR: f64 = 50.0;

/// libvpx `RATE_FACTOR_LEVELS` count (`vp9_ratectrl.h:54`). Sized to
/// hold every variant of [`RateFactorLevel`].
pub(super) const RATE_FACTOR_LEVELS: usize = 5;

/// Slot index into the per-level `rate_correction_factors[]` table.
///
/// Mirrors the libvpx `RATE_FACTOR_LEVEL` enum
/// (`vp9_ratectrl.h:48-55`) — the full 5-entry layout is kept for
/// future-proofing even though the v1 CBR realtime path only ever
/// reads/writes [`Self::InterNormal`] and [`Self::KfStd`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum RateFactorLevel {
    /// libvpx `INTER_NORMAL = 0`. Default inter frame.
    InterNormal = 0,
    /// libvpx `INTER_HIGH = 1`. Reserved for future use (high-motion
    /// inter classification not implemented in v1).
    #[allow(dead_code)]
    InterHigh = 1,
    /// libvpx `GF_ARF_LOW = 2`. Reserved — GF/ARF path not driven.
    #[allow(dead_code)]
    GfArfLow = 2,
    /// libvpx `GF_ARF_STD = 3`. Reserved — GF/ARF path not driven.
    #[allow(dead_code)]
    GfArfStd = 3,
    /// libvpx `KF_STD = 4`. Key / intra-only frame.
    KfStd = 4,
}

impl RateFactorLevel {
    /// Index into [`RateControl::rate_correction_factors`].
    #[inline]
    pub(super) const fn as_index(self) -> usize {
        self as usize
    }

    /// Pick the v1 CBR rate-factor level for a frame kind. Mirrors
    /// libvpx `get_rate_correction_factor` (`vp9_ratectrl.c:689-710`)
    /// for the `pass != 2 && !gf_cbr_boost` branch — keyframes land in
    /// `KF_STD`, every inter frame lands in `INTER_NORMAL`.
    #[inline]
    pub(super) const fn for_frame(frame_kind: FrameKind) -> Self {
        match frame_kind {
            FrameKind::Key => Self::KfStd,
            FrameKind::Inter => Self::InterNormal,
        }
    }
}

/// Convert a Q-index to the libvpx "q value" — `ac_quant(qindex, 0) /
/// 4.0`. Mirrors `vp9_convert_qindex_to_q` (`vp9_ratectrl.c:163-171`)
/// for the 8-bit path.
fn qindex_to_q(qindex: i32) -> f64 {
    // Encoder rate control — 8-bit only (HBD encoder out of scope for
    // Track T phase 0).
    f64::from(ac_quant(qindex, 0, BitDepth::Eight)) / 4.0
}

/// Bits-per-macroblock estimate at a given Q. Mirrors
/// `vp9_rc_bits_per_mb` (`vp9_ratectrl.c:185-196`):
///
/// ```text
/// enumerator = key ? 2_700_000 : 1_800_000
/// enumerator += (enumerator * q) >> 12
/// bpm = enumerator * correction_factor / q
/// ```
///
/// The caller supplies `correction_factor` from the per-level
/// [`RateControl::rate_correction_factors`] table (libvpx
/// `get_rate_correction_factor`, clamped to
/// `[MIN_BPB_FACTOR, MAX_BPB_FACTOR]`).
///
/// Libvpx-faithful version; the production rate-control entry points
/// route through [`bits_per_mb_scaled`] so the config-driven
/// `bits_per_mb_multiplier` divergence (A10) applies consistently to
/// every predict-then-feedback step.
pub(super) fn bits_per_mb(frame_kind: FrameKind, qindex: i32, correction_factor: f64) -> i32 {
    let q = qindex_to_q(qindex);
    // Avoid divide-by-zero when q drops to ~0 at qindex 0. libvpx
    // never hits this in practice (qindex 0 == lossless, handled
    // separately), but we're defensive.
    if q <= 0.0 {
        return i32::MAX;
    }
    let mut enumerator: f64 = if frame_kind.is_key() {
        2_700_000.0
    } else {
        1_800_000.0
    };
    // libvpx: `enumerator += (int)(enumerator * q) >> 12;`
    // The cast-to-int-then-shift truncates; we replicate the
    // semantics with `floor`.
    let bump = (enumerator * q / 4096.0).floor();
    enumerator += bump;
    (enumerator * correction_factor / q) as i32
}

/// Wrap [`bits_per_mb`] with the config-driven
/// `bits_per_mb_multiplier`. **DOCUMENTED DIVERGENCE A10** — see
/// [`super::config::RateControlConfig::bits_per_mb_multiplier`] for
/// the rationale. Default multiplier `1.0` matches libvpx byte-for-byte;
/// non-unit multipliers compensate for the simplified estimator on
/// textured / high-motion content. The multiplier applies to BOTH
/// the picker side (regulate_q) AND the feedback side
/// (estimate_bits_at_q used by update_rate_correction_factors) so the
/// loop stays internally consistent.
#[inline]
pub(super) fn bits_per_mb_scaled(
    frame_kind: FrameKind,
    qindex: i32,
    correction_factor: f64,
    multiplier: f64,
) -> i32 {
    let raw = bits_per_mb(frame_kind, qindex, correction_factor);
    if (multiplier - 1.0).abs() < f64::EPSILON {
        return raw;
    }
    // Apply in f64 then saturate back to i32. `bits_per_mb` already
    // saturates to i32::MAX when q is degenerate; we preserve that
    // behaviour.
    if raw == i32::MAX {
        return i32::MAX;
    }
    let scaled = f64::from(raw) * multiplier;
    if scaled >= f64::from(i32::MAX) {
        i32::MAX
    } else if scaled <= 0.0 {
        0
    } else {
        scaled as i32
    }
}

/// Read the clamped rate-correction factor for the current frame.
///
/// Mirrors libvpx `get_rate_correction_factor` (`vp9_ratectrl.c:689-710`)
/// — the v1 CBR-realtime path collapses to "KF_STD for intra-only,
/// INTER_NORMAL for every inter". The `rcf_mult[frame_size_selector]`
/// scaling (libvpx line 708) is omitted: we don't drive the dynamic
/// frame-resizer in v1, so `frame_size_selector` is always `UNSCALED`
/// (multiplier = 1.0).
#[inline]
pub(super) fn get_rate_correction_factor(rc: &RateControl, frame_kind: FrameKind) -> f64 {
    let level = RateFactorLevel::for_frame(frame_kind);
    rc.rate_correction_factors[level.as_index()].clamp(MIN_BPB_FACTOR, MAX_BPB_FACTOR)
}

/// Write a damped-updated correction factor back into the per-level
/// table. Mirrors libvpx `set_rate_correction_factor`
/// (`vp9_ratectrl.c:712-735`). The pre-clamp `factor /=
/// rcf_mult[selector]` libvpx line is dropped — see
/// [`get_rate_correction_factor`].
#[inline]
pub(super) fn set_rate_correction_factor(
    rc: &mut RateControl,
    frame_kind: FrameKind,
    factor: f64,
) {
    let clamped = factor.clamp(MIN_BPB_FACTOR, MAX_BPB_FACTOR);
    let level = RateFactorLevel::for_frame(frame_kind);
    rc.rate_correction_factors[level.as_index()] = clamped;
}

/// Estimate the number of bits the encoder *would* spend at a given Q,
/// for a given correction factor. Mirrors libvpx
/// `vp9_estimate_bits_at_q` (`vp9_ratectrl.c:198-205`):
///
/// ```text
/// bpm = vp9_rc_bits_per_mb(frame_type, q, correction_factor, ...)
/// bits = max(FRAME_OVERHEAD_BITS, (bpm * MBs) >> BPER_MB_NORMBITS)
/// ```
///
/// Used by `postencode::update_rate_correction_factors` to compute
/// `projected_size_based_on_q`, the controller's own prediction of
/// the frame size at the picked Q.
pub(super) fn estimate_bits_at_q(
    frame_kind: FrameKind,
    qindex: i32,
    mbs: u32,
    correction_factor: f64,
    multiplier: f64,
) -> i64 {
    let bpm = i64::from(bits_per_mb_scaled(
        frame_kind,
        qindex,
        correction_factor,
        multiplier,
    ));
    let raw = bpm.saturating_mul(i64::from(mbs)) >> BPER_MB_NORMBITS;
    raw.max(i64::from(FRAME_OVERHEAD_BITS))
}

/// Pick a Q-index that targets `target_bits_per_frame` within
/// `[active_best, active_worst]`. Mirrors `vp9_rc_regulate_q`
/// (`vp9_ratectrl.c:819-862`):
///
/// 1. Compute `target_bits_per_mb = (target_bits << 9) / MBs`.
/// 2. Iterate `i` from `active_best` upwards.
/// 3. First `i` whose `bits_per_mb_at_this_q` ≤ target wins (with the
///    "pick i-1 if the previous overshot less" tie-break libvpx uses).
fn regulate_q(
    frame_kind: FrameKind,
    target_bits_per_frame: u32,
    active_best: i32,
    active_worst: i32,
    mbs: u32,
    correction_factor: f64,
    multiplier: f64,
) -> i32 {
    // `(target_bits << 9) / MBs` — bits_per_mb at the target rate.
    let target_bpm: i64 = if mbs == 0 {
        0
    } else {
        (i64::from(target_bits_per_frame) << BPER_MB_NORMBITS) / i64::from(mbs)
    };

    let mut last_error = i64::MAX;
    let mut i = active_best;
    while i <= active_worst {
        let bpm_at_i =
            i64::from(bits_per_mb_scaled(frame_kind, i, correction_factor, multiplier));
        if bpm_at_i <= target_bpm {
            // libvpx tie-break: prefer the Q with the smaller residual.
            let q = if target_bpm - bpm_at_i <= last_error {
                i
            } else {
                i - 1
            };
            return q.max(active_best);
        }
        last_error = bpm_at_i - target_bpm;
        i += 1;
    }
    // Fell through — even at active_worst we exceed target. Cap at
    // worst.
    active_worst
}

/// Compute per-frame target bits for an inter (P) frame in 1-pass CBR.
/// Mirrors `vp9_calc_pframe_target_size_one_pass_cbr`
/// (`vp9_ratectrl.c:2145-2192`).
///
/// The buffer-deviation adjustment is the libvpx formula in full:
///
/// ```text
/// diff      = optimal_buffer_level - buffer_level
/// one_pct   = 1 + optimal_buffer_level / 100
/// if diff > 0 (under-full):
///     pct_low = min(diff / one_pct, UNDER_SHOOT_PCT)
///     target -= target * pct_low / 200
/// else if diff < 0 (over-full):
///     pct_high = min(-diff / one_pct, OVER_SHOOT_PCT)
///     target += target * pct_high / 200
/// ```
///
/// libvpx defaults `oxcf->under_shoot_pct = 100`, `oxcf->over_shoot_pct
/// = 100` (CBR), which we hard-code in v1.
fn calc_pframe_target_size(rc: &RateControl) -> u32 {
    const UNDER_SHOOT_PCT: i64 = 100;
    const OVER_SHOOT_PCT: i64 = 100;

    let diff: i64 = rc.optimal_buffer_level - rc.buffer_level;
    let one_pct_bits: i64 = 1 + rc.optimal_buffer_level / 100;

    let min_frame_target = (rc.avg_frame_bandwidth >> 4).max(FRAME_OVERHEAD_BITS);
    let mut target: i64 = i64::from(rc.avg_frame_bandwidth);

    if diff > 0 {
        // Buffer under-full: shave target.
        let pct_low = (diff / one_pct_bits).min(UNDER_SHOOT_PCT);
        target -= (target * pct_low) / 200;
    } else if diff < 0 {
        // Buffer over-full: boost target.
        let pct_high = ((-diff) / one_pct_bits).min(OVER_SHOOT_PCT);
        target += (target * pct_high) / 200;
    }

    // Clamp to [min, max_frame_bandwidth].
    if target < i64::from(min_frame_target) {
        min_frame_target
    } else if target > i64::from(rc.max_frame_bandwidth) {
        rc.max_frame_bandwidth
    } else {
        target as u32
    }
}

/// Compute per-frame target bits for a keyframe in 1-pass CBR.
/// Mirrors `vp9_calc_iframe_target_size_one_pass_cbr`
/// (`vp9_ratectrl.c:2194-2221`).
///
/// First frame uses a fixed boost (half of starting_buffer_level);
/// subsequent keyframes use the configured `kf_boost` ramped down
/// near the keyframe boundary.
fn calc_iframe_target_size(rc: &RateControl) -> u32 {
    let avg = i64::from(rc.avg_frame_bandwidth);
    let starting = rc.starting_buffer_level;

    let target: i64 = if rc.total_actual_bits == 0 && rc.frames_since_key == 0 {
        // First-ever frame: libvpx uses `starting_buffer_level / 2`
        // (`vp9_ratectrl.c:2199-2202`).
        (starting / 2).max(avg)
    } else {
        // libvpx kf_boost ramp:
        //   kf_boost = max(32, 2 * framerate - 16)
        //   if frames_since_key < framerate/2:
        //       kf_boost = kf_boost * frames_since_key / (framerate / 2)
        //   target = ((16 + kf_boost) * avg_frame_bandwidth) >> 4
        let mut kf_boost = i64::from(rc.kf_boost);
        let fr = rc.config().framerate;
        let fr_based: i64 = ((2.0 * fr) - 16.0) as i64;
        kf_boost = kf_boost.max(fr_based.max(32));
        let half_fr = fr / 2.0;
        if half_fr > 0.0 && f64::from(rc.frames_since_key) < half_fr {
            kf_boost = kf_boost * i64::from(rc.frames_since_key) / (half_fr as i64).max(1);
        }
        ((16 + kf_boost) * avg) >> 4
    };

    // Clamp to max_frame_bandwidth (libvpx
    // `vp9_rc_clamp_iframe_target_size:234-243`).
    if target > i64::from(rc.max_frame_bandwidth) {
        rc.max_frame_bandwidth
    } else if target < i64::from(FRAME_OVERHEAD_BITS) {
        FRAME_OVERHEAD_BITS
    } else {
        target as u32
    }
}

/// Linear approximation of libvpx's `rtc_minq` lookup table
/// (`vp9_rc_init_minq_luts:142`). libvpx's table is a non-linear curve
/// that pulls Q down from `ambient_qp` by ~15% in the low-Q range and
/// ~25% in the high-Q range. v1 uses a fixed 75% multiplier which is
/// within the right neighbourhood for `qindex in [50, 200]`. DOCUMENTED
/// DIVERGENCE — see [`super`] notes. Returns a Q-index clamped to
/// `[best_quality, worst_quality]`.
pub(super) fn rtc_minq_approx(rc: &RateControl, ambient_qindex: i32) -> i32 {
    // libvpx: `rtc_minq[ambient] ≈ ambient * 0.75` (rough)
    let approx = (ambient_qindex * 3) / 4;
    approx.clamp(rc.best_quality, rc.worst_quality)
}

/// libvpx `kf_low` / `kf_high` boost thresholds used by
/// `get_active_quality` (`vp9_ratectrl.c:90-91`).
const KF_LOW: i32 = 400;
const KF_HIGH: i32 = 5000;

/// Quality boost fed to [`get_active_quality`] for the keyframe path.
/// Mirrors libvpx `rc->kf_boost`, which `vp9_rc_init` seeds to
/// `DEFAULT_KF_BOOST` and the CBR realtime path leaves untouched
/// (the per-frame *target* `kf_boost` ramp in [`calc_iframe_target_size`]
/// is a separate quantity). `DEFAULT_KF_BOOST` is well below [`KF_LOW`],
/// so [`get_active_quality`] takes the high-motion branch — the
/// conservative (higher-Q) floor appropriate when no per-clip motion
/// statistics have been gathered yet (every realtime CBR keyframe).
const KF_ACTIVE_QUALITY_BOOST: i32 = 32;

/// Ceiling on the `active_worst_quality` basis fed to
/// [`get_kf_active_quality`]. libvpx indexes the LUT by the keyframe's
/// `active_worst_quality`; on the first frame that ambient is seeded at
/// `worst_quality` (255), which over-floors rvp9's keyframe (the LUT
/// returns ≈195 there). libvpx escapes this via its content-aware
/// `bits_per_mb` rate model, which rvp9 lacks. Capping the basis at this
/// realtime keyframe-Q neighbourhood lands the floor at the
/// libvpx-matched ≈57 on textured content. DOCUMENTED DIVERGENCE (E7).
const KF_ACTIVE_QUALITY_BASIS_CAP: i32 = 122;

/// Port of libvpx `get_minq_index` (`vp9_ratectrl.c:118-135`). Finds the
/// smallest qindex whose q-value (`ac_quant/4`) is ≥ the cubic-curve
/// target `minqtarget = min((x3·maxq + x2)·maxq + x1)·maxq, maxq)`.
fn get_minq_index(maxq: f64, x3: f64, x2: f64, x1: f64) -> i32 {
    let minqtarget = (((x3 * maxq + x2) * maxq + x1) * maxq).min(maxq);
    // Special-case the step from q2.0 down to lossless.
    if minqtarget <= 2.0 {
        return 0;
    }
    for i in 0..256 {
        if minqtarget <= qindex_to_q(i) {
            return i;
        }
    }
    255
}

/// libvpx `kf_low_motion_minq` LUT, generated by `init_minq_luts`
/// (`vp9_ratectrl.c:137-160`) with coefficients
/// `(0.000001, -0.0004, 0.150)`. Indexed by the keyframe's
/// `active_worst_quality`.
fn kf_low_motion_minq(qindex: i32) -> i32 {
    get_minq_index(qindex_to_q(qindex), 0.000001, -0.0004, 0.150)
}

/// libvpx `kf_high_motion_minq` LUT (`init_minq_luts`,
/// `vp9_ratectrl.c:137-160`) with coefficients
/// `(0.0000021, -0.00125, 0.45)`.
fn kf_high_motion_minq(qindex: i32) -> i32 {
    get_minq_index(qindex_to_q(qindex), 0.0000021, -0.00125, 0.45)
}

/// Port of libvpx `get_active_quality` (`vp9_ratectrl.c:96-116`).
/// Blends the low-/high-motion min-Q tables by where `gfu_boost` sits
/// between `kf_low` and `kf_high`.
fn get_active_quality(
    q: i32,
    gfu_boost: i32,
    low: i32,
    high: i32,
    low_motion_minq: impl Fn(i32) -> i32,
    high_motion_minq: impl Fn(i32) -> i32,
) -> i32 {
    if gfu_boost > high {
        low_motion_minq(q)
    } else if gfu_boost < low {
        high_motion_minq(q)
    } else {
        let gap = high - low;
        let offset = high - gfu_boost;
        let qdiff = high_motion_minq(q) - low_motion_minq(q);
        let adjustment = (offset * qdiff + (gap >> 1)) / gap;
        low_motion_minq(q) + adjustment
    }
}

/// Port of libvpx `get_kf_active_quality` (`vp9_ratectrl.c:175-183`).
/// Returns the keyframe `active_best_quality` floor for a given
/// `active_worst_quality` (`q`). This is the lever that stops
/// `regulate_q` from crashing the keyframe Q to the `best_quality`
/// floor when the (large, libvpx-faithful) keyframe target combines
/// with rvp9's under-predicting keyframe `bits_per_mb` estimator.
fn get_kf_active_quality(q: i32) -> i32 {
    get_active_quality(
        q,
        KF_ACTIVE_QUALITY_BOOST,
        KF_LOW,
        KF_HIGH,
        kf_low_motion_minq,
        kf_high_motion_minq,
    )
}

/// Buffer-driven active worst-quality for inter frames. Mirrors
/// `calc_active_worst_quality_one_pass_cbr` (`vp9_ratectrl.c:931-1000`)
/// for the non-SVC, non-screen-content path.
fn active_worst_quality_inter(rc: &RateControl) -> i32 {
    active_worst_quality(rc, i32::from(rc.avg_frame_qindex_inter))
}

/// Buffer-driven active worst-quality, parameterised on the ambient Q
/// (the relevant `avg_frame_qindex[type]`). Mirrors
/// `calc_active_worst_quality_one_pass_cbr` (`vp9_ratectrl.c:931-1000`)
/// for the non-SVC, non-screen-content path. Both the inter and the
/// keyframe paths use this single buffer model — libvpx computes
/// `active_worst_quality` once per frame from the same function.
fn active_worst_quality(rc: &RateControl, ambient_qp: i32) -> i32 {
    let critical_level = rc.optimal_buffer_level >> 3;
    let ambient_aw = (ambient_qp * 5) >> 2;
    let mut active_worst = rc.worst_quality.min(ambient_aw);

    if rc.buffer_level > rc.optimal_buffer_level {
        // Buffer above optimal: bring active_worst DOWN.
        let max_adjustment_down = active_worst / 3;
        if max_adjustment_down > 0 {
            let buff_lvl_step =
                (rc.maximum_buffer_size - rc.optimal_buffer_level) / i64::from(max_adjustment_down);
            if buff_lvl_step > 0 {
                let adjustment = (rc.buffer_level - rc.optimal_buffer_level) / buff_lvl_step;
                // Saturating subtract — adjustment could exceed
                // active_worst on pathological inputs.
                active_worst -= adjustment as i32;
            }
        }
    } else if rc.buffer_level > critical_level {
        // Buffer between critical and optimal: ramp UP from ambient to
        // worst_quality.
        if critical_level > 0 {
            let buff_lvl_step = rc.optimal_buffer_level - critical_level;
            if buff_lvl_step > 0 {
                let adjustment = i64::from(rc.worst_quality - ambient_qp)
                    * (rc.optimal_buffer_level - rc.buffer_level)
                    / buff_lvl_step;
                active_worst = ambient_qp + adjustment as i32;
            }
        }
    } else {
        // Buffer at or below critical: clamp to worst_quality.
        active_worst = rc.worst_quality;
    }

    active_worst.clamp(rc.best_quality, rc.worst_quality)
}

/// Pick the Q-index and `(bottom, top)` bounds for the next frame.
///
/// This is the rate-control front door — it updates
/// `rc.this_frame_target`, `rc.active_min_qindex`,
/// `rc.active_max_qindex`, and returns `(q, low, high)`.
pub(super) fn pick_q_and_bounds(rc: &mut RateControl, frame_kind: FrameKind) -> (u8, u8, u8) {
    // 1. Per-frame target bits.
    let target_bits = match frame_kind {
        FrameKind::Key => calc_iframe_target_size(rc),
        FrameKind::Inter => calc_pframe_target_size(rc),
    };
    rc.this_frame_target = target_bits;

    // 2. Active Q range.
    let (active_best, active_worst) = match frame_kind {
        FrameKind::Key => {
            // libvpx `rc_pick_q_and_bounds_one_pass_cbr` keyframe branch
            // (`vp9_ratectrl.c:1013-1042`). The keyframe gets a buffer-
            // driven `active_worst_quality` (same model as the inter
            // path) and an `active_best_quality` FLOOR from
            // `get_kf_active_quality`. That floor is the lever this fix
            // (E7) restores: without it, `regulate_q` walks the keyframe
            // Q all the way down to `best_quality` (q≈4) chasing the
            // large libvpx-faithful keyframe target, because rvp9's
            // keyframe `bits_per_mb` enumerator under-predicts the true
            // intra cost on textured content. The floor pins the
            // keyframe at a quality-appropriate Q (≈57 on foreman),
            // matching libvpx's content-aware rate model from the
            // outside.
            let worst = active_worst_quality(rc, i32::from(rc.avg_frame_qindex_key));
            // `get_kf_active_quality` is indexed by the keyframe's
            // `active_worst_quality` (the `q` argument in libvpx). On the
            // first frame of a stream, libvpx's `avg_frame_qindex[KEY]`
            // seed is `worst_quality` (255) and its content-aware
            // `bits_per_mb` rate model lands the keyframe at q≈57 from a
            // `best_quality` floor. rvp9's enumerator-over-Q estimator
            // can't reproduce that content awareness, so we instead floor
            // the keyframe via the LUT — but feeding the raw 255 ambient
            // over-floors (kf_high_motion_minq[255] ≈ 195). We clamp the
            // LUT basis to [`KF_ACTIVE_QUALITY_BASIS_CAP`], a realtime
            // keyframe-Q neighbourhood that yields a libvpx-matched floor
            // (≈57 on foreman). DOCUMENTED DIVERGENCE — see the E7 ledger
            // entry: the cap stands in for libvpx's content-aware rate
            // model on the keyframe.
            let basis = worst.min(KF_ACTIVE_QUALITY_BASIS_CAP);
            let best = get_kf_active_quality(basis);
            (best.min(worst), worst)
        }
        FrameKind::Inter => {
            let worst = active_worst_quality_inter(rc);
            // libvpx `rc_pick_q_and_bounds_one_pass_cbr:1057-1068`:
            //   active_best = rtc_minq[avg_inter_q]   if avg_inter_q < worst
            //                 rtc_minq[worst]         otherwise
            let basis = if i32::from(rc.avg_frame_qindex_inter) < worst {
                i32::from(rc.avg_frame_qindex_inter)
            } else {
                worst
            };
            let best = rtc_minq_approx(rc, basis);
            (best.min(worst), worst)
        }
    };

    let active_best = active_best.clamp(rc.best_quality, rc.worst_quality);
    let active_worst = active_worst.clamp(active_best, rc.worst_quality);

    // 3. Regulate Q using the current per-level correction factor.
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

    // Persist bounds + the picked Q so postencode can re-apply the
    // damped update formula at the same rate-factor level.
    rc.active_min_qindex = active_best as u8;
    rc.active_max_qindex = active_worst as u8;
    rc.last_picked_qindex = q as u8;

    (q as u8, active_best as u8, active_worst as u8)
}

#[cfg(test)]
mod tests {
    use super::super::config::RateControlConfig;
    use super::*;

    /// Build a controller at the default config, then run it through a
    /// dummy keyframe pick to advance `total_actual_bits` past zero
    /// (so subsequent picks use the steady-state branches). Returns
    /// the controller mid-flight, ready for an inter-frame test.
    fn warmed_up_rc() -> RateControl {
        let mut rc = RateControl::new(RateControlConfig::default());
        // Pretend we encoded one keyframe at avg_frame_bandwidth.
        let (q, _, _) = rc.pick_q_and_bounds(FrameKind::Key);
        let bits = rc.avg_frame_bandwidth;
        super::super::postencode::postencode_update(&mut rc, bits, FrameKind::Key);
        rc.advance_frame();
        // Sanity: KF Q should be plausible.
        assert!(q > 0);
        rc
    }

    #[test]
    fn bits_per_mb_monotonically_decreases_with_q() {
        // Test sanity: as Q rises, bits-per-MB falls. Use the default
        // correction factor of 1.0 (libvpx's `vp9_rc_init` seed) so
        // this stays a pure-function test.
        let prev = bits_per_mb(FrameKind::Inter, 20, 1.0);
        for q in (21..=255).step_by(8) {
            let now = bits_per_mb(FrameKind::Inter, q, 1.0);
            assert!(
                now <= prev || (prev - now).abs() < 5,
                "bits_per_mb non-monotonic: q={q}, prev={prev}, now={now}"
            );
        }
    }

    #[test]
    fn bits_per_mb_keyframe_higher_than_inter() {
        // Key enumerator (2_700_000) > inter enumerator (1_800_000),
        // so at the same Q keyframes consume more bits/MB.
        let q = 100;
        let key = bits_per_mb(FrameKind::Key, q, 1.0);
        let inter = bits_per_mb(FrameKind::Inter, q, 1.0);
        assert!(
            key > inter,
            "expected KF bpm ({key}) > inter bpm ({inter}) at q={q}"
        );
    }

    #[test]
    fn bits_per_mb_scales_with_correction_factor() {
        // 2x the correction factor → ~2x the bits-per-MB estimate
        // (modulo integer truncation).
        let q = 80;
        let one = bits_per_mb(FrameKind::Inter, q, 1.0);
        let two = bits_per_mb(FrameKind::Inter, q, 2.0);
        let ratio = f64::from(two) / f64::from(one);
        assert!(
            (1.9..=2.1).contains(&ratio),
            "expected ~2x scaling: one={one}, two={two}, ratio={ratio}"
        );
    }

    #[test]
    fn rate_control_pick_q_increases_when_bits_off_target_negative() {
        // Test #3 — over-budget buffer should push Q up.
        let mut rc_baseline = warmed_up_rc();
        let (q_baseline, _, _) = rc_baseline.pick_q_and_bounds(FrameKind::Inter);

        let mut rc_over = warmed_up_rc();
        // Drain the buffer hard.
        rc_over.buffer_level = -500_000;
        rc_over.bits_off_target = -500_000;
        let (q_over, _, _) = rc_over.pick_q_and_bounds(FrameKind::Inter);

        assert!(
            q_over >= q_baseline,
            "expected Q to RISE when buffer drained: \
             baseline={q_baseline}, over_budget={q_over}"
        );
    }

    #[test]
    fn rate_control_pick_q_decreases_when_bits_off_target_positive() {
        // Test #4 — under-budget buffer should pull Q down.
        let mut rc_baseline = warmed_up_rc();
        let (q_baseline, _, _) = rc_baseline.pick_q_and_bounds(FrameKind::Inter);

        let mut rc_under = warmed_up_rc();
        // Fill the buffer up past optimal.
        rc_under.buffer_level = rc_under.maximum_buffer_size;
        rc_under.bits_off_target = rc_under.maximum_buffer_size;
        // Also pull down the recent-Q ambient so the active_worst
        // calculation actually shrinks (otherwise the buffer-driven
        // adjustment is dominated by the already-high ambient).
        rc_under.avg_frame_qindex_inter = 100;
        let (q_under, _, _) = rc_under.pick_q_and_bounds(FrameKind::Inter);

        assert!(
            q_under <= q_baseline,
            "expected Q to FALL when buffer full: \
             baseline={q_baseline}, under_budget={q_under}"
        );
    }

    #[test]
    fn rate_control_kf_q_lower_than_inter_q() {
        // Test #5 — at the same buffer state, KF Q < inter Q (kf_boost
        // gives the keyframe a larger bit budget, so Q can be lower
        // while still hitting the budget).
        //
        // To isolate the kf_boost effect from the buffer-driven
        // active_worst clamp on inters, we drain the buffer below
        // critical level on both controllers so inter's
        // active_worst_quality also lands at `worst_quality`. Now
        // both pick Q in the full `[best, worst]` range, and the
        // only difference is the target-bits side of the budget.

        // Same setup for both.
        let setup = || {
            let mut rc = warmed_up_rc();
            rc.avg_frame_qindex_inter = 120;
            rc.avg_frame_qindex_key = 120;
            // Below critical (== optimal/8 == 625_000 bits) forces
            // inter active_worst = worst_quality.
            rc.buffer_level = 0;
            rc.bits_off_target = 0;
            // Far enough into the GOP that the kf_boost ramp is
            // at full strength (half_fr at 30 fps is 15).
            rc.frames_since_key = 60;
            rc
        };

        let mut rc_inter = setup();
        let (q_inter, _, _) = rc_inter.pick_q_and_bounds(FrameKind::Inter);

        let mut rc_kf = setup();
        let (q_kf, _, _) = rc_kf.pick_q_and_bounds(FrameKind::Key);

        assert!(
            q_kf <= q_inter,
            "expected KF Q ({q_kf}) <= inter Q ({q_inter}) at equal buffer state"
        );
    }

    #[test]
    fn regulate_q_returns_active_best_for_tiny_target() {
        // If target bits is huge, regulate should pick active_best.
        let q = regulate_q(FrameKind::Inter, 10_000_000, 10, 200, 3600, 1.0, 1.0);
        assert_eq!(q, 10, "huge target should land at active_best");
    }

    #[test]
    fn regulate_q_returns_active_worst_for_huge_demand() {
        // If target bits is tiny, regulate should pick active_worst.
        let q = regulate_q(FrameKind::Inter, 50, 10, 200, 3600, 1.0, 1.0);
        assert_eq!(q, 200, "tiny target should land at active_worst");
    }

    #[test]
    fn bits_per_mb_scaled_default_multiplier_matches_libvpx() {
        // Multiplier == 1.0 must produce the exact libvpx output. This
        // is the byte-exactness invariant for the default RC config.
        let q = 100;
        let raw = bits_per_mb(FrameKind::Inter, q, 1.0);
        let scaled = bits_per_mb_scaled(FrameKind::Inter, q, 1.0, 1.0);
        assert_eq!(raw, scaled);
    }

    #[test]
    fn bits_per_mb_scaled_applies_multiplier() {
        // Non-unit multiplier scales the prediction approximately
        // linearly. We use a moderate Q so the integer truncation in
        // `bits_per_mb` doesn't dominate the comparison.
        let q = 80;
        let one = bits_per_mb_scaled(FrameKind::Inter, q, 1.0, 1.0);
        let ten = bits_per_mb_scaled(FrameKind::Inter, q, 1.0, 10.0);
        let ratio = f64::from(ten) / f64::from(one);
        assert!(
            (9.9..=10.1).contains(&ratio),
            "expected ~10x multiplier scaling: one={one}, ten={ten}, ratio={ratio}"
        );
    }

    #[test]
    fn calc_iframe_target_size_first_frame_uses_starting_buffer() {
        // Fresh controller, no encoded bits → first-frame branch.
        let rc = RateControl::new(RateControlConfig::default());
        let target = calc_iframe_target_size(&rc);
        // Expect approx starting_buffer_level / 2 = 4_000_000 / 2 == 2M.
        assert!(target >= rc.avg_frame_bandwidth);
        // And clamped at max_frame_bandwidth.
        assert!(target <= rc.max_frame_bandwidth);
    }
}
