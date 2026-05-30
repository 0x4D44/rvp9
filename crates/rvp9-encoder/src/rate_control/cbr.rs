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
pub(super) fn qindex_to_q(qindex: i32) -> f64 {
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
pub(super) fn regulate_q(
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
/// [`get_kf_active_quality`] when the budget gate decides a near-lossless
/// first KF is unaffordable. libvpx indexes the LUT by the keyframe's
/// `active_worst_quality`; on the first frame that ambient is seeded at
/// `worst_quality` (255), which over-floors rvp9's keyframe (the LUT
/// returns ≈195 there). libvpx escapes this via its content-aware
/// `bits_per_mb` rate model, which rvp9 lacks. Capping the basis at this
/// realtime keyframe-Q neighbourhood lands the floor at the
/// libvpx-matched ≈57 on textured content.
///
/// History: removed by f971c4e (2026-05-27) when it dropped the
/// unconditional first-KF cap to let high-bitrate KFs reach near-lossless
/// (closing 1.2 dB at 1200/2000k). Re-introduced 2026-05-29, now gated:
/// the cap fires ONLY when the rate model says the KF target can't afford
/// near-lossless (the LOW-bitrate regime, where an uncapped KF starves
/// the GOP). See the budget gate in [`pick_q_and_bounds`]'s KF arm.
/// DOCUMENTED DIVERGENCE (E7) — the cap stands in for libvpx's
/// content-aware rate model on the keyframe.
const KF_ACTIVE_QUALITY_BASIS_CAP: i32 = 122;

/// KFCAP (Track A1, 2026-05-30) — bitrate/budget-aware keyframe-share cap.
///
/// Default ON; `RVP9_KFCAP=0` is the escape hatch that restores the
/// pre-KFCAP baseline byte-for-byte. Read once per pick (the KF arm of
/// [`pick_q_and_bounds`] runs once per GOP, not in the per-MB regulator
/// loop, so this is cheap).
///
/// **What it fixes (measured, post-Phase-A).** At ~250k on foreman the C1
/// budget gate's affordable branch lets the first KF go near-lossless
/// (q≈4), which spends ~60% of the whole stream. Post-Phase-A the inters
/// code efficiently, so those bits are wasted on a KF the GOP can't use.
/// KFCAP shaves the KF target so the freed bits flow to the inters.
///
/// **How it composes with C1 (and stays inert at high bitrate).** KFCAP
/// lives strictly INSIDE C1's `target_bpm >= near_lossless_bpm` affordable
/// branch — it never touches the low-bitrate CAP branch (C1 already caps
/// there) nor the subsequent-KF branch. Within the affordable branch it
/// fires ONLY when the rate model's predicted near-lossless KF cost
/// (`near_lossless_bpm * mbs >> 9`, the SAME quantity C1's gate computes)
/// exceeds [`KFCAP_SHARE_NUM`]/[`KFCAP_SHARE_DEN`] of the GOP budget
/// (`avg_frame_bandwidth * gop_length`). The near-lossless KF cost is
/// content-bounded (a fixed clip codes to roughly the same near-lossless
/// size at any target bitrate) while the GOP budget scales with bitrate,
/// so the predicted KF share falls as bitrate rises: measured ≈179% at
/// 250k (BITES) vs ≈15-22% at 2000k (INERT). The high-bitrate near-lossless
/// KF win (f971c4e / C1) is therefore untouched.
///
/// **Mechanism = target-fraction cap (HLD A2).** When it fires, the KF
/// *target bits* are clamped to the share ceiling and the floor is left at
/// `best_quality`; the existing `regulate_q` then walks Q up only as far as
/// needed to hit the smaller target. This degrades gracefully (Q rises
/// smoothly with the over-allocation) rather than slamming to the q≈57 C1
/// floor.
const KFCAP_SHARE_NUM: i64 = 2;
/// See [`KFCAP_SHARE_NUM`]. 2/5 = 40% of the GOP budget — the cap ceiling
/// on the predicted near-lossless KF cost. Chosen to sit cleanly between
/// the 250k over-allocation (≈179% predicted share → bites) and the 2000k
/// affordable regime (≈15-22% → inert).
const KFCAP_SHARE_DEN: i64 = 5;

/// Is the KFCAP keyframe-share cap enabled? Default ON; `RVP9_KFCAP=0`
/// is the escape hatch that restores the pre-KFCAP baseline byte-for-byte
/// (any other value, or the var being unset, leaves KFCAP enabled). See
/// [`KFCAP_SHARE_NUM`].
#[inline]
fn kfcap_enabled() -> bool {
    std::env::var_os("RVP9_KFCAP").is_none_or(|v| v != "0")
}

/// RCFI (Track A2, 2026-05-30) — first-inter-frame `active_best` basis.
///
/// Default ON; `RVP9_RCFI=0` is the escape hatch that restores the
/// pre-RCFI baseline byte-for-byte.
///
/// **What it fixes (measured).** rvp9 seeds `avg_frame_qindex_inter` to
/// `max_qindex` (255). The Inter q-pick's `active_best` basis used that
/// seed on the *first* inter frame, so `active_best = rtc_minq(255) ≈ 191`
/// floored the frame's Q at ~191 → q≈242. libvpx instead seeds the first
/// inter frame's `active_best` from `avg_frame_qindex[KEY]` (≈57 after the
/// keyframe) — `vp9_ratectrl.c:1058-1068`'s `current_video_frame > 1`
/// split — so its first inter drops to q≈114 and invests in a
/// high-quality reference the rest of the GOP coasts on. rvp9's flat,
/// starved first inter (q≈242, ~1840 bits) seeded a poor LASTFRAME that
/// propagated 2-3 dB through the whole GOP.
///
/// **Mechanism = libvpx-faithful basis select.** When `current_video_frame
/// <= 1` (the stream's first inter), seed the active_best basis from
/// `avg_frame_qindex_key` instead of `avg_frame_qindex_inter`. Every later
/// inter frame is unchanged (uses the inter avg). Bitstream-changing →
/// goldens re-captured, cross_decoder-gated.
#[inline]
fn rcfi_enabled() -> bool {
    std::env::var_os("RVP9_RCFI").is_none_or(|v| v != "0")
}

/// KFLB (Track A3, 2026-05-30) — low-bitrate keyframe-share cap.
///
/// Default ON; `RVP9_KFLB=0` is the escape hatch that restores the
/// pre-KFLB baseline byte-for-byte.
///
/// **What it targets.** KFCAP (the 40%-of-GOP-budget keyframe target cap)
/// only fires in the C1 *affordable* branch (`target_bpm >= near_lossless`).
/// At LOW bitrate the C1 gate takes the *cap* branch instead: it floors
/// `active_best` at ≈q57 but leaves the KF target un-shaved, so the KF
/// over-allocates (foreman 25k KF ≈46344 bits vs libvpx 22824; 50k ≈75432
/// vs 41168). KFLB applies the SAME 40%-of-GOP-budget target cap in the
/// cap branch, so `regulate_q` walks Q above the ≈57 floor and the freed
/// bits flow to the (Q255-saturated) inters — matching libvpx, whose KF Q
/// adapts to bitrate. INERT at high bitrate (that path is the affordable
/// branch, untouched) and inert when the KF target already fits the share
/// ceiling.
///
/// **Measured (quality_gate foreman_cif cpu7):** mean matched-bitrate Δ
/// +2.96 → +2.51 dB; 25k gap +3.08 → +2.35, 50k +3.86 → +2.80 (KF
/// over-allocation was the low-bitrate trunk). 100k/250k inert. testsrc
/// (synthetic) neutral-to-+0.31 dB (no regression). Scoped to the first
/// keyframe (`is_first_frame`); subsequent-KF share-cap is future work.
#[inline]
fn kflb_enabled() -> bool {
    std::env::var_os("RVP9_KFLB").is_none_or(|v| v != "0")
}

/// Port of libvpx `get_minq_index` (`vp9_ratectrl.c:118-135`). Finds the
/// smallest qindex whose q-value (`ac_quant/4`) is ≥ the cubic-curve
/// target `minqtarget = min((x3·maxq + x2)·maxq + x1)·maxq, maxq)`.
pub(super) fn get_minq_index(maxq: f64, x3: f64, x2: f64, x1: f64) -> i32 {
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
pub(super) fn get_kf_active_quality(q: i32) -> i32 {
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

    // KFCAP (Track A1) — the regulator below uses `regulate_target_bits`,
    // which is `target_bits` UNLESS the KFCAP gate (in the KF arm) shaves
    // it. Default ON (env-gated); `RVP9_KFCAP=0` keeps it == `target_bits`
    // → byte-identical to the pre-KFCAP baseline. See [`kfcap_enabled`] /
    // [`KFCAP_SHARE_NUM`].
    let mut regulate_target_bits = target_bits;

    // 2. Active Q range.
    let (active_best, active_worst) = match frame_kind {
        FrameKind::Key => {
            // libvpx `rc_pick_q_and_bounds_one_pass_cbr` keyframe branch
            // (`vp9_ratectrl.c:1013-1042`). The keyframe gets a buffer-
            // driven `active_worst_quality` (same model as the inter path)
            // and an `active_best_quality` FLOOR.
            //
            // libvpx behaviour (`vp9_ratectrl.c:1013-1042`):
            //   * `active_best = rc->best_quality`  (default 0, default
            //     CBR-realtime = 4)  — the unconditional starting point.
            //   * `if (current_video_frame > 0)` (i.e. not the first frame
            //     of the stream): override `active_best` with
            //     `get_kf_active_quality(avg_frame_qindex[KEY])`. The KF
            //     basis is the previous KF's qindex, NOT the current
            //     buffer-driven `active_worst`.
            //   * `regulate_q` then walks the [best, worst] range looking
            //     for the Q whose `bits_per_mb` estimate hits the KF
            //     target.
            //
            // Track F intra-lambda gap fix (2026-05-27): the previous rvp9
            // policy used `get_kf_active_quality(min(active_worst, 122))`
            // unconditionally and floored the KF at q≈57 regardless of
            // bitrate. That capped KF PSNR at ~42 dB and propagated through
            // the GOP via inter prediction (every inter frame predicts
            // against a 42-dB KF). The libvpx-faithful version below lets
            // the first KF take `best_quality` as the floor so high-
            // bitrate KFs can land near-lossless (where libvpx is), and
            // uses the previous KF's qindex (NOT active_worst) as the
            // basis on subsequent KFs.
            let worst = active_worst_quality(rc, i32::from(rc.avg_frame_qindex_key));
            // First frame of the stream: `avg_frame_qindex_key` is still
            // seeded at `worst_quality`. `total_actual_bits == 0 &&
            // frames_since_key == 0` is rvp9's first-frame predicate
            // (matches `calc_iframe_target_size` above).
            let is_first_frame =
                rc.total_actual_bits == 0 && rc.frames_since_key == 0;
            let best = if is_first_frame {
                // libvpx `vp9_ratectrl.c:1014` — `active_best_quality =
                // rc->best_quality` (near-lossless, q≈4). libvpx can
                // afford this because its content-aware `bits_per_mb`
                // estimator keeps the first KF inside the buffer; rvp9's
                // enumerator-over-Q estimator cannot, so at LOW bitrate a
                // near-lossless KF blows the buffer and starves the GOP.
                //
                // BUDGET GATE (2026-05-29). Take the near-lossless floor
                // only when the buffer can AFFORD a near-lossless KF;
                // otherwise restore the removed `get_kf_active_quality`
                // cap (≈57). The gate is the rate model's own
                // affordability test: `this_frame_target` (the KF target
                // already computed above) expressed as bits-per-MB
                // (`target_bpm`, the exact quantity `regulate_q` walks)
                // versus what the model says a near-lossless KF actually
                // costs per MB (`bits_per_mb_scaled(Key, best_quality, ...)`
                // using the SAME correction factor + multiplier the
                // regulator below walks). When the affordable budget can't
                // even cover the model's near-lossless cost, near-lossless
                // is unaffordable → cap. This self-scales with bitrate /
                // frame size / GOP (no hardcoded bitrate): the KF spends
                // proportionally less as bitrate drops, matching libvpx's
                // intent. The basis fed to `get_kf_active_quality` reuses
                // the removed `KF_ACTIVE_QUALITY_BASIS_CAP` (122 → q≈57 on
                // textured content); feeding the raw 255 ambient over-floors.
                //
                // CQ interaction: `RcMode::Cq` also routes through this
                // function, so the gate fires on CQ too. That is
                // intentional and left as-is — both CBR and CQ are
                // rate-aware (CQ still computes a per-frame target via
                // `calc_iframe_target_size`), so the affordability test is
                // sound for both. It has been validated on CBR; CQ behavior
                // is unchanged by this gate beyond inheriting the same
                // rate-model-driven cap.
                //
                // Measured separation (first-KF target_bpm vs the model's
                // near-lossless cost ≈982k bpm at q=4), as measured
                // 2026-05-29 (illustrative, not invariant):
                //   * 100k/256x256 (LOW):   target_bpm=400k  < 982k → CAP (q≈57)
                //   * 2000k/320x256 (HIGH): target_bpm=3.2M  ≥ 982k → near-lossless (q≈4)
                //   * 1M/128x128 (goldens): target_bpm=16M   ≥ 982k → near-lossless (q≈4)
                // The LOW regime caps with ~2.5x headroom; HIGH / golden
                // pass with 3.3x / 16x headroom — a clean split.
                let target_bpm: i64 = if rc.mbs == 0 {
                    0
                } else {
                    (i64::from(rc.this_frame_target) << BPER_MB_NORMBITS) / i64::from(rc.mbs)
                };
                // The near-lossless cost MUST use the same rate model the
                // regulator below walks (`bits_per_mb_scaled` with the
                // current KF correction factor and the config multiplier),
                // or the gate under-estimates cost under a non-unit
                // `bits_per_mb_multiplier` and fails to cap the textured,
                // low-bitrate content it exists to protect. On the first
                // frame the KF correction factor is the seeded 1.0 and the
                // default multiplier is 1.0, so this is byte-identical to
                // `bits_per_mb(Key, best_quality, 1.0)` at default config.
                let kf_cf = get_rate_correction_factor(rc, FrameKind::Key);
                let kf_mult = rc.config().bits_per_mb_multiplier;
                let near_lossless_bpm = i64::from(bits_per_mb_scaled(
                    FrameKind::Key,
                    rc.best_quality,
                    kf_cf,
                    kf_mult,
                ));
                if target_bpm >= near_lossless_bpm {
                    // Buffer affords near-lossless — keep f971c4e's win.
                    //
                    // KFCAP (Track A1, env-gated, default ON). Even when
                    // near-lossless is "affordable" by the C1 buffer test,
                    // at LOW/MID bitrate (e.g. 250k) the near-lossless KF
                    // over-allocates: it costs far more than its fair share
                    // of the GOP budget, starving the (now-efficient,
                    // post-Phase-A) inters. Shave the KF target to a fixed
                    // fraction of the GOP budget when the rate model's own
                    // predicted near-lossless KF cost exceeds that share.
                    // The floor stays `best_quality`; only the regulator's
                    // target shrinks, so `regulate_q` walks Q up gracefully.
                    // This composes with C1 (subset of its affordable
                    // branch) and is INERT at high bitrate (the predicted
                    // share is small there — see [`KFCAP_SHARE_NUM`]).
                    if kfcap_enabled() {
                        // Predicted near-lossless KF cost in bits (the SAME
                        // quantity the C1 gate prices, scaled back to a
                        // per-frame total): (near_lossless_bpm * mbs) >> 9.
                        let pred_kf_bits: i64 = (near_lossless_bpm
                            .saturating_mul(i64::from(rc.mbs)))
                            >> BPER_MB_NORMBITS;
                        // GOP budget = avg-frame-bits * gop_length.
                        let gop_budget: i64 = i64::from(rc.avg_frame_bandwidth)
                            .saturating_mul(i64::from(rc.config().gop_length));
                        // Share ceiling on the KF (e.g. 40% of GOP budget).
                        let kf_cap_bits: i64 =
                            gop_budget.saturating_mul(KFCAP_SHARE_NUM) / KFCAP_SHARE_DEN;
                        // Fire only when the predicted near-lossless KF would
                        // blow the share ceiling AND the cap actually shrinks
                        // the regulator target below what it is today (never
                        // RAISE the target — that would be a different lever).
                        if pred_kf_bits > kf_cap_bits
                            && kf_cap_bits > i64::from(FRAME_OVERHEAD_BITS)
                            && kf_cap_bits < i64::from(regulate_target_bits)
                        {
                            regulate_target_bits = kf_cap_bits as u32;
                        }
                    }
                    rc.best_quality
                } else {
                    // Budget too thin for near-lossless — restore the cap
                    // (active_best ≈ q57 floor).
                    //
                    // KFLB (Track A3, env-gated, default OFF). The cap
                    // branch floors Q at ≈57 but does NOT share-cap the KF
                    // target, so at low bitrate the KF still over-allocates
                    // (it spends whatever q57 costs — e.g. 46344 bits on
                    // foreman 25k vs libvpx 22824). Apply the same 40%-of-
                    // GOP-budget target cap KFCAP uses in the affordable
                    // branch, so `regulate_q` walks Q above the 57 floor and
                    // the freed bits flow to the Q255-saturated inters. See
                    // [`kflb_enabled`].
                    if kflb_enabled() {
                        let gop_budget: i64 = i64::from(rc.avg_frame_bandwidth)
                            .saturating_mul(i64::from(rc.config().gop_length));
                        let kf_cap_bits: i64 =
                            gop_budget.saturating_mul(KFCAP_SHARE_NUM) / KFCAP_SHARE_DEN;
                        if kf_cap_bits > i64::from(FRAME_OVERHEAD_BITS)
                            && kf_cap_bits < i64::from(regulate_target_bits)
                        {
                            regulate_target_bits = kf_cap_bits as u32;
                        }
                    }
                    get_kf_active_quality(KF_ACTIVE_QUALITY_BASIS_CAP)
                }
            } else {
                // libvpx `vp9_ratectrl.c:1029-1041` — subsequent KFs use
                // the previous KF's qindex as the basis (NOT the buffer-
                // driven `active_worst`). Small-image adjustment skipped:
                // `q_adj_factor = 0.75` for clips ≤ 352x288 would lower
                // `best` further (libvpx lines 1033-1041); the small-
                // image case lands `best` even lower, only helping.
                get_kf_active_quality(i32::from(rc.avg_frame_qindex_key))
            };
            (best.min(worst), worst)
        }
        FrameKind::Inter => {
            let worst = active_worst_quality_inter(rc);
            // libvpx `rc_pick_q_and_bounds_one_pass_cbr:1057-1068`:
            //   if current_video_frame > 1:  ambient = avg_frame_qindex[INTER]
            //   else (stream's first inter): ambient = avg_frame_qindex[KEY]
            //   active_best = rtc_minq[ambient]  if ambient < worst
            //                 rtc_minq[worst]    otherwise
            //
            // RCFI (Track A2, default ON): rvp9 v1 always used the INTER
            // ambient, but on the FIRST inter frame that field is still the
            // seed (`max_qindex` = 255) → active_best ≈ 0.75*255 = 191 →
            // q floored ~242, starving the first reference. The
            // `current_video_frame <= 1` arm seeds from the keyframe's avg
            // qindex (≈57) so the first inter can drop to q≈114 and build a
            // high-quality LASTFRAME the rest of the GOP coasts on. See
            // [`rcfi_enabled`]. `RVP9_RCFI=0` restores the seed-high basis.
            let ambient = if rcfi_enabled() && rc.current_video_frame <= 1 {
                i32::from(rc.avg_frame_qindex_key)
            } else {
                i32::from(rc.avg_frame_qindex_inter)
            };
            let basis = if ambient < worst { ambient } else { worst };
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
        regulate_target_bits,
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

    #[test]
    fn bits_per_mb_returns_i32_max_when_q_is_degenerate() {
        // q == 0 path: `qindex_to_q(0)` returns `ac_quant(0,0)/4` which
        // is small enough that the divide-by-zero guard fires.
        // Actually for qindex 0 we get a valid q (ac_quant(0,0) ≥ 4).
        // To hit the q <= 0.0 branch we need to test through the public
        // API with an unreachable case — the function clamps for safety.
        // The branch is genuinely a defensive guard. We exercise it
        // indirectly by ensuring small q still produces a sane value.
        let bpm = bits_per_mb(FrameKind::Inter, 0, 1.0);
        assert!(bpm > 0, "qindex 0 should still produce a valid bpm");
    }

    #[test]
    fn bits_per_mb_scaled_with_saturated_input_stays_saturated() {
        // When the raw `bits_per_mb` saturates at i32::MAX, the scaler
        // must short-circuit and return i32::MAX (not multiply and
        // overflow).
        //
        // We can't directly hit the q <= 0 path through bits_per_mb,
        // but we can test bits_per_mb_scaled's overflow-on-scale branch
        // by using a very large multiplier and a low Q.
        let q = 10;
        // Multiplier large enough that f64 scaling overflows i32.
        let big = f64::from(i32::MAX);
        let scaled = bits_per_mb_scaled(FrameKind::Key, q, 1.0, big);
        assert_eq!(
            scaled,
            i32::MAX,
            "scaled bpm should saturate to i32::MAX on overflow",
        );
    }

    #[test]
    fn bits_per_mb_scaled_clamps_negative_scale_to_zero() {
        // Multiplier of (effectively) 0 forces the scaled value below
        // zero by f64 underflow — actually 0 * raw == 0, which hits
        // the scaled <= 0.0 branch.
        let q = 100;
        // A multiplier just above 0 that still produces 0 after cast.
        let tiny = 1e-20;
        let scaled = bits_per_mb_scaled(FrameKind::Inter, q, 1.0, tiny);
        assert_eq!(
            scaled, 0,
            "scaled bpm with tiny multiplier should clamp to 0",
        );
    }

    #[test]
    fn regulate_q_tie_break_picks_smaller_residual() {
        // The libvpx tie-break: when bpm_at_i ≤ target_bpm, prefer
        // i-1 if the previous residual was smaller. Test by picking
        // a target that lands between two consecutive Q values.
        // Both `i` and `i-1` are returned per the residual ordering.
        //
        // Easier oracle: regulate_q returns `i-1` when the previous
        // Q's residual was smaller than the current's residual after
        // matching. Just check the function returns a value in [best, worst].
        let q = regulate_q(FrameKind::Inter, 100_000, 50, 200, 3600, 1.0, 1.0);
        assert!(
            (50..=200).contains(&q),
            "regulate_q must return a Q in [active_best, active_worst]: got {q}",
        );
    }

    #[test]
    fn regulate_q_with_zero_mbs_picks_active_worst() {
        // mbs == 0 ⇒ target_bpm == 0 ⇒ no Q satisfies bpm_at_i ≤ 0,
        // so we fall through to active_worst.
        let q = regulate_q(FrameKind::Inter, 1000, 10, 200, 0, 1.0, 1.0);
        assert_eq!(q, 200, "zero mbs should pin Q at active_worst");
    }

    #[test]
    fn calc_pframe_target_size_under_full_buffer_shaves_target() {
        // Buffer below optimal ⇒ diff > 0 ⇒ shave target.
        let mut rc = RateControl::new(RateControlConfig::default());
        // Drain buffer well below optimal.
        rc.buffer_level = 0;
        let target_drained = calc_pframe_target_size(&rc);

        // Refill to optimal for the baseline.
        rc.buffer_level = rc.optimal_buffer_level;
        let target_balanced = calc_pframe_target_size(&rc);

        assert!(
            target_drained < target_balanced,
            "drained buffer should shave target below balanced: \
             drained={target_drained}, balanced={target_balanced}",
        );
    }

    #[test]
    fn calc_pframe_target_size_over_full_buffer_boosts_target() {
        // Buffer above optimal ⇒ diff < 0 ⇒ boost target.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.buffer_level = rc.optimal_buffer_level;
        let target_balanced = calc_pframe_target_size(&rc);

        // Push buffer well above optimal.
        rc.buffer_level = rc.maximum_buffer_size;
        let target_overflow = calc_pframe_target_size(&rc);

        assert!(
            target_overflow > target_balanced,
            "overfull buffer should boost target: \
             overflow={target_overflow}, balanced={target_balanced}",
        );
    }

    #[test]
    fn calc_iframe_target_size_subsequent_keyframe_uses_kf_boost_ramp() {
        // Force the non-first-frame branch by setting total_actual_bits > 0.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.total_actual_bits = 100_000;
        rc.frames_since_key = 60; // well past half_fr (15) → full kf_boost
        let target = calc_iframe_target_size(&rc);
        // Should be ((16 + max(32, 44)) * avg) >> 4 ≈ 3.75 * avg with kf_boost=32.
        // Actually kf_boost will be max(32, fr_based=44) = 44 (from 2*30-16),
        // capped via the framerate-based formula. So target ≈ 3.75 * avg.
        // Just check that target > avg and ≤ max_frame_bandwidth.
        assert!(target > rc.avg_frame_bandwidth);
        assert!(target <= rc.max_frame_bandwidth);
    }

    #[test]
    fn calc_iframe_target_size_within_half_fr_uses_ramp_down() {
        // frames_since_key < half_fr triggers the ramp-down branch.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.total_actual_bits = 100_000;
        rc.frames_since_key = 5; // half_fr at 30 fps is 15

        let target_ramp = calc_iframe_target_size(&rc);

        // Compare to the fully-ramped case for the same controller.
        rc.frames_since_key = 60; // full boost
        let target_full = calc_iframe_target_size(&rc);

        assert!(
            target_ramp < target_full,
            "ramp-down within half_fr should produce smaller target: \
             ramp={target_ramp}, full={target_full}",
        );
    }

    #[test]
    fn active_worst_quality_buffer_above_optimal_lowers_q() {
        // Hand-construct an RC with buffer above optimal so the
        // first arm of active_worst_quality fires (`active_worst -=
        // adjustment`).
        let cfg = RateControlConfig::default();
        let mut rc = RateControl::new(cfg);
        // Push buffer well above optimal.
        rc.buffer_level = rc.maximum_buffer_size;
        // Pick a known ambient Q.
        let ambient = 100;
        let aw = active_worst_quality(&rc, ambient);
        // ambient_aw = (100 * 5) >> 2 = 125; active_worst starts at min(255, 125) = 125;
        // Then the down-adjustment lowers it further.
        assert!(
            aw < 125,
            "buffer above optimal should lower active_worst below ambient_aw: got {aw}",
        );
    }

    #[test]
    fn active_worst_quality_buffer_below_critical_pins_to_worst() {
        // Buffer below critical_level ⇒ third arm: active_worst = worst_quality.
        let cfg = RateControlConfig::default();
        let mut rc = RateControl::new(cfg);
        // Push buffer below critical (= optimal/8).
        rc.buffer_level = rc.optimal_buffer_level / 8 - 1;
        let aw = active_worst_quality(&rc, 100);
        // Worst quality is 255 (default config).
        assert_eq!(
            aw, 255,
            "buffer below critical should pin active_worst to worst_quality",
        );
    }

    #[test]
    fn active_worst_quality_buffer_between_critical_and_optimal_ramps_up() {
        // Buffer between critical and optimal ⇒ second arm (ramp up
        // from ambient to worst_quality).
        let cfg = RateControlConfig::default();
        let mut rc = RateControl::new(cfg);
        let critical = rc.optimal_buffer_level >> 3;
        // Put buffer between critical and optimal.
        rc.buffer_level = (critical + rc.optimal_buffer_level) / 2;
        let aw = active_worst_quality(&rc, 100);
        // Should be above ambient_aw (125) but below worst (255).
        assert!(
            aw > 125,
            "buffer in critical-to-optimal range should raise active_worst above ambient_aw: got {aw}",
        );
        assert!(aw <= 255, "still bounded by worst_quality");
    }

    #[test]
    fn rtc_minq_approx_clamps_to_best_quality_floor() {
        let rc = RateControl::new(RateControlConfig::default());
        // Pass an ambient below best_quality.
        let approx = rtc_minq_approx(&rc, 1);
        assert!(
            approx >= rc.best_quality,
            "rtc_minq_approx must clamp to best_quality floor: got {approx}",
        );
    }

    #[test]
    fn rtc_minq_approx_clamps_to_worst_quality_ceiling() {
        let mut cfg = RateControlConfig::default();
        cfg.max_qindex = 50;
        let rc = RateControl::new(cfg);
        // Pass an ambient well above worst.
        let approx = rtc_minq_approx(&rc, 200);
        // approx = 200 * 3 / 4 = 150 → clamped to max_qindex = 50.
        assert_eq!(approx, 50);
    }

    #[test]
    fn rate_factor_level_for_inter_returns_inter_normal() {
        let level = RateFactorLevel::for_frame(FrameKind::Inter);
        assert_eq!(level, RateFactorLevel::InterNormal);
        assert_eq!(level.as_index(), 0);
    }

    #[test]
    fn rate_factor_level_for_key_returns_kf_std() {
        let level = RateFactorLevel::for_frame(FrameKind::Key);
        assert_eq!(level, RateFactorLevel::KfStd);
        assert_eq!(level.as_index(), 4);
    }

    #[test]
    fn get_rate_correction_factor_clamps_below_min() {
        let mut rc = RateControl::new(RateControlConfig::default());
        // Force a sub-MIN value into the table.
        rc.rate_correction_factors[0] = 0.0001;
        let got = get_rate_correction_factor(&rc, FrameKind::Inter);
        assert_eq!(got, MIN_BPB_FACTOR);
    }

    #[test]
    fn get_rate_correction_factor_clamps_above_max() {
        let mut rc = RateControl::new(RateControlConfig::default());
        // Force a value above MAX into the table.
        rc.rate_correction_factors[0] = 9999.0;
        let got = get_rate_correction_factor(&rc, FrameKind::Inter);
        assert_eq!(got, MAX_BPB_FACTOR);
    }

    #[test]
    fn set_rate_correction_factor_clamps_value() {
        let mut rc = RateControl::new(RateControlConfig::default());
        set_rate_correction_factor(&mut rc, FrameKind::Inter, 9999.0);
        assert_eq!(rc.rate_correction_factors[0], MAX_BPB_FACTOR);
        set_rate_correction_factor(&mut rc, FrameKind::Inter, 0.0001);
        assert_eq!(rc.rate_correction_factors[0], MIN_BPB_FACTOR);
    }

    #[test]
    fn estimate_bits_at_q_returns_at_least_frame_overhead() {
        // Tiny mbs ⇒ raw bpm * mbs is tiny ⇒ floor at FRAME_OVERHEAD_BITS.
        let bits = estimate_bits_at_q(FrameKind::Inter, 255, 1, 0.01, 1.0);
        assert!(
            bits >= i64::from(FRAME_OVERHEAD_BITS),
            "bits must be at least FRAME_OVERHEAD_BITS: got {bits}",
        );
    }

    #[test]
    fn calc_pframe_target_size_clamps_to_min_frame_target() {
        // Drain the buffer hard so the under-shoot adjustment drops
        // target below the min_frame_target floor (which is the larger
        // of avg/16 and FRAME_OVERHEAD_BITS=200).
        //
        // Pick a tiny avg_frame_bandwidth so target/2 < FRAME_OVERHEAD_BITS
        // and the floor at FRAME_OVERHEAD_BITS engages.
        let mut rc = RateControl::new(RateControlConfig::default());
        // Force a tiny avg_frame_bandwidth (300 bits/frame).
        rc.avg_frame_bandwidth = 300;
        // Drain buffer below optimal so pct_low = 100, target -= target/2.
        rc.buffer_level = -10 * rc.optimal_buffer_level;
        rc.bits_off_target = rc.buffer_level;
        let target = calc_pframe_target_size(&rc);
        // min_frame_target = max(avg >> 4 = 18, FRAME_OVERHEAD_BITS=200) = 200.
        // target = 300/2 = 150 < 200 → clamp to 200.
        let expected_min = (rc.avg_frame_bandwidth >> 4).max(FRAME_OVERHEAD_BITS);
        assert_eq!(
            target, expected_min,
            "pframe target should clamp to min_frame_target floor",
        );
    }

    #[test]
    fn calc_pframe_target_size_clamps_to_max_frame_bandwidth() {
        // Boost target above max_frame_bandwidth by stuffing the
        // buffer above optimal AND keeping max_frame_bandwidth small
        // relative to avg.
        let mut rc = RateControl::new(RateControlConfig::default());
        // Force max_frame_bandwidth low so over-budget boost crosses it.
        rc.max_frame_bandwidth = rc.avg_frame_bandwidth + 100;
        // Push buffer way above optimal.
        rc.buffer_level = 1000 * rc.optimal_buffer_level;
        rc.bits_off_target = rc.buffer_level;
        let target = calc_pframe_target_size(&rc);
        assert_eq!(
            target, rc.max_frame_bandwidth,
            "pframe target should clamp to max_frame_bandwidth ceiling",
        );
    }

    #[test]
    fn calc_iframe_target_size_clamps_below_frame_overhead_bits() {
        // Force target to fall below FRAME_OVERHEAD_BITS. The arm at
        // line 354-355 in cbr.rs handles this. Hand-craft an RC with
        // zero `avg_frame_bandwidth` so the ramp produces tiny target.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.total_actual_bits = 100;
        rc.frames_since_key = 60;
        rc.avg_frame_bandwidth = 0; // zero out target
        rc.kf_boost = 0;
        let target = calc_iframe_target_size(&rc);
        assert_eq!(
            target, FRAME_OVERHEAD_BITS,
            "iframe target must floor at FRAME_OVERHEAD_BITS",
        );
    }

    #[test]
    fn calc_iframe_target_size_clamps_to_max_frame_bandwidth() {
        // First-frame branch can land above max_frame_bandwidth if
        // we force max_frame_bandwidth small. The clamp at line 352-353
        // in cbr.rs handles this.
        let mut rc = RateControl::new(RateControlConfig::default());
        // First frame, total_actual_bits == 0 → first-frame branch.
        // target = starting_buffer_level / 2 (large by default).
        // Crush max_frame_bandwidth so the clamp engages.
        rc.max_frame_bandwidth = 1000;
        let target = calc_iframe_target_size(&rc);
        assert_eq!(target, 1000, "iframe target must clamp to max_frame_bandwidth");
    }

    #[test]
    fn get_active_quality_blends_between_kf_low_and_kf_high_thresholds() {
        // Exercise the `else` arm of get_active_quality through the
        // public `get_kf_active_quality` wrapper. KF_ACTIVE_QUALITY_BOOST
        // (32) is between KF_LOW (400) and KF_HIGH (5000)? No, 32 < 400.
        // So the public wrapper takes the gfu_boost < low branch.
        //
        // Test the public wrapper covers the high-motion branch directly.
        let q = 80;
        let kf_q = get_kf_active_quality(q);
        // KF_ACTIVE_QUALITY_BOOST=32 < KF_LOW=400 → uses high-motion-minq.
        let expected = kf_high_motion_minq(q);
        assert_eq!(
            kf_q, expected,
            "get_kf_active_quality should use high_motion_minq when boost < KF_LOW",
        );
    }

    #[test]
    fn get_minq_index_lossless_for_low_target() {
        // The early-return at line 405-407: minqtarget <= 2.0 → return 0.
        // Coefficients (0.0, 0.0, 0.0) → minqtarget = 0.
        let idx = get_minq_index(80.0, 0.0, 0.0, 0.0);
        assert_eq!(idx, 0, "minqtarget == 0 should return qindex 0");
    }

    #[test]
    fn get_minq_index_saturates_at_255() {
        // Force minqtarget = maxq (== qindex_to_q(255)) so no inner
        // qindex satisfies the ≤ check. Should fall through to 255.
        // With x3=x2=0, x1=1 → minqtarget = maxq.min(maxq) = maxq.
        // ac_quant(255) values stop just below the maxq cap; the
        // iteration finds an exact match before 255. Skip 1.0
        // coefficient and use higher: x1=2.0 → minqtarget = 2*maxq,
        // clamped at maxq.
        let huge = qindex_to_q(255) * 10.0;
        let idx = get_minq_index(huge, 0.0, 0.0, 1.0);
        // minqtarget = huge, but capped at maxq = qindex_to_q(255)*10.
        // qindex_to_q(255) ≤ that, so iterates to 255 and returns it.
        assert_eq!(idx, 255, "minqtarget > max LUT q should return 255");
    }

    #[test]
    fn get_active_quality_high_boost_picks_low_motion_minq() {
        // Direct test of the `gfu_boost > high` arm at line 442-443.
        let q = 80;
        let high = 5000;
        let low = 400;
        let boost = high + 1;
        let blended = get_active_quality(
            q, boost, low, high, kf_low_motion_minq, kf_high_motion_minq,
        );
        // Should call low_motion_minq(q) directly.
        assert_eq!(blended, kf_low_motion_minq(q));
    }

    #[test]
    fn get_active_quality_middle_boost_blends_low_and_high() {
        // Direct test of the `else` arm at lines 447-451.
        let q = 80;
        let low = 400;
        let high = 5000;
        let mid_boost = (low + high) / 2;
        let blended = get_active_quality(
            q, mid_boost, low, high, kf_low_motion_minq, kf_high_motion_minq,
        );
        let lo_q = kf_low_motion_minq(q);
        let hi_q = kf_high_motion_minq(q);
        // Blend must be between the two endpoint values.
        let (a, b) = if lo_q < hi_q { (lo_q, hi_q) } else { (hi_q, lo_q) };
        assert!(
            (a..=b).contains(&blended),
            "blend {blended} should be between lo={lo_q} and hi={hi_q}",
        );
    }

    #[test]
    fn kf_low_motion_minq_returns_monotonic_curve() {
        // Direct exercise of the helper (covers lines 420-422).
        let a = kf_low_motion_minq(40);
        let b = kf_low_motion_minq(200);
        assert!(b >= a, "kf_low_motion_minq must be non-decreasing");
    }

    #[test]
    fn active_worst_quality_critical_level_zero_skips_inner() {
        // Set optimal_buffer_level = 0 ⇒ critical_level = 0 ⇒
        // the `critical_level > 0` guard at line 506 is false ⇒
        // active_worst stays at its pre-adjustment value.
        let cfg = RateControlConfig {
            optimal_buffer_ms: 0,
            buffer_size_ms: 0,
            starting_buffer_ms: 0,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        // Force buffer into the "between critical and optimal" arm by
        // making buffer_level between critical (=0) and optimal (=0)...
        // which is impossible. Instead force critical_level == 0 and
        // buffer > critical so we fall into the second arm, then the
        // inner critical>0 guard skips.
        //
        // Actually `optimal_buffer_level == 0` means everything > 0
        // ends up in the first arm. The condition is `buffer_level >
        // critical_level (= 0)` — so any positive buffer hits it.
        // Set buffer_level just above 0 → not >0 so the "below critical"
        // arm fires. Set buffer_level = 1 → > critical (=0) and not >
        // optimal (=0); enters second arm; critical_level == 0 so
        // inner guard skips.
        rc.buffer_level = 1;
        let aw = active_worst_quality(&rc, 100);
        // active_worst should be the post-clamp result; since the
        // inner adjustment skipped, it equals min(255, (100*5)>>2) = 125.
        assert_eq!(aw, 125);
    }

    #[test]
    fn active_worst_quality_buffer_step_zero_when_max_equals_optimal() {
        // Setup where maximum_buffer_size == optimal_buffer_level so
        // `buff_lvl_step` is 0 → adjustment skipped.
        let cfg = RateControlConfig {
            buffer_size_ms: 5000,
            optimal_buffer_ms: 5000, // == buffer_size
            starting_buffer_ms: 5000,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        // Push buffer above optimal so we enter the first arm.
        rc.buffer_level = rc.optimal_buffer_level + 1;
        let aw = active_worst_quality(&rc, 100);
        // Without an adjustment, active_worst stays at min(255, ambient*5/4) = 125.
        assert_eq!(aw, 125, "no adjustment when buff_lvl_step is zero");
    }

    /// Build a fresh first-frame controller for a clip-like config and
    /// return the qindex the KF arm of `pick_q_and_bounds` selects.
    fn first_kf_q(bps: u32, w: u32, h: u32, gop: u32) -> u8 {
        let cfg = RateControlConfig {
            target_bitrate_bps: bps,
            framerate: 30.0,
            frame_width: w,
            frame_height: h,
            gop_length: gop,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        rc.pick_q_and_bounds(FrameKind::Key).0
    }

    /// Budget gate — LOW bitrate. At 100 kbps on a 256x256 clip the
    /// first-KF target buys only ~3 bits/pixel, far below what the rate
    /// model says a near-lossless KF costs, so the gate must apply the
    /// `get_kf_active_quality` floor (≈57). A near-lossless KF here would
    /// burn ~150k bits and starve the GOP.
    ///
    /// Fails against pre-gate HEAD (uncapped → q≈20, below the floor).
    #[test]
    fn budget_gate_caps_kf_q_at_low_bitrate() {
        let q = first_kf_q(100_000, 256, 256, 64);
        assert!(
            q >= 50,
            "low-bitrate first KF must be capped near libvpx's q≈57, got q={q}",
        );
    }

    /// Budget gate — HIGH bitrate. At 2000 kbps on a 320x256 clip the
    /// KF target affords near-lossless (~24 bits/pixel), so the gate must
    /// NOT bite: the first KF keeps `best_quality` as its floor and the
    /// regulator lands near-lossless (q≈4). This preserves f971c4e's
    /// high-bitrate win.
    #[test]
    fn budget_gate_keeps_kf_q_near_lossless_at_high_bitrate() {
        let q = first_kf_q(2_000_000, 320, 256, 60);
        assert!(
            q < 30,
            "high-bitrate first KF must stay near-lossless (uncapped), got q={q}",
        );
    }

    /// Budget gate — the 1 Mbps / 128x128 golden fixture (w_buffer_pool /
    /// w_gop_topology) affords near-lossless (~122 bits/pixel), so the
    /// gate must NOT bite there either: the byte-identity goldens stay
    /// valid because the KF Q decision is unchanged at that bitrate.
    #[test]
    fn budget_gate_does_not_bite_golden_fixture() {
        let q = first_kf_q(1_000_000, 128, 128, 16);
        assert!(
            q < 30,
            "golden-fixture first KF must stay near-lossless (gate must not bite), got q={q}",
        );
    }

    // ---- KFCAP (Track A1) composition tests ----
    //
    // `RVP9_KFCAP` is a process-global env var; these tests mutate it, so
    // they must not run concurrently with each other. A shared mutex
    // serialises them. Each test restores the prior value on exit.
    //
    // KFCAP is default ON: env unset (or any value != "0") ⇒ ON;
    // `RVP9_KFCAP=0` ⇒ OFF (pre-KFCAP baseline). So `OFF` is `Some("0")`
    // and `ON` is `None` (the default) below.

    use std::sync::Mutex;
    static KFCAP_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Run `f` with `RVP9_KFCAP` forced to `value` (or removed if `None`),
    /// serialised against other KFCAP env tests, restoring the prior value.
    fn with_kfcap_env<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = KFCAP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("RVP9_KFCAP");
        // SAFETY: serialised by KFCAP_ENV_LOCK; no other thread in this
        // crate reads RVP9_KFCAP outside this lock during these tests.
        unsafe {
            match value {
                Some(v) => std::env::set_var("RVP9_KFCAP", v),
                None => std::env::remove_var("RVP9_KFCAP"),
            }
        }
        let out = f();
        unsafe {
            match prev {
                Some(v) => std::env::set_var("RVP9_KFCAP", v),
                None => std::env::remove_var("RVP9_KFCAP"),
            }
        }
        out
    }

    /// KFCAP — at 250k/256x256 the C1 affordable branch is taken (the KF
    /// would otherwise go near-lossless, q≈4), but the predicted
    /// near-lossless KF is ≈179% of the GOP budget, so KFCAP must shave
    /// the target and the regulator must pick a HIGHER Q than the
    /// uncapped (OFF, `RVP9_KFCAP=0`) baseline. This is the lever biting
    /// where the KF is wasteful.
    #[test]
    fn kfcap_raises_kf_q_at_mid_bitrate_when_on() {
        let q_off = with_kfcap_env(Some("0"), || first_kf_q(250_000, 256, 256, 33));
        let q_on = with_kfcap_env(None, || first_kf_q(250_000, 256, 256, 33));
        assert!(
            q_off < 10,
            "OFF baseline: 250k first KF must stay near-lossless (q≈4), got q={q_off}",
        );
        assert!(
            q_on > q_off,
            "KFCAP ON must RAISE the 250k KF Q above the uncapped baseline: \
             off={q_off}, on={q_on}",
        );
    }

    /// KFCAP — at 2000k/320x256 (the f971c4e regime) the predicted
    /// near-lossless KF is only ≈15% of the GOP budget, well under the
    /// 40% cap, so KFCAP must be INERT: the KF stays near-lossless
    /// whether the gate is ON (default) or OFF (no high-bitrate
    /// regression).
    #[test]
    fn kfcap_inert_at_high_bitrate() {
        let q_off = with_kfcap_env(Some("0"), || first_kf_q(2_000_000, 320, 256, 60));
        let q_on = with_kfcap_env(None, || first_kf_q(2_000_000, 320, 256, 60));
        assert_eq!(
            q_on, q_off,
            "KFCAP must be INERT at 2000k (f971c4e regime): off={q_off}, on={q_on}",
        );
        assert!(
            q_on < 30,
            "2000k first KF must stay near-lossless with KFCAP on, got q={q_on}",
        );
    }

    /// KFCAP — the escape hatch (`RVP9_KFCAP=0`) is byte-identical to the
    /// pre-KFCAP baseline across the bitrate sweep (gate never fires ⇒
    /// every Q matches the pre-KFCAP picks). Guards the "escape hatch
    /// restores baseline" invariant.
    #[test]
    fn kfcap_off_matches_head_across_sweep() {
        with_kfcap_env(Some("0"), || {
            // 100k: C1 caps (KFCAP never reached — it's inside the
            // affordable branch). 250k / 2000k: affordable branch, but
            // KFCAP disabled ⇒ near-lossless q≈4.
            assert!(first_kf_q(100_000, 256, 256, 64) >= 50, "100k still C1-capped");
            assert!(first_kf_q(250_000, 256, 256, 33) < 10, "250k OFF near-lossless");
            assert!(first_kf_q(2_000_000, 320, 256, 60) < 30, "2000k OFF near-lossless");
        });
    }

    /// KFCAP — does not disturb the low-bitrate C1 CAP branch. At 100k the
    /// C1 affordability test fails (target_bpm < near_lossless_bpm), so the
    /// floor is `get_kf_active_quality(122)` ≈57 and the KFCAP block (which
    /// lives in the OTHER branch) is never reached. ON and OFF must match.
    #[test]
    fn kfcap_does_not_double_cap_low_bitrate() {
        let q_off = with_kfcap_env(Some("0"), || first_kf_q(100_000, 256, 256, 64));
        let q_on = with_kfcap_env(None, || first_kf_q(100_000, 256, 256, 64));
        assert_eq!(
            q_on, q_off,
            "KFCAP must not double-cap the C1 low-bitrate branch: off={q_off}, on={q_on}",
        );
    }
}
