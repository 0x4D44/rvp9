//! `RateControl` — running rate-control state.
//!
//! Mirrors the `RATE_CONTROL` struct in `vp9/encoder/vp9_ratectrl.h:88`
//! for the field set actually touched by the 1-pass CBR path. Field
//! names are kept identical to libvpx wherever possible to ease
//! cross-referencing; Rust idioms (i64 → `i64`, no explicit `*const`
//! pointers, no SVC arrays) replace the C-isms.

use super::{FrameKind, RateControlConfig, TwoPassState, cbr, postencode, two_pass, vbr};
use cbr::RATE_FACTOR_LEVELS;

/// Running rate-control state.
///
/// Construct via [`RateControl::new`]; drive via:
///
/// 1. [`RateControl::pick_q_and_bounds`] — before each frame.
/// 2. The encoder picks Q somewhere in `[bottom, top]` and encodes.
/// 3. [`RateControl::postencode_update`] — with the actual bit count.
/// 4. [`RateControl::advance_frame`] — to roll forward frame counters.
#[derive(Debug, Clone)]
pub struct RateControl {
    // ---- Buffer state ----
    /// Current buffer fullness in bits. Mirrors libvpx `rc->buffer_level`
    /// (`vp9_ratectrl.h:131`). In libvpx, `buffer_level` is always
    /// equal to `bits_off_target` after every update (see
    /// `update_buffer_level_postencode:328`); we keep them in sync
    /// the same way.
    pub buffer_level: i64,

    /// Running budget tracker: positive == under-spent, negative ==
    /// over-spent. Mirrors libvpx `rc->bits_off_target`
    /// (`vp9_ratectrl.h:132`).
    pub bits_off_target: i64,

    // ---- Frame budget targets ----
    /// Average bits per frame. = `target_bitrate_bps / framerate`,
    /// rounded. Recomputed only if framerate / bitrate change. Mirrors
    /// libvpx `rc->avg_frame_bandwidth` (`vp9_ratectrl.h:120`).
    pub avg_frame_bandwidth: u32,

    /// Floor on per-frame target bits — used to keep tiny frames from
    /// going to zero. libvpx `rc->min_frame_bandwidth`
    /// (`vp9_ratectrl.h:121`). = `avg_frame_bandwidth / 4` in v1.
    pub min_frame_bandwidth: u32,

    /// Ceiling on per-frame target bits — burst rate. libvpx
    /// `rc->max_frame_bandwidth` (`vp9_ratectrl.h:122`). Computed in
    /// [`RateControl::new`] as `max(MBs * MAX_MB_RATE,
    /// MAXIMUM_FRAME_BANDWIDTH, avg * vbrmax/100)` — the libvpx
    /// `vp9_rc_update_framerate` formula. The `MAXIMUM_FRAME_BANDWIDTH`
    /// (2 Mbit) floor is what lets the keyframe target reach
    /// `starting_buffer_level / 2` instead of being clamped to ~4x avg.
    pub max_frame_bandwidth: u32,

    /// Target for the most recent `pick_q_and_bounds` call. libvpx
    /// `rc->this_frame_target`. Stored so `postencode_update` can
    /// compute the per-frame deviation.
    pub this_frame_target: u32,

    // ---- Q tracking ----
    /// Last keyframe Q-index. libvpx `rc->last_q[KEY_FRAME]`.
    pub last_q_keyframe: u8,

    /// Last inter Q-index. libvpx `rc->last_q[INTER_FRAME]`.
    pub last_q_inter: u8,

    /// Damped EWMA of recent keyframe Q. libvpx
    /// `rc->avg_frame_qindex[KEY_FRAME]` (`vp9_ratectrl.h:127`). Updated
    /// by `(3 * prev + new) >> 2` post-encode.
    pub avg_frame_qindex_key: u8,

    /// Damped EWMA of recent inter Q. libvpx
    /// `rc->avg_frame_qindex[INTER_FRAME]`.
    pub avg_frame_qindex_inter: u8,

    /// Active Q-range for the most recent pick. libvpx `bottom_index`
    /// / `top_index` in `vp9_rc_pick_q_and_bounds`.
    pub active_min_qindex: u8,
    /// See `active_min_qindex`.
    pub active_max_qindex: u8,

    /// Q-index actually picked by the most recent
    /// [`Self::pick_q_and_bounds`]. Mirrors libvpx
    /// `cm->base_qindex` at the moment `vp9_rc_update_rate_correction_factors`
    /// runs (`vp9_ratectrl.c:1859`) — the damped feedback loop
    /// needs the picked Q to compute `projected_size_based_on_q`.
    pub last_picked_qindex: u8,

    // ---- CBR Q-resonance history ----
    /// Q-index used by the most recent CBR frame. q_2_frame holds the
    /// preceding frame's Q-index. Both fields start at zero, matching the
    /// zeroed libvpx encoder state before `vp9_rc_init`; CBR's adjust_q_cbr
    /// uses them to clamp a candidate between the two Q values after
    /// alternating rate errors.
    pub(crate) q_1_frame: u8,

    /// Q-index used by the CBR frame before q_1_frame. It is shifted from
    /// q_1_frame after each CBR post-encode update and starts at zero,
    /// matching libvpx rc->q_2_frame initialization.
    pub(crate) q_2_frame: u8,

    /// Correction class for the most recent CBR frame: -1 means
    /// overshoot, +1 means undershoot, and 0 means the correction was
    /// within the neutral deadband or no frame has been classified yet.
    /// This is libvpx rc->rc_1_frame, initialized to zero.
    pub(crate) rc_1_frame: i8,

    /// Correction class for the CBR frame before rc_1_frame, with the same
    /// -1 overshoot, +1 undershoot, and 0 neutral encoding. This is libvpx
    /// rc->rc_2_frame, initialized to zero.
    pub(crate) rc_2_frame: i8,

    // ---- Rate-correction feedback ----
    /// Per-level rate-correction factor table.
    ///
    /// Mirrors libvpx `rc->rate_correction_factors[RATE_FACTOR_LEVELS]`
    /// (`vp9_ratectrl.h:103`). Indexed by the
    /// [`cbr::RateFactorLevel`] enum. Multiplies the libvpx
    /// `vp9_rc_bits_per_mb` enumerator-over-Q estimate to compensate
    /// for systematic over-/under-estimation; damped-updated
    /// post-encode by `postencode::update_rate_correction_factors`.
    ///
    /// libvpx seeds every slot to 1.0 in `vp9_rc_init`
    /// (`vp9_ratectrl.c:419-422`).
    pub rate_correction_factors: [f64; RATE_FACTOR_LEVELS],

    /// Per-level "have we seen any frame at this damping level yet" flag.
    /// Mirrors libvpx `rc->damped_adjustment[RATE_FACTOR_LEVELS]`
    /// (`vp9_ratectrl.h:203`). First update at each level uses
    /// `adjustment_limit = 1.0` (no damping); subsequent updates use
    /// the log-magnitude damping formula
    /// (`vp9_ratectrl.c:770-779`). One-pass VBR's GF-group level remains
    /// `INTER_NORMAL`, so its key, inter, and GOLDEN correction-factor
    /// updates deliberately share slot zero even though their factor values
    /// remain independent.
    pub damped_adjustment: [bool; RATE_FACTOR_LEVELS],

    // ---- GOP state ----
    /// Frames until the next keyframe. libvpx `rc->frames_to_key`
    /// (`vp9_ratectrl.h:112`). Decremented in `advance_frame`; reset
    /// to `gop_length - 1` whenever a keyframe is encoded.
    pub frames_to_key: u32,

    /// Frames since the last keyframe. libvpx `rc->frames_since_key`
    /// (`vp9_ratectrl.h:113`).
    pub frames_since_key: u32,

    /// Absolute display-order frame index. Mirrors libvpx
    /// `cm->current_video_frame` (`vp9_common.h`) — incremented once per
    /// encoded show frame in [`Self::advance_frame`]. The 1-pass CBR
    /// q-pick uses `current_video_frame > 1` to seed the first inter
    /// frame's `active_best_quality` from the keyframe's avg qindex
    /// rather than the (seed-high) inter avg (`vp9_ratectrl.c:1058-1068`).
    pub current_video_frame: u32,

    /// Current keyframe Q-boost. libvpx `rc->kf_boost`
    /// (`vp9_ratectrl.h:101`).
    pub kf_boost: i32,

    // ---- History ----
    /// Cumulative bits actually emitted. libvpx
    /// `rc->total_actual_bits` (`vp9_ratectrl.h:147`).
    pub total_actual_bits: i64,

    /// Cumulative bits we *aimed* to emit. libvpx
    /// `rc->total_target_bits` (`vp9_ratectrl.h:148`).
    pub total_target_bits: i64,

    /// Number of frames dropped post-encode (not used in v1 — present
    /// only to mirror the libvpx field shape so future drop-logic can
    /// land without an ABI break). libvpx `rc->drop_count`.
    pub drop_count: u32,

    // ---- Config snapshot ----
    /// Cached MB count for the configured frame size. Avoids
    /// recomputing in the per-frame `regulate_q` loop.
    pub(crate) mbs: u32,

    /// Worst-quality clamp (== config `max_qindex`). Mirrors
    /// `rc->worst_quality`. Cached `i32` for arithmetic in `cbr.rs`.
    pub(crate) worst_quality: i32,

    /// Best-quality clamp (== config `min_qindex`). Mirrors
    /// `rc->best_quality`.
    pub(crate) best_quality: i32,

    /// One-pass VBR GOLDEN allocation interval. Production retains the
    /// calibrated legacy value; the libvpx-aligned diagnostic path
    /// replaces it with the frame-rate-derived interval.
    pub(crate) baseline_gf_interval: i64,

    /// libvpx `rc->avg_frame_low_motion`, the smoothed percentage of
    /// LAST-referenced cells whose motion vector is below two pixels.
    /// Updated after each show inter frame and consumed by one-pass VBR's
    /// Golden-frame allocation heuristic.
    pub(crate) avg_frame_low_motion: i32,

    /// libvpx `rc->af_ratio_onepass_vbr`, reset at each Golden update and
    /// adjusted from the running low-motion estimate after frame 30.
    pub(crate) af_ratio_onepass_vbr: i32,

    /// libvpx `rc->gfu_boost`, which selects the low-/high-motion Golden
    /// active-best-quality table for one-pass VBR.
    pub(crate) gfu_boost: i32,

    /// Max buffer capacity in bits (`buffer_size_ms * target_bps /
    /// 1000`). Mirrors libvpx `rc->maximum_buffer_size`
    /// (`vp9_ratectrl.h:156`).
    pub(crate) maximum_buffer_size: i64,

    /// Optimal buffer level in bits. Mirrors libvpx
    /// `rc->optimal_buffer_level` (`vp9_ratectrl.h:155`).
    pub(crate) optimal_buffer_level: i64,

    /// Starting buffer level in bits. Mirrors libvpx
    /// `rc->starting_buffer_level` (`vp9_ratectrl.h:154`).
    pub(crate) starting_buffer_level: i64,

    /// Original config (snapshot for re-init).
    config: RateControlConfig,

    /// Two-pass driver state, present only under
    /// [`super::RcMode::VbrTwoPass`]. Carries the first-pass stats buffer
    /// and the precomputed per-frame bit allocation that drive the
    /// 2-pass `pick_q` arm. `None` for every one-pass mode (CBR / 1-pass
    /// VBR / CQ / Q), so those paths are byte-identical to before this
    /// field existed. Attached by the encoder's two-pass driver via
    /// [`RateControl::attach_two_pass`].
    pub(super) two_pass: Option<TwoPassState>,
}

impl RateControl {
    /// Initialize from `config`. Mirrors libvpx `vp9_rc_init`
    /// (`vp9_ratectrl.c:360`) for the one-pass CBR branch and the
    /// non-CBR branch.
    ///
    /// # Panics
    ///
    /// Panics if `config.validate()` fails — config is caller-supplied
    /// and a panic here is the simplest contract for v1; future work
    /// will plumb a fallible constructor.
    #[must_use]
    pub fn new(config: RateControlConfig) -> Self {
        config
            .validate()
            .expect("RateControl: invalid RateControlConfig");

        let avg_frame_bandwidth = avg_frame_bw(&config);
        let min_frame_bandwidth = avg_frame_bandwidth / 4;
        // Per-frame burst ceiling. Mirrors libvpx `vp9_rc_update_framerate`
        // (`vp9_ratectrl.c:2069-2083`):
        //   vbr_max_bits = avg_frame_bandwidth * two_pass_vbrmax_section / 100
        //   max_frame_bandwidth =
        //       max(max(MBs * MAX_MB_RATE, MAXIMUM_FRAME_BANDWIDTH), vbr_max_bits)
        // with `MAX_MB_RATE = 250`, `MAXIMUM_FRAME_BANDWIDTH = 2_000_000`,
        // and libvpx's `two_pass_vbrmax_section` default of 2000 (== 20x avg)
        // for VBR. Keep the established 4x ceiling for other modes while this
        // source-faithful one-pass VBR correction is measured in isolation.
        //
        // PREVIOUS rvp9 DIVERGENCE: `max_frame_bandwidth = 4 * avg` only.
        // That starved the keyframe — the iframe target clamp
        // (`calc_iframe_target_size`) pinned the KF at `4 * avg` (~13 kbit
        // on foreman @ 100 kbps) instead of letting it spend the
        // `starting_buffer_level / 2` it asks for, so the KF coded at q=177
        // and seeded a poor LASTFRAME. libvpx's 2-Mbit floor lets the KF
        // grow ~5x, matching the reference (E6 keyframe-boost fix).
        const MAX_MB_RATE: u64 = 250;
        const MAXIMUM_FRAME_BANDWIDTH: u64 = 2_000_000;
        const DEFAULT_MAX_SECTION: u64 = 400;
        const VBR_MAX_SECTION: u64 = 2000;
        let max_section = if config.rc_mode == super::RcMode::Vbr {
            VBR_MAX_SECTION
        } else {
            DEFAULT_MAX_SECTION
        };
        let vbr_max_bits = u64::from(avg_frame_bandwidth) * max_section / 100;
        let mb_rate_bits = u64::from(config.mbs()) * MAX_MB_RATE;
        let max_frame_bandwidth = mb_rate_bits
            .max(MAXIMUM_FRAME_BANDWIDTH)
            .max(vbr_max_bits)
            .min(u64::from(u32::MAX)) as u32;

        let maximum_buffer_size = ms_to_bits(config.buffer_size_ms, config.target_bitrate_bps);
        let optimal_buffer_level = ms_to_bits(config.optimal_buffer_ms, config.target_bitrate_bps);
        let starting_buffer_level =
            ms_to_bits(config.starting_buffer_ms, config.target_bitrate_bps);
        let avg_qindex_seed = if config.rc_mode == super::RcMode::Cbr {
            config.max_qindex
        } else {
            ((u16::from(config.max_qindex) + u16::from(config.min_qindex)) / 2) as u8
        };

        Self {
            buffer_level: starting_buffer_level,
            bits_off_target: starting_buffer_level,
            avg_frame_bandwidth,
            min_frame_bandwidth,
            max_frame_bandwidth,
            this_frame_target: avg_frame_bandwidth,

            // libvpx `vp9_rc_init`: CBR seeds avg_frame_qindex to
            // worst_allowed_q; one-pass VBR/CQ/Q/two-pass seed to the
            // midpoint of worst/best (`vp9_ratectrl.c:363-371`).
            last_q_keyframe: config.min_qindex,
            last_q_inter: config.max_qindex,
            avg_frame_qindex_key: avg_qindex_seed,
            avg_frame_qindex_inter: avg_qindex_seed,

            active_min_qindex: config.min_qindex,
            active_max_qindex: config.max_qindex,
            last_picked_qindex: config.max_qindex,
            // libvpx zeroes the complete encoder state before vp9_rc_init,
            // which also zero-initializes this history
            // (vp9_encoder.c:2380-2389).
            q_1_frame: 0,
            q_2_frame: 0,
            rc_1_frame: 0,
            rc_2_frame: 0,

            // libvpx `vp9_rc_init` seeds every slot of the
            // rate_correction_factors[] table to 1.0 and every
            // damped_adjustment[] flag to 0 (no damping applied to the
            // first frame at each level). `vp9_ratectrl.c:419-422`.
            rate_correction_factors: [1.0; RATE_FACTOR_LEVELS],
            damped_adjustment: [false; RATE_FACTOR_LEVELS],

            frames_to_key: 0, // first frame must be a keyframe
            frames_since_key: 0,
            current_video_frame: 0,
            kf_boost: config.kf_boost,

            total_actual_bits: 0,
            total_target_bits: 0,

            drop_count: 0,

            mbs: config.mbs(),
            worst_quality: i32::from(config.max_qindex),
            best_quality: i32::from(config.min_qindex),
            baseline_gf_interval: 15,
            avg_frame_low_motion: 0,
            af_ratio_onepass_vbr: 10,
            gfu_boost: 2000,
            maximum_buffer_size,
            optimal_buffer_level,
            starting_buffer_level,

            config,
            // One-pass modes leave this None (byte-identical path); the
            // two-pass driver attaches it via `attach_two_pass`.
            two_pass: None,
        }
    }

    /// Attach the [`TwoPassState`] for a [`super::RcMode::VbrTwoPass`]
    /// encode. The encoder's two-pass driver calls this once, after the
    /// first pass has gathered the per-frame [`super::FirstPassStats`],
    /// before the second pass begins. The 2-pass `pick_q_and_bounds` arm
    /// reads the per-frame bit allocation from here.
    pub fn attach_two_pass(&mut self, state: TwoPassState) {
        self.two_pass = Some(state);
    }

    /// Credit one cumulative temporal-layer frame budget before encoding.
    pub(crate) fn svc_preencode_buffer_update(&mut self) {
        self.bits_off_target = self
            .bits_off_target
            .saturating_add(i64::from(self.avg_frame_bandwidth))
            .min(self.maximum_buffer_size);
        self.buffer_level = self.bits_off_target;
    }

    /// Debit one encoded temporal-layer frame from a cumulative buffer.
    pub(crate) fn svc_postencode_buffer_debit(&mut self, actual_bits: u32) {
        self.bits_off_target = self.bits_off_target.saturating_sub(i64::from(actual_bits));
        self.bits_off_target = self
            .bits_off_target
            .clamp(-self.maximum_buffer_size, self.maximum_buffer_size);
        self.buffer_level = self.bits_off_target;
    }

    /// Raise the inter-Q ambient after a large temporal-SVC keyframe.
    pub(crate) fn svc_adjust_inter_ambient_after_keyframe(
        &mut self,
        actual_bits: u32,
        q_index: u8,
    ) -> Option<u8> {
        if i64::from(actual_bits) <= 3_i64.saturating_mul(i64::from(self.avg_frame_bandwidth)) {
            return None;
        }
        let corrected = self
            .avg_frame_qindex_inter
            .max(((i32::from(q_index) + self.worst_quality) >> 1) as u8);
        self.avg_frame_qindex_inter = corrected;
        Some(corrected)
    }

    /// Pick the Q-index for the next frame.
    ///
    /// Returns `(active_qindex, q_low_bound, q_high_bound)`. The caller
    /// is expected to use `active_qindex` as the starting point; the
    /// bounds describe the per-frame range the encoder may dither
    /// within (e.g. segment-level Q deltas, AQ adjustments).
    ///
    /// Wraps [`super::cbr::pick_q_and_bounds`].
    ///
    /// W1.6(vpxenc parity, A3 P0) — dispatches on `config.rc_mode`
    /// (libvpx `vp9_ratectrl.c:1587-1609`). All five arms are now live:
    /// the four one-pass regulators (`Cbr` / `Vbr` / `Cq` / `Q`,
    /// CLI-reachable via `--end-usage`) plus the two-pass driver
    /// (`VbrTwoPass`, reachable via `--passes 2` /
    /// [`crate::frame_loop::encode_two_pass`]). The 2-pass arm completes
    /// the rate-control family — it consumes the first-pass stats
    /// attached via [`Self::attach_two_pass`].
    pub fn pick_q_and_bounds(&mut self, frame_kind: FrameKind) -> (u8, u8, u8) {
        self.pick_q_and_bounds_with_model(frame_kind, None)
    }

    pub(crate) fn pick_q_and_bounds_with_model(
        &mut self,
        frame_kind: FrameKind,
        model: Option<super::CyclicRefreshRateModel>,
    ) -> (u8, u8, u8) {
        self.pick_q_and_bounds_with_model_and_target(frame_kind, model, None)
    }

    pub(crate) fn pick_q_and_bounds_with_model_and_target(
        &mut self,
        frame_kind: FrameKind,
        model: Option<super::CyclicRefreshRateModel>,
        svc_inter_target: Option<u32>,
    ) -> (u8, u8, u8) {
        use super::RcMode;
        match self.config.rc_mode {
            RcMode::Cbr => cbr::pick_q_and_bounds_with_model_and_target(
                self,
                frame_kind,
                model,
                svc_inter_target,
            ),
            // One-pass VBR. Port of libvpx
            // `rc_pick_q_and_bounds_one_pass_vbr`
            // (`vp9_ratectrl.c:1138-1303`). The two-pass arm
            // (`RcMode::VbrTwoPass`, port of
            // `rc_pick_q_and_bounds_two_pass`, `vp9_ratectrl.c:1427`)
            // is the separate `VbrTwoPass` dispatch below — it consumes
            // first-pass stats attached via `attach_two_pass`. The W1.6
            // one-pass VBR PROOF lives in `tests/w_vbr_rc_proof.rs`; the
            // two-pass PROOF in `tests/w_2pass_vbr.rs`.
            RcMode::Vbr => match (model, svc_inter_target) {
                (model, Some(target)) => vbr::pick_q_and_bounds_with_model_and_target(
                    self,
                    frame_kind,
                    model,
                    Some(target),
                ),
                (Some(model), None) => {
                    vbr::pick_q_and_bounds_with_model(self, frame_kind, Some(model))
                }
                (None, None) => vbr::pick_q_and_bounds(self, frame_kind),
            },
            // PHASE-3 integration: Constrained-Quality (CQ). Runs the
            // CBR regulator, then clamps the picked Q to be ≥
            // cq_level. Mirrors libvpx's CQ semantics: cq_level is a
            // quality FLOOR — the regulator can raise Q above it for
            // bitrate, but never go below it for quality. libvpx
            // `vp9_ratectrl.c::vp9_rc_pick_q_and_bounds_one_pass_cbr`
            // CQ branch (search for `cq_level`).
            RcMode::Cq => {
                debug_assert!(svc_inter_target.is_none());
                let (q, lo, hi) = match model {
                    Some(model) => cbr::pick_q_and_bounds_with_model(self, frame_kind, Some(model)),
                    None => cbr::pick_q_and_bounds(self, frame_kind),
                };
                let floor = self.config.cq_level;
                (q.max(floor), lo.max(floor), hi.max(floor))
            }
            // PHASE-3 integration: Constant-Quantizer (Q). Pins Q at
            // cq_level for every frame; keyframes get a libvpx-style
            // small qdelta (KF lower than inter for quality). Mirrors
            // libvpx's VPX_Q short-circuit (`vp9_ratectrl.c`, search
            // for `oxcf->rc_mode == VPX_Q`). No rate-control loop —
            // the regulator is bypassed entirely.
            RcMode::Q => {
                debug_assert!(svc_inter_target.is_none());
                let q_inter = self.config.cq_level;
                let q_kf = q_inter.saturating_sub(8); // libvpx KF qdelta ≈ -8
                let q = match frame_kind {
                    FrameKind::Key => q_kf,
                    FrameKind::Inter | FrameKind::Golden => q_inter,
                };
                (q, q, q)
            }
            // Two-pass VBR. Port of libvpx's `pass == 2` dispatch
            // (`vp9_ratectrl.c:1606-1608`) → `rc_pick_q_and_bounds_two_pass`
            // (`vp9_ratectrl.c:1427`). Consumes the first-pass stats
            // attached via `attach_two_pass`: the per-frame target is the
            // stats-driven bit allocation, NOT the flat 1-pass target.
            // This is the last rate-control mode to land — completing the
            // CBR / 1-pass VBR / CQ / Q / 2-pass-VBR family.
            RcMode::VbrTwoPass => two_pass::pick_q_and_bounds(self, frame_kind),
        }
    }

    /// Calculate a temporal-SVC CBR inter target from its incremental
    /// frame-size budget. Call this before the cumulative preencode credit.
    pub(crate) fn temporal_svc_inter_target(&self, avg_frame_size: u32) -> u32 {
        debug_assert!(matches!(
            self.config.rc_mode,
            super::RcMode::Cbr | super::RcMode::Vbr
        ));
        cbr::calc_pframe_target_size_with_average(self, avg_frame_size)
    }

    /// Update internal state after the frame has been encoded.
    ///
    /// `actual_bits` is the exact bit count of the emitted frame
    /// (compressed-header + tile data + frame overhead). Wraps
    /// [`super::postencode::postencode_update`].
    pub fn postencode_update(&mut self, actual_bits: u32, frame_kind: FrameKind) {
        postencode::postencode_update(self, actual_bits, frame_kind);
    }

    pub(crate) fn postencode_update_with_model(
        &mut self,
        actual_bits: u32,
        frame_kind: FrameKind,
        model: Option<super::CyclicRefreshRateModel>,
    ) {
        postencode::postencode_update_with_model(self, actual_bits, frame_kind, model);
    }

    pub(crate) fn postencode_update_svc_with_model(
        &mut self,
        actual_bits: u32,
        frame_kind: FrameKind,
        model: Option<super::CyclicRefreshRateModel>,
    ) {
        postencode::postencode_update_svc_with_model(self, actual_bits, frame_kind, model);
    }

    /// Advance frame counters. Called once per encoded *show* frame
    /// after [`Self::postencode_update`].
    ///
    /// Mirrors the `frames_since_key++` / `frames_to_key--` block at
    /// the bottom of `vp9_rc_postencode_update`
    /// (`vp9_ratectrl.c:1971-1975`). We split it out so callers can
    /// keep the post-encode update functional and the
    /// counter-bookkeeping explicit.
    pub fn advance_frame(&mut self) {
        if self.frames_to_key > 0 {
            self.frames_to_key -= 1;
        }
        self.frames_since_key = self.frames_since_key.saturating_add(1);
        // libvpx `cm->current_video_frame++` (per show frame). Used by
        // the CBR q-pick's first-inter-frame active_best special case.
        self.current_video_frame = self.current_video_frame.saturating_add(1);
        // Two-pass: advance the stats cursor so the next pick reads the
        // next frame's allocation. No-op for one-pass modes.
        if let Some(tp) = self.two_pass.as_mut() {
            tp.advance();
        }
    }

    /// Update libvpx's one-pass low-motion EWMA after a show inter frame.
    /// `low_motion_percent` is the percentage of 8×8 cells that use LAST
    /// with both MV components below two pixels.
    pub(crate) fn update_frame_low_motion(&mut self, low_motion_percent: u32) {
        let pct = low_motion_percent.min(100) as i32;
        self.avg_frame_low_motion = (3 * self.avg_frame_low_motion + pct) >> 2;
    }

    /// Read-only access to the snapshot config.
    #[must_use]
    pub fn config(&self) -> &RateControlConfig {
        &self.config
    }

    /// Diagnostic-only: predicted total bits at the most recent
    /// `pick_q_and_bounds`-selected Q, using the same
    /// `bits_per_mb_scaled` × `mbs` math the post-encode feedback loop
    /// uses to compute `projected_size_based_on_q`. Used by the A10
    /// `RVP9_DIAG_BPM_VERIFY=1` instrumentation in
    /// `frame_loop::encode_frame` to compare predicted-vs-actual per
    /// frame.
    #[must_use]
    pub fn predicted_bits_at_picked_q(&self, frame_kind: FrameKind) -> i64 {
        let correction = cbr::get_rate_correction_factor(self, frame_kind);
        let multiplier = self.config.bits_per_mb_multiplier;
        cbr::estimate_bits_at_q(
            frame_kind,
            i32::from(self.last_picked_qindex),
            self.mbs,
            correction,
            multiplier,
        )
    }

    /// Diagnostic-only (E8): snapshot of the buffer-state inputs and
    /// outputs of the inter `active_worst_quality` calculation, captured
    /// AFTER a `pick_q_and_bounds` call. Used by the
    /// `RVP9_E8_TRACE=1` instrumentation in `frame_loop::encode_frame`
    /// to discriminate "clamp artificially limiting" (case A) from
    /// "structural floor" (case B): does the buffer deplete below
    /// critical, and does `active_worst` ramp toward `worst_quality`?
    #[must_use]
    pub fn e8_trace_snapshot(&self) -> E8Trace {
        E8Trace {
            buffer_level: self.buffer_level,
            optimal_buffer_level: self.optimal_buffer_level,
            critical_level: self.optimal_buffer_level >> 3,
            maximum_buffer_size: self.maximum_buffer_size,
            ambient_qp_inter: i32::from(self.avg_frame_qindex_inter),
            ambient_cap: (i32::from(self.avg_frame_qindex_inter) * 5) >> 2,
            active_worst: i32::from(self.active_max_qindex),
            active_best: i32::from(self.active_min_qindex),
            picked_q: i32::from(self.last_picked_qindex),
            this_frame_target: self.this_frame_target,
        }
    }
}

/// Diagnostic-only (E8) snapshot of the inter `active_worst_quality`
/// buffer state. See [`RateControl::e8_trace_snapshot`].
#[derive(Copy, Clone, Debug)]
pub struct E8Trace {
    /// Current buffer fullness (bits).
    pub buffer_level: i64,
    /// Optimal buffer level (bits) — above this, ramp `active_worst` down.
    pub optimal_buffer_level: i64,
    /// Critical level (`optimal >> 3`) — below this, `active_worst` pins
    /// to `worst_quality`.
    pub critical_level: i64,
    /// Maximum buffer size (bits).
    pub maximum_buffer_size: i64,
    /// Ambient inter Q (the `avg_frame_qindex[INTER]` EWMA).
    pub ambient_qp_inter: i32,
    /// The `(ambient * 5) >> 2` base cap on `active_worst`.
    pub ambient_cap: i32,
    /// Chosen `active_worst_quality` (== `active_max_qindex`).
    pub active_worst: i32,
    /// Chosen `active_best_quality` (== `active_min_qindex`).
    pub active_best: i32,
    /// `regulate_q`'s clamped result (the Q actually used).
    pub picked_q: i32,
    /// Per-frame target bits.
    pub this_frame_target: u32,
}

/// `avg_frame_bandwidth = round(target_bps / framerate)`. Mirrors
/// libvpx `vp9_new_framerate` / `vp9_rc_update_framerate:2056`.
fn avg_frame_bw(cfg: &RateControlConfig) -> u32 {
    let raw = f64::from(cfg.target_bitrate_bps) / cfg.framerate;
    // Round-to-nearest. `f64::round` is the right call; the value is
    // always finite given `validate()`.
    let rounded = raw.round();
    // Clamp to u32::MAX to be defensive (caller could pass a huge
    // bitrate). At normal bitrates this is a no-op.
    if rounded < 0.0 {
        0
    } else if rounded > f64::from(u32::MAX) {
        u32::MAX
    } else {
        rounded as u32
    }
}

/// Convert a buffer-size-in-milliseconds to bits at the given bitrate.
fn ms_to_bits(ms: u32, bps: u32) -> i64 {
    (i64::from(ms) * i64::from(bps)) / 1000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_control_init_from_default_config() {
        // Test #1 — initialise from defaults.
        let rc = RateControl::new(RateControlConfig::default());
        assert_eq!(rc.frames_to_key, 0, "first frame must be KF");
        assert_eq!(rc.frames_since_key, 0);
        assert_eq!(rc.q_1_frame, 0);
        assert_eq!(rc.q_2_frame, 0);
        assert_eq!(rc.rc_1_frame, 0);
        assert_eq!(rc.rc_2_frame, 0);
        // Buffer starts at starting_buffer_ms.
        assert_eq!(rc.buffer_level, rc.starting_buffer_level);
        // Buffer level == bits_off_target post-init.
        assert_eq!(rc.buffer_level, rc.bits_off_target);
        // Optimal < max.
        assert!(rc.optimal_buffer_level <= rc.maximum_buffer_size);
        // Sanity on derived bandwidths.
        assert!(rc.avg_frame_bandwidth > 0);
        assert!(rc.max_frame_bandwidth >= rc.avg_frame_bandwidth);
    }

    #[test]
    fn svc_buffer_credit_and_debit_are_separate() {
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.bits_off_target = 0;
        rc.buffer_level = 0;
        let avg = rc.avg_frame_bandwidth;

        rc.svc_preencode_buffer_update();
        assert_eq!(rc.bits_off_target, i64::from(avg));
        rc.svc_postencode_buffer_debit(avg / 4);
        assert_eq!(rc.bits_off_target, i64::from(avg - avg / 4));
        assert_eq!(rc.buffer_level, rc.bits_off_target);
    }

    #[test]
    fn svc_large_keyframe_raises_inter_q_ambient() {
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.avg_frame_bandwidth = 1_000;
        rc.avg_frame_qindex_inter = 100;

        assert_eq!(rc.svc_adjust_inter_ambient_after_keyframe(3_000, 80), None);
        assert_eq!(rc.avg_frame_qindex_inter, 100);

        let corrected = ((80 + rc.worst_quality) >> 1) as u8;
        assert_eq!(
            rc.svc_adjust_inter_ambient_after_keyframe(3_001, 80),
            Some(corrected)
        );
        assert_eq!(rc.avg_frame_qindex_inter, corrected);
    }

    #[test]
    fn one_pass_vbr_uses_libvpx_default_max_section() {
        use super::super::RcMode;
        let config = RateControlConfig {
            rc_mode: RcMode::Vbr,
            target_bitrate_bps: 6_000_000,
            framerate: 30.0,
            ..RateControlConfig::default()
        };
        let rc = RateControl::new(config);
        assert_eq!(rc.avg_frame_bandwidth, 200_000);
        assert_eq!(rc.max_frame_bandwidth, 4_000_000);
    }

    #[test]
    fn low_motion_ewma_uses_libvpx_rounding() {
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.update_frame_low_motion(77);
        assert_eq!(rc.avg_frame_low_motion, 19);
        rc.update_frame_low_motion(58);
        assert_eq!(rc.avg_frame_low_motion, 28);
    }

    #[test]
    fn rate_control_avg_frame_bandwidth() {
        // Test #2 — 1 Mbps / 30 fps → ~33,333 bits/frame.
        let rc = RateControl::new(RateControlConfig::default());
        let expected = 1_000_000_u32 / 30;
        let actual = rc.avg_frame_bandwidth;
        assert!(
            actual.abs_diff(expected) <= 1,
            "avg_frame_bandwidth = {actual}, expected ~{expected}"
        );
    }

    #[test]
    fn qindex_history_init_matches_libvpx_cbr_branch() {
        use super::super::RcMode;
        let cfg = RateControlConfig {
            rc_mode: RcMode::Cbr,
            min_qindex: 4,
            max_qindex: 255,
            ..RateControlConfig::default()
        };
        let rc = RateControl::new(cfg);
        assert_eq!(rc.last_q_keyframe, 4);
        assert_eq!(rc.last_q_inter, 255);
        assert_eq!(rc.avg_frame_qindex_key, 255);
        assert_eq!(rc.avg_frame_qindex_inter, 255);
    }

    #[test]
    fn qindex_history_init_matches_libvpx_non_cbr_branch() {
        use super::super::RcMode;
        let cfg = RateControlConfig {
            rc_mode: RcMode::Vbr,
            min_qindex: 4,
            max_qindex: 255,
            ..RateControlConfig::default()
        };
        let rc = RateControl::new(cfg);
        let midpoint = ((u16::from(cfg.max_qindex) + u16::from(cfg.min_qindex)) / 2) as u8;
        assert_eq!(rc.last_q_keyframe, 4);
        assert_eq!(rc.last_q_inter, 255);
        assert_eq!(rc.avg_frame_qindex_key, midpoint);
        assert_eq!(rc.avg_frame_qindex_inter, midpoint);
    }

    #[test]
    fn ms_to_bits_for_default_config() {
        let cfg = RateControlConfig::default();
        // 6000 ms at 1 Mbps == 6,000,000 bits.
        assert_eq!(
            ms_to_bits(cfg.buffer_size_ms, cfg.target_bitrate_bps),
            6_000_000
        );
        // 5000 ms at 1 Mbps == 5,000,000 bits.
        assert_eq!(
            ms_to_bits(cfg.optimal_buffer_ms, cfg.target_bitrate_bps),
            5_000_000
        );
    }

    #[test]
    fn advance_frame_at_zero_does_not_underflow() {
        // Covers the `else` arm in `advance_frame` (frames_to_key == 0
        // path). Without the guard a saturating subtraction would still
        // be needed; the explicit branch leaves frames_to_key at 0.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.frames_to_key = 0;
        rc.frames_since_key = 5;
        rc.advance_frame();
        assert_eq!(rc.frames_to_key, 0, "frames_to_key must stay at zero");
        assert_eq!(rc.frames_since_key, 6, "frames_since_key must increment");
    }

    #[test]
    fn advance_frame_saturates_frames_since_key() {
        // Covers the saturating_add path in `advance_frame`.
        let mut rc = RateControl::new(RateControlConfig::default());
        rc.frames_to_key = 1;
        rc.frames_since_key = u32::MAX;
        rc.advance_frame();
        assert_eq!(
            rc.frames_since_key,
            u32::MAX,
            "frames_since_key must saturate at u32::MAX",
        );
        assert_eq!(rc.frames_to_key, 0, "frames_to_key must decrement");
    }

    #[test]
    fn avg_frame_bw_clamps_huge_bitrate_to_u32_max() {
        // Covers the `rounded > f64::from(u32::MAX)` arm in
        // `avg_frame_bw`. Use a tiny framerate so target/framerate
        // exceeds u32::MAX.
        let cfg = RateControlConfig {
            target_bitrate_bps: u32::MAX,
            framerate: 0.01, // forces avg_frame_bw = u32::MAX / 0.01 → > u32::MAX
            ..RateControlConfig::default()
        };
        let got = avg_frame_bw(&cfg);
        assert_eq!(got, u32::MAX, "avg_frame_bw must clamp at u32::MAX");
    }

    #[test]
    fn predicted_bits_at_picked_q_returns_positive_estimate() {
        // Exercises `RateControl::predicted_bits_at_picked_q` (diagnostic
        // accessor for the A10 instrumentation). Run a pick so
        // last_picked_qindex is populated; the predicted bits must be a
        // sane positive count.
        let mut rc = RateControl::new(RateControlConfig::default());
        let _ = rc.pick_q_and_bounds(FrameKind::Key);
        let predicted_key = rc.predicted_bits_at_picked_q(FrameKind::Key);
        assert!(
            predicted_key > 0,
            "predicted bits must be positive after a pick: got {predicted_key}",
        );
        let predicted_inter = rc.predicted_bits_at_picked_q(FrameKind::Inter);
        assert!(
            predicted_inter > 0,
            "predicted inter bits must be positive: got {predicted_inter}",
        );
    }

    #[test]
    fn e8_trace_snapshot_carries_buffer_state() {
        // Exercises `e8_trace_snapshot` (diagnostic E8 accessor). The
        // returned snapshot must match the underlying rc state.
        let mut rc = RateControl::new(RateControlConfig::default());
        let _ = rc.pick_q_and_bounds(FrameKind::Inter);
        let snap = rc.e8_trace_snapshot();
        assert_eq!(snap.buffer_level, rc.buffer_level);
        assert_eq!(snap.optimal_buffer_level, rc.optimal_buffer_level);
        assert_eq!(snap.critical_level, rc.optimal_buffer_level >> 3);
        assert_eq!(snap.maximum_buffer_size, rc.maximum_buffer_size);
        assert_eq!(snap.ambient_qp_inter, i32::from(rc.avg_frame_qindex_inter));
        assert_eq!(snap.active_worst, i32::from(rc.active_max_qindex));
        assert_eq!(snap.active_best, i32::from(rc.active_min_qindex));
        assert_eq!(snap.picked_q, i32::from(rc.last_picked_qindex));
        assert_eq!(snap.this_frame_target, rc.this_frame_target);
    }

    #[test]
    fn config_accessor_round_trips() {
        // The public `config` getter returns a borrow of the original
        // config snapshot. Verify the snapshot is byte-equal on a
        // representative non-default config.
        let cfg = RateControlConfig {
            target_bitrate_bps: 2_500_000,
            framerate: 24.0,
            ..RateControlConfig::default()
        };
        let rc = RateControl::new(cfg);
        let got = rc.config();
        assert_eq!(got.target_bitrate_bps, 2_500_000);
        assert!((got.framerate - 24.0).abs() < f64::EPSILON);
    }

    /// W1.6 PROOF — CBR is the default and works end-to-end through
    /// `pick_q_and_bounds`. Anti-G3: if a regression rewires the
    /// dispatcher to always panic, this fails immediately.
    #[test]
    fn w1_6_pick_q_and_bounds_cbr_path_works() {
        use super::super::RcMode;
        let cfg = RateControlConfig::default();
        assert_eq!(cfg.rc_mode, RcMode::Cbr, "Default must be CBR");
        let mut rc = RateControl::new(cfg);
        let (q, lo, hi) = rc.pick_q_and_bounds(FrameKind::Key);
        assert!(q >= lo && q <= hi, "picked q={q} must lie in [{lo},{hi}]");
    }

    /// W_VBR PROOF — VBR dispatch arm reaches the one-pass VBR
    /// implementation and returns a valid `(q, lo, hi)` triple. Replaces
    /// the previous `#[should_panic]` guard now that VBR is ported
    /// (libvpx `vp9_ratectrl.c:1138-1303`
    /// `rc_pick_q_and_bounds_one_pass_vbr`).
    #[test]
    fn w_vbr_dispatch_returns_valid_triple() {
        use super::super::RcMode;
        let cfg = RateControlConfig {
            rc_mode: RcMode::Vbr,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        // First frame is a KF in our model (frames_to_key == 0,
        // total_actual_bits == 0).
        let (q, lo, hi) = rc.pick_q_and_bounds(FrameKind::Key);
        assert!(lo <= q && q <= hi, "VBR pick must satisfy lo<=q<=hi");
        assert!(
            i32::from(lo) >= rc.best_quality && i32::from(hi) <= rc.worst_quality,
            "VBR bounds must lie within [best_quality, worst_quality]"
        );
    }

    /// PHASE-3 PROOF — Constrained-Quality mode clamps the picked Q
    /// to be ≥ cq_level (libvpx CQ semantics: cq_level is a quality
    /// FLOOR). The CBR regulator may legitimately pick a low Q when
    /// the buffer is full; CQ raises it. PROOF: with cq_level=100,
    /// the picked Q on a fresh CBR-style controller is ≥ 100.
    #[test]
    fn cq_mode_clamps_q_to_floor() {
        use super::super::RcMode;
        let cfg = RateControlConfig {
            rc_mode: RcMode::Cq,
            cq_level: 100,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        let (q, lo, hi) = rc.pick_q_and_bounds(FrameKind::Inter);
        assert!(
            q >= 100,
            "CQ PROOF: q={q} must be >= cq_level (100); got (q,lo,hi)=({q},{lo},{hi})"
        );
        assert!(lo >= 100, "CQ PROOF: lo={lo} must be >= cq_level (100)");
        assert!(hi >= 100, "CQ PROOF: hi={hi} must be >= cq_level (100)");
    }

    /// PHASE-3 PROOF — Constant-Quantizer mode pins Q at cq_level on
    /// inter frames; KF gets a libvpx-style small qdelta (≈ -8).
    /// libvpx VPX_Q short-circuit at `vp9_ratectrl.c`.
    #[test]
    fn q_mode_pins_qindex_at_cq_level() {
        use super::super::RcMode;
        let cfg = RateControlConfig {
            rc_mode: RcMode::Q,
            cq_level: 100,
            ..RateControlConfig::default()
        };
        let mut rc = RateControl::new(cfg);
        let (q_inter, lo_i, hi_i) = rc.pick_q_and_bounds(FrameKind::Inter);
        assert_eq!(q_inter, 100, "Q PROOF: inter q must be cq_level");
        assert_eq!(lo_i, 100);
        assert_eq!(hi_i, 100);
        let (q_kf, lo_k, hi_k) = rc.pick_q_and_bounds(FrameKind::Key);
        assert_eq!(
            q_kf, 92,
            "Q PROOF: KF q must be cq_level - 8 (libvpx KF qdelta)"
        );
        assert_eq!(lo_k, 92);
        assert_eq!(hi_k, 92);
    }

    /// W_VBR PROOF — VBR is now a ported arm (`super::vbr::
    /// pick_q_and_bounds`). This test guards that the dispatcher
    /// actually routes VBR through the VBR implementation, not the CBR
    /// or Q paths, by checking the active_worst VBR ambient-driven
    /// model produces a different `(active_min, active_max)` for VBR
    /// than CBR's buffer-driven model at the SAME starting state.
    /// libvpx `vp9_ratectrl.c:1587-1609` dispatch / `:905-928` VBR
    /// active_worst.
    #[test]
    fn vbr_dispatch_differs_from_cbr_at_first_frame() {
        use super::super::RcMode;
        // CBR baseline.
        let cfg_cbr = RateControlConfig {
            rc_mode: RcMode::Cbr,
            ..Default::default()
        };
        let mut rc_cbr = RateControl::new(cfg_cbr);
        let _ = rc_cbr.pick_q_and_bounds(FrameKind::Key);

        // VBR side-by-side.
        let cfg_vbr = RateControlConfig {
            rc_mode: RcMode::Vbr,
            ..RateControlConfig::default()
        };
        let mut rc_vbr = RateControl::new(cfg_vbr);
        let _ = rc_vbr.pick_q_and_bounds(FrameKind::Key);

        // The two rate models pick different per-frame targets on the
        // first KF (CBR: starting_buffer_level/2 ~= 2_000_000; VBR:
        // avg_bw * 25 ~= 833_333). That target difference + different
        // active_worst computation must produce some observable state
        // difference. Easiest oracle: `this_frame_target` differs.
        assert_ne!(
            rc_cbr.this_frame_target, rc_vbr.this_frame_target,
            "CBR and VBR must pick different per-frame KF targets \
             (CBR: starting_buf/2; VBR: avg * kf_ratio=25)"
        );
    }
}
