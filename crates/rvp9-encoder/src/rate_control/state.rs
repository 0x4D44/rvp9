//! `RateControl` — running rate-control state.
//!
//! Mirrors the `RATE_CONTROL` struct in `vp9/encoder/vp9_ratectrl.h:88`
//! for the field set actually touched by the 1-pass CBR path. Field
//! names are kept identical to libvpx wherever possible to ease
//! cross-referencing; Rust idioms (i64 → `i64`, no explicit `*const`
//! pointers, no SVC arrays) replace the C-isms.

use super::{FrameKind, RateControlConfig, cbr, postencode};

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
    /// `rc->max_frame_bandwidth` (`vp9_ratectrl.h:122`). Set to
    /// `avg_frame_bandwidth * max_burst_factor` where the factor
    /// is 4 (matches libvpx `set_frame_max_size` for CBR).
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

    // ---- GOP state ----
    /// Frames until the next keyframe. libvpx `rc->frames_to_key`
    /// (`vp9_ratectrl.h:112`). Decremented in `advance_frame`; reset
    /// to `gop_length - 1` whenever a keyframe is encoded.
    pub frames_to_key: u32,

    /// Frames since the last keyframe. libvpx `rc->frames_since_key`
    /// (`vp9_ratectrl.h:113`).
    pub frames_since_key: u32,

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
}

impl RateControl {
    /// Initialize from `config`. Mirrors libvpx `vp9_rc_init`
    /// (`vp9_ratectrl.c:360`) for the `pass == 0 && rc_mode == VPX_CBR`
    /// branch.
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
        // libvpx sets max via VP9E_SET_MAX_INTER_BITRATE_PCT (default
        // 0 == no clamp); we cap at 4x avg as a sane default.
        let max_frame_bandwidth = avg_frame_bandwidth.saturating_mul(4);

        let maximum_buffer_size = ms_to_bits(config.buffer_size_ms, config.target_bitrate_bps);
        let optimal_buffer_level = ms_to_bits(config.optimal_buffer_ms, config.target_bitrate_bps);
        let starting_buffer_level =
            ms_to_bits(config.starting_buffer_ms, config.target_bitrate_bps);

        Self {
            buffer_level: starting_buffer_level,
            bits_off_target: starting_buffer_level,
            avg_frame_bandwidth,
            min_frame_bandwidth,
            max_frame_bandwidth,
            this_frame_target: avg_frame_bandwidth,

            // libvpx `vp9_rc_init` for CBR: avg_frame_qindex initialized
            // to worst_allowed_q for both frame types
            // (`vp9_ratectrl.c:364-365`).
            last_q_keyframe: config.max_qindex,
            last_q_inter: config.max_qindex,
            avg_frame_qindex_key: config.max_qindex,
            avg_frame_qindex_inter: config.max_qindex,

            active_min_qindex: config.min_qindex,
            active_max_qindex: config.max_qindex,

            frames_to_key: 0, // first frame must be a keyframe
            frames_since_key: 0,
            kf_boost: config.kf_boost,

            total_actual_bits: 0,
            total_target_bits: 0,

            drop_count: 0,

            mbs: config.mbs(),
            worst_quality: i32::from(config.max_qindex),
            best_quality: i32::from(config.min_qindex),
            maximum_buffer_size,
            optimal_buffer_level,
            starting_buffer_level,

            config,
        }
    }

    /// Pick the Q-index for the next frame.
    ///
    /// Returns `(active_qindex, q_low_bound, q_high_bound)`. The caller
    /// is expected to use `active_qindex` as the starting point; the
    /// bounds describe the per-frame range the encoder may dither
    /// within (e.g. segment-level Q deltas, AQ adjustments).
    ///
    /// Wraps [`super::cbr::pick_q_and_bounds`].
    pub fn pick_q_and_bounds(&mut self, frame_kind: FrameKind) -> (u8, u8, u8) {
        cbr::pick_q_and_bounds(self, frame_kind)
    }

    /// Update internal state after the frame has been encoded.
    ///
    /// `actual_bits` is the exact bit count of the emitted frame
    /// (compressed-header + tile data + frame overhead). Wraps
    /// [`super::postencode::postencode_update`].
    pub fn postencode_update(&mut self, actual_bits: u32, frame_kind: FrameKind) {
        postencode::postencode_update(self, actual_bits, frame_kind);
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
    }

    /// Read-only access to the snapshot config.
    #[must_use]
    pub fn config(&self) -> &RateControlConfig {
        &self.config
    }
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
    fn ms_to_bits_for_default_config() {
        let cfg = RateControlConfig::default();
        // 6000 ms at 1 Mbps == 6,000,000 bits.
        assert_eq!(ms_to_bits(cfg.buffer_size_ms, cfg.target_bitrate_bps), 6_000_000);
        // 5000 ms at 1 Mbps == 5,000,000 bits.
        assert_eq!(ms_to_bits(cfg.optimal_buffer_ms, cfg.target_bitrate_bps), 5_000_000);
    }
}
