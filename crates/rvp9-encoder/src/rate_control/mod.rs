//! CBR (constant bit-rate) rate control.
//!
//! Libvpx-faithful port of the single-pass CBR paths in
//! `vp9/encoder/vp9_ratectrl.c`. The rate controller maintains a
//! leaky-bucket buffer model and picks a per-frame Q-index that targets
//! a configured average bitrate.
//!
//! ## Scope (F9.0 v1)
//!
//! What this module implements:
//!
//! - [`RateControlConfig`] — user-facing settings (target_bitrate,
//!   framerate, buffer geometry, GOP length).
//! - [`RateControl`] — running state (buffer_level, Q history,
//!   GOP counters) plus the three primary entry points
//!   ([`RateControl::pick_q_and_bounds`],
//!   [`RateControl::postencode_update`], [`RateControl::advance_frame`]).
//! - [`FrameKind`] — minimal Key / Inter discrimination for the
//!   public API (the encoder's wire-format `FrameKind` in
//!   `rvp9_core::bitstream` carries fields we don't need here).
//!
//! ## What's intentionally NOT here
//!
//! These libvpx behaviours are deferred to follow-up work and DOCUMENTED
//! AS DIVERGENCES (per HLD D8: every deliberate departure goes in
//! `notes/divergence-ledger.md`):
//!
//! - **Rate correction factor feedback loop.** libvpx maintains a
//!   per-rate-level `rate_correction_factors[]` table that is damped-
//!   updated after each frame in `vp9_rc_update_rate_correction_factors`
//!   (`vp9_ratectrl.c:712-817`). v1 uses a fixed `correction_factor =
//!   1.0` and relies on the buffer-driven active-worst-quality clamp to
//!   pull Q towards the right neighbourhood. The single-frame Q estimate
//!   may therefore over- or under-shoot by up to ~30 % at first; over a
//!   30-frame window the buffer feedback drives the average back to
//!   target (see [`tests::rate_control_simulation_30_frames`]).
//! - **Frame drop / decimation.** `vp9_rc_drop_frame` (`vp9_ratectrl.c:
//!   554-665`) and the `drop_count` / `decimation_*` machinery are
//!   stubbed (`drop_count` field carried for ABI shape only).
//! - **Aggressive overshoot rescue.** The CBR overshoot re-encode at
//!   `vp9_encodedframe_overshoot` is not ported.
//! - **GF/ARF boost.** `gf_cbr_boost_pct`, `baseline_gf_interval`, and
//!   the golden / alt-ref Q paths in
//!   `rc_pick_q_and_bounds_one_pass_cbr` (`vp9_ratectrl.c:1043-1068`)
//!   are skipped; v1 treats every non-keyframe as a normal inter.
//! - **Scene change detection** (`vp9_scene_detection_onepass`) and
//!   forced KF popping prevention (`this_key_frame_forced`).
//! - **HBD (high bit depth).** We mirror libvpx's 8-bit path only; the
//!   `vpx_bit_depth_t` parameter in `vp9_convert_qindex_to_q` is dropped
//!   (matches `rvp9_core::quant`'s deferred-HBD policy).
//!
//! ## Algorithm overview
//!
//! Per-frame Q selection (inter):
//!
//! 1. Compute `target_bits` from `avg_frame_bandwidth` adjusted for
//!    buffer fullness (libvpx `vp9_calc_pframe_target_size_one_pass_cbr`,
//!    `vp9_ratectrl.c:2145`).
//! 2. Compute `active_worst_quality` from buffer level (libvpx
//!    `calc_active_worst_quality_one_pass_cbr`,
//!    `vp9_ratectrl.c:931`). Above-optimal buffer pulls Q down; below-
//!    critical buffer pushes Q to worst_quality.
//! 3. Set `active_best_quality` from the recent inter-frame Q via the
//!    `rtc_minq` lookup. v1 uses a simplified linear approximation in
//!    [`cbr::rtc_minq_approx`].
//! 4. Run [`cbr::regulate_q`]: iterate from active_best upward; the
//!    first qindex whose `bits_per_mb` estimate falls below the
//!    target wins.
//!
//! Per-frame Q selection (keyframe):
//!
//! 1. Target bits = `((16 + kf_boost) * avg_frame_bandwidth) >> 4`
//!    (libvpx `vp9_calc_iframe_target_size_one_pass_cbr`,
//!    `vp9_ratectrl.c:2218`).
//! 2. Active best/worst use `best_quality`..`worst_quality` directly
//!    (libvpx `rc_pick_q_and_bounds_one_pass_cbr:1014`).
//! 3. Regulate as above.
//!
//! Post-encode update:
//!
//! 1. `bits_off_target += avg_frame_bandwidth - actual_bits`
//!    (= libvpx preencode `+ avg_frame_bandwidth`
//!    + postencode `- encoded_frame_size`).
//! 2. Clamp `bits_off_target` to `[-max_buffer, max_buffer]`.
//!    (libvpx clamps the upper bound at `maximum_buffer_size` only;
//!    we additionally clamp the lower bound so steady-state buffer
//!    underflow can't run away to `i64::MIN` in v1's no-drop world.
//!    This is a DOCUMENTED DIVERGENCE — see field docs.)
//! 3. `buffer_level = bits_off_target` (libvpx
//!    `update_buffer_level_postencode:328`).
//! 4. Update `last_q_keyframe` / `last_q_inter`.

pub mod cbr;
pub mod config;
pub mod postencode;
pub mod state;

pub use config::RateControlConfig;
pub use state::RateControl;

/// Frame-kind discriminant for the rate-control API.
///
/// Intentionally narrower than the wire-format
/// [`rvp9_core::bitstream::FrameKind`] — the rate controller only needs
/// to know "is this a keyframe" vs "is this an inter frame". `IntraOnly`
/// and `ShowExistingFrame` in the wire-format enum collapse here as
/// follows (mapping is the caller's job; the encoder framing layer will
/// do it):
///
/// - `ShowExistingFrame` → never reaches the rate controller (no bits
///   emitted by the encoder itself for the shown frame).
/// - `KeyFrame` → [`FrameKind::Key`].
/// - `IntraOnly` → [`FrameKind::Key`] (it's a fresh I-frame for RC purposes).
/// - `InterFrame` → [`FrameKind::Inter`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// Key frame (intra-only, GOP boundary). Higher target bits, lower Q.
    Key,
    /// Inter frame. Normal CBR target.
    Inter,
}

impl FrameKind {
    /// Is this a keyframe? Matches libvpx `frame_is_intra_only`.
    #[inline]
    #[must_use]
    pub const fn is_key(self) -> bool {
        matches!(self, Self::Key)
    }
}
