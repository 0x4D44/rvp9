//! GOP topology — libvpx-faithful golden-frame group schedule.
//!
//! This module introduces [`GfGroup`], the single source of truth for
//! "what TYPE is each frame in the current golden-frame group" (keyframe
//! / golden / alt-ref / overlay / normal-inter), the per-frame alt-ref
//! source offsets, and the GF interval. It is modelled on libvpx's
//! `GF_GROUP` struct so the TPL (A23) and 2-pass VBR consumers — which
//! both index `gf_group.update_type[...]`, `gf_group.arf_src_offset[...]`
//! and `gf_group.gf_group_size` — fit without reshaping.
//!
//! # Phase-I scope (1-pass subset)
//!
//! [`GfGroup`] is built from the encoder's CURRENT 1-pass schedule
//! (`gop_length` + `enable_alt_ref`) and reproduces it EXACTLY, so wiring
//! it into the frame loop is a **gate-safe, byte-identical refactor**.
//! The encoder's real 1-pass RT path runs two independent cadences which
//! libvpx's 2-pass `define_gf_group_structure` would merge:
//!
//! - **GOLDEN refresh**: every [`GOLDEN_REFRESH_PERIOD`] *inter* frames
//!   since the last keyframe / golden refresh, the frame additionally
//!   refreshes the GOLDEN slot (wire `refresh_frame_flags = 0x03`).
//!   Mirrors `cpi->refresh_golden_frame` (`vp9_firstpass.c::define_gf_group`).
//! - **ALT-REF**: when `enable_alt_ref`, every `gf_interval = max(gop/2, 2)`
//!   inter frames a non-show ARF is built (wire `refresh_frame_flags =
//!   0x04`) and superframe-prepended ahead of the show frame.
//!
//! Both cadences are deterministic functions of the frame's GOP-relative
//! offset, so the whole group is precomputed at the keyframe boundary.
//!
//! The 2-pass-only fields of libvpx's `GF_GROUP` (`rf_level`,
//! `layer_depth`, `bit_allocation`, `gfu_boost`, `arf_index_stack`,
//! `top_arf_idx`, `group_noise_energy`) are intentionally NOT modelled
//! here — they require first-pass statistics that the 1-pass path does
//! not produce. They are a documented follow-up for the 2-pass VBR wave.
//!
//! # libvpx file:line citations
//!
//! - `vp9/encoder/vp9_firstpass.h:62-71` — `FRAME_UPDATE_TYPE` enum
//!   (`KF_UPDATE = 0`, `LF_UPDATE = 1`, `GF_UPDATE = 2`, `ARF_UPDATE = 3`,
//!   `OVERLAY_UPDATE = 4`, `MID_OVERLAY_UPDATE = 5`, `USE_BUF_FRAME = 6`).
//! - `vp9/encoder/vp9_firstpass.h:80-100` — `GF_GROUP` struct
//!   (`index`, `update_type[]`, `arf_src_offset[]`, `frame_gop_index[]`,
//!   `gf_group_size`).
//! - `vp9/encoder/vp9_ratectrl.h:46` — `MAX_STATIC_GF_GROUP_LENGTH = 250`.
//! - `vp9/encoder/vp9_firstpass.c:2257-2302` — `define_gf_group_structure`
//!   (the per-frame `update_type` / `arf_src_offset` / `frame_gop_index`
//!   assignment this constructor mirrors for the 1-pass subset).
//! - `vp9/encoder/vp9_firstpass.c:2242-2255` — `set_gf_overlay_frame_type`
//!   (GF vs OVERLAY disposition at the group's first / last index).
//! - `vp9/encoder/vp9_firstpass.c:3426-3428` — KF handler
//!   (`update_type[0] = KF_UPDATE`).

/// libvpx `FRAME_UPDATE_TYPE` (`vp9_firstpass.h:62-71`). The discriminant
/// values match libvpx exactly so a consumer comparing against a stored
/// `gf_group.update_type[i]` reads the same integers.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrameUpdateType {
    /// Keyframe — refreshes every reference slot (`refresh_frame_flags =
    /// 0xFF`). libvpx `KF_UPDATE = 0`.
    KfUpdate = 0,
    /// Normal inter frame ("low-frequency" update) — refreshes LAST only
    /// (`refresh_frame_flags = 0x01`). libvpx `LF_UPDATE = 1`.
    LfUpdate = 1,
    /// Golden-frame update — a show inter frame that also refreshes the
    /// GOLDEN slot (`refresh_frame_flags = 0x03`). libvpx `GF_UPDATE = 2`.
    GfUpdate = 2,
    /// Alt-ref update — a non-show frame that refreshes the ALTREF slot
    /// (`refresh_frame_flags = 0x04`). libvpx `ARF_UPDATE = 3`.
    ArfUpdate = 3,
    /// Overlay update — the visible frame that "uses up" a prior ARF.
    /// Not emitted by the current 1-pass schedule (no show-existing /
    /// overlay frames); present for libvpx-faithful enum parity and the
    /// 2-pass consumer. libvpx `OVERLAY_UPDATE = 4`.
    OverlayUpdate = 4,
}

/// GOLDEN-slot refresh period for the inter path, in inter frames since
/// the last keyframe / golden refresh. Mirrors the encoder's prior
/// `A14_GOLDEN_REFRESH_PERIOD` constant (`frame_loop.rs`) and libvpx's
/// `cpi->refresh_golden_frame` cadence
/// (`vp9_firstpass.c::define_gf_group`, typically the 8-15 frame range;
/// pinned to 8 for the legacy CBR schedule). One-pass VBR supplies its
/// frame-rate-derived interval through [`GfGroup::new_with_golden_refresh_period`].
pub const GOLDEN_REFRESH_PERIOD: u32 = 8;

/// libvpx `MAX_STATIC_GF_GROUP_LENGTH` (`vp9_ratectrl.h:46`). The
/// schedule arrays in libvpx are sized `MAX_STATIC_GF_GROUP_LENGTH + 2`;
/// we cap GOP-relative lookups defensively at the same bound.
pub const MAX_STATIC_GF_GROUP_LENGTH: u32 = 250;

/// A single show-frame's place in the golden-frame group.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FrameSchedule {
    /// libvpx `gf_group.update_type[i]` — the frame's update disposition.
    pub update_type: FrameUpdateType,
    /// Wire-format `refresh_frame_flags` mask this frame emits. Derived
    /// 1:1 from `update_type`: KF → `0xFF`, GF → `0x03`, LF → `0x01`.
    /// (The ARF's `0x04` is carried by `ArfSchedule`, not here, since
    /// the ARF is a separate non-show frame in the superframe.)
    pub refresh_frame_flags: u8,
    /// libvpx `gf_group.frame_gop_index[i]` — the frame's offset from the
    /// start of the GOP (0 == keyframe).
    pub frame_gop_index: u32,
    /// `true` if a non-show ARF is emitted *immediately before* this show
    /// frame (superframe-prepended). When set, [`Self::arf_src_offset`]
    /// is meaningful.
    pub arf_before: bool,
    /// libvpx `gf_group.arf_src_offset[i]` — display-order distance from
    /// this frame's coded position to the ARF's source frame. `0` when no
    /// ARF precedes this frame.
    pub arf_src_offset: u8,
}

/// The golden-frame group: a per-GOP frame-type schedule indexed by
/// GOP-relative offset. Modelled on libvpx's `GF_GROUP`
/// (`vp9_firstpass.h:80-100`) — the 1-pass subset that reproduces the
/// encoder's current schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GfGroup {
    /// GOP length in display frames (the keyframe cadence: KF at GOP
    /// offsets `0, gop_length, 2*gop_length, ...`).
    gop_length: u32,
    /// `gf_interval = max(gop_length / 2, 2)` — the ARF cadence (inter
    /// frames between alt-ref frames). Mirrors libvpx's default
    /// `baseline_gf_interval` (`vp9_firstpass.c::define_gf_group`). Only
    /// consulted when `enable_alt_ref`.
    gf_interval: u32,
    /// Show-frame offset cadence for one-pass GOLDEN refreshes.
    golden_refresh_period: u32,
    /// `true` iff the encoder is configured for ALTREF emission.
    enable_alt_ref: bool,
}

impl GfGroup {
    /// Build the golden-frame group from the 1-pass schedule inputs.
    ///
    /// This is the `define_gf_group_structure`-equivalent for the 1-pass
    /// path (`vp9_firstpass.c:2257-2302`): it captures `gop_length` and
    /// `enable_alt_ref` and computes `gf_interval`. The per-frame
    /// disposition is then resolved on demand by [`Self::schedule`] (a
    /// pure function of the GOP-relative offset), which is equivalent to
    /// libvpx pre-filling the `update_type[]` / `arf_src_offset[]` arrays
    /// but avoids materialising a fixed-length array for unbounded GOPs.
    ///
    /// `gop_length` is clamped to `>= 1` (matching the frame loop's
    /// `self.config.gop_length.max(1)`).
    #[must_use]
    pub fn new(gop_length: u32, enable_alt_ref: bool) -> Self {
        Self::new_with_golden_refresh_period(gop_length, enable_alt_ref, GOLDEN_REFRESH_PERIOD)
    }

    /// Build a group with a rate-control-selected GOLDEN cadence.
    #[must_use]
    pub fn new_with_golden_refresh_period(
        gop_length: u32,
        enable_alt_ref: bool,
        golden_refresh_period: u32,
    ) -> Self {
        let gop_length = gop_length.max(1);
        // libvpx default `baseline_gf_interval` for the half-GOP ARF
        // cadence; `.max(2)` floors it so a 2-frame filter is viable.
        let gf_interval = (gop_length / 2).max(2);
        Self {
            gop_length,
            gf_interval,
            golden_refresh_period: golden_refresh_period.max(1),
            enable_alt_ref,
        }
    }

    /// GOP length in display frames (keyframe cadence).
    #[must_use]
    pub const fn gop_length(&self) -> u32 {
        self.gop_length
    }

    /// GF interval (`baseline_gf_interval`) — the ARF cadence in inter
    /// frames. Exposed for the TPL / 2-pass consumers.
    #[must_use]
    pub const fn gf_interval(&self) -> u32 {
        self.gf_interval
    }

    /// `true` iff alt-ref emission is enabled for this group.
    #[must_use]
    pub const fn enable_alt_ref(&self) -> bool {
        self.enable_alt_ref
    }

    /// libvpx `gf_group.gf_group_size` — the number of frames in the
    /// group. For the 1-pass keyframe cadence this is `gop_length` (KF at
    /// offset 0, then `gop_length - 1` inter frames before the next KF).
    #[must_use]
    pub const fn gf_group_size(&self) -> u32 {
        self.gop_length
    }

    /// GOP-relative offset of a display-order `frame_idx` (0 == this
    /// frame is the GOP's keyframe). The keyframe cadence is `frame_idx %
    /// gop_length`, matching the frame loop's
    /// `self.frame_idx.is_multiple_of(gop)` keyframe decision.
    #[must_use]
    pub const fn gop_offset(&self, frame_idx: u32) -> u32 {
        frame_idx % self.gop_length
    }

    /// Resolve the [`FrameSchedule`] for a display-order `frame_idx`.
    ///
    /// This is the single query the frame loop consults to decide the
    /// frame's type, refresh flags, and whether an ARF precedes it. It
    /// reproduces the encoder's CURRENT schedule exactly:
    ///
    /// - offset `0` → [`FrameUpdateType::KfUpdate`], `refresh = 0xFF`.
    /// - inter offset where `offset % golden_refresh_period == 0` →
    ///   [`FrameUpdateType::GfUpdate`], `refresh = 0x03`.
    /// - any other inter offset → [`FrameUpdateType::LfUpdate`],
    ///   `refresh = 0x01`.
    ///
    /// ARF disposition (`arf_before` / `arf_src_offset`) is set when
    /// `enable_alt_ref` and the frame is at the ARF cadence boundary
    /// (`offset % gf_interval == 0`, inter only).
    #[must_use]
    pub fn schedule(&self, frame_idx: u32) -> FrameSchedule {
        let offset = self.gop_offset(frame_idx);

        if offset == 0 {
            // Keyframe — refreshes every slot. libvpx
            // `vp9_firstpass.c:3426` `update_type[0] = KF_UPDATE`.
            return FrameSchedule {
                update_type: FrameUpdateType::KfUpdate,
                refresh_frame_flags: 0xFF,
                frame_gop_index: 0,
                arf_before: false,
                arf_src_offset: 0,
            };
        }

        // Inter frame. GOLDEN refresh fires every configured period
        // inter frames since the keyframe; because the keyframe sits at
        // offset 0 and resets the counter, the refresh lands at GOP
        // offsets that are exact multiples of that period.
        // This reproduces the frame loop's `frames_since_golden_refresh`
        // state machine (increment-then-check, reset on refresh): the
        // counter equals `offset % golden_refresh_period` after each
        // increment, hitting the period exactly at those offsets.
        let golden_refresh = offset.is_multiple_of(self.golden_refresh_period);

        // ARF cadence: when enabled, a non-show ARF is built once the
        // count of inter frames already coded since the keyframe reaches
        // `gf_interval`. The frame loop's `inter_frames_since_arf`
        // counter is reset to 0 at the KF, the trigger checks `counter >=
        // gf_interval` BEFORE the current inter frame is counted, and the
        // counter is incremented AFTER each inter frame. So by the time
        // the counter equals `gf_interval`, `gf_interval` inter frames
        // (GOP offsets `1..=gf_interval`) have been coded and the ARF
        // fires on the NEXT inter frame — GOP offset `gf_interval + 1`.
        // With `gf_interval = gop/2` at most one ARF fires per GOP (the
        // second would land beyond the next keyframe). The ARF source is
        // the just-popped frame (`alt_ref_index = 0` in the temporal
        // filter), so the display-order source offset is 0 in the
        // current minimal-filter wiring.
        let arf_before = self.enable_alt_ref && offset == self.gf_interval + 1;

        let (update_type, refresh_frame_flags) = if golden_refresh {
            (FrameUpdateType::GfUpdate, 0x03)
        } else {
            (FrameUpdateType::LfUpdate, 0x01)
        };

        FrameSchedule {
            update_type,
            refresh_frame_flags,
            frame_gop_index: offset,
            arf_before,
            arf_src_offset: 0,
        }
    }

    /// libvpx `gf_group.update_type[i]` accessor for a display-order
    /// `frame_idx`. Thin wrapper over [`Self::schedule`] for consumers
    /// that only need the update disposition.
    #[must_use]
    pub fn update_type(&self, frame_idx: u32) -> FrameUpdateType {
        self.schedule(frame_idx).update_type
    }

    /// libvpx `gf_group.arf_src_offset[i]` accessor. `0` when no ARF
    /// precedes the frame at `frame_idx`.
    #[must_use]
    pub fn arf_src_offset(&self, frame_idx: u32) -> u8 {
        self.schedule(frame_idx).arf_src_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_at_gop_boundaries() {
        let g = GfGroup::new(16, false);
        for &kf in &[0u32, 16, 32, 48] {
            let s = g.schedule(kf);
            assert_eq!(s.update_type, FrameUpdateType::KfUpdate);
            assert_eq!(s.refresh_frame_flags, 0xFF);
            assert_eq!(s.frame_gop_index, 0);
        }
    }

    #[test]
    fn golden_refresh_every_eight_inter_frames() {
        let g = GfGroup::new(16, false);
        // gop=16: KF at 0 and 16. Inter offsets 1..15; golden refresh at
        // offset 8 only (offset 16 is the next KF).
        for idx in 1..16u32 {
            let s = g.schedule(idx);
            if idx == 8 {
                assert_eq!(s.update_type, FrameUpdateType::GfUpdate, "f{idx}");
                assert_eq!(s.refresh_frame_flags, 0x03, "f{idx}");
            } else {
                assert_eq!(s.update_type, FrameUpdateType::LfUpdate, "f{idx}");
                assert_eq!(s.refresh_frame_flags, 0x01, "f{idx}");
            }
        }
    }

    #[test]
    fn golden_refresh_repeats_each_gop() {
        // gop=100: golden refresh at inter offsets 8,16,...,96.
        let g = GfGroup::new(100, false);
        for idx in 1..100u32 {
            let s = g.schedule(idx);
            if idx % GOLDEN_REFRESH_PERIOD == 0 {
                assert_eq!(s.refresh_frame_flags, 0x03, "f{idx}");
            } else {
                assert_eq!(s.refresh_frame_flags, 0x01, "f{idx}");
            }
        }
    }

    #[test]
    fn custom_golden_period_drives_vbr_schedule() {
        let g = GfGroup::new_with_golden_refresh_period(60, false, 11);
        for idx in 1..34 {
            assert_eq!(
                matches!(g.schedule(idx).update_type, FrameUpdateType::GfUpdate),
                idx % 11 == 0,
                "f{idx}"
            );
        }
    }

    #[test]
    fn alt_ref_disabled_never_schedules_arf() {
        let g = GfGroup::new(30, false);
        for idx in 0..60u32 {
            assert!(!g.schedule(idx).arf_before, "f{idx}");
        }
    }

    #[test]
    fn alt_ref_cadence_is_half_gop() {
        // gop=16 → gf_interval = max(8,2) = 8. The ARF fires once
        // `gf_interval` inter frames have been coded since the KF; the
        // trigger checks BEFORE counting the current frame, so the ARF
        // lands at GOP offset `gf_interval + 1 = 9` (matching the frame
        // loop's `inter_frames_since_arf >= arf_interval` semantics).
        let g = GfGroup::new(16, true);
        assert_eq!(g.gf_interval(), 8);
        for idx in 1..16u32 {
            let s = g.schedule(idx);
            assert_eq!(s.arf_before, idx == 9, "f{idx}");
        }
    }

    #[test]
    fn gf_interval_floor_is_two() {
        let g = GfGroup::new(2, true);
        assert_eq!(g.gf_interval(), 2);
        let g3 = GfGroup::new(3, true);
        assert_eq!(g3.gf_interval(), 2);
    }

    #[test]
    fn gop_length_clamped_to_one() {
        let g = GfGroup::new(0, false);
        assert_eq!(g.gop_length(), 1);
        assert_eq!(g.gf_group_size(), 1);
    }

    #[test]
    fn update_type_discriminants_match_libvpx() {
        assert_eq!(FrameUpdateType::KfUpdate as u8, 0);
        assert_eq!(FrameUpdateType::LfUpdate as u8, 1);
        assert_eq!(FrameUpdateType::GfUpdate as u8, 2);
        assert_eq!(FrameUpdateType::ArfUpdate as u8, 3);
        assert_eq!(FrameUpdateType::OverlayUpdate as u8, 4);
    }
}
