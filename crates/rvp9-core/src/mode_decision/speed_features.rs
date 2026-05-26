//! W0.3 — `SpeedFeatures` struct + `(speed, deadline, frame, content)`
//! dispatcher.
//!
//! libvpx splits the speed-feature setup across two large `switch`
//! functions:
//! `vp9_set_speed_features_framesize_independent`
//! (`vp9_speed_features.c:915-...`) and
//! `vp9_set_speed_features_framesize_dependent`
//! (`vp9_speed_features.c:869-913`). They run twice per frame and write
//! a giant `SPEED_FEATURES` struct that every downstream
//! mode-decision / partition / RDO path reads.
//!
//! Wave 0.3 lands the **dispatcher skeleton** for rvp9: a small
//! `SpeedFeatures` struct carrying the six flags the rvp9 encoder
//! actually consumes today (the §11 hardcodes at
//! `crates/rvp9-encoder/src/lib.rs:3689-3724`) plus the `Deadline` /
//! `ContentTier` enums. The dispatcher is pure — no `&mut cpi` — so it
//! is trivially testable per (speed, deadline, frame_size, content)
//! tuple.
//!
//! Wave 0.4 will wire this dispatcher per-frame and replace the
//! hardcoded literal at lib.rs:3713-3724.
//!
//! Subsequent waves grow the struct as additional libvpx SF rows are
//! ported (RDO trellis, denoiser thresholds, etc).

use crate::block::BlockSize;

/// vpxenc deadline class — speed-vs-quality preset (libvpx
/// `vpxenc.c:123-130`, mirroring
/// `VPX_DL_REALTIME` / `VPX_DL_GOOD_QUALITY` / `VPX_DL_BEST_QUALITY`).
///
/// Lives here (rather than rvp9-encoder) so the dispatcher can take it
/// as a parameter without forming a cycle — `rvp9-encoder` re-exports
/// this as `rvp9_encoder::Deadline` for back-compat.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Deadline {
    /// `--best` — best-quality deadline (libvpx `vpxenc.c:125-126`,
    /// `VPX_DL_BEST_QUALITY`).
    Best,
    /// `--good` — good-quality deadline (libvpx `vpxenc.c:127-128`,
    /// `VPX_DL_GOOD_QUALITY`). The vpxenc default.
    #[default]
    Good,
    /// `--rt` — real-time deadline (libvpx `vpxenc.c:129-130`,
    /// `VPX_DL_REALTIME`).
    Realtime,
}

/// `--tune-content` (libvpx `vpxenc.c:500-508`,
/// `VP9E_CONTENT_DEFAULT`/`SCREEN`/`FILM`).
///
/// The dispatcher consumes this in later waves (screen content gates
/// IntraBC-style picker shortcuts; film content suppresses sharpening
/// LF). Carried through W0.3 as the dispatcher parameter shape so
/// W0.4 doesn't need an API change.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum ContentTier {
    /// `default` — natural video (libvpx `VP9E_CONTENT_DEFAULT`).
    #[default]
    Default,
    /// `screen` — synthetic / screen-share content (libvpx
    /// `VP9E_CONTENT_SCREEN`).
    Screen,
    /// `film` — high-frame-rate / grainy film content (libvpx
    /// `VP9E_CONTENT_FILM`).
    Film,
}

/// Per-frame speed-feature decisions.
///
/// Mirrors the libvpx `SPEED_FEATURES` struct (`vp9_speed_features.h`)
/// at field granularity, but stops at the slice the rvp9 encoder
/// currently consumes (lib.rs:3713-3724). Later waves grow this to
/// match libvpx's full SF surface; each new field comes paired with
/// the libvpx `file:line` it was lifted from.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SpeedFeatures {
    /// libvpx `sf->limit_newmv_early_exit` (`vp9_speed_features.c:639,
    /// 790`). `1` at speed 5/6/7 CBR; `0` at speed 8.
    pub limit_newmv_early_exit: bool,
    /// libvpx `sf->bias_golden` (`vp9_speed_features.c:640`). `1` at
    /// speed 5+ CBR non-SVC.
    pub bias_golden: bool,
    /// libvpx `sf->cb_pred_filter_search` (`vp9_speed_features.c:658`).
    /// libvpx uses `int` (0/1/2); rvp9 currently consumes only the
    /// boolean "active" predicate (the `2` value enables the per-CB
    /// interp-filter early exit at speed 6 above 1280×720).
    pub cb_pred_filter_search: bool,
    /// libvpx `sf->disable_golden_ref` (`vp9_speed_features.c:811`).
    /// `1` only at speed 9 CBR. rvp9 op-points stop at 8, so this is
    /// `false` in W0.3 and ports forward when speed 9 lands.
    pub disable_golden_ref: bool,
    /// libvpx `sf->max_intra_bsize` (`vp9_speed_features.c:571`).
    /// `BLOCK_32X32` at speed 5+; `BLOCK_64X64` at the slower presets.
    pub max_intra_bsize: BlockSize,
    /// libvpx `sf->rt_intra_dc_only_low_content`
    /// (`vp9_speed_features.c:770`). `1` only at speed 8.
    pub rt_intra_dc_only_low_content: bool,
}

impl SpeedFeatures {
    /// Best-quality defaults — what libvpx's
    /// `vp9_set_speed_features_framesize_independent` writes before
    /// the speed-specific switch
    /// (`vp9_speed_features.c:924-988` "best quality defaults").
    ///
    /// `speed = 0` (good or best) lands on this configuration; the
    /// dispatcher overrides individual fields per (speed, deadline)
    /// switch arm.
    #[must_use]
    pub const fn best_quality_defaults() -> Self {
        Self {
            // All of the RT-path gates start "off" so the slow path
            // sees an unmodified `InterModeSearchParams`.
            limit_newmv_early_exit: false,
            bias_golden: false,
            cb_pred_filter_search: false,
            disable_golden_ref: false,
            // libvpx default at the best-quality preset (the speed-5+
            // RT path tightens to 32x32, see vp9_speed_features.c:571).
            max_intra_bsize: BlockSize::Block64x64,
            rt_intra_dc_only_low_content: false,
        }
    }
}

/// W0.3 — pure speed-features dispatcher.
///
/// Given the per-frame inputs the encoder already has (cpu-used,
/// deadline, coded resolution, content type), returns the
/// `SpeedFeatures` libvpx would write at the equivalent call to
/// `vp9_set_speed_features_framesize_{independent,dependent}`.
///
/// W0.3 covers the six fields rvp9 currently consumes. The argument
/// surface is the long-form libvpx tuple — adding new SF fields in
/// later waves does NOT require widening this signature.
///
/// libvpx file:line citations live on each `SpeedFeatures` field.
#[must_use]
pub fn dispatch(
    speed: i32,
    deadline: Deadline,
    width: u32,
    height: u32,
    content: ContentTier,
) -> SpeedFeatures {
    // libvpx's dispatcher branches on |speed|; negative cpu-used values
    // are reserved for tools (e.g. `-9` is "force best"; libvpx still
    // takes `abs(speed)` for the LUT lookups).
    let mag = speed.unsigned_abs();
    let mut sf = SpeedFeatures::best_quality_defaults();

    // Mode dispatch (libvpx `vp9_set_speed_features_framesize_independent`
    // top of file, `if (oxcf->mode == REALTIME)` switch at line 882-888).
    // We currently treat `Realtime` as the RT path and everything else
    // as Good — `Best` shares the Good preset until the RDO driver
    // (W4) brings a separate slow-path SF group online.
    match deadline {
        Deadline::Realtime => apply_rt(&mut sf, mag, width, height, content),
        Deadline::Good | Deadline::Best => apply_good(&mut sf, mag, width, height, content),
    }

    sf
}

/// libvpx `set_rt_speed_feature_framesize_{independent,dependent}`
/// (`vp9_speed_features.c:585-810` for the speed 5/6/7/8 RT arms).
///
/// Only the speed-specific overrides are listed; everything else
/// stays at [`SpeedFeatures::best_quality_defaults`].
fn apply_rt(
    sf: &mut SpeedFeatures,
    mag: u32,
    width: u32,
    height: u32,
    _content: ContentTier,
) {
    // libvpx `vp9_speed_features.c:571` — speed 5+ caps intra search
    // at 32x32 even on the RT path.
    if mag >= 5 {
        sf.max_intra_bsize = BlockSize::Block32x32;
    }

    // libvpx `vp9_speed_features.c:639, 640` — speed 5+ RT CBR turns
    // on the NEWMV early-exit + bias-golden refinements.
    if mag >= 5 {
        sf.limit_newmv_early_exit = true;
        sf.bias_golden = true;
    }

    // libvpx `vp9_speed_features.c:658` — speed 6 above 1280x720
    // upgrades cb_pred_filter_search to `2`. rvp9 carries it as a
    // boolean today; the resolution gate is preserved.
    if mag == 6 && width * height > 1280 * 720 {
        sf.cb_pred_filter_search = true;
    }

    // libvpx `vp9_speed_features.c:770` — speed 8 forces DC-only
    // intra on low-content blocks.
    if mag == 8 {
        sf.rt_intra_dc_only_low_content = true;
    }

    // libvpx `vp9_speed_features.c:790` — speed 8 turns OFF the
    // NEWMV early-exit refinement.
    if mag == 8 {
        sf.limit_newmv_early_exit = false;
    }

    // libvpx `vp9_speed_features.c:811` — speed 9 disables the
    // golden-ref slot entirely. rvp9 doesn't have a speed-9 op-point
    // today; the branch is here so the dispatcher behaves correctly
    // the moment one is added.
    if mag >= 9 {
        sf.disable_golden_ref = true;
    }
}

/// libvpx `set_good_speed_feature_framesize_{independent,dependent}`
/// (`vp9_speed_features.c:585-868` for the good-quality arms).
///
/// W0.3 covers only the slice of the Good path needed to differ from
/// the best-quality defaults at speed >= 5. Slow-path SF (speed 0..=4)
/// stays at defaults until Wave 4 brings the RDO driver online.
fn apply_good(
    sf: &mut SpeedFeatures,
    mag: u32,
    _width: u32,
    _height: u32,
    _content: ContentTier,
) {
    // libvpx `vp9_speed_features.c:571` — speed 5+ caps intra search
    // at 32x32 across both Good and RT paths.
    if mag >= 5 {
        sf.max_intra_bsize = BlockSize::Block32x32;
    }
    // No further Good-arm overrides land in W0.3. Wave 4 (RDO driver)
    // brings the speed 0..=4 good-quality switch arms with their full
    // libvpx citation set.
}

#[cfg(test)]
mod tests {
    //! PROOF tests for the W0.3 dispatcher.
    //!
    //! Each test pins a libvpx file:line gate so a regression that
    //! strips one of the W0.3 dispatcher arms is caught immediately.

    use super::*;

    /// Default dispatch — speed 0 / Good / SD natural content lands on
    /// the best-quality defaults (libvpx `vp9_speed_features.c:924-988`
    /// "best quality defaults").
    #[test]
    fn w0_3_speed0_good_default_is_best_quality_defaults() {
        let sf = dispatch(0, Deadline::Good, 640, 480, ContentTier::Default);
        assert_eq!(sf, SpeedFeatures::best_quality_defaults());
        assert_eq!(sf.max_intra_bsize, BlockSize::Block64x64);
        assert!(!sf.limit_newmv_early_exit);
        assert!(!sf.bias_golden);
    }

    /// Speed 7 / Realtime — the rvp9 §11 hardcodes at lib.rs:3713-3724.
    /// PROOF that the dispatcher reproduces them. libvpx citations:
    /// `vp9_speed_features.c:571, 639, 640`.
    #[test]
    fn w0_3_speed7_realtime_matches_rvp9_section_11_hardcodes() {
        let sf = dispatch(7, Deadline::Realtime, 1280, 720, ContentTier::Default);
        assert_eq!(sf.max_intra_bsize, BlockSize::Block32x32);
        assert!(sf.limit_newmv_early_exit);
        assert!(sf.bias_golden);
        assert!(!sf.cb_pred_filter_search); // exactly 1280x720, not "above"
        assert!(!sf.disable_golden_ref);
        assert!(!sf.rt_intra_dc_only_low_content);
    }

    /// Speed 6 above 1280×720 lights up cb_pred_filter_search
    /// (libvpx `vp9_speed_features.c:658`). The resolution gate is the
    /// load-bearing precondition — a regression that ignores width×
    /// height would fail this test.
    #[test]
    fn w0_3_speed6_above_720p_enables_cb_pred_filter_search() {
        let sub = dispatch(6, Deadline::Realtime, 1280, 720, ContentTier::Default);
        let over = dispatch(6, Deadline::Realtime, 1920, 1080, ContentTier::Default);
        assert!(!sub.cb_pred_filter_search);
        assert!(over.cb_pred_filter_search);
    }

    /// Speed 8 toggles two gates relative to speed 7
    /// (libvpx `vp9_speed_features.c:770, 790`).
    #[test]
    fn w0_3_speed8_realtime_dc_only_and_no_newmv_early_exit() {
        let sf = dispatch(8, Deadline::Realtime, 1280, 720, ContentTier::Default);
        assert!(sf.rt_intra_dc_only_low_content);
        assert!(!sf.limit_newmv_early_exit);
        // bias_golden stays on at speed 8 (libvpx :640 unchanged).
        assert!(sf.bias_golden);
    }

    /// W0.3 PROOF — different (speed, deadline) tuples produce
    /// different `SpeedFeatures`. If the dispatcher is stripped to a
    /// constant return, this test fails.
    #[test]
    fn w0_3_dispatcher_differs_across_speed_and_deadline() {
        let rt7 = dispatch(7, Deadline::Realtime, 1280, 720, ContentTier::Default);
        let good3 = dispatch(3, Deadline::Good, 1280, 720, ContentTier::Default);
        assert_ne!(rt7, good3, "speed/deadline must drive SF divergence");

        let rt6_hd = dispatch(6, Deadline::Realtime, 1920, 1080, ContentTier::Default);
        let rt6_sd = dispatch(6, Deadline::Realtime, 640, 480, ContentTier::Default);
        assert_ne!(rt6_hd, rt6_sd, "frame_size must drive SF divergence");
    }

    /// Deterministic: same input ⇒ same output. Guards against a future
    /// dispatcher that taps into a non-pure source (RNG, time, etc).
    #[test]
    fn w0_3_dispatcher_is_deterministic() {
        for (speed, dl) in [
            (0, Deadline::Best),
            (3, Deadline::Good),
            (6, Deadline::Realtime),
            (8, Deadline::Realtime),
            (-1, Deadline::Best),
        ] {
            let a = dispatch(speed, dl, 1280, 720, ContentTier::Default);
            let b = dispatch(speed, dl, 1280, 720, ContentTier::Default);
            assert_eq!(a, b, "non-deterministic dispatch at speed={speed}, dl={dl:?}");
        }
    }

    /// libvpx takes `abs(speed)` for LUT lookups, so cpu-used = -7
    /// (the "tool" path) should produce the same SF as cpu-used = 7.
    /// libvpx `vp9_speed_features.c:869-913` calls
    /// `set_..._framesize_dependent(cpi, sf, speed)` with the same
    /// signed value but the switch arms key off the magnitude.
    #[test]
    fn w0_3_negative_speed_uses_magnitude() {
        let pos = dispatch(7, Deadline::Realtime, 1280, 720, ContentTier::Default);
        let neg = dispatch(-7, Deadline::Realtime, 1280, 720, ContentTier::Default);
        assert_eq!(pos, neg);
    }
}
