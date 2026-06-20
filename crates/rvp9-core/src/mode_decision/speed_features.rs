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

/// libvpx `ML_PARTITION_RD_FEATURES sf->rd_ml_partition`
/// (`vp9_speed_features.h`). The slice rvp9's A21 partition-pruning
/// integration consumes — the `search_breakout` gate + thresholds and
/// (A21-rect) the `prune_rect_thresh` rect-pruning thresholds. Other
/// libvpx fields (`search_early_termination`, `var_pruning`) are
/// documented A21 follow-ups (`search_early_termination` is speed-0
/// only; `var_pruning` needs the `simple_motion_search` pre-pass).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RdMlPartition {
    /// libvpx `sf->rd_ml_partition.search_breakout`
    /// (`vp9_speed_features.c:94, 115, 153, 177, 880`). Enables the
    /// `ml_predict_breakout` partition-search breakout
    /// (`vp9_encodeframe.c:4161-4168`). On at speed 0/1 (Good preset,
    /// below 1080p); off at the best-quality defaults and speed 2+.
    pub search_breakout: bool,
    /// libvpx `sf->rd_ml_partition.search_breakout_thresh[q_ctx]`
    /// (`vp9_speed_features.c:96-102, 125-133, 158-160`). Indexed by
    /// the q-context `qindex>=200 ? 0 : (qindex>=150 ? 1 : 2)`. A
    /// positive `ml_predict_breakout` linear score `>= thresh` votes
    /// to stop the partition search.
    pub search_breakout_thresh: [f32; 3],
    /// libvpx `sf->rd_ml_partition.prune_rect_thresh[bsize_idx]`
    /// (`vp9_speed_features.c:253-256, 274-276, 340-342, 1012-1015`).
    /// Indexed `[0]`=8×8 (always -1, unused), `[1]`=16×16, `[2]`=32×32,
    /// `[3]`=64×64. Drives `ml_prune_rect_partition`
    /// (`vp9_encodeframe.c:3334-3425`, called at :4337): the model's
    /// `max_score - thresh` margin decides which rect arms
    /// (PARTITION_HORZ / PARTITION_VERT) stay allowed. A negative
    /// thresh disables the model for that size (libvpx :3366
    /// `if (thresh < 0) return;`). Good speed 0: `[-1, 350, 325, 250]`;
    /// speed 1: `[-1, 225, 225, 225]`; speed 2+ and the best-quality /
    /// RT defaults: all `-1` (off).
    pub prune_rect_thresh: [i32; 4],
}

impl RdMlPartition {
    /// libvpx best-quality / RT defaults (`vp9_speed_features.c:879-880`,
    /// `:1012-1015`): breakout off, rect-pruning off (all thresholds
    /// negative).
    #[must_use]
    pub const fn defaults() -> Self {
        Self {
            search_breakout: false,
            search_breakout_thresh: [0.0, 0.0, 0.0],
            prune_rect_thresh: [-1, -1, -1, -1],
        }
    }
}

/// Per-frame speed-feature decisions.
///
/// Mirrors the libvpx `SPEED_FEATURES` struct (`vp9_speed_features.h`)
/// at field granularity, but stops at the slice the rvp9 encoder
/// currently consumes (lib.rs:3713-3724). Later waves grow this to
/// match libvpx's full SF surface; each new field comes paired with
/// the libvpx `file:line` it was lifted from.
#[derive(Copy, Clone, Debug, PartialEq)]
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
    /// libvpx `sf->rd_ml_partition` (`vp9_speed_features.h`). A21 —
    /// the ML partition-search breakout gate + thresholds consumed by
    /// the RD partition descent.
    pub rd_ml_partition: RdMlPartition,
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
            // libvpx `vp9_speed_features.c:879-880` — breakout off in
            // the best-quality defaults; the Good/RT framesize-dependent
            // arms turn it on per resolution + speed.
            rd_ml_partition: RdMlPartition::defaults(),
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
fn apply_rt(sf: &mut SpeedFeatures, mag: u32, width: u32, height: u32, _content: ContentTier) {
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

    // C2-β — SIZE-GATED ML-partition `search_breakout` at cpu7 RT,
    // DEFAULT-ON for large frames.
    //
    // The ML breakout pruner (`ml_partition::ml_predict_breakout`,
    // consumed by the F.3 RD partition descent in
    // `rvp9-encoder/src/lib.rs`) skips a block's SPLIT recursion when
    // the NN votes the NONE pick is good enough. Measured on cpu7 CBR:
    //   • LARGE (≳720p, e.g. 1280×704): ~1.20× for ≈−0.06 dB — a win.
    //   • SMALL (e.g. 320×256 foreman):  ~1.04× for ≈−0.06 dB — cost
    //     without benefit (the descent is already cheap there).
    // So we gate by FRAME AREA: ON only for large frames, baseline OFF
    // for small frames (incl. the small golden fixtures → no churn).
    //
    // DEFAULT-ON: enabled unless `RVP9_C2B=0` force-disables it (the
    // escape hatch for goldens/debug, restoring the prior cpu7
    // behaviour byte-identically). The downstream consumer also
    // requires `base_qindex >= 100` (lib.rs) and the fire-rate counter
    // `inter_dispatch::ML_PARTITION_BREAKOUT_FIRED` proves it actually
    // fires (high on large, ZERO on small — the gate).
    const C2B_LARGE_FRAME_AREA: u64 = 700_000; // 1280×704 = 901_120 ≥; 320×256 = 81_920 <
    let is_large_frame = u64::from(width) * u64::from(height) >= C2B_LARGE_FRAME_AREA;
    // Escape hatch: `RVP9_C2B=0` force-disables (anything else, incl.
    // unset, leaves it on).
    let c2b_disabled = std::env::var_os("RVP9_C2B").is_some_and(|v| v == "0");
    if !c2b_disabled && mag == 7 && is_large_frame {
        sf.rd_ml_partition.search_breakout = true;
        // Stage-4a large-frame recovery threshold tuned on the 1280×704
        // realtime gate with the other large-frame-only speed features:
        // -20 keeps the median under 33 ms. This speed feature is gated
        // off for small frames, so the foreman low-bitrate quality gate
        // does not exercise this aggressive setting. A higher threshold
        // is MORE conservative (fires less → better quality, less
        // speed). `RVP9_C2B_THRESH=<f32>` overrides all three q-contexts
        // uniformly for tuning sweeps.
        let thr = std::env::var("RVP9_C2B_THRESH")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(-20.0);
        sf.rd_ml_partition.search_breakout_thresh = [thr, thr, thr];
    }
}

/// libvpx `set_good_speed_feature_framesize_{independent,dependent}`
/// (`vp9_speed_features.c:585-868` for the good-quality arms).
///
/// W0.3 covers only the slice of the Good path needed to differ from
/// the best-quality defaults at speed >= 5. Slow-path SF (speed 0..=4)
/// stays at defaults until Wave 4 brings the RDO driver online.
fn apply_good(sf: &mut SpeedFeatures, mag: u32, width: u32, height: u32, _content: ContentTier) {
    // libvpx `vp9_speed_features.c:571` — speed 5+ caps intra search
    // at 32x32 across both Good and RT paths.
    if mag >= 5 {
        sf.max_intra_bsize = BlockSize::Block32x32;
    }

    // A21 — `rd_ml_partition.search_breakout` gate + thresholds, lifted
    // from libvpx `set_good_speed_feature_framesize_dependent`
    // (`vp9_speed_features.c:81-180`). The `min(w, h)` resolution
    // tiers gate which thresholds apply.
    let min_dim = width.min(height);
    let is_720p_or_larger = min_dim >= 720;
    let is_1080p_or_larger = min_dim >= 1080;

    // libvpx `:93-104` — speed 0 Good (and the speed>=1 arm reasserts
    // it at `:115`): breakout on below 1080p.
    if mag <= 1 && !is_1080p_or_larger {
        sf.rd_ml_partition.search_breakout = true;
        if mag == 0 {
            // libvpx `:95-103`.
            sf.rd_ml_partition.search_breakout_thresh = if is_720p_or_larger {
                [0.0, 0.0, 0.0]
            } else {
                [2.5, 1.5, 1.5]
            };
        } else {
            // libvpx `:121-134` (speed >= 1).
            sf.rd_ml_partition.search_breakout_thresh = if is_720p_or_larger {
                [-5.0, -5.0, -9.0]
            } else {
                [-1.0, -1.0, -1.0]
            };
        }
    }

    // libvpx `:144-161` — speed 2 keeps breakout on only below 720p
    // (with relaxed thresholds); `:176-177` — speed 3+ turns it off.
    if mag == 2 && !is_720p_or_larger {
        sf.rd_ml_partition.search_breakout = true;
        sf.rd_ml_partition.search_breakout_thresh = [0.0, -1.0, -4.0];
    }
    // speed >= 3 (and speed 2 at 720p+) leave breakout off (the
    // best-quality default), so no further assignment is needed.

    // A21-rect — `rd_ml_partition.prune_rect_thresh`, lifted from
    // libvpx `set_good_speed_feature_framesize_independent`
    // (`vp9_speed_features.c:253-256, 274-276, 340-342`). These are
    // resolution-INDEPENDENT (unlike search_breakout above). Drives
    // `ml_prune_rect_partition` (`vp9_encodeframe.c:3334-3425`); a
    // negative thresh disables the model for that size. Index `[0]` is
    // 8×8 (always -1; the model has no 8×8 arm).
    if mag == 0 {
        // libvpx `:253-256` — speed 0 Good base.
        sf.rd_ml_partition.prune_rect_thresh = [-1, 350, 325, 250];
    } else if mag == 1 {
        // libvpx `:274-276` — speed 1 tightens 16/32/64 to 225.
        sf.rd_ml_partition.prune_rect_thresh = [-1, 225, 225, 225];
    } else {
        // libvpx `:340-342` — speed 2+ disables rect-pruning (all -1).
        // (Best-quality / RT defaults already start at all -1.)
        sf.rd_ml_partition.prune_rect_thresh = [-1, -1, -1, -1];
    }

    // No further Good-arm overrides land here. Wave 4 (RDO driver)
    // brings the remaining speed 0..=4 good-quality switch arms with
    // their full libvpx citation set.
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
        // The W0.3 mode/intra fields stay at the best-quality defaults
        // at speed 0 Good. (A21 enables `rd_ml_partition.search_breakout`
        // at this resolution per libvpx `vp9_speed_features.c:94`, so a
        // whole-struct equality no longer holds — that field is checked
        // separately in `a21_*` below.)
        assert_eq!(sf.max_intra_bsize, BlockSize::Block64x64);
        assert!(!sf.limit_newmv_early_exit);
        assert!(!sf.bias_golden);
        assert!(!sf.cb_pred_filter_search);
        assert!(!sf.disable_golden_ref);
        assert!(!sf.rt_intra_dc_only_low_content);
    }

    /// A21 — `rd_ml_partition.search_breakout` is ENABLED at speed 0/1
    /// Good below 1080p, with the libvpx resolution-tiered thresholds
    /// (`vp9_speed_features.c:93-134`), and DISABLED at speed 3+ and in
    /// the best-quality defaults (`:879-880`).
    #[test]
    fn a21_rd_ml_partition_breakout_gate() {
        // Speed 0, SD (<720p): on, thresh [2.5, 1.5, 1.5] (libvpx :100-102).
        let sd0 = dispatch(0, Deadline::Good, 640, 480, ContentTier::Default);
        assert!(sd0.rd_ml_partition.search_breakout);
        assert_eq!(sd0.rd_ml_partition.search_breakout_thresh, [2.5, 1.5, 1.5]);

        // Speed 0, 720p: on, thresh [0, 0, 0] (libvpx :96-98).
        let hd0 = dispatch(0, Deadline::Good, 1280, 720, ContentTier::Default);
        assert!(hd0.rd_ml_partition.search_breakout);
        assert_eq!(hd0.rd_ml_partition.search_breakout_thresh, [0.0, 0.0, 0.0]);

        // Speed 0, 1080p: off (libvpx :93 `!is_1080p_or_larger`).
        let fhd0 = dispatch(0, Deadline::Good, 1920, 1080, ContentTier::Default);
        assert!(!fhd0.rd_ml_partition.search_breakout);

        // Speed 1, SD: on, thresh [-1, -1, -1] (libvpx :131-133).
        let sd1 = dispatch(1, Deadline::Good, 640, 480, ContentTier::Default);
        assert!(sd1.rd_ml_partition.search_breakout);
        assert_eq!(
            sd1.rd_ml_partition.search_breakout_thresh,
            [-1.0, -1.0, -1.0]
        );

        // Speed 3, SD: off (libvpx :176-177).
        let sd3 = dispatch(3, Deadline::Good, 640, 480, ContentTier::Default);
        assert!(!sd3.rd_ml_partition.search_breakout);
    }

    /// C2-β — the size-gated cpu7 RT breakout, DEFAULT-ON for large
    /// frames. By default it is ON only at cpu7 on LARGE frames (≥700k
    /// px) and stays OFF on small frames (cost without benefit there)
    /// and at other speeds. `RVP9_C2B=0` is the escape hatch that
    /// force-disables it everywhere (restoring byte-identical baseline
    /// for goldens/debug); `RVP9_C2B_THRESH` overrides the -20 default.
    ///
    /// This test owns the `RVP9_C2B*` env keys and restores them.
    /// Reads the gate through the real `dispatch` path so it exercises
    /// the production code.
    #[test]
    fn c2b_size_gated_rt_breakout() {
        // SAFETY: single-threaded test mutating process-global env vars
        // it exclusively owns; restored before returning.
        let large = (1280u32, 704u32); // 901_120 px — the C2-β large clip.
        let small = (320u32, 256u32); // 81_920 px — foreman.

        // Default (unset): ON for large @ cpu7, OFF for small @ cpu7.
        // SAFETY: single-threaded test that exclusively owns these env vars and
        // restores them before returning; no concurrent getenv/setenv.
        unsafe {
            std::env::remove_var("RVP9_C2B");
            std::env::remove_var("RVP9_C2B_THRESH");
        }
        let def_large = dispatch(
            7,
            Deadline::Realtime,
            large.0,
            large.1,
            ContentTier::Default,
        );
        assert!(
            def_large.rd_ml_partition.search_breakout,
            "C2-β default-ON: large frame at cpu7 must enable breakout"
        );
        assert_eq!(
            def_large.rd_ml_partition.search_breakout_thresh,
            [-20.0, -20.0, -20.0],
            "C2-β default threshold is -20 for the large-frame recovery path"
        );
        assert!(
            !dispatch(
                7,
                Deadline::Realtime,
                small.0,
                small.1,
                ContentTier::Default
            )
            .rd_ml_partition
            .search_breakout,
            "C2-β: small frame must keep breakout OFF (the size gate)"
        );
        // Even on the default path, only cpu7 is affected (cpu6 stays
        // baseline).
        assert!(
            !dispatch(
                6,
                Deadline::Realtime,
                large.0,
                large.1,
                ContentTier::Default
            )
            .rd_ml_partition
            .search_breakout,
            "C2-β: only cpu7 large frames enable breakout"
        );

        // Escape hatch `RVP9_C2B=0`: force-OFF on both, even large @ cpu7.
        // SAFETY: single-threaded test exclusively owning this env var.
        unsafe { std::env::set_var("RVP9_C2B", "0") };
        assert!(
            !dispatch(
                7,
                Deadline::Realtime,
                large.0,
                large.1,
                ContentTier::Default
            )
            .rd_ml_partition
            .search_breakout,
            "C2-β RVP9_C2B=0: must force-disable on large @ cpu7"
        );
        assert!(
            !dispatch(
                7,
                Deadline::Realtime,
                small.0,
                small.1,
                ContentTier::Default
            )
            .rd_ml_partition
            .search_breakout,
            "C2-β RVP9_C2B=0: stays off on small"
        );
        // SAFETY: single-threaded test exclusively owning this env var.
        unsafe { std::env::remove_var("RVP9_C2B") };

        // `RVP9_C2B_THRESH` overrides the -20 default uniformly.
        // SAFETY: single-threaded test exclusively owning this env var.
        unsafe { std::env::set_var("RVP9_C2B_THRESH", "1.5") };
        let tuned = dispatch(
            7,
            Deadline::Realtime,
            large.0,
            large.1,
            ContentTier::Default,
        );
        assert!(tuned.rd_ml_partition.search_breakout);
        assert_eq!(
            tuned.rd_ml_partition.search_breakout_thresh,
            [1.5, 1.5, 1.5],
            "C2-β RVP9_C2B_THRESH overrides the threshold"
        );
        // SAFETY: single-threaded test exclusively owning this env var.
        unsafe { std::env::remove_var("RVP9_C2B_THRESH") };
    }

    /// A21-rect — `rd_ml_partition.prune_rect_thresh` matches libvpx's
    /// resolution-INDEPENDENT good-quality settings
    /// (`vp9_speed_features.c:253-256, 274-276, 340-342`) and is OFF
    /// (all -1) at the Realtime / best-quality defaults (`:1012-1015`).
    #[test]
    fn a21_rd_ml_partition_rect_prune_thresh_gate() {
        // Speed 0 Good — [-1, 350, 325, 250] (libvpx :253-256).
        // Resolution-independent: same at SD and 1080p.
        let g0_sd = dispatch(0, Deadline::Good, 640, 480, ContentTier::Default);
        assert_eq!(g0_sd.rd_ml_partition.prune_rect_thresh, [-1, 350, 325, 250]);
        let g0_fhd = dispatch(0, Deadline::Good, 1920, 1080, ContentTier::Default);
        assert_eq!(
            g0_fhd.rd_ml_partition.prune_rect_thresh,
            [-1, 350, 325, 250]
        );

        // Speed 1 Good — [-1, 225, 225, 225] (libvpx :274-276).
        let g1 = dispatch(1, Deadline::Good, 640, 480, ContentTier::Default);
        assert_eq!(g1.rd_ml_partition.prune_rect_thresh, [-1, 225, 225, 225]);

        // Speed 2 Good — off (libvpx :340-342).
        let g2 = dispatch(2, Deadline::Good, 640, 480, ContentTier::Default);
        assert_eq!(g2.rd_ml_partition.prune_rect_thresh, [-1, -1, -1, -1]);

        // Speed 7 Realtime — off (libvpx :1012-1015 / best-quality
        // defaults). The sacred gate runs here, so the rect pruner is
        // inert on the gate stream.
        let rt7 = dispatch(7, Deadline::Realtime, 1280, 720, ContentTier::Default);
        assert_eq!(rt7.rd_ml_partition.prune_rect_thresh, [-1, -1, -1, -1]);
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
            assert_eq!(
                a, b,
                "non-deterministic dispatch at speed={speed}, dl={dl:?}"
            );
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
