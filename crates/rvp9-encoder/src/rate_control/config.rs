//! User-facing rate-control configuration.

/// CBR rate-control configuration.
///
/// All buffer / lookahead values are in **milliseconds** at the
/// configured framerate — that matches the libvpx
/// `--buf-sz` / `--buf-initial-sz` / `--buf-optimal-sz` CLI shape and is
/// what WebRTC / RFC 6716 callers expect. Internal conversion to bits
/// happens once in [`super::RateControl::new`].
///
/// Defaults mirror libvpx's `VP9E_DEFAULT_TARGET_BANDWIDTH` /
/// `vp9_rc_init` initial settings for the WebRTC profile (see
/// `vp9_cx_iface.c::ctrl_set_*` and `vp9_ratectrl.c:360`).
#[derive(Debug, Clone, Copy)]
pub struct RateControlConfig {
    /// Target average bitrate, bits per second. libvpx
    /// `oxcf.target_bandwidth`.
    pub target_bitrate_bps: u32,

    /// Encoded framerate, frames per second. Must be in
    /// `0.0 < framerate <= MAX_FRAMERATE`. libvpx `oxcf.init_framerate` /
    /// `cpi->framerate`.
    pub framerate: f64,

    /// Total buffer capacity in **milliseconds at the target bitrate**.
    /// libvpx default for CBR is 6000 ms (`--buf-sz=6000`). Translates
    /// to bits via `(target_bitrate_bps * buffer_size_ms) / 1000`.
    pub buffer_size_ms: u32,

    /// Starting buffer fullness in milliseconds. libvpx default 4000 ms
    /// (`--buf-initial-sz=4000`).
    pub starting_buffer_ms: u32,

    /// Optimal / target buffer fullness in milliseconds. libvpx default
    /// 5000 ms (`--buf-optimal-sz=5000`). The CBR Q controller pulls
    /// Q up/down to push the buffer back towards this fullness.
    pub optimal_buffer_ms: u32,

    /// Best (lowest) allowed Q-index. libvpx `oxcf.best_allowed_q`,
    /// default 0 for CBR / lossless. We default to 4 to avoid lossless
    /// mode unintentionally (matches libvpx's CBR-realtime preset).
    pub min_qindex: u8,

    /// Worst (highest) allowed Q-index. libvpx `oxcf.worst_allowed_q`,
    /// default 255.
    pub max_qindex: u8,

    /// Keyframe Q-boost. Must be in `0..=MAX_KF_BOOST`.
    ///
    /// libvpx uses a per-frame computation in
    /// `vp9_calc_iframe_target_size_one_pass_cbr:2204-2218`:
    ///
    /// ```text
    /// kf_boost = max(32, 2 * framerate - 16)
    /// target_kf_bits = ((16 + kf_boost) * avg_frame_bandwidth) >> 4
    /// ```
    ///
    /// Our config stores the **multiplier in 1/16ths above 1.0**, i.e.
    /// libvpx's `kf_boost` directly. Default 32 (== 2x avg). With the
    /// libvpx formula at 30 fps you get `max(32, 44) = 44`, so 44 is
    /// more accurate; we expose this knob so callers can match.
    pub kf_boost: i32,

    /// Frames per GOP (keyframe interval). Must be in
    /// `1..=MAX_GOP_LENGTH`. libvpx `oxcf.key_freq`. Default 60 (== 2
    /// seconds at 30 fps). The rate controller decrements `frames_to_key`
    /// each frame; it's the caller's job to actually insert a keyframe when
    /// `frames_to_key == 0`.
    pub gop_length: u32,

    /// Coded luma frame width in pixels. Needed to convert
    /// "bits-per-mb" estimates into "bits-per-frame" (libvpx
    /// `vp9_estimate_bits_at_q:198-205`). Must be positive and within
    /// the supported coded-frame envelope; non-multiple dimensions are
    /// rounded up to macroblocks by [`Self::mbs`].
    pub frame_width: u32,

    /// Coded luma frame height in pixels. See [`Self::frame_width`].
    pub frame_height: u32,

    /// Scaling factor applied to the libvpx-faithful
    /// `vp9_rc_bits_per_mb` output. Default `1.0` = bit-exact libvpx
    /// faithful (DO NOT TOUCH if you care about cross-decoder parity
    /// of the *picked Q*; the bitstream stays decodable either way,
    /// only the RC trajectory changes).
    ///
    /// **DOCUMENTED DIVERGENCE (A10).** libvpx's per-MB bit estimator
    /// is content-aware (rate from quantised-coefficient energy), but
    /// the simplified enumerator-over-Q formula we port here can
    /// systematically under-predict actual cost on textured / high-
    /// motion content (foreman_cif: ~10× under-prediction measured
    /// empirically — see A10 instrumentation). Under-prediction →
    /// regulate_q picks too-low a Q → encoder massively overshoots
    /// target → RC raises Q → eventually pins at 255 → residuals
    /// quant to zero → PSNR locks at LASTFRAME-staleness floor.
    ///
    /// This knob lets callers correct the bias at the input side
    /// (`bits_per_mb *= multiplier`). The RC feedback loop in
    /// `postencode::update_rate_correction_factors` will still adapt
    /// the per-level `rate_correction_factors[]` on top of this, so
    /// the multiplier sets the *starting point* rather than a fixed
    /// scale.
    ///
    /// Must be `> 0.0`. `validate()` rejects non-positive / non-
    /// finite values.
    pub bits_per_mb_multiplier: f64,

    /// W1.6(vpxenc parity, A3 P0): rate-control mode selector.
    /// Mirrors libvpx's `oxcf.rc_mode` (`vp9_ratectrl.c:1587-1609`
    /// dispatch site). Only [`super::RcMode::Cbr`] has a ported path
    /// today; the other arms panic with a "not yet implemented"
    /// message naming the wave that will land them. Default = Cbr,
    /// matching rvp9's current behaviour and vpxenc's `--end-usage=cbr`
    /// CLI gate.
    pub rc_mode: super::RcMode,

    /// vpxenc `--cq-level` / libvpx `oxcf.cq_level` (`vp9_ratectrl.c`).
    /// Q-index used by [`super::RcMode::Q`] (pinned per-frame) and
    /// as the quality FLOOR for [`super::RcMode::Cq`] (CBR-style
    /// regulator can pick `q >= cq_level` only). Ignored by CBR.
    /// Default 0 (unused by CBR; if Q-mode is selected with the
    /// default, the encoder pins at qindex 0 which is the
    /// lossless edge — set cq_level explicitly).
    pub cq_level: u8,
}

/// Largest coded width or height this encoder currently supports.
///
/// The cap is intentionally explicit: it is above 8K UHD's 7680x4320 envelope,
/// while keeping derived frame-grid arithmetic and allocations bounded.
pub const MAX_CODED_DIMENSION: u32 = 16_384;

/// Largest coded luma sample count this encoder currently supports.
pub const MAX_CODED_LUMA_PIXELS: u64 = 8_192 * 8_192;

/// Largest frame rate accepted by the rate-control model.
pub const MAX_FRAMERATE: f64 = 1_000.0;

/// Largest keyframe boost accepted by the CBR keyframe target formula.
pub const MAX_KF_BOOST: i32 = 4_096;

/// Largest keyframe interval accepted by the encoder.
pub const MAX_GOP_LENGTH: u32 = 10_000;

const MAX_MS_TO_BITS_PRODUCT: u64 = i64::MAX as u64;

impl Default for RateControlConfig {
    fn default() -> Self {
        // 1 Mbps / 30 fps / 720p — a representative WebRTC default.
        // Buffer geometry tracks libvpx's CBR defaults
        // (`vp9_cx_iface.c::set_encoder_config`).
        Self {
            target_bitrate_bps: 1_000_000,
            framerate: 30.0,
            buffer_size_ms: 6000,
            starting_buffer_ms: 4000,
            optimal_buffer_ms: 5000,
            min_qindex: 4,
            max_qindex: 255,
            kf_boost: 32,
            gop_length: 60,
            frame_width: 1280,
            frame_height: 720,
            // Default 1.0 = libvpx-faithful. Callers encoding
            // textured / high-motion content can raise this to
            // compensate for the simplified estimator.
            bits_per_mb_multiplier: 1.0,
            // W1.6(vpxenc parity): CBR is the only ported RC mode
            // today; the other arms in `pick_q_and_bounds` panic.
            rc_mode: super::RcMode::Cbr,
            // Q + CQ rate-control floor. 0 means unused for CBR /
            // default-set-explicitly for Q/CQ.
            cq_level: 0,
        }
    }
}

impl RateControlConfig {
    /// Macroblock count for the configured frame size. Mirrors libvpx
    /// `cm->MBs = (width + 15) / 16 * (height + 15) / 16` from
    /// `vp9_alloc_compressor_data` setup. Cached internally by
    /// [`super::RateControl`].
    #[must_use]
    pub fn mbs(&self) -> u32 {
        let w_mb = ceil_div_16(self.frame_width);
        let h_mb = ceil_div_16(self.frame_height);
        let mbs = (w_mb as u64) * (h_mb as u64);
        mbs.min(u32::MAX as u64) as u32
    }

    /// Pinned libvpx's initial one-pass VBR GOLDEN interval for this
    /// resolution and frame rate (`vp9_rc_get_default_{min,max}_gf_interval`
    /// followed by `vp9_set_gf_update_one_pass_vbr`).
    #[must_use]
    pub(crate) fn one_pass_vbr_gf_interval(&self) -> u32 {
        const MIN_GF_INTERVAL: u32 = 4;
        const MAX_GF_INTERVAL: u32 = 16;
        const FACTOR_SAFE: f64 = 3840.0 * 2160.0 * 20.0;

        let default_min = (self.framerate * 0.125) as u32;
        let mut min_interval = default_min.clamp(MIN_GF_INTERVAL, MAX_GF_INTERVAL);
        let factor = f64::from(self.frame_width) * f64::from(self.frame_height) * self.framerate;
        if factor > FACTOR_SAFE {
            let scaled = (f64::from(MIN_GF_INTERVAL) * factor / FACTOR_SAFE + 0.5) as u32;
            min_interval = min_interval.max(scaled);
        }

        let mut max_interval = ((self.framerate * 0.75) as u32).min(MAX_GF_INTERVAL);
        max_interval += max_interval & 1;
        max_interval = max_interval.max(min_interval);

        ((min_interval + max_interval) / 2).clamp(10, 20)
    }

    /// Validate the config. Returns an error string on the first
    /// detected problem.
    ///
    /// # Errors
    ///
    /// Returns `Err(&'static str)` if any field is out of range, e.g.
    /// `framerate <= 0`, `min_qindex > max_qindex`, zero frame
    /// dimensions, or `optimal_buffer_ms > buffer_size_ms`.
    pub const fn validate(&self) -> Result<(), &'static str> {
        if self.target_bitrate_bps == 0 {
            return Err("target_bitrate_bps must be > 0");
        }
        // Use `is_sign_positive` + `is_finite` so NaN / negative /
        // infinite framerates are all rejected; the negated `>` form
        // upsets `clippy::neg_cmp_op_on_partial_ord`.
        if !self.framerate.is_finite() || self.framerate <= 0.0 {
            return Err("framerate must be a positive finite number");
        }
        if self.framerate > MAX_FRAMERATE {
            return Err("framerate exceeds supported maximum");
        }
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err("frame_width / frame_height must be > 0");
        }
        if !coded_dimensions_supported(self.frame_width, self.frame_height) {
            return Err("frame_width / frame_height exceed supported maximum");
        }
        if !macroblock_count_fits_u32(self.frame_width, self.frame_height) {
            return Err("macroblock count exceeds supported maximum");
        }
        if self.min_qindex > self.max_qindex {
            return Err("min_qindex must be <= max_qindex");
        }
        if self.optimal_buffer_ms > self.buffer_size_ms {
            return Err("optimal_buffer_ms must be <= buffer_size_ms");
        }
        if self.starting_buffer_ms > self.buffer_size_ms {
            return Err("starting_buffer_ms must be <= buffer_size_ms");
        }
        if self.gop_length == 0 {
            return Err("gop_length must be > 0");
        }
        if self.gop_length > MAX_GOP_LENGTH {
            return Err("gop_length exceeds supported maximum");
        }
        if self.kf_boost < 0 || self.kf_boost > MAX_KF_BOOST {
            return Err("kf_boost exceeds supported maximum");
        }
        if !ms_to_bits_product_fits(self.buffer_size_ms, self.target_bitrate_bps)
            || !ms_to_bits_product_fits(self.optimal_buffer_ms, self.target_bitrate_bps)
            || !ms_to_bits_product_fits(self.starting_buffer_ms, self.target_bitrate_bps)
        {
            return Err("buffer duration and bitrate exceed supported maximum");
        }
        if !self.bits_per_mb_multiplier.is_finite() || self.bits_per_mb_multiplier <= 0.0 {
            return Err("bits_per_mb_multiplier must be a positive finite number");
        }
        Ok(())
    }
}

const fn ceil_div_16(value: u32) -> u32 {
    (value / 16) + if value.is_multiple_of(16) { 0 } else { 1 }
}

const fn coded_dimensions_supported(width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    if width > MAX_CODED_DIMENSION || height > MAX_CODED_DIMENSION {
        return false;
    }
    (width as u64) <= MAX_CODED_LUMA_PIXELS / (height as u64)
}

const fn macroblock_count_fits_u32(width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    let w_mb = ceil_div_16(width) as u64;
    let h_mb = ceil_div_16(height) as u64;
    w_mb <= (u32::MAX as u64) / h_mb
}

const fn ms_to_bits_product_fits(ms: u32, bps: u32) -> bool {
    ms == 0 || bps == 0 || (ms as u64) <= MAX_MS_TO_BITS_PRODUCT / (bps as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = RateControlConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn default_mbs_count_for_720p() {
        // 1280x720 → 80 x 45 = 3600 MBs.
        let cfg = RateControlConfig::default();
        assert_eq!(cfg.mbs(), 80 * 45);
    }

    #[test]
    fn one_pass_vbr_gf_interval_matches_pinned_defaults() {
        let mut cfg = RateControlConfig {
            frame_width: 1280,
            frame_height: 720,
            framerate: 30.0,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.one_pass_vbr_gf_interval(), 10);

        cfg.frame_height = 704;
        cfg.framerate = 50.0;
        assert_eq!(cfg.one_pass_vbr_gf_interval(), 11);

        cfg.frame_width = 1920;
        cfg.frame_height = 1080;
        cfg.framerate = 60.0;
        assert_eq!(cfg.one_pass_vbr_gf_interval(), 11);
    }

    #[test]
    fn validate_rejects_zero_framerate() {
        let cfg = RateControlConfig {
            framerate: 0.0,
            ..RateControlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_inverted_qrange() {
        let cfg = RateControlConfig {
            min_qindex: 200,
            max_qindex: 100,
            ..RateControlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_target_bitrate() {
        let cfg = RateControlConfig {
            target_bitrate_bps: 0,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.validate(), Err("target_bitrate_bps must be > 0"),);
    }

    #[test]
    fn validate_rejects_negative_framerate() {
        let cfg = RateControlConfig {
            framerate: -1.0,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("framerate must be a positive finite number"),
        );
    }

    #[test]
    fn validate_rejects_nan_framerate() {
        let cfg = RateControlConfig {
            framerate: f64::NAN,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("framerate must be a positive finite number"),
        );
    }

    #[test]
    fn validate_rejects_infinite_framerate() {
        let cfg = RateControlConfig {
            framerate: f64::INFINITY,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("framerate must be a positive finite number"),
        );
    }

    #[test]
    fn validate_rejects_absurd_finite_framerate() {
        let cfg = RateControlConfig {
            framerate: f64::MAX,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.validate(), Err("framerate exceeds supported maximum"),);
    }

    #[test]
    fn validate_rejects_zero_frame_width() {
        let cfg = RateControlConfig {
            frame_width: 0,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("frame_width / frame_height must be > 0"),
        );
    }

    #[test]
    fn validate_rejects_zero_frame_height() {
        let cfg = RateControlConfig {
            frame_height: 0,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("frame_width / frame_height must be > 0"),
        );
    }

    #[test]
    fn validate_rejects_overflowing_macroblock_count() {
        let cfg = RateControlConfig {
            frame_width: u32::MAX,
            frame_height: u32::MAX,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("frame_width / frame_height exceed supported maximum"),
        );
    }

    #[test]
    fn mbs_does_not_panic_on_unvalidated_huge_dimensions() {
        let cfg = RateControlConfig {
            frame_width: u32::MAX,
            frame_height: u32::MAX,
            ..RateControlConfig::default()
        };
        let mbs =
            std::panic::catch_unwind(|| cfg.mbs()).expect("mbs should saturate rather than panic");
        assert_eq!(mbs, u32::MAX);
    }

    #[test]
    fn validate_rejects_buffer_size_ms_to_bits_overflow() {
        let cfg = RateControlConfig {
            target_bitrate_bps: u32::MAX,
            buffer_size_ms: u32::MAX,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("buffer duration and bitrate exceed supported maximum"),
        );
    }

    #[test]
    fn validate_rejects_optimal_buffer_above_buffer_size() {
        let cfg = RateControlConfig {
            buffer_size_ms: 1000,
            optimal_buffer_ms: 2000,
            starting_buffer_ms: 500,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("optimal_buffer_ms must be <= buffer_size_ms"),
        );
    }

    #[test]
    fn validate_rejects_starting_buffer_above_buffer_size() {
        let cfg = RateControlConfig {
            buffer_size_ms: 1000,
            optimal_buffer_ms: 500,
            starting_buffer_ms: 2000,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("starting_buffer_ms must be <= buffer_size_ms"),
        );
    }

    #[test]
    fn validate_rejects_zero_gop_length() {
        let cfg = RateControlConfig {
            gop_length: 0,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.validate(), Err("gop_length must be > 0"));
    }

    #[test]
    fn validate_rejects_unbounded_gop_length() {
        let cfg = RateControlConfig {
            gop_length: u32::MAX,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.validate(), Err("gop_length exceeds supported maximum"));
    }

    #[test]
    fn validate_rejects_unbounded_kf_boost() {
        let cfg = RateControlConfig {
            kf_boost: i32::MAX,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.validate(), Err("kf_boost exceeds supported maximum"));
    }

    #[test]
    fn validate_rejects_nonpositive_bpm_multiplier() {
        let cfg = RateControlConfig {
            bits_per_mb_multiplier: 0.0,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("bits_per_mb_multiplier must be a positive finite number"),
        );
    }

    #[test]
    fn validate_rejects_nan_bpm_multiplier() {
        let cfg = RateControlConfig {
            bits_per_mb_multiplier: f64::NAN,
            ..RateControlConfig::default()
        };
        assert_eq!(
            cfg.validate(),
            Err("bits_per_mb_multiplier must be a positive finite number"),
        );
    }

    #[test]
    fn mbs_rounds_up_for_non_multiple_of_16() {
        // 17 px wide ⇒ 2 MB columns (ceil-div). 16 px tall ⇒ 1 MB row.
        let cfg = RateControlConfig {
            frame_width: 17,
            frame_height: 16,
            ..RateControlConfig::default()
        };
        assert_eq!(cfg.mbs(), 2);
    }
}
