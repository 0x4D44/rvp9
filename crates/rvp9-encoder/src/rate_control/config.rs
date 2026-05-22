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

    /// Encoded framerate, frames per second. Must be > 0. libvpx
    /// `oxcf.init_framerate` / `cpi->framerate`.
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

    /// Keyframe Q-boost. libvpx uses a per-frame computation in
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

    /// Frames per GOP (keyframe interval). libvpx `oxcf.key_freq`.
    /// Default 60 (== 2 seconds at 30 fps). The rate controller
    /// decrements `frames_to_key` each frame; it's the caller's job
    /// to actually insert a keyframe when `frames_to_key == 0`.
    pub gop_length: u32,

    /// Coded luma frame width in pixels. Needed to convert
    /// "bits-per-mb" estimates into "bits-per-frame" (libvpx
    /// `vp9_estimate_bits_at_q:198-205`). Must be a positive multiple
    /// of 16 (one macroblock).
    pub frame_width: u32,

    /// Coded luma frame height in pixels. See `frame_width`.
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
}

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
        let w_mb = self.frame_width.div_ceil(16);
        let h_mb = self.frame_height.div_ceil(16);
        w_mb * h_mb
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
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err("frame_width / frame_height must be > 0");
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
        if !self.bits_per_mb_multiplier.is_finite() || self.bits_per_mb_multiplier <= 0.0 {
            return Err("bits_per_mb_multiplier must be a positive finite number");
        }
        Ok(())
    }
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
}
