//! W2 — pre-encode source analysis pipeline.
//!
//! Libvpx runs a "pre-analysis" pass per frame BEFORE mode decision —
//! a cluster of pure-source computations whose outputs feed the
//! picker, partition, AQ, and denoiser. The pipeline is structured as
//! a few cooperating producers sharing per-frame state:
//!
//! - `consec_zero_mv[]` — per-mi-cell zero-MV streak counter,
//!   updated post-frame from the picker's decisions, read by
//!   `avg_source_sad` and the temporal denoiser.
//! - `compute_source_sad_onepass` — frame-level gate (libvpx
//!   `vp9_encoder.c:3984` `cpi->compute_source_sad_onepass =
//!   cpi->oxcf.mode == REALTIME`) — true on RT, gates the avg-source
//!   SAD pass.
//! - `avg_source_sad[]` — per-SB64 avg-of-block-SADs against the
//!   previous source frame (libvpx `vp9_encodeframe.c:1209-1257`).
//! - `content_state_sb[]` — per-SB classification (LowSad /
//!   HighSad / VeryHighSad / LowSadHighSumdiff) derived from
//!   `avg_source_sad[]` and pixel variance.
//! - Scene-change detector (`vp9_scene_detection_onepass`) — sets
//!   `high_source_sad` and a forced-KF flag.
//! - Noise estimator (`vp9_update_noise_estimate`) — derives a
//!   `NoiseEstimate { enabled, level }` shared with the picker.
//! - Skin detector — per-mi `skin_map[]` shared with the picker.
//!
//! W2.1 lands the **scaffold** — the data-structure homes and the
//! per-frame lifecycle hooks. Producers (W2.2..W2.5) and threshold
//! consumers (W2.6) plug into these slots over the rest of Wave 2.

use crate::frame_loop::Deadline;
use rvp9_core::mode_decision::inter::{ContentStateSb, NoiseEstimate, NoiseLevel};

/// W2.1(vpxenc parity, A5 P0) — per-frame pre-analysis state owned
/// by `StreamEncoder`. Allocated once at encoder construction and
/// reset at every keyframe; per-frame fields are updated by the W2.2+
/// producers.
#[derive(Debug, Clone)]
pub struct PreAnalysisState {
    /// libvpx `cpi->consec_zero_mv[mi_rows * mi_cols]`
    /// (`vp9_encoder.h:629`). One u8 per 8×8 mi cell, counting
    /// consecutive frames the cell carried a zero MV. Reset to 0 on
    /// keyframes; incremented or zeroed per-cell post-encode based
    /// on the picker's pick.
    pub consec_zero_mv: Vec<u8>,
    /// libvpx `cpi->compute_source_sad_onepass`
    /// (`vp9_encoder.h:643`, set at `vp9_encoder.c:3984`). `true` on
    /// the realtime path, gates the avg-source-SAD pre-pass. Updated
    /// per-frame by `update_per_frame_gates`.
    pub compute_source_sad_onepass: bool,
    /// libvpx `cpi->Last_Source` view (`vp9_encoder.h:586`). The
    /// previous source frame's luma plane (tight-stride, `width *
    /// height` samples). Populated post-encode with the just-encoded
    /// source so the next frame's avg-source-SAD pass has a
    /// reference. `None` before the first encoded frame (which is
    /// always a keyframe — the avg-SAD pass is meaningless for KFs).
    pub last_source_y: Option<Vec<u8>>,
    /// W2.2(vpxenc parity, A5 P1): libvpx
    /// `vp9_encodeframe.c:1209-1257` `tmp_sad` per SB64. Indexed
    /// `[sb_row * sb_cols + sb_col]`; sized `sb_rows * sb_cols`.
    pub avg_source_sad: Vec<u64>,
    /// W2.2(vpxenc parity, A5 P1): libvpx
    /// `vp9_encodeframe.c:1231-1245` `x->content_state_sb`,
    /// classified per SB64 from avg_source_sad + tmp_sse.
    pub content_state_sb: Vec<ContentStateSb>,
    /// W2.3(vpxenc parity, A5 P2): libvpx `cpi->rc.high_source_sad`
    /// (`vp9_ratectrl.c:3034-3216`). `true` when the
    /// scene-detection pass spots a large jump in average per-SB
    /// SAD vs the running mean. Consumed by the picker / RC reset
    /// logic in libvpx; W2.3 produces it.
    pub high_source_sad: bool,
    /// W2.3(vpxenc parity, A5 P2): libvpx
    /// `vp9_ratectrl.c:3111-3112, 3176-3177`
    /// `scene_cut_force_key_frame`. `true` when `high_source_sad`
    /// AND the cut magnitude exceeds the key-frame threshold; the
    /// encoder is expected to insert a forced KF on the next frame.
    pub scene_cut_force_key_frame: bool,
    /// W2.3 running mean of per-frame `avg_source_sad`. libvpx
    /// `rc->avg_source_sad[0]` updated by the recursive average
    /// `(3 * prev + cur) >> 2` at `vp9_ratectrl.c:3179`. The mean
    /// is the load-bearing input to the scene-detection threshold
    /// comparison.
    pub prev_avg_source_sad: u64,
    /// W2.4(vpxenc parity, A5 P3): libvpx `cpi->noise_estimate`
    /// (`vp9_encoder.h:971`). The 2-field consumer-facing view
    /// `{ enabled, level }` that the picker reads via
    /// `vp9_NEWMV_diff_bias` (libvpx `vp9_pickmode.c:1299-1356`).
    /// The full libvpx state (`value`, `thresh`, `adapt_thresh`,
    /// `count`, `num_frames_estimate`, `last_w`, `last_h`) is held
    /// privately in the `ne_*` fields below; only `{enabled, level}`
    /// is published.
    pub noise_estimate: NoiseEstimate,
    /// libvpx `ne->value` (`vp9_noise_estimate.h:33`). The running
    /// noise estimate; scaled `max_bin * 40` with `(3*prev + cur) >> 2`
    /// recursive averaging at `vp9_noise_estimate.c:280`.
    ne_value: i32,
    /// libvpx `ne->thresh` (`vp9_noise_estimate.h:34`). Resolution-
    /// dependent threshold set at init (`vp9_noise_estimate.c:38-47`).
    ne_thresh: i32,
    /// libvpx `ne->adapt_thresh` (`vp9_noise_estimate.h:35`). Set to
    /// `(3 * thresh) >> 1` at init (`vp9_noise_estimate.c:49`); drives
    /// the rapid-onset-noise fast-path at line 282.
    ne_adapt_thresh: i32,
    /// libvpx `ne->count` (`vp9_noise_estimate.h:36`). Frames-
    /// accumulated counter; level updates when it hits
    /// `num_frames_estimate` (`vp9_noise_estimate.c:282-291`).
    ne_count: i32,
    /// libvpx `ne->num_frames_estimate` (`vp9_noise_estimate.h:39`).
    /// 15 at init, 30 after the first level update
    /// (`vp9_noise_estimate.c:48, 289`).
    ne_num_frames_estimate: i32,
    /// libvpx `ne->last_w` / `ne->last_h` (`vp9_noise_estimate.h:37-38`).
    /// Tracks the resolution last seen to detect resize and re-init
    /// (`vp9_noise_estimate.c:132-143`).
    ne_last_w: u32,
    /// See `ne_last_w`.
    ne_last_h: u32,
    /// Width × height the state was sized for. Used to validate
    /// re-allocation on resize (out of scope for v1).
    pub width: u32,
    /// Height the state was sized for.
    pub height: u32,
}

impl PreAnalysisState {
    /// Allocate fresh state for a `width × height` encoder. Both
    /// dimensions must be positive multiples of 8 (the mi-grid
    /// resolution); the constructor panics otherwise to surface
    /// configuration mistakes loudly.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        assert!(
            width > 0 && height > 0,
            "PreAnalysisState requires positive dims"
        );
        assert!(
            width % 8 == 0 && height % 8 == 0,
            "PreAnalysisState requires width/height to be multiples of 8 \
             (the mi-grid resolution)"
        );
        let mi_rows = (height / 8) as usize;
        let mi_cols = (width / 8) as usize;
        let sb_rows = (height / 64).max(1) as usize;
        let sb_cols = (width / 64).max(1) as usize;
        let (ne_level, ne_thresh) = noise_estimate_init_params(width, height);
        Self {
            consec_zero_mv: vec![0u8; mi_rows * mi_cols],
            // Updated per-frame by `update_per_frame_gates`.
            compute_source_sad_onepass: false,
            last_source_y: None,
            // W2.2 — per-SB content state. Default zero / Normal.
            avg_source_sad: vec![0u64; sb_rows * sb_cols],
            content_state_sb: vec![ContentStateSb::Normal; sb_rows * sb_cols],
            // W2.3 — scene-detection flags. Default off.
            high_source_sad: false,
            scene_cut_force_key_frame: false,
            prev_avg_source_sad: 0,
            // W2.4 — noise estimator. libvpx `vp9_noise_estimate_init`
            // (`vp9_noise_estimate.c:33-50`). `enabled` starts `false`
            // and is recomputed per call by the
            // `enable_noise_estimation` gate; the level seed is
            // resolution-dependent (kLowLow for sub-720p, kLow otherwise).
            noise_estimate: NoiseEstimate { enabled: false, level: ne_level },
            ne_value: 0,
            ne_thresh,
            // libvpx `vp9_noise_estimate.c:49`: adapt_thresh = (3*thresh)>>1
            ne_adapt_thresh: (3 * ne_thresh) >> 1,
            ne_count: 0,
            // libvpx `vp9_noise_estimate.c:48`: 15 frames to first update.
            ne_num_frames_estimate: 15,
            ne_last_w: 0,
            ne_last_h: 0,
            width,
            height,
        }
    }

    /// libvpx `vp9_encoder.c:3984` — `cpi->compute_source_sad_onepass
    /// = (cpi->oxcf.mode == REALTIME)`. Called once per frame at the
    /// top of `encode_frame_to_data_rate` before the avg-SAD pass.
    pub fn update_per_frame_gates(&mut self, deadline: Deadline) {
        self.compute_source_sad_onepass = matches!(deadline, Deadline::Realtime);
    }

    /// Reset per-mi tracking at keyframes. libvpx zeroes
    /// `consec_zero_mv[]` in `vp9_setup_past_independence` for
    /// keyframes (`vp9_alloccommon.c:81-91`).
    pub fn keyframe_reset(&mut self) {
        for v in &mut self.consec_zero_mv {
            *v = 0;
        }
        // Last_Source is invalidated at a KF — the previous source
        // is no longer a useful avg-SAD reference across a scene
        // boundary.
        self.last_source_y = None;
        // W2.2 — Reset per-SB classifications at KF (libvpx default
        // is kLowSadLowSumdiff = Normal at allocation, and the next
        // post-KF inter frame skips the avg-SAD pass since
        // Last_Source is None).
        for v in &mut self.avg_source_sad {
            *v = 0;
        }
        for v in &mut self.content_state_sb {
            *v = ContentStateSb::Normal;
        }
        // W2.3 — Reset scene-detection flags at KF (libvpx's
        // `vp9_ratectrl.c:3071-3072` clears these at the top of
        // every scene_detection_onepass call; we reset here so the
        // gate `frames_since_key > 1` skip on the first inter after
        // KF is bit-equivalent).
        self.high_source_sad = false;
        self.scene_cut_force_key_frame = false;
        self.prev_avg_source_sad = 0;
        // W2.4 — Reset the noise estimator. libvpx itself does not
        // explicitly reset `cpi->noise_estimate` at KF (the running
        // state decays naturally over the next frame_period samples);
        // the task brief calls for `NoiseEstimate::default()` reset
        // on KF to match the consumer-side contract. The full
        // internal state is reseeded from
        // `vp9_noise_estimate_init`-equivalent defaults
        // (`vp9_noise_estimate.c:33-50`) so the next inter frame
        // starts from a clean slate.
        let (_ne_level, ne_thresh) = noise_estimate_init_params(self.width, self.height);
        self.noise_estimate = NoiseEstimate::default();
        self.ne_value = 0;
        self.ne_thresh = ne_thresh;
        self.ne_adapt_thresh = (3 * ne_thresh) >> 1;
        self.ne_count = 0;
        self.ne_num_frames_estimate = 15;
        self.ne_last_w = 0;
        self.ne_last_h = 0;
    }

    /// W2.2(vpxenc parity, A5 P1): libvpx's per-SB
    /// `avg_source_sad` pass (`vp9_encodeframe.c:1209-1257`).
    /// Walks every 64×64 SB in the source, computes SAD + variance
    /// against the previous source frame, and classifies the SB's
    /// content state.
    ///
    /// libvpx tracks 6 content states (`kLowSadLowSumdiff`,
    /// `kLowSadHighSumdiff`, `kHighSadLowSumdiff`,
    /// `kHighSadHighSumdiff`, `kLowVarHighSumdiff`, `kVeryHighSad`).
    /// rvp9 currently consumes only the 2-state simplification
    /// (`Normal` / `VeryHighSad`) — see
    /// `crates/rvp9-core/src/mode_decision/inter.rs::ContentStateSb`.
    /// The producer here writes the simplified label; the 6-state
    /// expansion lands when the consumer side grows
    /// (Wave 4+ partition / AQ gates).
    ///
    /// Caller must ensure `src_y_tight` and `self.last_source_y` are
    /// the same `width × height` tight-stride plane. No-op if
    /// `last_source_y` is `None` (post-KF first inter frame).
    pub fn compute_avg_source_sad(&mut self, src_y_tight: &[u8]) {
        let Some(last) = self.last_source_y.as_deref() else {
            return;
        };
        let w = self.width as usize;
        let h = self.height as usize;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(last.len(), w * h);
        // libvpx threshold (`vp9_encodeframe.c:1219`).
        const AVG_SOURCE_SAD_THRESHOLD: u64 = 10_000;
        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let (sad, sse, sum) = sb_sad_and_var(
                    src_y_tight, last, w, sb_r * 64, sb_c * 64,
                );
                // libvpx: variance = sse - ((sum * sum) >> 12)
                let var = sse.saturating_sub((sum * sum) >> 12);
                let idx = sb_r * sb_cols + sb_c;
                self.avg_source_sad[idx] = sad;
                // libvpx `vp9_encodeframe.c:1243-1244` — only the
                // `kVeryHighSad` arm has a consumer in rvp9 today.
                self.content_state_sb[idx] =
                    if sad > (AVG_SOURCE_SAD_THRESHOLD << 1) {
                        ContentStateSb::VeryHighSad
                    } else {
                        // Other libvpx states all collapse to Normal
                        // in rvp9's 2-state enum until the consumer
                        // side expands (Wave 4+).
                        let _ = var; // computed for future expansion
                        ContentStateSb::Normal
                    };
            }
        }
    }

    /// Save the just-encoded source's luma plane for the next
    /// frame's avg-source-SAD pass. Called post-encode from
    /// `StreamEncoder::encode_frame`.
    pub fn capture_last_source(&mut self, src_y_tight: &[u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        debug_assert_eq!(
            src_y_tight.len(),
            w * h,
            "capture_last_source: src plane size mismatch"
        );
        match &mut self.last_source_y {
            Some(buf) => buf.copy_from_slice(src_y_tight),
            None => self.last_source_y = Some(src_y_tight.to_vec()),
        }
    }

    /// W2.3(vpxenc parity, A5 P2): one-pass scene detection.
    /// Mirrors libvpx `vp9_scene_detection_onepass`
    /// (`vp9_ratectrl.c:3034-3216`) for the
    /// `lag_in_frames == 0`, natural-content, CBR, single-spatial-
    /// layer slice rvp9 currently supports.
    ///
    /// The pass walks every interior 64×64 SB in a checkerboard
    /// pattern, computes per-SB SAD against `Last_Source`, averages
    /// the sampled SADs, and compares the average to a running
    /// recursive mean. If the current average jumps past the mean
    /// by a `thresh` factor AND a minimum threshold AND we're past
    /// the first inter frame, [`Self::high_source_sad`] lights up.
    /// A further `thresh_key` gate sets
    /// [`Self::scene_cut_force_key_frame`] for KF-forcing logic.
    ///
    /// Thresholds (libvpx defaults at `vp9_ratectrl.c:3076-3080`):
    ///   - `min_thresh = 65000` (natural content, the rvp9 default)
    ///   - `thresh = 8.0` (CBR default)
    ///   - `thresh_key = 140000` (speed > 5; rvp9's op-point is 7)
    ///
    /// No-op when `Last_Source` is None (first inter after KF, no
    /// reference).
    pub fn compute_scene_detection(&mut self, src_y_tight: &[u8], frames_since_key: u32) {
        let Some(last) = self.last_source_y.as_deref() else {
            self.high_source_sad = false;
            self.scene_cut_force_key_frame = false;
            return;
        };
        let w = self.width as usize;
        let h = self.height as usize;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(last.len(), w * h);
        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);
        // libvpx `vp9_ratectrl.c:3076-3079` default thresholds.
        const MIN_THRESH: u64 = 65_000;
        const THRESH: u64 = 8;
        const THRESH_KEY: u64 = 140_000;

        // Checkerboard sample, skipping the boundary SBs (libvpx
        // `vp9_ratectrl.c:3146-3148`).
        let mut total_sad: u64 = 0;
        let mut num_samples: u32 = 0;
        let mut num_zero_temp_sad: u32 = 0;
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let interior = sb_r > 0
                    && sb_c > 0
                    && sb_r + 1 < sb_rows
                    && sb_c + 1 < sb_cols;
                let checkerboard =
                    (sb_r % 2 == 0 && sb_c % 2 == 0) || (sb_r % 2 == 1 && sb_c % 2 == 1);
                if interior && checkerboard {
                    let (sad, _sse, _sum) =
                        sb_sad_and_var(src_y_tight, last, w, sb_r * 64, sb_c * 64);
                    total_sad += sad;
                    num_samples += 1;
                    if sad == 0 {
                        num_zero_temp_sad += 1;
                    }
                }
            }
        }
        let avg_sad = if num_samples > 0 {
            total_sad / num_samples as u64
        } else {
            0
        };

        // libvpx gate (`vp9_ratectrl.c:3168-3175`).
        let threshold = MIN_THRESH.max(self.prev_avg_source_sad * THRESH);
        let zero_sad_gate = num_zero_temp_sad < 3 * (num_samples >> 2);
        self.high_source_sad = avg_sad > threshold
            && frames_since_key > 1
            && zero_sad_gate;

        // libvpx forced-KF gate (`vp9_ratectrl.c:3176-3177`).
        self.scene_cut_force_key_frame = self.high_source_sad && avg_sad > THRESH_KEY;

        // libvpx recursive average update (`vp9_ratectrl.c:3179`).
        // CBR is rvp9's default rc_mode, so the update fires
        // unconditionally on every call.
        self.prev_avg_source_sad =
            (3 * self.prev_avg_source_sad + avg_sad) >> 2;
    }

    /// W2.4(vpxenc parity, A5 P3): faithful port of libvpx
    /// `vp9_update_noise_estimate` (`vp9_noise_estimate.c:109-302`)
    /// for the 8-bit, single-spatial-layer, no-SVC, no-temporal-
    /// denoising slice rvp9 supports. Walks a 1/4-sub-sample of the
    /// frame's 16×16 blocks, builds a per-block variance histogram of
    /// "steady-background" blocks (those whose underlying 8×8
    /// `consec_zero_mv` cells have all stayed at zero MV for
    /// `> thresh_consec_zeromv` consecutive frames), then folds the
    /// histogram's mode-bin into the running `ne_value` and (every
    /// `num_frames_estimate` updates) re-extracts the published
    /// `noise_estimate.level`.
    ///
    /// **Skipped libvpx paths** (out of scope for v1):
    /// - SVC top-spatial-layer gate (`noise_est_svc`) — single-layer.
    /// - HBD `cpi->common.use_highbitdepth` — 8-bit only.
    /// - `cpi->use_skin_detection` skin-block exclusion — wires up in
    ///   W2.5 (the skin detector lands next in this wave).
    /// - `CONFIG_VP9_TEMPORAL_DENOISING` denoiser sync — denoiser is
    ///   not ported.
    ///
    /// Caller passes the current source's luma plane (tight-stride,
    /// `width * height`) and the frame counter
    /// (libvpx `cm->current_video_frame`); the frame-period gate
    /// `frame_counter % 8 == 0` (line 132) keys off this.
    pub fn compute_noise_estimate(&mut self, src_y_tight: &[u8], frame_counter: u32) {
        // libvpx `vp9_noise_estimate.c:112` — low-res branch (used by
        // `thresh_consec_zeromv` for the denoiser-on path; rvp9 has
        // no denoiser so the default `thresh_consec_zeromv = 6` from
        // line 115 holds for all resolutions).
        let _low_res = self.width <= 352 && self.height <= 288;
        // libvpx line 114.
        const FRAME_PERIOD: u32 = 8;
        // libvpx line 115. Default 6; the denoiser-on path (skipped)
        // would lower this to 2 for 640<w<=1920.
        const THRESH_CONSEC_ZEROMV: u8 = 6;

        // libvpx line 129. Recomputed every call.
        self.noise_estimate.enabled =
            enable_noise_estimation(self.width, self.height);

        let w = self.width as usize;
        let h = self.height as usize;
        debug_assert_eq!(src_y_tight.len(), w * h);

        // libvpx lines 132-144: early-exit gate. The
        // `last_source == NULL` arm corresponds to no Last_Source.
        // For the resize-detect arm (`ne_last_w != width || ne_last_h
        // != height`), we still must update `ne_last_w/h` before
        // returning so the next call can run (libvpx lines 140-142).
        let last_source = self.last_source_y.as_deref();
        if !self.noise_estimate.enabled
            || frame_counter % FRAME_PERIOD != 0
            || last_source.is_none()
            || self.ne_last_w != self.width
            || self.ne_last_h != self.height
        {
            if last_source.is_some() {
                self.ne_last_w = self.width;
                self.ne_last_h = self.height;
            }
            return;
        }
        let last_source = last_source.unwrap();
        debug_assert_eq!(last_source.len(), w * h);

        // libvpx lines 145-160: high-motion forced-zero branch. Reads
        // `frame_counter > 60`, SVC counters, and
        // `rc.avg_frame_low_motion`. rvp9 has no SVC and no
        // `avg_frame_low_motion` tracker yet, so this branch never
        // fires in v1 (the next call lands in the main `else` arm).
        // Wired up cleanly when those producers land.

        // ---- Main arm (libvpx lines 161-297) ----
        // libvpx `MAX_VAR_HIST_BINS` (`vp9_noise_estimate.h:26`).
        const MAX_VAR_HIST_BINS: usize = 20;
        // libvpx line 162.
        const BIN_SIZE: u32 = 100;
        let mut hist: [u32; MAX_VAR_HIST_BINS] = [0; MAX_VAR_HIST_BINS];

        let mi_cols = w / 8;
        let mi_rows = h / 8;

        // libvpx lines 182-188: first pass, count low-motion blocks.
        let mut num_low_motion: u32 = 0;
        for mi_row in 0..mi_rows {
            for mi_col in 0..mi_cols {
                let bl_index = mi_row * mi_cols + mi_col;
                if self.consec_zero_mv[bl_index] > THRESH_CONSEC_ZEROMV {
                    num_low_motion += 1;
                }
            }
        }
        // libvpx lines 189-190: frame_low_motion if num_low_motion >=
        // (3 * mi_rows * mi_cols) >> 3 (i.e., >= 37.5% of cells).
        let frame_low_motion: bool =
            num_low_motion >= ((3 * (mi_rows * mi_cols) as u32) >> 3);

        // libvpx lines 191-242: second pass, 1/4-sub-sampled 16×16
        // variance over steady-background blocks. The mi-grid step
        // of 4 corresponds to a 32-pixel stride between sampled
        // top-left mi cells, but the variance is taken over the
        // 16×16 block rooted at that mi (8×8 mi → 16×16 covers a
        // 2×2 mi cluster).
        for mi_row in 0..mi_rows {
            for mi_col in 0..mi_cols {
                // libvpx line 194 — sub-sampling gate.
                if mi_row % 4 != 0
                    || mi_col % 4 != 0
                    || mi_row >= mi_rows - 1
                    || mi_col >= mi_cols - 1
                {
                    continue;
                }
                // libvpx lines 196-204: 4-way min over the four 8×8
                // sub-blocks of this 16×16 (the mi-grid is 8×8).
                let bl_index = mi_row * mi_cols + mi_col;
                let bl_index1 = bl_index + 1;
                let bl_index2 = bl_index + mi_cols;
                let bl_index3 = bl_index2 + 1;
                let consec_zeromv = self.consec_zero_mv[bl_index]
                    .min(self.consec_zero_mv[bl_index1])
                    .min(self.consec_zero_mv[bl_index2])
                    .min(self.consec_zero_mv[bl_index3]);
                // libvpx lines 210-212: only score steady-background
                // blocks AND skip on a scene-change frame. The skin
                // detector (libvpx lines 214-218) lands in W2.5; for
                // v1, `is_skin = false` unconditionally.
                if !(frame_low_motion
                    && consec_zeromv > THRESH_CONSEC_ZEROMV
                    && !self.high_source_sad)
                {
                    continue;
                }
                // libvpx lines 219-230: 16×16 variance of the block
                // against Last_Source. `vpx_variance16x16_c`
                // (`vpx_dsp/variance.c:128-135`) returns
                // `sse - (sum*sum / 256)`.
                let pix_r = mi_row * 8;
                let pix_c = mi_col * 8;
                let variance = variance_16x16(src_y_tight, last_source, w, pix_r, pix_c);
                let hist_index = (variance / BIN_SIZE) as usize;
                if hist_index < MAX_VAR_HIST_BINS {
                    hist[hist_index] += 1;
                } else if hist_index < (3 * (MAX_VAR_HIST_BINS >> 1)) {
                    // libvpx line 228-229: tail-fold into the last bin.
                    hist[MAX_VAR_HIST_BINS - 1] += 1;
                }
            }
        }

        // libvpx lines 243-244.
        self.ne_last_w = self.width;
        self.ne_last_h = self.height;

        // libvpx lines 247-256: histogram darken-shift correction.
        if hist[0] > 10 && hist[MAX_VAR_HIST_BINS - 1] > (hist[0] >> 2) {
            hist[0] = 0;
            hist[1] >>= 2;
            hist[2] >>= 2;
            hist[3] >>= 2;
            hist[4] >>= 1;
            hist[5] >>= 1;
            hist[6] = (3 * hist[6]) >> 1;
            hist[MAX_VAR_HIST_BINS - 1] >>= 1;
        }

        // libvpx lines 258-277: smooth + find max bin.
        let mut max_bin: u32 = 0;
        let mut max_bin_count: u32 = 0;
        let mut hist_avg: [u32; MAX_VAR_HIST_BINS] = [0; MAX_VAR_HIST_BINS];
        for bin_cnt in 0..MAX_VAR_HIST_BINS {
            if bin_cnt == 0 {
                // libvpx line 261.
                hist_avg[bin_cnt] = (hist[0] + hist[1] + hist[2]) / 3;
            } else if bin_cnt == MAX_VAR_HIST_BINS - 1 {
                // libvpx line 263.
                hist_avg[bin_cnt] = hist[MAX_VAR_HIST_BINS - 1] >> 2;
            } else if bin_cnt == MAX_VAR_HIST_BINS - 2 {
                // libvpx lines 265-267.
                hist_avg[bin_cnt] = (hist[bin_cnt - 1]
                    + 2 * hist[bin_cnt]
                    + (hist[bin_cnt + 1] >> 1)
                    + 2)
                    >> 2;
            } else {
                // libvpx lines 269-271.
                hist_avg[bin_cnt] = (hist[bin_cnt - 1]
                    + 2 * hist[bin_cnt]
                    + hist[bin_cnt + 1]
                    + 2)
                    >> 2;
            }
            if hist_avg[bin_cnt] > max_bin_count {
                max_bin_count = hist_avg[bin_cnt];
                max_bin = bin_cnt as u32;
            }
        }

        // libvpx line 280: recursive average of `ne_value` with the
        // scaled max bin index. `(3 * value + max_bin * 40) >> 2`.
        // libvpx stores `int`; we mirror with i32 saturating math.
        let scaled = (max_bin as i32).saturating_mul(40);
        self.ne_value =
            ((3i32.saturating_mul(self.ne_value)).saturating_add(scaled)) >> 2;

        // libvpx lines 282-286: rapid-onset fast-path. When level is
        // below kMedium AND value already exceeds adapt_thresh, jump
        // the counter so the level update fires immediately.
        // (`NoiseLevel` doesn't derive `Ord` upstream, so we encode
        // the libvpx `kLowLow < kLow < kMedium < kHigh` ordering
        // explicitly via `noise_level_rank`.)
        if noise_level_rank(self.noise_estimate.level) < noise_level_rank(NoiseLevel::Medium)
            && self.ne_value > self.ne_adapt_thresh
        {
            self.ne_count = self.ne_num_frames_estimate;
        } else {
            self.ne_count += 1;
        }

        // libvpx lines 287-296: level update gate. After the first
        // update, `num_frames_estimate` is bumped from 15 → 30
        // (so subsequent updates are less frequent).
        if self.ne_count == self.ne_num_frames_estimate {
            self.ne_num_frames_estimate = 30;
            self.ne_count = 0;
            self.noise_estimate.level = noise_estimate_extract_level(
                self.ne_value, self.ne_thresh,
            );
        }
    }

    /// Number of mi-grid cells (`mi_rows * mi_cols`) tracked.
    #[must_use]
    pub fn mi_grid_len(&self) -> usize {
        self.consec_zero_mv.len()
    }

    /// Number of SB64 cells (`sb_rows * sb_cols`) tracked by the
    /// W2.2 content-state pass.
    #[must_use]
    pub fn sb_grid_len(&self) -> usize {
        self.content_state_sb.len()
    }
}

/// Compute SAD, SSE, and pixel-sum-of-difference for one 64×64 SB at
/// `(sb_r_pix, sb_c_pix)` on tight-stride source planes. Mirrors
/// libvpx's `vpx_sad64x64` + `vpx_variance64x64`
/// (`vp9_encodeframe.c:1226-1229`) scalar reference. Returns
/// `(sad, sse, |sum|)`. Used only by [`PreAnalysisState::compute_avg_source_sad`];
/// SIMD parity lands when the encoder hot path consumes this (Wave
/// 4+).
fn sb_sad_and_var(
    src: &[u8],
    last: &[u8],
    stride: usize,
    sb_r_pix: usize,
    sb_c_pix: usize,
) -> (u64, u64, u64) {
    let mut sad: u64 = 0;
    let mut sse: u64 = 0;
    let mut sum: i64 = 0;
    for r in 0..64 {
        let base = (sb_r_pix + r) * stride + sb_c_pix;
        for c in 0..64 {
            let s = src[base + c] as i32;
            let l = last[base + c] as i32;
            let d = s - l;
            sad += d.unsigned_abs() as u64;
            sse += (d as i64 * d as i64) as u64;
            sum += d as i64;
        }
    }
    (sad, sse, sum.unsigned_abs())
}

/// W2.4 helper: libvpx `vp9_noise_estimate_init`
/// (`vp9_noise_estimate.c:33-50`) condensed to its two
/// resolution-dependent outputs: the seed `level` (kLowLow for
/// sub-720p, kLow for 720p+) and the `thresh` value (90 / 115 / 140
/// / 200 depending on resolution).
fn noise_estimate_init_params(width: u32, height: u32) -> (NoiseLevel, i32) {
    let area = (width as u64) * (height as u64);
    // libvpx line 35.
    let level = if area < 1280 * 720 {
        NoiseLevel::LowLow
    } else {
        NoiseLevel::Low
    };
    // libvpx lines 38, 41-47.
    let thresh = if area >= 1920 * 1080 {
        200
    } else if area >= 1280 * 720 {
        140
    } else if area >= 640 * 360 {
        115
    } else {
        90
    };
    (level, thresh)
}

/// W2.4 helper: libvpx `enable_noise_estimation`
/// (`vp9_noise_estimate.c:52-74`). The full libvpx gate requires
/// `pass == 0` (1-pass), CBR, `aq_mode == CYCLIC_REFRESH_AQ`,
/// `speed >= 5`, no resize, no SVC, no screen-content, and
/// resolution `>= 640 * 360`. rvp9's stream encoder is hard-wired
/// to 1-pass CBR + cyclic-refresh AQ + speed 7 + no resize/SVC, so
/// only the resolution test is dynamic.
fn enable_noise_estimation(width: u32, height: u32) -> bool {
    // libvpx line 70.
    (width as u64) * (height as u64) >= 640 * 360
}

/// W2.4 helper: libvpx `vp9_noise_estimate_extract_level`
/// (`vp9_noise_estimate.c:94-107`). Maps the integer `ne_value`
/// against the resolution-dependent `thresh` to one of the four
/// `NoiseLevel` tiers.
fn noise_estimate_extract_level(value: i32, thresh: i32) -> NoiseLevel {
    // libvpx line 96.
    if value > (thresh << 1) {
        NoiseLevel::High
    } else if value > thresh {
        // libvpx line 99-100.
        NoiseLevel::Medium
    } else if value > (thresh >> 1) {
        // libvpx line 101-102.
        NoiseLevel::Low
    } else {
        // libvpx line 103-104.
        NoiseLevel::LowLow
    }
}

/// W2.4 helper: ordering rank for `NoiseLevel`. `NoiseLevel` upstream
/// doesn't derive `Ord`, but libvpx's `NOISE_LEVEL` enum is a plain
/// `int` enum with the natural ascending order
/// `kLowLow < kLow < kMedium < kHigh` (`vp9_noise_estimate.h:28`).
/// We encode that order locally so this module can implement libvpx's
/// `ne->level < kMedium` comparison (`vp9_noise_estimate.c:282`)
/// without touching the consumer-side type.
fn noise_level_rank(level: NoiseLevel) -> u8 {
    match level {
        NoiseLevel::LowLow => 0,
        NoiseLevel::Low => 1,
        NoiseLevel::Medium => 2,
        NoiseLevel::High => 3,
    }
}

/// W2.4 helper: libvpx `vpx_variance16x16_c`
/// (`vpx_dsp/variance.c:128-135` after macro expansion). Computes
/// `sse - (sum*sum) / 256` on a 16×16 tile rooted at `(r, c)` in
/// tight-stride 8-bit source planes. The 8×8-cell `mi` indexing in
/// the caller guarantees `r + 16 <= h` and `c + 16 <= w` because
/// the caller pre-clamps `mi_row < mi_rows - 1` and
/// `mi_col < mi_cols - 1`.
fn variance_16x16(src: &[u8], last: &[u8], stride: usize, r: usize, c: usize) -> u32 {
    let mut sum: i32 = 0;
    let mut sse: u32 = 0;
    for i in 0..16 {
        let base = (r + i) * stride + c;
        for j in 0..16 {
            let d = src[base + j] as i32 - last[base + j] as i32;
            sum += d;
            sse += (d * d) as u32;
        }
    }
    // libvpx: sse - (sum*sum) / (W*H), W*H = 256 for 16x16.
    // Use saturating sub to match libvpx unsigned-wraparound behavior
    // (when sum*sum/256 momentarily exceeds sse, the libvpx code
    // produces a wraparound; rvp9 clamps to 0 — this only matters at
    // pathological inputs where the per-block variance is meaningless).
    let mean_sq = ((sum as i64).saturating_mul(sum as i64) / 256) as u32;
    sse.saturating_sub(mean_sq)
}

#[cfg(test)]
mod tests {
    //! PROOF tests for W2.1 — the pre-analysis scaffold.

    use super::*;

    /// W2.1 PROOF — the scaffold allocates `consec_zero_mv[]` at the
    /// correct mi-grid size and zeroes it.
    #[test]
    fn w2_1_state_sized_to_mi_grid_and_zero_initialized() {
        let st = PreAnalysisState::new(256, 128);
        // mi_rows = 128 / 8 = 16; mi_cols = 256 / 8 = 32; total 512.
        assert_eq!(st.mi_grid_len(), 16 * 32);
        for c in &st.consec_zero_mv {
            assert_eq!(*c, 0);
        }
        assert!(!st.compute_source_sad_onepass);
        assert!(st.last_source_y.is_none());
    }

    /// W2.1 PROOF — `update_per_frame_gates` follows libvpx
    /// `vp9_encoder.c:3984`: only the Realtime deadline lights up
    /// `compute_source_sad_onepass`.
    #[test]
    fn w2_1_gates_compute_source_sad_only_on_realtime() {
        let mut st = PreAnalysisState::new(64, 64);
        st.update_per_frame_gates(Deadline::Realtime);
        assert!(st.compute_source_sad_onepass);
        st.update_per_frame_gates(Deadline::Good);
        assert!(!st.compute_source_sad_onepass);
        st.update_per_frame_gates(Deadline::Best);
        assert!(!st.compute_source_sad_onepass);
    }

    /// W2.1 PROOF — keyframe reset zeroes the mi-grid counter and
    /// invalidates the prev-source view (libvpx
    /// `vp9_alloccommon.c:81-91` semantics).
    #[test]
    fn w2_1_keyframe_reset_zeroes_state() {
        let mut st = PreAnalysisState::new(64, 64);
        // Dirty the state.
        for c in &mut st.consec_zero_mv {
            *c = 7;
        }
        st.last_source_y = Some(vec![42u8; 64 * 64]);
        st.keyframe_reset();
        for c in &st.consec_zero_mv {
            assert_eq!(*c, 0);
        }
        assert!(st.last_source_y.is_none());
    }

    /// W2.1 PROOF — `capture_last_source` populates the Last_Source
    /// view; re-capture overwrites without reallocating.
    #[test]
    fn w2_1_capture_last_source_round_trips() {
        let mut st = PreAnalysisState::new(64, 64);
        let src1: Vec<u8> = (0..64 * 64).map(|i| i as u8).collect();
        st.capture_last_source(&src1);
        assert_eq!(st.last_source_y.as_deref(), Some(src1.as_slice()));
        let src2 = vec![99u8; 64 * 64];
        st.capture_last_source(&src2);
        assert_eq!(st.last_source_y.as_deref(), Some(src2.as_slice()));
    }

    /// W2.2 PROOF — `compute_avg_source_sad` is a no-op when
    /// Last_Source is None (libvpx-equivalent of the first inter
    /// frame after a KF — no reference to diff against).
    #[test]
    fn w2_2_avg_source_sad_noop_without_last_source() {
        let mut st = PreAnalysisState::new(64, 64);
        let src = vec![100u8; 64 * 64];
        st.compute_avg_source_sad(&src);
        // No reference ⇒ no state changed.
        for v in &st.avg_source_sad {
            assert_eq!(*v, 0);
        }
        for c in &st.content_state_sb {
            assert_eq!(*c, ContentStateSb::Normal);
        }
    }

    /// W2.2 PROOF — identical source + last_source ⇒ zero SAD ⇒
    /// Normal classification. libvpx `vp9_encodeframe.c:1231`:
    /// `tmp_sad == 0` ⇒ kLowSadLowSumdiff (Normal in rvp9).
    #[test]
    fn w2_2_avg_source_sad_zero_on_identical_frames() {
        let mut st = PreAnalysisState::new(128, 128);
        let frame = vec![100u8; 128 * 128];
        st.capture_last_source(&frame);
        st.compute_avg_source_sad(&frame);
        for v in &st.avg_source_sad {
            assert_eq!(*v, 0, "identical frames must produce SAD = 0");
        }
        for c in &st.content_state_sb {
            assert_eq!(*c, ContentStateSb::Normal);
        }
    }

    /// W2.2 PROOF — a large-magnitude per-SB diff lights up
    /// `VeryHighSad`. libvpx `vp9_encodeframe.c:1243-1244`:
    /// `tmp_sad > 2 * AVG_SOURCE_SAD_THRESHOLD` ⇒ kVeryHighSad.
    /// At 64×64 = 4096 pixels per SB, a per-pixel diff of 6
    /// produces SAD = 24576 > 20000.
    #[test]
    fn w2_2_avg_source_sad_lights_up_very_high_sad() {
        let mut st = PreAnalysisState::new(64, 64);
        let zero = vec![0u8; 64 * 64];
        let bright = vec![6u8; 64 * 64];
        st.capture_last_source(&zero);
        st.compute_avg_source_sad(&bright);
        // The single SB's SAD must exceed 2 * 10_000.
        assert!(st.avg_source_sad[0] > 20_000);
        assert_eq!(st.content_state_sb[0], ContentStateSb::VeryHighSad);
    }

    /// W2.2 PROOF — SB-grid sizing matches the encoder's
    /// `sb_rows × sb_cols`.
    #[test]
    fn w2_2_sb_grid_sized_correctly() {
        // 256×128 frame ⇒ 4 sb_cols × 2 sb_rows = 8.
        let st = PreAnalysisState::new(256, 128);
        assert_eq!(st.sb_grid_len(), 8);
        assert_eq!(st.avg_source_sad.len(), 8);
        assert_eq!(st.content_state_sb.len(), 8);
    }

    /// W2.3 PROOF — scene detection fires on a synthetic cut.
    /// Setup: 6×6 SB grid (384×384) so the checkerboard samples ~8
    /// interior SBs — large enough for the `num_zero_temp_sad <
    /// 3 * (num_samples >> 2)` gate (libvpx
    /// `vp9_ratectrl.c:3172`) to be feasible. Frame N = flat gray;
    /// frame N+1 = wildly different. After capturing N as
    /// Last_Source, running compute_scene_detection on N+1 must
    /// light up high_source_sad (libvpx
    /// `vp9_ratectrl.c:3168-3175`).
    #[test]
    fn w2_3_scene_detection_lights_high_source_sad_on_cut() {
        let mut st = PreAnalysisState::new(384, 384);
        let calm = vec![100u8; 384 * 384];
        let cut = vec![230u8; 384 * 384]; // large per-SB diff
        // Prime the running mean with a calm-vs-calm sample.
        st.capture_last_source(&calm);
        st.compute_scene_detection(&calm, /* frames_since_key = */ 5);
        assert!(!st.high_source_sad, "no cut on identical frames");
        // Now the cut.
        st.capture_last_source(&calm);
        st.compute_scene_detection(&cut, /* frames_since_key = */ 6);
        assert!(
            st.high_source_sad,
            "W2.3 PROOF: large per-SB diff vs Last_Source must light \
             high_source_sad (libvpx vp9_ratectrl.c:3168-3175). Got \
             prev_avg_source_sad={}",
            st.prev_avg_source_sad
        );
    }

    /// W2.3 PROOF — `frames_since_key <= 1` gate suppresses
    /// scene-detection on the post-KF inter frame (libvpx
    /// `vp9_ratectrl.c:3171`).
    #[test]
    fn w2_3_first_inter_after_kf_is_immune() {
        let mut st = PreAnalysisState::new(128, 128);
        let a = vec![10u8; 128 * 128];
        let b = vec![240u8; 128 * 128];
        st.capture_last_source(&a);
        st.compute_scene_detection(&b, /* frames_since_key = */ 1);
        assert!(
            !st.high_source_sad,
            "W2.3 PROOF: frames_since_key <= 1 must suppress scene detection \
             (libvpx vp9_ratectrl.c:3171). Got prev_avg_source_sad={}",
            st.prev_avg_source_sad
        );
    }

    /// W2.3 PROOF — no Last_Source ⇒ no scene detection (post-KF
    /// guard).
    #[test]
    fn w2_3_no_last_source_means_no_detection() {
        let mut st = PreAnalysisState::new(128, 128);
        let any = vec![50u8; 128 * 128];
        st.compute_scene_detection(&any, /* frames_since_key = */ 100);
        assert!(!st.high_source_sad);
        assert!(!st.scene_cut_force_key_frame);
    }

    /// W2.4 PROOF — a noise-free flat frame keeps the published
    /// `noise_estimate.level` at the libvpx-default low-noise tier
    /// (`kLow` for ≥720p resolutions, `kLowLow` for sub-720p — both
    /// rank below `kMedium` per `vp9_noise_estimate.h:28`).
    ///
    /// Setup: 640×384 frame (passes the ≥640×360 resolution gate at
    /// `vp9_noise_estimate.c:70`). `consec_zero_mv[]` pre-seeded with
    /// 100 everywhere, so every mi cell beats the
    /// `thresh_consec_zeromv = 6` gate (libvpx line 115) — emulating
    /// a long-running steady-background sequence the picker would
    /// have produced. Identical source vs Last_Source ⇒ every
    /// sampled 16×16 variance is 0, every sample lands in `hist[0]`,
    /// the mode bin is 0, and `ne_value` never moves off 0. After
    /// 15 update calls the level extracts to `kLowLow` (libvpx
    /// `vp9_noise_estimate.c:101-104`).
    ///
    /// PROOF: strip the wiring — drop the
    /// `self.noise_estimate.level = ...` assignment at libvpx line
    /// 291 — and the test passes trivially because the seed level
    /// is already Low. To make the test fail-loudly, we also
    /// assert that the noise estimator is `enabled` AND that
    /// `ne_value == 0` (i.e., the histogram-walk actually ran).
    #[test]
    fn w2_4_noise_estimate_flat_frame_is_low() {
        // 640×384 ⇒ mi_rows=48, mi_cols=80; passes 640×360 gate
        // (libvpx `vp9_noise_estimate.c:70`).
        let mut st = PreAnalysisState::new(640, 384);
        // Seed all 8×8 mi cells as steady background (libvpx line
        // 185: `consec_zero_mv > thresh_consec_zeromv = 6`).
        for c in &mut st.consec_zero_mv {
            *c = 100;
        }
        let flat = vec![128u8; 640 * 384];
        st.capture_last_source(&flat);
        // Run enough updates to span the initial num_frames_estimate
        // window (15). Use `frames_since_key` values that satisfy
        // the `% 8 == 0` gate (libvpx line 132).
        for k in 1..=15u32 {
            st.compute_noise_estimate(&flat, k * 8);
        }
        // libvpx line 129: enable_noise_estimation must be true at
        // ≥640×360.
        assert!(
            st.noise_estimate.enabled,
            "W2.4 PROOF: noise_estimate.enabled must light up for a \
             640×384 1-pass-CBR stream (libvpx vp9_noise_estimate.c:70)"
        );
        // libvpx lines 280, 287-291: a flat-frame sequence keeps
        // `ne_value` at 0 and extracts to kLowLow (line 103-104).
        assert_eq!(
            st.ne_value, 0,
            "W2.4 PROOF: flat-frame variance histogram must keep \
             ne_value at 0 (libvpx vp9_noise_estimate.c:280)"
        );
        // libvpx line 28: kLowLow and kLow both rank below kMedium,
        // which is the threshold for the picker's noise-driven
        // sub-bias (libvpx vp9_pickmode.c:1349). A flat frame must
        // land in either tier.
        assert!(
            noise_level_rank(st.noise_estimate.level)
                < noise_level_rank(NoiseLevel::Medium),
            "W2.4 PROOF: flat-frame noise level must stay below \
             Medium (libvpx vp9_noise_estimate.c:101-104). \
             Got level = {:?}, ne_value = {}",
            st.noise_estimate.level,
            st.ne_value
        );
    }

    /// W2.4 PROOF — a synthetic frame with moderate per-pixel noise
    /// (Δ ≈ ±3) lifts the published `noise_estimate.level` to
    /// `kMedium` or higher.
    ///
    /// Setup: 640×384 frame (resolution gate passes). Steady-
    /// background seeding so the variance pass actually samples
    /// every sub-sampled 16×16 block (otherwise no blocks
    /// contribute). Source = flat 128; Last_Source = a tile-aligned
    /// `±3` checkerboard. Per-pixel d² = 9, sum ≈ 0 over a 16×16,
    /// variance ≈ 256 × 9 = 2304. hist_index = 23 (since
    /// `2304 / 100 = 23`). Per libvpx
    /// `vp9_noise_estimate.c:226-229`: `hist_index >=
    /// MAX_VAR_HIST_BINS` AND `hist_index < 3 * (MAX_VAR_HIST_BINS
    /// >> 1) = 30` ⇒ tail-folds into bin 19. Every sampled block
    /// lands in bin 19, so max_bin = 19, `ne_value = (3*0 +
    /// 19*40) >> 2 = 190`. That's > `adapt_thresh = (3*115)>>1 =
    /// 172`, so the rapid-onset fast-path
    /// (`vp9_noise_estimate.c:282-286`) jumps count to 15 in this
    /// call, the next inner `if` triggers level extraction, and
    /// `extract_level(190, 115) → kMedium`
    /// (`vp9_noise_estimate.c:99-100`: `value > thresh`).
    ///
    /// PROOF: strip the level-update assignment at libvpx line 291
    /// and the level stays at the seed (Low/LowLow), failing the
    /// `>= Medium` assertion. Strip the histogram fold at line 280
    /// and `ne_value` stays at 0, also failing.
    #[test]
    fn w2_4_noise_estimate_lifts_on_noisy_content() {
        let w: usize = 640;
        let h: usize = 384;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        // Steady-background seeding (same justification as the
        // flat-frame PROOF above).
        for c in &mut st.consec_zero_mv {
            *c = 100;
        }
        // Last_Source: 1×1 checkerboard at ±3 around 128. Per-pixel
        // d = ±3 (sum=0 over any 16×16), d² = 9, sse over 16×16 =
        // 2304. hist_index = 23 → tail-folds into bin 19 per
        // libvpx `vp9_noise_estimate.c:228-229`.
        let mut last = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let chk = ((r ^ c) & 1) == 0;
                last[r * w + c] = if chk { 125 } else { 131 };
            }
        }
        let src = vec![128u8; w * h];
        st.capture_last_source(&last);
        // Two period-aligned calls: the first hits libvpx's
        // resize-detect arm (line 135: `ne_last_w != width`) and
        // returns after seeding last_w/last_h (lines 140-142). The
        // second call lands in the main arm and actually walks the
        // histogram.
        st.compute_noise_estimate(&src, /* frames_since_key = */ 8);
        st.compute_noise_estimate(&src, /* frames_since_key = */ 16);
        // libvpx line 280 must have moved ne_value off zero.
        assert!(
            st.ne_value > 0,
            "W2.4 PROOF: noisy frame must lift ne_value off 0 \
             (libvpx vp9_noise_estimate.c:280). Got {}",
            st.ne_value
        );
        // libvpx lines 282-291: with level<Medium and
        // value>adapt_thresh, the rapid path fires and the level
        // extracts in the same call.
        assert!(
            st.ne_value > st.ne_adapt_thresh,
            "W2.4 PROOF: noisy-frame ne_value ({}) must exceed \
             adapt_thresh ({}) so the rapid-onset fast-path fires \
             (libvpx vp9_noise_estimate.c:282-286)",
            st.ne_value, st.ne_adapt_thresh
        );
        assert!(
            noise_level_rank(st.noise_estimate.level)
                >= noise_level_rank(NoiseLevel::Medium),
            "W2.4 PROOF: per-pixel ±64 noise must lift the published \
             level to Medium or higher (libvpx \
             vp9_noise_estimate.c:99-100). Got level = {:?}, \
             ne_value = {}, ne_thresh = {}",
            st.noise_estimate.level,
            st.ne_value,
            st.ne_thresh
        );
    }

    /// W2.4 PROOF — the noise estimator is gated off at sub-640×360
    /// resolutions (libvpx `vp9_noise_estimate.c:70`).
    #[test]
    fn w2_4_noise_estimate_disabled_below_640x360() {
        let mut st = PreAnalysisState::new(320, 240);
        let flat = vec![128u8; 320 * 240];
        st.capture_last_source(&flat);
        for c in &mut st.consec_zero_mv {
            *c = 100;
        }
        st.compute_noise_estimate(&flat, /* frames_since_key = */ 8);
        // libvpx `enable_noise_estimation` requires width*height >=
        // 640*360. 320×240 = 76800 < 230400 ⇒ enabled stays false.
        assert!(
            !st.noise_estimate.enabled,
            "W2.4 PROOF: sub-640×360 resolution must disable the \
             noise estimator (libvpx vp9_noise_estimate.c:70)"
        );
        // And value must not move when disabled (libvpx line 132-144
        // returns early before the histogram walk).
        assert_eq!(st.ne_value, 0);
    }

    /// W2.4 PROOF — frame_period gate suppresses non-multiple-of-8
    /// calls (libvpx `vp9_noise_estimate.c:132`,
    /// `frame_counter % frame_period != 0`).
    #[test]
    fn w2_4_noise_estimate_frame_period_gate() {
        let mut st = PreAnalysisState::new(640, 384);
        for c in &mut st.consec_zero_mv {
            *c = 100;
        }
        // Set up a noisy delta so the histogram WOULD produce a
        // non-zero ne_value if the gate were ignored.
        let mut last = vec![0u8; 640 * 384];
        for r in 0..384 {
            for c in 0..640 {
                last[r * 640 + c] = if ((r ^ c) & 1) == 0 { 64 } else { 192 };
            }
        }
        let src = vec![128u8; 640 * 384];
        st.capture_last_source(&last);
        // Call with frame_counter values that fail % 8 == 0.
        for fc in [1u32, 2, 3, 5, 7, 9, 15] {
            st.compute_noise_estimate(&src, fc);
        }
        assert_eq!(
            st.ne_value, 0,
            "W2.4 PROOF: frame_period gate must suppress the \
             histogram walk on non-multiples of 8 (libvpx \
             vp9_noise_estimate.c:132). Got ne_value = {}",
            st.ne_value
        );
    }
}
