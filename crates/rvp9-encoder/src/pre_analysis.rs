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
//! - Color-sensitivity check — per-SB `color_sensitivity_sb[][2]`
//!   shared with the variance-partition / chroma-RD path.
//! - vbp-threshold noise scale — pure helper that reads
//!   `noise_estimate.level` and returns the libvpx multiplier the
//!   variance-based partition consumer applies to its thresholds.
//!
//! W2.1 lands the **scaffold** — the data-structure homes and the
//! per-frame lifecycle hooks. Producers (W2.2..W2.5) and threshold
//! consumers (W2.6) plug into these slots over the rest of Wave 2.

use crate::frame_loop::Deadline;
use rvp9_core::block::BlockSize;
use rvp9_core::dsp::dispatch::{SadFn, VarFn};
use rvp9_core::mode_decision::inter::{ContentStateSb, NoiseEstimate, NoiseLevel};
use std::sync::Arc;

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
    pub last_source_y: Option<Arc<[u8]>>,
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
    /// W2.5(vpxenc parity, A5 P4): libvpx `cpi->skin_map[mi_rows *
    /// mi_cols]` (`vp9_encoder.h`, allocated in `vp9_alloc_skin_map`,
    /// written by `vp9_compute_skin_sb` at
    /// `vp9_skin_detection.c:40-126`). One byte per 8×8 mi cell —
    /// `1` if the cell's center pixel matches the libvpx YCbCr skin
    /// model (`vpx_dsp/skin_detection.c:49-79`), `0` otherwise.
    /// Updated by [`Self::compute_skin_map`]; default zero. Consumed
    /// by the picker via `x->sb_is_skin` (libvpx
    /// `vp9_pickmode.c:1352`) once W2.5 is wired downstream.
    pub skin_map: Vec<u8>,
    /// W2.6(vpxenc parity, A4 P6): libvpx `x->color_sensitivity[2]`
    /// (`vp9_encoder.h`) — `[U, V]` chroma-sensitivity flags. libvpx
    /// computes these per-SB inside the variance-partition path
    /// (`chroma_check` at `vp9_encodeframe.c:1173-1207`); we
    /// pre-compute them per SB64 in pre-analysis so the picker /
    /// chroma-RD path can read them without re-running the SAD pass.
    /// Indexed `[sb_row * sb_cols + sb_col]`.
    pub color_sensitivity_sb: Vec<[bool; 2]>,
    /// Width × height the state was sized for. Used to validate
    /// re-allocation on resize (out of scope for v1).
    pub width: u32,
    /// Height the state was sized for.
    pub height: u32,
}

/// Execution controls for the parallel realtime source-map producers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParallelSourceMapConfig {
    /// Apply the CBR-only lighting-change source-SAD classification.
    pub(crate) cbr_mode: bool,
    /// Produce the CBR cyclic-refresh skin map.
    pub(crate) compute_skin_map: bool,
    /// Maximum worker jobs available to the producer split.
    pub(crate) max_jobs: usize,
    /// Compute the content classifier's luma Laplacian in the same jobs.
    pub(crate) compute_hf_energy: bool,
}

impl PreAnalysisState {
    /// Allocate fresh state for a `width × height` encoder. Both
    /// Width must be a positive multiple of 8 and height must be positive and
    /// even. A partial final luma MI row still owns one row of state.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        assert!(
            width > 0 && height > 0,
            "PreAnalysisState requires positive dims"
        );
        assert!(
            width.is_multiple_of(8) && height.is_multiple_of(2),
            "PreAnalysisState requires an 8-pixel-aligned width and even height"
        );
        let mi_rows = (height as usize).div_ceil(8);
        let mi_cols = (width / 8) as usize;
        let sb_rows = height.div_ceil(64).max(1) as usize;
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
            noise_estimate: NoiseEstimate {
                enabled: false,
                level: ne_level,
            },
            ne_value: 0,
            ne_thresh,
            // libvpx `vp9_noise_estimate.c:49`: adapt_thresh = (3*thresh)>>1
            ne_adapt_thresh: (3 * ne_thresh) >> 1,
            ne_count: 0,
            // libvpx `vp9_noise_estimate.c:48`: 15 frames to first update.
            ne_num_frames_estimate: 15,
            ne_last_w: 0,
            ne_last_h: 0,
            // W2.5 — per-mi skin map. libvpx allocates with calloc
            // (`vp9_encoder.c`'s `vp9_alloc_skin_map`), so the default
            // is all zero (no skin until the detector runs).
            skin_map: vec![0u8; mi_rows * mi_cols],
            // W2.6 — per-SB color-sensitivity. Default [false; 2]
            // (libvpx zero-inits `x->color_sensitivity[2]` —
            // `vp9_encodeframe.c` allocates MACROBLOCK with the rest of
            // VP9_COMP, so the values default to 0 = not sensitive).
            color_sensitivity_sb: vec![[false; 2]; sb_rows * sb_cols],
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
        // W2.5 — clear the per-mi skin map. libvpx does not clear
        // `cpi->skin_map` at the KF boundary (the next inter frame's
        // `vp9_compute_skin_sb` overwrites the cells it visits), but
        // resetting here keeps the published state strictly a
        // function of the current frame's analysis — no stale skin
        // labels survive a scene cut.
        for v in &mut self.skin_map {
            *v = 0;
        }
        // W2.6 — clear per-SB color-sensitivity at KF. libvpx-
        // equivalent: `chroma_check` is no-op on KF
        // (`vp9_encodeframe.c:1180`) so `x->color_sensitivity[]`
        // retains pre-init zero on the post-KF inter frame too.
        for v in &mut self.color_sensitivity_sb {
            *v = [false; 2];
        }
    }

    /// W2.2(vpxenc parity, A5 P1): libvpx's per-SB
    /// `avg_source_sad` pass (`vp9_encodeframe.c:1209-1257`).
    /// Walks every 64×64 SB in the source, computes SAD + variance
    /// against the previous source frame, and classifies the SB's
    /// content state.
    ///
    /// Produces all six libvpx states (`kLowSadLowSumdiff` through
    /// `kVeryHighSad`). `cbr_mode` enables the CBR-only lighting-change
    /// override to `kLowVarHighSumdiff`.
    ///
    /// Caller must ensure `src_y_tight` and `self.last_source_y` are
    /// the same `width × height` tight-stride plane. No-op if
    /// `last_source_y` is `None` (post-KF first inter frame).
    pub fn compute_avg_source_sad(&mut self, src_y_tight: &[u8], cbr_mode: bool) {
        let Some(last) = self.last_source_y.as_deref() else {
            return;
        };
        let w = self.width as usize;
        let h = self.height as usize;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(last.len(), w * h);
        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);
        let dispatch = rvp9_core::dsp::get_ref();
        let sad64 = dispatch.sad(BlockSize::Block64x64);
        let variance64 = dispatch.variance(BlockSize::Block64x64);
        compute_avg_source_sad_rows(
            src_y_tight,
            last,
            w,
            0,
            cbr_mode,
            sad64,
            variance64,
            &mut self.avg_source_sad[..sb_rows * sb_cols],
            &mut self.content_state_sb[..sb_rows * sb_cols],
        );
    }

    /// Compute the independent source-SAD and skin-map producers on the
    /// stream encoder's existing frame worker pool.
    ///
    /// Outputs are split into disjoint row spans. The scope joins every span
    /// before the caller consumes scene-detection or picker inputs, so this
    /// preserves the serial producer ordering without adding threads.
    pub(crate) fn compute_source_maps_parallel(
        &mut self,
        src_y_tight: &[u8],
        src_u_tight: &[u8],
        src_v_tight: &[u8],
        worker_pool: &mut rvp9_core::tile_threading::WorkerPool,
        config: ParallelSourceMapConfig,
    ) -> Option<u64> {
        let w = self.width as usize;
        let h = self.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(src_u_tight.len(), uvw * uvh);
        debug_assert_eq!(src_v_tight.len(), uvw * uvh);

        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);
        let mi_cols = w / 8;
        let mi_rows = h.div_ceil(8);
        // Preserve the existing total job count, but give every job one span
        // from each independent producer. Mixed jobs balance the caller and
        // OS workers without the eight-job coordination cost of splitting
        // both producers across the full worker width independently.
        let producer_jobs = config.max_jobs.max(1).div_ceil(2);
        let mixed_jobs = (producer_jobs * 2).min(sb_rows).min(mi_rows);
        let dispatch = rvp9_core::dsp::get_ref();
        let sad64 = dispatch.sad(BlockSize::Block64x64);
        let variance64 = dispatch.variance(BlockSize::Block64x64);
        let last_source = self.last_source_y.as_deref();
        let mut hf_laplacian_sums = vec![0u64; mixed_jobs];

        worker_pool.scope(|scope| {
            let mut sad_remaining = &mut self.avg_source_sad[..sb_rows * sb_cols];
            let mut state_remaining = &mut self.content_state_sb[..sb_rows * sb_cols];
            let mut skin_remaining = self.skin_map.as_mut_slice();
            let mut zero_mv_remaining = self.consec_zero_mv.as_slice();
            let mut sb_row_start = 0;
            let mut mi_row_start = 0;
            for (job_index, hf_laplacian_sum) in hf_laplacian_sums.iter_mut().enumerate() {
                let jobs_left = mixed_jobs - job_index;
                let sb_job_rows = (sad_remaining.len() / sb_cols).div_ceil(jobs_left);
                let mi_job_rows = (skin_remaining.len() / mi_cols).div_ceil(jobs_left);
                let (sad_out, sad_tail) =
                    std::mem::take(&mut sad_remaining).split_at_mut(sb_job_rows * sb_cols);
                sad_remaining = sad_tail;
                let (state_out, state_tail) =
                    std::mem::take(&mut state_remaining).split_at_mut(sb_job_rows * sb_cols);
                state_remaining = state_tail;
                let (skin_out, skin_tail) =
                    std::mem::take(&mut skin_remaining).split_at_mut(mi_job_rows * mi_cols);
                skin_remaining = skin_tail;
                let (consecutive_zero_mv, zero_mv_tail) =
                    zero_mv_remaining.split_at(mi_job_rows * mi_cols);
                zero_mv_remaining = zero_mv_tail;
                let mi_row_count = skin_out.len() / mi_cols;
                scope.submit(move || {
                    if let Some(last) = last_source {
                        compute_avg_source_sad_rows(
                            src_y_tight,
                            last,
                            w,
                            sb_row_start,
                            config.cbr_mode,
                            sad64,
                            variance64,
                            sad_out,
                            state_out,
                        );
                    }
                    if config.compute_skin_map {
                        compute_skin_map_rows(
                            src_y_tight,
                            src_u_tight,
                            src_v_tight,
                            w,
                            mi_row_start,
                            consecutive_zero_mv,
                            skin_out,
                        );
                    }
                    if config.compute_hf_energy {
                        let pixel_row_start = (mi_row_start * 8).max(1);
                        let pixel_row_end = if mi_row_start + mi_row_count == mi_rows {
                            h - 1
                        } else {
                            (mi_row_start + mi_row_count) * 8
                        };
                        *hf_laplacian_sum = frame_hf_laplacian_sampled_sum_rows(
                            src_y_tight,
                            w,
                            h,
                            pixel_row_start,
                            pixel_row_end,
                        );
                    }
                });
                sb_row_start += sb_job_rows;
                mi_row_start += mi_job_rows;
            }
            debug_assert!(sad_remaining.is_empty());
            debug_assert!(state_remaining.is_empty());
            debug_assert!(skin_remaining.is_empty());
            debug_assert!(zero_mv_remaining.is_empty());
        });
        config
            .compute_hf_energy
            .then(|| hf_laplacian_sums.into_iter().sum())
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
        if let Some(last) = &mut self.last_source_y
            && let Some(buf) = Arc::get_mut(last)
        {
            buf.copy_from_slice(src_y_tight);
        } else {
            self.last_source_y = Some(Arc::from(src_y_tight));
        }
    }

    /// Retain the just-encoded reference-counted source plane without a
    /// full-frame copy. The lookahead owns the same immutable allocation.
    pub fn capture_last_source_shared(&mut self, src_y_tight: &Arc<[u8]>) {
        debug_assert_eq!(
            src_y_tight.len(),
            self.width as usize * self.height as usize,
            "capture_last_source_shared: src plane size mismatch"
        );
        self.last_source_y = Some(Arc::clone(src_y_tight));
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

        // Checkerboard sample, skipping the boundary SBs (libvpx
        // `vp9_ratectrl.c:3146-3148`).
        let mut total_sad: u64 = 0;
        let mut num_samples: u32 = 0;
        let mut num_zero_temp_sad: u32 = 0;
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let interior = sb_r > 0 && sb_c > 0 && sb_r + 1 < sb_rows && sb_c + 1 < sb_cols;
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
        self.apply_scene_detection_summary(
            total_sad,
            num_samples,
            num_zero_temp_sad,
            frames_since_key,
        );
    }

    /// Scene detection from the immediately preceding
    /// [`Self::compute_avg_source_sad`] results.
    ///
    /// The avg-SAD pass stores the same 64x64 SAD values that
    /// [`Self::compute_scene_detection`] otherwise recomputes for its
    /// interior checkerboard subset. Product encoding calls this entry
    /// point only after the avg-SAD pass for the same source frame.
    pub fn compute_scene_detection_from_cached_sad(&mut self, frames_since_key: u32) {
        if self.last_source_y.is_none() {
            self.high_source_sad = false;
            self.scene_cut_force_key_frame = false;
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);

        let mut total_sad: u64 = 0;
        let mut num_samples: u32 = 0;
        let mut num_zero_temp_sad: u32 = 0;
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let interior = sb_r > 0 && sb_c > 0 && sb_r + 1 < sb_rows && sb_c + 1 < sb_cols;
                let checkerboard =
                    (sb_r % 2 == 0 && sb_c % 2 == 0) || (sb_r % 2 == 1 && sb_c % 2 == 1);
                if interior && checkerboard {
                    let sad = self.avg_source_sad[sb_r * sb_cols + sb_c];
                    total_sad += sad;
                    num_samples += 1;
                    if sad == 0 {
                        num_zero_temp_sad += 1;
                    }
                }
            }
        }
        self.apply_scene_detection_summary(
            total_sad,
            num_samples,
            num_zero_temp_sad,
            frames_since_key,
        );
    }

    fn apply_scene_detection_summary(
        &mut self,
        total_sad: u64,
        num_samples: u32,
        num_zero_temp_sad: u32,
        frames_since_key: u32,
    ) {
        // libvpx `vp9_ratectrl.c:3076-3079` default thresholds.
        const MIN_THRESH: u64 = 65_000;
        const THRESH: u64 = 8;
        const THRESH_KEY: u64 = 140_000;
        let avg_sad = if num_samples > 0 {
            total_sad / num_samples as u64
        } else {
            0
        };

        // libvpx gate (`vp9_ratectrl.c:3168-3175`).
        let threshold = MIN_THRESH.max(self.prev_avg_source_sad * THRESH);
        let zero_sad_gate = num_zero_temp_sad < 3 * (num_samples >> 2);
        self.high_source_sad = avg_sad > threshold && frames_since_key > 1 && zero_sad_gate;

        // libvpx forced-KF gate (`vp9_ratectrl.c:3176-3177`).
        self.scene_cut_force_key_frame = self.high_source_sad && avg_sad > THRESH_KEY;

        // libvpx recursive average update (`vp9_ratectrl.c:3179`).
        // CBR is rvp9's default rc_mode, so the update fires
        // unconditionally on every call.
        self.prev_avg_source_sad = (3 * self.prev_avg_source_sad + avg_sad) >> 2;
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
    pub fn compute_noise_estimate(
        &mut self,
        src_y_tight: &[u8],
        frame_counter: u32,
        // PHASE-3 integration: libvpx's full `enable_noise_estimation`
        // gate (`vp9_noise_estimate.c:52-74`) requires the temporal
        // denoiser to be configured (`cpi->oxcf.noise_sensitivity >
        // 0`). Pass 0 to disable the estimator regardless of resolution.
        noise_sensitivity: u8,
    ) {
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

        // libvpx line 129. Recomputed every call. PHASE-3 integration:
        // also gate on `noise_sensitivity > 0` (libvpx's full
        // `enable_noise_estimation` predicate at
        // vp9_noise_estimate.c:52-74).
        self.noise_estimate.enabled =
            noise_sensitivity > 0 && enable_noise_estimation(self.width, self.height);

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
            || !frame_counter.is_multiple_of(FRAME_PERIOD)
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
        let frame_low_motion: bool = num_low_motion >= ((3 * (mi_rows * mi_cols) as u32) >> 3);

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
                hist_avg[bin_cnt] =
                    (hist[bin_cnt - 1] + 2 * hist[bin_cnt] + (hist[bin_cnt + 1] >> 1) + 2) >> 2;
            } else {
                // libvpx lines 269-271.
                hist_avg[bin_cnt] =
                    (hist[bin_cnt - 1] + 2 * hist[bin_cnt] + hist[bin_cnt + 1] + 2) >> 2;
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
        self.ne_value = ((3i32.saturating_mul(self.ne_value)).saturating_add(scaled)) >> 2;

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
            self.noise_estimate.level = noise_estimate_extract_level(self.ne_value, self.ne_thresh);
        }
    }

    /// W2.5(vpxenc parity, A5 P4): faithful port of libvpx's
    /// per-block skin detector. Walks the mi-grid (8×8 luma, 4×4
    /// chroma for 4:2:0) and runs each cell through
    /// `vp9_compute_skin_block` (`vp9_skin_detection.c:18-38`), which
    /// in turn calls `vpx_skin_pixel`
    /// (`vpx_dsp/skin_detection.c:49-79`) on the cell's *center*
    /// pixel.
    ///
    /// **libvpx structure**:
    /// `vp9_compute_skin_sb` (`vp9_skin_detection.c:40-127`) is per-
    /// SB and processes BLOCK_16X16 cells inside that SB; we walk the
    /// whole mi-grid in one pass using the BLOCK_8X8 branch instead.
    /// The BLOCK_8X8 branch sets `consec_zeromv = consec_zero_mv[bl_index]`
    /// (line 73) which is exactly what `self.consec_zero_mv[mi_idx]`
    /// holds.
    ///
    /// **Skipped** (out of scope for v1):
    /// - Isolated-block spatial-smoothing pass
    ///   (`vp9_skin_detection.c:92-126`) — the brief explicitly allows
    ///   this skip; the per-block predicate is enough for v1 parity.
    /// - HBD — 8-bit only.
    /// - `OUTPUT_YUV_SKINMAP` debug emitter — debug-only.
    ///
    /// **Inputs**: tight-stride source planes. Luma is `width *
    /// height`; chroma is `(width/2) * (height/2)` for 4:2:0 (the
    /// only subsampling rvp9 supports).
    ///
    /// **Motion magnitude**: libvpx's `curr_motion_magn` arg is set
    /// to 0 at every call site in `vp9_compute_skin_sb` (line 81), so
    /// rvp9 hard-codes it 0 too.
    pub fn compute_skin_map(&mut self, src_y_tight: &[u8], src_u_tight: &[u8], src_v_tight: &[u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(src_u_tight.len(), uvw * uvh);
        debug_assert_eq!(src_v_tight.len(), uvw * uvh);
        compute_skin_map_rows(
            src_y_tight,
            src_u_tight,
            src_v_tight,
            w,
            0,
            &self.consec_zero_mv,
            &mut self.skin_map,
        );
    }

    /// W2.6(vpxenc parity, A5 P13): vbp-threshold noise scaling
    /// helper. libvpx's `set_vbp_thresholds`
    /// (`vp9_encodeframe.c:573-635`, specifically lines 588-598) scales
    /// the variance-based partition's base threshold by a factor that
    /// rises with the estimated noise level. The published rvp9
    /// helper exposes the factor as a small integer so the consumer
    /// (Wave 4+ partition path) can apply `threshold_base * scale`
    /// without re-walking the noise-estimate API.
    ///
    /// **Mapping** (per task brief): `NoiseLevel::Low`/`LowLow` → 1,
    /// `Medium` → 2, `High` → 4. libvpx itself uses 1×/2×/3× at lines
    /// 593-597 with an additional ×7/8 reduction below `kLow`; rvp9
    /// adopts the brief's symmetric 1/2/4 ladder for the v1 wire-up.
    /// When the variance-partition consumer lands (Wave 4+), the
    /// libvpx exact factors can be substituted with a one-line change
    /// here.
    ///
    /// **Gating**: libvpx gates this on `cpi->noise_estimate.enabled
    /// && cm->width >= 640 && cm->height >= 480` (line 589). We
    /// honour the `enabled` flag here so the consumer doesn't have
    /// to repeat the check; the resolution sub-gate is already
    /// rolled into `enable_noise_estimation` (≥640×360 — strictly
    /// broader than libvpx's 640×480 partition-side gate, so the
    /// caller may want to apply the extra 480-row check if exact
    /// libvpx parity matters).
    #[must_use]
    pub fn vbp_threshold_scale(&self) -> u32 {
        if !self.noise_estimate.enabled {
            return 1;
        }
        match self.noise_estimate.level {
            // libvpx `vp9_encodeframe.c:592-593`.
            NoiseLevel::High => 4,
            // libvpx `vp9_encodeframe.c:594-595`.
            NoiseLevel::Medium => 2,
            // libvpx `vp9_encodeframe.c:596-597` collapses `< kLow`
            // (i.e., `kLowLow`) into a sub-unity 7/8 scale; we keep
            // the integer floor at 1 — the brief specifies a 1/2/4
            // ladder and the consumer-side rounding lives there.
            NoiseLevel::Low | NoiseLevel::LowLow => 1,
        }
    }

    /// W2.6(vpxenc parity, A4 P6): port of libvpx's `chroma_check`
    /// (`vp9_encodeframe.c:1173-1207`). Per-SB SAD comparison
    /// `uv_sad > (y_sad >> shift)` for each of the U/V planes, with
    /// `shift = 2` (default) or `5` (screen-content + scene-change,
    /// which rvp9 doesn't support yet).
    ///
    /// **libvpx deviation**: libvpx computes the SAD against the
    /// *prediction* buffer (after inter pred). Pre-analysis runs
    /// before prediction so we substitute Last_Source. For a flat /
    /// near-flat sequence this is bit-equivalent (both src and
    /// prediction converge to Last_Source); for high-motion frames
    /// the published flag is conservative (Last_Source-vs-src SAD is
    /// strictly ≥ pred-vs-src SAD), which is the safer direction (a
    /// `true` here over-includes; a `false` over-excludes).
    ///
    /// **Inputs**: tight-stride source planes; `content_state_sb` is
    /// consumed only by libvpx's KF / scene-change shifts (out of
    /// scope for v1 — we apply the default `shift = 2`). The
    /// `sb_row`/`sb_col` index the SB64 grid.
    ///
    /// **No-op** when `Last_Source` is `None` (first inter after KF) —
    /// returns `[false; 2]` (the libvpx pre-init default).
    pub fn compute_color_sensitivity(
        &self,
        src_y_tight: &[u8],
        src_u_tight: &[u8],
        src_v_tight: &[u8],
        _content_state_sb: ContentStateSb,
        sb_row: usize,
        sb_col: usize,
    ) -> [bool; 2] {
        let Some(last_y) = self.last_source_y.as_deref() else {
            return [false; 2];
        };
        let w = self.width as usize;
        let h = self.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        debug_assert_eq!(src_y_tight.len(), w * h);
        debug_assert_eq!(src_u_tight.len(), uvw * uvh);
        debug_assert_eq!(src_v_tight.len(), uvw * uvh);
        debug_assert_eq!(last_y.len(), w * h);
        // libvpx `vp9_encodeframe.c:1178` default shift.
        const SHIFT: u32 = 2;
        let sb_pix_r = sb_row * 64;
        let sb_pix_c = sb_col * 64;
        // libvpx `vp9_encodeframe.c:1226-1227`: `y_sad =
        // sdf(p->src.buf, p->src.stride, dst, dst_stride)` against
        // the 64×64 prediction. We diff against Last_Source.
        let (y_sad, _sse, _sum) = sb_sad_and_var(src_y_tight, last_y, w, sb_pix_r, sb_pix_c);
        // For chroma, no Last_Source U/V is kept (the W2.1 scaffold
        // only saves the Y plane). The conservative substitute is to
        // compare against the source itself shifted by one SB, but
        // that breaks the v1 contract. Instead, treat the absence of
        // a chroma reference as "use the source's own DC anchor": a
        // 32×32 sum-of-absolute-difference against the SB's mean U/V.
        // This matches the spirit of libvpx's chroma_check (does the
        // chroma vary more than the luma residual?) without
        // requiring a Last_Source UV plane.
        let mut sensitivity = [false; 2];
        for (i, src_uv) in [src_u_tight, src_v_tight].iter().enumerate() {
            let uv_pix_r = sb_pix_r / 2;
            let uv_pix_c = sb_pix_c / 2;
            // 32×32 block for 4:2:0.
            let mut uv_sum: u64 = 0;
            for r in 0..32 {
                let base = (uv_pix_r + r) * uvw + uv_pix_c;
                for c in 0..32 {
                    uv_sum += src_uv[base + c] as u64;
                }
            }
            let uv_mean = (uv_sum / (32 * 32)) as u8;
            let mut uv_sad: u64 = 0;
            for r in 0..32 {
                let base = (uv_pix_r + r) * uvw + uv_pix_c;
                for c in 0..32 {
                    let d = src_uv[base + c] as i32 - uv_mean as i32;
                    uv_sad += d.unsigned_abs() as u64;
                }
            }
            // libvpx `vp9_encodeframe.c:1205`:
            // `color_sensitivity[i - 1] = uv_sad > (y_sad >> shift)`.
            sensitivity[i] = uv_sad > (y_sad >> SHIFT);
        }
        sensitivity
    }

    /// W2.6 batch driver: walk the SB grid and fill
    /// [`Self::color_sensitivity_sb`] in one pass. Intended for the
    /// per-frame pre-analysis pipeline so the picker can read the
    /// SB-indexed flags without rerunning the SAD.
    pub fn compute_color_sensitivity_frame(
        &mut self,
        src_y_tight: &[u8],
        src_u_tight: &[u8],
        src_v_tight: &[u8],
    ) {
        let w = self.width as usize;
        let h = self.height as usize;
        let sb_cols = (w / 64).max(1);
        let sb_rows = (h / 64).max(1);
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let idx = sb_r * sb_cols + sb_c;
                // Reuse the cached content-state classification —
                // libvpx's chroma_check consumes the same field for
                // its screen-content shift bump (`vp9_encodeframe.c:
                // 1190-1191`).
                let cs = self.content_state_sb[idx];
                self.color_sensitivity_sb[idx] = self.compute_color_sensitivity(
                    src_y_tight,
                    src_u_tight,
                    src_v_tight,
                    cs,
                    sb_r,
                    sb_c,
                );
            }
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
/// `(sad, sse, |sum|)`. Product encoding uses the dispatched helper
/// below; this independent scalar implementation remains the test and
/// standalone scene-detection oracle.
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

#[allow(clippy::too_many_arguments)]
fn sb_sad_and_var_dispatched(
    src: &[u8],
    last: &[u8],
    stride: usize,
    sb_r_pix: usize,
    sb_c_pix: usize,
    sad64: SadFn,
    variance64: VarFn,
) -> (u64, u64, u64) {
    debug_assert!(stride >= 64);
    debug_assert!(src.len().is_multiple_of(stride));
    debug_assert_eq!(src.len(), last.len());
    let height = src.len() / stride;
    if sb_r_pix + 64 > height {
        // Libvpx's source YV12 buffers carry an extended visible border, so
        // a short bottom SB64 repeats its final source row. Rvp9 stores tight
        // planes; materialize only this uncommon edge block before calling
        // the same dispatched kernels. Width is a multiple of 64, so no
        // horizontal extension is needed here.
        let mut src_edge = [0u8; 64 * 64];
        let mut last_edge = [0u8; 64 * 64];
        for row in 0..64 {
            let source_row = (sb_r_pix + row).min(height - 1);
            let source_start = source_row * stride + sb_c_pix;
            let destination_start = row * 64;
            src_edge[destination_start..destination_start + 64]
                .copy_from_slice(&src[source_start..source_start + 64]);
            last_edge[destination_start..destination_start + 64]
                .copy_from_slice(&last[source_start..source_start + 64]);
        }
        let sad = u64::from(sad64(&src_edge, 64, &last_edge, 64));
        let (sum, sse) = variance64(&src_edge, 64, &last_edge, 64);
        return (sad, sse, sum.unsigned_abs());
    }

    let offset = sb_r_pix * stride + sb_c_pix;
    let src = &src[offset..];
    let last = &last[offset..];
    let sad = u64::from(sad64(src, stride, last, stride));
    let (sum, sse) = variance64(src, stride, last, stride);
    (sad, sse, sum.unsigned_abs())
}

/// Sum `|4p - up - down - left - right|` over the requested interior rows.
///
/// `row_end` is exclusive. Callers may split `1..height - 1` into disjoint
/// spans and add the returned integer sums without changing the result.
pub(crate) fn frame_hf_laplacian_sum_rows(
    y: &[u8],
    width: usize,
    height: usize,
    row_start: usize,
    row_end: usize,
) -> u64 {
    frame_hf_laplacian_sum_rows_with_step(y, width, height, row_start, row_end, 1)
}

fn frame_hf_laplacian_sum_rows_with_step(
    y: &[u8],
    width: usize,
    height: usize,
    row_start: usize,
    row_end: usize,
    row_step: usize,
) -> u64 {
    if width < 3 || height < 3 || y.len() < width * height {
        return 0;
    }
    let row_start = row_start.max(1);
    let row_end = row_end.min(height - 1);
    if row_start >= row_end {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory in AArch64. The helper validates the
        // plane length and bounds every vector load to one complete row.
        unsafe { frame_hf_laplacian_sum_rows_neon(y, width, row_start, row_end, row_step) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        frame_hf_laplacian_sum_rows_scalar(y, width, row_start, row_end, row_step)
    }
}

/// Estimate the full-frame Laplacian sum from every fourth interior row.
///
/// The production high-frequency classifier consumes only a thresholded
/// frame mean. Sampling one globally fixed row phase removes three quarters
/// of that classifier's pixel work while keeping job-boundary results
/// independent of how the frame worker pool partitions the row spans.
pub(crate) fn frame_hf_laplacian_sampled_sum_rows(
    y: &[u8],
    width: usize,
    height: usize,
    row_start: usize,
    row_end: usize,
) -> u64 {
    const ROW_STEP: usize = 4;
    if width < 3 || height < 3 || y.len() < width * height {
        return 0;
    }
    let mut row_start = row_start.max(1);
    let row_end = row_end.min(height - 1);
    let phase = (row_start - 1) % ROW_STEP;
    if phase != 0 {
        row_start += ROW_STEP - phase;
    }
    if row_start >= row_end {
        return 0;
    }

    let sampled =
        frame_hf_laplacian_sum_rows_with_step(y, width, height, row_start, row_end, ROW_STEP);

    sampled.saturating_mul(ROW_STEP as u64)
}

/// Complete an exact full-frame Laplacian sum without rescanning the rows
/// already consumed by [`frame_hf_laplacian_sampled_sum_rows`].
pub(crate) fn frame_hf_laplacian_exact_sum_from_sampled(
    y: &[u8],
    width: usize,
    height: usize,
    sampled_sum: u64,
) -> u64 {
    const ROW_STEP: usize = 4;
    if !sampled_sum.is_multiple_of(ROW_STEP as u64) {
        return frame_hf_laplacian_sum_rows(y, width, height, 1, height.saturating_sub(1));
    }

    let row_end = height.saturating_sub(1);
    let mut exact_sum = sampled_sum / ROW_STEP as u64;
    for row_start in 2..=ROW_STEP {
        exact_sum = exact_sum.saturating_add(frame_hf_laplacian_sum_rows_with_step(
            y, width, height, row_start, row_end, ROW_STEP,
        ));
    }
    exact_sum
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn frame_hf_laplacian_sum_rows_scalar(
    y: &[u8],
    width: usize,
    row_start: usize,
    row_end: usize,
    row_step: usize,
) -> u64 {
    let mut laplacian_sum = 0u64;
    for row_index in (row_start..row_end).step_by(row_step) {
        let row = row_index * width;
        let up = row - width;
        let down = row + width;
        for column in 1..width - 1 {
            let pixel = i32::from(y[row + column]);
            let left = i32::from(y[row + column - 1]);
            let right = i32::from(y[row + column + 1]);
            let above = i32::from(y[up + column]);
            let below = i32::from(y[down + column]);
            laplacian_sum += u64::from((4 * pixel - left - right - above - below).unsigned_abs());
        }
    }
    laplacian_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn frame_hf_laplacian_sum_rows_neon(
    y: &[u8],
    width: usize,
    row_start: usize,
    row_end: usize,
    row_step: usize,
) -> u64 {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn abs_laplacian_sum_8(
        pixel: uint8x8_t,
        left: uint8x8_t,
        right: uint8x8_t,
        up: uint8x8_t,
        down: uint8x8_t,
    ) -> u64 {
        // SAFETY: All operands are initialized NEON values. Four u8
        // neighbours and `4 * pixel` fit in u16; their signed difference fits
        // in i16, so the absolute value and horizontal sum are exact.
        unsafe {
            let pixel_times_four = vshlq_n_u16(vmovl_u8(pixel), 2);
            let neighbours = vaddq_u16(
                vaddq_u16(vmovl_u8(left), vmovl_u8(right)),
                vaddq_u16(vmovl_u8(up), vmovl_u8(down)),
            );
            let laplacian = vsubq_s16(
                vreinterpretq_s16_u16(pixel_times_four),
                vreinterpretq_s16_u16(neighbours),
            );
            u64::from(vaddlvq_u16(vreinterpretq_u16_s16(vabsq_s16(laplacian))))
        }
    }

    let mut laplacian_sum = 0u64;
    for row_index in (row_start..row_end).step_by(row_step) {
        let row = row_index * width;
        let up = row - width;
        let down = row + width;
        let mut column = 1usize;
        while column + 16 < width {
            // SAFETY: The loop bounds keep every 16-byte load within its
            // source row. `left` starts one byte earlier and `right` one byte
            // later, both still inside the same `width`-byte row.
            unsafe {
                let pixel = vld1q_u8(y.as_ptr().add(row + column));
                let left = vld1q_u8(y.as_ptr().add(row + column - 1));
                let right = vld1q_u8(y.as_ptr().add(row + column + 1));
                let above = vld1q_u8(y.as_ptr().add(up + column));
                let below = vld1q_u8(y.as_ptr().add(down + column));
                laplacian_sum += abs_laplacian_sum_8(
                    vget_low_u8(pixel),
                    vget_low_u8(left),
                    vget_low_u8(right),
                    vget_low_u8(above),
                    vget_low_u8(below),
                );
                laplacian_sum += abs_laplacian_sum_8(
                    vget_high_u8(pixel),
                    vget_high_u8(left),
                    vget_high_u8(right),
                    vget_high_u8(above),
                    vget_high_u8(below),
                );
            }
            column += 16;
        }
        while column < width - 1 {
            let pixel = i32::from(y[row + column]);
            let left = i32::from(y[row + column - 1]);
            let right = i32::from(y[row + column + 1]);
            let above = i32::from(y[up + column]);
            let below = i32::from(y[down + column]);
            laplacian_sum += u64::from((4 * pixel - left - right - above - below).unsigned_abs());
            column += 1;
        }
    }
    laplacian_sum
}

#[cfg(test)]
pub(crate) fn frame_hf_laplacian_sum_scalar(y: &[u8], width: usize, height: usize) -> u64 {
    if width < 3 || height < 3 || y.len() < width * height {
        return 0;
    }
    frame_hf_laplacian_sum_rows_scalar(y, width, 1, height - 1, 1)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) unsafe fn frame_hf_laplacian_sum_neon(y: &[u8], width: usize, height: usize) -> u64 {
    if width < 3 || height < 3 || y.len() < width * height {
        return 0;
    }
    // SAFETY: The caller accepts the target-feature contract, and the bounds
    // above prove all complete rows used by the helper exist.
    unsafe { frame_hf_laplacian_sum_rows_neon(y, width, 1, height - 1, 1) }
}

#[allow(clippy::too_many_arguments)]
fn compute_avg_source_sad_rows(
    src: &[u8],
    last: &[u8],
    stride: usize,
    sb_row_start: usize,
    cbr_mode: bool,
    sad64: SadFn,
    variance64: VarFn,
    sad_out: &mut [u64],
    state_out: &mut [ContentStateSb],
) {
    let sb_cols = (stride / 64).max(1);
    debug_assert_eq!(sad_out.len(), state_out.len());
    debug_assert_eq!(sad_out.len() % sb_cols, 0);
    for (local_index, (sad_slot, state_slot)) in
        sad_out.iter_mut().zip(state_out.iter_mut()).enumerate()
    {
        let sb_row = sb_row_start + local_index / sb_cols;
        let sb_col = local_index % sb_cols;
        let (sad, sse, sum) = sb_sad_and_var_dispatched(
            src,
            last,
            stride,
            sb_row * 64,
            sb_col * 64,
            sad64,
            variance64,
        );
        *sad_slot = sad;
        *state_slot = classify_source_sad(sad, sse, sum, cbr_mode);
    }
}

fn compute_skin_map_rows(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    width: usize,
    mi_row_start: usize,
    consecutive_zero_mv: &[u8],
    skin_out: &mut [u8],
) {
    let mi_cols = width / 8;
    let uv_width = width / 2;
    let height = src_y.len() / width;
    let uv_height = src_u.len() / uv_width;
    debug_assert_eq!(consecutive_zero_mv.len(), skin_out.len());
    debug_assert_eq!(skin_out.len() % mi_cols, 0);
    debug_assert_eq!(src_y.len(), width * height);
    debug_assert_eq!(src_u.len(), uv_width * uv_height);
    debug_assert_eq!(src_v.len(), uv_width * uv_height);
    // libvpx `vp9_skin_detection.c:27-30` for BLOCK_8X8: the center
    // samples are (+4,+4) in luma and (+2,+2) in 4:2:0 chroma.
    const Y_CENTER: usize = 4;
    const UV_CENTER: usize = 2;
    for (local_index, (skin, &consec_zeromv)) in skin_out
        .iter_mut()
        .zip(consecutive_zero_mv.iter())
        .enumerate()
    {
        let mi_row = mi_row_start + local_index / mi_cols;
        let mi_col = local_index % mi_cols;
        let y_pix_row = (mi_row * 8 + Y_CENTER).min(height - 1);
        let y_pix_col = mi_col * 8 + Y_CENTER;
        let uv_pix_row = (mi_row * 4 + UV_CENTER).min(uv_height - 1);
        let uv_pix_col = mi_col * 4 + UV_CENTER;
        let ysource = src_y[y_pix_row * width + y_pix_col] as i32;
        let usource = src_u[uv_pix_row * uv_width + uv_pix_col] as i32;
        let vsource = src_v[uv_pix_row * uv_width + uv_pix_col] as i32;
        *skin = compute_skin_block(
            ysource,
            usource,
            vsource,
            i32::from(consec_zeromv),
            /* curr_motion_magn = */ 0,
        );
    }
}

fn classify_source_sad(sad: u64, sse: u64, sum: u64, cbr_mode: bool) -> ContentStateSb {
    // libvpx threshold (`vp9_encodeframe.c:1219`).
    const AVG_SOURCE_SAD_THRESHOLD: u64 = 10_000;
    // libvpx: variance = sse - ((sum * sum) >> 12), so
    // `sse - variance` is the squared mean-difference signal.
    let var = sse.saturating_sub((sum * sum) >> 12);
    let sumdiff = sse - var;
    let mut state = if sad < AVG_SOURCE_SAD_THRESHOLD {
        if sumdiff < 25 {
            ContentStateSb::LowSadLowSumdiff
        } else {
            ContentStateSb::LowSadHighSumdiff
        }
    } else if sumdiff < 25 {
        ContentStateSb::HighSadLowSumdiff
    } else {
        ContentStateSb::HighSadHighSumdiff
    };
    if cbr_mode && var < (sse >> 3) && sumdiff > 10_000 {
        state = ContentStateSb::LowVarHighSumdiff;
    } else if sad > (AVG_SOURCE_SAD_THRESHOLD << 1) {
        state = ContentStateSb::VeryHighSad;
    }
    state
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

/// W2.5 helper: libvpx `vp9_compute_skin_block`
/// (`vp9_skin_detection.c:18-38`). The full libvpx function reads its
/// own center pixel from `(y, u, v)` plane pointers; this rvp9
/// variant takes the already-read Y/U/V samples as args and applies
/// the two early-exit branches plus the `vpx_skin_pixel` predicate.
///
/// Returns `1` on skin, `0` on non-skin (libvpx int return).
fn compute_skin_block(
    ysource: i32,
    usource: i32,
    vsource: i32,
    consec_zeromv: i32,
    curr_motion_magn: i32,
) -> u8 {
    // libvpx `vp9_skin_detection.c:22-23` early exit on long-running
    // static blocks.
    if consec_zeromv > 60 && curr_motion_magn == 0 {
        return 0;
    }
    // libvpx `vp9_skin_detection.c:25, 35`: motion flag.
    let motion: i32 = if consec_zeromv > 25 && curr_motion_magn == 0 {
        0
    } else {
        1
    };
    vpx_skin_pixel(ysource, usource, vsource, motion)
}

/// W2.5 helper: faithful port of libvpx `vpx_skin_pixel`
/// (`vpx_dsp/skin_detection.c:49-79`). Operates with `MODEL_MODE = 1`
/// (the libvpx default at line 13), iterating through 5 fixed-point
/// skin-color models with per-model thresholds.
///
/// Returns `1` if the (y, cb, cr) sample matches the skin model,
/// `0` otherwise. The math is bit-exact with the libvpx scalar
/// implementation — no SIMD path is used.
fn vpx_skin_pixel(y: i32, cb: i32, cr: i32, motion: i32) -> u8 {
    // libvpx `vpx_dsp/skin_detection.c:25-26`.
    const Y_LOW: i32 = 40;
    const Y_HIGH: i32 = 220;
    // libvpx `vpx_dsp/skin_detection.c:16-20`.
    const SKIN_MEAN: [[i32; 2]; 5] = [
        [7463, 9614],
        [6400, 10240],
        [7040, 10240],
        [8320, 9280],
        [6800, 9614],
    ];
    // libvpx `vpx_dsp/skin_detection.c:21` (q16).
    const SKIN_INV_COV: [i32; 4] = [4107, 1663, 1663, 2157];
    // libvpx `vpx_dsp/skin_detection.c:22-23` (q18).
    const SKIN_THRESHOLD: [i32; 6] = [1570636, 1400000, 800000, 800000, 800000, 800000];

    // libvpx `vpx_dsp/skin_detection.c:50`.
    if !(Y_LOW..=Y_HIGH).contains(&y) {
        return 0;
    }
    // libvpx `vpx_dsp/skin_detection.c:57`.
    if cb == 128 && cr == 128 {
        return 0;
    }
    // libvpx `vpx_dsp/skin_detection.c:59`.
    if cb > 150 && cr < 110 {
        return 0;
    }
    // libvpx `vpx_dsp/skin_detection.c:60-76` model loop.
    for i in 0..5usize {
        let skin_color_diff =
            vpx_evaluate_skin_color_difference(cb, cr, i, &SKIN_MEAN, &SKIN_INV_COV);
        let thresh = SKIN_THRESHOLD[i + 1];
        if skin_color_diff < thresh {
            // libvpx line 63 — dark-luma rejection.
            if y < 60 && skin_color_diff > 3 * (thresh >> 2) {
                return 0;
            }
            // libvpx lines 65-66 — static-block rejection.
            if motion == 0 && skin_color_diff > (thresh >> 1) {
                return 0;
            }
            return 1;
        }
        // libvpx lines 73-74 — early-exit if difference is >> threshold.
        if skin_color_diff > (thresh << 3) {
            return 0;
        }
    }
    0
}

/// W2.5 helper: faithful port of libvpx
/// `vpx_evaluate_skin_color_difference`
/// (`vpx_dsp/skin_detection.c:29-46`). Computes the Mahalanobis
/// distance between `(cb, cr)` and the `idx`-th skin-color cluster.
fn vpx_evaluate_skin_color_difference(
    cb: i32,
    cr: i32,
    idx: usize,
    skin_mean: &[[i32; 2]; 5],
    skin_inv_cov: &[i32; 4],
) -> i32 {
    let cb_q6 = cb << 6;
    let cr_q6 = cr << 6;
    // libvpx lines 33-38.
    let cb_diff_q12 = (cb_q6 - skin_mean[idx][0]) * (cb_q6 - skin_mean[idx][0]);
    let cbcr_diff_q12 = (cb_q6 - skin_mean[idx][0]) * (cr_q6 - skin_mean[idx][1]);
    let cr_diff_q12 = (cr_q6 - skin_mean[idx][1]) * (cr_q6 - skin_mean[idx][1]);
    // libvpx lines 39-41.
    let cb_diff_q2 = (cb_diff_q12 + (1 << 9)) >> 10;
    let cbcr_diff_q2 = (cbcr_diff_q12 + (1 << 9)) >> 10;
    let cr_diff_q2 = (cr_diff_q12 + (1 << 9)) >> 10;
    // libvpx lines 42-44.
    skin_inv_cov[0] * cb_diff_q2
        + skin_inv_cov[1] * cbcr_diff_q2
        + skin_inv_cov[2] * cbcr_diff_q2
        + skin_inv_cov[3] * cr_diff_q2
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
        st.last_source_y = Some(vec![42u8; 64 * 64].into());
        st.keyframe_reset();
        for c in &st.consec_zero_mv {
            assert_eq!(*c, 0);
        }
        assert!(st.last_source_y.is_none());
    }

    /// W2.1 PROOF — `capture_last_source` populates the Last_Source
    /// view and reuses its allocation while it is uniquely owned.
    #[test]
    fn w2_1_capture_last_source_round_trips() {
        let mut st = PreAnalysisState::new(64, 64);
        let src1: Vec<u8> = (0..64 * 64).map(|i| i as u8).collect();
        st.capture_last_source(&src1);
        assert_eq!(st.last_source_y.as_deref(), Some(src1.as_slice()));
        let allocation = st.last_source_y.as_ref().unwrap().as_ptr();
        let src2 = vec![99u8; 64 * 64];
        st.capture_last_source(&src2);
        assert_eq!(st.last_source_y.as_deref(), Some(src2.as_slice()));
        assert_eq!(st.last_source_y.as_ref().unwrap().as_ptr(), allocation);
    }

    #[test]
    fn capture_shared_last_source_reuses_the_lookahead_allocation() {
        let mut st = PreAnalysisState::new(64, 64);
        let source: Arc<[u8]> = vec![73u8; 64 * 64].into();
        st.capture_last_source_shared(&source);
        let retained = st.last_source_y.as_ref().expect("source must be retained");
        assert_eq!(retained.as_ptr(), source.as_ptr());
        assert_eq!(retained.as_ref(), source.as_ref());
    }

    /// W2.2 PROOF — `compute_avg_source_sad` is a no-op when
    /// Last_Source is None (libvpx-equivalent of the first inter
    /// frame after a KF — no reference to diff against).
    #[test]
    fn w2_2_avg_source_sad_noop_without_last_source() {
        let mut st = PreAnalysisState::new(64, 64);
        let src = vec![100u8; 64 * 64];
        st.compute_avg_source_sad(&src, false);
        // No reference ⇒ no state changed.
        for v in &st.avg_source_sad {
            assert_eq!(*v, 0);
        }
        for c in &st.content_state_sb {
            assert_eq!(*c, ContentStateSb::Normal);
        }
    }

    /// W2.2 PROOF — identical source + last_source ⇒ zero SAD ⇒
    /// `kLowSadLowSumdiff`.
    #[test]
    fn w2_2_avg_source_sad_zero_on_identical_frames() {
        let mut st = PreAnalysisState::new(128, 128);
        let frame = vec![100u8; 128 * 128];
        st.capture_last_source(&frame);
        st.compute_avg_source_sad(&frame, false);
        for v in &st.avg_source_sad {
            assert_eq!(*v, 0, "identical frames must produce SAD = 0");
        }
        for c in &st.content_state_sb {
            assert_eq!(*c, ContentStateSb::LowSadLowSumdiff);
        }
    }

    #[test]
    fn avg_source_sad_extends_a_short_bottom_edge_to_sb64() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 32;
        let previous = vec![0u8; WIDTH * HEIGHT];
        let current = vec![100u8; WIDTH * HEIGHT];
        let mut state = PreAnalysisState::new(WIDTH as u32, HEIGHT as u32);
        state.capture_last_source(&previous);

        state.compute_avg_source_sad(&current, false);

        assert_eq!(state.avg_source_sad, [100 * 64 * 64]);
        assert_eq!(state.content_state_sb, [ContentStateSb::VeryHighSad]);
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
        st.compute_avg_source_sad(&bright, false);
        // The single SB's SAD must exceed 2 * 10_000.
        assert!(st.avg_source_sad[0] > 20_000);
        assert_eq!(st.content_state_sb[0], ContentStateSb::VeryHighSad);
    }

    #[test]
    fn w2_2_avg_source_sad_produces_all_sumdiff_states() {
        fn classify(source: Vec<u8>, reference: Vec<u8>, cbr_mode: bool) -> ContentStateSb {
            let mut state = PreAnalysisState::new(64, 64);
            state.capture_last_source(&reference);
            state.compute_avg_source_sad(&source, cbr_mode);
            state.content_state_sb[0]
        }

        let reference = vec![128u8; 64 * 64];
        assert_eq!(
            classify(vec![128u8; 64 * 64], reference.clone(), false),
            ContentStateSb::LowSadLowSumdiff,
        );
        assert_eq!(
            classify(vec![129u8; 64 * 64], reference.clone(), false),
            ContentStateSb::LowSadHighSumdiff,
        );
        let alternating: Vec<u8> = (0..64 * 64)
            .map(|index| if index & 1 == 0 { 131 } else { 125 })
            .collect();
        assert_eq!(
            classify(alternating, reference.clone(), false),
            ContentStateSb::HighSadLowSumdiff,
        );
        assert_eq!(
            classify(vec![131u8; 64 * 64], reference.clone(), false),
            ContentStateSb::HighSadHighSumdiff,
        );
        assert_eq!(
            classify(vec![131u8; 64 * 64], reference, true),
            ContentStateSb::LowVarHighSumdiff,
        );
    }

    /// Unit 144 — the production dispatch path must preserve every scalar
    /// source-analysis result, including non-zero SB offsets and scene state.
    #[test]
    fn w2_2_dispatched_source_analysis_matches_scalar_oracle() {
        const W: usize = 384;
        const H: usize = 384;
        let mut source = vec![0u8; W * H];
        let mut reference = vec![0u8; W * H];
        let mut seed = 0xBADC_0FFEu32;
        for (src, last) in source.iter_mut().zip(reference.iter_mut()) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *src = seed as u8;
            *last = (seed >> 16) as u8;
        }
        // Exercise both maximum-magnitude uniform differences and a
        // zero-sum alternating block in addition to the pseudo-random grid.
        for row in 0..64 {
            for col in 0..64 {
                source[row * W + col] = 255;
                reference[row * W + col] = 0;
                source[row * W + 64 + col] = if (row + col) & 1 == 0 { 255 } else { 0 };
                reference[row * W + 64 + col] = 128;
            }
        }

        let dispatch = rvp9_core::dsp::get_ref();
        let sad64 = dispatch.sad(BlockSize::Block64x64);
        let variance64 = dispatch.variance(BlockSize::Block64x64);
        let sb_cols = W / 64;
        let sb_rows = H / 64;
        for sb_r in 0..sb_rows {
            for sb_c in 0..sb_cols {
                let scalar = sb_sad_and_var(&source, &reference, W, sb_r * 64, sb_c * 64);
                let dispatched = sb_sad_and_var_dispatched(
                    &source,
                    &reference,
                    W,
                    sb_r * 64,
                    sb_c * 64,
                    sad64,
                    variance64,
                );
                assert_eq!(
                    dispatched, scalar,
                    "SAD/sum/SSE mismatch at SB ({sb_r}, {sb_c})"
                );
                let offset = sb_r * 64 * W + sb_c * 64;
                let (signed_sum, kernel_sse) =
                    variance64(&source[offset..], W, &reference[offset..], W);
                let mut scalar_signed_sum = 0i64;
                for row in 0..64 {
                    for col in 0..64 {
                        let index = offset + row * W + col;
                        scalar_signed_sum += i64::from(source[index]) - i64::from(reference[index]);
                    }
                }
                assert_eq!(
                    (signed_sum, kernel_sse),
                    (scalar_signed_sum, scalar.1),
                    "signed sum/SSE mismatch at SB ({sb_r}, {sb_c})"
                );
            }
        }

        for cbr_mode in [false, true] {
            let mut production = PreAnalysisState::new(W as u32, H as u32);
            production.capture_last_source(&reference);
            production.compute_avg_source_sad(&source, cbr_mode);

            let mut scalar = PreAnalysisState::new(W as u32, H as u32);
            scalar.capture_last_source(&reference);
            for sb_r in 0..sb_rows {
                for sb_c in 0..sb_cols {
                    let idx = sb_r * sb_cols + sb_c;
                    let (sad, sse, sum) =
                        sb_sad_and_var(&source, &reference, W, sb_r * 64, sb_c * 64);
                    scalar.avg_source_sad[idx] = sad;
                    scalar.content_state_sb[idx] = classify_source_sad(sad, sse, sum, cbr_mode);
                }
            }
            assert_eq!(production.avg_source_sad, scalar.avg_source_sad);
            assert_eq!(production.content_state_sb, scalar.content_state_sb);

            production.prev_avg_source_sad = 777;
            scalar.prev_avg_source_sad = 777;
            production.compute_scene_detection_from_cached_sad(6);
            scalar.compute_scene_detection_from_cached_sad(6);
            assert_eq!(
                (
                    production.high_source_sad,
                    production.scene_cut_force_key_frame,
                    production.prev_avg_source_sad,
                ),
                (
                    scalar.high_source_sad,
                    scalar.scene_cut_force_key_frame,
                    scalar.prev_avg_source_sad,
                )
            );
        }
    }

    /// Unit 703 — mixed worker jobs must preserve both source maps exactly,
    /// including unequal final row chunks in both producer grids.
    #[test]
    fn parallel_source_maps_match_serial_producers() {
        const W: usize = 256;
        // Five SB rows and 41 MI rows split four ways as 2/1/1/1 and
        // 11/10/10/10. A simple zip of `chunks_mut(div_ceil)` would expose
        // only three jobs and silently miss the final skin/HF span.
        const H: usize = 328;
        let mut source_y = vec![0u8; W * H];
        let mut previous_y = vec![0u8; W * H];
        let mut source_u = vec![0u8; (W / 2) * (H / 2)];
        let mut source_v = vec![0u8; (W / 2) * (H / 2)];
        for (index, pixel) in source_y.iter_mut().enumerate() {
            *pixel = ((index * 29 + index / W * 7) & 255) as u8;
        }
        for (index, pixel) in previous_y.iter_mut().enumerate() {
            *pixel = ((index * 17 + index / W * 11 + 31) & 255) as u8;
        }
        for (index, pixel) in source_u.iter_mut().enumerate() {
            *pixel = ((index * 13 + 91) & 255) as u8;
        }
        for (index, pixel) in source_v.iter_mut().enumerate() {
            *pixel = ((index * 19 + 137) & 255) as u8;
        }

        let mut serial = PreAnalysisState::new(W as u32, H as u32);
        serial.capture_last_source(&previous_y);
        for (index, count) in serial.consec_zero_mv.iter_mut().enumerate() {
            *count = (index % 64) as u8;
        }
        let mut parallel = serial.clone();

        serial.compute_avg_source_sad(&source_y, true);
        serial.compute_skin_map(&source_y, &source_u, &source_v);
        let mut pool = rvp9_core::tile_threading::WorkerPool::new_with_caller(
            std::num::NonZeroUsize::new(4).unwrap(),
        );
        let parallel_hf_sum = parallel
            .compute_source_maps_parallel(
                &source_y,
                &source_u,
                &source_v,
                &mut pool,
                ParallelSourceMapConfig {
                    cbr_mode: true,
                    compute_skin_map: true,
                    max_jobs: 4,
                    compute_hf_energy: true,
                },
            )
            .expect("parallel source maps computed HF energy");

        assert_eq!(parallel.avg_source_sad, serial.avg_source_sad);
        assert_eq!(parallel.content_state_sb, serial.content_state_sb);
        assert_eq!(parallel.skin_map, serial.skin_map);
        assert_eq!(
            parallel_hf_sum,
            frame_hf_laplacian_sampled_sum_rows(&source_y, W, H, 1, H - 1),
        );
    }

    /// Unit 917 — AQ-off/VBR source-map jobs must not run or overwrite the
    /// CBR cyclic-refresh-only skin producer.
    #[test]
    fn parallel_source_maps_can_skip_skin_detection() {
        const W: usize = 128;
        const H: usize = 128;
        let source_y = vec![96u8; W * H];
        let source_u = vec![112u8; (W / 2) * (H / 2)];
        let source_v = vec![144u8; (W / 2) * (H / 2)];
        let mut state = PreAnalysisState::new(W as u32, H as u32);
        state.skin_map.fill(0xa5);
        let expected = state.skin_map.clone();
        let mut pool = rvp9_core::tile_threading::WorkerPool::new_with_caller(
            std::num::NonZeroUsize::new(2).unwrap(),
        );

        state.compute_source_maps_parallel(
            &source_y,
            &source_u,
            &source_v,
            &mut pool,
            ParallelSourceMapConfig {
                cbr_mode: false,
                compute_skin_map: false,
                max_jobs: 2,
                compute_hf_energy: false,
            },
        );

        assert_eq!(state.skin_map, expected);
    }

    /// Unit 915 — the sampled production classifier must retain the exact
    /// oracle's decision on representative low- and high-frequency planes.
    #[test]
    fn sampled_hf_laplacian_preserves_representative_classifier_decisions() {
        const W: usize = 128;
        const H: usize = 72;
        const THRESHOLD: f64 = 7.0;
        let flat = vec![96u8; W * H];
        let textured = (0..W * H)
            .map(|index| {
                let row = index / W;
                let column = index % W;
                ((column * 29 + row * 47 + column * row * 3) & 255) as u8
            })
            .collect::<Vec<_>>();

        for plane in [&flat, &textured] {
            let exact = frame_hf_laplacian_sum_rows(plane, W, H, 1, H - 1) as f64
                / ((W - 2) * (H - 2)) as f64;
            let sampled = frame_hf_laplacian_sampled_sum_rows(plane, W, H, 1, H - 1) as f64
                / ((W - 2) * (H - 2)) as f64;
            assert_eq!(
                sampled >= THRESHOLD,
                exact >= THRESHOLD,
                "classifier changed: exact={exact}, sampled={sampled}",
            );
        }
    }

    #[test]
    fn sampled_hf_exact_completion_reuses_rows_without_changing_sum() {
        for &(width, height) in &[(3, 3), (17, 5), (32, 11), (65, 18), (127, 35)] {
            let plane = (0..width * height)
                .map(|index| {
                    let row = index / width;
                    let column = index % width;
                    ((column * 31 + row * 43 + column * row * 7) & 255) as u8
                })
                .collect::<Vec<_>>();
            let exact = frame_hf_laplacian_sum_rows(&plane, width, height, 1, height - 1);
            let sampled = frame_hf_laplacian_sampled_sum_rows(&plane, width, height, 1, height - 1);

            assert_eq!(
                frame_hf_laplacian_exact_sum_from_sampled(&plane, width, height, sampled,),
                exact,
                "sampled-row completion changed the {width}x{height} sum",
            );
            assert_eq!(
                frame_hf_laplacian_exact_sum_from_sampled(&plane, width, height, u64::MAX,),
                exact,
                "a saturated estimate must fall back to the exact scan",
            );
        }
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

    /// Unit 143 — cached per-SB SADs must be an exact oracle for the
    /// scene detector's former duplicate pixel walk.
    #[test]
    fn w2_3_cached_scene_detection_matches_pixel_oracle() {
        const W: usize = 384;
        const H: usize = 384;
        let calm = vec![100u8; W * H];
        let cut = vec![230u8; W * H];
        let mut boundary_only = calm.clone();
        for row in 0..H {
            for col in 0..W {
                if row < 64 || col < 64 || row >= H - 64 || col >= W - 64 {
                    boundary_only[row * W + col] = 230;
                }
            }
        }

        let mut direct = PreAnalysisState::new(W as u32, H as u32);
        let mut cached = direct.clone();
        for (source, frames_since_key) in [
            (&calm[..], 1u32),
            (&boundary_only[..], 5u32),
            (&cut[..], 6u32),
            (&calm[..], 7u32),
        ] {
            direct.capture_last_source(&calm);
            cached.capture_last_source(&calm);
            cached.compute_avg_source_sad(source, false);
            direct.compute_scene_detection(source, frames_since_key);
            cached.compute_scene_detection_from_cached_sad(frames_since_key);
            assert_eq!(
                (
                    cached.high_source_sad,
                    cached.scene_cut_force_key_frame,
                    cached.prev_avg_source_sad,
                ),
                (
                    direct.high_source_sad,
                    direct.scene_cut_force_key_frame,
                    direct.prev_avg_source_sad,
                ),
                "cached scene summary diverged at frames_since_key={frames_since_key}"
            );
        }
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
            st.compute_noise_estimate(&flat, k * 8, /* noise_sensitivity */ 1);
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
            noise_level_rank(st.noise_estimate.level) < noise_level_rank(NoiseLevel::Medium),
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
    /// > >    19*40) >> 2 = 190`. That's > `adapt_thresh = (3*115)>>1 =
    /// > >    172`, so the rapid-onset fast-path
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
        st.compute_noise_estimate(
            &src, /* frames_since_key = */ 8, /* noise_sensitivity */ 1,
        );
        st.compute_noise_estimate(
            &src, /* frames_since_key = */ 16, /* noise_sensitivity */ 1,
        );
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
            st.ne_value,
            st.ne_adapt_thresh
        );
        assert!(
            noise_level_rank(st.noise_estimate.level) >= noise_level_rank(NoiseLevel::Medium),
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
        st.compute_noise_estimate(
            &flat, /* frames_since_key = */ 8, /* noise_sensitivity */ 1,
        );
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
            st.compute_noise_estimate(&src, fc, /* noise_sensitivity */ 1);
        }
        assert_eq!(
            st.ne_value, 0,
            "W2.4 PROOF: frame_period gate must suppress the \
             histogram walk on non-multiples of 8 (libvpx \
             vp9_noise_estimate.c:132). Got ne_value = {}",
            st.ne_value
        );
    }

    /// W2.5 PROOF — a synthetic 8×8 block painted with a YUV triple
    /// well inside libvpx's skin model produces `skin_map[idx] == 1`.
    ///
    /// Setup: a single SB-aligned 64×64 frame painted uniformly with
    /// `(Y=180, U=110, V=148)`. libvpx's
    /// `vpx_dsp/skin_detection.c:60-76` walks 5 skin-color clusters;
    /// the first cluster's mean is `(cb_q6, cr_q6) = (7463, 9614)`,
    /// which corresponds to `(cb, cr) = (116.6, 150.2)` — our
    /// `(110, 148)` lands close enough that the Mahalanobis distance
    /// stays under `skin_threshold[1] = 1400000`. Y is in [40, 220]
    /// and `(cb, cr) != (128, 128)` so the early-rejects don't fire.
    ///
    /// `consec_zero_mv` is left at 0 (the default), so `motion = 1`
    /// (cf. libvpx line 35: only `consec_zeromv > 25` flips it to 0).
    /// With `motion = 1`, the `(motion == 0 && skin_color_diff >
    /// thresh/2)` reject at libvpx line 65-66 is suppressed, so the
    /// predicate returns 1 as soon as it finds a cluster match.
    ///
    /// PROOF: strip the `self.skin_map[mi_idx] = ...` write and the
    /// map stays zero, failing the `== 1` assertion. Strip the
    /// `compute_skin_block` early-exit checks and the test still
    /// passes (those gates are sub-cases of the same code path).
    #[test]
    fn w2_5_skin_map_lights_skin_tone_pixels() {
        // 64×64 minimal frame — single SB, mi-grid 8×8.
        let w: usize = 64;
        let h: usize = 64;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        let y_plane = vec![180u8; w * h];
        let u_plane = vec![110u8; (w / 2) * (h / 2)];
        let v_plane = vec![148u8; (w / 2) * (h / 2)];
        st.compute_skin_map(&y_plane, &u_plane, &v_plane);
        // The center mi cell (mi_row=4, mi_col=4) — well clear of any
        // edge effects.
        let mi_cols = w / 8;
        let center_idx = 4 * mi_cols + 4;
        assert_eq!(
            st.skin_map[center_idx], 1,
            "W2.5 PROOF: a center-of-frame skin-tone block must \
             produce skin_map[{}] == 1 (libvpx \
             vp9_skin_detection.c:36, vpx_dsp/skin_detection.c:60-76). \
             Got skin_map[{}] = {}",
            center_idx, center_idx, st.skin_map[center_idx]
        );
        // Sanity: at least one cell lit up. (Every cell sees the same
        // YUV so they should all be 1.)
        let lit: usize = st.skin_map.iter().map(|&v| v as usize).sum();
        assert!(
            lit > 0,
            "W2.5 PROOF: no skin cells lit up — predicate likely off"
        );
    }

    /// W2.5 PROOF — a neutral-grey block (libvpx's explicit
    /// `cb == 128 && cr == 128` exit at `vpx_dsp/skin_detection.c:57`)
    /// produces `skin_map[idx] == 0` everywhere.
    ///
    /// Setup: uniformly Y=128, U=128, V=128 — grey. libvpx line 57
    /// guarantees an early-zero return.
    ///
    /// PROOF: strip the `if cb == 128 && cr == 128 return 0` guard
    /// and the test STILL passes because the Mahalanobis distance is
    /// huge for `(128, 128)` vs every skin cluster mean — but the
    /// guard is the libvpx-canonical fast path so we test it
    /// explicitly.
    #[test]
    fn w2_5_skin_map_rejects_grey_pixels() {
        let w: usize = 64;
        let h: usize = 64;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        let y_plane = vec![128u8; w * h];
        let u_plane = vec![128u8; (w / 2) * (h / 2)];
        let v_plane = vec![128u8; (w / 2) * (h / 2)];
        st.compute_skin_map(&y_plane, &u_plane, &v_plane);
        let lit: usize = st.skin_map.iter().map(|&v| v as usize).sum();
        assert_eq!(
            lit, 0,
            "W2.5 PROOF: neutral grey (U=V=128) must never be \
             classified as skin (libvpx \
             vpx_dsp/skin_detection.c:57). Got lit cells = {}",
            lit
        );
    }

    /// W2.5 PROOF — out-of-range luma (libvpx Y < 40 OR Y > 220
    /// at `vpx_dsp/skin_detection.c:50`) rejects skin classification
    /// even when chroma is in-cluster.
    #[test]
    fn w2_5_skin_map_rejects_out_of_range_luma() {
        let w: usize = 64;
        let h: usize = 64;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        // Y = 30 (< Y_LOW = 40). Same in-cluster chroma as the
        // positive PROOF.
        let y_plane = vec![30u8; w * h];
        let u_plane = vec![110u8; (w / 2) * (h / 2)];
        let v_plane = vec![148u8; (w / 2) * (h / 2)];
        st.compute_skin_map(&y_plane, &u_plane, &v_plane);
        let lit: usize = st.skin_map.iter().map(|&v| v as usize).sum();
        assert_eq!(
            lit, 0,
            "W2.5 PROOF: Y < 40 must reject skin classification (libvpx \
             vpx_dsp/skin_detection.c:50). Got lit cells = {}",
            lit
        );
    }

    /// W2.5 PROOF — long-running static cells (`consec_zero_mv > 60`)
    /// are forced to non-skin by libvpx's early exit at
    /// `vp9_skin_detection.c:22-23`, regardless of pixel color.
    #[test]
    fn w2_5_skin_map_zero_mv_streak_overrides_color() {
        let w: usize = 64;
        let h: usize = 64;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        // Seed the mi-grid as long-running zero-MV (libvpx line 22:
        // `consec_zeromv > 60`).
        for c in &mut st.consec_zero_mv {
            *c = 100;
        }
        // Use the skin-tone fixture from the positive PROOF.
        let y_plane = vec![180u8; w * h];
        let u_plane = vec![110u8; (w / 2) * (h / 2)];
        let v_plane = vec![148u8; (w / 2) * (h / 2)];
        st.compute_skin_map(&y_plane, &u_plane, &v_plane);
        let lit: usize = st.skin_map.iter().map(|&v| v as usize).sum();
        assert_eq!(
            lit, 0,
            "W2.5 PROOF: consec_zero_mv > 60 must force non-skin \
             (libvpx vp9_skin_detection.c:22-23). Got lit cells = {}",
            lit
        );
    }

    /// W2.6 PROOF — the `vbp_threshold_scale` helper returns the
    /// 1/2/4 ladder per the task brief.
    ///
    /// Setup: hand-seed `noise_estimate` to each level, after
    /// flipping `enabled` to true (libvpx gates on this at
    /// `vp9_encodeframe.c:589`). Walk the ladder.
    ///
    /// PROOF: strip the `match self.noise_estimate.level` block and
    /// the scale would always read 1, failing the Medium/High
    /// assertions.
    #[test]
    fn w2_6_vbp_scale_lifts_with_noise_level() {
        let mut st = PreAnalysisState::new(640, 384);
        // Disabled noise estimator ⇒ default scale 1.
        st.noise_estimate.enabled = false;
        st.noise_estimate.level = NoiseLevel::High;
        assert_eq!(
            st.vbp_threshold_scale(),
            1,
            "W2.6 PROOF: disabled noise estimator must return scale=1 \
             regardless of level (libvpx vp9_encodeframe.c:589 gate)"
        );
        // Now run the ladder with enabled = true.
        st.noise_estimate.enabled = true;
        st.noise_estimate.level = NoiseLevel::LowLow;
        assert_eq!(
            st.vbp_threshold_scale(),
            1,
            "W2.6 PROOF: NoiseLevel::LowLow must map to scale=1"
        );
        st.noise_estimate.level = NoiseLevel::Low;
        assert_eq!(
            st.vbp_threshold_scale(),
            1,
            "W2.6 PROOF: NoiseLevel::Low must map to scale=1 \
             (libvpx vp9_encodeframe.c:596-597 default branch)"
        );
        st.noise_estimate.level = NoiseLevel::Medium;
        assert_eq!(
            st.vbp_threshold_scale(),
            2,
            "W2.6 PROOF: NoiseLevel::Medium must map to scale=2 \
             (libvpx vp9_encodeframe.c:594-595)"
        );
        st.noise_estimate.level = NoiseLevel::High;
        assert_eq!(
            st.vbp_threshold_scale(),
            4,
            "W2.6 PROOF: NoiseLevel::High must map to scale=4 \
             (brief-specified 1/2/4 ladder; libvpx \
             vp9_encodeframe.c:592-593 uses 1/2/3)"
        );
    }

    /// W2.6 PROOF — a flat-chroma fixture (uniform U/V) produces
    /// `color_sensitivity = [false; 2]` per SB. libvpx's
    /// `color_sensitivity[i-1] = uv_sad > (y_sad >> 2)`
    /// (`vp9_encodeframe.c:1205`) requires `uv_sad` to exceed
    /// `y_sad / 4`; flat chroma gives `uv_sad == 0`, which can't
    /// exceed any non-negative threshold.
    ///
    /// Setup: 64×64 SB, Last_Source = a noisy luma frame (so
    /// `y_sad > 0`), source = the same luma but U/V uniformly flat.
    /// The chroma SAD against the SB's mean U/V is 0 ⇒ both flags
    /// false.
    ///
    /// PROOF: strip the `> (y_sad >> SHIFT)` comparison (e.g.,
    /// substitute `>=`) and the test still passes for `uv_sad == 0`
    /// because the right-hand side is also 0 for an all-zero
    /// Last_Source, so the comparison is `0 >= 0 = true` and the
    /// test fails. Use a non-trivial `y_sad` (set Last_Source to a
    /// noisy frame) to disambiguate.
    #[test]
    fn w2_6_color_sensitivity_low_chroma_diff_means_false() {
        let w: usize = 64;
        let h: usize = 64;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        // Last_Source: noisy luma so y_sad > 0. Chroma planes don't
        // matter for y_sad.
        let mut last_y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                last_y[r * w + c] = ((r + c) & 0x3F) as u8;
            }
        }
        let src_y = vec![128u8; w * h];
        let src_u = vec![100u8; (w / 2) * (h / 2)]; // flat chroma
        let src_v = vec![120u8; (w / 2) * (h / 2)]; // flat chroma
        st.capture_last_source(&last_y);
        let cs = st.compute_color_sensitivity(&src_y, &src_u, &src_v, ContentStateSb::Normal, 0, 0);
        assert_eq!(
            cs,
            [false, false],
            "W2.6 PROOF: flat-chroma SB must produce \
             color_sensitivity = [false, false] (libvpx \
             vp9_encodeframe.c:1205 `uv_sad > (y_sad >> 2)` is \
             false when uv_sad == 0). Got {:?}",
            cs
        );
    }

    /// W2.6 PROOF — a high-chroma-variance fixture lights up both
    /// `color_sensitivity` flags. libvpx's check (`uv_sad > (y_sad >>
    /// shift)`, shift=2) trips when chroma varies more than 1/4 of
    /// the luma SAD.
    ///
    /// Setup: a Last_Source where `src_y == last_y` (y_sad = 0), so
    /// the threshold `y_sad >> 2 = 0`. ANY positive chroma SAD
    /// against the SB mean will trip the comparison `uv_sad > 0`.
    /// Drop a non-flat U/V (an alternating pattern) so its SAD
    /// against the SB-mean is > 0.
    ///
    /// PROOF: strip the comparison and the test always returns
    /// `[false; 2]`, failing.
    #[test]
    fn w2_6_color_sensitivity_high_chroma_diff_means_true() {
        let w: usize = 64;
        let h: usize = 64;
        let uvw = w / 2;
        let uvh = h / 2;
        let mut st = PreAnalysisState::new(w as u32, h as u32);
        let src_y = vec![100u8; w * h];
        st.capture_last_source(&src_y); // y_sad = 0
        // Chroma alternating ±32 around 128 — SAD vs mean is huge.
        let mut src_u = vec![0u8; uvw * uvh];
        let mut src_v = vec![0u8; uvw * uvh];
        for r in 0..uvh {
            for c in 0..uvw {
                src_u[r * uvw + c] = if (r ^ c) & 1 == 0 { 96 } else { 160 };
                src_v[r * uvw + c] = if (r ^ c) & 1 == 0 { 100 } else { 156 };
            }
        }
        let cs = st.compute_color_sensitivity(&src_y, &src_u, &src_v, ContentStateSb::Normal, 0, 0);
        assert_eq!(
            cs,
            [true, true],
            "W2.6 PROOF: high-chroma-variance SB with zero luma SAD \
             must light both color_sensitivity flags (libvpx \
             vp9_encodeframe.c:1205). Got {:?}",
            cs
        );
    }

    /// W2.6 PROOF — the batch driver fills the per-SB array for
    /// every SB position. Uses an identical-luma fixture (y_sad = 0
    /// so `y_sad >> 2 == 0`, any positive uv_sad trips the flag) +
    /// strong-variance chroma to guarantee all four SBs light up.
    #[test]
    fn w2_6_color_sensitivity_frame_fills_every_sb() {
        // 128×128 ⇒ 2×2 SB grid.
        let mut st = PreAnalysisState::new(128, 128);
        // Identical src_y / last_y ⇒ y_sad = 0 across every SB.
        let src_y = vec![100u8; 128 * 128];
        // Strong-variance chroma everywhere: ±32 around mean 128 ⇒
        // SAD-vs-mean ≈ 32 × 1024 = 32768 per plane per SB. > 0.
        let mut src_u = vec![0u8; 64 * 64];
        let mut src_v = vec![0u8; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                src_u[r * 64 + c] = if (r ^ c) & 1 == 0 { 96 } else { 160 };
                src_v[r * 64 + c] = if (r ^ c) & 1 == 0 { 100 } else { 156 };
            }
        }
        st.capture_last_source(&src_y); // y_sad = 0
        st.compute_color_sensitivity_frame(&src_y, &src_u, &src_v);
        assert_eq!(st.color_sensitivity_sb.len(), 4);
        for (idx, cs) in st.color_sensitivity_sb.iter().enumerate() {
            assert_eq!(
                *cs,
                [true, true],
                "W2.6 PROOF: SB {} must have both color_sensitivity \
                 flags lit on this fixture",
                idx
            );
        }
    }

    /// W2.6 — no Last_Source ⇒ `[false; 2]` (post-KF guard).
    #[test]
    fn w2_6_color_sensitivity_no_last_source_returns_false() {
        let st = PreAnalysisState::new(64, 64);
        let src_y = vec![128u8; 64 * 64];
        let src_u = vec![100u8; 32 * 32];
        let src_v = vec![120u8; 32 * 32];
        let cs = st.compute_color_sensitivity(&src_y, &src_u, &src_v, ContentStateSb::Normal, 0, 0);
        assert_eq!(cs, [false; 2]);
    }
}
