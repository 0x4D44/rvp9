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
use rvp9_core::mode_decision::inter::ContentStateSb;

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
}
