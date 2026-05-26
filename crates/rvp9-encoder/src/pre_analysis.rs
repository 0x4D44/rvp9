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
        Self {
            consec_zero_mv: vec![0u8; mi_rows * mi_cols],
            // Updated per-frame by `update_per_frame_gates`.
            compute_source_sad_onepass: false,
            last_source_y: None,
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

    /// Number of mi-grid cells (`mi_rows * mi_cols`) tracked.
    #[must_use]
    pub fn mi_grid_len(&self) -> usize {
        self.consec_zero_mv.len()
    }
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
}
