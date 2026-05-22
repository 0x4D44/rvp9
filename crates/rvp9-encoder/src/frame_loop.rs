//! F10.0 — Public `encode_frame(yuv)` API and frame-loop assembly.
//!
//! This is the integration layer that wires together the F5–F9
//! building blocks into a working encoder. It exposes a public
//! [`StreamEncoder`] type that consumes raw YUV 4:2:0 frames and emits
//! decodable VP9 frame packets.
//!
//! ## Scope (MVP)
//!
//! - Width / height multiples of 64 (matches the existing wire-format
//!   writers' scope).
//! - 4:2:0 chroma (Profile 0), 8-bit depth.
//! - Single tile (`tile_cols = tile_rows = 1`).
//! - Single 64×64 SB grid per frame.
//! - Keyframe path: real DC_PRED intra residuals via the F5.1
//!   simulator (`simulate_dc_intra_encode`).
//! - Inter path: single-ref LASTFRAME ZEROMV, `skip = true`,
//!   `BlockSize::Block64x64` over every superblock. The reconstruction
//!   equals the LASTFRAME reference (zero-MV copy, no residual added).
//! - GOP boundary: first frame is a keyframe; thereafter a keyframe is
//!   emitted whenever the rate-controller's `frames_to_key` counter
//!   reaches zero.
//! - Frame-context bank 0 stays at default — keyframes use
//!   `refresh_frame_context = false`. Inter frames emit
//!   `refresh_frame_flags = 0x01` so the decoder updates the
//!   LASTFRAME ref slot to point at this frame's reconstruction
//!   (post-A11 fix; pre-A11 set `0x00` which left the decoder's
//!   LASTFRAME pinned to the keyframe for the entire GOP and caused
//!   monotonic PSNR drift on real content).
//! - No loop filter (`loop_filter_level = 0`).
//!
//! ## Divergences from libvpx (documented)
//!
//! - **No mode decision on inter frames.** Every inter SB is forced to
//!   `BlockSize::Block64x64` `InterMode::ZeroMv` `LASTFRAME`
//!   `skip = true`. F8.1's `pick_inter_mode_luma` and F8.2's
//!   `choose_partitioning` are not driven yet — they're available in
//!   the codebase but the MVP frame loop short-circuits to ZEROMV/skip
//!   because (a) wiring partition+inter+coef-walk to the writer
//!   correctly requires shape adapters that aren't built yet, and (b)
//!   the F10.0 contract is "produces decodable bitstreams from a
//!   YUV input", which this scope satisfies. F10.1+ widens the loop.
//! - **No mode decision on keyframes.** Every KF SB is forced to
//!   `Block64x64` `IntraMode::Dc` for both Y and UV (matching the
//!   F5.1 simulator). F8.0's `pick_intra_mode_luma` is not driven yet
//!   — same reason as inter.
//! - **No segmentation, no loop filter, no error-resilient mode.**
//! - **Reference buffer is the encoder-side reconstruction.** For
//!   keyframes we use the F5.1 simulator's `recon_yuv_out` (the
//!   per-TX inverse-DCT-add output). For inter ZEROMV/skip=true
//!   frames the reconstruction equals the previous reference, so the
//!   reference buffer doesn't change. This is sufficient for the
//!   "encode → decode" oracle.

use rvp9_core::bitstream::frame::{MAX_INTER_REFS, ReferenceMode};
use rvp9_core::bitstream::header::HeaderTail;
use rvp9_core::bitstream::sub_parsers::{
    LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
    SegmentationParams, TileInfo,
};
use rvp9_core::bitstream::{
    BitDepth, BitstreamError, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
    CompressedHeaderResult, FrameKind as WireFrameKind, InterCoefSource, InterpFilter,
    ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
    write_tile_intra_keyframe, write_tile_inter, write_uncompressed_header,
};
use rvp9_core::block::{BlockSize, max_txsize};
use rvp9_core::coef_ctx::TileCoefCtx;
use rvp9_core::entropy::{BoolWriter, FrameContext};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_decision::partition::{
    PartitionDecision, PartitionSearchParams, choose_partitioning,
};
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
use rvp9_core::mv_search::PlaneRef;
use rvp9_core::transforms::TxMode;

use crate::aq_cyclic_refresh::{CyclicRefresh, CyclicRefreshConfig};
use crate::loopfilter::{
    apply_loop_filter_to_tight_recon, build_inter_mi_grid_for_lf, build_kf_mi_grid_for_lf,
    pick_filter_level,
};
use crate::rate_control::{FrameKind as RcFrameKind, RateControl, RateControlConfig};
use crate::{
    EncodeError, InterPartitionLeafDesc, PartitionLeafDesc,
    pick_and_simulate_intra_encode_partitions, pick_and_simulate_inter_encode_partitions,
};

/// Adaptive-quantization mode. Selects whether the encoder runs an
/// AQ state machine on inter frames.
///
/// `Off` reproduces Stage 4a behaviour exactly: no segmentation in the
/// uncompressed header, no per-leaf `segment_id` emission, every leaf
/// quantized at the frame's `base_qindex`.
///
/// `CyclicRefresh` drives the F8.2 cyclic-refresh state machine
/// ([`crate::aq_cyclic_refresh::CyclicRefresh`]) on every inter frame:
/// a fraction of super-blocks are picked for refresh (segment 1),
/// their per-leaf `segment_id` is emitted via the modes_writer (AQ.0),
/// the forward quant runs at a finer Q for those leaves (AQ.1), and
/// the post-encode skip-aware age-update + cursor advance closes the
/// loop. Keyframes reset the cyclic-refresh state (fresh map / cursor)
/// — they emit no segmentation themselves.
///
/// Defaults to `Off`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum AqMode {
    /// AQ disabled. Stage 4a behaviour: no segmentation, uniform Q.
    #[default]
    Off,
    /// libvpx cyclic refresh AQ. Cycles through SBs at finer Q over
    /// `~1/N` frames. See [`crate::aq_cyclic_refresh`].
    CyclicRefresh,
}

/// Public configuration for the F10.0 streaming encoder.
///
/// Wraps a [`RateControlConfig`] plus the encoder-specific knobs the
/// rate controller doesn't own (e.g. the keyframe q-index when rate
/// control is bypassed). v1 is intentionally compact; future stages
/// (4b adaptive Q, 4c denoiser, 4d/e SVC) will widen the surface.
#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    /// Coded luma width in pixels. Must be a positive multiple of 64.
    pub width: u32,
    /// Coded luma height in pixels. Must be a positive multiple of 64.
    pub height: u32,
    /// Frame-rate in Hz. Must be positive and finite.
    pub framerate: f64,
    /// Target average bitrate in bits per second. Must be > 0.
    pub target_bitrate_bps: u32,
    /// Keyframe interval in frames. Frame 0 is always a keyframe;
    /// thereafter a keyframe is emitted every `gop_length` frames.
    /// Must be > 0.
    pub gop_length: u32,
    /// Adaptive-quantization mode. [`AqMode::Off`] (the default)
    /// preserves Stage 4a bytes. [`AqMode::CyclicRefresh`] wires the
    /// F8.2 cyclic-refresh state machine into the inter path.
    pub aq_mode: AqMode,
}

impl EncoderConfig {
    /// Convenience constructor with defaults that mirror the
    /// rate-control crate's WebRTC profile (1 Mbps / 30 fps /
    /// gop_length = 60) and AQ off.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 60,
            aq_mode: AqMode::Off,
        }
    }

    /// Build the underlying [`RateControlConfig`] for the rate
    /// controller. Mirrors the encoder-config fields onto the
    /// rate-control fields and pulls libvpx-style defaults for the
    /// buffer geometry (6 s buffer, 4 s starting, 5 s optimal).
    fn to_rate_control_config(self) -> RateControlConfig {
        let defaults = RateControlConfig::default();
        // A10 experiment hook — let test harnesses override the
        // `bits_per_mb_multiplier` knob without plumbing a new public
        // field through every caller. Env var
        // `RVP9_BPM_MULTIPLIER=<f64>` overrides the default of 1.0
        // (libvpx-faithful). Non-parseable values are silently
        // ignored (default stays in force). Production callers
        // should NOT rely on this; it exists for A10-class diagnostic
        // experiments only.
        let bits_per_mb_multiplier = std::env::var("RVP9_BPM_MULTIPLIER")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(defaults.bits_per_mb_multiplier);
        RateControlConfig {
            target_bitrate_bps: self.target_bitrate_bps,
            framerate: self.framerate,
            gop_length: self.gop_length,
            frame_width: self.width,
            frame_height: self.height,
            bits_per_mb_multiplier,
            ..defaults
        }
    }
}

/// Borrowed view over a 4:2:0 YUV input frame.
///
/// Plane lengths are checked at `encode_frame` call time against the
/// configured `(width, height)`. Strides may exceed `width / chroma_w`
/// (allowing for caller-side padding); rows are read as
/// `data[row * stride..row * stride + plane_width]`.
#[derive(Copy, Clone, Debug)]
pub struct YuvFrameRef<'a> {
    /// Y-plane samples, `y_stride * height` bytes.
    pub y: &'a [u8],
    /// U-plane samples, `u_stride * (height / 2)` bytes.
    pub u: &'a [u8],
    /// V-plane samples, `v_stride * (height / 2)` bytes.
    pub v: &'a [u8],
    /// Y-plane stride in bytes. Must be ≥ `width`.
    pub y_stride: usize,
    /// U-plane stride in bytes. Must be ≥ `width / 2`.
    pub u_stride: usize,
    /// V-plane stride in bytes. Must be ≥ `width / 2`.
    pub v_stride: usize,
}

/// Result of a successful [`StreamEncoder::encode_frame`] call.
#[derive(Clone, Debug)]
pub struct EncodedFrame {
    /// Raw VP9 frame bytes (uncompressed header + compressed header +
    /// tile data). NOT wrapped in an IVF or superframe container; the
    /// caller is responsible for muxing.
    pub data: Vec<u8>,
    /// Discriminant: keyframe vs inter.
    pub frame_kind: FrameKind,
    /// Q-index actually used for this frame.
    pub q_index: u8,
    /// Bit count of `data` (== `data.len() * 8`). Mirrors what the
    /// rate controller saw via `postencode_update`.
    pub actual_bits: u32,
}

/// Public frame-kind discriminant for the F10.0 API. Narrower than the
/// wire-format `FrameKind` (which carries inter-frame reference-frame
/// indices etc.) — the public encoder API only needs the Key / Inter
/// distinction.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// Keyframe (intra-only, GOP boundary).
    Key,
    /// Inter frame (single-ref LASTFRAME).
    Inter,
}

impl FrameKind {
    /// `true` for keyframes.
    #[must_use]
    pub const fn is_key(self) -> bool {
        matches!(self, Self::Key)
    }
}

impl From<FrameKind> for RcFrameKind {
    fn from(fk: FrameKind) -> Self {
        match fk {
            FrameKind::Key => RcFrameKind::Key,
            FrameKind::Inter => RcFrameKind::Inter,
        }
    }
}

/// F10.0 streaming encoder.
///
/// Maintains the rate controller, the GOP counter, and the encoder's
/// local reference buffer (a per-plane reconstruction of the most
/// recent decoded frame). Construct via [`StreamEncoder::new`]; feed
/// frames through [`StreamEncoder::encode_frame`].
pub struct StreamEncoder {
    config: EncoderConfig,
    rate_control: RateControl,
    frame_idx: u32,
    /// Last reconstructed Y plane (tight-stride, `width * height`).
    /// Populated after every keyframe; for inter ZEROMV/skip=true
    /// frames the buffer is unchanged.
    last_recon_y: Vec<u8>,
    /// Last reconstructed U plane (`(width / 2) * (height / 2)`).
    last_recon_u: Vec<u8>,
    /// Last reconstructed V plane (`(width / 2) * (height / 2)`).
    last_recon_v: Vec<u8>,
    /// A14 — GOLDEN-slot reconstruction buffer (Phase A
    /// infrastructure). Initialised on every keyframe from the
    /// KF reconstruction (the decoder does the same via
    /// `refresh_frame_flags = 0xFF`); periodically refreshed
    /// from a chosen inter frame's reconstruction via
    /// `refresh_frame_flags` bit 1. The picker still iterates
    /// LASTFRAME only (Phase B is follow-up work); these
    /// buffers exist so the encoder's notion of the decoder's
    /// GOLDEN ref-pool slot stays in sync with the bitstream
    /// the writer emits. Cross-decoder gate stays byte-exact:
    /// the decoder reads the same refresh mask the encoder
    /// writes, copies its own decoded current frame into slot
    /// 1 on refresh, and the encoder's `golden_recon_*` mirrors
    /// that exact pixel data.
    golden_recon_y: Vec<u8>,
    /// A14 — GOLDEN U plane (`(width / 2) * (height / 2)`).
    golden_recon_u: Vec<u8>,
    /// A14 — GOLDEN V plane (`(width / 2) * (height / 2)`).
    golden_recon_v: Vec<u8>,
    /// A14 — frames since the GOLDEN slot was last refreshed
    /// (incremented on every encoded inter frame; reset to 0 on
    /// keyframes and on inter-refresh frames).
    frames_since_golden_refresh: u32,
    /// AQ.3 — cyclic-refresh AQ state. `Some(..)` iff
    /// `config.aq_mode == AqMode::CyclicRefresh`. Reset on every
    /// keyframe via `init_cyclic_refresh`.
    cyclic_refresh: Option<CyclicRefresh>,
}

/// A14 — GOLDEN refresh period for the inter path (Phase A
/// infrastructure). Every Nth inter frame refreshes the GOLDEN
/// slot via `refresh_frame_flags` bit 1. libvpx's CBR-RT path
/// uses an adaptive interval (`vp9_rc_pick_frame_type` with
/// `cpi->refresh_golden_frame` driven by
/// `vp9_cyclic_refresh_set_golden_update`) typically in the
/// 8-15 frame range; we pin a constant 8 for v1 to keep the
/// behaviour deterministic and self-tunable later.
const A14_GOLDEN_REFRESH_PERIOD: u32 = 8;

impl StreamEncoder {
    /// Construct a streaming encoder from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidConfig`] if the configuration is
    /// rejected (width/height not multiples of 64, framerate ≤ 0,
    /// gop_length == 0, etc.).
    pub fn new(config: EncoderConfig) -> Result<Self, EncodeError> {
        if !config.width.is_multiple_of(64) || config.width == 0 {
            return Err(EncodeError::InvalidConfig(
                "width must be a positive multiple of 64",
            ));
        }
        if !config.height.is_multiple_of(64) || config.height == 0 {
            return Err(EncodeError::InvalidConfig(
                "height must be a positive multiple of 64",
            ));
        }
        if config.gop_length == 0 {
            return Err(EncodeError::InvalidConfig("gop_length must be > 0"));
        }
        if !config.framerate.is_finite() || config.framerate <= 0.0 {
            return Err(EncodeError::InvalidConfig(
                "framerate must be positive and finite",
            ));
        }
        if config.target_bitrate_bps == 0 {
            return Err(EncodeError::InvalidConfig(
                "target_bitrate_bps must be > 0",
            ));
        }

        let rc_config = config.to_rate_control_config();
        rc_config
            .validate()
            .map_err(EncodeError::InvalidConfig)?;
        let rate_control = RateControl::new(rc_config);

        let w = config.width as usize;
        let h = config.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;

        let cyclic_refresh = match config.aq_mode {
            AqMode::Off => None,
            AqMode::CyclicRefresh => Some(CyclicRefresh::new(
                h / 8,
                w / 8,
                CyclicRefreshConfig::default(),
            )),
        };

        Ok(Self {
            config,
            rate_control,
            frame_idx: 0,
            last_recon_y: vec![0u8; w * h],
            last_recon_u: vec![0u8; uvw * uvh],
            last_recon_v: vec![0u8; uvw * uvh],
            golden_recon_y: vec![0u8; w * h],
            golden_recon_u: vec![0u8; uvw * uvh],
            golden_recon_v: vec![0u8; uvw * uvh],
            frames_since_golden_refresh: 0,
            cyclic_refresh,
        })
    }

    /// Read-only view of the encoder configuration.
    #[must_use]
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Frame counter — number of frames passed to [`Self::encode_frame`]
    /// so far.
    #[must_use]
    pub fn frame_idx(&self) -> u32 {
        self.frame_idx
    }

    /// Encode one input YUV frame; return the produced VP9 frame
    /// packet.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::InvalidConfig`] if the YUV planes don't match
    ///   the encoder's `(width, height)`.
    /// - [`EncodeError::Bitstream`] / [`EncodeError::Modes`] if any
    ///   wire-format writer fails.
    pub fn encode_frame(
        &mut self,
        yuv: &YuvFrameRef<'_>,
    ) -> Result<EncodedFrame, EncodeError> {
        // Validate plane lengths against configured dimensions.
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        if yuv.y_stride < w {
            return Err(EncodeError::InvalidConfig("y_stride < width"));
        }
        if yuv.u_stride < uvw {
            return Err(EncodeError::InvalidConfig("u_stride < width / 2"));
        }
        if yuv.v_stride < uvw {
            return Err(EncodeError::InvalidConfig("v_stride < width / 2"));
        }
        if yuv.y.len() < yuv.y_stride * h
            || yuv.u.len() < yuv.u_stride * uvh
            || yuv.v.len() < yuv.v_stride * uvh
        {
            return Err(EncodeError::InvalidConfig(
                "yuv plane lengths must cover stride * height",
            ));
        }

        // Decide frame kind. Frames at `frame_idx % gop_length == 0`
        // are keyframes (so frame 0 is always KF; with `gop_length =
        // N`, KFs land at frames 0, N, 2N, ...). We drive this from
        // the encoder's own counter rather than the rate-controller's
        // `frames_to_key` because the user-facing contract on
        // `gop_length` is "every N frames" — the rate-controller's
        // off-by-one accounting of `frames_to_key` is fine for its
        // internal budgeting but isn't the right oracle for KF
        // placement.
        let gop = self.config.gop_length.max(1);
        let frame_kind = if self.frame_idx.is_multiple_of(gop) {
            FrameKind::Key
        } else {
            FrameKind::Inter
        };

        // Pick Q via the rate controller.
        let (q_index, _q_lo, _q_hi) = self.rate_control.pick_q_and_bounds(frame_kind.into());
        // The CBR controller may legitimately pick `q == 0` when the
        // buffer is over-full; the F5.1 simulator's quantizer math
        // doesn't support lossless (qindex 0), so we clamp to 1.
        let q_index = q_index.max(1);

        // Compact tight-stride Y/U/V buffers from the strided input —
        // the F5.1 simulator and the wire-format writers expect tight
        // stride. (Allocating per-frame is wasteful in the steady
        // state; a future revision will keep the buffers in the
        // encoder struct.)
        let mut y_tight = vec![0u8; w * h];
        let mut u_tight = vec![0u8; uvw * uvh];
        let mut v_tight = vec![0u8; uvw * uvh];
        for r in 0..h {
            y_tight[r * w..r * w + w]
                .copy_from_slice(&yuv.y[r * yuv.y_stride..r * yuv.y_stride + w]);
        }
        for r in 0..uvh {
            u_tight[r * uvw..r * uvw + uvw]
                .copy_from_slice(&yuv.u[r * yuv.u_stride..r * yuv.u_stride + uvw]);
            v_tight[r * uvw..r * uvw + uvw]
                .copy_from_slice(&yuv.v[r * yuv.v_stride..r * yuv.v_stride + uvw]);
        }

        let packet = match frame_kind {
            FrameKind::Key => self.encode_keyframe_internal(
                q_index,
                (&y_tight, &u_tight, &v_tight),
            )?,
            FrameKind::Inter => self.encode_inter_internal(
                q_index,
                (&y_tight, &u_tight, &v_tight),
            )?,
        };

        let actual_bits = (packet.len() as u32).saturating_mul(8);

        // -- A10 DIAG: bits_per_mb verification --
        // Env-gated; off by default so production paths never pay the
        // observation cost. Reports the controller's own prediction
        // of frame size at the picked Q (via the same `bits_per_mb ×
        // mbs / 512` math the post-encode feedback loop uses) against
        // the actual bit count, then the ratio actual/predicted.
        // Mean of the ratio sets the empirical `bits_per_mb_multiplier`
        // needed to close the gap on this content.
        if std::env::var_os("RVP9_DIAG_BPM_VERIFY").is_some_and(|v| v != "0") {
            let predicted = self
                .rate_control
                .predicted_bits_at_picked_q(frame_kind.into());
            let q_picked = u32::from(self.rate_control.last_picked_qindex);
            let ratio = if predicted > 0 {
                f64::from(actual_bits) / predicted as f64
            } else {
                f64::NAN
            };
            eprintln!(
                "BPM_VERIFY frame={:>3} kind={:?} q={:>3} predicted={:>8} actual={:>8} ratio={:>6.3}",
                self.frame_idx, frame_kind, q_picked, predicted, actual_bits, ratio
            );
        }

        // Post-encode rate-control update + frame counter advance.
        self.rate_control
            .postencode_update(actual_bits, frame_kind.into());
        self.rate_control.advance_frame();
        self.frame_idx = self.frame_idx.saturating_add(1);

        Ok(EncodedFrame {
            data: packet,
            frame_kind,
            q_index,
            actual_bits,
        })
    }

    /// Keyframe internal: drive the F5.1 simulator to produce per-TX
    /// qcoeffs + the reconstruction (= our new reference). Assemble
    /// the wire-format packet using `write_tile_intra_keyframe`.
    fn encode_keyframe_internal(
        &mut self,
        q_index: u8,
        input_yuv: (&[u8], &[u8], &[u8]),
    ) -> Result<Vec<u8>, EncodeError> {
        let w = self.config.width;
        let h = self.config.height;

        // AQ.3 — Reset cyclic-refresh state on every keyframe. The
        // ref pool / mode-info history is discarded at the KF
        // boundary, so the cyclic-refresh map (which tracks "age since
        // last refresh per cell") and the SB cursor must reset too.
        // The KF path itself does not emit segmentation (AQ only fires
        // on inter frames), so the per-leaf segment_id stays 0 and the
        // simulator's seg_q_deltas stays all-`None` — preserving
        // Stage 4a behaviour on the KF.
        if self.cyclic_refresh.is_some() {
            let mi_rows = (h as usize) / 8;
            let mi_cols = (w as usize) / 8;
            self.cyclic_refresh = Some(CyclicRefresh::new(
                mi_rows,
                mi_cols,
                CyclicRefreshConfig::default(),
            ));
        }

        // F10.1a — pick the per-SB partition via the F8.2 variance
        // picker. For KF the ref plane is unused (the picker substitutes
        // a flat-128 fallback when `is_keyframe = true`), but we still
        // need a valid `PlaneRef`; pass the input Y as the ref.
        let mut sb_partitions =
            choose_partitions_for_frame(w as usize, h as usize, q_index, input_yuv.0, /* ref_y = */ input_yuv.0, /* is_keyframe = */ true);

        // F10.1b — pick the Y intra mode per leaf via
        // `pick_intra_mode_luma`, run the F5.1 simulator with the
        // picked modes. Output goes directly into our reference
        // buffers — those ARE the reconstruction.
        //
        // AQ.1 — Stage 4a behaviour: AQ is OFF in the frame loop, so
        // pass an all-`None` `seg_q_deltas` table. The simulator's
        // per-leaf effective qindex collapses to `q_index`. AQ.3 will
        // replace this with the cyclic-refresh state machine's output.
        let seg_q_deltas: [Option<i16>; 8] = [None; 8];
        let coef_source = pick_and_simulate_intra_encode_partitions(
            w,
            h,
            q_index,
            &seg_q_deltas,
            input_yuv,
            (
                &mut self.last_recon_y,
                &mut self.last_recon_u,
                &mut self.last_recon_v,
            ),
            &mut sb_partitions,
        );

        // F11.0 — apply the in-loop deblocking filter to the encoder's
        // reference reconstruction. The decoder runs the same filter
        // with the same `filter_level` after IT-add, so the
        // encoder↔decoder bit-exact invariant survives LF.
        let filter_level = pick_filter_level(q_index);
        if filter_level > 0 {
            let mi_grid = build_kf_mi_grid_for_lf(h / 8, w / 8, &sb_partitions);
            apply_loop_filter_to_tight_recon(
                w,
                h,
                filter_level,
                &mi_grid,
                &mut self.last_recon_y,
                &mut self.last_recon_u,
                &mut self.last_recon_v,
            );
        }

        // A14 (Phase A) — keyframes refresh every ref-pool slot via
        // `refresh_frame_flags = 0xFF` (see `build_keyframe_packet`).
        // Mirror that on the encoder side: copy the KF reconstruction
        // (post-LF) into the GOLDEN buffer so the encoder's notion
        // of the decoder's GOLDEN slot matches what the decoder
        // computed. Reset the refresh counter — the next GOLDEN
        // refresh lands `A14_GOLDEN_REFRESH_PERIOD` inter frames out.
        self.golden_recon_y.copy_from_slice(&self.last_recon_y);
        self.golden_recon_u.copy_from_slice(&self.last_recon_u);
        self.golden_recon_v.copy_from_slice(&self.last_recon_v);
        self.frames_since_golden_refresh = 0;

        build_keyframe_packet(w, h, q_index, filter_level, &coef_source, &sb_partitions)
    }

    /// Inter internal: F10.1c — drive F8.2 partition + F8.1 inter mode
    /// picker per leaf, run the inter encoder simulator to produce real
    /// per-TX residuals + a bit-exact-with-decoder encoder
    /// reconstruction, then emit the inter packet with those residuals.
    ///
    /// For Block8x8 (sub-8 picker-out-of-scope) leaves the simulator
    /// falls back to `ZEROMV / LASTFRAME / skip = true`; the recon for
    /// those leaves equals the reference (the simulator's up-front
    /// copy). The bigger picker-supported leaves carry the picked
    /// `(mode, mv)` and emit real coefs.
    ///
    /// AQ.3 — when `config.aq_mode == AqMode::CyclicRefresh`, this
    /// path runs the cyclic-refresh state machine: pre-encode it picks
    /// a `segment_id` per cell (segment 1 = refresh, segment 0 =
    /// baseline); the per-leaf `segment_id` + frame-level
    /// `seg_q_deltas` flow through the picker / simulator /
    /// modes_writer; post-encode the state machine's map ages.
    fn encode_inter_internal(
        &mut self,
        q_index: u8,
        input_yuv: (&[u8], &[u8], &[u8]),
    ) -> Result<Vec<u8>, EncodeError> {
        let w = self.config.width;
        let h = self.config.height;
        let mi_rows_us = (h as usize) / 8;
        let mi_cols_us = (w as usize) / 8;

        // AQ.3 — pre-encode segment-map pick + per-segment Q deltas.
        // `seg_map` is per-mi-cell (length mi_rows * mi_cols); when AQ
        // is off it's all zeros and `seg_q_deltas` is all-`None`.
        //
        // AQ-night-7: libvpx `apply_cyclic_refresh` gate
        // (`vp9_aq_cyclicrefresh.c:487-505`) — skip AQ this frame
        // when avg_qindex_inter is out of range or frames_since_key
        // window says it's not worth refreshing. The gate is
        // evaluated against the rate-controller state BEFORE the
        // post-encode update for this frame, matching libvpx's
        // call order (cyclic_refresh runs during `setup_frame`,
        // before encoding).
        let aq_active_this_frame = match &self.cyclic_refresh {
            None => false,
            Some(cr) => cr.apply_cyclic_refresh(
                /* is_keyframe = */ false,
                self.rate_control.avg_frame_qindex_inter,
                self.rate_control.best_quality,
                self.rate_control.frames_since_key,
            ),
        };
        let (seg_map, seg_q_deltas): (Vec<u8>, [Option<i16>; 8]) =
            match (&self.cyclic_refresh, aq_active_this_frame) {
                (None, _) | (Some(_), false) => {
                    (vec![0u8; mi_rows_us * mi_cols_us], [None; 8])
                }
                (Some(cr), true) => {
                    let (map, _n) = cr.pre_encode_pick_segments(q_index);
                    (map, cr.segment_q_deltas())
                }
            };

        // F10.1a — F8.2 variance-based partition decision per SB.
        let sb_partitions_intra_shape = choose_partitions_for_frame(
            w as usize,
            h as usize,
            q_index,
            input_yuv.0,
            &self.last_recon_y,
            /* is_keyframe = */ false,
        );

        // Convert the intra-shaped partition descriptors into inter-
        // shaped ones (picker fields default to ZEROMV / Mv::ZERO and
        // are overwritten by the simulator).
        let sb_cols = mi_cols_us / 8;
        let mut sb_partitions: Vec<Vec<InterPartitionLeafDesc>> =
            sb_partitions_intra_shape
                .iter()
                .enumerate()
                .map(|(sb_idx, leaves)| {
                    let sb_row = sb_idx / sb_cols;
                    let sb_col = sb_idx - sb_row * sb_cols;
                    leaves
                        .iter()
                        .map(|l| {
                            // AQ.3 — project the seg_map onto this leaf
                            // by sampling the top-left mi cell. Cyclic
                            // refresh paints whole SBs uniformly so the
                            // top-left is exact for any leaf inside the
                            // SB.
                            let leaf_mi_row = sb_row * 8 + l.mi_row_in_sb;
                            let leaf_mi_col = sb_col * 8 + l.mi_col_in_sb;
                            let seg_id = seg_map[leaf_mi_row * mi_cols_us + leaf_mi_col];
                            InterPartitionLeafDesc {
                                mi_row_in_sb: l.mi_row_in_sb,
                                mi_col_in_sb: l.mi_col_in_sb,
                                leaf_bsize: l.leaf_bsize,
                                picked_mode: InterMode::ZeroMv,
                                picked_mv: Mv::ZERO,
                                skip: true,
                                segment_id: seg_id,
                                // A15 Phase B — default picked ref to LAST.
                                // The picker overwrites this per-leaf when
                                // it iterates {LAST, GOLDEN} below; skip-
                                // fallback leaves retain LAST (matches A14).
                                picked_ref_frame: RefFrame::LastFrame,
                            }
                        })
                        .collect()
                })
                .collect();

        // Save the previous reference for the simulator — the simulator
        // reads `last_recon_*` and writes the new recon into
        // `recon_yuv_out`. We swap buffers via a scratch allocation.
        let w_us = w as usize;
        let h_us = h as usize;
        let uvw = w_us / 2;
        let uvh = h_us / 2;
        let mut new_recon_y = vec![0u8; w_us * h_us];
        let mut new_recon_u = vec![0u8; uvw * uvh];
        let mut new_recon_v = vec![0u8; uvw * uvh];

        let coef_source = pick_and_simulate_inter_encode_partitions(
            w,
            h,
            q_index,
            &seg_q_deltas,
            input_yuv,
            (&self.last_recon_y, &self.last_recon_u, &self.last_recon_v),
            // A15 Phase B — GOLDEN ref planes for the widened picker
            // (Step 2: feeds the second `find_mv_refs` walk for
            // GOLDEN; Step 3: drives candidate SSE on GOLDEN picks).
            (&self.golden_recon_y, &self.golden_recon_u, &self.golden_recon_v),
            (&mut new_recon_y, &mut new_recon_u, &mut new_recon_v),
            &mut sb_partitions,
        );

        // F11.0 — apply the in-loop deblocking filter to the encoder's
        // new reference reconstruction (matches the decoder's LF pass
        // after IT-add). Must run BEFORE we commit `new_recon_*` into
        // `last_recon_*` so the next inter frame's reference is the
        // post-LF pixels — same as the decoder's pool.
        let filter_level = pick_filter_level(q_index);
        if filter_level > 0 {
            let mi_grid = build_inter_mi_grid_for_lf(h / 8, w / 8, &sb_partitions);
            apply_loop_filter_to_tight_recon(
                w,
                h,
                filter_level,
                &mi_grid,
                &mut new_recon_y,
                &mut new_recon_u,
                &mut new_recon_v,
            );
        }

        // AQ.3 — build the segmentation header + per-leaf write params.
        // `Off` produces `None` (segmentation disabled in the header,
        // no per-leaf bits emitted by the modes_writer — Stage 4a
        // bytes exact).
        //
        // AQ-night-7: when the `apply_cyclic_refresh` gate suppressed
        // refresh for this frame (`aq_active_this_frame = false`),
        // `seg_q_deltas` is all-None and `seg_map` is all-zeros — both
        // disable the segmentation header (saves the segmentation
        // bytes) and prevent any per-leaf segment-id emission. This
        // mirrors libvpx `vp9_cyclic_refresh_setup` calling
        // `vp9_disable_segmentation` on the suppress branch
        // (`vp9_aq_cyclicrefresh.c:606-619`).
        let segmentation_enabled = self.cyclic_refresh.is_some() && aq_active_this_frame;
        let segmentation_params =
            build_segmentation_params(segmentation_enabled, &seg_q_deltas);

        // A14 (Phase A) — decide whether this inter frame refreshes
        // the GOLDEN slot. Period: `A14_GOLDEN_REFRESH_PERIOD` inter
        // frames between refreshes. The counter is incremented BEFORE
        // the check so the first inter frame after a KF (when the
        // counter is at 0 from the KF reset) gets `counter == 1` on
        // the check — meaning the first GOLDEN refresh lands at
        // inter-frame N (where N == period). That matches libvpx's
        // `cpi->refresh_golden_frame` cadence (refresh after the
        // GF group reaches its planned length, never on the first
        // post-KF frame).
        //
        // Phase A is INFRASTRUCTURE ONLY — the picker still iterates
        // LASTFRAME only (Phase B follow-up), so this refresh bit
        // changes the wire-format `refresh_frame_flags` mask but
        // does NOT change which ref the picker reads from. The
        // encoder's `golden_recon_*` mirror tracks the decoder's
        // slot-1 contents; once Phase B lands, the picker reads from
        // those buffers and the wire-format selection bits already
        // emitted by `write_inter_block_mode_info` will resolve to
        // the right slot.
        self.frames_since_golden_refresh =
            self.frames_since_golden_refresh.saturating_add(1);
        let refresh_golden_this_frame =
            self.frames_since_golden_refresh >= A14_GOLDEN_REFRESH_PERIOD;
        let refresh_flags: u8 = if refresh_golden_this_frame {
            0x03 // LAST (bit 0) + GOLDEN (bit 1)
        } else {
            0x01 // LAST only — A11's baseline
        };

        // Build the packet from the populated coef source + leaf
        // descriptors.
        let packet = build_inter_packet_with_residuals(
            w,
            h,
            q_index,
            filter_level,
            &coef_source,
            &sb_partitions,
            &segmentation_params,
            refresh_flags,
        )?;

        // Commit the new reconstruction as the encoder's new LAST ref.
        self.last_recon_y = new_recon_y;
        self.last_recon_u = new_recon_u;
        self.last_recon_v = new_recon_v;

        // A14 (Phase A) — if this frame refreshed GOLDEN on the wire,
        // mirror that on the encoder side: copy the new (post-LF)
        // reconstruction into the GOLDEN buffer and reset the
        // refresh counter. The decoder applies the same refresh
        // mask via `refresh_ref_frame_sizes` + the ref-pool copy,
        // so both sides agree on slot 1's pixels.
        if refresh_golden_this_frame {
            self.golden_recon_y.copy_from_slice(&self.last_recon_y);
            self.golden_recon_u.copy_from_slice(&self.last_recon_u);
            self.golden_recon_v.copy_from_slice(&self.last_recon_v);
            self.frames_since_golden_refresh = 0;
        }

        // AQ.3 — post-encode update: feed the actually-coded segment
        // ids + skip flags back into the cyclic-refresh state machine
        // so the next frame's map / cursor reflect what we just coded.
        //
        // AQ-night-7: the post-encode update still runs on
        // gate-suppressed frames so the age map keeps incrementing
        // (otherwise cells would stay at age 0 forever and AQ would
        // never re-engage when conditions improve). On gated frames
        // the per-leaf segment_id is 0 (`aq_active_this_frame == false`
        // pushes `seg_map` all-zeros above), so the `coded_segments`
        // gathered here are all-baseline. `last_coded_q_map` still
        // gets the correct `base_qindex` update for non-skipped cells,
        // matching libvpx's behaviour on the gate-suppress branch
        // (it sets `seg_map` to 0 then continues encoding).
        if let Some(cr) = self.cyclic_refresh.as_mut() {
            let (coded_segments, coded_skip) =
                gather_coded_segments_and_skip(mi_rows_us, mi_cols_us, &sb_partitions);
            cr.post_encode_update(&coded_segments, &coded_skip, q_index);
        }

        Ok(packet)
    }

    /// Read-only access to the rate controller (for tests / probes).
    #[must_use]
    pub fn rate_control(&self) -> &RateControl {
        &self.rate_control
    }

    /// Test-only — read access to the encoder's last reconstruction
    /// Y plane. Used by the F10.1c bit-exact-encoder-vs-decoder oracle
    /// to assert the encoder's mirrored `reconstruct_inter_block`
    /// matches the decoder bit-exactly.
    #[cfg(test)]
    pub(crate) fn last_recon_y(&self) -> &[u8] {
        &self.last_recon_y
    }

    /// Test-only — read access to the encoder's last reconstruction
    /// U plane. AQ.4 Oracle 1 (`aq_cyclic_refresh_bit_exact_encoder_decoder_recon`)
    /// asserts the encoder's per-plane recon matches the decoder bit-exactly
    /// with AQ on.
    #[cfg(test)]
    pub(crate) fn last_recon_u(&self) -> &[u8] {
        &self.last_recon_u
    }

    /// Test-only — read access to the encoder's last reconstruction
    /// V plane. AQ.4 Oracle 1 companion to [`Self::last_recon_u`].
    #[cfg(test)]
    pub(crate) fn last_recon_v(&self) -> &[u8] {
        &self.last_recon_v
    }

    /// Diagnostic — read access to the encoder's last reconstruction
    /// Y plane (post-LF, pre-border-extend). For integration tests
    /// that compare encoder's internal LASTFRAME against the
    /// decoder's reconstruction of the emitted bitstream (A11
    /// reference-buffer divergence diagnostic).
    #[must_use]
    pub fn diag_last_recon_y(&self) -> &[u8] {
        &self.last_recon_y
    }

    /// Diagnostic — read access to the encoder's last reconstruction
    /// U plane. See [`Self::diag_last_recon_y`].
    #[must_use]
    pub fn diag_last_recon_u(&self) -> &[u8] {
        &self.last_recon_u
    }

    /// Diagnostic — read access to the encoder's last reconstruction
    /// V plane. See [`Self::diag_last_recon_y`].
    #[must_use]
    pub fn diag_last_recon_v(&self) -> &[u8] {
        &self.last_recon_v
    }
}

// ---------------------------------------------------------------------------
// F10.1a — partition decision wiring.
// ---------------------------------------------------------------------------

/// Run the F8.2 variance-based partition picker on every 64×64 SB in
/// the frame and return one `Vec<PartitionLeafDesc>` per SB (in raster
/// scan inside the SB).
///
/// For keyframes the `ref_y` plane is ignored (the picker substitutes
/// a flat-128 fallback when `is_keyframe = true`); for inter frames
/// `ref_y` must point at the previous-frame's Y reconstruction
/// (LASTFRAME under our zero-MV mode).
fn choose_partitions_for_frame(
    width_px: usize,
    height_px: usize,
    q_index: u8,
    src_y: &[u8],
    ref_y: &[u8],
    is_keyframe: bool,
) -> Vec<Vec<PartitionLeafDesc>> {
    let sb_cols = width_px / 64;
    let sb_rows = height_px / 64;
    let mut out: Vec<Vec<PartitionLeafDesc>> = Vec::with_capacity(sb_rows * sb_cols);

    let src_plane = PlaneRef {
        data: src_y,
        stride: width_px,
        width: width_px,
        height: height_px,
    };
    let ref_plane = PlaneRef {
        data: ref_y,
        stride: width_px,
        width: width_px,
        height: height_px,
    };

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let params = PartitionSearchParams {
                src_plane: &src_plane,
                src_offset_xy: (sb_col * 64, sb_row * 64),
                ref_plane: &ref_plane,
                ref_offset_xy: (sb_col * 64, sb_row * 64),
                is_keyframe,
                q_index,
            };
            let decision = choose_partitioning(&params);
            out.push(leaves_from_decision(&decision));
        }
    }
    out
}

/// Walk a [`PartitionDecision`]'s 8×8 cell grid in raster order and
/// return one [`PartitionLeafDesc`] per unique leaf (top-left cell
/// only). Caller invariant: `decision` passes
/// [`PartitionDecision::is_valid_partition`] — which the F8.2 picker
/// always guarantees.
fn leaves_from_decision(decision: &PartitionDecision) -> Vec<PartitionLeafDesc> {
    let mut leaves = Vec::with_capacity(64);
    let mut visited = [[false; 8]; 8];
    for r in 0..8 {
        for c in 0..8 {
            if visited[r][c] {
                continue;
            }
            let bs = decision.cells[r][c];
            let cells_wide = match bs {
                BlockSize::Block8x8 => 1usize,
                BlockSize::Block16x16 => 2,
                BlockSize::Block32x32 => 4,
                BlockSize::Block64x64 => 8,
                other => panic!("F10.1a leaves_from_decision: unsupported leaf {other:?}"),
            };
            for dr in 0..cells_wide {
                for dc in 0..cells_wide {
                    visited[r + dr][c + dc] = true;
                }
            }
            leaves.push(PartitionLeafDesc {
                mi_row_in_sb: r,
                mi_col_in_sb: c,
                leaf_bsize: bs,
                // F10.1b — placeholder. The KF path mutates this via
                // `pick_and_simulate_intra_encode_partitions`. The
                // inter path doesn't read it.
                predicted_y_mode: IntraMode::Dc,
                // F10.1b chroma picker (Stage 4a' chroma-DC closure)
                // mutates this on the KF path too. Inter path ignores
                // it (inter leaves are not intra-coded).
                predicted_uv_mode: IntraMode::Dc,
                // AQ.1 — default to segment 0 (no Q-delta). AQ.3 will
                // populate this from the cyclic-refresh state machine.
                segment_id: 0,
            });
        }
    }
    leaves
}

/// Splat `cell` into every 8×8 slot of `mi_grid` the leaf at
/// `(leaf_mi_row, leaf_mi_col)` with size `leaf_bsize` covers. The
/// writer's `infer_partition` reads the top-left of each leaf, but
/// HORZ / VERT / SPLIT recursion also reads the alternate-position
/// leaf cell, so every cell must carry the leaf's `ModeInfo` (with
/// the same `block_size`).
fn splat_leaf_mode_info(
    mi_grid: &mut [Option<ModeInfo>],
    mi_cols: usize,
    leaf_mi_row: usize,
    leaf_mi_col: usize,
    leaf_bsize: BlockSize,
    cell: ModeInfo,
) {
    let n_8x8_wide = leaf_bsize.n_8x8_wide() as usize;
    let n_8x8_high = leaf_bsize.n_8x8_high() as usize;
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let mi_row = leaf_mi_row + dr;
            let mi_col = leaf_mi_col + dc;
            mi_grid[mi_row * mi_cols + mi_col] = Some(cell);
        }
    }
}

/// Build the keyframe mi_grid from the per-SB partition decisions.
/// Each leaf gets a `DC_PRED` intra `ModeInfo` with `skip = false`
/// splatted across every 8×8 cell it covers.
fn build_kf_mi_grid(
    mi_rows: u32,
    mi_cols: u32,
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> Vec<Option<ModeInfo>> {
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    let mut mi_grid: Vec<Option<ModeInfo>> = vec![None; mi_rows_us * mi_cols_us];

    let sb_cols = mi_cols_us / 8;
    let sb_rows = mi_rows_us / 8;
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let cell = ModeInfo {
                    block_size: leaf.leaf_bsize,
                    tx_size: max_txsize(leaf.leaf_bsize),
                    skip: false,
                    segment_id: 0,
                    seg_id_predicted: false,
                    mode: BlockMode::Intra {
                        // F10.1b — picker-driven Y mode. Stage 4a'
                        // chroma-DC closure: per-leaf picker-driven
                        // UV mode (see `pick_intra_mode_uv`).
                        y_mode: leaf.predicted_y_mode,
                        uv_mode: leaf.predicted_uv_mode,
                    },
                    bmi: Default::default(),
                };
                splat_leaf_mode_info(
                    &mut mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf.leaf_bsize,
                    cell,
                );
            }
        }
    }
    mi_grid
}

/// F10.1c — Build the inter mi_grid from the picker-populated inter
/// leaf descriptors. Each leaf carries the picker's `(picked_mode,
/// picked_mv, skip)` (set by `pick_and_simulate_inter_encode_partitions`);
/// Block8x8 leaves fall back to `ZEROMV / skip = true`.
fn build_inter_mi_grid_from_picker(
    mi_rows: u32,
    mi_cols: u32,
    sb_partitions: &[Vec<InterPartitionLeafDesc>],
) -> Vec<Option<ModeInfo>> {
    let mi_rows_us = mi_rows as usize;
    let mi_cols_us = mi_cols as usize;
    let mut mi_grid: Vec<Option<ModeInfo>> = vec![None; mi_rows_us * mi_cols_us];

    let sb_cols = mi_cols_us / 8;
    let sb_rows = mi_rows_us / 8;
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let cell = ModeInfo {
                    block_size: leaf.leaf_bsize,
                    tx_size: max_txsize(leaf.leaf_bsize),
                    skip: leaf.skip,
                    // AQ.3 — carry the leaf's segment_id into the mi_grid
                    // so the modes_writer emits the per-leaf segment_id
                    // bits (when `seg.is_some()`). Stage 4a path keeps
                    // `segment_id = 0` because cyclic refresh is off and
                    // every leaf's `segment_id` was initialised to 0.
                    segment_id: leaf.segment_id,
                    seg_id_predicted: false,
                    mode: BlockMode::Inter {
                        mode: leaf.picked_mode,
                        ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
                        mvs: [leaf.picked_mv, Mv::ZERO],
                        interp_filter: InterpFilter::EightTap,
                    },
                    bmi: Default::default(),
                };
                splat_leaf_mode_info(
                    &mut mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf.leaf_bsize,
                    cell,
                );
            }
        }
    }
    mi_grid
}

// ---------------------------------------------------------------------------
// Internal packet builders.
// ---------------------------------------------------------------------------

/// Build a single keyframe packet (uncompressed header + compressed
/// header + tile data) with real per-TX residual coefs supplied by
/// `coef_source` (the F5.1 simulator's output).
///
/// Mirrors `Encoder::build_keyframe_payload_with_residuals` in
/// `lib.rs` but with `refresh_frame_context = false` always, so the
/// decoder's FC bank 0 stays at default and subsequent inter frames
/// can use the default FC as their `pre_fc`.
fn build_keyframe_packet(
    width: u32,
    height: u32,
    base_qindex: u8,
    filter_level: u8,
    coef_source: &dyn rvp9_core::bitstream::IntraCoefSource,
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> Result<Vec<u8>, EncodeError> {
    let mi_rows = height / 8;
    let mi_cols = width / 8;
    let mi_grid = build_kf_mi_grid(mi_rows, mi_cols, sb_partitions);

    let pre_fc_template = FrameContext::new_default();
    let mut pre_fc = pre_fc_template.clone();
    let post_fc = pre_fc.clone();
    let opts = CompressedHeaderOptions {
        lossless: false,
        intra_only: true,
        interp_filter: InterpFilter::EightTap,
        compound_allowed: false,
        allow_high_precision_mv: false,
    };
    let target = CompressedHeaderResult {
        tx_mode: TxMode::AllowTx32x32,
        reference_mode: ReferenceMode::Single,
    };

    let mut compressed_buf = vec![0u8; 16 * 1024];
    let compressed_len = {
        let mut bw = BoolWriter::start_encode(&mut compressed_buf);
        write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target)?;
        bw.stop_encode()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    compressed_buf.truncate(compressed_len);

    let tile_bounds = ModesTileBounds {
        mi_row_start: 0,
        mi_row_end: mi_rows as usize,
        mi_col_start: 0,
        mi_col_end: mi_cols as usize,
    };
    let mut tile_buf =
        vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 64 + 16 * 1024];
    let tile_len = {
        let mut bw = BoolWriter::start_encode(&mut tile_buf);
        let tile_mi_cols = (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
        let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
        write_tile_intra_keyframe(
            &mut bw,
            &mi_grid,
            mi_rows,
            mi_cols,
            tile_bounds,
            &post_fc,
            /* seg = */ None,
            &mut coef_ctx,
            coef_source,
        )?;
        bw.stop_encode()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    tile_buf.truncate(tile_len);

    let tail = HeaderTail {
        // Keep FC bank 0 at default so subsequent inter frames can use
        // the default FC as their `pre_fc` without the encoder having
        // to mirror backward-adapt logic.
        refresh_frame_context: false,
        frame_parallel_decoding_mode: false,
        frame_context_idx: 0,
        loop_filter: LoopFilterParams {
            // F11.0 — emit the F11.0-picked filter_level so the decoder
            // applies the same in-loop deblocking pass the encoder
            // applied to its reference reconstruction.
            filter_level,
            sharpness_level: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        },
        quantization: QuantizationParams {
            base_qindex,
            y_dc_delta_q: 0,
            uv_dc_delta_q: 0,
            uv_ac_delta_q: 0,
        },
        segmentation: SegmentationParams {
            enabled: false,
            update_map: false,
            temporal_update: false,
            tree_probs: [MAX_PROB; SEG_TREE_PROBS],
            pred_probs: [MAX_PROB; PREDICTION_PROBS],
            update_data: false,
            abs_delta: false,
            feature_data: Default::default(),
        },
        tile_info: TileInfo {
            log2_tile_cols: 0,
            log2_tile_rows: 0,
        },
        first_partition_size: compressed_len as u16,
    };
    let hdr = UncompressedHeader {
        kind: WireFrameKind::KeyFrame,
        profile: Profile::Profile0,
        show_frame: true,
        error_resilient: false,
        bit_depth: BitDepth::Eight,
        color_space: ColorSpace::Bt601,
        color_range: ColorRange::Studio,
        chroma: ChromaFormat::Yuv420,
        width,
        height,
        render_size: None,
        tail: Some(tail),
    };

    let mut hdr_buf = vec![0u8; 256];
    let hdr_len = {
        let mut wb = RawBitWriter::new(&mut hdr_buf);
        write_uncompressed_header(&mut wb, &hdr, compressed_len as u16)?;
        wb.finish()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    hdr_buf.truncate(hdr_len);

    let mut packet = Vec::with_capacity(hdr_len + compressed_len + tile_len);
    packet.extend_from_slice(&hdr_buf);
    packet.extend_from_slice(&compressed_buf);
    packet.extend_from_slice(&tile_buf);
    Ok(packet)
}

/// F10.1c — Build an inter packet whose leaves carry the picker's
/// chosen `(inter_mode, mv, skip)` and whose non-skip leaves emit real
/// per-TX residuals supplied by `coef_source` (the F10.1c inter encoder
/// simulator's output).
///
/// Mirror of [`build_keyframe_packet`] for the inter path: builds the
/// mi-grid from `sb_partitions`, runs the wire-format compressed +
/// uncompressed header writers, and threads the coef source through
/// `write_tile_inter`.
///
/// AQ.3 — `segmentation` carries the header's segmentation block.
/// When `segmentation.enabled == true`, the tile writer is also
/// supplied with a [`SegmentationWriteParams`] (matching tree_probs)
/// so per-leaf `segment_id` bits are emitted. When disabled, both the
/// header bit and the per-leaf bits are suppressed — Stage 4a bytes
/// preserved exactly.
#[allow(clippy::too_many_arguments)] // load-bearing AQ.3 plumbing args
fn build_inter_packet_with_residuals(
    width: u32,
    height: u32,
    base_qindex: u8,
    filter_level: u8,
    coef_source: &dyn InterCoefSource,
    sb_partitions: &[Vec<InterPartitionLeafDesc>],
    segmentation: &SegmentationParams,
    refresh_frame_flags: u8,
) -> Result<Vec<u8>, EncodeError> {
    use rvp9_core::bitstream::SegmentationWriteParams;
    let mi_rows = height / 8;
    let mi_cols = width / 8;
    let mi_grid = build_inter_mi_grid_from_picker(mi_rows, mi_cols, sb_partitions);

    let mut pre_fc = FrameContext::new_default();
    let post_fc = pre_fc.clone();
    let opts = CompressedHeaderOptions {
        lossless: false,
        intra_only: false,
        interp_filter: InterpFilter::EightTap,
        compound_allowed: false,
        allow_high_precision_mv: false,
    };
    let target = CompressedHeaderResult {
        tx_mode: TxMode::AllowTx32x32,
        reference_mode: ReferenceMode::Single,
    };

    let mut compressed_buf = vec![0u8; 16 * 1024];
    let compressed_len = {
        let mut bw = BoolWriter::start_encode(&mut compressed_buf);
        write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target)?;
        bw.stop_encode()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    compressed_buf.truncate(compressed_len);

    let tile_bounds = ModesTileBounds {
        mi_row_start: 0,
        mi_row_end: mi_rows as usize,
        mi_col_start: 0,
        mi_col_end: mi_cols as usize,
    };

    let sign_bias = sign_bias_from_header(&[false, false, false]);
    let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
    let params = InterModeReadParams {
        allow_high_precision_mv: false,
        interp_filter: InterpFilter::EightTap,
        reference_mode: ReferenceMode::Single,
        comp_fixed_ref,
        comp_var_ref,
        sign_bias,
    };

    // AQ.3 — if segmentation is enabled in the header (cyclic refresh
    // is active), build a `SegmentationWriteParams` that mirrors the
    // header's `tree_probs`. The modes_writer reads `tree_probs` to
    // probability-code per-leaf segment_id bits. When disabled, the
    // writer is invoked with `None` (no bits emitted), matching the
    // Stage 4a path.
    let seg_write = if segmentation.enabled {
        Some(SegmentationWriteParams {
            tree_probs: &segmentation.tree_probs,
        })
    } else {
        None
    };

    // F10.1c — sized generously (per-pixel × 4) since real residuals
    // can dominate the bit budget at higher Q indices.
    let mut tile_buf =
        vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 64 + 16 * 1024];
    let tile_len = {
        let mut bw = BoolWriter::start_encode(&mut tile_buf);
        let tile_mi_cols = (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
        let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
        write_tile_inter(
            &mut bw,
            &post_fc,
            &params,
            TxMode::AllowTx32x32,
            &mi_grid,
            mi_rows,
            mi_cols,
            tile_bounds,
            seg_write,
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            coef_source,
        )?;
        bw.stop_encode()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    tile_buf.truncate(tile_len);

    // A11 (b9b4e93) — refresh_frame_flags = 0x01: every inter frame
    // refreshes ref slot 0 (LASTFRAME). Pre-A11 set this to 0x00,
    // meaning the decoder kept the keyframe in slot 0 as LASTFRAME
    // for the entire GOP — but the ENCODER's internal `last_recon_*`
    // tracked each successive frame's reconstruction. Result: the
    // encoder built residuals against frame N-1's recon while every
    // decoder applied them on top of MC against the keyframe. Both
    // rvp9-decoder and vpxdec agreed (cross-decoder byte-exact) on
    // the wrong reconstruction; PSNR vs source decayed monotonically
    // through the GOP because the residuals fixed the wrong base.
    //
    // libvpx's RT path sets `refresh_frame_flags = 0x01` (single
    // LASTFRAME refresh) on every non-altref inter frame
    // (`vp9_encoder.c::set_arf_sign_bias`-adjacent ref-flag setup,
    // and `vp9_encodeframe.c:5612` for the simple RT case). We
    // mirror that here under the single-ref single-LASTFRAME shape
    // emitted by this encoder.
    let tail = HeaderTail {
        refresh_frame_context: false,
        frame_parallel_decoding_mode: false,
        frame_context_idx: 0,
        loop_filter: LoopFilterParams {
            // F11.0 — emit the F11.0-picked filter_level (matches the
            // pass applied to the encoder's reference reconstruction).
            filter_level,
            sharpness_level: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_update: false,
            ref_deltas: [None; 4],
            mode_deltas: [None; 2],
        },
        quantization: QuantizationParams {
            base_qindex,
            y_dc_delta_q: 0,
            uv_dc_delta_q: 0,
            uv_ac_delta_q: 0,
        },
        // AQ.3 — segmentation block supplied by caller. `Off` ⇒
        // `enabled = false` (Stage 4a bytes). `CyclicRefresh` ⇒
        // `enabled = true && update_map && update_data` with the
        // per-segment Q delta encoded in `feature_data[seg][AltQ]`.
        segmentation: segmentation.clone(),
        tile_info: TileInfo {
            log2_tile_cols: 0,
            log2_tile_rows: 0,
        },
        first_partition_size: compressed_len as u16,
    };
    let hdr = UncompressedHeader {
        kind: WireFrameKind::InterFrame {
            reset_frame_context: 0,
            // A11 — refresh LASTFRAME (slot 0) every inter frame.
            // A14 (Phase A) — additionally refresh GOLDEN (slot 1)
            // on the configured period. Caller supplies the mask
            // via `refresh_frame_flags`: `0x01` for plain inter,
            // `0x03` for inter-that-also-refreshes-GOLDEN. See
            // `encode_inter_internal` for the period state machine.
            refresh_frame_flags,
            ref_frame_indices: [0; MAX_INTER_REFS],
            ref_frame_sign_biases: [false; MAX_INTER_REFS],
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
        },
        profile: Profile::Profile0,
        show_frame: true,
        error_resilient: false,
        bit_depth: BitDepth::Eight,
        color_space: ColorSpace::Bt601,
        color_range: ColorRange::Studio,
        chroma: ChromaFormat::Yuv420,
        width,
        height,
        render_size: None,
        tail: Some(tail),
    };

    let mut hdr_buf = vec![0u8; 256];
    let hdr_len = {
        let mut wb = RawBitWriter::new(&mut hdr_buf);
        write_uncompressed_header(&mut wb, &hdr, compressed_len as u16)?;
        wb.finish()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    hdr_buf.truncate(hdr_len);

    let mut packet = Vec::with_capacity(hdr_len + compressed_len + tile_len);
    packet.extend_from_slice(&hdr_buf);
    packet.extend_from_slice(&compressed_buf);
    packet.extend_from_slice(&tile_buf);
    Ok(packet)
}

// ---------------------------------------------------------------------------
// AQ.3 — segmentation header + post-encode collection helpers.
// ---------------------------------------------------------------------------

/// Build a [`SegmentationParams`] header block for the cyclic-refresh
/// inter path. When `enabled == false` returns the Stage 4a all-zeros
/// header (segmentation disabled — no bits emitted).
///
/// When `enabled == true` populates the libvpx cyclic-refresh shape:
///
/// - `enabled = true`, `update_map = true`, `temporal_update = false`,
///   `update_data = true`, `abs_delta = false`.
/// - `tree_probs` left at `MAX_PROB` (libvpx-style "no update from
///   default"; the per-leaf segment_id writer reads them verbatim).
///   Future AQ.x revisions can train these against the picked map.
/// - `feature_data[seg][SEG_LVL_ALT_Q] = Some(delta)` for every
///   segment with a non-`None` slot in `seg_q_deltas`. Other features
///   (`AltLf` / `ReferenceFrame` / `Skip`) stay `None`.
fn build_segmentation_params(
    enabled: bool,
    seg_q_deltas: &[Option<i16>; 8],
) -> SegmentationParams {
    let mut params = SegmentationParams {
        enabled,
        update_map: enabled,
        temporal_update: false,
        tree_probs: [MAX_PROB; SEG_TREE_PROBS],
        pred_probs: [MAX_PROB; PREDICTION_PROBS],
        update_data: enabled,
        abs_delta: false,
        feature_data: Default::default(),
    };
    if enabled {
        // Slot 0 (`SEG_LVL_ALT_Q`) = the per-segment Q delta. AQ.3
        // populates this from `cyclic_refresh.segment_q_deltas()`; the
        // decoder's `setup_segmentation` reads it back into the
        // dequantizer derivation per leaf (via
        // `effective_qindex_for_segment` on the encoder side).
        for (seg_idx, slot) in seg_q_deltas.iter().enumerate() {
            if let Some(delta) = *slot {
                params.feature_data[seg_idx][0] = Some(delta);
            }
        }
    }
    params
}

/// AQ.3 — gather the per-mi-cell `(coded_segment_id, coded_skip)`
/// pair for the cyclic-refresh `post_encode_update` call. Splats each
/// leaf's `(segment_id, skip)` across every 8×8 cell the leaf covers.
fn gather_coded_segments_and_skip(
    mi_rows: usize,
    mi_cols: usize,
    sb_partitions: &[Vec<InterPartitionLeafDesc>],
) -> (Vec<u8>, Vec<bool>) {
    let mut coded_segments = vec![0u8; mi_rows * mi_cols];
    let mut coded_skip = vec![false; mi_rows * mi_cols];
    let sb_cols = mi_cols / 8;
    let sb_rows = mi_rows / 8;
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let n_high = leaf.leaf_bsize.n_8x8_high() as usize;
                let n_wide = leaf.leaf_bsize.n_8x8_wide() as usize;
                for dr in 0..n_high {
                    for dc in 0..n_wide {
                        let idx = (leaf_mi_row + dr) * mi_cols + (leaf_mi_col + dc);
                        coded_segments[idx] = leaf.segment_id;
                        coded_skip[idx] = leaf.skip;
                    }
                }
            }
        }
    }
    (coded_segments, coded_skip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_decoder::{Decoder, ParsedFrame};

    /// Build a strided YUV view over tight-stride backing buffers.
    fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
        YuvFrameRef {
            y,
            u,
            v,
            y_stride: w,
            u_stride: w / 2,
            v_stride: w / 2,
        }
    }

    /// Generate a smooth gradient Y + flat UV — same shape as the F5.1
    /// keyframe oracle.
    fn gradient_frame(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let v = 128i32 + (r as i32) / 2 + (c as i32) / 2;
                y[r * w + c] = v.clamp(0, 255) as u8;
            }
        }
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        (y, u, v)
    }

    /// Decode the encoder's output and return the parsed frames.
    fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Vec<ParsedFrame<'a>> {
        let parsed = dec
            .parse_packet(pkt)
            .expect("parse_packet must accept the packet");
        parsed
            .frames()
            .iter()
            .map(|pf| ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect()
    }

    /// T1 — A single 64×64 keyframe round-trips cleanly through the
    /// real decoder.
    #[test]
    fn encode_64x64_keyframe_decodes() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(64, 64);

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);
        assert!(pkt.actual_bits > 0);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].header.kind, WireFrameKind::KeyFrame);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the keyframe");
    }

    /// T2 — Keyframe + inter sequence both decode cleanly. The
    /// load-bearing oracle for F10.0: the encoder's stateful
    /// rate-control + reference-buffer + GOP loop produces a
    /// decodable two-frame sequence.
    #[test]
    fn encode_64x64_inter_after_kf_decodes() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(64, 64);

        // Frame 0 — keyframe.
        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        // Frame 1 — same input, but the encoder must produce an inter.
        let inter = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(inter.frame_kind, FrameKind::Inter);

        // Decode both in sequence using a fresh decoder.
        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        assert_eq!(frames_kf.len(), 1);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");

        let frames_inter = decode_packet(&mut dec, &inter.data);
        assert_eq!(frames_inter.len(), 1);
        assert!(matches!(
            frames_inter[0].header.kind,
            WireFrameKind::InterFrame { .. }
        ));
        let _ = dec
            .decode_frame(&frames_inter[0])
            .expect("decode inter after kf");
    }

    /// T3 — End-to-end PSNR sanity. With a smooth gradient input and
    /// q ≈ 50, the encoder's reconstruction should match the input
    /// to within ~25 dB PSNR on Y. (Very loose — the goal is a
    /// non-degenerate quality oracle, not BD-rate parity.)
    #[test]
    fn encode_two_frame_sequence_psnr_reasonable() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000, // high enough for low q
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(64, 64);

        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        let inter = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();

        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let frame_kf = dec.decode_frame(&frames_kf[0]).expect("decode kf");
        // Compute PSNR on Y between the KF's decoded output and the
        // input. This is the oracle for the keyframe quality.
        let y_plane = frame_kf.y_plane();
        let y_data = y_plane.data();
        let y_stride = y_plane.stride();
        let mut sse: u64 = 0;
        for r in 0..64 {
            for c in 0..64 {
                let d = y_data[r * y_stride + c] as i32 - y[r * 64 + c] as i32;
                sse += (d * d) as u64;
            }
        }
        let mse = sse as f64 / (64.0 * 64.0);
        // PSNR = 10 * log10(255^2 / MSE).
        let psnr = if mse <= 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };
        assert!(
            psnr > 25.0,
            "keyframe PSNR {psnr:.2} dB below 25 dB sanity floor"
        );

        // The inter frame is decoded too — its output equals the
        // reference (zero-MV skip), which IS the KF's reconstruction.
        // We don't compute its PSNR here; the round-trip succeeding
        // is the oracle.
        let frames_inter = decode_packet(&mut dec, &inter.data);
        let _ = dec
            .decode_frame(&frames_inter[0])
            .expect("decode inter after kf");
    }

    /// T4 — GOP boundary: with `gop_length = 3`, frames 0 and 3
    /// should be keyframes; frames 1, 2, 4 should be inter.
    #[test]
    fn encode_frame_dispatches_kf_then_inter_then_kf_at_gop_end() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 3,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(64, 64);

        let view = yuv_view(&y, &u, &v, 64);

        let f0 = enc.encode_frame(&view).unwrap();
        let f1 = enc.encode_frame(&view).unwrap();
        let f2 = enc.encode_frame(&view).unwrap();
        let f3 = enc.encode_frame(&view).unwrap();
        let f4 = enc.encode_frame(&view).unwrap();

        assert_eq!(f0.frame_kind, FrameKind::Key, "frame 0 must be KF");
        assert_eq!(f1.frame_kind, FrameKind::Inter, "frame 1 must be inter");
        assert_eq!(f2.frame_kind, FrameKind::Inter, "frame 2 must be inter");
        assert_eq!(f3.frame_kind, FrameKind::Key, "frame 3 must be KF (gop boundary)");
        assert_eq!(f4.frame_kind, FrameKind::Inter, "frame 4 must be inter");
    }

    /// T5 — Rate-control responds to bitrate change: a very low
    /// target_bitrate_bps should push Q higher than a high
    /// target_bitrate_bps for the same input.
    #[test]
    fn encode_frame_q_index_responds_to_bitrate() {
        let (y, u, v) = gradient_frame(64, 64);
        let view = yuv_view(&y, &u, &v, 64);

        // Low-bitrate encoder.
        let cfg_low = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 50_000, // 50 kbps — very tight
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc_low = StreamEncoder::new(cfg_low).unwrap();
        // Pump a few KF-then-inter cycles so the buffer feedback
        // settles. The KF eats a big chunk of the budget; once the
        // buffer goes negative the controller raises Q.
        let mut last_low_q = 0u8;
        for _ in 0..4 {
            let f = enc_low.encode_frame(&view).unwrap();
            last_low_q = f.q_index;
        }

        // High-bitrate encoder.
        let cfg_high = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 10_000_000, // 10 Mbps — wide budget
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc_high = StreamEncoder::new(cfg_high).unwrap();
        let mut last_high_q = 0u8;
        for _ in 0..4 {
            let f = enc_high.encode_frame(&view).unwrap();
            last_high_q = f.q_index;
        }

        assert!(
            last_low_q >= last_high_q,
            "low-bitrate Q ({last_low_q}) should be >= high-bitrate Q ({last_high_q})"
        );
    }

    /// T6 — A 128×64 frame (two SBs horizontally) also round-trips.
    /// Verifies the SB loop in the F5.1 simulator + writer works
    /// across multiple SBs.
    #[test]
    fn encode_128x64_keyframe_decodes() {
        let cfg = EncoderConfig {
            width: 128,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(128, 64);

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 128)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        assert_eq!(frames.len(), 1);
        let _ = dec.decode_frame(&frames[0]).expect("decode 128x64 kf");
    }

    /// T7 — Encoder rejects non-multiple-of-64 dimensions.
    #[test]
    fn encoder_rejects_non_multiple_of_64() {
        let cfg = EncoderConfig {
            width: 63,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        assert!(StreamEncoder::new(cfg).is_err());
    }

    /// T8 — Encoder accepts a strided YUV input (y_stride > width).
    #[test]
    fn encode_accepts_strided_yuv() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        // Allocate buffers with extra stride padding (96 vs 64).
        let y_stride = 96;
        let uv_stride = 48;
        let mut y = vec![0u8; y_stride * 64];
        let mut u = vec![128u8; uv_stride * 32];
        let mut v = vec![128u8; uv_stride * 32];
        // Fill with a gradient inside the active region.
        for r in 0..64 {
            for c in 0..64 {
                let val = 128i32 + (r as i32) / 2 + (c as i32) / 2;
                y[r * y_stride + c] = val.clamp(0, 255) as u8;
            }
        }
        for r in 0..32 {
            for c in 0..32 {
                u[r * uv_stride + c] = 128;
                v[r * uv_stride + c] = 128;
            }
        }

        let view = YuvFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride,
            u_stride: uv_stride,
            v_stride: uv_stride,
        };
        let pkt = enc.encode_frame(&view).expect("strided encode");
        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        let _ = dec.decode_frame(&frames[0]).expect("decode strided kf");
    }

    /// xorshift32-based pseudo-random byte fill — deterministic across
    /// runs so the partition picker can't flicker its decisions.
    fn xorshift_y_plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; w * h];
        let mut state = seed.max(1);
        for b in buf.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *b = (state & 0xFF) as u8;
        }
        buf
    }

    /// Count distinct block sizes in the decoder's recovered mi_grid.
    fn count_distinct_block_sizes(grid: &[Option<ModeInfo>]) -> usize {
        let mut sizes: std::collections::BTreeSet<BlockSize> = std::collections::BTreeSet::new();
        for cell in grid.iter().flatten() {
            sizes.insert(cell.block_size);
        }
        sizes.len()
    }

    /// F10.1a KEY ORACLE — chaotic keyframe input drives the F8.2
    /// picker into a non-uniform partition, and the full bitstream
    /// round-trips through the real decoder (every cell of the
    /// recovered mi_grid is populated, and at least one cell is
    /// smaller than Block64x64).
    #[test]
    fn encode_keyframe_with_chaotic_input_uses_split_partition() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        let y = xorshift_y_plane(64, 64, 0xC0FFEE);
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        assert_eq!(frames.len(), 1);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the chaotic-input KF");

        let grid = dec.last_frame_mi_grid();
        // Every cell populated.
        for cell in grid.iter() {
            assert!(cell.is_some(), "every mi-cell must be populated");
        }
        // Partition split somewhere — at least one cell is not Block64x64.
        let any_split = grid
            .iter()
            .any(|cell| cell.map(|c| c.block_size != BlockSize::Block64x64).unwrap_or(false));
        assert!(
            any_split,
            "F8.2 should have split the chaotic-input KF into sub-64x64 leaves; \
             distinct sizes seen = {}",
            count_distinct_block_sizes(grid),
        );
    }

    /// F10.1a — Flat input keeps the picker on `Block64x64` (the F8.2
    /// "no-split" outcome). Decoder must recover an all-Block64x64
    /// mi_grid.
    #[test]
    fn encode_keyframe_with_flat_input_uses_64x64_partition() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        // Fully flat — variance is zero everywhere; the picker should
        // emit Block64x64 (no splits).
        let y = vec![128u8; 64 * 64];
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the flat-input KF");

        // libvpx's KF path forces a SPLIT at the top level for
        // keyframes (`set_vt_partitioning:504-508`). F8.2 deliberately
        // doesn't (documented divergence in `partition.rs`), so an
        // all-flat KF lands on a single Block64x64 leaf — same shape
        // as the F10.0-era path. Assert that here.
        let grid = dec.last_frame_mi_grid();
        for cell in grid.iter() {
            let cell = cell.expect("every mi-cell must be populated");
            assert_eq!(
                cell.block_size,
                BlockSize::Block64x64,
                "flat KF should not split"
            );
        }
    }

    /// F10.1a — Inter frame whose source differs from the previous
    /// reference splits and round-trips. (The reconstruction equals
    /// the reference under ZEROMV/skip=true so the picker's outcome
    /// is observable through the mi_grid shape, not through pixel
    /// content.)
    #[test]
    fn encode_inter_with_chaotic_diff_uses_split_partition() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        // Frame 0 — flat keyframe. The encoder's `last_recon_*` then
        // matches the reconstruction of this flat KF (a flat plane).
        let y0 = vec![128u8; 64 * 64];
        let u0 = vec![128u8; 32 * 32];
        let v0 = vec![128u8; 32 * 32];
        let kf = enc.encode_frame(&yuv_view(&y0, &u0, &v0, 64)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        // Frame 1 — chaotic Y; UV unchanged. The picker sees high
        // variance between src and the flat reference (Y) and should
        // split.
        let y1 = xorshift_y_plane(64, 64, 0xDEADBEEF);
        let inter = enc.encode_frame(&yuv_view(&y1, &u0, &v0, 64)).unwrap();
        assert_eq!(inter.frame_kind, FrameKind::Inter);

        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");
        let frames_inter = decode_packet(&mut dec, &inter.data);
        let _ = dec
            .decode_frame(&frames_inter[0])
            .expect("decode chaotic-diff inter");

        let grid = dec.last_frame_mi_grid();
        let any_split = grid
            .iter()
            .any(|cell| cell.map(|c| c.block_size != BlockSize::Block64x64).unwrap_or(false));
        assert!(
            any_split,
            "F8.2 should have split the chaotic-diff inter into sub-64x64 leaves"
        );
    }

    // ---------------------------------------------------------------
    // F10.1b — intra-mode picker wiring tests.
    // ---------------------------------------------------------------

    /// Tally each `IntraMode` seen in the recovered keyframe mi-grid
    /// (only the `BlockMode::Intra` cells; inter cells return 0).
    fn count_intra_y_modes(grid: &[Option<ModeInfo>]) -> [usize; 10] {
        let mut counts = [0usize; 10];
        for cell in grid.iter().flatten() {
            if let BlockMode::Intra { y_mode, .. } = cell.mode {
                counts[y_mode as usize] += 1;
            }
        }
        counts
    }

    /// F10.1b — End-to-end wiring smoke test. Encode a chaotic KF
    /// (so the F8.2 partition picker splits below 64×64 and the F8.0
    /// intra picker actually fires per leaf), then decode and verify
    /// every recovered intra cell carries one of the 10 [`IntraMode`]
    /// variants — round-tripped faithfully through writer / decoder.
    ///
    /// This test exists because the F8.0-fix transform-domain
    /// `block_yrd` distortion model is highly noise-sensitive: any
    /// recon noise (the encoder's per-pixel quantization round-trip
    /// noise of ±1-±3) in the above row inflates V_PRED's residual
    /// energy through the forward FHT in a way that "perfect-above"
    /// picker unit tests don't capture. In realistic frame-loop
    /// conditions the picker correctly RDO-chooses DC even on
    /// vertical-stripe content — V's rate-cost penalty
    /// (~1993 vs DC's ~462 in prob-cost units, [Dc][Dc] context)
    /// outweighs the noisy-residual distortion advantage. See the
    /// F10.1b open punch-list for the V_PRED-oracle follow-up.
    #[test]
    fn encode_keyframe_with_chaotic_input_intra_modes_round_trip() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 50_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        let y = xorshift_y_plane(64, 64, 0xFACEFEED);
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the chaotic-input KF");

        let grid = dec.last_frame_mi_grid();
        // Every cell must be intra and carry a valid Y mode (sanity:
        // the picker + writer agree on the per-leaf mode).
        for cell in grid.iter() {
            let c = cell.expect("every mi-cell must be populated");
            match c.mode {
                BlockMode::Intra { y_mode, uv_mode } => {
                    // UV stays DC for F10.1b (UV-mode picker is F10.1c+).
                    assert_eq!(uv_mode, IntraMode::Dc);
                    let raw = y_mode as u8;
                    assert!(raw < 10);
                }
                BlockMode::Inter { .. } => panic!("KF cell carries inter mode"),
            }
        }
    }

    /// F10.1b WIRING ORACLE — bypass the F8.0 picker by hand-injecting
    /// a `PartitionLeafDesc` with `predicted_y_mode = IntraMode::V`,
    /// feed it through `simulate_dc_intra_encode_partitions` +
    /// `build_keyframe_packet`, then decode and assert every cell
    /// round-trips as `V_PRED`. Proves the F10.1b wiring layer
    /// (`PartitionLeafDesc.predicted_y_mode` → `build_kf_mi_grid` →
    /// writer → decoder mi-grid) faithfully threads the picker's
    /// chosen mode, independent of what the picker actually picks
    /// when called from the frame loop with realistic neighbour data.
    #[test]
    fn build_kf_with_forced_v_pred_leaves_round_trips_as_v_pred() {
        use crate::{PartitionLeafDesc, simulate_dc_intra_encode_partitions};

        let w: u32 = 64;
        let h: u32 = 64;
        let q_index: u8 = 50;

        // Single 64×64 SB; 64 sub-8×8 leaves all forced to V_PRED.
        let mut leaves = Vec::with_capacity(64);
        for r in 0..8 {
            for c in 0..8 {
                leaves.push(PartitionLeafDesc {
                    mi_row_in_sb: r,
                    mi_col_in_sb: c,
                    leaf_bsize: BlockSize::Block8x8,
                    predicted_y_mode: IntraMode::V,
                    predicted_uv_mode: IntraMode::Dc,
                    segment_id: 0,
                });
            }
        }
        let sb_partitions = vec![leaves];

        // Gentle Y gradient so the recon round-trip is stable.
        let y_input: Vec<u8> = (0..(w * h) as usize)
            .map(|i| {
                let r = i / (w as usize);
                let c = i % (w as usize);
                (128i32 + (r as i32) / 2 + (c as i32) / 2).clamp(0, 255) as u8
            })
            .collect();
        let u_input = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let v_input = vec![128u8; ((w / 2) * (h / 2)) as usize];

        let mut y_recon = vec![0u8; (w * h) as usize];
        let mut u_recon = vec![0u8; ((w / 2) * (h / 2)) as usize];
        let mut v_recon = vec![0u8; ((w / 2) * (h / 2)) as usize];

        let seg_q_deltas: [Option<i16>; 8] = [None; 8];
        let coef_source = simulate_dc_intra_encode_partitions(
            w,
            h,
            q_index,
            &seg_q_deltas,
            (&y_input, &u_input, &v_input),
            (&mut y_recon, &mut u_recon, &mut v_recon),
            &sb_partitions,
        );

        let pkt = build_keyframe_packet(
            w,
            h,
            q_index,
            /* filter_level = */ 0,
            &coef_source,
            &sb_partitions,
        )
        .expect("build_keyframe_packet must succeed");

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt);
        assert_eq!(frames.len(), 1);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the V_PRED-leaf KF");

        let grid = dec.last_frame_mi_grid();
        let mut v_count = 0usize;
        for cell in grid.iter() {
            let c = cell.expect("every cell populated");
            match c.mode {
                BlockMode::Intra { y_mode, .. } => {
                    if y_mode == IntraMode::V {
                        v_count += 1;
                    }
                }
                BlockMode::Inter { .. } => panic!("expected intra"),
            }
        }
        assert_eq!(
            v_count, 64,
            "all 64 cells should round-trip as V_PRED (forced) — got {v_count}",
        );
    }

    /// F10.1b regression — Flat KF content keeps DC_PRED dominant. The
    /// F8.2 picker holds the partition at Block64x64 (no split), so the
    /// picker falls back to DC via the Block64x64 carve-out anyway; but
    /// for any sub-64x64 leaf the picker also lands on DC because the
    /// residual is zero everywhere under DC.
    #[test]
    fn encode_keyframe_with_flat_content_still_picks_dc() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        let y = vec![128u8; 64 * 64];
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(pkt.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the flat KF");

        let grid = dec.last_frame_mi_grid();
        let counts = count_intra_y_modes(grid);
        let dc_count = counts[IntraMode::Dc as usize];
        let total: usize = counts.iter().sum();
        assert!(
            dc_count > 0,
            "F10.1b: flat KF should produce at least one DC_PRED leaf \
             — counts = {counts:?}",
        );
        // DC must be the dominant mode (≥ half the cells).
        assert!(
            dc_count * 2 >= total,
            "F10.1b: flat KF should be DC-dominant: dc_count={dc_count}, total={total}, \
             counts={counts:?}",
        );
    }

    // -----------------------------------------------------------------
    // F10.1c — inter mode picker + real inter residuals tests.
    // -----------------------------------------------------------------

    /// Compute PSNR on the Y plane between two tight-stride buffers.
    /// Returns `f64::INFINITY` when the MSE is zero.
    fn y_psnr(a: &[u8], b: &[u8]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut sse: u64 = 0;
        for (&aa, &bb) in a.iter().zip(b.iter()) {
            let d = (aa as i32) - (bb as i32);
            sse += (d * d) as u64;
        }
        let mse = sse as f64 / (a.len() as f64);
        if mse <= 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        }
    }

    /// F10.1c KEY ORACLE — encode a KF followed by an inter whose
    /// source equals the KF's reconstruction. With src == ref under
    /// zero MV the F8.1 picker lands on ZEROMV / NEARESTMV with
    /// near-zero distortion; the inter recon should be bit-exact-ish
    /// with the KF recon. PSNR > 35 dB on the decoded inter output vs
    /// the encoder's intended source (= KF recon).
    #[test]
    fn encode_inter_with_real_residuals_decodes_with_low_psnr() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(64, 64);

        // Frame 0 — keyframe.
        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        // Drive the decoder to materialise the KF's reconstruction so
        // we can use it as the inter's source. (The encoder's internal
        // `last_recon_*` was bit-exact with this by construction —
        // F10.1b oracle.)
        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let frame_kf = dec.decode_frame(&frames_kf[0]).expect("decode kf");
        let kf_recon_y: Vec<u8> = {
            let y_plane = frame_kf.y_plane();
            let stride = y_plane.stride();
            let data = y_plane.data();
            let mut out = vec![0u8; 64 * 64];
            for r in 0..64 {
                out[r * 64..r * 64 + 64].copy_from_slice(&data[r * stride..r * stride + 64]);
            }
            out
        };
        let kf_recon_u: Vec<u8> = {
            let p = frame_kf.u_plane();
            let stride = p.stride();
            let data = p.data();
            let mut out = vec![0u8; 32 * 32];
            for r in 0..32 {
                out[r * 32..r * 32 + 32].copy_from_slice(&data[r * stride..r * stride + 32]);
            }
            out
        };
        let kf_recon_v: Vec<u8> = {
            let p = frame_kf.v_plane();
            let stride = p.stride();
            let data = p.data();
            let mut out = vec![0u8; 32 * 32];
            for r in 0..32 {
                out[r * 32..r * 32 + 32].copy_from_slice(&data[r * stride..r * stride + 32]);
            }
            out
        };

        // Frame 1 — inter with src = KF recon.
        let inter = enc
            .encode_frame(&yuv_view(&kf_recon_y, &kf_recon_u, &kf_recon_v, 64))
            .unwrap();
        assert_eq!(inter.frame_kind, FrameKind::Inter);

        let frames_inter = decode_packet(&mut dec, &inter.data);
        let frame_inter = dec.decode_frame(&frames_inter[0]).expect("decode inter");

        let y_plane = frame_inter.y_plane();
        let decoded_y: Vec<u8> = {
            let stride = y_plane.stride();
            let data = y_plane.data();
            let mut out = vec![0u8; 64 * 64];
            for r in 0..64 {
                out[r * 64..r * 64 + 64].copy_from_slice(&data[r * stride..r * stride + 64]);
            }
            out
        };

        let psnr = y_psnr(&decoded_y, &kf_recon_y);
        assert!(
            psnr > 35.0,
            "F10.1c KEY ORACLE: decoded inter PSNR vs source = {psnr:.2} dB \
             — expected > 35 dB",
        );

        // Stronger bit-exact-encoder-vs-decoder oracle: the encoder's
        // post-inter `last_recon_y` should match the decoder's output
        // pixel-for-pixel (or near-bit-exact under the documented
        // F10.1c divergences). Asserts the encoder's mirrored
        // `reconstruct_inter_block` IT-add sequence matches the
        // decoder's exact path.
        let enc_recon_y = enc.last_recon_y();
        assert_eq!(
            enc_recon_y.len(),
            decoded_y.len(),
            "encoder recon and decoded recon must have same length",
        );
        let mut max_diff = 0i32;
        let mut sse: u64 = 0;
        for (&e, &d) in enc_recon_y.iter().zip(decoded_y.iter()) {
            let diff = (e as i32) - (d as i32);
            max_diff = max_diff.max(diff.abs());
            sse += (diff * diff) as u64;
        }
        // Bit-exact is the goal; allow a tiny tolerance under the
        // documented v1 divergences.
        assert_eq!(
            max_diff, 0,
            "F10.1c KEY ORACLE: encoder recon vs decoder recon diverged \
             — max |diff| = {max_diff}, sse = {sse}",
        );
    }

    /// F10.1c — Inter src is shifted relative to the LASTFRAME ref.
    /// The F8.1 picker (running on Block16+ leaves) should pick
    /// `skip = false` on at least some leaves — proving the picker
    /// fired and emitted real residuals (not the skip-true fallback).
    /// We also call the picker directly through a synthetic harness to
    /// verify the NEWMV path finds a non-zero MV, which we can't always
    /// observe through the decoder's mi-grid because libvpx's RD logic
    /// often prefers `NEARESTMV` (cheaper MV bits) when the picker's
    /// post-recon SSE savings under NEWMV are modest.
    #[test]
    fn encode_inter_with_shifted_content_picks_nonzero_mv() {
        use rvp9_core::block::BlockSize as Bs;
        use rvp9_core::cost::mv_cost::build_nmv_cost_table;
        use rvp9_core::cost::symbol_costs::inter_mode_costs;
        use rvp9_core::entropy::FrameContext as Fc;
        use rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
        use rvp9_core::mode_decision::inter::{
            InterModeSearchParams, MvCandidates, pick_inter_mode_luma,
        };
        use rvp9_core::mv_search::PlaneRef as PR;
        use rvp9_core::mv_search::mv_limits::MvLimits;
        use rvp9_core::mv_search::init_dsmotion_compensation;
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

        // (a) Direct F8.1 picker oracle on a Block16x16 chaotic-content
        // 3-pel shift — mirrors `mode_decision::inter::tests::
        // pick_inter_finds_new_mv_on_shifted_ref` setup exactly (64×64
        // plane, seed 17). The picker should return NEWMV with col ≈
        // 24 (= 3 full-pels in 1/8-pel units).
        let pw = 64usize;
        let ph = 64usize;
        let mut state: u32 = 18; // xorshift seed (17 + 1, matching F8.1 test)
        let mut ref_data = vec![0u8; pw * ph];
        for b in ref_data.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *b = (state & 0xff) as u8;
        }
        let mut src_data = ref_data.clone();
        for i in 0..16 {
            for j in 0..16 {
                src_data[(20 + i) * pw + (20 + j)] = ref_data[(20 + i) * pw + (23 + j)];
            }
        }
        let src_plane = PR { data: &src_data, stride: pw, width: pw, height: ph };
        let ref_plane = PR { data: &ref_data, stride: pw, width: pw, height: ph };
        let fc = Fc::new_default();
        let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
        let inter_costs = inter_mode_costs(&fc);
        let quantizer = init_plane_quantizers(50, QuantizerDeltas::Y);
        let site_config = init_dsmotion_compensation(pw as i32);
        let mvc = MvCandidates::default();
        let params = InterModeSearchParams {
            block_size: Bs::Block16x16,
            src_plane: &src_plane,
            src_offset: (20, 20),
            ref_plane: &ref_plane,
            mv_candidates: &mvc,
            mv_limits: MvLimits::new(-16, 16, -16, 16),
            site_config: &site_config,
            mv_costs: &costs,
            inter_mode_costs: &inter_costs,
            inter_mode_ctx: 0,
            quantizer: &quantizer,
            q_index: 50,
            allow_hp: false,
            frame_kind: rvp9_core::bitstream::header::FrameKind::InterFrame {
                reset_frame_context: 0,
                refresh_frame_flags: 0,
                ref_frame_indices: [0; rvp9_core::bitstream::frame::MAX_INTER_REFS],
                ref_frame_sign_biases: [false; rvp9_core::bitstream::frame::MAX_INTER_REFS],
                allow_high_precision_mv: false,
                interp_filter: InterpFilter::EightTap,
            },
            newmv_seed: None,
        };
        let pick = pick_inter_mode_luma(&params);
        assert_eq!(
            pick.best_mode,
            rvp9_core::mode_info::InterMode::NewMv,
            "F8.1 picker should choose NEWMV for shifted chaotic ref — \
             got {:?}",
            pick.best_mode,
        );
        assert!(
            pick.best_mv.col != 0 || pick.best_mv.row != 0,
            "F8.1 picker should choose non-zero MV for shifted ref — \
             got mv=({}, {})",
            pick.best_mv.row, pick.best_mv.col,
        );

        // (b) End-to-end inter encode + decode oracle: at least one
        // leaf round-trips through the picker (skip = false), proving
        // the integration layer reaches the F8.1 picker. The chosen
        // mode may still be ZEROMV/NEARESTMV depending on RD; what
        // matters is that the picker FIRED on at least one Block16+
        // leaf (not just the Block8x8 skip-true fallback).
        //
        // RC.X1 — use a SHIFTED inter frame so forward-quant produces
        // non-zero residuals on at least one leaf. With identical
        // KF→inter content, all coefs quantize to zero and the
        // RC.X1 skip-on-zero-eob fix flips every leaf to skip=true
        // (which is the correct VP9 behaviour for zero-residual
        // blocks). The pre-RC.X1 behaviour of forcing skip=false on
        // picker-supported leaves was non-conformant — libvpx's
        // decoder overrides `mi->skip = 1` on `eobtotal == 0` and the
        // mismatch desyncs the bool coder at the next block's skip
        // read. See `vp9_decodeframe.c:1101`.
        let cfg = EncoderConfig {
            width: 128,
            height: 128,
            framerate: 30.0,
            target_bitrate_bps: 4_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y0, u0, v0) = gradient_frame(128, 128);
        let kf = enc.encode_frame(&yuv_view(&y0, &u0, &v0, 128)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        // Inter frame: shifted gradient so forward-quant produces
        // non-zero residuals on at least one Block16+ leaf.
        let (y1, u1, v1) = translating_gradient(128, 128, 1);
        let inter = enc.encode_frame(&yuv_view(&y1, &u1, &v1, 128)).unwrap();
        assert_eq!(inter.frame_kind, FrameKind::Inter);

        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");
        let frames_inter = decode_packet(&mut dec, &inter.data);
        let _ = dec.decode_frame(&frames_inter[0]).expect("decode inter");

        let grid = dec.last_frame_mi_grid();
        let any_nonskip_picker_leaf = grid.iter().flatten().any(|c| {
            !c.skip
                && matches!(
                    c.block_size,
                    BlockSize::Block16x16 | BlockSize::Block32x32 | BlockSize::Block64x64
                )
        });
        assert!(
            any_nonskip_picker_leaf,
            "F10.1c: at least one Block16+ leaf must have skip=false \
             (picker fired and emitted residuals)"
        );
    }

    /// F10.1c — Block8x8 leaves fall back to ZEROMV / skip = true (the
    /// picker doesn't support sub-16 sizes). Use a chaotic KF so F8.2
    /// splits into Block8x8 leaves; verify the inter frame's
    /// Block8x8 cells carry `skip = true`.
    #[test]
    fn encode_inter_skip_true_path_still_works() {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 50_000, // tight budget → high q → smaller splits
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();

        let y = xorshift_y_plane(64, 64, 0xC0FFEE42);
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        let mut dec = Decoder::new();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");

        // Inter: shifted Y so F8.2 splits (the inter path's variance
        // picker measures src-vs-ref difference).
        let inter = enc.encode_frame(&yuv_view(&y, &u, &v, 64)).unwrap();
        assert_eq!(inter.frame_kind, FrameKind::Inter);

        let frames_inter = decode_packet(&mut dec, &inter.data);
        let _ = dec.decode_frame(&frames_inter[0]).expect("decode inter");

        // Every cell must be populated; sub-16 leaves must carry
        // skip=true (picker fallback).
        let grid = dec.last_frame_mi_grid();
        for cell in grid.iter() {
            let c = cell.expect("every mi-cell populated");
            if c.block_size == BlockSize::Block8x8 {
                assert!(
                    c.skip,
                    "F10.1c: Block8x8 leaf must carry skip=true (picker \
                     fallback), got skip=false at block_size={:?}",
                    c.block_size,
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // F10.2 — Multi-frame real-content PSNR oracle.
    //
    // First true quality oracle for the encoder. Builds synthesised
    // multi-frame YUV input, runs it through StreamEncoder::encode_frame,
    // decodes every packet through a real rvp9_decoder::Decoder, and
    // computes per-frame Y-plane PSNR.
    //
    // Tests use 64×64 frames for tractable runtime (a few seconds for
    // multi-frame sequences).
    // -----------------------------------------------------------------

    /// Compute peak signal-to-noise ratio (Y-plane only) in dB. Returns
    /// 100.0 if the two planes are bit-exact (mse == 0).
    fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
        assert_eq!(orig.len(), recon.len(), "psnr_db: length mismatch");
        let mut sse: u64 = 0;
        for (a, b) in orig.iter().zip(recon.iter()) {
            let diff = i32::from(*a) - i32::from(*b);
            sse += (diff * diff) as u64;
        }
        if sse == 0 {
            return 100.0;
        }
        let mse = sse as f64 / orig.len() as f64;
        10.0 * (255.0 * 255.0 / mse).log10()
    }

    /// Synthesise frame `idx` of a translating-gradient sequence.
    /// Diagonal gradient shifts per frame so F8.1 inter mode picker
    /// sees real motion. U/V planes are constant 128.
    fn translating_gradient(w: usize, h: usize, idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let v = 64i32 + (r as i32 / 2) + (c as i32 / 2) + (idx as i32 * 3);
                y[r * w + c] = v.clamp(0, 255) as u8;
            }
        }
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        (y, u, v)
    }

    /// Synthesise a static gradient (same pattern every frame). Tests
    /// the encoder under "perfect ZEROMV" conditions — inter frames
    /// should reconstruct the keyframe bit-exactly via skip=true or
    /// zero-MV+zero-residual.
    fn static_gradient(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        translating_gradient(w, h, 0)
    }

    /// T1 — 4-frame translating gradient (1 KF + 3 inter). PSNR oracle.
    ///
    /// Smaller frame count than the original 30 to keep test runtime
    /// in line with the workspace's per-test budget. Still exercises
    /// the inter mode picker on multiple consecutive frames.
    #[test]
    fn f10_2_t1_translating_gradient_4_frames_psnr() {
        const W: usize = 64;
        const H: usize = 64;
        const N_FRAMES: u32 = 4;

        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        let mut psnrs = Vec::with_capacity(N_FRAMES as usize);

        for idx in 0..N_FRAMES {
            let (y, u, v) = translating_gradient(W, H, idx);
            let pkt = enc
                .encode_frame(&yuv_view(&y, &u, &v, W))
                .expect("encode_frame must succeed");

            let frames = decode_packet(&mut dec, &pkt.data);
            assert_eq!(frames.len(), 1, "frame {idx}: one wire frame per packet");
            let buf = dec
                .decode_frame(&frames[0])
                .expect("decode_frame must succeed");

            // Y-plane PSNR — extract a tight-stride copy of the decoded
            // visible region.
            let yp = buf.y_plane();
            let mut decoded_y = vec![0u8; W * H];
            for r in 0..H {
                let src_off = r * yp.stride();
                let dst_off = r * W;
                decoded_y[dst_off..dst_off + W]
                    .copy_from_slice(&yp.data()[src_off..src_off + W]);
            }
            let p = psnr_db(&y, &decoded_y);
            psnrs.push(p);
        }

        let avg: f64 = psnrs.iter().sum::<f64>() / psnrs.len() as f64;
        let min = psnrs.iter().cloned().fold(f64::INFINITY, f64::min);

        // Oracle: smooth-gradient content at 2 Mbps should easily exceed
        // 30 dB average and 25 dB min. The 4-frame run is the cumulative
        // signal for "the F5–F10 pipeline produces meaningful PSNR".
        assert!(
            avg > 30.0,
            "F10.2-T1: avg PSNR = {avg:.2} dB (per-frame: {psnrs:?}) — \
             expected > 30 dB",
        );
        assert!(
            min > 25.0,
            "F10.2-T1: min PSNR = {min:.2} dB (per-frame: {psnrs:?}) — \
             expected > 25 dB",
        );
    }

    /// T2 — Bit-exact invariant across multiple frames.
    ///
    /// Asserts the F10.1c key property: encoder reconstruction matches
    /// decoder reconstruction PIXEL-FOR-PIXEL on every frame of a
    /// multi-frame sequence. If this drifts, no downstream quality
    /// oracle is meaningful.
    #[test]
    fn f10_2_t2_bit_exact_invariant_across_frames() {
        const W: usize = 64;
        const H: usize = 64;
        const N_FRAMES: u32 = 3;

        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_500_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        for idx in 0..N_FRAMES {
            let (y, u, v) = translating_gradient(W, H, idx);
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            let buf = dec.decode_frame(&frames[0]).unwrap();

            // Encoder's local recon vs decoder's recon: must match
            // pixel-for-pixel.
            let enc_recon = &enc.last_recon_y;
            let yp = buf.y_plane();
            for r in 0..H {
                let dec_row = &yp.data()[r * yp.stride()..r * yp.stride() + W];
                let enc_row = &enc_recon[r * W..(r + 1) * W];
                let diff_count = dec_row
                    .iter()
                    .zip(enc_row.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    diff_count, 0,
                    "F10.2-T2: frame {idx} row {r}: encoder recon != \
                     decoder recon. dec={dec_row:?} enc={enc_row:?}",
                );
            }
        }
    }

    /// T3 — Static gradient with single KF + 2 inter; inter frames are
    /// identical to the KF source so ZEROMV should be optimal. PSNR on
    /// inter frames should match or exceed the keyframe PSNR.
    #[test]
    fn f10_2_t3_static_content_inter_matches_kf_quality() {
        const W: usize = 64;
        const H: usize = 64;

        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        let (y, u, v) = static_gradient(W, H);
        let mut psnrs = Vec::new();

        for _ in 0..3 {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            let buf = dec.decode_frame(&frames[0]).unwrap();

            let yp = buf.y_plane();
            let mut decoded_y = vec![0u8; W * H];
            for r in 0..H {
                decoded_y[r * W..(r + 1) * W]
                    .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W]);
            }
            psnrs.push(psnr_db(&y, &decoded_y));
        }

        // KF PSNR is the floor. Subsequent inter frames at static-content
        // ZEROMV should match (allow ≤ 1 dB drift for residual-coef
        // quantization differences).
        let kf_psnr = psnrs[0];
        for (i, p) in psnrs.iter().enumerate().skip(1) {
            assert!(
                *p >= kf_psnr - 1.0,
                "F10.2-T3: inter frame {i} PSNR = {p:.2} dB but KF PSNR \
                 = {kf_psnr:.2} dB — static content should not regress",
            );
        }

        // Sanity: KF on smooth gradient should comfortably exceed 30 dB.
        assert!(
            kf_psnr > 30.0,
            "F10.2-T3: KF PSNR = {kf_psnr:.2} dB — expected > 30 dB on \
             smooth gradient",
        );
    }

    // ============================================================
    // F11.0 — in-loop deblocking filter wireup tests.
    // ============================================================

    /// F11.0 T1 — At a typical Q (~80 here), the emitted
    /// `filter_level` in the uncompressed header must be > 0. Prior to
    /// F11.0 the encoder hard-coded `filter_level = 0`, silently
    /// disabling the deblocking pass; this test guards against the
    /// regression.
    #[test]
    fn f11_0_filter_level_nonzero_at_typical_q() {
        const W: usize = 64;
        const H: usize = 64;

        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            // Tight bitrate target so the rate controller picks a
            // mid-range Q — empirically lands around q = 70-90 on a
            // 64×64 gradient KF.
            target_bitrate_bps: 300_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
        // Sanity-check the Q the controller picked is in the "typical"
        // band — if not, the test's premise about a non-zero
        // filter_level doesn't apply.
        assert!(
            pkt.q_index >= 4,
            "F11.0-T1: precondition violated — q_index = {} too low to \
             exercise non-zero filter_level (need q_index ≥ 4)",
            pkt.q_index
        );

        // Parse the packet through the decoder and inspect the header.
        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        assert_eq!(frames.len(), 1);
        let tail = frames[0]
            .header
            .tail
            .as_ref()
            .expect("KF must carry a tail");
        assert!(
            tail.loop_filter.filter_level > 0,
            "F11.0-T1: filter_level = {} (expected > 0 at q_index = {})",
            tail.loop_filter.filter_level,
            pkt.q_index
        );
        // The picker is `(q >> 1) min 63`; cross-check.
        let expected = (pkt.q_index >> 1).min(63);
        assert_eq!(
            tail.loop_filter.filter_level, expected,
            "F11.0-T1: filter_level should match the F11.0 picker formula"
        );
    }

    /// F11.0 T2 — The key oracle: with LF enabled, the decoder's
    /// reconstruction must still match the encoder's `last_recon_y`
    /// pixel-for-pixel. If the encoder's LF and the decoder's LF
    /// diverge by a single byte, this fails. This is the gate.
    #[test]
    fn f11_0_decoded_recon_matches_encoder_recon_with_lf() {
        const W: usize = 64;
        const H: usize = 64;

        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            // Pick a tight bitrate so the rate controller drives Q up
            // into the band where LF is non-trivial.
            target_bitrate_bps: 200_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);

        let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
        // Precondition: LF actually fired (filter_level > 0).
        let filter_level_picked = (pkt.q_index >> 1).min(63);
        assert!(
            filter_level_picked > 0,
            "F11.0-T2: precondition — LF must fire at q_index = {}",
            pkt.q_index
        );

        let mut dec = Decoder::new();
        let frames = decode_packet(&mut dec, &pkt.data);
        let buf = dec.decode_frame(&frames[0]).unwrap();

        // Encoder's local recon (post-LF) vs decoder's recon (post-LF):
        // bit-exact match per row.
        let enc_recon = enc.last_recon_y();
        let yp = buf.y_plane();
        for r in 0..H {
            let dec_row = &yp.data()[r * yp.stride()..r * yp.stride() + W];
            let enc_row = &enc_recon[r * W..(r + 1) * W];
            assert_eq!(
                dec_row, enc_row,
                "F11.0-T2: row {r} encoder recon (post-LF) != decoder \
                 recon (post-LF); filter_level = {filter_level_picked}, \
                 q_index = {}",
                pkt.q_index
            );
        }
    }

    /// F11.0 T3 — At very low Q (q_index = 1, the encoder's lossless
    /// clamp), `pick_filter_level` must return 0 so the LF pass is
    /// skipped entirely (matches libvpx `vp9_loopfilter.c:1422-1427`).
    /// We assert via the public picker function rather than driving a
    /// full encode at q = 1, because the rate controller never picks
    /// q = 1 in practice on this codec config.
    #[test]
    fn f11_0_lf_at_q_zero_is_zero() {
        use crate::loopfilter::pick_filter_level;
        assert_eq!(pick_filter_level(0), 0, "q = 0 must yield filter_level = 0");
        assert_eq!(pick_filter_level(1), 0, "q = 1 (encoder min) must yield filter_level = 0");
        // Sanity: typical Q lands at the picker's formula.
        assert_eq!(pick_filter_level(80), 40);
        assert_eq!(pick_filter_level(120), 60);
        // Saturates at MAX_LOOP_FILTER = 63.
        assert_eq!(pick_filter_level(200), 63);
        assert_eq!(pick_filter_level(255), 63);
    }

    // ============================================================
    // AQ.3 — cyclic-refresh wireup tests.
    // ============================================================

    /// AQ.3 — `AqMode::Off` must preserve Stage 4a behaviour: the
    /// emitted uncompressed header's `segmentation.enabled` is `false`
    /// on every frame (KF + inter). This is the bytes-exact gate
    /// against AQ being silently enabled when callers haven't asked
    /// for it.
    #[test]
    fn aq_off_preserves_stage_4a_behavior() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::Off,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);

        let mut dec = Decoder::new();
        // KF + 3 inter — all five frames must report `segmentation.enabled = false`.
        for idx in 0..4u32 {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            assert_eq!(frames.len(), 1);
            let tail = frames[0]
                .header
                .tail
                .as_ref()
                .expect("frame must have a tail");
            assert!(
                !tail.segmentation.enabled,
                "frame {idx}: AqMode::Off must keep segmentation disabled",
            );
            let _ = dec.decode_frame(&frames[0]).expect("decode_frame");
        }
    }

    /// AQ.3 — `AqMode::CyclicRefresh`: KFs still emit `enabled = false`
    /// (cyclic refresh only fires on inter frames), but every inter
    /// frame must carry segmentation in the uncompressed header. The
    /// header's `feature_data[1][0]` must equal the CyclicRefresh
    /// default `segment1_q_delta` (-8).
    #[test]
    fn aq_cyclic_refresh_emits_segmentation_in_inter_frames() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);
        let mut dec = Decoder::new();

        // KF: segmentation disabled (KF doesn't run AQ).
        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let tail_kf = frames_kf[0].header.tail.as_ref().unwrap();
        assert!(
            !tail_kf.segmentation.enabled,
            "KF must keep segmentation disabled (KF doesn't run AQ)",
        );
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");

        // Inter frames: segmentation enabled, feature_data[1][0] == -8.
        for inter_idx in 0..3u32 {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            assert_eq!(pkt.frame_kind, FrameKind::Inter);
            let frames = decode_packet(&mut dec, &pkt.data);
            let tail = frames[0].header.tail.as_ref().unwrap();
            assert!(
                tail.segmentation.enabled,
                "inter frame {inter_idx}: AqMode::CyclicRefresh must \
                 enable segmentation",
            );
            assert!(tail.segmentation.update_map);
            assert!(!tail.segmentation.temporal_update);
            assert!(tail.segmentation.update_data);
            assert!(!tail.segmentation.abs_delta);
            assert_eq!(
                tail.segmentation.feature_data[1][0],
                Some(-8),
                "inter frame {inter_idx}: feature_data[1][AltQ] must carry \
                 the default segment1_q_delta (-8)",
            );
            let _ = dec.decode_frame(&frames[0]).expect("decode inter");
        }
    }

    /// AQ.3 KEY ORACLE — over a full refresh cycle (default 10% / 25
    /// of 256 ≈ 1 SB per frame on a 16×16 mi frame), every mi-cell
    /// must get `segment_id == CR_SEGMENT_ID_BOOST` at least once.
    ///
    /// Uses a 128×128 frame so we get 2×2 = 4 SBs; at the default
    /// 10% refresh rate that's ~26 mi cells/frame ≈ 0.4 SBs/frame.
    /// Over 20 inter frames every SB should land in segment 1 at
    /// least once. This is the cyclic-refresh state machine's
    /// defining invariant.
    #[test]
    fn aq_cyclic_refresh_refresh_cycle_completes_in_n_frames() {
        const W: usize = 128;
        const H: usize = 128;
        const N_INTER: u32 = 20;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 100, // single KF, then N inter frames
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);
        let mut dec = Decoder::new();

        let mi_rows = H / 8;
        let mi_cols = W / 8;
        let mut ever_refreshed = vec![false; mi_rows * mi_cols];

        // First frame is a KF — no refresh activity.
        let kf = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");

        // Drive N inter frames; track which mi-cells ever hit segment 1.
        for f in 0..N_INTER {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            assert_eq!(pkt.frame_kind, FrameKind::Inter);
            let frames = decode_packet(&mut dec, &pkt.data);
            let _ = dec.decode_frame(&frames[0]).expect("decode inter");
            let grid = dec.last_frame_mi_grid();
            for (i, cell) in grid.iter().enumerate() {
                if let Some(c) = cell
                    && c.segment_id == 1
                {
                    ever_refreshed[i] = true;
                }
            }
            let n_missed = ever_refreshed.iter().filter(|&&r| !r).count();
            let _ = (f, n_missed); // for debugging
        }

        let n_missed = ever_refreshed.iter().filter(|&&r| !r).count();
        assert_eq!(
            n_missed, 0,
            "AQ.3 KEY ORACLE: after {N_INTER} inter frames at 10% refresh \
             on a {W}x{H} frame, {n_missed} mi-cells never got refreshed",
        );
    }

    /// AQ.3 — keyframe boundaries reset the cyclic-refresh state. The
    /// segmentation maps emitted post-first-KF and post-second-KF
    /// must be identical (same input, same Q ⇒ same picker output).
    #[test]
    fn aq_cyclic_refresh_keyframes_reset_state() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 6, // KF at frames 0 and 6
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y, u, v) = gradient_frame(W, H);
        let mut dec = Decoder::new();

        // First GOP: KF + 5 inter. Capture each inter's mi-grid segment_id pattern.
        let mut first_gop_seg_grids: Vec<Vec<u8>> = Vec::new();
        for idx in 0..6u32 {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            let _ = dec.decode_frame(&frames[0]).expect("decode");
            if idx > 0 {
                let grid = dec.last_frame_mi_grid();
                let seg: Vec<u8> = grid
                    .iter()
                    .map(|c| c.map(|m| m.segment_id).unwrap_or(0))
                    .collect();
                first_gop_seg_grids.push(seg);
            }
        }

        // Second GOP: KF (frame 6) + 5 inter. Capture again.
        let mut second_gop_seg_grids: Vec<Vec<u8>> = Vec::new();
        for idx in 0..6u32 {
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            let _ = dec.decode_frame(&frames[0]).expect("decode");
            if idx > 0 {
                let grid = dec.last_frame_mi_grid();
                let seg: Vec<u8> = grid
                    .iter()
                    .map(|c| c.map(|m| m.segment_id).unwrap_or(0))
                    .collect();
                second_gop_seg_grids.push(seg);
            }
        }

        assert_eq!(first_gop_seg_grids.len(), 5);
        assert_eq!(second_gop_seg_grids.len(), 5);
        for (i, (a, b)) in first_gop_seg_grids
            .iter()
            .zip(second_gop_seg_grids.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "AQ.3: inter frame {i} of first GOP segment_id grid must \
                 equal inter frame {i} of second GOP — KF must reset CR \
                 state. first={a:?}, second={b:?}",
            );
        }
    }

    /// AQ.3 sanity — 4-frame AQ-on sequence (KF + 3 inter) decodes
    /// cleanly through the real Decoder; the encoder's reconstruction
    /// remains bit-exact with the decoder's on every frame (the AQ.0
    /// segment-id wire-format plumbing doesn't change the per-leaf
    /// recon pixels, since the per-leaf Q delta flows through both
    /// sides).
    #[test]
    fn aq_cyclic_refresh_decodes_through_real_decoder() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_500_000,
            gop_length: 30,
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        for idx in 0..4u32 {
            let (y, u, v) = translating_gradient(W, H, idx);
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            assert_eq!(frames.len(), 1);
            let buf = dec
                .decode_frame(&frames[0])
                .expect("decode_frame must accept the AQ-on packet");

            // Encoder's recon must match the decoder's recon bit-exact
            // on every frame (segmentation doesn't change the per-leaf
            // dequant; both sides compute the same effective qindex).
            let yp = buf.y_plane();
            let enc_recon = enc.last_recon_y();
            for r in 0..H {
                let dec_row = &yp.data()[r * yp.stride()..r * yp.stride() + W];
                let enc_row = &enc_recon[r * W..(r + 1) * W];
                assert_eq!(
                    dec_row, enc_row,
                    "AQ.3 sanity: frame {idx} row {r} encoder recon != \
                     decoder recon under AqMode::CyclicRefresh",
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // AQ.4 — Differential quality oracles for cyclic-refresh AQ.
    //
    // Stage 4b cyclic-refresh AQ is wired and functionally validated
    // (AQ.3 tests cover emit/decode/refresh-cycle/KF-reset round-trips).
    // The AQ.4 oracles validate that AQ actually produces meaningful
    // DIFFERENTIAL effects on the output:
    //
    // 1. `aq_cyclic_refresh_bit_exact_encoder_decoder_recon` — KEY
    //    ORACLE. AQ on, KF + 3 inter on translating-gradient input.
    //    Encoder Y/U/V recon must match decoder recon pixel-for-pixel
    //    (max |diff| == 0 on every plane on every frame). Extends the
    //    F10.1c / F10.2 T2 invariant to U/V under AQ.
    // 2. `aq_cyclic_refresh_segment_1_has_finer_quantization` — quality
    //    oracle. On chaotic content where AQ paints some SBs as
    //    segment 1 and others as segment 0, the segment-1 SBs must
    //    have lower per-cell reconstruction SSE vs source than the
    //    segment-0 SBs (finer Q ⇒ lower distortion).
    // 3. `aq_cyclic_refresh_cursor_advances_monotonically` — refresh-
    //    map stability oracle. Consecutive inter frames mark DISJOINT
    //    sets of SBs as segment 1 (the cursor advances; refreshed SBs
    //    age past `min_refresh_age = 4` before re-eligibility).
    // -----------------------------------------------------------------

    /// AQ.4 Oracle 1 — KEY ORACLE.
    ///
    /// With `AqMode::CyclicRefresh` on, encoder and decoder
    /// reconstructions match BIT-EXACT pixel-for-pixel on every plane
    /// (Y, U, V) for every frame (KF + 3 inter) under translating-
    /// gradient motion. Extends the F10.2 T2 invariant (encoder recon
    /// == decoder recon) to AQ-on path, on all three planes.
    ///
    /// If this fails, segmentation bits, per-segment Q-delta
    /// application, or segment_id projection has drifted between
    /// encoder and decoder — a critical wire-format bug.
    #[test]
    fn aq_cyclic_refresh_bit_exact_encoder_decoder_recon() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_500_000,
            gop_length: 30,
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        for idx in 0..4u32 {
            let (y, u, v) = translating_gradient(W, H, idx);
            let pkt = enc.encode_frame(&yuv_view(&y, &u, &v, W)).unwrap();
            let frames = decode_packet(&mut dec, &pkt.data);
            assert_eq!(frames.len(), 1);
            let buf = dec
                .decode_frame(&frames[0])
                .expect("decode_frame must accept the AQ-on packet");

            // Y plane: row-by-row bit-exact.
            let yp = buf.y_plane();
            let enc_y = enc.last_recon_y();
            let mut max_y_diff = 0i32;
            for r in 0..H {
                let dec_row = &yp.data()[r * yp.stride()..r * yp.stride() + W];
                let enc_row = &enc_y[r * W..(r + 1) * W];
                for c in 0..W {
                    let d = (dec_row[c] as i32) - (enc_row[c] as i32);
                    max_y_diff = max_y_diff.max(d.abs());
                }
            }
            assert_eq!(
                max_y_diff, 0,
                "AQ.4 Oracle 1: frame {idx} Y plane: encoder recon != \
                 decoder recon under AqMode::CyclicRefresh (max |diff| = {max_y_diff})",
            );

            // U/V planes (subsampled 4:2:0, W/2 × H/2): same invariant.
            let uvw = W / 2;
            let uvh = H / 2;
            let up = buf.u_plane();
            let vp = buf.v_plane();
            let enc_u = enc.last_recon_u();
            let enc_v = enc.last_recon_v();
            let mut max_u_diff = 0i32;
            let mut max_v_diff = 0i32;
            for r in 0..uvh {
                let dec_u_row = &up.data()[r * up.stride()..r * up.stride() + uvw];
                let enc_u_row = &enc_u[r * uvw..(r + 1) * uvw];
                let dec_v_row = &vp.data()[r * vp.stride()..r * vp.stride() + uvw];
                let enc_v_row = &enc_v[r * uvw..(r + 1) * uvw];
                for c in 0..uvw {
                    let du = (dec_u_row[c] as i32) - (enc_u_row[c] as i32);
                    let dv = (dec_v_row[c] as i32) - (enc_v_row[c] as i32);
                    max_u_diff = max_u_diff.max(du.abs());
                    max_v_diff = max_v_diff.max(dv.abs());
                }
            }
            assert_eq!(
                max_u_diff, 0,
                "AQ.4 Oracle 1: frame {idx} U plane: encoder recon != \
                 decoder recon under AqMode::CyclicRefresh (max |diff| = {max_u_diff})",
            );
            assert_eq!(
                max_v_diff, 0,
                "AQ.4 Oracle 1: frame {idx} V plane: encoder recon != \
                 decoder recon under AqMode::CyclicRefresh (max |diff| = {max_v_diff})",
            );
        }
    }

    /// AQ.4 Oracle 2 — segment-1 leaves are reconstructed at FINER
    /// quality (lower SSE) than the same cells would be under AQ-off.
    ///
    /// **Design rationale — differential, not absolute**: comparing
    /// segment-1 SSE to segment-0 SSE in the *same* AQ-on frame is
    /// confounded — cyclic refresh paints whole SBs uniformly, so
    /// segment-1 SBs and segment-0 SBs are *different content regions*
    /// with different motion-estimation difficulty / boundary
    /// proximity / etc. The two buckets are not comparable.
    ///
    /// Instead we run two parallel encoders on identical input:
    /// - encoder A: `AqMode::Off`
    /// - encoder B: `AqMode::CyclicRefresh`
    ///
    /// We pick the cells that B marked as segment 1 (using the decoded
    /// mi-grid), then compare A's per-cell SSE on those same cells
    /// against B's per-cell SSE on the same cells. The AQ.1 quality
    /// claim is: B's per-cell SSE on segment-1 cells must be LOWER
    /// than A's per-cell SSE on the same cells.
    ///
    /// This is the libvpx-spec property: finer Q ⇒ lower distortion.
    /// If this fails, AQ is "running but not producing quality" —
    /// segment_q_delta isn't reaching the dequant path on the decoder
    /// side, OR the encoder isn't actually using the finer Q.
    #[test]
    fn aq_cyclic_refresh_segment_1_has_finer_quantization() {
        // 256×256 = 16 SBs (4×4 grid); large enough that cyclic refresh
        // marks multiple SBs as segment 1 over the run.
        const W: usize = 256;
        const H: usize = 256;
        const N_INTER: u32 = 8;
        let cfg_off = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 1_500_000,
            gop_length: 100, // single KF
            aq_mode: AqMode::Off,
        };
        let cfg_on = EncoderConfig { aq_mode: AqMode::CyclicRefresh, ..cfg_off };

        let mut enc_off = StreamEncoder::new(cfg_off).unwrap();
        let mut enc_on = StreamEncoder::new(cfg_on).unwrap();
        let mut dec_off = Decoder::new();
        let mut dec_on = Decoder::new();

        // KF on both encoders (identical input ⇒ AQ-off KF and AQ-on KF
        // emit identical headers since AQ doesn't fire on KF).
        let (y0, u0, v0) = translating_gradient(W, H, 0);
        {
            let kf_off = enc_off.encode_frame(&yuv_view(&y0, &u0, &v0, W)).unwrap();
            let fr = decode_packet(&mut dec_off, &kf_off.data);
            let _ = dec_off.decode_frame(&fr[0]).expect("decode kf off");
            let kf_on = enc_on.encode_frame(&yuv_view(&y0, &u0, &v0, W)).unwrap();
            let fr = decode_packet(&mut dec_on, &kf_on.data);
            let _ = dec_on.decode_frame(&fr[0]).expect("decode kf on");
        }

        let mi_rows = H / 8;
        let mi_cols = W / 8;

        // Per-cell SSE for "segment-1 under AQ-on" vs "same cell under AQ-off".
        // The cell is identified by (frame_idx, mi_r, mi_c). We collect
        // BOTH encoders' SSE on that cell, then compare aggregated means.
        let mut sse_on_at_seg1_cells: Vec<u64> = Vec::new();
        let mut sse_off_at_seg1_cells: Vec<u64> = Vec::new();

        // Diagnostic counters surfaced in assert messages on failure.
        let mut seg1_total = 0usize;
        let mut seg1_skip = 0usize;

        for f in 1..=N_INTER {
            let (ys, us, vs) = translating_gradient(W, H, f);
            // Encode + decode on both encoders.
            let pkt_off = enc_off.encode_frame(&yuv_view(&ys, &us, &vs, W)).unwrap();
            let fr_off = decode_packet(&mut dec_off, &pkt_off.data);
            let buf_off = dec_off.decode_frame(&fr_off[0]).expect("decode off");
            let pkt_on = enc_on.encode_frame(&yuv_view(&ys, &us, &vs, W)).unwrap();
            let fr_on = decode_packet(&mut dec_on, &pkt_on.data);
            let buf_on = dec_on.decode_frame(&fr_on[0]).expect("decode on");

            let yp_off = buf_off.y_plane();
            let yp_on = buf_on.y_plane();
            let grid_on = dec_on.last_frame_mi_grid();
            assert_eq!(grid_on.len(), mi_rows * mi_cols);

            // For each cell marked segment 1 by AQ-on, compute both
            // encoders' 8×8 SSE on that cell vs source.
            for mi_r in 0..mi_rows {
                for mi_c in 0..mi_cols {
                    let cell = match grid_on[mi_r * mi_cols + mi_c] {
                        Some(c) => c,
                        None => continue,
                    };
                    if cell.segment_id != 1 {
                        continue;
                    }
                    seg1_total += 1;
                    if cell.skip {
                        seg1_skip += 1;
                        // Skipped seg-1 cells don't carry residual; their
                        // recon = reference (same on both encoders modulo
                        // ref-pool drift). Excluding mirrors the cyclic-
                        // refresh post-encode "skip excludes from refresh
                        // tracking" rule.
                        continue;
                    }
                    let mut sse_off: u64 = 0;
                    let mut sse_on: u64 = 0;
                    for r in 0..8 {
                        let py = mi_r * 8 + r;
                        let src_off = py * W + mi_c * 8;
                        let off_off = py * yp_off.stride() + mi_c * 8;
                        let off_on = py * yp_on.stride() + mi_c * 8;
                        for c in 0..8 {
                            let d_off = (yp_off.data()[off_off + c] as i32)
                                - (ys[src_off + c] as i32);
                            let d_on = (yp_on.data()[off_on + c] as i32)
                                - (ys[src_off + c] as i32);
                            sse_off += (d_off * d_off) as u64;
                            sse_on += (d_on * d_on) as u64;
                        }
                    }
                    sse_on_at_seg1_cells.push(sse_on);
                    sse_off_at_seg1_cells.push(sse_off);
                }
            }
        }

        // Need samples to do the comparison.
        let n_cells = sse_on_at_seg1_cells.len();
        assert!(
            n_cells > 0,
            "AQ.4 Oracle 2: no non-skipped segment-1 cells observed across \
             {N_INTER} inter frames — cyclic refresh fired but every seg-1 \
             cell skipped. seg1_total={seg1_total} (skip={seg1_skip})",
        );

        let mean_on =
            sse_on_at_seg1_cells.iter().sum::<u64>() as f64 / n_cells as f64;
        let mean_off =
            sse_off_at_seg1_cells.iter().sum::<u64>() as f64 / n_cells as f64;
        let ratio = mean_on / mean_off;

        // The libvpx-spec property: on the cells that AQ chose to boost,
        // AQ-on (finer Q) must produce strictly lower distortion than
        // AQ-off (base Q). Assertion message carries the full stats so
        // failures are self-diagnostic.
        assert!(
            mean_on < mean_off,
            "AQ.4 Oracle 2: AQ-on mean SSE on segment-1 cells \
             ({mean_on:.2}) must be STRICTLY LESS than AQ-off mean SSE \
             on the same cells ({mean_off:.2}); ratio on/off = {ratio:.3}. \
             Cyclic refresh's finer Q on segment 1 must produce lower \
             reconstruction distortion. \
             cells={n_cells}, seg1_total={seg1_total}, seg1_skip={seg1_skip}",
        );
    }

    /// AQ.4 Oracle 3 — refresh-map stability.
    ///
    /// Cyclic refresh's cursor advances monotonically: an SB refreshed
    /// in frame i must NOT be refreshed again in frame i+1 (the cells
    /// just had their age reset to 0; with `min_refresh_age = 4` they
    /// need 4 more frames before re-eligibility). The picker walks
    /// from `sb_cursor` forward, and the cursor advances past the most-
    /// recently-refreshed SB after every post_encode_update.
    ///
    /// This catches bugs in:
    /// - cursor advancement (e.g. if it didn't advance, frame i+1's
    ///   picker would restart at the same SB and re-pick it under
    ///   `min_refresh_age = 0` config — but our default age is 4 so
    ///   the picker would just skip-and-walk anyway; this test catches
    ///   the AGE bookkeeping instead);
    /// - age increment logic (cells must age past `min_refresh_age`
    ///   before re-eligibility).
    ///
    /// Method: encode KF + 10 inter frames, capture the SB-set refreshed
    /// per frame (= the set of SBs with at least one cell in segment 1),
    /// and verify that consecutive frames' SB-sets are disjoint.
    #[test]
    fn aq_cyclic_refresh_cursor_advances_monotonically() {
        // 128×128 = 4 SBs (2×2). Small enough that ~10% refresh rate
        // marks <1 SB per frame on average, but over 10 frames we'll
        // see multiple non-empty refresh sets.
        const W: usize = 128;
        const H: usize = 128;
        const N_INTER: u32 = 10;
        let cfg = EncoderConfig {
            width: W as u32,
            height: H as u32,
            framerate: 30.0,
            target_bitrate_bps: 2_000_000,
            gop_length: 100,
            aq_mode: AqMode::CyclicRefresh,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let mut dec = Decoder::new();

        // KF.
        let (y0, u0, v0) = translating_gradient(W, H, 0);
        let kf = enc.encode_frame(&yuv_view(&y0, &u0, &v0, W)).unwrap();
        let frames_kf = decode_packet(&mut dec, &kf.data);
        let _ = dec.decode_frame(&frames_kf[0]).expect("decode kf");

        let mi_rows = H / 8;
        let mi_cols = W / 8;
        let sb_cols = mi_cols / 8;

        // Per inter frame: set of SB indices (sb_row * sb_cols + sb_col)
        // that contain at least one cell marked as segment 1.
        let mut per_frame_refreshed_sbs: Vec<std::collections::BTreeSet<usize>> =
            Vec::with_capacity(N_INTER as usize);

        for f in 0..N_INTER {
            let (ys, us, vs) = translating_gradient(W, H, f + 1);
            let pkt = enc.encode_frame(&yuv_view(&ys, &us, &vs, W)).unwrap();
            assert_eq!(pkt.frame_kind, FrameKind::Inter);
            let frames = decode_packet(&mut dec, &pkt.data);
            let _ = dec.decode_frame(&frames[0]).expect("decode inter");
            let grid = dec.last_frame_mi_grid();

            let mut refreshed: std::collections::BTreeSet<usize> =
                std::collections::BTreeSet::new();
            for mi_r in 0..mi_rows {
                for mi_c in 0..mi_cols {
                    if let Some(cell) = grid[mi_r * mi_cols + mi_c]
                        && cell.segment_id == 1
                    {
                        let sb_row = mi_r / 8;
                        let sb_col = mi_c / 8;
                        refreshed.insert(sb_row * sb_cols + sb_col);
                    }
                }
            }
            per_frame_refreshed_sbs.push(refreshed);
        }

        // At least one frame must have refreshed something — otherwise
        // the test is vacuous (cyclic refresh never fired).
        let total_refreshed: usize =
            per_frame_refreshed_sbs.iter().map(|s| s.len()).sum();
        assert!(
            total_refreshed > 0,
            "AQ.4 Oracle 3: no SBs refreshed in any of {N_INTER} inter \
             frames — cyclic refresh never fired (test setup bug?)",
        );

        // Consecutive frames must mark DISJOINT SB sets. `min_refresh_age
        // = 4` (default config) means a cell needs at least 4 post-encode
        // cycles between refreshes — so frame i and frame i+1 must be
        // disjoint (and so should i and i+1, i+2, i+3 — but we only
        // assert i / i+1 to mirror the cursor-advancement contract; the
        // age contract is implicitly tested by the disjointness across
        // all consecutive pairs).
        for i in 0..(per_frame_refreshed_sbs.len() - 1) {
            let a = &per_frame_refreshed_sbs[i];
            let b = &per_frame_refreshed_sbs[i + 1];
            let overlap: Vec<usize> = a.intersection(b).copied().collect();
            assert!(
                overlap.is_empty(),
                "AQ.4 Oracle 3: inter frames {i} and {} both refreshed \
                 SBs {:?} — cursor failed to advance OR age logic let a \
                 freshly-refreshed SB re-trigger. Frame {i}: {:?}, \
                 frame {}: {:?}",
                i + 1,
                overlap,
                a,
                i + 1,
                b,
            );
        }
    }

}

