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
//!   `refresh_frame_context = false`, inter frames use
//!   `refresh_frame_flags = 0x00`.
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
    AllZeroCoefSource, BitDepth, BitstreamError, ChromaFormat, ColorRange, ColorSpace,
    CompressedHeaderOptions, CompressedHeaderResult, FrameKind as WireFrameKind, InterpFilter,
    ModesTileBounds, Profile, RawBitWriter, UncompressedHeader, write_compressed_header,
    write_tile_intra_keyframe, write_tile_inter, write_uncompressed_header,
};
use rvp9_core::block::BlockSize;
use rvp9_core::coef_ctx::TileCoefCtx;
use rvp9_core::entropy::{BoolWriter, FrameContext};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::{TxMode, TxSize};

use crate::rate_control::{FrameKind as RcFrameKind, RateControl, RateControlConfig};
use crate::{EncodeError, simulate_dc_intra_encode};

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
}

impl EncoderConfig {
    /// Convenience constructor with defaults that mirror the
    /// rate-control crate's WebRTC profile (1 Mbps / 30 fps /
    /// gop_length = 60).
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 60,
        }
    }

    /// Build the underlying [`RateControlConfig`] for the rate
    /// controller. Mirrors the encoder-config fields onto the
    /// rate-control fields and pulls libvpx-style defaults for the
    /// buffer geometry (6 s buffer, 4 s starting, 5 s optimal).
    fn to_rate_control_config(self) -> RateControlConfig {
        let defaults = RateControlConfig::default();
        RateControlConfig {
            target_bitrate_bps: self.target_bitrate_bps,
            framerate: self.framerate,
            gop_length: self.gop_length,
            frame_width: self.width,
            frame_height: self.height,
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
}

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

        Ok(Self {
            config,
            rate_control,
            frame_idx: 0,
            last_recon_y: vec![0u8; w * h],
            last_recon_u: vec![0u8; uvw * uvh],
            last_recon_v: vec![0u8; uvw * uvh],
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
            FrameKind::Inter => self.encode_inter_internal(q_index)?,
        };

        let actual_bits = (packet.len() as u32).saturating_mul(8);

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

        // Run the F5.1 simulator. Output goes directly into our
        // reference buffers — those ARE the reconstruction.
        let coef_source = simulate_dc_intra_encode(
            w,
            h,
            q_index,
            input_yuv,
            (
                &mut self.last_recon_y,
                &mut self.last_recon_u,
                &mut self.last_recon_v,
            ),
        );

        build_keyframe_packet(w, h, q_index, &coef_source)
    }

    /// Inter internal: emit a single-ref LASTFRAME ZEROMV `skip = true`
    /// frame over the full SB grid. The reconstruction equals the
    /// reference (zero-MV no-residual), so `last_recon_*` are unchanged.
    fn encode_inter_internal(&mut self, q_index: u8) -> Result<Vec<u8>, EncodeError> {
        build_inter_zero_mv_packet(self.config.width, self.config.height, q_index)
    }

    /// Read-only access to the rate controller (for tests / probes).
    #[must_use]
    pub fn rate_control(&self) -> &RateControl {
        &self.rate_control
    }
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
    coef_source: &dyn rvp9_core::bitstream::IntraCoefSource,
) -> Result<Vec<u8>, EncodeError> {
    let mi_rows = height / 8;
    let mi_cols = width / 8;
    let cell = ModeInfo {
        block_size: BlockSize::Block64x64,
        tx_size: TxSize::Tx32x32,
        skip: false,
        segment_id: 0,
        seg_id_predicted: false,
        mode: BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
        },
        bmi: Default::default(),
    };
    let mi_grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

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
            filter_level: 0,
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

/// Build an inter packet: every 64×64 SB is `InterMode::ZeroMv`
/// `LASTFRAME` `skip = true` `BlockSize::Block64x64`. Reconstruction
/// is a zero-MV copy of LASTFRAME — the encoder's reference buffer
/// remains identical across the boundary, which is the contract the
/// caller (the F10.0 frame loop) relies on.
fn build_inter_zero_mv_packet(
    width: u32,
    height: u32,
    base_qindex: u8,
) -> Result<Vec<u8>, EncodeError> {
    let mi_rows = height / 8;
    let mi_cols = width / 8;
    let inter_cell = ModeInfo {
        block_size: BlockSize::Block64x64,
        tx_size: TxSize::Tx32x32,
        skip: true,
        segment_id: 0,
        seg_id_predicted: false,
        mode: BlockMode::Inter {
            mode: InterMode::ZeroMv,
            ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
            mvs: [Mv::ZERO; 2],
            interp_filter: InterpFilter::EightTap,
        },
        bmi: Default::default(),
    };
    let mi_grid: Vec<Option<ModeInfo>> =
        vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

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

    let mut tile_buf = vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 8 + 1024];
    let tile_len = {
        let mut bw = BoolWriter::start_encode(&mut tile_buf);
        let tile_mi_cols = (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
        let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
        let coef_source = AllZeroCoefSource::new();
        write_tile_inter(
            &mut bw,
            &post_fc,
            &params,
            TxMode::AllowTx32x32,
            &mi_grid,
            mi_rows,
            mi_cols,
            tile_bounds,
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            &coef_source,
        )?;
        bw.stop_encode()
            .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
    };
    tile_buf.truncate(tile_len);

    // refresh_frame_flags = 0: inter frame doesn't update any ref slot.
    // The keyframe's pool-slot-0 LASTFRAME persists across the boundary.
    let tail = HeaderTail {
        refresh_frame_context: false,
        frame_parallel_decoding_mode: false,
        frame_context_idx: 0,
        loop_filter: LoopFilterParams {
            filter_level: 0,
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
        kind: WireFrameKind::InterFrame {
            reset_frame_context: 0,
            refresh_frame_flags: 0x00,
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
}
