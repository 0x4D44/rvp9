//! rvp9-encoder: VP9 real-time video encoder.
//!
//! Mirrors libvpx's tile-threaded real-time encode path (cpu-used 6..=8) for
//! WebRTC and livestream use cases. See `PLAN.md` for the staged buildout:
//!
//! - Stage 4a: 1-pass CBR MVP, single tile, 720p30 real-time.
//! - Stage 4b: Adaptive quantization (3 modes).
//! - Stage 4c: Content-aware denoiser.
//! - Stage 4d: Temporal SVC (2- and 3-layer).
//! - Stage 4e: Spatial SVC (2- and 3-layer).
//!
//! Design constraints from `PLAN.md` §3:
//!
//! - D6: Threading mirrors libvpx's pthread tile-worker model. Expect
//!   substantial `unsafe` in shared encoder state, gated by `UnsafeCell`
//!   and documented safety invariants.
//! - D7: Integer widths mirror libvpx exactly. Wrap-reliant sites use
//!   explicit `wrapping_*`; everywhere else, plain operators (debug panic
//!   on overflow is a feature, not a bug).
//! - D8: BD-rate parity is the release gate; byte parity is a debugging
//!   tool. The `notes/divergence-ledger.md` records every deliberate
//!   departure from libvpx.
//!
//! ## Current status (2026-05-19, Stage 4a in progress)
//!
//! The encoder is at SKELETON state — the wire-format layer is
//! functionally complete (keyframe + inter, every legal block size
//! down to BLOCK_4x4 sub-8 partitions, every partition shape,
//! superframe container assembly). What's missing is everything
//! that makes the encoder USEFUL: real residuals (coef integration
//! into inter leaves), mode decision, partition selection, motion
//! estimation, rate control, adaptive quantization, denoising.
//! Those land in the modules listed in `PLAN.md` Stage 4a tasks 2-8.
//!
//! Today the encoder can emit:
//!
//! - A synthetic all-DC-skip keyframe at any 64-multiple resolution
//!   ([`Encoder::encode_synthetic_dc_skip_keyframe`]).
//! - A 2-frame keyframe + inter sequence packed into one superframe
//!   packet, both decodable by `rvp9-decoder`
//!   ([`Encoder::encode_synthetic_two_frame_kf_inter`]). The inter
//!   frame uses single-ref LASTFRAME ZEROMV `skip=true` over
//!   PARTITION_NONE 64×64 super-blocks.
//!
//! Sub-8 partitioning, mixed inter modes, NEWMV, and compound-ref
//! inter frames are exercisable via the `rvp9-core` writers directly
//! (`write_tile_inter`, `write_inter_block_mode_info`, etc.); the
//! encoder's public API will widen to cover those shapes as the
//! mode-decision driver lands.

use rvp9_core::bitstream::frame::{MAX_INTER_REFS, ReferenceMode};
use rvp9_core::bitstream::header::HeaderTail;
use rvp9_core::bitstream::sub_parsers::{
    LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
    SegmentationParams, TileInfo,
};
use rvp9_core::bitstream::{
    AllZeroCoefSource, AllZeroIntraCoefSource, BitDepth, BitstreamError, ChromaFormat, ColorRange,
    ColorSpace, CompressedHeaderOptions, CompressedHeaderResult, FrameKind, InterpFilter,
    ModesTileBounds, ModesWriterError, Profile, RawBitWriter, SuperframeBuilder,
    UncompressedHeader, write_compressed_header, write_tile_intra_keyframe, write_tile_inter,
    write_uncompressed_header,
};
use rvp9_core::block::BlockSize;
use rvp9_core::coef_ctx::TileCoefCtx;
use rvp9_core::entropy::{BoolWriter, FrameContext};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::{TxMode, TxSize};

pub mod aq_cyclic_refresh;
pub mod frame_loop;
pub(crate) mod loopfilter;
pub mod rate_control;

pub use frame_loop::{AqMode, EncodedFrame, EncoderConfig, FrameKind as StreamFrameKind, StreamEncoder, YuvFrameRef};

/// Encoder configuration.
///
/// v1 is intentionally tiny — Stage 4a tasks 2-8 will widen this with
/// rate-control / partition / mode-decision / AQ / SVC knobs. For now the
/// only knobs that affect the (synthetic) output are `width`, `height`,
/// and `base_qindex`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    /// Coded luma frame width in pixels. Must be a multiple of 64
    /// (super-block boundary) until Stage 4a partition support lands.
    pub width: u32,
    /// Coded luma frame height in pixels. Must be a multiple of 64.
    pub height: u32,
    /// Base quantization index `0..=255`. `0` triggers lossless mode
    /// on the decoder side; the synthetic encoder doesn't support
    /// lossless yet, so callers should pick a small non-zero value
    /// (e.g. 50) for "near-lossless" or 100+ for normal lossy.
    pub base_qindex: u8,
}

impl Config {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(width: u32, height: u32, base_qindex: u8) -> Self {
        Self {
            width,
            height,
            base_qindex,
        }
    }
}

/// Encoder error type. Wraps the underlying wire-format writers'
/// errors; future variants will cover rate-control, mode-decision,
/// motion-estimation failures.
#[derive(Debug)]
pub enum EncodeError {
    /// Configuration was rejected before any bytes were emitted.
    InvalidConfig(&'static str),
    /// One of the bitstream-level writers (header / compressed / tile
    /// buffer / superframe) returned an error.
    Bitstream(BitstreamError),
    /// The per-tile mode-info writer rejected the input (typically
    /// because the synthesised mi_grid violated v1 scope).
    Modes(ModesWriterError),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid encoder config: {reason}"),
            Self::Bitstream(e) => write!(f, "bitstream writer error: {e}"),
            Self::Modes(e) => write!(f, "mode-info writer error: {e}"),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<BitstreamError> for EncodeError {
    fn from(e: BitstreamError) -> Self {
        Self::Bitstream(e)
    }
}

impl From<ModesWriterError> for EncodeError {
    fn from(e: ModesWriterError) -> Self {
        Self::Modes(e)
    }
}

/// VP9 encoder.
///
/// At v1 the encoder holds nothing but its configuration. As
/// rate-control / frame-context tracking land, this grows fields for
/// (a) the 4 `FrameContext` banks, (b) the rate-control state, and
/// (c) the reference-frame management. For now every call to an encode
/// method is self-contained.
#[derive(Clone, Copy, Debug)]
pub struct Encoder {
    config: Config,
}

impl Encoder {
    /// Build a new encoder from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::InvalidConfig`] if `width` or `height`
    /// is not a multiple of 64 — the synthetic encoder only emits
    /// PARTITION_NONE 64×64 super-blocks, so non-aligned dimensions
    /// can't be expressed.
    pub fn new(config: Config) -> Result<Self, EncodeError> {
        if !config.width.is_multiple_of(64) {
            return Err(EncodeError::InvalidConfig(
                "width must be a multiple of 64 (v1: PARTITION_NONE 64x64 SBs only)",
            ));
        }
        if !config.height.is_multiple_of(64) {
            return Err(EncodeError::InvalidConfig(
                "height must be a multiple of 64 (v1: PARTITION_NONE 64x64 SBs only)",
            ));
        }
        if config.width == 0 || config.height == 0 {
            return Err(EncodeError::InvalidConfig("width/height must be non-zero"));
        }
        if config.base_qindex == 0 {
            return Err(EncodeError::InvalidConfig(
                "base_qindex must be non-zero (v1 does not support lossless)",
            ));
        }
        Ok(Self { config })
    }

    /// Return the encoder configuration.
    #[must_use]
    pub fn config(&self) -> Config {
        self.config
    }

    /// Encode a single synthetic keyframe.
    ///
    /// The frame's `mi_grid` is filled with DC-PRED skip blocks at the
    /// 64×64 super-block boundary: every cell is `BlockSize::Block64x64`,
    /// `IntraMode::Dc` for both Y and UV, `skip=true`, `segment_id=0`,
    /// `tx_size=Tx32x32`. This is the SIMPLEST possible decodable VP9
    /// keyframe — the synthetic-encoder analogue of "Hello, world".
    ///
    /// Returns the **frame transport payload** (uncompressed header +
    /// compressed header + tile data), suitable for IVF chunking or RTP
    /// packetisation by the caller.
    ///
    /// As mode-decision / partition / motion-estimation land in
    /// follow-on commits, this entry point will widen to take a real
    /// YUV frame and the synthetic stub will move to a `_synthetic`
    /// suffix or behind a `#[cfg(test)]` gate.
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::Bitstream`] or [`EncodeError::Modes`]
    /// if any of the wire-format writers fail.
    pub fn encode_synthetic_dc_skip_keyframe(&self) -> Result<Vec<u8>, EncodeError> {
        self.build_keyframe_payload(/* refresh_frame_context = */ true)
    }

    /// F5.1 — Encode a single keyframe payload with **real DC_PRED
    /// residuals**.
    ///
    /// This is the first end-to-end exercise of the substrate landed in
    /// F4 (forward transforms) + F5.0 (forward quantizer) + S4.16g/m
    /// (residual emission). The encoder treats `input_yuv` as a 4:2:0
    /// YUV frame at `(self.config.width, self.config.height)`, runs the
    /// canonical per-TX-block intra-coding loop (predict → residual →
    /// fdct → forward-quant → dequant → inverse-DCT-add → record qcoeff),
    /// then emits a keyframe packet with `skip = false` on every leaf
    /// and the quantized coefficients threaded through the keyframe
    /// writer's new [`IntraCoefSource`] plumbing.
    ///
    /// Decoder reconstruction is bit-exact against the encoder's
    /// internal `recon_yuv` (the encoder simulator's per-TX inverse
    /// transform output) provided `input_yuv` and `recon_yuv` agree on
    /// the quant-noise envelope — that's the F5.1 oracle. The supplied
    /// `recon_yuv_out` buffers receive the encoder's reconstruction so
    /// callers can assert against the decoder's output.
    ///
    /// Scope: PARTITION_NONE 64×64 SBs only (mirrors v1's other
    /// synthetic encoders), `IntraMode::Dc` for both Y and UV,
    /// `tx_mode = AllowTx32x32` so Y uses Tx32x32 (per 32×32 luma
    /// quadrant) and UV uses Tx32x32 (a single 32×32 transform per
    /// chroma plane, since UV at 4:2:0 is a 32×32 block for a 64×64
    /// luma SB and `max_txsize(Block32x32) = Tx32x32`).
    /// Segmentation off, loop-filter off.
    ///
    /// # Parameters
    ///
    /// - `input_yuv`: three contiguous plane buffers `(y, u, v)`. The
    ///   Y plane is `width × height` bytes; U and V are each
    ///   `(width / 2) × (height / 2)` bytes. Tight stride (no
    ///   per-row padding).
    /// - `recon_yuv_out`: three plane buffers receiving the encoder's
    ///   per-TX reconstructed pixels — what the decoder is expected
    ///   to produce bit-exactly. Same shape as `input_yuv`.
    ///
    /// # Errors
    ///
    /// - [`EncodeError::InvalidConfig`] — `input_yuv` slice lengths
    ///   don't match the encoder's `(width, height)`.
    /// - [`EncodeError::Bitstream`] / [`EncodeError::Modes`] — wire-
    ///   format writer errors.
    pub fn encode_synthetic_dc_intra_with_residuals(
        &self,
        input_yuv: (&[u8], &[u8], &[u8]),
        recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    ) -> Result<Vec<u8>, EncodeError> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let uvw = w / 2;
        let uvh = h / 2;
        if input_yuv.0.len() != w * h
            || input_yuv.1.len() != uvw * uvh
            || input_yuv.2.len() != uvw * uvh
        {
            return Err(EncodeError::InvalidConfig(
                "input_yuv plane lengths must match (w*h, w/2*h/2, w/2*h/2)",
            ));
        }
        if recon_yuv_out.0.len() != w * h
            || recon_yuv_out.1.len() != uvw * uvh
            || recon_yuv_out.2.len() != uvw * uvh
        {
            return Err(EncodeError::InvalidConfig(
                "recon_yuv_out plane lengths must match (w*h, w/2*h/2, w/2*h/2)",
            ));
        }

        // Run the encoder simulator. Produces qcoeffs (one i32 vector
        // per (plane, mi_row, mi_col, tx_x4_local, tx_y4_local) — keyed
        // by raster position inside the SB) AND fills recon_yuv_out
        // with the per-TX reconstructed pixels.
        let coef_source = simulate_dc_intra_encode(
            self.config.width,
            self.config.height,
            self.config.base_qindex,
            input_yuv,
            recon_yuv_out,
        );

        self.build_keyframe_payload_with_residuals(
            /* refresh_frame_context = */ true,
            &coef_source,
        )
    }

    /// Encode a 2-frame sequence — keyframe + inter — packed into one
    /// superframe-container packet.
    ///
    /// The keyframe is the same synthetic DC-skip frame
    /// [`encode_synthetic_dc_skip_keyframe`] emits, but with
    /// `refresh_frame_context = false` so the decoder's frame-context
    /// bank 0 stays at [`FrameContext::new_default`] after the
    /// keyframe — and the inter frame can use that same default FC as
    /// its `pre_fc` without the encoder having to run backward
    /// adaptation internally.
    ///
    /// The inter frame is:
    /// - Single-ref `LASTFRAME` (pool slot 0 — populated by the
    ///   keyframe via `Decoder::decode_keyframe`'s implicit splat
    ///   across all 8 ref_frame_map slots).
    /// - `PARTITION_NONE` 64×64 super-blocks.
    /// - `InterMode::ZeroMv` (no MV bits emitted; reconstruction
    ///   collapses to MC from a zero-pixel reference, yielding a
    ///   zero-pixel output).
    /// - `skip = true` (no residual coefs).
    /// - `EightTap` interp filter (matches the keyframe).
    ///
    /// Both frame payloads are concatenated under a superframe index
    /// trailer (libvpx wire format). Callers feed the returned packet
    /// to `Decoder::parse_packet` once; the decoder then yields TWO
    /// `ParsedFrame`s (one keyframe, one inter).
    ///
    /// # Errors
    ///
    /// Returns the same error variants
    /// [`encode_synthetic_dc_skip_keyframe`] can produce, plus
    /// [`EncodeError::Bitstream`] if the superframe builder rejects
    /// either payload (only possible if the encoder buggily emitted
    /// an empty payload).
    ///
    /// [`encode_synthetic_dc_skip_keyframe`]: Self::encode_synthetic_dc_skip_keyframe
    pub fn encode_synthetic_two_frame_kf_inter(&self) -> Result<Vec<u8>, EncodeError> {
        let kf_payload =
            self.build_keyframe_payload(/* refresh_frame_context = */ false)?;
        let inter_payload =
            self.build_inter_after_default_keyframe_payload(/* inter_skip = */ true)?;

        let mut sf = SuperframeBuilder::new();
        sf.push_frame(kf_payload)?;
        sf.push_frame(inter_payload)?;
        Ok(sf.finish())
    }

    /// Encode a synthetic two-frame superframe **with `skip = false`**
    /// on the inter frame, exercising the encoder's residual-coef
    /// emission path end-to-end.
    ///
    /// Same shape as [`encode_synthetic_two_frame_kf_inter`] (default
    /// keyframe + single-ref LASTFRAME ZEROMV PARTITION_NONE 64×64
    /// inter), but the inter cell carries `skip = false` and the
    /// writer emits per-plane per-TX residual tokens via
    /// `write_inter_block_residual` (S4.16g phase 3a) →
    /// `write_block_tokens` (phase 2) → `encode_coefs` (S4.15).
    /// All coefs are zero (via `AllZeroCoefSource`), so the decoder's
    /// inverse transform produces zero and the pixel output is
    /// identical to the skip=true variant — but the entire coef-decode
    /// pipeline fires through the real `Decoder::decode_frame`
    /// (validated by S4.16k in the decoder crate's tests).
    ///
    /// This API surface gives external callers (tests, fuzzers,
    /// downstream tools) a way to exercise the encoder's non-skip
    /// path without driving a full mode-decision loop. Equivalent to
    /// what the rvp9-decoder's `s4_16k` test does at integration
    /// scope.
    ///
    /// # Errors
    ///
    /// Same as [`encode_synthetic_two_frame_kf_inter`].
    pub fn encode_synthetic_two_frame_kf_inter_non_skip(
        &self,
    ) -> Result<Vec<u8>, EncodeError> {
        let kf_payload =
            self.build_keyframe_payload(/* refresh_frame_context = */ false)?;
        let inter_payload =
            self.build_inter_after_default_keyframe_payload(/* inter_skip = */ false)?;

        let mut sf = SuperframeBuilder::new();
        sf.push_frame(kf_payload)?;
        sf.push_frame(inter_payload)?;
        Ok(sf.finish())
    }

    /// F5.1 — keyframe payload with `skip = false` on every leaf and
    /// real (caller-supplied) DC_PRED residual coefs.
    ///
    /// Same shape as [`build_keyframe_payload`] but the SB grid carries
    /// `skip = false` and the writer is invoked with the supplied
    /// `coef_source` instead of `AllZeroIntraCoefSource`. All other
    /// header / mode-info choices match: `tx_mode = AllowTx32x32`,
    /// `IntraMode::Dc` on Y and UV, no segmentation, no loop filter,
    /// `frame_context_idx = 0`, single tile.
    fn build_keyframe_payload_with_residuals(
        &self,
        refresh_frame_context: bool,
        coef_source: &dyn rvp9_core::bitstream::IntraCoefSource,
    ) -> Result<Vec<u8>, EncodeError> {
        let mi_rows = self.config.height / 8;
        let mi_cols = self.config.width / 8;
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
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
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
            refresh_frame_context,
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
                base_qindex: self.config.base_qindex,
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
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: self.config.width,
            height: self.config.height,
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

    /// Build a single keyframe payload (uncompressed header +
    /// compressed header + tile data). Used by both
    /// [`encode_synthetic_dc_skip_keyframe`] (which passes
    /// `refresh_frame_context = true` for backward compatibility) and
    /// [`encode_synthetic_two_frame_kf_inter`] (which passes `false`
    /// so the decoder's FC bank 0 stays at default for the follow-up
    /// inter frame's `pre_fc`).
    ///
    /// [`encode_synthetic_dc_skip_keyframe`]: Self::encode_synthetic_dc_skip_keyframe
    /// [`encode_synthetic_two_frame_kf_inter`]: Self::encode_synthetic_two_frame_kf_inter
    fn build_keyframe_payload(
        &self,
        refresh_frame_context: bool,
    ) -> Result<Vec<u8>, EncodeError> {
        // mi grid: 8 mi cells per 64-pixel SB row (8x8 mi units).
        let mi_rows = self.config.height / 8;
        let mi_cols = self.config.width / 8;
        let cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: 0,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mi_grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

        // Compressed header. tx_mode = AllowTx32x32 forces Tx32x32 on
        // every block without consuming bits. pre == post emits only
        // "no-update" gates for every probability.
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

        // Per-tile mode-info partition. Single tile covers the whole
        // frame; (mi_rows, mi_cols) is the full mi grid.
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut tile_buf = vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 8 + 1024];
        let tile_len = {
            let mut bw = BoolWriter::start_encode(&mut tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
            let coef_source = AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                /* seg = */ None,
                &mut coef_ctx,
                &coef_source,
            )?;
            bw.stop_encode()
                .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
        };
        tile_buf.truncate(tile_len);

        // Uncompressed header. Loop filter off, segmentation disabled,
        // single tile, frame_context_idx = 0. `first_partition_size`
        // is the compressed-header length we just measured.
        let tail = HeaderTail {
            refresh_frame_context,
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
                base_qindex: self.config.base_qindex,
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
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: self.config.width,
            height: self.config.height,
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

        // Assemble: uncompressed_header + compressed_header + tile_data.
        // No superframe trailer for a single-frame payload.
        let mut packet = Vec::with_capacity(hdr_len + compressed_len + tile_len);
        packet.extend_from_slice(&hdr_buf);
        packet.extend_from_slice(&compressed_buf);
        packet.extend_from_slice(&tile_buf);
        Ok(packet)
    }

    /// Build a single inter-frame payload that references the
    /// (default-FC) keyframe emitted by `build_keyframe_payload(false)`.
    /// Single-ref LASTFRAME, PARTITION_NONE 64×64, ZEROMV, EightTap.
    /// Caller's responsibility to ensure the decoder's
    /// `frame_contexts[0]` is at default before this packet is parsed
    /// — that's what the `refresh_frame_context = false` keyframe
    /// pairing guarantees.
    ///
    /// `inter_skip` controls the leaf cell's `skip` flag and gates
    /// whether the writer emits per-plane per-TX residual tokens
    /// (`AllZeroCoefSource` is no-op under `inter_skip = true`).
    fn build_inter_after_default_keyframe_payload(
        &self,
        inter_skip: bool,
    ) -> Result<Vec<u8>, EncodeError> {
        let mi_rows = self.config.height / 8;
        let mi_cols = self.config.width / 8;

        // mi_grid: every 8×8 cell carries the same Inter cell. The
        // PARTITION_NONE 64×64 walker splats one ModeInfo across all
        // 64 cells of each 64×64 SB.
        let inter_cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: inter_skip,
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

        // Compressed header. pre == post == new_default() so every
        // prob-diff-update emits a "no-update" gate bit. `tx_mode =
        // AllowTx32x32` to keep tx_size implicit. `reference_mode =
        // Single` matches the frame's single-ref usage.
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

        // Per-tile inter-frame mode-info partition.
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        // `InterModeReadParams` for a single-ref ZEROMV frame. The
        // compound-ref derivation is syntactically valid even though
        // we never emit compound bits (params.reference_mode =
        // Single).
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
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
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
                /* seg = */ None,
                /* prev_mv_grid = */ None,
                &mut coef_ctx,
                &coef_source,
            )?;
            bw.stop_encode()
                .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
        };
        tile_buf.truncate(tile_len);

        // Uncompressed header. Inter frame, single-ref LAST →
        // ref_frame_indices = [0, 0, 0] (all three slots point at
        // pool slot 0, which the keyframe splat-populated).
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
                base_qindex: self.config.base_qindex,
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
            kind: FrameKind::InterFrame {
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
            width: self.config.width,
            height: self.config.height,
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
}

// ---------------------------------------------------------------------------
// F5.1 — DC_PRED intra residual encoder simulator.
// ---------------------------------------------------------------------------
//
// Mirror of the decoder's `reconstruct_intra_planes` (rvp9-decoder::tile)
// inner per-TX loop, running in the encoder direction:
//
// For each plane in 0..3, walk each TX position inside each 64×64 SB
// (raster order). At each TX block:
//   1. Build the prediction edges from the local recon buffer (the
//      decoder's `build_intra_predictors` analogue).
//   2. Run `intra_predict` to fill the destination with prediction.
//   3. Compute residual = input - pred.
//   4. Forward DCT of the residual (per TX size).
//   5. Forward-quant the DCT coefs at the frame qindex.
//   6. Apply inverse-DCT-add of the dequantized coefs to the pred
//      buffer — produces the decoder's exact reconstruction.
//   7. Record the qcoeff buffer + eob for the writer's coef source.
//
// The qcoeff buffers are addressed by the same (mi_row, mi_col, plane,
// tx_x4_local, tx_y4_local) tuple the writer hands to
// `IntraCoefSource::get_block_coefs`.

/// Key into [`VecBackedIntraCoefSource::blocks`].
/// `(mi_row, mi_col, plane, tx_x4_local, tx_y4_local)`.
type IntraCoefBlockKey = (usize, usize, usize, usize, usize);

/// Stored block payload: `(qcoeff buffer, eob)`.
type IntraCoefBlockValue = (Vec<i32>, usize);

/// Encoder simulator output: a Vec-backed [`IntraCoefSource`].
///
/// Stores one qcoeff buffer per (plane, mi_row, mi_col, tx_local_x,
/// tx_local_y, TX size) tuple. The writer iterates these in the same
/// order the simulator produced them, so positional lookup is direct.
pub struct VecBackedIntraCoefSource {
    /// Frame width in pixels (must match the encoder's config).
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// One entry per stored TX block (plane, mi_row, mi_col,
    /// tx_x4_local, tx_y4_local) → (qcoeff buffer, eob).
    ///
    /// We use a flat `HashMap` keyed by the 6-tuple rather than a 5-D
    /// `Vec` to keep the data structure compact and easy to debug.
    blocks: std::collections::HashMap<IntraCoefBlockKey, IntraCoefBlockValue>,
}

impl VecBackedIntraCoefSource {
    /// Build an empty source for the given frame dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            blocks: std::collections::HashMap::new(),
        }
    }

    /// Record one TX block's quantized coefficients + eob. `plane`
    /// indexes 0/1/2 for Y/U/V; the local TX coordinates `(tx_x4_local,
    /// tx_y4_local)` are in plane-relative 4×4 sub-block units inside
    /// the SB the (mi_row, mi_col) cell anchors.
    #[allow(clippy::too_many_arguments)] // mirrors IntraCoefSource trait shape
    pub fn insert(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        qcoefs: Vec<i32>,
        eob: usize,
    ) {
        self.blocks
            .insert((mi_row, mi_col, plane, tx_x4_local, tx_y4_local), (qcoefs, eob));
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl rvp9_core::bitstream::IntraCoefSource for VecBackedIntraCoefSource {
    fn get_block_coefs(
        &self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        tx_size: TxSize,
    ) -> (&[i32], usize) {
        let entry = self
            .blocks
            .get(&(mi_row, mi_col, plane, tx_x4_local, tx_y4_local))
            .expect("VecBackedIntraCoefSource: missing TX block");
        let n = tx_size.n_coeffs();
        debug_assert!(entry.0.len() >= n);
        (&entry.0[..n], entry.1)
    }
}

/// F10.1c — Vec-backed [`rvp9_core::bitstream::InterCoefSource`].
///
/// Stores one qcoeff buffer per (mi_row, mi_col, plane, tx_x4_local,
/// tx_y4_local) tuple. Mirror of [`VecBackedIntraCoefSource`]'s shape,
/// but consumed by the inter-frame writer
/// ([`rvp9_core::bitstream::write_tile_inter`]).
///
/// Populated by [`pick_and_simulate_inter_encode_partitions`] — the
/// F10.1c inter encoder simulator.
pub struct VecBackedInterCoefSource {
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// One entry per stored TX block. Same key shape as
    /// [`VecBackedIntraCoefSource`].
    blocks: std::collections::HashMap<IntraCoefBlockKey, IntraCoefBlockValue>,
}

impl VecBackedInterCoefSource {
    /// Build an empty source for the given frame dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            blocks: std::collections::HashMap::new(),
        }
    }

    /// Record one TX block's quantized coefficients + eob.
    #[allow(clippy::too_many_arguments)] // mirrors InterCoefSource trait shape
    pub fn insert(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        qcoefs: Vec<i32>,
        eob: usize,
    ) {
        self.blocks
            .insert((mi_row, mi_col, plane, tx_x4_local, tx_y4_local), (qcoefs, eob));
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl rvp9_core::bitstream::InterCoefSource for VecBackedInterCoefSource {
    fn get_block_coefs(
        &self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        tx_size: TxSize,
    ) -> (&[i32], usize) {
        let entry = self
            .blocks
            .get(&(mi_row, mi_col, plane, tx_x4_local, tx_y4_local))
            .unwrap_or_else(|| {
                panic!(
                    "VecBackedInterCoefSource: missing TX block at \
                     (mi_row={mi_row}, mi_col={mi_col}, plane={plane}, \
                     tx_x4_local={tx_x4_local}, tx_y4_local={tx_y4_local})"
                )
            });
        let n = tx_size.n_coeffs();
        debug_assert!(entry.0.len() >= n);
        (&entry.0[..n], entry.1)
    }
}

/// Per-plane chroma subsampling for 4:2:0 (Profile 0). Mirrors
/// rvp9-decoder's `reconstruct_inter::SUBSAMPLING_*`.
const F5_SUBSAMPLING_X: [usize; 3] = [0, 1, 1];
const F5_SUBSAMPLING_Y: [usize; 3] = [0, 1, 1];

/// `min(luma_tx, max_txsize(uv_plane_block_size))` for UV planes.
fn f5_uv_plane_tx_size(bsize: BlockSize, luma_tx: TxSize) -> TxSize {
    let cap = rvp9_core::block::max_txsize(f5_uv_plane_block_size(bsize));
    if (luma_tx as u8) < (cap as u8) {
        luma_tx
    } else {
        cap
    }
}

/// Chroma plane block size under 4:2:0 (we only ever pass
/// `BlockSize::Block64x64` here, but the helper is written generally
/// to mirror the decoder).
fn f5_uv_plane_block_size(bsize: BlockSize) -> BlockSize {
    use BlockSize::*;
    match bsize {
        Block4x4 | Block4x8 | Block8x4 | Block8x8 => Block4x4,
        Block8x16 => Block4x8,
        Block16x8 => Block8x4,
        Block16x16 => Block8x8,
        Block16x32 => Block8x16,
        Block32x16 => Block16x8,
        Block32x32 => Block16x16,
        Block32x64 => Block16x32,
        Block64x32 => Block32x16,
        Block64x64 => Block32x32,
    }
}

/// AQ.1 — compute the effective qindex for a leaf, given the base
/// qindex and the per-segment Q-delta table from the uncompressed
/// header (`SegmentationParams.feature_data[seg][SEG_LVL_ALT_Q]`).
///
/// Mirrors the decoder's `tile.rs::plane_dequant` derivation in
/// delta-mode (`abs_delta = false`, which is the only mode the libvpx
/// cyclic-refresh path / AQ.0–AQ.3 ever emits today). The returned
/// value is clamped to `0..=MAX_Q` (where `MAX_Q == 255`).
///
/// `segment_id == 0` with `seg_q_deltas[0] = None` is the
/// AQ-disabled common case — returns `base_qindex` unchanged. This
/// preserves Stage 4a byte-exact behaviour when the frame loop passes
/// an all-`None` table.
#[inline]
#[must_use]
pub(crate) fn effective_qindex_for_segment(
    base_qindex: u8,
    segment_id: u8,
    seg_q_deltas: &[Option<i16>; 8],
) -> u8 {
    match seg_q_deltas[segment_id as usize] {
        Some(delta) => (i32::from(base_qindex) + i32::from(delta)).clamp(0, 255) as u8,
        None => base_qindex,
    }
}

/// Per-leaf descriptor handed to
/// [`simulate_dc_intra_encode_partitions`] — describes one intra leaf
/// inside a 64×64 SB.
///
/// `(mi_row_in_sb, mi_col_in_sb)` is the leaf's top-left in 8×8 mi
/// units relative to the SB's top-left (i.e. each is in `0..8`).
/// `leaf_bsize` is the leaf's [`BlockSize`] (square, ≥ 8×8 for F10.1a).
/// `predicted_y_mode` is the intra mode the simulator should use for
/// the leaf's prediction + forward / inverse transform pair. The single-
/// leaf form ([`simulate_dc_intra_encode`]) and the partition-aware
/// frame loop's pre-F10.1b path use `IntraMode::Dc`; F10.1b drives this
/// from [`crate::mode_decision::intra::pick_intra_mode_luma`].
/// `segment_id` selects the per-segment Q-delta entry in
/// `seg_q_deltas` that the forward-quant path uses for this leaf
/// (AQ.1). For AQ-disabled frames callers leave it `0` and the
/// simulator's `seg_q_deltas` will be all-`None`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct PartitionLeafDesc {
    /// Leaf's row inside the SB, in 8×8 mi units (0..=7).
    pub mi_row_in_sb: usize,
    /// Leaf's column inside the SB, in 8×8 mi units (0..=7).
    pub mi_col_in_sb: usize,
    /// Leaf [`BlockSize`] (square: `Block8x8` / `Block16x16` /
    /// `Block32x32` / `Block64x64`).
    pub leaf_bsize: BlockSize,
    /// Intra Y-mode the simulator should use for prediction +
    /// forward/inverse transforms (luma only — UV stays DC for F10.1b).
    pub predicted_y_mode: IntraMode,
    /// AQ.1 — segment id `0..=7` driving the per-leaf effective
    /// qindex via [`effective_qindex_for_segment`]. `0` for AQ-off.
    pub segment_id: u8,
}

/// Encoder simulator: per-TX-block predict → residual → fdct →
/// forward-quant → idct-add. Returns a [`VecBackedIntraCoefSource`]
/// populated with the per-TX qcoeff buffers; fills `recon_yuv_out`
/// with the per-TX reconstructed pixels (the exact bit pattern the
/// decoder will produce given the emitted bitstream).
///
/// Single-leaf form: every 64×64 SB is treated as a `Block64x64`
/// leaf. For F10.1a's partition-aware path, see
/// [`simulate_dc_intra_encode_partitions`].
///
/// Caller invariants (checked outside):
/// - `(width, height)` are multiples of 64.
/// - `input_yuv` plane lengths match the YUV420 tight-stride layout.
/// - `recon_yuv_out` plane lengths match `input_yuv`.
pub(crate) fn simulate_dc_intra_encode(
    width: u32,
    height: u32,
    base_qindex: u8,
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
) -> VecBackedIntraCoefSource {
    // Single Block64x64 leaf per SB; SBs walked in raster order.
    let sb_cols = (width as usize) / 64;
    let sb_rows = (height as usize) / 64;
    let mut sb_partitions: Vec<Vec<PartitionLeafDesc>> =
        Vec::with_capacity(sb_rows * sb_cols);
    for _ in 0..(sb_rows * sb_cols) {
        sb_partitions.push(vec![PartitionLeafDesc {
            mi_row_in_sb: 0,
            mi_col_in_sb: 0,
            leaf_bsize: BlockSize::Block64x64,
            predicted_y_mode: IntraMode::Dc,
            segment_id: 0,
        }]);
    }
    // AQ-off path — all-`None` deltas table preserves Stage 4a
    // behaviour (every leaf uses `base_qindex` verbatim).
    let seg_q_deltas: [Option<i16>; 8] = [None; 8];
    simulate_dc_intra_encode_partitions(
        width,
        height,
        base_qindex,
        &seg_q_deltas,
        input_yuv,
        recon_yuv_out,
        &sb_partitions,
    )
}

/// Partition-aware version of [`simulate_dc_intra_encode`].
///
/// `sb_partitions[sb_row * sb_cols + sb_col]` is the slice of leaves
/// (in raster scan inside the SB) for the SB at `(sb_row, sb_col)`.
/// Each leaf is encoded as `DC_PRED` intra; the simulator records
/// qcoeffs / eob per (mi_row, mi_col, plane, tx_x4_local, tx_y4_local)
/// — keyed off the leaf's top-left `(mi_row, mi_col)` in the frame —
/// and fills `recon_yuv_out` with the decoder-bit-exact reconstruction.
///
/// Leaves are walked in the order the slice provides them; for the
/// F10.1a wiring this is a raster scan inside the SB, which matches
/// the writer's tree-walk visit order (Z-order at SPLIT is equivalent
/// to raster for non-overlapping leaves at any depth).
pub(crate) fn simulate_dc_intra_encode_partitions(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> VecBackedIntraCoefSource {
    // Initialise the recon plane to zeroes (we'll fill TX-by-TX). The
    // F10.1b picker-driven path skips this and uses
    // [`simulate_dc_intra_encode_partitions_no_zero`] so previously-
    // encoded leaves' recon is preserved across calls.
    for px in recon_yuv_out.0.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.1.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.2.iter_mut() {
        *px = 0;
    }
    simulate_dc_intra_encode_partitions_no_zero(
        width,
        height,
        base_qindex,
        seg_q_deltas,
        input_yuv,
        recon_yuv_out,
        sb_partitions,
    )
}

/// `simulate_dc_intra_encode_partitions` without the up-front recon
/// zeroing — the F10.1b picker path calls this once per leaf and needs
/// the recon buffer from earlier leaves preserved across calls.
pub(crate) fn simulate_dc_intra_encode_partitions_no_zero(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> VecBackedIntraCoefSource {
    use rvp9_core::intra_pred::{
        DcAvailability, EdgeBuffers, IntraMode as PredIntraMode, build_intra_predictors,
        intra_predict,
    };
    use rvp9_core::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, scan_for_forward, vp9_quantize_b,
        vp9_quantize_b_32x32,
    };
    use rvp9_core::transforms::forward::{fdct32x32, fht4x4, fht8x8, fht16x16};
    use rvp9_core::transforms::inverse::{
        idct32x32_1024_add, idct4x4_16_add, idct8x8_64_add, idct16x16_256_add,
        iht4x4_16_add, iht8x8_64_add, iht16x16_256_add,
    };
    use rvp9_core::transforms::scans::scan_for;
    use rvp9_core::transforms::TxType;

    // G4.3.1 — `intra_predict` takes a `&SimdDispatch` so the DC 4×4
    // both-avail path can route through the SIMD slot. Fetch once per
    // call; `dsp::get()` is one atomic load.
    let dispatch = rvp9_core::dsp::get();

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;

    let mut out = VecBackedIntraCoefSource::new(width, height);

    let sb_cols = w / 64;
    let sb_rows = h / 64;
    assert_eq!(
        sb_partitions.len(),
        sb_rows * sb_cols,
        "sb_partitions must have one entry per 64x64 SB ({} rows × {} cols)",
        sb_rows,
        sb_cols,
    );

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            for leaf in leaves {
                let leaf_bsize = leaf.leaf_bsize;
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let luma_tx = rvp9_core::block::max_txsize(leaf_bsize);
                // F10.1b: Y plane uses the picked intra mode; UV stays
                // DC for v1 (UV mode picker is deferred). The Y mode
                // also drives `tx_type` for the forward transform +
                // scan order at sizes < Tx32x32 (Tx32x32 always uses
                // DctDct, mirroring the decoder + writer).
                let leaf_y_mode = leaf.predicted_y_mode;

                // AQ.1 — per-leaf effective qindex. When AQ is off
                // (`seg_q_deltas` all `None`), this collapses to
                // `base_qindex` and the recomputed `qp_y` / `qp_uv` are
                // byte-equal to the old per-frame values, so the
                // Stage 4a tests are unaffected.
                let leaf_qindex = i32::from(effective_qindex_for_segment(
                    base_qindex,
                    leaf.segment_id,
                    seg_q_deltas,
                ));
                let qp_y = init_plane_quantizers(leaf_qindex, QuantizerDeltas::Y);
                let qp_uv = init_plane_quantizers(leaf_qindex, QuantizerDeltas::Y);

                // Iterate planes: 0=Y, 1=U, 2=V.
                for plane in 0..3 {
                    let ss_x = F5_SUBSAMPLING_X[plane];
                    let ss_y = F5_SUBSAMPLING_Y[plane];

                    // Plane dimensions.
                    let plane_w = if plane == 0 { w } else { uvw };
                    let plane_h = if plane == 0 { h } else { uvh };

                    // Per-plane TX size and block size for this leaf.
                    let plane_tx = if plane == 0 {
                        luma_tx
                    } else {
                        f5_uv_plane_tx_size(leaf_bsize, luma_tx)
                    };
                    let plane_bsize = if plane == 0 {
                        leaf_bsize
                    } else {
                        f5_uv_plane_block_size(leaf_bsize)
                    };

                    // Plane 4×4 extents for this leaf. Mirrors the
                    // decoder's `reconstruct_intra_planes` derivation:
                    // sub-8 luma pins to `(2 >> ss)` but for F10.1a we
                    // only see ≥ 8×8 leaves so the regular path is fine.
                    let plane_bw_4x4 = plane_bsize.n_4x4_wide() as usize;
                    let plane_bh_4x4 = plane_bsize.n_4x4_high() as usize;
                    let tx_step_4x4 = 1usize << (plane_tx as usize);
                    let tx_side = plane_tx.side();

                    // Leaf top-left in plane pixels.
                    let leaf_plane_px_x = (leaf_mi_col * 8) >> ss_x;
                    let leaf_plane_px_y = (leaf_mi_row * 8) >> ss_y;

                    // Per-plane references — split mutable / immutable.
                    let (input_plane, recon_plane) = match plane {
                        0 => (input_yuv.0, &mut *recon_yuv_out.0),
                        1 => (input_yuv.1, &mut *recon_yuv_out.1),
                        _ => (input_yuv.2, &mut *recon_yuv_out.2),
                    };
                    let plane_stride = plane_w;

                    // Dequant pair / quantizer params: Y uses qp_y, UV
                    // uses qp_uv (delta-q zero → identical today).
                    let dequant_pair = if plane == 0 {
                        [qp_y.dequant[0], qp_y.dequant[1]]
                    } else {
                        [qp_uv.dequant[0], qp_uv.dequant[1]]
                    };
                    let (zbin, round, quant, quant_shift) = if plane == 0 {
                        (qp_y.zbin, qp_y.round, qp_y.quant, qp_y.quant_shift)
                    } else {
                        (qp_uv.zbin, qp_uv.round, qp_uv.quant, qp_uv.quant_shift)
                    };

                    // Leaf-level neighbour availability: single tile,
                    // single-slice frame, so it reduces to "the leaf
                    // is not at the top-left of the frame".
                    let have_above_block = leaf_mi_row > 0;
                    let have_left_block = leaf_mi_col > 0;

                    // Walk TX positions inside this plane's coding
                    // block. `loff` / `aoff` are in 4×4 units; the per-
                    // TX `have_top` / `have_left` formulas mirror
                    // `reconstruct_intra_planes`.
                    let mut tx_y4_local = 0usize;
                    while tx_y4_local < plane_bh_4x4 {
                        let mut tx_x4_local = 0usize;
                        while tx_x4_local < plane_bw_4x4 {
                            let tx_px_x = leaf_plane_px_x + tx_x4_local * 4;
                            let tx_px_y = leaf_plane_px_y + tx_y4_local * 4;

                            let have_top = have_above_block || tx_y4_local > 0;
                            let have_left = have_left_block || tx_x4_local > 0;
                            let have_right =
                                (tx_x4_local + tx_step_4x4) < plane_bw_4x4;

                            // Y plane uses the leaf's picked mode; UV
                            // is DC (UV-mode picker is F10.1c+).
                            let plane_mode = if plane == 0 {
                                leaf_y_mode
                            } else {
                                PredIntraMode::Dc
                            };
                            let mut edges = EdgeBuffers::new();
                            build_intra_predictors(
                                recon_plane,
                                plane_stride,
                                tx_px_x,
                                tx_px_y,
                                plane_w,
                                plane_h,
                                plane_tx,
                                plane_mode,
                                have_top,
                                have_left,
                                have_right,
                                &mut edges,
                            );
                            let pred_ctx = edges.pred_ctx(tx_side);
                            let dc_avail = DcAvailability {
                                above_available: have_top,
                                left_available: have_left,
                            };

                            // 2. Run prediction into recon.
                            let dst_off = tx_px_y * plane_stride + tx_px_x;
                            let dst = &mut recon_plane[dst_off..];
                            intra_predict(
                                plane_mode,
                                dst,
                                plane_stride,
                                tx_side,
                                pred_ctx,
                                dc_avail,
                                &dispatch,
                            );

                            // 3. Residual = input - pred.
                            let mut residual = vec![0i16; tx_side * tx_side];
                            for r in 0..tx_side {
                                for c in 0..tx_side {
                                    let px_x = tx_px_x + c;
                                    let px_y = tx_px_y + r;
                                    let in_off = px_y * plane_stride + px_x;
                                    let pred = recon_plane[in_off] as i16;
                                    let inp = input_plane[in_off] as i16;
                                    residual[r * tx_side + c] = inp - pred;
                                }
                            }

                            // TX type. Y plane: `plane_mode.tx_type()`
                            // (collapses to DctDct for Tx32x32 below).
                            // UV plane: always DctDct. Matches the
                            // writer's `tx_type` derivation at
                            // `modes_writer::write_intra_block_residual`.
                            let tx_type = if plane == 0 && plane_tx != TxSize::Tx32x32 {
                                plane_mode.tx_type()
                            } else {
                                TxType::DctDct
                            };

                            // 4. Forward DCT / FHT (per TX size).
                            //    Tx32x32 has no hybrid → fdct32x32. Smaller
                            //    sizes use the hybrid `fht*` family with
                            //    `tx_type`.
                            let n = plane_tx.n_coeffs();
                            let mut coefs_i32 = vec![0i32; n];
                            match plane_tx {
                                TxSize::Tx32x32 => {
                                    let mut buf = [0i32; 1024];
                                    fdct32x32(&residual, &mut buf, tx_side);
                                    coefs_i32.copy_from_slice(&buf[..]);
                                }
                                TxSize::Tx16x16 => {
                                    let mut buf = [0i32; 256];
                                    fht16x16(&residual, &mut buf, tx_side, tx_type);
                                    coefs_i32.copy_from_slice(&buf[..]);
                                }
                                TxSize::Tx8x8 => {
                                    let mut buf = [0i32; 64];
                                    fht8x8(&residual, &mut buf, tx_side, tx_type);
                                    coefs_i32.copy_from_slice(&buf[..]);
                                }
                                TxSize::Tx4x4 => {
                                    let mut buf = [0i32; 16];
                                    fht4x4(&residual, &mut buf, tx_side, tx_type);
                                    coefs_i32.copy_from_slice(&buf[..]);
                                }
                            }

                            // 5. Forward quant. Scan follows the writer:
                            //    Tx32x32 → DctDct scan; smaller sizes use
                            //    `scan_for(plane_tx, tx_type)`.
                            let scan = scan_for(plane_tx, tx_type);
                            let scan_slice = scan_for_forward(&scan);
                            let mut qcoeff_i16 = vec![0i16; n];
                            let mut dqcoeff_i16 = vec![0i16; n];
                            let eob = match plane_tx {
                                TxSize::Tx32x32 => vp9_quantize_b_32x32(
                                    &coefs_i32,
                                    /* skip_block = */ false,
                                    &zbin,
                                    &round,
                                    &quant,
                                    &quant_shift,
                                    &mut qcoeff_i16,
                                    &mut dqcoeff_i16,
                                    &dequant_pair,
                                    scan_slice,
                                ),
                                _ => vp9_quantize_b(
                                    &coefs_i32,
                                    n,
                                    /* skip_block = */ false,
                                    &zbin,
                                    &round,
                                    &quant,
                                    &quant_shift,
                                    &mut qcoeff_i16,
                                    &mut dqcoeff_i16,
                                    &dequant_pair,
                                    scan_slice,
                                ),
                            };
                            let eob_usize = eob as usize;

                            // 6. Inverse-DCT-add of the dequant coefs.
                            let mut dq_i32 = vec![0i32; n];
                            for i in 0..n {
                                dq_i32[i] = dqcoeff_i16[i] as i32;
                            }
                            if eob_usize > 0 {
                                match plane_tx {
                                    TxSize::Tx32x32 => {
                                        let mut buf = [0i32; 1024];
                                        buf.copy_from_slice(&dq_i32[..]);
                                        idct32x32_1024_add(
                                            &buf,
                                            &mut recon_plane[dst_off..],
                                            plane_stride,
                                        );
                                    }
                                    TxSize::Tx16x16 => {
                                        let mut buf = [0i32; 256];
                                        buf.copy_from_slice(&dq_i32[..]);
                                        if tx_type == TxType::DctDct {
                                            idct16x16_256_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                            );
                                        } else {
                                            iht16x16_256_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                                tx_type,
                                            );
                                        }
                                    }
                                    TxSize::Tx8x8 => {
                                        let mut buf = [0i32; 64];
                                        buf.copy_from_slice(&dq_i32[..]);
                                        if tx_type == TxType::DctDct {
                                            idct8x8_64_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                            );
                                        } else {
                                            iht8x8_64_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                                tx_type,
                                            );
                                        }
                                    }
                                    TxSize::Tx4x4 => {
                                        let mut buf = [0i32; 16];
                                        buf.copy_from_slice(&dq_i32[..]);
                                        if tx_type == TxType::DctDct {
                                            idct4x4_16_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                            );
                                        } else {
                                            iht4x4_16_add(
                                                &buf,
                                                &mut recon_plane[dst_off..],
                                                plane_stride,
                                                tx_type,
                                            );
                                        }
                                    }
                                }
                            }

                            // 7. Record qcoeff buffer + eob, keyed by
                            //    leaf top-left (mi_row, mi_col) so the
                            //    writer's `IntraCoefSource` lookup hits.
                            let qcoeff_i32: Vec<i32> = qcoeff_i16
                                .into_iter()
                                .map(i32::from)
                                .collect();
                            out.insert(
                                leaf_mi_row,
                                leaf_mi_col,
                                plane,
                                tx_x4_local,
                                tx_y4_local,
                                qcoeff_i32,
                                eob_usize,
                            );

                            tx_x4_local += tx_step_4x4;
                        }
                        tx_y4_local += tx_step_4x4;
                    }
                }
            }
        }
    }

    out
}

/// F10.1b — pick the Y intra mode per leaf via
/// [`rvp9_core::mode_decision::intra::pick_intra_mode_luma`], record it
/// into the leaf descriptor, AND encode the leaf in the same walk so
/// the running recon buffer feeds the next leaf's neighbour fetch.
///
/// Same shape as [`simulate_dc_intra_encode_partitions`] but with the
/// picker invocation interleaved per leaf. Walks SBs/leaves in raster
/// order; for each leaf:
///
/// 1. Materialise neighbour Y samples from the running
///    `recon_yuv_out.0` plane (above row + left column + top-left
///    pixel; the frame-edge leaves fall back to the picker's 127/129
///    libvpx sentinels via `have_above = false` / `have_left = false`).
/// 2. Look up the above-/left-mode contexts from the picked-mode grid.
///    libvpx's `vp9_rd.c:96-101` Y-mode-cost table is indexed
///    `[above_ctx][left_ctx][mode]`; out-of-frame neighbours fall back
///    to `DC_PRED`.
/// 3. Call `pick_intra_mode_luma` to choose the winning mode.
/// 4. Write the picked mode back into `sb_partitions[sb][leaf]
///    .predicted_y_mode` AND the picked-mode grid (splat across every
///    8×8 mi cell the leaf covers).
/// 5. Run the per-leaf encode pass (predict → fdct → quant → idct-add)
///    using that mode — populates both the recon buffer and the
///    qcoeff [`VecBackedIntraCoefSource`].
///
/// UV stays forced to `IntraMode::Dc` for v1 (the UV-mode picker is
/// further down the F8.x roadmap).
pub(crate) fn pick_and_simulate_intra_encode_partitions(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &mut [Vec<PartitionLeafDesc>],
) -> VecBackedIntraCoefSource {
    use rvp9_core::cost::intra_y_mode_costs_kf_table;
    use rvp9_core::mode_decision::intra::{
        IntraModeSearchParams, pick_intra_mode_luma,
    };
    use rvp9_core::mv_search::PlaneRef;
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    let w = width as usize;
    let h = height as usize;
    let sb_cols = w / 64;
    let sb_rows = h / 64;
    assert_eq!(
        sb_partitions.len(),
        sb_rows * sb_cols,
        "sb_partitions must have one entry per 64x64 SB ({} rows × {} cols)",
        sb_rows,
        sb_cols,
    );

    let mi_cols = w / 8;
    let mi_rows = h / 8;
    let mut picked_y_mode_grid: Vec<Option<IntraMode>> =
        vec![None; mi_rows * mi_cols];

    // First pass: walk SBs/leaves in raster, run the simulator one
    // leaf at a time, prepended by a picker call. The simulator
    // helper expects `sb_partitions` to drive *which* leaves get
    // encoded; we feed it a single-element slice per call so it only
    // touches one leaf per invocation.
    //
    // The simulator's neighbour-edge build reads from `recon_yuv_out`,
    // which has been populated by all previously-encoded leaves —
    // exactly the data the picker needs too. So the picker's `above_
    // pixels` / `left_pixels` come from the same buffer.

    let y_mode_costs = intra_y_mode_costs_kf_table();

    // Zero recon up-front — the per-leaf simulator call expects the
    // recon plane to be already populated for previously-encoded leaves
    // and zero outside.
    for px in recon_yuv_out.0.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.1.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.2.iter_mut() {
        *px = 0;
    }

    let mut coef_source = VecBackedIntraCoefSource::new(width, height);

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let sb_idx = sb_row * sb_cols + sb_col;
            for leaf_idx in 0..sb_partitions[sb_idx].len() {
                let (leaf_mi_row, leaf_mi_col, leaf_bsize) = {
                    let leaf = &sb_partitions[sb_idx][leaf_idx];
                    (
                        sb_row * 8 + leaf.mi_row_in_sb,
                        sb_col * 8 + leaf.mi_col_in_sb,
                        leaf.leaf_bsize,
                    )
                };
                let bs_pixels = leaf_bsize.width() as usize;
                debug_assert_eq!(
                    bs_pixels,
                    leaf_bsize.height() as usize,
                    "F10.1b picker expects square leaves"
                );
                let leaf_px_x = leaf_mi_col * 8;
                let leaf_px_y = leaf_mi_row * 8;

                let have_above = leaf_px_y > 0;
                let have_left = leaf_px_x > 0;
                let mut above_pixels = vec![0u8; bs_pixels];
                let mut left_pixels = vec![0u8; bs_pixels];
                if have_above {
                    let above_row_y = leaf_px_y - 1;
                    let row_off = above_row_y * w + leaf_px_x;
                    above_pixels
                        .copy_from_slice(&recon_yuv_out.0[row_off..row_off + bs_pixels]);
                }
                if have_left {
                    let left_col_x = leaf_px_x - 1;
                    for (r, slot) in left_pixels.iter_mut().enumerate() {
                        *slot = recon_yuv_out.0[(leaf_px_y + r) * w + left_col_x];
                    }
                }
                let top_left_pixel = if have_above && have_left {
                    recon_yuv_out.0[(leaf_px_y - 1) * w + (leaf_px_x - 1)]
                } else {
                    0u8
                };

                let above_mode_ctx = if leaf_mi_row > 0 {
                    picked_y_mode_grid
                        [(leaf_mi_row - 1) * mi_cols + leaf_mi_col]
                        .unwrap_or(IntraMode::Dc)
                } else {
                    IntraMode::Dc
                };
                let left_mode_ctx = if leaf_mi_col > 0 {
                    picked_y_mode_grid
                        [leaf_mi_row * mi_cols + (leaf_mi_col - 1)]
                        .unwrap_or(IntraMode::Dc)
                } else {
                    IntraMode::Dc
                };

                // F8.0 picker requires `block_size.side == tx_size.side`
                // (debug_assert at `mode_decision/intra.rs:246`). The
                // luma `max_txsize` for Block64x64 is Tx32x32 (side 32),
                // so a Block64x64 leaf falls outside the picker's
                // current scope — fall back to DC for those. Smaller
                // square leaves (≤ Block32x32) are handled by the
                // picker. Per-32×32 sub-block picking inside a 64×64
                // leaf is on the F8.0 picker's roadmap; for v1 we get
                // DC on the (rare) full-SB-no-split leaves.
                // AQ.1 — per-leaf effective qindex feeds BOTH the
                // picker's RD evaluation AND the simulator's forward-
                // quant init. The simulator gets it indirectly via the
                // per-leaf `seg_q_deltas` lookup against the leaf's
                // `segment_id`; the picker takes the resolved value
                // directly. When AQ is off (all-`None` deltas) this
                // collapses to `base_qindex`.
                let leaf_qindex = effective_qindex_for_segment(
                    base_qindex,
                    sb_partitions[sb_idx][leaf_idx].segment_id,
                    seg_q_deltas,
                );
                let leaf_quantizer = init_plane_quantizers(
                    i32::from(leaf_qindex),
                    QuantizerDeltas::Y,
                );
                let picked_mode = if leaf_bsize == BlockSize::Block64x64 {
                    IntraMode::Dc
                } else {
                    let src_plane = PlaneRef {
                        data: input_yuv.0,
                        stride: w,
                        width: w,
                        height: h,
                    };
                    let params = IntraModeSearchParams {
                        block_size: leaf_bsize,
                        src_plane: &src_plane,
                        src_offset: (leaf_px_x, leaf_px_y),
                        above_pixels: &above_pixels,
                        left_pixels: &left_pixels,
                        top_left_pixel,
                        have_above,
                        have_left,
                        quantizer: &leaf_quantizer,
                        q_index: leaf_qindex,
                        frame_kind:
                            rvp9_core::bitstream::header::FrameKind::KeyFrame,
                        above_mode_ctx,
                        left_mode_ctx,
                        y_mode_costs: &y_mode_costs,
                    };
                    pick_intra_mode_luma(&params).best_mode
                };

                // A0' diagnostic — dump per-leaf KF intra mode pick when
                // RVP9_DIAG_INTRA is set in the environment. Sampling at
                // mod-16 keeps the output bounded on a 256x256 frame.
                if std::env::var("RVP9_DIAG_INTRA").is_ok() {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "    intra_leaf sb=({},{}) px=({},{}) bs={:?} q={} → mode={:?}",
                            sb_row, sb_col, leaf_px_x, leaf_px_y, leaf_bsize,
                            leaf_qindex, picked_mode,
                        );
                    }
                }

                sb_partitions[sb_idx][leaf_idx].predicted_y_mode = picked_mode;
                let n_8x8 = (bs_pixels / 8).max(1);
                for dr in 0..n_8x8 {
                    for dc in 0..n_8x8 {
                        let mi_r = leaf_mi_row + dr;
                        let mi_c = leaf_mi_col + dc;
                        picked_y_mode_grid[mi_r * mi_cols + mi_c] =
                            Some(picked_mode);
                    }
                }

                // Encode this one leaf via the simulator. We construct
                // a per-SB layout containing only this leaf so the
                // simulator's outer loop walks exactly one element.
                // Recon buffer state from earlier leaves is preserved
                // (the simulator no longer zeros recon — we relocated
                // that to up-front, see below).
                let single_leaf_layout: Vec<Vec<PartitionLeafDesc>> = (0..(sb_rows
                    * sb_cols))
                    .map(|i| {
                        if i == sb_idx {
                            vec![sb_partitions[sb_idx][leaf_idx]]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect();
                let leaf_coefs = simulate_dc_intra_encode_partitions_no_zero(
                    width,
                    height,
                    base_qindex,
                    seg_q_deltas,
                    input_yuv,
                    (
                        &mut *recon_yuv_out.0,
                        &mut *recon_yuv_out.1,
                        &mut *recon_yuv_out.2,
                    ),
                    &single_leaf_layout,
                );
                // Merge `leaf_coefs` into the running `coef_source`.
                for ((mi_row, mi_col, plane, tx_x, tx_y), (qc, eob)) in
                    leaf_coefs.blocks.into_iter()
                {
                    coef_source.insert(
                        mi_row, mi_col, plane, tx_x, tx_y, qc, eob,
                    );
                }
            }
        }
    }

    coef_source
}

// ---------------------------------------------------------------------------
// F10.1c — inter-mode picker integration + real inter residual emission.
// ---------------------------------------------------------------------------

/// Per-leaf descriptor for the inter encoder simulator. Mirrors
/// [`PartitionLeafDesc`] but carries the inter mode info (mode + MV +
/// skip flag) the picker chose for the leaf.
///
/// `leaf_bsize` ∈ {Block8x8, Block16x16, Block32x32, Block64x64}.
/// F8.1 (the inter mode picker) only supports Block16x16 / Block32x32 /
/// Block64x64; for Block8x8 leaves the simulator falls back to a
/// `ZEROMV / LASTFRAME / skip = true` cell (the documented F10.1c
/// divergence, on the F10.2 punch-list).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct InterPartitionLeafDesc {
    /// Leaf's row inside the SB, in 8×8 mi units (0..=7).
    pub mi_row_in_sb: usize,
    /// Leaf's column inside the SB, in 8×8 mi units (0..=7).
    pub mi_col_in_sb: usize,
    /// Leaf [`BlockSize`] (square).
    pub leaf_bsize: BlockSize,
    /// Picked inter mode (one of `ZeroMv` / `NearestMv` / `NewMv`).
    /// Always `ZeroMv` on `skip = true` fallback leaves.
    pub picked_mode: InterMode,
    /// Picked MV in 1/8-pel units. `Mv::ZERO` on skip-true leaves.
    pub picked_mv: Mv,
    /// `true` ⇒ this leaf falls back to the skip-true / zero-residual
    /// path (e.g. Block8x8 leaves where F8.1 doesn't apply).
    pub skip: bool,
    /// AQ.1 — segment id `0..=7` driving the per-leaf effective
    /// qindex via [`effective_qindex_for_segment`]. `0` for AQ-off.
    pub segment_id: u8,
}

/// F10.1c — Drive the F8.1 inter mode picker per leaf and run the
/// inter encoder simulator that produces real residual coefficients +
/// a decoder-bit-exact encoder reconstruction.
///
/// For each leaf:
///
/// 1. If `leaf_bsize == Block8x8`, fall back to `ZeroMv / LASTFRAME /
///    skip = true`. The reconstruction is a zero-MV copy of the
///    reference plane (no residual emission); no coef-source entries
///    are recorded.
/// 2. Otherwise, run [`pick_inter_mode_luma`] against the leaf's source
///    block + the LASTFRAME ref plane to pick `(best_mode, best_mv)`.
///    Store the choice in `sb_partitions[sb][leaf]` so the caller can
///    splat it into the mi-grid.
/// 3. Per plane (Y, U, V): convolve8 the ref at the (chroma-scaled) MV
///    to produce the prediction, compute residual = src - pred,
///    forward-DCT + forward-quant, then inverse-DCT-add back into the
///    encoder's recon plane (mirror of the decoder's
///    `reconstruct_inter_block` IT-add sequence). Store
///    `(qcoeffs, eob)` per (plane, tx) in the returned coef source.
///
/// `last_recon_yuv` is the previous frame's reconstruction (the
/// LASTFRAME ref under our zero-MV-no-update reference scheme); the
/// encoder reads it for prediction. `recon_yuv_out` is filled with the
/// new frame's reconstruction (the encoder's "what the decoder will
/// produce" prediction-correctness oracle).
///
/// Walks SBs/leaves in raster order — same convention as the F10.1b
/// intra simulator + writer.
#[allow(clippy::too_many_arguments)] // AQ.1 added `seg_q_deltas`; the
// remaining args are all load-bearing per-frame inputs that don't
// reduce to a meaningful struct yet — revisit when AQ.3 lands.
pub(crate) fn pick_and_simulate_inter_encode_partitions(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    last_recon_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &mut [Vec<InterPartitionLeafDesc>],
) -> VecBackedInterCoefSource {
    use rvp9_core::bitstream::frame::{InterpFilter as WireInterpFilter, MAX_INTER_REFS};
    use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
    use rvp9_core::cost::mv_cost::build_nmv_cost_table;
    use rvp9_core::cost::symbol_costs::inter_mode_costs;
    use rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
    use rvp9_core::mode_decision::inter::{
        InterModeSearchParams, MvCandidates, pick_inter_mode_luma,
    };
    use rvp9_core::mv_search::mv_limits::MvLimits;
    use rvp9_core::mv_search::{PlaneRef, init_dsmotion_compensation};
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;

    let sb_cols = w / 64;
    let sb_rows = h / 64;

    // A1' — Frame-area gate on the Block8x8 inter picker. The
    // variance partitioner descends to Block8x8 at high-motion edges
    // (`partition.rs:331`), and A1 (`f7d5cb8`) lifted these into the
    // F8.1 picker. That recovers the testsrc PSNR (24 dB lock →
    // 35-52 dB sweep) but blows the 33 ms 720p30 RT budget — the
    // picker's NEWMV diamond + sub-pel refinement + per-plane
    // round-trip transform is too expensive to run on every
    // Block8x8 leaf at 720p (the variance partitioner descends
    // densely on natural-content frames).
    //
    // libvpx applies similar size-conditional restrictions in its
    // realtime path. The cheapest fix that preserves A1's
    // correctness gain on small frames where the budget can absorb
    // it: gate the Block8x8 picker on frame area. The threshold
    // (307_200 px == 640 × 480) cleanly separates the F12 small-
    // frame corpus from the RT-path 720p+ corpus:
    //   * testsrc 256×256 = 65_536 → enabled (PSNR sweep preserved)
    //   * foreman_cif 352×288 = 101_376 → enabled
    //   * 720p 1280×704 = 901_120 → disabled (RT gate recovers)
    // Block8x8 leaves on disabled frames fall through to the
    // pre-A1 ZEROMV-skip path (same as the sub-8 partitions emitted
    // wire-format-only by F8.2).
    let block8x8_picker_enabled = w * h <= 307_200;

    assert_eq!(
        sb_partitions.len(),
        sb_rows * sb_cols,
        "sb_partitions must have one entry per 64x64 SB ({} rows × {} cols)",
        sb_rows,
        sb_cols,
    );

    // Plane refs into the input + last-recon buffers.
    let src_y = PlaneRef { data: input_yuv.0, stride: w, width: w, height: h };
    let ref_y = PlaneRef { data: last_recon_yuv.0, stride: w, width: w, height: h };

    // Picker harness — built once per frame. The per-leaf
    // `PlaneQuantizers` is rebuilt inside the SB walk below (AQ.1)
    // because it depends on the leaf's `segment_id`.
    let inter_costs = inter_mode_costs(&FrameContext::new_default());
    let mv_costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, /* allow_hp = */ false);
    let site_config = init_dsmotion_compensation(w as i32);

    // Inter-frame `FrameKind` for `compute_rd_mult`. Only the variant
    // matters; the inner fields don't drive the picker.
    let frame_kind = WireFrameKind::InterFrame {
        reset_frame_context: 0,
        refresh_frame_flags: 0,
        ref_frame_indices: [0u8; MAX_INTER_REFS],
        ref_frame_sign_biases: [false; MAX_INTER_REFS],
        allow_high_precision_mv: false,
        interp_filter: WireInterpFilter::EightTap,
    };

    let mut coef_source = VecBackedInterCoefSource::new(width, height);

    // Initialise the recon planes from the LAST ref. ZERO-MV / skip
    // leaves get this as their final recon (no residual added); the
    // picker-driven leaves overwrite their region via the per-TX
    // IT-add. Mirrors the decoder's "MC writes into dst, then residual
    // IT-add applies on top" sequence.
    recon_yuv_out.0.copy_from_slice(last_recon_yuv.0);
    recon_yuv_out.1.copy_from_slice(last_recon_yuv.1);
    recon_yuv_out.2.copy_from_slice(last_recon_yuv.2);

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let sb_idx = sb_row * sb_cols + sb_col;
            for leaf_idx in 0..sb_partitions[sb_idx].len() {
                let (leaf_mi_row, leaf_mi_col, leaf_bsize) = {
                    let leaf = &sb_partitions[sb_idx][leaf_idx];
                    (
                        sb_row * 8 + leaf.mi_row_in_sb,
                        sb_col * 8 + leaf.mi_col_in_sb,
                        leaf.leaf_bsize,
                    )
                };
                let leaf_px_x = leaf_mi_col * 8;
                let leaf_px_y = leaf_mi_row * 8;
                let bw = usize::from(leaf_bsize.width());
                let bh = usize::from(leaf_bsize.height());

                // F8.1 picker supports Block8x8 / Block16x16 /
                // Block32x32 / Block64x64. Any other size (sub-8
                // partitions emitted wire-format-only by F8.2) falls
                // back to ZEROMV / skip = true; the recon for those
                // leaves equals the reference (already populated via
                // the up-front copy above), and no coef entries are
                // recorded. A1 — lifted Block8x8 into the picker so
                // high-motion 8×8 leaves get real MV + residual instead
                // of locking to LASTFRAME (A0 diagnosis: at peak motion
                // ~130-140 Block8x8 leaves per frame previously fell
                // through to ZEROMV-skip). A1' — frame-area-gated; see
                // `block8x8_picker_enabled` above (above the SB walk).
                let picker_supported = match leaf_bsize {
                    BlockSize::Block8x8 => block8x8_picker_enabled,
                    BlockSize::Block16x16
                    | BlockSize::Block32x32
                    | BlockSize::Block64x64 => true,
                    _ => false,
                };

                if !picker_supported {
                    sb_partitions[sb_idx][leaf_idx].picked_mode = InterMode::ZeroMv;
                    sb_partitions[sb_idx][leaf_idx].picked_mv = Mv::ZERO;
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                    if std::env::var("RVP9_DIAG_PICKER").is_ok() {
                        #[allow(clippy::print_stderr)]
                        {
                            eprintln!(
                                "    SKIP-fallback (Block8x8) leaf sb=({},{}) px=({},{}) bs={:?}",
                                sb_row, sb_col, leaf_px_x, leaf_px_y, leaf_bsize
                            );
                        }
                    }
                    continue;
                }

                // Constrain MV limits so the F8.1 picker's convolve
                // footprint stays inside the frame on EVERY plane
                // (luma + 4:2:0 chroma). The 8-tap kernel touches
                // `int_pel - 3 .. int_pel + bw + 4` in the
                // unscaled-with-subpel case.
                //
                // Luma:   `leaf_px + mv_int - 3 >= 0` AND
                //         `leaf_px + mv_int + bw + 4 <= plane_w`.
                // Chroma: same constraint on the chroma plane, where
                //         `chroma_leaf = leaf_px / 2`, `chroma_bw =
                //         bw / 2`, `chroma_w = w / 2`, and the chroma
                //         Q4 MV equals the luma Q3 MV. Because the
                //         picker's sub-pel refinement can drive the
                //         Q3 luma MV to `mv_fullpel*8 ± 7`, the chroma
                //         integer-pel base can drop one pixel below
                //         the luma integer-pel mapping when the
                //         physical motion lands on a half-chroma-pel.
                //         The arithmetic resolves to an envelope
                //         three luma full-pels TIGHTER on every side
                //         than the luma-only envelope. A1 — without
                //         this tightening, Block8x8 leaves at frame
                //         edges with the picked MV at the luma-only
                //         boundary underflowed `build_inter_pred_plane_unscaled`
                //         on the chroma plane (the simulator does not
                //         border-extend; see its caller-invariant
                //         comment).
                //
                // If the resulting envelope is degenerate (col_min >
                // col_max) the leaf is too close to a frame edge for
                // any non-zero MV; skip the picker and fall back to
                // ZEROMV / skip = true so the assert in F8.1's
                // `build_inter_prediction` doesn't fire.
                let col_min_luma = -(leaf_px_x as i32) + 3;
                let col_max_luma = (w as i32) - (leaf_px_x as i32) - (bw as i32) - 4;
                let row_min_luma = -(leaf_px_y as i32) + 3;
                let row_max_luma = (h as i32) - (leaf_px_y as i32) - (bh as i32) - 4;
                let col_min_chroma = 6 - leaf_px_x as i32;
                let col_max_chroma = (w as i32) - (leaf_px_x as i32) - (bw as i32) - 7;
                let row_min_chroma = 6 - leaf_px_y as i32;
                let row_max_chroma = (h as i32) - (leaf_px_y as i32) - (bh as i32) - 7;
                let col_min = col_min_luma.max(col_min_chroma);
                let col_max = col_max_luma.min(col_max_chroma);
                let row_min = row_min_luma.max(row_min_chroma);
                let row_max = row_max_luma.min(row_max_chroma);
                if col_min > col_max || row_min > row_max {
                    sb_partitions[sb_idx][leaf_idx].picked_mode = InterMode::ZeroMv;
                    sb_partitions[sb_idx][leaf_idx].picked_mv = Mv::ZERO;
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                    continue;
                }
                let mv_limits = MvLimits::new(col_min, col_max, row_min, row_max);

                // MV candidates — under our v1 LAST-frame-only ref
                // scheme the nearest / near refs default to zero (the
                // previous frame mi-grid carries only zero MVs and we
                // don't thread a prev_mv_grid through). For F10.1c
                // this matches the picker's `MvCandidates::default()`.
                let mvc = MvCandidates::default();

                // AQ.1 — per-leaf effective qindex from the leaf's
                // `segment_id` + the frame-level `seg_q_deltas` table.
                // Feeds BOTH the picker's RD evaluation and the
                // forward-quant `quantizer` used in
                // `inter_forward_quant_one_tx` below.
                let leaf_qindex = effective_qindex_for_segment(
                    base_qindex,
                    sb_partitions[sb_idx][leaf_idx].segment_id,
                    seg_q_deltas,
                );
                let quantizer = init_plane_quantizers(
                    i32::from(leaf_qindex),
                    QuantizerDeltas::Y,
                );

                let params = InterModeSearchParams {
                    block_size: leaf_bsize,
                    src_plane: &src_y,
                    src_offset: (leaf_px_x, leaf_px_y),
                    ref_plane: &ref_y,
                    mv_candidates: &mvc,
                    mv_limits,
                    site_config: &site_config,
                    mv_costs: &mv_costs,
                    inter_mode_costs: &inter_costs,
                    inter_mode_ctx: 0,
                    quantizer: &quantizer,
                    q_index: leaf_qindex,
                    allow_hp: false,
                    frame_kind,
                };
                let pick = pick_inter_mode_luma(&params);

                sb_partitions[sb_idx][leaf_idx].picked_mode = pick.best_mode;
                sb_partitions[sb_idx][leaf_idx].picked_mv = pick.best_mv;
                sb_partitions[sb_idx][leaf_idx].skip = false;

                // A0 diagnostic: dump per-leaf picker decision when
                // `RVP9_DIAG_PICKER` is set in the environment.
                if std::env::var("RVP9_DIAG_PICKER").is_ok() {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "    leaf sb=({},{}) px=({},{}) bs={:?} q={} → mode={:?} mv=({},{}) rd={} dist={} rate={}",
                            sb_row, sb_col, leaf_px_x, leaf_px_y, leaf_bsize, leaf_qindex,
                            pick.best_mode,
                            pick.best_mv.row, pick.best_mv.col,
                            pick.best_rd_cost,
                            pick.best_distortion,
                            pick.best_rate,
                        );
                    }
                }

                // RC.X1 — track total eob across all planes & TX blocks.
                // libvpx's decoder (`vp9_decodeframe.c:1101`) sets
                // `mi->skip = 1` AFTER coef-token reads if `eobtotal == 0`
                // (so neighbour blocks see `skip=1` when computing their
                // `skip_context`). We must mirror this on the encoder
                // side: if forward-quant produces all-zero coefs across
                // the entire leaf, mark the leaf `skip=true` so the
                // writer emits a `skip` bit instead of a sequence of
                // EOB-only TX tokens. Otherwise downstream blocks read
                // a different `skip_prob` than what we wrote, desyncing
                // the bool coder.
                let mut leaf_total_eob: usize = 0;

                // Per-plane recon + residual emission. Mirrors the
                // decoder's `reconstruct_inter_block` plane / TX walk
                // (rvp9-decoder::reconstruct_inter, lines 199-554).
                for plane in 0..3 {
                    let ss_x = F5_SUBSAMPLING_X[plane];
                    let ss_y = F5_SUBSAMPLING_Y[plane];
                    let plane_w = if plane == 0 { w } else { uvw };
                    let plane_h = if plane == 0 { h } else { uvh };
                    let plane_stride = plane_w;

                    let plane_bsize = if plane == 0 {
                        leaf_bsize
                    } else {
                        f5_uv_plane_block_size(leaf_bsize)
                    };
                    let luma_tx = rvp9_core::block::max_txsize(leaf_bsize);
                    let plane_tx = if plane == 0 {
                        luma_tx
                    } else {
                        f5_uv_plane_tx_size(leaf_bsize, luma_tx)
                    };

                    let plane_bw_4x4 = plane_bsize.n_4x4_wide() as usize;
                    let plane_bh_4x4 = plane_bsize.n_4x4_high() as usize;
                    let tx_step_4x4 = 1usize << (plane_tx as usize);
                    let tx_side = plane_tx.side();

                    // Plane-pixel coord of leaf's top-left.
                    let plane_px_x = (leaf_mi_col * 8) >> ss_x;
                    let plane_px_y = (leaf_mi_row * 8) >> ss_y;
                    let block_w_px = plane_bw_4x4 * 4;
                    let block_h_px = plane_bh_4x4 * 4;

                    // MV in Q4 units, with chroma scaling. Matches the
                    // decoder's non-sub-8 derivation:
                    //   mv_q4 = mv * (1 << (1 - ss))
                    // (i.e. luma stays at Q4, chroma doubles).
                    let mv_q4_col = i32::from(pick.best_mv.col) * (1i32 << (1 - ss_x as i32));
                    let mv_q4_row = i32::from(pick.best_mv.row) * (1i32 << (1 - ss_y as i32));

                    // Per-plane source / ref / recon slices.
                    let (input_plane, ref_plane_data, recon_plane) = match plane {
                        0 => (input_yuv.0, last_recon_yuv.0, &mut *recon_yuv_out.0),
                        1 => (input_yuv.1, last_recon_yuv.1, &mut *recon_yuv_out.1),
                        _ => (input_yuv.2, last_recon_yuv.2, &mut *recon_yuv_out.2),
                    };

                    // Build the leaf's MC prediction into the recon
                    // plane region (mirrors the decoder's MC step that
                    // writes into `dst_slice` before the residual
                    // IT-add). Uses the unscaled `inter_predictor` path
                    // (encoder ref dims = current dims, so identity
                    // scale).
                    build_inter_pred_plane_unscaled(
                        ref_plane_data,
                        plane_stride,
                        plane_w,
                        plane_h,
                        plane_px_x,
                        plane_px_y,
                        block_w_px,
                        block_h_px,
                        mv_q4_col,
                        mv_q4_row,
                        recon_plane,
                        plane_stride,
                    );

                    // Per-TX residual emission. Walk in plane 4×4
                    // raster (same order the writer walks).
                    let mut tx_y4_local = 0usize;
                    while tx_y4_local < plane_bh_4x4 {
                        let mut tx_x4_local = 0usize;
                        while tx_x4_local < plane_bw_4x4 {
                            let tx_px_x = plane_px_x + tx_x4_local * 4;
                            let tx_px_y = plane_px_y + tx_y4_local * 4;
                            let dst_off = tx_px_y * plane_stride + tx_px_x;

                            // Residual = input - pred. `recon_plane`
                            // currently holds the MC prediction.
                            let mut residual = vec![0i16; tx_side * tx_side];
                            for r in 0..tx_side {
                                for c in 0..tx_side {
                                    let off = (tx_px_y + r) * plane_stride
                                        + (tx_px_x + c);
                                    let s = input_plane[off] as i16;
                                    let p = recon_plane[off] as i16;
                                    residual[r * tx_side + c] = s - p;
                                }
                            }

                            let (qcoeff_i32, dqcoeff_i32, eob_usize) =
                                inter_forward_quant_one_tx(
                                    &residual,
                                    tx_side,
                                    plane_tx,
                                    &quantizer,
                                );

                            // Inverse-DCT-add into recon (the encoder's
                            // bit-exact mirror of the decoder's
                            // `inverse_transform_add_dct`). Skip when
                            // eob == 0 — matches the decoder's
                            // `if eob > 0` guard.
                            if eob_usize > 0 {
                                inter_inverse_dct_add(
                                    plane_tx,
                                    &dqcoeff_i32,
                                    eob_usize,
                                    &mut recon_plane[dst_off..],
                                    plane_stride,
                                );
                            }

                            coef_source.insert(
                                leaf_mi_row,
                                leaf_mi_col,
                                plane,
                                tx_x4_local,
                                tx_y4_local,
                                qcoeff_i32,
                                eob_usize,
                            );
                            leaf_total_eob += eob_usize;

                            tx_x4_local += tx_step_4x4;
                        }
                        tx_y4_local += tx_step_4x4;
                    }
                }

                // RC.X1 — if no coefs in the entire leaf were non-zero,
                // flip the leaf to skip=true. Mirrors libvpx's
                // `if (eobtotal == 0) mi->skip = 1` post-fact override
                // (`vp9_decodeframe.c:1101`). Without this, our writer
                // computes downstream `skip_context` from the original
                // `mi.skip = false` while the decoder computes it from
                // the overridden `mi.skip = true`, desyncing the bool
                // coder at the next block's skip read.
                if leaf_total_eob == 0 {
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                }
                if std::env::var("RVP9_DIAG_PICKER").is_ok() {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "      leaf_total_eob={} (skip={})",
                            leaf_total_eob,
                            sb_partitions[sb_idx][leaf_idx].skip
                        );
                    }
                }
            }
        }
    }

    coef_source
}

/// Build a `bw × bh` inter prediction at `(plane_px_x + mv)` into
/// `dst[..]`, mirroring the decoder's unscaled
/// `build_inter_predictor_with_mc_border` fast path (ref-dim == cur-dim,
/// so [`ScaleFactors`] is identity). MVs are in Q4 (1/16-pel) units in
/// the reference plane's coord system.
///
/// Caller invariant: the MV must be constrained so the convolve8
/// footprint stays inside the visible plane (no border-extend). The
/// F8.1 picker's `MvLimits` enforces this upstream.
#[allow(clippy::too_many_arguments)]
fn build_inter_pred_plane_unscaled(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    plane_px_x: usize,
    plane_px_y: usize,
    bw: usize,
    bh: usize,
    mv_q4_col: i32,
    mv_q4_row: i32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    use rvp9_core::bitstream::frame::InterpFilter;
    use rvp9_core::inter::filter::{SUBPEL_BITS, SUBPEL_MASK};
    use rvp9_core::inter::scale::{ScaleFactors, inter_predictor};

    // Identity scale (ref dims = cur dims).
    let sf = ScaleFactors::new(ref_w as u32, ref_h as u32, ref_w as u32, ref_h as u32);
    debug_assert!(!sf.is_scaled(), "F10.1c v1 encoder assumes identity ref scale");

    let total_x_q4 = sf.scaled_x_q4(plane_px_x as i32, mv_q4_col as i16);
    let total_y_q4 = sf.scaled_y_q4(plane_px_y as i32, mv_q4_row as i16);
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

    // Pre-shift the ref pointer by `(VP9_INTERP_EXTEND - 1) = 3` along
    // each axis where sub-pel is non-zero — mirrors the decoder's fast
    // path at `reconstruct_inter.rs:985-1023`. The encoder's MV-limits
    // bound enforces non-negative `int_y + row_pre` / `int_x + col_pre`.
    let row_pre = if subpel_y_q4 != 0 { -3i32 } else { 0 };
    let col_pre = if subpel_x_q4 != 0 { -3i32 } else { 0 };
    let row_base = (int_y + row_pre) as usize;
    let col_base = (int_x + col_pre) as usize;
    let src_base = row_base * ref_stride + col_base;

    // Destination slice starts at the leaf's top-left in the plane.
    let dst_off = plane_px_y * dst_stride + plane_px_x;

    inter_predictor(
        &ref_plane[src_base..],
        ref_stride,
        &mut dst[dst_off..],
        dst_stride,
        subpel_x_q4,
        subpel_y_q4,
        &sf,
        InterpFilter::EightTap,
        bw,
        bh,
        /* is_compound_avg = */ false,
    );
}

/// Forward DCT (DctDct) + forward quant of one TX block for the inter
/// path. Returns `(qcoeff_i32, dqcoeff_i32, eob)` — the qcoeff buffer
/// the writer will serialise, the dqcoeff buffer to inverse-transform
/// into the recon, and the end-of-block count.
fn inter_forward_quant_one_tx(
    residual: &[i16],
    tx_side: usize,
    plane_tx: TxSize,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
) -> (Vec<i32>, Vec<i32>, usize) {
    use rvp9_core::quant::forward::{scan_for_forward, vp9_quantize_b, vp9_quantize_b_32x32};
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::forward::{fdct8x8, fdct16x16, fdct32x32, fht4x4};
    use rvp9_core::transforms::scans::scan_for;

    let n = plane_tx.n_coeffs();
    let mut coefs = vec![0i32; n];
    match plane_tx {
        TxSize::Tx4x4 => {
            // Inter blocks always use DctDct (no ADST hybrid). `fht4x4`
            // with `TxType::DctDct` is the DCT-DCT variant.
            let mut buf = [0i32; 16];
            fht4x4(residual, &mut buf, tx_side, TxType::DctDct);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx8x8 => {
            let mut buf = [0i32; 64];
            fdct8x8(residual, &mut buf, tx_side);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx16x16 => {
            let mut buf = [0i32; 256];
            fdct16x16(residual, &mut buf, tx_side);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx32x32 => {
            let mut buf = [0i32; 1024];
            fdct32x32(residual, &mut buf, tx_side);
            coefs.copy_from_slice(&buf[..]);
        }
    }

    let scan = scan_for(plane_tx, TxType::DctDct);
    let scan_slice = scan_for_forward(&scan);
    let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
    let mut qcoeff_i16 = vec![0i16; n];
    let mut dqcoeff_i16 = vec![0i16; n];
    let eob = match plane_tx {
        TxSize::Tx32x32 => vp9_quantize_b_32x32(
            &coefs,
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            &mut qcoeff_i16,
            &mut dqcoeff_i16,
            &dequant_pair,
            scan_slice,
        ),
        _ => vp9_quantize_b(
            &coefs,
            n,
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            &mut qcoeff_i16,
            &mut dqcoeff_i16,
            &dequant_pair,
            scan_slice,
        ),
    };
    let qcoeff_i32: Vec<i32> = qcoeff_i16.into_iter().map(i32::from).collect();
    let dqcoeff_i32: Vec<i32> = dqcoeff_i16.into_iter().map(i32::from).collect();
    (qcoeff_i32, dqcoeff_i32, eob as usize)
}

/// Inverse-DCT-add for inter (DctDct only). Mirrors the decoder's
/// `inverse_transform_add_dct` at
/// `rvp9-decoder::reconstruct_inter::inverse_transform_add_dct`. `eob`
/// is the number of decoded scan positions (caller has already filtered
/// `eob > 0`).
fn inter_inverse_dct_add(
    plane_tx: TxSize,
    dq_input: &[i32],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    use rvp9_core::transforms::inverse::{
        idct4x4_1_add, idct4x4_16_add, idct8x8_1_add, idct8x8_64_add, idct16x16_1_add,
        idct16x16_256_add, idct32x32_1_add, idct32x32_1024_add,
    };

    debug_assert!(eob > 0, "inter_inverse_dct_add called with eob == 0");

    match plane_tx {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = dq_input.try_into().expect("Tx4x4 takes 16 coefs");
            if eob > 1 {
                idct4x4_16_add(coefs, dst, stride);
            } else {
                idct4x4_1_add(coefs[0], dst, stride);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = dq_input.try_into().expect("Tx8x8 takes 64 coefs");
            if eob == 1 {
                idct8x8_1_add(coefs[0], dst, stride);
            } else {
                idct8x8_64_add(coefs, dst, stride);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = dq_input.try_into().expect("Tx16x16 takes 256 coefs");
            if eob == 1 {
                idct16x16_1_add(coefs[0], dst, stride);
            } else {
                idct16x16_256_add(coefs, dst, stride);
            }
        }
        TxSize::Tx32x32 => {
            let coefs: &[i32; 1024] = dq_input.try_into().expect("Tx32x32 takes 1024 coefs");
            if eob == 1 {
                idct32x32_1_add(coefs[0], dst, stride);
            } else {
                idct32x32_1024_add(coefs, dst, stride);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_new_round_trips() {
        let c = Config::new(64, 64, 50);
        assert_eq!(c.width, 64);
        assert_eq!(c.height, 64);
        assert_eq!(c.base_qindex, 50);
    }

    #[test]
    fn encoder_rejects_non_multiple_of_64_width() {
        let err = Encoder::new(Config::new(63, 64, 50)).unwrap_err();
        match err {
            EncodeError::InvalidConfig(_) => {}
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encoder_rejects_non_multiple_of_64_height() {
        let err = Encoder::new(Config::new(64, 65, 50)).unwrap_err();
        match err {
            EncodeError::InvalidConfig(_) => {}
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encoder_rejects_zero_qindex() {
        let err = Encoder::new(Config::new(64, 64, 0)).unwrap_err();
        match err {
            EncodeError::InvalidConfig(_) => {}
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encoder_accepts_canonical_config() {
        let enc = Encoder::new(Config::new(64, 64, 50)).expect("valid config");
        assert_eq!(enc.config().width, 64);
    }

    #[test]
    fn encode_64x64_keyframe_produces_non_empty_packet() {
        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_dc_skip_keyframe().unwrap();
        // The packet has three pieces (uncompressed header, compressed
        // header, tile data); each is at least a few bytes, total well
        // above 16 bytes.
        assert!(pkt.len() >= 16, "packet len = {} < 16", pkt.len());
        // The first 2 bits of the uncompressed header are the frame
        // marker (= 0b10). High bit must be set; second bit must be
        // clear.
        assert_eq!(pkt[0] & 0xc0, 0x80, "frame marker mismatch in pkt[0] = {:#04x}", pkt[0]);
    }

    #[test]
    fn encode_keyframe_parses_back_through_decoder_path() {
        // Exercises the writer end-to-end against the
        // `parse_uncompressed_header` + `parse_compressed_header`
        // pair in rvp9-core. (The Decoder-driven whole-loop test
        // lives in rvp9-decoder; see
        // `crates/rvp9-decoder/src/lib.rs::tests::
        // s4_10_round_trip_synthetic_64x64_keyframe_through_decoder`.)
        use rvp9_core::bitstream::{
            parse_compressed_header, parse_uncompressed_header, RefFrameSizes,
        };

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_dc_skip_keyframe().unwrap();

        let no_refs = RefFrameSizes::default();
        let (hdr, hdr_end) = parse_uncompressed_header(&pkt, &no_refs)
            .expect("uncompressed header parses");
        assert_eq!(hdr.kind, FrameKind::KeyFrame);
        assert_eq!(hdr.width, 64);
        assert_eq!(hdr.height, 64);

        let tail = hdr.tail.expect("keyframe must carry a tail");
        let fp = tail.first_partition_size as usize;
        let compressed = &pkt[hdr_end..hdr_end + fp];

        let opts = CompressedHeaderOptions {
            lossless: false,
            intra_only: true,
            interp_filter: InterpFilter::EightTap,
            compound_allowed: false,
            allow_high_precision_mv: false,
        };
        let mut fc = FrameContext::new_default();
        let res = parse_compressed_header(compressed, &mut fc, opts)
            .expect("compressed header parses");
        assert_eq!(res.tx_mode, TxMode::AllowTx32x32);
        assert_eq!(res.reference_mode, ReferenceMode::Single);
    }

    #[test]
    fn encode_128x64_keyframe_works() {
        // Two SBs horizontally, one vertically.
        let enc = Encoder::new(Config::new(128, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_dc_skip_keyframe().unwrap();
        assert!(pkt.len() >= 16);
    }

    #[test]
    fn encode_64x128_keyframe_works() {
        // One SB horizontally, two vertically.
        let enc = Encoder::new(Config::new(64, 128, 50)).unwrap();
        let pkt = enc.encode_synthetic_dc_skip_keyframe().unwrap();
        assert!(pkt.len() >= 16);
    }

    #[test]
    fn encode_two_frame_kf_inter_produces_superframe_packet() {
        use rvp9_core::bitstream::{Superframe, parse_superframe_index};

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc
            .encode_synthetic_two_frame_kf_inter()
            .expect("two-frame encode should succeed");
        // Superframe trailer parser must agree this is a 2-frame
        // packet.
        let index = parse_superframe_index(&pkt).expect("superframe trailer parses");
        let (sizes, count) = match index {
            Superframe::Multi { sizes, count, .. } => (sizes, count),
            Superframe::Single => panic!("expected 2-frame superframe, got Single"),
        };
        assert_eq!(count, 2, "expected 2 frame entries, got {count}");

        // Each per-frame slice must parse as a VP9 frame (frame
        // marker 0b10 in the top 2 bits of the first byte).
        let kf_size = sizes[0] as usize;
        let inter_size = sizes[1] as usize;
        let kf_slice = &pkt[..kf_size];
        let inter_slice = &pkt[kf_size..kf_size + inter_size];
        assert_eq!(kf_slice[0] & 0xc0, 0x80, "kf frame marker");
        assert_eq!(inter_slice[0] & 0xc0, 0x80, "inter frame marker");

        // Parse both frame headers to confirm kind discriminant.
        use rvp9_core::bitstream::{RefFrameSizes, parse_uncompressed_header};
        let no_refs = RefFrameSizes::default();
        let (kf_hdr, _) = parse_uncompressed_header(kf_slice, &no_refs)
            .expect("kf header parses");
        assert_eq!(kf_hdr.kind, FrameKind::KeyFrame);

        // The inter frame requires the keyframe's pool-slot-0 size in
        // `ref_sizes`. The keyframe was 64×64.
        let mut ref_sizes = RefFrameSizes::default();
        ref_sizes[0] = Some((64, 64));
        let (inter_hdr, _) = parse_uncompressed_header(inter_slice, &ref_sizes)
            .expect("inter header parses");
        assert!(matches!(inter_hdr.kind, FrameKind::InterFrame { .. }));
    }

    /// End-to-end: encoder emits a 2-frame superframe packet; the
    /// real decoder consumes it and recovers BOTH frames cleanly.
    /// Compared to the rvp9-decoder-internal
    /// `s4_16d_round_trip_inter_after_keyframe_through_decoder`
    /// test (which builds the bitstream inline via rvp9-core writers),
    /// this exercises the encoder's PUBLIC API.
    #[test]
    fn encode_two_frame_kf_inter_decodes_through_real_decoder() {
        use rvp9_core::block::BlockSize;
        use rvp9_decoder::{Decoder, ParsedFrame};

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_two_frame_kf_inter().unwrap();

        let mut dec = Decoder::new();
        let parsed = dec
            .parse_packet(&pkt)
            .expect("parse_packet must accept the 2-frame superframe");
        assert_eq!(parsed.count(), 2, "expected 2 parsed frames");
        let frames: Vec<ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        assert_eq!(frames[0].header.kind, FrameKind::KeyFrame);
        assert!(matches!(frames[1].header.kind, FrameKind::InterFrame { .. }));

        // Decode both frames in order.
        dec.decode_frame(&frames[0])
            .expect("decode_frame must accept keyframe");
        dec.decode_frame(&frames[1])
            .expect("decode_frame must accept inter frame");

        // The inter frame's recovered mi_grid should be a single 64×64
        // `BlockMode::Inter` LASTFRAME ZEROMV skip=true cell splatted
        // across all 64 mi-cells.
        let (mi_rows, mi_cols) = dec.last_frame_mi_dims();
        assert_eq!((mi_rows, mi_cols), (8, 8));
        let grid = dec.last_frame_mi_grid();
        for cell_got in grid.iter() {
            let cell = cell_got.expect("every mi-cell must be populated");
            assert_eq!(cell.block_size, BlockSize::Block64x64);
            assert!(cell.skip);
            match cell.mode {
                BlockMode::Inter {
                    mode,
                    ref_frames,
                    interp_filter,
                    ..
                } => {
                    assert_eq!(mode, InterMode::ZeroMv);
                    assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                    assert_eq!(interp_filter, InterpFilter::EightTap);
                }
                BlockMode::Intra { .. } => panic!("expected inter mode"),
            }
        }
    }

    /// End-to-end test for the new public `encode_synthetic_two_frame_
    /// kf_inter_non_skip` API. Decodes through the real `Decoder` and
    /// asserts every cell of the recovered inter mi_grid comes back as
    /// `skip = false`. The encoder's residual-coef emission path
    /// (write_inter_block_residual → write_block_tokens → encode_coefs)
    /// fires for every plane × every TX block (6 total: 4 Y Tx32x32 +
    /// 1 U + 1 V).
    ///
    /// Public-API counterpart to the rvp9-decoder crate's S4.16k
    /// integration test (validates the same wire-format capability
    /// is reachable from the encoder crate's external surface).
    #[test]
    fn encode_two_frame_kf_inter_non_skip_decodes_through_real_decoder() {
        use rvp9_core::block::BlockSize;
        use rvp9_decoder::{Decoder, ParsedFrame};

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_two_frame_kf_inter_non_skip().unwrap();

        let mut dec = Decoder::new();
        let parsed = dec
            .parse_packet(&pkt)
            .expect("parse_packet must accept the non-skip 2-frame superframe");
        assert_eq!(parsed.count(), 2);
        let frames: Vec<ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        assert_eq!(frames[0].header.kind, FrameKind::KeyFrame);
        assert!(matches!(frames[1].header.kind, FrameKind::InterFrame { .. }));

        // Decode both frames; the inter frame must decode cleanly with
        // skip=false driving the coef-decode + IT-add pipeline.
        dec.decode_frame(&frames[0]).expect("decode kf");
        dec.decode_frame(&frames[1])
            .expect("decode_frame must accept non-skip inter with all-zero coefs");

        // Every cell of the 64×64 SB came back as Block64x64 inter
        // LASTFRAME ZEROMV EightTap with skip=FALSE.
        let grid = dec.last_frame_mi_grid();
        for cell_got in grid.iter() {
            let cell = cell_got.expect("every mi-cell must be populated");
            assert_eq!(cell.block_size, BlockSize::Block64x64);
            assert!(!cell.skip, "skip must round-trip as FALSE");
            match cell.mode {
                BlockMode::Inter {
                    mode,
                    ref_frames,
                    interp_filter,
                    ..
                } => {
                    assert_eq!(mode, InterMode::ZeroMv);
                    assert_eq!(ref_frames, [RefFrame::LastFrame, RefFrame::NoRefFrame]);
                    assert_eq!(interp_filter, InterpFilter::EightTap);
                }
                BlockMode::Intra { .. } => panic!("expected inter mode"),
            }
        }
    }

    /// F5.1 — end-to-end smoke test for the encoder simulator + the
    /// keyframe writer's `skip = false` + `IntraCoefSource` plumbing.
    ///
    /// Setup:
    /// - 64×64 frame (one SB), `base_qindex = 80`.
    /// - Y input: gradient `128 + r/2 + c/2`. U/V input: flat `128`.
    /// - Encoder simulator runs predict → residual → fdct → quant →
    ///   idct-add per TX block, producing real qcoeffs AND the
    ///   reconstructed pixels the decoder must produce bit-exactly.
    /// - Encoder emits the packet via
    ///   `encode_synthetic_dc_intra_with_residuals`.
    /// - Decoder parses + decodes the keyframe.
    ///
    /// Oracle: decoded Y/U/V must MATCH the encoder simulator's
    /// `recon_yuv_out` bit-exactly (every pixel, every plane).
    ///
    /// We separately compute the max `(decoded - input)` per-pixel
    /// error and assert it's within `dequant`-scaled quantization
    /// noise bound (a sanity check that the qindex isn't producing
    /// degenerate quantization).
    #[test]
    fn encode_synthetic_dc_intra_with_residuals_decodes_bit_exact() {
        use rvp9_decoder::{Decoder, ParsedFrame};

        let cfg = Config::new(64, 64, 80);
        let enc = Encoder::new(cfg).unwrap();

        // Synthesize a small gradient Y + flat UV. The gradient is
        // gentle enough that the residual stays small after DC_PRED,
        // exercising mostly low-magnitude qcoeffs.
        let w = 64;
        let h = 64;
        let mut y_input = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let v = 128i32 + (r as i32) / 2 + (c as i32) / 2;
                y_input[r * w + c] = v.clamp(0, 255) as u8;
            }
        }
        let u_input = vec![128u8; (w / 2) * (h / 2)];
        let v_input = vec![128u8; (w / 2) * (h / 2)];

        let mut y_recon = vec![0u8; w * h];
        let mut u_recon = vec![0u8; (w / 2) * (h / 2)];
        let mut v_recon = vec![0u8; (w / 2) * (h / 2)];

        let pkt = enc
            .encode_synthetic_dc_intra_with_residuals(
                (&y_input, &u_input, &v_input),
                (&mut y_recon, &mut u_recon, &mut v_recon),
            )
            .expect("F5.1 encode must succeed");
        assert!(pkt.len() >= 32, "packet too small: {} bytes", pkt.len());

        // Decode through the real Decoder.
        let mut dec = Decoder::new();
        let parsed = dec.parse_packet(&pkt).expect("parse_packet");
        assert_eq!(parsed.count(), 1);
        let frames: Vec<ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        assert_eq!(frames[0].header.kind, FrameKind::KeyFrame);

        let frame_buf = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the F5.1 keyframe");

        // Bit-exact oracle: decoded pixels == encoder's recon.
        let y_dec = frame_buf.y_plane();
        let u_dec = frame_buf.u_plane();
        let v_dec = frame_buf.v_plane();
        let y_dec_stride = y_dec.stride();
        let u_dec_stride = u_dec.stride();
        let v_dec_stride = v_dec.stride();
        let y_dec_data = y_dec.data();
        let u_dec_data = u_dec.data();
        let v_dec_data = v_dec.data();

        let mut max_err_input_y: i32 = 0;

        for r in 0..h {
            for c in 0..w {
                let want = y_recon[r * w + c];
                let got = y_dec_data[r * y_dec_stride + c];
                assert_eq!(
                    got, want,
                    "Y mismatch at (r={r},c={c}): decoded {got}, recon {want}"
                );
                let err = (got as i32 - y_input[r * w + c] as i32).abs();
                if err > max_err_input_y {
                    max_err_input_y = err;
                }
            }
        }
        for r in 0..h / 2 {
            for c in 0..w / 2 {
                let want = u_recon[r * (w / 2) + c];
                let got = u_dec_data[r * u_dec_stride + c];
                assert_eq!(got, want, "U mismatch at (r={r},c={c})");
                let want = v_recon[r * (w / 2) + c];
                let got = v_dec_data[r * v_dec_stride + c];
                assert_eq!(got, want, "V mismatch at (r={r},c={c})");
            }
        }

        // Sanity: the quant-noise envelope. At qindex=80, the dequant
        // (= dc_quant / ac_quant) is ~70 for Y. Reconstructed pixels
        // should be within ~1 dequant_ac magnitude of the input on
        // smooth content — we use a generous 32-step envelope to
        // avoid flaky regressions if quant params drift.
        // (Actual maxes observed on this gradient land around 1-3,
        // well within the bound.)
        assert!(
            max_err_input_y < 32,
            "max |decoded - input| (Y) = {max_err_input_y} exceeded loose bound"
        );
    }

    // -----------------------------------------------------------------
    // AQ.0 — segmentation-map round-trip via the real Decoder
    //
    // The encoder's segmentation writer (`write_segment_id` +
    // `SegmentationWriteParams`) emits per-leaf `segment_id` values
    // for the `update_map = true, temporal_update = false` case
    // (libvpx cyclic refresh / AQ mode 3). These integration tests
    // encode a keyframe whose 64×64 SB carries a non-zero
    // `segment_id`, decode it through the real `rvp9_decoder::Decoder`,
    // and assert the recovered `last_frame_mi_grid` carries the same
    // segment_id at every covered cell.
    // -----------------------------------------------------------------

    /// AQ.0 test-only helper: build a single-SB keyframe packet whose
    /// SB carries a uniform `target_segment_id` on every cell, with
    /// `enabled = true, update_map = true, temporal_update = false`
    /// and uniform-128 tree_probs. Matches the libvpx cyclic-refresh
    /// segmentation header but with `update_data = false` (no
    /// per-segment Q deltas — AQ.3 will wire those).
    fn build_kf_with_segments(
        enc: &Encoder,
        target_segment_id: u8,
    ) -> Result<Vec<u8>, EncodeError> {
        use rvp9_core::bitstream::{
            BitDepth, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
            CompressedHeaderResult, FrameKind, HeaderTail, LoopFilterParams, Profile,
            QuantizationParams, RawBitWriter, SEG_TREE_PROBS,
            SegmentationWriteParams as ModesSegParams, UncompressedHeader,
            write_compressed_header, write_tile_intra_keyframe, write_uncompressed_header,
        };

        let mi_rows = enc.config.height / 8;
        let mi_cols = enc.config.width / 8;
        let cell = ModeInfo {
            block_size: BlockSize::Block64x64,
            tx_size: TxSize::Tx32x32,
            skip: true,
            segment_id: target_segment_id,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let mi_grid: Vec<Option<ModeInfo>> = vec![Some(cell); (mi_rows * mi_cols) as usize];

        // Compressed header (default-FC, tx_mode = AllowTx32x32).
        let mut pre_fc = FrameContext::new_default();
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

        // Tile data: segmentation enabled with update_map=true,
        // tree_probs = uniform 128.
        let tree_probs = [128u8; SEG_TREE_PROBS];
        let tile_bounds = ModesTileBounds {
            mi_row_start: 0,
            mi_row_end: mi_rows as usize,
            mi_col_start: 0,
            mi_col_end: mi_cols as usize,
        };
        let mut tile_buf = vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 8 + 1024];
        let tile_len = {
            let mut bw = BoolWriter::start_encode(&mut tile_buf);
            let tile_mi_cols =
                (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
            let coef_source = AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                Some(ModesSegParams { tree_probs: &tree_probs }),
                &mut coef_ctx,
                &coef_source,
            )?;
            bw.stop_encode()
                .map_err(|_| EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge))?
        };
        tile_buf.truncate(tile_len);

        // Uncompressed header — segmentation enabled, update_map=true,
        // temporal_update=false, tree_probs[i]=128. `update_data =
        // false` (AQ.3 will wire feature_data for per-segment Q deltas).
        let tail = HeaderTail {
            refresh_frame_context: true,
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
                base_qindex: enc.config.base_qindex,
                y_dc_delta_q: 0,
                uv_dc_delta_q: 0,
                uv_ac_delta_q: 0,
            },
            segmentation: SegmentationParams {
                enabled: true,
                update_map: true,
                temporal_update: false,
                tree_probs,
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
            kind: FrameKind::KeyFrame,
            profile: Profile::Profile0,
            show_frame: true,
            error_resilient: false,
            bit_depth: BitDepth::Eight,
            color_space: ColorSpace::Bt601,
            color_range: ColorRange::Studio,
            chroma: ChromaFormat::Yuv420,
            width: enc.config.width,
            height: enc.config.height,
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

    /// AQ.0 — encode a 64×64 keyframe whose single SB carries
    /// `segment_id = 3` everywhere, then decode through the real
    /// `Decoder` and assert every cell of the recovered mi_grid has
    /// `segment_id = 3`. This is the key oracle: the
    /// encoder→header→bitstream→decoder loop carries per-leaf
    /// segment IDs end-to-end.
    #[test]
    fn aq0_encode_kf_with_segment_id_3_decodes_back_through_real_decoder() {
        use rvp9_decoder::{Decoder, ParsedFrame};

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = build_kf_with_segments(&enc, 3).expect("build_kf_with_segments");

        let mut dec = Decoder::new();
        let parsed = dec
            .parse_packet(&pkt)
            .expect("parse_packet must accept the seg-enabled keyframe");
        assert_eq!(parsed.count(), 1);
        let frames: Vec<ParsedFrame> = parsed
            .frames()
            .iter()
            .map(|pf| ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        assert_eq!(frames[0].header.kind, FrameKind::KeyFrame);

        let _ = dec
            .decode_frame(&frames[0])
            .expect("decode_frame must accept the seg-enabled keyframe");

        let grid = dec.last_frame_mi_grid();
        assert!(!grid.is_empty(), "decoded mi_grid must be populated");
        for (i, cell_got) in grid.iter().enumerate() {
            let cell = cell_got.unwrap_or_else(|| panic!("mi-cell [{i}] must be populated"));
            assert_eq!(
                cell.segment_id, 3,
                "cell [{i}] segment_id should round-trip as 3"
            );
        }
    }

    /// AQ.0 — round-trip every legal segment_id (0..=7) on a
    /// single-SB keyframe through the real Decoder. Verifies the
    /// full `SEGMENT_TREE` walk is reachable from public encoder
    /// state.
    #[test]
    fn aq0_encode_kf_with_each_segment_id_decodes_back_through_real_decoder() {
        use rvp9_core::bitstream::MAX_SEGMENTS;
        use rvp9_decoder::{Decoder, ParsedFrame};

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        for target in 0u8..(MAX_SEGMENTS as u8) {
            let pkt = build_kf_with_segments(&enc, target)
                .unwrap_or_else(|e| panic!("build_kf_with_segments({target}) failed: {e:?}"));

            let mut dec = Decoder::new();
            let parsed = dec
                .parse_packet(&pkt)
                .unwrap_or_else(|e| panic!("parse_packet seg={target}: {e:?}"));
            assert_eq!(parsed.count(), 1, "seg={target}");
            let frames: Vec<ParsedFrame> = parsed
                .frames()
                .iter()
                .map(|pf| ParsedFrame {
                    header: pf.header.clone(),
                    header_end_byte: pf.header_end_byte,
                    full_frame_data: pf.full_frame_data,
                })
                .collect();
            let _ = dec
                .decode_frame(&frames[0])
                .unwrap_or_else(|e| panic!("decode_frame seg={target}: {e:?}"));
            let grid = dec.last_frame_mi_grid();
            for (i, cell_got) in grid.iter().enumerate() {
                let cell = cell_got
                    .unwrap_or_else(|| panic!("seg={target} cell [{i}] must be populated"));
                assert_eq!(
                    cell.segment_id, target,
                    "seg={target} cell [{i}]: recovered segment_id mismatch"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // AQ.1 — per-segment Q-delta drives the forward quant init.
    //
    // The four scalar tests below pin down the
    // `effective_qindex_for_segment` helper's table semantics + clamp
    // behaviour. Test #5 is the key oracle: it runs the simulator on
    // two equally-sized leaves whose only difference is `segment_id`
    // and asserts that the simulator's per-TX dq coefficients differ
    // in the direction implied by the Q-delta. Without that final
    // assertion, the scalar tests are necessary-but-not-sufficient
    // — they prove the helper is correct in isolation but say nothing
    // about whether the simulator actually consults it on the hot path.
    // -----------------------------------------------------------------

    #[test]
    fn effective_qindex_for_segment_no_delta_returns_base() {
        let seg_q_deltas: [Option<i16>; 8] = [None; 8];
        assert_eq!(effective_qindex_for_segment(100, 0, &seg_q_deltas), 100);
        // Non-zero segment with no delta entry also returns base.
        assert_eq!(effective_qindex_for_segment(100, 4, &seg_q_deltas), 100);
    }

    #[test]
    fn effective_qindex_for_segment_applies_negative_delta() {
        let mut seg_q_deltas: [Option<i16>; 8] = [None; 8];
        seg_q_deltas[1] = Some(-10);
        assert_eq!(effective_qindex_for_segment(100, 1, &seg_q_deltas), 90);
        // Other segments still take the base path.
        assert_eq!(effective_qindex_for_segment(100, 0, &seg_q_deltas), 100);
        assert_eq!(effective_qindex_for_segment(100, 2, &seg_q_deltas), 100);
    }

    #[test]
    fn effective_qindex_for_segment_clamps_high() {
        let mut seg_q_deltas: [Option<i16>; 8] = [None; 8];
        seg_q_deltas[1] = Some(20);
        assert_eq!(effective_qindex_for_segment(250, 1, &seg_q_deltas), 255);
    }

    #[test]
    fn effective_qindex_for_segment_clamps_low() {
        let mut seg_q_deltas: [Option<i16>; 8] = [None; 8];
        seg_q_deltas[1] = Some(-20);
        assert_eq!(effective_qindex_for_segment(5, 1, &seg_q_deltas), 0);
    }

    /// AQ.1 KEY ORACLE — run the simulator on two equally-shaped 8×8
    /// leaves placed at two different positions in a single 64×64 SB,
    /// with the SAME input pixel values. Leaf A carries `segment_id =
    /// 0` (no Q-delta); leaf B carries `segment_id = 1` with
    /// `seg_q_deltas[1] = -20` (= finer quant, smaller dequant step).
    ///
    /// The simulator's per-TX `dequant` value scales with the
    /// effective qindex (`ac_quant(qindex, 0)` for AC bins, monotone
    /// non-decreasing in qindex per `rvp9_core::quant::tests::
    /// luts_are_monotonic_nondecreasing`). With a -20 delta on top of
    /// `base_qindex = 100` the per-segment-1 qindex is 80, so AC
    /// dequant for seg 1 should be strictly LESS than for seg 0.
    ///
    /// We don't compare full qcoeff vectors (those depend on edge
    /// neighbour state which differs by leaf position); instead we
    /// reach into the simulator's internals by computing the
    /// `PlaneQuantizers` for both qindices and asserting they differ
    /// in the expected direction. The simulator's per-leaf
    /// `init_plane_quantizers` call uses the same formula, so this
    /// proves the per-segment Q-delta math reaches the forward-quant
    /// init.
    ///
    /// As the integration oracle: we also check that the simulator's
    /// `qcoeff` outputs for leaf A vs leaf B genuinely differ when fed
    /// identical input pixels in identical neighbour contexts (i.e.
    /// the dq path is wired live, not a no-op).
    #[test]
    fn encode_kf_with_per_segment_q_delta_quantizes_differently_per_segment() {
        use rvp9_core::quant::forward::{
            QuantizerDeltas, init_plane_quantizers,
        };

        // (a) Helper-level proof: two qindices yield different
        // `PlaneQuantizers` (different `dequant` AC values).
        let base_qindex: u8 = 100;
        let mut seg_q_deltas: [Option<i16>; 8] = [None; 8];
        seg_q_deltas[1] = Some(-20);
        let q0 = effective_qindex_for_segment(base_qindex, 0, &seg_q_deltas);
        let q1 = effective_qindex_for_segment(base_qindex, 1, &seg_q_deltas);
        assert_eq!(q0, 100, "seg 0 qindex must equal base");
        assert_eq!(q1, 80, "seg 1 qindex must equal base + (-20)");
        let pq0 = init_plane_quantizers(i32::from(q0), QuantizerDeltas::Y);
        let pq1 = init_plane_quantizers(i32::from(q1), QuantizerDeltas::Y);
        assert!(
            pq1.dequant[1] < pq0.dequant[1],
            "seg 1 AC dequant (q={q1}) must be < seg 0 AC dequant (q={q0}): \
             got pq1.dequant[1] = {}, pq0.dequant[1] = {}",
            pq1.dequant[1],
            pq0.dequant[1],
        );

        // (b) Simulator-level proof: feed the simulator a SB whose two
        // 8×8 leaves carry identical input pixels but different
        // segment_ids. The simulator must produce different qcoeff
        // vectors for the two leaves — that's the "AQ.1 is wired"
        // signal. (Leaf neighbour state differs by position, so we
        // don't pin exact bit patterns; we only check that the
        // qcoeffs are non-equal.)
        //
        // Layout: single 64×64 SB, all 64 cells are 8×8 leaves. Cell
        // at (mi_row_in_sb=0, mi_col_in_sb=0) → seg 0. Cell at
        // (mi_row_in_sb=4, mi_col_in_sb=0) → seg 1. Both leaves
        // receive the same input pixel pattern (we tile-paint the
        // same 8×8 block over the whole 64×64 frame).
        let w: u32 = 64;
        let h: u32 = 64;
        let mut leaves = Vec::with_capacity(64);
        for r in 0..8usize {
            for c in 0..8usize {
                let seg = if r == 0 && c == 0 {
                    0
                } else if r == 4 && c == 0 {
                    1
                } else {
                    0
                };
                leaves.push(PartitionLeafDesc {
                    mi_row_in_sb: r,
                    mi_col_in_sb: c,
                    leaf_bsize: BlockSize::Block8x8,
                    predicted_y_mode: IntraMode::Dc,
                    segment_id: seg,
                });
            }
        }
        let sb_partitions = vec![leaves];

        // Tile-painted Y: same 8×8 block repeated across the frame so
        // the two seg-0 / seg-1 leaves see input data of the same
        // visual shape. (Note: their neighbour-recon contexts still
        // differ — but the residual is identical only at the top-left
        // 8×8 of the SB where there are no neighbours; for other
        // leaves the neighbour context is partial. We rely on the
        // leaves' qcoeff outputs simply being NOT equal to prove the
        // dq path is active per-leaf.)
        let y_input: Vec<u8> = (0..(w * h) as usize)
            .map(|i| {
                let r = i / (w as usize);
                let c = i % (w as usize);
                let r8 = r % 8;
                let c8 = c % 8;
                (128i32 + (r8 as i32) * 5 + (c8 as i32) * 3).clamp(0, 255) as u8
            })
            .collect();
        let u_input = vec![128u8; ((w / 2) * (h / 2)) as usize];
        let v_input = vec![128u8; ((w / 2) * (h / 2)) as usize];

        let mut y_recon = vec![0u8; (w * h) as usize];
        let mut u_recon = vec![0u8; ((w / 2) * (h / 2)) as usize];
        let mut v_recon = vec![0u8; ((w / 2) * (h / 2)) as usize];

        // Two simulator runs: one with `seg_q_deltas` empty (Stage 4a
        // baseline) and one with a -20 delta for seg 1. The seg 0
        // leaf at (0, 0) must produce identical qcoeffs across both
        // runs (its effective qindex doesn't change). The seg 1 leaf
        // at (mi_row=4, mi_col=0) must produce DIFFERENT qcoeffs
        // between the two runs (its effective qindex shifts from
        // 100 → 80).
        let mut y_recon_base = y_recon.clone();
        let mut u_recon_base = u_recon.clone();
        let mut v_recon_base = v_recon.clone();
        let baseline_deltas: [Option<i16>; 8] = [None; 8];
        let base_coefs = simulate_dc_intra_encode_partitions(
            w,
            h,
            base_qindex,
            &baseline_deltas,
            (&y_input, &u_input, &v_input),
            (&mut y_recon_base, &mut u_recon_base, &mut v_recon_base),
            &sb_partitions,
        );
        let aq_coefs = simulate_dc_intra_encode_partitions(
            w,
            h,
            base_qindex,
            &seg_q_deltas,
            (&y_input, &u_input, &v_input),
            (&mut y_recon, &mut u_recon, &mut v_recon),
            &sb_partitions,
        );

        // Pull the Y qcoeff vector for the seg-1 leaf
        // (mi_row=4, mi_col=0; plane 0; tx_x4_local=0, tx_y4_local=0).
        let key_seg1 = (4usize, 0usize, 0usize, 0usize, 0usize);
        let (base_qc_seg1, base_eob_seg1) =
            base_coefs.blocks.get(&key_seg1).expect("seg1 leaf coefs missing in baseline");
        let (aq_qc_seg1, aq_eob_seg1) =
            aq_coefs.blocks.get(&key_seg1).expect("seg1 leaf coefs missing in AQ run");
        // The two runs must produce DIFFERENT qcoeffs for the seg 1
        // leaf — that's the smoking gun that per-segment Q-delta
        // reaches the forward-quant init.
        assert!(
            base_qc_seg1 != aq_qc_seg1 || base_eob_seg1 != aq_eob_seg1,
            "AQ.1 KEY ORACLE: seg 1 leaf qcoeffs identical between baseline \
             (no delta) and AQ (delta=-20) runs — per-segment Q math is NOT \
             reaching the forward-quant init. \
             base_qc = {base_qc_seg1:?}, aq_qc = {aq_qc_seg1:?}, \
             base_eob = {base_eob_seg1}, aq_eob = {aq_eob_seg1}",
        );

        // The seg-0 leaf at (0, 0) must produce IDENTICAL qcoeffs
        // across the two runs — its qindex doesn't change.
        let key_seg0 = (0usize, 0usize, 0usize, 0usize, 0usize);
        let (base_qc_seg0, base_eob_seg0) =
            base_coefs.blocks.get(&key_seg0).expect("seg0 leaf coefs missing in baseline");
        let (aq_qc_seg0, aq_eob_seg0) =
            aq_coefs.blocks.get(&key_seg0).expect("seg0 leaf coefs missing in AQ run");
        assert_eq!(
            base_qc_seg0, aq_qc_seg0,
            "seg 0 leaf qcoeffs must be IDENTICAL across the two runs \
             (no Q-delta applies to segment 0)",
        );
        assert_eq!(
            base_eob_seg0, aq_eob_seg0,
            "seg 0 leaf eob must be IDENTICAL across the two runs",
        );
    }
}
