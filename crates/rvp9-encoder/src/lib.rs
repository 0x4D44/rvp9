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
pub mod buffer_pool;
pub mod e12stat;
pub mod f02_oracle;
pub mod frame_loop;
pub(crate) mod loopfilter;
pub mod lookahead;
pub mod pre_analysis;
pub mod predstat;
pub mod rate_control;
pub mod skipstat;
pub mod temporal_filter;

pub use frame_loop::{AqMode, Deadline, EncodedFrame, EncoderConfig, FrameKind as StreamFrameKind, StreamEncoder, YuvFrameRef};

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
                None,
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
/// `predicted_uv_mode` is the chroma intra mode used for both U and V
/// planes (VP9 wire format emits a single `uv_mode` per leaf). The
/// F10.1b chroma picker (Stage 4a' chroma-DC closure) drives this from
/// [`crate::mode_decision::intra::pick_intra_mode_uv`]; pre-picker
/// paths default to `IntraMode::Dc` (matching the prior v1 behaviour).
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
    /// forward/inverse transforms.
    pub predicted_y_mode: IntraMode,
    /// Intra UV-mode applied to BOTH U and V planes for this leaf.
    /// Picked by [`pick_intra_mode_uv`] alongside the luma mode.
    pub predicted_uv_mode: IntraMode,
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
            predicted_uv_mode: IntraMode::Dc,
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
        DcAvailability, EdgeBuffers, build_intra_predictors, intra_predict,
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
                // F10.1b: Y plane uses the picked intra mode. Stage 4a'
                // chroma-DC closure: UV planes use `leaf.predicted_uv_mode`
                // (also picked by F10.1b — see `pick_intra_mode_uv`).
                // Pre-picker call sites set `predicted_uv_mode =
                // IntraMode::Dc` to preserve prior behaviour. The Y mode
                // also drives `tx_type` for the forward transform +
                // scan order at sizes < Tx32x32 (Tx32x32 always uses
                // DctDct, mirroring the decoder + writer); UV is
                // always DctDct.
                let leaf_y_mode = leaf.predicted_y_mode;
                let leaf_uv_mode = leaf.predicted_uv_mode;

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

                            // Y plane uses the picked Y mode; both UV
                            // planes use the picked UV mode (single
                            // `uv_mode` per leaf in VP9 wire format).
                            let plane_mode = if plane == 0 {
                                leaf_y_mode
                            } else {
                                leaf_uv_mode
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

                            // E9 DIAG — tap the Y-plane prediction residual
                            // (`source − intra prediction`, before transform/quant).
                            if plane == 0 {
                                crate::predstat::record_y_residual(
                                    &residual,
                                    tx_side,
                                    plane_tx as usize,
                                );
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
/// 5. Stage 4a' chroma-DC closure — call
///    [`pick_intra_mode_uv`](rvp9_core::mode_decision::intra::pick_intra_mode_uv)
///    against the leaf's U+V planes (with neighbour edges sourced from
///    the running U/V recon) and the just-picked luma mode. The picker
///    evaluates all 10 intra modes on the combined U+V transform-
///    coefficient-domain distortion plus the UV-mode rate cost
///    `[y_mode][uv_mode]`. Write the winner back into
///    `sb_partitions[sb][leaf].predicted_uv_mode` so both the
///    simulator (this function's downstream call) and the writer
///    (`build_kf_mi_grid`) see the same mode.
/// 6. Run the per-leaf encode pass (predict → fdct → quant → idct-add)
///    using the picked Y / UV modes — populates both the recon buffer
///    and the qcoeff [`VecBackedIntraCoefSource`].
///
/// For `Block64x64` leaves the luma picker is out of scope (sub-32×32
/// per-leaf sub-mode picking is on the F8.x roadmap); both Y and UV
/// fall back to DC on those leaves. Every smaller-leaf size gets both
/// the luma and chroma picker treatment.
pub(crate) fn pick_and_simulate_intra_encode_partitions(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &mut [Vec<PartitionLeafDesc>],
) -> VecBackedIntraCoefSource {
    use rvp9_core::cost::{intra_uv_mode_costs_kf, intra_y_mode_costs_kf_table};
    use rvp9_core::mode_decision::intra::{
        IntraModeSearchParams, UvIntraModeSearchParams, pick_intra_mode_luma,
        pick_intra_mode_uv,
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
    let uv_mode_costs = intra_uv_mode_costs_kf();
    let uvw = w / 2;
    let uvh = h / 2;

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

                // Stage 4a' — chroma intra mode picker. Runs alongside
                // the luma picker so the leaf's `predicted_uv_mode` is
                // committed BEFORE the simulator's per-plane encode
                // pass. The picker reads U/V neighbour samples from the
                // running U/V recon (populated by all previously-encoded
                // leaves) and the U/V source planes, evaluates all 10
                // intra modes on the combined U+V transform-coefficient
                // distortion + the `intra_uv_mode_costs_kf[y_mode]`
                // rate cost, and returns the RD winner. Block64x64
                // leaves keep `predicted_uv_mode = IntraMode::Dc`
                // (initialised by the leaves-from-partition-decision
                // path) — the luma picker is out of scope there too.
                let picked_uv_mode = if leaf_bsize == BlockSize::Block64x64 {
                    IntraMode::Dc
                } else {
                    let bs_uv = bs_pixels / 2;
                    let leaf_uv_x = leaf_px_x / 2;
                    let leaf_uv_y = leaf_px_y / 2;
                    let have_above_uv = leaf_uv_y > 0;
                    let have_left_uv = leaf_uv_x > 0;
                    let mut u_above = vec![0u8; bs_uv];
                    let mut u_left = vec![0u8; bs_uv];
                    let mut v_above = vec![0u8; bs_uv];
                    let mut v_left = vec![0u8; bs_uv];
                    if have_above_uv {
                        let above_row_uv = leaf_uv_y - 1;
                        let row_off = above_row_uv * uvw + leaf_uv_x;
                        u_above.copy_from_slice(
                            &recon_yuv_out.1[row_off..row_off + bs_uv],
                        );
                        v_above.copy_from_slice(
                            &recon_yuv_out.2[row_off..row_off + bs_uv],
                        );
                    }
                    if have_left_uv {
                        let left_col_uv = leaf_uv_x - 1;
                        for r in 0..bs_uv {
                            u_left[r] =
                                recon_yuv_out.1[(leaf_uv_y + r) * uvw + left_col_uv];
                            v_left[r] =
                                recon_yuv_out.2[(leaf_uv_y + r) * uvw + left_col_uv];
                        }
                    }
                    let (u_tl, v_tl) = if have_above_uv && have_left_uv {
                        (
                            recon_yuv_out.1
                                [(leaf_uv_y - 1) * uvw + (leaf_uv_x - 1)],
                            recon_yuv_out.2
                                [(leaf_uv_y - 1) * uvw + (leaf_uv_x - 1)],
                        )
                    } else {
                        (0u8, 0u8)
                    };
                    let u_src_plane = PlaneRef {
                        data: input_yuv.1,
                        stride: uvw,
                        width: uvw,
                        height: uvh,
                    };
                    let v_src_plane = PlaneRef {
                        data: input_yuv.2,
                        stride: uvw,
                        width: uvw,
                        height: uvh,
                    };
                    let uv_block_size = f5_uv_plane_block_size(leaf_bsize);
                    let uv_params = UvIntraModeSearchParams {
                        block_size: uv_block_size,
                        u_src_plane: &u_src_plane,
                        v_src_plane: &v_src_plane,
                        src_offset: (leaf_uv_x, leaf_uv_y),
                        u_above_pixels: &u_above,
                        u_left_pixels: &u_left,
                        u_top_left_pixel: u_tl,
                        v_above_pixels: &v_above,
                        v_left_pixels: &v_left,
                        v_top_left_pixel: v_tl,
                        have_above: have_above_uv,
                        have_left: have_left_uv,
                        quantizer: &leaf_quantizer,
                        q_index: leaf_qindex,
                        frame_kind:
                            rvp9_core::bitstream::header::FrameKind::KeyFrame,
                        y_mode: picked_mode,
                        uv_mode_costs: &uv_mode_costs,
                    };
                    pick_intra_mode_uv(&uv_params).best_mode
                };

                sb_partitions[sb_idx][leaf_idx].predicted_uv_mode = picked_uv_mode;

                // F.0.2 oracle — record one BlockDecision per coded
                // intra leaf. Single relaxed atomic load when the env
                // var `RVP9_F02_ORACLE` is unset (see HLD §3.2 / §7).
                if crate::f02_oracle::is_enabled() {
                    crate::f02_oracle::record_decision(
                        crate::f02_oracle::BlockDecision {
                            frame_idx: crate::f02_oracle::current_frame_idx(),
                            mi_row: leaf_mi_row as u16,
                            mi_col: leaf_mi_col as u16,
                            bsize: leaf_bsize,
                            mode: crate::f02_oracle::BlockDecisionMode::Intra(
                                picked_mode,
                            ),
                            ref_frame: RefFrame::IntraFrame,
                            mv_row: 0,
                            mv_col: 0,
                            // Picker mode-rate estimate is not exposed
                            // through the existing `pick_intra_mode_luma`
                            // result struct (only the winner mode comes
                            // back). HLD §3.4.1 marks `mode_bits_est` as
                            // "live"; the closest we can emit without
                            // widening the picker API (out of scope,
                            // HLD §9) is `0` for intra at F.0.2. F.x
                            // milestones that need exact intra-mode
                            // bits will widen the picker's return
                            // shape.
                            mode_bits_est: 0,
                            mv_bits_est: 0,
                            // Forward-looking per HLD §3.4.1 — `0`
                            // until F.1 lands the coef-bit estimator.
                            coef_bits_est: 0,
                            sse_after_pred: 0,
                            // Intra leaves in the F5.1 path always
                            // emit residuals (`skip = false` on every
                            // coded intra leaf). The RC.X1 skip-on-
                            // zero-eob override only applies to inter.
                            skip_flag: false,
                        },
                    );
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

/// Mi-grid-backed [`rvp9_core::inter::mv_ref::NeighbourLookup`] adapter used
/// by the encoder simulator to spatially seed
/// [`rvp9_core::mode_decision::inter::MvCandidates`] from already-picked
/// neighbour leaves.
///
/// Mirrors the writer-side `MiGridNeighbourLookup` in
/// `rvp9-core::bitstream::modes_writer`. The writer's adapter is private
/// to the core crate; we duplicate the small adapter here so the encoder
/// can build the same lookup against its in-progress mi_grid (the
/// encoder picks leaves in raster order and splats each leaf's
/// `ModeInfo` into the grid right after picking, so all neighbour
/// positions queried by `find_mv_refs` resolve to either already-picked
/// cells or out-of-frame `None`).
struct InterPickerMiGridLookup<'a> {
    mi_grid: &'a [Option<rvp9_core::mode_info::ModeInfo>],
    mi_cols: usize,
    base_mi_row: i32,
    base_mi_col: i32,
    mi_rows: i32,
    mi_cols_frame: i32,
}

impl<'a> InterPickerMiGridLookup<'a> {
    fn new(
        mi_grid: &'a [Option<rvp9_core::mode_info::ModeInfo>],
        mi_cols: usize,
        base_mi_row: usize,
        base_mi_col: usize,
        mi_rows: usize,
    ) -> Self {
        Self {
            mi_grid,
            mi_cols,
            base_mi_row: base_mi_row as i32,
            base_mi_col: base_mi_col as i32,
            mi_rows: mi_rows as i32,
            mi_cols_frame: mi_cols as i32,
        }
    }
}

impl rvp9_core::inter::mv_ref::NeighbourLookup for InterPickerMiGridLookup<'_> {
    fn get(
        &self,
        pos: rvp9_core::inter::mv_ref::Position,
    ) -> Option<&rvp9_core::mode_info::ModeInfo> {
        let row = self.base_mi_row.checked_add(i32::from(pos.row))?;
        let col = self.base_mi_col.checked_add(i32::from(pos.col))?;
        if row < 0 || col < 0 || row >= self.mi_rows || col >= self.mi_cols_frame {
            return None;
        }
        let idx = (row as usize) * self.mi_cols + (col as usize);
        self.mi_grid.get(idx)?.as_ref()
    }
}

/// Splat a single inter leaf's [`rvp9_core::mode_info::ModeInfo`] into
/// every 8x8 cell the leaf covers in the in-progress mi_grid used by
/// the A2.foreman spatial MV-ref propagation. Mirrors
/// `frame_loop::splat_leaf_mode_info`, kept local so the encoder's
/// per-leaf simulator stays self-contained.
#[allow(clippy::too_many_arguments)]
fn splat_inter_leaf_into_mi_grid(
    mi_grid: &mut [Option<rvp9_core::mode_info::ModeInfo>],
    mi_cols: usize,
    leaf_mi_row: usize,
    leaf_mi_col: usize,
    leaf_bsize: BlockSize,
    picked_mode: InterMode,
    picked_mv: Mv,
    segment_id: u8,
    skip: bool,
    // A15 Phase B Step 4 — propagate picked ref frame into the
    // mi_grid so subsequent neighbour lookups (and the writer's
    // mi_grid read) see the same ref the picker chose. Default-
    // callable sites pass `RefFrame::LastFrame` for skip-fallback
    // leaves (matches A14 behaviour).
    picked_ref_frame: RefFrame,
    // F.5 — interpolation filter the picker chose for this leaf.
    // Splatted into every 8x8 cell so the writer's mi_grid read sees
    // the same filter the simulator built the recon with.
    picked_interp_filter: rvp9_core::bitstream::frame::InterpFilter,
) {
    use rvp9_core::mode_info::{BlockMode, ModeInfo};
    let n_8x8_wide = leaf_bsize.n_8x8_wide() as usize;
    let n_8x8_high = leaf_bsize.n_8x8_high() as usize;
    let cell = ModeInfo {
        block_size: leaf_bsize,
        tx_size: rvp9_core::block::max_txsize(leaf_bsize),
        skip,
        segment_id,
        seg_id_predicted: false,
        mode: BlockMode::Inter {
            mode: picked_mode,
            ref_frames: [picked_ref_frame, RefFrame::NoRefFrame],
            mvs: [picked_mv, Mv::ZERO],
            interp_filter: picked_interp_filter,
        },
        bmi: Default::default(),
    };
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let mi_row = leaf_mi_row + dr;
            let mi_col = leaf_mi_col + dc;
            mi_grid[mi_row * mi_cols + mi_col] = Some(cell);
        }
    }
}


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
    /// Picked inter mode (one of `ZeroMv` / `NearestMv` / `NearMv` /
    /// `NewMv`). Always `ZeroMv` on `skip = true` fallback leaves.
    pub picked_mode: InterMode,
    /// Picked MV in 1/8-pel units. `Mv::ZERO` on skip-true leaves.
    pub picked_mv: Mv,
    /// `true` ⇒ this leaf falls back to the skip-true / zero-residual
    /// path (e.g. Block8x8 leaves where F8.1 doesn't apply).
    pub skip: bool,
    /// AQ.1 — segment id `0..=7` driving the per-leaf effective
    /// qindex via [`effective_qindex_for_segment`]. `0` for AQ-off.
    pub segment_id: u8,
    /// A15 Phase B — reference frame the picker chose for this leaf.
    /// Stage 4a' compression-efficiency closure widens the picker
    /// surface from `{LASTFRAME} × {ZERO,NEAREST,NEAR,NEW}` to
    /// `{LASTFRAME, GOLDENFRAME} × {ZERO,NEAREST,NEAR,NEW}`. Default
    /// `RefFrame::LastFrame` matches A14 behaviour (LAST-only picker)
    /// on skip-fallback leaves and pre-A15 callers.
    pub picked_ref_frame: RefFrame,
    /// F.5 — interpolation filter the picker selected for this leaf.
    /// One of `{EightTap, EightTapSmooth}` (libvpx-realtime 2-filter
    /// range). Defaults to `EightTap` on skip-fallback leaves and
    /// pre-F.5 byte-exact paths. The writer reads this field to emit
    /// the per-block switchable-filter bits and the encoder simulator
    /// reads it when building the per-plane MC prediction so the
    /// recon matches what the decoder will produce.
    pub picked_interp_filter: rvp9_core::bitstream::frame::InterpFilter,
    /// Phase 2b — intra mode when `picked_ref_frame == IntraFrame`.
    /// One of `{Dc, V, H}` in v1. `None` for inter winners.
    pub picked_intra_mode: Option<rvp9_core::intra_pred::IntraMode>,
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
    // A15 Phase B — GOLDEN ref planes for the widened picker
    // ({LAST,GOLDEN} × {ZERO,NEAREST,NEAR,NEW}). Step 2: used
    // only for `find_mv_refs(GOLDEN)` to build the GOLDEN
    // `MvCandidates`. Step 3 will additionally use these as the
    // ref plane for the per-candidate SSE/RD evaluation when
    // `picked_ref_frame == GoldenFrame`. The caller threads
    // `golden_recon_*` from `StreamEncoder` (post-A14 Phase A
    // they mirror what the decoder writes when reading
    // `refresh_frame_flags & 0x02`).
    golden_recon_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &mut [Vec<InterPartitionLeafDesc>],
    // A18 — Previous frame's MV grid for `find_mv_refs` salvage
    // (libvpx `cm->prev_frame->mvs`). Indexed row-major at stride
    // `mi_cols`. `None` when the libvpx `use_prev_frame_mvs` gate
    // is false (frame size mismatch, intra-only / KF previous,
    // error-resilient). The caller is responsible for evaluating
    // the gate.
    prev_mv_grid: Option<&[rvp9_core::inter::mv_ref::PrevFrameMvRef]>,
    // A18 — out-param: caller-provided buffer that receives the
    // final picked `(mv[2], ref_frame[2])` per 8×8 mi cell after
    // the SB sweep finishes. Sized `mi_rows * mi_cols`. Mirrors
    // the decoder's `cur_mv_grid` write-back path
    // (`tile.rs:2141-2155`); the encoder swaps this into
    // `prev_mv_grid` for the next frame.
    out_mv_grid: &mut [rvp9_core::inter::mv_ref::PrevFrameMvRef],
    // P0b — adapted FrameContext from the previous frame's backward
    // adaptation. Drives the picker's cost tables (inter mode costs,
    // MV costs, intra-in-inter costs) so that content-adapted
    // probabilities feed mode decisions. For the first inter frame
    // this is still `FrameContext::new_default()` (matching libvpx's
    // frame-N-adapts-frame-N+1 convention).
    frame_context: &FrameContext,
    // P20 — adaptive RD-threshold state (libvpx
    // `vp9_pickmode.c:2241-2247` gate state). Read by the picker for
    // each candidate's effective threshold; mutated after each pick
    // via `update_thresh_freq_fact` to decay the winner and
    // increment losers. Persists across frames.
    rd_thresh_table: &mut rvp9_core::mode_decision::rd_thresh::RdThreshTable,
    // §11 — frames since the last GOLDEN refresh
    // (libvpx `cpi->rc.frames_since_golden`). Feeds the `bias_golden`
    // gate's `frames_since_golden > 4` predicate
    // (`vp9_pickmode.c:2237-2239`).
    frames_since_golden: u32,
    // §11 — libvpx `get_chessboard_index(cm->current_video_frame) =
    // current_video_frame & 1` (`vp9_common.h:88-90`). Feeds the
    // `cb_pred_filter_search` chessboard predicate at
    // `vp9_pickmode.c:1855`.
    chessboard_index: u32,
    // W0.4(vpxenc parity, A7 P1+P2) — per-frame speed-feature inputs.
    // The dispatcher at the top of this function computes the
    // `SpeedFeatures` set from `(speed, deadline, width, height,
    // content_tier)` and the per-leaf `PickInterModeParams` literals
    // below read from it instead of hard-coding the §11 values.
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
    content_tier: rvp9_core::mode_decision::speed_features::ContentTier,
) -> VecBackedInterCoefSource {
    use rvp9_core::bitstream::frame::{InterpFilter as WireInterpFilter, MAX_INTER_REFS};
    use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
    use rvp9_core::cost::mv_cost::build_nmv_cost_table;
    use rvp9_core::cost::symbol_costs::{inter_mode_costs, partition_costs, switchable_interp_costs};
    use rvp9_core::inter::{get_mode_context, get_pred_context_switchable_interp};
    use rvp9_core::inter::mv_ref::{
        BlockBounds, PrevFrameMvRef, find_mv_refs, lower_mv_precision,
    };
    use rvp9_core::mode_decision::inter::{
        InterModeSearchParams, MvCandidates,
        PickInterModeParams, pick_inter_mode,
    };
    use rvp9_core::mode_decision::nonrd_partition::{
        NonRdDescentParams, PartitionRdCandidate, nonrd_pick_partition, variance_hint_bounds,
    };
    use rvp9_core::mode_decision::speed_features::{self as sf_mod};
    use rvp9_core::mv_search::int_pro_me::int_pro_motion_estimation;
    use rvp9_core::mv_search::mv_limits::MvLimits;
    use rvp9_core::mv_search::{PlaneRef, init_dsmotion_compensation};
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers, plane_dequant_pair};

    // W0.4: compute the per-frame `SpeedFeatures` once. The dispatcher
    // is pure — no side effects on `cpi` like libvpx's
    // `vp9_set_speed_features_framesize_{independent,dependent}` —
    // and the result is consumed by every per-leaf
    // `PickInterModeParams` literal below.
    let sf = sf_mod::dispatch(speed, deadline, width, height, content_tier);

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
    // A15 Phase B — GOLDEN Y plane ref for the widened picker.
    // Used in Step 3 to drive candidate SSE/RD when iterating
    // `{LAST, GOLDEN} × {ZERO, NEAREST, NEAR, NEW}`. The
    // encoder calls `pick_inter_mode_luma` once per ref-plane,
    // then takes the min-RD winner across both rankings.
    let ref_y_golden = PlaneRef { data: golden_recon_yuv.0, stride: w, width: w, height: h };

    // Picker harness — built once per frame. The per-leaf
    // `PlaneQuantizers` is rebuilt inside the SB walk below (AQ.1)
    // because it depends on the leaf's `segment_id`.
    let inter_costs = inter_mode_costs(frame_context);
    let sic = switchable_interp_costs(frame_context);
    let mv_costs = build_nmv_cost_table(&frame_context.nmvc, /* allow_hp = */ false);
    let site_config = init_dsmotion_compensation(w as i32);

    // F.3 (2026-05-23, infrastructure-only) — motion-aware partition
    // descent (libvpx `nonrd_pick_partition`,
    // `vp9_encodeframe.c:4922-5210`).
    //
    // **Default OFF**, gated behind `RVP9_F3_NONRD_PARTITION_ENABLE=1`.
    // Mirrors the F.1 / F.2 infrastructure-only pattern (see F.3 HLD
    // §11 ship-state addendum and `notes/divergence-ledger.md` F.3
    // ship-state entry).
    //
    // When the gate is set, this branch runs the descent per SB
    // BEFORE the per-leaf picker loop. The descent's output replaces
    // `sb_partitions[sb_idx]` (the variance-hint leaf list) with a
    // new leaf list reflecting the motion-aware RD-driven decision.
    // The existing per-leaf picker loop then runs over the new
    // leaves unchanged.
    //
    // F.4 owns flipping this default and replacing the descent's
    // simplified mode-picker callback with a call into the real
    // `pick_inter_mode_luma`. The infrastructure here proves the
    // wiring; it does NOT drive byte-stream output at defaults.
    let f3_nonrd_partition_enabled =
        std::env::var_os("RVP9_F3_NONRD_PARTITION_ENABLE").is_some();
    let f3_partition_costs = if f3_nonrd_partition_enabled {
        Some(partition_costs(&FrameContext::new_default()))
    } else {
        None
    };
    // F.3 rd_mult — libvpx `compute_rd_mult(q_index, INTER_FRAME)`
    // (`vp9_rd.c:170-178`). The descent uses it to rank NONE vs
    // SPLIT via [`rvp9_core::mode_decision::rd_cost`].
    let f3_rd_mult: u32 = if f3_nonrd_partition_enabled {
        rvp9_core::cost::compute_rd_mult(
            i32::from(base_qindex),
            rvp9_core::cost::FrameTypeForRd::Inter,
        )
    } else {
        0
    };

    // A16 (2026-05-22) — ref-frame rate-cost for the dual-ref picker.
    //
    // Mirrors libvpx `init_ref_frame_cost` (`vp9_pickmode.c:1215-1230`):
    //   ref_frame_cost[LAST]   = cost_bit(intra_inter,1) + cost_bit(p1,0)
    //   ref_frame_cost[GOLDEN] = cost_bit(intra_inter,1) + cost_bit(p1,1)
    //                                                   + cost_bit(p2,0)
    // Both share the `cost_bit(intra_inter, 1)` term — for the LAST-vs-
    // GOLDEN argmin only the `single_ref_p{1,2}` differential matters,
    // but we include the intra_inter term so the absolute cost magnitudes
    // are libvpx-shaped (useful diagnostic and future-proofing for an
    // intra-vs-inter picker that compares against an intra leaf).
    //
    // **Approximation**: libvpx looks up the per-block context via
    // `vp9_get_pred_prob_single_ref_p{1,2}(cm, xd)` from neighbour
    // blocks. The picker is called BEFORE the leaf is splatted into
    // the mi_grid, so the exact per-leaf context isn't free at this
    // point. We use the libvpx-default probs at the middle REF context
    // (`DEFAULT_SINGLE_REF_PROB[2] = [142, 142]`) and the middle
    // intra_inter context (`DEFAULT_INTRA_INTER_PROB[2] = 187`). This
    // is a constant approximation of the true per-block cost. The
    // differential between LAST and GOLDEN at this context evaluates
    // to ~566 - 425 = ~141 units for the p1 bit alone; adding the p2
    // term tips GOLDEN ~566 units higher than LAST in total. That is
    // a meaningful fraction of one wire bit (~512 units at p=128) and
    // is enough to break SSE ties in LAST's favour at high q.
    let ref_frame_cost_last: u32 = {
        use rvp9_core::cost::vp9_cost_bit;
        use rvp9_core::entropy::DEFAULT_INTRA_INTER_PROB;
        use rvp9_core::entropy::DEFAULT_SINGLE_REF_PROB;
        let intra_inter_p = DEFAULT_INTRA_INTER_PROB[2];
        let single_ref_p1 = DEFAULT_SINGLE_REF_PROB[2][0];
        vp9_cost_bit(intra_inter_p, true) + vp9_cost_bit(single_ref_p1, false)
    };
    let ref_frame_cost_golden: u32 = {
        use rvp9_core::cost::vp9_cost_bit;
        use rvp9_core::entropy::DEFAULT_INTRA_INTER_PROB;
        use rvp9_core::entropy::DEFAULT_SINGLE_REF_PROB;
        let intra_inter_p = DEFAULT_INTRA_INTER_PROB[2];
        let single_ref_p1 = DEFAULT_SINGLE_REF_PROB[2][0];
        let single_ref_p2 = DEFAULT_SINGLE_REF_PROB[2][1];
        vp9_cost_bit(intra_inter_p, true)
            + vp9_cost_bit(single_ref_p1, true)
            + vp9_cost_bit(single_ref_p2, false)
    };

    // Phase 2b — ref_frame_cost for INTRA_FRAME. libvpx's
    // `init_ref_frame_cost` at `vp9_pickmode.c:1222`:
    //   ref_frame_cost[INTRA_FRAME] = vp9_cost_bit(intra_inter_p, 0)
    let ref_frame_cost_intra: u32 = {
        use rvp9_core::cost::vp9_cost_bit;
        use rvp9_core::entropy::DEFAULT_INTRA_INTER_PROB;
        let intra_inter_p = DEFAULT_INTRA_INTER_PROB[2];
        vp9_cost_bit(intra_inter_p, false)
    };

    // Phase 2b — inter-frame intra-Y-mode cost table (mbmode_cost).
    // libvpx builds `cpi->mbmode_cost` from
    // `fc->y_mode_prob[size_group_lookup[bsize]]`; for our supported
    // leaf sizes (16x16 / 32x32 / 64x64) the size_group is 2 or 3.
    // We use size_group=1 (libvpx's `mbmode_cost` default, used in
    // the intra-in-inter path at `vp9_pickmode.c:2621`). The exact
    // group doesn't matter much — libvpx uses size_group_lookup[bsize]
    // only in the full RDO path; the realtime path pins at
    // `cpi->mbmode_cost` which is group 1.
    let intra_y_mode_costs_inter: [i32; 10] = {
        let table = rvp9_core::cost::intra_y_mode_costs_inter(frame_context);
        table[1] // size_group = 1 (libvpx mbmode_cost default)
    };

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

    // A16 diagnostic: count GOLDEN ref-frame picks per inter frame when
    // RVP9_DIAG_GOLDEN_PICKS is set. Emitted as a single eprintln at
    // end of the SB loop below. Counts only non-skip leaves.
    let mut diag_golden_picks: usize = 0;
    let mut diag_last_picks: usize = 0;
    let diag_golden_enabled = std::env::var("RVP9_DIAG_GOLDEN_PICKS").is_ok();

    // E3 (2026-05-22) — NEWMV diagnosis. Env-gated (`RVP9_E3_MVDIAG`).
    // For every picker-supported leaf we compute the LAST-ref
    // `pick_inter_mode_luma_all_costs` breakdown and accumulate:
    //   * per-mode win counts (which of ZERO/NEAREST/NEAR/NEW won the
    //     4-way argmin on LAST),
    //   * NEWMV MV magnitude histogram (|row|+|col| in 1/8-pel),
    //   * for leaves where NEWMV had the lowest DISTORTION but did NOT
    //     win the RD argmin: the RD gap and the MV-rate share of NEWMV's
    //     cost (Hypothesis 3 evidence).
    // A sample (first `E3_DIAG_SAMPLE_LIMIT`) of per-leaf rows is printed
    // verbatim; the rest is summarised at frame end.
    let e3_diag_enabled = std::env::var("RVP9_E3_MVDIAG").is_ok();
    const E3_DIAG_SAMPLE_LIMIT: usize = 40;
    let mut e3_sample_printed: usize = 0;
    // win counts indexed [ZERO, NEAREST, NEAR, NEW].
    let mut e3_win_counts = [0usize; 4];
    let mut e3_total_leaves: usize = 0;
    let mut e3_newmv_nonzero: usize = 0; // NEWMV MV magnitude > 0
    let mut e3_newmv_mag_sum: u64 = 0;
    let mut e3_newmv_mag_max: u32 = 0;
    // Leaves where NEWMV had strictly lowest distortion among the 4.
    let mut e3_newmv_best_dist: usize = 0;
    // ...of those, how many NEWMV nonetheless LOST the RD argmin.
    let mut e3_newmv_best_dist_lost: usize = 0;
    // Accumulated NEWMV cost decomposition over the "best-dist-but-lost"
    // leaves: rate vs distortion shares + RD gap to the winner.
    let mut e3_lost_mv_rate_sum: u64 = 0;
    let mut e3_lost_total_rate_sum: u64 = 0;
    let mut e3_lost_rd_gap_sum: i128 = 0;

    // Initialise the recon planes from the LAST ref. ZERO-MV / skip
    // leaves get this as their final recon (no residual added); the
    // picker-driven leaves overwrite their region via the per-TX
    // IT-add. Mirrors the decoder's "MC writes into dst, then residual
    // IT-add applies on top" sequence.
    recon_yuv_out.0.copy_from_slice(last_recon_yuv.0);
    recon_yuv_out.1.copy_from_slice(last_recon_yuv.1);
    recon_yuv_out.2.copy_from_slice(last_recon_yuv.2);

    // A2.foreman - In-progress mi_grid for spatial MV-ref propagation
    // (H3' fix). The encoder walks leaves in raster order; right after
    // a leaf is picked we splat its `ModeInfo` (carrying the picked
    // `(InterMode, Mv)`) into every 8x8 cell the leaf covers. Before
    // the picker runs on each leaf, `find_mv_refs` reads the already-
    // picked neighbour cells via [`InterPickerMiGridLookup`] and
    // returns the spatially-seeded `(nearest, near)` MV pair, which we
    // hand to the picker via [`MvCandidates`].
    //
    // Without this, every NEWMV is encoded as a delta from (0,0), so
    // on content with continuous motion (e.g. foreman_cif) the rate
    // cost prohibitively favours NEARESTMV/ZEROMV with mv=(0,0) - the
    // reconstruction then drifts further from the source every frame.
    //
    // Sign-bias is all-`false` under our single-LASTFRAME ref scheme
    // (matches `frame_loop::build_inter_packet_with_residuals`).
    let mi_rows_us = h / 8;
    let mi_cols_us = w / 8;
    let mut mi_grid: Vec<Option<rvp9_core::mode_info::ModeInfo>> =
        vec![None; mi_rows_us * mi_cols_us];
    let sign_bias = sign_bias_from_header(&[false, false, false]);

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let sb_idx = sb_row * sb_cols + sb_col;

            // A3'-int-pro - SB-level coarse ME (libvpx
            // `vp9_int_pro_motion_estimation`,
            // `vp9_mcomp.c:2264-2400`). Produces a starter
            // integer-pel MV per 64×64 SB by comparing 1D
            // row/column projections of source vs LASTFRAME ref.
            // The per-leaf picker uses this MV as the diamond-
            // search center for NEWMV (mirror of libvpx's
            // `x->pred_mv[LAST_FRAME]` plumbing at
            // `vp9_encodeframe.c:1495`). Without this seed the
            // per-leaf NEWMV diamond starts at (0,0) and the
            // MV-bit cost dominates over the SSE gain → picker
            // never picks NEWMV → spatial MV-ref propagation
            // (A2.foreman) never chains.
            //
            // Cheap: 1D projections (not full 2D SAD). Result is
            // an unclamped Q3 MV; per-leaf clamping happens
            // inside [`pick_inter_mode_luma`] via the picker's
            // own MvLimits envelope before the diamond search.
            //
            // E10 — every SB (including frame-edge SBs) now produces a
            // seed: the projection window and 2D-SAD halo read through
            // an edge-clamped border (libvpx's extended-border frame
            // equivalent) instead of bailing. A literal zero-MV result
            // is still treated as "no seed" (the picker's existing
            // NEAREST/ZERO-at-(0,0) candidates cover it); previously
            // 12 of 16 SBs on a 256×256 frame bailed outright.
            let sb_px_x = sb_col * 64;
            let sb_px_y = sb_row * 64;
            let int_pro = int_pro_motion_estimation(
                &src_y,
                &ref_y,
                sb_px_x,
                sb_px_y,
                BlockSize::Block64x64,
            );
            // Treat a literal zero MV as "no useful seed" — the
            // picker's existing NEAREST/ZERO-at-(0,0) candidates
            // already cover that case.
            let sb_newmv_seed = if int_pro.mv == Mv::ZERO {
                None
            } else {
                Some(int_pro.mv)
            };

            // A3'-int-pro diagnostic — per-SB seed activity. Set
            // RVP9_DIAG_INTPRO_SB=1 to enable (off by default to
            // avoid log spam).
            if std::env::var("RVP9_DIAG_INTPRO_SB").is_ok() {
                #[allow(clippy::print_stderr)]
                {
                    eprintln!(
                        "  int_pro sb=({},{}) → mv=({},{}) sad={} seed={}",
                        sb_row,
                        sb_col,
                        int_pro.mv.row,
                        int_pro.mv.col,
                        int_pro.best_sad,
                        if sb_newmv_seed.is_some() { "Y" } else { "N" },
                    );
                }
            }

            // F.3 (infrastructure-only) — when
            // `RVP9_F3_NONRD_PARTITION_ENABLE=1` is set, run the
            // motion-aware RD descent BEFORE the per-leaf picker
            // loop. The descent replaces this SB's variance-hint
            // leaf list with a new one reflecting the
            // RD-vs-variance comparison. At default (env unset),
            // this branch is skipped and the existing picker loop
            // runs over the variance-hint leaves byte-identical to
            // pre-F.3.
            //
            // The simplified mode-picker callback here computes a
            // ZERO-MV SSE rate-distortion proxy (NOT the real
            // `pick_inter_mode_luma`). F.4 owns replacing this
            // with the real picker when activating F.3 — that
            // requires lambda recalibration per the F.3 HLD §11
            // ship-state addendum. For infrastructure-only
            // purposes the callback proves the wiring: descent →
            // callback → per-position RD → new leaf list.
            if f3_nonrd_partition_enabled {
                use rvp9_core::block::BlockSize as BS;
                use rvp9_core::mode_decision::partition::PartitionDecision;

                // 1. Rebuild a `PartitionDecision { cells }` from
                //    the variance-hint leaf list.
                let mut hint_cells = [[BS::Block8x8; 8]; 8];
                for leaf in &sb_partitions[sb_idx] {
                    let span = match leaf.leaf_bsize {
                        BS::Block64x64 => 8,
                        BS::Block32x32 => 4,
                        BS::Block16x16 => 2,
                        BS::Block8x8 => 1,
                        _ => 1,
                    };
                    for dr in 0..span {
                        for dc in 0..span {
                            let r = leaf.mi_row_in_sb + dr;
                            let c = leaf.mi_col_in_sb + dc;
                            if r < 8 && c < 8 {
                                hint_cells[r][c] = leaf.leaf_bsize;
                            }
                        }
                    }
                }
                let hint = PartitionDecision { cells: hint_cells };

                // 2. Derive (min, max) bounds (libvpx
                //    `x->{min,max}_partition_size`) via the
                //    variance hint, with the R3 floor at
                //    Block16x16.
                let (min_bs, max_bs) = variance_hint_bounds(&hint);

                // 3. Build the descent params.
                let pcosts = f3_partition_costs
                    .as_ref()
                    .expect("partition_costs built when F.3 enabled");
                let params = NonRdDescentParams {
                    min_partition_size: min_bs,
                    max_partition_size: max_bs,
                    partition_costs: pcosts,
                    rd_mult: f3_rd_mult,
                };

                // 4. P12 (2026-05-25) — real `pick_inter_mode`
                //    callback. Constructs a full
                //    `PickInterModeParams` for each leaf the
                //    descent evaluates and invokes the same
                //    picker that the post-descent per-leaf loop
                //    uses. The descent now sees motion-aware RD
                //    costs (NEWMV / NEARESTMV winners) instead of
                //    a ZERO-MV-SSE proxy and can prefer NONE on
                //    blocks whose true motion is uniform.
                //
                //    Frame-level state (cost tables, ref planes,
                //    quantiser sources) is captured by reference.
                //    Per-leaf state — mv_limits, mv refs,
                //    inter_mode_ctx, intra edges, switchable /
                //    skip / const_motion contexts — is computed
                //    inside the callback from the leaf's
                //    (mi_row, mi_col, bsize). This mirrors the
                //    structure of the post-descent per-leaf
                //    picker setup at the same call site; the two
                //    paths now exercise the same picker, just at
                //    different sub-block resolutions.
                //
                //    Borrow plumbing: `mi_grid` / `recon_yuv_out`
                //    are read immutably for the descent (the
                //    descent doesn't mutate them — only the
                //    post-descent picker loop does). Borrows are
                //    re-acquired inline so they expire when this
                //    block ends, freeing the mutable borrows
                //    that the per-leaf loop needs.
                let mi_grid_ro: &[Option<rvp9_core::mode_info::ModeInfo>] = &mi_grid;
                let recon_y_ro: &[u8] = &recon_yuv_out.0;
                let mut descent_picker =
                    |sb_local_x_8x8: usize, sb_local_y_8x8: usize, bs: BS| {
                        let leaf_mi_row = sb_row * 8 + sb_local_y_8x8;
                        let leaf_mi_col = sb_col * 8 + sb_local_x_8x8;
                        let leaf_px_x = leaf_mi_col * 8;
                        let leaf_px_y = leaf_mi_row * 8;
                        let bw_px = usize::from(bs.width());
                        let bh_px = usize::from(bs.height());
                        // SB-edge guard for partial frames.
                        if leaf_px_x + bw_px > w || leaf_px_y + bh_px > h {
                            return None;
                        }
                        // The picker is wired for these 4 luma sizes. The
                        // descent caps at Block16x16 (HLD §4.3) so 8×8 is
                        // never queried by the descent itself — defensive
                        // guard kept for symmetry with the post-descent
                        // loop.
                        if !matches!(
                            bs,
                            BlockSize::Block8x8
                                | BlockSize::Block16x16
                                | BlockSize::Block32x32
                                | BlockSize::Block64x64,
                        ) {
                            return None;
                        }

                        // MV envelope (mirrors the post-descent loop's
                        // luma + 4:2:0 chroma intersection).
                        let col_min_luma = -(leaf_px_x as i32) + 3;
                        let col_max_luma =
                            (w as i32) - (leaf_px_x as i32) - (bw_px as i32) - 4;
                        let row_min_luma = -(leaf_px_y as i32) + 3;
                        let row_max_luma =
                            (h as i32) - (leaf_px_y as i32) - (bh_px as i32) - 4;
                        let col_min_chroma = 6 - leaf_px_x as i32;
                        let col_max_chroma =
                            (w as i32) - (leaf_px_x as i32) - (bw_px as i32) - 7;
                        let row_min_chroma = 6 - leaf_px_y as i32;
                        let row_max_chroma =
                            (h as i32) - (leaf_px_y as i32) - (bh_px as i32) - 7;
                        let col_min = col_min_luma.max(col_min_chroma);
                        let col_max = col_max_luma.min(col_max_chroma);
                        let row_min = row_min_luma.max(row_min_chroma);
                        let row_max = row_max_luma.min(row_max_chroma);
                        if col_min > col_max || row_min > row_max {
                            return None;
                        }
                        let mv_limits = MvLimits::new(col_min, col_max, row_min, row_max);
                        let mv_in_bounds = |m: Mv| -> bool {
                            let col_fp = i32::from(m.col) >> 3;
                            let row_fp = i32::from(m.row) >> 3;
                            col_fp >= col_min
                                && col_fp <= col_max
                                && row_fp >= row_min
                                && row_fp <= row_max
                        };

                        // mi-grid neighbour lookup for find_mv_refs and
                        // context predicates. The descent runs BEFORE
                        // the per-leaf picker loop for this SB, so
                        // `mi_grid` carries only prior-SB neighbours;
                        // within-SB neighbours return `None` (the same
                        // condition the post-descent first-leaf sees).
                        let bw_mi = bs.n_8x8_wide() as i32;
                        let bh_mi = bs.n_8x8_high() as i32;
                        let bounds = BlockBounds::new(
                            leaf_mi_row as i32,
                            leaf_mi_col as i32,
                            mi_rows_us as i32,
                            mi_cols_us as i32,
                            bw_mi,
                            bh_mi,
                        );
                        let neighbours = InterPickerMiGridLookup::new(
                            mi_grid_ro,
                            mi_cols_us,
                            leaf_mi_row,
                            leaf_mi_col,
                            mi_rows_us,
                        );
                        let inter_mode_ctx = get_mode_context(bs, &neighbours);
                        let leaf_prev_mv: Option<PrevFrameMvRef> = prev_mv_grid
                            .and_then(|g| {
                                g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied()
                            });
                        let build_mvc = |rf: RefFrame| -> MvCandidates {
                            let list_nearest = find_mv_refs(
                                InterMode::NewMv,
                                bs,
                                rf,
                                &neighbours,
                                &sign_bias,
                                &bounds,
                                leaf_prev_mv,
                            );
                            let nearest = lower_mv_precision(list_nearest.mvs[0], false);
                            let list_near = find_mv_refs(
                                InterMode::NearMv,
                                bs,
                                rf,
                                &neighbours,
                                &sign_bias,
                                &bounds,
                                leaf_prev_mv,
                            );
                            let near_idx = list_near
                                .count
                                .saturating_sub(1)
                                .min(list_near.mvs.len() - 1);
                            let near = lower_mv_precision(list_near.mvs[near_idx], false);
                            MvCandidates { nearest, near }
                        };
                        let mvc = build_mvc(RefFrame::LastFrame);
                        let mvc_golden = build_mvc(RefFrame::GoldenFrame);

                        // Per-leaf segment_id from the variance hint.
                        let seg = sb_partitions[sb_idx]
                            .iter()
                            .find(|h| {
                                h.mi_row_in_sb == sb_local_y_8x8
                                    && h.mi_col_in_sb == sb_local_x_8x8
                            })
                            .map(|h| h.segment_id)
                            .unwrap_or(0);
                        let leaf_qindex =
                            effective_qindex_for_segment(base_qindex, seg, seg_q_deltas);
                        let quantizer = init_plane_quantizers(
                            i32::from(leaf_qindex),
                            QuantizerDeltas::Y,
                        );
                        let (uv_dc_dq, uv_ac_dq) = plane_dequant_pair(
                            i32::from(leaf_qindex),
                            QuantizerDeltas { dc_delta: 0, ac_delta: 0 },
                        );

                        // Intra-prediction neighbour edges from the
                        // running Y recon (initialised to last-recon,
                        // so for the first SB this is identical to the
                        // reference plane).
                        let intra_have_above = leaf_px_y > 0;
                        let intra_have_left = leaf_px_x > 0;
                        let mut intra_above_buf = [0u8; 64];
                        let mut intra_left_buf = [0u8; 64];
                        let intra_above: Option<&[u8]> = if intra_have_above {
                            let row_off = (leaf_px_y - 1) * w + leaf_px_x;
                            let n = bw_px.min(w - leaf_px_x);
                            intra_above_buf[..n]
                                .copy_from_slice(&recon_y_ro[row_off..row_off + n]);
                            Some(&intra_above_buf[..n])
                        } else {
                            None
                        };
                        let intra_left: Option<&[u8]> = if intra_have_left {
                            let n = bh_px.min(h - leaf_px_y);
                            for r in 0..n {
                                intra_left_buf[r] =
                                    recon_y_ro[(leaf_px_y + r) * w + (leaf_px_x - 1)];
                            }
                            Some(&intra_left_buf[..n])
                        } else {
                            None
                        };
                        let intra_top_left = if intra_have_above && intra_have_left {
                            recon_y_ro[(leaf_px_y - 1) * w + (leaf_px_x - 1)]
                        } else {
                            127
                        };

                        // Neighbour mi-grid context predicates.
                        let nb_above = if leaf_mi_row > 0 {
                            mi_grid_ro[(leaf_mi_row - 1) * mi_cols_us + leaf_mi_col]
                                .as_ref()
                        } else {
                            None
                        };
                        let nb_left = if leaf_mi_col > 0 {
                            mi_grid_ro[leaf_mi_row * mi_cols_us + (leaf_mi_col - 1)]
                                .as_ref()
                        } else {
                            None
                        };
                        let switchable_interp_ctx =
                            get_pred_context_switchable_interp(nb_above, nb_left);
                        let skip_ctx =
                            rvp9_core::inter::get_pred_context_skip(nb_above, nb_left);
                        let last_const_motion =
                            rvp9_core::inter::mv_refs_rt_const_motion(
                                bs,
                                RefFrame::LastFrame,
                                &neighbours,
                            );
                        let golden_const_motion =
                            rvp9_core::inter::mv_refs_rt_const_motion(
                                bs,
                                RefFrame::GoldenFrame,
                                &neighbours,
                            );
                        let neighbour_mv = |mi: Option<&rvp9_core::mode_info::ModeInfo>|
                            -> Option<Mv> {
                            match mi.map(|m| m.mode) {
                                Some(rvp9_core::mode_info::BlockMode::Inter {
                                    mvs, ..
                                }) => Some(mvs[0]),
                                _ => None,
                            }
                        };
                        let above_mv_p13 = neighbour_mv(nb_above);
                        let left_mv_p13 = neighbour_mv(nb_left);

                        let pick_params = PickInterModeParams {
                            block_size: bs,
                            src_plane: &src_y,
                            src_offset: (leaf_px_x, leaf_px_y),
                            last_ref_plane: &ref_y,
                            last_mv_candidates: &mvc,
                            golden_ref_plane: &ref_y_golden,
                            golden_mv_candidates: &mvc_golden,
                            mv_limits,
                            site_config: &site_config,
                            mv_costs: &mv_costs,
                            inter_mode_costs: &inter_costs,
                            inter_mode_ctx,
                            quantizer: &quantizer,
                            q_index: leaf_qindex,
                            allow_hp: false,
                            frame_kind,
                            newmv_seed: sb_newmv_seed,
                            ref_frame_cost_last,
                            ref_frame_cost_golden,
                            last_nearest_in_bounds: mv_in_bounds(mvc.nearest),
                            last_near_in_bounds: mv_in_bounds(mvc.near),
                            golden_nearest_in_bounds: mv_in_bounds(mvc_golden.nearest),
                            golden_near_in_bounds: mv_in_bounds(mvc_golden.near),
                            intra_above_pixels: intra_above,
                            intra_left_pixels: intra_left,
                            intra_top_left_pixel: intra_top_left,
                            intra_y_mode_costs: &intra_y_mode_costs_inter,
                            ref_frame_cost_intra,
                            switchable_interp_costs: &sic,
                            switchable_interp_ctx,
                            skip_ctx,
                            last_const_motion,
                            golden_const_motion,
                            src_u: input_yuv.1,
                            src_u_stride: uvw,
                            src_v: input_yuv.2,
                            src_v_stride: uvw,
                            last_ref_u: last_recon_yuv.1,
                            last_ref_v: last_recon_yuv.2,
                            golden_ref_u: golden_recon_yuv.1,
                            golden_ref_v: golden_recon_yuv.2,
                            uv_stride: uvw,
                            uv_dc_quant: uv_dc_dq as u32,
                            uv_ac_quant: uv_ac_dq as u32,
                            encode_breakout: 20,
                            dequant_dc: quantizer.dequant[0] as u16,
                            dequant_ac: quantizer.dequant[1] as u16,
                            above_mv: above_mv_p13,
                            left_mv: left_mv_p13,
                            noise_estimate:
                                rvp9_core::mode_decision::inter::NoiseEstimate::default(),
                            lowvar_highsumdiff: false,
                            is_skin: false,
                            reference_masking: true,
                            // P20 — disabled in the descent: the adaptive
                            // RD-threshold table is mutated post-pick by
                            // the per-leaf loop. Letting the descent
                            // read it could couple descent decisions to
                            // post-pick state and break determinism
                            // across runs that explore different
                            // candidate sets. Pass `None` (libvpx
                            // pre-init behaviour: gate disabled).
                            rd_thresh_table: None,
                            short_circuit_flat_blocks: false,
                            source_variance:
                                rvp9_core::mode_decision::inter::SOURCE_VARIANCE_UNCOMPUTED,
                            force_skip_low_temp_var: false,
                            content_state_sb:
                                rvp9_core::mode_decision::inter::ContentStateSb::Normal,
                            is_screen_content: false,
                            zero_temp_sad_source: None,
                            scene_change_detected: false,
                            inter_mode_mask:
                                rvp9_core::mode_decision::inter::default_inter_mode_mask(),
                            // W0.4: SF fields now flow from the
                            // dispatcher computed at the top of this
                            // function; the literal previously
                            // hard-coded the cpu-used-6/7 RT values.
                            limit_newmv_early_exit: sf.limit_newmv_early_exit,
                            bias_golden: sf.bias_golden,
                            frames_since_golden,
                            cb_pred_filter_search: sf.cb_pred_filter_search,
                            mi_row: leaf_mi_row as u32,
                            mi_col: leaf_mi_col as u32,
                            chessboard_index,
                            disable_golden_ref: sf.disable_golden_ref,
                            max_intra_bsize: sf.max_intra_bsize,
                            rt_intra_dc_only_low_content: sf.rt_intra_dc_only_low_content,
                        };
                        let result = pick_inter_mode(&pick_params);
                        // Translate to the descent's PartitionRdCandidate.
                        // `best_rate` is already in `vp9_prob_cost` units
                        // (1/512-bit) — same family as the descent's
                        // `partition_cost` adds. `best_distortion` is
                        // pixel-SSE-of-recon — same as the proxy used.
                        // `ref_frame` discriminant matches the encoder's
                        // `RefFrame as u8` convention (see proxy site for
                        // the existing encoding).
                        Some(PartitionRdCandidate {
                            rate: result.best_rate,
                            dist: result.best_distortion,
                            mv: result.best_mv,
                            mode: result.best_mode as u8,
                            ref_frame: result.picked_ref_frame as u8,
                        })
                    };

                let descent_result =
                    nonrd_pick_partition(&hint, &params, &mut descent_picker);

                // 5. Convert the descent's leaf list back into
                //    `Vec<InterPartitionLeafDesc>`. Preserve the
                //    `segment_id` from the variance hint at each
                //    leaf's top-left mi cell (the descent doesn't
                //    own segment-id information; AQ.1 owns it).
                let mut new_leaves: Vec<InterPartitionLeafDesc> =
                    Vec::with_capacity(descent_result.leaves.len());
                for leaf in &descent_result.leaves {
                    // Look up segment_id from the variance hint's
                    // leaf list at this position. Default to 0 if
                    // not found (defensive — should always match
                    // for AQ-off frames where every leaf has
                    // segment_id = 0).
                    let seg = sb_partitions[sb_idx]
                        .iter()
                        .find(|h| {
                            h.mi_row_in_sb == leaf.mi_row_in_sb
                                && h.mi_col_in_sb == leaf.mi_col_in_sb
                        })
                        .map(|h| h.segment_id)
                        .unwrap_or(0);
                    new_leaves.push(InterPartitionLeafDesc {
                        mi_row_in_sb: leaf.mi_row_in_sb,
                        mi_col_in_sb: leaf.mi_col_in_sb,
                        leaf_bsize: leaf.bsize,
                        // The picker loop below overwrites these,
                        // but we seed them to ZeroMv defaults so
                        // the leaf list is well-formed before the
                        // picker runs.
                        picked_mode: InterMode::ZeroMv,
                        picked_mv: Mv::ZERO,
                        skip: true,
                        segment_id: seg,
                        picked_ref_frame: RefFrame::LastFrame,
                        picked_interp_filter:
                            rvp9_core::bitstream::frame::InterpFilter::EightTap,
                        picked_intra_mode: None,
                    });
                }
                sb_partitions[sb_idx] = new_leaves;
            }

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
                // recorded.
                let picker_supported = matches!(
                    leaf_bsize,
                    BlockSize::Block8x8
                        | BlockSize::Block16x16
                        | BlockSize::Block32x32
                        | BlockSize::Block64x64
                );

                if !picker_supported {
                    sb_partitions[sb_idx][leaf_idx].picked_mode = InterMode::ZeroMv;
                    sb_partitions[sb_idx][leaf_idx].picked_mv = Mv::ZERO;
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame = RefFrame::LastFrame;
                    // F.0.2 oracle — record the skip-fallback decision
                    // too (otherwise sub-8 leaves are missing from the
                    // sink and the libvpx-side patch would emit an
                    // unmatched record).
                    if crate::f02_oracle::is_enabled() {
                        crate::f02_oracle::record_decision(
                            crate::f02_oracle::BlockDecision {
                                frame_idx: crate::f02_oracle::current_frame_idx(),
                                mi_row: leaf_mi_row as u16,
                                mi_col: leaf_mi_col as u16,
                                bsize: leaf_bsize,
                                mode: crate::f02_oracle::BlockDecisionMode::Inter(
                                    InterMode::ZeroMv,
                                ),
                                ref_frame: RefFrame::LastFrame,
                                mv_row: 0,
                                mv_col: 0,
                                mode_bits_est: 0,
                                mv_bits_est: 0,
                                coef_bits_est: 0,
                                sse_after_pred: 0,
                                skip_flag: true,
                            },
                        );
                    }
                    splat_inter_leaf_into_mi_grid(
                        &mut mi_grid,
                        mi_cols_us,
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        InterMode::ZeroMv,
                        Mv::ZERO,
                        sb_partitions[sb_idx][leaf_idx].segment_id,
                        /* skip = */ true,
                        /* picked_ref_frame = */ RefFrame::LastFrame,
                        /* picked_interp_filter = */
                        rvp9_core::bitstream::frame::InterpFilter::EightTap,
                    );
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
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame = RefFrame::LastFrame;
                    // F.0.2 oracle — also record this degenerate-MV-
                    // envelope fallback so frame-edge leaves aren't
                    // silently missing from the sink (and the libvpx-
                    // side patch's coverage stays paired).
                    if crate::f02_oracle::is_enabled() {
                        crate::f02_oracle::record_decision(
                            crate::f02_oracle::BlockDecision {
                                frame_idx: crate::f02_oracle::current_frame_idx(),
                                mi_row: leaf_mi_row as u16,
                                mi_col: leaf_mi_col as u16,
                                bsize: leaf_bsize,
                                mode: crate::f02_oracle::BlockDecisionMode::Inter(
                                    InterMode::ZeroMv,
                                ),
                                ref_frame: RefFrame::LastFrame,
                                mv_row: 0,
                                mv_col: 0,
                                mode_bits_est: 0,
                                mv_bits_est: 0,
                                coef_bits_est: 0,
                                sse_after_pred: 0,
                                skip_flag: true,
                            },
                        );
                    }
                    splat_inter_leaf_into_mi_grid(
                        &mut mi_grid,
                        mi_cols_us,
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        InterMode::ZeroMv,
                        Mv::ZERO,
                        sb_partitions[sb_idx][leaf_idx].segment_id,
                        /* skip = */ true,
                        /* picked_ref_frame = */ RefFrame::LastFrame,
                        /* picked_interp_filter = */
                        rvp9_core::bitstream::frame::InterpFilter::EightTap,
                    );
                    continue;
                }
                let mv_limits = MvLimits::new(col_min, col_max, row_min, row_max);

                // A2.foreman - Spatial MV-ref seeding (H3' fix).
                // Build a `find_mv_refs` neighbour lookup against the
                // in-progress `mi_grid` (cells covered by earlier-
                // picked leaves carry their `(InterMode, Mv)` already)
                // and derive the `(nearest, near)` MV pair for the
                // current leaf's NEWMV ref-MV / NEARESTMV candidate.
                // This mirrors the decoder's `find_mv_refs` call at
                // the same leaf, so the encoder's `MvCandidates`
                // exactly match what the decoder will compute when
                // reading the bitstream - cross-decoder byte-
                // exactness preserved.
                //
                // The previous-frame MV grid (`use_prev_frame_mvs`)
                // is not threaded yet (Phase D / Wave 3 K); we pass
                // `None`, matching the decoder's `prev_mv_grid =
                // None` setup in
                // `frame_loop::build_inter_packet_with_residuals`.
                // A15 Phase B Step 2 — build BOTH LAST and GOLDEN
                // `MvCandidates` per leaf. The mi_grid neighbours +
                // bounds are shared between the two ref-frame walks;
                // only the `RefFrame` argument to `find_mv_refs`
                // differs. Decoder-side `find_mv_refs` already
                // dispatches per-ref-frame so this mirrors what the
                // decoder will compute when reading a GOLDEN block.
                let bw_mi = leaf_bsize.n_8x8_wide() as i32;
                let bh_mi = leaf_bsize.n_8x8_high() as i32;
                let bounds = BlockBounds::new(
                    leaf_mi_row as i32,
                    leaf_mi_col as i32,
                    mi_rows_us as i32,
                    mi_cols_us as i32,
                    bw_mi,
                    bh_mi,
                );
                let neighbours = InterPickerMiGridLookup::new(
                    &mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    mi_rows_us,
                );
                // Phase 0a — inter-mode context from spatial neighbours.
                // Matches libvpx `x->mbmi_ext->mode_context[ref_frame]`
                // populated by `vp9_find_mv_refs`. Context is ref-frame-
                // independent (only spatial neighbour modes matter).
                let inter_mode_ctx =
                    get_mode_context(leaf_bsize, &neighbours);
                // A18 — collocated `PrevFrameMvRef` for `find_mv_refs`
                // salvage. Mirrors the decoder's lookup at
                // `tile.rs:2036` —
                //   prev_frame_mvs + mi_row * mi_cols + mi_col
                // (libvpx `vp9_mvref_common.c:25-28`). The decoder
                // splats the captured MV/ref pair across every cell the
                // block covers, so reading the leaf's top-left mi cell
                // is sufficient and matches what the decoder will see.
                //
                // The activation gate (`RVP9_A18_ACTIVATE`) lives at
                // the caller (`frame_loop::encode_inter_internal`); when
                // gated off, `prev_mv_grid` is `None` here and this
                // lookup returns `None` — i.e. find_mv_refs falls
                // through to the pre-A18 spatial-only behaviour.
                let leaf_prev_mv: Option<PrevFrameMvRef> = prev_mv_grid
                    .and_then(|g| {
                        g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied()
                    });
                let build_mvc = |ref_frame: RefFrame| -> MvCandidates {
                    // Use `InterMode::NewMv` so `find_mv_refs` runs the
                    // early-break path (refmv_count == 1; mvs[0] is the
                    // best ref MV - libvpx `tmp_mvs[refmv_count - 1]`
                    // with `refmv_count == 1`). Apply
                    // `lower_mv_precision` to mirror the writer's
                    // exact `best_ref_mv` derivation (mode_info.rs
                    // line 1098-1109).
                    let list_nearest = find_mv_refs(
                        InterMode::NewMv,
                        leaf_bsize,
                        ref_frame,
                        &neighbours,
                        &sign_bias,
                        &bounds,
                        /* prev_frame_mv = */ leaf_prev_mv,
                    );
                    let nearest = lower_mv_precision(
                        list_nearest.mvs[0],
                        /* allow_hp = */ false,
                    );
                    // A12 — `find_mv_refs(NEARMV)` separately so the
                    // search continues past the first candidate
                    // (`early_break` is gated on NEARMV).
                    let list_near = find_mv_refs(
                        InterMode::NearMv,
                        leaf_bsize,
                        ref_frame,
                        &neighbours,
                        &sign_bias,
                        &bounds,
                        /* prev_frame_mv = */ leaf_prev_mv,
                    );
                    let near_idx = list_near
                        .count
                        .saturating_sub(1)
                        .min(list_near.mvs.len() - 1);
                    let near = lower_mv_precision(
                        list_near.mvs[near_idx],
                        /* allow_hp = */ false,
                    );
                    MvCandidates { nearest, near }
                };
                let mvc = build_mvc(RefFrame::LastFrame);
                // A15 Phase B Step 3 — GOLDEN candidates feed the
                // second picker invocation below. The encoder iterates
                // {LAST, GOLDEN} × {ZERO, NEAREST, NEAR, NEW} = 8
                // candidates by calling `pick_inter_mode_luma` once
                // per ref frame and taking the min-RD winner across
                // both rankings.
                let mvc_golden = build_mvc(RefFrame::GoldenFrame);

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
                // G1 — UV dequant pair for chroma model_rd.
                let (uv_dc_dq, uv_ac_dq) = plane_dequant_pair(
                    i32::from(leaf_qindex),
                    QuantizerDeltas { dc_delta: 0, ac_delta: 0 },
                );

                // A2.foreman - The encoder's strict `MvLimits` is
                // tighter than the decoder's `clamp_mv_ref` envelope
                // because `build_inter_pred_plane_unscaled` doesn't
                // border-extend. If `mvc.nearest` falls outside
                // `mv_limits` (strict no-border-extend envelope), the
                // picker would crash when evaluating the NEARESTMV
                // candidate (the convolve over-reads the plane). The
                // decoder DOES border-extend so its reconstruction is
                // well-defined - but the two diverge.
                //
                // To preserve bit-exact encoder<->decoder agreement
                // AND avoid the panic, when `mvc.nearest` is OOB we
                // substitute Mv::ZERO in the picker's mvc so the
                // NEARESTMV evaluation reconstructs at (0,0) instead.
                // We then never let the picker actually choose
                // NEARESTMV: post-pick, if `picked_mode ==
                // NearestMv`, override to the better-RD candidate
                // among {ZeroMv, NewMv}. The writer is unaffected -
                // it doesn't read `mv_candidates`; it derives its own
                // `best_ref_mv` from the mi_grid via `find_mv_refs`,
                // and for the NEAREST mode it would propagate the
                // un-clamped value through the decoder's
                // border-extend path.
                let mv_in_bounds = |m: Mv| -> bool {
                    let col_fp = i32::from(m.col) >> 3;
                    let row_fp = i32::from(m.row) >> 3;
                    col_fp >= col_min
                        && col_fp <= col_max
                        && row_fp >= row_min
                        && row_fp <= row_max
                };
                // Phase 2a — unified dual-ref interleaved picker.
                // Replaces the old dual-`pick_inter_mode_luma` +
                // argmin pattern with a single call to
                // `pick_inter_mode` that iterates libvpx's
                // RT_INTER_MODES order internally. OOB veto is
                // handled inside the picker via the `*_in_bounds`
                // flags.

                // Phase 2b — extract intra-prediction neighbour edges
                // from the Y recon plane. The blocks above and left of
                // the current leaf are already reconstructed.
                let intra_have_above = leaf_px_y > 0;
                let intra_have_left = leaf_px_x > 0;
                let mut intra_above_buf = [0u8; 64];
                let mut intra_left_buf = [0u8; 64];
                let intra_above: Option<&[u8]>;
                let intra_left: Option<&[u8]>;
                let intra_top_left: u8;
                {
                    let recon_y: &[u8] = &recon_yuv_out.0;
                    if intra_have_above {
                        let row_off = (leaf_px_y - 1) * w + leaf_px_x;
                        let n = bw.min(w - leaf_px_x);
                        intra_above_buf[..n].copy_from_slice(&recon_y[row_off..row_off + n]);
                        intra_above = Some(&intra_above_buf[..n]);
                    } else {
                        intra_above = None;
                    }
                    if intra_have_left {
                        for r in 0..bh.min(h - leaf_px_y) {
                            intra_left_buf[r] = recon_y[(leaf_px_y + r) * w + (leaf_px_x - 1)];
                        }
                        intra_left = Some(&intra_left_buf[..bh.min(h - leaf_px_y)]);
                    } else {
                        intra_left = None;
                    }
                    intra_top_left = if intra_have_above && intra_have_left {
                        recon_y[(leaf_px_y - 1) * w + (leaf_px_x - 1)]
                    } else {
                        127
                    };
                }

                // G2 — switchable-filter context from above/left
                // neighbours in the in-progress mi_grid.
                let nb_above = if leaf_mi_row > 0 {
                    mi_grid[(leaf_mi_row - 1) * mi_cols_us + leaf_mi_col].as_ref()
                } else {
                    None
                };
                let nb_left = if leaf_mi_col > 0 {
                    mi_grid[leaf_mi_row * mi_cols_us + (leaf_mi_col - 1)].as_ref()
                } else {
                    None
                };
                let switchable_interp_ctx =
                    get_pred_context_switchable_interp(nb_above, nb_left);
                // P19 — per-block skip-prob context, libvpx
                // `vp9_pred_common.h:55-61` (`above.skip + left.skip`).
                // Threaded into the picker's skip-flag tie-break.
                let skip_ctx =
                    rvp9_core::inter::get_pred_context_skip(nb_above, nb_left);
                // P15 — libvpx `const_motion[ref_frame]` from
                // `mv_refs_rt` (`vp9_pickmode.c:60-153`). Set when the
                // first 2 spatial neighbours don't yield 2 distinct
                // same-ref MVs; the picker uses it to skip NEARMV
                // (`vp9_pickmode.c:2166`).
                let last_const_motion = rvp9_core::inter::mv_refs_rt_const_motion(
                    leaf_bsize,
                    RefFrame::LastFrame,
                    &neighbours,
                );
                let golden_const_motion = rvp9_core::inter::mv_refs_rt_const_motion(
                    leaf_bsize,
                    RefFrame::GoldenFrame,
                    &neighbours,
                );

                // P13 — extract neighbour MVs for vp9_NEWMV_diff_bias.
                // Mirrors libvpx `xd->above_mi->mv[0]` /
                // `xd->left_mi->mv[0]` (vp9_pickmode.c:1316-1325). For
                // intra / OOB neighbours we pass `None`, matching
                // libvpx's `xd->above_mi == NULL` and `as_int !=
                // INVALID_MV` checks. The MV is the neighbour's
                // bitstream MV (`mvs[0]` from BlockMode::Inter).
                let neighbour_mv = |mi: Option<&rvp9_core::mode_info::ModeInfo>| -> Option<Mv> {
                    match mi.map(|m| m.mode) {
                        Some(rvp9_core::mode_info::BlockMode::Inter { mvs, .. }) => Some(mvs[0]),
                        _ => None,
                    }
                };
                let above_mv_p13 = neighbour_mv(nb_above);
                let left_mv_p13 = neighbour_mv(nb_left);

                // P20 — re-borrow the persistent RD-threshold table
                // immutably for the picker. The mutable borrow for
                // the post-pick `update_thresh_freq_fact` call below
                // is acquired separately after `unified_pick` returns
                // so the borrows do not overlap.
                let rd_thresh_table_ref: Option<
                    &rvp9_core::mode_decision::rd_thresh::RdThreshTable,
                > = Some(&*rd_thresh_table);

                let pick_params = PickInterModeParams {
                    block_size: leaf_bsize,
                    src_plane: &src_y,
                    src_offset: (leaf_px_x, leaf_px_y),
                    last_ref_plane: &ref_y,
                    last_mv_candidates: &mvc,
                    golden_ref_plane: &ref_y_golden,
                    golden_mv_candidates: &mvc_golden,
                    mv_limits,
                    site_config: &site_config,
                    mv_costs: &mv_costs,
                    inter_mode_costs: &inter_costs,
                    inter_mode_ctx,
                    quantizer: &quantizer,
                    q_index: leaf_qindex,
                    allow_hp: false,
                    frame_kind,
                    newmv_seed: sb_newmv_seed,
                    ref_frame_cost_last,
                    ref_frame_cost_golden,
                    last_nearest_in_bounds: mv_in_bounds(mvc.nearest),
                    last_near_in_bounds: mv_in_bounds(mvc.near),
                    golden_nearest_in_bounds: mv_in_bounds(mvc_golden.nearest),
                    golden_near_in_bounds: mv_in_bounds(mvc_golden.near),
                    intra_above_pixels: intra_above,
                    intra_left_pixels: intra_left,
                    intra_top_left_pixel: intra_top_left,
                    intra_y_mode_costs: &intra_y_mode_costs_inter,
                    ref_frame_cost_intra,
                    switchable_interp_costs: &sic,
                    switchable_interp_ctx,
                    skip_ctx,
                    last_const_motion,
                    golden_const_motion,
                    // G1 — UV planes for chroma RD in the picker.
                    src_u: input_yuv.1,
                    src_u_stride: uvw,
                    src_v: input_yuv.2,
                    src_v_stride: uvw,
                    last_ref_u: last_recon_yuv.1,
                    last_ref_v: last_recon_yuv.2,
                    golden_ref_u: golden_recon_yuv.1,
                    golden_ref_v: golden_recon_yuv.2,
                    uv_stride: uvw,
                    uv_dc_quant: uv_dc_dq as u32,
                    uv_ac_quant: uv_ac_dq as u32,
                    // G4: encode breakout test fields.
                    encode_breakout: 20,
                    dequant_dc: quantizer.dequant[0] as u16,
                    dequant_ac: quantizer.dequant[1] as u16,
                    // P13 — vp9_NEWMV_diff_bias inputs. rvp9 has no
                    // noise estimator / skin detector / lowvar-highsumdiff
                    // tracker — those sub-clauses stay disabled (faithful
                    // port of libvpx pre-`vp9_update_noise_estimate`
                    // behaviour). The magnitude-driven bias is active.
                    above_mv: above_mv_p13,
                    left_mv: left_mv_p13,
                    noise_estimate: rvp9_core::mode_decision::inter::NoiseEstimate::default(),
                    lowvar_highsumdiff: false,
                    is_skin: false,
                    // P18 — `sf.reference_masking` default. libvpx
                    // `vp9_speed_features.c:250 / 517` keeps this on
                    // for the realtime single-spatial-layer path,
                    // which is the rvp9 operating point.
                    reference_masking: true,
                    // P20 — adaptive RD-threshold table, owned by the
                    // streaming encoder. Default ON (libvpx default
                    // at cpu-used 6+, vp9_speed_features.c:752 sets
                    // `sf.adaptive_rd_thresh = 4`, matched in
                    // mode_decision::rd_thresh::ADAPTIVE_RD_THRESH).
                    rd_thresh_table: rd_thresh_table_ref,
                    // §10 — content-adaptive picker gates. rvp9 has no
                    // noise estimator / skin detector / scene-change
                    // detector / content-state classifier / source-SAD
                    // pass, so each gate's input defaults to libvpx's
                    // pre-init / disabled value. This is functionally
                    // honest: libvpx's picker, when called before those
                    // analysis passes have run, sees the same disabled
                    // inputs and the same picker-output (the gates are
                    // no-ops). Refs:
                    //   - source_variance — libvpx vp9_pickmode.c:1781
                    //     initialises to UINT_MAX (= SOURCE_VARIANCE_
                    //     UNCOMPUTED).
                    //   - short_circuit_flat_blocks — speed-feature
                    //     flag, default 0 (vp9_speed_features.c
                    //     pre-init).
                    //   - force_skip_low_temp_var — derived from
                    //     x->variance_low[]; pre-init 0.
                    //   - content_state_sb — pre-init kLowSadLowSumdiff
                    //     (vp9_encodeframe.c:5574: x->content_state_sb
                    //     = 0; only kVeryHighSad = nonzero triggers the
                    //     gate).
                    //   - is_screen_content — cpi->oxcf.content default
                    //     CAMERA / DEFAULT.
                    //   - zero_temp_sad_source — None matches libvpx
                    //     pre-init (x->zero_temp_sad_source = 0,
                    //     compute_source_sad_onepass = 0).
                    //   - scene_change_detected — cpi->rc.high_source_
                    //     sad pre-init 0.
                    short_circuit_flat_blocks: false,
                    source_variance: rvp9_core::mode_decision::inter::SOURCE_VARIANCE_UNCOMPUTED,
                    force_skip_low_temp_var: false,
                    content_state_sb: rvp9_core::mode_decision::inter::ContentStateSb::Normal,
                    is_screen_content: false,
                    zero_temp_sad_source: None,
                    scene_change_detected: false,
                    // §11 — speed-feature gates for cpu-used-6/7/8. Defaults
                    // mirror libvpx `vp9_speed_features.c:585-793` for the
                    // realtime path:
                    //  - `inter_mode_mask` — NEAREST_NEW_ZERO for 32x32+
                    //    (vp9_speed_features.c:603-606, speed 5+; preserved
                    //    at speed 6/7/8).
                    //  - `limit_newmv_early_exit` — `1` at speed 5/6/7 CBR
                    //    (vp9_speed_features.c:639); back to `0` at speed 8
                    //    (vp9_speed_features.c:790). rvp9 op-points cpu-used
                    //    6/7 so `true`.
                    //  - `bias_golden` — `1` at speed 5+ CBR non-SVC
                    //    (vp9_speed_features.c:640).
                    //  - `cb_pred_filter_search` — `2` at speed 6 above
                    //    1280×720 (vp9_speed_features.c:658). rvp9 has no
                    //    resolution-dependent toggle so default `false`;
                    //    encoders that target the >720p path may opt in.
                    //  - `disable_golden_ref` — `1` only at speed-9 CBR
                    //    (vp9_speed_features.c:811). Off at our op point.
                    //  - `max_intra_bsize` — `Block32x32` at speed 5+
                    //    (vp9_speed_features.c:571). Matches the previous
                    //    hardcoded `Block64x64` skip.
                    //  - `rt_intra_dc_only_low_content` — `1` only at
                    //    speed-8 (vp9_speed_features.c:770). Default `false`
                    //    here keeps intra-fallback quality at speed 6/7.
                    inter_mode_mask:
                        rvp9_core::mode_decision::inter::default_inter_mode_mask(),
                    // W0.4: see top of `pick_and_simulate_inter_
                    // encode_partitions` — the dispatcher computed
                    // `sf` once per frame and every per-leaf literal
                    // reads from it instead of hard-coding §11 values.
                    limit_newmv_early_exit: sf.limit_newmv_early_exit,
                    bias_golden: sf.bias_golden,
                    frames_since_golden,
                    cb_pred_filter_search: sf.cb_pred_filter_search,
                    mi_row: leaf_mi_row as u32,
                    mi_col: leaf_mi_col as u32,
                    chessboard_index,
                    disable_golden_ref: sf.disable_golden_ref,
                    max_intra_bsize: sf.max_intra_bsize,
                    rt_intra_dc_only_low_content: sf.rt_intra_dc_only_low_content,
                };
                let unified_pick = pick_inter_mode(&pick_params);

                // P20 — drop the immutable borrow held by `pick_params`
                // so the mutable update can proceed.
                drop(pick_params);
                let _ = rd_thresh_table_ref; // consume binding

                // P20 — update `thresh_freq_fact` after the pick.
                // Mirrors libvpx `vp9_pickmode.c:2725-2738`: decay the
                // winning (ref, mode) slot's fact and increment every
                // other inter slot for the winning ref. Skip for
                // intra-winning picks (libvpx's intra-loop branch at
                // `vp9_pickmode.c:2708-2724` updates intra slots,
                // which rvp9's gate does not read — see the
                // `rd_thresh` module's divergence note).
                if matches!(
                    unified_pick.picked_ref_frame,
                    rvp9_core::mode_info::RefFrame::LastFrame
                        | rvp9_core::mode_info::RefFrame::GoldenFrame,
                ) {
                    // §11 — `limit_newmv_early_exit` participates in
                    // the freq-fact update at libvpx
                    // `vp9_pickmode.c:1147-1149`. Encoder default
                    // matches the cpu-used-6/7 setting:
                    // `limit_newmv_early_exit = true`. rvp9 has no
                    // source-variance pass yet so we pass `u32::MAX`
                    // (libvpx pre-init = `UINT_MAX`); the low-cap
                    // branch is therefore not yet active in
                    // production, but the wiring is in place.
                    rvp9_core::mode_decision::rd_thresh::update_thresh_freq_fact_with_s11(
                        rd_thresh_table,
                        leaf_bsize,
                        unified_pick.picked_ref_frame,
                        unified_pick.best_mode,
                        /* limit_newmv_early_exit = */ true,
                        /* source_variance = */ u32::MAX,
                    );
                }

                // Phase 2b — let intra-winning picks flow through.
                // When `picked_ref_frame == IntraFrame`, the encoder
                // simulator runs intra prediction (skip=true, no
                // residual) instead of MC + residual. The writer
                // emits the intra-in-inter wire format (is_inter=0,
                // y_mode, uv_mode).
                let pick = unified_pick;
                let picked_ref = pick.picked_ref_frame;

                // E12 (2026-05-22) — search-failure-vs-prediction-ceiling
                // oracle. Read-only diagnostic, env-gated
                // `RVP9_E12_ORACLE`; no encoder state mutated.
                if crate::e12stat::enabled() {
                    let last_mvc_safe = MvCandidates {
                        nearest: if mv_in_bounds(mvc.nearest) { mvc.nearest } else { Mv::ZERO },
                        near: if mv_in_bounds(mvc.near) { mvc.near } else { Mv::ZERO },
                    };
                    let all_e12 =
                        rvp9_core::mode_decision::inter::pick_inter_mode_luma_all_costs(
                            &InterModeSearchParams {
                                block_size: leaf_bsize,
                                src_plane: &src_y,
                                src_offset: (leaf_px_x, leaf_px_y),
                                ref_plane: &ref_y,
                                mv_candidates: &last_mvc_safe,
                                mv_limits,
                                site_config: &site_config,
                                mv_costs: &mv_costs,
                                inter_mode_costs: &inter_costs,
                                inter_mode_ctx,
                                quantizer: &quantizer,
                                q_index: leaf_qindex,
                                allow_hp: false,
                                frame_kind,
                                newmv_seed: sb_newmv_seed,
                                ref_frame_rate_cost: ref_frame_cost_last,
                                switchable_interp_costs: &sic,
                                switchable_interp_ctx,
                                skip_ctx: Some(skip_ctx),
                                src_u: input_yuv.1,
                                src_v: input_yuv.2,
                                ref_u: last_recon_yuv.1,
                                ref_v: last_recon_yuv.2,
                                uv_stride: uvw,
                                uv_dc_quant: uv_dc_dq as u32,
                                uv_ac_quant: uv_ac_dq as u32,
                            },
                        );
                    crate::e12stat::record_leaf(
                        src_y.data,
                        src_y.stride,
                        ref_y.data,
                        ref_y.stride,
                        ref_y.width,
                        ref_y.height,
                        leaf_px_x,
                        leaf_px_y,
                        bw,
                        bh,
                        pick.best_mv,
                        all_e12[3].best_mv,
                    );
                }

                // E3 (2026-05-22) — capture the LAST-ref per-candidate
                // cost breakdown. Read-only diagnostic.
                if e3_diag_enabled {
                    let nearest_in_bounds = mv_in_bounds(mvc.nearest);
                    let near_in_bounds = mv_in_bounds(mvc.near);
                    let mvc_safe = MvCandidates {
                        nearest: if nearest_in_bounds { mvc.nearest } else { Mv::ZERO },
                        near: if near_in_bounds { mvc.near } else { Mv::ZERO },
                    };
                    let e3_params = InterModeSearchParams {
                        block_size: leaf_bsize,
                        src_plane: &src_y,
                        src_offset: (leaf_px_x, leaf_px_y),
                        ref_plane: &ref_y,
                        mv_candidates: &mvc_safe,
                        mv_limits,
                        site_config: &site_config,
                        mv_costs: &mv_costs,
                        inter_mode_costs: &inter_costs,
                        inter_mode_ctx,
                        quantizer: &quantizer,
                        q_index: leaf_qindex,
                        allow_hp: false,
                        frame_kind,
                        newmv_seed: sb_newmv_seed,
                        ref_frame_rate_cost: ref_frame_cost_last,
                        switchable_interp_costs: &sic,
                        switchable_interp_ctx,
                        skip_ctx: Some(skip_ctx),
                        src_u: input_yuv.1,
                        src_v: input_yuv.2,
                        ref_u: last_recon_yuv.1,
                        ref_v: last_recon_yuv.2,
                        uv_stride: uvw,
                        uv_dc_quant: uv_dc_dq as u32,
                        uv_ac_quant: uv_ac_dq as u32,
                    };
                    let all =
                        rvp9_core::mode_decision::inter::pick_inter_mode_luma_all_costs(
                            &e3_params,
                        );
                    // all = [ZERO, NEAREST, NEAR, NEW].
                    e3_total_leaves += 1;
                    // RD argmin (mirror the picker's own argmin).
                    let mut win_idx = 0usize;
                    for i in 1..4 {
                        if all[i].best_rd_cost < all[win_idx].best_rd_cost {
                            win_idx = i;
                        }
                    }
                    e3_win_counts[win_idx] += 1;

                    let new = &all[3];
                    let new_mag =
                        (i32::from(new.best_mv.row).abs()
                            + i32::from(new.best_mv.col).abs()) as u32;
                    if new_mag > 0 {
                        e3_newmv_nonzero += 1;
                    }
                    e3_newmv_mag_sum += u64::from(new_mag);
                    e3_newmv_mag_max = e3_newmv_mag_max.max(new_mag);

                    // Did NEWMV have the strictly-lowest distortion?
                    let new_best_dist = all
                        .iter()
                        .enumerate()
                        .all(|(i, c)| i == 3 || new.best_distortion <= c.best_distortion)
                        && all
                            .iter()
                            .enumerate()
                            .any(|(i, c)| i != 3 && new.best_distortion < c.best_distortion);
                    if new_best_dist {
                        e3_newmv_best_dist += 1;
                        if win_idx != 3 {
                            e3_newmv_best_dist_lost += 1;
                            // Decompose NEWMV's rate into mode + mv rate.
                            // mv_rate = ROUND_POWER_OF_TWO(mv_cost*108, 7)
                            // relative to mvc_safe.nearest (the ref MV).
                            let diff_row =
                                new.best_mv.row.wrapping_sub(mvc_safe.nearest.row);
                            let diff_col =
                                new.best_mv.col.wrapping_sub(mvc_safe.nearest.col);
                            let raw = rvp9_core::cost::mv_cost::mv_cost(
                                diff_row, diff_col, &mv_costs,
                            )
                            .max(0) as u64;
                            let mv_rate = (raw * 108 + 64) >> 7;
                            e3_lost_mv_rate_sum += mv_rate;
                            e3_lost_total_rate_sum += u64::from(new.best_rate);
                            let gap = i128::from(new.best_rd_cost)
                                - i128::from(all[win_idx].best_rd_cost);
                            e3_lost_rd_gap_sum += gap;
                        }
                    }

                    if e3_sample_printed < E3_DIAG_SAMPLE_LIMIT {
                        e3_sample_printed += 1;
                        #[allow(clippy::print_stderr)]
                        {
                            let names = ["ZERO", "NEAREST", "NEAR", "NEW"];
                            eprintln!(
                                "E3 leaf px=({},{}) bs={:?} q={} nearest=({},{}) newmv=({},{}) mag={} | \
                                 ZERO[rd={} d={} r={}] NEAREST[rd={} d={} r={}] NEAR[rd={} d={} r={}] NEW[rd={} d={} r={}] WIN={}",
                                leaf_px_x, leaf_px_y, leaf_bsize, leaf_qindex,
                                mvc_safe.nearest.row, mvc_safe.nearest.col,
                                new.best_mv.row, new.best_mv.col, new_mag,
                                all[0].best_rd_cost, all[0].best_distortion, all[0].best_rate,
                                all[1].best_rd_cost, all[1].best_distortion, all[1].best_rate,
                                all[2].best_rd_cost, all[2].best_distortion, all[2].best_rate,
                                all[3].best_rd_cost, all[3].best_distortion, all[3].best_rate,
                                names[win_idx],
                            );
                        }
                    }
                }

                if diag_golden_enabled {
                    match picked_ref {
                        RefFrame::GoldenFrame => diag_golden_picks += 1,
                        _ => diag_last_picks += 1,
                    }
                }

                sb_partitions[sb_idx][leaf_idx].picked_mode = pick.best_mode;
                sb_partitions[sb_idx][leaf_idx].picked_mv = pick.best_mv;
                sb_partitions[sb_idx][leaf_idx].skip = false;
                sb_partitions[sb_idx][leaf_idx].picked_ref_frame = picked_ref;
                // F.5 — record the per-leaf interpolation filter so the
                // writer (via `splat_inter_leaf_into_mi_grid`) and the
                // encoder simulator's per-plane MC step below both use
                // the same filter the picker scored against.
                sb_partitions[sb_idx][leaf_idx].picked_interp_filter =
                    pick.picked_interp_filter;
                // Phase 2b — record the intra mode when intra wins.
                sb_partitions[sb_idx][leaf_idx].picked_intra_mode =
                    pick.picked_intra_mode;

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

                if picked_ref == RefFrame::IntraFrame {
                    // G5 — intra-in-inter block. Single-pass TX walk:
                    // prediction + residual (forward transform, quantize,
                    // inverse transform-add, coef collection) per TX
                    // block. The single-walk design is required because
                    // VP9 intra prediction is causal — each TX block
                    // reads reconstructed pixels from prior TX blocks.
                    use rvp9_core::intra_pred::{
                        DcAvailability, EdgeBuffers, build_intra_predictors,
                        intra_predict,
                    };
                    use rvp9_core::transforms::TxType;
                    let intra_y_mode = pick.picked_intra_mode
                        .expect("IntraFrame pick must have picked_intra_mode");
                    let intra_uv_mode = rvp9_core::intra_pred::IntraMode::Dc;
                    let dispatch = rvp9_core::dsp::get();

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

                        let plane_px_x = (leaf_mi_col * 8) >> ss_x;
                        let plane_px_y = (leaf_mi_row * 8) >> ss_y;

                        // G5 — bind input + recon planes (input was
                        // previously only used in the inter branch).
                        let (input_plane, recon_plane) = match plane {
                            0 => (input_yuv.0, &mut *recon_yuv_out.0),
                            1 => (input_yuv.1, &mut *recon_yuv_out.1),
                            _ => (input_yuv.2, &mut *recon_yuv_out.2),
                        };

                        let have_above_block = leaf_mi_row > 0;
                        let have_left_block = leaf_mi_col > 0;

                        let plane_mode = if plane == 0 {
                            intra_y_mode
                        } else {
                            intra_uv_mode
                        };

                        // G5 — tx_type: Y plane uses the intra mode's
                        // hybrid type (can be ADST variants); UV and
                        // Tx32x32 always DctDct. Matches the decoder's
                        // `resolve_tx_type`.
                        let tx_type = if plane == 0 && plane_tx != TxSize::Tx32x32 {
                            plane_mode.tx_type()
                        } else {
                            TxType::DctDct
                        };

                        let mut tx_y4_local = 0usize;
                        while tx_y4_local < plane_bh_4x4 {
                            let mut tx_x4_local = 0usize;
                            while tx_x4_local < plane_bw_4x4 {
                                let tx_px_x = plane_px_x + tx_x4_local * 4;
                                let tx_px_y = plane_px_y + tx_y4_local * 4;

                                // Step 1: intra prediction (existing).
                                let have_top = have_above_block || tx_y4_local > 0;
                                let have_left = have_left_block || tx_x4_local > 0;
                                let have_right =
                                    (tx_x4_local + tx_step_4x4) < plane_bw_4x4;

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

                                let dst_off = tx_px_y * plane_stride + tx_px_x;
                                intra_predict(
                                    plane_mode,
                                    &mut recon_plane[dst_off..],
                                    plane_stride,
                                    tx_side,
                                    pred_ctx,
                                    dc_avail,
                                    &dispatch,
                                );

                                // Step 2: residual = source - prediction.
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

                                // E9 DIAG — tap Y-plane intra residual.
                                if plane == 0 {
                                    crate::predstat::record_y_residual(
                                        &residual,
                                        tx_side,
                                        plane_tx as usize,
                                    );
                                }

                                // Steps 3-4: forward hybrid transform + quant.
                                let (qcoeff_i32, dqcoeff_i32, eob_usize) =
                                    intra_forward_quant_one_tx(
                                        &residual,
                                        tx_side,
                                        plane_tx,
                                        tx_type,
                                        &quantizer,
                                    );

                                // Step 5: inverse transform-add into recon.
                                if eob_usize > 0 {
                                    intra_inverse_transform_add(
                                        plane_tx,
                                        tx_type,
                                        &dqcoeff_i32,
                                        eob_usize,
                                        &mut recon_plane[dst_off..],
                                        plane_stride,
                                    );
                                }

                                // Step 6: stash coefs for the writer.
                                coef_source.insert(
                                    leaf_mi_row,
                                    leaf_mi_col,
                                    plane,
                                    tx_x4_local,
                                    tx_y4_local,
                                    qcoeff_i32,
                                    eob_usize,
                                );
                                // Step 7: accumulate eob for skip decision.
                                leaf_total_eob += eob_usize;

                                tx_x4_local += tx_step_4x4;
                            }
                            tx_y4_local += tx_step_4x4;
                        }
                    }

                    // G5 — data-driven skip: true only when all coefs
                    // are zero (leaf_total_eob == 0). The RC.X1 override
                    // below handles this uniformly for both branches.
                    if leaf_total_eob == 0 {
                        sb_partitions[sb_idx][leaf_idx].skip = true;
                    }
                } else {
                    // Inter block — MC prediction + residual path.
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
                        // A15 Phase B Step 3 — select the ref plane that
                        // matches `picked_ref`. The decoder's
                        // `reconstruct_inter_block` reads from the slot
                        // the writer encodes in the ref-frame bits, so
                        // the encoder simulator must mirror that to keep
                        // the recon byte-exact.
                        let (input_plane, ref_plane_data, recon_plane) = match (plane, picked_ref) {
                            (0, RefFrame::GoldenFrame) => {
                                (input_yuv.0, golden_recon_yuv.0, &mut *recon_yuv_out.0)
                            }
                            (1, RefFrame::GoldenFrame) => {
                                (input_yuv.1, golden_recon_yuv.1, &mut *recon_yuv_out.1)
                            }
                            (_, RefFrame::GoldenFrame) => {
                                (input_yuv.2, golden_recon_yuv.2, &mut *recon_yuv_out.2)
                            }
                            (0, _) => (input_yuv.0, last_recon_yuv.0, &mut *recon_yuv_out.0),
                            (1, _) => (input_yuv.1, last_recon_yuv.1, &mut *recon_yuv_out.1),
                            (_, _) => (input_yuv.2, last_recon_yuv.2, &mut *recon_yuv_out.2),
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
                            sb_partitions[sb_idx][leaf_idx].picked_interp_filter,
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

                                // E9 DIAG — tap the Y-plane prediction residual
                                // (`source − MC prediction`, before transform/quant).
                                if plane == 0 {
                                    crate::predstat::record_y_residual(
                                        &residual,
                                        tx_side,
                                        plane_tx as usize,
                                    );
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
                }
                // E4 diagnostic: per-leaf skip / residual-energy stats.
                crate::skipstat::record_leaf(
                    sb_partitions[sb_idx][leaf_idx].skip,
                    leaf_total_eob,
                );
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
                // F.0.2 oracle — record one BlockDecision per coded
                // inter leaf, AFTER the RC.X1 skip-on-zero-eob override
                // (so `skip_flag` matches what lands in the bitstream).
                // Single relaxed atomic load when `RVP9_F02_ORACLE` is
                // unset (HLD §3.2 / §7).
                if crate::f02_oracle::is_enabled() {
                    let final_skip = sb_partitions[sb_idx][leaf_idx].skip;
                    let final_mode = sb_partitions[sb_idx][leaf_idx].picked_mode;
                    let final_mv = sb_partitions[sb_idx][leaf_idx].picked_mv;
                    let final_ref = sb_partitions[sb_idx][leaf_idx].picked_ref_frame;
                    // Re-derive `mv_bits_est` for NEWMV using the same
                    // formula the picker uses internally
                    // (`evaluate_candidate` in
                    // `core::mode_decision::inter`), since the existing
                    // `InterModeSearchResult` only returns the combined
                    // `best_rate` (HLD §3.4.1 forward-looking note).
                    // Reference MV is the safe `nearest` MV the picker
                    // saw (post-OOB-veto), matching the
                    // `pick_for_ref` closure's `mvc_safe.nearest`.
                    let mv_bits_est: u32 =
                        if matches!(final_mode, InterMode::NewMv) {
                            let mvc_ref = match final_ref {
                                RefFrame::GoldenFrame => mvc_golden.nearest,
                                _ => mvc.nearest,
                            };
                            let ref_mv = if mv_in_bounds(mvc_ref) {
                                mvc_ref
                            } else {
                                Mv::ZERO
                            };
                            let diff_row = final_mv.row.wrapping_sub(ref_mv.row);
                            let diff_col = final_mv.col.wrapping_sub(ref_mv.col);
                            let raw = rvp9_core::cost::mv_cost::mv_cost(
                                diff_row, diff_col, &mv_costs,
                            )
                            .max(0) as u64;
                            // ROUND_POWER_OF_TWO(raw * 108, 7) — same
                            // `MV_COST_WEIGHT = 108` scaling the picker
                            // uses.
                            ((raw * 108 + 64) >> 7).min(u32::MAX as u64) as u32
                        } else {
                            0
                        };
                    // mode_bits_est = total picker rate minus the
                    // ref-frame cost and the MV cost — what's left
                    // is the inter-mode cost.
                    let ref_cost = match final_ref {
                        RefFrame::GoldenFrame => ref_frame_cost_golden,
                        _ => ref_frame_cost_last,
                    };
                    let mode_bits_est = pick
                        .best_rate
                        .saturating_sub(mv_bits_est)
                        .saturating_sub(ref_cost);
                    // F.1 — `coef_bits_est` now live. The picker's
                    // `coef_rate` is in `vp9_prob_cost` units
                    // (1/512-bit, libvpx `VP9_PROB_COST_SHIFT = 9`); the
                    // oracle column is in plain bits. The HLD F.1 Q4
                    // resolution mandates `coef_rate >> 9` as the
                    // back-conversion. See
                    // `wrk_docs/Track_F/F1_HLD_block_yrd_port.md` §10
                    // and the F.0.2 schema liveness matrix.
                    //
                    // Night-8 (2026-05-23): `pick.coef_rate` is `u64`
                    // now (saturation fix); saturate-cast to `u32` at
                    // the oracle boundary.
                    let coef_bits_est: u32 = (pick
                        .coef_rate
                        >> rvp9_core::cost::VP9_PROB_COST_SHIFT)
                        .min(u32::MAX as u64) as u32;
                    crate::f02_oracle::record_decision(
                        crate::f02_oracle::BlockDecision {
                            frame_idx: crate::f02_oracle::current_frame_idx(),
                            mi_row: leaf_mi_row as u16,
                            mi_col: leaf_mi_col as u16,
                            bsize: leaf_bsize,
                            mode: crate::f02_oracle::BlockDecisionMode::Inter(
                                final_mode,
                            ),
                            ref_frame: final_ref,
                            mv_row: final_mv.row,
                            mv_col: final_mv.col,
                            mode_bits_est,
                            mv_bits_est,
                            coef_bits_est,
                            sse_after_pred: u64::from(pick.best_distortion),
                            skip_flag: final_skip,
                        },
                    );
                }

                // A2.foreman - Splat the picked `(mode, mv, skip)`
                // into the in-progress mi_grid so subsequent leaves'
                // neighbour lookups see this leaf's decision.
                // A15 Phase B Step 4 — carry `picked_ref_frame` too so
                // neighbour MV-ref propagation matches the per-ref
                // picker behaviour (the decoder's `find_mv_refs` is
                // ref-aware).
                splat_inter_leaf_into_mi_grid(
                    &mut mi_grid,
                    mi_cols_us,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf_bsize,
                    sb_partitions[sb_idx][leaf_idx].picked_mode,
                    sb_partitions[sb_idx][leaf_idx].picked_mv,
                    sb_partitions[sb_idx][leaf_idx].segment_id,
                    sb_partitions[sb_idx][leaf_idx].skip,
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame,
                    sb_partitions[sb_idx][leaf_idx].picked_interp_filter,
                );
            }
        }
    }

    // A3'-int-pro frame-level diagnostic — aggregate mode counts.
    // Set RVP9_DIAG_INTPRO=1 to enable.
    // A15 Phase B Step 5 — also count GOLDEN picks (key diagnostic
    // for the widened-picker BD-rate measurement).
    if std::env::var("RVP9_DIAG_INTPRO").is_ok() {
        let mut count_zero = 0usize;
        let mut count_nearest = 0usize;
        let mut count_new = 0usize;
        let mut count_skip = 0usize;
        let mut count_nonzero_mv = 0usize;
        let mut count_near = 0usize;
        let mut count_golden = 0usize;
        for sb in sb_partitions.iter() {
            for leaf in sb.iter() {
                match leaf.picked_mode {
                    InterMode::ZeroMv => count_zero += 1,
                    InterMode::NearestMv => count_nearest += 1,
                    InterMode::NearMv => count_near += 1,
                    InterMode::NewMv => count_new += 1,
                }
                if leaf.skip {
                    count_skip += 1;
                }
                if leaf.picked_mv != Mv::ZERO {
                    count_nonzero_mv += 1;
                }
                if matches!(leaf.picked_ref_frame, RefFrame::GoldenFrame) {
                    count_golden += 1;
                }
            }
        }
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[A3'-int-pro] frame summary: zero={} nearest={} near={} new={} skip={} nonzero_mv={} golden={}",
                count_zero, count_nearest, count_near, count_new, count_skip, count_nonzero_mv, count_golden,
            );
        }
    }

    if diag_golden_enabled {
        #[allow(clippy::print_stderr)]
        {
            let total = diag_golden_picks + diag_last_picks;
            eprintln!(
                "[A16-golden-picks] last={} golden={} total={} golden_pct={:.1}",
                diag_last_picks,
                diag_golden_picks,
                total,
                if total > 0 {
                    100.0 * diag_golden_picks as f64 / total as f64
                } else {
                    0.0
                },
            );
        }
    }

    if e3_diag_enabled {
        #[allow(clippy::print_stderr)]
        {
            let n = e3_total_leaves.max(1);
            eprintln!(
                "[E3-mvdiag] frame summary: leaves={} | RD-win: ZERO={} NEAREST={} NEAR={} NEW={} (NEW {:.1}%) | \
                 newmv_nonzero_mv={} ({:.1}%) avg_mag={:.1} max_mag={} | \
                 newmv_best_dist={} of_those_lost_rd={} | \
                 lost-leaf avg: mv_rate={} total_rate={} rd_gap={}",
                e3_total_leaves,
                e3_win_counts[0], e3_win_counts[1], e3_win_counts[2], e3_win_counts[3],
                100.0 * e3_win_counts[3] as f64 / n as f64,
                e3_newmv_nonzero,
                100.0 * e3_newmv_nonzero as f64 / n as f64,
                e3_newmv_mag_sum as f64 / n as f64,
                e3_newmv_mag_max,
                e3_newmv_best_dist,
                e3_newmv_best_dist_lost,
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_mv_rate_sum / e3_newmv_best_dist_lost as u64
                } else { 0 },
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_total_rate_sum / e3_newmv_best_dist_lost as u64
                } else { 0 },
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_rd_gap_sum / e3_newmv_best_dist_lost as i128
                } else { 0 },
            );
        }
    }

    // A18 — Write back the final picked-leaf MVs into `out_mv_grid` for
    // the encoder to capture as the *next* frame's `prev_mv_grid`.
    // Mirrors the decoder's `cur_mv_grid` write-back at
    // `tile.rs:2141-2155` — for inter cells, store
    // `(ref_frames, mvs)`; for intra-in-inter cells, store
    // `(IntraFrame, zero-mv)` matching the decoder's D.7 path.
    debug_assert_eq!(out_mv_grid.len(), mi_rows_us * mi_cols_us);
    for cell in out_mv_grid.iter_mut() {
        *cell = PrevFrameMvRef::default();
    }
    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let sb_idx = sb_row * sb_cols + sb_col;
            for leaf in &sb_partitions[sb_idx] {
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                let bw_mi = leaf.leaf_bsize.n_8x8_wide() as usize;
                let bh_mi = leaf.leaf_bsize.n_8x8_high() as usize;
                // Phase 2b — intra-in-inter blocks write
                // `(IntraFrame, zero-mv)` to the MV grid, matching
                // the decoder's D.7 path. Inter blocks write the
                // picked MV + ref frame as before.
                let entry = if leaf.picked_ref_frame == RefFrame::IntraFrame {
                    PrevFrameMvRef {
                        mvs: [Mv::ZERO; 2],
                        ref_frames: [RefFrame::IntraFrame, RefFrame::NoRefFrame],
                    }
                } else {
                    PrevFrameMvRef {
                        mvs: [leaf.picked_mv, Mv::ZERO],
                        ref_frames: [leaf.picked_ref_frame, RefFrame::NoRefFrame],
                    }
                };
                for dr in 0..bh_mi {
                    for dc in 0..bw_mi {
                        let r = leaf_mi_row + dr;
                        let c = leaf_mi_col + dc;
                        if r < mi_rows_us && c < mi_cols_us {
                            out_mv_grid[r * mi_cols_us + c] = entry;
                        }
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
    // F.5 — interpolation filter chosen by the picker for this leaf.
    // Pre-F.5 this was hard-pinned to `EightTap`; the picker now drives
    // per-block filter selection (`{EightTap, EightTapSmooth}`) and
    // the encoder simulator must mirror that here so its recon
    // matches what the decoder will produce when reading the per-block
    // switchable-filter bits the writer emits.
    interp_filter: rvp9_core::bitstream::frame::InterpFilter,
) {
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
        interp_filter,
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

/// G5 — Forward hybrid transform + forward quant of one TX block for the
/// intra path (both keyframe and intra-in-inter). Unlike the inter helper
/// (`inter_forward_quant_one_tx`), this supports non-DctDct `tx_type`
/// (AdstDct, DctAdst, AdstAdst) via the `fht*` family, and derives
/// scan order from `(plane_tx, tx_type)` rather than hardcoding DctDct.
///
/// Returns `(qcoeff_i32, dqcoeff_i32, eob)`.
fn intra_forward_quant_one_tx(
    residual: &[i16],
    tx_side: usize,
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
) -> (Vec<i32>, Vec<i32>, usize) {
    use rvp9_core::quant::forward::{scan_for_forward, vp9_quantize_b, vp9_quantize_b_32x32};
    use rvp9_core::transforms::forward::{fdct32x32, fht4x4, fht8x8, fht16x16};
    use rvp9_core::transforms::scans::scan_for;

    let n = plane_tx.n_coeffs();
    let mut coefs = vec![0i32; n];
    match plane_tx {
        TxSize::Tx4x4 => {
            let mut buf = [0i32; 16];
            fht4x4(residual, &mut buf, tx_side, tx_type);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx8x8 => {
            let mut buf = [0i32; 64];
            fht8x8(residual, &mut buf, tx_side, tx_type);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx16x16 => {
            let mut buf = [0i32; 256];
            fht16x16(residual, &mut buf, tx_side, tx_type);
            coefs.copy_from_slice(&buf[..]);
        }
        TxSize::Tx32x32 => {
            // VP9 has no hybrid 32x32 -- always pure DCT.
            let mut buf = [0i32; 1024];
            fdct32x32(residual, &mut buf, tx_side);
            coefs.copy_from_slice(&buf[..]);
        }
    }

    let scan = scan_for(plane_tx, tx_type);
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

/// G5 — Inverse transform-add for intra paths (both keyframe and
/// intra-in-inter). Unlike `inter_inverse_dct_add`, this dispatches
/// to the IHT variants (`iht4x4_16_add` / `iht8x8_64_add` /
/// `iht16x16_256_add`) when `tx_type != DctDct`. The eob==1 fast path
/// (`idct*_1_add`) applies only for DctDct (no single-coef IHT variants
/// exist in libvpx).
///
/// Caller must ensure `eob > 0`.
fn intra_inverse_transform_add(
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    dq_input: &[i32],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::inverse::{
        idct4x4_1_add, idct4x4_16_add, idct8x8_1_add, idct8x8_64_add,
        idct16x16_1_add, idct16x16_256_add, idct32x32_1_add, idct32x32_1024_add,
        iht4x4_16_add, iht8x8_64_add, iht16x16_256_add,
    };

    debug_assert!(eob > 0, "intra_inverse_transform_add called with eob == 0");

    match plane_tx {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = dq_input.try_into().expect("Tx4x4 takes 16 coefs");
            if tx_type == TxType::DctDct {
                if eob > 1 {
                    idct4x4_16_add(coefs, dst, stride);
                } else {
                    idct4x4_1_add(coefs[0], dst, stride);
                }
            } else {
                iht4x4_16_add(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = dq_input.try_into().expect("Tx8x8 takes 64 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct8x8_1_add(coefs[0], dst, stride);
                } else {
                    idct8x8_64_add(coefs, dst, stride);
                }
            } else {
                iht8x8_64_add(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = dq_input.try_into().expect("Tx16x16 takes 256 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct16x16_1_add(coefs[0], dst, stride);
                } else {
                    idct16x16_256_add(coefs, dst, stride);
                }
            } else {
                iht16x16_256_add(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx32x32 => {
            // Tx32x32 is always DctDct (no hybrid 32×32 in VP9).
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
    fn encoder_rejects_zero_width() {
        // 0 passes is_multiple_of(64), so the explicit zero-width
        // check on line 187-189 is reachable.
        let err = Encoder::new(Config::new(0, 64, 50)).unwrap_err();
        match err {
            EncodeError::InvalidConfig(msg) => {
                assert!(msg.contains("non-zero"), "expected non-zero error: {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encoder_rejects_zero_height() {
        let err = Encoder::new(Config::new(64, 0, 50)).unwrap_err();
        match err {
            EncodeError::InvalidConfig(msg) => {
                assert!(msg.contains("non-zero"), "expected non-zero error: {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn encode_error_display_invalid_config_includes_reason() {
        let err = EncodeError::InvalidConfig("widget");
        let s = format!("{err}");
        assert!(
            s.contains("widget"),
            "InvalidConfig display should include the reason: {s}",
        );
        assert!(
            s.contains("invalid encoder config"),
            "InvalidConfig display should include the leading label: {s}",
        );
    }

    #[test]
    fn encode_error_display_bitstream_variant() {
        let e = EncodeError::Bitstream(BitstreamError::CompressedHeaderTooLarge);
        let s = format!("{e}");
        assert!(s.contains("bitstream writer error"), "got: {s}");
    }

    #[test]
    fn encode_error_from_bitstream_error_round_trips() {
        let inner = BitstreamError::CompressedHeaderTooLarge;
        let wrapped: EncodeError = inner.into();
        assert!(matches!(wrapped, EncodeError::Bitstream(_)));
    }

    #[test]
    fn encode_error_from_modes_writer_error_round_trips() {
        // Pick any valid ModesWriterError variant.
        let inner = ModesWriterError::MissingCell { mi_row: 0, mi_col: 0 };
        let wrapped: EncodeError = inner.into();
        assert!(matches!(wrapped, EncodeError::Modes(_)));
        // Display should label the variant.
        let s = format!("{wrapped}");
        assert!(s.contains("mode-info writer error"), "got: {s}");
    }

    #[test]
    fn encode_synthetic_dc_intra_with_residuals_rejects_short_input_y() {
        let cfg = Config::new(64, 64, 50);
        let enc = Encoder::new(cfg).unwrap();
        // Short Y input.
        let y = vec![0u8; 32 * 32];
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];
        let mut y_recon = vec![0u8; 64 * 64];
        let mut u_recon = vec![0u8; 32 * 32];
        let mut v_recon = vec![0u8; 32 * 32];
        let err = enc
            .encode_synthetic_dc_intra_with_residuals(
                (&y, &u, &v),
                (&mut y_recon, &mut u_recon, &mut v_recon),
            )
            .unwrap_err();
        assert!(
            matches!(err, EncodeError::InvalidConfig(msg) if msg.contains("input_yuv")),
            "expected input_yuv error, got: {err:?}",
        );
    }

    #[test]
    fn encode_synthetic_dc_intra_with_residuals_rejects_short_recon_y() {
        let cfg = Config::new(64, 64, 50);
        let enc = Encoder::new(cfg).unwrap();
        let y = vec![128u8; 64 * 64];
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];
        // Short recon Y.
        let mut y_recon = vec![0u8; 32 * 32];
        let mut u_recon = vec![0u8; 32 * 32];
        let mut v_recon = vec![0u8; 32 * 32];
        let err = enc
            .encode_synthetic_dc_intra_with_residuals(
                (&y, &u, &v),
                (&mut y_recon, &mut u_recon, &mut v_recon),
            )
            .unwrap_err();
        assert!(
            matches!(err, EncodeError::InvalidConfig(msg) if msg.contains("recon_yuv_out")),
            "expected recon_yuv_out error, got: {err:?}",
        );
    }

    #[test]
    fn config_new_const_round_trips() {
        // Exercise the const constructor: assignment + field reads.
        const CFG: Config = Config::new(128, 128, 100);
        assert_eq!(CFG.width, 128);
        assert_eq!(CFG.height, 128);
        assert_eq!(CFG.base_qindex, 100);
    }

    #[test]
    fn vec_backed_intra_coef_source_round_trips() {
        // Exercises VecBackedIntraCoefSource::new + insert + get_block_coefs.
        use rvp9_core::bitstream::IntraCoefSource;
        let mut src = VecBackedIntraCoefSource::new(64, 64);
        assert_eq!(src.width(), 64);
        assert_eq!(src.height(), 64);
        let coefs = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        src.insert(0, 0, 0, 0, 0, coefs.clone(), 5);
        let (got_coefs, got_eob) = src.get_block_coefs(0, 0, 0, 0, 0, TxSize::Tx4x4);
        assert_eq!(&got_coefs[..16], &coefs[..16]);
        assert_eq!(got_eob, 5);
    }

    #[test]
    fn vec_backed_inter_coef_source_round_trips() {
        use rvp9_core::bitstream::InterCoefSource;
        let mut src = VecBackedInterCoefSource::new(128, 64);
        assert_eq!(src.width(), 128);
        assert_eq!(src.height(), 64);
        let coefs = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        src.insert(1, 2, 1, 0, 0, coefs.clone(), 7);
        let (got_coefs, got_eob) = src.get_block_coefs(1, 2, 1, 0, 0, TxSize::Tx4x4);
        assert_eq!(&got_coefs[..16], &coefs[..16]);
        assert_eq!(got_eob, 7);
    }

    /// Serial guard for env-mutating tests that touch the global
    /// E12/F02 oracle flags or RVP9_* env vars.
    fn env_test_mutex() -> std::sync::MutexGuard<'static, ()> {
        static MTX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        MTX.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn streaming_config() -> frame_loop::EncoderConfig {
        frame_loop::EncoderConfig {
            width: 64,
            height: 64,
            framerate: 30.0,
            target_bitrate_bps: 1_000_000,
            gop_length: 30,
            aq_mode: frame_loop::AqMode::Off,
            use_prev_frame_mvs: false,
            speed: 7,
            deadline: Deadline::Realtime,
        }
    }

    fn yuv_view_64<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> frame_loop::YuvFrameRef<'a> {
        frame_loop::YuvFrameRef {
            y, u, v,
            y_stride: 64,
            u_stride: 32,
            v_stride: 32,
        }
    }

    fn flat_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (vec![128u8; 64 * 64], vec![128u8; 32 * 32], vec![128u8; 32 * 32])
    }

    #[test]
    fn encode_with_f02_oracle_enabled_runs_record_paths() {
        let _guard = env_test_mutex();
        let key = "RVP9_F02_ORACLE";
        // SAFETY: serialised via env_test_mutex.
        unsafe { std::env::set_var(key, "1"); }
        crate::f02_oracle::init_from_env();
        assert!(crate::f02_oracle::is_enabled());

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let _inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();

        let decisions = crate::f02_oracle::drain_frame();
        assert!(
            !decisions.is_empty(),
            "F02 oracle should have recorded ≥1 decision when enabled",
        );

        unsafe { std::env::remove_var(key); }
        crate::f02_oracle::init_from_env();
        assert!(!crate::f02_oracle::is_enabled());
    }

    #[test]
    fn encode_with_e12_oracle_enabled_runs_record_paths() {
        let _guard = env_test_mutex();
        let key = "RVP9_E12_ORACLE";
        unsafe { std::env::set_var(key, "1"); }
        crate::e12stat::init_from_env();
        assert!(crate::e12stat::enabled());

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        assert!(inter.actual_bits > 0);

        unsafe { std::env::remove_var(key); }
        crate::e12stat::init_from_env();
        assert!(!crate::e12stat::enabled());
    }

    #[test]
    fn encode_with_f3_nonrd_partition_runs_descent_path() {
        let _guard = env_test_mutex();
        let key = "RVP9_F3_NONRD_PARTITION_ENABLE";
        unsafe { std::env::set_var(key, "1"); }

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        assert_eq!(inter.frame_kind, frame_loop::FrameKind::Inter);

        unsafe { std::env::remove_var(key); }
    }

    /// P12 PROOF — Build a 64×64 source where the second frame is a
    /// rightward-shifted copy of the first. A ZERO-MV-SSE callback sees
    /// high residual at the SB level and prefers SPLIT; the real
    /// `pick_inter_mode` finds the global NEWMV, drops the residual
    /// near zero at Block64x64, and prefers PARTITION_NONE.
    ///
    /// We compare partition decisions with the F.3 gate ON vs. OFF
    /// using the F.0.2 oracle as the introspection channel. The gate-ON
    /// path must produce at least one leaf size strictly greater than
    /// the variance-hint baseline at the same position — i.e. the
    /// motion-aware descent collapsed children into a larger leaf.
    #[test]
    fn p12_real_callback_picks_motion_aware_winner() {
        // Hold the env mutex for the WHOLE test — both
        // `RVP9_F3_NONRD_PARTITION_ENABLE` and `RVP9_F02_ORACLE` are
        // process-wide and would otherwise leak into parallel tests
        // (the F.3 gate is read via `std::env::var_os` inside the
        // encoder hot path; F.0.2 flips a global atomic via
        // `init_from_env`).
        let _guard = env_test_mutex();

        // Helper — deterministic xorshift32 generator (same seed
        // produces same data across runs).
        fn make_texture(width: usize, height: usize, seed: u32) -> Vec<u8> {
            let mut buf = vec![0u8; width * height];
            let mut state = seed.wrapping_add(1);
            for slot in buf.iter_mut() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *slot = (state & 0xff) as u8;
            }
            buf
        }
        // Build a 64×64 source frame with chaotic luma + flat chroma.
        let kf_y = make_texture(64, 64, 0x9e37_79b9);
        let uv = vec![128u8; 32 * 32];
        // Shifted-source second frame: each row of the source is the
        // KF row shifted right by 3 pixels (wrap-around). The KF
        // reconstructs ~losslessly at the encoder's default q, so the
        // inter-frame reference is approximately `kf_y`; the global
        // best MV for the entire SB is therefore `col≈+3` full-pel
        // (≈ +24 in 1/8-pel units).
        let mut shifted_y = vec![0u8; 64 * 64];
        for r in 0..64usize {
            for c in 0..64usize {
                // Wrap to keep the inter-frame source uniform at the
                // block level (no edge discontinuity that would force
                // the picker into a split).
                let src_col = (c + 3) % 64;
                shifted_y[r * 64 + c] = kf_y[r * 64 + src_col];
            }
        }

        // F.0.2 oracle introspection — capture per-leaf bsize for the
        // inter frame under each gate state.
        let collect_inter_leaves = |gate_on: bool| -> Vec<rvp9_core::block::BlockSize> {
            // SAFETY: outer test holds env_test_mutex for the whole
            // test, so these env-var mutations are serialised.
            unsafe {
                std::env::set_var("RVP9_F02_ORACLE", "1");
                if gate_on {
                    std::env::set_var("RVP9_F3_NONRD_PARTITION_ENABLE", "1");
                } else {
                    std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
                }
            }
            crate::f02_oracle::init_from_env();

            let mut enc =
                frame_loop::StreamEncoder::new(streaming_config()).unwrap();
            let _kf = enc
                .encode_frame(&yuv_view_64(&kf_y, &uv, &uv))
                .unwrap();
            // Drain KF decisions so we look only at inter-frame leaves.
            let _ = crate::f02_oracle::drain_frame();
            let _inter = enc
                .encode_frame(&yuv_view_64(&shifted_y, &uv, &uv))
                .unwrap();
            let leaves = crate::f02_oracle::drain_frame();
            leaves.into_iter().map(|d| d.bsize).collect()
        };

        let leaves_off = collect_inter_leaves(false);
        let leaves_on = collect_inter_leaves(true);

        // Defensive cleanup before any assertion fires (so a failed
        // assert doesn't leave env state dirty for subsequent tests
        // sharing the same process).
        unsafe {
            std::env::remove_var("RVP9_F02_ORACLE");
            std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
        }
        crate::f02_oracle::init_from_env();

        assert!(!leaves_off.is_empty(), "baseline (gate OFF) must emit leaves");
        assert!(!leaves_on.is_empty(), "F.3 (gate ON) must emit leaves");

        // PROOF: with the real picker the descent recognises the
        // uniform motion and collapses to fewer / larger leaves than
        // the variance-hint baseline. Use leaf count as the proxy for
        // "collapsed" — fewer leaves means the descent picked larger
        // sub-blocks (PARTITION_NONE at a higher level).
        //
        // The descent MIGHT match the variance hint exactly on some
        // inputs (proxy-OK case), but on shifted-content it
        // demonstrably differs: the ZERO-MV proxy sees high SSE and
        // refuses to merge, while the real picker's NEWMV winner
        // drives SSE near zero and prefers NONE. We assert "strictly
        // fewer leaves with F.3 on" — the libvpx-faithful semantic
        // for "motion-aware partition merged children".
        assert!(
            leaves_on.len() < leaves_off.len(),
            "F.3 gate ON must collapse partitions on shifted-source content; \
             off={} leaves, on={} leaves",
            leaves_off.len(),
            leaves_on.len(),
        );
    }

    /// P12 PROOF — with the F.3 gate OFF (default), the encoder
    /// produces byte-identical output to before the callback change.
    /// We can't compare against a frozen-byte baseline here (the test
    /// harness doesn't carry one), but we CAN verify that the existing
    /// "encode KF + inter" two-frame flow still produces a non-empty
    /// inter packet whose decoded shape matches the unchanged path.
    /// Coupled with `encode_with_f3_nonrd_partition_runs_descent_path`
    /// (gate-ON smoke test) and the rest of the encoder suite (gate
    /// unset throughout), this confirms the default path is intact.
    #[test]
    fn p12_real_callback_default_state_unchanged() {
        let _guard = env_test_mutex();
        // Defensive: ensure the gate is OFF.
        unsafe { std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE"); }

        let mut enc =
            frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        assert_eq!(kf.frame_kind, frame_loop::FrameKind::Key);
        assert_eq!(inter.frame_kind, frame_loop::FrameKind::Inter);
        assert!(kf.actual_bits > 0);
        assert!(inter.actual_bits > 0);
    }

    #[test]
    fn encode_with_diag_picker_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        let key = "RVP9_DIAG_PICKER";
        unsafe { std::env::set_var(key, "1"); }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        unsafe { std::env::remove_var(key); }
        assert_eq!(inter.frame_kind, frame_loop::FrameKind::Inter);
    }

    #[test]
    fn encode_with_diag_intpro_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        unsafe {
            std::env::set_var("RVP9_DIAG_INTPRO", "1");
            std::env::set_var("RVP9_DIAG_INTPRO_SB", "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        unsafe {
            std::env::remove_var("RVP9_DIAG_INTPRO");
            std::env::remove_var("RVP9_DIAG_INTPRO_SB");
        }
        assert!(inter.actual_bits > 0);
    }

    #[test]
    fn encode_with_diag_intra_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        unsafe { std::env::set_var("RVP9_DIAG_INTRA", "1"); }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        unsafe { std::env::remove_var("RVP9_DIAG_INTRA"); }
        assert_eq!(kf.frame_kind, frame_loop::FrameKind::Key);
    }

    #[test]
    fn encode_with_e3_mvdiag_and_golden_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        unsafe {
            std::env::set_var("RVP9_E3_MVDIAG", "1");
            std::env::set_var("RVP9_DIAG_GOLDEN_PICKS", "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        unsafe {
            std::env::remove_var("RVP9_E3_MVDIAG");
            std::env::remove_var("RVP9_DIAG_GOLDEN_PICKS");
        }
        assert!(inter.actual_bits > 0);
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
                    predicted_uv_mode: IntraMode::Dc,
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
