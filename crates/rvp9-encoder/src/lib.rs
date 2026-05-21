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

pub mod rate_control;

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

/// Encoder simulator: per-TX-block predict → residual → fdct →
/// forward-quant → idct-add. Returns a [`VecBackedIntraCoefSource`]
/// populated with the per-TX qcoeff buffers; fills `recon_yuv_out`
/// with the per-TX reconstructed pixels (the exact bit pattern the
/// decoder will produce given the emitted bitstream).
///
/// Caller invariants (checked outside):
/// - `(width, height)` are multiples of 64.
/// - `input_yuv` plane lengths match the YUV420 tight-stride layout.
/// - `recon_yuv_out` plane lengths match `input_yuv`.
fn simulate_dc_intra_encode(
    width: u32,
    height: u32,
    base_qindex: u8,
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
) -> VecBackedIntraCoefSource {
    use rvp9_core::intra_pred::{
        DcAvailability, EdgeBuffers, IntraMode as PredIntraMode, build_intra_predictors,
        intra_predict,
    };
    use rvp9_core::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, scan_for_forward, vp9_quantize_b,
        vp9_quantize_b_32x32,
    };
    use rvp9_core::transforms::forward::{fdct16x16, fdct32x32};
    use rvp9_core::transforms::inverse::{idct16x16_256_add, idct32x32_1024_add};
    use rvp9_core::transforms::scans::scan_for;
    use rvp9_core::transforms::TxType;

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let qi = base_qindex as i32;

    // Initialise the recon plane to zeroes (we'll fill TX-by-TX).
    for px in recon_yuv_out.0.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.1.iter_mut() {
        *px = 0;
    }
    for px in recon_yuv_out.2.iter_mut() {
        *px = 0;
    }

    let qp_y = init_plane_quantizers(qi, QuantizerDeltas::Y);
    let qp_uv = init_plane_quantizers(qi, QuantizerDeltas::Y); // delta-q all zero

    let mut out = VecBackedIntraCoefSource::new(width, height);

    // The mi-grid is splatted with `BlockSize::Block64x64` cells; we
    // iterate over each SB.
    let sb_cols = w / 64;
    let sb_rows = h / 64;

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mi_row = sb_row * 8;
            let mi_col = sb_col * 8;

            // The writer's `IntraCoefSource::get_block_coefs` receives
            // `(mi_row, mi_col)` of the leaf's top-left.

            // Iterate planes: 0=Y, 1=U, 2=V.
            for plane in 0..3 {
                let ss_x = F5_SUBSAMPLING_X[plane];
                let ss_y = F5_SUBSAMPLING_Y[plane];

                // Plane dimensions and TX size.
                let plane_w = if plane == 0 { w } else { uvw };
                let plane_h = if plane == 0 { h } else { uvh };

                let plane_tx = if plane == 0 {
                    TxSize::Tx32x32
                } else {
                    f5_uv_plane_tx_size(BlockSize::Block64x64, TxSize::Tx32x32)
                };
                // For Block64x64 + Y Tx32x32: UV plane size is
                // Block32x32, max_txsize = Tx32x32, so UV TX is also
                // Tx32x32 (a single 32×32 transform covers the whole
                // UV plane of a 64×64 SB).
                debug_assert_eq!(plane_tx, TxSize::Tx32x32);

                let plane_bw_4x4 = (64 >> ss_x) / 4;
                let plane_bh_4x4 = (64 >> ss_y) / 4;
                let tx_step_4x4 = 1usize << (plane_tx as usize);
                let tx_side = plane_tx.side();

                // Plane top-left in pixels (within the SB).
                let plane_px_base_x = (mi_col * 8) >> ss_x;
                let plane_px_base_y = (mi_row * 8) >> ss_y;

                // Per-plane references — split mutable / immutable so
                // we can borrow recon (mut) and input (read).
                let (input_plane, recon_plane) = match plane {
                    0 => (input_yuv.0, &mut *recon_yuv_out.0),
                    1 => (input_yuv.1, &mut *recon_yuv_out.1),
                    _ => (input_yuv.2, &mut *recon_yuv_out.2),
                };
                let plane_stride = plane_w;

                // Dequant pair: Y uses qp_y, UV uses qp_uv. Both
                // identical here (delta-q zero).
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

                // Walk TX positions inside this plane's coding block.
                let mut tx_y4_local = 0usize;
                while tx_y4_local < plane_bh_4x4 {
                    let mut tx_x4_local = 0usize;
                    while tx_x4_local < plane_bw_4x4 {
                        let tx_px_x = plane_px_base_x + tx_x4_local * 4;
                        let tx_px_y = plane_px_base_y + tx_y4_local * 4;

                        // 1. Build prediction edges from the local
                        //    recon buffer. Neighbour availability
                        //    flags mirror the decoder's
                        //    `have_top = loff > 0 || have_above_block;
                        //     have_left = aoff > 0 || have_left_block;`
                        let have_above_block =
                            sb_row > 0 || tx_y4_local > 0; // any prior SB row OR prior TX row inside SB
                        let have_left_block =
                            sb_col > 0 || tx_x4_local > 0;
                        // For TX-internal (within-SB) walks, the
                        // `have_top` / `have_left` formulas reduce to
                        // the same booleans (loff/aoff cover the
                        // within-coding-block portion; have_*_block
                        // covers crossings of the coding-block edge,
                        // i.e. crossings of the SB edge). For our
                        // single-SB tests `sb_row == sb_col == 0` so
                        // both are exactly `tx_*_local > 0`.
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
                            PredIntraMode::Dc,
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

                        // 2. Run prediction into the recon plane at
                        //    (tx_px_x, tx_px_y).
                        let dst_off = tx_px_y * plane_stride + tx_px_x;
                        let dst = &mut recon_plane[dst_off..];
                        intra_predict(
                            PredIntraMode::Dc,
                            dst,
                            plane_stride,
                            tx_side,
                            pred_ctx,
                            dc_avail,
                        );

                        // 3. Residual = input - pred. Pred lives in
                        //    recon[..] (just written by intra_predict).
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

                        // 4. Forward DCT.
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
                                fdct16x16(&residual, &mut buf, tx_side);
                                coefs_i32.copy_from_slice(&buf[..]);
                            }
                            // Tx8x8 / Tx4x4 aren't reachable for our
                            // 64×64 SB / Block64x64 leaf; assert if
                            // they ever are.
                            _ => unreachable!(
                                "F5.1 simulator only handles Tx32x32 (Y) and Tx16x16 (UV); got {:?}",
                                plane_tx
                            ),
                        }

                        // 5. Forward quant. Encoder qcoeff / dqcoeff
                        //    are i16. We then widen qcoeff to i32 for
                        //    the writer.
                        let scan = scan_for(plane_tx, TxType::DctDct);
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
                            TxSize::Tx16x16 => vp9_quantize_b(
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
                            _ => unreachable!(),
                        };
                        let eob_usize = eob as usize;

                        // 6. Inverse-DCT-add of the dequantized coefs.
                        //    Writes pred + residual into recon. The
                        //    decoder will produce the exact same
                        //    bytes given (a) the same qcoeffs (which
                        //    we emit via the coef source) and (b) the
                        //    same intra prediction (which we matched
                        //    by walking from the same recon-buffer
                        //    state).
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
                                    idct16x16_256_add(
                                        &buf,
                                        &mut recon_plane[dst_off..],
                                        plane_stride,
                                    );
                                }
                                _ => unreachable!(),
                            }
                        }
                        // If eob == 0, decoder uses fast-skip path
                        // (prediction stays as the reconstruction).

                        // 7. Record qcoeff buffer + eob.
                        let qcoeff_i32: Vec<i32> = qcoeff_i16
                            .into_iter()
                            .map(i32::from)
                            .collect();
                        out.insert(
                            mi_row,
                            mi_col,
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

    out
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
}
