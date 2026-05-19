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
//! The encoder is at SKELETON state. The wire-format layer is complete
//! (uncompressed header, compressed header, tile-buffer assembly,
//! superframe container, mode-info encoding for the simplest keyframe
//! case). What's missing is everything that makes the encoder USEFUL:
//! mode decision, partition selection, motion estimation, rate control,
//! adaptive quantization, denoising. Those land in the modules listed
//! in `PLAN.md` Stage 4a tasks 2-8.
//!
//! Today the encoder can emit ONE thing: a synthetic all-DC-skip
//! keyframe at any 64-multiple resolution. See
//! [`Encoder::encode_synthetic_dc_skip_keyframe`].

use rvp9_core::bitstream::frame::{MAX_INTER_REFS, ReferenceMode};
use rvp9_core::bitstream::header::HeaderTail;
use rvp9_core::bitstream::sub_parsers::{
    LoopFilterParams, MAX_PROB, PREDICTION_PROBS, QuantizationParams, SEG_TREE_PROBS,
    SegmentationParams, TileInfo,
};
use rvp9_core::bitstream::{
    BitDepth, BitstreamError, ChromaFormat, ColorRange, ColorSpace, CompressedHeaderOptions,
    CompressedHeaderResult, FrameKind, InterpFilter, ModesTileBounds, ModesWriterError, Profile,
    RawBitWriter, SuperframeBuilder, UncompressedHeader, write_compressed_header,
    write_tile_intra_keyframe, write_tile_inter, write_uncompressed_header,
};
use rvp9_core::block::BlockSize;
use rvp9_core::entropy::{BoolWriter, FrameContext};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, InterMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::{TxMode, TxSize};

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
        let inter_payload = self.build_inter_after_default_keyframe_payload()?;

        let mut sf = SuperframeBuilder::new();
        sf.push_frame(kf_payload)?;
        sf.push_frame(inter_payload)?;
        Ok(sf.finish())
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
            write_tile_intra_keyframe(&mut bw, &mi_grid, mi_rows, mi_cols, tile_bounds, &post_fc)?;
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
    /// Single-ref LASTFRAME, PARTITION_NONE 64×64, ZEROMV, skip=1,
    /// EightTap. Caller's responsibility to ensure the decoder's
    /// `frame_contexts[0]` is at default before this packet is
    /// parsed — that's what the `refresh_frame_context = false`
    /// keyframe pairing guarantees.
    fn build_inter_after_default_keyframe_payload(&self) -> Result<Vec<u8>, EncodeError> {
        let mi_rows = self.config.height / 8;
        let mi_cols = self.config.width / 8;

        // mi_grid: every 8×8 cell carries the same Inter cell. The
        // PARTITION_NONE 64×64 walker splats one ModeInfo across all
        // 64 cells of each 64×64 SB.
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
}
