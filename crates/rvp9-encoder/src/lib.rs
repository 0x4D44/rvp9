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
    UncompressedHeader, record_inter_leaf_coefficients, tile_bounds, write_compressed_header,
    write_tile_inter, write_tile_intra_keyframe, write_uncompressed_header,
};
use rvp9_core::block::BlockSize;
use rvp9_core::coef_ctx::TileCoefCtx;
use rvp9_core::entropy::{BoolWriter, FrameContext, FrameCounts, RetainedCoefTokens};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::PrevFrameMvRef;
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::intra_pred::IntraMode;
use rvp9_core::mode_info::{BlockMode, BmiInfo, InterMode, ModeInfo, Mv, RefFrame};
use rvp9_core::transforms::{TxMode, TxSize};

/// Positive-control counter for the `RVP9_TRELLIS` coefficient
/// optimizer: total number of (qcoeff) coefficients the trellis CHANGED
/// across the whole encode. Zero ⇒ trellis is inert (a no-op); non-zero
/// ⇒ trellis bit the bitstream. Read via [`trellis_changed_count`].
pub static TRELLIS_CHANGED_COEFFS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

struct InterEmitScratch {
    residual: [i16; 64 * 64],
    quant: InterEmitQuantScratch,
}

impl InterEmitScratch {
    const fn new() -> Self {
        Self {
            residual: [0; 64 * 64],
            quant: InterEmitQuantScratch {
                coefs: [0; 32 * 32],
                qcoeff: [0; 32 * 32],
                dqcoeff: [0; 32 * 32],
            },
        }
    }

    #[inline]
    fn with_len<R>(
        &mut self,
        len: usize,
        operation: impl FnOnce(&mut [i16], &mut InterEmitQuantScratch) -> R,
    ) -> R {
        assert!(
            len <= 32 * 32,
            "inter emit residual length {len} exceeds 32x32 scratch"
        );
        let Self { residual, quant } = self;
        operation(&mut residual[..len], quant)
    }
}

struct InterEmitQuantScratch {
    coefs: [i32; 32 * 32],
    qcoeff: [i16; 32 * 32],
    dqcoeff: [i16; 32 * 32],
}

/// Read the running total of coefficients changed by the `RVP9_TRELLIS`
/// trellis (positive control). See [`TRELLIS_CHANGED_COEFFS`].
#[must_use]
pub fn trellis_changed_count() -> u64 {
    TRELLIS_CHANGED_COEFFS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Serial guard for unit tests that mutate or depend on process-global
/// encoder environment variables.
#[cfg(test)]
pub(crate) fn env_test_mutex() -> std::sync::MutexGuard<'static, ()> {
    static MTX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    MTX.lock().unwrap_or_else(|p| p.into_inner())
}

/// Frame-wide high-precision MV gate.
///
/// This is off by default while Stage 4a recovery measures it. With
/// `RVP9_HIGHMV=1`, frames below libvpx's q<200 threshold emit and search
/// eighth-pel MVs; every MV-precision site must derive from this function.
#[must_use]
pub(crate) fn frame_allow_hp(q_index: u8) -> bool {
    std::env::var_os("RVP9_HIGHMV").is_some_and(|v| v != "0") && q_index < 200
}

pub mod aq_cyclic_refresh;
pub mod bitstream_gates;
pub mod buffer_pool;
pub mod compound;
pub mod denoiser;
pub mod e12stat;
pub mod f02_oracle;
pub mod frame_loop;
pub mod gop;
pub mod lookahead;
pub(crate) mod loopfilter;
pub mod ml_partition;
pub mod pre_analysis;
pub mod predstat;
pub mod rate_control;
pub mod rdo;
pub mod row_mt;
pub mod segmentation;
pub mod skipstat;
pub mod svc;
pub mod temporal_filter;
pub mod tpl;

pub use frame_loop::{
    AqMode, Deadline, EncodedFrame, EncoderConfig, EncoderParallelism,
    FrameKind as StreamFrameKind, Stage4aProductSplitOverride, StreamEncoder, YuvFrameRef,
    encode_two_pass,
};
// Re-audit pass 3: `EncoderConfig::rc_mode` is a public field of type
// `rate_control::RcMode`, so the discriminator must be reachable from
// the crate root for callers (CLI, tests, benches) constructing a
// config literal without reaching into the `rate_control` submodule.
pub use rate_control::RcMode;

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
        let kf_payload = self.build_keyframe_payload(/* refresh_frame_context = */ false)?;
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
    pub fn encode_synthetic_two_frame_kf_inter_non_skip(&self) -> Result<Vec<u8>, EncodeError> {
        let kf_payload = self.build_keyframe_payload(/* refresh_frame_context = */ false)?;
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
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target, None)?;
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
        let mut tile_buf = vec![0u8; (mi_rows as usize) * (mi_cols as usize) * 64 + 16 * 1024];
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
                target.tx_mode,
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
    fn build_keyframe_payload(&self, refresh_frame_context: bool) -> Result<Vec<u8>, EncodeError> {
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
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target, None)?;
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
            let tile_mi_cols = (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
            let coef_source = AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                target.tx_mode,
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
        let mi_grid: Vec<Option<ModeInfo>> = vec![Some(inter_cell); (mi_rows * mi_cols) as usize];

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
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target, None)?;
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
            seg_ref_frame: None,
            seg_skip: false,
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

/// Stored block payload inside [`VecBackedInterCoefSource::coefficients`].
#[derive(Copy, Clone, Debug)]
struct IntraCoefBlockValue {
    offset: u32,
    len: u16,
    eob: u16,
}

/// Backing storage for zero-EOB inter TX markers. When the encoder's
/// quantizer returns `eob == 0`, the writer only needs an all-zero slice
/// of the requested TX size; storing a full zero vector per block would
/// allocate in the hottest emit path for no information.
static INTER_ZERO_COEFS: [i16; 1024] = [0; 1024];

/// Empty marker in [`VecBackedInterCoefSource::slots`].
const INTER_COEF_SLOT_EMPTY: u32 = u32::MAX;

/// Encoder simulator output: a direct-slot [`IntraCoefSource`].
///
/// Intra and inter writers address coefficients with the same physical
/// transform position tuple. Reuse the inter source's dense slot map so the
/// keyframe path does not hash every transform on insertion and emission.
pub struct VecBackedIntraCoefSource {
    inner: VecBackedInterCoefSource,
}

impl VecBackedIntraCoefSource {
    /// Build an empty source for the given frame dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            inner: VecBackedInterCoefSource::new(width, height),
        }
    }

    /// Build an empty source containing only one column tile's physical slots.
    #[must_use]
    pub(crate) fn new_for_mi_columns(
        width: u32,
        height: u32,
        mi_col_start: usize,
        mi_col_end: usize,
    ) -> Self {
        Self {
            inner: VecBackedInterCoefSource::new_for_mi_columns(
                width,
                height,
                mi_col_start,
                mi_col_end,
            ),
        }
    }

    /// Merge one disjoint tile-owned coefficient source into this frame source.
    pub(crate) fn merge_from(&mut self, other: Self) {
        self.inner.merge_from(other.inner);
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
        self.inner
            .insert(mi_row, mi_col, plane, tx_x4_local, tx_y4_local, qcoefs, eob);
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.inner.height()
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
    ) -> (&[i16], usize) {
        rvp9_core::bitstream::InterCoefSource::get_block_coefs(
            &self.inner,
            mi_row,
            mi_col,
            plane,
            tx_x4_local,
            tx_y4_local,
            tx_size,
        )
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
    /// Per-plane offsets into `slots`.
    plane_offsets: [usize; 3],
    /// First global plane-relative 4x4 column represented in `slots`.
    plane_col_starts: [usize; 3],
    /// Per-plane widths in plane-relative 4x4 units.
    plane_strides: [usize; 3],
    /// Per-plane heights in plane-relative 4x4 units.
    plane_rows: [usize; 3],
    /// Dense physical-TX-position to `blocks` index mapping. Luma uses one
    /// slot per 4x4 luma cell; U and V use one per 4x4 chroma cell.
    slots: Vec<u32>,
    /// Coefficient payloads in insertion order.
    blocks: Vec<IntraCoefBlockValue>,
    /// Contiguous coefficient arena. Keeping all transform payloads here
    /// avoids one heap allocation per transform in the inter emit hot path.
    coefficients: Vec<i16>,
}

impl VecBackedInterCoefSource {
    /// Build an empty source for the given frame dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let luma_cols = (width as usize).div_ceil(4);
        let chroma_cols = (width as usize).div_ceil(8);
        Self::new_for_plane_columns(width, height, [0; 3], [luma_cols, chroma_cols, chroma_cols])
    }

    /// Build an empty source containing only a column tile's physical slots.
    #[must_use]
    pub(crate) fn new_for_mi_columns(
        width: u32,
        height: u32,
        mi_col_start: usize,
        mi_col_end: usize,
    ) -> Self {
        let width_usize = width as usize;
        let mi_cols = width_usize.div_ceil(8);
        assert!(
            mi_col_start <= mi_col_end && mi_col_end <= mi_cols,
            "inter coefficient tile columns must lie inside the frame",
        );
        let luma_cols = width_usize.div_ceil(4);
        let chroma_cols = width_usize.div_ceil(8);
        Self::new_for_plane_columns(
            width,
            height,
            [mi_col_start * 2, mi_col_start, mi_col_start],
            [
                (mi_col_end * 2).min(luma_cols),
                mi_col_end.min(chroma_cols),
                mi_col_end.min(chroma_cols),
            ],
        )
    }

    fn new_for_plane_columns(
        width: u32,
        height: u32,
        plane_col_starts: [usize; 3],
        plane_col_ends: [usize; 3],
    ) -> Self {
        let width = width as usize;
        let height = height as usize;
        let luma_rows = height.div_ceil(4);
        let chroma_rows = height.div_ceil(8);
        let plane_strides = std::array::from_fn(|plane| {
            plane_col_ends[plane]
                .checked_sub(plane_col_starts[plane])
                .expect("inter coefficient plane column range must be ordered")
        });
        let luma_slots = plane_strides[0] * luma_rows;
        let chroma_u_slots = plane_strides[1] * chroma_rows;
        let chroma_v_slots = plane_strides[2] * chroma_rows;
        let total_slots = luma_slots + chroma_u_slots + chroma_v_slots;
        Self {
            width: width as u32,
            height: height as u32,
            plane_offsets: [0, luma_slots, luma_slots + chroma_u_slots],
            plane_col_starts,
            plane_strides,
            plane_rows: [luma_rows, chroma_rows, chroma_rows],
            slots: vec![INTER_COEF_SLOT_EMPTY; total_slots],
            blocks: Vec::with_capacity(total_slots),
            // Every physical 4x4 slot accounts for 16 coefficients. Larger
            // transforms occupy several slots but store the same total area.
            coefficients: Vec::with_capacity(total_slots * 16),
        }
    }

    /// Map a leaf-local TX coordinate to its unique physical plane slot.
    fn slot_index(
        &self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
    ) -> usize {
        assert!(plane < 3, "coefficient plane must be 0, 1, or 2");
        // One 8x8 luma MI spans two luma 4x4 cells, or one chroma 4x4
        // cell under Profile-0 4:2:0 subsampling.
        let mi_scale = if plane == 0 { 2 } else { 1 };
        let x4 = mi_col
            .checked_mul(mi_scale)
            .and_then(|x| x.checked_add(tx_x4_local))
            .expect("inter coefficient x coordinate overflow");
        let y4 = mi_row
            .checked_mul(mi_scale)
            .and_then(|y| y.checked_add(tx_y4_local))
            .expect("inter coefficient y coordinate overflow");
        let plane_col_start = self.plane_col_starts[plane];
        assert!(
            x4 >= plane_col_start
                && x4 - plane_col_start < self.plane_strides[plane]
                && y4 < self.plane_rows[plane],
            "inter coefficient coordinate outside plane: \
             mi=({mi_row},{mi_col}) plane={plane} local=({tx_x4_local},{tx_y4_local})",
        );
        self.plane_offsets[plane] + y4 * self.plane_strides[plane] + (x4 - plane_col_start)
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
        self.insert_coefficients(
            mi_row,
            mi_col,
            plane,
            tx_x4_local,
            tx_y4_local,
            &qcoefs,
            eob,
            |value| i16::try_from(value).expect("quantized coefficient exceeds i16"),
        );
    }

    /// Record one transform directly from the quantizer's i16 scratch.
    #[allow(clippy::too_many_arguments)]
    fn insert_i16(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        qcoefs: &[i16],
        eob: usize,
    ) {
        self.insert_coefficients(
            mi_row,
            mi_col,
            plane,
            tx_x4_local,
            tx_y4_local,
            qcoefs,
            eob,
            |value| value,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_coefficients<T: Copy>(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        plane: usize,
        tx_x4_local: usize,
        tx_y4_local: usize,
        qcoefs: &[T],
        eob: usize,
        convert: impl Fn(T) -> i16,
    ) {
        let slot = self.slot_index(mi_row, mi_col, plane, tx_x4_local, tx_y4_local);
        let block_index = self.slots[slot];
        let len = u16::try_from(qcoefs.len()).expect("transform coefficient count exceeds u16");
        let eob = u16::try_from(eob).expect("transform EOB exceeds u16");
        if block_index != INTER_COEF_SLOT_EMPTY {
            let entry = self.blocks[block_index as usize];
            if entry.len == len {
                let start = entry.offset as usize;
                let end = start + usize::from(len);
                for (dst, &src) in self.coefficients[start..end].iter_mut().zip(qcoefs) {
                    *dst = convert(src);
                }
                self.blocks[block_index as usize].eob = eob;
                return;
            }
        }

        let offset = u32::try_from(self.coefficients.len())
            .expect("inter coefficient arena exceeds u32 capacity");
        self.coefficients
            .extend(qcoefs.iter().copied().map(convert));
        let value = IntraCoefBlockValue { offset, len, eob };
        if block_index == INTER_COEF_SLOT_EMPTY {
            let block_index = u32::try_from(self.blocks.len())
                .expect("inter coefficient source exceeds u32 block capacity");
            self.blocks.push(value);
            self.slots[slot] = block_index;
        } else {
            self.blocks[block_index as usize] = value;
        }
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

    /// Merge disjoint tile-owned coefficient blocks into this frame
    /// source. Tile jobs use global MI coordinates, so no key rebasing
    /// is required.
    pub(crate) fn merge_from(&mut self, mut other: Self) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        assert_eq!(self.plane_rows, other.plane_rows);
        let block_base = u32::try_from(self.blocks.len())
            .expect("inter coefficient source exceeds u32 block capacity");
        let coefficient_base = u32::try_from(self.coefficients.len())
            .expect("inter coefficient arena exceeds u32 capacity");
        for block in &mut other.blocks {
            block.offset = block
                .offset
                .checked_add(coefficient_base)
                .expect("inter coefficient arena exceeds u32 capacity");
        }
        self.coefficients.extend(other.coefficients);
        self.blocks.extend(other.blocks);
        for plane in 0..3 {
            let self_col_start = self.plane_col_starts[plane];
            let other_col_start = other.plane_col_starts[plane];
            let other_col_offset = other_col_start
                .checked_sub(self_col_start)
                .expect("merged coefficient source must contain the tile columns");
            assert!(
                other_col_offset + other.plane_strides[plane] <= self.plane_strides[plane],
                "merged coefficient source must contain the tile columns",
            );
            for row in 0..other.plane_rows[plane] {
                for col in 0..other.plane_strides[plane] {
                    let other_slot = other.slots
                        [other.plane_offsets[plane] + row * other.plane_strides[plane] + col];
                    if other_slot == INTER_COEF_SLOT_EMPTY {
                        continue;
                    }
                    let slot = &mut self.slots[self.plane_offsets[plane]
                        + row * self.plane_strides[plane]
                        + other_col_offset
                        + col];
                    assert_eq!(
                        *slot, INTER_COEF_SLOT_EMPTY,
                        "column tile coefficient sources must have disjoint physical TX slots",
                    );
                    *slot = block_base
                        .checked_add(other_slot)
                        .expect("inter coefficient source exceeds u32 block capacity");
                }
            }
        }
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
    ) -> (&[i16], usize) {
        let slot = self.slot_index(mi_row, mi_col, plane, tx_x4_local, tx_y4_local);
        let block_index = self.slots[slot];
        assert_ne!(
            block_index, INTER_COEF_SLOT_EMPTY,
            "VecBackedInterCoefSource: missing TX block at \
             (mi_row={mi_row}, mi_col={mi_col}, plane={plane}, \
             tx_x4_local={tx_x4_local}, tx_y4_local={tx_y4_local})",
        );
        let entry = &self.blocks[block_index as usize];
        let n = tx_size.n_coeffs();
        if entry.eob == 0 && entry.len == 0 {
            return (&INTER_ZERO_COEFS[..n], 0);
        }
        debug_assert!(usize::from(entry.len) >= n);
        let start = entry.offset as usize;
        (&self.coefficients[start..start + n], usize::from(entry.eob))
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

/// Materialize the zero-motion LAST prediction for a leaf that bypasses the
/// normal picker and reconstruction path.
fn copy_last_ref_leaf_to_recon(
    width: usize,
    height: usize,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    last_recon_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
) {
    for plane in 0..3 {
        let ss_x = F5_SUBSAMPLING_X[plane];
        let ss_y = F5_SUBSAMPLING_Y[plane];
        let plane_width = width >> ss_x;
        let plane_height = height >> ss_y;
        let plane_bsize = if plane == 0 {
            bsize
        } else {
            f5_uv_plane_block_size(bsize)
        };
        let block_width = usize::from(plane_bsize.width());
        let block_height = usize::from(plane_bsize.height());
        let pixel_x = (mi_col * 8) >> ss_x;
        let pixel_y = (mi_row * 8) >> ss_y;
        debug_assert!(pixel_x + block_width <= plane_width);
        debug_assert!(pixel_y + block_height <= plane_height);
        let (src, dst) = match plane {
            0 => (last_recon_yuv.0, &mut *recon_yuv_out.0),
            1 => (last_recon_yuv.1, &mut *recon_yuv_out.1),
            _ => (last_recon_yuv.2, &mut *recon_yuv_out.2),
        };
        for row in pixel_y..pixel_y + block_height {
            let start = row * plane_width + pixel_x;
            let end = start + block_width;
            dst[start..end].copy_from_slice(&src[start..end]);
        }
    }
}

/// Largest transform allowed by libvpx's realtime keyframe policy.
///
/// Speed 4+ combines `USE_LARGESTALL` with frame `tx_mode = ALLOW_16X16`,
/// so the chosen size is the leaf maximum capped at 16x16.
#[inline]
fn realtime_keyframe_tx_size(bsize: BlockSize) -> TxSize {
    rvp9_core::block::max_txsize(bsize).min(TxSize::Tx16x16)
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
/// `predicted_y_modes` holds the four raster-order 4×4 BMI modes for a
/// sub-8 leaf. Larger leaves keep all four slots equal. The single-
/// leaf form ([`simulate_dc_intra_encode`]) and the partition-aware
/// frame loop's pre-F10.1b path use `IntraMode::Dc`; realtime keyframes drive
/// this from `pick_realtime_intra_mode_luma`'s DC/V/H subset.
/// `predicted_uv_mode` is the chroma intra mode used for both U and V
/// planes (VP9 wire format emits a single `uv_mode` per leaf). Realtime
/// keyframes and pre-picker paths keep it at `IntraMode::Dc`.
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
    /// Raster-order 4×4 luma modes. All slots are equal for leaves at
    /// least 8×8; a Block4x4 leaf carries four causal BMI modes.
    pub predicted_y_modes: [IntraMode; 4],
    /// Intra UV-mode applied to BOTH U and V planes for this leaf.
    /// Realtime keyframes set it to [`IntraMode::Dc`].
    pub predicted_uv_mode: IntraMode,
    /// Whether all Y/U/V transform blocks quantized to zero. Realtime
    /// keyframes mirror libvpx's post-encode `mi->skip` update and signal
    /// `skip=true` for such leaves.
    pub skip: bool,
    /// AQ.1 — segment id `0..=7` driving the per-leaf effective
    /// qindex via [`effective_qindex_for_segment`]. `0` for AQ-off.
    pub segment_id: u8,
    /// Luma transform size for the KF/intra path. Realtime keyframes fill
    /// this from [`realtime_keyframe_tx_size`];
    /// [`crate::frame_loop::build_kf_mi_grid`] /
    /// [`crate::loopfilter::build_kf_mi_grid_for_lf`] splat it into
    /// `mi.tx_size`. Frame `ALLOW_16X16` makes the realtime size implicit.
    /// Non-picker test simulator paths default to `max_txsize(leaf_bsize)`.
    pub picked_tx_size: TxSize,
}

impl PartitionLeafDesc {
    #[inline]
    fn parent_y_mode(self) -> IntraMode {
        if self.leaf_bsize < BlockSize::Block8x8 {
            self.predicted_y_modes[3]
        } else {
            self.predicted_y_modes[0]
        }
    }

    #[inline]
    fn y_mode_for_tx(self, tx_x4_local: usize, tx_y4_local: usize) -> IntraMode {
        if self.leaf_bsize < BlockSize::Block8x8 {
            self.predicted_y_modes[(tx_y4_local << 1) | tx_x4_local]
        } else {
            self.predicted_y_modes[0]
        }
    }

    #[inline]
    fn bmi(self) -> [BmiInfo; 4] {
        if self.leaf_bsize < BlockSize::Block8x8 {
            core::array::from_fn(|index| BmiInfo {
                as_mode: self.predicted_y_modes[index] as u8,
                ..BmiInfo::default()
            })
        } else {
            [BmiInfo::default(); 4]
        }
    }
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
    let mut sb_partitions: Vec<Vec<PartitionLeafDesc>> = Vec::with_capacity(sb_rows * sb_cols);
    for _ in 0..(sb_rows * sb_cols) {
        sb_partitions.push(vec![PartitionLeafDesc {
            mi_row_in_sb: 0,
            mi_col_in_sb: 0,
            leaf_bsize: BlockSize::Block64x64,
            predicted_y_modes: [IntraMode::Dc; 4],
            predicted_uv_mode: IntraMode::Dc,
            skip: false,
            segment_id: 0,
            picked_tx_size: rvp9_core::block::max_txsize(BlockSize::Block64x64),
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

fn column_tile_bounds(mi_rows: usize, mi_cols: usize, log2_tile_cols: u8) -> Vec<ModesTileBounds> {
    (0..(1u32 << log2_tile_cols))
        .map(|tile_col| tile_bounds(mi_rows, mi_cols, 0, log2_tile_cols, 0, tile_col))
        .collect()
}

fn tile_for_mi_col(bounds: &[ModesTileBounds], mi_col: usize) -> ModesTileBounds {
    bounds
        .iter()
        .copied()
        .find(|tile| mi_col >= tile.mi_col_start && mi_col < tile.mi_col_end)
        .expect("every frame MI column belongs to exactly one VP9 tile")
}

fn inter_row_job_sb_columns(
    bounds: &[ModesTileBounds],
    active_tile_col: Option<usize>,
    job: crate::row_mt::RowJob,
) -> std::ops::Range<usize> {
    let tile = *bounds
        .get(job.tile_col)
        .expect("inter row job tile column is out of bounds");
    assert!(
        active_tile_col.is_none_or(|active| active == job.tile_col),
        "inter row job does not belong to the active tile"
    );
    assert!(
        tile.mi_col_start.is_multiple_of(8) && tile.mi_col_end.is_multiple_of(8),
        "inter row jobs require superblock-aligned tile columns"
    );
    tile.mi_col_start / 8..tile.mi_col_end / 8
}

/// `simulate_dc_intra_encode_partitions` without the up-front recon zeroing.
pub(crate) fn simulate_dc_intra_encode_partitions_no_zero(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &[Vec<PartitionLeafDesc>],
) -> VecBackedIntraCoefSource {
    let mut out = VecBackedIntraCoefSource::new(width, height);
    simulate_dc_intra_encode_partitions_no_zero_with_tiles(
        width,
        height,
        base_qindex,
        seg_q_deltas,
        input_yuv,
        recon_yuv_out,
        sb_partitions,
        0,
        None,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn simulate_dc_intra_encode_partitions_no_zero_with_tiles(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &[Vec<PartitionLeafDesc>],
    log2_tile_cols: u8,
    active_leaf: Option<(usize, usize)>,
    out: &mut VecBackedIntraCoefSource,
) {
    use rvp9_core::intra_pred::{
        DcAvailability, EdgeBuffers, build_intra_predictors, intra_predict,
    };
    use rvp9_core::quant::forward::{
        QuantizerDeltas, init_plane_quantizers, vp9_quantize_b_32x32_with_order,
        vp9_quantize_b_with_order,
    };
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::forward::{fdct32x32, fht4x4, fht8x8, fht16x16};
    use rvp9_core::transforms::scans::scan_for;

    // G4.3.1 — `intra_predict` takes a `&SimdDispatch` so the DC 4×4
    // both-avail path can route through the SIMD slot. Fetch once per
    // call; `dsp::get()` is one atomic load.
    let dispatch = rvp9_core::dsp::get();

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;

    let sb_cols = w / 64;
    let sb_rows = h.div_ceil(64);
    let tile_bounds_by_col = column_tile_bounds(h / 8, w / 8, log2_tile_cols);
    assert_eq!(
        sb_partitions.len(),
        sb_rows * sb_cols,
        "sb_partitions must have one entry per 64x64 SB ({} rows × {} cols)",
        sb_rows,
        sb_cols,
    );

    let active_sb = active_leaf.map(|(sb_idx, _)| sb_idx);
    assert!(
        active_sb.is_none_or(|sb_idx| sb_idx < sb_rows * sb_cols),
        "active intra leaf must be inside the frame"
    );
    let sb_row_start = active_sb.map_or(0, |sb_idx| sb_idx / sb_cols);
    let sb_row_end = active_sb.map_or(sb_rows, |sb_idx| sb_idx / sb_cols + 1);
    for sb_row in sb_row_start..sb_row_end {
        let sb_col_start = active_sb.map_or(0, |sb_idx| sb_idx % sb_cols);
        let sb_col_end = active_sb.map_or(sb_cols, |sb_idx| sb_idx % sb_cols + 1);
        for sb_col in sb_col_start..sb_col_end {
            let current_tile = tile_for_mi_col(&tile_bounds_by_col, sb_col * 8);
            let leaves = &sb_partitions[sb_row * sb_cols + sb_col];
            let leaf_start = active_leaf.map_or(0, |(_, leaf_idx)| leaf_idx);
            let leaf_end = active_leaf.map_or(leaves.len(), |(_, leaf_idx)| leaf_idx + 1);
            assert!(leaf_end <= leaves.len(), "active intra leaf must exist");
            for leaf in &leaves[leaf_start..leaf_end] {
                let leaf_bsize = leaf.leaf_bsize;
                let leaf_mi_row = sb_row * 8 + leaf.mi_row_in_sb;
                let leaf_mi_col = sb_col * 8 + leaf.mi_col_in_sb;
                // Realtime keyframe policy sets `leaf.picked_tx_size` to the
                // block maximum capped at Tx16x16 before this simulator codes
                // the leaf. Non-picker tests may supply any valid size. UV is
                // capped to the Y size.
                let luma_tx = leaf.picked_tx_size;
                debug_assert!(
                    (luma_tx as u8) <= (rvp9_core::block::max_txsize(leaf_bsize) as u8),
                    "picked_tx_size {:?} exceeds max for {:?}",
                    luma_tx,
                    leaf_bsize,
                );
                // F10.1b: Y uses the realtime-picked intra mode. A sub-8
                // leaf may carry a different mode for each 4x4 TX. UV uses
                // `leaf.predicted_uv_mode`, which realtime keyframes keep at
                // DC. The per-TX Y mode also drives the forward-transform
                // scan order at sizes < Tx32x32 (Tx32x32 always uses
                // DctDct, mirroring the decoder + writer); UV is
                // always DctDct.
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

                // Reuse one set of maximum-size transform buffers for every
                // TX block in this leaf. The prior path allocated five Vecs
                // per block and copied each transform's stack output into a
                // second heap buffer. Only the final qcoeff buffer escapes
                // into `out`; all other storage is scratch.
                let mut residual = [0i16; 1024];
                let mut coefs_i32 = [0i32; 1024];
                let mut qcoeff_i16 = [0i16; 1024];
                let mut dqcoeff_i16 = [0i16; 1024];
                let mut dq_i32 = [0i32; 1024];

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
                    // decoder's `reconstruct_intra_planes` derivation.
                    // Sub-8 ModeInfo occupies one 8x8 cell: four luma
                    // 4x4 TXs and one 4x4 TX per chroma plane.
                    let is_sub8 = leaf_bsize < BlockSize::Block8x8;
                    let (plane_bw_4x4, plane_bh_4x4) = if is_sub8 {
                        (2 >> ss_x, 2 >> ss_y)
                    } else {
                        (
                            plane_bsize.n_4x4_wide() as usize,
                            plane_bsize.n_4x4_high() as usize,
                        )
                    };
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
                    let have_left_block = leaf_mi_col > current_tile.mi_col_start;

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
                            let have_right = (tx_x4_local + tx_step_4x4) < plane_bw_4x4;

                            // Sub-8 Y uses the causal mode for this BMI;
                            // both UV planes use the single wire `uv_mode`.
                            let plane_mode = if plane == 0 {
                                leaf.y_mode_for_tx(tx_x4_local, tx_y4_local)
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
                            let residual = &mut residual[..tx_side * tx_side];
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
                                    residual,
                                    tx_side,
                                    plane_tx as usize,
                                );
                                // E9/stage-SSE DIAG — capture the Y prediction
                                // pixels (pre-residual) into the whole-frame
                                // `diag_pred` snapshot. `recon_plane` holds
                                // pure prediction here.
                                crate::predstat::record_y_pred_block(
                                    recon_plane,
                                    plane_stride,
                                    tx_px_x,
                                    tx_px_y,
                                    tx_side,
                                    plane_w,
                                    plane_h,
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
                            let coefs_i32 = &mut coefs_i32[..n];
                            match plane_tx {
                                TxSize::Tx32x32 => {
                                    fdct32x32(
                                        residual,
                                        coefs_i32.try_into().expect("Tx32 coefficient extent"),
                                        tx_side,
                                    );
                                }
                                TxSize::Tx16x16 => {
                                    fht16x16(
                                        residual,
                                        coefs_i32.try_into().expect("Tx16 coefficient extent"),
                                        tx_side,
                                        tx_type,
                                    );
                                }
                                TxSize::Tx8x8 => {
                                    fht8x8(
                                        residual,
                                        coefs_i32.try_into().expect("Tx8 coefficient extent"),
                                        tx_side,
                                        tx_type,
                                    );
                                }
                                TxSize::Tx4x4 => {
                                    fht4x4(
                                        residual,
                                        coefs_i32.try_into().expect("Tx4 coefficient extent"),
                                        tx_side,
                                        tx_type,
                                    );
                                }
                            }

                            // 5. Forward quant. Scan follows the writer:
                            //    Tx32x32 → DctDct scan; smaller sizes use
                            //    `scan_for(plane_tx, tx_type)`.
                            let scan = scan_for(plane_tx, tx_type);
                            let qcoeff_i16 = &mut qcoeff_i16[..n];
                            let dqcoeff_i16 = &mut dqcoeff_i16[..n];
                            let eob = match plane_tx {
                                TxSize::Tx32x32 => vp9_quantize_b_32x32_with_order(
                                    coefs_i32,
                                    /* skip_block = */ false,
                                    &zbin,
                                    &round,
                                    &quant,
                                    &quant_shift,
                                    qcoeff_i16,
                                    dqcoeff_i16,
                                    &dequant_pair,
                                    &scan,
                                ),
                                _ => vp9_quantize_b_with_order(
                                    coefs_i32,
                                    n,
                                    /* skip_block = */ false,
                                    &zbin,
                                    &round,
                                    &quant,
                                    &quant_shift,
                                    qcoeff_i16,
                                    dqcoeff_i16,
                                    &dequant_pair,
                                    &scan,
                                ),
                            };
                            let eob_usize = eob as usize;

                            // 6. Inverse-DCT-add of the dequant coefs.
                            //    DRIFT-FIX — route through the shared
                            //    `intra_inverse_transform_add` so the KF
                            //    reconstruction is byte-identical to the
                            //    decoder's eob-bucketed, SIMD-dispatched
                            //    `inverse_transform_add` (was: scalar full
                            //    kernel only, which diverged ±1 from the
                            //    decoder's sparse/SIMD kernels and seeded
                            //    the keyframe intra-prediction cascade).
                            let dq_i32 = &mut dq_i32[..n];
                            for i in 0..n {
                                dq_i32[i] = dqcoeff_i16[i] as i32;
                            }
                            if eob_usize > 0 {
                                intra_inverse_transform_add(
                                    plane_tx,
                                    tx_type,
                                    dq_i32,
                                    eob_usize,
                                    &mut recon_plane[dst_off..],
                                    plane_stride,
                                );
                            }

                            // 7. Record qcoeff buffer + eob, keyed by
                            //    leaf top-left (mi_row, mi_col) so the
                            //    writer's `IntraCoefSource` lookup hits.
                            let qcoeff_i32: Vec<i32> =
                                qcoeff_i16.iter().copied().map(i32::from).collect();
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
}

fn update_intra_picker_luma_coef_context(
    coef_ctx: &mut TileCoefCtx,
    coef_source: &VecBackedIntraCoefSource,
    mi_row: usize,
    mi_col: usize,
    tile_mi_col_start: usize,
    leaf_bsize: BlockSize,
    tx_size: TxSize,
) {
    let extent_4x4 = if leaf_bsize < BlockSize::Block8x8 {
        2
    } else {
        leaf_bsize.n_4x4_wide() as usize
    };
    let tx_step_4x4 = 1usize << tx_size as usize;
    let x4_base = (mi_col - tile_mi_col_start) * 2;
    let y4_in_sb_base = (mi_row & 7) * 2;

    for tx_y4_local in (0..extent_4x4).step_by(tx_step_4x4) {
        for tx_x4_local in (0..extent_4x4).step_by(tx_step_4x4) {
            let (_, eob) = rvp9_core::bitstream::IntraCoefSource::get_block_coefs(
                coef_source,
                mi_row,
                mi_col,
                0,
                tx_x4_local,
                tx_y4_local,
                tx_size,
            );
            coef_ctx.set_coef_flags(
                0,
                x4_base + tx_x4_local,
                y4_in_sb_base + tx_y4_local,
                tx_step_4x4,
                tx_step_4x4,
                eob > 0,
            );
        }
    }
}

#[inline]
fn extend_standalone_tx8_above(above: &mut [u8; 16]) {
    let top_right = above[7];
    above[8..].fill(top_right);
}

/// F10.1b — pick the Y intra mode per leaf via
/// [`rvp9_core::mode_decision::intra::pick_realtime_intra_mode_luma`], record it
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
/// 3. Call `pick_realtime_intra_mode_luma` to choose among libvpx's
///    speed-4+ keyframe subset: DC, V, and H.
/// 4. Write the picked mode back into `sb_partitions[sb][leaf]
///    .predicted_y_modes` AND the picked-mode grid (splat across every
///    8×8 mi cell the leaf covers).
/// 5. Keep chroma at DC, matching libvpx's speed-4+ realtime keyframe
///    policy (`intra_uv_mode_mask[*] = INTRA_DC`).
/// 6. Run the per-leaf encode pass (predict → fdct → quant → idct-add)
///    using the picked Y / UV modes — populates both the recon buffer
///    and the qcoeff [`VecBackedIntraCoefSource`].
///
/// For `Block64x64` leaves the luma picker is out of scope (sub-32×32
/// per-leaf sub-mode picking is on the F8.x roadmap); both Y and UV
/// fall back to DC on those leaves. Every smaller-leaf size gets both
/// the luma and chroma picker treatment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_and_simulate_intra_encode_partitions(
    width: u32,
    height: u32,
    base_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    recon_yuv_out: (&mut [u8], &mut [u8], &mut [u8]),
    sb_partitions: &mut [Vec<PartitionLeafDesc>],
    log2_tile_cols: u8,
    active_tile_col: Option<usize>,
) -> VecBackedIntraCoefSource {
    use rvp9_core::coef_ctx::combine_entropy_contexts;
    use rvp9_core::cost::{
        FrameTypeForRd, coef_cost::build_coef_cost_table, compute_rd_mult,
        intra_y_mode_costs_kf_table,
    };
    use rvp9_core::entropy::FrameContext;
    use rvp9_core::intra_pred::{ABOVE_UNAVAILABLE_FILL, LEFT_UNAVAILABLE_FILL};
    use rvp9_core::mode_decision::intra::{IntraModeSearchParams, pick_realtime_intra_mode_luma};
    use rvp9_core::mv_search::PlaneRef;
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

    use crate::rdo::kf_intra8::{PickIntra8x8Params, pick_intra8x8_y_mode};
    use crate::rdo::sub8_kf_intra::{PickIntraSub8x8Params, pick_intra_sub_8x8_y_mode};

    let w = width as usize;
    let h = height as usize;
    let sb_cols = w / 64;
    let sb_rows = h.div_ceil(64);
    assert_eq!(
        sb_partitions.len(),
        sb_rows * sb_cols,
        "sb_partitions must have one entry per 64x64 SB ({} rows × {} cols)",
        sb_rows,
        sb_cols,
    );

    let mi_cols = w / 8;
    let mi_rows = h / 8;
    let tile_bounds_by_col = column_tile_bounds(mi_rows, mi_cols, log2_tile_cols);
    let active_bounds = active_tile_col.map(|tile_col| {
        *tile_bounds_by_col
            .get(tile_col)
            .expect("active keyframe tile column must exist")
    });
    let active_sb_cols = active_bounds.map_or(0..sb_cols, |bounds| {
        debug_assert_eq!(bounds.mi_col_start % 8, 0);
        debug_assert_eq!(bounds.mi_col_end % 8, 0);
        bounds.mi_col_start / 8..bounds.mi_col_end / 8
    });
    let mut picked_y_mode_grid: Vec<Option<[IntraMode; 4]>> = vec![None; mi_rows * mi_cols];

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
    let exact_keyframe_sub8 =
        std::env::var_os("RVP9_EXACT_KF_SUB8").is_some_and(|value| value != "0");
    let coef_costs = sb_partitions
        .iter()
        .flatten()
        .any(|leaf| {
            leaf.leaf_bsize < BlockSize::Block8x8
                || (exact_keyframe_sub8 && leaf.leaf_bsize == BlockSize::Block8x8)
        })
        .then(|| build_coef_cost_table(&FrameContext::new_default()));
    let mut picker_coef_contexts = coef_costs.as_ref().map(|_| {
        tile_bounds_by_col
            .iter()
            .map(|bounds| TileCoefCtx::new(bounds.mi_col_end - bounds.mi_col_start))
            .collect::<Vec<_>>()
    });

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

    let mut coef_source = active_bounds.map_or_else(
        || VecBackedIntraCoefSource::new(width, height),
        |bounds| {
            VecBackedIntraCoefSource::new_for_mi_columns(
                width,
                height,
                bounds.mi_col_start,
                bounds.mi_col_end,
            )
        },
    );

    // Every keyframe leaf needs at most one 64-pixel edge. Reuse fixed
    // function-local scratch instead of allocating two tiny vectors per
    // leaf. Unavailable edges are ignored under their explicit flags.
    let mut above_pixels = [0u8; 64];
    let mut left_pixels = [0u8; 64];
    for sb_row in 0..sb_rows {
        if let Some(contexts) = picker_coef_contexts.as_mut() {
            for context in contexts {
                context.reset_left_coef();
            }
        }
        for sb_col in active_sb_cols.clone() {
            let current_tile = tile_for_mi_col(&tile_bounds_by_col, sb_col * 8);
            let current_tile_index = picker_coef_contexts.as_ref().map_or(0, |_| {
                tile_bounds_by_col
                    .iter()
                    .position(|bounds| {
                        sb_col * 8 >= bounds.mi_col_start && sb_col * 8 < bounds.mi_col_end
                    })
                    .expect("current keyframe superblock belongs to one tile")
            });
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
                let prediction_side = if leaf_bsize < BlockSize::Block8x8 {
                    8
                } else {
                    bs_pixels
                };

                let have_above = leaf_px_y > 0;
                let have_left = leaf_mi_col > current_tile.mi_col_start;
                if have_above {
                    let above_row_y = leaf_px_y - 1;
                    let row_off = above_row_y * w + leaf_px_x;
                    above_pixels[..prediction_side]
                        .copy_from_slice(&recon_yuv_out.0[row_off..row_off + prediction_side]);
                }
                if have_left {
                    let left_col_x = leaf_px_x - 1;
                    for (r, slot) in left_pixels[..prediction_side].iter_mut().enumerate() {
                        *slot = recon_yuv_out.0[(leaf_px_y + r) * w + left_col_x];
                    }
                }
                let top_left_pixel = if have_above && have_left {
                    recon_yuv_out.0[(leaf_px_y - 1) * w + (leaf_px_x - 1)]
                } else {
                    0u8
                };

                let above_modes = if leaf_mi_row > 0 {
                    picked_y_mode_grid[(leaf_mi_row - 1) * mi_cols + leaf_mi_col]
                        .unwrap_or([IntraMode::Dc; 4])
                } else {
                    [IntraMode::Dc; 4]
                };
                let left_modes = if leaf_mi_col > current_tile.mi_col_start {
                    picked_y_mode_grid[leaf_mi_row * mi_cols + (leaf_mi_col - 1)]
                        .unwrap_or([IntraMode::Dc; 4])
                } else {
                    [IntraMode::Dc; 4]
                };
                let above_mode_ctx = above_modes[2];
                let left_mode_ctx = left_modes[1];

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
                let leaf_quantizer =
                    init_plane_quantizers(i32::from(leaf_qindex), QuantizerDeltas::Y);
                let picked_y_modes = if leaf_bsize < BlockSize::Block8x8 {
                    let mut src = [0u8; 64];
                    for row in 0..8 {
                        let src_start = (leaf_px_y + row) * w + leaf_px_x;
                        src[row * 8..row * 8 + 8]
                            .copy_from_slice(&input_yuv.0[src_start..src_start + 8]);
                    }
                    let mut above = [ABOVE_UNAVAILABLE_FILL; 16];
                    if have_above {
                        let above_start = (leaf_px_y - 1) * w;
                        for (column, slot) in above.iter_mut().enumerate() {
                            *slot = recon_yuv_out.0[above_start + (leaf_px_x + column).min(w - 1)];
                        }
                    }
                    let mut left = [LEFT_UNAVAILABLE_FILL; 8];
                    if have_left {
                        left.copy_from_slice(&left_pixels[..8]);
                    }
                    let above_left = if have_above && have_left {
                        top_left_pixel
                    } else if have_above {
                        LEFT_UNAVAILABLE_FILL
                    } else {
                        ABOVE_UNAVAILABLE_FILL
                    };
                    let picker_coef_ctx = picker_coef_contexts
                        .as_ref()
                        .expect("sub-8 leaves require picker coefficient contexts")
                        .get(current_tile_index)
                        .expect("current keyframe tile has coefficient context");
                    let x4_base = (leaf_mi_col - current_tile.mi_col_start) * 2;
                    let y4_in_sb_base = (leaf_mi_row & 7) * 2;
                    let params = PickIntraSub8x8Params {
                        parent_bsize: BlockSize::Block8x8,
                        src: &src,
                        above: &above,
                        left: &left,
                        above_left,
                        have_above,
                        have_left,
                        y_mode_costs: &y_mode_costs,
                        above_modes,
                        left_modes,
                        quant: &leaf_quantizer,
                        coef_costs: coef_costs
                            .as_ref()
                            .expect("sub-8 leaves require coefficient costs"),
                        above_coef_contexts: [
                            picker_coef_ctx.above_coef(0, x4_base),
                            picker_coef_ctx.above_coef(0, x4_base + 1),
                        ],
                        left_coef_contexts: [
                            picker_coef_ctx.left_coef(0, y4_in_sb_base),
                            picker_coef_ctx.left_coef(0, y4_in_sb_base + 1),
                        ],
                        rdmult: i64::from(compute_rd_mult(
                            i32::from(leaf_qindex),
                            FrameTypeForRd::Key,
                        )),
                        best_rd: i64::MAX,
                    };
                    let result = pick_intra_sub_8x8_y_mode(&params)
                        .expect("unbounded sub-8 keyframe pick must choose four modes");
                    core::array::from_fn(|index| result.sub_winners[index].mode)
                } else if exact_keyframe_sub8 && leaf_bsize == BlockSize::Block8x8 {
                    let mut src = [0u8; 64];
                    for row in 0..8 {
                        let src_start = (leaf_px_y + row) * w + leaf_px_x;
                        src[row * 8..row * 8 + 8]
                            .copy_from_slice(&input_yuv.0[src_start..src_start + 8]);
                    }
                    let mut above = [ABOVE_UNAVAILABLE_FILL; 16];
                    if have_above {
                        let above_start = (leaf_px_y - 1) * w;
                        above[..8].copy_from_slice(
                            &recon_yuv_out.0[above_start + leaf_px_x..above_start + leaf_px_x + 8],
                        );
                        // A Block8x8 leaf contains one Tx8x8, so there is no
                        // causally available transform to its right. VP9
                        // extends the final top sample for directional
                        // predictors instead of reading unreconstructed
                        // pixels from the next leaf.
                        extend_standalone_tx8_above(&mut above);
                    }
                    let mut left = [LEFT_UNAVAILABLE_FILL; 8];
                    if have_left {
                        left.copy_from_slice(&left_pixels[..8]);
                    }
                    let above_left = if have_above && have_left {
                        top_left_pixel
                    } else if have_above {
                        LEFT_UNAVAILABLE_FILL
                    } else {
                        ABOVE_UNAVAILABLE_FILL
                    };
                    let picker_coef_ctx = picker_coef_contexts
                        .as_ref()
                        .expect("exact 8x8 leaves require picker coefficient contexts")
                        .get(current_tile_index)
                        .expect("current keyframe tile has coefficient context");
                    let x4_base = (leaf_mi_col - current_tile.mi_col_start) * 2;
                    let y4_in_sb_base = (leaf_mi_row & 7) * 2;
                    let above_ctx = picker_coef_ctx.above_coef(0, x4_base)
                        | picker_coef_ctx.above_coef(0, x4_base + 1);
                    let left_ctx = picker_coef_ctx.left_coef(0, y4_in_sb_base)
                        | picker_coef_ctx.left_coef(0, y4_in_sb_base + 1);
                    let result = pick_intra8x8_y_mode(&PickIntra8x8Params {
                        src: &src,
                        above: &above,
                        left: &left,
                        above_left,
                        have_above,
                        have_left,
                        bmode_costs: &y_mode_costs[above_mode_ctx as usize][left_mode_ctx as usize],
                        quant: &leaf_quantizer,
                        coef_costs: coef_costs
                            .as_ref()
                            .expect("exact 8x8 leaves require coefficient costs"),
                        coeff_ctx: combine_entropy_contexts(above_ctx, left_ctx),
                        rdmult: i64::from(compute_rd_mult(
                            i32::from(leaf_qindex),
                            FrameTypeForRd::Key,
                        )),
                        best_rd: i64::MAX,
                    })
                    .expect("unbounded 8x8 keyframe pick must choose a mode");
                    [result.mode; 4]
                } else if leaf_bsize == BlockSize::Block64x64 {
                    [IntraMode::Dc; 4]
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
                        above_pixels: &above_pixels[..prediction_side],
                        left_pixels: &left_pixels[..prediction_side],
                        top_left_pixel,
                        have_above,
                        have_left,
                        quantizer: &leaf_quantizer,
                        q_index: leaf_qindex,
                        frame_kind: rvp9_core::bitstream::header::FrameKind::KeyFrame,
                        above_mode_ctx,
                        left_mode_ctx,
                        y_mode_costs: &y_mode_costs,
                    };
                    [pick_realtime_intra_mode_luma(&params).best_mode; 4]
                };
                let picked_mode = if leaf_bsize < BlockSize::Block8x8 {
                    picked_y_modes[3]
                } else {
                    picked_y_modes[0]
                };

                // A0' diagnostic — dump per-leaf KF intra mode pick when
                // RVP9_DIAG_INTRA is set in the environment. Sampling at
                // mod-16 keeps the output bounded on a 256x256 frame.
                if std::env::var("RVP9_DIAG_INTRA").is_ok() {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "    intra_leaf sb=({},{}) px=({},{}) bs={:?} q={} → mode={:?}",
                            sb_row,
                            sb_col,
                            leaf_px_x,
                            leaf_px_y,
                            leaf_bsize,
                            leaf_qindex,
                            picked_mode,
                        );
                    }
                }

                sb_partitions[sb_idx][leaf_idx].predicted_y_modes = picked_y_modes;
                let n_8x8 = (bs_pixels / 8).max(1);
                for dr in 0..n_8x8 {
                    for dc in 0..n_8x8 {
                        let mi_r = leaf_mi_row + dr;
                        let mi_c = leaf_mi_col + dc;
                        picked_y_mode_grid[mi_r * mi_cols + mi_c] = Some(picked_y_modes);
                    }
                }

                // libvpx's speed-4+ keyframe path uses USE_LARGESTALL with
                // frame tx_mode ALLOW_16X16. Pick the largest transform that
                // both the leaf and that frame-level cap allow; no per-leaf
                // transform-size RD sweep is run in this realtime policy.
                let chosen_intra_tx = realtime_keyframe_tx_size(leaf_bsize);
                sb_partitions[sb_idx][leaf_idx].picked_tx_size = chosen_intra_tx;

                // libvpx's speed-4+ realtime policy masks every chroma TX
                // size to DC. Avoid building UV edges and scoring modes the
                // canonical encoder never evaluates in this configuration.
                let picked_uv_mode = IntraMode::Dc;
                sb_partitions[sb_idx][leaf_idx].predicted_uv_mode = picked_uv_mode;

                // Encode this leaf directly into the running coefficient
                // source. Recon state from earlier leaves is preserved and
                // remains the causal neighbour input for the next pick.
                simulate_dc_intra_encode_partitions_no_zero_with_tiles(
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
                    sb_partitions,
                    log2_tile_cols,
                    Some((sb_idx, leaf_idx)),
                    &mut coef_source,
                );
                sb_partitions[sb_idx][leaf_idx].skip = intra_leaf_all_zero(
                    &coef_source,
                    leaf_mi_row,
                    leaf_mi_col,
                    leaf_bsize,
                    chosen_intra_tx,
                );
                // F.0.2 oracle — record after residual generation so its
                // skip state matches the syntax emitted for this leaf.
                if crate::f02_oracle::is_enabled() {
                    let luma_eob_nonzero = intra_leaf_luma_eob_nonzero(
                        &coef_source,
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        chosen_intra_tx,
                    );
                    crate::f02_oracle::record_decision(crate::f02_oracle::BlockDecision {
                        frame_idx: crate::f02_oracle::current_frame_idx(),
                        mi_row: leaf_mi_row as u16,
                        mi_col: leaf_mi_col as u16,
                        bsize: leaf_bsize,
                        mode: crate::f02_oracle::BlockDecisionMode::Intra(picked_mode),
                        ref_frame: RefFrame::IntraFrame,
                        mv_row: 0,
                        mv_col: 0,
                        // The keyframe picker does not expose these F.0.2
                        // diagnostic estimates through its result shape.
                        mode_bits_est: 0,
                        mv_bits_est: 0,
                        ref_bits_est: 0,
                        filter_bits_est: 0,
                        skip_bits_est: 0,
                        coef_bits_est: 0,
                        picker_rate: 0,
                        picker_dist: 0,
                        picker_rd_cost: 0,
                        model_var_y: 0,
                        model_tx_size: chosen_intra_tx,
                        model_skip_txfm: 0,
                        luma_coef_rate: 0,
                        luma_coef_dist: 0,
                        uv_model_rate: 0,
                        uv_model_dist: 0,
                        uv_u_sensitive: false,
                        uv_v_sensitive: false,
                        partition_u_sensitive: false,
                        partition_v_sensitive: false,
                        last_candidates: [Default::default(); 4],
                        sse_after_pred: 0,
                        interp_filter: None,
                        skip_flag: sb_partitions[sb_idx][leaf_idx].skip,
                        q_index: leaf_qindex,
                        tx_size: chosen_intra_tx,
                        luma_eob_nonzero,
                    });
                }
                if let Some(contexts) = picker_coef_contexts.as_mut() {
                    update_intra_picker_luma_coef_context(
                        &mut contexts[current_tile_index],
                        &coef_source,
                        leaf_mi_row,
                        leaf_mi_col,
                        current_tile.mi_col_start,
                        leaf_bsize,
                        chosen_intra_tx,
                    );
                }
            }
        }
    }

    coef_source
}

fn intra_leaf_all_zero(
    coef_source: &VecBackedIntraCoefSource,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    luma_tx: TxSize,
) -> bool {
    use rvp9_core::bitstream::IntraCoefSource;

    for plane in 0..3 {
        let ss_x = F5_SUBSAMPLING_X[plane];
        let ss_y = F5_SUBSAMPLING_Y[plane];
        let plane_bsize = if plane == 0 {
            leaf_bsize
        } else {
            f5_uv_plane_block_size(leaf_bsize)
        };
        let plane_tx = if plane == 0 {
            luma_tx
        } else {
            f5_uv_plane_tx_size(leaf_bsize, luma_tx)
        };
        let is_sub8 = leaf_bsize < BlockSize::Block8x8;
        let (bw_4x4, bh_4x4) = if is_sub8 {
            (2 >> ss_x, 2 >> ss_y)
        } else {
            (
                plane_bsize.n_4x4_wide() as usize,
                plane_bsize.n_4x4_high() as usize,
            )
        };
        let step = 1usize << plane_tx as usize;
        for y4 in (0..bh_4x4).step_by(step) {
            for x4 in (0..bw_4x4).step_by(step) {
                let (_, eob) = coef_source.get_block_coefs(mi_row, mi_col, plane, x4, y4, plane_tx);
                if eob != 0 {
                    return false;
                }
            }
        }
    }
    true
}

fn intra_leaf_luma_eob_nonzero(
    coef_source: &VecBackedIntraCoefSource,
    mi_row: usize,
    mi_col: usize,
    leaf_bsize: BlockSize,
    luma_tx: TxSize,
) -> bool {
    use rvp9_core::bitstream::IntraCoefSource;

    let (bw_4x4, bh_4x4) = if leaf_bsize < BlockSize::Block8x8 {
        (2, 2)
    } else {
        (
            leaf_bsize.n_4x4_wide() as usize,
            leaf_bsize.n_4x4_high() as usize,
        )
    };
    let step = 1usize << luma_tx as usize;
    for y4 in (0..bh_4x4).step_by(step) {
        for x4 in (0..bw_4x4).step_by(step) {
            let (_, eob) = coef_source.get_block_coefs(mi_row, mi_col, 0, x4, y4, luma_tx);
            if eob != 0 {
                return true;
            }
        }
    }
    false
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
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct InterPickerMiGridLayout {
    mi_col_start: usize,
    mi_cols: usize,
}

impl InterPickerMiGridLayout {
    fn new(mi_col_start: usize, mi_cols: usize) -> Self {
        Self {
            mi_col_start,
            mi_cols,
        }
    }

    fn index(self, mi_row: usize, mi_col: usize) -> Option<usize> {
        let local_col = mi_col.checked_sub(self.mi_col_start)?;
        (local_col < self.mi_cols).then_some(mi_row * self.mi_cols + local_col)
    }

    fn get(
        self,
        mi_grid: &[Option<rvp9_core::mode_info::ModeInfo>],
        mi_row: usize,
        mi_col: usize,
    ) -> Option<&rvp9_core::mode_info::ModeInfo> {
        mi_grid.get(self.index(mi_row, mi_col)?)?.as_ref()
    }
}

/// Serial ownership adapter for the causal mode and previous-MV grids.
///
/// Row-MT will replace the implementation behind this seam with row-disjoint
/// writes and wavefront-ordered reads. Keeping both grids together ensures a
/// committed leaf cannot update mode propagation without also updating the
/// next-frame MV history.
struct InterRowGrids<'a> {
    mi_grid: Vec<Option<rvp9_core::mode_info::ModeInfo>>,
    layout: InterPickerMiGridLayout,
    out_mv_grid: &'a mut [PrevFrameMvRef],
    out_mi_cols: usize,
}

impl<'a> InterRowGrids<'a> {
    fn new(
        mi_rows: usize,
        layout: InterPickerMiGridLayout,
        out_mv_grid: &'a mut [PrevFrameMvRef],
        out_mi_cols: usize,
    ) -> Self {
        Self {
            mi_grid: vec![None; mi_rows * layout.mi_cols],
            layout,
            out_mv_grid,
            out_mi_cols,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[Option<rvp9_core::mode_info::ModeInfo>] {
        &self.mi_grid
    }

    #[inline]
    fn get(&self, mi_row: usize, mi_col: usize) -> Option<&rvp9_core::mode_info::ModeInfo> {
        self.layout.get(&self.mi_grid, mi_row, mi_col)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn splat_mode(
        &mut self,
        leaf_mi_row: usize,
        leaf_mi_col: usize,
        leaf_bsize: BlockSize,
        picked_mode: InterMode,
        picked_mv: Mv,
        segment_id: u8,
        skip: bool,
        picked_ref_frame: RefFrame,
        picked_intra_mode: Option<IntraMode>,
        picked_interp_filter: InterpFilter,
    ) {
        splat_inter_leaf_into_mi_grid(
            &mut self.mi_grid,
            self.layout,
            leaf_mi_row,
            leaf_mi_col,
            leaf_bsize,
            picked_mode,
            picked_mv,
            segment_id,
            skip,
            picked_ref_frame,
            picked_intra_mode,
            picked_interp_filter,
        );
    }

    #[inline]
    fn splat_final(
        &mut self,
        leaf_mi_row: usize,
        leaf_mi_col: usize,
        leaf: InterPartitionLeafDesc,
    ) {
        splat_final_inter_leaf_into_grids(
            &mut self.mi_grid,
            self.layout,
            self.out_mv_grid,
            self.out_mi_cols,
            leaf_mi_row,
            leaf_mi_col,
            leaf,
        );
    }

    fn into_simulation_grid(self) -> InterSimulationMiGrid {
        InterSimulationMiGrid {
            cells: self.mi_grid,
            mi_col_start: self.layout.mi_col_start,
            mi_cols: self.layout.mi_cols,
        }
    }
}

struct InterPickerMiGridLookup<'a> {
    mi_grid: &'a [Option<rvp9_core::mode_info::ModeInfo>],
    layout: InterPickerMiGridLayout,
    base_mi_row: i32,
    base_mi_col: i32,
    tile_bounds: ModesTileBounds,
}

impl<'a> InterPickerMiGridLookup<'a> {
    fn new(
        mi_grid: &'a [Option<rvp9_core::mode_info::ModeInfo>],
        layout: InterPickerMiGridLayout,
        base_mi_row: usize,
        base_mi_col: usize,
        tile_bounds: ModesTileBounds,
    ) -> Self {
        Self {
            mi_grid,
            layout,
            base_mi_row: base_mi_row as i32,
            base_mi_col: base_mi_col as i32,
            tile_bounds,
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
        if row < self.tile_bounds.mi_row_start as i32
            || col < self.tile_bounds.mi_col_start as i32
            || row >= self.tile_bounds.mi_row_end as i32
            || col >= self.tile_bounds.mi_col_end as i32
        {
            return None;
        }
        self.layout.get(self.mi_grid, row as usize, col as usize)
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
    layout: InterPickerMiGridLayout,
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
    // Phase 2b — an intra winner must enter the causal grid as intra.
    // Otherwise later NEAREST/NEAR resolution sees an invalid inter
    // neighbour while the writer and decoder see `BlockMode::Intra`.
    picked_intra_mode: Option<IntraMode>,
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
        mode: match picked_intra_mode {
            Some(y_mode) => BlockMode::Intra {
                y_mode,
                uv_mode: IntraMode::Dc,
            },
            None => BlockMode::Inter {
                mode: picked_mode,
                ref_frames: [picked_ref_frame, RefFrame::NoRefFrame],
                mvs: [picked_mv, Mv::ZERO],
                interp_filter: picked_interp_filter,
            },
        },
        bmi: Default::default(),
    };
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let mi_row = leaf_mi_row + dr;
            let mi_col = leaf_mi_col + dc;
            let idx = layout
                .index(mi_row, mi_col)
                .expect("inter leaf must stay inside its causal MI grid");
            mi_grid[idx] = Some(cell);
        }
    }
}

fn f02_oracle_mode(
    picked_ref_frame: RefFrame,
    picked_mode: InterMode,
    picked_intra_mode: Option<IntraMode>,
) -> crate::f02_oracle::BlockDecisionMode {
    match picked_ref_frame {
        RefFrame::IntraFrame => crate::f02_oracle::BlockDecisionMode::Intra(
            picked_intra_mode.expect("IntraFrame pick must have picked_intra_mode"),
        ),
        _ => crate::f02_oracle::BlockDecisionMode::Inter(picked_mode),
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
    /// W-COMPOUND — second reference for a compound (bidirectional)
    /// winner (mi-grid slot 1 = `AltrefFrame`). `RefFrame::NoRefFrame`
    /// for single-ref leaves (the default). When this is a real inter
    /// ref the leaf is compound: the writer derives `is_compound` from
    /// `ref_frames[1].is_inter()` (`mode_info.rs:1013`) and the encoder
    /// simulator builds the averaged {slot0, slot1} prediction.
    pub picked_ref_frame2: RefFrame,
    /// W-COMPOUND — MV for the slot 1 (ALTREF) reference of a compound
    /// winner; `Mv::ZERO` for single-ref leaves.
    pub picked_mv2: Mv,
    /// Picked luma transform size splatted by both mi-grid builders and
    /// signalled by the modes writer under `tx_mode = TxModeSelect`.
    /// The default realtime emit path reuses the picker's
    /// `calculate_tx_size` model decision; `RVP9_RT_EMIT_TX_RD=1` enables
    /// the focused emit-side RD redecision for diagnostics.
    pub picked_tx_size: TxSize,
}

impl InterPartitionLeafDesc {
    pub(crate) fn mode_info(self) -> rvp9_core::mode_info::ModeInfo {
        use rvp9_core::mode_info::{BlockMode, ModeInfo};

        ModeInfo {
            block_size: self.leaf_bsize,
            tx_size: if self.skip {
                rvp9_core::block::max_txsize(self.leaf_bsize)
            } else {
                self.picked_tx_size
            },
            skip: self.skip,
            segment_id: self.segment_id,
            seg_id_predicted: false,
            mode: if let Some(y_mode) = self.picked_intra_mode {
                BlockMode::Intra {
                    y_mode,
                    uv_mode: IntraMode::Dc,
                }
            } else {
                BlockMode::Inter {
                    mode: self.picked_mode,
                    ref_frames: [self.picked_ref_frame, self.picked_ref_frame2],
                    mvs: [self.picked_mv, self.picked_mv2],
                    interp_filter: self.picked_interp_filter,
                }
            },
            bmi: Default::default(),
        }
    }

    #[inline]
    fn mv_ref(self) -> PrevFrameMvRef {
        if self.picked_ref_frame == RefFrame::IntraFrame {
            PrevFrameMvRef {
                mvs: [Mv::ZERO; 2],
                ref_frames: [RefFrame::IntraFrame, RefFrame::NoRefFrame],
            }
        } else {
            PrevFrameMvRef {
                mvs: [self.picked_mv, self.picked_mv2],
                ref_frames: [self.picked_ref_frame, self.picked_ref_frame2],
            }
        }
    }
}

fn splat_final_inter_leaf_into_grids(
    mi_grid: &mut [Option<rvp9_core::mode_info::ModeInfo>],
    layout: InterPickerMiGridLayout,
    out_mv_grid: &mut [PrevFrameMvRef],
    out_mi_cols: usize,
    leaf_mi_row: usize,
    leaf_mi_col: usize,
    leaf: InterPartitionLeafDesc,
) {
    let n_8x8_wide = leaf.leaf_bsize.n_8x8_wide() as usize;
    let n_8x8_high = leaf.leaf_bsize.n_8x8_high() as usize;
    let mode_info = leaf.mode_info();
    let mv_ref = leaf.mv_ref();
    for dr in 0..n_8x8_high {
        for dc in 0..n_8x8_wide {
            let mi_row = leaf_mi_row + dr;
            let mi_col = leaf_mi_col + dc;
            let local_index = layout
                .index(mi_row, mi_col)
                .expect("inter leaf must stay inside its causal MI grid");
            mi_grid[local_index] = Some(mode_info);
            out_mv_grid[mi_row * out_mi_cols + mi_col] = mv_ref;
        }
    }
}

pub(crate) struct RecordedCoefTile {
    pub(crate) tile_col: usize,
    pub(crate) retained: RetainedCoefTokens,
    pub(crate) counts: FrameCounts,
    coef_ctx: TileCoefCtx,
    last_sb_row: Option<usize>,
    mi_col_start: usize,
    mi_col_end: usize,
}

pub(crate) struct InterSimulationResult {
    pub(crate) coefs: VecBackedInterCoefSource,
    pub(crate) recorded_coef_tile: Option<RecordedCoefTile>,
    pub(crate) mi_grid: InterSimulationMiGrid,
}

/// Final causal mode-info grid produced while the inter picker commits leaves.
///
/// Parallel tile jobs retain only their local column span; the frame loop
/// merges those disjoint row slices instead of rebuilding every cell from the
/// partition descriptors after the picker barrier.
pub(crate) struct InterSimulationMiGrid {
    pub(crate) cells: Vec<Option<rvp9_core::mode_info::ModeInfo>>,
    pub(crate) mi_col_start: usize,
    pub(crate) mi_cols: usize,
}

fn record_encoded_inter_leaf(
    tile: &mut RecordedCoefTile,
    bw: &mut BoolWriter<'_>,
    frame_context: &FrameContext,
    coef_source: &VecBackedInterCoefSource,
    mi_row: usize,
    mi_col: usize,
    leaf: InterPartitionLeafDesc,
) {
    debug_assert!(mi_col >= tile.mi_col_start && mi_col < tile.mi_col_end);
    let sb_row = mi_row / 8;
    if tile.last_sb_row != Some(sb_row) {
        tile.coef_ctx.reset_left_coef();
        tile.last_sb_row = Some(sb_row);
    }
    let mi = leaf.mode_info();
    record_inter_leaf_coefficients(
        bw,
        frame_context,
        TxMode::TxModeSelect,
        &mut tile.coef_ctx,
        coef_source,
        mi_row,
        mi_col,
        tile.mi_col_start,
        leaf.leaf_bsize,
        &mi,
        &mut tile.counts,
        &mut tile.retained,
    );
}

fn segment_id_for_hint_cell(
    hints: &[InterPartitionLeafDesc],
    mi_row_in_sb: usize,
    mi_col_in_sb: usize,
) -> u8 {
    hints
        .iter()
        .find(|h| {
            let row_end = h.mi_row_in_sb + h.leaf_bsize.n_8x8_high() as usize;
            let col_end = h.mi_col_in_sb + h.leaf_bsize.n_8x8_wide() as usize;
            mi_row_in_sb >= h.mi_row_in_sb
                && mi_row_in_sb < row_end
                && mi_col_in_sb >= h.mi_col_in_sb
                && mi_col_in_sb < col_end
        })
        .map(|h| h.segment_id)
        .unwrap_or(0)
}

/// Pinned libvpx restricts the first post-refresh realtime pick to LAST
/// (`vp9_pickmode.c:1907-1911`). Require exact storage identity for every
/// prediction plane as well as the matching non-SVC/non-ALTREF topology;
/// a denoiser-owned ME view therefore keeps GOLDEN eligible.
fn golden_ref_duplicates_last(
    allow_last_only_topology: bool,
    frames_since_golden: u32,
    me_ref_yuv: (&[u8], &[u8], &[u8]),
    last_recon_yuv: (&[u8], &[u8], &[u8]),
    golden_recon_yuv: (&[u8], &[u8], &[u8]),
) -> bool {
    fn same_storage(a: (&[u8], &[u8], &[u8]), b: (&[u8], &[u8], &[u8])) -> bool {
        a.0.as_ptr() == b.0.as_ptr() && a.1.as_ptr() == b.1.as_ptr() && a.2.as_ptr() == b.2.as_ptr()
    }

    allow_last_only_topology
        && frames_since_golden == 0
        && same_storage(me_ref_yuv, last_recon_yuv)
        && same_storage(golden_recon_yuv, last_recon_yuv)
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
/// Reference-buffer split (libvpx parity, denoiser / filtered-ALTREF
/// prerequisite): the LAST slot is exposed as two parallel views.
/// `me_ref_yuv` is the ME-side reference read by motion search,
/// `find_mv_refs`, candidate SSE/RD, and the partition picker (libvpx
/// `running_avg_y`, `vp9/encoder/vp9_denoiser.c:294-304`).
/// `last_recon_yuv` is the residual-coding reference — the bit-exact
/// decoded reconstruction the final MC prediction reads from before the
/// residual IT-add (libvpx `cm->frame_refs[]`). The cross-decoder gate
/// pins `last_recon_yuv` byte-exact against the decoder; `me_ref_yuv`
/// is free to be a filtered/denoised view in a later wave. Until then
/// the caller (`StreamEncoder`) keeps them byte-identical.
/// `recon_yuv_out` is filled with the new frame's reconstruction (the
/// encoder's "what the decoder will produce" prediction-correctness
/// oracle).
///
/// Walks SBs/leaves in raster order — same convention as the F10.1b
/// intra simulator + writer.
fn p8_default_enabled(
    rt_large_recovery: bool,
    rc_mode: crate::rate_control::RcMode,
    base_qindex: u8,
) -> bool {
    !rt_large_recovery
        && (!matches!(rc_mode, crate::rate_control::RcMode::Vbr) || base_qindex <= 200)
}

fn p8_vbr_qmax_enabled(
    rt_large_recovery: bool,
    rc_mode: crate::rate_control::RcMode,
    base_qindex: u8,
    qmax: u8,
) -> bool {
    !rt_large_recovery && matches!(rc_mode, crate::rate_control::RcMode::Vbr) && base_qindex <= qmax
}

#[inline(always)]
fn build_mv_candidates_if_available(
    available: bool,
    build: impl FnOnce() -> rvp9_core::mode_decision::inter::MvCandidates,
) -> rvp9_core::mode_decision::inter::MvCandidates {
    if available {
        build()
    } else {
        rvp9_core::mode_decision::inter::MvCandidates::default()
    }
}

fn mv_in_strict_limits_or_visible_full_chroma_pel(
    mv: Mv,
    limits: rvp9_core::mv_search::MvLimits,
    block_origin: (usize, usize),
    block_size: (usize, usize),
    frame_size: (usize, usize),
) -> bool {
    let col_fp = i32::from(mv.col) >> 3;
    let row_fp = i32::from(mv.row) >> 3;
    if col_fp >= limits.col_min
        && col_fp <= limits.col_max
        && row_fp >= limits.row_min
        && row_fp <= limits.row_max
    {
        return true;
    }
    // A Q3 vector must be a multiple of 16 to land on an integer sample
    // after 4:2:0 chroma subsampling. Odd whole-luma-pel vectors still
    // require a half-pel chroma filter halo.
    if (mv.row | mv.col) & 15 != 0 {
        return false;
    }

    let ref_x = block_origin.0 as i64 + i64::from(col_fp);
    let ref_y = block_origin.1 as i64 + i64::from(row_fp);
    ref_x >= 0
        && ref_y >= 0
        && ref_x + block_size.0 as i64 <= frame_size.0 as i64
        && ref_y + block_size.1 as i64 <= frame_size.1 as i64
}

/// Per-block unrestricted-motion-vector window from libvpx
/// `vp9_encodeframe.c:set_offsets`. Units are full pixels. The reference
/// planes are tight, but both picker and reconstruction prediction paths
/// edge-replicate reads that cross the visible frame boundary.
fn libvpx_inter_mv_limits(
    mi_row: usize,
    mi_col: usize,
    block_mi_w: usize,
    block_mi_h: usize,
    frame_mi_rows: usize,
    frame_mi_cols: usize,
) -> rvp9_core::mv_search::MvLimits {
    const MI_SIZE: i32 = 8;
    const VP9_INTERP_EXTEND: i32 = 4;
    rvp9_core::mv_search::MvLimits::new(
        -(((mi_col + block_mi_w) as i32 * MI_SIZE) + VP9_INTERP_EXTEND),
        ((frame_mi_cols - mi_col) as i32 * MI_SIZE) + VP9_INTERP_EXTEND,
        -(((mi_row + block_mi_h) as i32 * MI_SIZE) + VP9_INTERP_EXTEND),
        ((frame_mi_rows - mi_row) as i32 * MI_SIZE) + VP9_INTERP_EXTEND,
    )
}

fn tight_inter_mv_limits(
    block_origin: (usize, usize),
    block_size: (usize, usize),
    frame_size: (usize, usize),
) -> rvp9_core::mv_search::MvLimits {
    let (x, y) = block_origin;
    let (bw, bh) = block_size;
    let (width, height) = frame_size;
    let col_min_luma = -(x as i32) + 3;
    let col_max_luma = width as i32 - x as i32 - bw as i32 - 4;
    let row_min_luma = -(y as i32) + 3;
    let row_max_luma = height as i32 - y as i32 - bh as i32 - 4;
    let col_min_chroma = 6 - x as i32;
    let col_max_chroma = width as i32 - x as i32 - bw as i32 - 7;
    let row_min_chroma = 6 - y as i32;
    let row_max_chroma = height as i32 - y as i32 - bh as i32 - 7;
    rvp9_core::mv_search::MvLimits::new(
        col_min_luma.max(col_min_chroma),
        col_max_luma.min(col_max_chroma),
        row_min_luma.max(row_min_chroma),
        row_max_luma.min(row_max_chroma),
    )
}

fn stage4a_block16_split_admission(
    p8_enabled: bool,
    rt_large_recovery: bool,
    rc_mode: crate::rate_control::RcMode,
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
    override_policy: crate::frame_loop::Stage4aProductSplitOverride,
) -> rvp9_core::mode_decision::nonrd_partition::Block16SplitAdmission {
    use rvp9_core::mode_decision::nonrd_partition::Block16SplitAdmission;

    if matches!(
        override_policy,
        crate::frame_loop::Stage4aProductSplitOverride::ForceStrict
    ) {
        return Block16SplitAdmission::Strict;
    }
    if p8_enabled
        && !rt_large_recovery
        && matches!(rc_mode, crate::rate_control::RcMode::Vbr)
        && matches!(
            deadline,
            rvp9_core::mode_decision::speed_features::Deadline::Realtime
        )
        && speed >= 7
    {
        Block16SplitAdmission::Slack1Over8
    } else {
        Block16SplitAdmission::Strict
    }
}

fn stage4a_descent_breakout_16_dist_qmul(
    p8_enabled: bool,
    rt_large_recovery: bool,
    rc_mode: crate::rate_control::RcMode,
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
    override_qmul: Option<u32>,
) -> Option<u32> {
    match override_qmul {
        Some(0) => None,
        Some(qmul) => Some(qmul),
        None if p8_enabled
            && !rt_large_recovery
            && matches!(rc_mode, crate::rate_control::RcMode::Cbr)
            && speed == 7
            && matches!(
                deadline,
                rvp9_core::mode_decision::speed_features::Deadline::Realtime
            ) =>
        {
            Some(1600)
        }
        None => None,
    }
}

fn stage4a_suppress_final_newmv(
    block_size: BlockSize,
    cached_mode: InterMode,
    cached_partition_distortion: u32,
    base_qindex: u8,
    descent_breakout_qmul: u32,
) -> bool {
    const FINAL_NEWMV_DIST_QMUL: u32 = 600;

    matches!(block_size, BlockSize::Block16x16)
        && cached_mode != InterMode::NewMv
        && cached_partition_distortion
            <= u32::from(base_qindex)
                .saturating_mul(descent_breakout_qmul.min(FINAL_NEWMV_DIST_QMUL))
}

fn a4_partition_rd_mult_for_env(base_rd_mult: u32) -> u32 {
    let Some(num) = std::env::var("RVP9_A4_RDMULT_NUM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    else {
        return base_rd_mult;
    };
    let Some(den) = std::env::var("RVP9_A4_RDMULT_DEN")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    else {
        return base_rd_mult;
    };
    if den == 0 {
        return base_rd_mult;
    }
    let scaled = (u64::from(base_rd_mult) * u64::from(num) + u64::from(den / 2)) / u64::from(den);
    scaled.min(u64::from(u32::MAX)) as u32
}

fn inter_rd_calibration_for_mode(
    rc_mode: crate::rate_control::RcMode,
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
    cyclic_refresh_active: bool,
) -> rvp9_core::mode_decision::inter::InterRdCalibration {
    let speed = speed.unsigned_abs();
    let calibrated_rc = match rc_mode {
        crate::rate_control::RcMode::Cbr | crate::rate_control::RcMode::Vbr => speed >= 6,
        crate::rate_control::RcMode::Cq
        | crate::rate_control::RcMode::Q
        | crate::rate_control::RcMode::VbrTwoPass => false,
    };
    if calibrated_rc
        && !cyclic_refresh_active
        && matches!(
            deadline,
            rvp9_core::mode_decision::speed_features::Deadline::Realtime
        )
    {
        rvp9_core::mode_decision::inter::InterRdCalibration::realtime()
            .with_large_block_model(matches!(rc_mode, crate::rate_control::RcMode::Cbr))
    } else {
        rvp9_core::mode_decision::inter::InterRdCalibration::default()
    }
}

fn golden_real_block_yrd_enabled(
    rc_mode: crate::rate_control::RcMode,
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
    cyclic_refresh_active: bool,
    is_golden_refresh: bool,
    override_value: Option<bool>,
) -> bool {
    if !is_golden_refresh {
        return false;
    }
    override_value.unwrap_or_else(|| {
        matches!(rc_mode, crate::rate_control::RcMode::Vbr)
            && speed.unsigned_abs() == 7
            && matches!(
                deadline,
                rvp9_core::mode_decision::speed_features::Deadline::Realtime
            )
            && !cyclic_refresh_active
    })
}

fn newmv_diff_bias_for_mode(rc_mode: crate::rate_control::RcMode, env_enabled: bool) -> bool {
    env_enabled && matches!(rc_mode, crate::rate_control::RcMode::Cbr)
}

fn partition_color_sensitivity(
    src_uv: (&[u8], &[u8]),
    ref_uv: (&[u8], &[u8]),
    frame_size: (usize, usize),
    sb_origin: (usize, usize),
    partition_mv: Mv,
    y_sad: u32,
) -> [bool; 2] {
    let (width, height) = frame_size;
    let uv_width = width / 2;
    let uv_height = height / 2;
    let uv_x = sb_origin.0 / 2;
    let uv_y = sb_origin.1 / 2;
    let mut sensitivity = [false; 2];

    for (plane, (src, reference)) in [src_uv.0, src_uv.1]
        .into_iter()
        .zip([ref_uv.0, ref_uv.1])
        .enumerate()
    {
        let mut predictor = [0u8; 32 * 32];
        build_inter_pred_block_unscaled(
            reference,
            uv_width,
            uv_width,
            uv_height,
            uv_x,
            uv_y,
            32,
            32,
            i32::from(partition_mv.col),
            i32::from(partition_mv.row),
            &mut predictor,
            32,
            InterpFilter::Bilinear,
        );
        let sad32 = rvp9_core::dsp::get_ref().sad(BlockSize::Block32x32);
        let uv_sad = if uv_x + 32 <= uv_width && uv_y + 32 <= uv_height {
            sad32(&src[uv_y * uv_width + uv_x..], uv_width, &predictor, 32)
        } else {
            let mut source_block = [0u8; 32 * 32];
            for row in 0..32 {
                let src_y = (uv_y + row).min(uv_height - 1);
                for col in 0..32 {
                    let src_x = (uv_x + col).min(uv_width - 1);
                    source_block[row * 32 + col] = src[src_y * uv_width + src_x];
                }
            }
            sad32(&source_block, 32, &predictor, 32)
        };
        sensitivity[plane] = uv_sad > (y_sad >> 2);
    }

    sensitivity
}

fn zero_mv_sad64_edge_clamped(
    src: &[u8],
    reference: &[u8],
    frame_size: (usize, usize),
    sb_origin: (usize, usize),
) -> u32 {
    let (width, height) = frame_size;
    let mut sad = 0u32;
    for row in 0..64 {
        let y = (sb_origin.1 + row).min(height - 1);
        for col in 0..64 {
            let x = (sb_origin.0 + col).min(width - 1);
            let offset = y * width + x;
            sad += u32::from(src[offset].abs_diff(reference[offset]));
        }
    }
    sad
}

fn full_pixel_search_method_for_mode(
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
) -> rvp9_core::mode_decision::inter::FullPixelSearchMethod {
    if matches!(
        deadline,
        rvp9_core::mode_decision::speed_features::Deadline::Realtime
    ) && speed.unsigned_abs() >= 7
    {
        rvp9_core::mode_decision::inter::FullPixelSearchMethod::FastDiamond
    } else {
        rvp9_core::mode_decision::inter::FullPixelSearchMethod::Nstep
    }
}

fn a4_descent_independent_mv_default(
    rc_mode: crate::rate_control::RcMode,
    speed: i32,
    deadline: rvp9_core::mode_decision::speed_features::Deadline,
) -> bool {
    matches!(rc_mode, crate::rate_control::RcMode::Vbr)
        && matches!(
            deadline,
            rvp9_core::mode_decision::speed_features::Deadline::Realtime
        )
        && speed.unsigned_abs() >= 7
}

fn variance_hint_cell_code(bs: BlockSize) -> &'static str {
    match bs {
        BlockSize::Block8x8 => "8",
        BlockSize::Block16x16 => "16",
        BlockSize::Block32x32 => "32",
        BlockSize::Block64x64 => "64",
        _ => "other",
    }
}

fn dump_variance_hint_cells(
    base_qindex: u8,
    sb_row: usize,
    sb_col: usize,
    hint: &rvp9_core::mode_decision::partition::PartitionDecision,
) {
    let mut area8_counts = [0usize; 4];
    let mut grid = String::new();
    for (r, row) in hint.cells.iter().enumerate() {
        if r != 0 {
            grid.push('/');
        }
        for (c, &bs) in row.iter().enumerate() {
            if c != 0 {
                grid.push(',');
            }
            grid.push_str(variance_hint_cell_code(bs));
            match bs {
                BlockSize::Block8x8 => area8_counts[0] += 1,
                BlockSize::Block16x16 => area8_counts[1] += 1,
                BlockSize::Block32x32 => area8_counts[2] += 1,
                BlockSize::Block64x64 => area8_counts[3] += 1,
                _ => {}
            }
        }
    }

    #[allow(clippy::print_stderr)]
    {
        eprintln!(
            "RVP9_VARIANCE_HINT_DUMP q={base_qindex} sb=({sb_row},{sb_col}) \
             area8_hist 8x8={} 16x16={} 32x32={} 64x64={} grid={grid}",
            area8_counts[0], area8_counts[1], area8_counts[2], area8_counts[3]
        );
    }
}

fn should_skip_realtime_luma_transform(
    quant_fp: bool,
    skip_txfm: rvp9_core::mode_decision::model_rd::SkipTxfm,
) -> bool {
    quant_fp
        && matches!(
            skip_txfm,
            rvp9_core::mode_decision::model_rd::SkipTxfm::AcDc
        )
}

fn active_single_ref_frame_costs(
    frame_context: &FrameContext,
    above: Option<&ModeInfo>,
    left: Option<&ModeInfo>,
) -> [u32; 4] {
    use crate::rdo::ref_costs::{
        RefFrameProbabilities, ReferenceMode as CostReferenceMode,
        estimate_ref_frame_costs_from_probs,
    };
    use rvp9_core::inter::{
        get_intra_inter_context, get_pred_context_single_ref_p1, get_pred_context_single_ref_p2,
    };

    let intra_inter_ctx = get_intra_inter_context(above, left);
    let single_ref_p1_ctx = get_pred_context_single_ref_p1(above, left);
    let single_ref_p2_ctx = get_pred_context_single_ref_p2(above, left);
    estimate_ref_frame_costs_from_probs(
        false,
        CostReferenceMode::SingleReference,
        RefFrameProbabilities {
            intra_inter: frame_context.intra_inter_prob[intra_inter_ctx],
            single_ref_p1: frame_context.single_ref_prob[single_ref_p1_ctx][0],
            single_ref_p2: frame_context.single_ref_prob[single_ref_p2_ctx][1],
            // Unused in single-reference mode.
            comp_ref: 128,
            comp_inter: 128,
        },
    )
    .single
}

const DESCENT_CACHE_SLOTS: usize = 8 * 8 * rvp9_core::block::BLOCK_SIZES;
const DESCENT_CACHE_EMPTY: u16 = u16::MAX;

/// Per-superblock direct-index cache for partition-descent picker results.
///
/// The fixed slot table maps the complete 64-cell × 13-block-size key space
/// to an insertion-order value index, avoiding randomized hashing for this
/// bounded trusted domain.
struct DescentPickCache<T> {
    slots: [u16; DESCENT_CACHE_SLOTS],
    values: Vec<T>,
}

impl<T> DescentPickCache<T> {
    #[inline]
    fn new() -> Self {
        Self {
            slots: [DESCENT_CACHE_EMPTY; DESCENT_CACHE_SLOTS],
            values: Vec::with_capacity(128),
        }
    }

    #[inline(always)]
    fn slot(mi_row: usize, mi_col: usize, bsize: BlockSize) -> usize {
        (((mi_row & 7) * 8 + (mi_col & 7)) * rvp9_core::block::BLOCK_SIZES) + bsize.idx()
    }

    #[inline]
    fn insert(&mut self, mi_row: usize, mi_col: usize, bsize: BlockSize, value: T) {
        let slot = Self::slot(mi_row, mi_col, bsize);
        let value_index = self.slots[slot];
        if value_index != DESCENT_CACHE_EMPTY {
            self.values[value_index as usize] = value;
            return;
        }

        debug_assert!(self.values.len() < DESCENT_CACHE_SLOTS);
        let value_index = self.values.len() as u16;
        self.values.push(value);
        self.slots[slot] = value_index;
    }

    #[inline]
    fn get(&self, mi_row: usize, mi_col: usize, bsize: BlockSize) -> Option<&T> {
        let value_index = self.slots[Self::slot(mi_row, mi_col, bsize)];
        if value_index == DESCENT_CACHE_EMPTY {
            None
        } else {
            self.values.get(value_index as usize)
        }
    }
}

/// Select the transform size used by final realtime residual emission.
///
/// Large-frame recovery historically forced every leaf to its maximum
/// transform. Block32 is the measured exception: its realtime picker runs the
/// same tiled `block_yrd` path as libvpx and carries the transform decision the
/// residual walk must reuse. Other sizes retain the existing recovery policy.
fn realtime_emit_luma_tx_size(
    force_max_tx: bool,
    bsize: BlockSize,
    picked_tx_size: TxSize,
) -> TxSize {
    if force_max_tx && bsize != BlockSize::Block32x32 {
        rvp9_core::block::max_txsize(bsize)
    } else {
        picked_tx_size
    }
}

/// Decide whether large-frame recovery should force the legacy maximum-TX
/// emission policy.
///
/// The exact variance tree has enough rate-distortion fidelity to run the
/// focused TX redecision below. Other large-frame paths retain their measured
/// speed shortcut unless an explicit diagnostic override says otherwise.
fn realtime_force_max_emit_tx(
    override_value: Option<bool>,
    large_frame_recovery: bool,
    exact_vbp_tree: bool,
) -> bool {
    override_value.unwrap_or(large_frame_recovery && !exact_vbp_tree)
}

/// Reuse the realtime picker's transform by default. The focused emit-side
/// TX8/TX16 search remains available only as an explicit diagnostic.
fn realtime_emit_tx_rd_enabled(override_value: Option<bool>) -> bool {
    override_value.unwrap_or(false)
}

/// Immutable snapshot of diagnostic and A/B environment knobs shared by all
/// tile jobs for one frame.
pub(crate) struct InterEncodeEnv {
    rt_disable_golden: bool,
    rt_disable_newmv: bool,
    rt_disable_newmv_max_edge: Option<u8>,
    rt_disable_nearmv: bool,
    a4_descent_independent_mv: Option<bool>,
    trellis: Option<bool>,
    a4_rd_partition_disable: bool,
    f3_nonrd_partition_enable: bool,
    phase2_reuse_disable: bool,
    p8: Option<bool>,
    p8_qmax: Option<u8>,
    a4_live_partition_cost: bool,
    diag_golden: bool,
    diag_picker: bool,
    diag_intpro_sb: bool,
    e3_mvdiag: bool,
    s2_xtab: bool,
    rt_profile: bool,
    exact_vbp_tree: Option<bool>,
    live_ref_costs: Option<bool>,
    rt_emit_max_tx: Option<bool>,
    rt_emit_tx_rd: Option<bool>,
    rt_quant_fp: Option<bool>,
    rt_reuse_descent_pick: Option<bool>,
    a4_descent_cb_filter: Option<bool>,
    a4_descent_disable_golden: Option<bool>,
    a4_descent_disable_rect_newmv: Option<bool>,
    a4_breakout_16_dist_qmul: Option<u32>,
    rt_skip_intmv_filter_search: Option<bool>,
    newmv_diff_bias: Option<bool>,
    rt_min_partition32: Option<bool>,
    golden_real_block_yrd: Option<bool>,
    emit_profile: bool,
    a4_profile: bool,
    variance_hint_dump: bool,
    a4_rect_disable: bool,
    a4_split_stats: bool,
    diag_intpro: bool,
    p8_stats: bool,
}

impl InterEncodeEnv {
    pub(crate) fn capture() -> Self {
        let bool_override = |key: &str| std::env::var_os(key).map(|value| value != "0");
        Self {
            rt_disable_golden: std::env::var_os("RVP9_RT_DISABLE_GOLDEN").is_some(),
            rt_disable_newmv: std::env::var_os("RVP9_RT_DISABLE_NEWMV").is_some(),
            rt_disable_newmv_max_edge: std::env::var("RVP9_RT_DISABLE_NEWMV_MAX_EDGE")
                .ok()
                .and_then(|value| value.parse().ok()),
            rt_disable_nearmv: std::env::var_os("RVP9_RT_DISABLE_NEARMV").is_some(),
            a4_descent_independent_mv: bool_override("RVP9_A4_DESCENT_INDEPENDENT_MV"),
            trellis: bool_override("RVP9_TRELLIS"),
            a4_rd_partition_disable: std::env::var_os("RVP9_A4_RD_PARTITION_DISABLE").is_some(),
            f3_nonrd_partition_enable: std::env::var_os("RVP9_F3_NONRD_PARTITION_ENABLE").is_some(),
            phase2_reuse_disable: std::env::var_os("RVP9_PHASE2_REUSE_DISABLE").is_some(),
            p8: bool_override("RVP9_P8"),
            p8_qmax: std::env::var("RVP9_P8_QMAX")
                .ok()
                .and_then(|value| value.parse().ok()),
            a4_live_partition_cost: bool_override("RVP9_A4_LIVE_PARTITION_COST") == Some(true),
            diag_golden: std::env::var("RVP9_DIAG_GOLDEN_PICKS").is_ok(),
            diag_picker: std::env::var_os("RVP9_DIAG_PICKER").is_some(),
            diag_intpro_sb: std::env::var_os("RVP9_DIAG_INTPRO_SB").is_some(),
            e3_mvdiag: std::env::var("RVP9_E3_MVDIAG").is_ok(),
            s2_xtab: std::env::var("RVP9_S2_XTAB").is_ok(),
            rt_profile: std::env::var_os("RVP9_RT_PROFILE").is_some(),
            exact_vbp_tree: bool_override("RVP9_EXACT_VBP_TREE"),
            live_ref_costs: bool_override("RVP9_LIVE_REF_COSTS"),
            rt_emit_max_tx: bool_override("RVP9_RT_EMIT_MAX_TX"),
            rt_emit_tx_rd: bool_override("RVP9_RT_EMIT_TX_RD"),
            rt_quant_fp: bool_override("RVP9_RT_QUANT_FP"),
            rt_reuse_descent_pick: bool_override("RVP9_RT_REUSE_DESCENT_PICK"),
            a4_descent_cb_filter: bool_override("RVP9_A4_DESCENT_CB_FILTER"),
            a4_descent_disable_golden: bool_override("RVP9_A4_DESCENT_DISABLE_GOLDEN"),
            a4_descent_disable_rect_newmv: bool_override("RVP9_A4_DESCENT_DISABLE_RECT_NEWMV"),
            a4_breakout_16_dist_qmul: std::env::var("RVP9_A4_BREAKOUT_16_DIST_QMUL")
                .ok()
                .and_then(|value| value.parse().ok()),
            rt_skip_intmv_filter_search: bool_override("RVP9_RT_SKIP_INTMV_FILTER_SEARCH"),
            newmv_diff_bias: bool_override("RVP9_NEWMV_DIFF_BIAS"),
            rt_min_partition32: bool_override("RVP9_RT_MIN_PARTITION32"),
            golden_real_block_yrd: bool_override("RVP9_GOLDEN_REAL_BLOCK_YRD"),
            emit_profile: std::env::var_os("RVP9_EMIT_PROFILE").is_some(),
            a4_profile: std::env::var_os("RVP9_A4_PROFILE").is_some(),
            variance_hint_dump: bool_override("RVP9_VARIANCE_HINT_DUMP") == Some(true),
            a4_rect_disable: std::env::var_os("RVP9_A4_RECT_DISABLE").is_some(),
            a4_split_stats: std::env::var_os("RVP9_A4_SPLIT_STATS").is_some(),
            diag_intpro: std::env::var("RVP9_DIAG_INTPRO").is_ok(),
            p8_stats: std::env::var_os("RVP9_P8_STATS").is_some(),
        }
    }

    #[inline]
    fn flag_or(&self, value: Option<bool>, default: bool) -> bool {
        value.unwrap_or(default)
    }

    /// Mirror libvpx realtime `search_filter_ref`: integer-pel motion
    /// never needs interpolation-filter search. The environment override
    /// remains available for differential diagnostics.
    fn skip_integer_mv_filter_search(&self) -> bool {
        self.flag_or(self.rt_skip_intmv_filter_search, true)
    }

    /// Use libvpx's chessboard filter-search proxy throughout realtime
    /// partition descent. Rvp9 applies it below libvpx's resolution gate
    /// because the final emitted picker still performs the full search.
    fn descent_cb_filter_search(&self) -> bool {
        self.flag_or(self.a4_descent_cb_filter, true)
    }

    /// Keep the speculative partition proxy on LAST for ordinary content.
    /// The final emitted picker still evaluates GOLDEN, and the core picker's
    /// very-high-SAD exception keeps GOLDEN available to scene transitions.
    fn descent_disable_golden(&self) -> bool {
        self.flag_or(self.a4_descent_disable_golden, true)
    }

    /// Rank rectangular partition arms with inherited/zero motion while
    /// retaining NEWMV on every square NONE/SPLIT candidate and final leaf.
    fn descent_disable_rect_newmv(&self) -> bool {
        self.flag_or(self.a4_descent_disable_rect_newmv, true)
    }
}

#[allow(clippy::too_many_arguments)] // AQ.1 added `seg_q_deltas`; the
// remaining args are all load-bearing per-frame inputs that don't
// reduce to a meaningful struct yet — revisit when AQ.3 lands.
pub(crate) fn pick_and_simulate_inter_encode_partitions(
    env: &InterEncodeEnv,
    width: u32,
    height: u32,
    base_qindex: u8,
    avg_inter_qindex: u8,
    seg_q_deltas: &[Option<i16>; 8],
    input_yuv: (&[u8], &[u8], &[u8]),
    // Reference-buffer split — the LAST slot exposes TWO views to the
    // picker:
    //
    //   * `me_ref_yuv`: the ME-side reference (libvpx
    //     `vp9_denoiser.c:294-304` `running_avg_y`). Read by motion
    //     analysis paths only — `find_mv_refs`, motion search seeds,
    //     candidate SSE/RD, the diamond/subpel refinement, the chroma
    //     context the picker hands to `pick_inter_mode`. May diverge
    //     from `last_recon_yuv` once the denoiser / filtered-ALTREF
    //     wave fills it with filtered pixels; until then the caller
    //     keeps them byte-identical.
    //
    //   * `last_recon_yuv`: the residual-coding reference (libvpx
    //     `cm->frame_refs[]`). The bit-exact decoded pool the final MC
    //     prediction reads from before the residual IT-add. Must never
    //     diverge from the decoder — the cross-decoder gate verifies
    //     this. Read by the up-front `recon_yuv_out` seed copy and the
    //     per-leaf `build_inter_pred_plane_unscaled` plane selector
    //     (the residual-coding MC).
    //
    // Aliasing the same buffer for both would not be a real split; the
    // caller allocates separate `Vec`s in `StreamEncoder`. See the
    // `me_ref_y` field doc in `frame_loop.rs` and the
    // `reference_split_me_and_pred_identical_initially` PROOF.
    me_ref_yuv: (&[u8], &[u8], &[u8]),
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
    // W-ALTREF — ALTREF (ref-pool slot 2) reference planes for the
    // single-ref ALTREF picker candidate. `Some(..)` ONLY when
    // `enable_alt_ref` is set AND a real ARF has already been emitted
    // (so the decoder's slot 2 holds a meaningful reconstruction and the
    // show frame's header points `ref_frame_indices[2]` at slot 2 with
    // `sign_bias[ALTREF] = true`). `None` ⇒ the ALTREF candidate is
    // never enumerated and the picker behaviour is byte-identical to the
    // LAST/GOLDEN-only path (the gate-safety invariant: the sacred
    // cross-decoder gate runs with `enable_alt_ref = false`, so this is
    // always `None` there). libvpx parity: `setup_buffer_inter`
    // (`vp9_rdopt.c:2511`) wires the ALTREF buffer into the candidate
    // loop; `cpi->ref_frame_flags & VP9_ALT_FLAG` (`vp9_rdopt.c:3535`)
    // gates whether ALTREF is iterated.
    altref_recon_yuv: Option<(&[u8], &[u8], &[u8])>,
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
    // Frame-context-derived NMV costs shared by every tile job. The caller
    // builds this once per frame because `frame_context`, `base_qindex`, and
    // the high-precision-MV gate are identical across tiles.
    mv_costs: &rvp9_core::cost::mv_cost::MvCosts,
    // P20 — adaptive RD-threshold state (libvpx
    // `vp9_pickmode.c:2241-2247` gate state). Read by the picker for
    // each candidate's effective threshold; mutated after each pick
    // via `update_thresh_freq_fact` to decay the winner and
    // increment losers. Persists across frames.
    rd_thresh_tables: &mut [rvp9_core::mode_decision::rd_thresh::RdThreshTable],
    // §11 — frames since the last GOLDEN refresh
    // (libvpx `cpi->rc.frames_since_golden`). Feeds the `bias_golden`
    // gate's `frames_since_golden > 4` predicate
    // (`vp9_pickmode.c:2237-2239`).
    frames_since_golden: u32,
    // Show-frame GOLDEN refresh marker. CPU-used 7 realtime VBR uses the
    // source-faithful block_yrd path on these frames while retaining the
    // measured model-only approximation on ordinary inter frames.
    is_golden_refresh: bool,
    // Pinned-libvpx `gf_temporal_ref` topology gate. True only for the
    // single-layer, non-ALTREF show-frame path; exact buffer identity is
    // checked inside this function before GOLDEN is suppressed.
    allow_duplicate_golden_prune: bool,
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
    // Rate-control mode feeds mode-decision feature defaults where
    // fixed-Q and rate-aware RD curves diverge.
    rc_mode: crate::rate_control::RcMode,
    // PHASE-3 integration: thread W2.x pre-analysis producer outputs
    // into the per-leaf PickInterModeParams literals. Mirrors libvpx's
    // `cpi->noise_estimate` (vp9_encoder.h:971), `x->content_state_sb`
    // (vp9_encodeframe.c:1209-1257), and `cpi->rc.high_source_sad`
    // (vp9_ratectrl.c:3034-3216). Previously the picker saw default
    // values (NoiseEstimate::default, ContentStateSb::Normal,
    // scene_change_detected=false), which masked the W2.x producer
    // work behind a no-op consumer — the gap the re-audit caught.
    noise_estimate_frame: rvp9_core::mode_decision::inter::NoiseEstimate,
    scene_change_detected_frame: bool,
    // Per-SB `content_state_sb[sb_rows * sb_cols]` from W2.2.
    content_state_sb_grid: &[rvp9_core::mode_decision::inter::ContentStateSb],
    // Per-mi `skin_map[mi_rows * mi_cols]` from W2.5 (0 = non-skin).
    skin_map: &[u8],
    // Exact pinned-libvpx VBP copy eligibility and persistent show-frame
    // history. `None` isolates non-show ALTREF coding and disables the exact
    // adapter even if its diagnostic environment flag is set.
    frames_since_key: u32,
    mut exact_vbp_history: Option<
        &mut [rvp9_core::mode_decision::partition::FixedTreeHistoryEntry],
    >,
    // A23 — opt-in TPL rdmult adjustment. `Some((tpl_frame, r0))` only
    // when `config.enable_tpl` is set AND the single-GOP forward TPL pass
    // produced a valid frame grid for this frame; `None` otherwise (the
    // gate config, byte-identical to the pre-A23 path). When `Some`, the
    // per-64x64-SB `cb_rdmult` from `tpl::get_rdmult_delta`
    // (`vp9_encodeframe.c:3719-3768`) substitutes the base rdmult fed to
    // the inter RDO leaf (the consumer hook, `vp9_encodeframe.c:2235-2242`).
    tpl_rdmult: Option<(&crate::tpl::TplDepFrame, f64)>,
    // Stage 4a product-only strict-baseline override.
    stage4a_product_split_override: crate::frame_loop::Stage4aProductSplitOverride,
    // Typed split-admission stats sink for product reports.
    stage4a_split_stats_out: &mut rvp9_core::mode_decision::nonrd_partition::NonRdSplitStats,
    record_coef_tokens: bool,
    log2_tile_cols: u8,
    active_tile_col: Option<usize>,
) -> InterSimulationResult {
    use rvp9_core::bitstream::frame::{InterpFilter as WireInterpFilter, MAX_INTER_REFS};
    use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
    use rvp9_core::cost::symbol_costs::{
        inter_mode_costs, partition_costs, switchable_interp_costs,
    };
    use rvp9_core::inter::mv_ref::{
        BlockBounds, RealtimeRefMvs, find_last_golden_nearest_near_mvs_with_context,
        find_last_nearest_near_mvs_with_context, find_nearest_near_mvs,
    };
    use rvp9_core::inter::{get_pred_context_switchable_interp, get_pred_filter_ref};
    use rvp9_core::mode_decision::inter::{
        InterModeSearchParams, MvCandidates, PickInterModeParams, pick_inter_mode,
    };
    use rvp9_core::mode_decision::nonrd_partition::{
        NonRdDescentParams, NonRdSplitStats, PartitionRdCandidate,
        nonrd_pick_partition_with_pruners, variance_hint_bounds,
    };
    use rvp9_core::mode_decision::speed_features::{self as sf_mod};
    use rvp9_core::mv_search::int_pro_me::{
        int_pro_motion_estimation, int_pro_motion_estimation_seed,
    };
    use rvp9_core::mv_search::mv_limits::MvLimits;
    use rvp9_core::mv_search::{PlaneRef, init_dsmotion_compensation};
    use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers, plane_dequant_pair};

    // One tile worker owns this complete call. Reuse its emission buffers
    // directly instead of entering thread-local scratch for every transform.
    let mut inter_emit_scratch = InterEmitScratch::new();

    // W0.4: compute the per-frame `SpeedFeatures` once. The dispatcher
    // is pure — no side effects on `cpi` like libvpx's
    // `vp9_set_speed_features_framesize_{independent,dependent}` —
    // and the result is consumed by every per-leaf
    // `PickInterModeParams` literal below.
    let sf = sf_mod::dispatch(speed, deadline, width, height, content_tier);
    let rt_disable_golden_ref = sf.disable_golden_ref || env.rt_disable_golden;
    let golden_ref_is_last = golden_ref_duplicates_last(
        allow_duplicate_golden_prune,
        frames_since_golden,
        me_ref_yuv,
        last_recon_yuv,
        golden_recon_yuv,
    );
    let mut rt_inter_mode_mask = rvp9_core::mode_decision::inter::default_inter_mode_mask();
    if env.rt_disable_newmv {
        let newmv_bit = 1u8 << InterMode::NewMv.idx();
        for mask in &mut rt_inter_mode_mask {
            *mask &= !newmv_bit;
        }
    }
    if let Some(max_edge) = env.rt_disable_newmv_max_edge {
        let newmv_bit = 1u8 << InterMode::NewMv.idx();
        for bsize in [
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ] {
            if bsize.width() <= max_edge && bsize.height() <= max_edge {
                rt_inter_mode_mask[bsize.idx()] &= !newmv_bit;
            }
        }
    }
    if env.rt_disable_nearmv {
        let nearmv_bit = 1u8 << InterMode::NearMv.idx();
        for mask in &mut rt_inter_mode_mask {
            *mask &= !nearmv_bit;
        }
    }
    let mut a4_descent_inter_mode_mask = rt_inter_mode_mask;
    if env.descent_disable_rect_newmv() {
        let newmv_bit = 1u8 << InterMode::NewMv.idx();
        for bsize in [
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
        ] {
            a4_descent_inter_mode_mask[bsize.idx()] &= !newmv_bit;
        }
    }
    // A4 evaluates every candidate before the current SB has populated the
    // causal mi-grid. Using inherited MVs only where a candidate touches a
    // prior SB makes the partition proxy depend on boundary position rather
    // than comparable evidence. Give every speculative shape the same neutral
    // predictor; ZERO and NEWMV remain fully evaluated. The final emitted
    // picker still uses the real causal neighbours. `=0` preserves the old
    // diagnostic.
    let a4_descent_independent_mv = env
        .a4_descent_independent_mv
        .unwrap_or_else(|| a4_descent_independent_mv_default(rc_mode, speed, deadline));

    let w = width as usize;
    let h = height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let rt_large_recovery = matches!(deadline, sf_mod::Deadline::Realtime)
        && speed.unsigned_abs() >= 7
        && matches!(content_tier, sf_mod::ContentTier::Default)
        && u64::from(width) * u64::from(height) >= 700_000;
    let rt_v1_umv_border = matches!(deadline, sf_mod::Deadline::Realtime)
        && (6..=8).contains(&speed.unsigned_abs())
        && matches!(content_tier, sf_mod::ContentTier::Default)
        && u64::from(width) * u64::from(height) >= 700_000;
    let cyclic_refresh_active = seg_q_deltas.iter().any(Option::is_some);
    let mut inter_rd_calibration =
        inter_rd_calibration_for_mode(rc_mode, speed, deadline, cyclic_refresh_active);
    if golden_real_block_yrd_enabled(
        rc_mode,
        speed,
        deadline,
        cyclic_refresh_active,
        is_golden_refresh,
        env.golden_real_block_yrd,
    ) {
        inter_rd_calibration.model_only = false;
    }
    let full_pixel_search_method = full_pixel_search_method_for_mode(speed, deadline);

    let sb_cols = w / 64;
    let sb_rows = h.div_ceil(64);

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

    // Plane refs into the input + ME-side LAST reference buffer.
    //
    // Reference-buffer split: `ref_y` is the LAST-slot Y plane fed to
    // motion search, `find_mv_refs`, the per-leaf NEWMV diamond /
    // subpel refinement, the candidate SSE accumulation, and the
    // partition-side RD analysis. All of these are ME-side accesses
    // (libvpx `running_avg_y`); the residual-coding MC reads from
    // `last_recon_yuv` directly via the per-plane selector inside the
    // SB walk further down. Until the denoiser wave wires a real
    // filter, `me_ref_yuv` and `last_recon_yuv` are byte-identical.
    let src_y = PlaneRef {
        data: input_yuv.0,
        stride: w,
        width: w,
        height: h,
    };
    let ref_y = PlaneRef {
        data: me_ref_yuv.0,
        stride: w,
        width: w,
        height: h,
    };
    // A15 Phase B — GOLDEN Y plane ref for the widened picker.
    // Used in Step 3 to drive candidate SSE/RD when iterating
    // `{LAST, GOLDEN} × {ZERO, NEAREST, NEAR, NEW}`. The
    // encoder calls `pick_inter_mode_luma` once per ref-plane,
    // then takes the min-RD winner across both rankings.
    let ref_y_golden = PlaneRef {
        data: golden_recon_yuv.0,
        stride: w,
        width: w,
        height: h,
    };
    // W-ALTREF — ALTREF Y plane ref for the single-ref ALTREF candidate.
    // `altref_available` is the single gate: `true` only when the caller
    // passed `Some(..)` (= enable_alt_ref && ARF emitted). When `false`,
    // the picker never enumerates ALTREF and the bitstream is unchanged.
    let altref_available = altref_recon_yuv.is_some();
    let ref_y_altref = altref_recon_yuv.map(|(y, _, _)| PlaneRef {
        data: y,
        stride: w,
        width: w,
        height: h,
    });

    // Picker harness — built once per frame. The per-leaf
    // `PlaneQuantizers` is rebuilt inside the SB walk below (AQ.1)
    // because it depends on the leaf's `segment_id`.
    let inter_costs = inter_mode_costs(frame_context);
    let sic = switchable_interp_costs(frame_context);
    let allow_hp = crate::frame_allow_hp(base_qindex);
    let site_config = init_dsmotion_compensation(w as i32);

    // W4 RDO dispatch — speed≤4 routes the per-leaf mode decision through
    // the transform-domain RDO picker (`rdo::pick_inter_sb` driven by the
    // now-real `handle_inter`) instead of the realtime `pick_inter_mode`.
    // Built once per frame: the coef-bit cost table, skip-bit costs, and
    // a per-actual-size tx_size signalling cost (constant across a leaf's
    // mode candidates since `handle_inter` pins the tx to
    // `max_txsize(bsize)`; it cancels in the per-leaf argmin). libvpx
    // `vp9_rdopt.c:3438-4187` (driver) / `vp9_encodeframe.c:1975-2107`
    // (rd_pick_sb_modes dispatch).
    let rdo_speed_gated = speed.unsigned_abs() <= 4;
    let rdo_coef_costs = rvp9_core::cost::coef_cost::build_coef_cost_table(frame_context);
    // TRELLIS (2026-05-30) — post-quant `vp9_optimize_b` coefficient
    // optimizer in the inter + intra-in-inter emit paths. libvpx sets
    // `optimize_coefficients = 1` at good-mode speed<=4 (and 0 for RT
    // speed>=5), so the trellis is a GOOD-path feature. The prior port was
    // INERT (no-op) because it omitted libvpx's look-ahead `next_bits`
    // term + the real neighbour context; completing those makes it bite.
    // Default ON on the cpu<=4 GOOD path (foreman +0.08 dB matched,
    // strictly Pareto-better; synthetic positive; cross-decoder byte-exact
    // — adversarially confirmed). NOT run on the RT path (speed>=5),
    // matching libvpx and keeping the cpu7 goldens byte-identical.
    // `RVP9_TRELLIS=0` restores the byte-identical baseline.
    let trellis_enabled = speed.unsigned_abs() <= 4 && env.trellis.unwrap_or(true);
    let rdo_skip_costs = rvp9_core::cost::skip_costs(frame_context);
    let rdo_tx_size_costs = rvp9_core::cost::tx_size_costs(frame_context);
    // Skip-bit cost at the middle skip context (the per-leaf skip_ctx
    // shifts this equally for every candidate, so the argmin is
    // unaffected; the middle context is the libvpx-neutral choice).
    let rdo_skip_cost0 = rdo_skip_costs[1][0].max(0) as u32;
    let rdo_skip_cost1 = rdo_skip_costs[1][1].max(0) as u32;
    // Flat per-actual-size tx_size cost at (max=TX_32X32, ctx=0). Indexed
    // by `TxSize as usize`; only ≤ max entries are meaningful.
    let rdo_tx_size_cost: [u32; 4] = {
        let row = &rdo_tx_size_costs.data[2][0];
        [
            row[0].max(0) as u32,
            row[1].max(0) as u32,
            row[2].max(0) as u32,
            row[3].max(0) as u32,
        ]
    };

    // A4 (2026-05-27) — recursive RD-optimal partition descent
    // (libvpx `rd_pick_partition` / `nonrd_pick_partition`,
    // `vp9_encodeframe.c:4922-5210`). This is the real recursive
    // NONE-vs-SPLIT RD comparison: for each 64×64 SB it evaluates
    // PARTITION_NONE at the current size via the real
    // `pick_inter_mode` leaf picker, recurses into 4 quadrants for
    // PARTITION_SPLIT (64→32→16), and keeps the minimum-RD partition.
    // The chosen tree replaces `sb_partitions[sb_idx]` (the
    // variance-hint leaf list) before the per-leaf picker loop runs.
    //
    // **A4 promotes this from the F.3 infrastructure-only default-OFF
    // state to the encoder's default partition decision.** Pre-A4 the
    // partition SHAPE was always the variance heuristic
    // (`choose_partitioning`, `vp9_encodeframe.c:1261`); the RDO leaf
    // machinery only re-priced MODES inside fixed leaves. Now the
    // SHAPE itself is RD-chosen.
    //
    // Set `RVP9_A4_RD_PARTITION_DISABLE=1` to opt OUT (fall back to the
    // pure variance-hint partition — the pre-A4 behaviour). The legacy
    // `RVP9_F3_NONRD_PARTITION_ENABLE` env var is still honoured as a
    // force-ON for back-compat with the F.3 diagnostic tests, but the
    // descent now runs by default regardless.
    //
    // ## Scope (A4 first cut — honest scope-down, recorded)
    //
    // - **Square NONE / SPLIT only.** No PARTITION_HORZ / PARTITION_VERT
    //   (libvpx defers rect at `speed >= 5`, `vp9_encodeframe.c:4957`).
    //   The rect arms are a documented A4 follow-up; the descent's
    //   square 64→32→16 NONE/SPLIT comparison is the highest-value core.
    // - **Recursion caps at Block16x16.** Block8x8 leaves are inherited
    //   from the variance hint (the leaf picker bails to ZERO+skip at
    //   8×8 — see the `picker_supported` guard below).
    // - **Inter frames only.** The keyframe path keeps the variance /
    //   intra-shape partitioner; a KF RD descent is a follow-up.
    // - **min/max block-size clamps** come from `variance_hint_bounds`
    //   (libvpx `x->{min,max}_partition_size`), with the R3 floor at
    //   Block16x16.
    let f3_nonrd_partition_enabled = !env.a4_rd_partition_disable || env.f3_nonrd_partition_enable;
    // Phase-2 option-A — cache + reuse the descent's NEWMV search
    // result for the final per-leaf pass (seed-only / cross-decoder-safe
    // variant). Default ON; opt OUT with `RVP9_PHASE2_REUSE_DISABLE=1`
    // (the standing A/B knob for re-confirming the speed/quality split).
    // Only meaningful when the descent runs (`f3_nonrd_partition_enabled`),
    // since the cache is populated by the descent's picker calls.
    let phase2_reuse_enabled = !env.phase2_reuse_disable;
    // Phase A — enable 8×8 inter partitions in the A4 descent. Default
    // ON for fixed-Q / CBR-style paths where matched-Q Foreman improves,
    // and now also for one-pass VBR only at low-q/high-rate frames
    // (`base_qindex <= 200`). The bounded VBR gate is the measured-safe
    // slice: it gives a small Stage 4a BD-rate movement without reopening
    // the low-rate overspend seen in the full-P8 and DIRECT probes. Set
    // `RVP9_P8=1` or `RVP9_P8=0` as an explicit override.
    // `RVP9_P8_QMAX=<q>` remains a VBR diagnostic override for sweeping the
    // threshold. When ON, the descent lowers its `min_partition_size` floor
    // to Block8x8 so Block16x16 may SPLIT into four 8×8 NONE leaves and 8×8
    // becomes a valid NONE candidate.
    let p8_enabled = env.p8.unwrap_or_else(|| {
        env.p8_qmax
            .map(|qmax| p8_vbr_qmax_enabled(rt_large_recovery, rc_mode, base_qindex, qmax))
            .unwrap_or_else(|| p8_default_enabled(rt_large_recovery, rc_mode, base_qindex))
    });
    let block16_split_admission = stage4a_block16_split_admission(
        p8_enabled,
        rt_large_recovery,
        rc_mode,
        speed,
        deadline,
        stage4a_product_split_override,
    );
    // The full 16x16-to-8x8/rectangular search has a poor return on smooth
    // low-distortion parents. Scale the breakout threshold by qindex so
    // low-rate frames prune more aggressively while high-quality frames keep
    // the detail search. `=0` restores the pre-breakout differential path.
    let a4_breakout_16_dist_qmul = stage4a_descent_breakout_16_dist_qmul(
        p8_enabled,
        rt_large_recovery,
        rc_mode,
        speed,
        deadline,
        env.a4_breakout_16_dist_qmul,
    );
    let f3_partition_costs = if f3_nonrd_partition_enabled {
        // Diagnostic only: libvpx sources partition_cost from the live frame
        // context, but the Stage 4a VBR gate is BD-negative with that source
        // until the surrounding A4 calibration moves with it.
        if env.a4_live_partition_cost {
            Some(partition_costs(frame_context))
        } else {
            Some(partition_costs(&FrameContext::new_default()))
        }
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
    let f3_rd_mult = a4_partition_rd_mult_for_env(f3_rd_mult);

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
    // W-ALTREF — single-ref ALTREF signalling cost. libvpx
    // `vp9_rdopt.c:2464-2468` (`estimate_ref_frame_costs`): ALTREF takes
    // the `single_ref_p1 = 1` (GOLDEN/ALTREF subtree) then
    // `single_ref_p2 = 1` (ALTREF, not GOLDEN) branch.
    let ref_frame_cost_altref: u32 = {
        use rvp9_core::cost::vp9_cost_bit;
        use rvp9_core::entropy::DEFAULT_INTRA_INTER_PROB;
        use rvp9_core::entropy::DEFAULT_SINGLE_REF_PROB;
        let intra_inter_p = DEFAULT_INTRA_INTER_PROB[2];
        let single_ref_p1 = DEFAULT_SINGLE_REF_PROB[2][0];
        let single_ref_p2 = DEFAULT_SINGLE_REF_PROB[2][1];
        vp9_cost_bit(intra_inter_p, true)
            + vp9_cost_bit(single_ref_p1, true)
            + vp9_cost_bit(single_ref_p2, true)
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
        allow_high_precision_mv: allow_hp,
        interp_filter: WireInterpFilter::EightTap,
    };

    // A16 diagnostic: count GOLDEN ref-frame picks per inter frame when
    // RVP9_DIAG_GOLDEN_PICKS is set. Emitted as a single eprintln at
    // end of the SB loop below. Counts only non-skip leaves.
    let mut diag_golden_picks: usize = 0;
    let mut diag_last_picks: usize = 0;
    let diag_golden_enabled = env.diag_golden;
    // This diagnostic gate is consulted several times per leaf. Snapshot it
    // once per tile job instead of entering the process environment lock in
    // the mode-decision hot path.
    let diag_picker_enabled = env.diag_picker;
    // This seed diagnostic is emitted once per SB. Snapshot its process-env
    // gate here with the other diagnostics instead of locking the environment
    // for every superblock on normal encodes.
    let diag_intpro_sb_enabled = env.diag_intpro_sb;

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
    let e3_diag_enabled = env.e3_mvdiag;
    const E3_DIAG_SAMPLE_LIMIT: usize = 40;
    let mut e3_sample_printed: usize = 0;

    // S2/F.0.2 diagnostic flags, read once. The luma-EOB split is needed
    // by either diagnostic, but remains zero-cost in the hot emit TX walk
    // when both are disabled (the default).
    let s2_xtab_enabled = env.s2_xtab;
    let f02_oracle_enabled = crate::f02_oracle::is_enabled();
    rvp9_core::mode_decision::inter::candidate_trace::reset(f02_oracle_enabled);
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
    // Sign-bias: LAST/GOLDEN point at past display-order frames (no
    // bias). W-ALTREF — when the ALTREF candidate is live the ALTREF
    // points at a FUTURE display-order frame, so `sign_bias[ALTREF] =
    // true` (libvpx `set_ref_sign_bias`, `vp9_encoder.c:5034-5049`). This
    // MUST match the show-frame header's `ref_frame_sign_biases` and the
    // decoder's per-block `find_mv_refs` so the MV-ref candidate scan
    // applies the same sign flip on both sides
    // (`frame_loop::build_inter_packet_with_residuals`). When ALTREF is
    // NOT live (the gate's config) this stays `[false, false, false]` —
    // byte-identical to today. Note: the flip only affects candidate
    // scans that involve an ALTREF-referencing neighbour, which can only
    // exist once an ALTREF leaf is selected — impossible when ALTREF is
    // off — so the LAST/GOLDEN scans are unaffected regardless.
    let mi_rows_us = h / 8;
    let mi_cols_us = w / 8;
    let tile_bounds_by_col = column_tile_bounds(mi_rows_us, mi_cols_us, log2_tile_cols);
    assert_eq!(
        rd_thresh_tables.len(),
        tile_bounds_by_col.len() * sb_rows,
        "one adaptive RD-threshold table is required per tile SB row",
    );
    let mi_grid_layout = if let Some(tile_col) = active_tile_col {
        let bounds = tile_bounds_by_col
            .get(tile_col)
            .expect("active tile column must have frame bounds");
        InterPickerMiGridLayout::new(bounds.mi_col_start, bounds.mi_col_end - bounds.mi_col_start)
    } else {
        InterPickerMiGridLayout::new(0, mi_cols_us)
    };
    let mut coef_source = if let Some(tile_col) = active_tile_col {
        let bounds = tile_bounds_by_col
            .get(tile_col)
            .expect("active tile column must have frame bounds");
        VecBackedInterCoefSource::new_for_mi_columns(
            width,
            height,
            bounds.mi_col_start,
            bounds.mi_col_end,
        )
    } else {
        VecBackedInterCoefSource::new(width, height)
    };
    let mut recorded_coef_tile = record_coef_tokens.then(|| {
        let tile_col = active_tile_col.expect("coefficient recording requires one active tile");
        let bounds = tile_bounds_by_col[tile_col];
        RecordedCoefTile {
            tile_col,
            retained: RetainedCoefTokens::default(),
            counts: FrameCounts {
                coef_full: Some(rvp9_core::entropy::new_coef12_bank()),
                ..FrameCounts::default()
            },
            coef_ctx: TileCoefCtx::new(bounds.mi_col_end - bounds.mi_col_start),
            last_sb_row: None,
            mi_col_start: bounds.mi_col_start,
            mi_col_end: bounds.mi_col_end,
        }
    });
    let mut coef_record_sink = [];
    let mut coef_record_bw = BoolWriter::start_count_only(&mut coef_record_sink);
    debug_assert_eq!(out_mv_grid.len(), mi_rows_us * mi_cols_us);
    let mut row_grids = InterRowGrids::new(mi_rows_us, mi_grid_layout, out_mv_grid, mi_cols_us);
    let sign_bias = sign_bias_from_header(&[false, false, altref_available]);

    let rt_profile_enabled = env.rt_profile;
    let rt_profile_frame_start = std::time::Instant::now();
    let mut rt_profile_descent_time = std::time::Duration::ZERO;
    let mut rt_profile_final_pick_time = std::time::Duration::ZERO;
    let mut rt_profile_emit_time = std::time::Duration::ZERO;
    let mut rt_profile_final_leaves = 0usize;
    let mut rt_profile_picker_leaves = 0usize;
    let mut rt_profile_fallback_leaves = 0usize;
    // Exact pinned-libvpx cpu-used 7 variance-tree experiment. This stays
    // opt-in until the fixed-Q and product gates show a reproducible Pareto
    // improvement. The pure tree itself is differential-oracle exact; this
    // product adapter supplies its source, selected LAST/GOLDEN predictor,
    // thresholds, and causal leaf list.
    let exact_vbp_tree_enabled = env.flag_or(env.exact_vbp_tree, false)
        && exact_vbp_history.is_some()
        && speed == 7
        && deadline == rvp9_core::mode_decision::speed_features::Deadline::Realtime
        && content_tier == rvp9_core::mode_decision::speed_features::ContentTier::Default;
    if exact_vbp_tree_enabled {
        assert_eq!(
            exact_vbp_history.as_deref().map(|history| history.len()),
            Some((w / 64) * h.div_ceil(64)),
            "exact VBP history must have one entry per superblock"
        );
    }
    let live_ref_costs_enabled = exact_vbp_tree_enabled && env.flag_or(env.live_ref_costs, true);
    let rt_emit_max_tx = realtime_force_max_emit_tx(
        env.rt_emit_max_tx,
        rt_large_recovery,
        exact_vbp_tree_enabled,
    );
    let rt_emit_tx_rd = realtime_emit_tx_rd_enabled(env.rt_emit_tx_rd);
    // libvpx speed >= 4 uses `quant_fp` for every inter frame. Preserve
    // `=0` as the regular-quantizer diagnostic without making the slower
    // good-quality quantizer the realtime default.
    let rt_quant_fp = env.flag_or(env.rt_quant_fp, true);
    let rt_reuse_descent_pick = env.flag_or(env.rt_reuse_descent_pick, rt_large_recovery);
    let a4_descent_cb_filter_search = env.descent_cb_filter_search();
    let a4_descent_disable_golden = env.descent_disable_golden();
    let rt_skip_integer_mv_filter_search = env.skip_integer_mv_filter_search();
    let newmv_diff_bias_enabled =
        newmv_diff_bias_for_mode(rc_mode, env.flag_or(env.newmv_diff_bias, true));
    // Large-frame recovery: skip the A4 descent's 16x16 picker layer.
    // This removes the dominant 720p timing work while leaving the
    // small-frame quality gates untouched. `=0` restores the full descent.
    let rt_min_partition32 = env.flag_or(env.rt_min_partition32, rt_large_recovery);
    let emit_profile_enabled = env.emit_profile;
    let mut emit_profile_pred_time = std::time::Duration::ZERO;
    let mut emit_profile_tx_select_time = std::time::Duration::ZERO;
    let mut emit_profile_residual_time = std::time::Duration::ZERO;
    let mut emit_profile_quant_time = std::time::Duration::ZERO;
    let mut emit_profile_inverse_time = std::time::Duration::ZERO;
    let mut emit_profile_coef_insert_time = std::time::Duration::ZERO;
    let mut emit_profile_inter_leaves = 0usize;
    let mut emit_profile_tx_blocks = 0usize;
    let mut emit_profile_nonzero_tx_blocks = 0usize;
    let mut emit_profile_eob_sum = 0usize;
    let mut emit_profile_plane_tx_blocks = [0usize; 3];
    let a4_profile_enabled = env.a4_profile;
    let variance_hint_dump_enabled = env.variance_hint_dump;
    let a4_rect_disable = env.a4_rect_disable;
    let a4_split_stats_enabled = env.a4_split_stats;
    let mut a4_profile_calls = [0usize; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_pick_time = [std::time::Duration::ZERO; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_committed = [0usize; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_same_pick = [0usize; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_rate_delta = [0i64; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_dist_delta = [0i64; rvp9_core::block::BLOCK_SIZES];
    let mut a4_profile_coef_rate_delta = [0i128; rvp9_core::block::BLOCK_SIZES];
    let mut a4_split_stats = NonRdSplitStats::default();
    let rt_profile_phase2_reused_start = if rt_profile_enabled {
        crate::rdo::inter_dispatch::PHASE2_NEWMV_REUSED.load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    };
    let rt_profile_phase2_fresh_start = if rt_profile_enabled {
        crate::rdo::inter_dispatch::PHASE2_NEWMV_FRESH.load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    };

    let mut encode_job = |job: crate::row_mt::RowJob| {
        let current_tile = tile_bounds_by_col[job.tile_col];
        let sb_columns = inter_row_job_sb_columns(&tile_bounds_by_col, active_tile_col, job);
        let sb_row = job.sb_row;
        for sb_col in sb_columns {
            let tile_col = job.tile_col;
            let rd_thresh_table = &mut rd_thresh_tables[tile_col * sb_rows + sb_row];
            let sb_idx = sb_row * sb_cols + sb_col;

            // A23 — per-64x64-SB TPL `cb_rdmult` (libvpx
            // `get_rdmult_delta`, `vp9_encodeframe.c:3719-3768`). Computed
            // once per SB when the TPL pass produced a valid grid; reused
            // for every leaf in the SB (libvpx caches it on `x->cb_rdmult`
            // before descending the SB). `None` when TPL is off — the
            // base rdmult is then used unchanged (byte-identical path).
            let tpl_sb_cb_rdmult: Option<i32> = tpl_rdmult.and_then(|(tf, r0)| {
                // libvpx feeds `BLOCK_64X64` at the SB's mi top-left, the
                // base q-index rdmult (Inter frame), and gates the
                // adjustment to `gf_group_index < MAX_ARF_GOP_SIZE` and
                // `layer_depth <= 1` (a linear-P-chain leaf is layer 0,
                // index 1, both inside the gate).
                let orig = rvp9_core::cost::compute_rd_mult(
                    i32::from(base_qindex),
                    rvp9_core::cost::FrameTypeForRd::Inter,
                ) as i32;
                let delta = crate::tpl::get_rdmult_delta(crate::tpl::RdmultDeltaParams {
                    tpl_frame: tf,
                    mi_row: sb_row * 8,
                    mi_col: sb_col * 8,
                    bsize: rvp9_core::block::BlockSize::Block64x64,
                    orig_rdmult: orig,
                    r0,
                    frame_mi_rows: mi_rows_us,
                    frame_mi_cols: mi_cols_us,
                    gf_group_index: 1,
                    layer_depth: 0,
                });
                if delta.used_tpl && delta.cb_rdmult != orig {
                    Some(delta.cb_rdmult)
                } else {
                    None
                }
            });

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
            // The large-frame realtime path keeps the projection match and
            // its center SAD, but leaves neighbouring full-block refinement
            // to the later per-leaf NEWMV search.
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
            let int_pro = if sb_px_y + 64 > h {
                rvp9_core::mv_search::int_pro_me::IntProMeResult {
                    mv: Mv::ZERO,
                    best_sad: zero_mv_sad64_edge_clamped(
                        input_yuv.0,
                        last_recon_yuv.0,
                        (w, h),
                        (sb_px_x, sb_px_y),
                    ),
                }
            } else if rt_large_recovery {
                int_pro_motion_estimation_seed(
                    &src_y,
                    &ref_y,
                    sb_px_x,
                    sb_px_y,
                    BlockSize::Block64x64,
                )
            } else {
                int_pro_motion_estimation(&src_y, &ref_y, sb_px_x, sb_px_y, BlockSize::Block64x64)
            };
            // Treat a literal zero MV as "no useful seed" — the
            // picker's existing NEAREST/ZERO-at-(0,0) candidates
            // already cover that case.
            let sb_newmv_seed = if int_pro.mv == Mv::ZERO {
                None
            } else {
                Some(int_pro.mv)
            };
            // The partition tree and its chroma-sensitivity probe select the
            // same LAST-vs-GOLDEN predictor. Production dimensions are
            // multiples of 64, so compute GOLDEN's zero-MV SAD once with the
            // dispatched full-block kernel; the exact-tree path below reuses
            // this result instead of scanning the superblock a second time.
            let golden_distinct = golden_recon_yuv.0.as_ptr() != last_recon_yuv.0.as_ptr();
            let sb_golden_sad = if !rt_disable_golden_ref && golden_distinct && sb_px_y + 64 <= h {
                let src_offset = sb_px_y * w + sb_px_x;
                rvp9_core::dsp::get_ref().sad(BlockSize::Block64x64)(
                    &input_yuv.0[src_offset..],
                    w,
                    &golden_recon_yuv.0[src_offset..],
                    w,
                )
            } else if !rt_disable_golden_ref && golden_distinct {
                zero_mv_sad64_edge_clamped(
                    input_yuv.0,
                    golden_recon_yuv.0,
                    (w, h),
                    (sb_px_x, sb_px_y),
                )
            } else {
                u32::MAX
            };
            let last_bias_num = if rc_mode == crate::rate_control::RcMode::Cbr {
                7
            } else {
                8
            };
            let sb_partition_use_golden =
                u64::from(sb_golden_sad) < (u64::from(int_pro.best_sad) * last_bias_num) >> 3;
            let sb_partition_color_sensitivity = {
                let (partition_ref_uv, partition_mv, selected_sad) = if sb_partition_use_golden {
                    (
                        (golden_recon_yuv.1, golden_recon_yuv.2),
                        Mv::ZERO,
                        sb_golden_sad,
                    )
                } else {
                    (
                        (last_recon_yuv.1, last_recon_yuv.2),
                        int_pro.mv,
                        int_pro.best_sad,
                    )
                };
                partition_color_sensitivity(
                    (input_yuv.1, input_yuv.2),
                    partition_ref_uv,
                    (w, h),
                    (sb_px_x, sb_px_y),
                    partition_mv,
                    selected_sad,
                )
            };

            let mut used_exact_vbp_tree = false;
            let mut exact_variance_low = [false; 25];
            if exact_vbp_tree_enabled && sb_px_y + 64 <= h {
                use rvp9_core::bitstream::frame::InterpFilter;
                use rvp9_core::mode_decision::partition::{
                    Cpu7InterVbpThresholdInput, FixedTreeCopyConditions, FixedTreeInterInput,
                    FixedTreeLeaf, PartitionDecision, choose_fixed_tree_inter,
                    cpu7_inter_vbp_thresholds,
                };
                use rvp9_core::mv_search::set_subpel_mv_search_range;

                let segment_id = segment_id_for_hint_cell(&sb_partitions[sb_idx], 0, 0);
                let effective_qindex =
                    effective_qindex_for_segment(base_qindex, segment_id, seg_q_deltas);
                let content_state = content_state_sb_grid
                    .get(sb_idx)
                    .copied()
                    .unwrap_or(rvp9_core::mode_decision::inter::ContentStateSb::Normal);
                let thresholds = cpu7_inter_vbp_thresholds(Cpu7InterVbpThresholdInput {
                    q_index: effective_qindex,
                    avg_inter_qindex,
                    frame_width: w,
                    frame_height: h,
                    noise_estimate: noise_estimate_frame,
                    content_state,
                });
                let history = &mut exact_vbp_history
                    .as_deref_mut()
                    .expect("exact VBP history checked above")[sb_idx];
                let copy_conditions = FixedTreeCopyConditions {
                    frames_since_key,
                    segment_id,
                    max_copied_frames: 2,
                    resize_pending: false,
                    svc_copy_allowed: true,
                };
                let low_source_sad = matches!(
                    content_state,
                    rvp9_core::mode_decision::inter::ContentStateSb::LowSadLowSumdiff
                        | rvp9_core::mode_decision::inter::ContentStateSb::LowSadHighSumdiff
                );
                let low_source_copy = if low_source_sad && !scene_change_detected_frame {
                    history.try_copy(copy_conditions)
                } else {
                    None
                };

                let (exact_leaves, variance_low) = if let Some(copy) = low_source_copy {
                    (copy.leaves, copy.variance_low)
                } else {
                    // `set_offsets(..., BLOCK_64X64)` defines this full-pel
                    // UMV envelope. int-pro returns Q3, then libvpx derives a
                    // Q3 subpel envelope around the zero reference MV and clamps.
                    let fullpel_limits = MvLimits::new(
                        -(((sb_col * 8 + 8) * 8) as i32 + VP9_INTERP_EXTEND),
                        ((mi_cols_us - sb_col * 8) * 8) as i32 + VP9_INTERP_EXTEND,
                        -(((sb_row * 8 + 8) * 8) as i32 + VP9_INTERP_EXTEND),
                        ((mi_rows_us - sb_row * 8) * 8) as i32 + VP9_INTERP_EXTEND,
                    );
                    let mut subpel_limits = MvLimits::default();
                    set_subpel_mv_search_range(&mut subpel_limits, &fullpel_limits, 0, 0);
                    let partition_mv = Mv {
                        row: i32::from(int_pro.mv.row)
                            .clamp(subpel_limits.row_min, subpel_limits.row_max)
                            as i16,
                        col: i32::from(int_pro.mv.col)
                            .clamp(subpel_limits.col_min, subpel_limits.col_max)
                            as i16,
                    };

                    // cpu7 considers GOLDEN at zero MV and keeps LAST unless
                    // GOLDEN is strictly better than 7/8 of int-pro LAST SAD
                    // (the CBR short-circuit bias; VBR uses the unbiased SAD).
                    let (partition_ref, partition_mv, selected_sad) = if sb_partition_use_golden {
                        (golden_recon_yuv.0, Mv::ZERO, sb_golden_sad)
                    } else {
                        (last_recon_yuv.0, partition_mv, int_pro.best_sad)
                    };

                    let mut predictor_block = [0u8; 64 * 64];
                    build_inter_pred_block_unscaled(
                        partition_ref,
                        w,
                        w,
                        h,
                        sb_px_x,
                        sb_px_y,
                        64,
                        64,
                        i32::from(partition_mv.col) * 2,
                        i32::from(partition_mv.row) * 2,
                        &mut predictor_block,
                        64,
                        InterpFilter::Bilinear,
                    );

                    let (_, ac) =
                        plane_dequant_pair(i32::from(effective_qindex), QuantizerDeltas::Y);
                    let vbp_threshold_sad = if scene_change_detected_frame {
                        0
                    } else if w <= 352 && h <= 288 {
                        10
                    } else {
                        (i32::from(ac) << 1).max(1000) as u32
                    };
                    let vbp_threshold_copy = if scene_change_detected_frame {
                        0
                    } else if w <= 352 && h <= 288 {
                        4000
                    } else if w <= 640 && h <= 360 {
                        8000
                    } else {
                        (i32::from(ac) << 3).max(8000) as u32
                    };
                    let choose_fresh_tree = || {
                        let mut source_block = [0u8; 64 * 64];
                        for row in 0..64 {
                            source_block[row * 64..row * 64 + 64].copy_from_slice(
                                &input_yuv.0[(sb_px_y + row) * w + sb_px_x
                                    ..(sb_px_y + row) * w + sb_px_x + 64],
                            );
                        }
                        choose_fixed_tree_inter(&FixedTreeInterInput {
                            source: &source_block,
                            predictor: &predictor_block,
                            visible_width: 64,
                            visible_height: 64,
                            frame_width: w,
                            thresholds: thresholds.thresholds,
                            minmax_threshold: thresholds.minmax_threshold,
                            force_64x64_split: scene_change_detected_frame,
                            compute_minmax_variance: true,
                            segment_is_boosted: segment_id != 0
                                && seg_q_deltas[usize::from(segment_id)].is_some(),
                            partition_ref_is_last: !sb_partition_use_golden,
                            partition_mv_row: partition_mv.row,
                            partition_mv_col: partition_mv.col,
                            low_temp_var_mode: u8::from(
                                rc_mode == crate::rate_control::RcMode::Cbr,
                            ),
                            noise_level: noise_estimate_frame.level,
                            low_resolution: w <= 352 && h <= 288,
                            frame_height_le_360: h <= 360,
                        })
                    };

                    if segment_id == 0 && selected_sad < vbp_threshold_sad {
                        let decision = PartitionDecision::all_64x64();
                        let mut variance_low = [false; 25];
                        variance_low[0] = true;
                        history.update(decision, segment_id, variance_low);
                        (
                            vec![FixedTreeLeaf {
                                row_8x8: 0,
                                col_8x8: 0,
                                block_size: BlockSize::Block64x64,
                            }],
                            variance_low,
                        )
                    } else if int_pro.best_sad < vbp_threshold_copy {
                        if let Some(copy) = history.try_copy(copy_conditions) {
                            (copy.leaves, copy.variance_low)
                        } else {
                            let trace = choose_fresh_tree();
                            history.update(trace.decision.clone(), segment_id, trace.variance_low);
                            (trace.leaves, trace.variance_low)
                        }
                    } else {
                        let trace = choose_fresh_tree();
                        history.update(trace.decision.clone(), segment_id, trace.variance_low);
                        (trace.leaves, trace.variance_low)
                    }
                };
                exact_variance_low = variance_low;
                sb_partitions[sb_idx] = exact_leaves
                    .into_iter()
                    .map(|leaf| InterPartitionLeafDesc {
                        mi_row_in_sb: leaf.row_8x8,
                        mi_col_in_sb: leaf.col_8x8,
                        leaf_bsize: leaf.block_size,
                        picked_mode: InterMode::ZeroMv,
                        picked_mv: Mv::ZERO,
                        skip: true,
                        segment_id: segment_id_for_hint_cell(
                            &sb_partitions[sb_idx],
                            leaf.row_8x8,
                            leaf.col_8x8,
                        ),
                        picked_ref_frame: RefFrame::LastFrame,
                        picked_interp_filter: InterpFilter::EightTap,
                        picked_intra_mode: None,
                        picked_ref_frame2: RefFrame::NoRefFrame,
                        picked_mv2: Mv::ZERO,
                        picked_tx_size: rvp9_core::block::max_txsize(leaf.block_size),
                    })
                    .collect();
                used_exact_vbp_tree = true;
            }

            // Phase-2 option-A — per-SB cache of the descent's picker
            // results, keyed by (leaf_mi_row, leaf_mi_col, bsize). The
            // descent_picker callback inserts the full PickInterModeResult
            // for every (position, bsize) it evaluates; the final per-leaf
            // loop looks up the COMMITTED leaf's (position, bsize) and
            // reuses its NEWMV search result (R7: committed leaves are a
            // subset of evaluated (position, bsize) pairs, and the descent
            // never queries the same key twice, so no overwrite ambiguity).
            // Cleared per SB. Empty when the descent is off or reuse is
            // gated off — the final loop then runs the full search.
            // Value: (full picker result, descent LAST MvCandidates,
            // descent GOLDEN MvCandidates). Reuse of the cached NEWMV is
            // gated on the final pass's CAUSAL (nearest, near) matching
            // the descent's for that ref — the diamond seed derives from
            // {nearest, near} and the rate cost is priced against nearest,
            // so a full match means the fresh search would land on the
            // same MV (faithful reuse). Any mismatch -> re-search (R2).
            let descent_enabled =
                f3_nonrd_partition_enabled && !used_exact_vbp_tree && sb_px_y + 64 <= h;
            let mut descent_pick_cache = descent_enabled.then(DescentPickCache::new);

            // A3'-int-pro diagnostic — per-SB seed activity. Set
            // RVP9_DIAG_INTPRO_SB=1 to enable (off by default to
            // avoid log spam).
            if diag_intpro_sb_enabled {
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

            // A4 — recursive RD partition descent (default ON; opt out
            // with `RVP9_A4_RD_PARTITION_DISABLE=1`). Runs the
            // NONE-vs-SPLIT RD comparison BEFORE the per-leaf picker
            // loop and replaces this SB's variance-hint leaf list with
            // the RD-chosen tree. When opted out, this branch is
            // skipped and the picker loop runs over the variance-hint
            // leaves (pre-A4 behaviour).
            //
            // The mode-picker callback below is the REAL `pick_inter_mode`
            // (the same picker the post-descent per-leaf loop uses), so
            // the descent ranks partitions by genuine motion-aware RD
            // (NEWMV / NEARESTMV winners + coef/mode/mv rate), not a
            // synthetic proxy. The two paths exercise the same picker at
            // different sub-block resolutions.
            // The descent currently evaluates a complete 64x64 source
            // region. Keep the edge-safe variance tree for a partial
            // bottom SB; every retained leaf then fits the visible frame.
            if descent_enabled {
                let rt_profile_descent_start = if rt_profile_enabled {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
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
                if variance_hint_dump_enabled {
                    dump_variance_hint_cells(base_qindex, sb_row, sb_col, &hint);
                }

                // 2. Derive (min, max) bounds (libvpx
                //    `x->{min,max}_partition_size`) via the
                //    variance hint, with the R3 floor at
                //    Block16x16.
                let (min_bs, max_bs) = variance_hint_bounds(&hint);
                // Phase A (`RVP9_P8`) — lower the min floor to Block8x8 so
                // the descent can SPLIT Block16x16 into four 8×8 leaves and
                // treat 8×8 as a NONE candidate. Default policy is computed
                // once above; `RVP9_P8=0` keeps the R3 Block16x16 floor
                // exactly as before.
                let mut min_bs = if p8_enabled { BS::Block8x8 } else { min_bs };
                let mut max_bs = max_bs;
                if rt_large_recovery && rt_min_partition32 {
                    // The large-frame RT path has already disabled P8,
                    // so clamping the floor to 32x32 skips 16x16 descent
                    // calls without changing the small-frame Foreman gate.
                    min_bs = BS::Block32x32;
                    if (max_bs as u8) < (min_bs as u8) {
                        max_bs = min_bs;
                    }
                }

                // 3. Build the descent params.
                let pcosts = f3_partition_costs
                    .as_ref()
                    .expect("partition_costs built when F.3 enabled");
                let params = NonRdDescentParams {
                    min_partition_size: min_bs,
                    max_partition_size: max_bs,
                    partition_costs: pcosts,
                    rd_mult: f3_rd_mult,
                    allow_8x8_leaf: p8_enabled,
                    block16_split_admission,
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
                //    Borrow plumbing: `mi_grid` is read immutably for the
                //    descent (only the post-descent picker loop mutates it).
                //    The borrow is re-acquired inline so it expires before
                //    the per-leaf loop needs mutable access.
                let mi_grid_ro = row_grids.as_slice();
                let mut descent_picker = |sb_local_x_8x8: usize, sb_local_y_8x8: usize, bs: BS| {
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
                    // The picker handles all square sizes plus the
                    // six rect sizes the A4-rect descent evaluates
                    // (PARTITION_HORZ / PARTITION_VERT subsizes of
                    // 64/32/16: Block64x32 / Block32x16 / Block16x8 and
                    // their VERT mirrors). The descent caps at
                    // Block16x16 (HLD §4.3) so the smallest queried
                    // rect is 16x8 / 8x16. Sub-8 sizes are never
                    // queried — defensive guard returns None.
                    if matches!(
                        bs,
                        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4,
                    ) {
                        return None;
                    }

                    // Match libvpx's per-block UMV window. Edge-crossing
                    // candidates are safe because the picker now mirrors
                    // the reference frame's replicated border.
                    let mv_limits = if rt_v1_umv_border {
                        libvpx_inter_mv_limits(
                            leaf_mi_row,
                            leaf_mi_col,
                            usize::from(bs.n_8x8_wide()),
                            usize::from(bs.n_8x8_high()),
                            mi_rows_us,
                            mi_cols_us,
                        )
                    } else {
                        tight_inter_mv_limits((leaf_px_x, leaf_px_y), (bw_px, bh_px), (w, h))
                    };
                    if mv_limits.col_min > mv_limits.col_max
                        || mv_limits.row_min > mv_limits.row_max
                    {
                        return None;
                    }
                    let mv_in_bounds = |m: Mv| -> bool {
                        mv_in_strict_limits_or_visible_full_chroma_pel(
                            m,
                            mv_limits,
                            (leaf_px_x, leaf_px_y),
                            (bw_px, bh_px),
                            (w, h),
                        )
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
                        row_grids.layout,
                        leaf_mi_row,
                        leaf_mi_col,
                        current_tile,
                    );
                    let leaf_prev_mv: Option<PrevFrameMvRef> = prev_mv_grid
                        .and_then(|g| g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied());
                    let sb_cols = (width / 64).max(1) as usize;
                    let sb_idx = (leaf_mi_row / 8) * sb_cols + (leaf_mi_col / 8);
                    let content_state_sb = content_state_sb_grid
                        .get(sb_idx)
                        .copied()
                        .unwrap_or(rvp9_core::mode_decision::inter::ContentStateSb::Normal);
                    let (last_ref_mvs, golden_ref_mvs, inter_mode_ctx) =
                        if a4_descent_disable_golden
                            && content_state_sb
                                != rvp9_core::mode_decision::inter::ContentStateSb::VeryHighSad
                        {
                            let (last, mode_context) = find_last_nearest_near_mvs_with_context(
                                bs,
                                &neighbours,
                                &sign_bias,
                                &bounds,
                                leaf_prev_mv,
                                allow_hp,
                            );
                            (
                                last,
                                RealtimeRefMvs {
                                    nearest: Mv::ZERO,
                                    near: Mv::ZERO,
                                    const_motion: true,
                                },
                                mode_context,
                            )
                        } else {
                            let context = find_last_golden_nearest_near_mvs_with_context(
                                bs,
                                &neighbours,
                                &sign_bias,
                                &bounds,
                                leaf_prev_mv,
                                allow_hp,
                            );
                            (context.refs[0], context.refs[1], context.mode_context)
                        };
                    let mvc = MvCandidates {
                        nearest: last_ref_mvs.nearest,
                        near: last_ref_mvs.near,
                    };
                    let mvc_golden = MvCandidates {
                        nearest: golden_ref_mvs.nearest,
                        near: golden_ref_mvs.near,
                    };
                    let descent_mvc = if a4_descent_independent_mv {
                        MvCandidates {
                            nearest: Mv::ZERO,
                            near: Mv::ZERO,
                        }
                    } else {
                        mvc
                    };
                    let descent_mvc_golden = if a4_descent_independent_mv {
                        MvCandidates {
                            nearest: Mv::ZERO,
                            near: Mv::ZERO,
                        }
                    } else {
                        mvc_golden
                    };

                    // Per-leaf segment_id from the variance hint leaf
                    // covering this candidate's top-left cell.
                    let seg = segment_id_for_hint_cell(
                        &sb_partitions[sb_idx],
                        sb_local_y_8x8,
                        sb_local_x_8x8,
                    );
                    let leaf_qindex = effective_qindex_for_segment(base_qindex, seg, seg_q_deltas);
                    let quantizer =
                        init_plane_quantizers(i32::from(leaf_qindex), QuantizerDeltas::Y);
                    let (uv_dc_dq, uv_ac_dq) = plane_dequant_pair(
                        i32::from(leaf_qindex),
                        QuantizerDeltas {
                            dc_delta: 0,
                            ac_delta: 0,
                        },
                    );

                    // Neighbour mi-grid context predicates.
                    let nb_above = if leaf_mi_row > 0 {
                        row_grids
                            .layout
                            .get(mi_grid_ro, leaf_mi_row - 1, leaf_mi_col)
                    } else {
                        None
                    };
                    let nb_left = if leaf_mi_col > current_tile.mi_col_start {
                        row_grids
                            .layout
                            .get(mi_grid_ro, leaf_mi_row, leaf_mi_col - 1)
                    } else {
                        None
                    };
                    let (
                        ref_frame_cost_intra,
                        ref_frame_cost_last,
                        ref_frame_cost_golden,
                        _ref_frame_cost_altref,
                    ) = if live_ref_costs_enabled {
                        let costs = active_single_ref_frame_costs(frame_context, nb_above, nb_left);
                        (costs[0], costs[1], costs[2], costs[3])
                    } else {
                        (
                            ref_frame_cost_intra,
                            ref_frame_cost_last,
                            ref_frame_cost_golden,
                            ref_frame_cost_altref,
                        )
                    };
                    let switchable_interp_ctx =
                        get_pred_context_switchable_interp(nb_above, nb_left);
                    let filter_ref = get_pred_filter_ref(nb_above, nb_left);
                    let skip_ctx = rvp9_core::inter::get_pred_context_skip(nb_above, nb_left);
                    let last_const_motion = last_ref_mvs.const_motion;
                    let golden_const_motion = golden_ref_mvs.const_motion;
                    let neighbour_mv = |mi: Option<&rvp9_core::mode_info::ModeInfo>| -> Option<Mv> {
                        match mi.map(|m| m.mode) {
                            Some(rvp9_core::mode_info::BlockMode::Inter { mvs, .. }) => {
                                Some(mvs[0])
                            }
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
                        last_mv_candidates: &descent_mvc,
                        golden_ref_plane: &ref_y_golden,
                        golden_mv_candidates: &descent_mvc_golden,
                        mv_limits,
                        site_config: &site_config,
                        full_pixel_search_method,
                        mv_costs,
                        inter_mode_costs: &inter_costs,
                        inter_mode_ctx,
                        quantizer: &quantizer,
                        q_index: leaf_qindex,
                        allow_hp,
                        frame_kind,
                        newmv_seed: sb_newmv_seed,
                        ref_frame_cost_last,
                        ref_frame_cost_golden,
                        last_nearest_in_bounds: mv_in_bounds(descent_mvc.nearest),
                        last_near_in_bounds: mv_in_bounds(descent_mvc.near),
                        golden_nearest_in_bounds: mv_in_bounds(descent_mvc_golden.nearest),
                        golden_near_in_bounds: mv_in_bounds(descent_mvc_golden.near),
                        enable_intra_fallback: false,
                        intra_above_pixels: None,
                        intra_left_pixels: None,
                        intra_top_left_pixel: 127,
                        intra_y_mode_costs: &intra_y_mode_costs_inter,
                        ref_frame_cost_intra,
                        switchable_interp_costs: &sic,
                        switchable_interp_ctx,
                        filter_ref,
                        search_golden_filter: rc_mode == crate::rate_control::RcMode::Vbr,
                        skip_ctx,
                        skip_prob: frame_context.skip_probs[skip_ctx],
                        rd_calibration: inter_rd_calibration,
                        last_const_motion,
                        golden_const_motion,
                        // The recursive descent is a speculative luma proxy:
                        // score partition shapes without rebuilding/modeling
                        // both chroma planes for every rejected candidate.
                        // The selected leaves are picked again below with the
                        // real U/V inputs, so emitted mode and residual choices
                        // still include full chroma RD.
                        src_u: &[],
                        src_u_stride: uvw,
                        src_v: &[],
                        src_v_stride: uvw,
                        // Reference-buffer split — picker chroma
                        // SSE/RD reads the ME-side LAST view.
                        last_ref_u: me_ref_yuv.1,
                        last_ref_v: me_ref_yuv.2,
                        golden_ref_u: golden_recon_yuv.1,
                        golden_ref_v: golden_recon_yuv.2,
                        uv_stride: uvw,
                        color_sensitivity: [false; 2],
                        uv_dc_quant: uv_dc_dq as u32,
                        uv_ac_quant: uv_ac_dq as u32,
                        encode_breakout: 20,
                        dequant_dc: quantizer.dequant[0] as u16,
                        dequant_ac: quantizer.dequant[1] as u16,
                        above_mv: above_mv_p13,
                        left_mv: left_mv_p13,
                        // PHASE-3 integration: frame-level noise tier
                        // from W2.4 noise estimator.
                        noise_estimate: noise_estimate_frame,
                        lowvar_highsumdiff: false,
                        // PHASE-3 integration: per-mi skin bit from
                        // W2.5 skin detector.
                        is_skin: {
                            let mi_cols = (width / 8) as usize;
                            let mi_idx = leaf_mi_row * mi_cols + leaf_mi_col;
                            skin_map.get(mi_idx).copied().unwrap_or(0) != 0
                        },
                        newmv_diff_bias_enabled,
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
                        force_skip_low_temp_var: exact_vbp_tree_enabled
                            && rvp9_core::mode_decision::partition::fixed_tree_force_skip_low_temp_var(
                                &exact_variance_low,
                                leaf_mi_row,
                                leaf_mi_col,
                                bs,
                            ),
                        short_circuit_low_temp_var: u8::from(
                            exact_vbp_tree_enabled
                                && rc_mode == crate::rate_control::RcMode::Cbr,
                        ),
                        // PHASE-3 integration: per-SB content_state
                        // from W2.2 avg_source_sad pass. Index via
                        // (leaf_mi_row/8, leaf_mi_col/8) — the SB
                        // grid resolution is 64×64 luma.
                        content_state_sb,
                        is_screen_content: false,
                        zero_temp_sad_source: None,
                        scene_change_detected: scene_change_detected_frame,
                        inter_mode_mask: a4_descent_inter_mode_mask,
                        // W0.4: SF fields now flow from the
                        // dispatcher computed at the top of this
                        // function; the literal previously
                        // hard-coded the cpu-used-6/7 RT values.
                        limit_newmv_early_exit: sf.limit_newmv_early_exit,
                        bias_golden: sf.bias_golden,
                        frames_since_golden,
                        golden_ref_is_last,
                        // Apply the chessboard shortcut only to the A4
                        // partition proxy. The final emitted picker below
                        // keeps the resolution-gated libvpx speed feature and
                        // therefore performs the full small-frame search.
                        cb_pred_filter_search: sf.cb_pred_filter_search
                            || a4_descent_cb_filter_search,
                        skip_integer_mv_filter_search: rt_skip_integer_mv_filter_search,
                        mi_row: leaf_mi_row as u32,
                        mi_col: leaf_mi_col as u32,
                        chessboard_index,
                        // The speculative partition proxy uses LAST only for
                        // ordinary content. The final picker below retains the
                        // configured GOLDEN search, and VeryHighSad bypasses
                        // this core speed feature for scene transitions.
                        disable_golden_ref: rt_disable_golden_ref || a4_descent_disable_golden,
                        max_intra_bsize: sf.max_intra_bsize,
                        rt_intra_dc_only_low_content: sf.rt_intra_dc_only_low_content,
                        // Phase-2 option-A: the descent computes the
                        // NEWMV search fresh (it has no prior cache).
                        newmv_override_last: None,
                        newmv_override_golden: None,
                    };
                    let a4_profile_pick_start = if a4_profile_enabled {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let result = pick_inter_mode(&pick_params);
                    if let Some(t0) = a4_profile_pick_start {
                        a4_profile_calls[bs.idx()] += 1;
                        a4_profile_pick_time[bs.idx()] += t0.elapsed();
                    }
                    // Phase-2 option-A — stash the FULL picker result
                    // for this (position, bsize) so the final per-leaf
                    // loop can reuse the NEWMV search if it commits this
                    // exact leaf. Keyed by absolute mi position + bsize;
                    // the descent never queries the same key twice.
                    if phase2_reuse_enabled {
                        descent_pick_cache
                            .as_mut()
                            .expect("enabled partition descent owns a picker cache")
                            .insert(
                                leaf_mi_row,
                                leaf_mi_col,
                                bs,
                                (result, descent_mvc, descent_mvc_golden),
                            );
                    }
                    // Translate to the descent's PartitionRdCandidate.
                    // `best_rate` is already in `vp9_prob_cost` units
                    // (1/512-bit) — same family as the descent's
                    // `partition_cost` adds. The compatibility path carries
                    // pixel-SSE-of-recon; the Hadamard-only experiment
                    // carries the picker's coefficient-domain distortion.
                    // `ref_frame` discriminant matches the encoder's
                    // `RefFrame as u8` convention (see proxy site for
                    // the existing encoding).
                    Some(PartitionRdCandidate {
                        rate: result.best_rate,
                        dist: result.partition_distortion,
                        mv: result.best_mv,
                        mode: result.best_mode as u8,
                        ref_frame: result.picked_ref_frame as u8,
                    })
                };

                // A21 — ML partition-search breakout pruner
                // (`ml_partition::ml_predict_breakout`, libvpx
                // `vp9_encodeframe.c:4161-4168`). Active only when the
                // speed feature `sf.rd_ml_partition.search_breakout` is
                // on AND `cm->base_qindex >= 100` (libvpx :4163). When
                // active, after the descent's NONE pick wins at a block
                // the real NN breakout model votes — a positive linear
                // score `>= search_breakout_thresh[q_ctx]` forces
                // `do_split = 0`, skipping that block's SPLIT recursion.
                //
                // Features are REAL: source-block Y variance vs. zeros
                // (`vf(src, zeros) >> num_pels_log2`, libvpx :3301-3304),
                // the NONE pick's rate/dist, `x->rdmult` (= f3_rd_mult),
                // and `vp9_ac_quant(base_qindex)`. No synthetic shortcut.
                let ml_breakout_active =
                    sf.rd_ml_partition.search_breakout && i32::from(base_qindex) >= 100;
                // A21-rect — ML rectangular-partition pruner
                // (`ml_partition::ml_prune_rect_partition`, libvpx
                // `vp9_encodeframe.c:3334-3425`; gate at :4332-4339).
                // Active when ANY of the 16/32/64 `prune_rect_thresh`
                // entries is non-negative (libvpx leaves the model off
                // per-size via `thresh < 0`). The gate / RT presets set
                // all thresholds to -1 so the pruner is INERT there;
                // Good speed 0/1 turn it on. libvpx also gates on
                // `!frame_is_intra_only` — the inter descent only runs
                // on non-KF frames, so that holds here.
                let ml_rect_active = sf.rd_ml_partition.prune_rect_thresh[1] >= 0
                    || sf.rd_ml_partition.prune_rect_thresh[2] >= 0
                    || sf.rd_ml_partition.prune_rect_thresh[3] >= 0;

                let descent_result = {
                    use rvp9_core::block::BlockSize as BS;
                    use rvp9_core::mode_decision::nonrd_partition::RectPruneRdContext;
                    let src_y_buf: &[u8] = input_yuv.0;
                    let q_idx = i32::from(base_qindex);
                    let ac_q = i32::from(rvp9_core::quant::ac_quant(q_idx, 0, BitDepth::Eight));
                    let dc_q = i32::from(rvp9_core::quant::dc_quant(q_idx, 0, BitDepth::Eight));
                    let breakout_thresh = sf.rd_ml_partition.search_breakout_thresh;
                    let prune_rect_thresh = sf.rd_ml_partition.prune_rect_thresh;
                    // 64-byte zero reference (libvpx `vp9_64_zeros`,
                    // :3253) — variance core reads it row-by-row with a
                    // zero stride so the same 64 bytes cover any block.
                    let zeros = [0u8; 64];
                    let mut breakout = |sb_local_x_8x8: usize,
                                        sb_local_y_8x8: usize,
                                        bs: BS,
                                        none_rate: u32,
                                        none_dist: u32|
                     -> bool {
                        if let Some(dist_qmul) = a4_breakout_16_dist_qmul
                            && matches!(bs, BS::Block16x16)
                            && none_dist <= u32::from(base_qindex).saturating_mul(dist_qmul)
                        {
                            return true;
                        }
                        if !ml_breakout_active {
                            return false;
                        }
                        // libvpx only consults breakout at 64/32/16/8.
                        if !matches!(
                            bs,
                            BS::Block8x8 | BS::Block16x16 | BS::Block32x32 | BS::Block64x64,
                        ) {
                            return false;
                        }
                        let px_x = (sb_col * 8 + sb_local_x_8x8) * 8;
                        let px_y = (sb_row * 8 + sb_local_y_8x8) * 8;
                        let bw_px = usize::from(bs.width());
                        let bh_px = usize::from(bs.height());
                        if px_x + bw_px > w || px_y + bh_px > h {
                            return false;
                        }
                        // Source-block variance vs. zeros, shifted by
                        // num_pels_log2 (libvpx :3301-3304).
                        let src_off = px_y * w + px_x;
                        let (var, _sse) = rvp9_core::dsp::distortion::variance(
                            &src_y_buf[src_off..],
                            w,
                            &zeros,
                            0,
                            bw_px,
                            bh_px,
                        );
                        let num_pels_log2 = bs.width_log2() as u32 + bs.height_log2() as u32;
                        let var_shifted = var >> num_pels_log2;
                        // q_ctx (libvpx :3259).
                        let q_ctx = if q_idx >= 200 {
                            0usize
                        } else if q_idx >= 150 {
                            1
                        } else {
                            2
                        };
                        let bin = crate::ml_partition::BreakoutInput {
                            bsize: bs,
                            base_qindex: q_idx,
                            width: w as i32,
                            height: h as i32,
                            ac_q,
                            var_shifted,
                            rd_rate: none_rate.min(i32::MAX as u32) as i32,
                            // libvpx :3309 — clamp dist to INT_MAX, THEN
                            // shift right by num_pels_log2.
                            rd_dist_shifted: i64::from(none_dist.min(i32::MAX as u32))
                                >> num_pels_log2,
                            rdmult: f3_rd_mult.min(i32::MAX as u32) as i32,
                            breakout_thresh: breakout_thresh[q_ctx],
                        };
                        let fired = crate::ml_partition::ml_predict_breakout(&bin);
                        if fired {
                            crate::rdo::inter_dispatch::ML_PARTITION_BREAKOUT_FIRED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        fired
                    };

                    // A21-rect — real `ml_prune_rect_partition` adapter.
                    // Features are REAL: `vp9_dc_quant(base_qindex)`,
                    // `pc_tree->partitioning == NONE`, `ref_rd`
                    // (best_rdc.rdcost), the NONE arm's RD, and the 4
                    // SPLIT children's NONE-arm RDs (libvpx :3380-3401).
                    // The model's `max_score - thresh` margin decides
                    // which rect arms stay allowed; a pruned arm skips
                    // its picker invocations. The ANTI-G3 counter bumps
                    // once per arm the model removes (so a stub that
                    // never pruned leaves it at 0).
                    let mut rect_prune = |sb_local_x_8x8: usize,
                                          sb_local_y_8x8: usize,
                                          bs: BS,
                                          rd: RectPruneRdContext,
                                          allow_horz_in: bool,
                                          allow_vert_in: bool|
                     -> (bool, bool) {
                        if a4_rect_disable {
                            return (false, false);
                        }
                        if !ml_rect_active {
                            return (allow_horz_in, allow_vert_in);
                        }
                        // Model only covers 16/32/64 (libvpx 8×8 = break).
                        let thresh = match bs {
                            BS::Block16x16 => prune_rect_thresh[1],
                            BS::Block32x32 => prune_rect_thresh[2],
                            BS::Block64x64 => prune_rect_thresh[3],
                            _ => return (allow_horz_in, allow_vert_in),
                        };
                        if thresh < 0 {
                            return (allow_horz_in, allow_vert_in);
                        }
                        // `ref_rd` is libvpx `best_rdc.rdcost`; if
                        // nothing was feasible the pruner is a no-op.
                        let ref_rd = match rd.best_rd_cost {
                            Some(r) => r,
                            None => return (allow_horz_in, allow_vert_in),
                        };
                        let _ = (sb_local_x_8x8, sb_local_y_8x8);
                        let rin = crate::ml_partition::RectPruneInput {
                            bsize: bs,
                            dc_q,
                            thresh,
                            ref_rd,
                            partitioning_is_none: rd.partitioning_is_none,
                            // libvpx reads `pc_tree->none.rdcost`; -1 (an
                            // out-of-range sentinel the model treats as
                            // "invalid" → rd_ratio = 2.0) when NONE was
                            // infeasible, matching libvpx's uninitialised
                            // `INT64_MAX` guard at :3393.
                            none_rdcost: rd.none_rd_cost.unwrap_or(-1),
                            split_none_rdcost: [
                                rd.split_none_rd_cost[0].unwrap_or(-1),
                                rd.split_none_rd_cost[1].unwrap_or(-1),
                                rd.split_none_rd_cost[2].unwrap_or(-1),
                                rd.split_none_rd_cost[3].unwrap_or(-1),
                            ],
                        };
                        let dec = crate::ml_partition::ml_prune_rect_partition(
                            &rin,
                            allow_horz_in,
                            allow_vert_in,
                        );
                        // ANTI-G3: count each arm the model removed.
                        if allow_horz_in && !dec.allow_horz {
                            crate::rdo::inter_dispatch::ML_PARTITION_RECT_PRUNED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        if allow_vert_in && !dec.allow_vert {
                            crate::rdo::inter_dispatch::ML_PARTITION_RECT_PRUNED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        (dec.allow_horz, dec.allow_vert)
                    };

                    nonrd_pick_partition_with_pruners(
                        &hint,
                        &params,
                        &mut descent_picker,
                        &mut breakout,
                        &mut rect_prune,
                    )
                };

                a4_split_stats.accumulate(&descent_result.split_stats);

                // A4 anti-G3 diagnostic: count this SB if the RD
                // descent chose a partition SHAPE that differs from the
                // variance hint. A stubbed / fixed descent that echoed
                // the hint would never bump this; the PROOF test
                // (`tests/w_rd_partition.rs`) asserts it advances on
                // content where RD and variance disagree.
                if descent_result.cells != hint.cells {
                    crate::rdo::inter_dispatch::RD_PARTITION_RESHAPED_SBS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                // 5. Convert the descent's leaf list back into
                //    `Vec<InterPartitionLeafDesc>`. Preserve the
                //    `segment_id` from the variance hint at each
                //    leaf's top-left mi cell (the descent doesn't
                //    own segment-id information; AQ.1 owns it).
                let mut new_leaves: Vec<InterPartitionLeafDesc> =
                    Vec::with_capacity(descent_result.leaves.len());
                for leaf in &descent_result.leaves {
                    // A4-rect anti-G3 diagnostic: count each rectangular
                    // leaf (`width != height`) the descent committed. A
                    // HORZ/VERT decision contributes 2 (the two halves).
                    // The PROOF test (`tests/w_rd_partition_rect.rs`)
                    // asserts this advances on a horizontal/vertical-edge
                    // clip — a descent that never picked rect would leave
                    // it at zero and fail.
                    if leaf.bsize.width() != leaf.bsize.height() {
                        crate::rdo::inter_dispatch::RD_PARTITION_RECT_LEAVES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // Phase A (`RVP9_P8`) diagnostic — count each 8×8 leaf
                    // the descent committed (descent-chosen, distinct from
                    // variance-hint-inherited 8×8). Dumped via RVP9_P8_STATS.
                    if matches!(leaf.bsize, BlockSize::Block8x8) {
                        crate::rdo::inter_dispatch::RD_PARTITION_8X8_LEAVES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // Preserve the segment_id from the variance hint
                    // leaf covering this descent leaf's top-left cell.
                    // RD descent can split a hinted leaf; exact-origin
                    // lookup would drop those child leaves back to
                    // segment 0 and silently disable AQ for them.
                    let seg = segment_id_for_hint_cell(
                        &sb_partitions[sb_idx],
                        leaf.mi_row_in_sb,
                        leaf.mi_col_in_sb,
                    );
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
                        picked_interp_filter: rvp9_core::bitstream::frame::InterpFilter::EightTap,
                        picked_intra_mode: None,
                        picked_ref_frame2: RefFrame::NoRefFrame,
                        picked_mv2: Mv::ZERO,
                        // Track-F iter7 — default to max; the simulator's
                        // TX-size RD loop overwrites it.
                        picked_tx_size: rvp9_core::block::max_txsize(leaf.bsize),
                    });
                }
                sb_partitions[sb_idx] = new_leaves;
                if let Some(t0) = rt_profile_descent_start {
                    rt_profile_descent_time += t0.elapsed();
                }
            }

            for leaf_idx in 0..sb_partitions[sb_idx].len() {
                rt_profile_final_leaves += 1;
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

                // The picker + recon path support every square size plus
                // the six rect sizes the A4-rect descent can emit
                // (PARTITION_HORZ / PARTITION_VERT subsizes:
                // Block64x32 / Block32x16 / Block16x8 / Block32x64 /
                // Block16x32 / Block8x16). The recon TX walk, UV-plane
                // helpers (`f5_uv_plane_block_size` / `f5_uv_plane_tx_size`)
                // and the bitstream writer (`infer_inter_partition`,
                // which maps a rect leaf size to PARTITION_HORZ/VERT) are
                // all rect-aware. Only the three sub-8 sizes (which rvp9
                // never emits) fall back to ZEROMV / skip = true.
                let picker_supported = !matches!(
                    leaf_bsize,
                    BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
                );

                if !picker_supported {
                    rt_profile_fallback_leaves += 1;
                    sb_partitions[sb_idx][leaf_idx].picked_mode = InterMode::ZeroMv;
                    sb_partitions[sb_idx][leaf_idx].picked_mv = Mv::ZERO;
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame = RefFrame::LastFrame;
                    copy_last_ref_leaf_to_recon(
                        w,
                        h,
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        last_recon_yuv,
                        (
                            &mut *recon_yuv_out.0,
                            &mut *recon_yuv_out.1,
                            &mut *recon_yuv_out.2,
                        ),
                    );
                    // F.0.2 oracle — record the skip-fallback decision
                    // too (otherwise sub-8 leaves are missing from the
                    // sink and the libvpx-side patch would emit an
                    // unmatched record).
                    if f02_oracle_enabled {
                        crate::f02_oracle::record_decision(crate::f02_oracle::BlockDecision {
                            frame_idx: crate::f02_oracle::current_frame_idx(),
                            mi_row: leaf_mi_row as u16,
                            mi_col: leaf_mi_col as u16,
                            bsize: leaf_bsize,
                            mode: crate::f02_oracle::BlockDecisionMode::Inter(InterMode::ZeroMv),
                            ref_frame: RefFrame::LastFrame,
                            mv_row: 0,
                            mv_col: 0,
                            mode_bits_est: 0,
                            mv_bits_est: 0,
                            ref_bits_est: 0,
                            filter_bits_est: 0,
                            skip_bits_est: 0,
                            coef_bits_est: 0,
                            picker_rate: 0,
                            picker_dist: 0,
                            picker_rd_cost: 0,
                            model_var_y: 0,
                            model_tx_size: rvp9_core::transforms::TxSize::Tx4x4,
                            model_skip_txfm: 2,
                            luma_coef_rate: 0,
                            luma_coef_dist: 0,
                            uv_model_rate: 0,
                            uv_model_dist: 0,
                            uv_u_sensitive: false,
                            uv_v_sensitive: false,
                            partition_u_sensitive: false,
                            partition_v_sensitive: false,
                            last_candidates: [Default::default(); 4],
                            sse_after_pred: 0,
                            interp_filter: Some(
                                rvp9_core::bitstream::frame::InterpFilter::EightTap,
                            ),
                            skip_flag: true,
                            q_index: effective_qindex_for_segment(
                                base_qindex,
                                sb_partitions[sb_idx][leaf_idx].segment_id,
                                seg_q_deltas,
                            ),
                            tx_size: rvp9_core::transforms::TxSize::Tx4x4,
                            luma_eob_nonzero: false,
                        });
                    }
                    row_grids.splat_mode(
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        InterMode::ZeroMv,
                        Mv::ZERO,
                        sb_partitions[sb_idx][leaf_idx].segment_id,
                        /* skip = */ true,
                        /* picked_ref_frame = */ RefFrame::LastFrame,
                        /* picked_intra_mode = */ None,
                        /* picked_interp_filter = */
                        rvp9_core::bitstream::frame::InterpFilter::EightTap,
                    );
                    if diag_picker_enabled {
                        #[allow(clippy::print_stderr)]
                        {
                            eprintln!(
                                "    SKIP-fallback (Block8x8) leaf sb=({},{}) px=({},{}) bs={:?}",
                                sb_row, sb_col, leaf_px_x, leaf_px_y, leaf_bsize
                            );
                        }
                    }
                    if let Some(recorded) = recorded_coef_tile.as_mut() {
                        record_encoded_inter_leaf(
                            recorded,
                            &mut coef_record_bw,
                            frame_context,
                            &coef_source,
                            leaf_mi_row,
                            leaf_mi_col,
                            sb_partitions[sb_idx][leaf_idx],
                        );
                    }
                    continue;
                }

                // Match libvpx's unrestricted-MV window. Unlike the former
                // tight-plane envelope, this admits filtered vectors at frame
                // edges; picker and reconstruction paths edge-replicate the
                // reference footprint just as libvpx's extended YV12 buffers
                // do. Keep the degenerate check as defensive protection for
                // malformed geometry even though valid frames cannot hit it.
                let mv_limits = if rt_v1_umv_border {
                    libvpx_inter_mv_limits(
                        leaf_mi_row,
                        leaf_mi_col,
                        usize::from(leaf_bsize.n_8x8_wide()),
                        usize::from(leaf_bsize.n_8x8_high()),
                        mi_rows_us,
                        mi_cols_us,
                    )
                } else {
                    tight_inter_mv_limits((leaf_px_x, leaf_px_y), (bw, bh), (w, h))
                };
                let col_min = mv_limits.col_min;
                let col_max = mv_limits.col_max;
                let row_min = mv_limits.row_min;
                let row_max = mv_limits.row_max;
                if col_min > col_max || row_min > row_max {
                    rt_profile_fallback_leaves += 1;
                    sb_partitions[sb_idx][leaf_idx].picked_mode = InterMode::ZeroMv;
                    sb_partitions[sb_idx][leaf_idx].picked_mv = Mv::ZERO;
                    sb_partitions[sb_idx][leaf_idx].skip = true;
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame = RefFrame::LastFrame;
                    copy_last_ref_leaf_to_recon(
                        w,
                        h,
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        last_recon_yuv,
                        (
                            &mut *recon_yuv_out.0,
                            &mut *recon_yuv_out.1,
                            &mut *recon_yuv_out.2,
                        ),
                    );
                    // F.0.2 oracle — also record this degenerate-MV-
                    // envelope fallback so frame-edge leaves aren't
                    // silently missing from the sink (and the libvpx-
                    // side patch's coverage stays paired).
                    if f02_oracle_enabled {
                        crate::f02_oracle::record_decision(crate::f02_oracle::BlockDecision {
                            frame_idx: crate::f02_oracle::current_frame_idx(),
                            mi_row: leaf_mi_row as u16,
                            mi_col: leaf_mi_col as u16,
                            bsize: leaf_bsize,
                            mode: crate::f02_oracle::BlockDecisionMode::Inter(InterMode::ZeroMv),
                            ref_frame: RefFrame::LastFrame,
                            mv_row: 0,
                            mv_col: 0,
                            mode_bits_est: 0,
                            mv_bits_est: 0,
                            ref_bits_est: 0,
                            filter_bits_est: 0,
                            skip_bits_est: 0,
                            coef_bits_est: 0,
                            picker_rate: 0,
                            picker_dist: 0,
                            picker_rd_cost: 0,
                            model_var_y: 0,
                            model_tx_size: rvp9_core::transforms::TxSize::Tx4x4,
                            model_skip_txfm: 2,
                            luma_coef_rate: 0,
                            luma_coef_dist: 0,
                            uv_model_rate: 0,
                            uv_model_dist: 0,
                            uv_u_sensitive: false,
                            uv_v_sensitive: false,
                            partition_u_sensitive: false,
                            partition_v_sensitive: false,
                            last_candidates: [Default::default(); 4],
                            sse_after_pred: 0,
                            interp_filter: Some(
                                rvp9_core::bitstream::frame::InterpFilter::EightTap,
                            ),
                            skip_flag: true,
                            q_index: effective_qindex_for_segment(
                                base_qindex,
                                sb_partitions[sb_idx][leaf_idx].segment_id,
                                seg_q_deltas,
                            ),
                            tx_size: rvp9_core::transforms::TxSize::Tx4x4,
                            luma_eob_nonzero: false,
                        });
                    }
                    row_grids.splat_mode(
                        leaf_mi_row,
                        leaf_mi_col,
                        leaf_bsize,
                        InterMode::ZeroMv,
                        Mv::ZERO,
                        sb_partitions[sb_idx][leaf_idx].segment_id,
                        /* skip = */ true,
                        /* picked_ref_frame = */ RefFrame::LastFrame,
                        /* picked_intra_mode = */ None,
                        /* picked_interp_filter = */
                        rvp9_core::bitstream::frame::InterpFilter::EightTap,
                    );
                    if let Some(recorded) = recorded_coef_tile.as_mut() {
                        record_encoded_inter_leaf(
                            recorded,
                            &mut coef_record_bw,
                            frame_context,
                            &coef_source,
                            leaf_mi_row,
                            leaf_mi_col,
                            sb_partitions[sb_idx][leaf_idx],
                        );
                    }
                    continue;
                }
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
                    row_grids.as_slice(),
                    row_grids.layout,
                    leaf_mi_row,
                    leaf_mi_col,
                    current_tile,
                );
                // Phase 0a — inter-mode context from spatial neighbours.
                // Matches libvpx `x->mbmi_ext->mode_context[ref_frame]`
                // populated by `vp9_find_mv_refs`. Context is ref-frame-
                // independent (only spatial neighbour modes matter).
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
                    .and_then(|g| g.get(leaf_mi_row * mi_cols_us + leaf_mi_col).copied());
                let realtime_mv_context = find_last_golden_nearest_near_mvs_with_context(
                    leaf_bsize,
                    &neighbours,
                    &sign_bias,
                    &bounds,
                    leaf_prev_mv,
                    allow_hp,
                );
                let [last_ref_mvs, golden_ref_mvs] = realtime_mv_context.refs;
                let inter_mode_ctx = realtime_mv_context.mode_context;
                let mvc = MvCandidates {
                    nearest: last_ref_mvs.nearest,
                    near: last_ref_mvs.near,
                };
                let mvc_golden = MvCandidates {
                    nearest: golden_ref_mvs.nearest,
                    near: golden_ref_mvs.near,
                };
                let build_mvc = |ref_frame: RefFrame| -> MvCandidates {
                    // One complete NEARMV walk returns the same first
                    // candidate as NEWMV's early-break walk plus the second
                    // NEARMV candidate. Apply precision lowering to both.
                    let (nearest, near) = find_nearest_near_mvs(
                        leaf_bsize,
                        ref_frame,
                        &neighbours,
                        &sign_bias,
                        &bounds,
                        leaf_prev_mv,
                        allow_hp,
                    );
                    MvCandidates { nearest, near }
                };
                // W-ALTREF — ALTREF MV candidates. `build_mvc` is
                // ref-agnostic; the `sign_bias[ALTREF]` flip (set above
                // when ALTREF is live) is applied inside `find_mv_refs`.
                // Only build them when the reference is live. The default
                // single-layer path marks ALTREF unavailable, so an eager
                // walk here did work that every downstream gate discarded.
                let mvc_altref = build_mv_candidates_if_available(altref_available, || {
                    build_mvc(RefFrame::AltrefFrame)
                });

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
                let quantizer = init_plane_quantizers(i32::from(leaf_qindex), QuantizerDeltas::Y);
                // Track-F iter7 — rdmult for the per-block TX-size RD loop
                // (`choose_inter_luma_tx_size`). libvpx `compute_rd_mult`
                // at the leaf's effective qindex (inter frame).
                let tx_rd_rdmult = i64::from(rvp9_core::cost::compute_rd_mult(
                    i32::from(leaf_qindex),
                    rvp9_core::cost::FrameTypeForRd::Inter,
                ));
                // G1 — UV dequant pair for chroma model_rd.
                let (uv_dc_dq, uv_ac_dq) = plane_dequant_pair(
                    i32::from(leaf_qindex),
                    QuantizerDeltas {
                        dc_delta: 0,
                        ac_delta: 0,
                    },
                );

                // A2.foreman — reject causal MV references outside libvpx's
                // per-block UMV window. The prediction paths safely handle
                // any vector inside this window through edge replication.
                // For a genuinely out-of-window reference, substitute zero
                // in the picker and prevent NEARESTMV from winning below.
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
                    mv_in_strict_limits_or_visible_full_chroma_pel(
                        m,
                        mv_limits,
                        (leaf_px_x, leaf_px_y),
                        (bw, bh),
                        (w, h),
                    )
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
                let intra_have_left = leaf_mi_col > current_tile.mi_col_start;
                let mut intra_above_buf = [0u8; 64];
                let mut intra_left_buf = [0u8; 64];
                let intra_above: Option<&[u8]>;
                let intra_left: Option<&[u8]>;
                let intra_top_left: u8;
                {
                    let recon_y: &[u8] = recon_yuv_out.0;
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
                    row_grids.get(leaf_mi_row - 1, leaf_mi_col)
                } else {
                    None
                };
                let nb_left = if leaf_mi_col > current_tile.mi_col_start {
                    row_grids.get(leaf_mi_row, leaf_mi_col - 1)
                } else {
                    None
                };
                let (
                    ref_frame_cost_intra,
                    ref_frame_cost_last,
                    ref_frame_cost_golden,
                    ref_frame_cost_altref,
                ) = if live_ref_costs_enabled {
                    let costs = active_single_ref_frame_costs(frame_context, nb_above, nb_left);
                    (costs[0], costs[1], costs[2], costs[3])
                } else {
                    (
                        ref_frame_cost_intra,
                        ref_frame_cost_last,
                        ref_frame_cost_golden,
                        ref_frame_cost_altref,
                    )
                };
                let switchable_interp_ctx = get_pred_context_switchable_interp(nb_above, nb_left);
                let filter_ref = get_pred_filter_ref(nb_above, nb_left);
                // P19 — per-block skip-prob context, libvpx
                // `vp9_pred_common.h:55-61` (`above.skip + left.skip`).
                // Threaded into the picker's skip-flag tie-break.
                let skip_ctx = rvp9_core::inter::get_pred_context_skip(nb_above, nb_left);
                // P15 — libvpx `const_motion[ref_frame]` from
                // `mv_refs_rt` (`vp9_pickmode.c:60-153`). Set when the
                // first 2 spatial neighbours don't yield 2 distinct
                // same-ref MVs; the picker uses it to skip NEARMV
                // (`vp9_pickmode.c:2166`).
                let last_const_motion = last_ref_mvs.const_motion;
                let golden_const_motion = golden_ref_mvs.const_motion;

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

                // W4 RDO dispatch (speed≤4). For blocks in `handle_inter`'s
                // single-tile scope (8×8 / 16×16 / 32×32) route the per-leaf
                // mode decision through the transform-domain RDO picker.
                // The winner `(mode, mv, ref, filter)` is mapped into a
                // `PickInterModeResult` so the rest of this loop (leaf
                // field writes, the shared emit TX walk, the mi-grid splat)
                // runs UNCHANGED — the coef-source bridge is shared. When
                // the RDO picker can't decide the leaf (both refs OOB-only)
                // we fall through to the RT picker. 64×64 always falls
                // through (out of `handle_inter` scope). libvpx
                // `vp9_rdopt.c:3438-4187` driver; `vp9_encodeframe.c:1975-
                // 2107` rd_pick_sb_modes dispatch. 8×8 is treated as a
                // single tile (Tx8x8 luma + Tx4x4 chroma) matching the
                // bitstream the writer emits; libvpx's sub-8 driver
                // (`vp9_rdopt.c:4285-5060`) is for sub-8 partitions
                // (Block4x4/Block4x8/Block8x4) which rvp9 never emits.
                // W-COMPOUND — capture a compound winner's second
                // reference + MV (mi-grid slot 1 = ALTREF). `None` for a
                // single-ref winner. The `PickInterModeResult` mapping
                // below drops the second ref/MV (that struct is single-ref
                // only), so we hoist it into this sibling capture and apply
                // it to the leaf desc after the pick resolves.
                let mut rdo_compound_second: Option<(RefFrame, Mv)> = None;
                let rdo_unified: Option<rvp9_core::mode_decision::inter::PickInterModeResult> =
                    if rdo_speed_gated
                        && crate::rdo::inter_dispatch::InterDispatchLeaf::rdo_supported(leaf_bsize)
                    {
                        use crate::rdo::inter_dispatch::{
                            InterDispatchLeaf, RefInputs, pick_inter_sb_leaf,
                        };
                        // A23 — consumer hook (libvpx `vp9_encodeframe.c:2235-
                        // 2242`): `x->rdmult = x->cb_rdmult` when the TPL model
                        // is enabled. `tpl_sb_cb_rdmult` is `Some` only when
                        // TPL is on AND the SB's `get_rdmult_delta` produced a
                        // value distinct from the base rdmult; otherwise the
                        // base q-index rdmult is used (byte-identical path).
                        let base_rdmult = rvp9_core::cost::compute_rd_mult(
                            i32::from(leaf_qindex),
                            rvp9_core::cost::FrameTypeForRd::Inter,
                        ) as i32;
                        let rdmult = if let Some(cb) = tpl_sb_cb_rdmult {
                            crate::tpl::TPL_RDMULT_ADJUSTED_BLOCKS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            i64::from(cb)
                        } else {
                            i64::from(base_rdmult)
                        };
                        let last_inputs = RefInputs {
                            available: true,
                            ref_y: &ref_y,
                            // Reference-buffer split — RDO picker chroma
                            // reads the ME-side LAST view.
                            ref_u: me_ref_yuv.1,
                            ref_v: me_ref_yuv.2,
                            nearest: mvc.nearest,
                            near: mvc.near,
                            nearest_in_bounds: mv_in_bounds(mvc.nearest),
                            near_in_bounds: mv_in_bounds(mvc.near),
                        };
                        let golden_inputs = RefInputs {
                            // GOLDEN availability mirrors the RT picker's
                            // `disable_golden_ref` speed-feature.
                            available: !sf.disable_golden_ref,
                            ref_y: &ref_y_golden,
                            ref_u: golden_recon_yuv.1,
                            ref_v: golden_recon_yuv.2,
                            nearest: mvc_golden.nearest,
                            near: mvc_golden.near,
                            nearest_in_bounds: mv_in_bounds(mvc_golden.nearest),
                            near_in_bounds: mv_in_bounds(mvc_golden.near),
                        };
                        // W-ALTREF — ALTREF single-ref candidate bundle.
                        // `available` is the single gate (`altref_available`,
                        // i.e. enable_alt_ref && ARF emitted). When `false`
                        // the planes are placeholders the runner never reads
                        // (the driver skips unavailable refs), so the picker
                        // is byte-identical to the LAST/GOLDEN-only path.
                        let altref_inputs = RefInputs {
                            available: altref_available,
                            ref_y: ref_y_altref.as_ref().unwrap_or(&ref_y),
                            ref_u: altref_recon_yuv.map_or(me_ref_yuv.1, |(_, u, _)| u),
                            ref_v: altref_recon_yuv.map_or(me_ref_yuv.2, |(_, _, v)| v),
                            nearest: mvc_altref.nearest,
                            near: mvc_altref.near,
                            nearest_in_bounds: mv_in_bounds(mvc_altref.nearest),
                            near_in_bounds: mv_in_bounds(mvc_altref.near),
                        };
                        let dispatch_leaf = InterDispatchLeaf {
                            bsize: leaf_bsize,
                            src_y: &src_y,
                            src_offset: (leaf_px_x, leaf_px_y),
                            src_u: input_yuv.1,
                            src_v: input_yuv.2,
                            uv_stride: uvw,
                            last: last_inputs,
                            golden: golden_inputs,
                            altref: altref_inputs,
                            // W-COMPOUND — the {LAST, ALTREF} compound
                            // candidate is enumerated iff the ALTREF slot is
                            // live (= `enable_alt_ref && ARF emitted`), the
                            // same gate that flips the frame's sign bias and
                            // reference_mode to Select. When `false` no
                            // compound candidate is scored and the picker is
                            // byte-identical to the single-ref path.
                            compound_enabled: altref_available,
                            mv_limits,
                            site_config: &site_config,
                            mv_costs,
                            newmv_seed: sb_newmv_seed,
                            // UV uses zero quant deltas (matches the RT path's
                            // `plane_dequant_pair` call with zero deltas), so
                            // the Y quantizer doubles as the chroma quantizer.
                            y_quant: &quantizer,
                            uv_quant: &quantizer,
                            coef_costs: &rdo_coef_costs,
                            inter_mode_costs: &inter_costs,
                            inter_mode_ctx,
                            switchable_interp_costs: &sic,
                            switchable_interp_ctx,
                            rdmult,
                            tx_size_cost: rdo_tx_size_cost,
                            skip_cost0: rdo_skip_cost0,
                            skip_cost1: rdo_skip_cost1,
                            error_per_bit: 16,
                            sad_per_bit: 16,
                            allow_hp,
                        };
                        pick_inter_sb_leaf(&dispatch_leaf).map(|r| {
                            // W-COMPOUND — hoist the second ref/MV out of the
                            // dispatch result before it's flattened into the
                            // single-ref `PickInterModeResult`.
                            if r.ref_frame2.is_inter() {
                                rdo_compound_second = Some((r.ref_frame2, r.mv2));
                            }
                            rvp9_core::mode_decision::inter::PickInterModeResult {
                                best_mode: r.mode,
                                best_mv: r.mv,
                                // RD/dist/rate fields are consumed only by the
                                // env-gated per-leaf diagnostics below; the
                                // shared emit walk + writer read only
                                // (mode, mv, ref, filter). The RDO picker's
                                // internal RD is not surfaced here.
                                best_rd_cost: 0,
                                best_distortion: 0,
                                partition_distortion: 0,
                                best_rate: 0,
                                coef_rate: 0,
                                coef_dist: 0,
                                uv_rate: 0,
                                uv_dist: 0,
                                picked_interp_filter: r.interp_filter,
                                picked_tx_size: rvp9_core::block::max_txsize(leaf_bsize),
                                skip_txfm: rvp9_core::mode_decision::model_rd::SkipTxfm::None,
                                picked_ref_frame: r.ref_frame,
                                picked_intra_mode: None,
                                var_y: 0,
                                sse_y: 0,
                                // S2 diag fields — the RDO (cpu<=4) path does
                                // not surface the RT picker's skip/eob diag;
                                // inert here (cross-tab targets the RT path).
                                dct_eob_total: 0,
                                picker_skip: false,
                                // Phase-2 option-A: the RDO (cpu<=4) path
                                // doesn't surface per-ref NEWMV searches.
                                newmv_search_last: None,
                                newmv_search_golden: None,
                                candidate_trace_id: 0,
                            }
                        })
                    } else {
                        None
                    };

                // Phase-2 option-A — reuse the partition descent's cached
                // NEWMV search for this committed leaf. The cache holds the
                // descent's full picker result keyed by (position, bsize);
                // a hit means the descent already ran the diamond + sub-pel
                // search for THIS exact leaf. We inject its per-ref NEWMV
                // results as overrides so the final picker skips the search
                // (the costly part) while STILL re-deriving NEAREST/NEAR,
                // skip/inter-mode context, and the DRIFT-FIX on the causal
                // grid — preserving the cross-decoder invariant (R1). The
                // overrides are only CONSUMED if the picker reaches a NEWMV
                // candidate for that ref (gated NEWMV candidates ignore
                // them); and the descent's mv_limits == this pass's
                // mv_limits, so the override is always in-bounds (R7-safe).
                // Only the RT picker (`rdo_unified == None`) honours the
                // overrides; the cpu<=4 RDO path runs its own search.
                let (phase2_override_last, phase2_override_golden) =
                    if phase2_reuse_enabled && rdo_unified.is_none() {
                        match descent_pick_cache
                            .as_ref()
                            .and_then(|cache| cache.get(leaf_mi_row, leaf_mi_col, leaf_bsize))
                        {
                            Some((cached, desc_mvc_last, desc_mvc_golden)) => {
                                // MATCH GUARD (R2): only reuse the descent's
                                // NEWMV when the final pass's CAUSAL
                                // (nearest, near) pair equals the descent's
                                // for that ref. The diamond search seed
                                // derives from {nearest, near} and the rate
                                // cost is priced against nearest; a full
                                // match means the fresh search would land on
                                // the SAME MV, so reuse is faithful. Any
                                // mismatch -> re-search (avoids the R2 quality
                                // wobble from re-pricing a stale-context MV).
                                let ov_last = if mvc.nearest == desc_mvc_last.nearest
                                    && mvc.near == desc_mvc_last.near
                                {
                                    cached.newmv_search_last
                                } else {
                                    None
                                };
                                let ov_golden = if mvc_golden.nearest == desc_mvc_golden.nearest
                                    && mvc_golden.near == desc_mvc_golden.near
                                {
                                    cached.newmv_search_golden
                                } else {
                                    None
                                };
                                if ov_last.is_some() || ov_golden.is_some() {
                                    crate::rdo::inter_dispatch::PHASE2_NEWMV_REUSED
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                } else {
                                    crate::rdo::inter_dispatch::PHASE2_NEWMV_FRESH
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                (ov_last, ov_golden)
                            }
                            None => {
                                crate::rdo::inter_dispatch::PHASE2_NEWMV_FRESH
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                (None, None)
                            }
                        }
                    } else {
                        (None, None)
                    };

                // Unit 430: a very smooth 16x16 descent winner that did not
                // need NEWMV is a bounded negative signal for repeating the
                // expensive NEWMV search under the causal grid. Keep the
                // final search on every motion-selected or less-smooth leaf.
                // `RVP9_A4_BREAKOUT_16_DIST_QMUL=0` leaves this disabled with
                // the parent breakout, preserving one differential escape.
                let mut final_inter_mode_mask = rt_inter_mode_mask;
                if matches!(leaf_bsize, BlockSize::Block16x16)
                    && a4_breakout_16_dist_qmul.is_some_and(|dist_qmul| {
                        descent_pick_cache
                            .as_ref()
                            .and_then(|cache| cache.get(leaf_mi_row, leaf_mi_col, leaf_bsize))
                            .is_some_and(|(cached, _, _)| {
                                stage4a_suppress_final_newmv(
                                    leaf_bsize,
                                    cached.best_mode,
                                    cached.partition_distortion,
                                    base_qindex,
                                    dist_qmul,
                                )
                            })
                    })
                {
                    final_inter_mode_mask[leaf_bsize.idx()] &= !(1u8 << InterMode::NewMv.idx());
                }

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
                    full_pixel_search_method,
                    mv_costs,
                    inter_mode_costs: &inter_costs,
                    inter_mode_ctx,
                    quantizer: &quantizer,
                    q_index: leaf_qindex,
                    allow_hp,
                    frame_kind,
                    newmv_seed: sb_newmv_seed,
                    ref_frame_cost_last,
                    ref_frame_cost_golden,
                    last_nearest_in_bounds: mv_in_bounds(mvc.nearest),
                    last_near_in_bounds: mv_in_bounds(mvc.near),
                    golden_nearest_in_bounds: mv_in_bounds(mvc_golden.nearest),
                    golden_near_in_bounds: mv_in_bounds(mvc_golden.near),
                    // Final-leaf intra fallback is safe now that the causal
                    // and loop-filter grids preserve intra winners. Keep it
                    // out of the speculative descent above; libvpx evaluates
                    // it after the inter-mode search on the selected leaf.
                    enable_intra_fallback: true,
                    intra_above_pixels: intra_above,
                    intra_left_pixels: intra_left,
                    intra_top_left_pixel: intra_top_left,
                    intra_y_mode_costs: &intra_y_mode_costs_inter,
                    ref_frame_cost_intra,
                    switchable_interp_costs: &sic,
                    switchable_interp_ctx,
                    filter_ref,
                    search_golden_filter: rc_mode == crate::rate_control::RcMode::Vbr,
                    skip_ctx,
                    skip_prob: frame_context.skip_probs[skip_ctx],
                    rd_calibration: inter_rd_calibration,
                    last_const_motion,
                    golden_const_motion,
                    // G1 — UV planes for chroma RD in the picker.
                    src_u: input_yuv.1,
                    src_u_stride: uvw,
                    src_v: input_yuv.2,
                    src_v_stride: uvw,
                    // Reference-buffer split — picker chroma SSE/RD
                    // reads the ME-side LAST view.
                    last_ref_u: me_ref_yuv.1,
                    last_ref_v: me_ref_yuv.2,
                    golden_ref_u: golden_recon_yuv.1,
                    golden_ref_v: golden_recon_yuv.2,
                    uv_stride: uvw,
                    color_sensitivity: sb_partition_color_sensitivity,
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
                    // PHASE-3 integration: frame-level noise tier from
                    // W2.4 noise estimator.
                    noise_estimate: noise_estimate_frame,
                    lowvar_highsumdiff: false,
                    // PHASE-3 integration: per-mi skin bit from W2.5.
                    is_skin: {
                        let mi_cols = (width / 8) as usize;
                        let mi_idx = leaf_mi_row * mi_cols + leaf_mi_col;
                        skin_map.get(mi_idx).copied().unwrap_or(0) != 0
                    },
                    newmv_diff_bias_enabled,
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
                    rd_thresh_table: Some(&*rd_thresh_table),
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
                    force_skip_low_temp_var: exact_vbp_tree_enabled
                        && rvp9_core::mode_decision::partition::fixed_tree_force_skip_low_temp_var(
                            &exact_variance_low,
                            leaf_mi_row,
                            leaf_mi_col,
                            leaf_bsize,
                        ),
                    short_circuit_low_temp_var: u8::from(
                        exact_vbp_tree_enabled && rc_mode == crate::rate_control::RcMode::Cbr,
                    ),
                    // PHASE-3 integration: per-SB content_state from
                    // W2.2 avg_source_sad pass. Look up by SB grid.
                    content_state_sb: {
                        let sb_cols = (width / 64).max(1) as usize;
                        let sb_idx = (leaf_mi_row / 8) * sb_cols + (leaf_mi_col / 8);
                        content_state_sb_grid
                            .get(sb_idx)
                            .copied()
                            .unwrap_or(rvp9_core::mode_decision::inter::ContentStateSb::Normal)
                    },
                    is_screen_content: false,
                    zero_temp_sad_source: None,
                    scene_change_detected: scene_change_detected_frame,
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
                    //  - `cb_pred_filter_search` — `2` above 1280×720 at
                    //    speed 5/6, above 640×480 at speed 7, and above
                    //    352×288 at speed 8 (`vp9_speed_features.c:658,
                    //    742, 800`).
                    //  - `disable_golden_ref` — `1` only at speed-9 CBR
                    //    (vp9_speed_features.c:811). Off at our op point.
                    //  - `max_intra_bsize` — `Block32x32` at speed 5+
                    //    (vp9_speed_features.c:571). Matches the previous
                    //    hardcoded `Block64x64` skip.
                    //  - `rt_intra_dc_only_low_content` — `1` only at
                    //    speed-8 (vp9_speed_features.c:770). Default `false`
                    //    here keeps intra-fallback quality at speed 6/7.
                    inter_mode_mask: final_inter_mode_mask,
                    // W0.4: see top of `pick_and_simulate_inter_
                    // encode_partitions` — the dispatcher computed
                    // `sf` once per frame and every per-leaf literal
                    // reads from it instead of hard-coding §11 values.
                    limit_newmv_early_exit: sf.limit_newmv_early_exit,
                    bias_golden: sf.bias_golden,
                    frames_since_golden,
                    golden_ref_is_last,
                    cb_pred_filter_search: sf.cb_pred_filter_search,
                    skip_integer_mv_filter_search: rt_skip_integer_mv_filter_search,
                    mi_row: leaf_mi_row as u32,
                    mi_col: leaf_mi_col as u32,
                    chessboard_index,
                    disable_golden_ref: rt_disable_golden_ref,
                    max_intra_bsize: sf.max_intra_bsize,
                    rt_intra_dc_only_low_content: sf.rt_intra_dc_only_low_content,
                    // Phase-2 option-A — descent NEWMV-search reuse.
                    newmv_override_last: phase2_override_last,
                    newmv_override_golden: phase2_override_golden,
                };
                // W4 RDO dispatch: at speed≤4 the RDO picker (if it
                // decided this leaf) is the source of truth; otherwise run
                // the RT picker. `pick_params` is still built above so the
                // RT fallback (and its borrows) are ready, but the RT
                // picker only runs when RDO didn't decide the leaf — a
                // dispatch that ran RDO then discarded it for the RT result
                // would be the forbidden cheat.
                let cached_descent_pick = if rt_reuse_descent_pick && rdo_unified.is_none() {
                    descent_pick_cache
                        .as_ref()
                        .and_then(|cache| cache.get(leaf_mi_row, leaf_mi_col, leaf_bsize))
                        .map(|(cached, _, _)| *cached)
                } else {
                    None
                };
                let rt_profile_final_pick_start =
                    if rt_profile_enabled && rdo_unified.is_none() && cached_descent_pick.is_none()
                    {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                let unified_pick = match (rdo_unified, cached_descent_pick) {
                    (Some(rdo), _) => rdo,
                    (None, Some(cached)) => cached,
                    (None, None) => pick_inter_mode(&pick_params),
                };
                if let Some(t0) = rt_profile_final_pick_start {
                    rt_profile_final_pick_time += t0.elapsed();
                    rt_profile_picker_leaves += 1;
                }
                if a4_profile_enabled
                    && let Some((cached, _, _)) = descent_pick_cache
                        .as_ref()
                        .and_then(|cache| cache.get(leaf_mi_row, leaf_mi_col, leaf_bsize))
                {
                    let idx = leaf_bsize.idx();
                    a4_profile_committed[idx] += 1;
                    a4_profile_same_pick[idx] += usize::from(
                        unified_pick.best_mode == cached.best_mode
                            && unified_pick.best_mv == cached.best_mv
                            && unified_pick.picked_ref_frame == cached.picked_ref_frame
                            && unified_pick.picked_interp_filter == cached.picked_interp_filter,
                    );
                    a4_profile_rate_delta[idx] +=
                        i64::from(unified_pick.best_rate) - i64::from(cached.best_rate);
                    a4_profile_dist_delta[idx] +=
                        i64::from(unified_pick.best_distortion) - i64::from(cached.best_distortion);
                    a4_profile_coef_rate_delta[idx] +=
                        i128::from(unified_pick.coef_rate) - i128::from(cached.coef_rate);
                }
                // P20 — update `thresh_freq_fact` after the pick.
                // Mirrors libvpx `vp9_pickmode.c:2708-2738`: update the
                // three realtime intra slots after an intra winner, or
                // the four inter slots belonging to the winning ref.
                if unified_pick.picked_ref_frame == rvp9_core::mode_info::RefFrame::IntraFrame {
                    if let Some(intra_mode) = unified_pick.picked_intra_mode {
                        rvp9_core::mode_decision::rd_thresh::update_intra_thresh_freq_fact(
                            rd_thresh_table,
                            leaf_bsize,
                            intra_mode,
                        );
                    }
                } else if matches!(
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
                let mut pick = unified_pick;
                let picked_ref = pick.picked_ref_frame;
                let rt_profile_emit_start = if rt_profile_enabled {
                    Some(std::time::Instant::now())
                } else {
                    None
                };

                // DRIFT-FIX — NEAREST/NEAR carry NO MV bits on the wire;
                // the DECODER reconstructs their MV via `find_mv_refs` on
                // its CAUSAL mode-info grid (the cells decoded so far in
                // z-order). The encoder's recon must use the IDENTICAL
                // value, otherwise its reconstructed reference diverges
                // from the true decode (inter-frame drift). The picker's
                // `best_mv` is NOT a reliable source: the RT picker reads
                // a causal grid that matches, but the cpu≤4 RDO descent
                // picks during the partition search with a grid that lacks
                // the within-SB neighbours present at decode time — so its
                // NEAREST/NEAR `best_mv` can disagree with the decoder
                // (observed at frame edges). `mvc*` here are rebuilt from
                // the recon loop's causal `mi_grid` (committed up to this
                // leaf in z-order — exactly the decoder's view), so we
                // re-pin the picked MV to the matching candidate. This is
                // a no-op for the RT picker (its `mvc` already drove the
                // pick) and corrects the RDO descent. (NEWMV codes an
                // explicit delta on top of the ref-MV, so its `best_mv`
                // stays authoritative; ZEROMV is always (0,0).)
                if matches!(pick.best_mode, InterMode::NearestMv | InterMode::NearMv) {
                    let causal = match picked_ref {
                        RefFrame::GoldenFrame => &mvc_golden,
                        RefFrame::AltrefFrame => &mvc_altref,
                        _ => &mvc,
                    };
                    pick.best_mv = match pick.best_mode {
                        InterMode::NearestMv => causal.nearest,
                        _ => causal.near,
                    };
                }

                let f02_last_candidates = if f02_oracle_enabled {
                    rvp9_core::mode_decision::inter::candidate_trace::take(pick.candidate_trace_id)
                        .map(|candidate| {
                            candidate.map_or_else(Default::default, |candidate| {
                                crate::f02_oracle::CandidateProbe {
                                    available: true,
                                    mv_row: candidate.best_mv.row,
                                    mv_col: candidate.best_mv.col,
                                    rate: candidate.best_rate,
                                    dist: candidate.ranking_distortion.min(i64::MAX as u64) as i64,
                                    rd_cost: candidate.best_rd_cost,
                                    luma_rate: candidate.coef_rate,
                                    luma_dist: candidate.coef_dist,
                                    uv_rate: candidate.uv_rate,
                                    uv_dist: candidate.uv_dist,
                                    skip_txfm: match candidate.skip_txfm {
                                        rvp9_core::mode_decision::model_rd::SkipTxfm::None => 0,
                                        rvp9_core::mode_decision::model_rd::SkipTxfm::AcOnly => 1,
                                        rvp9_core::mode_decision::model_rd::SkipTxfm::AcDc => 2,
                                    },
                                }
                            })
                        })
                } else {
                    [Default::default(); 4]
                };
                // E12 (2026-05-22) — search-failure-vs-prediction-ceiling
                // oracle. Read-only diagnostic, env-gated
                // `RVP9_E12_ORACLE`; no encoder state mutated.
                if crate::e12stat::enabled() {
                    let last_mvc_safe = MvCandidates {
                        nearest: if mv_in_bounds(mvc.nearest) {
                            mvc.nearest
                        } else {
                            Mv::ZERO
                        },
                        near: if mv_in_bounds(mvc.near) {
                            mvc.near
                        } else {
                            Mv::ZERO
                        },
                    };
                    let all_e12 = rvp9_core::mode_decision::inter::pick_inter_mode_luma_all_costs(
                        &InterModeSearchParams {
                            block_size: leaf_bsize,
                            src_plane: &src_y,
                            src_offset: (leaf_px_x, leaf_px_y),
                            ref_plane: &ref_y,
                            mv_candidates: &last_mvc_safe,
                            mv_limits,
                            site_config: &site_config,
                            full_pixel_search_method,
                            mv_costs,
                            inter_mode_costs: &inter_costs,
                            inter_mode_ctx,
                            quantizer: &quantizer,
                            q_index: leaf_qindex,
                            allow_hp,
                            frame_kind,
                            newmv_seed: sb_newmv_seed,
                            ref_frame_rate_cost: ref_frame_cost_last,
                            switchable_interp_costs: &sic,
                            switchable_interp_ctx,
                            skip_ctx: Some(skip_ctx),
                            skip_prob: Some(frame_context.skip_probs[skip_ctx]),
                            rd_calibration: inter_rd_calibration,
                            src_u: input_yuv.1,
                            src_v: input_yuv.2,
                            // Reference-buffer split — diagnostic
                            // (e12) candidate sweep is ME-side.
                            ref_u: me_ref_yuv.1,
                            ref_v: me_ref_yuv.2,
                            uv_stride: uvw,
                            color_sensitivity: sb_partition_color_sensitivity,
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
                        nearest: if nearest_in_bounds {
                            mvc.nearest
                        } else {
                            Mv::ZERO
                        },
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
                        full_pixel_search_method,
                        mv_costs,
                        inter_mode_costs: &inter_costs,
                        inter_mode_ctx,
                        quantizer: &quantizer,
                        q_index: leaf_qindex,
                        allow_hp,
                        frame_kind,
                        newmv_seed: sb_newmv_seed,
                        ref_frame_rate_cost: ref_frame_cost_last,
                        switchable_interp_costs: &sic,
                        switchable_interp_ctx,
                        skip_ctx: Some(skip_ctx),
                        skip_prob: Some(frame_context.skip_probs[skip_ctx]),
                        rd_calibration: inter_rd_calibration,
                        src_u: input_yuv.1,
                        src_v: input_yuv.2,
                        // Reference-buffer split — diagnostic (e3)
                        // candidate sweep is ME-side.
                        ref_u: me_ref_yuv.1,
                        ref_v: me_ref_yuv.2,
                        uv_stride: uvw,
                        color_sensitivity: sb_partition_color_sensitivity,
                        uv_dc_quant: uv_dc_dq as u32,
                        uv_ac_quant: uv_ac_dq as u32,
                    };
                    let all =
                        rvp9_core::mode_decision::inter::pick_inter_mode_luma_all_costs(&e3_params);
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
                    let new_mag = (i32::from(new.best_mv.row).abs()
                        + i32::from(new.best_mv.col).abs())
                        as u32;
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
                            let diff_row = new.best_mv.row.wrapping_sub(mvc_safe.nearest.row);
                            let diff_col = new.best_mv.col.wrapping_sub(mvc_safe.nearest.col);
                            let raw =
                                rvp9_core::cost::mv_cost::mv_cost(diff_row, diff_col, mv_costs)
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
                                leaf_px_x,
                                leaf_px_y,
                                leaf_bsize,
                                leaf_qindex,
                                mvc_safe.nearest.row,
                                mvc_safe.nearest.col,
                                new.best_mv.row,
                                new.best_mv.col,
                                new_mag,
                                all[0].best_rd_cost,
                                all[0].best_distortion,
                                all[0].best_rate,
                                all[1].best_rd_cost,
                                all[1].best_distortion,
                                all[1].best_rate,
                                all[2].best_rd_cost,
                                all[2].best_distortion,
                                all[2].best_rate,
                                all[3].best_rd_cost,
                                all[3].best_distortion,
                                all[3].best_rate,
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
                // W-COMPOUND — record the second reference + MV for a
                // compound winner (slot 1 = ALTREF). The leaf desc carries
                // `NoRefFrame` / `Mv::ZERO` for single-ref leaves. The
                // recon path + mi-grid splat both branch on this.
                if let Some((ref2, mv2)) = rdo_compound_second {
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame2 = ref2;
                    sb_partitions[sb_idx][leaf_idx].picked_mv2 = mv2;
                } else {
                    sb_partitions[sb_idx][leaf_idx].picked_ref_frame2 = RefFrame::NoRefFrame;
                    sb_partitions[sb_idx][leaf_idx].picked_mv2 = Mv::ZERO;
                }
                // F.5 — record the per-leaf interpolation filter so the
                // writer (via `splat_inter_leaf_into_mi_grid`) and the
                // encoder simulator's per-plane MC step below both use
                // the same filter the picker scored against.
                sb_partitions[sb_idx][leaf_idx].picked_interp_filter = pick.picked_interp_filter;
                // Phase 2b — record the intra mode when intra wins.
                sb_partitions[sb_idx][leaf_idx].picked_intra_mode = pick.picked_intra_mode;

                // A0 diagnostic: dump per-leaf picker decision when
                // `RVP9_DIAG_PICKER` is set in the environment.
                if diag_picker_enabled {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "    leaf sb=({},{}) px=({},{}) bs={:?} q={} → mode={:?} mv=({},{}) rd={} dist={} rate={}",
                            sb_row,
                            sb_col,
                            leaf_px_x,
                            leaf_px_y,
                            leaf_bsize,
                            leaf_qindex,
                            pick.best_mode,
                            pick.best_mv.row,
                            pick.best_mv.col,
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
                // Split the emitted EOB total by plane when either the S2
                // cross-tab or F.0.2 oracle needs it. This attributes skip
                // disagreement to Y versus chroma without adding work when
                // both diagnostics are disabled.
                let mut leaf_y_eob: usize = 0;

                if picked_ref == RefFrame::IntraFrame {
                    // G5 — intra-in-inter block. Single-pass TX walk:
                    // prediction + residual (forward transform, quantize,
                    // inverse transform-add, coef collection) per TX
                    // block. The single-walk design is required because
                    // VP9 intra prediction is causal — each TX block
                    // reads reconstructed pixels from prior TX blocks.
                    use rvp9_core::intra_pred::{
                        DcAvailability, EdgeBuffers, build_intra_predictors, intra_predict,
                    };
                    use rvp9_core::transforms::TxType;
                    let intra_y_mode = pick
                        .picked_intra_mode
                        .expect("IntraFrame pick must have picked_intra_mode");
                    let intra_uv_mode = rvp9_core::intra_pred::IntraMode::Dc;
                    let dispatch = rvp9_core::dsp::get_ref();

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
                        let have_left_block = leaf_mi_col > current_tile.mi_col_start;

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
                                let have_right = (tx_x4_local + tx_step_4x4) < plane_bw_4x4;

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
                                    dispatch,
                                );

                                // Step 2: residual = source - prediction.
                                let residual_len = tx_side * tx_side;
                                // Steps 3-4: forward hybrid transform + quant.
                                // RVP9_TRELLIS — intra-in-inter emit path
                                // (KF intra block / intra block in inter
                                // frame). ref_type = 0 (intra).
                                let trellis_ctx = if trellis_enabled {
                                    Some(TrellisCtx {
                                        costs: &rdo_coef_costs,
                                        mb_rdmult: tx_rd_rdmult,
                                        ref_type: 0,
                                    })
                                } else {
                                    None
                                };
                                let eob_usize = inter_emit_scratch.with_len(
                                    residual_len,
                                    |residual, quant_scratch| {
                                        #[cfg(target_arch = "aarch64")]
                                        if matches!(tx_side, 8 | 16 | 32) {
                                            let _ = rvp9_core::dsp::aarch64::subtract_square_neon_with_nonzero(
                                                &input_plane[dst_off..],
                                                plane_stride,
                                                &recon_plane[dst_off..],
                                                plane_stride,
                                                residual,
                                                tx_side,
                                            );
                                        } else {
                                            for r in 0..tx_side {
                                                for c in 0..tx_side {
                                                    let off =
                                                        (tx_px_y + r) * plane_stride + tx_px_x + c;
                                                    residual[r * tx_side + c] = input_plane[off]
                                                        as i16
                                                        - recon_plane[off] as i16;
                                                }
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for r in 0..tx_side {
                                            for c in 0..tx_side {
                                                let off =
                                                    (tx_px_y + r) * plane_stride + tx_px_x + c;
                                                residual[r * tx_side + c] = input_plane[off] as i16
                                                    - recon_plane[off] as i16;
                                            }
                                        }

                                        // E9 DIAG — tap Y-plane intra residual.
                                        if plane == 0 {
                                            crate::predstat::record_y_residual(
                                                residual,
                                                tx_side,
                                                plane_tx as usize,
                                            );
                                            crate::predstat::record_y_pred_block(
                                                recon_plane,
                                                plane_stride,
                                                tx_px_x,
                                                tx_px_y,
                                                tx_side,
                                                plane_w,
                                                plane_h,
                                            );
                                        }

                                        intra_forward_quant_one_tx_with_scratch(
                                            quant_scratch,
                                            residual,
                                            tx_side,
                                            plane_tx,
                                            tx_type,
                                            &quantizer,
                                            usize::from(plane != 0),
                                            trellis_ctx,
                                            |qcoeff, dqcoeff, eob| {
                                                // Step 5: inverse transform-add into recon.
                                                if eob > 0 {
                                                    intra_inverse_transform_add(
                                                        plane_tx,
                                                        tx_type,
                                                        dqcoeff,
                                                        eob,
                                                        &mut recon_plane[dst_off..],
                                                        plane_stride,
                                                    );
                                                }
                                                coef_source.insert_i16(
                                                    leaf_mi_row,
                                                    leaf_mi_col,
                                                    plane,
                                                    tx_x4_local,
                                                    tx_y4_local,
                                                    qcoeff,
                                                    eob,
                                                );
                                                eob
                                            },
                                        )
                                    },
                                );

                                // Step 6: coefficients were stashed directly
                                // from the quantizer scratch above. Step 7:
                                // accumulate eob for the skip decision.
                                leaf_total_eob += eob_usize;
                                if (s2_xtab_enabled || f02_oracle_enabled) && plane == 0 {
                                    leaf_y_eob += eob_usize;
                                }

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
                    //
                    // Start from the RT picker's model-RD TX decision,
                    // matching libvpx's reuse of `mi->tx_size`. The optional
                    // exact variance-tree diagnostic below redecides only the
                    // measured TX8/TX16 leaf class.
                    if emit_profile_enabled {
                        emit_profile_inter_leaves += 1;
                    }
                    let mut chosen_luma_tx =
                        realtime_emit_luma_tx_size(rt_emit_max_tx, leaf_bsize, pick.picked_tx_size);
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
                        // Y plane TX size is RD-chosen below (after the MC
                        // prediction is built); `plane_tx` / `tx_step_4x4` /
                        // `tx_side` are derived from `chosen_luma_tx` at that
                        // point. UV is capped to the Y size.
                        let plane_bw_4x4 = plane_bsize.n_4x4_wide() as usize;
                        let plane_bh_4x4 = plane_bsize.n_4x4_high() as usize;

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
                        // W-ALTREF — when the picker selected ALTREF the
                        // residual-coding MC must read from slot 2's
                        // reconstruction (the decoder's
                        // `reconstruct_inter_block` reads
                        // `refs[ALTREF]`). `altref_recon_yuv` is
                        // guaranteed `Some` here: a leaf can only carry
                        // `picked_ref == AltrefFrame` when `altref_
                        // available` was true (the picker's `altref`
                        // slot was unavailable otherwise). The `.expect`
                        // documents that structural invariant.
                        let altref_yuv = || {
                            altref_recon_yuv.expect(
                                "ALTREF leaf selected but altref_recon_yuv is None — \
                                 the picker must not select ALTREF unless its slot is available",
                            )
                        };
                        let (input_plane, ref_plane_data, recon_plane) = match (plane, picked_ref) {
                            (0, RefFrame::AltrefFrame) => {
                                (input_yuv.0, altref_yuv().0, &mut *recon_yuv_out.0)
                            }
                            (1, RefFrame::AltrefFrame) => {
                                (input_yuv.1, altref_yuv().1, &mut *recon_yuv_out.1)
                            }
                            (_, RefFrame::AltrefFrame) => {
                                (input_yuv.2, altref_yuv().2, &mut *recon_yuv_out.2)
                            }
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
                        //
                        // W-COMPOUND — for a compound leaf this builds the
                        // slot-0 (LAST) prediction first (`is_compound_avg =
                        // false`); the slot-1 (ALTREF) prediction is
                        // averaged into it below. This matches the decoder's
                        // `for r in 0..num_refs` recon loop
                        // (`reconstruct_inter.rs:331`, `is_compound_avg = r >
                        // 0`): slot 0 writes, slot 1 averages.
                        let emit_pred_start = if emit_profile_enabled {
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
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
                            /* is_compound_avg = */ false,
                        );

                        // W-COMPOUND — second-reference averaging pass. A
                        // compound leaf carries `picked_ref_frame2 ==
                        // AltrefFrame` (the {LAST, ALTREF} pair, where slot
                        // 0 = LAST = `picked_ref`). Build the ALTREF
                        // prediction at `picked_mv2` and average it into the
                        // recon plane `(dst + pred + 1) >> 1` via the `_avg`
                        // convolve kernels, exactly as the decoder does for
                        // `r == 1` (`reconstruct_inter.rs:409`,
                        // `is_compound_avg = r > 0`). The ALTREF slot is
                        // guaranteed available (a leaf can only carry a
                        // compound second ref when the picker's `altref`
                        // bundle — gated on `altref_available` — was live).
                        let leaf_ref2 = sb_partitions[sb_idx][leaf_idx].picked_ref_frame2;
                        if leaf_ref2.is_inter() {
                            debug_assert_eq!(
                                leaf_ref2,
                                RefFrame::AltrefFrame,
                                "compound second ref must be ALTREF (the \
                                 {{LAST, ALTREF}} pair's comp_fixed_ref)"
                            );
                            debug_assert_eq!(
                                picked_ref,
                                RefFrame::LastFrame,
                                "compound first ref (slot 0) must be LAST"
                            );
                            let mv2 = sb_partitions[sb_idx][leaf_idx].picked_mv2;
                            let mv2_q4_col = i32::from(mv2.col) * (1i32 << (1 - ss_x as i32));
                            let mv2_q4_row = i32::from(mv2.row) * (1i32 << (1 - ss_y as i32));
                            let ref2_plane_data = match plane {
                                0 => altref_yuv().0,
                                1 => altref_yuv().1,
                                _ => altref_yuv().2,
                            };
                            build_inter_pred_plane_unscaled(
                                ref2_plane_data,
                                plane_stride,
                                plane_w,
                                plane_h,
                                plane_px_x,
                                plane_px_y,
                                block_w_px,
                                block_h_px,
                                mv2_q4_col,
                                mv2_q4_row,
                                recon_plane,
                                plane_stride,
                                sb_partitions[sb_idx][leaf_idx].picked_interp_filter,
                                /* is_compound_avg = */ true,
                            );
                        }
                        if let Some(t0) = emit_pred_start {
                            emit_profile_pred_time += t0.elapsed();
                        }

                        // The default reuses the RT picker's model-RD TX
                        // decision, matching libvpx pickmode's stored
                        // `mi->tx_size`. Diagnostic
                        // `RVP9_RT_EMIT_TX_RD=1` redecides TX8/TX16 only for
                        // TX16-capable leaves; other leaf classes retain the
                        // picker transform. Other large-frame paths keep
                        // their established maximum-TX shortcut.
                        if plane == 0
                            && should_redecide_inter_luma_tx(
                                rt_emit_max_tx,
                                rt_emit_tx_rd,
                                leaf_bsize,
                            )
                        {
                            let emit_tx_select_start = if emit_profile_enabled {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            chosen_luma_tx = choose_inter_luma_tx_size(InterLumaTxInput {
                                input_plane,
                                recon_plane,
                                plane_stride,
                                plane_px_x,
                                plane_px_y,
                                leaf_bsize,
                                quantizer: &quantizer,
                                coef_costs: &rdo_coef_costs,
                                rdmult: tx_rd_rdmult,
                                tx_size_cost: rdo_tx_size_cost,
                                quant_fp: rt_quant_fp,
                                scratch: &mut inter_emit_scratch,
                            });
                            if let Some(t0) = emit_tx_select_start {
                                emit_profile_tx_select_time += t0.elapsed();
                            }
                        }
                        let plane_tx = if plane == 0 {
                            chosen_luma_tx
                        } else {
                            f5_uv_plane_tx_size(leaf_bsize, chosen_luma_tx)
                        };
                        let tx_step_4x4 = 1usize << (plane_tx as usize);
                        let tx_side = plane_tx.side();

                        // libvpx `vp9_encodemb.c:590-604`: in realtime
                        // `quant_fp` mode, `SKIP_TXFM_AC_DC` skips only the
                        // luma forward transform. Chroma is still quantized,
                        // and if chroma produces coefficients the VP9 block
                        // remains non-skip, so the writer still needs explicit
                        // zero-EOB markers for every luma TX. This decision is
                        // independent of whether the transform size came from
                        // the picker, emit-side RD, or the large-frame maximum
                        // transform shortcut.
                        if plane == 0
                            && should_skip_realtime_luma_transform(rt_quant_fp, pick.skip_txfm)
                        {
                            let mut tx_y4_local = 0usize;
                            while tx_y4_local < plane_bh_4x4 {
                                let mut tx_x4_local = 0usize;
                                while tx_x4_local < plane_bw_4x4 {
                                    coef_source.insert(
                                        leaf_mi_row,
                                        leaf_mi_col,
                                        plane,
                                        tx_x4_local,
                                        tx_y4_local,
                                        Vec::new(),
                                        0,
                                    );
                                    tx_x4_local += tx_step_4x4;
                                }
                                tx_y4_local += tx_step_4x4;
                            }
                            continue;
                        }

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
                                if emit_profile_enabled {
                                    emit_profile_tx_blocks += 1;
                                    emit_profile_plane_tx_blocks[plane] += 1;
                                }
                                let residual_len = tx_side * tx_side;
                                let trellis_ctx = if trellis_enabled {
                                    Some(TrellisCtx {
                                        costs: &rdo_coef_costs,
                                        mb_rdmult: tx_rd_rdmult,
                                        ref_type: 1,
                                    })
                                } else {
                                    None
                                };
                                let eob_usize = inter_emit_scratch.with_len(
                                    residual_len,
                                    |residual, quant_scratch| {
                                        let emit_residual_start = if emit_profile_enabled {
                                            Some(std::time::Instant::now())
                                        } else {
                                            None
                                        };
                                        #[cfg(target_arch = "aarch64")]
                                        let residual_nonzero = if matches!(tx_side, 8 | 16 | 32) {
                                            rvp9_core::dsp::aarch64::subtract_square_neon_with_nonzero(
                                                &input_plane[dst_off..],
                                                plane_stride,
                                                &recon_plane[dst_off..],
                                                plane_stride,
                                                residual,
                                                tx_side,
                                            )
                                        } else {
                                            for r in 0..tx_side {
                                                for c in 0..tx_side {
                                                    let off = (tx_px_y + r) * plane_stride
                                                        + (tx_px_x + c);
                                                    residual[r * tx_side + c] = input_plane[off]
                                                        as i16
                                                        - recon_plane[off] as i16;
                                                }
                                            }
                                            residual.iter().any(|&value| value != 0)
                                        };
                                        #[cfg(not(target_arch = "aarch64"))]
                                        let residual_nonzero = {
                                            let mut nonzero = false;
                                            for r in 0..tx_side {
                                                for c in 0..tx_side {
                                                    let off = (tx_px_y + r) * plane_stride
                                                        + (tx_px_x + c);
                                                    let value = input_plane[off] as i16
                                                        - recon_plane[off] as i16;
                                                    residual[r * tx_side + c] = value;
                                                    nonzero |= value != 0;
                                                }
                                            }
                                            nonzero
                                        };
                                        if let Some(t0) = emit_residual_start {
                                            emit_profile_residual_time += t0.elapsed();
                                        }

                                        // E9 DIAG — tap the Y-plane prediction residual
                                        // (`source − MC prediction`, before transform/quant).
                                        if plane == 0 {
                                            crate::predstat::record_y_residual(
                                                residual,
                                                tx_side,
                                                plane_tx as usize,
                                            );
                                            // E9/stage-SSE DIAG — capture Y MC
                                            // prediction pixels into `diag_pred`.
                                            // `recon_plane` holds pure MC prediction
                                            // for this TX (residual not yet added).
                                            crate::predstat::record_y_pred_block(
                                                recon_plane,
                                                plane_stride,
                                                tx_px_x,
                                                tx_px_y,
                                                tx_side,
                                                plane_w,
                                                plane_h,
                                            );
                                        }

                                        // A pixel-exact prediction has an
                                        // identically zero transform and
                                        // quantized coefficient block. The
                                        // coefficient source represents that
                                        // case with its existing empty marker.
                                        if !residual_nonzero {
                                            let emit_coef_insert_start = emit_profile_enabled
                                                .then(std::time::Instant::now);
                                            coef_source.insert(
                                                leaf_mi_row,
                                                leaf_mi_col,
                                                plane,
                                                tx_x4_local,
                                                tx_y4_local,
                                                Vec::new(),
                                                0,
                                            );
                                            if let Some(t0) = emit_coef_insert_start {
                                                emit_profile_coef_insert_time += t0.elapsed();
                                            }
                                            return 0;
                                        }

                                        // RVP9_TRELLIS — inter emit path.
                                        // ref_type = 1 (inter).
                                        let emit_quant_start = if emit_profile_enabled {
                                            Some(std::time::Instant::now())
                                        } else {
                                            None
                                        };
                                        inter_forward_quant_one_tx_with_scratch(
                                            quant_scratch,
                                            residual,
                                            tx_side,
                                            plane_tx,
                                            &quantizer,
                                            usize::from(plane != 0),
                                            trellis_ctx,
                                            rt_quant_fp,
                                            |qcoeff, dqcoeff, eob| {
                                                if let Some(t0) = emit_quant_start {
                                                    emit_profile_quant_time += t0.elapsed();
                                                }
                                                let emit_inverse_start =
                                                    if emit_profile_enabled && eob > 0 {
                                                        Some(std::time::Instant::now())
                                                    } else {
                                                        None
                                                    };
                                                if eob > 0 {
                                                    inter_inverse_dct_add(
                                                        plane_tx,
                                                        dqcoeff,
                                                        eob,
                                                        &mut recon_plane[dst_off..],
                                                        plane_stride,
                                                    );
                                                }
                                                if let Some(t0) = emit_inverse_start {
                                                    emit_profile_inverse_time += t0.elapsed();
                                                }
                                                let emit_coef_insert_start = emit_profile_enabled
                                                    .then(std::time::Instant::now);
                                                coef_source.insert_i16(
                                                    leaf_mi_row,
                                                    leaf_mi_col,
                                                    plane,
                                                    tx_x4_local,
                                                    tx_y4_local,
                                                    qcoeff,
                                                    eob,
                                                );
                                                if let Some(t0) = emit_coef_insert_start {
                                                    emit_profile_coef_insert_time += t0.elapsed();
                                                }
                                                eob
                                            },
                                        )
                                    },
                                );
                                if emit_profile_enabled {
                                    emit_profile_eob_sum += eob_usize;
                                    if eob_usize > 0 {
                                        emit_profile_nonzero_tx_blocks += 1;
                                    }
                                }

                                leaf_total_eob += eob_usize;
                                if (s2_xtab_enabled || f02_oracle_enabled) && plane == 0 {
                                    leaf_y_eob += eob_usize;
                                }

                                tx_x4_local += tx_step_4x4;
                            }
                            tx_y4_local += tx_step_4x4;
                        }
                    }

                    // Track-F iter7 — record the RD-chosen Y tx_size so the
                    // mi-grid builders (`build_inter_mi_grid_from_picker` /
                    // `_for_lf`) splat it and the modes writer signals it
                    // under `TxModeSelect`.
                    sb_partitions[sb_idx][leaf_idx].picked_tx_size = chosen_luma_tx;

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
                crate::skipstat::record_leaf(sb_partitions[sb_idx][leaf_idx].skip, leaf_total_eob);
                // S2 cross-tab diag — one machine-parseable line per
                // *inter* leaf: picker's skip decision + picker Y DCT eob
                // vs the emit's per-plane eob. Parsed offline into the
                // (picker_skip, emit_code) cross-tab decomposed by cause
                // (HLD §S1). Inter-only: `pick`'s S2 diag fields are
                // meaningless for intra leaves. Env-gated; no behaviour.
                if picked_ref != RefFrame::IntraFrame && s2_xtab_enabled {
                    #[allow(clippy::print_stderr)]
                    {
                        let final_skip = sb_partitions[sb_idx][leaf_idx].skip;
                        eprintln!(
                            "S2XTAB pskip={} pdcteob={} ey={} euv={} etot={} fskip={} bs={:?}",
                            u8::from(pick.picker_skip),
                            pick.dct_eob_total,
                            leaf_y_eob,
                            leaf_total_eob - leaf_y_eob,
                            leaf_total_eob,
                            u8::from(final_skip),
                            leaf_bsize,
                        );
                    }
                }
                if diag_picker_enabled {
                    #[allow(clippy::print_stderr)]
                    {
                        eprintln!(
                            "      leaf_total_eob={} (skip={})",
                            leaf_total_eob, sb_partitions[sb_idx][leaf_idx].skip
                        );
                    }
                }
                if let Some(recorded) = recorded_coef_tile.as_mut() {
                    record_encoded_inter_leaf(
                        recorded,
                        &mut coef_record_bw,
                        frame_context,
                        &coef_source,
                        leaf_mi_row,
                        leaf_mi_col,
                        sb_partitions[sb_idx][leaf_idx],
                    );
                }
                // F.0.2 oracle — record one BlockDecision per coded
                // inter leaf, AFTER the RC.X1 skip-on-zero-eob override
                // (so `skip_flag` matches what lands in the bitstream).
                // Single relaxed atomic load when `RVP9_F02_ORACLE` is
                // unset (HLD §3.2 / §7).
                if f02_oracle_enabled {
                    let final_skip = sb_partitions[sb_idx][leaf_idx].skip;
                    let final_mode = sb_partitions[sb_idx][leaf_idx].picked_mode;
                    let final_mv = sb_partitions[sb_idx][leaf_idx].picked_mv;
                    let final_ref = sb_partitions[sb_idx][leaf_idx].picked_ref_frame;
                    let oracle_mode = f02_oracle_mode(
                        final_ref,
                        final_mode,
                        sb_partitions[sb_idx][leaf_idx].picked_intra_mode,
                    );
                    // Re-derive `mv_bits_est` for NEWMV using the same
                    // formula the picker uses internally
                    // (`evaluate_candidate` in
                    // `core::mode_decision::inter`), since the existing
                    // `InterModeSearchResult` only returns the combined
                    // `best_rate` (HLD §3.4.1 forward-looking note).
                    // Reference MV is the safe `nearest` MV the picker
                    // saw (post-OOB-veto), matching the
                    // `pick_for_ref` closure's `mvc_safe.nearest`.
                    let mv_bits_est: u32 = if matches!(final_mode, InterMode::NewMv) {
                        let mvc_ref = match final_ref {
                            RefFrame::GoldenFrame => mvc_golden.nearest,
                            RefFrame::AltrefFrame => mvc_altref.nearest,
                            _ => mvc.nearest,
                        };
                        let ref_mv = if mv_in_bounds(mvc_ref) {
                            mvc_ref
                        } else {
                            Mv::ZERO
                        };
                        let diff_row = final_mv.row.wrapping_sub(ref_mv.row);
                        let diff_col = final_mv.col.wrapping_sub(ref_mv.col);
                        let raw = rvp9_core::cost::mv_cost::mv_cost(diff_row, diff_col, mv_costs)
                            .max(0) as u64;
                        // ROUND_POWER_OF_TWO(raw * 108, 7) — same
                        // `MV_COST_WEIGHT = 108` scaling the picker
                        // uses.
                        ((raw * 108 + 64) >> 7).min(u32::MAX as u64) as u32
                    } else {
                        0
                    };
                    let ref_cost = match final_ref {
                        RefFrame::GoldenFrame => ref_frame_cost_golden,
                        RefFrame::AltrefFrame => ref_frame_cost_altref,
                        RefFrame::LastFrame => ref_frame_cost_last,
                        _ => 0,
                    };
                    let mode_bits_est = if final_ref == RefFrame::IntraFrame {
                        0
                    } else {
                        inter_costs[inter_mode_ctx][final_mode.idx()].max(0) as u32
                    };
                    let filter_bits_est = if final_ref != RefFrame::IntraFrame
                        && (final_mv.row | final_mv.col) & 0x07 != 0
                    {
                        sic[switchable_interp_ctx][pick.picked_interp_filter as usize].max(0) as u32
                    } else {
                        0
                    };
                    let skip_bits_est = if final_ref == RefFrame::IntraFrame {
                        0
                    } else {
                        let skip_prob = frame_context.skip_probs[skip_ctx];
                        rvp9_core::cost::vp9_cost_bit(skip_prob, pick.picker_skip)
                    };
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
                    let coef_bits_est: u32 = (pick.coef_rate
                        >> rvp9_core::cost::VP9_PROB_COST_SHIFT)
                        .min(u32::MAX as u64) as u32;
                    crate::f02_oracle::record_decision(crate::f02_oracle::BlockDecision {
                        frame_idx: crate::f02_oracle::current_frame_idx(),
                        mi_row: leaf_mi_row as u16,
                        mi_col: leaf_mi_col as u16,
                        bsize: leaf_bsize,
                        mode: oracle_mode,
                        ref_frame: final_ref,
                        mv_row: if final_ref == RefFrame::IntraFrame {
                            0
                        } else {
                            final_mv.row
                        },
                        mv_col: if final_ref == RefFrame::IntraFrame {
                            0
                        } else {
                            final_mv.col
                        },
                        mode_bits_est,
                        mv_bits_est,
                        ref_bits_est: ref_cost,
                        filter_bits_est,
                        skip_bits_est,
                        coef_bits_est,
                        picker_rate: pick.best_rate,
                        picker_dist: i64::from(pick.best_distortion),
                        picker_rd_cost: pick.best_rd_cost,
                        model_var_y: pick.var_y,
                        model_tx_size: pick.picked_tx_size,
                        model_skip_txfm: match pick.skip_txfm {
                            rvp9_core::mode_decision::model_rd::SkipTxfm::None => 0,
                            rvp9_core::mode_decision::model_rd::SkipTxfm::AcOnly => 1,
                            rvp9_core::mode_decision::model_rd::SkipTxfm::AcDc => 2,
                        },
                        luma_coef_rate: pick.coef_rate,
                        luma_coef_dist: pick.coef_dist,
                        uv_model_rate: pick.uv_rate,
                        uv_model_dist: pick.uv_dist,
                        uv_u_sensitive: final_ref != RefFrame::IntraFrame
                            && sb_partition_color_sensitivity[0],
                        uv_v_sensitive: final_ref != RefFrame::IntraFrame
                            && sb_partition_color_sensitivity[1],
                        partition_u_sensitive: sb_partition_color_sensitivity[0],
                        partition_v_sensitive: sb_partition_color_sensitivity[1],
                        last_candidates: f02_last_candidates,
                        sse_after_pred: u64::from(pick.sse_y),
                        interp_filter: if final_ref == RefFrame::IntraFrame {
                            None
                        } else {
                            Some(pick.picked_interp_filter)
                        },
                        skip_flag: final_skip,
                        q_index: leaf_qindex,
                        // Emission can replace the picker's model-RD
                        // transform choice. Report the transform stored in
                        // the final leaf because that is what the modes
                        // writer signals and the coefficient walker used.
                        tx_size: sb_partitions[sb_idx][leaf_idx].picked_tx_size,
                        luma_eob_nonzero: leaf_y_eob > 0,
                    });
                }

                // A2.foreman - Splat the picked `(mode, mv, skip)`
                // into the in-progress mi_grid so subsequent leaves'
                // neighbour lookups see this leaf's decision.
                // A15 Phase B Step 4 — carry `picked_ref_frame` too so
                // neighbour MV-ref propagation matches the per-ref
                // picker behaviour (the decoder's `find_mv_refs` is
                // ref-aware).
                row_grids.splat_final(leaf_mi_row, leaf_mi_col, sb_partitions[sb_idx][leaf_idx]);
                if let Some(t0) = rt_profile_emit_start {
                    rt_profile_emit_time += t0.elapsed();
                }
            }
        }
    };
    if let Some(tile_col) = active_tile_col {
        for sb_row in 0..sb_rows {
            encode_job(crate::row_mt::RowJob { sb_row, tile_col });
        }
    } else {
        let row_context = crate::row_mt::RowMtContext::new(tile_bounds_by_col.len(), sb_rows);
        let mut tile_col = row_context.initial_tile_for_worker(0);
        while let Some(job) = row_context.claim_for_worker(&mut tile_col) {
            encode_job(job);
        }
        debug_assert!(row_context.queues_drained());
        debug_assert_eq!(row_context.remaining_jobs(), 0);
    }

    if rt_profile_enabled {
        let total = rt_profile_frame_start.elapsed();
        let accounted = rt_profile_descent_time + rt_profile_final_pick_time + rt_profile_emit_time;
        let other = total.saturating_sub(accounted);
        let reused = crate::rdo::inter_dispatch::PHASE2_NEWMV_REUSED
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(rt_profile_phase2_reused_start);
        let fresh = crate::rdo::inter_dispatch::PHASE2_NEWMV_FRESH
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(rt_profile_phase2_fresh_start);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1_000.0;
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[RT_PROFILE] total={:.2}ms descent={:.2}ms final_pick={:.2}ms emit={:.2}ms other={:.2}ms leaves={} picker_leaves={} fallback_leaves={} phase2_reused={} phase2_fresh={}",
                ms(total),
                ms(rt_profile_descent_time),
                ms(rt_profile_final_pick_time),
                ms(rt_profile_emit_time),
                ms(other),
                rt_profile_final_leaves,
                rt_profile_picker_leaves,
                rt_profile_fallback_leaves,
                reused,
                fresh,
            );
        }
    }
    if emit_profile_enabled {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1_000.0;
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[EMIT_PROFILE] inter_leaves={} tx_blocks={} y_tx={} u_tx={} v_tx={} nonzero_tx={} eob_sum={} pred={:.2}ms tx_select={:.2}ms residual={:.2}ms quant={:.2}ms inverse={:.2}ms coef_insert={:.2}ms",
                emit_profile_inter_leaves,
                emit_profile_tx_blocks,
                emit_profile_plane_tx_blocks[0],
                emit_profile_plane_tx_blocks[1],
                emit_profile_plane_tx_blocks[2],
                emit_profile_nonzero_tx_blocks,
                emit_profile_eob_sum,
                ms(emit_profile_pred_time),
                ms(emit_profile_tx_select_time),
                ms(emit_profile_residual_time),
                ms(emit_profile_quant_time),
                ms(emit_profile_inverse_time),
                ms(emit_profile_coef_insert_time),
            );
        }
    }
    if a4_profile_enabled {
        let labels = [
            "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16", "32x32",
            "32x64", "64x32", "64x64",
        ];
        let mut parts = Vec::new();
        let mut committed_parts = Vec::new();
        let total_time = a4_profile_pick_time
            .iter()
            .copied()
            .fold(std::time::Duration::ZERO, |acc, d| acc + d);
        for (idx, &calls) in a4_profile_calls.iter().enumerate() {
            if calls == 0 {
                continue;
            }
            parts.push(format!(
                "{}={}({:.2}ms)",
                labels[idx],
                calls,
                a4_profile_pick_time[idx].as_secs_f64() * 1_000.0,
            ));
            let committed = a4_profile_committed[idx];
            if committed > 0 {
                committed_parts.push(format!(
                    "{}={} same={} d_rate={:.1} d_dist={:.1} d_coef_rate={:.1}",
                    labels[idx],
                    committed,
                    a4_profile_same_pick[idx],
                    a4_profile_rate_delta[idx] as f64 / committed as f64,
                    a4_profile_dist_delta[idx] as f64 / committed as f64,
                    a4_profile_coef_rate_delta[idx] as f64 / committed as f64,
                ));
            }
        }
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[A4_PROFILE] total_pick={:.2}ms {}",
                total_time.as_secs_f64() * 1_000.0,
                parts.join(" "),
            );
            eprintln!("[A4_COMMITTED] {}", committed_parts.join(" "));
        }
    }

    // A3'-int-pro frame-level diagnostic — aggregate mode counts.
    // Set RVP9_DIAG_INTPRO=1 to enable.
    // A15 Phase B Step 5 — also count GOLDEN picks (key diagnostic
    // for the widened-picker BD-rate measurement).
    if env.diag_intpro {
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
                count_zero,
                count_nearest,
                count_near,
                count_new,
                count_skip,
                count_nonzero_mv,
                count_golden,
            );
        }
    }

    if a4_split_stats_enabled {
        let comparable =
            a4_split_stats.block16_split_beats_none + a4_split_stats.block16_split_loses_to_none;
        let avg_margin = if comparable == 0 {
            0.0
        } else {
            a4_split_stats.block16_margin_sum as f64 / f64::from(comparable)
        };
        let avg_rate_margin = if comparable == 0 {
            0.0
        } else {
            a4_split_stats.block16_rate_margin_sum as f64 / f64::from(comparable)
        };
        let avg_dist_margin = if comparable == 0 {
            0.0
        } else {
            a4_split_stats.block16_dist_margin_sum as f64 / f64::from(comparable)
        };
        let min_margin = a4_split_stats
            .block16_margin_min
            .map_or_else(|| "n/a".to_string(), |v| v.to_string());
        let max_margin = a4_split_stats
            .block16_margin_max
            .map_or_else(|| "n/a".to_string(), |v| v.to_string());
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[A4-split] q={base_qindex} block16 attempts={} breakout_suppressed={} feasible={} infeasible_or_early={} child_infeasible={} partial_rd_early_out={} partial_after_child={:?} split_beats_none={} split_loses_to_none={} split_committed={} rect_overrides_split={} no_none={} slack_would_continue={} slack_completed={} slack_split_eligible={} slack_split_committed={} slack_rect_override={} slack_rejected={} avg_margin={avg_margin:.1} avg_rate_margin={avg_rate_margin:.1} avg_dist_margin={avg_dist_margin:.1} min_margin={min_margin} max_margin={max_margin}",
                a4_split_stats.block16_attempts,
                a4_split_stats.block16_breakout_suppressed,
                a4_split_stats.block16_feasible,
                a4_split_stats.block16_infeasible_or_early,
                a4_split_stats.block16_child_infeasible,
                a4_split_stats.block16_partial_rd_early_out,
                a4_split_stats.block16_partial_early_out_after_child,
                a4_split_stats.block16_split_beats_none,
                a4_split_stats.block16_split_loses_to_none,
                a4_split_stats.block16_split_committed,
                a4_split_stats.block16_rect_overrides_split,
                a4_split_stats.block16_no_none,
                a4_split_stats.block16_slack_would_continue,
                a4_split_stats.block16_slack_completed,
                a4_split_stats.block16_slack_split_eligible,
                a4_split_stats.block16_slack_split_committed,
                a4_split_stats.block16_slack_rect_override,
                a4_split_stats.block16_slack_rejected,
            );
        }
    }

    stage4a_split_stats_out.accumulate(&a4_split_stats);

    // Phase A (`RVP9_P8_STATS`) — per-frame partition-size distribution
    // over the final leaf list (counts leaves, not area). Reports the
    // descent-chosen 8×8 cumulative counter alongside, distinguishing
    // "8×8 appears" mechanics from the variance-hint baseline.
    if env.p8_stats {
        let mut n8 = 0usize;
        let mut n16 = 0usize;
        let mut n32 = 0usize;
        let mut n64 = 0usize;
        let mut nrect = 0usize;
        for sb in sb_partitions.iter() {
            for leaf in sb.iter() {
                match leaf.leaf_bsize {
                    BlockSize::Block8x8 => n8 += 1,
                    BlockSize::Block16x16 => n16 += 1,
                    BlockSize::Block32x32 => n32 += 1,
                    BlockSize::Block64x64 => n64 += 1,
                    _ => nrect += 1,
                }
            }
        }
        let descent_8x8 = crate::rdo::inter_dispatch::RD_PARTITION_8X8_LEAVES
            .load(std::sync::atomic::Ordering::Relaxed);
        #[allow(clippy::print_stderr)]
        {
            eprintln!(
                "[P8] frame leaves: 8x8={n8} 16x16={n16} 32x32={n32} 64x64={n64} rect={nrect} | descent_8x8_cumulative={descent_8x8}",
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
                e3_win_counts[0],
                e3_win_counts[1],
                e3_win_counts[2],
                e3_win_counts[3],
                100.0 * e3_win_counts[3] as f64 / n as f64,
                e3_newmv_nonzero,
                100.0 * e3_newmv_nonzero as f64 / n as f64,
                e3_newmv_mag_sum as f64 / n as f64,
                e3_newmv_mag_max,
                e3_newmv_best_dist,
                e3_newmv_best_dist_lost,
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_mv_rate_sum / e3_newmv_best_dist_lost as u64
                } else {
                    0
                },
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_total_rate_sum / e3_newmv_best_dist_lost as u64
                } else {
                    0
                },
                if e3_newmv_best_dist_lost > 0 {
                    e3_lost_rd_gap_sum / e3_newmv_best_dist_lost as i128
                } else {
                    0
                },
            );
        }
    }

    InterSimulationResult {
        coefs: coef_source,
        recorded_coef_tile,
        mi_grid: row_grids.into_simulation_grid(),
    }
}

/// Build a `bw × bh` inter prediction at `(plane_px_x + mv)` into
/// `dst[..]`, mirroring the decoder's unscaled
/// `build_inter_predictor_with_mc_border` fast path (ref-dim == cur-dim,
/// so [`ScaleFactors`] is identity). MVs are in Q4 (1/16-pel) units in
/// the reference plane's coord system.
///
/// DRIFT-FIX — MVs whose convolve8 footprint spills past the visible
/// plane edge are handled by an edge-replicated scratch (libvpx
/// `build_mc_border`), so the encoder's prediction is byte-identical to
/// the decoder's border-extended one even when the MV points off-edge.
/// (Previously this assumed the footprint always stayed in-bounds, which
/// silently read wrong pixels near edges and caused inter-frame drift.)
///
/// `VP9_INTERP_EXTEND` (libvpx `yv12config.h:25`). The 8-tap convolve
/// footprint extends by `VP9_INTERP_EXTEND - 1 = 3` leading and
/// `VP9_INTERP_EXTEND = 4` trailing samples on any sub-pel-filtered axis.
const VP9_INTERP_EXTEND: i32 = 4;

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
    // W-COMPOUND — `true` for the second (ALTREF) reference of a
    // compound leaf: the prediction is averaged into the existing `dst`
    // contents `(dst + pred + 1) >> 1` via the `_avg` convolve kernels,
    // matching the decoder's `is_compound_avg = r > 0` recon
    // (`reconstruct_inter.rs:331-476`). `false` for single-ref leaves and
    // for the first (LAST) reference of a compound leaf.
    is_compound_avg: bool,
) {
    build_inter_pred_plane_unscaled_at_dst(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        plane_px_x,
        plane_px_y,
        plane_px_x,
        plane_px_y,
        bw,
        bh,
        mv_q4_col,
        mv_q4_row,
        dst,
        dst_stride,
        interp_filter,
        is_compound_avg,
    );
}

/// Product-side partition predictor variant. The reference origin remains in
/// full-frame coordinates while the destination may start at `(0, 0)` in a
/// compact scratch block.
#[allow(clippy::too_many_arguments)]
fn build_inter_pred_block_unscaled(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    ref_px_x: usize,
    ref_px_y: usize,
    bw: usize,
    bh: usize,
    mv_q4_col: i32,
    mv_q4_row: i32,
    dst: &mut [u8],
    dst_stride: usize,
    interp_filter: rvp9_core::bitstream::frame::InterpFilter,
) {
    build_inter_pred_plane_unscaled_at_dst(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        ref_px_x,
        ref_px_y,
        0,
        0,
        bw,
        bh,
        mv_q4_col,
        mv_q4_row,
        dst,
        dst_stride,
        interp_filter,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn build_inter_pred_plane_unscaled_at_dst(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    ref_px_x: usize,
    ref_px_y: usize,
    dst_px_x: usize,
    dst_px_y: usize,
    bw: usize,
    bh: usize,
    mv_q4_col: i32,
    mv_q4_row: i32,
    dst: &mut [u8],
    dst_stride: usize,
    interp_filter: rvp9_core::bitstream::frame::InterpFilter,
    is_compound_avg: bool,
) {
    use rvp9_core::inter::filter::{SUBPEL_BITS, SUBPEL_MASK};
    use rvp9_core::inter::scale::{ScaleFactors, inter_predictor_with_dispatch};

    // Identity scale (ref dims = cur dims).
    let sf = ScaleFactors::new(ref_w as u32, ref_h as u32, ref_w as u32, ref_h as u32);
    debug_assert!(
        !sf.is_scaled(),
        "F10.1c v1 encoder assumes identity ref scale"
    );

    let total_x_q4 = sf.scaled_x_q4(ref_px_x as i32, mv_q4_col as i16);
    let total_y_q4 = sf.scaled_y_q4(ref_px_y as i32, mv_q4_row as i16);
    let int_x = total_x_q4 >> SUBPEL_BITS;
    let int_y = total_y_q4 >> SUBPEL_BITS;
    let subpel_x_q4 = total_x_q4 & SUBPEL_MASK;
    let subpel_y_q4 = total_y_q4 & SUBPEL_MASK;

    // Destination slice starts at the leaf's top-left in the plane.
    let dst_off = dst_px_y * dst_stride + dst_px_x;

    // DRIFT-FIX — the encoder's reference planes are TIGHT (no border
    // padding; see `buffer_pool::RefSlotPlanes`), but the DECODER reads
    // from an edge-extended reference (`extend_borders` replicates edge
    // pixels into a ±32 Y / ±16 UV border) and falls back to an
    // edge-replicated scratch for any MV whose convolve footprint spills
    // past that border (`build_inter_predictor_with_mc_border`). With a
    // tight encoder plane, ANY footprint cell outside `[0, ref_w) × [0,
    // ref_h)` would read the wrong (adjacent-row / out-of-bounds) pixel,
    // producing a prediction that differs from the decoder's
    // edge-replicated one — the source of the inter-frame drift. We
    // mirror the decoder's algorithm exactly: compute the convolve
    // footprint, and whenever it leaves the visible region, build an
    // edge-replicated scratch (libvpx `build_mc_border`) and predict
    // from that. The fast path (footprint fully inside) is byte-identical
    // to the prior code.
    let x_pad = subpel_x_q4 != 0;
    let y_pad = subpel_y_q4 != 0;
    // Footprint (one-past-last touched sample) — same as the decoder's
    // unscaled path: `x1 = int_x + bw`, extended by the 8-tap halo when
    // the axis is sub-pel filtered.
    let mut x0 = int_x;
    let mut y0 = int_y;
    let mut x1 = int_x + bw as i32;
    let mut y1 = int_y + bh as i32;
    if x_pad {
        x0 -= VP9_INTERP_EXTEND - 1;
        x1 += VP9_INTERP_EXTEND;
    }
    if y_pad {
        y0 -= VP9_INTERP_EXTEND - 1;
        y1 += VP9_INTERP_EXTEND;
    }

    let needs_scratch = x0 < 0 || y0 < 0 || x1 > ref_w as i32 || y1 > ref_h as i32;

    if !needs_scratch {
        // Fast path: footprint is fully inside the tight plane. Pre-shift
        // the ref pointer by `(VP9_INTERP_EXTEND - 1) = 3` along each
        // sub-pel axis (mirrors the decoder's fast path).
        let row_pre = if y_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let col_pre = if x_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
        let row_base = (int_y + row_pre) as usize;
        let col_base = (int_x + col_pre) as usize;
        let src_base = row_base * ref_stride + col_base;

        inter_predictor_with_dispatch(
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
            is_compound_avg,
            rvp9_core::dsp::get_ref(),
        );
        return;
    }

    // Scratch path: replicate the decoder's `build_mc_border` +
    // predict-from-scratch. `b_w × b_h` is the footprint size (including
    // the 8-tap halo); the scratch's top-left maps to footprint cell
    // `(x0, y0)`. The kernel then reads with its usual `-3` pre-shift
    // relative to the scratch origin where the axis is sub-pel filtered.
    let b_w = (x1 - x0) as usize;
    let b_h = (y1 - y0) as usize;
    let mut scratch = vec![0u8; b_w * b_h];
    enc_build_mc_border(
        ref_plane,
        ref_stride,
        &mut scratch,
        b_w,
        x0,
        y0,
        b_w,
        b_h,
        ref_w as i32,
        ref_h as i32,
    );
    // Within the scratch, the integer-pel base sits at `(int_x - x0,
    // int_y - y0)`; the kernel pre-shift is folded into that offset.
    let scratch_col = (int_x - x0) + if x_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
    let scratch_row = (int_y - y0) + if y_pad { -(VP9_INTERP_EXTEND - 1) } else { 0 };
    let scratch_base = (scratch_row as usize) * b_w + (scratch_col as usize);

    inter_predictor_with_dispatch(
        &scratch[scratch_base..],
        b_w,
        &mut dst[dst_off..],
        dst_stride,
        subpel_x_q4,
        subpel_y_q4,
        &sf,
        interp_filter,
        bw,
        bh,
        is_compound_avg,
        rvp9_core::dsp::get_ref(),
    );
}

/// libvpx `build_mc_border` (`vp9_decodeframe.c:456-490`), encoder-side
/// port for a TIGHT (unbordered) reference plane. Copies a `b_w × b_h`
/// rectangle of reference pixels with top-left at visible-region
/// coordinate `(x, y)` into `dst` (tight stride `dst_stride == b_w`),
/// edge-replicating any cell that falls outside `[0, w) × [0, h)`. This
/// matches the decoder's `build_mc_border` byte-for-byte; the only
/// difference is that the decoder's source already carries a static
/// edge-extended border, whereas here the visible region is the whole
/// plane and all padding is produced by this replication.
#[allow(clippy::too_many_arguments)] // mirrors libvpx's `build_mc_border` 1:1
fn enc_build_mc_border(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    x: i32,
    mut y: i32,
    b_w: usize,
    b_h: usize,
    w: i32,
    h: i32,
) {
    let mut y_clamped = if y >= h {
        h - 1
    } else if y > 0 {
        y
    } else {
        0
    };

    let mut dst_off = 0usize;
    for _ in 0..b_h {
        let left = if x < 0 { (-x) as usize } else { 0 };
        let left = left.min(b_w);
        let right_raw = x + b_w as i32 - w;
        let right = if right_raw > 0 { right_raw as usize } else { 0 };
        let right = right.min(b_w);
        let copy = b_w.saturating_sub(left + right);

        let row_base = (y_clamped as usize) * src_stride;
        let row = &src[row_base..];

        if left > 0 {
            let fill = row[0];
            for c in 0..left {
                dst[dst_off + c] = fill;
            }
        }
        if copy > 0 {
            let src_start = (x + left as i32) as usize;
            dst[dst_off + left..dst_off + left + copy]
                .copy_from_slice(&row[src_start..src_start + copy]);
        }
        if right > 0 {
            let fill = row[(w - 1) as usize];
            for c in 0..right {
                dst[dst_off + left + copy + c] = fill;
            }
        }

        dst_off += dst_stride;
        y += 1;
        if y > 0 && y < h {
            y_clamped = y;
        }
    }
}

/// Trellis context for the `RVP9_TRELLIS` post-quant coefficient
/// optimizer, threaded into the inter/intra emit paths. `None` ⇒ trellis
/// OFF (byte-identical baseline). When `Some`, the trellis runs AFTER
/// forward-quant and BEFORE the coef_source insert + reconstruction, so
/// the recon uses the trellis-adjusted dqcoeff (encoder recon ==
/// decoder).
#[derive(Copy, Clone)]
struct TrellisCtx<'a> {
    costs: &'a rvp9_core::cost::coef_cost::CoefCostTable,
    /// libvpx `mb->rdmult` (the `compute_rd_mult` value); the
    /// `plane_rd_mult` scaling is applied inside `optimize_b_inter`.
    mb_rdmult: i64,
    /// 0 = intra, 1 = inter.
    ref_type: usize,
}

/// Forward DCT (DctDct) + forward quant of one TX block for the inter
/// path. Returns `(qcoeff_i32, dqcoeff_i32, eob)` — the qcoeff buffer
/// the writer will serialise, the dqcoeff buffer to inverse-transform
/// into the recon, and the end-of-block count.
///
/// `trellis` is `Some` only when `RVP9_TRELLIS` is set; it runs
/// `vp9_optimize_b` on the post-quant coefficients before they are
/// returned (so both the serialised qcoeff AND the recon-driving dqcoeff
/// are the trellis-adjusted values).
#[allow(clippy::too_many_arguments)] // transform/quant inputs plus the scoped output callback
fn inter_forward_quant_one_tx_with_scratch<R>(
    scratch: &mut InterEmitQuantScratch,
    residual: &[i16],
    tx_side: usize,
    plane_tx: TxSize,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
    plane_type: usize,
    trellis: Option<TrellisCtx<'_>>,
    quant_fp: bool,
    operation: impl FnOnce(&[i16], &[i32], usize) -> R,
) -> R {
    use rvp9_core::quant::forward::{
        vp9_quantize_b_32x32_with_order, vp9_quantize_b_with_order,
        vp9_quantize_fp_32x32_with_order, vp9_quantize_fp_with_order,
    };
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::forward::{fdct8x8_libvpx_lowbd, fdct16x16, fdct32x32, fht4x4};
    use rvp9_core::transforms::scans::scan_for;

    {
        let n = plane_tx.n_coeffs();
        match plane_tx {
            TxSize::Tx4x4 => {
                // Inter blocks always use DctDct (no ADST hybrid).
                let output: &mut [i32; 16] = (&mut scratch.coefs[..n])
                    .try_into()
                    .expect("Tx4x4 coefficient prefix has 16 entries");
                fht4x4(residual, output, tx_side, TxType::DctDct);
            }
            TxSize::Tx8x8 => {
                let output: &mut [i32; 64] = (&mut scratch.coefs[..n])
                    .try_into()
                    .expect("Tx8x8 coefficient prefix has 64 entries");
                fdct8x8_libvpx_lowbd(residual, output, tx_side);
            }
            TxSize::Tx16x16 => {
                let output: &mut [i32; 256] = (&mut scratch.coefs[..n])
                    .try_into()
                    .expect("Tx16x16 coefficient prefix has 256 entries");
                fdct16x16(residual, output, tx_side);
            }
            TxSize::Tx32x32 => fdct32x32(residual, &mut scratch.coefs, tx_side),
        }

        let scan = scan_for(plane_tx, TxType::DctDct);
        let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
        let coefs = &scratch.coefs[..n];
        let qcoeff = &mut scratch.qcoeff[..n];
        let dqcoeff = &mut scratch.dqcoeff[..n];
        let eob = if quant_fp {
            match plane_tx {
                TxSize::Tx32x32 => vp9_quantize_fp_32x32_with_order(
                    coefs,
                    /* skip_block = */ false,
                    &quantizer.round_fp,
                    &quantizer.quant_fp,
                    qcoeff,
                    dqcoeff,
                    &dequant_pair,
                    &scan,
                ),
                _ => vp9_quantize_fp_with_order(
                    coefs,
                    n,
                    /* skip_block = */ false,
                    &quantizer.round_fp,
                    &quantizer.quant_fp,
                    qcoeff,
                    dqcoeff,
                    &dequant_pair,
                    &scan,
                ),
            }
        } else {
            match plane_tx {
                TxSize::Tx32x32 => vp9_quantize_b_32x32_with_order(
                    coefs,
                    /* skip_block = */ false,
                    &quantizer.zbin,
                    &quantizer.round,
                    &quantizer.quant,
                    &quantizer.quant_shift,
                    qcoeff,
                    dqcoeff,
                    &dequant_pair,
                    &scan,
                ),
                _ => vp9_quantize_b_with_order(
                    coefs,
                    n,
                    /* skip_block = */ false,
                    &quantizer.zbin,
                    &quantizer.round,
                    &quantizer.quant,
                    &quantizer.quant_shift,
                    qcoeff,
                    dqcoeff,
                    &dequant_pair,
                    &scan,
                ),
            }
        };

        let eob = maybe_trellis(
            coefs, qcoeff, dqcoeff, eob, plane_tx, &scan, quantizer, plane_type, trellis,
        );

        let eob = eob as usize;
        if eob == 0 {
            return operation(&[], &[], 0);
        }

        for (output, &value) in scratch.coefs[..n].iter_mut().zip(dqcoeff.iter()) {
            *output = i32::from(value);
        }
        operation(qcoeff, &scratch.coefs[..n], eob)
    }
}

/// Shared `RVP9_TRELLIS` post-quant step for the inter + intra-in-inter
/// emit paths. Runs `vp9_optimize_b` on the post-quant coefficients
/// (libvpx good-mode runs quantize_b THEN trellis on every block) when
/// `trellis` is `Some`, mutating `qcoeff`/`dqcoeff` in place and
/// returning the (possibly reduced) eob. `None` ⇒ byte-identical
/// baseline. The adjusted coeffs feed BOTH the serialised stream and the
/// recon, keeping encoder recon == decoder.
#[allow(clippy::too_many_arguments)]
fn maybe_trellis(
    coefs: &[i32],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    eob: u16,
    plane_tx: TxSize,
    scan: &rvp9_core::transforms::scans::ScanOrder,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
    plane_type: usize,
    trellis: Option<TrellisCtx<'_>>,
) -> u16 {
    let Some(tc) = trellis else {
        return eob;
    };
    let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
    // Snapshot to count coefficients the trellis changes (positive
    // control via `TRELLIS_CHANGED_COEFFS`).
    let pre_q = qcoeff.to_vec();
    let new_eob = crate::rdo::trellis::optimize_b_inter(
        coefs,
        qcoeff,
        dqcoeff,
        eob,
        plane_tx,
        scan,
        &dequant_pair,
        tc.mb_rdmult,
        tc.costs,
        plane_type,
        tc.ref_type,
        /* ctx0 = */ 0,
    );
    let changed = pre_q
        .iter()
        .zip(qcoeff.iter())
        .filter(|(a, b)| a != b)
        .count() as u64;
    if changed > 0 {
        TRELLIS_CHANGED_COEFFS.fetch_add(changed, std::sync::atomic::Ordering::Relaxed);
    }
    new_eob
}

/// Track-F iter8 — per-block transform-size RD selection for the KF /
/// intra Y plane. libvpx ref: `vp9/encoder/vp9_rdopt.c`
/// `rd_pick_intra_sby_mode` → `super_block_yrd` intra branch →
/// `txfm_rd_in_plane` with the intra `vp9_predict_intra_block` inner
/// loop (the good-quality RD-domain TX-size loop that re-predicts +
/// reconstructs each candidate TX size).
///
/// The intra wrinkle vs the inter path: intra prediction is CAUSAL —
/// each TX block predicts from the *reconstructed* pixels of prior TX
/// blocks WITHIN the leaf (and the committed neighbour recon outside).
/// So a candidate TX size cannot reuse a fixed residual; the whole
/// predict → fdct → quant → idct-add chain must be simulated per
/// candidate, feeding each sub-block's reconstruction into the next
/// sub-block's prediction. We do this on a per-candidate scratch copy of
/// the recon Y plane: the leaf footprint is overwritten by the causal
/// recon while the surrounding committed neighbour pixels (above row,
/// left column, above-right) are read straight from the clone.
///
/// The RD metric mirrors libvpx's `block_rd_txfm` / `dist_block`
/// transform-domain branch: distortion is `vp9_block_error(coeff,
/// dqcoeff) >> shift` summed across the leaf's tiles, and the rate is
/// `cost_coeffs` summed with a leaf-local propagating entropy context
/// (seeded at zero — the absolute above/left offset cancels across
/// candidates for the SIZE decision, same approximation as the inter
/// loop). `tx_type` follows the intra mode's hybrid type at sizes <
/// Tx32x32 (DctDct at Tx32x32), so each candidate uses the same
/// transform the leaf will be coded with.
///
/// Returns the chosen luma `TxSize`. The caller (the KF intra picker)
/// stores it on the leaf and re-codes the Y plane with it; the modes
/// writer signals it under `tx_mode = TxModeSelect`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn choose_intra_luma_tx_size(
    input_plane: &[u8],
    recon_plane: &[u8],
    plane_stride: usize,
    plane_w: usize,
    plane_h: usize,
    plane_px_x: usize,
    plane_px_y: usize,
    leaf_bsize: BlockSize,
    intra_mode: IntraMode,
    have_above_block: bool,
    have_left_block: bool,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
    coef_costs: &rvp9_core::cost::coef_cost::CoefCostTable,
    rdmult: i64,
) -> TxSize {
    use rvp9_core::intra_pred::{
        DcAvailability, EdgeBuffers, build_intra_predictors, intra_predict,
    };
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::scans::scan_for;

    let max_tx = rvp9_core::block::max_txsize(leaf_bsize);
    // Sub-8 leaves only have Tx4x4 — no decision to make.
    if max_tx == TxSize::Tx4x4 {
        return TxSize::Tx4x4;
    }

    let dispatch = rvp9_core::dsp::get();
    let plane_bw_4x4 = leaf_bsize.n_4x4_wide() as usize;
    let plane_bh_4x4 = leaf_bsize.n_4x4_high() as usize;

    // libvpx `RDCOST(RM, DM, R, D) = ((RM * R + 256) >> 9) + (D << 7)`.
    let rdcost = |rate: u32, dist: i64| -> i64 {
        let rate_term = (rdmult.saturating_mul(i64::from(rate)) + 256) >> 9;
        rate_term.saturating_add(dist << 7)
    };
    let mut best_tx = max_tx;
    let mut best_rd = i64::MAX;

    // Reuse one maximum-size transform workspace across every candidate and
    // TX block in this leaf. The previous path allocated one residual Vec per
    // candidate and five coefficient Vecs per block (raw, qcoeff, dqcoeff,
    // widened dqcoeff, then a narrowed distortion copy). A 32x32 leaf visits
    // 85 blocks while walking Tx32x32 down to Tx4x4, so those short-lived
    // allocations dominated keyframe TX selection.
    let mut residual = [0i16; 1024];
    let mut coefs_i32 = [0i32; 1024];
    let mut qcoeff_i16 = [0i16; 1024];
    let mut dqcoeff_i16 = [0i16; 1024];
    let mut dqcoeff_i32 = [0i32; 1024];

    // Per-candidate scratch recon. Intra prediction reads only the leaf plus
    // its one-pixel causal above/left halo; cloning the entire frame for every
    // leaf made keyframe TX selection scale with frame area squared. Keep a
    // compact local plane instead, with the leaf at (1, 1).
    let leaf_w = plane_bw_4x4 * 4;
    let leaf_h = plane_bh_4x4 * 4;
    debug_assert!(plane_px_x + leaf_w <= plane_w);
    debug_assert!(plane_px_y + leaf_h <= plane_h);
    let scratch_stride = leaf_w + 1;
    let scratch_h = leaf_h + 1;
    let mut scratch = vec![0u8; scratch_stride * scratch_h];
    if have_above_block {
        let src_row = (plane_px_y - 1) * plane_stride + plane_px_x;
        scratch[1..1 + leaf_w].copy_from_slice(&recon_plane[src_row..src_row + leaf_w]);
    }
    if have_left_block {
        let src_col = plane_px_x - 1;
        for r in 0..leaf_h {
            scratch[(r + 1) * scratch_stride] =
                recon_plane[(plane_px_y + r) * plane_stride + src_col];
        }
    }
    if have_above_block && have_left_block {
        scratch[0] = recon_plane[(plane_px_y - 1) * plane_stride + plane_px_x - 1];
    }

    for n in (0..=(max_tx as usize)).rev() {
        let tx_size = match n {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            _ => TxSize::Tx32x32,
        };
        let tx_side = tx_size.side();
        let tx_step_4x4 = 1usize << (tx_size as usize);
        // Y plane uses the intra mode's hybrid tx_type at sizes <
        // Tx32x32 (Tx32x32 has no hybrid → DctDct). Matches the
        // simulator's `tx_type` derivation + the decoder's
        // `resolve_tx_type`.
        let tx_type = if tx_size != TxSize::Tx32x32 {
            intra_mode.tx_type()
        } else {
            TxType::DctDct
        };
        let scan = scan_for(tx_size, tx_type);
        let shift = if matches!(tx_size, TxSize::Tx32x32) {
            0
        } else {
            2
        };

        // Refresh the leaf footprint in the scratch buffer from the
        // committed recon so each candidate starts from the same
        // neighbour state (the prior candidate overwrote the footprint).
        for r in 0..leaf_h {
            let src_row = (plane_px_y + r) * plane_stride + plane_px_x;
            let dst_row = (r + 1) * scratch_stride + 1;
            scratch[dst_row..dst_row + leaf_w]
                .copy_from_slice(&recon_plane[src_row..src_row + leaf_w]);
        }

        let mut total_dist: i64 = 0;
        let mut total_rate: u32 = 0;
        // Leaf-local propagating entropy context: above[col4], left[row4].
        let mut above_ctx = [0u8; 16];
        let mut left_ctx = [0u8; 16];

        let mut ty4 = 0usize;
        while ty4 < plane_bh_4x4 {
            let mut tx4 = 0usize;
            while tx4 < plane_bw_4x4 {
                let tx_px_x = 1 + tx4 * 4;
                let tx_px_y = 1 + ty4 * 4;
                let dst_off = tx_px_y * scratch_stride + tx_px_x;

                // Causal intra prediction into the scratch recon — reads
                // already-reconstructed sub-blocks of this leaf (and the
                // committed neighbour recon). Mirrors the simulator's
                // per-TX `build_intra_predictors` + `intra_predict`.
                let have_top = have_above_block || ty4 > 0;
                let have_left = have_left_block || tx4 > 0;
                let have_right = (tx4 + tx_step_4x4) < plane_bw_4x4;
                let mut edges = EdgeBuffers::new();
                build_intra_predictors(
                    &scratch,
                    scratch_stride,
                    tx_px_x,
                    tx_px_y,
                    scratch_stride,
                    scratch_h,
                    tx_size,
                    intra_mode,
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
                intra_predict(
                    intra_mode,
                    &mut scratch[dst_off..],
                    scratch_stride,
                    tx_side,
                    pred_ctx,
                    dc_avail,
                    &dispatch,
                );

                // Residual = source − prediction.
                for r in 0..tx_side {
                    let src_row = (plane_px_y + ty4 * 4 + r) * plane_stride + plane_px_x + tx4 * 4;
                    let pred_row = (tx_px_y + r) * scratch_stride + tx_px_x;
                    for c in 0..tx_side {
                        let s = i16::from(input_plane[src_row + c]);
                        let p = i16::from(scratch[pred_row + c]);
                        residual[r * tx_side + c] = s - p;
                    }
                }

                // Forward hybrid transform (raw coefs needed for the
                // transform-domain distortion `vp9_block_error(coeff,
                // dqcoeff)`) then forward quant.
                let ncoeff = tx_size.n_coeffs();
                intra_forward_coefs_one_tx(
                    &residual[..ncoeff],
                    tx_side,
                    tx_size,
                    tx_type,
                    &mut coefs_i32,
                );
                let eob = intra_quant_one_tx(
                    &coefs_i32,
                    tx_size,
                    tx_type,
                    quantizer,
                    &mut qcoeff_i16,
                    &mut dqcoeff_i16,
                );
                for (dst, &src) in dqcoeff_i32[..ncoeff].iter_mut().zip(&dqcoeff_i16[..ncoeff]) {
                    *dst = i32::from(src);
                }

                // Transform-domain distortion: vp9_block_error >> shift.
                let (err, _ssz) = crate::rdo::block_error::vp9_block_error(
                    &coefs_i32[..ncoeff],
                    &dqcoeff_i16[..ncoeff],
                );
                total_dist += err >> shift;

                // Coefficient rate via cost_coeffs with the leaf-local
                // entropy context (above[col] + left[row], capped to 0..2).
                let above = above_ctx[tx4];
                let left = left_ctx[ty4];
                let coeff_ctx = rvp9_core::coef_ctx::combine_entropy_contexts(above, left);
                total_rate = total_rate.saturating_add(crate::rdo::cost_coeffs::cost_coeffs(
                    &qcoeff_i16[..ncoeff],
                    scan.scan(),
                    eob as u16,
                    tx_size,
                    /* plane_type = */ 0,
                    /* ref_type (intra) = */ 0,
                    coeff_ctx,
                    coef_costs,
                ));

                // Reconstruct this sub-block into the scratch recon so the
                // next sub-block predicts from the true decode (causal).
                if eob > 0 {
                    intra_inverse_transform_add(
                        tx_size,
                        tx_type,
                        &dqcoeff_i32[..ncoeff],
                        eob,
                        &mut scratch[dst_off..],
                        scratch_stride,
                    );
                }

                // Propagate the "had non-zero coef" flag across the tile's
                // 4×4 footprint in the leaf-local above/left arrays.
                let nz = u8::from(eob > 0);
                for c in 0..tx_step_4x4 {
                    if tx4 + c < plane_bw_4x4 {
                        above_ctx[tx4 + c] = nz;
                    }
                }
                for r in 0..tx_step_4x4 {
                    if ty4 + r < plane_bh_4x4 {
                        left_ctx[ty4 + r] = nz;
                    }
                }

                tx4 += tx_step_4x4;
            }
            ty4 += tx_step_4x4;
        }

        let rd = rdcost(total_rate, total_dist);
        // Strict `<` keeps the larger TX on ties (libvpx walks high→low
        // and keeps the first/biggest on equality).
        if rd < best_rd {
            best_rd = rd;
            best_tx = tx_size;
        }
    }

    best_tx
}

/// Track-F iter8 helper — forward hybrid transform of one intra TX block
/// writing the raw transform coefficients (pre-quant) into caller scratch. Used by
/// [`choose_intra_luma_tx_size`] for the `vp9_block_error(coeff,
/// dqcoeff)` transform-domain distortion (which needs the un-quantized
/// `coeff`, not the qcoeff). Mirrors the forward-transform half of
/// [`intra_forward_quant_one_tx`].
#[cfg(test)]
fn intra_forward_coefs_one_tx(
    residual: &[i16],
    tx_side: usize,
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    coefs: &mut [i32; 1024],
) {
    use rvp9_core::transforms::forward::{fdct32x32, fht4x4, fht8x8, fht16x16};
    match plane_tx {
        TxSize::Tx4x4 => {
            let mut buf = [0i32; 16];
            fht4x4(residual, &mut buf, tx_side, tx_type);
            coefs[..16].copy_from_slice(&buf);
        }
        TxSize::Tx8x8 => {
            let mut buf = [0i32; 64];
            fht8x8(residual, &mut buf, tx_side, tx_type);
            coefs[..64].copy_from_slice(&buf);
        }
        TxSize::Tx16x16 => {
            let mut buf = [0i32; 256];
            fht16x16(residual, &mut buf, tx_side, tx_type);
            coefs[..256].copy_from_slice(&buf);
        }
        TxSize::Tx32x32 => {
            fdct32x32(residual, coefs, tx_side);
        }
    }
}

/// Track-F iter8 helper — forward quant of raw transform coefficients for
/// one intra TX block. Writes qcoeff/dqcoeff into caller scratch and returns
/// the EOB. Mirrors
/// the quant half of [`intra_forward_quant_one_tx`] but takes the raw
/// `coefs` (already transformed by [`intra_forward_coefs_one_tx`]) so the
/// RD loop transforms exactly once per sub-block.
#[cfg(test)]
fn intra_quant_one_tx(
    coefs: &[i32],
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
    qcoeff_i16: &mut [i16; 1024],
    dqcoeff_i16: &mut [i16; 1024],
) -> usize {
    use rvp9_core::quant::forward::{vp9_quantize_b_32x32_with_order, vp9_quantize_b_with_order};
    use rvp9_core::transforms::scans::scan_for;
    let n = plane_tx.n_coeffs();
    let scan = scan_for(plane_tx, tx_type);
    let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
    let eob = match plane_tx {
        TxSize::Tx32x32 => vp9_quantize_b_32x32_with_order(
            &coefs[..n],
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            &mut qcoeff_i16[..n],
            &mut dqcoeff_i16[..n],
            &dequant_pair,
            &scan,
        ),
        _ => vp9_quantize_b_with_order(
            &coefs[..n],
            n,
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            &mut qcoeff_i16[..n],
            &mut dqcoeff_i16[..n],
            &dequant_pair,
            &scan,
        ),
    };
    eob as usize
}

/// Track-F iter7 — per-block transform-size RD selection for the inter Y
/// plane. libvpx ref: `vp9/encoder/vp9_rdopt.c` `super_block_yrd` /
/// `choose_tx_size_from_rd` / `txfm_rd_in_plane` (the good-quality
/// RD-domain TX-size loop).
///
/// For an INTER leaf the motion-compensated prediction (already written
/// into `recon_plane`) is TX-size-independent, so the residual `src -
/// pred` is identical for every candidate TX size. This lets us walk all
/// legal TX sizes — `Tx4x4` up to `max_txsize(leaf_bsize)` — compute the
/// per-size RD cost (transform-domain quant distortion + coefficient
/// rate + the per-size tx_size signalling bit), and return the size
/// minimising `D + lambda * R`.
///
/// The RD metric mirrors libvpx's `block_rd_txfm` / `dist_block`
/// transform-domain branch: distortion is `vp9_block_error(coeff,
/// dqcoeff) >> shift` summed across the leaf's tiles, and the rate is
/// `cost_coeffs` summed across tiles with a leaf-local propagating
/// entropy context (the absolute above/left offset from the persistent
/// `TileCoefCtx` is small relative to the distortion-driven difference
/// between sizes and cancels across candidates, so a leaf-local context
/// seeded at zero is a faithful approximation for the SIZE decision).
///
/// Returns the chosen luma `TxSize`. The caller then codes the leaf with
/// it and the writer signals it (under `tx_mode = TxModeSelect`).
struct InterLumaTxInput<'a> {
    input_plane: &'a [u8],
    recon_plane: &'a [u8],
    plane_stride: usize,
    plane_px_x: usize,
    plane_px_y: usize,
    leaf_bsize: BlockSize,
    quantizer: &'a rvp9_core::quant::forward::PlaneQuantizers,
    coef_costs: &'a rvp9_core::cost::coef_cost::CoefCostTable,
    rdmult: i64,
    tx_size_cost: [u32; 4],
    quant_fp: bool,
    scratch: &'a mut InterEmitScratch,
}

fn realtime_inter_tx_candidate(max_tx: TxSize, candidate: TxSize) -> bool {
    max_tx == TxSize::Tx16x16 && matches!(candidate, TxSize::Tx8x8 | TxSize::Tx16x16)
}

fn should_redecide_inter_luma_tx(force_max_tx: bool, tx_rd: bool, bsize: BlockSize) -> bool {
    !force_max_tx && tx_rd && rvp9_core::block::max_txsize(bsize) == TxSize::Tx16x16
}

fn choose_inter_luma_tx_size(input: InterLumaTxInput<'_>) -> TxSize {
    let max_tx = rvp9_core::block::max_txsize(input.leaf_bsize);
    if max_tx != TxSize::Tx16x16 {
        return max_tx;
    }
    choose_inter_luma_tx_size_impl::<true>(input, true)
}

fn choose_inter_luma_tx_size_impl<const PRUNE: bool>(
    input: InterLumaTxInput<'_>,
    realtime_candidate_subset: bool,
) -> TxSize {
    use rvp9_core::quant::forward::{
        vp9_quantize_b_32x32_with_order, vp9_quantize_b_with_order,
        vp9_quantize_fp_32x32_with_order, vp9_quantize_fp_with_order,
    };
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::forward::{fdct8x8_libvpx_lowbd, fdct16x16, fdct32x32, fht4x4};
    use rvp9_core::transforms::scans::scan_for;

    let InterLumaTxInput {
        input_plane,
        recon_plane,
        plane_stride,
        plane_px_x,
        plane_px_y,
        leaf_bsize,
        quantizer,
        coef_costs,
        rdmult,
        tx_size_cost,
        quant_fp,
        scratch,
    } = input;

    let max_tx = rvp9_core::block::max_txsize(leaf_bsize);
    // Sub-8 leaves only have Tx4x4 — no decision to make.
    if max_tx == TxSize::Tx4x4 {
        return TxSize::Tx4x4;
    }
    let plane_bw_4x4 = leaf_bsize.n_4x4_wide() as usize;
    let plane_bh_4x4 = leaf_bsize.n_4x4_high() as usize;
    let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
    let block_w_px = plane_bw_4x4 * 4;
    let block_h_px = plane_bh_4x4 * 4;
    let InterEmitScratch { residual, quant } = scratch;
    let leaf_residual = &mut residual[..block_w_px * block_h_px];
    #[cfg(target_arch = "aarch64")]
    if block_w_px == block_h_px {
        let pixel_offset = plane_px_y * plane_stride + plane_px_x;
        rvp9_core::dsp::aarch64::subtract_square_neon(
            &input_plane[pixel_offset..],
            plane_stride,
            &recon_plane[pixel_offset..],
            plane_stride,
            leaf_residual,
            block_w_px,
        );
    } else {
        for r in 0..block_h_px {
            let src_row = (plane_px_y + r) * plane_stride + plane_px_x;
            let dst_row = r * block_w_px;
            for c in 0..block_w_px {
                let s = i16::from(input_plane[src_row + c]);
                let p = i16::from(recon_plane[src_row + c]);
                leaf_residual[dst_row + c] = s - p;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for r in 0..block_h_px {
        let src_row = (plane_px_y + r) * plane_stride + plane_px_x;
        let dst_row = r * block_w_px;
        for c in 0..block_w_px {
            let s = i16::from(input_plane[src_row + c]);
            let p = i16::from(recon_plane[src_row + c]);
            leaf_residual[dst_row + c] = s - p;
        }
    }

    // libvpx `RDCOST(RM, DM, R, D) = ((RM * R + 256) >> 9) + (D << 7)`.
    let rdcost = |rate: u32, dist: i64| -> i64 {
        let rate_term = (rdmult.saturating_mul(i64::from(rate)) + 256) >> 9;
        rate_term.saturating_add(dist << 7)
    };
    let mut best_tx = max_tx;
    let mut best_rd = i64::MAX;

    // Walk candidate sizes from max down to Tx4x4. Exhaustive diagnostics
    // retain the full libvpx good-quality depth; the measured realtime policy
    // admits only TX16 and TX8 for the eligible leaf class.
    'candidate_sizes: for n in (0..=(max_tx as usize)).rev() {
        let tx_size = match n {
            0 => TxSize::Tx4x4,
            1 => TxSize::Tx8x8,
            2 => TxSize::Tx16x16,
            _ => TxSize::Tx32x32,
        };
        if realtime_candidate_subset && !realtime_inter_tx_candidate(max_tx, tx_size) {
            continue;
        }
        let tx_step_4x4 = 1usize << (tx_size as usize);
        let ncoef = tx_size.n_coeffs();
        let scan = scan_for(tx_size, TxType::DctDct);
        let shift = if matches!(tx_size, TxSize::Tx32x32) {
            0
        } else {
            2
        };

        let mut total_dist: i64 = 0;
        let mut total_rate: u32 = 0;
        // Leaf-local propagating entropy context: above[col4], left[row4].
        let mut above_ctx = [0u8; 16];
        let mut left_ctx = [0u8; 16];

        let InterEmitQuantScratch {
            coefs,
            qcoeff,
            dqcoeff,
        } = quant;

        let mut ty4 = 0usize;
        while ty4 < plane_bh_4x4 {
            let mut tx4 = 0usize;
            while tx4 < plane_bw_4x4 {
                let residual_off = (ty4 * 4) * block_w_px + tx4 * 4;
                let residual_tile = &leaf_residual[residual_off..];
                // Forward transform (inter = DctDct).
                match tx_size {
                    TxSize::Tx4x4 => {
                        fht4x4(
                            residual_tile,
                            (&mut coefs[..16])
                                .try_into()
                                .expect("4x4 transform prefix has 16 coefficients"),
                            block_w_px,
                            TxType::DctDct,
                        );
                    }
                    TxSize::Tx8x8 => {
                        fdct8x8_libvpx_lowbd(
                            residual_tile,
                            (&mut coefs[..64])
                                .try_into()
                                .expect("8x8 transform prefix has 64 coefficients"),
                            block_w_px,
                        );
                    }
                    TxSize::Tx16x16 => {
                        fdct16x16(
                            residual_tile,
                            (&mut coefs[..256])
                                .try_into()
                                .expect("16x16 transform prefix has 256 coefficients"),
                            block_w_px,
                        );
                    }
                    TxSize::Tx32x32 => {
                        fdct32x32(residual_tile, coefs, block_w_px);
                    }
                }
                let eob = if quant_fp {
                    match tx_size {
                        TxSize::Tx32x32 => vp9_quantize_fp_32x32_with_order(
                            &coefs[..ncoef],
                            false,
                            &quantizer.round_fp,
                            &quantizer.quant_fp,
                            &mut qcoeff[..ncoef],
                            &mut dqcoeff[..ncoef],
                            &dequant_pair,
                            &scan,
                        ),
                        _ => vp9_quantize_fp_with_order(
                            &coefs[..ncoef],
                            ncoef,
                            false,
                            &quantizer.round_fp,
                            &quantizer.quant_fp,
                            &mut qcoeff[..ncoef],
                            &mut dqcoeff[..ncoef],
                            &dequant_pair,
                            &scan,
                        ),
                    }
                } else {
                    match tx_size {
                        TxSize::Tx32x32 => vp9_quantize_b_32x32_with_order(
                            &coefs[..ncoef],
                            false,
                            &quantizer.zbin,
                            &quantizer.round,
                            &quantizer.quant,
                            &quantizer.quant_shift,
                            &mut qcoeff[..ncoef],
                            &mut dqcoeff[..ncoef],
                            &dequant_pair,
                            &scan,
                        ),
                        _ => vp9_quantize_b_with_order(
                            &coefs[..ncoef],
                            ncoef,
                            false,
                            &quantizer.zbin,
                            &quantizer.round,
                            &quantizer.quant,
                            &quantizer.quant_shift,
                            &mut qcoeff[..ncoef],
                            &mut dqcoeff[..ncoef],
                            &dequant_pair,
                            &scan,
                        ),
                    }
                };
                // Transform-domain distortion: vp9_block_error >> shift.
                let (err, _ssz) =
                    crate::rdo::block_error::vp9_block_error(&coefs[..ncoef], &dqcoeff[..ncoef]);
                total_dist += err >> shift;

                // Coefficient rate via cost_coeffs with the leaf-local
                // entropy context (above[col] + left[row], capped to 0..2).
                let above = above_ctx[tx4];
                let left = left_ctx[ty4];
                let coeff_ctx = rvp9_core::coef_ctx::combine_entropy_contexts(above, left);
                total_rate = total_rate.saturating_add(crate::rdo::cost_coeffs::cost_coeffs(
                    &qcoeff[..ncoef],
                    scan.scan(),
                    eob as u16,
                    tx_size,
                    /* plane_type = */ 0,
                    /* ref_type (inter) = */ 1,
                    coeff_ctx,
                    coef_costs,
                ));
                // Propagate the "had non-zero coef" flag across the tile's
                // 4×4 footprint in the leaf-local above/left arrays.
                let nz = u8::from(eob > 0);
                for c in 0..tx_step_4x4 {
                    if tx4 + c < plane_bw_4x4 {
                        above_ctx[tx4 + c] = nz;
                    }
                }
                for r in 0..tx_step_4x4 {
                    if ty4 + r < plane_bh_4x4 {
                        left_ctx[ty4 + r] = nz;
                    }
                }

                // All remaining tile-rate and distortion contributions are
                // non-negative. Once this candidate's partial lower bound
                // cannot beat the best complete size, its unevaluated tiles
                // cannot change the strict argmin.
                if PRUNE && best_rd != i64::MAX {
                    let partial_rate = total_rate.saturating_add(tx_size_cost[tx_size as usize]);
                    if rdcost(partial_rate, total_dist) >= best_rd {
                        continue 'candidate_sizes;
                    }
                }

                tx4 += tx_step_4x4;
            }
            ty4 += tx_step_4x4;
        }

        // Charge the per-candidate tx-size SIGNALING cost, exactly as
        // libvpx `choose_tx_size_from_rd` does (`r1 = r0 + r_tx_size`,
        // `vp9_rdopt.c:957-1003`; rvp9's faithful port at
        // `rdo/block_yrd.rs:214-228`). Without this term the smaller
        // transforms look artificially cheap because the tx_probs tree
        // charges more bits to code "the smaller size" than the max —
        // omitting it biased the emit walk toward TX_4x4. `tx_size_cost`
        // is the flat (max=TX_32X32, ctx=0) row already built once per
        // frame for the RDO picker (`rdo_tx_size_cost`); ctx=0 is a
        // documented approximation — the *relative* cost across sizes is
        // what drives the argmin.
        let total_rate = total_rate.saturating_add(tx_size_cost[tx_size as usize]);

        let rd = rdcost(total_rate, total_dist);
        // Strict `<` keeps the larger TX on ties (libvpx walks high→low
        // and keeps the first/biggest on equality — matches the prior
        // always-max behaviour for residual-free leaves).
        if rd < best_rd {
            best_rd = rd;
            best_tx = tx_size;
        }
    }

    best_tx
}

/// DRIFT-FIX — inverse-transform-add that is BYTE-IDENTICAL to the
/// decoder's `inverse_transform_add` (`rvp9-decoder::reconstruct.rs`).
///
/// The encoder's stored reconstruction MUST equal the true decode of its
/// own bitstream, otherwise the next inter frame predicts from a
/// reference that drifts away from what the decoder reconstructs — the
/// SEVERITY-1 inter-frame drift. The drift originated here: the encoder
/// used to call the *scalar* full kernels (`idct16x16_256_add`, …)
/// unconditionally, while the decoder routes every TX block through
/// [`SimdDispatch`] with the SAME eob-bucketed fast paths it picks at
/// decode time (`idct16x16_10_add` / `_38_add` / dense SSE2 / etc.).
/// Those SIMD/eob-bucketed kernels are NOT bit-exact to the scalar full
/// kernel (e.g. `idct16x16_38_add` differs by ±1 per pixel), so any TX
/// block whose eob lands in a different bucket reconstructs ±1 in the
/// encoder vs the decoder. That ±1 seeds an intra-prediction cascade on
/// the keyframe and compounds across every inter frame.
///
/// This helper replicates the decoder's dispatch exactly:
///   * same `tx_type`-keyed DCT-vs-IHT split,
///   * same eob buckets (`==1` DC fast path, `≤10/≤38` 16×16 sparse,
///     `≤12` 8×8 sparse, `≤34/≤135` 32×32 sparse),
///   * same [`SimdDispatch`] slots (so the encoder picks the AVX2/SSE2
///     kernel the decoder will pick on the same host).
///
/// Non-lossless 8-bit only (the encoder never emits lossless here).
/// Caller must ensure `eob > 0`.
fn enc_inverse_transform_add(
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    dq_input: &[i32],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    use rvp9_core::transforms::TxType;
    use rvp9_core::transforms::inverse::{
        idct4x4_1_add, idct8x8_1_add, idct16x16_1_add, idct32x32_1_add,
    };

    debug_assert!(eob > 0, "enc_inverse_transform_add called with eob == 0");
    let dispatch = rvp9_core::dsp::get();

    match plane_tx {
        TxSize::Tx4x4 => {
            let coefs: &[i32; 16] = dq_input.try_into().expect("Tx4x4 takes 16 coefs");
            if tx_type == TxType::DctDct {
                if eob > 1 {
                    (dispatch.idct4x4_add())(coefs, dst, stride);
                } else {
                    idct4x4_1_add(coefs[0], dst, stride);
                }
            } else {
                (dispatch.iht4x4_add())(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx8x8 => {
            let coefs: &[i32; 64] = dq_input.try_into().expect("Tx8x8 takes 64 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct8x8_1_add(coefs[0], dst, stride);
                } else if eob <= 12 {
                    (dispatch.idct8x8_12_add())(coefs, dst, stride);
                } else {
                    (dispatch.idct8x8_add())(coefs, dst, stride);
                }
            } else {
                (dispatch.iht8x8_add())(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx16x16 => {
            let coefs: &[i32; 256] = dq_input.try_into().expect("Tx16x16 takes 256 coefs");
            if tx_type == TxType::DctDct {
                if eob == 1 {
                    idct16x16_1_add(coefs[0], dst, stride);
                } else if eob <= 10 {
                    (dispatch.idct16x16_10_add())(coefs, dst, stride);
                } else if eob <= 38 {
                    (dispatch.idct16x16_38_add())(coefs, dst, stride);
                } else {
                    (dispatch.idct16x16_add())(coefs, dst, stride);
                }
            } else {
                (dispatch.iht16x16_add())(coefs, dst, stride, tx_type);
            }
        }
        TxSize::Tx32x32 => {
            // No 32×32 hybrid exists; always pure IDCT.
            let coefs: &[i32; 1024] = dq_input.try_into().expect("Tx32x32 takes 1024 coefs");
            if eob == 1 {
                idct32x32_1_add(coefs[0], dst, stride);
            } else if eob <= 34 {
                (dispatch.idct32x32_34_add())(coefs, dst, stride);
            } else if eob <= 135 {
                (dispatch.idct32x32_135_add())(coefs, dst, stride);
            } else {
                (dispatch.idct32x32_add())(coefs, dst, stride);
            }
        }
    }
}

/// Inverse-DCT-add for inter (DctDct only). Delegates to
/// [`enc_inverse_transform_add`] so the inter committed reconstruction is
/// byte-identical to the decoder. `eob` is the number of decoded scan
/// positions (caller has already filtered `eob > 0`).
fn inter_inverse_dct_add(
    plane_tx: TxSize,
    dq_input: &[i32],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    enc_inverse_transform_add(
        plane_tx,
        rvp9_core::transforms::TxType::DctDct,
        dq_input,
        eob,
        dst,
        stride,
    );
}

/// G5 — Forward hybrid transform + forward quant of one TX block for the
/// intra path (both keyframe and intra-in-inter). Unlike the inter helper
/// (`inter_forward_quant_one_tx`), this supports non-DctDct `tx_type`
/// (AdstDct, DctAdst, AdstAdst) via the `fht*` family, and derives
/// scan order from `(plane_tx, tx_type)` rather than hardcoding DctDct.
///
/// The caller supplies tile-worker scratch and consumes the active qcoeff and
/// widened dqcoeff prefixes before that scratch is reused.
#[allow(clippy::too_many_arguments)]
fn intra_forward_quant_one_tx_with_scratch<R>(
    scratch: &mut InterEmitQuantScratch,
    residual: &[i16],
    tx_side: usize,
    plane_tx: TxSize,
    tx_type: rvp9_core::transforms::TxType,
    quantizer: &rvp9_core::quant::forward::PlaneQuantizers,
    plane_type: usize,
    trellis: Option<TrellisCtx<'_>>,
    operation: impl FnOnce(&[i16], &[i32], usize) -> R,
) -> R {
    use rvp9_core::quant::forward::{vp9_quantize_b_32x32_with_order, vp9_quantize_b_with_order};
    use rvp9_core::transforms::forward::{fdct32x32, fht4x4, fht8x8, fht16x16};
    use rvp9_core::transforms::scans::scan_for;

    let n = plane_tx.n_coeffs();
    match plane_tx {
        TxSize::Tx4x4 => {
            let output: &mut [i32; 16] = (&mut scratch.coefs[..n])
                .try_into()
                .expect("Tx4x4 coefficient prefix has 16 entries");
            fht4x4(residual, output, tx_side, tx_type);
        }
        TxSize::Tx8x8 => {
            let output: &mut [i32; 64] = (&mut scratch.coefs[..n])
                .try_into()
                .expect("Tx8x8 coefficient prefix has 64 entries");
            fht8x8(residual, output, tx_side, tx_type);
        }
        TxSize::Tx16x16 => {
            let output: &mut [i32; 256] = (&mut scratch.coefs[..n])
                .try_into()
                .expect("Tx16x16 coefficient prefix has 256 entries");
            fht16x16(residual, output, tx_side, tx_type);
        }
        TxSize::Tx32x32 => {
            // VP9 has no hybrid 32x32 -- always pure DCT.
            fdct32x32(residual, &mut scratch.coefs, tx_side);
        }
    }

    let scan = scan_for(plane_tx, tx_type);
    let dequant_pair = [quantizer.dequant[0], quantizer.dequant[1]];
    let coefs = &scratch.coefs[..n];
    let qcoeff = &mut scratch.qcoeff[..n];
    let dqcoeff = &mut scratch.dqcoeff[..n];
    let eob = match plane_tx {
        TxSize::Tx32x32 => vp9_quantize_b_32x32_with_order(
            coefs,
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            qcoeff,
            dqcoeff,
            &dequant_pair,
            &scan,
        ),
        _ => vp9_quantize_b_with_order(
            coefs,
            n,
            /* skip_block = */ false,
            &quantizer.zbin,
            &quantizer.round,
            &quantizer.quant,
            &quantizer.quant_shift,
            qcoeff,
            dqcoeff,
            &dequant_pair,
            &scan,
        ),
    };

    let eob = maybe_trellis(
        coefs, qcoeff, dqcoeff, eob, plane_tx, &scan, quantizer, plane_type, trellis,
    );

    let eob = eob as usize;
    if eob == 0 {
        return operation(&[], &[], 0);
    }
    for (output, &value) in scratch.coefs[..n].iter_mut().zip(dqcoeff.iter()) {
        *output = i32::from(value);
    }
    operation(qcoeff, &scratch.coefs[..n], eob)
}

/// G5 — Inverse transform-add for intra paths (both keyframe and
/// intra-in-inter). Delegates to [`enc_inverse_transform_add`] so the
/// intra committed reconstruction is byte-identical to the decoder's
/// eob-bucketed, SIMD-dispatched `inverse_transform_add` (incl. the
/// `tx_type`-keyed IHT split for non-DctDct hybrid blocks).
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
    enc_inverse_transform_add(plane_tx, tx_type, dq_input, eob, dst, stride);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inter_row_jobs_partition_unequal_tiles_into_disjoint_sb_ranges() {
        let bounds = column_tile_bounds(16, 30 * 8, 2);
        let mut observed = Vec::new();
        for tile_col in 0..bounds.len() {
            observed.extend(inter_row_job_sb_columns(
                &bounds,
                None,
                crate::row_mt::RowJob {
                    sb_row: 1,
                    tile_col,
                },
            ));
        }
        assert_eq!(observed, (0..30).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "inter row job does not belong to the active tile")]
    fn inter_row_job_rejects_another_active_tile() {
        let bounds = column_tile_bounds(16, 20 * 8, 2);
        let _ = inter_row_job_sb_columns(
            &bounds,
            Some(2),
            crate::row_mt::RowJob {
                sb_row: 0,
                tile_col: 1,
            },
        );
    }

    #[test]
    fn inter_row_grids_commit_mode_and_mv_cells_together() {
        let mut out_mv_grid = vec![PrevFrameMvRef::default(); 4 * 8];
        let layout = InterPickerMiGridLayout::new(2, 4);
        let leaf = InterPartitionLeafDesc {
            mi_row_in_sb: 0,
            mi_col_in_sb: 0,
            leaf_bsize: BlockSize::Block16x16,
            picked_mode: InterMode::NewMv,
            picked_mv: Mv { row: 8, col: -16 },
            skip: false,
            segment_id: 3,
            picked_ref_frame: RefFrame::GoldenFrame,
            picked_interp_filter: InterpFilter::EightTapSmooth,
            picked_intra_mode: None,
            picked_ref_frame2: RefFrame::NoRefFrame,
            picked_mv2: Mv::ZERO,
            picked_tx_size: TxSize::Tx8x8,
        };
        let expected_mode = leaf.mode_info();
        let expected_mv = leaf.mv_ref();

        let mut grids = InterRowGrids::new(4, layout, &mut out_mv_grid, 8);
        grids.splat_final(1, 2, leaf);
        for row in 1..3 {
            for col in 2..4 {
                assert_eq!(grids.get(row, col), Some(&expected_mode));
            }
        }
        assert_eq!(grids.get(0, 2), None);
        let simulation_grid = grids.into_simulation_grid();

        assert_eq!(simulation_grid.mi_col_start, 2);
        assert_eq!(simulation_grid.mi_cols, 4);
        for row in 1..3 {
            for col in 2..4 {
                assert_eq!(out_mv_grid[row * 8 + col], expected_mv);
            }
        }
        assert_eq!(out_mv_grid[0], PrevFrameMvRef::default());
    }

    #[test]
    fn duplicate_golden_shortcut_requires_topology_refresh_and_all_plane_aliases() {
        let y = [1u8; 64];
        let u = [2u8; 16];
        let v = [3u8; 16];
        let shared = (&y[..], &u[..], &v[..]);

        assert!(golden_ref_duplicates_last(true, 0, shared, shared, shared));
        assert!(!golden_ref_duplicates_last(
            false, 0, shared, shared, shared
        ));
        assert!(!golden_ref_duplicates_last(true, 1, shared, shared, shared));

        let other_y = [1u8; 64];
        let other_u = [2u8; 16];
        let other_v = [3u8; 16];
        let distinct = (&other_y[..], &other_u[..], &other_v[..]);
        assert!(!golden_ref_duplicates_last(
            true, 0, distinct, shared, shared
        ));
        assert!(!golden_ref_duplicates_last(
            true, 0, shared, shared, distinct
        ));
    }

    #[test]
    fn unavailable_reference_skips_mv_candidate_builder() {
        let calls = std::cell::Cell::new(0);
        let unavailable = build_mv_candidates_if_available(false, || {
            calls.set(calls.get() + 1);
            rvp9_core::mode_decision::inter::MvCandidates {
                nearest: Mv { row: 8, col: 16 },
                near: Mv { row: 24, col: 32 },
            }
        });
        assert_eq!(
            unavailable,
            rvp9_core::mode_decision::inter::MvCandidates::default()
        );
        assert_eq!(calls.get(), 0);

        let available = build_mv_candidates_if_available(true, || {
            calls.set(calls.get() + 1);
            rvp9_core::mode_decision::inter::MvCandidates {
                nearest: Mv { row: 8, col: 16 },
                near: Mv { row: 24, col: 32 },
            }
        });
        assert_eq!(available.nearest, Mv { row: 8, col: 16 });
        assert_eq!(available.near, Mv { row: 24, col: 32 });
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn zero_motion_fallback_copies_only_the_leaf_footprint() {
        const W: usize = 128;
        const H: usize = 64;
        const SENTINEL: u8 = 0xee;
        let y: Vec<u8> = (0..W * H).map(|index| index as u8).collect();
        let u: Vec<u8> = (0..W * H / 4)
            .map(|index| (index as u8).wrapping_mul(3))
            .collect();
        let v: Vec<u8> = (0..W * H / 4)
            .map(|index| (index as u8).wrapping_mul(5))
            .collect();
        let mut recon_y = vec![SENTINEL; y.len()];
        let mut recon_u = vec![SENTINEL; u.len()];
        let mut recon_v = vec![SENTINEL; v.len()];

        copy_last_ref_leaf_to_recon(
            W,
            H,
            2,
            6,
            BlockSize::Block32x16,
            (&y, &u, &v),
            (&mut recon_y, &mut recon_u, &mut recon_v),
        );

        for (plane, (src, recon)) in [(&y, &recon_y), (&u, &recon_u), (&v, &recon_v)]
            .into_iter()
            .enumerate()
        {
            let ss = usize::from(plane != 0);
            let stride = W >> ss;
            let plane_height = H >> ss;
            let x = 48 >> ss;
            let y = 16 >> ss;
            let block_width = 32 >> ss;
            let block_height = 16 >> ss;
            for row in 0..plane_height {
                for col in 0..stride {
                    let index = row * stride + col;
                    if (y..y + block_height).contains(&row) && (x..x + block_width).contains(&col) {
                        assert_eq!(recon[index], src[index]);
                    } else {
                        assert_eq!(recon[index], SENTINEL);
                    }
                }
            }
        }
    }

    #[test]
    fn standalone_tx8_extends_only_the_unavailable_top_right() {
        let mut above = [
            181, 181, 181, 181, 184, 186, 188, 189, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        extend_standalone_tx8_above(&mut above);
        assert_eq!(
            above,
            [
                181, 181, 181, 181, 184, 186, 188, 189, 189, 189, 189, 189, 189, 189, 189, 189,
            ]
        );
    }

    #[test]
    fn intra_tx_size_pick_ignores_pixels_outside_causal_leaf_halo() {
        use rvp9_core::cost::{FrameTypeForRd, coef_cost::build_coef_cost_table};
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

        const W: usize = 128;
        const H: usize = 128;
        const X: usize = 64;
        const Y: usize = 64;
        const SIDE: usize = 32;

        let input_a: Vec<u8> = (0..W * H)
            .map(|i| ((i * 29 + i / W * 17 + 31) & 0xff) as u8)
            .collect();
        let mut input_b: Vec<u8> = (0..W * H)
            .map(|i| ((i * 7 + i / W * 43 + 193) & 0xff) as u8)
            .collect();
        let recon_a: Vec<u8> = (0..W * H)
            .map(|i| ((i * 11 + i / W * 23 + 71) & 0xff) as u8)
            .collect();
        let mut recon_b: Vec<u8> = (0..W * H)
            .map(|i| ((i * 47 + i / W * 5 + 13) & 0xff) as u8)
            .collect();

        // The source footprint and the reconstruction's leaf plus causal
        // above/left halo are identical. Every other pixel deliberately
        // differs and must not affect this leaf's transform-size decision.
        for r in 0..SIDE {
            let row = (Y + r) * W + X;
            input_b[row..row + SIDE].copy_from_slice(&input_a[row..row + SIDE]);
            recon_b[row..row + SIDE].copy_from_slice(&recon_a[row..row + SIDE]);
            recon_b[(Y + r) * W + X - 1] = recon_a[(Y + r) * W + X - 1];
        }
        let above = (Y - 1) * W + X - 1;
        recon_b[above..above + SIDE + 1].copy_from_slice(&recon_a[above..above + SIDE + 1]);
        assert_ne!(input_a[0], input_b[0]);
        assert_ne!(recon_a[0], recon_b[0]);

        let q_index = 80;
        let quantizer = init_plane_quantizers(q_index, QuantizerDeltas::Y);
        let coef_costs = build_coef_cost_table(&FrameContext::new_default());
        let rdmult = i64::from(rvp9_core::cost::compute_rd_mult(
            q_index,
            FrameTypeForRd::Key,
        ));
        let pick = |input: &[u8], recon: &[u8]| {
            choose_intra_luma_tx_size(
                input,
                recon,
                W,
                W,
                H,
                X,
                Y,
                BlockSize::Block32x32,
                IntraMode::V,
                true,
                true,
                &quantizer,
                &coef_costs,
                rdmult,
            )
        };

        assert_eq!(pick(&input_a, &recon_a), pick(&input_b, &recon_b));
    }

    #[test]
    fn intra_tx_rd_scratch_is_bounded_and_independent_of_prior_tx_size() {
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};
        use rvp9_core::transforms::TxType;

        let residual: Vec<i16> = (0..1024)
            .map(|index| ((index * 37 + index / 32 * 19) % 511) as i16 - 255)
            .collect();
        let quantizer = init_plane_quantizers(80, QuantizerDeltas::Y);
        let cases = [
            (TxSize::Tx4x4, TxType::AdstAdst),
            (TxSize::Tx8x8, TxType::DctAdst),
            (TxSize::Tx16x16, TxType::AdstDct),
            (TxSize::Tx32x32, TxType::DctDct),
        ];

        let run = |tx_size: TxSize, tx_type: TxType, poison: i16| {
            let ncoeff = tx_size.n_coeffs();
            let mut coefs = [i32::from(poison); 1024];
            let mut qcoeff = [poison; 1024];
            let mut dqcoeff = [poison; 1024];
            intra_forward_coefs_one_tx(
                &residual[..ncoeff],
                tx_size.side(),
                tx_size,
                tx_type,
                &mut coefs,
            );
            let eob = intra_quant_one_tx(
                &coefs,
                tx_size,
                tx_type,
                &quantizer,
                &mut qcoeff,
                &mut dqcoeff,
            );
            assert!(
                coefs[ncoeff..]
                    .iter()
                    .all(|&value| value == i32::from(poison))
            );
            assert!(qcoeff[ncoeff..].iter().all(|&value| value == poison));
            assert!(dqcoeff[ncoeff..].iter().all(|&value| value == poison));
            (
                coefs[..ncoeff].to_vec(),
                qcoeff[..ncoeff].to_vec(),
                dqcoeff[..ncoeff].to_vec(),
                eob,
            )
        };

        for (tx_size, tx_type) in cases {
            assert_eq!(
                run(tx_size, tx_type, 11),
                run(tx_size, tx_type, -17),
                "active TX output must not depend on a previous block's scratch contents"
            );
        }
    }

    #[test]
    fn descent_pick_cache_covers_complete_superblock_key_space() {
        let block_sizes = [
            BlockSize::Block4x4,
            BlockSize::Block4x8,
            BlockSize::Block8x4,
            BlockSize::Block8x8,
            BlockSize::Block8x16,
            BlockSize::Block16x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block32x64,
            BlockSize::Block64x32,
            BlockSize::Block64x64,
        ];
        let mut keys = Vec::new();
        for bsize in block_sizes {
            for row in 0..8 {
                for col in 0..8 {
                    keys.push((row, col, bsize));
                }
            }
        }
        assert_eq!(keys.len(), DESCENT_CACHE_SLOTS);

        let mut cache = DescentPickCache::new();
        for (value, &(row, col, bsize)) in keys.iter().enumerate() {
            cache.insert(16 + row, 24 + col, bsize, value);
        }
        for (expected, &(row, col, bsize)) in keys.iter().enumerate() {
            assert_eq!(cache.get(16 + row, 24 + col, bsize), Some(&expected));
        }

        cache.insert(16, 24, BlockSize::Block64x64, usize::MAX);
        assert_eq!(cache.values.len(), DESCENT_CACHE_SLOTS);
        assert_eq!(
            cache.get(16, 24, BlockSize::Block64x64),
            Some(&usize::MAX),
            "repeated keys preserve HashMap replacement semantics"
        );

        let mut next_superblock = DescentPickCache::new();
        next_superblock.insert(24, 32, BlockSize::Block64x64, 7usize);
        assert_eq!(
            next_superblock.get(24, 32, BlockSize::Block64x64),
            Some(&7),
            "local-slot aliases are isolated by per-superblock cache scope"
        );
    }

    #[test]
    fn inter_emit_residual_scratch_is_prefix_bounded_and_causal() {
        let mut scratch = InterEmitScratch::new();
        let run_small = |scratch: &mut InterEmitScratch| {
            scratch.with_len(8 * 8, |residual, _| {
                assert_eq!(residual.len(), 8 * 8);
                for (index, value) in residual.iter_mut().enumerate() {
                    *value = index as i16 - 31;
                }
                residual.to_vec()
            })
        };

        scratch.with_len(32 * 32, |residual, _| residual.fill(i16::MAX));
        let after_positive = run_small(&mut scratch);
        scratch.with_len(32 * 32, |residual, _| residual.fill(i16::MIN));
        let after_negative = run_small(&mut scratch);

        assert_eq!(
            after_negative, after_positive,
            "active residual prefix must be independent of wider prior transforms"
        );
    }

    #[test]
    fn inter_emit_quant_scratch_is_independent_of_prior_transform_size() {
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

        let quantizer = init_plane_quantizers(117, QuantizerDeltas::Y);
        let mut scratch = InterEmitScratch::new();
        let run = |scratch: &mut InterEmitScratch, tx_size: TxSize, salt: i16| {
            let side = tx_size.side();
            let residual: Vec<i16> = (0..side * side)
                .map(|index| ((index as i16 * 13 + salt) & 0xff) - 128)
                .collect();
            inter_forward_quant_one_tx_with_scratch(
                &mut scratch.quant,
                &residual,
                side,
                tx_size,
                &quantizer,
                0,
                None,
                true,
                |qcoeff, dqcoeff, eob| (qcoeff.to_vec(), dqcoeff.to_vec(), eob),
            )
        };

        for tx_size in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16] {
            let before = run(&mut scratch, tx_size, 7);
            let _contaminating_full_scratch_transform = run(&mut scratch, TxSize::Tx32x32, 93);
            let after = run(&mut scratch, tx_size, 7);
            assert_eq!(
                after, before,
                "{tx_size:?} result must not depend on the prior scratch contents"
            );
        }
    }

    #[test]
    fn inter_tx_size_pick_is_independent_of_prior_scratch_contents() {
        use rvp9_core::cost::{FrameTypeForRd, coef_cost::build_coef_cost_table};
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

        const SIDE: usize = 64;
        const BLOCK_X: usize = 16;
        const BLOCK_Y: usize = 16;
        let input: Vec<u8> = (0..SIDE * SIDE)
            .map(|index| ((index * 29 + index / SIDE * 17 + 31) & 0xff) as u8)
            .collect();
        let recon: Vec<u8> = (0..SIDE * SIDE)
            .map(|index| ((index * 11 + index / SIDE * 23 + 71) & 0xff) as u8)
            .collect();
        let q_index = 96;
        let quantizer = init_plane_quantizers(q_index, QuantizerDeltas::Y);
        let coef_costs = build_coef_cost_table(&FrameContext::new_default());
        let rdmult = i64::from(rvp9_core::cost::compute_rd_mult(
            q_index,
            FrameTypeForRd::Inter,
        ));
        let mut scratch = InterEmitScratch::new();
        let run = |scratch: &mut InterEmitScratch, poison: i16, quant_fp: bool, prune: bool| {
            scratch.residual.fill(poison);
            scratch.quant.coefs.fill(i32::from(poison));
            scratch.quant.qcoeff.fill(poison);
            scratch.quant.dqcoeff.fill(poison);
            let pick_input = InterLumaTxInput {
                input_plane: &input,
                recon_plane: &recon,
                plane_stride: SIDE,
                plane_px_x: BLOCK_X,
                plane_px_y: BLOCK_Y,
                leaf_bsize: BlockSize::Block32x32,
                quantizer: &quantizer,
                coef_costs: &coef_costs,
                rdmult,
                tx_size_cost: [0, 17, 31, 47],
                quant_fp,
                scratch,
            };
            if prune {
                choose_inter_luma_tx_size_impl::<true>(pick_input, false)
            } else {
                choose_inter_luma_tx_size_impl::<false>(pick_input, false)
            }
        };

        for quant_fp in [false, true] {
            assert_eq!(
                run(&mut scratch, i16::MIN, quant_fp, true),
                run(&mut scratch, i16::MAX, quant_fp, true),
                "TX choice must not depend on prior scratch contents (quant_fp={quant_fp})"
            );
            assert_eq!(
                run(&mut scratch, 0x1234, quant_fp, false),
                run(&mut scratch, -0x1234, quant_fp, true),
                "pruning must preserve the exhaustive TX argmin (quant_fp={quant_fp})"
            );
        }
    }

    #[test]
    fn inter_tx_size_pruning_matches_exhaustive_search() {
        use rvp9_core::cost::{FrameTypeForRd, coef_cost::build_coef_cost_table};
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

        const SIDE: usize = 80;
        let input: Vec<u8> = (0..SIDE * SIDE)
            .map(|index| ((index * 37 + index / SIDE * 19 + 23) & 0xff) as u8)
            .collect();
        let recon: Vec<u8> = (0..SIDE * SIDE)
            .map(|index| ((index * 13 + index / SIDE * 31 + 79) & 0xff) as u8)
            .collect();
        let coef_costs = build_coef_cost_table(&FrameContext::new_default());

        for leaf_bsize in [
            BlockSize::Block8x8,
            BlockSize::Block16x16,
            BlockSize::Block16x32,
            BlockSize::Block32x16,
            BlockSize::Block32x32,
            BlockSize::Block64x64,
        ] {
            for q_index in [20, 96, 200] {
                let quantizer = init_plane_quantizers(q_index, QuantizerDeltas::Y);
                let rdmult = i64::from(rvp9_core::cost::compute_rd_mult(
                    q_index,
                    FrameTypeForRd::Inter,
                ));
                for quant_fp in [false, true] {
                    let pick = |scratch: &mut InterEmitScratch, prune: bool| {
                        let input = InterLumaTxInput {
                            input_plane: &input,
                            recon_plane: &recon,
                            plane_stride: SIDE,
                            plane_px_x: 8,
                            plane_px_y: 8,
                            leaf_bsize,
                            quantizer: &quantizer,
                            coef_costs: &coef_costs,
                            rdmult,
                            tx_size_cost: [0, 17, 31, 47],
                            quant_fp,
                            scratch,
                        };
                        if prune {
                            choose_inter_luma_tx_size_impl::<true>(input, false)
                        } else {
                            choose_inter_luma_tx_size_impl::<false>(input, false)
                        }
                    };
                    let exhaustive = pick(&mut InterEmitScratch::new(), false);
                    let pruned = pick(&mut InterEmitScratch::new(), true);
                    assert_eq!(
                        pruned, exhaustive,
                        "pruning changed {leaf_bsize:?} q={q_index} quant_fp={quant_fp}"
                    );
                }
            }
        }
    }

    #[test]
    fn intra_emit_quant_scratch_is_independent_of_prior_contents() {
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};
        use rvp9_core::transforms::TxType;

        let quantizer = init_plane_quantizers(117, QuantizerDeltas::Y);
        let mut scratch = InterEmitScratch::new();
        let run =
            |scratch: &mut InterEmitScratch, tx_size: TxSize, tx_type: TxType, poison: i16| {
                let side = tx_size.side();
                let residual: Vec<i16> = (0..side * side)
                    .map(|index| ((index as i16 * 29 + 17) & 0xff) - 128)
                    .collect();
                scratch.quant.coefs.fill(i32::from(poison));
                scratch.quant.qcoeff.fill(poison);
                scratch.quant.dqcoeff.fill(poison);
                intra_forward_quant_one_tx_with_scratch(
                    &mut scratch.quant,
                    &residual,
                    side,
                    tx_size,
                    tx_type,
                    &quantizer,
                    0,
                    None,
                    |qcoeff, dqcoeff, eob| (qcoeff.to_vec(), dqcoeff.to_vec(), eob),
                )
            };

        for (tx_size, tx_type) in [
            (TxSize::Tx4x4, TxType::AdstAdst),
            (TxSize::Tx8x8, TxType::DctAdst),
            (TxSize::Tx16x16, TxType::AdstDct),
            (TxSize::Tx32x32, TxType::DctDct),
        ] {
            assert_eq!(
                run(&mut scratch, tx_size, tx_type, 11),
                run(&mut scratch, tx_size, tx_type, -17),
                "{tx_size:?}/{tx_type:?} output must overwrite active scratch",
            );
        }
    }

    #[test]
    fn active_ref_costs_use_neighbour_selected_frame_context_bank() {
        let mut fc = FrameContext::new_default();
        // No neighbours selects intra-inter ctx 0 and single-ref ctx 2.
        // These values are a pinned-libvpx q120 oracle row.
        fc.intra_inter_prob[0] = 9;
        fc.single_ref_prob[2] = [238, 142];
        assert_eq!(
            active_single_ref_frame_costs(&fc, None, None),
            [2473, 80, 2422, 2585],
        );
    }

    #[test]
    fn compact_partition_predictor_matches_full_frame_destination() {
        let width = 128usize;
        let height = 128usize;
        let reference: Vec<u8> = (0..width * height)
            .map(|i| ((i * 29 + i / width * 17) & 0xff) as u8)
            .collect();
        let mut full = vec![0u8; width * height];
        let mut compact = [0u8; 64 * 64];
        let mv_col_q4 = 11;
        let mv_row_q4 = -7;

        build_inter_pred_plane_unscaled(
            &reference,
            width,
            width,
            height,
            32,
            32,
            64,
            64,
            mv_col_q4,
            mv_row_q4,
            &mut full,
            width,
            InterpFilter::Bilinear,
            false,
        );
        build_inter_pred_block_unscaled(
            &reference,
            width,
            width,
            height,
            32,
            32,
            64,
            64,
            mv_col_q4,
            mv_row_q4,
            &mut compact,
            64,
            InterpFilter::Bilinear,
        );

        for row in 0..64 {
            assert_eq!(
                &compact[row * 64..row * 64 + 64],
                &full[(32 + row) * width + 32..(32 + row) * width + 96],
            );
        }
    }

    #[test]
    fn config_new_round_trips() {
        let c = Config::new(64, 64, 50);
        assert_eq!(c.width, 64);
        assert_eq!(c.height, 64);
        assert_eq!(c.base_qindex, 50);
    }

    #[test]
    fn causal_grid_splats_intra_winner_as_intra() {
        let mut grid = vec![None; 8 * 8];
        splat_inter_leaf_into_mi_grid(
            &mut grid,
            InterPickerMiGridLayout::new(0, 8),
            0,
            0,
            BlockSize::Block16x16,
            InterMode::ZeroMv,
            Mv::ZERO,
            0,
            false,
            RefFrame::IntraFrame,
            Some(IntraMode::H),
            InterpFilter::EightTap,
        );

        for cell in [grid[0], grid[1], grid[8], grid[9]] {
            assert!(matches!(
                cell.expect("intra winner must cover its causal-grid cells")
                    .mode,
                BlockMode::Intra {
                    y_mode: IntraMode::H,
                    uv_mode: IntraMode::Dc,
                }
            ));
        }
    }

    #[test]
    fn tile_local_causal_grid_matches_full_grid_lookup_and_splat() {
        use rvp9_core::inter::mv_ref::{NeighbourLookup, Position};

        const MI_ROWS: usize = 4;
        const MI_COLS: usize = 64;
        let tile = tile_bounds(MI_ROWS, MI_COLS, 0, 2, 0, 2);
        let local_layout =
            InterPickerMiGridLayout::new(tile.mi_col_start, tile.mi_col_end - tile.mi_col_start);
        let full_layout = InterPickerMiGridLayout::new(0, MI_COLS);
        let mut local_grid = vec![None; MI_ROWS * local_layout.mi_cols];
        let mut full_grid = vec![None; MI_ROWS * MI_COLS];

        for (grid, layout) in [
            (&mut local_grid[..], local_layout),
            (&mut full_grid[..], full_layout),
        ] {
            splat_inter_leaf_into_mi_grid(
                grid,
                layout,
                0,
                tile.mi_col_start,
                BlockSize::Block16x16,
                InterMode::NewMv,
                Mv { row: 8, col: -16 },
                0,
                false,
                RefFrame::LastFrame,
                None,
                InterpFilter::EightTap,
            );
        }

        let local_lookup =
            InterPickerMiGridLookup::new(&local_grid, local_layout, 1, tile.mi_col_start + 1, tile);
        let full_lookup =
            InterPickerMiGridLookup::new(&full_grid, full_layout, 1, tile.mi_col_start + 1, tile);
        for position in [
            Position { row: -1, col: 0 },
            Position { row: 0, col: -1 },
            Position { row: 0, col: 0 },
        ] {
            assert_eq!(local_lookup.get(position), full_lookup.get(position));
        }
        assert_eq!(local_grid.len() * 4, full_grid.len());

        let tile_edge_lookup =
            InterPickerMiGridLookup::new(&local_grid, local_layout, 0, tile.mi_col_start, tile);
        assert!(
            tile_edge_lookup.get(Position { row: 0, col: -1 }).is_none(),
            "tile-local storage must preserve the VP9 tile-edge barrier"
        );
    }

    #[test]
    fn f02_oracle_serializes_intra_winner_as_intra_mode() {
        assert_eq!(
            f02_oracle_mode(RefFrame::IntraFrame, InterMode::ZeroMv, Some(IntraMode::V),),
            crate::f02_oracle::BlockDecisionMode::Intra(IntraMode::V),
        );
        assert_eq!(
            f02_oracle_mode(RefFrame::LastFrame, InterMode::NearMv, None),
            crate::f02_oracle::BlockDecisionMode::Inter(InterMode::NearMv),
        );
    }

    #[test]
    fn visible_full_chroma_pel_mv_is_safe_outside_subpel_halo_limits() {
        let edge_limits = rvp9_core::mv_search::MvLimits::new(6, 20, 6, 20);
        let is_safe = |mv, origin, size| {
            mv_in_strict_limits_or_visible_full_chroma_pel(
                mv,
                edge_limits,
                origin,
                size,
                (256, 256),
            )
        };

        assert!(is_safe(Mv::ZERO, (0, 0), (16, 16)));
        assert!(is_safe(Mv { row: -16, col: -16 }, (16, 16), (16, 16)));
        assert!(!is_safe(Mv { row: -16, col: -16 }, (0, 0), (16, 16)));
        assert!(!is_safe(Mv { row: -8, col: -8 }, (16, 16), (16, 16)));
        assert!(!is_safe(Mv { row: 1, col: 0 }, (0, 0), (16, 16)));
        assert!(!is_safe(Mv::ZERO, (248, 248), (16, 16)));
        assert!(is_safe(Mv { row: 48, col: 48 }, (0, 0), (16, 16)));
    }

    #[test]
    fn unrestricted_mv_limits_match_libvpx_top_left_block32() {
        let limits = libvpx_inter_mv_limits(0, 0, 4, 4, 88, 160);
        assert_eq!(
            limits,
            rvp9_core::mv_search::MvLimits::new(-36, 1284, -36, 708)
        );
    }

    #[test]
    fn p8_default_is_vbr_low_q_and_non_vbr_only() {
        assert!(p8_default_enabled(false, RcMode::Vbr, 200));
        assert!(p8_default_enabled(false, RcMode::Vbr, 180));
        assert!(!p8_default_enabled(false, RcMode::Vbr, 201));
        assert!(p8_default_enabled(false, RcMode::Cbr, 255));
        assert!(p8_default_enabled(false, RcMode::Cq, 255));
        assert!(p8_default_enabled(false, RcMode::Q, 255));
        assert!(!p8_default_enabled(true, RcMode::Vbr, 180));
        assert!(!p8_default_enabled(true, RcMode::Cbr, 180));
    }

    #[test]
    fn p8_qmax_is_vbr_low_q_only() {
        assert!(p8_vbr_qmax_enabled(false, RcMode::Vbr, 180, 180));
        assert!(p8_vbr_qmax_enabled(false, RcMode::Vbr, 170, 180));
        assert!(!p8_vbr_qmax_enabled(false, RcMode::Vbr, 181, 180));
        assert!(!p8_vbr_qmax_enabled(false, RcMode::Cbr, 170, 180));
        assert!(!p8_vbr_qmax_enabled(true, RcMode::Vbr, 170, 180));
    }

    #[test]
    fn block16_distortion_breakout_is_cpu7_cbr_realtime_only() {
        use rvp9_core::mode_decision::speed_features::Deadline;

        let policy = |p8, large, rc, speed, deadline, override_qmul| {
            stage4a_descent_breakout_16_dist_qmul(p8, large, rc, speed, deadline, override_qmul)
        };
        assert_eq!(
            policy(true, false, RcMode::Cbr, 7, Deadline::Realtime, None),
            Some(1600)
        );
        assert_eq!(
            policy(false, false, RcMode::Cbr, 7, Deadline::Realtime, None),
            None
        );
        assert_eq!(
            policy(true, true, RcMode::Cbr, 7, Deadline::Realtime, None),
            None
        );
        assert_eq!(
            policy(true, false, RcMode::Vbr, 7, Deadline::Realtime, None),
            None
        );
        assert_eq!(
            policy(true, false, RcMode::Cbr, 6, Deadline::Realtime, None),
            None
        );
        assert_eq!(
            policy(true, false, RcMode::Cbr, 7, Deadline::Good, None),
            None
        );
        assert_eq!(
            policy(false, true, RcMode::Vbr, 6, Deadline::Good, Some(600)),
            Some(600)
        );
        assert_eq!(
            policy(true, false, RcMode::Cbr, 7, Deadline::Realtime, Some(0)),
            None
        );
    }

    #[test]
    fn final_newmv_suppression_requires_smooth_non_newmv_block16() {
        let suppress = |bsize, mode, dist, qindex, qmul| {
            stage4a_suppress_final_newmv(bsize, mode, dist, qindex, qmul)
        };
        assert!(suppress(
            BlockSize::Block16x16,
            InterMode::NearestMv,
            120_000,
            200,
            1000,
        ));
        assert!(!suppress(
            BlockSize::Block16x16,
            InterMode::NewMv,
            120_000,
            200,
            1000,
        ));
        assert!(!suppress(
            BlockSize::Block8x8,
            InterMode::NearestMv,
            120_000,
            200,
            1000,
        ));
        assert!(!suppress(
            BlockSize::Block16x16,
            InterMode::NearestMv,
            120_001,
            200,
            1000,
        ));
    }

    #[test]
    fn stage4a_split_admission_is_product_vbr_realtime_p8_only() {
        use rvp9_core::mode_decision::nonrd_partition::Block16SplitAdmission;
        use rvp9_core::mode_decision::speed_features::Deadline;

        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Slack1Over8
        );
        assert_eq!(
            stage4a_block16_split_admission(
                false,
                false,
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
        assert_eq!(
            stage4a_block16_split_admission(
                true,
                true,
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Cbr,
                7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Vbr,
                6,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Vbr,
                -7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Vbr,
                7,
                Deadline::Good,
                Stage4aProductSplitOverride::Auto,
            ),
            Block16SplitAdmission::Strict
        );
    }

    #[test]
    fn stage4a_split_force_strict_override_closes_product_policy() {
        use rvp9_core::mode_decision::nonrd_partition::Block16SplitAdmission;
        use rvp9_core::mode_decision::speed_features::Deadline;

        assert_eq!(
            stage4a_block16_split_admission(
                true,
                false,
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                Stage4aProductSplitOverride::ForceStrict,
            ),
            Block16SplitAdmission::Strict
        );
    }

    #[test]
    fn inter_rd_calibration_covers_realtime_v1_presets() {
        use rvp9_core::mode_decision::inter::InterRdCalibration;
        use rvp9_core::mode_decision::speed_features::Deadline;

        let active = inter_rd_calibration_for_mode(RcMode::Vbr, 7, Deadline::Realtime, false);
        assert_eq!(
            active,
            InterRdCalibration::realtime().with_large_block_model(false)
        );
        assert!(active.s3_domain);
        assert_eq!(
            active.s3_skip_shift, 4,
            "libvpx block_yrd compares against prediction SSE << 4"
        );
        assert!(active.skip_modecost);
        assert!(
            active.hadamard_only,
            "Stage 4a uses libvpx's single Hadamard block_yrd pipeline"
        );
        assert_eq!(
            (active.lambda_num, active.lambda_den),
            (3, 2),
            "Stage 4a's measured natural-content lambda optimum is 1.5x"
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Cbr, 6, Deadline::Realtime, false),
            InterRdCalibration::realtime()
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Vbr, 7, Deadline::Good, false),
            InterRdCalibration::default()
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Cbr, 7, Deadline::Realtime, false),
            InterRdCalibration::realtime()
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Cbr, 8, Deadline::Realtime, false),
            InterRdCalibration::realtime()
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Vbr, 6, Deadline::Realtime, false),
            InterRdCalibration::realtime().with_large_block_model(false)
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Cq, 7, Deadline::Realtime, false),
            InterRdCalibration::default()
        );
        assert_eq!(
            inter_rd_calibration_for_mode(RcMode::Cbr, 7, Deadline::Realtime, true),
            InterRdCalibration::default()
        );
    }

    #[test]
    fn golden_real_block_yrd_is_scoped_to_cpu7_realtime_vbr() {
        use rvp9_core::mode_decision::speed_features::Deadline;

        let cases = [
            (RcMode::Vbr, 7, Deadline::Realtime, false, true, None, true),
            (
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                false,
                false,
                None,
                false,
            ),
            (RcMode::Vbr, 6, Deadline::Realtime, false, true, None, false),
            (RcMode::Cbr, 7, Deadline::Realtime, false, true, None, false),
            (RcMode::Vbr, 7, Deadline::Good, false, true, None, false),
            (RcMode::Vbr, 7, Deadline::Realtime, true, true, None, false),
            (
                RcMode::Vbr,
                7,
                Deadline::Realtime,
                false,
                true,
                Some(false),
                false,
            ),
            (
                RcMode::Vbr,
                8,
                Deadline::Realtime,
                false,
                true,
                Some(true),
                true,
            ),
        ];
        for (rc_mode, speed, deadline, aq, golden, override_value, expected) in cases {
            assert_eq!(
                golden_real_block_yrd_enabled(rc_mode, speed, deadline, aq, golden, override_value,),
                expected
            );
        }
    }

    #[test]
    fn newmv_diff_bias_matches_libvpx_cbr_caller_gate() {
        assert!(newmv_diff_bias_for_mode(RcMode::Cbr, true));
        assert!(!newmv_diff_bias_for_mode(RcMode::Vbr, true));
        assert!(!newmv_diff_bias_for_mode(RcMode::Cq, true));
        assert!(!newmv_diff_bias_for_mode(RcMode::Q, true));
        assert!(!newmv_diff_bias_for_mode(RcMode::Cbr, false));
    }

    #[test]
    fn partition_color_sensitivity_uses_real_uv_prediction_sad() {
        let src_u = vec![80u8; 32 * 24];
        let src_v = vec![120u8; 32 * 24];
        let ref_u = src_u.clone();
        let ref_v = vec![80u8; 32 * 24];

        assert_eq!(
            partition_color_sensitivity(
                (&src_u, &src_v),
                (&ref_u, &ref_v),
                (64, 48),
                (0, 0),
                Mv::ZERO,
                4_000,
            ),
            [false, true],
        );
    }

    #[test]
    fn partition_color_sensitivity_reads_full_source_block_in_place() {
        let src_u = vec![80u8; 40 * 40];
        let src_v = vec![120u8; 40 * 40];
        let ref_u = src_u.clone();
        let ref_v = vec![80u8; 40 * 40];

        assert_eq!(
            partition_color_sensitivity(
                (&src_u, &src_v),
                (&ref_u, &ref_v),
                (80, 80),
                (0, 0),
                Mv::ZERO,
                4_000,
            ),
            [false, true],
        );
    }

    #[test]
    fn zero_mv_sad64_replicates_partial_frame_edges() {
        let src = vec![80u8; 80 * 72];
        let reference = vec![79u8; 80 * 72];
        assert_eq!(
            zero_mv_sad64_edge_clamped(&src, &reference, (80, 72), (64, 64)),
            64 * 64,
        );
    }

    #[test]
    fn zero_mv_sad64_full_block_matches_dispatched_kernel() {
        let src = (0..64 * 64)
            .map(|i| (i as u8).wrapping_mul(37))
            .collect::<Vec<_>>();
        let reference = (0..64 * 64)
            .map(|i| (i as u8).wrapping_mul(19).wrapping_add(11))
            .collect::<Vec<_>>();
        let edge_clamped = zero_mv_sad64_edge_clamped(&src, &reference, (64, 64), (0, 0));
        let dispatched =
            rvp9_core::dsp::get_ref().sad(BlockSize::Block64x64)(&src, 64, &reference, 64);
        assert_eq!(dispatched, edge_clamped);
    }

    #[test]
    fn full_pixel_search_matches_realtime_speed_policy() {
        use rvp9_core::mode_decision::inter::FullPixelSearchMethod;
        use rvp9_core::mode_decision::speed_features::Deadline;

        assert_eq!(
            full_pixel_search_method_for_mode(6, Deadline::Realtime),
            FullPixelSearchMethod::Nstep
        );
        assert_eq!(
            full_pixel_search_method_for_mode(7, Deadline::Realtime),
            FullPixelSearchMethod::FastDiamond
        );
        assert_eq!(
            full_pixel_search_method_for_mode(8, Deadline::Realtime),
            FullPixelSearchMethod::FastDiamond
        );
        assert_eq!(
            full_pixel_search_method_for_mode(8, Deadline::Good),
            FullPixelSearchMethod::Nstep
        );
    }

    #[test]
    fn realtime_integer_mv_filter_search_is_skipped_by_default() {
        let _guard = env_test_mutex();
        let key = "RVP9_RT_SKIP_INTMV_FILTER_SEARCH";
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
        assert!(InterEncodeEnv::capture().skip_integer_mv_filter_search());

        // Preserve the diagnostic escape hatch used by differential sweeps.
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::set_var(key, "0") };
        assert!(!InterEncodeEnv::capture().skip_integer_mv_filter_search());
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn realtime_descent_uses_chessboard_filter_search_by_default() {
        let _guard = env_test_mutex();
        let key = "RVP9_A4_DESCENT_CB_FILTER";
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
        assert!(InterEncodeEnv::capture().descent_cb_filter_search());

        // Preserve the full-search differential escape hatch.
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::set_var(key, "0") };
        assert!(!InterEncodeEnv::capture().descent_cb_filter_search());
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn realtime_descent_disables_golden_by_default() {
        let _guard = env_test_mutex();
        let key = "RVP9_A4_DESCENT_DISABLE_GOLDEN";
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
        assert!(InterEncodeEnv::capture().descent_disable_golden());

        // Preserve a full-reference differential escape hatch. This changes
        // only the speculative descent; the final picker remains untouched.
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::set_var(key, "0") };
        assert!(!InterEncodeEnv::capture().descent_disable_golden());
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn realtime_descent_disables_rectangular_newmv_by_default() {
        let _guard = env_test_mutex();
        let key = "RVP9_A4_DESCENT_DISABLE_RECT_NEWMV";
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
        assert!(
            InterEncodeEnv::capture().descent_disable_rect_newmv(),
            "rectangular proxy candidates should avoid independent motion search",
        );

        // Preserve a full-motion differential escape hatch. Square proxy
        // candidates and the final picker are unaffected by either state.
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::set_var(key, "0") };
        assert!(!InterEncodeEnv::capture().descent_disable_rect_newmv());
        // SAFETY: the mutex above serialises all encoder environment tests.
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn realtime_luma_transform_honors_acdc_skip_under_fast_quantization() {
        use rvp9_core::mode_decision::model_rd::SkipTxfm;

        assert!(should_skip_realtime_luma_transform(true, SkipTxfm::AcDc));
        assert!(!should_skip_realtime_luma_transform(false, SkipTxfm::AcDc));
        assert!(!should_skip_realtime_luma_transform(true, SkipTxfm::AcOnly));
        assert!(!should_skip_realtime_luma_transform(true, SkipTxfm::None));
    }

    #[test]
    fn realtime_block32_emit_reuses_picker_transform() {
        assert_eq!(
            realtime_emit_luma_tx_size(true, BlockSize::Block32x32, TxSize::Tx8x8),
            TxSize::Tx8x8,
        );
        assert_eq!(
            realtime_emit_luma_tx_size(true, BlockSize::Block64x64, TxSize::Tx8x8),
            TxSize::Tx32x32,
        );
        assert_eq!(
            realtime_emit_luma_tx_size(true, BlockSize::Block16x16, TxSize::Tx8x8),
            TxSize::Tx16x16,
        );
        assert_eq!(
            realtime_emit_luma_tx_size(false, BlockSize::Block64x64, TxSize::Tx8x8),
            TxSize::Tx8x8,
        );
    }

    #[test]
    fn exact_vbp_large_frames_release_max_tx_shortcut() {
        assert!(realtime_force_max_emit_tx(None, true, false));
        assert!(!realtime_force_max_emit_tx(None, true, true));
        assert!(!realtime_force_max_emit_tx(Some(false), true, false));
        assert!(realtime_force_max_emit_tx(Some(true), true, true));
    }

    #[test]
    fn realtime_tx_redecision_searches_only_tx8_and_tx16() {
        assert!(!realtime_emit_tx_rd_enabled(None));
        assert!(realtime_emit_tx_rd_enabled(Some(true)));
        assert!(!realtime_emit_tx_rd_enabled(Some(false)));
        assert!(should_redecide_inter_luma_tx(
            false,
            true,
            BlockSize::Block16x16,
        ));
        assert!(!should_redecide_inter_luma_tx(
            false,
            true,
            BlockSize::Block32x32,
        ));
        assert!(!should_redecide_inter_luma_tx(
            true,
            true,
            BlockSize::Block16x16,
        ));
        assert!(!should_redecide_inter_luma_tx(
            false,
            false,
            BlockSize::Block16x16,
        ));
        assert!(!realtime_inter_tx_candidate(TxSize::Tx16x16, TxSize::Tx4x4));
        assert!(realtime_inter_tx_candidate(TxSize::Tx16x16, TxSize::Tx8x8));
        assert!(realtime_inter_tx_candidate(
            TxSize::Tx16x16,
            TxSize::Tx16x16
        ));
        assert!(!realtime_inter_tx_candidate(TxSize::Tx32x32, TxSize::Tx8x8));
    }

    #[test]
    fn a4_partition_rd_mult_env_scales_only_with_valid_ratio() {
        let _guard = env_test_mutex();
        let num_key = "RVP9_A4_RDMULT_NUM";
        let den_key = "RVP9_A4_RDMULT_DEN";
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::remove_var(num_key);
            std::env::remove_var(den_key);
        }
        assert_eq!(a4_partition_rd_mult_for_env(1000), 1000);
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::set_var(num_key, "3");
            std::env::set_var(den_key, "4");
        }
        assert_eq!(a4_partition_rd_mult_for_env(1000), 750);
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::set_var(num_key, "0");
        }
        assert_eq!(a4_partition_rd_mult_for_env(1000), 0);
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::set_var(num_key, "3");
            std::env::set_var(den_key, "0");
        }
        assert_eq!(a4_partition_rd_mult_for_env(1000), 1000);
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::set_var(den_key, "not-a-denominator");
        }
        assert_eq!(a4_partition_rd_mult_for_env(1000), 1000);
        // SAFETY: the mutex above serialises all env_test_mutex callers.
        unsafe {
            std::env::remove_var(num_key);
            std::env::remove_var(den_key);
        }
    }

    #[test]
    fn a4_independent_mv_proxy_is_scoped_to_stage4a_vbr() {
        use rvp9_core::mode_decision::speed_features::Deadline;

        assert!(a4_descent_independent_mv_default(
            crate::rate_control::RcMode::Vbr,
            7,
            Deadline::Realtime,
        ));
        assert!(a4_descent_independent_mv_default(
            crate::rate_control::RcMode::Vbr,
            8,
            Deadline::Realtime,
        ));
        assert!(!a4_descent_independent_mv_default(
            crate::rate_control::RcMode::Vbr,
            6,
            Deadline::Realtime,
        ));
        assert!(!a4_descent_independent_mv_default(
            crate::rate_control::RcMode::Cbr,
            7,
            Deadline::Realtime,
        ));
        assert!(!a4_descent_independent_mv_default(
            crate::rate_control::RcMode::Vbr,
            7,
            Deadline::Good,
        ));
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
        let inner = ModesWriterError::MissingCell {
            mi_row: 0,
            mi_col: 0,
        };
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
        assert_eq!(
            got_coefs,
            &[1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(got_eob, 5);
    }

    #[test]
    fn vec_backed_inter_coef_source_round_trips() {
        use rvp9_core::bitstream::InterCoefSource;
        let mut src = VecBackedInterCoefSource::new(128, 64);
        assert_eq!(src.width(), 128);
        assert_eq!(src.height(), 64);
        let coefs = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        src.insert(1, 2, 1, 0, 0, coefs.clone(), 7);
        let (got_coefs, got_eob) = src.get_block_coefs(1, 2, 1, 0, 0, TxSize::Tx4x4);
        assert_eq!(
            got_coefs,
            &[
                10i16, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160
            ]
        );
        assert_eq!(got_eob, 7);
    }

    #[test]
    fn vec_backed_inter_coef_source_empty_zero_marker_reads_as_zeros() {
        use rvp9_core::bitstream::InterCoefSource;
        let mut src = VecBackedInterCoefSource::new(128, 64);
        src.insert(1, 2, 1, 0, 0, Vec::new(), 0);

        for tx_size in [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx32x32,
        ] {
            let (got_coefs, got_eob) = src.get_block_coefs(1, 2, 1, 0, 0, tx_size);
            assert_eq!(got_eob, 0);
            assert_eq!(got_coefs.len(), tx_size.n_coeffs());
            assert!(got_coefs.iter().all(|&c| c == 0));
        }
    }

    #[test]
    fn inter_coef_direct_slots_cover_every_plane_position_once() {
        let src = VecBackedInterCoefSource::new(68, 36);
        let mut observed = std::collections::HashSet::new();

        for plane in 0..3 {
            let mi_scale = if plane == 0 { 2 } else { 1 };
            for y4 in 0..src.plane_rows[plane] {
                for x4 in 0..src.plane_strides[plane] {
                    let slot = src.slot_index(
                        y4 / mi_scale,
                        x4 / mi_scale,
                        plane,
                        x4 % mi_scale,
                        y4 % mi_scale,
                    );
                    assert!(observed.insert(slot), "slot {slot} aliases");
                }
            }
        }

        assert_eq!(observed.len(), src.slots.len());
        assert!((0..src.slots.len()).all(|slot| observed.contains(&slot)));
    }

    #[test]
    fn inter_coef_direct_slots_replace_and_merge() {
        use rvp9_core::bitstream::InterCoefSource;

        let mut left = VecBackedInterCoefSource::new(128, 64);
        left.insert(1, 2, 0, 0, 0, vec![1; 16], 1);
        left.insert(1, 2, 0, 0, 0, vec![7; 16], 7);

        let mut right = VecBackedInterCoefSource::new(128, 64);
        right.insert(1, 12, 2, 0, 0, vec![9; 16], 9);
        left.merge_from(right);

        let (left_coefs, left_eob) = left.get_block_coefs(1, 2, 0, 0, 0, TxSize::Tx4x4);
        assert_eq!(left_coefs, &[7; 16]);
        assert_eq!(left_eob, 7);
        let (right_coefs, right_eob) = left.get_block_coefs(1, 12, 2, 0, 0, TxSize::Tx4x4);
        assert_eq!(right_coefs, &[9; 16]);
        assert_eq!(right_eob, 9);
        assert_eq!(
            left.blocks.len(),
            2,
            "replacement must reuse its value slot"
        );
    }

    #[test]
    fn inter_coef_tile_sources_store_and_merge_only_their_columns() {
        use rvp9_core::bitstream::InterCoefSource;

        let mut frame = VecBackedInterCoefSource::new(128, 64);
        let mut left = VecBackedInterCoefSource::new_for_mi_columns(128, 64, 0, 8);
        let mut right = VecBackedInterCoefSource::new_for_mi_columns(128, 64, 8, 16);
        assert_eq!(
            left.slots.len() + right.slots.len(),
            frame.slots.len(),
            "disjoint tile sources must allocate one frame of slots in total",
        );

        left.insert(1, 2, 0, 0, 0, vec![3; 16], 3);
        right.insert(1, 12, 2, 0, 0, vec![9; 16], 9);
        frame.merge_from(left);
        frame.merge_from(right);

        let (left_coefs, left_eob) = frame.get_block_coefs(1, 2, 0, 0, 0, TxSize::Tx4x4);
        assert_eq!(left_coefs, &[3; 16]);
        assert_eq!(left_eob, 3);
        let (right_coefs, right_eob) = frame.get_block_coefs(1, 12, 2, 0, 0, TxSize::Tx4x4);
        assert_eq!(right_coefs, &[9; 16]);
        assert_eq!(right_eob, 9);
    }

    #[test]
    fn inter_coef_worker_sources_merge_disjoint_rows_in_one_tile() {
        use rvp9_core::bitstream::InterCoefSource;

        let mut frame = VecBackedInterCoefSource::new(128, 128);
        let mut upper_worker = VecBackedInterCoefSource::new_for_mi_columns(128, 128, 0, 16);
        let mut lower_worker = VecBackedInterCoefSource::new_for_mi_columns(128, 128, 0, 16);
        upper_worker.insert(1, 2, 0, 0, 0, vec![4; 16], 4);
        lower_worker.insert(9, 2, 0, 0, 0, vec![8; 16], 8);

        // Worker completion order is deliberately the inverse of tile-raster
        // row order. Dense physical slots, not merge order, drive replay.
        frame.merge_from(lower_worker);
        frame.merge_from(upper_worker);

        let (upper, upper_eob) = frame.get_block_coefs(1, 2, 0, 0, 0, TxSize::Tx4x4);
        let (lower, lower_eob) = frame.get_block_coefs(9, 2, 0, 0, 0, TxSize::Tx4x4);
        assert_eq!(upper, &[4; 16]);
        assert_eq!(upper_eob, 4);
        assert_eq!(lower, &[8; 16]);
        assert_eq!(lower_eob, 8);
    }

    #[test]
    #[should_panic(expected = "disjoint physical TX slots")]
    fn inter_coef_direct_slot_merge_rejects_overlap() {
        let mut left = VecBackedInterCoefSource::new(128, 64);
        left.insert(1, 2, 0, 0, 0, vec![1; 16], 1);
        let mut overlapping = VecBackedInterCoefSource::new(128, 64);
        overlapping.insert(1, 2, 0, 0, 0, vec![2; 16], 2);
        left.merge_from(overlapping);
    }

    /// Serial guard for env-mutating tests that touch the global
    /// E12/F02 oracle flags or RVP9_* env vars.
    fn env_test_mutex() -> std::sync::MutexGuard<'static, ()> {
        crate::env_test_mutex()
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
            noise_sensitivity: 0,
            bit_depth: rvp9_core::BitDepth::Bd8,
            lag_in_frames: 0,
            enable_alt_ref: false,
            rc_mode: RcMode::Cbr,
            cq_level: 0,
            temporal_layers: 1,
            enable_tpl: false,
            stage4a_product_split_override: Default::default(),
            min_qindex: 4,
            max_qindex: 255,
        }
    }

    fn yuv_view_64<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> frame_loop::YuvFrameRef<'a> {
        frame_loop::YuvFrameRef {
            y,
            u,
            v,
            y_stride: 64,
            u_stride: 32,
            v_stride: 32,
        }
    }

    fn flat_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            vec![128u8; 64 * 64],
            vec![128u8; 32 * 32],
            vec![128u8; 32 * 32],
        )
    }

    fn textured_luma_64(seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..64 * 64)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn encode_with_f02_oracle_enabled_runs_record_paths() {
        let _guard = env_test_mutex();
        let key = "RVP9_F02_ORACLE";
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var(key, "1");
            std::env::set_var("RVP9_RT_EMIT_TX_RD", "1");
            std::env::remove_var("RVP9_RT_EMIT_MAX_TX");
        }
        crate::f02_oracle::init_from_env();
        assert!(crate::f02_oracle::is_enabled());

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let y = textured_luma_64(0x9e37_79b9);
        let mut shifted_y = vec![0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                shifted_y[row * 64 + col] = y[row * 64 + (col + 1).min(63)];
            }
        }
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let _inter = enc.encode_frame(&yuv_view_64(&shifted_y, &u, &v)).unwrap();

        let decisions = crate::f02_oracle::drain_frame();
        assert!(
            !decisions.is_empty(),
            "F02 oracle should have recorded ≥1 decision when enabled",
        );
        assert!(
            decisions.iter().any(|decision| decision.luma_eob_nonzero),
            "F02 oracle should report a non-zero emitted luma EOB",
        );
        assert!(
            decisions
                .iter()
                .any(|decision| decision.tx_size < rvp9_core::block::max_txsize(decision.bsize)),
            "deterministic texture should exercise an emit-side transform smaller than \
             the picker's maximum transform",
        );

        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var(key);
            std::env::remove_var("RVP9_RT_EMIT_TX_RD");
        }
        crate::f02_oracle::init_from_env();
        assert!(!crate::f02_oracle::is_enabled());
    }

    #[test]
    fn encode_with_e12_oracle_enabled_runs_record_paths() {
        let _guard = env_test_mutex();
        let key = "RVP9_E12_ORACLE";
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var(key, "1");
        }
        crate::e12stat::init_from_env();
        assert!(crate::e12stat::enabled());

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        assert!(inter.actual_bits > 0);

        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var(key);
        }
        crate::e12stat::init_from_env();
        assert!(!crate::e12stat::enabled());
    }

    #[test]
    fn encode_with_f3_nonrd_partition_runs_descent_path() {
        let _guard = env_test_mutex();
        let key = "RVP9_F3_NONRD_PARTITION_ENABLE";
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var(key, "1");
        }

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        assert_eq!(inter.frame_kind, frame_loop::FrameKind::Inter);

        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var(key);
        }
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
                // A4: the descent is now default-ON, so "gate off" must
                // explicitly DISABLE it via the opt-out var; "gate on"
                // is the default but we clear the opt-out to be sure.
                if gate_on {
                    std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
                } else {
                    std::env::set_var("RVP9_A4_RD_PARTITION_DISABLE", "1");
                }
            }
            crate::f02_oracle::init_from_env();

            let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
            let _kf = enc.encode_frame(&yuv_view_64(&kf_y, &uv, &uv)).unwrap();
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
        // SAFETY: outer test holds env_test_mutex for the whole test.
        unsafe {
            std::env::remove_var("RVP9_F02_ORACLE");
            std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
            std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        }
        crate::f02_oracle::init_from_env();

        assert!(
            !leaves_off.is_empty(),
            "baseline (gate OFF) must emit leaves"
        );
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

    /// Partition descent deliberately uses a luma-only proxy, while the
    /// committed-leaf picker still sees chroma. Two inter frames with the
    /// same luma and uniformly shifted chroma must therefore keep the same
    /// partition layout but produce different final chroma RD evidence.
    #[test]
    fn a4_descent_is_luma_only_but_final_pick_uses_chroma() {
        let _guard = env_test_mutex();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var("RVP9_F02_ORACLE", "1");
            std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        }
        crate::f02_oracle::init_from_env();

        let y = textured_luma_64(0xa4d3_5c17);
        let reference_uv = vec![128u8; 32 * 32];
        let shifted_uv = vec![160u8; 32 * 32];

        let collect = |inter_uv: &[u8]| {
            let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
            let _key = enc
                .encode_frame(&yuv_view_64(&y, &reference_uv, &reference_uv))
                .unwrap();
            let _ = crate::f02_oracle::drain_frame();
            let inter = enc
                .encode_frame(&yuv_view_64(&y, inter_uv, inter_uv))
                .unwrap();
            let decisions = crate::f02_oracle::drain_frame();
            let layout = decisions
                .iter()
                .map(|decision| (decision.mi_row, decision.mi_col, decision.bsize))
                .collect::<Vec<_>>();
            let chroma_rd = decisions
                .iter()
                .map(|decision| (decision.uv_model_rate, decision.uv_model_dist))
                .collect::<Vec<_>>();
            (layout, chroma_rd, inter.data)
        };

        let (reference_layout, reference_chroma_rd, reference_packet) = collect(&reference_uv);
        let (shifted_layout, shifted_chroma_rd, shifted_packet) = collect(&shifted_uv);

        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var("RVP9_F02_ORACLE");
        }
        crate::f02_oracle::init_from_env();

        assert!(!reference_layout.is_empty(), "inter frame must emit leaves");
        assert_eq!(
            reference_layout, shifted_layout,
            "uniform chroma changes must not alter the luma-only descent layout",
        );
        assert_ne!(
            reference_chroma_rd, shifted_chroma_rd,
            "final committed-leaf picks must retain chroma RD",
        );
        assert_ne!(
            reference_packet, shifted_packet,
            "changed chroma residuals must reach the emitted stream",
        );
    }

    /// A4 smoke — the default encode path (RD partition descent now
    /// default-ON, no env vars set) still produces a well-formed KF +
    /// inter packet pair. Pre-A4 this asserted "descent OFF = unchanged
    /// output"; A4 promotes the descent to the default, so this is now
    /// a default-path liveness check (the descent runs; the two-frame
    /// flow still yields non-empty Key + Inter packets). The
    /// cross-decoder gate proves the default stream is valid VP9.
    #[test]
    fn p12_real_callback_default_state_unchanged() {
        let _guard = env_test_mutex();
        // Default path: no descent env vars set (descent is default-ON).
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var("RVP9_F3_NONRD_PARTITION_ENABLE");
            std::env::remove_var("RVP9_A4_RD_PARTITION_DISABLE");
        }

        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
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
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var(key, "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var(key);
        }
        assert_eq!(inter.frame_kind, frame_loop::FrameKind::Inter);
    }

    #[test]
    fn encode_with_diag_intpro_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var("RVP9_DIAG_INTPRO", "1");
            std::env::set_var("RVP9_DIAG_INTPRO_SB", "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var("RVP9_DIAG_INTPRO");
            std::env::remove_var("RVP9_DIAG_INTPRO_SB");
        }
        assert!(inter.actual_bits > 0);
    }

    #[test]
    fn encode_with_diag_intra_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var("RVP9_DIAG_INTRA", "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::remove_var("RVP9_DIAG_INTRA");
        }
        assert_eq!(kf.frame_kind, frame_loop::FrameKind::Key);
    }

    #[test]
    fn encode_with_e3_mvdiag_and_golden_env_runs_eprintln_paths() {
        let _guard = env_test_mutex();
        // SAFETY: serialised via env_test_mutex.
        unsafe {
            std::env::set_var("RVP9_E3_MVDIAG", "1");
            std::env::set_var("RVP9_DIAG_GOLDEN_PICKS", "1");
        }
        let mut enc = frame_loop::StreamEncoder::new(streaming_config()).unwrap();
        let (y, u, v) = flat_frame();
        let _kf = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        let inter = enc.encode_frame(&yuv_view_64(&y, &u, &v)).unwrap();
        // SAFETY: serialised via env_test_mutex.
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
        assert_eq!(
            pkt[0] & 0xc0,
            0x80,
            "frame marker mismatch in pkt[0] = {:#04x}",
            pkt[0]
        );
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
            RefFrameSizes, parse_compressed_header, parse_uncompressed_header,
        };

        let enc = Encoder::new(Config::new(64, 64, 50)).unwrap();
        let pkt = enc.encode_synthetic_dc_skip_keyframe().unwrap();

        let no_refs = RefFrameSizes::default();
        let (hdr, hdr_end) =
            parse_uncompressed_header(&pkt, &no_refs).expect("uncompressed header parses");
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
        let res =
            parse_compressed_header(compressed, &mut fc, opts).expect("compressed header parses");
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
        let (kf_hdr, _) = parse_uncompressed_header(kf_slice, &no_refs).expect("kf header parses");
        assert_eq!(kf_hdr.kind, FrameKind::KeyFrame);

        // The inter frame requires the keyframe's pool-slot-0 size in
        // `ref_sizes`. The keyframe was 64×64.
        let mut ref_sizes = RefFrameSizes::default();
        ref_sizes[0] = Some((64, 64));
        let (inter_hdr, _) =
            parse_uncompressed_header(inter_slice, &ref_sizes).expect("inter header parses");
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
        assert!(matches!(
            frames[1].header.kind,
            FrameKind::InterFrame { .. }
        ));

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
        assert!(matches!(
            frames[1].header.kind,
            FrameKind::InterFrame { .. }
        ));

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
            SegmentationWriteParams as ModesSegParams, UncompressedHeader, write_compressed_header,
            write_tile_intra_keyframe, write_uncompressed_header,
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
            write_compressed_header(&mut bw, &mut pre_fc, &post_fc, opts, &target, None)?;
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
            let tile_mi_cols = (tile_bounds.mi_col_end - tile_bounds.mi_col_start).max(1);
            let mut coef_ctx = TileCoefCtx::new(tile_mi_cols);
            let coef_source = AllZeroIntraCoefSource::new();
            write_tile_intra_keyframe(
                &mut bw,
                &mi_grid,
                mi_rows,
                mi_cols,
                tile_bounds,
                &post_fc,
                rvp9_core::transforms::TxMode::AllowTx32x32,
                Some(ModesSegParams {
                    tree_probs: &tree_probs,
                }),
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
                let cell =
                    cell_got.unwrap_or_else(|| panic!("seg={target} cell [{i}] must be populated"));
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
    #[test]
    fn realtime_keyframe_tx_size_caps_leaf_maximum_at_16x16() {
        assert_eq!(
            realtime_keyframe_tx_size(BlockSize::Block8x8),
            TxSize::Tx8x8
        );
        assert_eq!(
            realtime_keyframe_tx_size(BlockSize::Block16x16),
            TxSize::Tx16x16
        );
        assert_eq!(
            realtime_keyframe_tx_size(BlockSize::Block32x32),
            TxSize::Tx16x16
        );
        assert_eq!(
            realtime_keyframe_tx_size(BlockSize::Block64x64),
            TxSize::Tx16x16
        );
    }

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
        use rvp9_core::bitstream::IntraCoefSource;
        use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

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
                    predicted_y_modes: [IntraMode::Dc; 4],
                    predicted_uv_mode: IntraMode::Dc,
                    skip: false,
                    segment_id: seg,
                    picked_tx_size: rvp9_core::block::max_txsize(BlockSize::Block8x8),
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
        let (base_qc_seg1, base_eob_seg1) =
            base_coefs.get_block_coefs(4, 0, 0, 0, 0, TxSize::Tx8x8);
        let (aq_qc_seg1, aq_eob_seg1) = aq_coefs.get_block_coefs(4, 0, 0, 0, 0, TxSize::Tx8x8);
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
        let (base_qc_seg0, base_eob_seg0) =
            base_coefs.get_block_coefs(0, 0, 0, 0, 0, TxSize::Tx8x8);
        let (aq_qc_seg0, aq_eob_seg0) = aq_coefs.get_block_coefs(0, 0, 0, 0, 0, TxSize::Tx8x8);
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
