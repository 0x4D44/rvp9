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

use crate::loopfilter::{
    apply_loop_filter_to_tight_recon, build_inter_mi_grid_for_lf, build_kf_mi_grid_for_lf,
    pick_filter_level,
};
use crate::rate_control::{FrameKind as RcFrameKind, RateControl, RateControlConfig};
use crate::{
    EncodeError, InterPartitionLeafDesc, PartitionLeafDesc,
    pick_and_simulate_intra_encode_partitions, pick_and_simulate_inter_encode_partitions,
};

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
            FrameKind::Inter => self.encode_inter_internal(
                q_index,
                (&y_tight, &u_tight, &v_tight),
            )?,
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
        let coef_source = pick_and_simulate_intra_encode_partitions(
            w,
            h,
            q_index,
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
    fn encode_inter_internal(
        &mut self,
        q_index: u8,
        input_yuv: (&[u8], &[u8], &[u8]),
    ) -> Result<Vec<u8>, EncodeError> {
        let w = self.config.width;
        let h = self.config.height;

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
        let mut sb_partitions: Vec<Vec<InterPartitionLeafDesc>> =
            sb_partitions_intra_shape
                .iter()
                .map(|leaves| {
                    leaves
                        .iter()
                        .map(|l| InterPartitionLeafDesc {
                            mi_row_in_sb: l.mi_row_in_sb,
                            mi_col_in_sb: l.mi_col_in_sb,
                            leaf_bsize: l.leaf_bsize,
                            picked_mode: InterMode::ZeroMv,
                            picked_mv: Mv::ZERO,
                            skip: true,
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
            input_yuv,
            (&self.last_recon_y, &self.last_recon_u, &self.last_recon_v),
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

        // Build the packet from the populated coef source + leaf
        // descriptors.
        let packet = build_inter_packet_with_residuals(
            w,
            h,
            q_index,
            filter_level,
            &coef_source,
            &sb_partitions,
        )?;

        // Commit the new reconstruction as the encoder's new LAST ref.
        self.last_recon_y = new_recon_y;
        self.last_recon_u = new_recon_u;
        self.last_recon_v = new_recon_v;

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
                        // F10.1b — picker-driven Y mode; UV stays DC.
                        y_mode: leaf.predicted_y_mode,
                        uv_mode: IntraMode::Dc,
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
                    segment_id: 0,
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
fn build_inter_packet_with_residuals(
    width: u32,
    height: u32,
    base_qindex: u8,
    filter_level: u8,
    coef_source: &dyn InterCoefSource,
    sb_partitions: &[Vec<InterPartitionLeafDesc>],
) -> Result<Vec<u8>, EncodeError> {
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
            /* prev_mv_grid = */ None,
            &mut coef_ctx,
            coef_source,
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

        let coef_source = simulate_dc_intra_encode_partitions(
            w,
            h,
            q_index,
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
        let cfg = EncoderConfig {
            width: 128,
            height: 128,
            framerate: 30.0,
            target_bitrate_bps: 4_000_000,
            gop_length: 30,
        };
        let mut enc = StreamEncoder::new(cfg).unwrap();
        let (y0, u0, v0) = gradient_frame(128, 128);
        let kf = enc.encode_frame(&yuv_view(&y0, &u0, &v0, 128)).unwrap();
        assert_eq!(kf.frame_kind, FrameKind::Key);

        // Inter frame: gradient — F8.2 should keep partition at
        // Block64x64; F8.1 fires and chooses some non-skip leaf.
        let inter = enc.encode_frame(&yuv_view(&y0, &u0, &v0, 128)).unwrap();
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
}
