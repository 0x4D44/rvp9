//! rvp9-decoder: VP9 video decoder.
//!
//! Implements the VP9 Bitstream & Decoding Process Specification.
//! Validated end-to-end against the official VP9 conformance test
//! vectors.
//!
//! ## Stage 3 status
//!
//! Stage 3 Task 8.1 lands the [`Decoder`] skeleton — the top-level
//! state container that owns the [`DecoderFramePool`], the per-frame
//! probability banks ([`FrameContext`] × 4), and the running
//! loop-filter delta state.
//!
//! What works today:
//!
//! - [`Decoder::parse_packet`] splits a superframe (1..=8 frames per
//!   transport packet) into individual frame slices via
//!   [`parse_superframe_index`], then parses each frame's
//!   uncompressed header via [`parse_uncompressed_header`]. Returns
//!   the [`PacketFrames`] iterator of `(header, byte_range)` per
//!   frame.
//!
//! - [`Decoder::frame_pool`] / `frame_pool_mut` accessors for the
//!   ref-counted frame buffer pool.
//!
//! What does NOT work yet (deferred to later Stage 3 tasks):
//!
//! - Compressed-header parsing (Task 1e — boolean-coder driver
//!   reading tx_mode, coef-prob updates, frame-ref-mode, intra/inter
//!   probs, MV probs).
//! - Tile-level coefficient decoding (Tasks 8.2+).
//! - Pixel reconstruction (intra-pred + transform + add + loopfilter)
//!   per superblock (Task 8.3).
//! - Conformance gate against vpxdec (Task 9).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod detokenize;
pub mod reconstruct;
pub mod reconstruct_inter;
pub mod tile;
pub mod tile_ctx;

use rvp9_core::bitstream::{
    BitstreamError, ChromaFormat, CompressedHeaderOptions, CompressedHeaderResult, FrameKind,
    InterpFilter, Superframe, UncompressedHeader, parse_compressed_header,
    parse_superframe_index, parse_tile_buffers, parse_uncompressed_header,
};
use rvp9_core::entropy::{FRAME_CONTEXTS, FrameContext};
use rvp9_core::frame::{BorderUsage, ByteAlignment, DecoderFramePool, FrameBuf, Yv12Buffer};
use rvp9_core::inter::mode_info::{InterModeReadParams, setup_compound_reference_mode};
use rvp9_core::inter::mv_ref::sign_bias_from_header;
use rvp9_core::frame::extend_frame_borders;
use rvp9_core::loopfilter::{
    LoopFilterInfo, LoopFilterMask, ResolvedLfDeltas, SegmentationLfView, adjust_mask,
    loop_filter_frame, resolve_filter_levels, setup_mask,
};
use rvp9_core::block::BlockSize;
use rvp9_core::mode_info::ModeInfo;

use crate::tile::{
    FrameDecodeParams, InterFrameDecodeParams, TileDecodeError, decode_tile_inter,
    decode_tile_intra, tile_bounds,
};
use crate::tile_ctx::TileDecodeCtx;

/// Top-level VP9 decoder state.
///
/// Mirrors the role of libvpx's `VP9Decoder` + `VP9_COMMON`
/// — owns every per-stream and per-frame-history data structure the
/// decoder needs.
///
/// Construction does not require a stream header — call
/// [`Decoder::new`], then feed packets via
/// [`Decoder::parse_packet`].
pub struct Decoder {
    /// 9-slot pool of decoded reference frames.
    frame_pool: DecoderFramePool,
    /// 8-slot reference-frame map. `ref_frame_map[i]` is the
    /// [`FrameBuf`] currently bound to logical ref slot `i`, or
    /// `None` if that slot has never been written. Mirrors libvpx's
    /// `cm->ref_frame_map[i]` — see `vp9_decodeframe.c:2940`'s
    /// `cm->frame_refs[i].buf = &cm->buffer_pool->frame_bufs[
    /// cm->ref_frame_map[ref_idx]].buf`. Updated post-decode in
    /// [`Decoder::decode_keyframe`] (every slot per
    /// `refresh_frame_flags = 0xFF`) and [`Decoder::decode_inter_frame`]
    /// (per the parsed `refresh_frame_flags` mask).
    ref_frame_map: [Option<FrameBuf>; rvp9_core::bitstream::REF_FRAMES],
    /// Cached (width, height) of every frame currently held in the
    /// 8-slot reference-frame-map. `None` means the corresponding
    /// ref-map slot has no decoded frame yet. Mirrors the
    /// `cm->ref_frame_map[i] -> frame_bufs[i].buf->y_crop_{width,height}`
    /// lookup chain libvpx uses inside `setup_frame_size_with_refs`
    /// (`vp9_decodeframe.c:1559-1574`). Updated by `decode_keyframe`
    /// (which sets every slot per `refresh_frame_flags = 0xFF`).
    /// Decoupled from the frame_pool slot pool: this is the minimum
    /// state needed by the uncompressed-header parser, and does NOT
    /// require any actual pixels to be stored.
    ref_frame_sizes: [Option<(u32, u32)>; rvp9_core::bitstream::REF_FRAMES],
    /// Four independent FRAME_CONTEXT banks, selected per-frame by
    /// the 2-bit `frame_context_idx` header field. Each is reset to
    /// libvpx defaults on construction and updated by backward
    /// adaptation after each frame's residuals are decoded.
    frame_contexts: [FrameContext; FRAME_CONTEXTS],
    /// Accumulated motion-vector decode counts for the in-flight
    /// inter frame. Zeroed at the start of each inter frame; consumed
    /// by [`adapt_mv_probs`](rvp9_core::entropy::adapt::adapt_mv_probs)
    /// at end-of-frame to produce the next frame's `nmvc` table.
    /// Mirrors libvpx `cm->counts.mv` (`vp9_entropymode.h:78`).
    mv_counts: rvp9_core::entropy::mv_probs::NmvCounts,
    /// Accumulated coefficient decode counts for the in-flight frame.
    /// Zeroed at the start of every frame (libvpx clears `cm->counts`
    /// for any frame with `frame_parallel_decoding_mode == 0` —
    /// `vp9_decodeframe.c:2784`). Consumed by
    /// [`adapt_coef_probs`](rvp9_core::entropy::adapt::adapt_coef_probs)
    /// at end-of-frame to produce the next frame's `coef_probs` table.
    /// Both keyframes and inter frames adapt coefficient probabilities
    /// (only the `(count_sat, max_update_factor)` pair differs — see
    /// [`coef_adapt_params`](rvp9_core::entropy::adapt::coef_adapt_params)).
    /// Mirrors libvpx `cm->counts.coef` + `cm->counts.eob_branch`
    /// (`vp9_entropymode.h:66-68`).
    coef_counts: rvp9_core::entropy::adapt::CoefCountsBank,
    /// `true` iff the most recently decoded frame was a `KEY_FRAME`
    /// (libvpx `cm->last_frame_type`). Read by
    /// [`coef_adapt_params`](rvp9_core::entropy::adapt::coef_adapt_params)
    /// to pick the AFTER_KEY update factor on the first inter frame
    /// after a keyframe. Updated post-decode in
    /// [`Decoder::decode_keyframe`] (= `true`) and
    /// [`Decoder::decode_inter_frame`] (= `false`).
    last_frame_was_key: bool,
    /// Accumulated mode / partition / skip / inter / ref / tx
    /// counts for the in-flight inter frame. Zeroed at the start of
    /// each inter frame; consumed by
    /// [`adapt_mode_probs`](rvp9_core::entropy::adapt::adapt_mode_probs)
    /// at end-of-frame to produce the next frame's mode probabilities.
    /// Mirrors libvpx `cm->counts` (the y_mode / uv_mode / partition /
    /// inter_mode / intra_inter / comp_inter / comp_ref / single_ref /
    /// switchable_interp / skip / tx fields of `vp9_entropymode.h`
    /// `FRAME_COUNTS`). Keyframes do not call `adapt_mode_probs` (libvpx
    /// `vp9_decodeframe.c:3050`: gated on `!frame_is_intra_only(cm)`),
    /// so this field is only meaningfully populated on inter-frame
    /// decode.
    mode_counts: rvp9_core::entropy::adapt::FrameCounts,
    /// Current per-ref / per-mode loop-filter deltas. The
    /// uncompressed header overlays these via the `Option<i8>` slots
    /// in `LoopFilterParams`; we maintain the resolved running state
    /// here.
    lf_deltas: ResolvedLfDeltas,
    /// `true` once the decoder has seen a successful keyframe.
    /// `parse_packet` blocks inter-frame headers from being acted on
    /// before a keyframe (libvpx `pbi->need_resync`).
    have_resync: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Construct a fresh decoder. The frame pool starts empty; the
    /// four FRAME_CONTEXT banks are pre-populated with libvpx
    /// defaults; loop-filter deltas are all zero and disabled.
    #[must_use]
    pub fn new() -> Self {
        // Default::default() doesn't work on a 4-array if FrameContext
        // isn't Copy (it isn't — it contains a large coef_probs table).
        // Build each bank explicitly.
        let frame_contexts = [
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
            FrameContext::new_default(),
        ];
        Self {
            frame_pool: DecoderFramePool::new(),
            ref_frame_map: Default::default(),
            ref_frame_sizes: [None; rvp9_core::bitstream::REF_FRAMES],
            frame_contexts,
            lf_deltas: ResolvedLfDeltas::default(),
            have_resync: true, // until we've seen any frame, treat as needing resync
            mv_counts: rvp9_core::entropy::mv_probs::NmvCounts::default(),
            coef_counts: rvp9_core::entropy::adapt::CoefCountsBank::default(),
            last_frame_was_key: false,
            mode_counts: rvp9_core::entropy::adapt::FrameCounts::default(),
        }
    }

    /// Borrow the decoder's frame pool (immutable).
    #[must_use]
    pub fn frame_pool(&self) -> &DecoderFramePool {
        &self.frame_pool
    }

    /// Borrow the decoder's frame pool (mutable). Used by tests and
    /// by the eventual pixel-reconstruction path.
    pub fn frame_pool_mut(&mut self) -> &mut DecoderFramePool {
        &mut self.frame_pool
    }

    /// Borrow a specific frame-context bank (0..=3).
    ///
    /// # Panics
    ///
    /// If `idx >= FRAME_CONTEXTS`.
    #[must_use]
    pub fn frame_context(&self, idx: usize) -> &FrameContext {
        &self.frame_contexts[idx]
    }

    /// `true` if the decoder is waiting for a keyframe (the initial
    /// state, plus after a bitstream resync). Inter frames received
    /// in this state are recognised but their compressed payloads
    /// are not consumed.
    #[must_use]
    pub fn needs_resync(&self) -> bool {
        self.have_resync
    }

    /// Parse one transport packet (an IVF chunk, RTP packet, etc.)
    /// into its 1..=8 component VP9 frames.
    ///
    /// For each component frame, parses the uncompressed header and
    /// returns the result via [`PacketFrames`]. The compressed
    /// portion of each frame is NOT parsed yet (Task 1e); the
    /// `compressed_data` slice in the returned [`ParsedFrame`] is the
    /// raw bytes following the uncompressed header.
    ///
    /// # Errors
    ///
    /// Propagates any [`BitstreamError`] from the superframe-index
    /// parse or the per-frame header parse.
    pub fn parse_packet<'a>(&mut self, data: &'a [u8]) -> Result<PacketFrames<'a>, BitstreamError> {
        let sf = parse_superframe_index(data)?;
        let mut frame_slices: Vec<&'a [u8]> = Vec::new();

        match sf {
            Superframe::Single => {
                frame_slices.push(data);
            }
            Superframe::Multi {
                sizes,
                count,
                index_size,
            } => {
                let payload_end = data.len() - index_size;
                let mut offset = 0;
                for &sz in sizes.iter().take(count) {
                    let sz = sz as usize;
                    if offset + sz > payload_end {
                        return Err(BitstreamError::InvalidSuperframeIndex {
                            reason:
                                rvp9_core::bitstream::SuperframeIndexErrorReason::TruncatedIndex {
                                    declared: offset + sz,
                                    available: payload_end,
                                },
                        });
                    }
                    frame_slices.push(&data[offset..offset + sz]);
                    offset += sz;
                }
            }
        }

        // Pass the running per-slot ref-frame sizes to the parser so
        // it can resolve `setup_frame_size_with_refs` (`vp9_decodeframe.c:1559-1574`)
        // when the header inherits size from a previous frame.
        let mut parsed: Vec<ParsedFrame<'a>> = Vec::with_capacity(frame_slices.len());
        for frame_data in frame_slices {
            let (header, header_end) =
                parse_uncompressed_header(frame_data, &self.ref_frame_sizes)?;
            // Keyframe and intra-only frames advertise a fresh width
            // and refresh some/all ref-frame-map slots; mirror libvpx's
            // `setup_frame_size` propagation (`vp9_decodeframe.c:2698`)
            // so subsequent inter frames in the same packet (or the
            // next packet) can resolve their sizes during parse.
            // Keyframes set every slot (`refresh_frame_flags = 0xFF` per
            // spec); intra-only carries an explicit 8-bit flag in the
            // header that we surface via `FrameKind::IntraOnly`.
            self.refresh_ref_frame_sizes(&header);
            parsed.push(ParsedFrame {
                header,
                header_end_byte: header_end,
                full_frame_data: frame_data,
            });
        }

        Ok(PacketFrames { frames: parsed })
    }

    /// Propagate the just-parsed header's frame size into the
    /// `ref_frame_sizes` cache, respecting the `refresh_frame_flags`
    /// mask. Mirrors the post-parse update libvpx performs at
    /// `vp9_decodeframe.c:2691` (keyframes set every slot) and the
    /// intra-only / inter mask handling at `:2730`.
    fn refresh_ref_frame_sizes(&mut self, header: &UncompressedHeader) {
        match header.kind {
            FrameKind::KeyFrame => {
                let size = Some((header.width, header.height));
                for slot in self.ref_frame_sizes.iter_mut() {
                    *slot = size;
                }
            }
            FrameKind::IntraOnly {
                refresh_frame_flags,
                ..
            } => {
                let size = (header.width, header.height);
                for (i, slot) in self.ref_frame_sizes.iter_mut().enumerate() {
                    if (refresh_frame_flags >> i) & 1 != 0 {
                        *slot = Some(size);
                    }
                }
            }
            FrameKind::InterFrame {
                refresh_frame_flags,
                ..
            } => {
                // Inter frames refresh slots per the mask. The width
                // and height stored here is the frame's decoded /
                // inherited (width, height) — see `parse_uncompressed
                // _header`. This lets subsequent inter frames in the
                // same packet resolve `setup_frame_size_with_refs`.
                let size = (header.width, header.height);
                for (i, slot) in self.ref_frame_sizes.iter_mut().enumerate() {
                    if (refresh_frame_flags >> i) & 1 != 0 {
                        *slot = Some(size);
                    }
                }
            }
            FrameKind::ShowExistingFrame { .. } => {}
        }
    }

    /// Update the running loop-filter deltas with a parsed
    /// `LoopFilterParams` from this frame's header. `None` slots in
    /// the params are interpreted as "keep previous"; `Some(v)`
    /// overlay onto the existing state.
    ///
    /// Mirrors libvpx's handling at `vp9_decodeframe.c:2819` (the
    /// `setup_loopfilter` block plus the running-state plumbing in
    /// `cm->lf`).
    pub fn merge_lf_params(&mut self, lf: &rvp9_core::bitstream::LoopFilterParams) {
        self.lf_deltas.enabled = lf.mode_ref_delta_enabled;
        if lf.mode_ref_delta_update {
            for (i, slot) in lf.ref_deltas.iter().enumerate() {
                if let Some(v) = slot {
                    self.lf_deltas.ref_deltas[i] = *v;
                }
            }
            for (i, slot) in lf.mode_deltas.iter().enumerate() {
                if let Some(v) = slot {
                    self.lf_deltas.mode_deltas[i] = *v;
                }
            }
        }
    }

    /// Read-only access to the running loop-filter delta state.
    /// Required by the per-frame filter-level resolver
    /// [`rvp9_core::loopfilter::resolve_filter_levels`].
    #[must_use]
    pub fn lf_deltas(&self) -> ResolvedLfDeltas {
        self.lf_deltas
    }

    /// Parse the compressed-header (first-partition) bytes of a
    /// non-show-existing frame into the decoder's selected
    /// `FrameContext` bank. Mirrors libvpx's call sequence at
    /// `vp9_decodeframe.c:2944-2948`:
    ///
    ///     read_uncompressed_header(...);
    ///     // ...
    ///     read_compressed_header(...);    // ← this method
    ///     // ...
    ///     decode_tiles(...);
    ///
    /// The selected bank is `frame_contexts[header.tail.frame_context_idx]`.
    /// The compressed-header byte slice is `frame.full_frame_data
    /// [header_end_byte .. header_end_byte + first_partition_size]`.
    ///
    /// The `compound_allowed` input to `parse_compressed_header` is
    /// conservatively set to `false` for now — proper POC-direction
    /// tracking lands with the reference-frame plumbing in Task 8.2+.
    /// This matches libvpx behaviour for keyframes / intra-only
    /// frames (where compound is always disallowed) but is
    /// conservative for inter frames.
    ///
    /// Returns `(tx_mode, reference_mode)` so the caller can wire
    /// them into per-block decode.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::WrongFrameKind`] if `frame.header.kind` is
    ///   `ShowExistingFrame` (those frames have no compressed
    ///   header).
    /// - [`DecodeError::TruncatedCompressedHeader`] if the declared
    ///   `first_partition_size` extends past the frame's data slice.
    /// - [`DecodeError::Bitstream`] for any error surfaced by
    ///   [`parse_compressed_header`] (marker-bit fail / read past end).
    pub fn parse_compressed_partition(
        &mut self,
        frame: &ParsedFrame<'_>,
    ) -> Result<CompressedHeaderResult, DecodeError> {
        let tail = frame.header.tail.as_ref().ok_or(DecodeError::WrongFrameKind)?;

        // Slice out [header_end .. header_end + first_partition_size].
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let total = frame.full_frame_data.len();
        let needed_end = header_end
            .checked_add(fps)
            .ok_or(DecodeError::TruncatedCompressedHeader {
                declared: usize::MAX,
                available: total,
            })?;
        if needed_end > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: needed_end,
                available: total,
            });
        }
        let compressed = &frame.full_frame_data[header_end..needed_end];

        // Compute parser options from the uncompressed header.
        let lossless = tail.quantization.is_lossless();
        let intra_only = frame.header.kind.is_intra();
        let (interp_filter, allow_high_precision_mv) = match frame.header.kind {
            FrameKind::InterFrame {
                interp_filter,
                allow_high_precision_mv,
                ..
            } => (interp_filter, allow_high_precision_mv),
            // Intra-only / keyframe: these fields are unused — pass
            // any sane defaults. `parse_compressed_header` ignores
            // them when `intra_only == true`.
            _ => (InterpFilter::EightTap, false),
        };
        let opts = CompressedHeaderOptions {
            lossless,
            intra_only,
            interp_filter,
            // Conservative pending POC tracking (Task 8.2+).
            compound_allowed: false,
            allow_high_precision_mv,
        };

        // Drive the parser against the selected FrameContext bank.
        let idx = tail.frame_context_idx as usize;
        let fc = &mut self.frame_contexts[idx];
        parse_compressed_header(compressed, fc, opts).map_err(DecodeError::Bitstream)
    }

    /// Handle a `show_existing_frame` parsed header by looking up
    /// the referenced slot in the frame pool and returning its
    /// [`FrameBuf`]. This is the simplest possible decoder output
    /// path — no compressed-header / tile decoding required.
    ///
    /// Mirrors libvpx's `vp9_decodeframe.c:2664-2679` early-exit:
    /// after recording the show-existing buffer index, libvpx
    /// returns the buffer at that slot.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::WrongFrameKind`] if `header.kind` is not
    ///   `FrameKind::ShowExistingFrame`.
    /// - [`DecodeError::EmptySlot`] if the referenced slot is empty
    ///   (libvpx's "Buffer N does not contain a decoded frame"
    ///   rejection at `vp9_decodeframe.c:2667`).
    pub fn decode_show_existing_frame(
        &self,
        header: &UncompressedHeader,
    ) -> Result<FrameBuf, DecodeError> {
        let slot = match header.kind {
            FrameKind::ShowExistingFrame { slot } => slot,
            _ => return Err(DecodeError::WrongFrameKind),
        };
        let slot_usize = slot as usize;
        self.frame_pool
            .get(slot_usize)
            .ok_or(DecodeError::EmptySlot { slot })
    }

    /// Drive one parsed frame end-to-end and return the decoded
    /// [`FrameBuf`] that should be shown (or `Ok(None)` for a
    /// not-shown frame whose payload still references something).
    ///
    /// This is the single entry point the conformance harness uses
    /// to walk a packet's worth of frames. Today only the
    /// `ShowExistingFrame` path is plumbed through; all other paths
    /// return [`DecodeError::Unimplemented`] with a short reason
    /// string. As the decoder fills in across Stage 3 (compressed-
    /// header parse, tile decode, pixel reconstruction, loopfilter)
    /// those branches will be replaced with real decode paths.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Unimplemented`] for every non-`ShowExistingFrame`
    ///   path until the wiring lands.
    /// - [`DecodeError::EmptySlot`] from the `ShowExistingFrame`
    ///   sub-path.
    pub fn decode_frame(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        match frame.header.kind {
            FrameKind::ShowExistingFrame { .. } => self.decode_show_existing_frame(&frame.header),
            FrameKind::KeyFrame => self.decode_keyframe(frame),
            FrameKind::IntraOnly { .. } => Err(DecodeError::Unimplemented {
                reason: "intra-only frame decode (compressed header + tile decode + reconstruction)",
            }),
            FrameKind::InterFrame { .. } => self.decode_inter_frame(frame),
        }
    }

    /// Decode a keyframe end-to-end, returning the freshly-allocated
    /// [`FrameBuf`] that holds the reconstructed pixels.
    ///
    /// Mirrors libvpx's keyframe decode driver from
    /// `vp9/decoder/vp9_decodeframe.c:read_compressed_header` →
    /// `decode_tiles` for the `tile_cols == 1 && tile_rows == 1` path
    /// only (multi-tile vectors surface as
    /// [`DecodeError::TileDecode`]).
    ///
    /// ## Stage K2 status
    ///
    /// - **K2.a (this commit)** wires the partition+tile walker, reads
    ///   per-block ModeInfo, and writes neighbour-context state.
    ///   Pixels are NOT yet reconstructed; the returned `FrameBuf`'s
    ///   plane contents are still the all-zero allocation.
    /// - **K2.b** adds per-plane prediction + token decode + IT-add.
    /// - **K2.c** wires loop-filter apply (gated on J landing).
    fn decode_keyframe(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        // Lossless decode (WHT) not yet wired in Task H — gracefully
        // return Unimplemented rather than letting H's debug_assert
        // fire downstream. See `notes/overnight-2026-05-17-stage3.md`
        // for the lossless follow-up tracking; this is an explicit
        // skip rather than a wedge.
        if tail.quantization.is_lossless() {
            return Err(DecodeError::Unimplemented {
                reason: "lossless decode (WHT inverse transform path) not yet wired",
            });
        }

        // 1. Parse the compressed header into the selected FrameContext
        //    bank to get `tx_mode` (and the coef-prob updates etc.).
        let compressed_result = self.parse_compressed_partition(frame)?;

        // 2. Slice out the tile data (everything past the compressed
        //    header).
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let tile_data_start = header_end + fps;
        let total = frame.full_frame_data.len();
        if tile_data_start > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: tile_data_start,
                available: total,
            });
        }
        let tile_section = &frame.full_frame_data[tile_data_start..];

        // 3. Compute frame mi-dimensions + tile-count.
        let mi_cols = ((frame.header.width as usize) + 7) >> 3;
        let mi_rows = ((frame.header.height as usize) + 7) >> 3;
        let tile_cols = 1u32 << tail.tile_info.log2_tile_cols;
        let tile_rows = 1u32 << tail.tile_info.log2_tile_rows;

        // 4. Reject multi-tile in v1.
        if tile_cols > 1 || tile_rows > 1 {
            return Err(DecodeError::TileDecode(TileDecodeError::MultiTileUnsupported {
                log2_tile_cols: tail.tile_info.log2_tile_cols,
                log2_tile_rows: tail.tile_info.log2_tile_rows,
            }));
        }

        // 5. Parse tile buffers (single tile: takes the whole tile
        //    section).
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        let tile_buf = buffers
            .first()
            .ok_or(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }))?;

        // 6. Allocate the destination FrameBuf.
        let mut yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;

        // 7. Drive the tile decode loop, writing into `yv12`.
        let bounds = tile_bounds(&tail.tile_info, mi_cols, mi_rows, 0, 0);
        // Per-segment alt-Q feature data: feature_data[seg][SEG_LVL_ALT_Q].
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if tail.segmentation.enabled {
            for (i, s) in tail.segmentation.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0]; // SEG_LVL_ALT_Q == 0
            }
        }
        let params = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: tail.segmentation.enabled && tail.segmentation.update_map,
            seg_tree_probs: tail.segmentation.tree_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: tail.segmentation.abs_delta,
        };
        // Stage 3 Task E.1.b — zero the coef counts at the start of
        // every frame and pass an accumulator into the tile decoder.
        // libvpx clears `cm->counts` in `read_uncompressed_header`
        // (`vp9_decodeframe.c:2784`) for every frame whose
        // `frame_parallel_decoding_mode == 0`. We don't model that
        // flag yet (treated as 0), so we zero unconditionally — matches
        // libvpx's behaviour on the single-decoder conformance gate.
        self.coef_counts = rvp9_core::entropy::adapt::CoefCountsBank::default();
        let fc_idx = tail.frame_context_idx as usize;
        let decode_ctx = {
            let fc = &self.frame_contexts[fc_idx];
            decode_tile_intra(
                tile_buf.data,
                bounds,
                &params,
                fc,
                &mut yv12,
                Some(&mut self.coef_counts),
            )
            .map_err(DecodeError::TileDecode)?
        };

        // Backward-adapt `coef_probs` per `vp9_adapt_coef_probs`
        // (`vp9_entropy.c:1084-1100`). Same gate as `adapt_mv_probs`:
        // fires whenever `refresh_frame_context` is on and the frame is
        // not error-resilient. For keyframes the kernel runs with the
        // `_KEY` constants — `intra_only = true` in `coef_adapt_params`.
        if tail.refresh_frame_context && !frame.header.error_resilient {
            let (count_sat, max_update_factor) =
                rvp9_core::entropy::adapt::coef_adapt_params(
                    /* intra_only = */ true,
                    self.last_frame_was_key,
                );
            let new_coef_probs = rvp9_core::entropy::adapt::adapt_coef_probs(
                &self.frame_contexts[fc_idx].coef_probs,
                &self.coef_counts,
                count_sat,
                max_update_factor,
            );
            self.frame_contexts[fc_idx].coef_probs = new_coef_probs;
        }

        // 8. K2.c — loop-filter apply driver. Skip when filter_level
        //    is zero (LF is a frame-level no-op — see libvpx
        //    `vp9_loopfilter.c:1422-1427`).
        //
        //    Keyframes invoke libvpx's `vp9_setup_past_independence`
        //    (`vp9_entropymode.c:424-469`), which resets the LF deltas
        //    to the keyframe defaults `{INTRA:+1, LAST:0, GOLDEN:-1,
        //    ALTREF:-1}` with `mode_ref_delta_enabled = true` BEFORE
        //    `setup_loopfilter` parses the header. We mirror that
        //    here so the header's "no update" path inherits the
        //    correct baseline rather than the all-zeros startup state.
        self.lf_deltas = ResolvedLfDeltas {
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
            enabled: true,
        };
        self.merge_lf_params(&tail.loop_filter);
        let filter_level = tail.loop_filter.filter_level;
        // Stage 3 stretch M.13 — `RVP9_DISABLE_LF=1` skips the loop-filter
        // apply driver, exposing pre-LF reconstruction for cascade
        // analysis (dump_frame_diff vs vpxdec's post-LF YUV).
        let lf_disabled = std::env::var("RVP9_DISABLE_LF").is_ok();
        if filter_level > 0 && !lf_disabled {
            apply_loop_filter_to_frame(
                &mut yv12,
                &decode_ctx,
                mi_rows,
                mi_cols,
                filter_level,
                tail.loop_filter.sharpness_level,
                self.lf_deltas,
                SegmentationLfView::from_segmentation(&tail.segmentation),
            );
        }

        // 9. Extend the frame's borders so future inter frames using
        //    this keyframe as a reference can read MC halos past the
        //    visible region (libvpx `vpx_extend_frame_borders_c`,
        //    `yv12extend.c:169`).
        extend_frame_borders(&mut yv12);

        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);

        // Note: ref_frame_sizes is already populated by
        // `parse_packet -> refresh_ref_frame_sizes`; no work here.

        // Keyframes refresh every ref-map slot (libvpx implicit
        // `refresh_frame_flags = 0xFF` for FRAME_TYPE_KEY at
        // `vp9_decodeframe.c:2691`). Splat the new FrameBuf across all
        // eight slots so subsequent inter frames can resolve any
        // ref_frame_indices[i] against this keyframe.
        for slot in &mut self.ref_frame_map {
            *slot = Some(frame_buf.clone());
        }

        self.have_resync = false;
        self.last_frame_was_key = true;
        Ok(frame_buf)
    }

    /// Decode an inter frame end-to-end, returning the freshly-
    /// allocated [`FrameBuf`] that holds the reconstructed pixels.
    ///
    /// Mirrors libvpx's inter-frame decode driver:
    /// `read_compressed_header` → `decode_tiles` (`vp9_decodeframe.c`)
    /// for the single-tile path. Multi-tile vectors surface as
    /// [`DecodeError::TileDecode`].
    ///
    /// ## Stage D.3 status
    ///
    /// - **D.3.a (this commit)** wires the partition+tile walker, the
    ///   per-block ModeInfo reader, and (via D.2) the per-inter-block
    ///   reconstruction primitive. Intra-in-inter blocks consume their
    ///   bitstream bits but skip pixel reconstruction (deferred to
    ///   D.4). Lossless and any frame referencing a ref slot whose
    ///   pool entry is `None` returns [`DecodeError::Unimplemented`].
    /// - **D.3.b** rewires the per-block refs from "single-ref LAST
    ///   only" to the actual decoded `ref_frames` pair.
    /// - **D.3.c** wires `extend_frame_borders` + `refresh_frame_flags`
    ///   handling.
    fn decode_inter_frame(&mut self, frame: &ParsedFrame<'_>) -> Result<FrameBuf, DecodeError> {
        let tail = frame
            .header
            .tail
            .as_ref()
            .ok_or(DecodeError::WrongFrameKind)?;

        let (
            ref_frame_indices,
            ref_frame_sign_biases,
            interp_filter,
            allow_high_precision_mv,
            refresh_frame_flags,
        ) = match frame.header.kind {
            FrameKind::InterFrame {
                ref_frame_indices,
                ref_frame_sign_biases,
                interp_filter,
                allow_high_precision_mv,
                refresh_frame_flags,
                ..
            } => (
                ref_frame_indices,
                ref_frame_sign_biases,
                interp_filter,
                allow_high_precision_mv,
                refresh_frame_flags,
            ),
            _ => return Err(DecodeError::WrongFrameKind),
        };

        // Lossless inter not yet wired (mirrors keyframe gating).
        if tail.quantization.is_lossless() {
            return Err(DecodeError::Unimplemented {
                reason: "lossless inter frame decode (WHT inverse transform path)",
            });
        }

        // 1. Resolve the three logical inter refs (LAST / GOLDEN /
        //    ALTREF) into actual FrameBuf handles via ref_frame_map.
        //    Empty slots surface as Unimplemented for now (libvpx would
        //    treat as bitstream-corrupt; in practice every real stream
        //    populates these via the prior keyframe).
        let mut refs: [Option<FrameBuf>; 3] = [None, None, None];
        for (i, &ref_idx) in ref_frame_indices.iter().enumerate() {
            let slot_idx = ref_idx as usize;
            let buf = self
                .ref_frame_map
                .get(slot_idx)
                .and_then(|slot| slot.clone());
            match buf {
                Some(b) => refs[i] = Some(b),
                None => {
                    return Err(DecodeError::Unimplemented {
                        reason: "inter frame referenced empty ref_frame_map slot",
                    });
                }
            }
        }

        // 2. Compute compound_allowed (libvpx
        //    `vp9_compound_reference_allowed`): true iff any two of
        //    the three sign biases differ.
        let compound_allowed = ref_frame_sign_biases[1] != ref_frame_sign_biases[0]
            || ref_frame_sign_biases[2] != ref_frame_sign_biases[0];

        // 3. Parse compressed header into the selected FC bank.
        let compressed_result = self.parse_compressed_partition_inter(
            frame,
            compound_allowed,
            interp_filter,
            allow_high_precision_mv,
        )?;

        // 4. Slice tile data (everything past the compressed header).
        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let tile_data_start = header_end + fps;
        let total = frame.full_frame_data.len();
        if tile_data_start > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: tile_data_start,
                available: total,
            });
        }
        let tile_section = &frame.full_frame_data[tile_data_start..];

        let mi_cols = ((frame.header.width as usize) + 7) >> 3;
        let mi_rows = ((frame.header.height as usize) + 7) >> 3;
        let tile_cols = 1u32 << tail.tile_info.log2_tile_cols;
        let tile_rows = 1u32 << tail.tile_info.log2_tile_rows;
        if tile_cols > 1 || tile_rows > 1 {
            return Err(DecodeError::TileDecode(TileDecodeError::MultiTileUnsupported {
                log2_tile_cols: tail.tile_info.log2_tile_cols,
                log2_tile_rows: tail.tile_info.log2_tile_rows,
            }));
        }
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        let tile_buf = buffers
            .first()
            .ok_or(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }))?;

        // 5. Allocate destination FrameBuf.
        let mut yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;

        let cur_y_width = yv12.aligned_y_width();
        let cur_y_height = yv12.aligned_y_height();

        // 6. Per-frame inter params.
        let sign_bias = sign_bias_from_header(&ref_frame_sign_biases);
        let (comp_fixed_ref, comp_var_ref) = setup_compound_reference_mode(&sign_bias);
        let inter_params = InterModeReadParams {
            allow_high_precision_mv,
            interp_filter,
            reference_mode: compressed_result.reference_mode,
            comp_fixed_ref,
            comp_var_ref,
            sign_bias,
        };

        // 7. Seg alt-Q feature data.
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if tail.segmentation.enabled {
            for (i, s) in tail.segmentation.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0];
            }
        }

        let base = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: tail.segmentation.enabled && tail.segmentation.update_map,
            seg_tree_probs: tail.segmentation.tree_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: tail.segmentation.abs_delta,
        };

        // Borrow refs as &Yv12Buffer for InterFrameDecodeParams. The
        // FrameBufs we cloned above keep the underlying allocations
        // alive for the lifetime of this scope.
        let refs_borrowed: [Option<&Yv12Buffer>; 3] = [
            refs[0].as_deref(),
            refs[1].as_deref(),
            refs[2].as_deref(),
        ];
        let params = InterFrameDecodeParams {
            base,
            cur_y_width,
            cur_y_height,
            refs: refs_borrowed,
            inter_params,
        };

        // Zero MV + coef + mode counts at start of frame, then decode
        // the tile. After the tile completes the counts hold this
        // frame's NEWMV component tallies + per-(tx, plane, ref)
        // coef-token tallies + per-block mode/partition/skip/inter/ref/
        // tx tallies, which feed `adapt_mv_probs` / `adapt_coef_probs`
        // / `adapt_mode_probs` below.
        self.mv_counts = rvp9_core::entropy::mv_probs::NmvCounts::default();
        self.coef_counts = rvp9_core::entropy::adapt::CoefCountsBank::default();
        self.mode_counts = rvp9_core::entropy::adapt::FrameCounts::default();
        let fc_idx = tail.frame_context_idx as usize;
        let bounds = tile_bounds(&tail.tile_info, mi_cols, mi_rows, 0, 0);
        {
            let fc = &self.frame_contexts[fc_idx];
            decode_tile_inter(
                tile_buf.data,
                bounds,
                &params,
                fc,
                &mut yv12,
                Some(&mut self.mv_counts),
                Some(&mut self.coef_counts),
            )
            .map_err(DecodeError::TileDecode)?;
        }

        // Backward-adapt the per-frame probability banks per
        // `vp9_adapt_*_probs` (`vp9_decodeframe.c:3046-3053`). Same gate
        // for all kernels: `refresh_frame_context && !error_resilient`.
        // `adapt_coef_probs` fires here on inter frames (the keyframe
        // path runs the kernel separately with the `_KEY` constants);
        // `adapt_mode_probs` / `adapt_mv_probs` only fire on
        // non-intra-only frames.
        if tail.refresh_frame_context && !frame.header.error_resilient {
            // mode_probs — libvpx `vp9_adapt_mode_probs`. Returns a
            // freshly built FrameContext with the mode / partition /
            // skip / tx / intra_inter / comp_inter / single_ref /
            // comp_ref / inter_mode / y_mode / uv_mode /
            // switchable_interp fields folded; `coef_probs` and `nmvc`
            // are copied through unchanged from `pre_fc` (see
            // `adapt_mode_probs` docs) — we re-install the coef / mv
            // adapts on top of this baseline below.
            let interp_switchable = matches!(
                params.inter_params.interp_filter,
                rvp9_core::bitstream::InterpFilter::Switchable
            );
            let tx_mode_select = matches!(
                params.base.tx_mode,
                rvp9_core::transforms::TxMode::TxModeSelect
            );
            let new_fc = rvp9_core::entropy::adapt::adapt_mode_probs(
                &self.frame_contexts[fc_idx],
                &self.mode_counts,
                interp_switchable,
                tx_mode_select,
            );
            self.frame_contexts[fc_idx] = new_fc;

            // coef_probs — inter-frame adapt (`intra_only = false`).
            // `coef_adapt_params` picks AFTER_KEY constants on the first
            // inter frame after a keyframe, default constants otherwise.
            let (count_sat, max_update_factor) =
                rvp9_core::entropy::adapt::coef_adapt_params(
                    /* intra_only = */ false,
                    self.last_frame_was_key,
                );
            let new_coef_probs = rvp9_core::entropy::adapt::adapt_coef_probs(
                &self.frame_contexts[fc_idx].coef_probs,
                &self.coef_counts,
                count_sat,
                max_update_factor,
            );
            self.frame_contexts[fc_idx].coef_probs = new_coef_probs;

            // nmvc — libvpx `vp9_adapt_mv_probs`. `allow_high_precision_mv`
            // gates the class0_hp / hp slots.
            let new_nmvc = rvp9_core::entropy::adapt::adapt_mv_probs(
                &self.frame_contexts[fc_idx].nmvc,
                &self.mv_counts,
                allow_high_precision_mv,
            );
            self.frame_contexts[fc_idx].nmvc = new_nmvc;
        }

        // D.3.c — frame finalisation:
        //
        // 1. Extend the frame's borders so future inter frames using
        //    this frame as a reference can read MC halos past the
        //    visible region (libvpx `vpx_extend_frame_borders_c`,
        //    `yv12extend.c:169`).
        extend_frame_borders(&mut yv12);

        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);

        // 2. Update `ref_frame_map` per `refresh_frame_flags` mask
        //    (libvpx `vp9_decodeframe.c:2914` — for each bit `i` set in
        //    the mask, `cm->ref_frame_map[i] = cm->new_fb_idx`). We
        //    splat the freshly-decoded FrameBuf into every slot the
        //    mask selects.
        for (i, slot) in self.ref_frame_map.iter_mut().enumerate() {
            if (refresh_frame_flags >> i) & 1 != 0 {
                *slot = Some(frame_buf.clone());
            }
        }

        self.last_frame_was_key = false;
        Ok(frame_buf)
    }

    /// Parse the compressed-header for an inter frame with the
    /// supplied `compound_allowed` / `interp_filter` /
    /// `allow_high_precision_mv` overrides. Companion to
    /// [`Decoder::parse_compressed_partition`] which conservatively
    /// pins `compound_allowed = false`.
    fn parse_compressed_partition_inter(
        &mut self,
        frame: &ParsedFrame<'_>,
        compound_allowed: bool,
        interp_filter: InterpFilter,
        allow_high_precision_mv: bool,
    ) -> Result<CompressedHeaderResult, DecodeError> {
        let tail = frame.header.tail.as_ref().ok_or(DecodeError::WrongFrameKind)?;

        let header_end = frame.header_end_byte;
        let fps = tail.first_partition_size as usize;
        let total = frame.full_frame_data.len();
        let needed_end = header_end
            .checked_add(fps)
            .ok_or(DecodeError::TruncatedCompressedHeader {
                declared: usize::MAX,
                available: total,
            })?;
        if needed_end > total {
            return Err(DecodeError::TruncatedCompressedHeader {
                declared: needed_end,
                available: total,
            });
        }
        let compressed = &frame.full_frame_data[header_end..needed_end];

        let opts = CompressedHeaderOptions {
            lossless: tail.quantization.is_lossless(),
            intra_only: false,
            interp_filter,
            compound_allowed,
            allow_high_precision_mv,
        };

        let idx = tail.frame_context_idx as usize;
        let fc = &mut self.frame_contexts[idx];
        parse_compressed_header(compressed, fc, opts).map_err(DecodeError::Bitstream)
    }
}

/// Mode-info side length of a 64×64 superblock (mirrors libvpx
/// `MI_BLOCK_SIZE` at `vp9_enums.h:25`).
const SB_MI_SIZE: usize = 8;

/// Drive the per-SB loop-filter mask build + frame-wide apply. Mirrors
/// the libvpx sequence in `vp9_loop_filter_frame` →
/// `loop_filter_rows` (`vp9_loopfilter.c:1422-1483`): build the
/// `[seg][ref][mode]` filter-level cube once per frame, walk each
/// superblock to populate its [`LoopFilterMask`] via `setup_mask` +
/// `adjust_mask`, then dispatch [`loop_filter_frame`] in place on the
/// reconstructed planes.
///
/// `decode_ctx` carries the per-tile (which equals per-frame for the
/// single-tile path) mi grid produced by [`decode_tile_intra`].
#[allow(clippy::too_many_arguments)]
fn apply_loop_filter_to_frame(
    yv12: &mut Yv12Buffer,
    decode_ctx: &TileDecodeCtx,
    mi_rows: usize,
    mi_cols: usize,
    filter_level: u8,
    sharpness_level: u8,
    lf_deltas: ResolvedLfDeltas,
    seg_view: SegmentationLfView,
) {
    let levels = resolve_filter_levels(filter_level, lf_deltas, seg_view);
    let info = LoopFilterInfo::new(sharpness_level);

    let sb_rows = mi_rows.div_ceil(SB_MI_SIZE);
    let sb_cols = mi_cols.div_ceil(SB_MI_SIZE);
    let mi_stride = decode_ctx.tile_mi_cols();
    let mi_grid = decode_ctx.mi_grid();

    // Materialise the full-frame ModeInfo grid (one entry per 8×8
    // luma cell, row-major with stride = mi_stride). `setup_mask`
    // expects a flat `&[ModeInfo]` indexed as `mi[r*stride + c]`,
    // mirroring libvpx's `cm->mi_grid_visible` pointer array
    // (`vp9_onyxc_int.h:mi_grid_visible`). For the single-tile path
    // the tile grid IS the frame grid.
    //
    // Cells the partition walker won't reach (past-frame quadrants,
    // bounded by `mi_rows` / `mi_cols`) get a placeholder; the
    // walker's `max_rows` / `max_cols` clamp guarantees those cells
    // are never read. Sub-leaf MI cells inside a >8×8 partition that
    // weren't explicitly written by the decoder (libvpx's "mi pointer
    // duplication" convention) likewise stay as the placeholder, but
    // `setup_mask` only reads the partition leaf's top-left MI at
    // each leaf boundary, so the duplication isn't required.
    let placeholder = mi_grid
        .iter()
        .find_map(|cell| cell.as_ref().cloned())
        .unwrap_or_else(default_padding_modeinfo);
    let mut frame_mi: Vec<ModeInfo> = Vec::with_capacity(mi_grid.len().max(1));
    for cell in mi_grid {
        match cell {
            Some(mi) => frame_mi.push(*mi),
            None => frame_mi.push(placeholder),
        }
    }

    let mut lfm_grid: Vec<LoopFilterMask> =
        vec![LoopFilterMask::default(); sb_rows * sb_cols];

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mi_row = sb_row * SB_MI_SIZE;
            let mi_col = sb_col * SB_MI_SIZE;

            let lfm = &mut lfm_grid[sb_row * sb_cols + sb_col];
            setup_mask(
                mi_row as u32,
                mi_col as u32,
                &frame_mi,
                mi_stride,
                mi_rows as u32,
                mi_cols as u32,
                &levels,
                lfm,
            );
            adjust_mask(
                mi_row as u32,
                mi_col as u32,
                mi_rows as u32,
                mi_cols as u32,
                lfm,
            );
            dump_lf_masks(lfm, &frame_mi, mi_stride, mi_rows, mi_cols, sb_row, sb_col);
        }
    }

    loop_filter_frame(
        yv12,
        &lfm_grid,
        sb_cols,
        sb_rows,
        sb_cols,
        mi_rows,
        mi_cols,
        &info,
        0,
        mi_rows,
    );
}

/// Stage 3 stretch M.10 diagnostic: dump the per-SB
/// [`LoopFilterMask`] + the visible mi-grid block-size tile, gated by
/// `RVP9_DUMP_LF_MASKS=1`. Used to audit K2.c's wireup against
/// libvpx's `vp9_setup_mask` + `vp9_adjust_mask` output. Cheap when
/// the env var is unset (one `std::env::var` lookup per SB).
fn dump_lf_masks(
    lfm: &LoopFilterMask,
    frame_mi: &[ModeInfo],
    mi_stride: usize,
    mi_rows: usize,
    mi_cols: usize,
    sb_row: usize,
    sb_col: usize,
) {
    if std::env::var("RVP9_DUMP_LF_MASKS").is_err() {
        return;
    }
    let mi_row = sb_row * SB_MI_SIZE;
    let mi_col = sb_col * SB_MI_SIZE;
    eprintln!(
        "SB ({sb_row},{sb_col}) mi=({mi_row},{mi_col}) mi_rows={mi_rows} mi_cols={mi_cols}"
    );
    eprintln!(
        "  left_y    = [{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
        lfm.left_y[0], lfm.left_y[1], lfm.left_y[2], lfm.left_y[3]
    );
    eprintln!(
        "  above_y   = [{:#018x}, {:#018x}, {:#018x}, {:#018x}]",
        lfm.above_y[0], lfm.above_y[1], lfm.above_y[2], lfm.above_y[3]
    );
    eprintln!("  int_4x4_y = {:#018x}", lfm.int_4x4_y);
    eprintln!(
        "  left_uv   = [{:#06x}, {:#06x}, {:#06x}, {:#06x}]",
        lfm.left_uv[0], lfm.left_uv[1], lfm.left_uv[2], lfm.left_uv[3]
    );
    eprintln!(
        "  above_uv  = [{:#06x}, {:#06x}, {:#06x}, {:#06x}]",
        lfm.above_uv[0], lfm.above_uv[1], lfm.above_uv[2], lfm.above_uv[3]
    );
    eprintln!("  int_4x4_uv= {:#06x}", lfm.int_4x4_uv);
    eprintln!("  lfl_y     = {:?}", lfm.lfl_y);
    eprintln!("  mi_grid block_size (visible only):");
    for r in 0..mi_rows {
        let mut row_str = String::new();
        for c in 0..mi_cols {
            let mi = &frame_mi[r * mi_stride + c];
            let tag = match mi.block_size {
                BlockSize::Block4x4 => "4x4 ",
                BlockSize::Block8x8 => "8x8 ",
                BlockSize::Block16x16 => "16x16",
                BlockSize::Block32x32 => "32x32",
                BlockSize::Block64x64 => "64x64",
                BlockSize::Block4x8 => "4x8 ",
                BlockSize::Block8x4 => "8x4 ",
                BlockSize::Block8x16 => "8x16",
                BlockSize::Block16x8 => "16x8",
                BlockSize::Block16x32 => "16x32",
                BlockSize::Block32x16 => "32x16",
                BlockSize::Block32x64 => "32x64",
                BlockSize::Block64x32 => "64x32",
            };
            row_str.push_str(&format!(" {tag}"));
        }
        eprintln!("    r{r}: {row_str}");
    }
}

/// Placeholder [`ModeInfo`] for the rare path where the entire mi
/// grid is empty (e.g. a degenerate tile that produced no leaves).
/// Mirrors libvpx's `vpx_calloc`-initialised `MODE_INFO` slot
/// (`vp9_onyxc_int.h:setup_mi`): an intra DC 8×8 block with
/// segment_id = 0. The partition walker in `setup_mask` only reads
/// this slot if `max_rows == 0 || max_cols == 0`, in which case the
/// walker early-exits before touching the data.
fn default_padding_modeinfo() -> ModeInfo {
    use rvp9_core::block::BlockSize;
    use rvp9_core::intra_pred::IntraMode;
    use rvp9_core::mode_info::BlockMode;
    use rvp9_core::transforms::TxSize;
    ModeInfo {
        block_size: BlockSize::Block8x8,
        tx_size: TxSize::Tx4x4,
        skip: false,
        segment_id: 0,
        seg_id_predicted: false,
        mode: BlockMode::Intra {
            y_mode: IntraMode::Dc,
            uv_mode: IntraMode::Dc,
        },
        bmi: Default::default(),
    }
}

/// Errors from the per-frame decode entry points.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Called against a header whose `kind` was inappropriate for
    /// the entry point. Examples:
    /// - `decode_show_existing_frame` on a non-`ShowExistingFrame`.
    /// - `parse_compressed_partition` on a `ShowExistingFrame`
    ///   (which has no `tail` / compressed header).
    WrongFrameKind,
    /// `decode_show_existing_frame` referenced an empty slot.
    /// Mirrors libvpx's `vp9_decodeframe.c:2667` corrupt-frame
    /// rejection ("Buffer N does not contain a decoded frame").
    EmptySlot {
        /// The 3-bit slot index that was requested.
        slot: u8,
    },
    /// `parse_compressed_partition` saw a `first_partition_size`
    /// that runs past the end of the frame's byte slice.
    TruncatedCompressedHeader {
        /// The declared end offset (`header_end + first_partition_size`).
        declared: usize,
        /// The frame's actual data length.
        available: usize,
    },
    /// A bitstream-layer error surfaced during compressed-header
    /// parsing. Carries the underlying `BitstreamError` for
    /// diagnostics.
    Bitstream(BitstreamError),
    /// A decode path that has not yet been implemented in this
    /// stage of the Stage 3 build-out. The `reason` describes which
    /// path. Used by [`Decoder::decode_frame`] for non-show-existing
    /// branches until the full pipeline lands.
    Unimplemented {
        /// Short human-readable description of the missing path.
        reason: &'static str,
    },
    /// A tile-decode-layer error surfaced during keyframe decode.
    /// Carries the underlying [`TileDecodeError`] for diagnostics.
    TileDecode(TileDecodeError),
    /// Failed to allocate the destination [`FrameBuf`]. Surfaces e.g.
    /// dimensions exceeding the safety cap in [`Yv12Buffer::alloc`].
    FrameAlloc,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongFrameKind => f.write_str(
                "frame kind is inappropriate for this entry point",
            ),
            Self::EmptySlot { slot } => {
                write!(f, "frame pool slot {slot} contains no decoded frame")
            }
            Self::TruncatedCompressedHeader {
                declared,
                available,
            } => write!(
                f,
                "compressed header declared {declared} bytes but only {available} available"
            ),
            Self::Bitstream(e) => write!(f, "bitstream error: {e}"),
            Self::Unimplemented { reason } => write!(f, "unimplemented: {reason}"),
            Self::TileDecode(e) => write!(f, "tile decode: {e}"),
            Self::FrameAlloc => f.write_str("frame allocation failed"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// One frame extracted from a transport packet by
/// [`Decoder::parse_packet`].
pub struct ParsedFrame<'a> {
    /// Parsed uncompressed-header struct.
    pub header: UncompressedHeader,
    /// Byte offset within `full_frame_data` where the uncompressed
    /// header ended and the compressed header begins. The
    /// `first_partition_size` field of `header.tail` (when present)
    /// tells you the size of the compressed-header partition.
    pub header_end_byte: usize,
    /// The frame's full byte slice (everything from `frame_marker` to
    /// the last tile byte). Tile data starts at
    /// `header_end_byte + header.tail.first_partition_size`.
    pub full_frame_data: &'a [u8],
}

/// Result of [`Decoder::parse_packet`]. Iterable over the 1..=8
/// frames found in the packet.
pub struct PacketFrames<'a> {
    frames: Vec<ParsedFrame<'a>>,
}

impl<'a> PacketFrames<'a> {
    /// Number of frames in the packet (`1..=8`).
    #[must_use]
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Borrow the parsed frames.
    #[must_use]
    pub fn frames(&self) -> &[ParsedFrame<'a>] {
        &self.frames
    }
}

impl<'a> IntoIterator for PacketFrames<'a> {
    type Item = ParsedFrame<'a>;
    type IntoIter = std::vec::IntoIter<ParsedFrame<'a>>;
    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::bitstream::{
        ColorSpace, FrameKind, Profile, RawBitWriter, frame::sync,
    };

    /// Build a minimal valid keyframe header buffer (returns the
    /// raw `&[u8]` payload). Mirrors the helper in
    /// `bitstream::header::tests`.
    fn build_minimal_keyframe() -> Vec<u8> {
        // 320 × 240 keyframe, profile 0, BT.601, studio.
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing = 0
        w.write_bit(0); // frame_type = KEY
        w.write_bit(1); // show_frame = 1
        w.write_bit(0); // error_resilient = 0
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(ColorSpace::Bt601 as u32, 3);
        w.write_bit(0); // color_range = Studio
        w.write_literal(319, 16); // width-1
        w.write_literal(239, 16); // height-1
        w.write_bit(0); // no render size

        // Tail: refresh_frame_context (1), frame_parallel_decoding_mode (1),
        // frame_context_idx (2), loopfilter (10), quant (11),
        // segmentation (1), tile cols (0 — width 320 → mi_cols 40
        // → sb64_cols 5 → min=max=0), tile rows (1), first_partition_size (16=1).
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(0, 2);
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        w.write_literal(0, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0); // log2_tile_rows = 0
        w.write_literal(1, 16); // first_partition_size = 1
        // Pad.
        for _ in 0..8 {
            w.write_literal(0, 8);
        }
        let used = w.bytes_written();
        buf.truncate(used + 8);
        buf
    }

    #[test]
    fn decoder_construction() {
        let dec = Decoder::new();
        assert_eq!(dec.frame_pool().occupied_count(), 0);
        // Default lf deltas: disabled, all zeros.
        let lf = dec.lf_deltas();
        assert!(!lf.enabled);
        assert_eq!(lf.ref_deltas, [0; 4]);
        assert_eq!(lf.mode_deltas, [0; 2]);
        // Default FRAME_CONTEXT is the libvpx-initialised bank.
        for i in 0..FRAME_CONTEXTS {
            assert!(dec.frame_context(i).initialized);
        }
    }

    #[test]
    fn parse_packet_single_keyframe() {
        let buf = build_minimal_keyframe();
        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&buf).expect("valid keyframe");
        assert_eq!(frames.count(), 1);
        let f = &frames.frames()[0];
        assert_eq!(f.header.kind, FrameKind::KeyFrame);
        assert_eq!(f.header.profile, Profile::Profile0);
        assert_eq!(f.header.width, 320);
        assert_eq!(f.header.height, 240);
        assert!(f.header_end_byte > 0);
        assert!(f.full_frame_data.len() >= f.header_end_byte);
    }

    /// Build a lossless keyframe with `first_partition_size = N`
    /// bytes of zero-padded compressed-header. The 1-byte minimal
    /// version produced by [`build_minimal_keyframe`] is too small
    /// for the bool reader to parse without crossing EOF; this
    /// helper gives the parser breathing room.
    fn build_lossless_keyframe_with_compressed_size(n: u16) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(0); // show_existing = 0
        w.write_bit(0); // frame_type = KEY
        w.write_bit(1); // show_frame = 1
        w.write_bit(0); // error_resilient = 0
        w.write_literal(sync::SYNC_CODE_0 as u32, 8);
        w.write_literal(sync::SYNC_CODE_1 as u32, 8);
        w.write_literal(sync::SYNC_CODE_2 as u32, 8);
        w.write_literal(ColorSpace::Bt601 as u32, 3);
        w.write_bit(0); // color_range = Studio
        w.write_literal(319, 16);
        w.write_literal(239, 16);
        w.write_bit(0); // no render size
        // Tail.
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(0, 2);
        w.write_literal(0, 6);
        w.write_literal(0, 3);
        w.write_bit(0);
        w.write_literal(0, 8);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_bit(0);
        w.write_literal(u32::from(n), 16);
        let used = w.bytes_written();
        buf.truncate(used + n as usize + 4);
        buf
    }

    /// `parse_compressed_partition` on a lossless keyframe with a
    /// 16-byte zero-padded compressed-header (plenty of room for
    /// the bool reader to issue all "no-update" gates):
    /// - Returns `tx_mode = OnlyTx4x4` (lossless forces this).
    /// - Returns `reference_mode = Single` (intra branch).
    /// - Leaves the selected FrameContext bank at its libvpx defaults.
    #[test]
    fn parse_compressed_partition_lossless_keyframe_no_updates() {
        use rvp9_core::bitstream::ReferenceMode;
        use rvp9_core::transforms::TxMode;

        let buf = build_lossless_keyframe_with_compressed_size(16);
        let mut dec = Decoder::new();

        let frames = dec.parse_packet(&buf).expect("valid keyframe");
        let header = frames.frames()[0].header.clone();
        let header_end = frames.frames()[0].header_end_byte;
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        drop(frames);

        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };

        let original = dec.frame_context(0).clone();
        let result = dec
            .parse_compressed_partition(&parsed)
            .expect("compressed header parse");

        assert_eq!(result.tx_mode, TxMode::OnlyTx4x4);
        assert_eq!(result.reference_mode, ReferenceMode::Single);
        assert_eq!(dec.frame_context(0), &original);
    }

    /// `parse_compressed_partition` on a `ShowExistingFrame` header
    /// fails with `WrongFrameKind` (those frames have no tail and
    /// no compressed header).
    #[test]
    fn parse_compressed_partition_show_existing_rejected() {
        // Build a tiny `show_existing_frame` packet: 1 byte with the
        // marker (0b10) + show_existing_flag (1) + slot (0..7=3 bits).
        let mut buf = vec![0u8; 16];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(1); // show_existing = 1
        w.write_literal(2, 3); // slot = 2
        let used = w.bytes_written();
        buf.truncate(used);

        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&buf).expect("valid show-existing");
        let header = frames.frames()[0].header.clone();
        let header_end = frames.frames()[0].header_end_byte;
        let frame_data: Vec<u8> = frames.frames()[0].full_frame_data.to_vec();
        drop(frames);

        let parsed = ParsedFrame {
            header,
            header_end_byte: header_end,
            full_frame_data: &frame_data,
        };
        let err = dec.parse_compressed_partition(&parsed).expect_err("must fail");
        assert_eq!(err, DecodeError::WrongFrameKind);
    }

    #[test]
    fn parse_packet_superframe_splits_into_frames() {
        // Build a 2-frame superframe: two copies of the keyframe.
        let k = build_minimal_keyframe();
        // Strip trailing padding to a real frame size (the header
        // parser doesn't care about trailing bytes, but the
        // superframe builder needs accurate sizes).
        let frame_size = k.len();
        // Build superframe: frame A | frame B | trailing index.
        let mut packet = Vec::new();
        packet.extend_from_slice(&k);
        packet.extend_from_slice(&k);
        // mag = 2 (since frame_size > 255 for this header). Build
        // the trailing index manually.
        let mag = 2u8;
        let frames_count = 2u8;
        let marker = 0xc0 | ((mag - 1) << 3) | (frames_count - 1);
        packet.push(marker);
        // Size fields (little-endian, mag bytes each).
        for _ in 0..frames_count {
            packet.push((frame_size & 0xff) as u8);
            packet.push(((frame_size >> 8) & 0xff) as u8);
        }
        packet.push(marker);

        let mut dec = Decoder::new();
        let frames = dec.parse_packet(&packet).expect("valid superframe");
        assert_eq!(frames.count(), 2);
        for f in frames.frames() {
            assert_eq!(f.header.kind, FrameKind::KeyFrame);
            assert_eq!(f.header.width, 320);
            assert_eq!(f.header.height, 240);
        }
    }

    #[test]
    fn merge_lf_params_applies_only_some_slots() {
        let mut dec = Decoder::new();
        let lf = rvp9_core::bitstream::LoopFilterParams {
            filter_level: 30,
            sharpness_level: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(2), None, Some(-3), None],
            mode_deltas: [None, Some(1)],
        };
        dec.merge_lf_params(&lf);
        let deltas = dec.lf_deltas();
        assert!(deltas.enabled);
        assert_eq!(deltas.ref_deltas[0], 2);
        assert_eq!(deltas.ref_deltas[1], 0); // None → keep previous (0).
        assert_eq!(deltas.ref_deltas[2], -3);
        assert_eq!(deltas.ref_deltas[3], 0);
        assert_eq!(deltas.mode_deltas[0], 0);
        assert_eq!(deltas.mode_deltas[1], 1);
    }

    /// Build a single byte-buffer for a `show_existing_frame` header
    /// pointing at the given slot. Profile 0, marker correct,
    /// show_existing = 1, then 3 bits for the slot.
    fn build_show_existing_frame(slot: u8) -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let mut w = RawBitWriter::new(&mut buf);
        w.write_literal(sync::FRAME_MARKER as u32, 2);
        w.write_bit(0); // profile low
        w.write_bit(0); // profile high → Profile 0
        w.write_bit(1); // show_existing_frame = 1
        w.write_literal(slot as u32, 3);
        // Pad.
        for _ in 0..4 {
            w.write_literal(0, 8);
        }
        let used = w.bytes_written();
        buf.truncate(used + 4);
        buf
    }

    #[test]
    fn decode_show_existing_frame_emits_slot() {
        use rvp9_core::bitstream::ChromaFormat;
        use rvp9_core::frame::{BorderUsage, ByteAlignment, FrameBuf, Yv12Buffer};
        use std::sync::Arc;

        let mut dec = Decoder::new();
        // Populate slot 4 with a frame.
        let f: FrameBuf = Arc::new(
            Yv12Buffer::alloc(
                320,
                240,
                ChromaFormat::Yuv420,
                BorderUsage::Decoder,
                ByteAlignment::Legacy,
            )
            .unwrap(),
        );
        dec.frame_pool_mut().assign_slot(4, f.clone()).unwrap();

        // Build a packet pointing at slot 4.
        let buf = build_show_existing_frame(4);
        let parsed = dec.parse_packet(&buf).expect("valid");
        assert_eq!(parsed.count(), 1);
        let frame = &parsed.frames()[0];
        assert_eq!(frame.header.kind, FrameKind::ShowExistingFrame { slot: 4 });

        // Decode emits the existing FrameBuf.
        let out = dec.decode_show_existing_frame(&frame.header).unwrap();
        assert!(Arc::ptr_eq(&out, &f));
    }

    #[test]
    fn decode_show_existing_frame_empty_slot_rejected() {
        let mut dec = Decoder::new();
        // No slots populated.
        let buf = build_show_existing_frame(2);
        let parsed = dec.parse_packet(&buf).unwrap();
        let frame = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&frame.header).unwrap_err();
        assert_eq!(err, DecodeError::EmptySlot { slot: 2 });
    }

    #[test]
    fn decode_show_existing_frame_rejects_non_show_existing() {
        let mut dec = Decoder::new();
        let buf = build_minimal_keyframe();
        let parsed = dec.parse_packet(&buf).unwrap();
        let kf = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&kf.header).unwrap_err();
        assert_eq!(err, DecodeError::WrongFrameKind);
    }

    /// Stage 3 stretch D.6: backward-adapt `nmvc` end-of-inter-frame
    /// shifts the bank away from defaults when counts accumulate.
    /// Verifies the plumbing: a synthetic count bump on the joint
    /// "NEWMV used horizontally" tally feeds `adapt_mv_probs` and
    /// pushes the `joints` probabilities toward libvpx's expected
    /// merged value.
    #[test]
    fn adapt_mv_probs_post_inter_frame_shifts_nmvc() {
        use rvp9_core::entropy::adapt::adapt_mv_probs;
        use rvp9_core::entropy::mv_probs::NmvCounts;
        let mut counts = NmvCounts::default();
        counts.joints[1] = 10; // 10 horizontal-only MV reads
        counts.joints[3] = 5; // 5 both-component reads
        let pre = rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
        let post = adapt_mv_probs(&pre, &counts, /* allow_hp */ true);
        // Some prob in joints[] must differ — counts are non-zero so
        // mode-merge moves the prob from its default.
        assert!(
            pre.joints != post.joints,
            "expected adapt_mv_probs to shift joints with non-zero counts"
        );
    }

    /// Stage 3 Task E.1.b: a fresh [`Decoder`] starts with all-zero
    /// `coef_counts` and `last_frame_was_key == false`. Tests the
    /// Decoder-level state invariants the adapt pipeline relies on.
    #[test]
    fn decoder_new_initializes_coef_counts_zero() {
        use rvp9_core::entropy::adapt::CoefCountsBank;
        let dec = Decoder::new();
        // Every coef cell is zero on construction.
        assert_eq!(dec.coef_counts, CoefCountsBank::default());
        // Every eob_branch cell is zero too (CoefCountsBank::default
        // sets both axes to zero — pin the round-trip).
        for tx in dec.coef_counts.coef.iter() {
            for plane in tx.iter() {
                for ref_ty in plane.iter() {
                    for band in ref_ty.iter() {
                        for ctx in band.iter() {
                            assert_eq!(ctx, &[0u32; 4]);
                        }
                    }
                }
            }
        }
        for tx in dec.coef_counts.eob_branch.iter() {
            for plane in tx.iter() {
                for ref_ty in plane.iter() {
                    for band in ref_ty.iter() {
                        for &cell in band.iter() {
                            assert_eq!(cell, 0);
                        }
                    }
                }
            }
        }
        assert!(!dec.last_frame_was_key);
    }

    /// Stage 3 Task E.1.b: `adapt_coef_probs` fed a [`CoefCountsBank`]
    /// with non-zero entries produces a different bank than the
    /// default. Verifies the adapt kernel sees the count tally we
    /// route to it, independently of the per-frame plumbing.
    #[test]
    fn adapt_coef_probs_with_nonzero_counts_shifts_probs() {
        use rvp9_core::entropy::adapt::{
            CoefCountsBank, adapt_coef_probs, coef_adapt_params,
        };
        let pre = FrameContext::new_default();
        let mut counts = CoefCountsBank::default();
        // Inject a single (tx=Tx4x4, plane=Y, ref=intra, band=1, ctx=0)
        // tally — n0=3, n1=1, n2=1, neob=5, eob_total=10 (same shape
        // as the adapt.rs hand-trace test).
        counts.coef[0][0][0][1][0][rvp9_core::entropy::ZERO_TOKEN] = 3;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::ONE_TOKEN] = 1;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::TWO_TOKEN] = 1;
        counts.coef[0][0][0][1][0][rvp9_core::entropy::EOB_MODEL_TOKEN] = 5;
        counts.eob_branch[0][0][0][1][0] = 10;

        let (cs, uf) = coef_adapt_params(true, false); // intra-only keyframe
        let post = adapt_coef_probs(&pre.coef_probs, &counts, cs, uf);
        // The injected slot must differ.
        assert_ne!(
            post[0][0][0][1][0], pre.coef_probs[0][0][0][1][0],
            "expected adapted (tx=4x4, plane=Y, intra, band=1, ctx=0) slot to differ"
        );
        // A slot we didn't touch should match the default.
        assert_eq!(
            post[3][1][1][5][5], pre.coef_probs[3][1][1][5][5],
            "untouched (tx=32x32, UV, inter, band=5, ctx=5) slot must be unchanged"
        );
    }

    /// Stage 3 Task E.1.b: synthetic counts fed directly through the
    /// adapt kernel + write-back sequence (mirroring the end-of-frame
    /// adapt step) shift the active FrameContext's `coef_probs` and
    /// leave the `coef_counts` field on the Decoder unchanged on a
    /// keyframe-success run. This verifies the integration handshake
    /// between the bank and the adapt entry point without depending
    /// on `build_minimal_keyframe` running a full conformance-class
    /// decode (the minimal-keyframe synthesis exists for header
    /// round-trips, not full tile execution).
    #[test]
    fn coef_counts_adapt_round_trip_via_decoder_fields() {
        use rvp9_core::entropy::adapt::{adapt_coef_probs, coef_adapt_params};
        let mut dec = Decoder::new();
        // Seed the bank — mirrors what the per-block plumbing would
        // do for a single (Tx4x4, Y, intra, band=2, ctx=1) decision
        // chain on a real keyframe. Use ZERO_TOKEN / ONE_TOKEN /
        // TWO_TOKEN / EOB_MODEL_TOKEN tallies on a non-default cell so
        // the adapt step can't return the input unchanged.
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ZERO_TOKEN] = 7;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ONE_TOKEN] = 3;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::TWO_TOKEN] = 1;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::EOB_MODEL_TOKEN] = 2;
        dec.coef_counts.eob_branch[0][0][0][2][1] = 5;

        // Run the keyframe adapt step inline — matches the post-tile
        // adapt block in `decode_keyframe`.
        let (cs, uf) = coef_adapt_params(/* intra_only = */ true, dec.last_frame_was_key);
        let pre_defaults = FrameContext::new_default().coef_probs;
        let new_probs =
            adapt_coef_probs(&dec.frame_contexts[0].coef_probs, &dec.coef_counts, cs, uf);
        dec.frame_contexts[0].coef_probs = new_probs;
        dec.last_frame_was_key = true;

        // Adapt must have shifted the seeded slot.
        assert_ne!(
            dec.frame_contexts[0].coef_probs[0][0][0][2][1], pre_defaults[0][0][0][2][1],
            "the seeded (tx=4x4, Y, intra, band=2, ctx=1) slot must shift after adapt"
        );
        // And `last_frame_was_key` is true now (driver responsibility).
        assert!(dec.last_frame_was_key);
    }

    /// Stage 3 Task E.1.c.1: a fresh [`Decoder`] starts with all-zero
    /// `mode_counts`. Mirrors the analogous coef-counts init test.
    #[test]
    fn decoder_new_initializes_mode_counts_zero() {
        use rvp9_core::entropy::adapt::FrameCounts;
        let dec = Decoder::new();
        assert_eq!(dec.mode_counts, FrameCounts::default());
    }

    /// Stage 3 Task E.1.c.1: feed synthetic `mode_counts` through
    /// `adapt_mode_probs` + write-back and verify the active
    /// FrameContext's mode-related fields shift. Mirrors the
    /// `coef_counts_adapt_round_trip_via_decoder_fields` pattern for
    /// the third (and final) adapt kernel.
    #[test]
    fn mode_counts_adapt_round_trip_via_decoder_fields() {
        use rvp9_core::entropy::adapt::adapt_mode_probs;
        let mut dec = Decoder::new();
        // Inject a non-zero skip[0] tally — synthesises one block per
        // skip-context decision. `adapt_mode_probs` uses
        // `mode_mv_merge_probs` for skip; non-zero counts ⇒ shift.
        dec.mode_counts.skip[0] = [4, 6];
        // Inject a non-zero intra_inter[1] tally too — exercises a
        // second `mode_mv_merge_probs` slot.
        dec.mode_counts.intra_inter[1] = [5, 3];

        let pre = dec.frame_contexts[0].clone();
        let new_fc = adapt_mode_probs(
            &dec.frame_contexts[0],
            &dec.mode_counts,
            /* interp_switchable = */ false,
            /* tx_mode_select   = */ false,
        );
        dec.frame_contexts[0] = new_fc;

        assert_ne!(
            dec.frame_contexts[0].skip_probs[0], pre.skip_probs[0],
            "skip_probs[0] must shift after adapt with non-zero counts"
        );
        assert_ne!(
            dec.frame_contexts[0].intra_inter_prob[1], pre.intra_inter_prob[1],
            "intra_inter_prob[1] must shift after adapt with non-zero counts"
        );
        // coef_probs / nmvc must be unchanged — adapt_mode_probs copies
        // them through (separate kernels handle those).
        assert_eq!(dec.frame_contexts[0].coef_probs, pre.coef_probs);
        assert_eq!(dec.frame_contexts[0].nmvc, pre.nmvc);
    }

    #[test]
    fn merge_lf_params_disabled_means_no_carry() {
        let mut dec = Decoder::new();
        // First frame: enable + set deltas.
        dec.merge_lf_params(&rvp9_core::bitstream::LoopFilterParams {
            filter_level: 30,
            sharpness_level: 0,
            mode_ref_delta_enabled: true,
            mode_ref_delta_update: true,
            ref_deltas: [Some(5), Some(5), Some(5), Some(5)],
            mode_deltas: [Some(5), Some(5)],
        });
        assert_eq!(dec.lf_deltas().ref_deltas, [5, 5, 5, 5]);

        // Second frame: disabled. Existing deltas survive but
        // `enabled` flips.
        dec.merge_lf_params(&rvp9_core::bitstream::LoopFilterParams::default());
        let deltas = dec.lf_deltas();
        assert!(!deltas.enabled);
        // Persistent storage preserved.
        assert_eq!(deltas.ref_deltas, [5, 5, 5, 5]);
    }
}
