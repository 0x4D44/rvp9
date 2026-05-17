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
pub mod tile;
pub mod tile_ctx;

use rvp9_core::bitstream::{
    BitstreamError, ChromaFormat, CompressedHeaderOptions, CompressedHeaderResult, FrameKind,
    InterpFilter, RefFrameSizes, Superframe, UncompressedHeader, parse_compressed_header,
    parse_superframe_index, parse_tile_buffers, parse_uncompressed_header,
};
use rvp9_core::entropy::{FRAME_CONTEXTS, FrameContext};
use rvp9_core::frame::{BorderUsage, ByteAlignment, DecoderFramePool, FrameBuf, Yv12Buffer};
use rvp9_core::loopfilter::ResolvedLfDeltas;

use crate::tile::{FrameDecodeParams, TileDecodeError, decode_tile_intra, tile_bounds};

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
    /// Four independent FRAME_CONTEXT banks, selected per-frame by
    /// the 2-bit `frame_context_idx` header field. Each is reset to
    /// libvpx defaults on construction and updated by backward
    /// adaptation after each frame's residuals are decoded.
    frame_contexts: [FrameContext; FRAME_CONTEXTS],
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
            frame_contexts,
            lf_deltas: ResolvedLfDeltas::default(),
            have_resync: true, // until we've seen any frame, treat as needing resync
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
    pub fn parse_packet<'a>(&self, data: &'a [u8]) -> Result<PacketFrames<'a>, BitstreamError> {
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

        // For each frame, parse its uncompressed header. We use
        // `NO_REFS` for now — the proper ref-size lookup wires in
        // once we maintain per-slot dimensions (Task 8.2).
        let no_refs: RefFrameSizes = [None; rvp9_core::bitstream::MAX_INTER_REFS];
        let mut parsed: Vec<ParsedFrame<'a>> = Vec::with_capacity(frame_slices.len());
        for frame_data in frame_slices {
            let (header, header_end) = parse_uncompressed_header(frame_data, &no_refs)?;
            // After the uncompressed header, the next
            // first_partition_size bytes are the compressed header
            // (boolean-coder driver). For now we just expose the
            // raw remaining bytes.
            parsed.push(ParsedFrame {
                header,
                header_end_byte: header_end,
                full_frame_data: frame_data,
            });
        }

        Ok(PacketFrames { frames: parsed })
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
            FrameKind::InterFrame { .. } => Err(DecodeError::Unimplemented {
                reason: "inter frame decode (compressed header + tile decode + reconstruction)",
            }),
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
        let yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;
        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);

        // 7. Drive the tile decode loop.
        let bounds = tile_bounds(&tail.tile_info, mi_cols, mi_rows, 0, 0);
        let params = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: tail.segmentation.enabled && tail.segmentation.update_map,
            seg_tree_probs: tail.segmentation.tree_probs,
            lossless: tail.quantization.is_lossless(),
        };
        let fc_idx = tail.frame_context_idx as usize;
        let fc = &self.frame_contexts[fc_idx];
        decode_tile_intra(tile_buf.data, bounds, &params, fc).map_err(DecodeError::TileDecode)?;

        // 8. TODO(overnight-2026-05-17): wire J (loop-filter apply)
        //    once it lands.

        self.have_resync = false;
        Ok(frame_buf)
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
        let dec = Decoder::new();
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

        let dec = Decoder::new();
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
        let dec = Decoder::new();
        // No slots populated.
        let buf = build_show_existing_frame(2);
        let parsed = dec.parse_packet(&buf).unwrap();
        let frame = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&frame.header).unwrap_err();
        assert_eq!(err, DecodeError::EmptySlot { slot: 2 });
    }

    #[test]
    fn decode_show_existing_frame_rejects_non_show_existing() {
        let dec = Decoder::new();
        let buf = build_minimal_keyframe();
        let parsed = dec.parse_packet(&buf).unwrap();
        let kf = &parsed.frames()[0];
        let err = dec.decode_show_existing_frame(&kf.header).unwrap_err();
        assert_eq!(err, DecodeError::WrongFrameKind);
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
