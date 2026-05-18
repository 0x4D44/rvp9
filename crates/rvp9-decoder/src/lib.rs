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
    /// Per-ref-slot collocated-MV grid. Parallel to
    /// [`Decoder::ref_frame_map`]; `mv_grids[i]` holds the MV grid that
    /// was written by the frame currently bound at ref slot `i`. Used
    /// to populate the `prev_mv` parameter of
    /// [`read_inter_block_mode_info`](rvp9_core::inter::mode_info::read_inter_block_mode_info)
    /// when `use_prev_frame_mvs` fires.
    ///
    /// Mirrors libvpx `cm->cur_frame->mvs` + the
    /// `cm->prev_frame = cm->cur_frame` swap that happens after each
    /// frame finishes decoding (`vp9_decoder.c:478`). We store one grid
    /// per ref slot rather than a single "prev_frame" pointer because
    /// the libvpx model swaps before `refresh_frame_flags` is applied,
    /// so the prev frame is just whatever was at LAST_FRAME's slot
    /// last cycle.
    mv_grids: [Option<std::sync::Arc<Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>>>;
        rvp9_core::bitstream::REF_FRAMES],
    /// Width / height of the most recently decoded frame. Used by
    /// `decode_inter_frame` to compute `use_prev_frame_mvs` per libvpx
    /// `vp9_decodeframe.c:2983`. `None` before any frame decodes.
    last_frame_size: Option<(u32, u32)>,
    /// `true` iff the most recently decoded frame was `intra_only`
    /// (libvpx `cm->last_intra_only`). Bypasses the prev-frame-MV
    /// salvage for the frame after an intra-only.
    last_intra_only: bool,
    /// `true` iff the most recently decoded frame was a shown frame
    /// (libvpx `cm->last_show_frame`). Always true for keyframes (which
    /// must be shown) and most inter frames; false for ALTREF /
    /// non-shown frames.
    last_show_frame: bool,
    /// Previous frame's per-`(mi_row, mi_col)` segment_id grid, row-
    /// major in 8×8 mi units at stride `last_frame_mi_cols`. Used by
    /// inter frames as the `predicted_segment_id` source for
    /// segmentation inheritance (libvpx
    /// `cm->last_frame_seg_map`, `vp9_decodemv.c:115-126`).
    ///
    /// D.10 (Wave 5) — populated at frame end by swapping the just-
    /// decoded `current_segment_ids` slab into this field. Keyframes
    /// and intra-only frames zero this field at frame end (mirroring
    /// libvpx's `vp9_setup_past_independence` at
    /// `vp9_entropymode.c:424-437`).
    ///
    /// Empty (`Vec::new`) before any frame has decoded; in that state
    /// the inter segment-id reader treats every cell as zero (matches
    /// libvpx `cm->last_frame_seg_map == NULL`).
    prev_segment_ids: Vec<u8>,
    /// `(mi_rows, mi_cols)` of `prev_segment_ids`. Used to detect
    /// resize / frame-dim mismatch — when the new frame's mi-dims
    /// differ, the prev-frame seg map is treated as all-zeros
    /// (matches libvpx, which reallocates `last_frame_seg_map` on
    /// resize and the new allocation is zeroed).
    prev_segment_ids_dims: (usize, usize),
    /// Persistent segmentation feature data across frames. libvpx's
    /// `setup_segmentation` (`vp9_decodeframe.c:1332-1383`) mutates
    /// `cm->seg` in place and only rewrites `feature_data` /
    /// `abs_delta` when the per-frame `update_data` bit is set; on
    /// frames with `update_data == 0` the feature data INHERITS from
    /// the previous frame. Our parser returns a fresh
    /// [`SegmentationParams`] per frame, so we maintain this running
    /// copy here and overlay it onto the parsed value when
    /// `update_data == false`. Needed for AQ-2 vectors where the
    /// keyframe writes the per-segment alt_Q (cyclic refresh) and
    /// every subsequent inter frame leaves the data untouched.
    seg_feature_data: [[Option<i16>; rvp9_core::bitstream::SEG_LVL_MAX];
        rvp9_core::bitstream::MAX_SEGMENTS],
    /// `abs_delta` companion to [`Self::seg_feature_data`]. Same
    /// inheritance semantics.
    seg_abs_delta: bool,
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
            mv_grids: Default::default(),
            last_frame_size: None,
            last_intra_only: false,
            last_show_frame: false,
            prev_segment_ids: Vec::new(),
            prev_segment_ids_dims: (0, 0),
            seg_feature_data: [[None; rvp9_core::bitstream::SEG_LVL_MAX];
                rvp9_core::bitstream::MAX_SEGMENTS],
            seg_abs_delta: false,
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
        //
        // **D.9 (cascade-fix preamble):** snapshot the bank *before* the
        // compressed-header diff-updates land, so the backward-adapt
        // pass below can use it as the `pre` baseline. libvpx's
        // `read_compressed_header` operates on a working copy `cm->fc`
        // (loaded from the bank at the top of `vp9_decode_frame`), and
        // `vp9_adapt_coef_probs` then reads `pre_fc = cm->frame_contexts
        // [frame_context_idx]` — the original bank value, *unaffected*
        // by the compressed-header deltas (`vp9_entropy.c:1058`). Our
        // [`parse_compressed_partition`] mutates the bank in place, so
        // without this snapshot the adapt pass would use the modified
        // bank as `pre`, causing the chained-merge baseline to drift
        // away from libvpx after every frame.
        let fc_idx = tail.frame_context_idx as usize;
        let pre_fc = self.frame_contexts[fc_idx].clone();
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

        // 4. Parse per-tile byte payloads. Stage 3 Task E.2 — wire up
        //    single-threaded multi-tile decode. libvpx splits the tile
        //    section by the 4-byte BE size prefix before every tile
        //    except the last (`get_tile_buffers`,
        //    `vp9_decodeframe.c:1649-1697`); we walk the resulting
        //    `Vec<TileBuffer>` in row-major order below.
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        if buffers.is_empty() {
            return Err(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }));
        }

        // 5. Allocate the destination FrameBuf.
        let mut yv12 = Yv12Buffer::alloc(
            frame.header.width,
            frame.header.height,
            ChromaFormat::Yuv420,
            BorderUsage::Decoder,
            ByteAlignment::Legacy,
        )
        .map_err(|_| DecodeError::FrameAlloc)?;

        // 6. Per-frame state — drive every tile through a shared
        //    accumulator (counts) + shared output (`yv12`,
        //    `cur_segment_ids`). Each tile gets its own fresh
        //    `TileDecodeCtx` / `TileCoefCtx` / `BoolReader` allocated
        //    inside `decode_tile_intra`. This mirrors libvpx
        //    `decode_tiles` (`vp9_decodeframe.c:2200-2240`) which
        //    constructs a fresh `TileWorkerData` per tile but writes
        //    to the shared `cm->cur_frame->yv12` and accumulates
        //    counts into `cm->counts`.
        // D.10 — resolve the per-frame segmentation against the
        // running state, then use the RESOLVED view everywhere
        // downstream (per-segment alt_Q lookup + the loop-filter
        // segmentation view). Without this, AQ-2 inter frames lose
        // the keyframe's `feature_data` (encoder sets it once at the
        // keyframe with `update_data = 1`, then leaves it alone on
        // every inter frame).
        let resolved_seg = self.resolve_segmentation(&tail.segmentation, /* is_keyframe = */ true);
        // Per-segment alt-Q feature data: feature_data[seg][SEG_LVL_ALT_Q].
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if resolved_seg.enabled {
            for (i, s) in resolved_seg.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0]; // SEG_LVL_ALT_Q == 0
            }
        }
        let params = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: resolved_seg.enabled && resolved_seg.update_map,
            segmentation_enabled: resolved_seg.enabled,
            segmentation_temporal_update: resolved_seg.temporal_update,
            seg_tree_probs: resolved_seg.tree_probs,
            seg_pred_probs: resolved_seg.pred_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: resolved_seg.abs_delta,
        };
        // Stage 3 Task E.1.b — zero the coef counts at the start of
        // every frame and pass an accumulator into the tile decoder.
        // libvpx clears `cm->counts` in `read_uncompressed_header`
        // (`vp9_decodeframe.c:2784`) for every frame whose
        // `frame_parallel_decoding_mode == 0`. We don't model that
        // flag yet (treated as 0), so we zero unconditionally — matches
        // libvpx's behaviour on the single-decoder conformance gate.
        self.coef_counts = rvp9_core::entropy::adapt::CoefCountsBank::default();
        // D.10 — allocate the current frame's segment_id slab. Keyframes
        // either write a fresh ID per block (when `update_map = true`)
        // or default every cell to 0. After decode finishes we wipe
        // `prev_segment_ids` (libvpx `vp9_setup_past_independence` at
        // `vp9_entropymode.c:424-437`) — keyframes RESET the seg map.
        let mut cur_segment_ids = vec![0u8; mi_rows * mi_cols];

        // Stage 3 Task E.2 — iterate `(tile_row, tile_col)` in raster
        // order, decoding each tile sequentially with a fresh per-tile
        // `TileDecodeCtx` + `TileCoefCtx` + `BoolReader` (allocated
        // inside `decode_tile_intra`). Per-tile state is intentionally
        // isolated — VP9 specifies that the `above_*` / `left_*`
        // entropy contexts reset at every tile boundary
        // (`vp9_decodeframe.c:2200-2240`). Keep the per-tile mi grids
        // so we can stitch them into a frame-wide mi grid for the LF
        // driver below.
        let mut tile_decode_ctxs: Vec<(tile::TileBounds, TileDecodeCtx)> =
            Vec::with_capacity(buffers.len());
        for tile_buf in &buffers {
            let bounds = tile_bounds(
                &tail.tile_info,
                mi_cols,
                mi_rows,
                tile_buf.row,
                tile_buf.col,
            );
            let fc = &self.frame_contexts[fc_idx];
            let decode_ctx = decode_tile_intra(
                tile_buf.data,
                bounds,
                &params,
                fc,
                &mut yv12,
                Some(&mut self.coef_counts),
                &mut cur_segment_ids,
            )
            .map_err(DecodeError::TileDecode)?;
            tile_decode_ctxs.push((bounds, decode_ctx));
        }

        // Backward-adapt `coef_probs` per `vp9_adapt_coef_probs`
        // (`vp9_entropy.c:1084-1100`). Same gate as `adapt_mv_probs`:
        // fires whenever `refresh_frame_context` is on and the frame is
        // not error-resilient. For keyframes the kernel runs with the
        // `_KEY` constants — `intra_only = true` in `coef_adapt_params`.
        //
        // D.9 — use `pre_fc` (snapshot taken before
        // `parse_compressed_partition`) as the `pre` baseline. libvpx's
        // `pre_fc = cm->frame_contexts[fc_idx]` carries the *prior*
        // adapt result, unaffected by this frame's compressed-header
        // diff-updates (see snapshot comment above).
        if tail.refresh_frame_context && !frame.header.error_resilient {
            let (count_sat, max_update_factor) =
                rvp9_core::entropy::adapt::coef_adapt_params(
                    /* intra_only = */ true,
                    self.last_frame_was_key,
                );
            let new_coef_probs = rvp9_core::entropy::adapt::adapt_coef_probs(
                &pre_fc.coef_probs,
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
            // Stage 3 Task E.2 — stitch per-tile mi grids into a
            // frame-wide grid for the LF mask builder. Mirrors libvpx
            // `cm->mi_grid_visible` which spans the entire frame
            // regardless of tile count.
            let tile_grids: Vec<(tile::TileBounds, &[Option<ModeInfo>])> =
                tile_decode_ctxs
                    .iter()
                    .map(|(b, c)| (*b, c.mi_grid()))
                    .collect();
            let frame_mi_grid = build_frame_mi_grid(mi_rows, mi_cols, &tile_grids);
            apply_loop_filter_to_frame(
                &mut yv12,
                &frame_mi_grid,
                mi_cols,
                mi_rows,
                mi_cols,
                filter_level,
                tail.loop_filter.sharpness_level,
                self.lf_deltas,
                SegmentationLfView::from_segmentation(&resolved_seg),
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
        // Keyframes don't populate the MV grid — libvpx never reads
        // `cm->cur_frame->mvs` for intra frames in `vp9_read_mode_info`
        // (the `frame_is_intra_only` early-return at
        // `vp9_decodemv.c:822`). Clear every ref-slot MV grid so the
        // first inter frame after this keyframe finds no salvageable
        // collocated MVs and falls through to the default zero-init
        // (matching libvpx `use_prev_frame_mvs == false` for
        // `cm->last_frame_type == KEY_FRAME`).
        for slot in &mut self.mv_grids {
            *slot = None;
        }

        self.have_resync = false;
        self.last_frame_was_key = true;
        self.last_frame_size = Some((frame.header.width, frame.header.height));
        self.last_intra_only = false; // keyframes are not intra-only
        self.last_show_frame = frame.header.show_frame;
        // D.10 — promote the keyframe's just-decoded seg map into
        // `prev_segment_ids` so the next inter frame's
        // `read_inter_segment_id` inherits from it (when
        // `update_map = false`). This mirrors libvpx's per-frame
        // `vp9_swap_current_and_last_seg_map` at
        // `vp9_decoder.c:479`, which fires after every (shown) decode
        // whose `seg.enabled == true`. The keyframe's
        // `vp9_setup_past_independence` at frame START already zeroed
        // both maps; we then decoded into `cur_segment_ids`, so the
        // swap-equivalent is just a move into `prev_segment_ids`.
        if std::env::var("RVP9_TRACE_SEG").is_ok() {
            let nz = cur_segment_ids.iter().filter(|&&v| v != 0).count();
            eprintln!(
                "[rvp9_seg] keyframe end: mi={}x{} non-zero seg cells = {} / {}",
                mi_rows, mi_cols, nz, cur_segment_ids.len()
            );
        }
        self.prev_segment_ids = cur_segment_ids;
        self.prev_segment_ids_dims = (mi_rows, mi_cols);
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

        if std::env::var("RVP9_TRACE_INTER_MV").is_ok() {
            eprintln!(
                "[rvp9_inter_frame] indices={:?} sign_biases={:?} refresh=0x{:02x} last_key={} last_size={:?}",
                ref_frame_indices, ref_frame_sign_biases, refresh_frame_flags,
                self.last_frame_was_key, self.last_frame_size,
            );
            for i in 0..rvp9_core::bitstream::REF_FRAMES {
                let row0: Option<Vec<u8>> = self.ref_frame_map[i].as_ref().map(|b| {
                    let yp = b.y_plane();
                    let stride = yp.stride();
                    yp.data()[..stride.min(8)].to_vec()
                });
                eprintln!("  ref_frame_map[{i}] row0={:?}", row0);
            }
        }

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

        // D.9 — snapshot the bank *before* compressed-header diff-updates
        // land, so the backward-adapt block below can use it as `pre_fc`
        // (libvpx `vp9_entropy.c:1058` / `vp9_entropymode.c:343-407`,
        // where `pre_fc = cm->frame_contexts[cm->frame_context_idx]`).
        // Our [`parse_compressed_partition_inter`] mutates the bank in
        // place — without the snapshot the adapt pass would chain-merge
        // from the bank's *post-diff* value, accumulating drift across
        // inter frames. This was the 32×32 "4 pass then cascade"
        // pattern: by frame 5 the probability tables had drifted enough
        // that the parent-mode bool-decode took a wrong branch, putting
        // the cursor out of sync and producing a wholly-wrong block at
        // mi=(1, 3).
        let fc_idx = tail.frame_context_idx as usize;
        let pre_fc = self.frame_contexts[fc_idx].clone();

        // 3. Parse compressed header into the selected FC bank.
        let compressed_result = self.parse_compressed_partition_inter(
            frame,
            compound_allowed,
            interp_filter,
            allow_high_precision_mv,
        )?;

        // D.16 — env-gated dump of FrameContext state immediately after
        // compressed-header parse, mirroring libvpx's RVP9_TRACE_FC
        // instrumentation in `vp9_decodeframe.c::read_compressed_header`.
        // Diff line-by-line vs the libvpx trace to localize any
        // bitstream-interpretation divergence at the per-frame FC
        // diff-update layer (separate from per-block decode bugs).
        if std::env::var("RVP9_TRACE_FC").ok().as_deref() == Some("1") {
            let fc = &self.frame_contexts[fc_idx];
            eprintln!(
                "RVP9_TRACE_FC compressed_header_done kf=0 intra_only=0"
            );
            eprintln!(
                "RVP9_TRACE_FC tx_mode={} ref_mode={}",
                compressed_result.tx_mode as u8,
                compressed_result.reference_mode as u8
            );
            eprintln!(
                "RVP9_TRACE_FC skip_probs=[{},{},{}]",
                fc.skip_probs[0], fc.skip_probs[1], fc.skip_probs[2]
            );
            eprintln!(
                "RVP9_TRACE_FC intra_inter_prob=[{},{},{},{}]",
                fc.intra_inter_prob[0],
                fc.intra_inter_prob[1],
                fc.intra_inter_prob[2],
                fc.intra_inter_prob[3]
            );
            eprintln!(
                "RVP9_TRACE_FC comp_inter_prob=[{},{},{},{},{}]",
                fc.comp_inter_prob[0], fc.comp_inter_prob[1],
                fc.comp_inter_prob[2], fc.comp_inter_prob[3],
                fc.comp_inter_prob[4]
            );
            for i in 0..5 {
                eprintln!(
                    "RVP9_TRACE_FC single_ref_prob[{i}]=[{},{}]",
                    fc.single_ref_prob[i][0], fc.single_ref_prob[i][1]
                );
            }
            eprintln!(
                "RVP9_TRACE_FC comp_ref_prob=[{},{},{},{},{}]",
                fc.comp_ref_prob[0], fc.comp_ref_prob[1],
                fc.comp_ref_prob[2], fc.comp_ref_prob[3],
                fc.comp_ref_prob[4]
            );
            for j in 0..7 {
                eprintln!(
                    "RVP9_TRACE_FC inter_mode_probs[{j}]=[{},{},{}]",
                    fc.inter_mode_probs[j][0],
                    fc.inter_mode_probs[j][1],
                    fc.inter_mode_probs[j][2]
                );
            }
            eprintln!(
                "RVP9_TRACE_FC nmvc.joints=[{},{},{}]",
                fc.nmvc.joints[0], fc.nmvc.joints[1], fc.nmvc.joints[2]
            );
            for i in 0..2 {
                let c = &fc.nmvc.comps[i];
                eprintln!(
                    "RVP9_TRACE_FC nmvc.comp[{i}].sign={} classes=[{},{},{},{},{},{},{},{},{},{}]",
                    c.sign,
                    c.classes[0], c.classes[1], c.classes[2], c.classes[3],
                    c.classes[4], c.classes[5], c.classes[6], c.classes[7],
                    c.classes[8], c.classes[9]
                );
                eprintln!(
                    "RVP9_TRACE_FC nmvc.comp[{i}].class0=[{}] bits=[{},{},{},{},{},{},{},{},{},{}]",
                    c.class0[0],
                    c.bits[0], c.bits[1], c.bits[2], c.bits[3],
                    c.bits[4], c.bits[5], c.bits[6], c.bits[7],
                    c.bits[8], c.bits[9]
                );
            }
        }

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
        // Stage 3 Task E.2 — wire up single-threaded multi-tile decode
        // for inter frames. Per-tile state (BoolReader / TileDecodeCtx
        // / TileCoefCtx) resets at every tile boundary; frame-level
        // accumulators (counts / cur_mv_grid / cur_segment_ids / yv12)
        // are shared across all tiles. Mirrors libvpx `decode_tiles`
        // (`vp9_decodeframe.c:2200-2240`).
        let buffers = parse_tile_buffers(tile_section, tile_cols, tile_rows)
            .map_err(DecodeError::Bitstream)?;
        if buffers.is_empty() {
            return Err(DecodeError::Bitstream(BitstreamError::Truncated {
                needed: 1,
                available: 0,
            }));
        }

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

        // 7. D.10 — resolve per-frame segmentation against the running
        // state (inherits `feature_data` / `abs_delta` when the parsed
        // value's `update_data == false`).
        let resolved_seg = self.resolve_segmentation(&tail.segmentation, /* is_keyframe = */ false);

        // Per-segment alt-Q from the RESOLVED feature data.
        let mut seg_alt_q = [None; rvp9_core::loopfilter::MAX_SEGMENTS];
        if resolved_seg.enabled {
            for (i, s) in resolved_seg.feature_data.iter().enumerate() {
                seg_alt_q[i] = s[0];
            }
        }

        let base = FrameDecodeParams {
            mi_cols,
            mi_rows,
            tx_mode: compressed_result.tx_mode,
            segmentation_update_map: resolved_seg.enabled && resolved_seg.update_map,
            segmentation_enabled: resolved_seg.enabled,
            segmentation_temporal_update: resolved_seg.temporal_update,
            seg_tree_probs: resolved_seg.tree_probs,
            seg_pred_probs: resolved_seg.pred_probs,
            lossless: tail.quantization.is_lossless(),
            base_qindex: i32::from(tail.quantization.base_qindex),
            y_dc_delta_q: i32::from(tail.quantization.y_dc_delta_q),
            uv_dc_delta_q: i32::from(tail.quantization.uv_dc_delta_q),
            uv_ac_delta_q: i32::from(tail.quantization.uv_ac_delta_q),
            seg_alt_q,
            seg_abs_delta: resolved_seg.abs_delta,
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

        // D.7 — compute `use_prev_frame_mvs` per libvpx
        // `vp9_decodeframe.c:2983-2986`:
        //   !error_resilient && width == last_width && height ==
        //   last_height && !last_intra_only && last_show_frame &&
        //   last_frame_type != KEY_FRAME.
        //
        // If the gate fires, look up the LAST_FRAME ref slot's MV grid
        // and pass it to the tile decoder so `find_mv_refs` can salvage
        // collocated MV candidates. Otherwise pass `None` (matching
        // pre-D.7 behaviour).
        let use_prev_frame_mvs = !frame.header.error_resilient
            && self
                .last_frame_size
                .map(|(w, h)| w == frame.header.width && h == frame.header.height)
                .unwrap_or(false)
            && !self.last_intra_only
            && self.last_show_frame
            && !self.last_frame_was_key;
        let prev_mv_grid: Option<&[rvp9_core::inter::mv_ref::PrevFrameMvRef]> =
            if use_prev_frame_mvs {
                // libvpx pulls `cm->prev_frame` which is the just-
                // decoded frame's buffer, swapped from `cm->cur_frame`.
                // For our model that's whichever frame the last cycle
                // ended up storing — equivalent to the LAST_FRAME ref
                // slot's grid (slot index = ref_frame_indices[0]).
                let last_slot = ref_frame_indices[0] as usize;
                self.mv_grids
                    .get(last_slot)
                    .and_then(|s| s.as_ref())
                    .map(|g| g.as_slice())
            } else {
                None
            };
        // Verify the prev grid is correctly sized for this frame.
        let prev_mv_grid = prev_mv_grid.filter(|g| g.len() >= mi_rows * mi_cols);
        // Allocate fresh current-frame MV grid (zero-init = all
        // `(NoRefFrame, NoRefFrame)` entries with zero MVs).
        let mut cur_mv_grid = vec![
            rvp9_core::inter::mv_ref::PrevFrameMvRef::default();
            mi_rows * mi_cols
        ];

        // D.10 — populate `prev_segment_ids` view for segment-id
        // inheritance and allocate the current frame's seg-id slab.
        //
        // Libvpx's `read_inter_segment_id` reads `cm->last_frame_seg_map`;
        // when that pointer is NULL (first frame after past-independence
        // reset, or right after allocation) every cell behaves as zero
        // (`predicted_segment_id = cm->last_frame_seg_map ? ... : 0` at
        // `vp9_decodemv.c:156-159`). We model NULL as an empty slice.
        //
        // Resize check: if the prev seg map's dims don't match this
        // frame's mi-dims, treat it as missing — libvpx reallocates +
        // zero-fills `last_frame_seg_map` whenever mi_rows / mi_cols
        // change (`vp9_alloccommon.c:vp9_alloc_context_buffers`).
        let prev_seg_ids_slice: &[u8] =
            if self.prev_segment_ids_dims == (mi_rows, mi_cols)
                && self.prev_segment_ids.len() == mi_rows * mi_cols
            {
                &self.prev_segment_ids
            } else {
                &[]
            };
        let mut cur_segment_ids = vec![0u8; mi_rows * mi_cols];

        // Stage 3 Task E.2 — iterate `(tile_row, tile_col)` in raster
        // order. Each tile is decoded sequentially with its own fresh
        // `BoolReader` / `TileDecodeCtx` / `TileCoefCtx` (allocated
        // inside `decode_tile_inter`). Shared frame-level state
        // (`yv12`, `cur_mv_grid`, `cur_segment_ids`, counts) accumulates
        // across all tiles. Per-tile mi grids are collected here so
        // we can stitch them into a frame-wide mi grid for the LF
        // driver below.
        let mut tile_decode_ctxs: Vec<(tile::TileBounds, TileDecodeCtx)> =
            Vec::with_capacity(buffers.len());
        for tile_buf in &buffers {
            let bounds = tile_bounds(
                &tail.tile_info,
                mi_cols,
                mi_rows,
                tile_buf.row,
                tile_buf.col,
            );
            let fc = &self.frame_contexts[fc_idx];
            let decode_ctx = decode_tile_inter(
                tile_buf.data,
                bounds,
                &params,
                fc,
                &mut yv12,
                Some(&mut self.mv_counts),
                Some(&mut self.coef_counts),
                Some(&mut self.mode_counts),
                prev_mv_grid,
                &mut cur_mv_grid,
                prev_seg_ids_slice,
                &mut cur_segment_ids,
            )
            .map_err(DecodeError::TileDecode)?;
            tile_decode_ctxs.push((bounds, decode_ctx));
        }

        // Backward-adapt the per-frame probability banks per
        // `vp9_adapt_*_probs` (`vp9_decodeframe.c:3046-3053`). Same gate
        // for all kernels: `refresh_frame_context && !error_resilient`.
        // `adapt_coef_probs` fires here on inter frames (the keyframe
        // path runs the kernel separately with the `_KEY` constants);
        // `adapt_mode_probs` / `adapt_mv_probs` only fire on
        // non-intra-only frames.
        //
        // D.9 — use `pre_fc` (snapshot taken before
        // `parse_compressed_partition_inter`) as the `pre` baseline.
        // libvpx's adapt kernels read `pre_fc = cm->frame_contexts
        // [frame_context_idx]` — the bank value at frame *start*,
        // *unaffected* by the compressed-header diff-updates that
        // `read_compressed_header` applies to its working copy `cm->fc`.
        // Because [`parse_compressed_partition_inter`] mutates the bank
        // in place, reading `&self.frame_contexts[fc_idx]` here would
        // give the bank *plus* this frame's diff updates — drifting
        // the chained-merge baseline away from libvpx each frame.
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
                &pre_fc,
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
                &pre_fc.coef_probs,
                &self.coef_counts,
                count_sat,
                max_update_factor,
            );
            self.frame_contexts[fc_idx].coef_probs = new_coef_probs;

            // nmvc — libvpx `vp9_adapt_mv_probs`. `allow_high_precision_mv`
            // gates the class0_hp / hp slots.
            let new_nmvc = rvp9_core::entropy::adapt::adapt_mv_probs(
                &pre_fc.nmvc,
                &self.mv_counts,
                allow_high_precision_mv,
            );
            self.frame_contexts[fc_idx].nmvc = new_nmvc;
        }

        // D.8 — loop-filter apply (inter path). Mirrors the keyframe
        // wireup at the top of `decode_keyframe`: build per-SB filter
        // masks via `setup_mask` + `adjust_mask`, then dispatch
        // `loop_filter_frame`. libvpx applies the LF for inter frames
        // through the same `vp9_loopfilter.c:vp9_loop_filter_frame`
        // entry point as keyframes (`vp9_decodeframe.c:2867`).
        //
        // Without this wireup the inter pre-LF reconstruction was
        // being MD5'd against libvpx's post-LF dump — a per-frame +5/-5
        // edge-smoothing diff cascading across the whole frame. Same
        // pattern as M.9's "filter_level > 0 keyframes fail" finding,
        // here for the inter path.
        self.merge_lf_params(&tail.loop_filter);
        let filter_level = tail.loop_filter.filter_level;
        let lf_disabled = std::env::var("RVP9_DISABLE_LF").is_ok();
        if filter_level > 0 && !lf_disabled {
            // Stage 3 Task E.2 — stitch per-tile mi grids into a
            // frame-wide grid for the LF mask builder. Mirrors libvpx
            // `cm->mi_grid_visible` which spans the entire frame
            // regardless of tile count.
            let tile_grids: Vec<(tile::TileBounds, &[Option<ModeInfo>])> =
                tile_decode_ctxs
                    .iter()
                    .map(|(b, c)| (*b, c.mi_grid()))
                    .collect();
            let frame_mi_grid = build_frame_mi_grid(mi_rows, mi_cols, &tile_grids);
            apply_loop_filter_to_frame(
                &mut yv12,
                &frame_mi_grid,
                mi_cols,
                mi_rows,
                mi_cols,
                filter_level,
                tail.loop_filter.sharpness_level,
                self.lf_deltas,
                SegmentationLfView::from_segmentation(&resolved_seg),
            );
        }

        // D.3.c — frame finalisation:
        //
        // 1. Extend the frame's borders so future inter frames using
        //    this frame as a reference can read MC halos past the
        //    visible region (libvpx `vpx_extend_frame_borders_c`,
        //    `yv12extend.c:169`).
        extend_frame_borders(&mut yv12);

        let frame_buf: FrameBuf = std::sync::Arc::new(yv12);
        let cur_mv_grid_arc: std::sync::Arc<
            Vec<rvp9_core::inter::mv_ref::PrevFrameMvRef>,
        > = std::sync::Arc::new(cur_mv_grid);

        // 2. Update `ref_frame_map` per `refresh_frame_flags` mask
        //    (libvpx `vp9_decodeframe.c:2914` — for each bit `i` set in
        //    the mask, `cm->ref_frame_map[i] = cm->new_fb_idx`). We
        //    splat the freshly-decoded FrameBuf into every slot the
        //    mask selects. The parallel `mv_grids` array gets the same
        //    splat so the next inter frame can salvage collocated MVs
        //    via `use_prev_frame_mvs`.
        for (i, (slot, mv_slot)) in self
            .ref_frame_map
            .iter_mut()
            .zip(self.mv_grids.iter_mut())
            .enumerate()
        {
            if (refresh_frame_flags >> i) & 1 != 0 {
                *slot = Some(frame_buf.clone());
                *mv_slot = Some(cur_mv_grid_arc.clone());
            }
        }

        self.last_frame_was_key = false;
        self.last_frame_size = Some((frame.header.width, frame.header.height));
        self.last_intra_only = false;
        self.last_show_frame = frame.header.show_frame;
        // D.10 — swap the just-decoded segment_id slab into
        // `prev_segment_ids` for the next frame's inheritance reads.
        // Mirrors libvpx's seg-map double-buffer at end-of-frame
        // (`vp9_alloccommon.c:vp9_swap_current_and_last_seg_map`).
        if std::env::var("RVP9_TRACE_SEG").is_ok() {
            let nz = cur_segment_ids.iter().filter(|&&v| v != 0).count();
            eprintln!(
                "[rvp9_seg] inter end: mi={}x{} non-zero seg cells = {} / {}",
                mi_rows, mi_cols, nz, cur_segment_ids.len()
            );
        }
        self.prev_segment_ids = cur_segment_ids;
        self.prev_segment_ids_dims = (mi_rows, mi_cols);
        Ok(frame_buf)
    }

    /// D.10 — resolve the per-frame segmentation by overlaying the
    /// running `seg_feature_data` / `seg_abs_delta` onto the parsed
    /// `tail.segmentation` when this frame did NOT update those fields
    /// (`update_data == false`). Also refreshes the running copy when
    /// this frame DID update them. Mirrors libvpx's in-place mutation
    /// of `cm->seg` in `setup_segmentation`
    /// (`vp9_decodeframe.c:1332-1383`).
    ///
    /// Keyframes always start fresh: libvpx's
    /// `vp9_setup_past_independence` calls `vp9_clearall_segfeatures`
    /// before `setup_segmentation` runs, so any prior feature data is
    /// wiped. Callers should pass `is_keyframe = true` to mirror that.
    fn resolve_segmentation(
        &mut self,
        parsed: &rvp9_core::bitstream::SegmentationParams,
        is_keyframe: bool,
    ) -> rvp9_core::bitstream::SegmentationParams {
        if is_keyframe {
            self.seg_feature_data =
                [[None; rvp9_core::bitstream::SEG_LVL_MAX]; rvp9_core::bitstream::MAX_SEGMENTS];
            self.seg_abs_delta = false;
        }

        let mut resolved = parsed.clone();
        if resolved.enabled {
            if resolved.update_data {
                // This frame's parser already filled `feature_data` /
                // `abs_delta`; refresh our running copy from it.
                self.seg_feature_data = resolved.feature_data;
                self.seg_abs_delta = resolved.abs_delta;
            } else {
                // Inherit from the running state.
                resolved.feature_data = self.seg_feature_data;
                resolved.abs_delta = self.seg_abs_delta;
            }
        }
        resolved
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

/// Stitch per-tile mi grids into a single frame-level mi grid suitable
/// for downstream consumers (loop-filter mask builder, etc.).
///
/// `tile_grids` is a row-major list of `(bounds, tile_grid)` covering
/// the frame's `tile_rows × tile_cols` tiles. Each `tile_grid` is the
/// row-major `Option<ModeInfo>` slab returned by
/// [`tile_ctx::TileDecodeCtx::mi_grid`], sized
/// `(bounds.mi_row_end - bounds.mi_row_start) * (bounds.mi_col_end -
/// bounds.mi_col_start)` and addressed at stride =
/// `bounds.mi_col_end - bounds.mi_col_start`.
///
/// Returns a frame-wide `Vec<Option<ModeInfo>>` of length
/// `mi_rows * mi_cols` at stride `mi_cols`, where each cell's value
/// comes from the tile that covers `(mi_row, mi_col)`. Cells outside
/// any tile (shouldn't happen for valid `TileInfo`) stay `None`.
///
/// Mirrors libvpx's `cm->mi_grid_visible` (`vp9_onyxc_int.h:
/// mi_grid_visible`) which is allocated frame-wide and populated by
/// each tile's `vp9_decodeframe.c:decode_partition` calls writing
/// through the frame-wide stride. Our `TileDecodeCtx::write_block`
/// stores at tile-local stride; this stitcher merges them back to
/// frame stride after the tile loop completes.
fn build_frame_mi_grid(
    mi_rows: usize,
    mi_cols: usize,
    tile_grids: &[(tile::TileBounds, &[Option<ModeInfo>])],
) -> Vec<Option<ModeInfo>> {
    let mut frame: Vec<Option<ModeInfo>> = vec![None; mi_rows * mi_cols];
    for (bounds, tile_grid) in tile_grids {
        let tile_mi_cols = bounds.mi_col_end - bounds.mi_col_start;
        let tile_mi_rows = bounds.mi_row_end - bounds.mi_row_start;
        debug_assert_eq!(tile_grid.len(), tile_mi_cols * tile_mi_rows);
        for r in 0..tile_mi_rows {
            let frame_row = bounds.mi_row_start + r;
            if frame_row >= mi_rows {
                break;
            }
            let src_base = r * tile_mi_cols;
            let dst_base = frame_row * mi_cols + bounds.mi_col_start;
            let copy_cols = tile_mi_cols.min(mi_cols - bounds.mi_col_start);
            frame[dst_base..dst_base + copy_cols]
                .copy_from_slice(&tile_grid[src_base..src_base + copy_cols]);
        }
    }
    frame
}

/// Drive the per-SB loop-filter mask build + frame-wide apply. Mirrors
/// the libvpx sequence in `vp9_loop_filter_frame` →
/// `loop_filter_rows` (`vp9_loopfilter.c:1422-1483`): build the
/// `[seg][ref][mode]` filter-level cube once per frame, walk each
/// superblock to populate its [`LoopFilterMask`] via `setup_mask` +
/// `adjust_mask`, then dispatch [`loop_filter_frame`] in place on the
/// reconstructed planes.
///
/// `mi_grid` is the frame-level row-major `Option<ModeInfo>` slab
/// (stride = `mi_stride = mi_cols`). For single-tile frames this is
/// the tile's grid; for multi-tile frames it's stitched together by
/// [`build_frame_mi_grid`] across all tiles. Mirrors libvpx
/// `cm->mi_grid_visible` (`vp9_onyxc_int.h:mi_grid_visible`) which
/// spans the entire frame regardless of tile count.
#[allow(clippy::too_many_arguments)]
fn apply_loop_filter_to_frame(
    yv12: &mut Yv12Buffer,
    mi_grid: &[Option<ModeInfo>],
    mi_stride: usize,
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

    /// Stage 3 stretch D.9: the backward-adapt `pre` baseline must be
    /// the FrameContext value *before* the compressed-header diff-updates
    /// land, not after. Mirrors libvpx's
    /// `pre_fc = cm->frame_contexts[cm->frame_context_idx]` in
    /// `adapt_coef_probs` (`vp9_entropy.c:1058`) — `pre_fc` reads from
    /// `cm->frame_contexts[fc_idx]` while `cm->fc` (the working copy
    /// that the compressed-header parser mutates) is separate. Without
    /// the pre-snapshot, our adapt would chain-merge from `bank +
    /// current_frame_diff_updates`, accumulating drift across inter
    /// frames and (in the 32×32 conformance vector) corrupting the
    /// bool-coder cursor by frame 5.
    ///
    /// This test exercises the snapshot-vs-mutated-bank distinction at
    /// the adapt kernel level: same counts, same write-back target,
    /// but two different `pre` inputs (snapshot vs. mutated bank)
    /// produce two distinguishable adapt results.
    #[test]
    fn adapt_pre_snapshot_diverges_from_mutated_bank() {
        use rvp9_core::entropy::adapt::{adapt_coef_probs, coef_adapt_params};
        let mut dec = Decoder::new();
        // Seed counts so adapt is forced to merge (zero counts would
        // produce `merge_probs(pre, _, _) == pre`, masking the
        // distinction this test wants to surface).
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ZERO_TOKEN] = 6;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::ONE_TOKEN] = 2;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::TWO_TOKEN] = 1;
        dec.coef_counts.coef[0][0][0][2][1][rvp9_core::entropy::EOB_MODEL_TOKEN] = 3;
        dec.coef_counts.eob_branch[0][0][0][2][1] = 4;

        // Snapshot the bank BEFORE the simulated compressed-header
        // delta. This is what D.9's `pre_fc` captures.
        let pre_snapshot = dec.frame_contexts[0].clone();

        // Simulate the compressed-header diff-update: bump one
        // coef_probs cell by a small delta directly on the bank.
        // libvpx applies these to `cm->fc` (working copy); we apply
        // them in place on the bank, matching our actual decode path.
        let delta_cell = (0usize, 0, 0, 2, 1, 0);
        let original = dec.frame_contexts[0].coef_probs[delta_cell.0][delta_cell.1]
            [delta_cell.2][delta_cell.3][delta_cell.4][delta_cell.5];
        dec.frame_contexts[0].coef_probs[delta_cell.0][delta_cell.1][delta_cell.2]
            [delta_cell.3][delta_cell.4][delta_cell.5] = original.saturating_sub(20);

        let (cs, uf) = coef_adapt_params(/* intra_only = */ false, /* last_was_key = */ true);

        // The D.9 codepath: adapt uses the pre-snapshot as the `pre`
        // baseline.
        let adapt_from_snapshot =
            adapt_coef_probs(&pre_snapshot.coef_probs, &dec.coef_counts, cs, uf);
        // The pre-D.9 (buggy) codepath: adapt would use the post-
        // compressed-header bank as the `pre` baseline.
        let adapt_from_mutated_bank =
            adapt_coef_probs(&dec.frame_contexts[0].coef_probs, &dec.coef_counts, cs, uf);

        // The two must differ — same counts, same merge kernel, but
        // different `pre` inputs ⇒ different merged outputs.
        assert_ne!(
            adapt_from_snapshot[delta_cell.0][delta_cell.1][delta_cell.2][delta_cell.3]
                [delta_cell.4][delta_cell.5],
            adapt_from_mutated_bank[delta_cell.0][delta_cell.1][delta_cell.2][delta_cell.3]
                [delta_cell.4][delta_cell.5],
            "adapt result must differ between snapshot-as-pre and \
             mutated-bank-as-pre — D.9's snapshot guards against the \
             compressed-header-diff drift cascade"
        );
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

    /// Stage 3 Task E.2: the multi-tile mi-grid stitcher must place
    /// each per-tile grid at the right `(mi_row_start, mi_col_start)`
    /// offset in the frame-wide grid, using the frame's `mi_cols` as
    /// the stride.
    ///
    /// Builds a 16×16 frame split into two 16×8 tiles (1 row × 2 cols),
    /// fills each tile's grid with a distinguishable `ModeInfo` tag,
    /// runs `build_frame_mi_grid`, and verifies that every frame cell
    /// resolves to the correct per-tile origin.
    #[test]
    fn build_frame_mi_grid_stitches_two_tiles_in_row() {
        use rvp9_core::block::BlockSize;
        use rvp9_core::intra_pred::IntraMode;
        use rvp9_core::mode_info::{BlockMode, ModeInfo};
        use rvp9_core::transforms::TxSize;
        use crate::tile::TileBounds;

        let mi_rows = 16usize;
        let mi_cols = 16usize;
        let tile_a_bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let tile_b_bounds = TileBounds {
            mi_row_start: 0,
            mi_row_end: 16,
            mi_col_start: 8,
            mi_col_end: 16,
        };
        // Use `segment_id` as a tile tag (1 = tile A, 2 = tile B).
        let mk_mi = |seg: u8| ModeInfo {
            block_size: BlockSize::Block8x8,
            tx_size: TxSize::Tx4x4,
            skip: false,
            segment_id: seg,
            seg_id_predicted: false,
            mode: BlockMode::Intra {
                y_mode: IntraMode::Dc,
                uv_mode: IntraMode::Dc,
            },
            bmi: Default::default(),
        };
        let tile_a_grid: Vec<Option<ModeInfo>> = vec![Some(mk_mi(1)); 16 * 8];
        let tile_b_grid: Vec<Option<ModeInfo>> = vec![Some(mk_mi(2)); 16 * 8];

        let frame = build_frame_mi_grid(
            mi_rows,
            mi_cols,
            &[
                (tile_a_bounds, tile_a_grid.as_slice()),
                (tile_b_bounds, tile_b_grid.as_slice()),
            ],
        );

        assert_eq!(frame.len(), mi_rows * mi_cols);
        // Spot-check: top-left of tile A.
        assert_eq!(frame[0].unwrap().segment_id, 1);
        // Top-right cell of tile A (col 7).
        assert_eq!(frame[7].unwrap().segment_id, 1);
        // First cell of tile B (col 8).
        assert_eq!(frame[8].unwrap().segment_id, 2);
        // Bottom row of tile B.
        assert_eq!(frame[15 * mi_cols + 15].unwrap().segment_id, 2);
        // Every cell on the tile A / tile B border respects the split.
        for r in 0..mi_rows {
            for c in 0..mi_cols {
                let expected_tag = if c < 8 { 1 } else { 2 };
                assert_eq!(
                    frame[r * mi_cols + c].unwrap().segment_id,
                    expected_tag,
                    "mismatch at (r={r}, c={c})"
                );
            }
        }
    }

}
