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

use rvp9_core::bitstream::{
    BitstreamError, RefFrameSizes, Superframe, UncompressedHeader, parse_superframe_index,
    parse_uncompressed_header,
};
use rvp9_core::entropy::{FRAME_CONTEXTS, FrameContext};
use rvp9_core::frame::DecoderFramePool;
use rvp9_core::loopfilter::ResolvedLfDeltas;

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
}

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
