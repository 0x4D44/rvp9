//! VP9 superframe-index parser.
//!
//! A "superframe" is a single transport packet (IVF chunk, RTP packet,
//! etc.) that contains 2..=8 independently-decodable VP9 frames
//! concatenated together. The boundaries between them are signalled
//! by a trailing **superframe index** appended to the packet by the
//! encoder, and the decoder uses this index to split the packet into
//! per-frame byte ranges.
//!
//! Mirrors libvpx `vp9_parse_superframe_index`
//! (`vp9/decoder/vp9_decoder.c:527-585`). The decryption-callback
//! branch is dropped (per `notes/divergence-ledger.md` — same policy
//! as the bool reader; rvp9 takes a plain `&[u8]`).
//!
//! ## Wire format
//!
//! - The very last byte of the packet is the **trailing marker**.
//! - If the marker's top 3 bits are `0b110` (`marker & 0xe0 == 0xc0`),
//!   the packet ends with an index; otherwise the packet contains a
//!   single frame occupying the whole buffer.
//! - For a superframe index, the remaining 5 bits of the marker
//!   encode:
//!   - `frames = (marker & 0x7) + 1` — number of frames (1..=8).
//!   - `mag = ((marker >> 3) & 0x3) + 1` — bytes per size field (1..=4).
//! - The full index occupies `2 + mag * frames` trailing bytes:
//!   1 leading marker + `frames × mag` little-endian size fields +
//!   1 trailing marker (must equal the leading marker; else corrupt).
//! - The frame data starts at byte 0 and ends `index_sz` bytes before
//!   the end of the packet. Frame `i` has size `sizes[i]`.

use super::error::BitstreamError;

/// Maximum number of frames in a single superframe (libvpx hard-coded
/// limit). Mirrors the implicit `(marker & 0x7) + 1` range.
pub const MAX_SUPERFRAME_FRAMES: usize = 8;

/// Maximum bytes per per-frame size field. Mirrors
/// `((marker >> 3) & 0x3) + 1` range.
pub const MAX_SIZE_FIELD_BYTES: usize = 4;

/// Result of [`parse_superframe_index`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Superframe {
    /// Not a superframe — the packet contains a single VP9 frame.
    /// Decoder should treat the entire buffer as one frame.
    Single,
    /// Superframe — multiple concatenated frames. `sizes[0..count]`
    /// gives each frame's byte length in order. The frames start at
    /// byte 0 of the original packet; frame `i` lives at offset
    /// `sum(sizes[0..i]) .. sum(sizes[0..=i])`.
    Multi {
        /// Per-frame byte sizes, in order.
        sizes: [u32; MAX_SUPERFRAME_FRAMES],
        /// Number of frames (`1..=8`).
        count: usize,
        /// Trailing index size, in bytes — the data the decoder should
        /// trim off the end of the packet before parsing frames. Equals
        /// `2 + mag * frames`.
        index_size: usize,
    },
}

/// Parse the trailing superframe index of a VP9 transport packet.
///
/// Mirrors libvpx `vp9_parse_superframe_index`
/// (`vp9/decoder/vp9_decoder.c:527-585`).
///
/// # Errors
///
/// - [`BitstreamError::Truncated`] if `data` is empty.
/// - [`BitstreamError::InvalidSuperframeIndex`] if the trailing marker
///   indicates a superframe but the data is too short to contain the
///   declared index OR the leading marker doesn't match the trailing
///   marker.
pub fn parse_superframe_index(data: &[u8]) -> Result<Superframe, BitstreamError> {
    if data.is_empty() {
        return Err(BitstreamError::Truncated {
            needed: 1,
            available: 0,
        });
    }

    let marker = data[data.len() - 1];
    if (marker & 0xe0) != 0xc0 {
        // Not a superframe.
        return Ok(Superframe::Single);
    }

    let frames = ((marker & 0x07) + 1) as usize;
    let mag = (((marker >> 3) & 0x03) + 1) as usize;
    let index_size = 2 + mag * frames;

    if data.len() < index_size {
        return Err(BitstreamError::InvalidSuperframeIndex {
            reason: SuperframeIndexErrorReason::TruncatedIndex {
                declared: index_size,
                available: data.len(),
            },
        });
    }

    // Verify the leading marker byte matches.
    let leading_marker = data[data.len() - index_size];
    if leading_marker != marker {
        return Err(BitstreamError::InvalidSuperframeIndex {
            reason: SuperframeIndexErrorReason::MarkerMismatch {
                leading: leading_marker,
                trailing: marker,
            },
        });
    }

    // Decode the `frames * mag` size bytes (little-endian per frame).
    let mut sizes = [0u32; MAX_SUPERFRAME_FRAMES];
    let mut x = data.len() - index_size + 1;
    for size_slot in sizes.iter_mut().take(frames) {
        let mut this_sz = 0u32;
        for j in 0..mag {
            this_sz |= (data[x] as u32) << (j * 8);
            x += 1;
        }
        *size_slot = this_sz;
    }

    Ok(Superframe::Multi {
        sizes,
        count: frames,
        index_size,
    })
}

/// Encoder-side companion to [`parse_superframe_index`]. Accumulates
/// 1..=[`MAX_SUPERFRAME_FRAMES`] frames and emits a transport packet
/// whose trailing superframe index parses back to the same set of
/// boundaries.
///
/// Single-frame outputs carry NO trailing index (decoder reads
/// [`Superframe::Single`]). Multi-frame outputs carry the minimum-mag
/// index that fits every frame size.
///
/// Mirrors libvpx `vp9/encoder/vp9_encoder.c:5093-5142` `vp9_pack_bitstream`'s
/// superframe-trailer emission, with the same wire format
/// documented in this module's header.
#[derive(Clone, Debug, Default)]
pub struct SuperframeBuilder {
    frames: Vec<Vec<u8>>,
}

impl SuperframeBuilder {
    /// Construct an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of frames currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` if no frames have been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Append a frame.
    ///
    /// # Errors
    ///
    /// - [`BitstreamError::SuperframeOverflow`] if pushing would
    ///   exceed [`MAX_SUPERFRAME_FRAMES`].
    /// - [`BitstreamError::SuperframeEmptyFrame`] if `bytes` is empty
    ///   (the wire format encodes each frame's size as a 1..=4 byte
    ///   little-endian field; a zero-length frame is decodable but
    ///   meaningless, and libvpx never emits one — reject for parity).
    pub fn push_frame(&mut self, bytes: Vec<u8>) -> Result<(), BitstreamError> {
        if self.frames.len() >= MAX_SUPERFRAME_FRAMES {
            return Err(BitstreamError::SuperframeOverflow {
                limit: MAX_SUPERFRAME_FRAMES,
            });
        }
        if bytes.is_empty() {
            return Err(BitstreamError::SuperframeEmptyFrame);
        }
        self.frames.push(bytes);
        Ok(())
    }

    /// Consume the builder and emit the transport packet bytes. Returns
    /// an empty `Vec` when no frames were pushed.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        match self.frames.len() {
            0 => Vec::new(),
            1 => self.frames.into_iter().next().expect("len == 1"),
            n => {
                let max_size = self.frames.iter().map(Vec::len).max().expect("non-empty") as u32;
                let mag = size_field_bytes_for(max_size);
                let total_body: usize = self.frames.iter().map(Vec::len).sum();
                let mut buf = Vec::with_capacity(total_body + 2 + mag * n);
                for f in &self.frames {
                    buf.extend_from_slice(f);
                }
                let marker = 0xc0u8 | (((mag as u8) - 1) << 3) | ((n as u8) - 1);
                buf.push(marker);
                for f in &self.frames {
                    let mut sz = f.len() as u32;
                    for _ in 0..mag {
                        buf.push((sz & 0xff) as u8);
                        sz >>= 8;
                    }
                }
                buf.push(marker);
                buf
            }
        }
    }
}

/// Smallest `mag` value (1..=4) such that `size` fits in `mag` bytes.
/// Mirrors libvpx's implicit selection in `vp9_pack_bitstream`.
#[must_use]
fn size_field_bytes_for(max_size: u32) -> usize {
    if max_size <= 0xff {
        1
    } else if max_size <= 0xffff {
        2
    } else if max_size <= 0x00ff_ffff {
        3
    } else {
        4
    }
}

/// Sub-classification for [`BitstreamError::InvalidSuperframeIndex`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SuperframeIndexErrorReason {
    /// Trailing marker claimed a superframe index but the buffer is
    /// too short to hold it.
    TruncatedIndex {
        /// Bytes the index would require (`2 + mag * frames`).
        declared: usize,
        /// Bytes actually available in the buffer.
        available: usize,
    },
    /// The leading marker byte did not match the trailing marker —
    /// indicates a corrupted bitstream that happens to end in `0xc0..`.
    MarkerMismatch {
        /// The byte at `data[data.len() - index_size]`.
        leading: u8,
        /// The byte at `data[data.len() - 1]`.
        trailing: u8,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic superframe: concatenate `frames`, then
    /// append the trailing index encoding their sizes.
    fn build_superframe(frames: &[&[u8]], mag: usize) -> Vec<u8> {
        assert!(!frames.is_empty() && frames.len() <= MAX_SUPERFRAME_FRAMES);
        assert!((1..=4).contains(&mag));
        let mut buf = Vec::new();
        for f in frames {
            buf.extend_from_slice(f);
        }
        let marker = 0xc0 | (((mag - 1) as u8) << 3) | ((frames.len() - 1) as u8);
        buf.push(marker);
        for f in frames {
            let mut sz = f.len() as u32;
            for _ in 0..mag {
                buf.push((sz & 0xff) as u8);
                sz >>= 8;
            }
        }
        buf.push(marker);
        buf
    }

    #[test]
    fn empty_buffer_rejected() {
        let err = parse_superframe_index(&[]).unwrap_err();
        assert!(matches!(err, BitstreamError::Truncated { .. }));
    }

    #[test]
    fn single_frame_no_index() {
        // A "normal" frame: just data bytes, last byte not 0xc0..0xdf.
        let frame = [0x82, 0x49, 0x83, 0x42, 0x00, 0x00];
        let sf = parse_superframe_index(&frame).unwrap();
        assert_eq!(sf, Superframe::Single);
    }

    #[test]
    fn single_frame_with_payload_ending_in_co_byte_safe_if_not_marker_pattern() {
        // 0xc0 is the start of the marker range. The encoder must add a
        // 0 byte to avoid this; but if a packet legitimately ends with
        // a byte in `[0xc0, 0xdf]` AND the front marker byte doesn't
        // match, the parser returns MarkerMismatch. We test the legal
        // case here: a byte in some other range.
        let frame = [0xff, 0x00];
        let sf = parse_superframe_index(&frame).unwrap();
        assert_eq!(sf, Superframe::Single);
    }

    #[test]
    fn two_frame_superframe_mag_1() {
        // Frame A = 8 bytes, Frame B = 12 bytes; mag = 1.
        let a = [0xaa; 8];
        let b = [0xbb; 12];
        let buf = build_superframe(&[&a, &b], 1);
        let sf = parse_superframe_index(&buf).unwrap();
        match sf {
            Superframe::Multi { sizes, count, index_size } => {
                assert_eq!(count, 2);
                assert_eq!(sizes[0], 8);
                assert_eq!(sizes[1], 12);
                // index_size = 2 + 1 * 2 = 4
                assert_eq!(index_size, 4);
            }
            other => panic!("expected Multi, got {other:?}"),
        }
    }

    #[test]
    fn eight_frame_superframe_mag_4() {
        let frames: Vec<Vec<u8>> = (0..8).map(|i| vec![i as u8; 32 + i]).collect();
        let frame_refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let buf = build_superframe(&frame_refs, 4);
        let sf = parse_superframe_index(&buf).unwrap();
        match sf {
            Superframe::Multi { sizes, count, index_size } => {
                assert_eq!(count, 8);
                for (i, &sz) in sizes.iter().take(8).enumerate() {
                    assert_eq!(sz as usize, 32 + i);
                }
                // index_size = 2 + 4 * 8 = 34
                assert_eq!(index_size, 34);
            }
            _ => panic!("expected Multi"),
        }
    }

    #[test]
    fn truncated_index_rejected() {
        // Trailing marker claims 2 frames, mag = 4 → index_size = 10,
        // but the buffer is only 5 bytes long.
        let marker = 0xc0 | (3 << 3) | 1;
        let buf = [0u8, 0u8, 0u8, 0u8, marker];
        let err = parse_superframe_index(&buf).unwrap_err();
        match err {
            BitstreamError::InvalidSuperframeIndex { reason } => {
                assert!(matches!(
                    reason,
                    SuperframeIndexErrorReason::TruncatedIndex {
                        declared: 10,
                        available: 5
                    }
                ));
            }
            other => panic!("expected InvalidSuperframeIndex, got {other:?}"),
        }
    }

    #[test]
    fn marker_mismatch_rejected() {
        // Build a valid 1-frame superframe then corrupt the leading
        // marker.
        let mut buf = build_superframe(&[&[0x00; 16]], 1);
        // Leading marker is at data.len() - index_size; for 1 frame
        // mag=1 index_size=3.
        let idx_size = 2 + 1;
        let pos = buf.len() - idx_size;
        buf[pos] = 0x00;
        let err = parse_superframe_index(&buf).unwrap_err();
        assert!(matches!(
            err,
            BitstreamError::InvalidSuperframeIndex {
                reason: SuperframeIndexErrorReason::MarkerMismatch { .. }
            }
        ));
    }

    #[test]
    fn multi_byte_size_round_trip() {
        // One frame of 257 bytes, mag = 2 → size encoded as `[0x01,
        // 0x01]` little-endian.
        let frame = vec![0xee; 257];
        let buf = build_superframe(&[&frame], 2);
        let sf = parse_superframe_index(&buf).unwrap();
        match sf {
            Superframe::Multi { sizes, count, .. } => {
                assert_eq!(count, 1);
                assert_eq!(sizes[0], 257);
            }
            _ => panic!("expected Multi"),
        }
    }

    // -----------------------------------------------------------------
    // SuperframeBuilder tests
    // -----------------------------------------------------------------

    #[test]
    fn builder_empty_finish_yields_empty_vec() {
        let b = SuperframeBuilder::new();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert!(b.finish().is_empty());
    }

    #[test]
    fn builder_single_frame_emits_no_index() {
        let mut b = SuperframeBuilder::new();
        b.push_frame(vec![1, 2, 3, 4, 5]).unwrap();
        assert_eq!(b.len(), 1);
        let out = b.finish();
        // No trailing marker — parses as Single.
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
        assert_eq!(parse_superframe_index(&out).unwrap(), Superframe::Single);
    }

    #[test]
    fn builder_multi_frame_round_trips_through_parser() {
        let mut b = SuperframeBuilder::new();
        b.push_frame(vec![0xaa; 10]).unwrap();
        b.push_frame(vec![0xbb; 25]).unwrap();
        b.push_frame(vec![0xcc; 13]).unwrap();
        let out = b.finish();
        let sf = parse_superframe_index(&out).unwrap();
        match sf {
            Superframe::Multi { sizes, count, index_size } => {
                assert_eq!(count, 3);
                assert_eq!(&sizes[..3], &[10, 25, 13]);
                // mag=1 (all sizes fit in 1 byte) + 1 leading + 1 trailing + 3 size bytes.
                assert_eq!(index_size, 2 + 3); // 2 + mag(=1) * frames(=3)
            }
            _ => panic!("expected Multi"),
        }
    }

    #[test]
    fn builder_picks_mag_2_when_a_frame_exceeds_255_bytes() {
        let mut b = SuperframeBuilder::new();
        b.push_frame(vec![0xaa; 300]).unwrap(); // > 255 -> needs mag >= 2
        b.push_frame(vec![0xbb; 50]).unwrap();
        let out = b.finish();
        let sf = parse_superframe_index(&out).unwrap();
        match sf {
            Superframe::Multi { sizes, count, index_size } => {
                assert_eq!(count, 2);
                assert_eq!(&sizes[..2], &[300, 50]);
                assert_eq!(index_size, 2 + 2 * 2); // mag=2, 2 frames
            }
            _ => panic!("expected Multi"),
        }
    }

    #[test]
    fn builder_picks_mag_3_then_4_at_thresholds() {
        // Just above 16-bit limit -> mag = 3.
        let mut b = SuperframeBuilder::new();
        b.push_frame(vec![0xaa; 0x10000]).unwrap();
        b.push_frame(vec![0xbb; 1]).unwrap();
        let out = b.finish();
        match parse_superframe_index(&out).unwrap() {
            Superframe::Multi { sizes, index_size, .. } => {
                assert_eq!(sizes[0], 0x10000);
                assert_eq!(sizes[1], 1);
                assert_eq!(index_size, 2 + 3 * 2);
            }
            _ => panic!("expected Multi"),
        }
        // Just above 24-bit limit -> mag = 4.
        let mut b = SuperframeBuilder::new();
        b.push_frame(vec![0xaa; 0x0100_0000]).unwrap();
        b.push_frame(vec![0xbb; 1]).unwrap();
        let out = b.finish();
        match parse_superframe_index(&out).unwrap() {
            Superframe::Multi { sizes, index_size, .. } => {
                assert_eq!(sizes[0], 0x0100_0000);
                assert_eq!(sizes[1], 1);
                assert_eq!(index_size, 2 + 4 * 2);
            }
            _ => panic!("expected Multi"),
        }
    }

    #[test]
    fn builder_rejects_ninth_frame() {
        let mut b = SuperframeBuilder::new();
        for _ in 0..MAX_SUPERFRAME_FRAMES {
            b.push_frame(vec![0xaa]).unwrap();
        }
        let err = b.push_frame(vec![0xbb]).unwrap_err();
        assert_eq!(
            err,
            BitstreamError::SuperframeOverflow { limit: MAX_SUPERFRAME_FRAMES }
        );
    }

    #[test]
    fn builder_rejects_empty_frame() {
        let mut b = SuperframeBuilder::new();
        let err = b.push_frame(Vec::new()).unwrap_err();
        assert_eq!(err, BitstreamError::SuperframeEmptyFrame);
    }

    #[test]
    fn builder_round_trips_max_frames() {
        let mut b = SuperframeBuilder::new();
        for i in 0..MAX_SUPERFRAME_FRAMES {
            b.push_frame(vec![i as u8; 10 + i]).unwrap();
        }
        let out = b.finish();
        let sf = parse_superframe_index(&out).unwrap();
        match sf {
            Superframe::Multi { sizes, count, .. } => {
                assert_eq!(count, MAX_SUPERFRAME_FRAMES);
                for (i, &sz) in sizes.iter().enumerate().take(MAX_SUPERFRAME_FRAMES) {
                    assert_eq!(sz as usize, 10 + i);
                }
            }
            _ => panic!("expected Multi"),
        }
    }

    #[test]
    fn builder_byte_layout_matches_legacy_helper() {
        // The legacy test helper `build_superframe(frames, mag=1)`
        // produces the canonical wire form. The builder picks mag
        // automatically; for small frames (<=255 bytes) it should pick
        // mag=1 and produce byte-identical output.
        let a = vec![0x11u8; 5];
        let b = vec![0x22u8; 7];
        let want = build_superframe(&[&a, &b], 1);

        let mut sb = SuperframeBuilder::new();
        sb.push_frame(a).unwrap();
        sb.push_frame(b).unwrap();
        let got = sb.finish();
        assert_eq!(got, want);
    }
}
