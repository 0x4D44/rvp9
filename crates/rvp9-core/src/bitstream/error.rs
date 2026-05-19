//! Bitstream-layer error type.
//!
//! Per LLD-bitstream §3.9. Replaces libvpx's `setjmp`/`longjmp` error-handler
//! callback (`vpx_dsp/bitreader_buffer.h:22`) with a stable `Result`-based
//! Rust API (D-09 RESOLVED — see `notes/divergence-ledger.md` and LLD §7.2).

use std::fmt;

/// Errors produced by the bitstream-layout primitives.
///
/// `Send + Sync + 'static + Clone` so it can move freely across threads
/// and be cloned for diagnostic logging.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BitstreamError {
    /// [`super::RawBitWriter`] overflowed its destination buffer. The
    /// writer is *poisoned* after the first overflow — further `write_*`
    /// calls are silently dropped (matching libvpx semantics at
    /// `bitwriter_buffer.c:35-43`).
    BufferFull,

    /// Reader hit end-of-buffer before the requested number of bits could
    /// be consumed. `needed` and `available` are byte counts measured
    /// from the buffer start to aid diagnostics. Mirrors the silent-zero
    /// behaviour at `bitreader_buffer.c:21-28` but surfaces it via
    /// `Result` rather than calling an error handler.
    Truncated {
        /// Bytes that would be needed to satisfy the read.
        needed: usize,
        /// Bytes actually available in the reader's slice.
        available: usize,
    },

    /// Top 2 bits of the first header byte were not `VP9_FRAME_MARKER`
    /// (= 2). Mirrors libvpx's `Invalid frame marker` rejection at
    /// `vp9_decodeframe.c:2648`.
    InvalidFrameMarker {
        /// The two bits actually read.
        found: u8,
    },

    /// The bitstream profile field decoded to a value the decoder
    /// rejects. In rvp9 v1 we reject everything except `Profile0`
    /// (D-01 RESOLVED: HBD deferred). Profile 4+ (reserved) is also
    /// rejected at libvpx parity (`vp9_decodeframe.c:2654`).
    UnsupportedProfile {
        /// The raw profile value as decoded from the header (0..=3 if
        /// representable; 4 is the sentinel for "out-of-range").
        profile: u8,
    },

    /// The 24-bit sync-code field did not match `0x49 0x83 0x42`.
    /// Mirrors libvpx's `Invalid frame sync code` rejection at
    /// `vp9_decodeframe.c:2688`.
    InvalidSyncCode {
        /// The three sync bytes actually read.
        found: [u8; 3],
    },

    /// The decoded `bit_depth` field was not one of `{8, 10, 12}`.
    /// VP9 only defines those three; any other value is malformed.
    UnsupportedBitDepth {
        /// The raw 2-bit `bit_depth_one_more_than_8` field decoded
        /// (0 → 8-bit, 1 → 10-bit, 2 → 12-bit, 3 → reserved).
        raw: u8,
    },

    /// The decoded color space was `Srgb` but the profile or chroma
    /// subsampling is inconsistent with it. Per spec §6.2 `Srgb`
    /// requires Profile 1/3 + 4:4:4.
    InconsistentSrgb,

    /// `log2_tile_cols` exceeded 6 after applying the increment-bit
    /// chain (libvpx rejects this at `vp9_decodeframe.c:1638-1640`).
    /// 6 is the spec-imposed maximum (64 tile columns).
    InvalidTileCols {
        /// The decoded `log2_tile_cols` value.
        log2_tile_cols: u8,
    },

    /// The 16-bit `first_partition_size` field at the end of the
    /// uncompressed header decoded to `0`. libvpx rejects this with
    /// `Invalid header size` (`vp9_decodeframe.c:2856-2858`).
    InvalidFirstPartitionSize,

    /// An inter-frame header set a `size_in_found` bit for a logical
    /// inter-ref whose pool slot is empty (no decoded frame yet).
    /// Mirrors libvpx's "Failed to decode frame size" rejection at
    /// `vp9_decodeframe.c:1567-1570` (the `INVALID_IDX` branch of
    /// `setup_frame_size_with_refs`).
    InvalidRefFrameSize {
        /// Which logical inter-ref signalled the missing size
        /// (0 = LAST, 1 = GOLDEN, 2 = ALTREF).
        logical_ref: u8,
        /// The 3-bit pool slot index the header referenced.
        pool_slot: u8,
    },

    /// The trailing superframe-index byte indicated a multi-frame
    /// packet but the index was malformed (truncated or marker
    /// mismatch). Mirrors the `VPX_CODEC_CORRUPT_FRAME` returns at
    /// `vp9/decoder/vp9_decoder.c:550, 559`.
    InvalidSuperframeIndex {
        /// Sub-reason — see [`super::superframe::SuperframeIndexErrorReason`].
        reason: super::superframe::SuperframeIndexErrorReason,
    },

    /// Encoder-side [`super::superframe::SuperframeBuilder`] received
    /// a [`super::superframe::SuperframeBuilder::push_frame`] call
    /// after already holding [`super::superframe::MAX_SUPERFRAME_FRAMES`]
    /// frames. The wire format reserves 3 bits for the frame count
    /// (`marker & 0x7`), capping the payload at 8 frames per packet.
    SuperframeOverflow {
        /// `MAX_SUPERFRAME_FRAMES` (always 8) — surfaced so error
        /// callers don't have to import the constant.
        limit: usize,
    },

    /// Encoder-side [`super::superframe::SuperframeBuilder::push_frame`]
    /// received a zero-length frame. The wire format CAN encode size
    /// `0` but libvpx never emits an empty frame; reject for parity.
    SuperframeEmptyFrame,

    /// Encoder-side [`super::tile::write_tile_buffers`] received a
    /// tile payload larger than `u32::MAX` bytes. The 4-byte size
    /// prefix can't represent it. In practice impossible; the variant
    /// exists so callers can propagate `Result` cleanly.
    TileSizeOverflow {
        /// Zero-based index of the offending tile (row-major).
        tile_index: usize,
        /// Length of the payload that overflowed.
        bytes: usize,
    },

    /// The compressed-header boolean decoder failed to initialise:
    /// the leading marker bit at probability 128 wasn't zero.
    /// Mirrors libvpx's bool-coder init failure at
    /// `vpx_dsp/bitreader.c:21-36`, surfaced as `VPX_CODEC_MEM_ERROR`
    /// in `vp9_decodeframe.c:2871-2874`.
    CompressedHeaderInvalidMarker,

    /// The compressed-header boolean decoder ran past the end of its
    /// partition during parse. Mirrors libvpx's
    /// `vpx_reader_has_error` post-decode check at
    /// `vp9_decodeframe.c:2910`.
    CompressedHeaderReaderError,

    /// The compressed-header boolean encoder ran past the end of its
    /// output buffer during write. Mirrors libvpx's
    /// `vpx_writer.error` sticky flag (`vpx_dsp/bitwriter.h:74`).
    /// Also surfaced when the produced header would exceed the
    /// 16-bit `first_partition_size` field (`vp9_bitstream.c:1441-1443`).
    CompressedHeaderTooLarge,

    /// A per-tile size prefix declared more bytes than the input
    /// slice had remaining. Mirrors libvpx
    /// `vp9_decodeframe.c:1669-1671`
    /// ("Truncated packet or corrupt tile size").
    InvalidTileSize {
        /// Tile row index where the failure occurred.
        tile_row: u32,
        /// Tile column index where the failure occurred.
        tile_col: u32,
        /// The size the 4-byte BE prefix decoded to.
        declared: usize,
        /// Bytes actually remaining when the prefix was read.
        available: usize,
    },
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull => f.write_str("raw bit writer buffer overflow"),
            Self::Truncated { needed, available } => write!(
                f,
                "raw bit reader truncated: needed {needed} bytes, only {available} available"
            ),
            Self::InvalidFrameMarker { found } => {
                write!(f, "invalid VP9 frame marker: expected 0b10, found 0b{found:02b}")
            }
            Self::UnsupportedProfile { profile } => {
                write!(f, "unsupported VP9 bitstream profile: {profile}")
            }
            Self::InvalidSyncCode { found } => write!(
                f,
                "invalid VP9 sync code: expected 0x49 0x83 0x42, found 0x{:02x} 0x{:02x} 0x{:02x}",
                found[0], found[1], found[2]
            ),
            Self::UnsupportedBitDepth { raw } => {
                write!(f, "unsupported bit depth raw value: {raw}")
            }
            Self::InconsistentSrgb => f.write_str("sRGB color space requires Profile 1/3 with 4:4:4 sampling"),
            Self::InvalidTileCols { log2_tile_cols } => {
                write!(f, "invalid log2_tile_cols: {log2_tile_cols} (max 6)")
            }
            Self::InvalidFirstPartitionSize => {
                f.write_str("invalid first_partition_size: 0 (compressed header must be non-empty)")
            }
            Self::InvalidRefFrameSize {
                logical_ref,
                pool_slot,
            } => write!(
                f,
                "inter-ref {logical_ref} requested size from empty pool slot {pool_slot}"
            ),
            Self::CompressedHeaderInvalidMarker => {
                f.write_str("compressed-header boolean decoder: invalid marker bit")
            }
            Self::CompressedHeaderReaderError => {
                f.write_str("compressed-header boolean decoder: read past end of partition")
            }
            Self::CompressedHeaderTooLarge => {
                f.write_str("compressed-header boolean encoder: output buffer overflow")
            }
            Self::InvalidTileSize {
                tile_row,
                tile_col,
                declared,
                available,
            } => write!(
                f,
                "invalid tile size at row {tile_row}, col {tile_col}: declared {declared} bytes, only {available} available"
            ),
            Self::InvalidSuperframeIndex { reason } => match reason {
                super::superframe::SuperframeIndexErrorReason::TruncatedIndex {
                    declared,
                    available,
                } => write!(
                    f,
                    "invalid superframe index: declared {declared} bytes, only {available} available"
                ),
                super::superframe::SuperframeIndexErrorReason::MarkerMismatch {
                    leading,
                    trailing,
                } => write!(
                    f,
                    "invalid superframe index: leading marker 0x{leading:02x} != trailing 0x{trailing:02x}"
                ),
            },
            Self::SuperframeOverflow { limit } => write!(
                f,
                "superframe builder: cannot push more than {limit} frames per packet"
            ),
            Self::SuperframeEmptyFrame => {
                f.write_str("superframe builder: refusing zero-length frame")
            }
            Self::TileSizeOverflow { tile_index, bytes } => write!(
                f,
                "tile-buffer write: tile {tile_index} payload is {bytes} bytes, exceeds 4-byte size prefix"
            ),
        }
    }
}

impl std::error::Error for BitstreamError {}
