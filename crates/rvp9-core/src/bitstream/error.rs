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
        }
    }
}

impl std::error::Error for BitstreamError {}
