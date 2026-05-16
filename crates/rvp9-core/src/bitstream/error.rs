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
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull => f.write_str("raw bit writer buffer overflow"),
            Self::Truncated { needed, available } => write!(
                f,
                "raw bit reader truncated: needed {needed} bytes, only {available} available"
            ),
        }
    }
}

impl std::error::Error for BitstreamError {}
