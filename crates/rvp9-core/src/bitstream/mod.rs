//! VP9 bitstream-layout primitives.
//!
//! This module owns the **uncompressed-header / superframe-index / tile-buffer**
//! machinery — everything *outside* the boolean coder. The boolean coder
//! itself lives in [`crate::entropy`] (Stage 3 task 2). See
//! `notes/lld/bitstream.md` for the design; `notes/vp9-analysis/04-bitstream-entropy.md`
//! has the libvpx reference.
//!
//! ## Status
//!
//! Stage 3 task 1 in progress. This commit lands:
//!
//! - [`error::BitstreamError`] — error variants for the raw bit
//!   reader/writer + a couple of forward-declared variants used by header
//!   parsing (added empty for now; constructors land with their producers).
//! - [`raw::RawBitWriter`] / [`raw::RawBitReader`] — MSB-first byte-level
//!   bit packing, mirrors libvpx `vpx_dsp/bitwriter_buffer.{h,c}` and
//!   `bitreader_buffer.{h,c}` bit-for-bit.
//!
//! Frame header / compressed-header / tile-buffer / superframe assembly
//! will land in follow-up commits to this Stage 3 task.

pub mod error;
pub mod raw;

pub use error::BitstreamError;
pub use raw::{RawBitReader, RawBitWriter};
