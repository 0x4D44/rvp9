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
//! - [`frame`] — frame-header type definitions (Profile / FrameType /
//!   ColorSpace / ColorRange / BitDepth / ChromaFormat + sync constants).
//!
//! Frame-header parser (the function that consumes a byte slice and
//! produces a populated header struct), compressed-header glue,
//! tile-buffer, and superframe assembly will land in follow-up commits.

pub mod error;
pub mod frame;
pub mod header;
pub mod raw;
pub mod sub_parsers;

pub use error::BitstreamError;
pub use frame::{
    BitDepth, ChromaFormat, ColorRange, ColorSpace, FrameType, MAX_INTER_REFS, Profile,
    REF_FRAMES, REF_FRAMES_LOG2,
};
pub use header::{FrameKind, UncompressedHeader, parse_uncompressed_header};
pub use raw::{RawBitReader, RawBitWriter};
pub use sub_parsers::{
    LoopFilterParams, QuantizationParams, SegFeature, SegmentationParams, TileInfo, parse_loopfilter,
    parse_quantization, parse_segmentation, parse_tile_info,
};
