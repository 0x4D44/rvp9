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

pub mod compressed_header;
pub mod error;
pub mod frame;
pub mod header;
pub mod inter_coef_source;
pub mod intra_coef_source;
pub mod modes_writer;
pub mod raw;
pub mod sub_parsers;
pub mod superframe;
pub mod tile;

pub use compressed_header::{
    CoefProbTxPlan, CoefProbUpdatePlan, CoefSlotUpdate, CompressedHeaderOptions,
    CompressedHeaderResult, MV_UPDATE_PROB, TX_SIZE_CONTEXTS, parse_compressed_header,
    read_coef_probs, read_frame_reference_mode, read_frame_reference_mode_probs,
    read_inter_mode_probs, read_intra_inter_probs, read_mv_probs, read_partition_probs,
    read_skip_probs, read_switchable_interp_probs, read_tx_mode, read_tx_mode_probs,
    read_y_mode_probs, tx_mode_to_biggest_tx_size, write_coef_probs, write_compressed_header,
    write_frame_reference_mode, write_frame_reference_mode_probs, write_inter_mode_probs,
    write_intra_inter_probs, write_mv_probs, write_partition_probs, write_skip_probs,
    write_switchable_interp_probs, write_tx_mode, write_tx_mode_probs, write_y_mode_probs,
};
pub use error::BitstreamError;
pub use frame::{
    BitDepth, ChromaFormat, ColorRange, ColorSpace, FrameType, InterpFilter, MAX_INTER_REFS,
    Profile, REF_FRAMES, REF_FRAMES_LOG2, ReferenceMode,
};
pub use header::{
    FRAME_CONTEXTS_LOG2, FrameKind, HeaderTail, RefFrameSizes, UncompressedHeader,
    parse_uncompressed_header, write_uncompressed_header,
};
pub use inter_coef_source::{AllZeroCoefSource, InterCoefSource};
pub use intra_coef_source::{AllZeroIntraCoefSource, IntraCoefSource};
pub use modes_writer::{
    ModesWriterError, SegmentationWriteParams, TileBounds as ModesTileBounds,
    record_inter_leaf_coefficients, tile_bounds, tile_mi_offset, write_tile_inter,
    write_tile_inter_with_retained, write_tile_intra_keyframe,
    write_tile_intra_keyframe_with_counts,
};
pub use raw::{RawBitReader, RawBitWriter};
pub use sub_parsers::{
    LoopFilterParams, MAX_SEGMENTS, PREDICTION_PROBS, QuantizationParams, SEG_LVL_MAX,
    SEG_TREE_PROBS, SegFeature, SegmentationParams, TileInfo, get_tile_n_bits, parse_loopfilter,
    parse_quantization, parse_segmentation, parse_tile_info, write_loopfilter, write_quantization,
    write_segmentation, write_tile_info,
};
pub use superframe::{
    MAX_SIZE_FIELD_BYTES, MAX_SUPERFRAME_FRAMES, Superframe, SuperframeBuilder,
    SuperframeIndexErrorReason, parse_superframe_index,
};
pub use tile::{TileBuffer, parse_tile_buffers, write_tile_buffers};
