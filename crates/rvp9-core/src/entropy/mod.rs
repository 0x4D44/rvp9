//! VP9 entropy coder: boolean (arithmetic) writer + reader.
//!
//! This module owns the **byte-stream-level arithmetic codec** used for
//! VP9's compressed header and every coefficient/mode/MV symbol. Mirrors
//! libvpx `vpx_dsp/bitwriter.{h,c}` and `bitreader.{h,c}` bit-for-bit; the
//! shared `vpx_norm[256]` table lives in [`tables`]. The frame-context
//! probability machinery (`FRAME_CONTEXT`, `FRAME_COUNTS`, tree merges,
//! `adapt_*_probs`) will land in follow-up commits to this Stage 3 task.
//!
//! ## Locked design decisions
//!
//! - **`BD_VALUE = u64`** (entropy LLD §7.1, divergence-ledger index).
//!   32-bit targets are explicitly out of scope for v1.
//! - **No decryption callback** (entropy LLD §7.2). libvpx's
//!   `vpx_decrypt_cb` + `decrypt_state` fields are dropped; the reader
//!   takes a plain `&[u8]`.
//! - **Writer panic-mid-write is not poison-safe** (entropy LLD §7.3).
//!   We mirror libvpx: the `error` flag is checked at byte-emit time,
//!   not on every method.
//! - **`LOTS_OF_BITS` and EOF semantics preserved** (entropy LLD §7.6)
//!   — see [`bool_reader`].
//! - **Counts double-count quirk preserved bug-for-bug** (entropy LLD
//!   §7.4 / threading LLD §5.6). Affects backward probability
//!   adaptation, not the bool codec itself; documented here for the
//!   record.

pub mod bool_reader;
pub mod bool_writer;
pub mod default_probs;
pub mod tables;

pub use bool_reader::BoolReader;
pub use bool_writer::BoolWriter;
pub use default_probs::{
    BLOCK_SIZE_GROUPS, COMP_INTER_CONTEXTS, DEFAULT_COMP_INTER_PROB, DEFAULT_COMP_REF_PROB,
    DEFAULT_IF_UV_PROBS, DEFAULT_IF_Y_PROBS, DEFAULT_INTER_MODE_PROBS, DEFAULT_INTRA_INTER_PROB,
    DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROB, DEFAULT_SKIP_PROBS,
    DEFAULT_SWITCHABLE_INTERP_PROB, DEFAULT_TX_PROBS, DefaultTxProbs, INTER_MODE_CONTEXTS,
    INTER_MODES, INTRA_INTER_CONTEXTS, INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES,
    REF_CONTEXTS, SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
pub use tables::VPX_NORM;

/// The probability value `128` — the maximum-entropy point that
/// [`BoolWriter::write_bit`] and [`BoolReader::read_bit`] use when they
/// don't have a model. Mirrors libvpx `vpx_prob_half`.
pub const VPX_PROB_HALF: u8 = 128;
