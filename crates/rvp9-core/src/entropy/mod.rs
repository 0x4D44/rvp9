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

pub mod adapt;
pub mod bitacct;
pub mod bool_reader;
pub mod bool_writer;
pub mod coef_probs;
pub mod default_probs;
pub mod detokenize;
pub mod diff_update;
pub mod frame_context;
pub mod kf_probs;
pub mod mv_probs;
pub mod pareto;
pub mod prob_diff_writer;
pub mod tables;

pub use adapt::{
    Coef12Bank, CoefCountsBank, FrameCounts, MODE_MV_COUNT_SAT, TX_SIZE_CONTEXTS, TxCounts,
    adapt_coef_probs, adapt_mode_probs, adapt_mv_probs, coef_adapt_params, get_binary_prob,
    get_prob, merge_probs, mode_mv_merge_probs, new_coef12_bank, tree_merge_probs, weighted_prob,
};
pub use bool_reader::{BoolReader, BoolReaderSnapshot};
pub use bool_writer::BoolWriter;
pub use coef_probs::{
    COEF_BANDS, COEFF_CONTEXTS, CoeffProbsModel, DEFAULT_COEF_PROBS_4X4, DEFAULT_COEF_PROBS_8X8,
    DEFAULT_COEF_PROBS_16X16, DEFAULT_COEF_PROBS_32X32, PLANE_TYPES, REF_TYPES,
    UNCONSTRAINED_NODES,
};
pub use default_probs::{
    BLOCK_SIZE_GROUPS, COMP_INTER_CONTEXTS, DEFAULT_COMP_INTER_PROB, DEFAULT_COMP_REF_PROB,
    DEFAULT_IF_UV_PROBS, DEFAULT_IF_Y_PROBS, DEFAULT_INTER_MODE_PROBS, DEFAULT_INTRA_INTER_PROB,
    DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROB, DEFAULT_SKIP_PROBS,
    DEFAULT_SWITCHABLE_INTERP_PROB, DEFAULT_TX_PROBS, DefaultTxProbs, INTER_MODE_CONTEXTS,
    INTER_MODES, INTRA_INTER_CONTEXTS, INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES,
    REF_CONTEXTS, SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
pub use detokenize::{
    CoefBandCtxProbs, CoefCountsBlock, CoefFullCountsBlock, CoefTokenPass, DecodeCoefCounts,
    ENTROPY_TOKENS, EOB_MODEL_TOKEN, EobBranchCountsBlock, ONE_TOKEN, RetainedCoefTokens,
    TWO_TOKEN, VP9_CAT1_PROB, VP9_CAT2_PROB, VP9_CAT3_PROB, VP9_CAT4_PROB, VP9_CAT5_PROB,
    VP9_CAT6_PROB, VP9_COEFBAND_TRANS_4X4, VP9_COEFBAND_TRANS_8X8PLUS, ZERO_TOKEN, decode_coefs,
    encode_coefs, write_block_tokens, write_block_tokens_with_retained,
};
pub use diff_update::{DIFF_UPDATE_PROB, diff_update_prob, inv_remap_prob};
pub use frame_context::{FRAME_CONTEXTS, FRAME_CONTEXTS_LOG2, FrameContext, TX_SIZES};
pub use kf_probs::{VP9_KF_PARTITION_PROBS, VP9_KF_UV_MODE_PROB, VP9_KF_Y_MODE_PROB};
pub use mv_probs::{
    CLASS0_BITS, CLASS0_SIZE, DEFAULT_NMV_CONTEXT, MV_CLASSES, MV_FP_SIZE, MV_JOINTS,
    MV_OFFSET_BITS, NmvComponent, NmvComponentCounts, NmvContext, NmvCounts,
};
pub use pareto::{COEFF_PROB_MODELS, MODEL_NODES, VP9_PARETO8_FULL};
pub use prob_diff_writer::{cond_prob_diff_update, remap_prob, write_prob_diff_update};
pub use tables::VPX_NORM;

/// The probability value `128` — the maximum-entropy point that
/// [`BoolWriter::write_bit`] and [`BoolReader::read_bit`] use when they
/// don't have a model. Mirrors libvpx `vpx_prob_half`.
pub const VPX_PROB_HALF: u8 = 128;
