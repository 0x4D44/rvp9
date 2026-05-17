//! VP9 `FRAME_CONTEXT` aggregate — the per-frame probability bank
//! (Task 2b.5).
//!
//! Mirrors libvpx `struct frame_contexts`
//! (`vp9/common/vp9_entropymode.h:44-60`). The decoder maintains
//! [`FRAME_CONTEXTS_LOG2`] banks of these (`= 4` slots indexed by the
//! 2-bit `frame_context_idx` from the uncompressed header) and
//! initialises them via [`FrameContext::new_default`] which fills in
//! every field from the constants defined in
//! [`super::default_probs`], [`super::coef_probs`], and
//! [`super::mv_probs`].
//!
//! Backward adaptation (`vp9_adapt_*_probs` and `merge_probs`) is
//! deferred to Task 1e — that's the boolean-coder driver layer.

use super::coef_probs::{
    CoeffProbsModel, DEFAULT_COEF_PROBS_16X16, DEFAULT_COEF_PROBS_32X32, DEFAULT_COEF_PROBS_4X4,
    DEFAULT_COEF_PROBS_8X8,
};
use super::default_probs::{
    BLOCK_SIZE_GROUPS, COMP_INTER_CONTEXTS, DEFAULT_COMP_INTER_PROB, DEFAULT_COMP_REF_PROB,
    DEFAULT_IF_UV_PROBS, DEFAULT_IF_Y_PROBS, DEFAULT_INTER_MODE_PROBS, DEFAULT_INTRA_INTER_PROB,
    DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROB, DEFAULT_SKIP_PROBS,
    DEFAULT_SWITCHABLE_INTERP_PROB, DEFAULT_TX_PROBS, DefaultTxProbs, INTER_MODE_CONTEXTS,
    INTER_MODES, INTRA_INTER_CONTEXTS, INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES,
    REF_CONTEXTS, SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
use super::mv_probs::{DEFAULT_NMV_CONTEXT, NmvContext};

/// `log2(FRAME_CONTEXTS)` where `FRAME_CONTEXTS = 4` is the number of
/// independent probability banks the decoder keeps around (selected by
/// the 2-bit `frame_context_idx`). Mirrors libvpx
/// `vp9_entropymode.h` (`FRAME_CONTEXTS_LOG2 = 2`).
pub const FRAME_CONTEXTS_LOG2: u8 = 2;

/// `1 << FRAME_CONTEXTS_LOG2 = 4`.
pub const FRAME_CONTEXTS: usize = 1 << FRAME_CONTEXTS_LOG2;

/// Number of VP9 TX sizes (`TX_4X4`, `TX_8X8`, `TX_16X16`, `TX_32X32`).
pub const TX_SIZES: usize = 4;

/// Per-frame probability bank. Mirrors libvpx `FRAME_CONTEXT`
/// (`vp9_entropymode.h:44-60`). The `initialized` bool is the
/// equivalent of libvpx's `int initialized` field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameContext {
    /// Intra Y-mode probabilities per block-size group.
    pub y_mode_prob: [[u8; INTRA_MODES - 1]; BLOCK_SIZE_GROUPS],
    /// Intra UV-mode probabilities given Y mode.
    pub uv_mode_prob: [[u8; INTRA_MODES - 1]; INTRA_MODES],
    /// Partition tree probabilities.
    pub partition_prob: [[u8; PARTITION_TYPES - 1]; PARTITION_CONTEXTS],
    /// Coefficient probabilities, indexed by `[TX_SIZE]`.
    pub coef_probs: [CoeffProbsModel; TX_SIZES],
    /// Switchable interpolation-filter tree probabilities.
    pub switchable_interp_prob: [[u8; SWITCHABLE_FILTERS - 1]; SWITCHABLE_FILTER_CONTEXTS],
    /// Inter-mode tree probabilities.
    pub inter_mode_probs: [[u8; INTER_MODES - 1]; INTER_MODE_CONTEXTS],
    /// Intra/inter classification probabilities.
    pub intra_inter_prob: [u8; INTRA_INTER_CONTEXTS],
    /// Compound-inter probabilities.
    pub comp_inter_prob: [u8; COMP_INTER_CONTEXTS],
    /// Single-ref probabilities.
    pub single_ref_prob: [[u8; 2]; REF_CONTEXTS],
    /// Compound-ref probabilities.
    pub comp_ref_prob: [u8; REF_CONTEXTS],
    /// TX-size probabilities (32×32 / 16×16 / 8×8 sub-tables).
    pub tx_probs: DefaultTxProbs,
    /// Skip-flag probabilities.
    pub skip_probs: [u8; SKIP_CONTEXTS],
    /// Motion-vector probabilities.
    pub nmvc: NmvContext,
    /// `true` once the context has been initialised (libvpx
    /// `initialized`).
    pub initialized: bool,
}

impl FrameContext {
    /// Build a fresh frame context populated with libvpx's
    /// `init_mode_probs` defaults plus `vp9_init_mv_probs`. After this
    /// call `self.initialized == true`. Mirrors libvpx
    /// `init_mode_probs` (`vp9_entropymode.c:323-342`) + the
    /// `vp9_copy(cm->fc->coef_probs[...])` block at
    /// `vp9_entropy.c:1041-1045`.
    #[must_use]
    pub fn new_default() -> Self {
        Self {
            y_mode_prob: DEFAULT_IF_Y_PROBS,
            uv_mode_prob: DEFAULT_IF_UV_PROBS,
            partition_prob: DEFAULT_PARTITION_PROBS,
            coef_probs: [
                DEFAULT_COEF_PROBS_4X4,
                DEFAULT_COEF_PROBS_8X8,
                DEFAULT_COEF_PROBS_16X16,
                DEFAULT_COEF_PROBS_32X32,
            ],
            switchable_interp_prob: DEFAULT_SWITCHABLE_INTERP_PROB,
            inter_mode_probs: DEFAULT_INTER_MODE_PROBS,
            intra_inter_prob: DEFAULT_INTRA_INTER_PROB,
            comp_inter_prob: DEFAULT_COMP_INTER_PROB,
            single_ref_prob: DEFAULT_SINGLE_REF_PROB,
            comp_ref_prob: DEFAULT_COMP_REF_PROB,
            tx_probs: DEFAULT_TX_PROBS,
            skip_probs: DEFAULT_SKIP_PROBS,
            nmvc: DEFAULT_NMV_CONTEXT,
            initialized: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::FRAME_CONTEXTS_LOG2 as BITSTREAM_FRAME_CONTEXTS_LOG2;

    #[test]
    fn frame_contexts_constants() {
        assert_eq!(FRAME_CONTEXTS_LOG2, 2);
        assert_eq!(FRAME_CONTEXTS, 4);
        assert_eq!(TX_SIZES, 4);
    }

    /// The bitstream layer (header.rs) declares its own
    /// `FRAME_CONTEXTS_LOG2` — it must match the entropy layer's
    /// definition. (Both originate from libvpx; this test pins them
    /// in sync so future drift gets caught at build time.)
    #[test]
    fn frame_contexts_log2_consistent_with_bitstream_layer() {
        assert_eq!(FRAME_CONTEXTS_LOG2, BITSTREAM_FRAME_CONTEXTS_LOG2);
    }

    #[test]
    fn new_default_populates_all_fields() {
        let fc = FrameContext::new_default();
        assert!(fc.initialized);

        // Cross-check pointers / values against the source defaults.
        assert_eq!(fc.y_mode_prob, DEFAULT_IF_Y_PROBS);
        assert_eq!(fc.uv_mode_prob, DEFAULT_IF_UV_PROBS);
        assert_eq!(fc.partition_prob, DEFAULT_PARTITION_PROBS);
        assert_eq!(fc.coef_probs[0], DEFAULT_COEF_PROBS_4X4);
        assert_eq!(fc.coef_probs[1], DEFAULT_COEF_PROBS_8X8);
        assert_eq!(fc.coef_probs[2], DEFAULT_COEF_PROBS_16X16);
        assert_eq!(fc.coef_probs[3], DEFAULT_COEF_PROBS_32X32);
        assert_eq!(fc.switchable_interp_prob, DEFAULT_SWITCHABLE_INTERP_PROB);
        assert_eq!(fc.inter_mode_probs, DEFAULT_INTER_MODE_PROBS);
        assert_eq!(fc.intra_inter_prob, DEFAULT_INTRA_INTER_PROB);
        assert_eq!(fc.comp_inter_prob, DEFAULT_COMP_INTER_PROB);
        assert_eq!(fc.single_ref_prob, DEFAULT_SINGLE_REF_PROB);
        assert_eq!(fc.comp_ref_prob, DEFAULT_COMP_REF_PROB);
        assert_eq!(fc.tx_probs, DEFAULT_TX_PROBS);
        assert_eq!(fc.skip_probs, DEFAULT_SKIP_PROBS);
        assert_eq!(fc.nmvc, DEFAULT_NMV_CONTEXT);
    }

    /// Two independently-constructed defaults are bit-for-bit equal.
    #[test]
    fn default_is_deterministic() {
        let a = FrameContext::new_default();
        let b = FrameContext::new_default();
        assert_eq!(a, b);
    }
}
