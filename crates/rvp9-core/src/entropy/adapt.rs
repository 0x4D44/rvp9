//! VP9 backward-adaptation: fold a frame's decoded `FRAME_COUNTS` into
//! the previous `FRAME_CONTEXT`'s probability tables to produce the
//! next frame's tables.
//!
//! Mirrors libvpx, file:line cited per kernel:
//!
//! - [`weighted_prob`] / [`get_prob`] / [`get_binary_prob`] /
//!   [`merge_probs`] / [`mode_mv_merge_probs`] —
//!   `vpx_dsp/prob.h:48-95`.
//! - [`tree_merge_probs`] / `tree_merge_probs_impl` —
//!   `vpx_dsp/prob.c:26-47`.
//! - [`adapt_coef_probs`] — `vp9/common/vp9_entropy.c:1048-1100`
//!   (constants `:1048-1053`, per-tx-size loop `:1055-1082`, driver
//!   `:1084-1100`).
//! - [`adapt_mode_probs`] — `vp9/common/vp9_entropymode.c:340-409`
//!   (the `tx_counts_to_branch_counts_*` helpers at `:288-311`).
//! - [`adapt_mv_probs`] — `vp9/common/vp9_entropymv.c:154-189`.
//!
//! ## Pure-compute scope (Stage 3 Task E.1.a)
//!
//! These are **pure functions**. They consume an immutable
//! `&FrameContext` (the previous-frame bank) plus an immutable
//! `&FrameCounts` (the just-decoded frame's tally) and return a new
//! `FrameContext`. No decoder integration here: Task E.1.b will plumb
//! the per-block counts through the decode call chain; E.1.c will hook
//! the call site at end-of-frame. This module is the math kernel.
//!
//! ## "Counts double-count quirk" preserved (entropy LLD §7.4)
//!
//! Backward adaptation reads `counts` exactly as libvpx does — including
//! the per-worker baseline-double-count described in
//! `notes/lld/entropy.md` §7.4 / `notes/lld/tile-threading.md` §5.6.
//! When the threading integration lands, the merged `counts` will be
//! identical (bit-for-bit) to what libvpx produces under the same
//! double-count. This kernel does NOT compensate.
//!
//! ## Output shape
//!
//! Each `adapt_*_probs` returns a fresh, owned probability bank:
//!
//! - [`adapt_coef_probs`] returns `[CoeffProbsModel; TX_SIZES]`
//!   (1728 `u8`s).
//! - [`adapt_mode_probs`] returns a freshly-allocated [`FrameContext`]
//!   with the mode/tx/skip/partition fields freshly folded.
//!   `coef_probs` and `nmvc` are **copied through unchanged** from
//!   `pre_fc`; the caller is expected to install
//!   [`adapt_coef_probs`] / [`adapt_mv_probs`] outputs separately.
//! - [`adapt_mv_probs`] returns [`NmvContext`].
//!
//! This split mirrors libvpx's three independent entry points
//! (`vp9_adapt_coef_probs`, `vp9_adapt_mode_probs`,
//! `vp9_adapt_mv_probs`), each called separately by
//! `vp9_decode_frame` at end-of-frame.

use super::coef_probs::{
    COEF_BANDS, COEFF_CONTEXTS, CoeffProbsModel, PLANE_TYPES, REF_TYPES, UNCONSTRAINED_NODES,
};
use super::default_probs::{
    BLOCK_SIZE_GROUPS, COMP_INTER_CONTEXTS, DefaultTxProbs, INTER_MODE_CONTEXTS, INTER_MODES,
    INTRA_INTER_CONTEXTS, INTRA_MODES, PARTITION_CONTEXTS, PARTITION_TYPES, REF_CONTEXTS,
    SKIP_CONTEXTS, SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS,
};
use super::detokenize::{
    CoefCountsBlock, CoefFullCountsBlock, EOB_MODEL_TOKEN, EobBranchCountsBlock, ONE_TOKEN,
    TWO_TOKEN, ZERO_TOKEN,
};
use super::frame_context::{FrameContext, TX_SIZES};
use super::mv_probs::{
    CLASS0_SIZE, MV_OFFSET_BITS, NmvComponent, NmvComponentCounts, NmvContext, NmvCounts,
};

// -----------------------------------------------------------------------------
// Constants (libvpx vp9_entropy.c:1048-1053 + vpx_dsp/prob.h:37-82).
// -----------------------------------------------------------------------------

/// `MODE_MV_COUNT_SAT` (`vpx_dsp/prob.h:37`). Saturation cap on the
/// total count fed into [`mode_mv_merge_probs`].
pub const MODE_MV_COUNT_SAT: u32 = 20;

/// libvpx `count_to_update_factor[MODE_MV_COUNT_SAT + 1]`
/// (`vpx_dsp/prob.h:79-82`). Pre-computed table:
/// `count_to_update_factor[i] = MODE_MV_MAX_UPDATE_FACTOR (128) * i /
/// MODE_MV_COUNT_SAT`, rounded toward zero (integer division).
pub const COUNT_TO_UPDATE_FACTOR: [u32; MODE_MV_COUNT_SAT as usize + 1] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

/// `COEF_COUNT_SAT` (`vp9_entropy.c:1048`). Inter (non-key) coef count
/// saturation cap.
pub const COEF_COUNT_SAT: u32 = 24;
/// `COEF_MAX_UPDATE_FACTOR` (`vp9_entropy.c:1049`). Inter (non-key)
/// coef weighting.
pub const COEF_MAX_UPDATE_FACTOR: u32 = 112;
/// `COEF_COUNT_SAT_KEY` (`vp9_entropy.c:1050`). Coef count cap for
/// intra-only frames.
pub const COEF_COUNT_SAT_KEY: u32 = 24;
/// `COEF_MAX_UPDATE_FACTOR_KEY` (`vp9_entropy.c:1051`).
pub const COEF_MAX_UPDATE_FACTOR_KEY: u32 = 112;
/// `COEF_COUNT_SAT_AFTER_KEY` (`vp9_entropy.c:1052`). The first inter
/// frame after a keyframe adapts slightly faster (uf=128 vs 112).
pub const COEF_COUNT_SAT_AFTER_KEY: u32 = 24;
/// `COEF_MAX_UPDATE_FACTOR_AFTER_KEY` (`vp9_entropy.c:1053`).
pub const COEF_MAX_UPDATE_FACTOR_AFTER_KEY: u32 = 128;

/// `TX_SIZE_CONTEXTS` (`vp9_entropymode.h:25`).
pub const TX_SIZE_CONTEXTS: usize = 2;

// -----------------------------------------------------------------------------
// Count structs — the subset of libvpx `FRAME_COUNTS` consumed here.
// -----------------------------------------------------------------------------

/// Per-`(tx_size, plane_type, ref_type)` coefficient + EOB tallies.
/// Mirrors libvpx `cm->counts.coef[tx][type][ref]` +
/// `cm->counts.eob_branch[tx][type][ref]` (`vp9_entropymode.h:66-68`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefCountsBank {
    /// `[tx_size][plane_type][ref_type][band][ctx][token]` token tally
    /// (4 tokens: ZERO / ONE / TWO / EOB_MODEL).
    pub coef: [[[CoefCountsBlock; REF_TYPES]; PLANE_TYPES]; TX_SIZES],
    /// `[tx_size][plane_type][ref_type][band][ctx]` EOB-branch tally.
    pub eob_branch: [[[EobBranchCountsBlock; REF_TYPES]; PLANE_TYPES]; TX_SIZES],
}

impl Default for CoefCountsBank {
    fn default() -> Self {
        Self {
            coef: [[[[[[0u32; UNCONSTRAINED_NODES + 1]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
                PLANE_TYPES]; TX_SIZES],
            eob_branch: [[[[[0u32; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES]; PLANE_TYPES];
                TX_SIZES],
        }
    }
}

/// ENCODER-ONLY full 12-token coefficient histogram, the
/// `[tx_size][plane_type][ref_type]` bank of [`CoefFullCountsBlock`].
/// Mirrors libvpx `cpi->td.rd_counts.coef_counts` (`vp9_encoder.h:379`,
/// `vp9_coeff_count coef_counts[TX_SIZES][PLANE_TYPES]`). The decoder
/// never builds this — it only needs the 4-token model histogram for
/// backward adaptation. The encoder's TWO_LOOP coef-prob writer needs
/// the full distribution so its PIVOT_NODE model search can score the 8
/// downstream Pareto-8 nodes (`build_coef_branch_cts`). Boxed because it
/// is ~55 KiB and is only allocated when the encoder requests it.
pub type Coef12Bank = [[[CoefFullCountsBlock; REF_TYPES]; PLANE_TYPES]; TX_SIZES];

/// Allocate a zeroed [`Coef12Bank`] on the heap.
#[must_use]
pub fn new_coef12_bank() -> Box<Coef12Bank> {
    Box::new(
        [[[[[[0u32; super::detokenize::ENTROPY_TOKENS]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
            PLANE_TYPES]; TX_SIZES],
    )
}

impl CoefCountsBank {
    /// G5.1: additive merge for per-tile-worker count accumulation.
    /// Each tile worker decodes into its own owned `CoefCountsBank`;
    /// after the scope joins, the main thread folds the per-tile banks
    /// back into the decoder's running counter via repeated `add` calls.
    /// Mirrors libvpx's `cm->counts` merge in
    /// `vp9_decode_tiles_mt`'s post-join accumulation loop.
    pub fn add(&mut self, other: &Self) {
        for (tx_a, tx_b) in self.coef.iter_mut().zip(other.coef.iter()) {
            for (pt_a, pt_b) in tx_a.iter_mut().zip(tx_b.iter()) {
                for (rt_a, rt_b) in pt_a.iter_mut().zip(pt_b.iter()) {
                    for (band_a, band_b) in rt_a.iter_mut().zip(rt_b.iter()) {
                        for (ctx_a, ctx_b) in band_a.iter_mut().zip(band_b.iter()) {
                            for (a, b) in ctx_a.iter_mut().zip(ctx_b.iter()) {
                                *a += *b;
                            }
                        }
                    }
                }
            }
        }
        for (tx_a, tx_b) in self.eob_branch.iter_mut().zip(other.eob_branch.iter()) {
            for (pt_a, pt_b) in tx_a.iter_mut().zip(tx_b.iter()) {
                for (rt_a, rt_b) in pt_a.iter_mut().zip(pt_b.iter()) {
                    for (band_a, band_b) in rt_a.iter_mut().zip(rt_b.iter()) {
                        for (a, b) in band_a.iter_mut().zip(band_b.iter()) {
                            *a += *b;
                        }
                    }
                }
            }
        }
    }
}

/// TX-size selection counts. Mirrors libvpx `struct tx_counts`
/// (`vp9_entropymode.h:37-42`). `tx_totals` is libvpx bookkeeping for
/// `vp9_tx_counts_to_branch_counts_*` callers; backward adaptation
/// itself does not consume `tx_totals`, but we carry it for parity with
/// the upcoming decoder integration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TxCounts {
    /// 32×32-max contexts: per-context tally of which TX size was
    /// selected. Indexed `[ctx][TX_SIZE]`.
    pub p32x32: [[u32; TX_SIZES]; TX_SIZE_CONTEXTS],
    /// 16×16-max contexts: TX size up to 16×16 (3 outcomes).
    pub p16x16: [[u32; TX_SIZES - 1]; TX_SIZE_CONTEXTS],
    /// 8×8-max contexts: TX size up to 8×8 (2 outcomes).
    pub p8x8: [[u32; TX_SIZES - 2]; TX_SIZE_CONTEXTS],
    /// `tx_totals[TX_SIZE]` — total count of each TX size across all
    /// contexts. Bookkeeping only; not consumed by adapt.
    pub tx_totals: [u32; TX_SIZES],
}

impl TxCounts {
    /// G5.1: additive merge — see [`CoefCountsBank::add`].
    pub fn add(&mut self, other: &Self) {
        for (ra, rb) in self.p32x32.iter_mut().zip(other.p32x32.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.p16x16.iter_mut().zip(other.p16x16.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.p8x8.iter_mut().zip(other.p8x8.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (a, b) in self.tx_totals.iter_mut().zip(other.tx_totals.iter()) {
            *a += *b;
        }
    }
}

/// Aggregated per-frame decoded counts. Mirrors libvpx `FRAME_COUNTS`
/// (`vp9_entropymode.h:62-79`) at field level — every field is the
/// minimum needed by [`adapt_coef_probs`], [`adapt_mode_probs`], and
/// [`adapt_mv_probs`].
///
/// Field ordering preserves libvpx's declaration order; new fields
/// (e.g. segmentation counts) can be appended without disturbing
/// existing layout.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameCounts {
    /// `y_mode[BLOCK_SIZE_GROUPS][INTRA_MODES]` — counts per
    /// (block-size group, decoded Y intra mode).
    pub y_mode: [[u32; INTRA_MODES]; BLOCK_SIZE_GROUPS],
    /// `uv_mode[INTRA_MODES][INTRA_MODES]` — counts per
    /// (Y-mode context, decoded UV mode).
    pub uv_mode: [[u32; INTRA_MODES]; INTRA_MODES],
    /// `partition[PARTITION_CONTEXTS][PARTITION_TYPES]` — per-context
    /// partition-type counts.
    pub partition: [[u32; PARTITION_TYPES]; PARTITION_CONTEXTS],
    /// Coefficient counts (per-tx-size / plane / ref).
    pub coef: CoefCountsBank,
    /// `switchable_interp[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS]`.
    pub switchable_interp: [[u32; SWITCHABLE_FILTERS]; SWITCHABLE_FILTER_CONTEXTS],
    /// `inter_mode[INTER_MODE_CONTEXTS][INTER_MODES]`.
    pub inter_mode: [[u32; INTER_MODES]; INTER_MODE_CONTEXTS],
    /// `intra_inter[INTRA_INTER_CONTEXTS][2]` — `[#intra, #inter]`.
    pub intra_inter: [[u32; 2]; INTRA_INTER_CONTEXTS],
    /// `comp_inter[COMP_INTER_CONTEXTS][2]` — `[#single, #compound]`.
    pub comp_inter: [[u32; 2]; COMP_INTER_CONTEXTS],
    /// `single_ref[REF_CONTEXTS][2][2]` — first axis is the two
    /// pyramid branches, last axis is `[bit=0, bit=1]`.
    pub single_ref: [[[u32; 2]; 2]; REF_CONTEXTS],
    /// `comp_ref[REF_CONTEXTS][2]`.
    pub comp_ref: [[u32; 2]; REF_CONTEXTS],
    /// `tx` — per-context TX-size selection counts.
    pub tx: TxCounts,
    /// `skip[SKIP_CONTEXTS][2]` — `[#not-skip, #skip]`.
    pub skip: [[u32; 2]; SKIP_CONTEXTS],
    /// MV counts.
    pub mv: NmvCounts,
    /// ENCODER-ONLY full 12-token coefficient histogram. `None` on the
    /// decoder path and on encoder paths that don't need TWO_LOOP coef
    /// updates. When `Some`, the tile writer's per-TX `encode_coefs`
    /// bumps the exact 12-token bucket in parallel with the 4-token
    /// `coef` model histogram, feeding the compressed-header TWO_LOOP
    /// savings search. Boxed + optional so the decoder's
    /// `FrameCounts::default()` stays a cheap all-zero value and its
    /// `==` / `add` behaviour is unchanged (always `None`).
    pub coef_full: Option<Box<Coef12Bank>>,
}

impl FrameCounts {
    /// G5.1: additive merge — sum every field of `other` into `self`.
    /// Used after per-tile-worker scope join to fold worker-local
    /// counts back into the decoder's running `mode_counts` field.
    /// Mirrors libvpx `cm->counts` accumulation in
    /// `vp9_decode_tiles_mt`.
    pub fn add(&mut self, other: &Self) {
        self.add_non_coef(other);
        self.add_coef(other);
    }

    /// Merge only coefficient histograms from a tile-local count bank.
    ///
    /// Encoders that record coefficient tokens during tile-local block
    /// encoding use this before the compressed header is written. Their
    /// mode/MV banks are still empty, so walking those banks would be wasted
    /// work.
    pub fn add_coef(&mut self, other: &Self) {
        self.coef.add(&other.coef);
        // ENCODER-ONLY: fold the full 12-token histogram when present on
        // both sides (per-tile-worker merge). The decoder leaves this
        // `None`, so this is a no-op on the decode path.
        if let (Some(dst), Some(src)) = (self.coef_full.as_mut(), other.coef_full.as_ref()) {
            for (tx_a, tx_b) in dst.iter_mut().zip(src.iter()) {
                for (pt_a, pt_b) in tx_a.iter_mut().zip(tx_b.iter()) {
                    for (rt_a, rt_b) in pt_a.iter_mut().zip(pt_b.iter()) {
                        for (band_a, band_b) in rt_a.iter_mut().zip(rt_b.iter()) {
                            for (ctx_a, ctx_b) in band_a.iter_mut().zip(band_b.iter()) {
                                for (a, b) in ctx_a.iter_mut().zip(ctx_b.iter()) {
                                    *a += *b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Merge every non-coefficient field from a tile-local count bank.
    ///
    /// Retained-token replay gathers coefficient histograms before the
    /// compressed header, then gathers mode/MV counts during final packing.
    /// This avoids scanning the known-zero coefficient banks of those final
    /// tile-local counts.
    pub fn add_non_coef(&mut self, other: &Self) {
        for (ra, rb) in self.y_mode.iter_mut().zip(other.y_mode.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.uv_mode.iter_mut().zip(other.uv_mode.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.partition.iter_mut().zip(other.partition.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self
            .switchable_interp
            .iter_mut()
            .zip(other.switchable_interp.iter())
        {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.inter_mode.iter_mut().zip(other.inter_mode.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.intra_inter.iter_mut().zip(other.intra_inter.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.comp_inter.iter_mut().zip(other.comp_inter.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        for (ra, rb) in self.single_ref.iter_mut().zip(other.single_ref.iter()) {
            for (rra, rrb) in ra.iter_mut().zip(rb.iter()) {
                for (a, b) in rra.iter_mut().zip(rrb.iter()) {
                    *a += *b;
                }
            }
        }
        for (ra, rb) in self.comp_ref.iter_mut().zip(other.comp_ref.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        self.tx.add(&other.tx);
        for (ra, rb) in self.skip.iter_mut().zip(other.skip.iter()) {
            for (a, b) in ra.iter_mut().zip(rb.iter()) {
                *a += *b;
            }
        }
        self.mv.add(&other.mv);
    }
}

// -----------------------------------------------------------------------------
// Primitives — `vpx_dsp/prob.h:48-95`, `vpx_dsp/prob.c:26-47`.
// -----------------------------------------------------------------------------

/// `vpx_dsp/prob.h:48-56`. Numerator/denominator → probability with
/// the clamp `p ∈ [1, 255]`.
///
/// # Panics
///
/// Panics if `den == 0` — matches the `assert(den != 0)` libvpx
/// upholds at every call site.
#[inline]
#[must_use]
pub fn get_prob(num: u32, den: u32) -> u8 {
    assert!(den != 0, "get_prob: denominator must be non-zero");
    // p = (num * 256 + den/2) / den  (with u64 promotion to avoid overflow)
    let p = ((u64::from(num) * 256 + u64::from(den >> 1)) / u64::from(den)) as i32;
    // Bit-twiddle clamp identical to libvpx:
    //   const int clipped = p | ((255 - p) >> 23) | (p == 0);
    // - If p > 255: ((255 - p) >> 23) is a long string of 1s (arithmetic
    //   sign-extension on the negative result), saturating to 255.
    // - If p == 0: ored with 1 to lift it to 1.
    let clipped = p | ((255 - p) >> 23) | i32::from(p == 0);
    clipped as u8
}

/// `vpx_dsp/prob.h:58-62`. Binary-branch shortcut. Returns
/// [`super::VPX_PROB_HALF`] when both counts are zero.
#[inline]
#[must_use]
pub fn get_binary_prob(n0: u32, n1: u32) -> u8 {
    let den = n0 + n1;
    if den == 0 {
        return 128;
    }
    get_prob(n0, den)
}

/// `vpx_dsp/prob.h:64-67`. Weighted average of two probs:
/// `ROUND_POWER_OF_TWO(prob1 * (256 - factor) + prob2 * factor, 8)`.
///
/// `factor` is expected to be in `[0, 256]` (caller's responsibility);
/// values outside that range still produce a defined `u8` result via
/// saturating semantics from the truncating cast.
#[inline]
#[must_use]
pub fn weighted_prob(prob1: u8, prob2: u8, factor: u32) -> u8 {
    let sum = u32::from(prob1) * (256 - factor) + u32::from(prob2) * factor;
    ((sum + 128) >> 8) as u8
}

/// `vpx_dsp/prob.h:69-76`. The canonical "merge a count into a prob"
/// fold used for coefficient adaptation.
///
/// - `pre_prob`: the previous frame's probability for this slot.
/// - `ct`: `[count_of_zero_branch, count_of_one_branch]`.
/// - `count_sat`: saturates the total count fed to the update factor.
/// - `max_update_factor`: caps the weighting toward the new evidence.
#[inline]
#[must_use]
pub fn merge_probs(pre_prob: u8, ct: [u32; 2], count_sat: u32, max_update_factor: u32) -> u8 {
    let prob = get_binary_prob(ct[0], ct[1]);
    let count = (ct[0] + ct[1]).min(count_sat);
    let factor = max_update_factor * count / count_sat;
    weighted_prob(pre_prob, prob, factor)
}

/// `vpx_dsp/prob.h:84-95`. The mode/MV-specific merge: capped at
/// [`MODE_MV_COUNT_SAT`] and indexed into [`COUNT_TO_UPDATE_FACTOR`]
/// (a pre-divided LUT, which saves the per-call division
/// `merge_probs` does). Distinct behaviour from `merge_probs`:
/// if `ct[0] + ct[1] == 0`, returns `pre_prob` unchanged (no
/// half-prob neutral output).
#[inline]
#[must_use]
pub fn mode_mv_merge_probs(pre_prob: u8, ct: [u32; 2]) -> u8 {
    let den = ct[0] + ct[1];
    if den == 0 {
        return pre_prob;
    }
    let count = den.min(MODE_MV_COUNT_SAT);
    let factor = COUNT_TO_UPDATE_FACTOR[count as usize];
    let prob = get_prob(ct[0], den);
    weighted_prob(pre_prob, prob, factor)
}

// -----------------------------------------------------------------------------
// `tree_merge_probs` — `vpx_dsp/prob.c:26-47`.
// -----------------------------------------------------------------------------

fn tree_merge_probs_impl(
    i: usize,
    tree: &[i8],
    pre_probs: &[u8],
    counts: &[u32],
    probs: &mut [u8],
) -> u32 {
    let l = tree[i];
    let left_count = if l <= 0 {
        counts[(-l) as usize]
    } else {
        tree_merge_probs_impl(l as usize, tree, pre_probs, counts, probs)
    };
    let r = tree[i + 1];
    let right_count = if r <= 0 {
        counts[(-r) as usize]
    } else {
        tree_merge_probs_impl(r as usize, tree, pre_probs, counts, probs)
    };
    probs[i >> 1] = mode_mv_merge_probs(pre_probs[i >> 1], [left_count, right_count]);
    left_count + right_count
}

/// `vpx_tree_merge_probs` (`vpx_dsp/prob.c:44-47`). Walks the binary
/// tree encoded in `tree` (libvpx's flat `vpx_tree_index` layout —
/// non-positive entries are leaf-token negations, positive entries are
/// internal-node indices), gathers per-leaf counts, and writes one
/// merged probability into `probs[i >> 1]` for each internal node.
///
/// The tree shape constants live alongside their probability tables:
/// `vp9_intra_mode_tree`, `vp9_inter_mode_tree`, `vp9_partition_tree`,
/// `vp9_switchable_interp_tree` (entropymode.c), and
/// `vp9_mv_joint_tree`, `vp9_mv_class_tree`, `vp9_mv_class0_tree`,
/// `vp9_mv_fp_tree` (entropymv.c).
pub fn tree_merge_probs(tree: &[i8], pre_probs: &[u8], counts: &[u32], probs: &mut [u8]) {
    let _ = tree_merge_probs_impl(0, tree, pre_probs, counts, probs);
}

// -----------------------------------------------------------------------------
// Tree definitions — verbatim from libvpx (entropymode.c:245-264,
// entropymv.c:14-28). Re-stated here so the adapt kernels have
// freestanding access; the bitstream/mode-info layers can adopt these
// in a follow-up consolidation.
// -----------------------------------------------------------------------------

/// `vp9_intra_mode_tree` (`vp9_entropymode.c:245-255`). Size:
/// `TREE_SIZE(INTRA_MODES) = 2 * 10 - 2 = 18`.
#[rustfmt::skip]
pub const VP9_INTRA_MODE_TREE: [i8; 18] = [
    -0,  2,  // -DC_PRED, ...
    -9,  4,  // -TM_PRED, ...
    -1,  6,  // -V_PRED, ...
     8, 12,  // COM_NODE
    -2, 10,  // -H_PRED, ...
    -4, -5,  // -D135_PRED, -D117_PRED
    -3, 14,  // -D45_PRED, ...
    -8, 16,  // -D63_PRED, ...
    -6, -7,  // -D153_PRED, -D207_PRED
];

/// `vp9_inter_mode_tree` (`vp9_entropymode.c:257-260`). Size:
/// `TREE_SIZE(INTER_MODES) = 6`. `INTER_OFFSET(mode) = mode - NEARESTMV`
/// gives the 0-based indexes used here.
#[rustfmt::skip]
pub const VP9_INTER_MODE_TREE: [i8; 6] = [
    -2, 2,    // -INTER_OFFSET(ZEROMV) = -2,  ...
    -0, 4,    // -INTER_OFFSET(NEARESTMV) = 0, ...
    -1, -3,   // -INTER_OFFSET(NEARMV) = -1, -INTER_OFFSET(NEWMV) = -3
];

/// `vp9_partition_tree` (`vp9_entropymode.c:262-264`). Size:
/// `TREE_SIZE(PARTITION_TYPES) = 6`.
#[rustfmt::skip]
pub const VP9_PARTITION_TREE: [i8; 6] = [
    -0, 2,    // -PARTITION_NONE, ...
    -1, 4,    // -PARTITION_HORZ, ...
    -2, -3,   // -PARTITION_VERT, -PARTITION_SPLIT
];

/// `vp9_switchable_interp_tree` (`vp9_entropymode.c:337-338`). Size:
/// `TREE_SIZE(SWITCHABLE_FILTERS) = 4`.
#[rustfmt::skip]
pub const VP9_SWITCHABLE_INTERP_TREE: [i8; 4] = [
    -0, 2,    // -EIGHTTAP, ...
    -1, -2,   // -EIGHTTAP_SMOOTH, -EIGHTTAP_SHARP
];

/// `vp9_mv_joint_tree` (`vp9_entropymv.c:14-16`). Size:
/// `TREE_SIZE(MV_JOINTS) = 6`.
#[rustfmt::skip]
pub const VP9_MV_JOINT_TREE: [i8; 6] = [
    -0, 2,    // -MV_JOINT_ZERO, ...
    -1, 4,    // -MV_JOINT_HNZVZ, ...
    -2, -3,   // -MV_JOINT_HZVNZ, -MV_JOINT_HNZVNZ
];

/// `vp9_mv_class_tree` (`vp9_entropymv.c:18-23`). Size:
/// `TREE_SIZE(MV_CLASSES) = 20`.
#[rustfmt::skip]
pub const VP9_MV_CLASS_TREE: [i8; 20] = [
    -0,  2,   // -MV_CLASS_0, ...
    -1,  4,   // -MV_CLASS_1, ...
     6,  8,   // ...
    -2, -3,   // -MV_CLASS_2, -MV_CLASS_3
    10, 12,
    -4, -5,   // -MV_CLASS_4, -MV_CLASS_5
    -6, 14,   // -MV_CLASS_6, ...
    16, 18,
    -7, -8,   // -MV_CLASS_7, -MV_CLASS_8
    -9, -10,  // -MV_CLASS_9, -MV_CLASS_10
];

/// `vp9_mv_class0_tree` (`vp9_entropymv.c:25`). Size:
/// `TREE_SIZE(CLASS0_SIZE) = 2`.
pub const VP9_MV_CLASS0_TREE: [i8; 2] = [-0, -1];

/// `vp9_mv_fp_tree` (`vp9_entropymv.c:27-28`). Size:
/// `TREE_SIZE(MV_FP_SIZE) = 6`.
#[rustfmt::skip]
pub const VP9_MV_FP_TREE: [i8; 6] = [
    -0, 2,
    -1, 4,
    -2, -3,
];

// -----------------------------------------------------------------------------
// `adapt_coef_probs` — `vp9_entropy.c:1055-1100`.
// -----------------------------------------------------------------------------

#[inline]
const fn band_coeff_contexts(band: usize) -> usize {
    if band == 0 { 3 } else { COEFF_CONTEXTS }
}

/// Selection of `(count_sat, max_update_factor)` per
/// `vp9_adapt_coef_probs` (`vp9_entropy.c:1088-1097`).
///
/// - `intra_only`: this frame is a keyframe or intra-only frame
///   (`frame_is_intra_only(cm)` in libvpx).
/// - `prev_was_key`: the previous frame's `frame_type` was
///   `KEY_FRAME` (libvpx `cm->last_frame_type`).
#[inline]
#[must_use]
pub const fn coef_adapt_params(intra_only: bool, prev_was_key: bool) -> (u32, u32) {
    if intra_only {
        (COEF_COUNT_SAT_KEY, COEF_MAX_UPDATE_FACTOR_KEY)
    } else if prev_was_key {
        (COEF_COUNT_SAT_AFTER_KEY, COEF_MAX_UPDATE_FACTOR_AFTER_KEY)
    } else {
        (COEF_COUNT_SAT, COEF_MAX_UPDATE_FACTOR)
    }
}

/// Backward-adapt the 4 per-TX-size coefficient probability tables in
/// one pass. Mirrors libvpx `vp9_adapt_coef_probs` +
/// `adapt_coef_probs` (`vp9_entropy.c:1055-1100`).
///
/// Returns a freshly-allocated `[CoeffProbsModel; TX_SIZES]`. The
/// caller picks the correct `(count_sat, max_update_factor)` via
/// [`coef_adapt_params`].
///
/// Note: libvpx skips this kernel entirely on the very first decoded
/// frame (it's the keyframe-reset path). Backward adapt is only called
/// AFTER a frame's symbols have been decoded into `counts`. The pure
/// math handles the keyframe-counts case correctly (it returns
/// `pre_probs` essentially unchanged because keyframe `counts` is zero
/// in libvpx's call sequence — `vp9_adapt_coef_probs` is still invoked
/// for keyframes, but the counts have just been reset).
#[must_use]
pub fn adapt_coef_probs(
    pre_probs: &[CoeffProbsModel; TX_SIZES],
    counts: &CoefCountsBank,
    count_sat: u32,
    max_update_factor: u32,
) -> [CoeffProbsModel; TX_SIZES] {
    let mut out = *pre_probs;

    for tx in 0..TX_SIZES {
        for i in 0..PLANE_TYPES {
            for j in 0..REF_TYPES {
                for k in 0..COEF_BANDS {
                    let valid_ctx = band_coeff_contexts(k);
                    for l in 0..valid_ctx {
                        let token_counts = &counts.coef[tx][i][j][k][l];
                        let n0 = token_counts[ZERO_TOKEN];
                        let n1 = token_counts[ONE_TOKEN];
                        let n2 = token_counts[TWO_TOKEN];
                        let neob = token_counts[EOB_MODEL_TOKEN];
                        let eob_total = counts.eob_branch[tx][i][j][k][l];
                        // Libvpx encodes the three unconstrained-tree
                        // branch counts as:
                        //   node 0 (EOB):  ct = [#eob,   #non-eob]
                        //   node 1 (ZERO): ct = [#zero,  #one+#two]
                        //   node 2 (ONE):  ct = [#one,   #two]
                        // `eob_total` is "every position at which an
                        // EOB decision was attempted"; `neob` is "we
                        // emitted an EOB". So `eob_total - neob` is
                        // the non-eob count. libvpx uses signed `int`
                        // here; we mirror with saturating sub for
                        // safety (if eob_total < neob the caller's
                        // counts are corrupt; libvpx would underflow).
                        let branch_ct: [[u32; 2]; UNCONSTRAINED_NODES] = [
                            [neob, eob_total.saturating_sub(neob)],
                            [n0, n1 + n2],
                            [n1, n2],
                        ];
                        for m in 0..UNCONSTRAINED_NODES {
                            out[tx][i][j][k][l][m] = merge_probs(
                                pre_probs[tx][i][j][k][l][m],
                                branch_ct[m],
                                count_sat,
                                max_update_factor,
                            );
                        }
                    }
                }
            }
        }
    }

    out
}

// -----------------------------------------------------------------------------
// `adapt_mode_probs` — `vp9_entropymode.c:340-409`.
// -----------------------------------------------------------------------------

/// `tx_counts_to_branch_counts_32x32` (`vp9_entropymode.c:288-297`).
#[inline]
fn tx_counts_to_branch_counts_32x32(c: &[u32; TX_SIZES]) -> [[u32; 2]; TX_SIZES - 1] {
    [
        [c[0], c[1] + c[2] + c[3]],
        [c[1], c[2] + c[3]],
        [c[2], c[3]],
    ]
}

/// `tx_counts_to_branch_counts_16x16` (`vp9_entropymode.c:299-305`).
#[inline]
fn tx_counts_to_branch_counts_16x16(c: &[u32; TX_SIZES - 1]) -> [[u32; 2]; TX_SIZES - 2] {
    [[c[0], c[1] + c[2]], [c[1], c[2]]]
}

/// `tx_counts_to_branch_counts_8x8` (`vp9_entropymode.c:307-311`).
#[inline]
fn tx_counts_to_branch_counts_8x8(c: &[u32; TX_SIZES - 2]) -> [[u32; 2]; TX_SIZES - 3] {
    [[c[0], c[1]]]
}

/// Backward-adapt the mode/partition/skip/tx-size/ref probabilities
/// in `FrameContext`. Mirrors `vp9_adapt_mode_probs`
/// (`vp9_entropymode.c:340-409`).
///
/// - `pre_fc`: the previous-frame probability bank.
/// - `counts`: this frame's tallies.
/// - `interp_switchable`: `cm->interp_filter == SWITCHABLE` (only
///   then do switchable_interp_prob slots get adapted).
/// - `tx_mode_select`: `cm->tx_mode == TX_MODE_SELECT` (only then do
///   `tx_probs` get adapted).
///
/// Returns a freshly-built [`FrameContext`]. **The returned context's
/// `coef_probs` and `nmvc` fields are copied unchanged from
/// `pre_fc`** — those are folded by [`adapt_coef_probs`] and
/// [`adapt_mv_probs`] independently.
#[must_use]
pub fn adapt_mode_probs(
    pre_fc: &FrameContext,
    counts: &FrameCounts,
    interp_switchable: bool,
    tx_mode_select: bool,
) -> FrameContext {
    let mut fc = pre_fc.clone();

    // intra_inter_prob — line 346-348.
    for i in 0..INTRA_INTER_CONTEXTS {
        fc.intra_inter_prob[i] =
            mode_mv_merge_probs(pre_fc.intra_inter_prob[i], counts.intra_inter[i]);
    }
    // comp_inter_prob — line 349-351.
    for i in 0..COMP_INTER_CONTEXTS {
        fc.comp_inter_prob[i] =
            mode_mv_merge_probs(pre_fc.comp_inter_prob[i], counts.comp_inter[i]);
    }
    // comp_ref_prob — line 352-354.
    for i in 0..REF_CONTEXTS {
        fc.comp_ref_prob[i] = mode_mv_merge_probs(pre_fc.comp_ref_prob[i], counts.comp_ref[i]);
    }
    // single_ref_prob — line 355-358.
    for i in 0..REF_CONTEXTS {
        for j in 0..2 {
            fc.single_ref_prob[i][j] =
                mode_mv_merge_probs(pre_fc.single_ref_prob[i][j], counts.single_ref[i][j]);
        }
    }

    // inter_mode_probs — line 360-362 (tree merge).
    for i in 0..INTER_MODE_CONTEXTS {
        tree_merge_probs(
            &VP9_INTER_MODE_TREE,
            &pre_fc.inter_mode_probs[i],
            &counts.inter_mode[i],
            &mut fc.inter_mode_probs[i],
        );
    }

    // y_mode_prob — line 364-366.
    for i in 0..BLOCK_SIZE_GROUPS {
        tree_merge_probs(
            &VP9_INTRA_MODE_TREE,
            &pre_fc.y_mode_prob[i],
            &counts.y_mode[i],
            &mut fc.y_mode_prob[i],
        );
    }

    // uv_mode_prob — line 368-370.
    for i in 0..INTRA_MODES {
        tree_merge_probs(
            &VP9_INTRA_MODE_TREE,
            &pre_fc.uv_mode_prob[i],
            &counts.uv_mode[i],
            &mut fc.uv_mode_prob[i],
        );
    }

    // partition_prob — line 372-374.
    for i in 0..PARTITION_CONTEXTS {
        tree_merge_probs(
            &VP9_PARTITION_TREE,
            &pre_fc.partition_prob[i],
            &counts.partition[i],
            &mut fc.partition_prob[i],
        );
    }

    // switchable_interp_prob — line 376-381.
    if interp_switchable {
        for i in 0..SWITCHABLE_FILTER_CONTEXTS {
            tree_merge_probs(
                &VP9_SWITCHABLE_INTERP_TREE,
                &pre_fc.switchable_interp_prob[i],
                &counts.switchable_interp[i],
                &mut fc.switchable_interp_prob[i],
            );
        }
    }

    // tx_probs — line 383-404. Per-context, per-sub-table fold.
    if tx_mode_select {
        adapt_tx_probs(&pre_fc.tx_probs, &counts.tx, &mut fc.tx_probs);
    }

    // skip_probs — line 406-408.
    for i in 0..SKIP_CONTEXTS {
        fc.skip_probs[i] = mode_mv_merge_probs(pre_fc.skip_probs[i], counts.skip[i]);
    }

    fc
}

fn adapt_tx_probs(pre: &DefaultTxProbs, counts: &TxCounts, out: &mut DefaultTxProbs) {
    for i in 0..TX_SIZE_CONTEXTS {
        // p8x8 — TX_SIZES - 3 = 1 prob per context.
        let ct8 = tx_counts_to_branch_counts_8x8(&counts.p8x8[i]);
        for (j, ct) in ct8.iter().enumerate() {
            out.p8x8[i][j] = mode_mv_merge_probs(pre.p8x8[i][j], *ct);
        }

        // p16x16 — TX_SIZES - 2 = 2 probs per context.
        let ct16 = tx_counts_to_branch_counts_16x16(&counts.p16x16[i]);
        for (j, ct) in ct16.iter().enumerate() {
            out.p16x16[i][j] = mode_mv_merge_probs(pre.p16x16[i][j], *ct);
        }

        // p32x32 — TX_SIZES - 1 = 3 probs per context.
        let ct32 = tx_counts_to_branch_counts_32x32(&counts.p32x32[i]);
        for (j, ct) in ct32.iter().enumerate() {
            out.p32x32[i][j] = mode_mv_merge_probs(pre.p32x32[i][j], *ct);
        }
    }
}

// -----------------------------------------------------------------------------
// `adapt_mv_probs` — `vp9_entropymv.c:154-189`.
// -----------------------------------------------------------------------------

/// Backward-adapt the motion-vector probabilities. Mirrors
/// `vp9_adapt_mv_probs` (`vp9_entropymv.c:154-189`). Returns a freshly
/// built [`NmvContext`].
///
/// - `pre`: the previous-frame MV bank.
/// - `counts`: this frame's MV tallies.
/// - `allow_hp`: `cm->allow_high_precision_mv`. When false, the
///   `class0_hp` / `hp` probabilities are copied through unchanged
///   (libvpx skips the merge for those two slots when `allow_hp == 0`).
#[must_use]
pub fn adapt_mv_probs(pre: &NmvContext, counts: &NmvCounts, allow_hp: bool) -> NmvContext {
    let mut fc = pre.clone();

    // joints — line 161-162.
    tree_merge_probs(
        &VP9_MV_JOINT_TREE,
        &pre.joints,
        &counts.joints,
        &mut fc.joints,
    );

    for i in 0..2 {
        adapt_mv_component(&pre.comps[i], &counts.comps[i], &mut fc.comps[i], allow_hp);
    }

    fc
}

fn adapt_mv_component(
    pre: &NmvComponent,
    counts: &NmvComponentCounts,
    out: &mut NmvComponent,
    allow_hp: bool,
) {
    // sign — line 169.
    out.sign = mode_mv_merge_probs(pre.sign, counts.sign);

    // classes — line 170-171.
    tree_merge_probs(
        &VP9_MV_CLASS_TREE,
        &pre.classes,
        &counts.classes,
        &mut out.classes,
    );

    // class0 (binary tree, CLASS0_BITS = 1) — line 172-173.
    tree_merge_probs(
        &VP9_MV_CLASS0_TREE,
        &pre.class0,
        &counts.class0,
        &mut out.class0,
    );

    // bits — line 175-176.
    for j in 0..MV_OFFSET_BITS {
        out.bits[j] = mode_mv_merge_probs(pre.bits[j], counts.bits[j]);
    }

    // class0_fp — line 178-180.
    for j in 0..CLASS0_SIZE {
        tree_merge_probs(
            &VP9_MV_FP_TREE,
            &pre.class0_fp[j],
            &counts.class0_fp[j],
            &mut out.class0_fp[j],
        );
    }

    // fp — line 182.
    tree_merge_probs(&VP9_MV_FP_TREE, &pre.fp, &counts.fp, &mut out.fp);

    // class0_hp / hp — line 184-187. Skipped when allow_hp == 0
    // (the previous values stay; we did `out = pre.clone()` above so
    // that's already the case — but spelling it out for clarity).
    if allow_hp {
        out.class0_hp = mode_mv_merge_probs(pre.class0_hp, counts.class0_hp);
        out.hp = mode_mv_merge_probs(pre.hp, counts.hp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::mv_probs::DEFAULT_NMV_CONTEXT;

    // -------------------------------------------------------------------------
    // weighted_prob / get_prob / get_binary_prob — primitive math.
    // -------------------------------------------------------------------------

    #[test]
    fn weighted_prob_endpoints() {
        // factor=0  -> all weight on prob1.
        assert_eq!(weighted_prob(100, 200, 0), 100);
        // factor=256 -> all weight on prob2.
        assert_eq!(weighted_prob(100, 200, 256), 200);
        // factor=128 -> equal weight, round-half-up.
        // (100 * 128 + 200 * 128 + 128) >> 8 = (12800+25600+128)/256 = 38528/256 = 150.5 -> 150
        // Hand-trace: 100 * (256-128) = 12800; 200 * 128 = 25600; sum=38400; +128=38528; >>8=150.
        assert_eq!(weighted_prob(100, 200, 128), 150);
    }

    #[test]
    fn get_prob_clamps_to_one_and_255() {
        // Tiny num -> would be 0, clamped to 1.
        assert_eq!(get_prob(0, 1), 1);
        // num == den / 2 -> ~128.
        assert_eq!(get_prob(50, 100), 128);
        // num == den -> 256 raw, clamped to 255.
        assert_eq!(get_prob(100, 100), 255);
        // num > den would similarly clamp.
        assert_eq!(get_prob(200, 100), 255);
    }

    #[test]
    fn get_binary_prob_zero_counts_is_half() {
        assert_eq!(get_binary_prob(0, 0), 128);
        // 1:1 ratio -> ~128.
        assert_eq!(get_binary_prob(10, 10), 128);
        // All zero-branch -> clamped to 255.
        assert_eq!(get_binary_prob(10, 0), 255);
    }

    // -------------------------------------------------------------------------
    // merge_probs — count_sat=24, max_uf=112 (the COEF_* numbers).
    // -------------------------------------------------------------------------

    /// Hand-traced from libvpx semantics:
    ///
    /// ```text
    /// pre=100, ct=[10, 20], count_sat=24, max_uf=112.
    /// prob = get_binary_prob(10, 20) = get_prob(10, 30) =
    ///   (10 * 256 + 15) / 30 = 2575 / 30 = 85 (in [1,255]).
    /// count = min(30, 24) = 24.
    /// factor = 112 * 24 / 24 = 112.
    /// weighted = ROUND_POW2(100 * (256-112) + 85 * 112, 8)
    ///          = ROUND_POW2(100*144 + 85*112, 8)
    ///          = ROUND_POW2(14400 + 9520, 8)
    ///          = ROUND_POW2(23920, 8)
    ///          = (23920 + 128) >> 8 = 24048 >> 8 = 93.9... -> 93
    /// ```
    #[test]
    fn merge_probs_hand_trace_24_112() {
        assert_eq!(merge_probs(100, [10, 20], 24, 112), 93);
    }

    /// Zero counts -> get_binary_prob returns 128; count=0, factor=0;
    /// weighted_prob(pre, 128, 0) = pre. So zero counts is a no-op.
    #[test]
    fn merge_probs_zero_counts_no_op() {
        for pre in [1u8, 50, 128, 200, 255] {
            assert_eq!(merge_probs(pre, [0, 0], 24, 112), pre);
        }
    }

    /// Saturated count: ct=[100, 100] -> total=200, clamped to 24,
    /// factor=112; prob=get_prob(100,200)=128; weighted(pre,128,112).
    #[test]
    fn merge_probs_saturated_count() {
        // weighted(50, 128, 112) = ((50*(256-112)) + 128*112 + 128) >> 8
        //                        = (50*144 + 128*112 + 128) >> 8
        //                        = (7200 + 14336 + 128) >> 8
        //                        = 21664 >> 8 = 84.625 -> 84
        assert_eq!(merge_probs(50, [100, 100], 24, 112), 84);
    }

    // -------------------------------------------------------------------------
    // mode_mv_merge_probs — count_to_update_factor LUT.
    // -------------------------------------------------------------------------

    #[test]
    fn mode_mv_merge_zero_counts_passthrough() {
        for pre in [1u8, 50, 128, 255] {
            assert_eq!(mode_mv_merge_probs(pre, [0, 0]), pre);
        }
    }

    /// ct=[5, 5], total=10. count=10, factor=COUNT_TO_UPDATE_FACTOR[10]=64.
    /// prob=get_prob(5,10)=128. weighted(100, 128, 64) =
    ///   (100*192 + 128*64 + 128) >> 8 = (19200 + 8192 + 128) >> 8 =
    ///   27520 >> 8 = 107.5 -> 107
    #[test]
    fn mode_mv_merge_hand_trace() {
        assert_eq!(mode_mv_merge_probs(100, [5, 5]), 107);
    }

    /// ct=[20, 30], total=50 -> capped at 20, factor=128. prob=get_prob(20,50)=103.
    /// (20*256+25)/50 = 5145/50 = 102.9 -> 102. Recompute:
    /// 20*256 = 5120; + (50>>1)=25 -> 5145; /50 = 102 (integer truncate).
    /// Then bit-twiddle clamp: 102 not 0, not > 255, returns 102.
    /// weighted(150, 102, 128) = (150 * 128 + 102 * 128 + 128) >> 8 =
    ///                         = (19200 + 13056 + 128) >> 8 =
    ///                         = 32384 >> 8 = 126.5 -> 126
    #[test]
    fn mode_mv_merge_saturates() {
        assert_eq!(mode_mv_merge_probs(150, [20, 30]), 126);
    }

    // -------------------------------------------------------------------------
    // tree_merge_probs — exercise the recursive walk against a known
    // tree (partition tree: 4 leaves, 3 internal nodes).
    // -------------------------------------------------------------------------

    #[test]
    fn tree_merge_partition_tree_zero_counts() {
        let pre_probs = [50u8, 100, 150];
        let counts = [0u32; PARTITION_TYPES];
        let mut probs = [0u8; PARTITION_TYPES - 1];
        tree_merge_probs(&VP9_PARTITION_TREE, &pre_probs, &counts, &mut probs);
        // Zero counts -> mode_mv_merge_probs returns pre.
        assert_eq!(probs, pre_probs);
    }

    /// All counts on PARTITION_NONE (index 0) -> first internal node
    /// sees ct=[lots, 0]; second/third see ct=[0, 0] and stay at pre.
    #[test]
    fn tree_merge_partition_tree_skewed() {
        let pre_probs = [100u8, 120, 140];
        let mut counts = [0u32; PARTITION_TYPES];
        counts[0] = 8; // PARTITION_NONE
        let mut probs = [0u8; PARTITION_TYPES - 1];
        tree_merge_probs(&VP9_PARTITION_TREE, &pre_probs, &counts, &mut probs);

        // Node 0 (root): left=counts[0]=8, right = subtree count = 0.
        // ct=[8, 0]. den=8. count=min(8, 20)=8. factor=COUNT_TO_UPDATE_FACTOR[8]=51.
        // prob = get_prob(8, 8) = (8 * 256 + 4) / 8 = 256.5 -> 256; clamp -> 255.
        // weighted(100, 255, 51) = (100 * (256-51) + 255 * 51 + 128) >> 8
        //                        = (100*205 + 255*51 + 128) >> 8
        //                        = (20500 + 13005 + 128) >> 8
        //                        = 33633 >> 8 = 131.4 -> 131
        assert_eq!(probs[0], 131);
        // Nodes 1, 2 saw zero counts -> unchanged.
        assert_eq!(probs[1], 120);
        assert_eq!(probs[2], 140);
    }

    // -------------------------------------------------------------------------
    // adapt_coef_probs — synthetic FrameContext + counts.
    // -------------------------------------------------------------------------

    #[test]
    fn adapt_coef_probs_zero_counts_returns_input() {
        let pre = [[[[[[100u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
            PLANE_TYPES]; TX_SIZES];
        let counts = CoefCountsBank::default();
        let out = adapt_coef_probs(&pre, &counts, COEF_COUNT_SAT, COEF_MAX_UPDATE_FACTOR);
        // With zero counts, merge_probs(pre, [0,0], 24, 112) =
        //   prob = get_binary_prob(0,0) = 128
        //   count = 0, factor = 0
        //   weighted(pre, 128, 0) = pre.
        // So output equals input (for every valid ctx in every band;
        // out-of-range ctxs are also untouched since we only touched
        // `valid_ctx` slots).
        assert_eq!(out, pre);
    }

    /// Inject a single (tx=0, plane=0, ref=0, band=1, ctx=0) entry
    /// with neob=5, eob_total=10, n0=3, n1=1, n2=1. Verify the three
    /// node probabilities hand-computed against libvpx semantics.
    #[test]
    fn adapt_coef_probs_single_slot_hand_trace() {
        let mut pre = [[[[[[100u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
            PLANE_TYPES]; TX_SIZES];
        // Make the target slot distinguishable.
        pre[0][0][0][1][0] = [50, 100, 150];

        let mut counts = CoefCountsBank::default();
        counts.coef[0][0][0][1][0][ZERO_TOKEN] = 3;
        counts.coef[0][0][0][1][0][ONE_TOKEN] = 1;
        counts.coef[0][0][0][1][0][TWO_TOKEN] = 1;
        counts.coef[0][0][0][1][0][EOB_MODEL_TOKEN] = 5;
        counts.eob_branch[0][0][0][1][0] = 10;

        let out = adapt_coef_probs(&pre, &counts, 24, 112);

        // Hand-trace each node:
        // Node 0 (EOB): ct=[5, 10-5=5]. den=10. min(10,24)=10. factor=112*10/24=46.
        //   prob=get_binary_prob(5,5) = get_prob(5,10) = (5*256+5)/10 = 1285/10 = 128.
        //   weighted(50, 128, 46) = (50*(256-46) + 128*46 + 128) >> 8
        //                         = (50*210 + 128*46 + 128) >> 8
        //                         = (10500 + 5888 + 128) >> 8
        //                         = 16516 >> 8 = 64.5 -> 64
        assert_eq!(out[0][0][0][1][0][0], 64);

        // Node 1 (ZERO): ct=[3, 1+1=2]. den=5. min(5,24)=5. factor=112*5/24=23.
        //   prob=get_binary_prob(3,2) = get_prob(3,5) = (3*256+2)/5 = 770/5 = 154.
        //   weighted(100, 154, 23) = (100*233 + 154*23 + 128) >> 8
        //                          = (23300 + 3542 + 128) >> 8
        //                          = 26970 >> 8 = 105.4 -> 105
        assert_eq!(out[0][0][0][1][0][1], 105);

        // Node 2 (ONE): ct=[1, 1]. den=2. min(2,24)=2. factor=112*2/24=9.
        //   prob=get_binary_prob(1,1)=get_prob(1,2)=(1*256+1)/2=128.
        //   weighted(150, 128, 9) = (150*247 + 128*9 + 128) >> 8
        //                         = (37050 + 1152 + 128) >> 8
        //                         = 38330 >> 8 = 149.7 -> 149
        assert_eq!(out[0][0][0][1][0][2], 149);

        // Untouched slots should remain at 100 (the pre-fill).
        assert_eq!(out[0][0][0][0][0], [100, 100, 100]);
        assert_eq!(out[3][1][1][5][5], [100, 100, 100]);
    }

    /// All-zero counts but `count_sat=COEF_COUNT_SAT_KEY=24,
    /// update_factor=COEF_MAX_UPDATE_FACTOR_KEY=112` (intra-only
    /// frame). Should still be a no-op.
    #[test]
    fn adapt_coef_probs_intra_only_zero_counts() {
        let pre = [[[[[[200u8; UNCONSTRAINED_NODES]; COEFF_CONTEXTS]; COEF_BANDS]; REF_TYPES];
            PLANE_TYPES]; TX_SIZES];
        let counts = CoefCountsBank::default();
        let (cs, uf) = coef_adapt_params(true, false);
        assert_eq!((cs, uf), (24, 112));
        let out = adapt_coef_probs(&pre, &counts, cs, uf);
        assert_eq!(out, pre);
    }

    #[test]
    fn coef_adapt_params_matches_libvpx() {
        // intra_only -> KEY constants.
        assert_eq!(coef_adapt_params(true, false), (24, 112));
        assert_eq!(coef_adapt_params(true, true), (24, 112));
        // first inter after key -> AFTER_KEY constants.
        assert_eq!(coef_adapt_params(false, true), (24, 128));
        // steady-state inter -> default constants.
        assert_eq!(coef_adapt_params(false, false), (24, 112));
    }

    // -------------------------------------------------------------------------
    // adapt_mode_probs — synthetic full FrameContext / FrameCounts.
    // -------------------------------------------------------------------------

    #[test]
    fn adapt_mode_probs_zero_counts_unchanged() {
        let pre = FrameContext::new_default();
        let counts = FrameCounts::default();
        // interp_switchable=false, tx_select=false -> only the
        // mode_mv_merge_probs slots get touched; all of those are
        // no-op on zero counts; tree_merge_probs slots are also no-op
        // (mode_mv_merge_probs short-circuits at den=0 at every
        // internal node).
        let out = adapt_mode_probs(&pre, &counts, false, false);
        // Verify the mode-related fields are unchanged. (coef_probs
        // and nmvc are copied through, so they trivially match.)
        assert_eq!(out.intra_inter_prob, pre.intra_inter_prob);
        assert_eq!(out.comp_inter_prob, pre.comp_inter_prob);
        assert_eq!(out.comp_ref_prob, pre.comp_ref_prob);
        assert_eq!(out.single_ref_prob, pre.single_ref_prob);
        assert_eq!(out.inter_mode_probs, pre.inter_mode_probs);
        assert_eq!(out.y_mode_prob, pre.y_mode_prob);
        assert_eq!(out.uv_mode_prob, pre.uv_mode_prob);
        assert_eq!(out.partition_prob, pre.partition_prob);
        assert_eq!(out.switchable_interp_prob, pre.switchable_interp_prob);
        assert_eq!(out.skip_probs, pre.skip_probs);
        assert_eq!(out.tx_probs, pre.tx_probs);
        assert_eq!(out, pre);
    }

    /// Adapt skip_probs[0] with counts.skip[0] = [3, 7]. Hand-trace:
    /// pre.skip_probs[0] = 192 (libvpx default).
    /// mode_mv_merge_probs(192, [3, 7]):
    ///   den=10, count=10, factor=COUNT_TO_UPDATE_FACTOR[10]=64.
    ///   prob=get_prob(3,10) = (3*256+5)/10 = 773/10 = 77.
    ///   weighted(192, 77, 64) = (192*(256-64) + 77*64 + 128) >> 8
    ///                         = (192*192 + 77*64 + 128) >> 8
    ///                         = (36864 + 4928 + 128) >> 8
    ///                         = 41920 >> 8 = 163.75 -> 163
    #[test]
    fn adapt_mode_probs_skip_hand_trace() {
        let pre = FrameContext::new_default();
        let mut counts = FrameCounts::default();
        counts.skip[0] = [3, 7];
        let out = adapt_mode_probs(&pre, &counts, false, false);
        assert_eq!(out.skip_probs[0], 163);
        // Other skip ctxs untouched.
        assert_eq!(out.skip_probs[1], pre.skip_probs[1]);
        assert_eq!(out.skip_probs[2], pre.skip_probs[2]);
    }

    /// Adapt intra_inter_prob[2] with counts.intra_inter[2] = [4, 6].
    /// pre = 187 (libvpx DEFAULT_INTRA_INTER_PROB[2]).
    /// mode_mv_merge_probs(187, [4, 6]):
    ///   den=10, factor=COUNT_TO_UPDATE_FACTOR[10]=64.
    ///   prob=get_prob(4,10) = (4*256+5)/10 = 1029/10 = 102.
    ///   weighted(187, 102, 64) = (187*192 + 102*64 + 128) >> 8
    ///                          = (35904 + 6528 + 128) >> 8
    ///                          = 42560 >> 8 = 166.25 -> 166
    #[test]
    fn adapt_mode_probs_intra_inter_hand_trace() {
        let pre = FrameContext::new_default();
        let mut counts = FrameCounts::default();
        counts.intra_inter[2] = [4, 6];
        let out = adapt_mode_probs(&pre, &counts, false, false);
        assert_eq!(out.intra_inter_prob[2], 166);
    }

    #[test]
    fn adapt_mode_probs_passes_through_coef_and_mv() {
        let pre = FrameContext::new_default();
        let mut counts = FrameCounts::default();
        counts.skip[0] = [3, 7];
        let out = adapt_mode_probs(&pre, &counts, true, true);
        // coef_probs and nmvc untouched.
        assert_eq!(out.coef_probs, pre.coef_probs);
        assert_eq!(out.nmvc, pre.nmvc);
    }

    // -------------------------------------------------------------------------
    // adapt_mv_probs — verify joint + component path.
    // -------------------------------------------------------------------------

    #[test]
    fn adapt_mv_probs_zero_counts_unchanged() {
        let pre = DEFAULT_NMV_CONTEXT;
        let counts = NmvCounts::default();
        let out = adapt_mv_probs(&pre, &counts, true);
        assert_eq!(out, pre);
        // Also with allow_hp=false.
        let out2 = adapt_mv_probs(&pre, &counts, false);
        assert_eq!(out2, pre);
    }

    /// Hand-trace MV joint adapt with all counts on MV_JOINT_ZERO
    /// (`counts.joints[0] = 8`).
    ///
    /// ```text
    /// pre.joints = [32, 64, 96].
    /// VP9_MV_JOINT_TREE = [-0, 2, -1, 4, -2, -3].
    /// Internal node at i=2: left=counts[1]=0, right=subtree(i=4).
    ///   Subtree i=4: left=counts[2]=0, right=counts[3]=0 -> ct=[0,0]
    ///     -> probs[2] = mode_mv_merge(96, [0,0]) = 96 (unchanged).
    ///     Returns 0+0=0.
    ///   Back at i=2: ct=[0, 0] -> probs[1] = mode_mv_merge(64, [0,0]) = 64.
    ///   Returns 0.
    /// Root i=0: left=counts[0]=8, right=0. ct=[8, 0]. den=8.
    ///   factor=COUNT_TO_UPDATE_FACTOR[8]=51. prob=get_prob(8,8) =
    ///   (8*256+4)/8 = 256.5 -> 256, clamped to 255.
    ///   weighted(32, 255, 51) = (32*(256-51) + 255*51 + 128) >> 8
    ///                         = (32*205 + 255*51 + 128) >> 8
    ///                         = (6560 + 13005 + 128) >> 8
    ///                         = 19693 >> 8 = 76.9 -> 76
    /// ```
    #[test]
    fn adapt_mv_probs_joints_skewed_to_zero() {
        let pre = DEFAULT_NMV_CONTEXT;
        let mut counts = NmvCounts::default();
        counts.joints[0] = 8;
        let out = adapt_mv_probs(&pre, &counts, true);
        assert_eq!(out.joints[0], 76);
        assert_eq!(out.joints[1], 64);
        assert_eq!(out.joints[2], 96);
    }

    /// Allow-HP flag: with allow_hp=false, class0_hp and hp slots
    /// should NOT be folded even if counts are present.
    #[test]
    fn adapt_mv_probs_allow_hp_false_skips_hp() {
        let pre = DEFAULT_NMV_CONTEXT;
        let mut counts = NmvCounts::default();
        // Stage counts that would otherwise change hp/class0_hp.
        counts.comps[0].class0_hp = [10, 0];
        counts.comps[0].hp = [10, 0];
        let out = adapt_mv_probs(&pre, &counts, false);
        assert_eq!(out.comps[0].class0_hp, pre.comps[0].class0_hp);
        assert_eq!(out.comps[0].hp, pre.comps[0].hp);
        // Now with allow_hp=true the same counts DO mutate.
        let out2 = adapt_mv_probs(&pre, &counts, true);
        assert_ne!(out2.comps[0].class0_hp, pre.comps[0].class0_hp);
    }

    /// Sign bit fold: simple binary mode_mv_merge_probs path.
    /// pre.sign[0] = 128 (default vertical). counts = [3, 5].
    /// den=8, factor=COUNT_TO_UPDATE_FACTOR[8]=51. prob=get_prob(3,8)
    ///   = (3*256+4)/8 = 772/8 = 96.
    /// weighted(128, 96, 51) = (128*205 + 96*51 + 128) >> 8
    ///                       = (26240 + 4896 + 128) >> 8
    ///                       = 31264 >> 8 = 122.1 -> 122
    #[test]
    fn adapt_mv_probs_sign_hand_trace() {
        let pre = DEFAULT_NMV_CONTEXT;
        let mut counts = NmvCounts::default();
        counts.comps[0].sign = [3, 5];
        let out = adapt_mv_probs(&pre, &counts, true);
        assert_eq!(out.comps[0].sign, 122);
    }

    // -------------------------------------------------------------------------
    // Sanity: every produced prob lies in [1, 255] (libvpx invariant).
    // -------------------------------------------------------------------------

    #[test]
    fn merge_probs_output_in_valid_range() {
        // Exhaustive small-grid sanity (not a full property test).
        for pre in [1u8, 50, 128, 200, 255] {
            for c0 in 0..5u32 {
                for c1 in 0..5u32 {
                    let p = merge_probs(pre, [c0, c1], 24, 112);
                    assert!(
                        p > 0 || pre == 0,
                        "produced 0 from pre={pre} ct=[{c0},{c1}]"
                    );
                    let p2 = mode_mv_merge_probs(pre, [c0, c1]);
                    // mode_mv_merge_probs only returns pre when den==0;
                    // get_prob clamps to [1,255]; weighted_prob preserves
                    // that bound when inputs are already in [1,255].
                    if c0 + c1 > 0 {
                        assert!(p2 > 0, "mode_mv produced 0 from pre={pre} ct=[{c0},{c1}]");
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // G5.1 — additive merge methods (per-tile-worker count accumulation).
    //
    // The invariant: `merge` is associative and identity-on-default.
    // For three independently-built counts banks A, B, C we should have
    // `(A + B) + C == A + (B + C) == sum(A, B, C)`. We verify with a
    // hand-constructed synthetic FrameCounts.
    // -------------------------------------------------------------------------

    fn populate_synthetic(fc: &mut FrameCounts, seed: u32) {
        // Sprinkle non-zero values across every field path. Numbers
        // chosen so each `add` produces a unique result.
        fc.y_mode[0][0] = seed;
        fc.y_mode[1][3] = seed * 2;
        fc.uv_mode[2][4] = seed * 3;
        fc.partition[5][1] = seed * 4;
        fc.coef.coef[0][0][0][2][1][ZERO_TOKEN] = seed * 5;
        fc.coef.eob_branch[0][0][0][2][1] = seed * 6;
        fc.switchable_interp[0][0] = seed * 7;
        fc.inter_mode[2][1] = seed * 8;
        fc.intra_inter[1][0] = seed * 9;
        fc.comp_inter[0][1] = seed * 10;
        fc.single_ref[1][0][1] = seed * 11;
        fc.comp_ref[2][0] = seed * 12;
        fc.tx.p32x32[0][3] = seed * 13;
        fc.tx.p16x16[1][2] = seed * 14;
        fc.tx.p8x8[0][1] = seed * 15;
        fc.tx.tx_totals[2] = seed * 16;
        fc.skip[1][1] = seed * 17;
        fc.mv.joints[2] = seed * 18;
        fc.mv.comps[0].classes[5] = seed * 19;
        fc.mv.comps[1].bits[3][1] = seed * 20;
    }

    #[test]
    fn frame_counts_add_identity_on_default() {
        let mut a = FrameCounts::default();
        populate_synthetic(&mut a, 7);
        let snapshot = a.clone();
        let b = FrameCounts::default();
        a.add(&b);
        assert_eq!(a, snapshot, "adding default must be the identity");
    }

    #[test]
    fn frame_counts_add_is_associative() {
        let mut a = FrameCounts::default();
        let mut b = FrameCounts::default();
        let mut c = FrameCounts::default();
        populate_synthetic(&mut a, 3);
        populate_synthetic(&mut b, 11);
        populate_synthetic(&mut c, 23);

        // (a + b) + c
        let mut left = a.clone();
        left.add(&b);
        left.add(&c);

        // a + (b + c)
        let mut bc = b.clone();
        bc.add(&c);
        let mut right = a.clone();
        right.add(&bc);

        assert_eq!(left, right, "merge must be associative");
    }

    #[test]
    fn frame_counts_add_bit_exact_against_explicit_sum() {
        let mut a = FrameCounts::default();
        let mut b = FrameCounts::default();
        populate_synthetic(&mut a, 5);
        populate_synthetic(&mut b, 17);

        let mut merged = a.clone();
        merged.add(&b);

        // Spot-check a representative slot per field path.
        assert_eq!(merged.y_mode[0][0], 5 + 17);
        assert_eq!(merged.uv_mode[2][4], 5 * 3 + 17 * 3);
        assert_eq!(merged.partition[5][1], 5 * 4 + 17 * 4);
        assert_eq!(merged.coef.coef[0][0][0][2][1][ZERO_TOKEN], 5 * 5 + 17 * 5);
        assert_eq!(merged.coef.eob_branch[0][0][0][2][1], 5 * 6 + 17 * 6);
        assert_eq!(merged.switchable_interp[0][0], 5 * 7 + 17 * 7);
        assert_eq!(merged.tx.p32x32[0][3], 5 * 13 + 17 * 13);
        assert_eq!(merged.skip[1][1], 5 * 17 + 17 * 17);
        assert_eq!(merged.mv.joints[2], 5 * 18 + 17 * 18);
        assert_eq!(merged.mv.comps[1].bits[3][1], 5 * 20 + 17 * 20);
    }

    #[test]
    fn frame_counts_add_non_coef_leaves_coefficient_banks_unchanged() {
        let mut destination = FrameCounts::default();
        let mut source = FrameCounts::default();
        populate_synthetic(&mut destination, 5);
        populate_synthetic(&mut source, 17);
        let coef_before = destination.coef.clone();

        destination.add_non_coef(&source);

        assert_eq!(destination.coef, coef_before);
        assert_eq!(destination.partition[5][1], 5 * 4 + 17 * 4);
        assert_eq!(destination.mv.joints[2], 5 * 18 + 17 * 18);
    }

    #[test]
    fn coef_counts_bank_add_identity_and_sum() {
        // Smaller scope: just the CoefCountsBank slot.
        let mut a = CoefCountsBank::default();
        let mut b = CoefCountsBank::default();
        a.coef[0][0][0][1][2][ONE_TOKEN] = 7;
        a.eob_branch[2][1][0][3][0] = 9;
        b.coef[0][0][0][1][2][ONE_TOKEN] = 11;
        b.eob_branch[2][1][0][3][0] = 13;

        let zero = CoefCountsBank::default();
        let mut snap = a.clone();
        snap.add(&zero);
        assert_eq!(snap, a);

        let mut merged = a.clone();
        merged.add(&b);
        assert_eq!(merged.coef[0][0][0][1][2][ONE_TOKEN], 7 + 11);
        assert_eq!(merged.eob_branch[2][1][0][3][0], 9 + 13);
    }
}
