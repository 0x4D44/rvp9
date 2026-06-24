//! W7.6 — ML-driven partition pruners.
//!
//! Three standalone pruners ported from libvpx
//! `vp9/encoder/vp9_encodeframe.c:3155-3427`:
//!
//! - [`ml_pruning_partition`] — `vp9_encodeframe.c:3157-3245` (`FEATURES=7`):
//!   linear + NN model that votes "skip split/rect partitions" at
//!   64×64 / 32×32 / 16×16.
//! - [`ml_predict_breakout`] — `vp9_encodeframe.c:3248-3329` (`FEATURES=4`):
//!   linear model that votes "stop the partition search now" at
//!   64×64 / 32×32 / 16×16 / 8×8 based on the current RD cost + source
//!   variance + AC quantiser.
//! - [`ml_prune_rect_partition`] — `vp9_encodeframe.c:3332-3427` (`FEATURES=8`,
//!   `LABELS=4`): NN model that votes whether the rectangular
//!   (`PARTITION_HORZ` / `PARTITION_VERT`) arms should be searched at
//!   16×16 / 32×32 / 64×64.
//!
//! ## Scope (recorded deviations)
//!
//! - **8-bit only.** HBD `vp9_high_get_sby_variance` and `bd-8` shifts
//!   are skipped; callers must provide an 8-bit-derived variance.
//! - **No `MACROBLOCKD` / `VP9_COMMON` / `PC_TREE` plumbing.** The
//!   ports take plain Rust structs of the features each pruner needs.
//!   Production callers are responsible for filling these from the
//!   real encoder state.
//! - **Linear breakout threshold is the caller's input.** libvpx reads
//!   `cpi->sf.rd_ml_partition.search_breakout_thresh[q_ctx]`; we accept
//!   it as a parameter to keep the module standalone.
//! - **No `vpx_clear_system_state` calls.** These are SSE/MMX clean-up
//!   no-ops on x86; Rust scalar code doesn't need them.

use rvp9_core::block::BlockSize;

use super::nn_predict::{
    NnConfig, PARTITION_BREAKOUT_WEIGHTS_8, PARTITION_BREAKOUT_WEIGHTS_16,
    PARTITION_BREAKOUT_WEIGHTS_32, PARTITION_BREAKOUT_WEIGHTS_64, PARTITION_FEATURE_MEAN,
    PARTITION_FEATURE_STD, PARTITION_LINEAR_WEIGHTS, PARTITION_NN_CONFIG_16X16,
    PARTITION_NN_CONFIG_32X32, PARTITION_NN_CONFIG_64X64, RECT_PART_NN_CONFIG_16,
    RECT_PART_NN_CONFIG_32, RECT_PART_NN_CONFIG_64, nn_predict,
};

// ============================================================================
// ml_pruning_partition — vp9_encodeframe.c:3157-3245
// ============================================================================

/// Inputs to [`ml_pruning_partition`]. Mirrors the libvpx feature
/// extraction at `vp9_encodeframe.c:3222-3230`.
#[derive(Debug, Clone, Copy)]
pub struct PruningPartitionInput {
    /// Current `bsize` — must be 64×64, 32×32, or 16×16.
    pub bsize: BlockSize,
    /// `ctx->rate` (rate of the NONE leaf at this block).
    pub ctx_rate: i32,
    /// `ctx->dist` (distortion of the NONE leaf).
    pub ctx_dist: i64,
    /// `abs(ctx->mic.mv[0].as_mv.col) + abs(...row)` — magnitude of
    /// the leaf's MV.
    pub mag_mv: i32,
    /// `ctx->sum_y_eobs` — sum of EOBs for the luma block.
    pub sum_y_eobs: i32,
    /// `cm->base_qindex`.
    pub base_qindex: i32,
    /// Whether the left neighbour is in the image (`!!xd->left_mi`).
    pub left_in_image: bool,
    /// Whether the above neighbour is in the image.
    pub above_in_image: bool,
    /// Whether the previous-frame mi at `(mi_row, mi_col)` exists.
    pub prev_mi_in_image: bool,
    /// `xd->left_mi->sb_type` if `left_in_image`.
    pub left_sb_type: BlockSize,
    /// `xd->above_mi->sb_type` if `above_in_image`.
    pub above_sb_type: BlockSize,
    /// `prev_mi[0]->sb_type` if `prev_mi_in_image`.
    pub prev_mi_sb_type: BlockSize,
}

/// Result of [`ml_pruning_partition`].
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PruningDecision {
    /// Continue the full partition search.
    Continue,
    /// Skip split and rect partitions — only PARTITION_NONE is kept.
    /// libvpx return value `1` from `ml_pruning_partition`.
    Skip,
}

/// libvpx `ml_pruning_partition` (`vp9_encodeframe.c:3157-3245`).
///
/// Returns [`PruningDecision::Skip`] if both the linear and NN models
/// vote to skip split/rect partition search at this block. Only valid
/// for 64×64 / 32×32 / 16×16 — other bsizes return
/// [`PruningDecision::Continue`].
#[must_use]
pub fn ml_pruning_partition(input: &PruningPartitionInput) -> PruningDecision {
    // Map bsize → (offset into linear tables, NN config).
    // libvpx `vp9_encodeframe.c:3182-3196`.
    let (offset, nn_config): (usize, &NnConfig) = match input.bsize {
        BlockSize::Block64x64 => (0, &PARTITION_NN_CONFIG_64X64),
        BlockSize::Block32x32 => (8, &PARTITION_NN_CONFIG_32X32),
        BlockSize::Block16x16 => (16, &PARTITION_NN_CONFIG_16X16),
        _ => return PruningDecision::Continue,
    };

    // Neighbour context. libvpx `vp9_encodeframe.c:3198-3220`:
    // par = 2 if neighbour_size < bsize, 1 if equal, 0 if larger.
    let par_for = |present: bool, ctx: BlockSize| -> i32 {
        if !present {
            return 0;
        }
        // libvpx compares enum-ordered BLOCK_SIZE values.
        if (ctx as u8) < (input.bsize as u8) {
            2
        } else if (ctx as u8) == (input.bsize as u8) {
            1
        } else {
            0
        }
    };

    let above_par = par_for(input.above_in_image, input.above_sb_type);
    let left_par = par_for(input.left_in_image, input.left_sb_type);
    let last_par = par_for(input.prev_mi_in_image, input.prev_mi_sb_type);

    // Feature normalisation. libvpx `vp9_encodeframe.c:3222-3230`.
    // Note: libvpx mixes "(x - mean)/sd" (for features 0,1,4) with
    // "(x - mean)*sd" (for features 2,3,5,6). This is a quirk of the
    // training pipeline — we replicate verbatim.
    let mean = &PARTITION_FEATURE_MEAN[offset..offset + 7];
    let sd = &PARTITION_FEATURE_STD[offset..offset + 7];

    let mut features = [0.0f32; 7];
    features[0] = (input.ctx_rate as f32 - mean[0]) / sd[0];
    features[1] = (input.ctx_dist as f32 - mean[1]) / sd[1];
    features[2] = (input.mag_mv as f32 / 2.0 - mean[2]) * sd[2];
    features[3] = ((left_par + above_par) as f32 / 2.0 - mean[3]) * sd[3];
    features[4] = (input.sum_y_eobs as f32 - mean[4]) / sd[4];
    features[5] = (input.base_qindex as f32 - mean[5]) * sd[5];
    features[6] = (last_par as f32 - mean[6]) * sd[6];

    // Linear model. libvpx `vp9_encodeframe.c:3232-3237`.
    // linear_weights[FEATURES] is the bias; the 7 preceding are weights.
    let linear_weights = &PARTITION_LINEAR_WEIGHTS[offset..offset + 8];
    let mut linear_score = linear_weights[7];
    for i in 0..7 {
        linear_score += linear_weights[i] * features[i];
    }
    if linear_score > 0.1 {
        return PruningDecision::Continue;
    }

    // NN model. libvpx `vp9_encodeframe.c:3240`.
    let mut nn_score = [0.0f32; 1];
    nn_predict(&features, nn_config, &mut nn_score);

    // Joint decision. libvpx `vp9_encodeframe.c:3242-3244`.
    // (`< -0.0f` is just `< 0.0f` for finite floats — copy the test
    // verbatim for parity.)
    if linear_score < -0.0 && nn_score[0] < 0.1 {
        return PruningDecision::Skip;
    }
    if nn_score[0] < -0.0 && linear_score < 0.1 {
        return PruningDecision::Skip;
    }
    PruningDecision::Continue
}

// ============================================================================
// ml_predict_breakout — vp9_encodeframe.c:3248-3329
// ============================================================================

/// Inputs to [`ml_predict_breakout`]. Mirrors libvpx
/// `vp9_encodeframe.c:3280-3318`.
#[derive(Debug, Clone, Copy)]
pub struct BreakoutInput {
    /// Current bsize — must be 64×64, 32×32, 16×16, or 8×8.
    pub bsize: BlockSize,
    /// `cm->base_qindex`.
    pub base_qindex: i32,
    /// `cm->width`, `cm->height`. Used to derive the `is_720p_or_larger`
    /// resolution context (libvpx `vp9_encodeframe.c:3260-3261`).
    pub width: i32,
    /// `cm->height`.
    pub height: i32,
    /// Source-block AC quantiser (`vp9_ac_quant(base_qindex, 0, bit_depth)`).
    /// 8-bit only — caller resolves any HBD shift.
    pub ac_q: i32,
    /// Y-plane variance vs. the all-zero reference block (`vf(src,
    /// zeros, ...)`), shifted right by `num_pels_log2_lookup[bsize]`.
    pub var_shifted: u32,
    /// `rd_cost->rate`, clamped to `INT_MAX`.
    pub rd_rate: i32,
    /// `rd_cost->dist`, clamped to `INT_MAX` then `>> num_pels_log2`.
    pub rd_dist_shifted: i64,
    /// `x->rdmult`.
    pub rdmult: i32,
    /// `cpi->sf.rd_ml_partition.search_breakout_thresh[q_ctx]` — the
    /// threshold at which a positive linear score earns a breakout
    /// vote. Caller plugs this in from speed-feature state.
    pub breakout_thresh: f32,
}

/// libvpx `ml_predict_breakout` (`vp9_encodeframe.c:3250-3329`).
///
/// Returns `true` if the model votes to stop the partition search now.
#[must_use]
pub fn ml_predict_breakout(input: &BreakoutInput) -> bool {
    // libvpx `vp9_encodeframe.c:3258-3261`.
    let q_ctx = if input.base_qindex >= 200 {
        0
    } else if input.base_qindex >= 150 {
        1
    } else {
        2
    };
    let is_720p_or_larger = input.width.min(input.height) >= 720;
    let resolution_ctx = usize::from(is_720p_or_larger);

    // libvpx `vp9_encodeframe.c:3263-3277`.
    let weights: &[f32; 5] = match input.bsize {
        BlockSize::Block64x64 => &PARTITION_BREAKOUT_WEIGHTS_64[resolution_ctx][q_ctx],
        BlockSize::Block32x32 => &PARTITION_BREAKOUT_WEIGHTS_32[resolution_ctx][q_ctx],
        BlockSize::Block16x16 => &PARTITION_BREAKOUT_WEIGHTS_16[resolution_ctx][q_ctx],
        BlockSize::Block8x8 => &PARTITION_BREAKOUT_WEIGHTS_8[resolution_ctx][q_ctx],
        _ => return false,
    };

    // libvpx `vp9_encodeframe.c:3286-3318`.
    let bsize_pels_log2 = (input.bsize.width_log2() + input.bsize.height_log2()) as i32;
    let pels_shift = 1i64 << bsize_pels_log2;

    let rate_f = input.rd_rate as f32;
    let dist_f = input.rd_dist_shifted as f32;
    // libvpx applies rdmult/128/512/(1<<num_pels_log2) — we mirror exactly.
    let rate_norm = (input.rdmult as f32 / 128.0 / 512.0 / (pels_shift as f32)) * rate_f;

    let features = [
        rate_norm,
        dist_f,
        input.var_shifted as f32,
        input.ac_q as f32,
    ];

    // libvpx `vp9_encodeframe.c:3321-3326`.
    let mut linear_score = weights[4]; // bias
    for i in 0..4 {
        linear_score += weights[i] * features[i];
    }

    linear_score >= input.breakout_thresh
}

// ============================================================================
// ml_prune_rect_partition — vp9_encodeframe.c:3332-3427
// ============================================================================

/// Inputs to [`ml_prune_rect_partition`]. Mirrors the libvpx feature
/// extraction at `vp9_encodeframe.c:3380-3401`.
#[derive(Debug, Clone, Copy)]
pub struct RectPruneInput {
    /// Current bsize — 16×16 / 32×32 / 64×64. (8×8 is a no-op in libvpx.)
    pub bsize: BlockSize,
    /// `vp9_dc_quant(cm->base_qindex, 0, bit_depth)`. 8-bit only.
    pub dc_q: i32,
    /// `cpi->sf.rd_ml_partition.prune_rect_thresh[bsize_idx]` — caller
    /// resolves the speed-feature lookup. Negative or unset → caller
    /// returns Continue without invoking the model.
    pub thresh: i32,
    /// Reference RD cost (`pc_tree->none` etc), the divisor used to
    /// normalise sub-block RDs.
    pub ref_rd: i64,
    /// Whether the current `pc_tree->partitioning == PARTITION_NONE`.
    pub partitioning_is_none: bool,
    /// `pc_tree->none.rdcost`.
    pub none_rdcost: i64,
    /// `pc_tree->split[i]->none.rdcost` — the 4 split children's NONE-arm RD.
    pub split_none_rdcost: [i64; 4],
}

/// Output mask from [`ml_prune_rect_partition`]. Each field is `true`
/// when the corresponding rect-partition arm should still be searched.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RectPruneDecision {
    /// `*allow_horz` — search PARTITION_HORZ.
    pub allow_horz: bool,
    /// `*allow_vert` — search PARTITION_VERT.
    pub allow_vert: bool,
}

/// libvpx `ml_prune_rect_partition` (`vp9_encodeframe.c:3334-3425`).
///
/// `allow_horz_in` / `allow_vert_in` are the caller's pre-existing
/// masks; the function ANDs them with the model verdict (libvpx lines
/// 3422-3423: `*allow_horz = *allow_horz && horz`).
#[must_use]
pub fn ml_prune_rect_partition(
    input: &RectPruneInput,
    allow_horz_in: bool,
    allow_vert_in: bool,
) -> RectPruneDecision {
    // libvpx `vp9_encodeframe.c:3347`. Early bail on degenerate ref_rd.
    if input.ref_rd <= 0 || input.ref_rd > 1_000_000_000 {
        return RectPruneDecision {
            allow_horz: allow_horz_in,
            allow_vert: allow_vert_in,
        };
    }

    // libvpx `vp9_encodeframe.c:3349-3364`.
    let nn_config: &NnConfig = match input.bsize {
        BlockSize::Block8x8 => {
            // libvpx case BLOCK_8X8: break; (no config → no pruning).
            return RectPruneDecision {
                allow_horz: allow_horz_in,
                allow_vert: allow_vert_in,
            };
        }
        BlockSize::Block16x16 => &RECT_PART_NN_CONFIG_16,
        BlockSize::Block32x32 => &RECT_PART_NN_CONFIG_32,
        BlockSize::Block64x64 => &RECT_PART_NN_CONFIG_64,
        _ => {
            return RectPruneDecision {
                allow_horz: allow_horz_in,
                allow_vert: allow_vert_in,
            };
        }
    };

    if input.thresh < 0 {
        return RectPruneDecision {
            allow_horz: allow_horz_in,
            allow_vert: allow_vert_in,
        };
    }

    // Feature extraction. libvpx `vp9_encodeframe.c:3376-3404`.
    let bs = 4 * (input.bsize.n_4x4_wide() as i32); // 4 * num_4x4_blocks_wide.
    let mut features = [0.0f32; 8];
    let mut idx = 0usize;
    features[idx] = (input.dc_q as f32 + 1.0).ln();
    idx += 1;
    features[idx] = if input.partitioning_is_none { 1.0 } else { 0.0 };
    idx += 1;
    features[idx] = (input.ref_rd as f32 / (bs as f32) / (bs as f32) + 1.0).ln();
    idx += 1;

    let norm_factor = 1.0f32 / (input.ref_rd as f32 + 1.0);
    let mut rd_ratio = 2.0f32;
    if input.none_rdcost > 0 && input.none_rdcost < 1_000_000_000 {
        rd_ratio = input.none_rdcost as f32 * norm_factor;
    }
    features[idx] = rd_ratio.min(2.0);
    idx += 1;

    for i in 0..4 {
        let this_rd = input.split_none_rdcost[i];
        let rd_valid = this_rd > 0 && this_rd < 1_000_000_000;
        features[idx] = if rd_valid {
            this_rd as f32 * norm_factor
        } else {
            1.0
        };
        idx += 1;
    }
    debug_assert_eq!(idx, 8);

    let mut score = [0.0f32; 4];
    nn_predict(&features, nn_config, &mut score);

    // Decision: bit 0 of label index → horz; bit 1 → vert.
    // libvpx `vp9_encodeframe.c:3408-3423`.
    let mut max_int = i32::MIN;
    let mut int_score = [0i32; 4];
    for i in 0..4 {
        int_score[i] = (100.0 * score[i]) as i32;
        if int_score[i] > max_int {
            max_int = int_score[i];
        }
    }
    let thresh = max_int - input.thresh;
    let mut horz = false;
    let mut vert = false;
    for (i, s) in int_score.iter().enumerate() {
        if *s >= thresh {
            if (i & 1) != 0 {
                horz = true;
            }
            if ((i >> 1) & 1) != 0 {
                vert = true;
            }
        }
    }

    RectPruneDecision {
        allow_horz: allow_horz_in && horz,
        allow_vert: allow_vert_in && vert,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// W7.6 — `ml_predict_breakout` returns a deterministic decision
    /// for a sane breakout input.
    ///
    /// libvpx reference: `vp9_encodeframe.c:3250-3329`.
    #[test]
    fn w7_6_ml_predict_breakout_returns_skip_decision() {
        // High base_qindex + small variance → linear_score is dominated
        // by the negative bias term; breakout should not trigger when
        // we set the threshold above the achievable score.
        let high_thresh = BreakoutInput {
            bsize: BlockSize::Block32x32,
            base_qindex: 210,
            width: 1280,
            height: 720,
            ac_q: 100,
            var_shifted: 8,
            rd_rate: 1000,
            rd_dist_shifted: 200,
            rdmult: 100,
            breakout_thresh: 100.0,
        };
        assert!(!ml_predict_breakout(&high_thresh));

        // Mirror call with the threshold set to a very low value:
        // the same score should now pass it.
        let low_thresh = BreakoutInput {
            breakout_thresh: -1000.0,
            ..high_thresh
        };
        assert!(ml_predict_breakout(&low_thresh));

        // Out-of-range bsize returns false (libvpx `default` arm).
        let bad_bsize = BreakoutInput {
            bsize: BlockSize::Block4x4,
            ..high_thresh
        };
        assert!(!ml_predict_breakout(&bad_bsize));
    }

    /// W7.6 — `ml_pruning_partition` and `ml_prune_rect_partition`
    /// both return a sane subset of the partition arms.
    ///
    /// libvpx reference: `vp9_encodeframe.c:3157-3245` and
    /// `vp9_encodeframe.c:3334-3425`.
    #[test]
    fn w7_6_ml_pruning_partition_picks_subset() {
        // Pruning partition: feed a neutral feature set at 32×32 and
        // check the decision is one of the two enum variants (not a
        // panic / NaN).
        let pruning_input = PruningPartitionInput {
            bsize: BlockSize::Block32x32,
            ctx_rate: 100,
            ctx_dist: 5_000,
            mag_mv: 4,
            sum_y_eobs: 32,
            base_qindex: 100,
            left_in_image: true,
            above_in_image: true,
            prev_mi_in_image: false,
            left_sb_type: BlockSize::Block32x32,
            above_sb_type: BlockSize::Block16x16,
            prev_mi_sb_type: BlockSize::Block32x32,
        };
        let decision = ml_pruning_partition(&pruning_input);
        // Always one of two values — the test is that the call doesn't
        // panic and that an unrelated bsize returns Continue.
        let _ = matches!(decision, PruningDecision::Skip | PruningDecision::Continue);

        let unsupported = PruningPartitionInput {
            bsize: BlockSize::Block8x8,
            ..pruning_input
        };
        assert_eq!(
            ml_pruning_partition(&unsupported),
            PruningDecision::Continue
        );

        // Rect-prune: feed a sane RD set, both arms allowed by caller.
        let rect_input = RectPruneInput {
            bsize: BlockSize::Block32x32,
            dc_q: 50,
            thresh: 50,
            ref_rd: 1_000_000,
            partitioning_is_none: true,
            none_rdcost: 500_000,
            split_none_rdcost: [250_000, 260_000, 240_000, 255_000],
        };
        // Caller already disallowed horz → must stay disallowed.
        let rect_h_off = ml_prune_rect_partition(&rect_input, false, true);
        assert!(!rect_h_off.allow_horz);
        // Caller already disallowed vert → must stay disallowed.
        let rect_v_off = ml_prune_rect_partition(&rect_input, true, false);
        assert!(!rect_v_off.allow_vert);

        // Degenerate ref_rd → pass-through (no pruning).
        let degenerate = RectPruneInput {
            ref_rd: 0,
            ..rect_input
        };
        let pass = ml_prune_rect_partition(&degenerate, true, true);
        assert_eq!(
            pass,
            RectPruneDecision {
                allow_horz: true,
                allow_vert: true,
            }
        );
    }
}
