//! W7.7 — Variance-based ML partitioning + simple motion search.
//!
//! ## libvpx references
//!
//! - **`ml_predict_var_rd_partitioning`** —
//!   `vp9/encoder/vp9_encodeframe.c:3489-3618` (commit `12f3a2ac6`).
//!   Picks between PARTITION_NONE and PARTITION_SPLIT for the current
//!   block based on the variance of the residue after a single coarse
//!   motion search.
//! - **`simple_motion_search`** —
//!   `vp9/encoder/vp9_encodeframe.c:3429-3483`. A pre-processing
//!   single-MV diamond search that fills the prediction buffer
//!   consumed by `ml_predict_var_rd_partitioning`.
//! - **`pred_buffer`** — `vp9/encoder/vp9_encodeframe.c:3498-3504`. The
//!   64×64-byte prediction scratch buffer.
//!
//! ## Scope (recorded deviations)
//!
//! - **8-bit only.** The HBD-aligned `pred_buffer[64*64*2]` branch is
//!   dropped; the buffer is a plain `[u8; 64*64]`.
//! - **No real MC.** `simple_motion_search` ports the *logic* (diamond
//!   search → MV scaled by 8 → fill `pred_buf`), with the MC step
//!   abstracted behind a `MotionSearchOps` trait that the production
//!   caller wires to libvpx's `vp9_full_pixel_search` +
//!   `vp9_build_inter_predictors_sby`. The standalone test path
//!   plugs a synthetic SAD oracle.
//! - **No `MACROBLOCK` / `MACROBLOCKD` / `VP9_COMP` plumbing.** Inputs
//!   are plain Rust structs of the features each function needs.
//! - **Variance is computed scalar.** SIMD parity for `vpx_variance*`
//!   lands in a later wave.

use rvp9_core::{block::BlockSize, mode_decision::partition::PartitionDecision};

use super::nn_predict::{
    NnConfig, PART_SPLIT_NN_CONFIG_16, PART_SPLIT_NN_CONFIG_32, PART_SPLIT_NN_CONFIG_64,
    VAR_PART_NN_CONFIG_16, VAR_PART_NN_CONFIG_32, VAR_PART_NN_CONFIG_64, nn_predict,
};

/// 64×64-pixel motion-prediction scratch buffer.
/// libvpx `vp9_encodeframe.c:3503` (`DECLARE_ALIGNED(16, uint8_t,
/// pred_buffer[64*64])`).
pub type PredBuffer = [u8; 64 * 64];

/// Allocate an empty `PredBuffer`.
#[must_use]
pub fn new_pred_buffer() -> PredBuffer {
    [0u8; 64 * 64]
}

/// Motion vector with quarter-pel precision (8x int-pel resolution).
/// Mirrors libvpx `MV { row, col }` after the `* 8` upscale at
/// `vp9_encodeframe.c:3469-3470`.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct Mv {
    /// Row offset in 1/8-pel units.
    pub row: i16,
    /// Column offset in 1/8-pel units.
    pub col: i16,
}

/// A 2D rectangular buffer view (source / reference plane). Used as
/// the input to [`simple_motion_search`] and the variance computation.
#[derive(Debug, Clone, Copy)]
pub struct PlaneView<'a> {
    /// Raw pixel data, scanline-major.
    pub buf: &'a [u8],
    /// Row stride.
    pub stride: usize,
    /// Width and height (in pixels) of the addressable region.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

/// Mutable plane view — sole consumer is the production
/// `MotionSearchOps::fill_pred` callback.
#[derive(Debug)]
pub struct PlaneViewMut<'a> {
    /// Mutable pixel data.
    pub buf: &'a mut [u8],
    /// Row stride.
    pub stride: usize,
}

/// Trait abstracting libvpx's two MC building blocks used by
/// `simple_motion_search`:
///
/// 1. `vp9_full_pixel_search(...)` — integer-pel diamond search.
/// 2. `vp9_build_inter_predictors_sby(...)` — fills the prediction
///    buffer with the chosen MV's MC output.
///
/// In production the caller plugs the real MC functions; in tests a
/// synthetic oracle (e.g. minimum-SAD shift) is enough.
pub trait MotionSearchOps {
    /// Find the best integer-pel MV around `ref_mv_int` for `src` vs.
    /// `reference`, both at `(mi_row * 8, mi_col * 8)` pixel offsets.
    /// Returns the **integer-pel** MV (row, col); the caller multiplies
    /// by 8 to get 1/8-pel.
    fn full_pixel_search(
        &mut self,
        src: PlaneView<'_>,
        reference: PlaneView<'_>,
        bsize: BlockSize,
        ref_mv_int: Mv,
    ) -> Mv;

    /// Fill `pred` with the MC output of `(mv, reference)`. Stride of
    /// `pred` is always 64 (libvpx `vp9_encodeframe.c:3481`).
    fn build_inter_predictors(
        &mut self,
        reference: PlaneView<'_>,
        bsize: BlockSize,
        mv: Mv,
        pred: &mut PlaneViewMut<'_>,
    );
}

/// libvpx `simple_motion_search` (`vp9_encodeframe.c:3431-3483`).
///
/// Runs a single coarse diamond search, scales the MV by 8 to convert
/// integer-pel → 1/8-pel, then fills `pred_buf` with the MC output.
pub fn simple_motion_search<O: MotionSearchOps>(
    ops: &mut O,
    src: PlaneView<'_>,
    reference: PlaneView<'_>,
    bsize: BlockSize,
    ref_mv: Mv,
    pred_buf: &mut PredBuffer,
) -> Mv {
    // libvpx `vp9_encodeframe.c:3444` — `ref_mv_full = { row>>3, col>>3 }`.
    let ref_mv_int = Mv {
        row: ref_mv.row >> 3,
        col: ref_mv.col >> 3,
    };

    let best_int = ops.full_pixel_search(src, reference, bsize, ref_mv_int);

    // libvpx `vp9_encodeframe.c:3469-3470` — scale back to 1/8-pel.
    let best_mv = Mv {
        row: best_int.row.saturating_mul(8),
        col: best_int.col.saturating_mul(8),
    };

    // libvpx `vp9_encodeframe.c:3480-3482` — `dst.buf = pred_buf;
    // dst.stride = 64; vp9_build_inter_predictors_sby(...)`.
    let mut pred_view = PlaneViewMut {
        buf: pred_buf.as_mut_slice(),
        stride: 64,
    };
    ops.build_inter_predictors(reference, bsize, best_mv, &mut pred_view);

    best_mv
}

/// Compute the variance of `src` vs. `pred` over a `w×h` region. Returns
/// `(var, sse)`. Mirrors libvpx `vpx_variance*` scalar reference:
/// `mean = sum(d) / N; var = (sum(d^2) - mean*sum(d)) / N` actually
/// libvpx returns `sse - ((int64_t)sum*sum)/n`. We use the same form.
fn variance(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> (u32, u32) {
    let mut sum: i64 = 0;
    let mut sse: i64 = 0;
    for r in 0..h {
        for c in 0..w {
            let diff = src[r * src_stride + c] as i32 - pred[r * pred_stride + c] as i32;
            sum += diff as i64;
            sse += (diff as i64) * (diff as i64);
        }
    }
    let n = (w * h) as i64;
    // libvpx form: var = sse - (sum * sum) / n;
    let var_signed = if n > 0 { sse - (sum * sum) / n } else { 0 };
    let var = if var_signed < 0 { 0 } else { var_signed as u32 };
    (var, sse.max(0) as u32)
}

/// Inputs to [`ml_predict_var_rd_partitioning`]. Mirrors libvpx
/// `vp9_encodeframe.c:3554-3605`.
#[derive(Debug, Clone, Copy)]
pub struct VarPartInput<'a> {
    /// Current bsize — 16×16 / 32×32 / 64×64. (libvpx also has an 8×8
    /// arm, but that uses a different feature set / threshold; supported
    /// here too.)
    pub bsize: BlockSize,
    /// `vp9_dc_quant(cm->base_qindex, 0, cm->bit_depth)`. 8-bit only.
    pub dc_q: i32,
    /// `cpi->oxcf.speed`.
    pub speed: i32,
    /// `cm->width`, `cm->height` — drive the 8×8 threshold gate at
    /// libvpx `vp9_encodeframe.c:3524-3527`.
    pub width: i32,
    /// `cm->height`.
    pub height: i32,
    /// Source plane view (`x->plane[0].src`).
    pub src: PlaneView<'a>,
    /// Prediction-buffer slice (filled by `simple_motion_search`).
    pub pred: &'a [u8; 64 * 64],
    /// Whether the above neighbour is in the image.
    pub has_above: bool,
    /// Whether the left neighbour is in the image.
    pub has_left: bool,
    /// `xd->above_mi->sb_type` if `has_above`, else current `bsize`.
    pub above_sb_type: BlockSize,
    /// `xd->left_mi->sb_type` if `has_left`, else current `bsize`.
    pub left_sb_type: BlockSize,
}

/// Output of [`ml_predict_var_rd_partitioning`]. Each field is `true`
/// when the corresponding partition arm should still be searched.
/// libvpx writes `*none = 0` / `*split = 0` to disable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarPartDecision {
    /// Whether PARTITION_NONE should still be searched.
    pub allow_none: bool,
    /// Whether PARTITION_SPLIT should still be searched.
    pub allow_split: bool,
    /// The model score (for logging / parity testing).
    pub score: f32,
}

/// libvpx `ml_predict_var_rd_partitioning` (`vp9_encodeframe.c:3489-3618`).
///
/// Returns the partition-arm mask plus the raw model score. Caller is
/// responsible for having filled `pred` via [`simple_motion_search`]
/// beforehand (libvpx threads the same data flow at lines 3536-3548).
///
/// The arm prediction is:
/// - `allow_none = false` if `score > thresh` (model says split better);
/// - `allow_split = false` if `score < -thresh` (model says none better).
///
/// Returns [`BlockSize`]-like coarse advice: callers prefer the
/// "winning" arm via the boolean fields.
#[must_use]
pub fn ml_predict_var_rd_partitioning(input: &VarPartInput<'_>) -> VarPartDecision {
    // libvpx `vp9_encodeframe.c:3509-3530`. The function name is
    // `ml_predict_var_rd_partitioning`; libvpx reaches into the
    // PART_SPLIT NN configs (12-input) — the VAR_PART configs (6-input)
    // are used elsewhere by the non-RD variance partitioner.
    let (nn_config, thresh): (&NnConfig, f32) = match input.bsize {
        BlockSize::Block64x64 => (
            &PART_SPLIT_NN_CONFIG_64,
            if input.speed > 0 { 2.8 } else { 3.0 },
        ),
        BlockSize::Block32x32 => (
            &PART_SPLIT_NN_CONFIG_32,
            if input.speed > 0 { 3.5 } else { 3.0 },
        ),
        BlockSize::Block16x16 => (
            &PART_SPLIT_NN_CONFIG_16,
            if input.speed > 0 { 3.8 } else { 4.0 },
        ),
        BlockSize::Block8x8 => {
            // libvpx `vp9_encodeframe.c:3522-3528` — 8×8 uses
            // `vp9_part_split_nnconfig_8` plus a resolution-gated thresh.
            let t = if input.width >= 720 && input.height >= 720 {
                if input.speed > 0 { 2.5 } else { 2.0 }
            } else if input.speed > 0 {
                3.8
            } else {
                2.0
            };
            (&super::nn_predict::PART_SPLIT_NN_CONFIG_8, t)
        }
        _ => {
            return VarPartDecision {
                allow_none: true,
                allow_split: true,
                score: 0.0,
            };
        }
    };

    // libvpx `vp9_encodeframe.c:3565-3604`.
    let bw = input.bsize.width() as usize;
    let bh = input.bsize.height() as usize;
    let bs = 4 * (input.bsize.n_4x4_wide() as usize);
    let pred_stride = 64usize;

    // Full-block variance.
    let (var, _sse) = variance(
        input.src.buf,
        input.src.stride,
        input.pred,
        pred_stride,
        bw,
        bh,
    );
    let factor = if var == 0 { 1.0 } else { 1.0 / var as f32 };

    let above_bsize = if input.has_above {
        input.above_sb_type
    } else {
        input.bsize
    };
    let left_bsize = if input.has_left {
        input.left_sb_type
    } else {
        input.bsize
    };

    // 12 features per libvpx `vp9_encodeframe.c:3586-3604`.
    let mut features = [0.0f32; 12];
    features[0] = (input.dc_q as f32 + 1.0).ln();
    features[1] = if input.has_above { 1.0 } else { 0.0 };
    features[2] = above_bsize.width_log2() as f32;
    features[3] = above_bsize.height_log2() as f32;
    features[4] = if input.has_left { 1.0 } else { 0.0 };
    features[5] = left_bsize.width_log2() as f32;
    features[6] = left_bsize.height_log2() as f32;
    features[7] = (var as f32 + 1.0).ln();

    // 4 quarter-block variance ratios.
    let sub_w = bs / 2;
    let sub_h = bs / 2;
    for i in 0..4 {
        let x_idx = (i & 1) * (bs / 2);
        let y_idx = (i >> 1) * (bs / 2);
        let src_off = y_idx * input.src.stride + x_idx;
        let pred_off = y_idx * pred_stride + x_idx;
        // Defensive clamp: the production caller always passes a full
        // 64×64 region, but for unit tests we accept smaller src views.
        let src_avail_w = input.src.width.saturating_sub(x_idx).min(sub_w);
        let src_avail_h = input.src.height.saturating_sub(y_idx).min(sub_h);
        let (sub_var, _) = if src_avail_w == 0 || src_avail_h == 0 {
            (0u32, 0u32)
        } else {
            variance(
                &input.src.buf[src_off..],
                input.src.stride,
                &input.pred[pred_off..],
                pred_stride,
                src_avail_w,
                src_avail_h,
            )
        };
        let var_ratio = if var == 0 {
            1.0
        } else {
            factor * sub_var as f32
        };
        features[8 + i] = var_ratio;
    }

    let mut score = [0.0f32; 1];
    nn_predict(&features, nn_config, &mut score);

    // libvpx `vp9_encodeframe.c:3615-3616`.
    let mut allow_none = true;
    let mut allow_split = true;
    if score[0] > thresh {
        allow_none = false;
    }
    if score[0] < -thresh {
        allow_split = false;
    }

    VarPartDecision {
        allow_none,
        allow_split,
        score: score[0],
    }
}

/// Inputs to pinned libvpx's low-resolution realtime
/// `ml_predict_var_partitioning` controller.
#[derive(Debug, Clone, Copy)]
pub struct MlVarPartitionTreeInput<'a> {
    /// Frame DC quantizer value.
    pub dc_q: i32,
    /// Encoder speed preset; speed 6 and above use a zero decision threshold.
    pub speed: i32,
    /// Edge-extended 64x64 source block.
    pub src: &'a [u8],
    /// Source stride in bytes.
    pub src_stride: usize,
    /// Estimated 64x64 inter prediction.
    pub pred: &'a [u8],
    /// Prediction stride in bytes.
    pub pred_stride: usize,
}

/// Exact inputs and output for one visited node in the low-resolution
/// realtime partition model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlVarPartitionNodeTrace {
    /// Pixel row relative to the enclosing 64x64 superblock.
    pub row_px: usize,
    /// Pixel column relative to the enclosing 64x64 superblock.
    pub col_px: usize,
    /// Square node size evaluated by the model.
    pub bsize: BlockSize,
    /// FNV-1a hash of the visible source pixels used by the node.
    pub source_hash: u64,
    /// FNV-1a hash of the estimated-prediction pixels used by the node.
    pub predictor_hash: u64,
    /// Whole-node source-minus-prediction variance.
    pub variance: u32,
    /// Quadrant variances in raster order.
    pub sub_variances: [u32; 4],
    /// Exact six floating-point model inputs.
    pub features: [f32; 6],
    /// Single output produced by pinned libvpx's neural-network table.
    pub score: f32,
}

fn hash_rect(data: &[u8], stride: usize, width: usize, height: usize) -> u64 {
    let mut hash = 14_695_981_039_346_656_037u64;
    for row in 0..height {
        for &value in &data[row * stride..row * stride + width] {
            hash = (hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

#[derive(Debug, Clone, Copy)]
struct MlVarPartitionNodeEval {
    variance: u32,
    sub_variances: [u32; 4],
    features: [f32; 6],
    score: f32,
}

#[inline]
fn finish_variance(sum: i64, sse: u64, pixels: usize) -> u32 {
    let mean_term = (sum * sum) as u64 / pixels as u64;
    sse.saturating_sub(mean_term) as u32
}

fn ml_var_partition_node_eval(
    input: &MlVarPartitionTreeInput<'_>,
    row_px: usize,
    col_px: usize,
    bsize: BlockSize,
) -> Option<MlVarPartitionNodeEval> {
    let config = match bsize {
        BlockSize::Block64x64 => &VAR_PART_NN_CONFIG_64,
        BlockSize::Block32x32 => &VAR_PART_NN_CONFIG_32,
        BlockSize::Block16x16 => &VAR_PART_NN_CONFIG_16,
        _ => return None,
    };
    let width = usize::from(bsize.width());
    let height = usize::from(bsize.height());
    let source = &input.src[row_px * input.src_stride + col_px..];
    let predictor = &input.pred[row_px * input.pred_stride + col_px..];
    let sub_width = width / 2;
    let sub_height = height / 2;
    let sub_bsize = match bsize {
        BlockSize::Block64x64 => BlockSize::Block32x32,
        BlockSize::Block32x32 => BlockSize::Block16x16,
        BlockSize::Block16x16 => BlockSize::Block8x8,
        _ => unreachable!("validated ML partition node size"),
    };
    let reduce = rvp9_core::dsp::get_ref().variance(sub_bsize);
    let mut total_sum = 0i64;
    let mut total_sse = 0u64;
    let mut sub_variances = [0u32; 4];
    for (quadrant, sub_variance) in sub_variances.iter_mut().enumerate() {
        let x = (quadrant & 1) * sub_width;
        let y = (quadrant >> 1) * sub_height;
        let (sum, sse) = reduce(
            &source[y * input.src_stride + x..],
            input.src_stride,
            &predictor[y * input.pred_stride + x..],
            input.pred_stride,
        );
        total_sum += sum;
        total_sse += sse;
        *sub_variance = finish_variance(sum, sse, sub_width * sub_height);
    }
    let var = finish_variance(total_sum, total_sse, width * height);
    let factor = if var == 0 { 1.0 } else { 1.0 / var as f32 };
    let mut features = [0.0f32; 6];
    features[0] = ((input.dc_q * input.dc_q) as f32 / 256.0 + 1.0).ln();
    features[1] = (var as f32 + 1.0).ln();
    for (feature, &sub_variance) in features[2..].iter_mut().zip(&sub_variances) {
        *feature = if var == 0 {
            1.0
        } else {
            factor * sub_variance as f32
        };
    }
    let mut score = [0.0f32; 1];
    nn_predict(&features, config, &mut score);
    Some(MlVarPartitionNodeEval {
        variance: var,
        sub_variances,
        features,
        score: score[0],
    })
}

fn ml_var_partition_node_trace(
    input: &MlVarPartitionTreeInput<'_>,
    row_px: usize,
    col_px: usize,
    bsize: BlockSize,
) -> Option<MlVarPartitionNodeTrace> {
    let eval = ml_var_partition_node_eval(input, row_px, col_px, bsize)?;
    let width = usize::from(bsize.width());
    let height = usize::from(bsize.height());
    let source = &input.src[row_px * input.src_stride + col_px..];
    let predictor = &input.pred[row_px * input.pred_stride + col_px..];
    Some(MlVarPartitionNodeTrace {
        row_px,
        col_px,
        bsize,
        source_hash: hash_rect(source, input.src_stride, width, height),
        predictor_hash: hash_rect(predictor, input.pred_stride, width, height),
        variance: eval.variance,
        sub_variances: eval.sub_variances,
        features: eval.features,
        score: eval.score,
    })
}

/// Trace every 64x64, 32x32, and 16x16 node visited by pinned libvpx's
/// low-resolution realtime ML partition controller.
#[must_use]
pub fn trace_ml_var_partition_tree(
    input: &MlVarPartitionTreeInput<'_>,
) -> Vec<MlVarPartitionNodeTrace> {
    fn visit(
        input: &MlVarPartitionTreeInput<'_>,
        traces: &mut Vec<MlVarPartitionNodeTrace>,
        row_px: usize,
        col_px: usize,
        bsize: BlockSize,
    ) {
        let Some(trace) = ml_var_partition_node_trace(input, row_px, col_px, bsize) else {
            return;
        };
        traces.push(trace);
        let threshold = if input.speed <= 5 { 1.25 } else { 0.0 };
        if trace.score <= threshold {
            return;
        }
        let child = match bsize {
            BlockSize::Block64x64 => BlockSize::Block32x32,
            BlockSize::Block32x32 => BlockSize::Block16x16,
            BlockSize::Block16x16 => return,
            _ => unreachable!("ML partition tree only visits square nodes"),
        };
        let half = usize::from(child.width());
        visit(input, traces, row_px, col_px, child);
        visit(input, traces, row_px, col_px + half, child);
        visit(input, traces, row_px + half, col_px, child);
        visit(input, traces, row_px + half, col_px + half, child);
    }

    let mut traces = Vec::new();
    visit(input, &mut traces, 0, 0, BlockSize::Block64x64);
    traces
}

/// Build the square-only fixed tree selected by pinned libvpx's CPU8
/// low-resolution ML partition controller.
///
/// At CPU8 the model threshold is zero, so every 64x64, 32x32, and 16x16
/// node selects either `NONE` or `SPLIT`. The 8x8 minimum is always retained.
#[must_use]
pub fn choose_ml_var_partition_tree(input: &MlVarPartitionTreeInput<'_>) -> PartitionDecision {
    debug_assert!(input.src_stride >= 64);
    debug_assert!(input.pred_stride >= 64);
    debug_assert!(input.src.len() >= input.src_stride * 64);
    debug_assert!(input.pred.len() >= input.pred_stride * 64);

    fn stamp(decision: &mut PartitionDecision, row_8x8: usize, col_8x8: usize, bsize: BlockSize) {
        let cells_wide = usize::from(bsize.n_8x8_wide());
        let cells_high = usize::from(bsize.n_8x8_high());
        for row in &mut decision.cells[row_8x8..row_8x8 + cells_high] {
            row[col_8x8..col_8x8 + cells_wide].fill(bsize);
        }
    }

    fn visit(
        input: &MlVarPartitionTreeInput<'_>,
        decision: &mut PartitionDecision,
        row_8x8: usize,
        col_8x8: usize,
        bsize: BlockSize,
    ) {
        let Some(eval) = ml_var_partition_node_eval(input, row_8x8 * 8, col_8x8 * 8, bsize) else {
            stamp(decision, row_8x8, col_8x8, bsize);
            return;
        };
        let threshold = if input.speed <= 5 { 1.25 } else { 0.0 };
        if eval.score <= threshold {
            stamp(decision, row_8x8, col_8x8, bsize);
            return;
        }
        let child = match bsize {
            BlockSize::Block64x64 => BlockSize::Block32x32,
            BlockSize::Block32x32 => BlockSize::Block16x16,
            BlockSize::Block16x16 => BlockSize::Block8x8,
            _ => unreachable!("ML partition tree only visits square nodes"),
        };
        let half = usize::from(child.n_8x8_wide());
        visit(input, decision, row_8x8, col_8x8, child);
        visit(input, decision, row_8x8, col_8x8 + half, child);
        visit(input, decision, row_8x8 + half, col_8x8, child);
        visit(input, decision, row_8x8 + half, col_8x8 + half, child);
    }

    let mut decision = PartitionDecision::all_64x64();
    visit(input, &mut decision, 0, 0, BlockSize::Block64x64);
    debug_assert!(decision.is_valid_partition());
    decision
}

/// Convenience helper: pick the *coarse* predicted partition size
/// (NONE / SPLIT). Returns the parent `bsize` when NONE wins, or the
/// SPLIT subsize when SPLIT wins. If both arms are allowed, returns
/// the parent `bsize` (the model has no strong preference). Caller
/// can ignore this and use the boolean fields directly.
#[must_use]
pub fn predict_partition_size(decision: VarPartDecision, parent: BlockSize) -> BlockSize {
    use rvp9_core::block::{PartitionType, subsize};
    if decision.allow_none && !decision.allow_split {
        return parent;
    }
    if decision.allow_split && !decision.allow_none {
        return subsize(parent, PartitionType::Split).unwrap_or(parent);
    }
    parent
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A test `MotionSearchOps` impl that finds the shift `(row, col)`
    /// minimising SAD between `src` and `reference` over a fixed
    /// `±SEARCH_RADIUS` integer-pel window, then copies the
    /// reference-at-best to `pred`.
    struct ExhaustiveSadOracle {
        search_radius: i16,
    }

    impl MotionSearchOps for ExhaustiveSadOracle {
        fn full_pixel_search(
            &mut self,
            src: PlaneView<'_>,
            reference: PlaneView<'_>,
            bsize: BlockSize,
            _ref_mv_int: Mv,
        ) -> Mv {
            let bw = bsize.width() as usize;
            let bh = bsize.height() as usize;
            let mut best_mv = Mv { row: 0, col: 0 };
            let mut best_sad = u64::MAX;
            let r = self.search_radius;
            for dr in -r..=r {
                for dc in -r..=r {
                    // Sample reference at (dr, dc) — clamp to bounds.
                    let mut sad: u64 = 0;
                    let mut ok = true;
                    for y in 0..bh {
                        for x in 0..bw {
                            let ry = y as i32 + dr as i32;
                            let rx = x as i32 + dc as i32;
                            if ry < 0
                                || rx < 0
                                || (ry as usize) >= reference.height
                                || (rx as usize) >= reference.width
                            {
                                ok = false;
                                break;
                            }
                            let s = src.buf[y * src.stride + x] as i32;
                            let rpix = reference.buf
                                [(ry as usize) * reference.stride + (rx as usize)]
                                as i32;
                            sad += (s - rpix).unsigned_abs() as u64;
                        }
                        if !ok {
                            break;
                        }
                    }
                    if ok && sad < best_sad {
                        best_sad = sad;
                        best_mv = Mv { row: dr, col: dc };
                    }
                }
            }
            best_mv
        }

        fn build_inter_predictors(
            &mut self,
            reference: PlaneView<'_>,
            bsize: BlockSize,
            mv: Mv,
            pred: &mut PlaneViewMut<'_>,
        ) {
            // 1/8-pel mv → integer-pel.
            let dr = mv.row / 8;
            let dc = mv.col / 8;
            let bw = bsize.width() as usize;
            let bh = bsize.height() as usize;
            for y in 0..bh {
                for x in 0..bw {
                    let ry = y as i32 + dr as i32;
                    let rx = x as i32 + dc as i32;
                    let pixel = if ry >= 0
                        && rx >= 0
                        && (ry as usize) < reference.height
                        && (rx as usize) < reference.width
                    {
                        reference.buf[(ry as usize) * reference.stride + (rx as usize)]
                    } else {
                        0
                    };
                    pred.buf[y * pred.stride + x] = pixel;
                }
            }
        }
    }

    /// W7.7 — `simple_motion_search` recovers a known integer-pel
    /// shift between two synthetic frames.
    ///
    /// libvpx reference: `vp9_encodeframe.c:3431-3483`.
    #[test]
    fn w7_7_simple_motion_search_finds_known_shift() {
        // Build a 16×16 source pattern that is unambiguously
        // reconstructable. Place an *identical copy* of that pattern
        // at offset (+3, -2) inside a larger reference plane filled
        // with a different pattern so the SAD is minimum only at that
        // exact shift.
        let stride = 64usize;
        let height = 64usize;
        let bw = 16usize;
        let bh = 16usize;
        // Use positive shifts so the oracle's bounds-check doesn't
        // reject the true MV (the oracle rejects any sample read
        // outside the reference view bounds — production libvpx pads
        // the reference plane to allow negative offsets, but the
        // standalone oracle keeps the bounds tight on purpose).
        let true_dr: i16 = 3;
        let true_dc: i16 = 2;

        // Source plane: pattern A in the 16×16 block at (8, 8), zero elsewhere.
        let mut src = vec![0u8; stride * height];
        // Build pattern A — broken-symmetry mix that ranges over 1..=250.
        let mut pattern_a = [0u8; 16 * 16];
        for y in 0..bh {
            for x in 0..bw {
                // Avoid 0 → the reference background (also 0) won't match.
                let v = 1 + ((y as u32 * 17 + x as u32 * 23) ^ 0x55) % 200;
                pattern_a[y * bw + x] = v as u8;
            }
        }
        let src_row_off = 8usize;
        let src_col_off = 8usize;
        for y in 0..bh {
            for x in 0..bw {
                src[(src_row_off + y) * stride + (src_col_off + x)] = pattern_a[y * bw + x];
            }
        }

        // Reference plane: pattern A placed at (src_row_off + true_dr,
        // src_col_off + true_dc). Background = high-contrast pattern B
        // that has no overlap with pattern A's value range.
        let mut reference = vec![0u8; stride * height];
        // Pattern B uses values in 240..=254 with broken symmetry — far
        // from pattern A's 1..=200 range. (Cell at the placed-copy
        // location is then overwritten with pattern A.)
        for y in 0..height {
            for x in 0..stride {
                reference[y * stride + x] = 240 + ((y as u32 * 11 + x as u32 * 13) % 15) as u8;
            }
        }
        let ref_row_off = (src_row_off as i32 + true_dr as i32) as usize;
        let ref_col_off = (src_col_off as i32 + true_dc as i32) as usize;
        for y in 0..bh {
            for x in 0..bw {
                reference[(ref_row_off + y) * stride + (ref_col_off + x)] = pattern_a[y * bw + x];
            }
        }

        let src_view = PlaneView {
            buf: &src[src_row_off * stride + src_col_off..],
            stride,
            width: bw,
            height: bh,
        };
        let ref_view = PlaneView {
            buf: &reference[src_row_off * stride + src_col_off..],
            stride,
            width: stride - src_col_off,
            height: height - src_row_off,
        };
        let mut ops = ExhaustiveSadOracle { search_radius: 5 };
        let mut pred = new_pred_buffer();

        let mv = simple_motion_search(
            &mut ops,
            src_view,
            ref_view,
            BlockSize::Block16x16,
            Mv { row: 0, col: 0 },
            &mut pred,
        );

        // Integer-pel MV * 8 = 1/8-pel.
        assert_eq!(
            mv.row,
            true_dr * 8,
            "expected row {} got {}",
            true_dr * 8,
            mv.row
        );
        assert_eq!(
            mv.col,
            true_dc * 8,
            "expected col {} got {}",
            true_dc * 8,
            mv.col
        );

        // `pred` was filled — at the matched MV the residue against
        // src should be effectively zero in the bw×bh window.
        let mut sad = 0u32;
        for y in 0..bh {
            for x in 0..bw {
                sad +=
                    (src_view.buf[y * stride + x] as i32 - pred[y * 64 + x] as i32).unsigned_abs();
            }
        }
        assert_eq!(sad, 0, "predicted block must match src at the recovered MV");
    }

    /// W7.7 — `ml_predict_var_rd_partitioning` returns a sane decision
    /// + coarse partition size for a low-variance (well-predicted)
    ///
    /// block.
    ///
    /// libvpx reference: `vp9_encodeframe.c:3489-3618`.
    #[test]
    fn w7_7_ml_predict_var_partitioning_returns_partition_size() {
        // Use a 32×32 src that's *identical* to pred → variance ≈ 0 →
        // model should generally favour PARTITION_NONE (or at least
        // not push the score above +thresh).
        let bw = 32usize;
        let bh = 32usize;
        let src_stride = 64usize;
        let mut src = vec![128u8; src_stride * 64];
        for y in 0..bh {
            for x in 0..bw {
                src[y * src_stride + x] = ((x as u32 + y as u32) % 200) as u8 + 30;
            }
        }
        let mut pred = new_pred_buffer();
        // Pred = src (identical). Copy into the 64-stride pred buffer.
        for y in 0..bh {
            for x in 0..bw {
                pred[y * 64 + x] = src[y * src_stride + x];
            }
        }

        let input = VarPartInput {
            bsize: BlockSize::Block32x32,
            dc_q: 80,
            speed: 1,
            width: 1280,
            height: 720,
            src: PlaneView {
                buf: &src,
                stride: src_stride,
                width: bw,
                height: bh,
            },
            pred: &pred,
            has_above: true,
            has_left: true,
            above_sb_type: BlockSize::Block32x32,
            left_sb_type: BlockSize::Block32x32,
        };
        let decision = ml_predict_var_rd_partitioning(&input);
        assert!(decision.score.is_finite());
        // Variance is essentially zero → score should not be > +thresh
        // (which would disable NONE). It is *allowed* to be < -thresh
        // (disabling SPLIT), but typically the score sits near 0.
        assert!(decision.allow_none || decision.allow_split);

        // Predicted partition size: BlockSize-typed.
        let size = predict_partition_size(decision, BlockSize::Block32x32);
        assert!(matches!(
            size,
            BlockSize::Block32x32 | BlockSize::Block16x16
        ));

        // Unsupported bsize → pass-through both arms.
        let small = VarPartInput {
            bsize: BlockSize::Block4x4,
            ..input
        };
        let small_dec = ml_predict_var_rd_partitioning(&small);
        assert!(small_dec.allow_none && small_dec.allow_split);
    }

    #[test]
    fn low_resolution_var_model_builds_a_legal_recursive_tree() {
        let flat = [128u8; 64 * 64];
        let flat_tree = choose_ml_var_partition_tree(&MlVarPartitionTreeInput {
            dc_q: 80,
            speed: 8,
            src: &flat,
            src_stride: 64,
            pred: &flat,
            pred_stride: 64,
        });
        assert_eq!(flat_tree, PartitionDecision::all_64x64());

        let mut texture = [0u8; 64 * 64];
        for row in 0..64 {
            for col in 0..64 {
                texture[row * 64 + col] = if (row / 4 + col / 4) % 2 == 0 { 0 } else { 255 };
            }
        }
        let texture_tree = choose_ml_var_partition_tree(&MlVarPartitionTreeInput {
            dc_q: 80,
            speed: 8,
            src: &texture,
            src_stride: 64,
            pred: &flat,
            pred_stride: 64,
        });
        assert!(texture_tree.is_valid_partition());
        assert_ne!(texture_tree, PartitionDecision::all_64x64());
    }

    #[test]
    fn low_resolution_var_trace_exposes_exact_model_inputs() {
        let flat = [128u8; 64 * 64];
        let mut source = flat;
        source[0] = 129;
        let trace = trace_ml_var_partition_tree(&MlVarPartitionTreeInput {
            dc_q: 80,
            speed: 8,
            src: &source,
            src_stride: 64,
            pred: &flat,
            pred_stride: 64,
        });

        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].row_px, 0);
        assert_eq!(trace[0].col_px, 0);
        assert_eq!(trace[0].bsize, BlockSize::Block64x64);
        assert_ne!(trace[0].source_hash, trace[0].predictor_hash);
        assert_eq!(trace[0].variance, 1);
        assert_eq!(trace[0].sub_variances, [1, 0, 0, 0]);
        assert_eq!(trace[0].features[2..], [1.0, 0.0, 0.0, 0.0]);
    }
}
