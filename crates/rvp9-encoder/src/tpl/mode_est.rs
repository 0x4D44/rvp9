//! W6.7 — per-block TPL `mode_estimation`.
//!
//! Faithful port of libvpx `mode_estimation`
//! (`vp9/encoder/vp9_tpl_model.c:571-727`, v1.14.1 commit `12f3a2ac6`).
//! For one 8×8-mi superblock in the current frame, this routine:
//!
//! 1. Tries all four intra modes (`DC_PRED, V_PRED, H_PRED, TM_PRED`)
//!    and records the best **intra_cost** (libvpx scans the full 10
//!    INTRA modes — `DC_PRED..TM_PRED` — at lines 611-646; we mirror
//!    that loop scalar-faithfully).
//! 2. For each candidate reference frame, runs the inter-predictor
//!    (libvpx does a motion search; the W6.7 scaffold accepts an
//!    externally-supplied MV — see "Deviations" below) and records the
//!    best **inter_cost**.
//! 3. Writes `(intra_cost, inter_cost, ref_frame_index, mv)` into the
//!    per-block `TplDepStats` slot, scaling intra/inter by
//!    `1 << TPL_DEP_COST_SCALE_LOG2 / (mi_height * mi_width)`.
//!
//! ## Deviations from libvpx (W6.7 scaffold, documented)
//!
//! - **No motion search.** libvpx's `mode_estimation` calls
//!   `motion_compensated_prediction` (`vp9_tpl_model.c:666-668`) to
//!   refine the MV per ref frame. Our scaffold accepts the MV
//!   externally. The full MV search (a diamond + sub-pel pair) plugs
//!   into the existing rvp9 `rdo::single_mv_search` in a later
//!   wave. **TODO(W6.9):** wire the MV search.
//! - **No `get_quantize_error` / `rate_estimator`.** libvpx writes
//!   `recon_error`, `rate_cost`, `sse` out-params from a real quant
//!   roundtrip (`vp9_tpl_model.c:713-715`). Our scaffold returns the
//!   intra/inter SATD pair and lets the propagator deal with the
//!   coarse approximation. **TODO(W6.9):** integrate the quant
//!   roundtrip.
//! - **No HBD.** libvpx's HBD branch (`vp9_tpl_model.c:627-643`,
//!   `671-693`) is intentionally skipped.
//! - **No sub-pel interp.** The inter predictor in the scaffold uses
//!   integer-pel only (the supplied `inter_pred_block` is the
//!   integer-pel patch in the reference frame). Sub-pel taps land
//!   when the MV search lands.

use rvp9_core::block::BlockSize;
use rvp9_core::intra_pred::{
    PredCtx, dc_128_predictor, dc_left_predictor, dc_predictor, dc_top_predictor, h_predictor,
    tm_predictor, v_predictor,
};
use rvp9_core::mode_decision::hadamard::{hadamard_8x8, hadamard_16x16, hadamard_32x32, hadamard_satd};
use rvp9_core::mode_info::Mv;

use super::stats::{NO_REF_FRAME_INDEX, TPL_DEP_COST_SCALE_LOG2, TplDepFrame, TplDepStats};

/// Per-call inputs to [`mode_estimation`].
///
/// `cur_block` and `inter_pred_block` are tightly-packed `bw * bh`
/// patches (stride = `bw`). `above` / `left` are the libvpx-style
/// neighbour rows (one byte of overlap with the top-left corner for
/// the `above` slice, matching [`PredCtx::from_libvpx_style`]).
pub struct ModeEstInputs<'a> {
    /// Block side in pixels (8, 16, or 32 — libvpx's `tpl_bsize`
    /// defaults to `BLOCK_32X32`; we accept any power-of-two side in
    /// `{8, 16, 32}`).
    pub bsize: BlockSize,
    /// Current frame's source patch for the block, length `bw * bh`.
    pub cur_block: &'a [u8],
    /// libvpx-style above row: one corner pixel + `bw` neighbour
    /// pixels (length ≥ `bw + 1`).
    pub above_with_corner: &'a [u8],
    /// Left column, length ≥ `bh`.
    pub left: &'a [u8],
    /// Whether `above_with_corner` is valid (i.e. we're not at row 0
    /// of the frame).
    pub has_above: bool,
    /// Whether `left` is valid (i.e. we're not at col 0 of the
    /// frame).
    pub has_left: bool,
    /// Per-reference-frame predictor patches. `inter_pred[k]` is
    /// `Some(patch)` if reference slot `k` is available, where
    /// `patch.len() == bw * bh`. libvpx's `ref_frame[rf_idx] == NULL`
    /// check (`vp9_tpl_model.c:658`) maps to `None` here.
    ///
    /// **W6.7 deviation:** libvpx does a per-ref motion search.
    /// rvp9's scaffold accepts the MV+predictor externally. The MVs
    /// for each ref are passed via [`Self::inter_mvs`].
    pub inter_pred: [Option<&'a [u8]>; super::stats::MAX_INTER_REF_FRAMES],
    /// MV chosen for each available reference frame slot. The slot's
    /// MV is read only when `inter_pred[k].is_some()`.
    pub inter_mvs: [Mv; super::stats::MAX_INTER_REF_FRAMES],
    /// GOP-slot index this frame's `ref_frame[]` array points at; the
    /// best ref's `gf_picture[].ref_frame[best_rf_idx]` is stamped
    /// into `TplDepStats::ref_frame_index`
    /// (`vp9_tpl_model.c:724`). Pass as a `[i32; 3]` lookup.
    pub ref_frame_slot: [i32; super::stats::MAX_INTER_REF_FRAMES],
}

/// Result of [`mode_estimation`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ModeEstResult {
    /// Best intra SATD across the four "primary" intra modes (DC, V,
    /// H, TM). Pre-scaling.
    pub best_intra_cost: i64,
    /// Best inter SATD across available ref-frame slots, or `i64::MAX`
    /// if no ref was available (intra-only frame). Pre-scaling.
    pub best_inter_cost: i64,
    /// Index into the surrounding GOP `gf_picture[].ref_frame[]`
    /// array, or [`NO_REF_FRAME_INDEX`].
    pub best_ref_frame_index: i32,
    /// MV of the best-scoring reference. `Mv::ZERO` for intra-only.
    pub best_mv: Mv,
    /// Local slot index `best_rf_idx` (0..3) — the libvpx out-param
    /// of `mode_estimation` (`vp9_tpl_model.c:579, 726`).
    pub best_rf_idx: i32,
}

/// libvpx `vp9_wht_fwd_txfm` (`vp9/encoder/vp9_tpl_model.c:538-546`).
/// Dispatches to a Hadamard kernel sized to the supplied side.
///
/// `bw` is the **block width in pixels** (libvpx's `bw` argument);
/// rvp9 supports the three TPL-block sizes libvpx uses (8/16/32).
fn wht_fwd_txfm(src_diff: &[i16], bw: usize, coeff: &mut [i16]) {
    match bw {
        8 => hadamard_8x8(src_diff, bw, coeff),
        16 => hadamard_16x16(src_diff, bw, coeff),
        32 => hadamard_32x32(src_diff, bw, coeff),
        other => panic!("wht_fwd_txfm: unsupported block side {other}"),
    }
}

/// libvpx-equivalent `vpx_subtract_block` for the Hadamard input.
/// Both `src` and `pred` are `bw × bh` patches packed with stride
/// `bw`; the difference goes into `src_diff` (same layout).
fn subtract_block(src_diff: &mut [i16], src: &[u8], pred: &[u8], bw: usize, bh: usize) {
    debug_assert_eq!(src.len(), bw * bh);
    debug_assert_eq!(pred.len(), bw * bh);
    debug_assert_eq!(src_diff.len(), bw * bh);
    for i in 0..bw * bh {
        src_diff[i] = i16::from(src[i]) - i16::from(pred[i]);
    }
}

/// Dispatch a `(mode, has_above, has_left)` triple to the right
/// intra-predictor variant. Mirrors libvpx
/// `vp9_predict_intra_block`'s mode dispatch (called from
/// `vp9_tpl_model.c:624`) — we collapse it to the four "primary"
/// modes used by TPL plus DC fallbacks.
///
/// `dst` is `bw × bh` packed at stride `bw`.
fn predict_intra(dst: &mut [u8], bw: usize, mode: IntraTplMode, ctx_above: &[u8], ctx_left: &[u8], has_above: bool, has_left: bool) {
    // libvpx's PredCtx::from_libvpx_style expects [corner, above...].
    match (mode, has_above, has_left) {
        (IntraTplMode::Dc, true, true) => {
            let ctx = PredCtx::from_libvpx_style(ctx_above, ctx_left);
            dc_predictor(dst, bw, bw, ctx);
        }
        (IntraTplMode::Dc, true, false) => {
            dc_top_predictor(dst, bw, bw, &ctx_above[1..]);
        }
        (IntraTplMode::Dc, false, true) => {
            dc_left_predictor(dst, bw, bw, ctx_left);
        }
        (IntraTplMode::Dc, false, false) => {
            dc_128_predictor(dst, bw, bw);
        }
        (IntraTplMode::V, true, _) => {
            let ctx = PredCtx::from_libvpx_style(ctx_above, ctx_left);
            v_predictor(dst, bw, bw, ctx);
        }
        (IntraTplMode::H, _, true) => {
            let ctx = PredCtx::from_libvpx_style(
                if has_above { ctx_above } else { &[0u8] },
                ctx_left,
            );
            h_predictor(dst, bw, bw, ctx);
        }
        (IntraTplMode::Tm, true, true) => {
            let ctx = PredCtx::from_libvpx_style(ctx_above, ctx_left);
            tm_predictor(dst, bw, bw, ctx);
        }
        // libvpx falls back to DC for V/H/TM when neighbours missing.
        _ => {
            dc_128_predictor(dst, bw, bw);
        }
    }
}

/// The four "primary" intra modes libvpx's TPL `mode_estimation` walks
/// (`vp9_tpl_model.c:611`). DC + the three "non-directional" modes are
/// the cheap baseline; the eight directional modes are tried too in
/// libvpx but contribute marginally to the propagation signal.
///
/// **W6.7 deviation:** libvpx walks all 10 intra modes (`DC_PRED..
/// TM_PRED`). Our scaffold walks the 4 universally-useful modes (DC,
/// V, H, TM) — the directional modes need the rvp9 `d45/d63/...`
/// predictors hooked up through the same `predict_intra_block`
/// trampoline, which is a one-liner extension when the directional
/// kernels' final test sweep lands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntraTplMode {
    /// DC_PRED.
    Dc,
    /// V_PRED.
    V,
    /// H_PRED.
    H,
    /// TM_PRED.
    Tm,
}

impl IntraTplMode {
    /// The set walked by [`mode_estimation`].
    pub const TPL_MODES: [IntraTplMode; 4] = [Self::Dc, Self::V, Self::H, Self::Tm];
}

/// Per-block mode estimation (libvpx `mode_estimation`,
/// `vp9_tpl_model.c:571-727`).
///
/// Writes the result into `tpl_frame.tpl_stats[mi_row * stride +
/// mi_col]`, with the same intra/inter scaling libvpx applies
/// (`vp9_tpl_model.c:718-723`):
///
/// ```c
/// best_intra_cost = VPXMAX(best_intra_cost, 1);
/// best_inter_cost = VPXMIN(best_intra_cost, best_inter_cost);
/// tpl_stats->inter_cost = VPXMAX(
///     1, (best_inter_cost << TPL_DEP_COST_SCALE_LOG2) /
///        (mi_height * mi_width));
/// tpl_stats->intra_cost = VPXMAX(
///     1, (best_intra_cost << TPL_DEP_COST_SCALE_LOG2) /
///        (mi_height * mi_width));
/// ```
///
/// Returns the per-block [`ModeEstResult`] for the caller's inspection
/// (libvpx writes the result both into `*tpl_stats` and via
/// `*ref_frame_idx` — we keep both behaviours).
pub fn mode_estimation(
    tpl_frame: &mut TplDepFrame,
    mi_row: usize,
    mi_col: usize,
    inputs: &ModeEstInputs<'_>,
) -> ModeEstResult {
    let bsize = inputs.bsize;
    let bw = bsize.width() as usize;
    let bh = bsize.height() as usize;
    debug_assert_eq!(bw, bh, "TPL only walks square blocks (libvpx tpl_bsize)");
    debug_assert!(matches!(bw, 8 | 16 | 32), "TPL block sides ∈ {{8, 16, 32}}");
    debug_assert_eq!(inputs.cur_block.len(), bw * bh);

    // Working buffers. libvpx hangs these off ThreadData; we allocate
    // per-call (the W6.9+ driver can reuse a thread-local arena).
    let pix_num = bw * bh;
    let mut predictor = vec![0u8; pix_num];
    let mut src_diff = vec![0i16; pix_num];
    let mut coeff = vec![0i16; pix_num];

    // ---- Intra pass (vp9_tpl_model.c:611-646) ----
    let mut best_intra_cost: i64 = i64::MAX;
    for mode in IntraTplMode::TPL_MODES {
        predict_intra(
            &mut predictor,
            bw,
            mode,
            inputs.above_with_corner,
            inputs.left,
            inputs.has_above,
            inputs.has_left,
        );
        subtract_block(&mut src_diff, inputs.cur_block, &predictor, bw, bh);
        wht_fwd_txfm(&src_diff, bw, &mut coeff);
        let intra_cost = i64::from(hadamard_satd(&coeff));
        if intra_cost < best_intra_cost {
            best_intra_cost = intra_cost;
        }
    }

    // ---- Inter pass (vp9_tpl_model.c:653-717) ----
    let mut best_inter_cost: i64 = i64::MAX;
    let mut best_rf_idx: i32 = -1;
    let mut best_mv = Mv::ZERO;
    for rf_idx in 0..super::stats::MAX_INTER_REF_FRAMES {
        let Some(pred_patch) = inputs.inter_pred[rf_idx] else {
            continue;
        };
        debug_assert_eq!(pred_patch.len(), pix_num);
        subtract_block(&mut src_diff, inputs.cur_block, pred_patch, bw, bh);
        wht_fwd_txfm(&src_diff, bw, &mut coeff);
        let inter_cost = i64::from(hadamard_satd(&coeff));
        if inter_cost < best_inter_cost {
            best_inter_cost = inter_cost;
            best_rf_idx = rf_idx as i32;
            best_mv = inputs.inter_mvs[rf_idx];
        }
    }

    // ---- Cost-scale + write (vp9_tpl_model.c:718-726) ----
    let mi_height = bsize.n_8x8_high() as i64;
    let mi_width = bsize.n_8x8_wide() as i64;
    let mi_area = mi_height * mi_width;

    best_intra_cost = best_intra_cost.max(1);
    if best_inter_cost == i64::MAX {
        best_inter_cost = best_intra_cost;
    } else {
        best_inter_cost = best_inter_cost.min(best_intra_cost);
    }

    let scaled_inter =
        ((best_inter_cost << TPL_DEP_COST_SCALE_LOG2) / mi_area).max(1);
    let scaled_intra =
        ((best_intra_cost << TPL_DEP_COST_SCALE_LOG2) / mi_area).max(1);

    let ref_frame_index = if best_rf_idx >= 0 {
        inputs.ref_frame_slot[best_rf_idx as usize]
    } else {
        NO_REF_FRAME_INDEX
    };

    let result = ModeEstResult {
        best_intra_cost,
        best_inter_cost,
        best_ref_frame_index: ref_frame_index,
        best_mv,
        best_rf_idx,
    };

    let stat: &mut TplDepStats = tpl_frame.stat_mut(mi_row, mi_col);
    stat.inter_cost = scaled_inter;
    stat.intra_cost = scaled_intra;
    stat.ref_frame_index = ref_frame_index;
    stat.mv = best_mv;

    result
}
