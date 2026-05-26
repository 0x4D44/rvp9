//! W6.8 — `get_rdmult_delta` + `vp9_get_adaptive_rdmult` hook.
//!
//! Faithful port of:
//!
//! - `get_rdmult_delta` — `vp9/encoder/vp9_encodeframe.c:3719-3768`
//!   (v1.14.1 commit `12f3a2ac6`). Reads the propagated `intra_cost`
//!   and `mc_dep_cost` from the TPL state for the current superblock
//!   and computes a `dr` adjustment of the original `rdmult`, capped
//!   to `[orig_rdmult / 2, orig_rdmult * 3/2]`.
//! - `vp9_get_adaptive_rdmult` — `vp9/encoder/vp9_rd.c:304-310`.
//!   Scales the base-qindex `rdmult` by `1 / beta`.
//! - RDO consumer hook — `vp9_encodeframe.c:2235-2242`. The picker
//!   reads `x->cb_rdmult` into `x->rdmult` when the TPL model is
//!   enabled and the AQ mode is compatible. This file ports the
//!   *production* side (compute `cb_rdmult`); the picker side
//!   integration is a one-liner when the rvp9 RDO driver lands.

use super::stats::{MAX_ARF_GOP_SIZE, TplDepFrame};
use rvp9_core::block::BlockSize;

/// libvpx `vp9_get_adaptive_rdmult` (`vp9/encoder/vp9_rd.c:304-310`):
///
/// ```c
/// int vp9_get_adaptive_rdmult(const VP9_COMP *cpi, double beta) {
///   int rdmult = vp9_compute_rd_mult_based_on_qindex(cpi, cpi->common.base_qindex);
///   rdmult = (int)((double)rdmult / beta);
///   rdmult = rdmult > 0 ? rdmult : 1;
///   return modulate_rdmult(cpi, rdmult);
/// }
/// ```
///
/// rvp9 scaffold takes `base_rdmult` (the q-index-derived value)
/// directly rather than recomputing from `cpi->common.base_qindex` —
/// the rate-control pipeline owns `base_rdmult`. The `modulate_rdmult`
/// post-step (used by `tune=ssim`) is out of scope.
///
/// `beta` is the TPL "intra-to-dep" ratio; values < 1 mean the block
/// has high dep cost (downstream depends on it heavily) and the
/// rdmult should shrink.
#[must_use]
pub fn vp9_get_adaptive_rdmult(base_rdmult: i32, beta: f64) -> i32 {
    if !(beta.is_finite()) || beta <= 0.0 {
        return base_rdmult.max(1);
    }
    let scaled = (f64::from(base_rdmult) / beta) as i32;
    scaled.max(1)
}

/// Output of [`get_rdmult_delta`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RdmultDelta {
    /// The new per-superblock `cb_rdmult` (libvpx
    /// `vp9_encodeframe.c:3767` returns this as `dr`). Always > 0.
    pub cb_rdmult: i32,
    /// Whether the TPL model was consulted (`true`) or the original
    /// rdmult was returned unchanged (`false`).
    pub used_tpl: bool,
}

/// libvpx `get_rdmult_delta` (`vp9/encoder/vp9_encodeframe.c:3719-3768`).
///
/// For the superblock at `(mi_row, mi_col)` of `bsize`, sums
/// `intra_cost` and `mc_dep_cost` over the in-bounds 8×8 sub-mis,
/// computes `rk = intra_cost / mc_dep_cost`, then `beta = r0 / rk`,
/// and returns `vp9_get_adaptive_rdmult(beta)` capped to `[orig/2,
/// orig*3/2]`.
///
/// Mirrors:
///
/// ```c
/// for (row = mi_row; row < mi_row + mi_high; ++row) {
///   for (col = mi_col; col < mi_col + mi_wide; ++col) {
///     TplDepStats *this_stats = &tpl_stats[row * tpl_stride + col];
///     if (row >= cpi->common.mi_rows || col >= cpi->common.mi_cols) continue;
///     intra_cost += this_stats->intra_cost;
///     mc_dep_cost += this_stats->mc_dep_cost;
///   }
/// }
/// r0 = cpi->rd.r0;
/// rk = (double)intra_cost / mc_dep_cost;
/// beta = r0 / rk;
/// dr = vp9_get_adaptive_rdmult(cpi, beta);
/// dr = VPXMIN(dr, orig_rdmult * 3 / 2);
/// dr = VPXMAX(dr, orig_rdmult * 1 / 2);
/// dr = VPXMAX(1, dr);
/// ```
///
/// Inputs:
/// - `tpl_frame`: TPL state for the current frame (libvpx
///   `cpi->tpl_stats[gf_group_index]`). If `is_valid` is false the
///   function returns `orig_rdmult` unchanged.
/// - `mi_row` / `mi_col`: top-left mi-unit of the superblock.
/// - `bsize`: superblock size (libvpx feeds `BLOCK_64X64` here).
/// - `orig_rdmult`: the picker's base `rdmult` (q-index-derived).
/// - `r0`: libvpx `cpi->rd.r0`, the frame-level RD ratio. The
///   "perfect" `rk` baseline.
/// - `frame_mi_rows` / `frame_mi_cols`: extents to clamp against
///   (libvpx `cpi->common.mi_rows` / `cpi->common.mi_cols`).
/// - `gf_group_index` / `layer_depth`: libvpx gates the rdmult
///   adjustment to `gf_group_index < MAX_ARF_GOP_SIZE` and
///   `layer_depth <= 1`. We mirror the gate exactly.
pub fn get_rdmult_delta(
    tpl_frame: &TplDepFrame,
    mi_row: usize,
    mi_col: usize,
    bsize: BlockSize,
    orig_rdmult: i32,
    r0: f64,
    frame_mi_rows: usize,
    frame_mi_cols: usize,
    gf_group_index: usize,
    layer_depth: u8,
) -> RdmultDelta {
    // libvpx vp9_encodeframe.c:3735 gate.
    if gf_group_index >= MAX_ARF_GOP_SIZE {
        return RdmultDelta {
            cb_rdmult: orig_rdmult.max(1),
            used_tpl: false,
        };
    }
    // libvpx vp9_encodeframe.c:3738.
    if !tpl_frame.is_valid {
        return RdmultDelta {
            cb_rdmult: orig_rdmult.max(1),
            used_tpl: false,
        };
    }
    // libvpx vp9_encodeframe.c:3742 — only the base layer participates.
    if layer_depth > 1 {
        return RdmultDelta {
            cb_rdmult: orig_rdmult.max(1),
            used_tpl: false,
        };
    }

    let mi_high = bsize.n_8x8_high() as usize;
    let mi_wide = bsize.n_8x8_wide() as usize;
    let stride = tpl_frame.stride;

    let mut intra_cost: i64 = 0;
    let mut mc_dep_cost: i64 = 0;
    for row in mi_row..mi_row + mi_high {
        for col in mi_col..mi_col + mi_wide {
            if row >= frame_mi_rows || col >= frame_mi_cols {
                continue;
            }
            // Clamp against the TPL grid extent (defensive: the libvpx
            // assertion is that `frame_mi_*` matches `tpl_frame->mi_*`).
            if row >= tpl_frame.mi_rows || col >= tpl_frame.mi_cols {
                continue;
            }
            let this = &tpl_frame.tpl_stats[row * stride + col];
            intra_cost += this.intra_cost;
            mc_dep_cost += this.mc_dep_cost;
        }
    }

    // libvpx vp9_encodeframe.c:3758-3760. Guard against division by
    // zero — libvpx relies on the propagator's invariant that
    // `mc_dep_cost >= intra_cost > 0`, but our scaffold is also
    // exercised by tests with unpopulated frames.
    if mc_dep_cost == 0 || intra_cost == 0 || r0 <= 0.0 {
        return RdmultDelta {
            cb_rdmult: orig_rdmult.max(1),
            used_tpl: false,
        };
    }

    let rk = intra_cost as f64 / mc_dep_cost as f64;
    let beta = r0 / rk;
    let mut dr = vp9_get_adaptive_rdmult(orig_rdmult, beta);

    // libvpx vp9_encodeframe.c:3762-3765 cap.
    let upper = orig_rdmult.saturating_mul(3) / 2;
    let lower = orig_rdmult / 2;
    dr = dr.min(upper);
    dr = dr.max(lower);
    dr = dr.max(1);

    RdmultDelta {
        cb_rdmult: dr,
        used_tpl: true,
    }
}

/// libvpx RDO consumer hook (`vp9_encodeframe.c:2235-2242`):
///
/// ```c
/// if (cpi->sf.enable_tpl_model &&
///     (cpi->oxcf.aq_mode == NO_AQ || cpi->oxcf.aq_mode == PERCEPTUAL_AQ)) {
///   x->rdmult = x->cb_rdmult;
///   if (oxcf->tuning == VP8_TUNE_SSIM) {
///     set_ssim_rdmult(cpi, x, bsize, mi_row, mi_col, &x->rdmult);
///   }
/// }
/// ```
///
/// The rvp9 scaffold version takes the gating flags as plain inputs
/// (no `VP9_COMP` to walk yet) and returns the rdmult the picker
/// should load. `cb_rdmult` is the [`RdmultDelta::cb_rdmult`] returned
/// by [`get_rdmult_delta`].
///
/// Returns the rdmult to set on `MACROBLOCK::rdmult` for the
/// superblock RDO loop. `tune_ssim` adjustment is **TODO(W6.9+)**.
#[must_use]
pub fn rdo_apply_cb_rdmult(
    base_rdmult: i32,
    cb_rdmult: i32,
    enable_tpl_model: bool,
    aq_compatible: bool,
) -> i32 {
    if enable_tpl_model && aq_compatible {
        cb_rdmult.max(1)
    } else {
        base_rdmult.max(1)
    }
}
