//! A23 — single-GOP forward TPL pass driver (`mc_flow_dispenser`
//! equivalent).
//!
//! This is the *driver half* of the TPL subsystem: it ties the ported
//! standalone primitives ([`super::mode_est::mode_estimation`],
//! [`super::update::tpl_model_store`] /
//! [`super::update::tpl_model_update`]) into a real per-frame pass that
//! libvpx names `mc_flow_dispenser`
//! (`vp9/encoder/vp9_tpl_model.c:1138-1255`, v1.14.1).
//!
//! ## What it does
//!
//! Given a forward window of GOP SOURCE frames (the frame being coded
//! plus the inter frames that follow it, in display order) and the LAST
//! reference reconstruction for the window's first frame, the driver:
//!
//! 1. For every 8×8 block of every frame in the window runs
//!    [`mode_estimation`](super::mode_est::mode_estimation): the four
//!    "primary" intra modes (SATD vs the block's own source neighbours)
//!    and an inter candidate (SATD vs an integer-pel motion-compensated
//!    predictor patch produced by a real diamond motion search —
//!    [`diamond_search_sad`], the encoder's shipped ME). Mirrors libvpx
//!    `mode_estimation` (`vp9_tpl_model.c:571-727`), whose inter pass
//!    calls `motion_compensated_prediction` (`:280-330`,
//!    `vp9_full_pixel_search` NSTEP + sub-pel).
//! 2. [`tpl_model_store`](super::update::tpl_model_store) recomputes
//!    `mc_dep_cost = intra_cost + mc_flow` (`vp9_tpl_model.c:372-390`).
//! 3. [`tpl_model_update`](super::update::tpl_model_update) projects the
//!    block's dependency flow into its reference frame's grid
//!    (`vp9_tpl_model.c:475-489`).
//!
//! The window is processed in **reverse display order** (the last frame
//! first), so the dependency flow propagates *backward* toward the frame
//! being coded — exactly libvpx's forward-frames / backward-propagation
//! structure. After the pass the coded frame's grid carries real
//! `mc_flow` accumulated from its downstream dependents, so
//! `mc_dep_cost > intra_cost` and [`super::rdmult::get_rdmult_delta`]
//! produces a non-trivial per-superblock rdmult.
//!
//! ## Recon staging (matches libvpx)
//!
//! libvpx's TPL does **not** reconstruct the GOP. Verified in
//! `init_gop_frames` (`vp9_tpl_model.c:73-96`): the golden anchor
//! (`gf_picture[0].frame`) is the existing GOLDEN *recon* buffer, but the
//! base ARF (`gf_picture[1].frame = cpi->Source`) and every P-frame
//! (`gf_picture[frame_idx].frame = &buf->img`, the *lookahead source
//! image*) predict from **SOURCE** pixels.
//! `motion_compensated_prediction` (`:304-307`) and `mode_estimation`
//! (`:666-696`) read `ref_frame[rf_idx]->y_buffer` directly. So the only
//! "recon" the linear-P-chain TPL touches is the prior committed LAST
//! reference (the start of the chain); every intra-window reference is
//! the source frame. This driver matches that exactly: the window is
//! source frames; the first frame's reference is the committed LAST
//! recon the caller threads in.
//!
//! ## Scope (bounded single-GOP increment, documented follow-ups)
//!
//! - **Linear P-chain only.** No nested-ARF stack, no OVERLAY /
//!   USE_BUF_FRAME branches (`vp9_tpl_model.c:102-119`). The window is a
//!   straight `F, F+1, …` display-order chain where frame `k` references
//!   frame `k-1`. Multi-ARF GOP topology is a documented follow-up
//!   (~400 LOC: the `arf_index_stack` walk + per-layer ref wiring).
//! - **Integer-pel inter predictor.** libvpx refines to sub-pel
//!   (`find_fractional_mv_step`, `:323-327`); this driver uses the
//!   full-pel diamond result and extracts the integer-pel patch. Sub-pel
//!   MC for the TPL predictor is a follow-up (~80 LOC: route through the
//!   8-tap MC kernel). The propagation maths is unaffected — it consumes
//!   the SATD pair, not the MV fraction.
//! - **No quant roundtrip.** libvpx writes `recon_error` / `rate_cost`
//!   from `get_quantize_error` (`:713-715`); those feed only the Qmode
//!   sidecar (`tpl_store_before_propagation`), which rvp9 does not
//!   expose. The propagator reads only intra/inter SATD.
//! - **Y (luma) only.** libvpx's TPL is luma-only too
//!   (`motion_compensated_prediction` operates on `plane[0]`).

use rvp9_core::block::BlockSize;
use rvp9_core::cost::mv_cost::{MvCosts, build_nmv_cost_table};
use rvp9_core::entropy::FrameContext;
use rvp9_core::mode_info::Mv;
use rvp9_core::mv_search::{
    FullPixelSearchParams, MvLimits, PlaneRef, SearchSiteConfig, diamond_search_sad,
    init_dsmotion_compensation, set_mv_search_range,
};

use super::mode_est::{ModeEstInputs, mode_estimation};
use super::stats::{MAX_INTER_REF_FRAMES, NO_REF_FRAME_INDEX, TplDepFrame};
use super::update::{tpl_model_store, tpl_model_update};

/// TPL block size — the 8×8 mi-unit grid the propagation operates on
/// (libvpx's `tpl_model_update` inner loop is `BLOCK_8X8`,
/// `vp9_tpl_model.c:486`; the dispenser default `tpl_bsize` is
/// `BLOCK_32X32`, but running `mode_estimation` at 8×8 keeps the whole
/// pipeline on a single uniform grid and the propagation identical).
const TPL_BSIZE: BlockSize = BlockSize::Block8x8;
const TPL_BW: usize = 8;

/// libvpx `sadperbit16` seed for the TPL full-pixel search
/// (`vp9_tpl_model.c:290`, `x->sadperbit16`). A fixed coarse weight is
/// faithful for the bounded pass (the real value is q-derived via
/// `vp9_initialize_me_consts`; 16 matches the inter dispatch path's
/// `sad_per_bit` literal in `lib.rs`).
const TPL_SAD_PER_BIT: i32 = 16;

/// One frame of the TPL forward window: a tight-stride luma SOURCE plane.
pub struct TplWindowFrame<'a> {
    /// Luma source pixels, tight stride = `width`.
    pub y: &'a [u8],
}

/// Output of [`run_single_gop_tpl`].
pub struct TplPassResult {
    /// Per-8×8-block dependency stats for the frame being coded
    /// (window slot 0), with `mc_flow` / `mc_dep_cost` populated from the
    /// backward propagation. `is_valid` is set so
    /// [`super::rdmult::get_rdmult_delta`] consults it.
    pub coded_frame: TplDepFrame,
    /// Frame-level `r0 = Σ intra_cost / Σ mc_dep_cost` over the coded
    /// frame's grid (libvpx `cpi->rd.r0`, `vp9_encodeframe.c:6162`). The
    /// "perfect" `rk` baseline `get_rdmult_delta` divides by.
    pub r0: f64,
}

/// Extract the libvpx-style intra neighbour rows for an 8×8 block at
/// pixel `(px, py)` of a tight-stride plane (`stride = w`). Returns
/// `(above_with_corner, left, has_above, has_left)`.
///
/// `above_with_corner[0]` is the top-left corner pixel (libvpx
/// `above[-1]`); `above_with_corner[1..=bw]` is the row directly above.
/// At the frame edge libvpx substitutes 129 (above) / 127 (left) — the
/// `PredCtx` consumers handle the missing-neighbour case via the
/// `has_above` / `has_left` flags, so the filler value is inert.
fn intra_neighbours(
    plane: &[u8],
    w: usize,
    h: usize,
    px: usize,
    py: usize,
) -> ([u8; TPL_BW + 1], [u8; TPL_BW], bool, bool) {
    let has_above = py > 0;
    let has_left = px > 0;

    let mut above = [129u8; TPL_BW + 1];
    if has_above {
        let arow = py - 1;
        // Corner: above-left pixel (or left frame edge → repeat above[0]).
        above[0] = if has_left {
            plane[arow * w + (px - 1)]
        } else {
            plane[arow * w + px]
        };
        for (c, slot) in above[1..].iter_mut().enumerate() {
            let ac = px + c;
            *slot = if ac < w {
                plane[arow * w + ac]
            } else {
                plane[arow * w + (w - 1)]
            };
        }
    }

    let mut left = [127u8; TPL_BW];
    if has_left {
        for (r, slot) in left.iter_mut().enumerate() {
            let lr = py + r;
            *slot = if lr < h {
                plane[lr * w + (px - 1)]
            } else {
                plane[(h - 1) * w + (px - 1)]
            };
        }
    }

    (above, left, has_above, has_left)
}

/// Extract a tight `bw × bw` block from a tight-stride plane at pixel
/// `(px, py)`, clamping reads at the frame edge (the TPL block grid can
/// straddle the padded right/bottom margin of a non-multiple-of-8 frame;
/// rvp9 frames are 64-aligned so this is exact, but the clamp is
/// defensive).
fn copy_src_block(plane: &[u8], w: usize, h: usize, px: usize, py: usize, out: &mut [u8]) {
    for r in 0..TPL_BW {
        let sy = (py + r).min(h - 1);
        for c in 0..TPL_BW {
            let sx = (px + c).min(w - 1);
            out[r * TPL_BW + c] = plane[sy * w + sx];
        }
    }
}

/// Run a real full-pixel diamond motion search for the 8×8 block at
/// `(px, py)` of `src` against `reference`, returning the best full-pel
/// MV and the integer-pel predictor patch extracted at that MV. Returns
/// `None` if the search produced no in-bounds candidate (the block is
/// then treated as intra-only by the caller).
fn motion_search_and_predict(
    src_plane: &PlaneRef<'_>,
    ref_plane: &PlaneRef<'_>,
    px: usize,
    py: usize,
    site_config: &SearchSiteConfig,
    mv_costs: &MvCosts,
    out_patch: &mut [u8],
) -> Option<Mv> {
    // libvpx `motion_compensated_prediction` (`vp9_tpl_model.c:312`):
    // best_ref_mv1 = {0,0}; search range set around it.
    let mut mv_limits = MvLimits::default();
    set_mv_search_range(&mut mv_limits, 0, 0);

    let params = FullPixelSearchParams {
        block_size: TPL_BSIZE,
        src_plane,
        src_offset: (px, py),
        ref_plane,
        center_mv: Mv::ZERO,
        start_mv: None,
        mv_limits,
        sad_per_bit: TPL_SAD_PER_BIT,
        mv_costs,
        max_iters: site_config.total_steps as i32,
        site_config,
    };
    let res = diamond_search_sad(&params);
    if res.best_sad == u32::MAX {
        return None;
    }

    let mv_row = res.best_mv_row;
    let mv_col = res.best_mv_col;

    // Extract the integer-pel predictor patch from the reference at the
    // chosen MV. If any pixel falls outside the reference plane the
    // candidate is rejected (intra-only).
    let w = ref_plane.width;
    let h = ref_plane.height;
    let stride = ref_plane.stride;
    let data = ref_plane.data;
    for r in 0..TPL_BW {
        let ry = py as isize + isize::from(mv_row) + r as isize;
        if ry < 0 || ry as usize >= h {
            return None;
        }
        for c in 0..TPL_BW {
            let rx = px as isize + isize::from(mv_col) + c as isize;
            if rx < 0 || rx as usize >= w {
                return None;
            }
            out_patch[r * TPL_BW + c] = data[ry as usize * stride + rx as usize];
        }
    }

    // MV is returned in 1/8-pel units (full-pel << 3).
    Some(Mv {
        row: mv_row.saturating_mul(8),
        col: mv_col.saturating_mul(8),
    })
}

/// Build the `mode_estimation` inputs + run it for one 8×8 block of the
/// `cur` frame, motion-searching against `reference` (a SOURCE plane for
/// intra-window frames, or the committed LAST recon for the window's
/// first frame). `ref_slot` is the GOP-slot index of the reference for
/// the `TplDepStats::ref_frame_index` stamp.
#[allow(clippy::too_many_arguments)]
fn estimate_block(
    tpl_frame: &mut TplDepFrame,
    cur_y: &[u8],
    ref_plane: Option<&PlaneRef<'_>>,
    w: usize,
    h: usize,
    mi_row: usize,
    mi_col: usize,
    site_config: &SearchSiteConfig,
    mv_costs: &MvCosts,
    ref_slot: i32,
) {
    let px = mi_col * TPL_BW;
    let py = mi_row * TPL_BW;

    let mut cur_block = [0u8; TPL_BW * TPL_BW];
    copy_src_block(cur_y, w, h, px, py, &mut cur_block);

    let (above, left, has_above, has_left) = intra_neighbours(cur_y, w, h, px, py);

    let src_plane = PlaneRef {
        data: cur_y,
        stride: w,
        width: w,
        height: h,
    };

    let mut patch = [0u8; TPL_BW * TPL_BW];
    let mut inter_pred: [Option<&[u8]>; MAX_INTER_REF_FRAMES] = [None, None, None];
    let mut inter_mvs = [Mv::ZERO; MAX_INTER_REF_FRAMES];
    let mut ref_frame_slot = [NO_REF_FRAME_INDEX; MAX_INTER_REF_FRAMES];

    if let Some(rp) = ref_plane
        && let Some(mv) =
            motion_search_and_predict(&src_plane, rp, px, py, site_config, mv_costs, &mut patch)
    {
        inter_pred[0] = Some(&patch[..]);
        inter_mvs[0] = mv;
        ref_frame_slot[0] = ref_slot;
    }

    let inputs = ModeEstInputs {
        bsize: TPL_BSIZE,
        cur_block: &cur_block,
        above_with_corner: &above,
        left: &left,
        has_above,
        has_left,
        inter_pred,
        inter_mvs,
        ref_frame_slot,
    };
    mode_estimation(tpl_frame, mi_row, mi_col, &inputs);
}

/// Driver — run the single-GOP forward TPL pass.
///
/// `window[0]` is the frame being coded; `window[1..]` are the inter
/// frames that follow it in display order (the linear P-chain). Frame
/// `k` references frame `k-1`'s SOURCE plane; frame 0 references the
/// committed LAST reconstruction `last_recon_y` (libvpx's "GOLDEN
/// anchor"-equivalent chain start). `width` / `height` are the coded
/// luma dimensions (tight stride = `width`).
///
/// Returns the coded frame's populated [`TplDepFrame`] + the frame-level
/// `r0`. Requires `window.len() >= 1`. With `window.len() == 1` no
/// downstream dependent exists, so no propagation occurs and the coded
/// frame's `mc_dep_cost == intra_cost` → `r0 == 1.0` and
/// `get_rdmult_delta` leaves rdmult unchanged (an honest no-op, not a
/// fake stat).
///
/// Mirrors libvpx `vp9_setup_tpl_stats` →
/// `mc_flow_dispenser` (`vp9_tpl_model.c:1138-1255`, :1466-1513) over a
/// linear GF group.
#[must_use]
pub fn run_single_gop_tpl(
    window: &[TplWindowFrame<'_>],
    last_recon_y: &[u8],
    width: usize,
    height: usize,
    base_qindex: i32,
    frame_context: &FrameContext,
) -> TplPassResult {
    assert!(!window.is_empty(), "TPL window must have >= 1 frame");
    let mi_cols = width / TPL_BW;
    let mi_rows = height / TPL_BW;

    // ME harness — built once (libvpx hangs these off ThreadData;
    // `init_dsmotion_compensation` builds the 4-site diamond config for
    // the plane stride, `vp9_tpl_model.c:1147` `vp9_setup_scale_factors`
    // + the MB's `sadperbit`/MV-cost setup at :1201-1203).
    let site_config = init_dsmotion_compensation(width as i32);
    let mv_costs = build_nmv_cost_table(&frame_context.nmvc, /* allow_hp = */ false);

    let n = window.len();
    // Per-frame TPL stat grids for the whole window.
    let mut frames: Vec<TplDepFrame> = (0..n).map(|_| TplDepFrame::new(mi_rows, mi_cols)).collect();
    for f in &mut frames {
        f.base_qindex = base_qindex;
    }

    // Forward window / backward propagation: process the last frame
    // first so each frame's dependency flow reaches its predecessor,
    // ultimately accumulating into frame 0 (the frame being coded).
    // libvpx walks `frame_idx` forward in `vp9_setup_tpl_stats` but the
    // dependency it propagates is into earlier (reference) frames; the
    // reverse iteration here produces the same backward accumulation for
    // a linear chain in a single sweep.
    for k in (0..n).rev() {
        // mode_estimation for every block of frame k, motion-searching
        // against frame k-1 (its reference). Frame 0's reference is the
        // committed LAST recon.
        let cur_y = window[k].y;
        let (ref_plane, ref_slot): (Option<PlaneRef<'_>>, i32) = if k == 0 {
            (
                Some(PlaneRef {
                    data: last_recon_y,
                    stride: width,
                    width,
                    height,
                }),
                NO_REF_FRAME_INDEX, // anchor's reference is outside the window
            )
        } else {
            (
                Some(PlaneRef {
                    data: window[k - 1].y,
                    stride: width,
                    width,
                    height,
                }),
                (k - 1) as i32,
            )
        };

        for mi_row in 0..mi_rows {
            for mi_col in 0..mi_cols {
                estimate_block(
                    &mut frames[k],
                    cur_y,
                    ref_plane.as_ref(),
                    width,
                    height,
                    mi_row,
                    mi_col,
                    &site_config,
                    &mv_costs,
                    ref_slot,
                );
                // tpl_model_store recomputes mc_dep_cost = intra + mc_flow
                // for this block (libvpx :1243).
                tpl_model_store(&mut frames[k], mi_row, mi_col, TPL_BSIZE);
            }
        }
        frames[k].is_valid = true;

        // Propagate frame k's dependency into its reference frame k-1's
        // grid (libvpx :1251 `tpl_model_update`). The anchor (k == 0)
        // has no in-window reference, so it only accumulates.
        if k >= 1 {
            // Split the borrow: cur = frames[k] (read), ref = frames[k-1]
            // (write). `split_at_mut` keeps both live.
            let (lo, hi) = frames.split_at_mut(k);
            let cur_frame = &hi[0]; // frames[k]
            let ref_frame = &mut lo[k - 1]; // frames[k-1]
            for mi_row in 0..mi_rows {
                for mi_col in 0..mi_cols {
                    tpl_model_update(cur_frame, ref_frame, mi_row, mi_col, TPL_BSIZE);
                }
            }
        }
    }

    // r0 = Σ intra_cost / Σ mc_dep_cost over the coded frame's grid
    // (libvpx vp9_encodeframe.c:6151-6162).
    let coded = frames.into_iter().next().expect("window non-empty");
    let mut intra_base: i64 = 0;
    let mut mc_dep_base: i64 = 0;
    for s in &coded.tpl_stats {
        intra_base += s.intra_cost;
        mc_dep_base += s.mc_dep_cost;
    }
    let r0 = if mc_dep_base > 0 {
        intra_base as f64 / mc_dep_base as f64
    } else {
        1.0
    };

    TplPassResult {
        coded_frame: coded,
        r0,
    }
}
