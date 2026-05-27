//! Wave 6.7 + 6.8 — TPL (Temporal-dependency model) scaffold.
//!
//! Faithful port of libvpx's `vp9/encoder/vp9_tpl_model.{c,h}` and the
//! TPL-aware bits of `vp9_encodeframe.c` (`get_rdmult_delta` + RDO
//! consumer hook), v1.14.1 commit `12f3a2ac6`.
//!
//! The TPL machinery propagates a backward `mc_dep_cost` from leaf
//! frames up to the GOP's golden anchor so the RDO picker can scale
//! `rdmult` per-superblock by how much downstream distortion depends
//! on the current block being coded faithfully.
//!
//! ## libvpx mapping
//!
//! | rvp9 sub-module | libvpx source                                       |
//! |-----------------|-----------------------------------------------------|
//! | [`stats`]       | `vp9_encoder.h:292-326` (struct defs) + `vp9_tpl_model.c:25-166` (init_gop_frames + init_tpl_stats) |
//! | [`mode_est`]    | `vp9_tpl_model.c:571-727` (mode_estimation)         |
//! | [`update`]      | `vp9_tpl_model.c:372-489` (tpl_model_store + tpl_model_update) |
//! | [`rdmult`]      | `vp9_encodeframe.c:3717-3768` (get_rdmult_delta) + `vp9_rd.c:304-310` (vp9_get_adaptive_rdmult) + `vp9_encodeframe.c:2235-2242` (RDO cb_rdmult hook) |
//!
//! ## Scope (W6.7 + W6.8 scaffold)
//!
//! - **Standalone primitives.** No encoder integration yet — the
//!   `mc_flow_dispenser` driver that ties everything together (libvpx
//!   `vp9_tpl_model.c:1019-1226`) lands in W6.9.
//! - **8-bit only.** HBD branches are out of scope.
//! - **Externally-supplied MVs.** libvpx's `mode_estimation` runs a
//!   per-ref-frame motion search via
//!   `motion_compensated_prediction`; the W6.7 scaffold takes MVs
//!   from the caller and just computes the SATD given a precomputed
//!   inter-predictor patch. Hooking the MV search into
//!   `rdo::single_mv_search` is **TODO(W6.9)**.
//! - **No quant roundtrip.** libvpx writes `recon_error` / `rate_cost`
//!   out-params from `get_quantize_error` + `rate_estimator`. Our
//!   scaffold leaves those as zero (the propagator doesn't read
//!   them). **TODO(W6.9)**.
//! - **No `tpl_store_before_propagation` (Qmode telemetry).**
//!   `VpxTplGopStats` is a Qmode-only sidecar; rvp9 doesn't expose
//!   Qmode yet.
//!
//! ## PROOF tests
//!
//! Live in [`tests`](#) at the bottom of this file rather than in a
//! sibling crate, so they share the private internals of the
//! sub-modules.

pub mod driver;
pub mod mode_est;
pub mod rdmult;
pub mod stats;
pub mod update;

pub use driver::{TplPassResult, TplWindowFrame, run_single_gop_tpl};
pub use mode_est::{IntraTplMode, ModeEstInputs, ModeEstResult, mode_estimation};

/// A23 anti-G3 diagnostic — count of inter-leaf RDO dispatches that
/// received a TPL-adjusted rdmult DIFFERENT from the base rdmult (i.e.
/// `get_rdmult_delta` returned `used_tpl == true` AND `cb_rdmult !=
/// base_rdmult`). Bumped at the `cb_rdmult` consumer hook in
/// `lib.rs::pick_and_simulate_inter_encode_partitions` only when
/// `enable_tpl` is on. With `enable_tpl` off it never advances (the
/// hook substitution is gated). The PROOF test
/// (`tests/w_tpl.rs`) asserts it is `> 0` on content with motion when
/// TPL is enabled and `== 0` when disabled — a stubbed dispenser (all
/// `mc_dep_cost == intra_cost` → `beta == 1` → no delta) would leave it
/// at 0, failing the test.
pub static TPL_RDMULT_ADJUSTED_BLOCKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub use rdmult::{RdmultDelta, get_rdmult_delta, rdo_apply_cb_rdmult, vp9_get_adaptive_rdmult};
pub use stats::{
    GfPicture, MAX_ARF_GOP_SIZE, MAX_ARF_LAYERS, MAX_INTER_REF_FRAMES, MAX_LAG_BUFFERS, MI_SIZE,
    MI_SIZE_LOG2, NO_REF_FRAME_INDEX, REFS_PER_FRAME, TPL_DEP_COST_SCALE_LOG2, TplDepFrame,
    TplDepStats, TplGopState, init_gop_frames, update_type,
};
pub use update::{tpl_model_store, tpl_model_update};

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::block::BlockSize;
    use rvp9_core::mode_info::Mv;

    /// libvpx `init_gop_frames` (`vp9_tpl_model.c:25-155`) +
    /// `vp9_setup_tpl_stats` allocation (`vp9_tpl_model.c:1492-1507`):
    /// the TPL state pre-allocates per-frame `TplDepStats` arrays
    /// sized to the frame's mi grid for every GOP slot it intends to
    /// model. Verifies the rvp9 [`TplGopState::new`] /
    /// [`init_gop_frames`] pair sizes and chains correctly.
    #[test]
    fn w6_7_tpl_dep_frame_init_gop_frames_sizes_correctly() {
        // 1080p in mi-units: 1920/8 = 240 cols, 1088/8 = 136 rows.
        // We use a smaller grid for test speed.
        const MI_ROWS: usize = 9;
        const MI_COLS: usize = 16;
        const GOP_FRAMES: usize = 8;

        let mut state = TplGopState::new(GOP_FRAMES, MI_ROWS, MI_COLS);
        assert_eq!(state.frames.len(), GOP_FRAMES);
        for f in &state.frames {
            assert_eq!(f.mi_rows, MI_ROWS);
            assert_eq!(f.mi_cols, MI_COLS);
            assert_eq!(f.stride, MI_COLS);
            assert_eq!(f.tpl_stats.len(), MI_ROWS * MI_COLS);
            assert!(!f.is_valid);
        }

        let last_lst = init_gop_frames(&mut state, GOP_FRAMES);
        assert_eq!(state.tpl_group_frames, GOP_FRAMES);

        // Slot 0 = GF anchor, no refs (libvpx vp9_tpl_model.c:73-76).
        assert_eq!(
            state.gf_picture[0].ref_frame,
            [NO_REF_FRAME_INDEX; MAX_INTER_REF_FRAMES]
        );
        assert_eq!(state.gf_picture[0].update_type, update_type::GF_UPDATE);

        // Slot 1 = base ARF, refs GLD via slot 0 (libvpx 80-86).
        assert_eq!(state.gf_picture[1].ref_frame[0], 0);
        assert_eq!(state.gf_picture[1].update_type, update_type::ARF_UPDATE);

        // Slot 7 = last LF frame chains through slot 6 (libvpx 108).
        assert_eq!(state.gf_picture[GOP_FRAMES - 1].update_type, update_type::LF_UPDATE);
        assert_eq!(last_lst, (GOP_FRAMES - 1) as i32);
    }

    /// libvpx `init_tpl_stats` (`vp9_tpl_model.c:157-166`) bulk-zeroes
    /// every per-block stat slot. Verify [`TplGopState::new`] starts
    /// us in that state and [`TplDepFrame::reset`] restores it.
    #[test]
    fn w6_7_tpl_dep_stats_zero_at_init() {
        let mut state = TplGopState::new(4, 4, 4);
        for f in &state.frames {
            for s in &f.tpl_stats {
                assert_eq!(*s, TplDepStats::zero());
                assert_eq!(s.ref_frame_index, NO_REF_FRAME_INDEX);
            }
        }

        // Poison one frame, reset, verify clean.
        let s = state.frames[0].stat_mut(0, 0);
        s.intra_cost = 12345;
        s.mc_flow = -99;
        s.mv = Mv { row: 7, col: -3 };
        state.frames[0].is_valid = true;

        state.reset_all_stats();
        for f in &state.frames {
            assert!(!f.is_valid);
            for s in &f.tpl_stats {
                assert_eq!(*s, TplDepStats::zero());
            }
        }
    }

    /// libvpx `mode_estimation` (`vp9_tpl_model.c:571-727`): when no
    /// inter ref is supplied, only the intra pass runs and the
    /// resulting `intra_cost` is the SATD of the best-of-4 intra
    /// modes. Verify the scaled value matches the post-`VPXMAX(.., 1)`
    /// + cost-scale formula at libvpx lines 718-723.
    #[test]
    fn w6_7_mode_estimation_produces_intra_cost_baseline() {
        // 8×8 mi grid covering one 64-pixel block.
        let mut frame = TplDepFrame::new(2, 2);
        let bsize = BlockSize::Block8x8;
        let bw = 8;

        // Synthetic source: a sharp vertical edge — V_PRED should
        // win cleanly.
        let mut cur_block = vec![0u8; bw * bw];
        for r in 0..bw {
            for c in 0..bw {
                cur_block[r * bw + c] = if c < bw / 2 { 16 } else { 200 };
            }
        }
        // libvpx-style: corner + 8 above pixels.
        let mut above_with_corner = vec![16u8; bw + 1];
        for (c, slot) in above_with_corner[1..].iter_mut().enumerate() {
            *slot = if c < bw / 2 { 16 } else { 200 };
        }
        let left = vec![16u8; bw];

        let inputs = ModeEstInputs {
            bsize,
            cur_block: &cur_block,
            above_with_corner: &above_with_corner,
            left: &left,
            has_above: true,
            has_left: true,
            inter_pred: [None, None, None],
            inter_mvs: [Mv::ZERO; MAX_INTER_REF_FRAMES],
            ref_frame_slot: [NO_REF_FRAME_INDEX; MAX_INTER_REF_FRAMES],
        };

        let result = mode_estimation(&mut frame, 0, 0, &inputs);

        // Intra-only: best_rf_idx should be -1, and the stat's
        // intra/inter both equal scaled(best_intra_cost), with
        // ref_frame_index = NO_REF_FRAME_INDEX.
        assert_eq!(result.best_rf_idx, -1);
        assert_eq!(result.best_ref_frame_index, NO_REF_FRAME_INDEX);

        let stat = frame.stat(0, 0);
        // Both intra and inter should equal the scaled intra cost
        // (libvpx VPXMIN(best_intra_cost, best_inter_cost) when
        // best_inter_cost was MAX in our scaffold falls back to intra).
        assert!(stat.intra_cost >= 1, "intra_cost must be >= 1");
        assert!(stat.inter_cost >= 1, "inter_cost must be >= 1");
        assert_eq!(
            stat.intra_cost, stat.inter_cost,
            "intra-only frame: inter_cost falls back to intra"
        );
        assert_eq!(stat.ref_frame_index, NO_REF_FRAME_INDEX);

        // V_PRED predicts perfectly (above row matches the source
        // top row), so best_intra_cost (post-VPXMAX(.., 1)) should be 1.
        // Then scaled = (1 << 4) / (1 * 1) = 16.
        assert_eq!(stat.intra_cost, 16, "V_PRED perfect predict, expected SATD=0 → scaled to 16");
    }

    /// libvpx `tpl_model_update` (`vp9_tpl_model.c:475-489`) +
    /// `tpl_model_update_b` (`vp9_tpl_model.c:419-473`): propagating
    /// a populated current-frame stat into a future frame's
    /// mc_dep_cost grid increases the reference cell's `mc_flow`
    /// (and `mc_ref_cost`) by the integer-overlap weighted
    /// contribution.
    #[test]
    fn w6_8_tpl_model_update_propagates_to_future_frame() {
        // 2×2 mi grid. The current-frame block at (0,0) carries a
        // zero MV → ref_pos_row/col = 0, grid_pos_base = (0, 0),
        // so block 0 overlaps cell (0, 0) of the reference frame
        // fully (overlap_area == pix_num == 64), block 1/2/3 fall
        // off the edge.
        let mut cur_frame = TplDepFrame::new(2, 2);
        let mut ref_frame = TplDepFrame::new(2, 2);

        // Populate the cur stat:
        //   intra_cost = 100, inter_cost = 50, mc_dep_cost = 100,
        //   mc_flow = 0 (typical post-mode_estimation, pre-propagation
        //   state), zero MV. We expect:
        //   mc_flow_contrib = 100 - (100 * 50) / 100 = 100 - 50 = 50.
        //   overlap_area / pix_num = 1.
        //   → ref.mc_flow += 50, ref.mc_ref_cost += (100 - 50) * 1 = 50.
        let cur = cur_frame.stat_mut(0, 0);
        cur.intra_cost = 100;
        cur.inter_cost = 50;
        cur.mc_dep_cost = 100;
        cur.mc_flow = 0;
        cur.mv = Mv::ZERO;

        let ref_before = *ref_frame.stat(0, 0);
        assert_eq!(ref_before.mc_flow, 0);
        assert_eq!(ref_before.mc_ref_cost, 0);

        tpl_model_update(&cur_frame, &mut ref_frame, 0, 0, BlockSize::Block8x8);

        let ref_after = ref_frame.stat(0, 0);
        assert_eq!(ref_after.mc_flow, 50, "expected mc_flow += 50");
        assert_eq!(ref_after.mc_ref_cost, 50, "expected mc_ref_cost += 50");

        // Other cells untouched.
        let other = ref_frame.stat(1, 1);
        assert_eq!(other.mc_flow, 0);
        assert_eq!(other.mc_ref_cost, 0);
    }

    /// libvpx `get_rdmult_delta` (`vp9_encodeframe.c:3719-3768`) +
    /// `vp9_get_adaptive_rdmult` (`vp9_rd.c:304-310`): when the TPL
    /// stat indicates a high-dependency block (large mc_dep_cost vs
    /// intra_cost), `beta = r0 / rk` shrinks and the rdmult is
    /// scaled down. The result must remain ≥ 1 (libvpx
    /// `dr = VPXMAX(1, dr)` at line 3765) and bounded below by
    /// `orig_rdmult / 2`.
    #[test]
    fn w6_8_get_rdmult_delta_returns_nonneg_lambda_scale() {
        // 8×8 SB at (0,0) covers an 8×8 mi region. Populate it with
        // intra_cost = 100, mc_dep_cost = 800 → rk = 100/800 = 0.125.
        // With r0 = 0.5 → beta = 4.0 → dr = base / 4.
        let mut tpl_frame = TplDepFrame::new(8, 8);
        tpl_frame.is_valid = true;
        for r in 0..8 {
            for c in 0..8 {
                let s = tpl_frame.stat_mut(r, c);
                s.intra_cost = 100;
                s.mc_dep_cost = 800;
            }
        }

        let orig_rdmult = 200;
        let delta = get_rdmult_delta(
            &tpl_frame,
            0,
            0,
            BlockSize::Block64x64,
            orig_rdmult,
            0.5,
            8,
            8,
            0,
            0,
        );
        assert!(delta.used_tpl, "TPL gating predicates all satisfied");
        assert!(delta.cb_rdmult >= 1, "rdmult must always be ≥ 1");
        // libvpx cap: dr ∈ [orig/2, orig*3/2].
        assert!(
            delta.cb_rdmult >= orig_rdmult / 2,
            "expected dr >= orig_rdmult/2 = {}, got {}",
            orig_rdmult / 2,
            delta.cb_rdmult,
        );
        assert!(
            delta.cb_rdmult <= orig_rdmult * 3 / 2,
            "expected dr <= orig_rdmult*3/2 = {}, got {}",
            orig_rdmult * 3 / 2,
            delta.cb_rdmult,
        );
        // High-dep block: dr should be SMALLER than orig (the picker
        // codes it more carefully). Without the cap this would be
        // orig / beta = 200 / 4 = 50, then capped UP to orig/2 = 100.
        assert_eq!(delta.cb_rdmult, 100, "high-dep block clamps to orig/2");
    }

    /// libvpx `tpl_model_store` (`vp9_tpl_model.c:372-390`):
    /// broadcasting an anchor stat to every 8×8 sub-mi of a 16×16
    /// block preserves the destination cells' pre-existing `mc_flow`
    /// and `mc_ref_cost` accumulators and recomputes
    /// `mc_dep_cost = intra_cost + mc_flow`.
    #[test]
    fn w6_8_tpl_model_store_preserves_accumulators_and_broadcasts() {
        let mut frame = TplDepFrame::new(2, 2);
        // Pre-populate destination (1, 1) with a non-zero
        // accumulator (downstream propagation already wrote it).
        let prev_flow = 777_i64;
        let prev_ref_cost = 333_i64;
        {
            let s = frame.stat_mut(1, 1);
            s.mc_flow = prev_flow;
            s.mc_ref_cost = prev_ref_cost;
        }
        // Anchor at (0, 0): intra_cost=50, inter_cost=20, mv non-zero.
        {
            let s = frame.stat_mut(0, 0);
            s.intra_cost = 50;
            s.inter_cost = 20;
            s.mv = Mv { row: 8, col: -16 };
            s.ref_frame_index = 1;
        }

        tpl_model_store(&mut frame, 0, 0, BlockSize::Block16x16);

        // Every cell now sees intra_cost=50, inter_cost=20, mv, ref_idx
        // from the anchor; (1, 1)'s accumulators survived.
        for (r, c) in [(0usize, 0), (0, 1), (1, 0), (1, 1)] {
            let s = frame.stat(r, c);
            assert_eq!(s.intra_cost, 50, "broadcast intra_cost at ({r}, {c})");
            assert_eq!(s.inter_cost, 20, "broadcast inter_cost at ({r}, {c})");
            assert_eq!(s.mv, Mv { row: 8, col: -16 });
            assert_eq!(s.ref_frame_index, 1);
        }
        let preserved = frame.stat(1, 1);
        assert_eq!(preserved.mc_flow, prev_flow, "mc_flow preserved");
        assert_eq!(
            preserved.mc_ref_cost, prev_ref_cost,
            "mc_ref_cost preserved"
        );
        // mc_dep_cost = intra_cost + mc_flow
        assert_eq!(
            preserved.mc_dep_cost,
            preserved.intra_cost + preserved.mc_flow,
            "mc_dep_cost recomputed",
        );
        // For untouched (0, 0): mc_flow stays 0 → mc_dep_cost = intra_cost.
        assert_eq!(frame.stat(0, 0).mc_dep_cost, 50);
    }

    /// libvpx `get_rdmult_delta` gating (`vp9_encodeframe.c:3735-3742`):
    /// when the TPL frame is invalid, or the layer depth exceeds 1,
    /// or `gf_group_index` is out of range, the function returns
    /// `orig_rdmult` unchanged and signals `used_tpl == false`.
    #[test]
    fn w6_8_get_rdmult_delta_gates_when_tpl_invalid() {
        let tpl_frame = TplDepFrame::new(8, 8); // is_valid == false
        let r = get_rdmult_delta(
            &tpl_frame,
            0,
            0,
            BlockSize::Block64x64,
            321,
            0.5,
            8,
            8,
            0,
            0,
        );
        assert!(!r.used_tpl);
        assert_eq!(r.cb_rdmult, 321);

        // valid but layer_depth > 1.
        let mut tpl_frame = TplDepFrame::new(8, 8);
        tpl_frame.is_valid = true;
        let r = get_rdmult_delta(
            &tpl_frame,
            0,
            0,
            BlockSize::Block64x64,
            321,
            0.5,
            8,
            8,
            0,
            2,
        );
        assert!(!r.used_tpl);
        assert_eq!(r.cb_rdmult, 321);

        // gf_group_index >= MAX_ARF_GOP_SIZE.
        let r = get_rdmult_delta(
            &tpl_frame,
            0,
            0,
            BlockSize::Block64x64,
            321,
            0.5,
            8,
            8,
            MAX_ARF_GOP_SIZE,
            0,
        );
        assert!(!r.used_tpl);
        assert_eq!(r.cb_rdmult, 321);
    }

    /// libvpx RDO consumer hook (`vp9_encodeframe.c:2235-2242`):
    /// when `enable_tpl_model && (aq_mode ∈ {NO_AQ, PERCEPTUAL_AQ})`
    /// the picker reads `x->cb_rdmult` into `x->rdmult`; otherwise
    /// it stays on the base rdmult.
    #[test]
    fn w6_8_rdo_cb_rdmult_hook_dispatches_correctly() {
        assert_eq!(rdo_apply_cb_rdmult(100, 50, true, true), 50);
        assert_eq!(rdo_apply_cb_rdmult(100, 50, false, true), 100);
        assert_eq!(rdo_apply_cb_rdmult(100, 50, true, false), 100);
        assert_eq!(rdo_apply_cb_rdmult(100, 50, false, false), 100);
        // Always clamp to ≥ 1.
        assert_eq!(rdo_apply_cb_rdmult(0, 0, true, true), 1);
    }
}
