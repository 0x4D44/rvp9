#![allow(clippy::print_stderr)]

//! Direct test: call `pick_inter_mode_luma` with a known shifted source
//! / ref and check what mode/MV it picks and what dist it reports.

use rvp9_core::bitstream::frame::{InterpFilter as WireInterpFilter, MAX_INTER_REFS};
use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
use rvp9_core::block::BlockSize;
use rvp9_core::cost::mv_cost::build_nmv_cost_table;
use rvp9_core::cost::symbol_costs::inter_mode_costs;
use rvp9_core::entropy::FrameContext;
use rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
use rvp9_core::mode_decision::inter::{
    InterModeSearchParams, MvCandidates, pick_inter_mode_luma, pick_inter_mode_luma_all_costs,
};
use rvp9_core::mv_search::PlaneRef;
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::mv_search::init_dsmotion_compensation;
use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

#[test]
fn picker_finds_motion_on_shifted_block() {
    // Build a 64x64 reference plane with a high-contrast pattern.
    const W: usize = 64;
    const H: usize = 64;
    let mut reference = vec![128u8; W * H];
    // Add a step at column 16: left half 50, right half 200.
    for r in 0..H {
        for c in 0..W {
            reference[r * W + c] = if c < W / 2 { 50 } else { 200 };
        }
    }
    // Source = reference shifted by (3 cols, 2 rows). So the step now
    // lives at column 16+3 = 19, and the pattern shifts down 2 rows.
    let mut source = vec![128u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let sr = r.saturating_sub(2);
            let sc = c.saturating_sub(3);
            source[r * W + c] = reference[sr * W + sc];
        }
    }

    let src = PlaneRef { data: &source, stride: W, width: W, height: H };
    let refp = PlaneRef { data: &reference, stride: W, width: W, height: H };

    let inter_costs = inter_mode_costs(&FrameContext::new_default());
    let mv_costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, /* allow_hp = */ false);
    let site_config = init_dsmotion_compensation(W as i32);
    let mvc = MvCandidates::default();
    let quantizer = init_plane_quantizers(100, QuantizerDeltas::Y);

    let frame_kind = WireFrameKind::InterFrame {
        reset_frame_context: 0,
        refresh_frame_flags: 0,
        ref_frame_indices: [0u8; MAX_INTER_REFS],
        ref_frame_sign_biases: [false; MAX_INTER_REFS],
        allow_high_precision_mv: false,
        interp_filter: WireInterpFilter::EightTap,
    };

    // Block at (0,0) — but we need enough border for the search; use a
    // 32x32 block centred so the search can swing in any direction.
    let leaf_px_x = 16usize;
    let leaf_px_y = 16usize;
    let bw = 32usize;
    let bh = 32usize;
    let col_min = -(leaf_px_x as i32) + 3;
    let col_max = (W as i32) - (leaf_px_x as i32) - (bw as i32) - 4;
    let row_min = -(leaf_px_y as i32) + 3;
    let row_max = (H as i32) - (leaf_px_y as i32) - (bh as i32) - 4;
    let mv_limits = MvLimits::new(col_min, col_max, row_min, row_max);

    let params = InterModeSearchParams {
        block_size: BlockSize::Block32x32,
        src_plane: &src,
        src_offset: (leaf_px_x, leaf_px_y),
        ref_plane: &refp,
        mv_candidates: &mvc,
        mv_limits,
        site_config: &site_config,
        mv_costs: &mv_costs,
        inter_mode_costs: &inter_costs,
        inter_mode_ctx: 0,
        quantizer: &quantizer,
        q_index: 100,
        allow_hp: false,
        frame_kind,
        newmv_seed: None,
        ref_frame_rate_cost: 0,
    };

    let all = pick_inter_mode_luma_all_costs(&params);
    eprintln!("All candidates:");
    for (i, c) in all.iter().enumerate() {
        eprintln!("  cand {i}: mode={:?} mv=({},{}) rd={} dist={} rate={}",
            c.best_mode, c.best_mv.row, c.best_mv.col, c.best_rd_cost, c.best_distortion, c.best_rate);
    }
    let pick = pick_inter_mode_luma(&params);
    eprintln!("Winner: mode={:?} mv=({},{}) rd={} dist={} rate={}",
        pick.best_mode, pick.best_mv.row, pick.best_mv.col, pick.best_rd_cost, pick.best_distortion, pick.best_rate);

    // We expect NEWMV with MV close to (16, 24) Q3 (== 2 row, 3 col).
}
