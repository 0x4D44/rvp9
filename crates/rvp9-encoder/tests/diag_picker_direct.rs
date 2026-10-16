#![allow(clippy::print_stderr)]

//! Direct test: call `pick_inter_mode_luma` with a known shifted source
//! / ref and check what mode/MV it picks and what dist it reports.

use rvp9_core::bitstream::frame::{InterpFilter as WireInterpFilter, MAX_INTER_REFS};
use rvp9_core::bitstream::header::FrameKind as WireFrameKind;
use rvp9_core::block::BlockSize;
use rvp9_core::cost::mv_cost::build_nmv_cost_table;
use rvp9_core::cost::symbol_costs::{inter_mode_costs, switchable_interp_costs};
use rvp9_core::entropy::FrameContext;
use rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
use rvp9_core::mode_decision::inter::{
    InterModeSearchParams, InterRdCalibration, MvCandidates, pick_inter_mode_luma,
    pick_inter_mode_luma_all_costs,
};
use rvp9_core::mode_info::InterMode;
use rvp9_core::mv_search::PlaneRef;
use rvp9_core::mv_search::init_dsmotion_compensation;
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

#[test]
fn picker_finds_motion_on_shifted_block() {
    // Two perpendicular steps make both motion components observable.
    const W: usize = 64;
    const H: usize = 64;
    let mut reference = vec![0u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let row_step = if r >= H / 2 { 120 } else { 0 };
            let column_step = if c >= W / 2 { 80 } else { 0 };
            reference[r * W + c] = 20 + row_step + column_step;
        }
    }

    // Each source sample comes from two rows up and three columns left.
    // The centered source block therefore matches the reference at MV
    // (-2, -3) pixels, or (-16, -24) in VP9's 1/8-pel units.
    let mut source = vec![128u8; W * H];
    for r in 0..H {
        for c in 0..W {
            let sr = r.saturating_sub(2);
            let sc = c.saturating_sub(3);
            source[r * W + c] = reference[sr * W + sc];
        }
    }

    let src = PlaneRef {
        data: &source,
        stride: W,
        width: W,
        height: H,
    };
    let refp = PlaneRef {
        data: &reference,
        stride: W,
        width: W,
        height: H,
    };

    let fc = FrameContext::new_default();
    let inter_costs = inter_mode_costs(&fc);
    let sic = switchable_interp_costs(&fc);
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

    // Center the 32x32 block so both expected negative offsets are valid.
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
        full_pixel_search_method: Default::default(),
        mv_costs: &mv_costs,
        inter_mode_costs: &inter_costs,
        inter_mode_ctx: 0,
        quantizer: &quantizer,
        q_index: 100,
        allow_hp: false,
        frame_kind,
        newmv_seed: None,
        svc_non_reference_frame: false,
        ref_frame_rate_cost: 0,
        switchable_interp_costs: &sic,
        switchable_interp_ctx: 3,
        skip_ctx: None,
        skip_prob: None,
        rd_calibration: InterRdCalibration::default(),
        src_u: &[],
        src_v: &[],
        ref_u: &[],
        ref_v: &[],
        uv_stride: 0,
        color_sensitivity: [false; 2],
        uv_dc_quant: 0,
        uv_ac_quant: 0,
    };

    let all = pick_inter_mode_luma_all_costs(&params);
    eprintln!("All candidates:");
    for (i, c) in all.iter().enumerate() {
        eprintln!(
            "  cand {i}: mode={:?} mv=({},{}) rd={} dist={} rate={}",
            c.best_mode,
            c.best_mv.row,
            c.best_mv.col,
            c.best_rd_cost,
            c.best_distortion,
            c.best_rate
        );
    }

    // Independently confirm that the source block has a zero raw-SAD
    // match at the expected reference origin (row 14, column 13).
    let mut expected_raw_sad = 0u32;
    for r in 0..bh {
        for c in 0..bw {
            let src_sample = source[(leaf_px_y + r) * W + leaf_px_x + c];
            let ref_sample = reference[(leaf_px_y + r - 2) * W + leaf_px_x + c - 3];
            expected_raw_sad += u32::from(src_sample.abs_diff(ref_sample));
        }
    }
    assert_eq!(
        expected_raw_sad, 0,
        "fixture's expected offset must be exact"
    );

    let new = &all[3];
    assert_eq!(new.best_mode, InterMode::NewMv, "candidate 3 is NEWMV");
    assert_eq!(
        (new.best_mv.row, new.best_mv.col),
        (-16, -24),
        "source samples reference at row -2, column -3, in Q3 units",
    );
    assert_eq!(
        new.best_distortion, 0,
        "exact prediction must have zero distortion"
    );

    for (index, zero_motion) in all[..3].iter().enumerate() {
        assert_eq!(
            (zero_motion.best_mv.row, zero_motion.best_mv.col),
            (0, 0),
            "candidate {index} should be a zero-motion competitor",
        );
        assert!(
            new.best_rd_cost < zero_motion.best_rd_cost,
            "NEWMV RD cost {} must beat zero-motion candidate {index} at {}",
            new.best_rd_cost,
            zero_motion.best_rd_cost,
        );
    }

    let pick = pick_inter_mode_luma(&params);
    eprintln!(
        "Winner: mode={:?} mv=({},{}) rd={} dist={} rate={}",
        pick.best_mode,
        pick.best_mv.row,
        pick.best_mv.col,
        pick.best_rd_cost,
        pick.best_distortion,
        pick.best_rate
    );

    assert_eq!(
        pick.best_mode,
        InterMode::NewMv,
        "picker should choose NEWMV"
    );
    assert_eq!((pick.best_mv.row, pick.best_mv.col), (-16, -24));
    assert_eq!(pick.best_distortion, 0);
    assert_eq!(pick.best_rd_cost, new.best_rd_cost);
    assert_eq!(
        pick, *new,
        "winner and per-candidate entrypoints should agree"
    );
}
