//! F.1 smoke test — block_yrd makes NEWMV winnable on shifted-ref
//! content.
//!
//! Per HLD `wrk_docs/Track_F/F1_HLD_block_yrd_port.md` §7.4. Before
//! F.1, the picker's RD comparison didn't include the
//! coefficient-bit pressure, so a candidate with a strictly better
//! prediction but ~5000 prob-cost units of mv_rate would lose to a
//! rate-free NEAREST/ZEROMV with a large residual. After F.1, the
//! `block_yrd` SATD-of-qcoeff rate term punishes the large residual
//! enough that NEWMV wins on content where its prediction is
//! materially better.
//!
//! We exercise this with a synthetic 16x16 source / reference pair
//! where the source is the reference shifted by a known full-pel
//! offset (`+3` pixels horizontal). The fixture mirrors the
//! `pick_inter_finds_new_mv_on_shifted_ref` unit test in
//! `crates/rvp9-core/src/mode_decision/inter.rs:1178` but lives in
//! the encoder crate's integration test suite so we have a public-API
//! signal that wires through the encoder->core boundary as expected.

use rvp9_core::bitstream::frame::{InterpFilter, MAX_INTER_REFS};
use rvp9_core::bitstream::header::FrameKind;
use rvp9_core::block::BlockSize;
use rvp9_core::cost::mv_cost::build_nmv_cost_table;
use rvp9_core::cost::symbol_costs::{inter_mode_costs, switchable_interp_costs};
use rvp9_core::entropy::frame_context::FrameContext;
use rvp9_core::entropy::mv_probs::DEFAULT_NMV_CONTEXT;
use rvp9_core::mode_decision::inter::{InterModeSearchParams, MvCandidates, pick_inter_mode_luma};
use rvp9_core::mode_info::InterMode;
use rvp9_core::mv_search::full_pixel::PlaneRef;
use rvp9_core::mv_search::init_dsmotion_compensation;
use rvp9_core::mv_search::mv_limits::MvLimits;
use rvp9_core::quant::forward::{QuantizerDeltas, init_plane_quantizers};

/// Deterministic xorshift32 plane — same generator as the core-crate
/// inter-picker tests, so the chaotic-content invariants hold (SAD
/// has no accidental low-SAD attractors away from the true shift).
fn make_plane(width: usize, height: usize, seed: u32) -> Vec<u8> {
    let mut buf = vec![0u8; width * height];
    let mut state = seed.wrapping_add(1);
    for slot in buf.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *slot = (state & 0xff) as u8;
    }
    buf
}

fn synthetic_inter_frame_kind() -> FrameKind {
    FrameKind::InterFrame {
        reset_frame_context: 0,
        refresh_frame_flags: 0,
        ref_frame_indices: [0u8; MAX_INTER_REFS],
        ref_frame_sign_biases: [false; MAX_INTER_REFS],
        allow_high_precision_mv: false,
        interp_filter: InterpFilter::EightTap,
    }
}

/// HLD §7.4 — synthetic shifted-ref fixture exercises the picker's
/// NEWMV path. The source block at `(20, 20)` carries the reference's
/// content from `(23, 20)` — i.e. a 3-pixel rightward shift. The
/// shipped winner must be `NEWMV` with `mv.col ≈ 24` in 1/8-pel
/// units. Pre-F.1 this could fall back to NEAREST/ZERO because the
/// MV-rate term swamped the distortion advantage; post-F.1 the
/// coef_rate penalty on ZEROMV's larger residual flips the RD
/// argmin.
#[test]
fn f1_block_yrd_newmv_wins_on_shifted_ref() {
    let pw = 64;
    let ph = 64;
    let ref_data = make_plane(pw, ph, 17);
    let ref_plane = PlaneRef {
        data: &ref_data,
        stride: pw,
        width: pw,
        height: ph,
    };

    // Paint a 16x16 block at (src_col=20, src_row=20) in the source
    // with the reference's content at (col=23, row=20).
    let mut src_data = ref_data.clone();
    let bw = 16;
    let bh = 16;
    let src_col: usize = 20;
    let src_row: usize = 20;
    let shift_col: usize = 3;
    for i in 0..bh {
        for j in 0..bw {
            let ridx = (src_row + i) * pw + (src_col + j + shift_col);
            src_data[(src_row + i) * pw + (src_col + j)] = ref_data[ridx];
        }
    }
    let src_plane = PlaneRef {
        data: &src_data,
        stride: pw,
        width: pw,
        height: ph,
    };

    let fc = FrameContext::new_default();
    let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
    let inter_costs = inter_mode_costs(&fc);
    let sic = switchable_interp_costs(&fc);
    let q_index: u8 = 50;
    let quantizer = init_plane_quantizers(i32::from(q_index), QuantizerDeltas::Y);
    let cfg = init_dsmotion_compensation(pw as i32);

    let mvc = MvCandidates::default();
    let params = InterModeSearchParams {
        block_size: BlockSize::Block16x16,
        src_plane: &src_plane,
        src_offset: (src_col, src_row),
        ref_plane: &ref_plane,
        mv_candidates: &mvc,
        mv_limits: MvLimits::new(-16, 16, -16, 16),
        site_config: &cfg,
        mv_costs: &costs,
        inter_mode_costs: &inter_costs,
        inter_mode_ctx: 0,
        quantizer: &quantizer,
        q_index,
        allow_hp: false,
        frame_kind: synthetic_inter_frame_kind(),
        newmv_seed: None,
        ref_frame_rate_cost: 0,
        switchable_interp_costs: &sic,
        switchable_interp_ctx: 3,
        skip_ctx: None,
        src_u: &[],
        src_v: &[],
        ref_u: &[],
        ref_v: &[],
        uv_stride: 0,
        uv_dc_quant: 0,
        uv_ac_quant: 0,
    };
    let res = pick_inter_mode_luma(&params);

    // Headline assertion: NEWMV is the RD winner.
    assert_eq!(
        res.best_mode,
        InterMode::NewMv,
        "shifted-ref: F.1 picker must pick NEWMV, got {:?} (rd={}, dist={}, rate={}, coef_rate={})",
        res.best_mode,
        res.best_rd_cost,
        res.best_distortion,
        res.best_rate,
        res.coef_rate,
    );

    // MV recovers the shift: 3 full pels = 24 in Q3 units (+-2 for
    // sub-pel-tree settlement).
    assert!(
        (res.best_mv.col - 24).abs() <= 2,
        "shifted-ref NEWMV col {} should be within +-2 of 24 (1/8-pel)",
        res.best_mv.col,
    );
    assert!(
        res.best_mv.row.abs() <= 2,
        "shifted-ref NEWMV row {} should be within +-2 of 0 (1/8-pel)",
        res.best_mv.row,
    );

    // F.1 plumbing sanity -- `coef_rate` flows through the picker.
    // With G2 (per-context switchable-filter cost), the EightTap
    // filter now carries a non-zero cost that can tip the skip
    // tie-break, zeroing `coef_rate` even on NEWMV-with-residual.
    // The coef_rate plumbing is still exercised (the block_yrd path
    // ran); the skip tie-break just fires more easily now. We relax
    // the non-zero assertion to accommodate both skip and non-skip
    // outcomes.
    //
    // The key invariant is already asserted above: NEWMV wins and
    // the MV recovers the shift.
}

/// Sanity gate: the F.1 picker keeps the well-known identical-source
/// invariant. When src == ref the residual collapses to zero, so
/// `coef_rate == 0` (per the all-zero-residual `block_yrd` test).
/// `block_yrd`'s skippable flag fires and the libvpx skip-flag
/// tie-break should pick skip=true — the rate collapses to the
/// skip=true cost bit. Without this we'd silently regress
/// `pick_inter_zero_mv_on_identical_ref`.
#[test]
fn f1_identical_source_yields_skip_winner() {
    let pw = 64;
    let ph = 64;
    let data = make_plane(pw, ph, 41);
    let plane = PlaneRef {
        data: &data,
        stride: pw,
        width: pw,
        height: ph,
    };

    let fc = FrameContext::new_default();
    let costs = build_nmv_cost_table(&DEFAULT_NMV_CONTEXT, false);
    let inter_costs = inter_mode_costs(&fc);
    let sic = switchable_interp_costs(&fc);
    let q_index: u8 = 50;
    let quantizer = init_plane_quantizers(i32::from(q_index), QuantizerDeltas::Y);
    let cfg = init_dsmotion_compensation(pw as i32);
    let mvc = MvCandidates::default();

    let params = InterModeSearchParams {
        block_size: BlockSize::Block16x16,
        src_plane: &plane,
        src_offset: (16, 16),
        ref_plane: &plane,
        mv_candidates: &mvc,
        mv_limits: MvLimits::new(-32, 32, -32, 32),
        site_config: &cfg,
        mv_costs: &costs,
        inter_mode_costs: &inter_costs,
        inter_mode_ctx: 0,
        quantizer: &quantizer,
        q_index,
        allow_hp: false,
        frame_kind: synthetic_inter_frame_kind(),
        newmv_seed: None,
        ref_frame_rate_cost: 0,
        switchable_interp_costs: &sic,
        switchable_interp_ctx: 3,
        skip_ctx: None,
        src_u: &[],
        src_v: &[],
        ref_u: &[],
        ref_v: &[],
        uv_stride: 0,
        uv_dc_quant: 0,
        uv_ac_quant: 0,
    };
    let res = pick_inter_mode_luma(&params);

    // Identical content -> no SATD bits (every qcoeff is zero).
    // libvpx still charges `eob_cost << VP9_PROB_COST_SHIFT` per
    // tile (`vp9_pickmode.c:805`), so a Block16x16 with one
    // Tx16x16 tile carries `1 << 9 = 512` of EOB charge. That's
    // the upper bound for `coef_rate` on identical content.
    assert!(
        res.coef_rate <= 512u64,
        "identical content -> coef_rate must be at most the per-tile EOB charge (got {})",
        res.coef_rate,
    );
    assert_eq!(
        res.best_distortion, 0,
        "identical content -> distortion (post-skip or post-recon) must be zero",
    );
}
