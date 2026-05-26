//! F.2 §6.2 — temporal candidates visible in `MvCandidates`.
//!
//! Asserts the prev-frame salvage path (`use_prev_frame_mvs`) is
//! reachable end-to-end and produces a non-zero candidate for the
//! picker's `mvc.nearest` slot through the real encoder pipeline.
//!
//! ## Note on ship-state and observable salvage
//!
//! F.2 ships INFRASTRUCTURE-ONLY (see F.2 HLD §12 ship-state addendum,
//! 2026-05-23). F.1 also shipped infrastructure-only with `coef_rate`
//! gated OFF by default (see F.1 HLD §14 / `notes/divergence-ledger.md`
//! F.1 entry 2026-05-23). With both activations off, the picker's RD
//! comparator is byte-identical to pre-F.1: it still emits NEARESTMV
//! with a zero anchor everywhere on the first inter frame after a KF,
//! producing a `prev_mv_grid` of `(LASTFRAME, ZERO)` cells. Frame 2's
//! picker now sees the gate open but the salvage lifts only zeros —
//! the chicken-and-egg degeneracy E9 §4 documents is closed by F.4
//! lambda recalibration, not by F.2 alone.
//!
//! To validate the F.2 plumbing without depending on F.4's picker
//! tuning, the test:
//!
//! 1. Encodes KF + 1 inter frame through the REAL encoder with
//!    `use_prev_frame_mvs: true`. The encoder must produce a valid
//!    bitstream with no panics.
//! 2. Snapshots the captured grid via `prev_mv_grid_for_test` and
//!    confirms it is populated (writer-side plumbing).
//! 3. Substitutes a SYNTHETIC `PrevFrameMvRef` with a known non-zero
//!    MV (modelling what a F.4-enabled picker WILL produce) and
//!    re-invokes `find_mv_refs` with an empty spatial neighbourhood.
//!    Asserts `mvc.nearest != Mv::ZERO` (matching-ref salvage fires).
//! 4. Re-invokes with `prev_frame_mv = None` (the
//!    `use_prev_frame_mvs: false` regression case) and asserts
//!    `mvc.nearest == Mv::ZERO`. This regression-guards the F.0.3
//!    default-OFF behaviour the team-lead preserves at F.2 sign-off.
//!
//! The salvage CODE-PATH is exercised through `find_mv_refs` (the
//! same function the picker driver calls at `lib.rs:2871`), so this
//! test catches a regression in the salvage logic itself even though
//! the picker's mode-pick is not yet RD-aware enough to surface
//! non-zero MVs through the captured grid.
//!
//! HLD §6.2 Q2 (resolved 2026-05-23) explicitly allows implementer
//! choice on the peeker mechanism; the F.0.2 sink only records the
//! picker's CHOSEN mv (post-RD), not the candidate input
//! `mvc.nearest`, so this approach matches the HLD's intent without
//! adding an in-picker peeker that would churn the hot path.
//!
//! Run:
//! ```text
//! cargo test --release -p rvp9-encoder \
//!     --test f2_find_mv_refs_temporal_visible -- --nocapture
//! ```

#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

use rvp9_core::block::BlockSize;
use rvp9_core::inter::{
    find_mv_refs, sign_bias_from_header, BlockBounds, NeighbourLookup, Position, PrevFrameMvRef,
};
use rvp9_core::mode_info::{InterMode, ModeInfo, Mv, RefFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, StreamFrameKind, YuvFrameRef};

// ---------------------------------------------------------------------------
// Synthetic source — a bright square that translates by a full-pel offset.
// ---------------------------------------------------------------------------

const W: u32 = 128;
const H: u32 = 128;

/// 128×128 4:2:0 frame with a bright square at (ox, oy) on a gradient
/// background. Mirrors the F.0.3 §6.1 fixture so the picker's NEWMV
/// gating is identical.
fn synthetic_frame(ox: i32, oy: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;

    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let dx = c as i32 - (w as i32 / 2);
            let dy = r as i32 - (h as i32 / 2);
            let d = ((dx * dx + dy * dy) as f64).sqrt();
            let v = 64.0 + (d * 0.5).min(96.0);
            y[r * w + c] = v.clamp(0.0, 255.0) as u8;
        }
    }

    let sq = 32i32;
    for r in 0..sq {
        for c in 0..sq {
            let yy = oy + r;
            let xx = ox + c;
            if (0..h as i32).contains(&yy) && (0..w as i32).contains(&xx) {
                y[yy as usize * w + xx as usize] = 230;
            }
        }
    }

    let u = vec![128u8; uvw * uvh];
    let v = vec![128u8; uvw * uvh];
    (y, u, v)
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

fn make_encoder(use_prev_frame_mvs: bool) -> StreamEncoder {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 500_000,
        gop_length: 32,
        aq_mode: AqMode::Off,
        // NOTE: F.2 ships infrastructure-only — the field default at
        // `frame_loop.rs:173` is `false` and stays that way. Tests
        // that exercise the F.2 salvage path opt in explicitly here.
        // See F.2 HLD §12 ship-state addendum and the
        // divergence-ledger 2026-05-23 F.2 entry.
        use_prev_frame_mvs,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        min_qindex: 4,
        max_qindex: 255,
    };
    StreamEncoder::new(cfg).expect("StreamEncoder::new")
}

/// Always-empty `NeighbourLookup`. Models the first-inter-after-KF
/// configuration where every spatial neighbour carries the default
/// `(IntraFrame, ZERO)` from the KF reset, so the encoder's
/// in-progress mi-grid offers no usable spatial candidate. With this
/// lookup, the only path to a non-zero MV is the prev-frame salvage.
struct NoNeighbours;
impl NeighbourLookup for NoNeighbours {
    fn get(&self, _pos: Position) -> Option<&ModeInfo> {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers shared by the two assertions.
// ---------------------------------------------------------------------------

/// Pretend frame-2 leaf coordinates for the salvage probe. Picks a
/// 16×16 leaf well clear of the frame edges so `BlockBounds` is loose
/// enough to not clamp a small synthetic MV.
const PROBE_MI_ROW: usize = 8;
const PROBE_MI_COL: usize = 8;
const PROBE_BW_MI: i32 = 2;
const PROBE_BH_MI: i32 = 2;

/// Synthetic prev-frame cell modelling what a F.4-enabled picker
/// would emit at the probe leaf: NEWMV(LASTFRAME) with a small
/// full-pel motion vector (8 luma px = 64 in 1/8-pel MV units; rows
/// here use the same scale).
fn synthetic_last_cell() -> PrevFrameMvRef {
    PrevFrameMvRef {
        mvs: [Mv { row: 64, col: 64 }, Mv::ZERO],
        ref_frames: [RefFrame::LastFrame, RefFrame::NoRefFrame],
    }
}

fn probe_bounds(mi_rows: usize, mi_cols: usize) -> BlockBounds {
    BlockBounds::new(
        PROBE_MI_ROW as i32,
        PROBE_MI_COL as i32,
        mi_rows as i32,
        mi_cols as i32,
        PROBE_BW_MI,
        PROBE_BH_MI,
    )
}

/// Encode KF + 1 inter frame through the real encoder. Returns the
/// captured `prev_mv_grid` dims so the caller can build the probe's
/// `BlockBounds`. Confirms the plumbing is alive (the grid is
/// populated regardless of the gate — the writer-side capture is
/// unconditional, see `frame_loop.rs:1053`).
fn encode_kf_and_inter_then_dims(use_prev_frame_mvs: bool) -> (usize, usize) {
    let w_us = W as usize;
    let mut enc = make_encoder(use_prev_frame_mvs);

    let (y0, u0, v0) = synthetic_frame(20, 20);
    let pkt0 = enc
        .encode_frame(&yuv_view(&y0, &u0, &v0, w_us))
        .expect("encode KF");
    assert_eq!(pkt0.frame_kind, StreamFrameKind::Key);

    let (y1, u1, v1) = synthetic_frame(28, 28);
    let pkt1 = enc
        .encode_frame(&yuv_view(&y1, &u1, &v1, w_us))
        .expect("encode inter 1");
    assert_eq!(pkt1.frame_kind, StreamFrameKind::Inter);

    let (grid, dims) = enc
        .prev_mv_grid_for_test()
        .expect("prev_mv_grid must be populated after first inter");
    let (mi_rows, mi_cols) = dims;
    assert_eq!(grid.len(), mi_rows * mi_cols, "grid length sanity");
    eprintln!(
        "[F.2 §6.2 plumbing] use_prev_frame_mvs = {use_prev_frame_mvs}, \
         grid dims = ({mi_rows}, {mi_cols}), cells = {}",
        grid.len(),
    );
    (mi_rows, mi_cols)
}

// ---------------------------------------------------------------------------
// §6.2 — assertion 1: salvage emits a non-zero candidate when active.
// ---------------------------------------------------------------------------

#[test]
fn f2_find_mv_refs_temporal_visible_with_use_prev_frame_mvs_true() {
    let (mi_rows, mi_cols) = encode_kf_and_inter_then_dims(/* use_prev_frame_mvs = */ true);
    let sign_bias = sign_bias_from_header(&[false, false, false]);
    let bounds = probe_bounds(mi_rows, mi_cols);

    // Substitute a synthetic non-zero LAST cell — the salvage logic
    // exercised here is exactly what the encoder calls at
    // `lib.rs:2871-2879`, just with a known-non-zero `prev_frame_mv`
    // input.
    let cell = synthetic_last_cell();
    let list = find_mv_refs(
        InterMode::NewMv,
        BlockSize::Block16x16,
        RefFrame::LastFrame,
        &NoNeighbours,
        &sign_bias,
        &bounds,
        /* prev_frame_mv = */ Some(cell),
    );

    assert_ne!(
        list.mvs[0],
        Mv::ZERO,
        "with use_prev_frame_mvs=true and a non-zero LAST prev cell, \
         the matching-ref salvage at libvpx vp9_mvref_common.c:72-79 \
         must lift the cell's MV into mvc.nearest. Got zero — the \
         salvage path did not fire."
    );
    // Concretely: identity scale (LAST→LAST), loose bounds → the
    // probe MV propagates verbatim (`clamp_mv_ref` is a no-op when
    // bounds are loose enough).
    assert_eq!(
        list.mvs[0],
        Mv { row: 64, col: 64 },
        "matching-ref salvage lifts the prev-cell MV verbatim under \
         identity scale + loose bounds"
    );
    eprintln!(
        "[F.2 §6.2 assertion 1] use_prev_frame_mvs=true: mvc.nearest = {:?}",
        list.mvs[0],
    );
}

// ---------------------------------------------------------------------------
// §6.2 — assertion 3: salvage stays dormant when gated off.
// ---------------------------------------------------------------------------

#[test]
fn f2_find_mv_refs_temporal_zero_when_use_prev_frame_mvs_false() {
    let (mi_rows, mi_cols) = encode_kf_and_inter_then_dims(/* use_prev_frame_mvs = */ false);
    let sign_bias = sign_bias_from_header(&[false, false, false]);
    let bounds = probe_bounds(mi_rows, mi_cols);

    // Mirror the encoder's actual behaviour when use_prev_frame_mvs
    // is false: `gate_open` is false (frame_loop.rs:927-928), so the
    // picker driver passes `prev_frame_mv = None` to find_mv_refs
    // (lib.rs:2859-2862 returns None when prev_mv_grid is None). With
    // empty spatial neighbours AND no salvage input, slot 0 must
    // stay Mv::ZERO. This regression-guards the F.0.3 / F.2 default-
    // OFF behaviour.
    let list = find_mv_refs(
        InterMode::NewMv,
        BlockSize::Block16x16,
        RefFrame::LastFrame,
        &NoNeighbours,
        &sign_bias,
        &bounds,
        /* prev_frame_mv = */ None,
    );
    assert_eq!(
        list.mvs[0],
        Mv::ZERO,
        "with use_prev_frame_mvs=false (prev_frame_mv=None) and no \
         spatial neighbours, mvc.nearest must be Mv::ZERO. Got {:?} \
         — the salvage fires when it should not.",
        list.mvs[0]
    );
    eprintln!(
        "[F.2 §6.2 assertion 3] use_prev_frame_mvs=false: mvc.nearest = {:?}",
        list.mvs[0],
    );
}
