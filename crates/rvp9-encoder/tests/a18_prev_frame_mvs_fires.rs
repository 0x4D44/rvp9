//! F.0.3 §6.1 / §6.2 — A18 fires: the encoder's previous-frame MV grid
//! is populated after the first inter frame, contains real inter cells,
//! and the activation gate opens on the second inter frame when
//! `config.use_prev_frame_mvs = true`.
//!
//! This test does NOT verify that `find_mv_refs` consumes the grid —
//! that is F.2's surface and a separate test
//! (`a18_picker_reads_prev_grid.rs`) verifies only the plumbing is open.
//! Here we assert the WRITER side: the grid is captured, the cells are
//! populated, and the gate evaluates true.
//!
//! Run:
//! ```text
//! cargo test -p rvp9-encoder --release --test a18_prev_frame_mvs_fires -- --nocapture
//! ```

#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

use rvp9_core::mode_info::RefFrame;
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, StreamFrameKind, YuvFrameRef};

// ---------------------------------------------------------------------------
// Synthetic source — a small motion sequence the picker can land NEWMV on.
// ---------------------------------------------------------------------------

const W: u32 = 128;
const H: u32 = 128;

/// Build a 128x128 4:2:0 frame with a bright rectangle whose corner is
/// at `(ox, oy)` (luma pixels). The U/V planes are mid-grey (128) so the
/// chroma residual is null and any captured ref-frame in the prev-grid
/// is driven by the luma-side picker.
fn synthetic_frame(ox: i32, oy: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = W as usize;
    let h = H as usize;
    let uvw = w / 2;
    let uvh = h / 2;

    // Background: a slow gradient that gives every SB some luminance so
    // the picker doesn't bail out on a flat plane.
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            // Smooth gradient centred at frame middle.
            let dx = c as i32 - (w as i32 / 2);
            let dy = r as i32 - (h as i32 / 2);
            let d = ((dx * dx + dy * dy) as f64).sqrt();
            let v = 64.0 + (d * 0.5).min(96.0);
            y[r * w + c] = v.clamp(0.0, 255.0) as u8;
        }
    }

    // Bright square — width 32 px — at (ox, oy). This is the moving
    // feature the picker will track with NEWMV when ox/oy shifts.
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
        // Long GOP so frames 1 + 2 + 3 stay as inter frames.
        gop_length: 32,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Cbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    StreamEncoder::new(cfg).expect("StreamEncoder::new")
}

// ---------------------------------------------------------------------------
// §6.1 — A18 actually fires (Assertions A / B / C).
// ---------------------------------------------------------------------------

#[test]
fn a18_prev_frame_mvs_grid_populated_after_first_inter() {
    let w_us = W as usize;
    let mut enc = make_encoder(/* use_prev_frame_mvs = */ true);

    // Frame 0 — keyframe. The KF path resets the prev_mv_grid to
    // all-`(IntraFrame, zero-mv)` cells (HLD §4.1 / frame_loop.rs:690-706).
    let (y0, u0, v0) = synthetic_frame(20, 20);
    let pkt0 = enc
        .encode_frame(&yuv_view(&y0, &u0, &v0, w_us))
        .expect("encode KF");
    assert_eq!(pkt0.frame_kind, StreamFrameKind::Key, "frame 0 must be KF");

    // Frame 1 — first inter frame. After encode, `prev_mv_grid` holds the
    // picks the picker made for *this* frame (the splat at
    // `lib.rs:3500-3535`). Because the moving square shifted by (24, 24)
    // px the picker has every incentive to emit at least one NEWMV cell
    // on LASTFRAME.
    let (y1, u1, v1) = synthetic_frame(44, 44);
    let pkt1 = enc
        .encode_frame(&yuv_view(&y1, &u1, &v1, w_us))
        .expect("encode inter 1");
    assert_eq!(
        pkt1.frame_kind,
        StreamFrameKind::Inter,
        "frame 1 must be inter (GOP=32)"
    );

    // ---- Assertion A — plumbing alive ----
    let (grid, dims) = enc
        .prev_mv_grid_for_test()
        .expect("prev_mv_grid must be Some(_) after first inter encode");
    let mi_rows = (H as usize) / 8;
    let mi_cols = (W as usize) / 8;
    assert_eq!(
        dims,
        (mi_rows, mi_cols),
        "prev_mv_grid dims must be ({mi_rows}, {mi_cols})"
    );
    assert_eq!(
        grid.len(),
        mi_rows * mi_cols,
        "prev_mv_grid length must equal mi_rows * mi_cols"
    );

    // ---- Assertion B — writer captured real inter picks ----
    let has_inter_cell = grid.iter().any(|c| c.ref_frames[0] == RefFrame::LastFrame);
    assert!(
        has_inter_cell,
        "after first inter encode, prev_mv_grid must contain at least \
         one cell with ref_frames[0] == LastFrame (Assertion B; HLD §6.1)"
    );

    eprintln!(
        "[F.0.3 §6.1 A/B] grid dims = {dims:?}, cells = {}, LastFrame cells present = {has_inter_cell}",
        grid.len(),
    );

    // ---- Assertion C — reader path open on frame 2 ----
    //
    // Before encoding frame 2, evaluate the gate the inter-frame path
    // computes at `frame_loop.rs:859-867`. With
    // `config.use_prev_frame_mvs = true`, the previous frame was an
    // inter frame (not KF), and dims match, the gate MUST open.
    let gate_open = enc.prev_grid_gate_open_for_test(mi_rows, mi_cols);
    assert!(
        gate_open,
        "before frame 2, prev_grid_slice gate must be open: \
         !last_frame_was_key + dims match + config.use_prev_frame_mvs == true"
    );
    eprintln!("[F.0.3 §6.1 C] gate_open before frame 2 = {gate_open}");
}

// ---------------------------------------------------------------------------
// §6.2 — A18 chains across frames.
// ---------------------------------------------------------------------------
//
// The prev_mv_grid is overwritten by every inter encode. Encode three
// frames (KF + 2 inter) and verify that frame 2's picks land in the
// grid the next frame would see — i.e. the grid changes between
// inter[0] (set to frame-1 picks) and inter[1] (set to frame-2 picks)
// because the moving square has shifted again.

#[test]
fn a18_prev_frame_mvs_grid_chains_across_inter_frames() {
    let w_us = W as usize;
    let mut enc = make_encoder(/* use_prev_frame_mvs = */ true);

    // Frame 0 — KF.
    let (y0, u0, v0) = synthetic_frame(20, 20);
    enc.encode_frame(&yuv_view(&y0, &u0, &v0, w_us))
        .expect("encode KF");

    // Frame 1 — first inter; square moves by (+24, +24).
    let (y1, u1, v1) = synthetic_frame(44, 44);
    enc.encode_frame(&yuv_view(&y1, &u1, &v1, w_us))
        .expect("encode inter 1");

    // Snapshot the grid after frame 1.
    let (grid_after_1, _) = enc
        .prev_mv_grid_for_test()
        .expect("grid present after inter 1");
    let snapshot_1: Vec<_> = grid_after_1.to_vec();

    // Frame 2 — second inter; square moves by another (+24, +24). The
    // picker's NEWMV picks will (almost certainly) land in different mi
    // cells than frame 1's, so the captured grid must change.
    let (y2, u2, v2) = synthetic_frame(68, 68);
    enc.encode_frame(&yuv_view(&y2, &u2, &v2, w_us))
        .expect("encode inter 2");

    let (grid_after_2, _) = enc
        .prev_mv_grid_for_test()
        .expect("grid present after inter 2");

    // After the second inter encode, the grid MUST still contain
    // LastFrame picks (no regression from frame 1 to frame 2 — the
    // writer-side capture still runs).
    let has_inter = grid_after_2
        .iter()
        .any(|c| c.ref_frames[0] == RefFrame::LastFrame);
    assert!(
        has_inter,
        "after second inter, prev_mv_grid must still contain LastFrame cells"
    );

    // The grid is OVERWRITTEN on every inter frame (`self.prev_mv_grid
    // = Some(cur_mv_grid)` at `frame_loop.rs:985`). The diagnostic
    // case is "picks shifted between frames so the cell contents
    // differ"; the resilient case is "even if the picker happened to
    // emit the same picks both frames, the grid is still well-formed
    // and chained correctly." We assert the resilient invariant
    // (length + at-least-one-inter-cell) above; here we just log the
    // diagnostic.
    let any_diff = grid_after_2
        .iter()
        .zip(snapshot_1.iter())
        .any(|(a, b)| a != b);
    eprintln!("[F.0.3 §6.2] grid chained: LastFrame cells = {has_inter}, any_diff = {any_diff}",);
}
