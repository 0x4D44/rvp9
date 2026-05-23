//! F.0.3 §6.3 — A18 picker reads prev_grid plumbing test.
//!
//! Encodes a 2-frame sequence (KF + inter) with
//! `config.use_prev_frame_mvs = true` and verifies that on the second
//! inter frame the activation gate at `frame_loop.rs:859-867` would
//! open — i.e. the picker would have received `Some(prev_grid_slice)`
//! and threaded it through to `find_mv_refs` for the per-leaf prev-frame
//! salvage lookup at `lib.rs:2767-2770`.
//!
//! This is a PLUMBING test. It proves the path is open from
//! writer (frame 1 splat) → storage (`self.prev_mv_grid`) → reader
//! (`prev_grid_slice` is non-None when the gate is on). It does NOT
//! prove `find_mv_refs` returns a non-trivial candidate list — that is
//! F.2's surface (HLD §6.3b is explicitly deferred).
//!
//! Run:
//! ```text
//! cargo test -p rvp9-encoder --release --test a18_picker_reads_prev_grid -- --nocapture
//! ```

#![allow(clippy::print_stderr)]
#![allow(missing_docs)]

use rvp9_core::mode_info::RefFrame;
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, StreamFrameKind, YuvFrameRef};

const W: u32 = 128;
const H: u32 = 128;

/// Same synthetic source as `a18_prev_frame_mvs_fires`: gradient
/// background + a moving 32×32 bright square that gives the picker a
/// strong incentive to emit LASTFRAME picks (often NEWMV; at minimum
/// NEARESTMV with `ref_frames[0] = LastFrame`).
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

#[test]
fn a18_picker_reads_prev_grid_when_activated() {
    let w_us = W as usize;
    let mi_rows = (H as usize) / 8;
    let mi_cols = (W as usize) / 8;

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 500_000,
        gop_length: 32,
        aq_mode: AqMode::Off,
        // F.0.3 — activate the prev-frame MV salvage path.
        use_prev_frame_mvs: true,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    // Frame 0 — KF.
    let (y0, u0, v0) = synthetic_frame(20, 20);
    let pkt0 = enc
        .encode_frame(&yuv_view(&y0, &u0, &v0, w_us))
        .expect("encode KF");
    assert_eq!(pkt0.frame_kind, StreamFrameKind::Key);

    // After the KF, the gate must be CLOSED — the previous frame was a
    // keyframe, so libvpx's `use_prev_frame_mvs` predicate (and rvp9's
    // mirror at `frame_loop.rs:859-860`) returns false even when the
    // config flag is on.
    assert!(
        !enc.prev_grid_gate_open_for_test(mi_rows, mi_cols),
        "after a KF, the gate must be CLOSED (last_frame_was_key = true)"
    );

    // Frame 1 — first inter. Picker runs without prev_grid_slice (gate
    // closed because frame 0 was a KF). After encode, `prev_mv_grid`
    // holds frame 1's picks.
    let (y1, u1, v1) = synthetic_frame(44, 44);
    let pkt1 = enc
        .encode_frame(&yuv_view(&y1, &u1, &v1, w_us))
        .expect("encode inter 1");
    assert_eq!(pkt1.frame_kind, StreamFrameKind::Inter);

    // ---- Plumbing assertion 1 — the gate opens on frame 2 ----
    let gate_open = enc.prev_grid_gate_open_for_test(mi_rows, mi_cols);
    assert!(
        gate_open,
        "before frame 2 the gate must be OPEN: \
         !last_frame_was_key + dims match + config.use_prev_frame_mvs = true"
    );

    // ---- Plumbing assertion 2 — the slice the picker will see is sized
    //      correctly and contains the picks from frame 1 ----
    let (grid, dims) = enc
        .prev_mv_grid_for_test()
        .expect("prev_mv_grid must be Some after inter 1");
    assert_eq!(dims, (mi_rows, mi_cols));
    assert_eq!(
        grid.len(),
        mi_rows * mi_cols,
        "slice length must equal mi_rows * mi_cols (plumbing assertion 2 / HLD §6.3)"
    );

    // ---- Plumbing assertion 3 — at least one cell is a real inter
    //      pick (not the post-KF intra-cell default), proving frame 1's
    //      picks landed in the grid the picker will read on frame 2 ----
    let has_last_cell = grid
        .iter()
        .any(|c| c.ref_frames[0] == RefFrame::LastFrame);
    assert!(
        has_last_cell,
        "at least one cell must have ref_frames[0] != IntraFrame \
         (frame 1's picks landed in the grid the picker sees; HLD §6.3 assertion 3)"
    );
    eprintln!(
        "[F.0.3 §6.3] gate_open = {gate_open}, grid_len = {}, has_last_cell = {has_last_cell}",
        grid.len(),
    );

    // Frame 2 — second inter. The picker runs with the prev_grid_slice
    // we just verified is open + populated. F.0.3 deliverable is "the
    // path is wired and the slice arrives"; what `find_mv_refs` does
    // with it (HLD §6.3b) is F.2's surface — we just confirm the
    // encode still produces a valid inter frame.
    let (y2, u2, v2) = synthetic_frame(68, 68);
    let pkt2 = enc
        .encode_frame(&yuv_view(&y2, &u2, &v2, w_us))
        .expect("encode inter 2");
    assert_eq!(pkt2.frame_kind, StreamFrameKind::Inter);
}

#[test]
fn a18_picker_reads_nothing_when_default_off() {
    // Same harness but with the default `use_prev_frame_mvs = false`.
    // The gate must STAY CLOSED on frame 2 even though the dims match
    // and the previous frame was an inter frame.
    let w_us = W as usize;
    let mi_rows = (H as usize) / 8;
    let mi_cols = (W as usize) / 8;

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 500_000,
        gop_length: 32,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    let (y0, u0, v0) = synthetic_frame(20, 20);
    enc.encode_frame(&yuv_view(&y0, &u0, &v0, w_us))
        .expect("encode KF");

    let (y1, u1, v1) = synthetic_frame(44, 44);
    enc.encode_frame(&yuv_view(&y1, &u1, &v1, w_us))
        .expect("encode inter 1");

    // With config OFF, the gate must NOT open on frame 2.
    assert!(
        !enc.prev_grid_gate_open_for_test(mi_rows, mi_cols),
        "default-off config must keep prev_grid_slice = None on frame 2 \
         (do-no-harm gate; HLD §6.4)"
    );
}
