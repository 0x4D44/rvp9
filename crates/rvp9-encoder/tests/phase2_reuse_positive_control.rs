//! Phase-2 option-A positive control (HLD §4.3).
//!
//! Proves the descent-pick NEWMV reuse is actually wired: encoding inter
//! frames with the descent ON (default) must REUSE a non-zero number of
//! NEWMV searches. A no-op masquerade (the reuse branch never firing)
//! would leave `PHASE2_NEWMV_REUSED` at zero and fail this test.
//!
//! Run with: `cargo test -p rvp9-encoder --test phase2_reuse_positive_control`

use rvp9_encoder::rdo::inter_dispatch::{PHASE2_NEWMV_FRESH, PHASE2_NEWMV_REUSED};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::probe_foreman_cif;
use std::sync::atomic::Ordering;

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: usize = 8;

fn crop_center(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let x0 = (sw - dw) / 2;
    let y0 = (sh - dh) / 2;
    let mut out = Vec::with_capacity(dw * dh);
    for r in 0..dh {
        let so = (y0 + r) * sw + x0;
        out.extend_from_slice(&src[so..so + dw]);
    }
    out
}

fn config() -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        framerate: 30.0,
        target_bitrate_bps: 1_000_000,
        gop_length: (N_FRAMES as u32) + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
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
        min_qindex: 4,
        max_qindex: 255,
    }
}

#[test]
fn phase2_newmv_reuse_fires() {
    let Some(path) = probe_foreman_cif() else {
        return;
    };
    // Read foreman_cif (352x288) Y planes and center-crop to 256x256.
    let data = std::fs::read(&path).expect("read foreman_cif");
    let nl = data.iter().position(|&b| b == b'\n').expect("hdr");
    let hdr = &data[..nl];
    let toks: Vec<&[u8]> = hdr.split(|&b| b == b' ').collect();
    let mut sw = 0usize;
    let mut sh = 0usize;
    for t in &toks[1..] {
        if t.first() == Some(&b'W') {
            sw = std::str::from_utf8(&t[1..]).unwrap().parse().unwrap();
        } else if t.first() == Some(&b'H') {
            sh = std::str::from_utf8(&t[1..]).unwrap().parse().unwrap();
        }
    }
    let ysz = sw * sh;
    let csz = (sw / 2) * (sh / 2);
    let fsz = ysz + 2 * csz;
    let mut pos = nl + 1;
    let mut frames = Vec::new();
    while pos < data.len() && frames.len() < N_FRAMES {
        let fnl = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| pos + p);
        let Some(fnl) = fnl else { break };
        let base = fnl + 1;
        if base + fsz > data.len() {
            break;
        }
        let y = crop_center(&data[base..base + ysz], sw, sh, W as usize, H as usize);
        let u = crop_center(
            &data[base + ysz..base + ysz + csz],
            sw / 2,
            sh / 2,
            (W / 2) as usize,
            (H / 2) as usize,
        );
        let v = crop_center(
            &data[base + ysz + csz..base + fsz],
            sw / 2,
            sh / 2,
            (W / 2) as usize,
            (H / 2) as usize,
        );
        frames.push((y, u, v));
        pos = base + fsz;
    }
    assert_eq!(frames.len(), N_FRAMES, "need {N_FRAMES} frames");

    let before_reuse = PHASE2_NEWMV_REUSED.load(Ordering::Relaxed);
    let before_fresh = PHASE2_NEWMV_FRESH.load(Ordering::Relaxed);

    let mut enc = StreamEncoder::new(config()).expect("StreamEncoder::new");
    for (y, u, v) in &frames {
        let view = YuvFrameRef {
            y,
            u,
            v,
            y_stride: W as usize,
            u_stride: (W / 2) as usize,
            v_stride: (W / 2) as usize,
        };
        enc.encode_frame(&view).expect("encode_frame");
    }

    let reused = PHASE2_NEWMV_REUSED.load(Ordering::Relaxed) - before_reuse;
    let fresh = PHASE2_NEWMV_FRESH.load(Ordering::Relaxed) - before_fresh;
    assert!(
        reused > 0,
        "expected NEWMV reuse to fire on inter frames (descent ON), got reused={reused} fresh={fresh}"
    );
    // Every final-pass RT-picker leaf is attributed to reuse or fresh.
    assert!(
        reused + fresh > 0,
        "no leaves attributed: reused={reused} fresh={fresh}"
    );
}
