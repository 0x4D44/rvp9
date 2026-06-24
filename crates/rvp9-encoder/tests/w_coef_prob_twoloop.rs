// Integration test: stderr IS the output mechanism for the corpus /
// vpxdec leg; the workspace's `print_stderr = "deny"` lint targets
// library code, not this test binary.
#![allow(clippy::print_stderr)]

//! A6 — coef-prob TWO_LOOP PROOF.
//!
//! The compressed-header coefficient-probability writer now runs
//! libvpx's TWO_LOOP `update_coef_probs` / `update_coef_probs_common`
//! (`vp9/encoder/vp9_bitstream.c:683-699` + `:545-625`, TWO_LOOP
//! branch): per TX size, a dry-run savings search decides which probs
//! to signal — updating where `savings > 0` and SUPPRESSING the whole
//! table when the aggregate `savings < 0`. This replaces the prior
//! "Strategy B" force-update-on-diff policy as the LIVE writer for
//! inter frames.
//!
//! Two proofs:
//!
//! 1. **Anti-G3 (Strategy-B-impossible outcome).** On a deterministic moving
//!    textured encode,
//!    assert the TWO_LOOP plan both UPDATES some coef-prob tables AND
//!    SUPPRESSES others *despite a would-update slot* (aggregate
//!    `savings < 0`). Strategy B (force-update-on-diff) emits gate
//!    bit 1 whenever ANY slot differs, so it can NEVER produce the
//!    "suppressed despite a diff" outcome. If Strategy B were still
//!    live, `COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF` would stay 0 and
//!    this test FAILS.
//!
//! 2. **Oracle (byte-exact through vpxdec).** The TWO_LOOP stream is
//!    valid VP9 that libvpx-stock `vpxdec` decodes byte-identically to
//!    rvp9-decoder. (The differential `cross_decoder_gate` 4/4 is the
//!    full version; this is the A6-scoped restatement.)

use rvp9_encoder::bitstream_gates::{
    COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF, COEF_PROB_TABLES_UPDATED,
};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::Ordering::Relaxed;

type OwnedYuvFrame = (Vec<u8>, Vec<u8>, Vec<u8>);
type OwnedYuvClip = Vec<OwnedYuvFrame>;

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

fn gate_config(w: u32, h: u32, framerate: f64, gop: u32) -> EncoderConfig {
    EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps: 1_000_000,
        gop_length: gop,
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

fn synthetic_motion_frames(w: u32, h: u32, n_frames: u32) -> (OwnedYuvClip, f64) {
    let w_us = w as usize;
    let h_us = h as usize;
    let uv_w = w_us / 2;
    let uv_h = h_us / 2;
    let mut frames = Vec::with_capacity(n_frames as usize);

    for frame_idx in 0..n_frames as usize {
        let x_shift = frame_idx * 5;
        let y_shift = frame_idx * 3;
        let mut y = vec![0u8; w_us * h_us];
        for row in 0..h_us {
            for col in 0..w_us {
                let sx = (col + x_shift) & 255;
                let sy = (row + y_shift) & 255;
                let v = sx
                    .wrapping_mul(37)
                    .wrapping_add(sy.wrapping_mul(19))
                    .wrapping_add((sx ^ sy).wrapping_mul(11))
                    .wrapping_add(((sx.wrapping_mul(sy)) >> 3) & 255);
                y[row * w_us + col] = v as u8;
            }
        }

        let mut u = vec![0u8; uv_w * uv_h];
        let mut v_plane = vec![0u8; uv_w * uv_h];
        for row in 0..uv_h {
            for col in 0..uv_w {
                u[row * uv_w + col] = (96 + ((col + frame_idx) & 31)) as u8;
                v_plane[row * uv_w + col] = (144 + ((row + frame_idx * 2) & 31)) as u8;
            }
        }

        frames.push((y, u, v_plane));
    }

    (frames, 30.0)
}

/// A6 anti-G3 — on a real multi-frame encode the TWO_LOOP coef-prob
/// writer must BOTH update some tables AND suppress at least one table
/// that had a would-update slot (aggregate `savings < 0`). The latter
/// is structurally impossible under Strategy B (force-update-on-diff),
/// so a green run is the live-TWO_LOOP proof; a Strategy-B regression
/// turns `COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF` to 0 → FAIL.
#[test]
fn a6_twoloop_updates_some_and_suppresses_with_diff() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;

    let (frames, framerate) = synthetic_motion_frames(W, H, N_FRAMES);

    COEF_PROB_TABLES_UPDATED.store(0, Relaxed);
    COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF.store(0, Relaxed);

    let cfg = gate_config(W, H, framerate, N_FRAMES + 1);
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    for (y, u, v) in &frames {
        enc.encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
    }

    let updated = COEF_PROB_TABLES_UPDATED.load(Relaxed);
    let suppressed_with_diff = COEF_PROB_TABLES_SUPPRESSED_WITH_DIFF.load(Relaxed);
    eprintln!(
        "A6 TWO_LOOP: tables_updated={updated}, tables_suppressed_with_diff={suppressed_with_diff}"
    );

    assert!(
        updated > 0,
        "TWO_LOOP must UPDATE at least one coef-prob table on a real encode \
         (got {updated})"
    );
    assert!(
        suppressed_with_diff > 0,
        "TWO_LOOP must SUPPRESS at least one coef-prob table despite a \
         would-update slot (aggregate savings < 0) — the Strategy-B-impossible \
         outcome. Got 0, which means either Strategy B is live (force-update \
         on diff) or the savings search never gates a table off."
    );
}

fn vpxdec_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    if Command::new("vpxdec").arg("--help").output().is_ok() {
        return Some(PathBuf::from("vpxdec"));
    }
    None
}

/// A6 oracle — the TWO_LOOP stream is valid VP9 that vpxdec decodes
/// byte-identically to rvp9-decoder. `#[ignore]` (needs vpxdec binary),
/// invoked via `--ignored`.
#[test]
#[ignore = "A6 oracle — needs corpus + vpxdec; run with --ignored"]
fn a6_twoloop_stream_byte_exact_through_vpxdec() {
    use rvp9_decoder::Decoder;
    use rvp9_harness::container::{IvfHeader, write_ivf};

    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 8;

    let (frames, framerate) = synthetic_motion_frames(W, H, N_FRAMES);

    let cfg = gate_config(W, H, framerate, N_FRAMES + 1);
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut packets: Vec<Vec<u8>> = Vec::with_capacity(N_FRAMES as usize);
    for (y, u, v) in &frames {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        packets.push(pkt.data);
    }

    // rvp9-decoder baseline.
    let mut dec = Decoder::new();
    let mut rvp9_yuv: Vec<u8> = Vec::new();
    for pkt in &packets {
        let parsed = dec.parse_packet(pkt).expect("parse_packet");
        let owned: Vec<_> = parsed
            .frames()
            .iter()
            .map(|pf| rvp9_decoder::ParsedFrame {
                header: pf.header.clone(),
                header_end_byte: pf.header_end_byte,
                full_frame_data: pf.full_frame_data,
            })
            .collect();
        for pf in &owned {
            let buf = dec.decode_frame(pf).expect("decode_frame");
            let yp = buf.y_plane();
            let up = buf.u_plane();
            let vp = buf.v_plane();
            let (wu, hu) = (W as usize, H as usize);
            for r in 0..hu {
                let o = r * yp.stride();
                rvp9_yuv.extend_from_slice(&yp.data()[o..o + wu]);
            }
            for r in 0..hu / 2 {
                let o = r * up.stride();
                rvp9_yuv.extend_from_slice(&up.data()[o..o + wu / 2]);
            }
            for r in 0..hu / 2 {
                let o = r * vp.stride();
                rvp9_yuv.extend_from_slice(&vp.data()[o..o + wu / 2]);
            }
        }
    }

    let Some(vpxdec) = vpxdec_path() else {
        eprintln!("[SKIPPED] vpxdec not found; rvp9-decoder produced a valid stream.");
        return;
    };

    let tmp = std::env::temp_dir();
    let ivf = tmp.join("rvp9_a6_twoloop.ivf");
    {
        let mut f = std::fs::File::create(&ivf).expect("create ivf");
        let hdr = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
        let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
        write_ivf(&mut f, &hdr, &refs).expect("write_ivf");
    }
    let out = tmp.join("rvp9_a6_twoloop_vpx.yuv");
    let status = Command::new(&vpxdec)
        .args(["--codec=vp9", "--i420", "-o"])
        .arg(&out)
        .arg(&ivf)
        .output()
        .expect("run vpxdec");
    assert!(
        status.status.success(),
        "vpxdec rejected the TWO_LOOP stream: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let vpx_yuv = std::fs::read(&out).expect("read vpxdec output");
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&out);

    assert_eq!(
        vpx_yuv,
        rvp9_yuv,
        "A6 oracle: vpxdec decode of the TWO_LOOP stream must be byte-identical \
         to rvp9-decoder (vpx={} bytes, rvp9={} bytes)",
        vpx_yuv.len(),
        rvp9_yuv.len()
    );
}
