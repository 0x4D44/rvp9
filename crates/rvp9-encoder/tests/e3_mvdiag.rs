//! E3 (2026-05-22) — diagnostic-only harness for the "0% NEWMV on
//! foreman" investigation. Drives the encoder over a 256×256 centre crop
//! of foreman_cif (64 frames, 100 kbps target — mirroring
//! `bdrate_gate_stage4a_realcontent`) with the `RVP9_E3_MVDIAG` env-gated
//! instrumentation active, so the picker's per-candidate cost/MV
//! breakdown is dumped to stderr.
//!
//! This is a THROWAWAY diagnostic test (delete after E3 closes). It does
//! NOT assert anything — it exists purely to provide a deterministic
//! driver for the instrumentation. Gated `#[ignore]` so it never runs on
//! `cargo test`. Invoke:
//!
//!   $env:RVP9_E3_MVDIAG=1
//!   cargo test --test e3_mvdiag -- --ignored --nocapture
//!
//! stderr IS the output mechanism here.
#![allow(clippy::print_stderr)]

use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped};

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
#[ignore = "E3 diagnostic-only; set RVP9_E3_MVDIAG and run with --ignored --nocapture"]
fn e3_foreman_newmv_diag() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;
    // Target 100 kbps — the operating point the E1/E2 audits flagged.
    // Override via RVP9_E3_TARGET_BPS to confirm NEWMV activates once q
    // drops (low-lambda regime). Diagnostic only.
    let target_bps: u32 = std::env::var("RVP9_E3_TARGET_BPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let Some(path) = probe_foreman_cif() else {
        eprintln!("[E3 skip] foreman_cif.y4m not found at tests/corpus/foreman_cif.y4m");
        return;
    };

    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "[E3 skip] foreman_cif only yielded {}/{N_FRAMES} frames",
                f.len()
            );
            return;
        }
        Err(e) => {
            eprintln!("[E3 skip] foreman_cif read failed: {e}");
            return;
        }
    };

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: target_bps,
        gop_length: frames.len() as u32 + 1,
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
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    eprintln!("[E3] foreman {W}x{H} {N_FRAMES} frames @ {framerate} fps target {target_bps} bps",);
    for (i, (y, u, v)) in frames.iter().enumerate() {
        eprintln!("[E3] === frame {i} ===");
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .expect("encode_frame");
        eprintln!("[E3] frame {i} actual_bits={}", pkt.actual_bits);
    }
}
