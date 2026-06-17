#![allow(clippy::print_stderr)]

//! Budget-gate KF-cap validation (2026-05-29).
//!
//! The keyframe budget gate in `rate_control/cbr.rs` caps the first-KF
//! active-quality floor at LOW bitrate (restoring the removed
//! `KF_ACTIVE_QUALITY_BASIS_CAP`, q≈57) while letting HIGH-bitrate KFs
//! stay near-lossless (preserving f971c4e's win). This test encodes the
//! two regimes through the real `StreamEncoder`, decodes each packet, and
//! reports the first-KF qindex / KF bits / per-frame mean Y-PSNR so the
//! gate's effect can be measured with real numbers.
//!
//! It is `#[ignore]` (a measurement oracle, not a CI gate — the CI gate
//! is the `budget_gate_*` unit test in `cbr.rs`). It also asserts the
//! load-bearing invariants so a regression here still trips:
//!   * LOW (foreman 256x256 @ 100k):  first KF q >= 50, decode_failed == 0.
//!   * HIGH (foreman_320x256 @ 2000k): first KF q < 30, decode_failed == 0.
//!
//! Run with:
//! ```text
//! cargo test --release --test budget_gate_validation -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::read_cropped;
use std::path::PathBuf;

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

fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Vec<ParsedFrame<'a>> {
    let parsed = dec.parse_packet(pkt).expect("parse_packet");
    parsed
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect()
}

fn cfg(w: u32, h: u32, fr: f64, tb: u32, gop: u32) -> EncoderConfig {
    EncoderConfig {
        width: w,
        height: h,
        framerate: fr,
        target_bitrate_bps: tb,
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

/// Encode `frames` at `cfg`, decode every packet, and return
/// `(first_kf_q, first_kf_bits, mean_psnr_db, decode_failed)`.
fn run_regime(
    label: &str,
    frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    config: EncoderConfig,
) -> (u8, u32, f64, u32) {
    let w = config.width as usize;
    let h = config.height as usize;
    let fr = config.framerate;
    let mut enc = StreamEncoder::new(config).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    let mut first_kf_q = 0u8;
    let mut first_kf_bits = 0u32;
    let mut total_bits: u64 = 0;
    let mut total_sse: u64 = 0;
    let mut decode_failed = 0u32;

    for (i, (y, u, v)) in frames.iter().enumerate() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w))
            .expect("encode_frame");
        if i == 0 {
            first_kf_q = pkt.q_index;
            first_kf_bits = pkt.actual_bits;
        }
        total_bits += u64::from(pkt.actual_bits);

        let parsed = decode_packet(&mut dec, &pkt.data);
        match dec.decode_frame(&parsed[0]) {
            Ok(buf) => {
                let yp = buf.y_plane();
                let mut sse: u64 = 0;
                for r in 0..h {
                    let row = &yp.data()[r * yp.stride()..r * yp.stride() + w];
                    for (a, b) in y[r * w..r * w + w].iter().zip(row.iter()) {
                        let d = i32::from(*a) - i32::from(*b);
                        sse += (d * d) as u64;
                    }
                }
                total_sse += sse;
            }
            Err(_) => decode_failed += 1,
        }
    }

    let n = frames.len() as f64;
    let actual_kbps = (total_bits as f64 * fr) / (n * 1000.0);
    let mse = total_sse as f64 / (n * (w * h) as f64);
    let mean_psnr = if mse == 0.0 {
        100.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    eprintln!(
        "[{label}] first_kf_q={first_kf_q} first_kf_bits={first_kf_bits} \
         actual={actual_kbps:.1}kbps mean_psnr={mean_psnr:.2}dB decode_failed={decode_failed}",
    );
    (first_kf_q, first_kf_bits, mean_psnr, decode_failed)
}

#[test]
#[ignore = "budget-gate KF-cap real-encode validation (low + high regime)"]
fn budget_gate_low_and_high_regime() {
    // Run BOTH regimes and print their numbers BEFORE asserting, so a
    // baseline run (where the LOW assert would trip) still reports the
    // HIGH numbers — the no-regression comparison needs both legs.
    let mut pending = Vec::new();
    // Count regimes that actually ran. A missing fixture must NOT let the
    // test report success while validating nothing — if neither leg runs we
    // panic at the end.
    let mut regimes_run = 0u32;

    // HIGH regime: foreman_320x256_60f full-frame @ 2000 kbps, 60 frames.
    // Run first so its numbers are always captured.
    {
        let src = PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_320x256_60f.y4m");
        if !src.is_file() {
            eprintln!("[skip HIGH] foreman_320x256_60f.y4m not found");
        } else {
            regimes_run += 1;
            let (frames, fr) = read_cropped(&src, 320, 256, 60).expect("read foreman_320x256");
            let (kf_q, _bits, psnr, failed) = run_regime(
                "HIGH 2000k 320x256",
                &frames,
                cfg(320, 256, fr, 2_000_000, 61),
            );
            if failed != 0 {
                pending.push(format!("HIGH regime decode_failed={failed} (must be 0)"));
            }
            if kf_q >= 30 {
                pending.push(format!(
                    "HIGH-bitrate first KF must stay near-lossless (got q={kf_q}); PSNR={psnr:.2}dB"
                ));
            }
        }
    }

    // LOW regime: foreman 256x256 centre-crop @ 100 kbps, 64 frames.
    {
        let src = PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_cif.y4m");
        if !src.is_file() {
            eprintln!("[skip LOW] foreman_cif.y4m not found");
        } else {
            regimes_run += 1;
            let (frames, fr) = read_cropped(&src, 256, 256, 64).expect("read foreman_cif");
            let (kf_q, _bits, psnr, failed) =
                run_regime("LOW 100k 256x256", &frames, cfg(256, 256, fr, 100_000, 65));
            if failed != 0 {
                pending.push(format!("LOW regime decode_failed={failed} (must be 0)"));
            }
            if kf_q < 50 {
                pending.push(format!(
                    "LOW-bitrate first KF must be capped (got q={kf_q}); PSNR={psnr:.2}dB"
                ));
            }
        }
    }

    // Fail loudly if no fixture was present — otherwise this test would
    // "pass" while validating nothing (a vacuous green).
    assert!(
        regimes_run > 0,
        "budget-gate validation ran ZERO regimes: required corpus fixtures \
         (foreman_320x256_60f.y4m / foreman_cif.y4m) are missing under \
         tests/corpus — cannot validate the gate",
    );
    assert!(
        pending.is_empty(),
        "budget-gate validation failed:\n{}",
        pending.join("\n")
    );
}
