#![allow(clippy::print_stderr)]

//! KFCAP (Track A1) measurement harness — keyframe-share cap.
//!
//! Encodes foreman (CIF centre-crop 256x256 + the 320x256 60-frame
//! variant) at 100k / 250k / 2000k CBR cpu7, real-decodes each frame
//! (rvp9 decoder), and reports per-bitrate:
//!   - KF q / bits / Y-PSNR / byte-share
//!   - GOP-mean Y-PSNR (the metric that decides if KFCAP lands)
//!   - achieved kbps (overshoot factor)
//!   - the inter-Q trajectory tail (buffer-health proxy)
//!
//! The SAME binary is run twice by the driver: `RVP9_KFCAP` unset (OFF =
//! HEAD baseline) and `RVP9_KFCAP=1` (ON). The cap is env-gated so this
//! one harness measures both arms.
//!
//! Run:
//! ```text
//! cargo test --release -p rvp9-encoder --test diag_kfcap_measure -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped as y4m_read_cropped, YuvFrameBuf};
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

fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut sse: u64 = 0;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = i32::from(*a) - i32::from(*b);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn probe_foreman_320x256() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("tests/corpus/foreman_320x256_60f.y4m"),
        PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_320x256_60f.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Encode one clip at one target, real-decode, return per-frame
/// (is_key, q, bits, psnr) plus totals.
struct ClipResult {
    per_frame: Vec<(bool, u8, u32, f64)>,
    total_bits: u64,
    gop_mean_psnr: f64,
    achieved_kbps: f64,
}

fn encode_measure(
    frames: &[YuvFrameBuf],
    w: u32,
    h: u32,
    framerate: f64,
    target_bps: u32,
    gop_length: u32,
) -> ClipResult {
    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps: target_bps,
        gop_length,
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
    let mut dec = Decoder::new();
    let mut per_frame = Vec::with_capacity(frames.len());
    let mut total_bits: u64 = 0;
    let mut total_sse: u64 = 0;

    for (y, u, v) in frames.iter() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let yp = buf.y_plane();
        let mut dec_y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            dec_y[r * w as usize..(r + 1) * w as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + w as usize]);
        }
        let psnr_y = psnr_db(y, &dec_y);
        let mut sse: u64 = 0;
        for (a, b) in y.iter().zip(dec_y.iter()) {
            let d = i32::from(*a) - i32::from(*b);
            sse += (d * d) as u64;
        }
        per_frame.push((
            pkt.frame_kind.is_key(),
            pkt.q_index,
            pkt.actual_bits,
            psnr_y,
        ));
        total_bits += u64::from(pkt.actual_bits);
        total_sse += sse;
    }

    let achieved_kbps = (total_bits as f64 * framerate) / (frames.len() as f64 * 1000.0);
    let mse = total_sse as f64 / (frames.len() as f64 * (w * h) as f64);
    let gop_mean_psnr = if mse == 0.0 {
        100.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    ClipResult {
        per_frame,
        total_bits,
        gop_mean_psnr,
        achieved_kbps,
    }
}

fn report(label: &str, target_kbps: u32, r: &ClipResult) {
    let (kf_is_key, kf_q, kf_bits, kf_psnr) = r.per_frame[0];
    assert!(kf_is_key, "frame 0 must be a keyframe");
    let kf_share = if r.total_bits > 0 {
        kf_bits as f64 / r.total_bits as f64 * 100.0
    } else {
        0.0
    };
    // Inter-Q tail: last 4 frames' Q (buffer-health proxy — pinning at
    // 255 means the buffer drained and the inters starved).
    let n = r.per_frame.len();
    let tail: Vec<u8> = r.per_frame[n.saturating_sub(4)..]
        .iter()
        .map(|f| f.1)
        .collect();
    let inter_mean_psnr = {
        let inters: Vec<f64> = r.per_frame[1..].iter().map(|f| f.3).collect();
        inters.iter().sum::<f64>() / inters.len().max(1) as f64
    };
    eprintln!(
        "  [{label}] {target_kbps}k: KF q={kf_q} bits={kf_bits} ({kf_psnr:.2}dB) share={kf_share:.1}%  \
         GOP-mean={:.3}dB  inter-mean={inter_mean_psnr:.3}dB  achieved={:.1}kbps (x{:.2})  inter-Q-tail={tail:?}",
        r.gop_mean_psnr,
        r.achieved_kbps,
        r.achieved_kbps / (target_kbps as f64),
    );
}

#[test]
#[ignore = "KFCAP Track A1 measurement — run with RVP9_KFCAP unset then =1"]
fn diag_kfcap_measure() {
    let kfcap = std::env::var_os("RVP9_KFCAP").is_some_and(|v| v != "0");
    let arm = if kfcap { "ON " } else { "OFF" };
    eprintln!("================ KFCAP arm = {arm} (RVP9_KFCAP={kfcap}) ================");

    let targets: [u32; 3] = [100_000, 250_000, 2_000_000];

    // Clip 1: foreman_cif 256x256 centre-crop, 32 frames.
    if let Some(src) = probe_foreman_cif() {
        const W: u32 = 256;
        const H: u32 = 256;
        const N: usize = 32;
        let (frames, fr) = y4m_read_cropped(&src, W, H, N).expect("read foreman_cif");
        eprintln!("\n--- foreman_cif {W}x{H} {fr}fps {} frames ---", frames.len());
        for &tb in &targets {
            let r = encode_measure(&frames, W, H, fr, tb, frames.len() as u32 + 1);
            report("cif", tb / 1000, &r);
        }
    } else {
        eprintln!("[skip] foreman_cif.y4m not found");
    }

    // Clip 2: foreman_320x256_60f (native size, 60 frames).
    if let Some(src) = probe_foreman_320x256() {
        const W: u32 = 320;
        const H: u32 = 256;
        const N: usize = 60;
        let (frames, fr) = y4m_read_cropped(&src, W, H, N).expect("read foreman_320x256");
        eprintln!("\n--- foreman_320x256 {W}x{H} {fr}fps {} frames ---", frames.len());
        for &tb in &targets {
            let r = encode_measure(&frames, W, H, fr, tb, frames.len() as u32 + 1);
            report("320", tb / 1000, &r);
        }
    } else {
        eprintln!("[skip] foreman_320x256_60f.y4m not found");
    }
}
