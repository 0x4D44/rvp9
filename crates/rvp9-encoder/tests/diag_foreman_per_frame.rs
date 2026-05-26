#![allow(clippy::print_stderr)]

//! A0' — Per-frame foreman_cif diagnostic.
//!
//! foreman_cif has a textured face + real motion; testsrc is mostly flat
//! and the A1 Block8x8 fix closed its picker gap. But foreman PSNR is
//! still stuck at 14-16 dB across the 25-250 kbps sweep. The picker gap
//! is closed yet PSNR doesn't move with bitrate — so we have a
//! DIFFERENT dominant bug.
//!
//! This test:
//! 1. Encodes foreman_cif 256×256 centre-crop at a sweep of targets
//!    (25, 50, 100, 250 kbps), reporting per-frame:
//!    - rvp9 qindex, bits, Y-PSNR, KF flag
//!    - SSE(recon, src), SSE(recon, prev_src) — the diagonal "recon
//!      tracks lagged source by N frames" signature
//! 2. Mirrors libvpx --rt --target-bitrate at the same targets and
//!    prints per-frame bits + PSNR side-by-side.
//!
//! Run with:
//! ```text
//! cargo test --release --test diag_foreman_per_frame -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped as y4m_read_cropped, YuvFrameBuf};
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn vpxenc_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxenc.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    None
}

fn vpxdec_path() -> Option<PathBuf> {
    let canonical = PathBuf::from(r"D:\language\libvpx-stock\vpxdec.exe");
    if canonical.is_file() {
        return Some(canonical);
    }
    None
}

fn write_i420_raw(path: &Path, frames: &[YuvFrameBuf]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for (y, u, v) in frames {
        f.write_all(y)?;
        f.write_all(u)?;
        f.write_all(v)?;
    }
    Ok(())
}

fn parse_ivf_sizes(ivf: &[u8]) -> Vec<u32> {
    let mut sizes = Vec::new();
    if ivf.len() < 32 {
        return sizes;
    }
    let mut p = 32usize;
    while p + 12 <= ivf.len() {
        let frame_size = u32::from_le_bytes([ivf[p], ivf[p + 1], ivf[p + 2], ivf[p + 3]]);
        p += 12;
        if p + frame_size as usize > ivf.len() {
            break;
        }
        sizes.push(frame_size);
        p += frame_size as usize;
    }
    sizes
}

fn read_y4m_yplanes(path: &Path, w: u32, h: u32, count: usize) -> Vec<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    // Y4M: skip header line.
    let nl = buf.iter().position(|&b| b == b'\n').unwrap();
    let mut p = nl + 1;
    let yuv_per_frame = (w * h) as usize + 2 * ((w / 2) * (h / 2)) as usize;
    let mut out = Vec::with_capacity(count);
    while out.len() < count && p < buf.len() {
        // skip "FRAME\n"
        let nl2 = buf[p..].iter().position(|&b| b == b'\n').map(|n| p + n);
        let Some(nl2) = nl2 else { break };
        p = nl2 + 1;
        if p + yuv_per_frame > buf.len() {
            break;
        }
        out.push(buf[p..p + (w * h) as usize].to_vec());
        p += yuv_per_frame;
    }
    out
}

#[test]
#[ignore = "A0' — Foreman per-frame diagnostic"]
fn diag_foreman_per_frame() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 32;

    let Some(src) = probe_foreman_cif() else {
        eprintln!("    [skip] foreman_cif.y4m not found");
        return;
    };
    let (frames, framerate) = y4m_read_cropped(&src, W, H, N_FRAMES as usize)
        .expect("read foreman_cif");
    eprintln!("foreman_cif {W}x{H} {framerate} fps {} frames", frames.len());

    let targets: [u32; 4] = [25_000, 50_000, 100_000, 250_000];

    for &tb in &targets {
        eprintln!("\n=== target {} kbps ===", tb / 1000);
        // rvp9 side.
        let cfg = EncoderConfig {
            width: W,
            height: H,
            framerate,
            target_bitrate_bps: tb,
            gop_length: N_FRAMES + 1,
            aq_mode: AqMode::Off,
            use_prev_frame_mvs: false,
            speed: 0,
            deadline: rvp9_encoder::Deadline::Good,
        };
        let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
        let mut dec = Decoder::new();
        let mut rvp_total_bits: u64 = 0;
        let mut rvp_total_sse: u64 = 0;

        for (i, (y, u, v)) in frames.iter().enumerate() {
            let pkt = enc
                .encode_frame(&yuv_view(y, u, v, W as usize))
                .expect("encode_frame");
            let parsed = decode_packet(&mut dec, &pkt.data);
            let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
            let yp = buf.y_plane();
            let mut dec_y = vec![0u8; (W * H) as usize];
            for r in 0..H as usize {
                dec_y[r * W as usize..(r + 1) * W as usize]
                    .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
            }
            let psnr_y = psnr_db(y, &dec_y);
            let mut sse_self: u64 = 0;
            for (a, b) in y.iter().zip(dec_y.iter()) {
                let d = i32::from(*a) - i32::from(*b);
                sse_self += (d * d) as u64;
            }
            let mut sse_vs_prev_src: u64 = 0;
            if i > 0 {
                let prev = &frames[i - 1].0;
                for (a, b) in prev.iter().zip(dec_y.iter()) {
                    let d = i32::from(*a) - i32::from(*b);
                    sse_vs_prev_src += (d * d) as u64;
                }
            }
            // Source motion: SSE(src_i, src_{i-1}) — characterises what the
            // encoder needs to compensate.
            let mut sse_src_motion: u64 = 0;
            if i > 0 {
                let prev = &frames[i - 1].0;
                for (a, b) in prev.iter().zip(y.iter()) {
                    let d = i32::from(*a) - i32::from(*b);
                    sse_src_motion += (d * d) as u64;
                }
            }
            eprintln!(
                "  f{:2}: q={:3} bits={:6} PSNR={:5.2}dB sse(r,s)={:10} sse(r,s_-1)={:10} sse(s,s_-1)={:10} kf={}",
                i, pkt.q_index, pkt.actual_bits, psnr_y, sse_self, sse_vs_prev_src,
                sse_src_motion, pkt.frame_kind.is_key(),
            );
            rvp_total_bits += u64::from(pkt.actual_bits);
            rvp_total_sse += sse_self;
        }
        let rvp_actual_kbps = (rvp_total_bits as f64 * framerate) / (frames.len() as f64 * 1000.0);
        let avg_mse = rvp_total_sse as f64 / (frames.len() as f64 * (W * H) as f64);
        let avg_psnr = if avg_mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0 * 255.0 / avg_mse).log10()
        };
        eprintln!(
            "  RVP9 TOTAL: {} bits  actual={:.1} kbps  avg_psnr={:.2} dB",
            rvp_total_bits, rvp_actual_kbps, avg_psnr,
        );

        // libvpx side.
        let (Some(vpxenc), Some(vpxdec)) = (vpxenc_path(), vpxdec_path()) else {
            eprintln!("  [libvpx skipped — binaries not found]");
            continue;
        };
        let tmp = std::env::temp_dir();
        let raw = tmp.join("a0prime_foreman.yuv");
        let ivf = tmp.join("a0prime_foreman.ivf");
        let dec_y4m = tmp.join("a0prime_foreman.y4m");
        let _ = write_i420_raw(&raw, &frames);
        let mut cmd = Command::new(&vpxenc);
        cmd.args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--passes=1",
            "--i420",
            "-y",
        ]);
        cmd.arg(format!("--width={W}"));
        cmd.arg(format!("--height={H}"));
        cmd.arg(format!("--fps={}/1", framerate.round() as u32));
        cmd.arg(format!("--target-bitrate={}", tb / 1000));
        cmd.arg(format!("--limit={}", frames.len()));
        cmd.arg("-o");
        cmd.arg(ivf.as_os_str());
        cmd.arg(raw.as_os_str());
        let out = cmd.output().expect("vpxenc");
        if !out.status.success() {
            eprintln!("  [libvpx fail: {}]", String::from_utf8_lossy(&out.stderr));
            continue;
        }
        let ivf_bytes = std::fs::read(&ivf).unwrap();
        let sizes = parse_ivf_sizes(&ivf_bytes);
        let dec_out = Command::new(&vpxdec)
            .args([
                "--codec=vp9",
                "-o",
                dec_y4m.to_str().unwrap(),
                ivf.to_str().unwrap(),
            ])
            .output()
            .expect("vpxdec");
        if !dec_out.status.success() {
            eprintln!("  [libvpx dec fail]");
            continue;
        }
        let lv_y = read_y4m_yplanes(&dec_y4m, W, H, frames.len());
        let mut lv_total_bits: u64 = 0;
        let mut lv_total_sse: u64 = 0;
        for (i, ((src_y, _u, _v), recon_y)) in frames.iter().zip(lv_y.iter()).enumerate() {
            let psnr = psnr_db(src_y, recon_y);
            let bits = sizes.get(i).copied().unwrap_or(0) as u64 * 8;
            let mut sse: u64 = 0;
            for (a, b) in src_y.iter().zip(recon_y.iter()) {
                let d = i32::from(*a) - i32::from(*b);
                sse += (d * d) as u64;
            }
            eprintln!(
                "  LV f{:2}: bits={:6} PSNR={:5.2}dB sse={}",
                i, bits, psnr, sse,
            );
            lv_total_bits += bits;
            lv_total_sse += sse;
        }
        let lv_kbps = (lv_total_bits as f64 * framerate) / (frames.len() as f64 * 1000.0);
        let lv_mse = lv_total_sse as f64 / (frames.len() as f64 * (W * H) as f64);
        let lv_psnr = if lv_mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0 * 255.0 / lv_mse).log10()
        };
        eprintln!(
            "  LIBVPX TOTAL: {} bits  actual={:.1} kbps  avg_psnr={:.2} dB",
            lv_total_bits, lv_kbps, lv_psnr,
        );
    }
}
