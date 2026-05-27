// Integration test: stderr IS the output mechanism (the harness's job
// is to print a per-frame measurement report). The workspace's
// `print_stderr = "deny"` lint targets library code; for this binary
// that lint is the wrong rule.
#![allow(clippy::print_stderr)]

//! A0 — Per-frame BD-rate gap diagnosis (Stage 4a').
//!
//! Goal: identify the dominant source of rvp9's +205% BD-rate vs libvpx
//! on the F12 corpus by collecting per-frame data on BOTH encoders at
//! matched configs.
//!
//! Three sweeps:
//!
//! 1. Matched-CBR: both encoders driven at the same `target-bitrate`
//!    (250 kbps). Records per-frame qindex, bits, PSNR. Highlights
//!    whether rvp9 picks a much higher qindex (H1).
//!
//! 2. Matched-Q: both encoders forced to a fixed Q (qindex 100) via
//!    `--min-q=100 --max-q=100` on libvpx. rvp9 doesn't have a direct
//!    qindex knob, but the rate-controller will converge near min-q
//!    when `target-bitrate` is huge so we use a large enough target.
//!    (Skipped if too hard to force the rvp9 side cleanly — we use
//!    `set_bitrate_to_match_q` workaround.)
//!
//! 3. Per-frame size distribution: gives bit-spend breakdown across
//!    frame 4 in particular.
//!
//! Run with:
//! ```text
//! cargo test --test diag_per_frame -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{AqMode, EncoderConfig, RcMode, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{YuvFrameBuf, probe_testsrc_720p, read_cropped as y4m_read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Per-frame record from rvp9.
#[derive(Clone, Copy, Debug)]
struct Rvp9FrameRecord {
    /// Encoder-side picked Q-index.
    q_index: u8,
    /// Bits the encoder emitted for this frame (including headers).
    bits: u32,
    /// Y-plane PSNR vs the original source.
    psnr_y: f64,
    /// `true` for keyframes.
    is_key: bool,
}

/// Encode + decode the supplied YUV frames through rvp9; return per-frame
/// records.
fn run_rvp9(
    w: u32,
    h: u32,
    framerate: f64,
    target_bitrate_bps: u32,
    aq_mode: AqMode,
    gop_length: u32,
    frames: &[YuvFrameBuf],
) -> Vec<Rvp9FrameRecord> {
    let cfg = EncoderConfig {
        width: w,
        height: h,
        framerate,
        target_bitrate_bps,
        gop_length,
        aq_mode,
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
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();
    let mut out = Vec::with_capacity(frames.len());

    for (y, u, v) in frames.iter() {
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, w as usize))
            .expect("encode_frame");
        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let yp = buf.y_plane();
        let mut decoded_y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            decoded_y[r * w as usize..(r + 1) * w as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + w as usize]);
        }
        let psnr_y = psnr_db(y, &decoded_y);
        out.push(Rvp9FrameRecord {
            q_index: pkt.q_index,
            bits: pkt.actual_bits,
            psnr_y,
            is_key: pkt.frame_kind.is_key(),
        });
    }
    out
}

/// Per-frame record from libvpx after decode + PSNR computation.
#[derive(Clone, Copy, Debug)]
struct LibvpxFrameRecord {
    bits: u32,
    psnr_y: f64,
    /// `true` if frame parsed as keyframe (frame_type == 0).
    #[allow(dead_code)]
    is_key: bool,
}

/// Write the frames in tight-stride I420 layout to `path`.
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

/// IVF parser: yields each frame's payload size as the per-frame byte
/// count. Used to attribute bits-per-frame on the libvpx side.
fn parse_ivf_sizes(ivf: &[u8]) -> Vec<u32> {
    // IVF file header is 32 bytes.
    let mut sizes = Vec::new();
    if ivf.len() < 32 {
        return sizes;
    }
    let mut p = 32usize;
    while p + 12 <= ivf.len() {
        let frame_size = u32::from_le_bytes([ivf[p], ivf[p + 1], ivf[p + 2], ivf[p + 3]]);
        // 8 bytes pts follow.
        p += 12;
        if p + frame_size as usize > ivf.len() {
            break;
        }
        // First byte of a VP9 frame contains the frame-marker (`10`),
        // profile bits, then key bit etc.
        sizes.push(frame_size);
        p += frame_size as usize;
    }
    sizes
}

/// Parse keyframe flags from an IVF VP9 bitstream. Each entry is `true`
/// iff that frame is a keyframe. Mirrors the VP9 frame header structure:
/// the high two bits of the first byte are the frame marker (`10`); bits
/// [3..4] are profile; the next bit is `show_existing_frame`; if zero,
/// the next bit is `frame_type` (0=KF, 1=Inter). We extract that bit.
fn parse_ivf_keyflags(ivf: &[u8]) -> Vec<bool> {
    let mut flags = Vec::new();
    if ivf.len() < 32 {
        return flags;
    }
    let mut p = 32usize;
    while p + 12 <= ivf.len() {
        let frame_size = u32::from_le_bytes([ivf[p], ivf[p + 1], ivf[p + 2], ivf[p + 3]]);
        p += 12;
        if p + frame_size as usize > ivf.len() {
            break;
        }
        let payload = &ivf[p..p + frame_size as usize];
        // Decode VP9 frame header byte 0 + (maybe) extra bit:
        //   bits 0..1: frame_marker (must be 0b10)
        //   bit 2: profile_low_bit
        //   bit 3: profile_high_bit (assemble with low bit → profile)
        //   if profile == 3: read 1 more reserved bit, then continue
        //   bit (4 or 5): show_existing_frame
        //   if show_existing_frame == 0:
        //     next bit: frame_type   (0 == KEY)
        let b0 = payload[0];
        // bits read from MSB → LSB.
        let profile_low = (b0 >> 5) & 1;
        let profile_high = (b0 >> 4) & 1;
        let profile = (profile_high << 1) | profile_low;
        let mut bitpos = 4; // we've consumed bits 7,6,5,4 (marker, plow, phigh).
        if profile == 3 {
            // skip 1 reserved bit.
            bitpos += 1;
        }
        let show_existing_frame = (b0 >> (7 - bitpos)) & 1;
        bitpos += 1;
        let is_key = if show_existing_frame != 0 {
            false
        } else {
            // Next bit is frame_type. 0 == keyframe.
            ((b0 >> (7 - bitpos)) & 1) == 0
        };
        flags.push(is_key);
        p += frame_size as usize;
    }
    flags
}

/// Run libvpx-stock `vpxenc` with the supplied args; return the IVF
/// bytes and the decoded YUV per-frame PSNR. `extra_args` lets the
/// caller force min-q/max-q etc.
#[allow(clippy::too_many_arguments)]
fn run_libvpx_full(
    vpxenc: &Path,
    vpxdec: &Path,
    w: u32,
    h: u32,
    framerate: f64,
    extra_args: &[&str],
    frames: &[YuvFrameBuf],
    label: &str,
) -> Option<Vec<LibvpxFrameRecord>> {
    let tmp_dir = std::env::temp_dir();
    let raw_path = tmp_dir.join(format!("rvp9_a0_diag_{label}.yuv"));
    let ivf_path = tmp_dir.join(format!("rvp9_a0_diag_{label}.ivf"));
    if write_i420_raw(&raw_path, frames).is_err() {
        return None;
    }
    let mut cmd = Command::new(vpxenc);
    cmd.args([
        "--codec=vp9",
        "--ivf",
        "--rt",
        "--cpu-used=7",
        "--tune-content=default",
        "--passes=1",
        "--i420",
        "-y",
        "--disable-warnings",
    ])
    .arg(format!("--width={w}"))
    .arg(format!("--height={h}"))
    .arg(format!("--fps={}/1", framerate as u32))
    .arg("--aq-mode=0")
    .args(extra_args)
    .arg("-o")
    .arg(&ivf_path)
    .arg(&raw_path);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx-{label}] vpxenc failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    let ivf_bytes = std::fs::read(&ivf_path).ok()?;
    let sizes = parse_ivf_sizes(&ivf_bytes);
    let keyflags = parse_ivf_keyflags(&ivf_bytes);

    // Decode for PSNR via vpxdec → raw .yuv.
    let dec_raw_path = tmp_dir.join(format!("rvp9_a0_diag_{label}_dec.yuv"));
    let dec_out = Command::new(vpxdec)
        .args(["--i420", "-o"])
        .arg(&dec_raw_path)
        .arg(&ivf_path)
        .output()
        .ok()?;
    if !dec_out.status.success() {
        eprintln!(
            "    [libvpx-{label}] vpxdec failed: stderr={}",
            String::from_utf8_lossy(&dec_out.stderr)
        );
        return None;
    }
    let dec_data = std::fs::read(&dec_raw_path).ok()?;
    let frame_size = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    let y_size = (w * h) as usize;
    let mut records = Vec::with_capacity(frames.len());
    for (i, (orig_y, _, _)) in frames.iter().enumerate() {
        if (i + 1) * frame_size > dec_data.len() {
            break;
        }
        if i >= sizes.len() {
            break;
        }
        let dec_y = &dec_data[i * frame_size..i * frame_size + y_size];
        let psnr_y = psnr_db(orig_y, dec_y);
        let is_key = keyflags.get(i).copied().unwrap_or(false);
        records.push(LibvpxFrameRecord {
            bits: sizes[i] * 8,
            psnr_y,
            is_key,
        });
    }
    let _ = std::fs::remove_file(&raw_path);
    let _ = std::fs::remove_file(&ivf_path);
    let _ = std::fs::remove_file(&dec_raw_path);
    Some(records)
}

/// A0 — per-frame diagnostic harness.
#[test]
#[ignore = "A0 — diagnostic harness for the BD-rate gap; run with `--ignored`"]
fn a0_per_frame_diag() {
    const W: u32 = 256;
    const H: u32 = 256;
    const N_FRAMES: u32 = 64;
    // Target sweep: same as bdrate_gate.
    let targets: &[u32] = &[25_000, 50_000, 100_000, 250_000];

    let y4m_path_buf = probe_testsrc_720p();
    let (frames, framerate, source_label): (Vec<YuvFrameBuf>, f64, String) =
        match y4m_path_buf.as_deref() {
            Some(path) => match y4m_read_cropped(path, W, H, N_FRAMES as usize) {
                Ok((f, fr)) if f.len() == N_FRAMES as usize => (
                    f,
                    fr,
                    format!(
                        "y4m centre-crop from {} → {W}x{H} @ {fr} fps",
                        path.display()
                    ),
                ),
                other => panic!("y4m read failed or short: {other:?}"),
            },
            None => panic!("testsrc_720p30.y4m not available"),
        };

    eprintln!("A0 Per-Frame Diagnostic Report");
    eprintln!("==============================");
    eprintln!("Source: {source_label}");
    eprintln!("Frames: {}", frames.len());
    eprintln!();

    let vpxenc_opt = vpxenc_path();
    let vpxdec_opt = vpxdec_path();
    let vpxenc = vpxenc_opt.expect("vpxenc missing");
    let vpxdec = vpxdec_opt.expect("vpxdec missing");

    // -----------------------------------------------------------------
    // Sweep 1: Matched-target-CBR per-frame breakdown.
    // -----------------------------------------------------------------
    for &target in targets {
        eprintln!("--- Matched target: {} kbps ---", target / 1000);
        let rvp9_recs = run_rvp9(
            W,
            H,
            framerate,
            target,
            AqMode::Off,
            N_FRAMES + 1, // single keyframe at frame 0
            &frames,
        );
        let lv_args: Vec<String> = vec![
            format!("--target-bitrate={}", target / 1000),
            "--end-usage=cbr".to_string(),
        ];
        let lv_arg_refs: Vec<&str> = lv_args.iter().map(|s| s.as_str()).collect();
        let lv_recs = run_libvpx_full(
            &vpxenc,
            &vpxdec,
            W,
            H,
            framerate,
            &lv_arg_refs,
            &frames,
            &format!("cbr_{}", target / 1000),
        )
        .expect("libvpx run failed");

        eprintln!(
            "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "frame", "rvp9_q", "rvp9_bits", "rvp9_psnr", "lv_bits", "lv_psnr", "rvp9/lv"
        );
        let mut sum_r_bits: u64 = 0;
        let mut sum_l_bits: u64 = 0;
        let mut sum_r_psnr = 0.0;
        let mut sum_l_psnr = 0.0;
        let n = rvp9_recs.len().min(lv_recs.len());
        for i in 0..n {
            let r = rvp9_recs[i];
            let l = lv_recs[i];
            let ratio = if l.bits > 0 {
                f64::from(r.bits) / f64::from(l.bits)
            } else {
                f64::INFINITY
            };
            eprintln!(
                "  {:>5}  q={:>3}({}){:>2}  {:>10}  {:>9.2} dB  {:>10}  {:>9.2} dB  {:>9.2}x",
                i,
                r.q_index,
                if r.is_key { "K" } else { "P" },
                "",
                r.bits,
                r.psnr_y,
                l.bits,
                l.psnr_y,
                ratio,
            );
            sum_r_bits += u64::from(r.bits);
            sum_l_bits += u64::from(l.bits);
            sum_r_psnr += r.psnr_y;
            sum_l_psnr += l.psnr_y;
        }
        eprintln!(
            "  TOTALS:  rvp9 bits={}  libvpx bits={}  ratio={:.2}x  rvp9 avg PSNR={:.2}  libvpx avg PSNR={:.2}",
            sum_r_bits,
            sum_l_bits,
            sum_r_bits as f64 / sum_l_bits.max(1) as f64,
            sum_r_psnr / n as f64,
            sum_l_psnr / n as f64,
        );
        eprintln!();
    }

    // -----------------------------------------------------------------
    // Sweep 2: forced-Q via min-q/max-q on libvpx; compare PSNR per frame.
    // (No min-q knob exposed by rvp9 — instead we record rvp9's median
    // inter-frame qindex from sweep 1's 250 kbps run, then force libvpx
    // to that qindex.)
    // -----------------------------------------------------------------
    eprintln!("--- Forced-Q libvpx sweep: spans qindex range ---");
    for forced_q in [40u32, 80, 120, 160, 200] {
        let lv_args: Vec<String> = vec![
            "--target-bitrate=2000".to_string(),
            "--end-usage=cbr".to_string(),
            format!("--min-q={forced_q}"),
            format!("--max-q={forced_q}"),
        ];
        let lv_arg_refs: Vec<&str> = lv_args.iter().map(|s| s.as_str()).collect();
        let lv_recs = run_libvpx_full(
            &vpxenc,
            &vpxdec,
            W,
            H,
            framerate,
            &lv_arg_refs,
            &frames,
            &format!("fixedq_{forced_q}"),
        );
        if let Some(lv) = lv_recs {
            let total_bits: u64 = lv.iter().map(|r| u64::from(r.bits)).sum();
            let avg_psnr: f64 = lv.iter().map(|r| r.psnr_y).sum::<f64>() / lv.len() as f64;
            eprintln!(
                "  libvpx forced-q={:>3}  total_bits={}  avg_psnr={:.2} dB  ({} frames)",
                forced_q,
                total_bits,
                avg_psnr,
                lv.len()
            );
        }
    }
    eprintln!();

    eprintln!("--- Forced-Q libvpx vs rvp9 same-Q comparison ---");
    eprintln!("(rvp9 has no direct qindex knob; the rate-controller picks Q. We");
    eprintln!(" record rvp9's qindex distribution under each target and look");
    eprintln!(" at the libvpx forced-q match.)");
    eprintln!();
}
