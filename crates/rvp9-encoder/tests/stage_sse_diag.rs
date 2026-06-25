// STAGE-SSE DIAG — per-stage SSE oracle. stderr IS the output mechanism
// (this binary's job is to print a localization report).
#![allow(clippy::print_stderr)]

//! STAGE-SSE DIAG — localize the rvp9 encoder's matched-Q PSNR gap to
//! libvpx to a specific PIPELINE STAGE.
//!
//! Track-F iteration 6. Iters 1-5 ruled out the mode-decision levers (KF
//! cap was the only win; intra / lambda / skip-plumbing / model_rd-skip
//! all null). The combined iter4+5 finding was that the encoder is NOT
//! residual-bound at CBR — the gap lives in transform/quant precision or
//! reference-frame quality. This oracle measures, for a FIXED-Q encode of
//! `foreman_320x256_60f.y4m`, the per-frame **SSE-to-source at three
//! pipeline stages**, and compares rvp9's final reconstruction to
//! **vpxenc's** at the SAME quantizer:
//!
//!   1. **pred-SSE** — `SSE(source, prediction)` BEFORE any residual.
//!      How good is the motion-compensated / intra prediction? Captured
//!      via the whole-frame `diag_pred_y()` snapshot (predstat taps).
//!   2. **post-quant-SSE** — `SSE(source, prediction + dequant(quant(res)))`,
//!      the pre-loop-filter reconstruction the encoder commits. How much
//!      does quantization lose? Captured via `diag_prelf_recon()`.
//!   3. **post-recon-SSE** — `SSE(source, final reconstruction incl.
//!      in-loop deblock)`, the committed reference. Captured via
//!      `diag_last_recon_y()`; equals the decode (drift_diag SSE=0).
//!
//! The KEY comparison: rvp9 post-recon-SSE vs **vpxenc** post-recon-SSE
//! at the same internal qindex (`vpxenc --end-usage=q --cq-level=N`,
//! decoded with vpxdec). If vpxenc is closer to source at the same Q,
//! rvp9 loses quality in pred / transform / quant / loop-filter. The
//! within-rvp9 progression (pred → post-quant → post-recon) localizes
//! WHERE.
//!
//! Diagnostic-only; instrumentation adds read-only accessors. Run:
//! ```text
//! cargo test --release -p rvp9-encoder --test stage_sse_diag -- --ignored --nocapture
//! ```

use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{
    AqMode, EncoderConfig, RcMode, StreamEncoder, StreamFrameKind, YuvFrameRef, predstat,
};
use rvp9_harness::y4m::{YuvFrameBuf, read_cropped};
use std::path::{Path, PathBuf};
use std::process::Command;

const W: u32 = 320;
const H: u32 = 256;
const N_FRAMES: usize = 16;

/// Fixed internal qindex (full VP9 0..255 space). 80 maps onto libvpx
/// CLI quantizer 20 (`quantizer_to_qindex[N] == 4·N` over this range), a
/// mid-range op-point where the prediction is informative and the skip
/// fraction is low enough that pred-SSE covers nearly the whole frame.
const Q_POINTS: [u8; 2] = [80, 160];

fn yuv_view<'a>(f: &'a YuvFrameBuf, w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y: &f.0,
        u: &f.1,
        v: &f.2,
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

/// Y-plane SSE between two equal-length buffers.
fn sse_u8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = i64::from(*x) - i64::from(*y);
        s += (d * d) as u64;
    }
    s
}

/// Convert a Y-plane SSE over `pixels` samples to PSNR (dB).
fn psnr_from_sse(sse: u64, pixels: u64) -> f64 {
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / pixels as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Tighten a decoder plane (stride may exceed width) to `w × h`.
fn tighten(plane: &rvp9_core::frame::PlaneRef<'_>, w: usize, h: usize) -> Vec<u8> {
    let stride = plane.stride();
    let data = plane.data();
    let mut out = vec![0u8; w * h];
    for r in 0..h {
        out[r * w..r * w + w].copy_from_slice(&data[r * stride..r * stride + w]);
    }
    out
}

/// Per-frame aggregate of the three-stage Y SSE, plus coverage.
#[derive(Default, Clone, Copy)]
struct StageSse {
    pred_sse: u64,
    pred_px: u64,
    post_quant_sse: u64,
    post_recon_sse: u64,
}

/// Aggregate rvp9 stage measurement over a fixed-Q encode.
struct Rvp9Stages {
    /// Per-frame rows (for the table).
    per_frame: Vec<(StreamFrameKind, u8, usize, StageSse)>,
    /// Inter-frame aggregate.
    inter: StageSse,
    /// Keyframe aggregate.
    kf: StageSse,
    /// All-frame post-recon SSE (for the vpxenc comparison).
    all_recon_sse: u64,
    /// Total coded bytes.
    total_bytes: u64,
}

fn measure_rvp9(fixed_q: u8, frames: &[YuvFrameBuf], framerate: f64) -> Rvp9Stages {
    // Pin Q and enable the prediction-capture gate for this process.
    // SAFETY: this diagnostic test owns these process env knobs for the
    // duration of the test binary invocation.
    unsafe {
        std::env::set_var("RVP9_FIXED_Q", fixed_q.to_string());
        std::env::set_var("RVP9_E9_PREDSTAT", "1");
    }
    predstat::init_from_env();

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 100_000, // unused: RVP9_FIXED_Q overrides Q
        gop_length: N_FRAMES as u32 + 1, // single keyframe at frame 0
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
        stage4a_product_split_override: Default::default(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");
    let mut dec = Decoder::new();

    let w = W as usize;
    let h = H as usize;
    let y_pixels = (w * h) as u64;

    let mut per_frame = Vec::with_capacity(frames.len());
    let mut inter = StageSse::default();
    let mut kf = StageSse::default();
    let mut all_recon_sse = 0u64;
    let mut total_bytes = 0u64;

    for f in frames.iter() {
        let pkt = enc.encode_frame(&yuv_view(f, w)).expect("encode_frame");
        total_bytes += pkt.data.len() as u64;

        // Decode the emitted bitstream so post-recon-SSE is the true
        // decode (and so the drift invariant — enc stored ref == decode —
        // holds: see drift_diag).
        let parsed = decode_packet(&mut dec, &pkt.data);
        assert_eq!(parsed.len(), 1);
        let buf = dec.decode_frame(&parsed[0]).expect("decode_frame");
        let dec_y = tighten(&buf.y_plane(), w, h);

        // STAGE 3: post-recon-SSE (final, incl. loop filter == decode).
        let post_recon_sse = sse_u8(&f.0, &dec_y);

        // STAGE 2: post-quant-SSE (pre-LF recon = pred + dequant(quant)).
        let (prelf_y, _, _) = enc.diag_prelf_recon().expect("prelf snapshot present");
        let post_quant_sse = sse_u8(&f.0, prelf_y);

        // STAGE 1: pred-SSE (whole-frame Y prediction, pre-residual).
        let pred_y = enc.diag_pred_y().expect("diag_pred snapshot present");
        // The prediction buffer is sized w*h once the gate populates it.
        // If empty (gate off / no coded blocks), fall back to recon as a
        // visible sentinel — but with the gate ON it is always full.
        let (pred_sse, pred_px) = if pred_y.len() == w * h {
            (sse_u8(&f.0, pred_y), y_pixels)
        } else {
            (0, 0)
        };

        let row = StageSse {
            pred_sse,
            pred_px,
            post_quant_sse,
            post_recon_sse,
        };
        per_frame.push((pkt.frame_kind, pkt.q_index, pkt.data.len(), row));

        all_recon_sse += post_recon_sse;
        let agg = if pkt.frame_kind == StreamFrameKind::Inter {
            &mut inter
        } else {
            &mut kf
        };
        agg.pred_sse += pred_sse;
        agg.pred_px += pred_px;
        agg.post_quant_sse += post_quant_sse;
        agg.post_recon_sse += post_recon_sse;
    }

    // SAFETY: see the setup block above; this restores the fixed-Q env knob.
    unsafe { std::env::remove_var("RVP9_FIXED_Q") };

    Rvp9Stages {
        per_frame,
        inter,
        kf,
        all_recon_sse,
        total_bytes,
    }
}

fn libvpx_available() -> Option<PathBuf> {
    let p = PathBuf::from(r"D:\language\libvpx-stock\vpxenc.exe");
    if p.is_file() { Some(p) } else { None }
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

/// libvpx fixed-Q result: per-frame decoded Y SSE (post-recon) and total
/// coded bytes. `psnr_all` is the all-frame post-recon PSNR.
struct LibvpxResult {
    /// All-frame post-recon Y SSE.
    all_recon_sse: u64,
    /// All-frame post-recon Y PSNR (dB).
    psnr_all: f64,
    total_bytes: u64,
}

/// Drive libvpx at fixed internal qindex (CLI quantizer = qindex/4),
/// decode with vpxdec, and compute the post-recon Y SSE against source.
fn measure_libvpx(
    vpxenc: &Path,
    qindex: u8,
    framerate: f64,
    frames: &[YuvFrameBuf],
) -> Option<LibvpxResult> {
    let cli_q = (qindex / 4).clamp(0, 63);
    let tmp = std::env::temp_dir();
    let raw = tmp.join(format!("stagesse_{W}x{H}_q{qindex}.yuv"));
    let ivf = tmp.join(format!("stagesse_{W}x{H}_q{qindex}.ivf"));
    let dec_raw = tmp.join(format!("stagesse_{W}x{H}_q{qindex}_dec.yuv"));

    if write_i420_raw(&raw, frames).is_err() {
        return None;
    }

    let out = Command::new(vpxenc)
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--passes=1",
            "--end-usage=q",
            "--i420",
            "--aq-mode=0",
            "--lag-in-frames=0",
        ])
        .arg(format!("--width={W}"))
        .arg(format!("--height={H}"))
        .arg(format!("--fps={}/1", framerate as u32))
        .arg(format!("--cq-level={cli_q}"))
        .arg(format!("--kf-min-dist={N_FRAMES}"))
        .arg(format!("--kf-max-dist={N_FRAMES}"))
        .arg("-o")
        .arg(&ivf)
        .arg(&raw)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxenc failed q={qindex}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }

    let ivf_bytes = std::fs::metadata(&ivf).ok()?.len();

    let vpxdec = vpxenc.with_file_name("vpxdec.exe");
    let out = Command::new(&vpxdec)
        .args(["--i420", "-o"])
        .arg(&dec_raw)
        .arg(&ivf)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "    [libvpx skipped] vpxdec failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let dec_data = std::fs::read(&dec_raw).ok()?;
    let frame_size = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    let y_size = (W * H) as usize;
    if dec_data.len() < frame_size * frames.len() {
        return None;
    }
    let mut sse_total = 0u64;
    for (i, (orig_y, _, _)) in frames.iter().enumerate() {
        let dec_y = &dec_data[i * frame_size..i * frame_size + y_size];
        sse_total += sse_u8(orig_y, dec_y);
    }
    let psnr_all = psnr_from_sse(sse_total, frames.len() as u64 * y_size as u64);

    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&ivf);
    let _ = std::fs::remove_file(&dec_raw);

    Some(LibvpxResult {
        all_recon_sse: sse_total,
        psnr_all,
        total_bytes: ivf_bytes,
    })
}

fn corpus_path() -> Option<PathBuf> {
    let p = PathBuf::from("tests/corpus/foreman_320x256_60f.y4m");
    if p.is_file() {
        return Some(p);
    }
    let p2 = PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_320x256_60f.y4m");
    if p2.is_file() {
        return Some(p2);
    }
    None
}

#[test]
#[ignore = "STAGE-SSE DIAG — per-stage SSE oracle; run with `--ignored --nocapture`"]
fn stage_sse_diag_foreman() {
    let Some(path) = corpus_path() else {
        eprintln!("    [skip] foreman_320x256_60f.y4m corpus not available");
        return;
    };
    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES) {
        Ok((f, fr)) if f.len() == N_FRAMES => (f, fr),
        Ok((f, _)) => {
            eprintln!(
                "    [skip] corpus only yielded {}/{} frames",
                f.len(),
                N_FRAMES
            );
            return;
        }
        Err(e) => {
            eprintln!("    [skip] corpus read failed: {e}");
            return;
        }
    };

    let w = W as usize;
    let h = H as usize;
    let y_pixels = (w * h) as u64;

    eprintln!("STAGE-SSE DIAG — per-stage SSE oracle (foreman 320x256, {N_FRAMES} frames)");
    eprintln!("source: {} @ {framerate} fps", path.display());
    eprintln!(
        "Stages: 1=pred (source vs prediction)  2=post-quant (vs pre-LF recon)  3=post-recon (vs decode)"
    );
    eprintln!("================================================================================");

    let vpxenc = libvpx_available();

    for &q in Q_POINTS.iter() {
        let r = measure_rvp9(q, &frames, framerate);

        eprintln!();
        eprintln!("=== rvp9 @ fixed qindex {q} (cpu-used 7) ===");
        eprintln!(
            "  frame kind  q  bytes |  pred-SSE  pred-PSNR | post-quant-SSE  PSNR | post-recon-SSE  PSNR"
        );
        for (idx, (kind, qi, bytes, s)) in r.per_frame.iter().enumerate() {
            let pred_psnr = psnr_from_sse(s.pred_sse, s.pred_px.max(1));
            let pq_psnr = psnr_from_sse(s.post_quant_sse, y_pixels);
            let pr_psnr = psnr_from_sse(s.post_recon_sse, y_pixels);
            eprintln!(
                "  {idx:>3} {kind:>5?} {qi:>3} {bytes:>5} | {:>9} {:>7.2} | {:>13} {:>6.2} | {:>13} {:>6.2}",
                s.pred_sse, pred_psnr, s.post_quant_sse, pq_psnr, s.post_recon_sse, pr_psnr,
            );
        }

        // Aggregate progression (inter frames — the cleaner matched-Q
        // comparison; the KF is pinned to the same Q with no KF boost).
        let n_inter = (frames.len() - 1) as u64;
        let inter_pred_psnr = psnr_from_sse(r.inter.pred_sse, r.inter.pred_px.max(1));
        let inter_pq_psnr = psnr_from_sse(r.inter.post_quant_sse, n_inter * y_pixels);
        let inter_pr_psnr = psnr_from_sse(r.inter.post_recon_sse, n_inter * y_pixels);
        let kf_pred_psnr = psnr_from_sse(r.kf.pred_sse, r.kf.pred_px.max(1));
        let kf_pq_psnr = psnr_from_sse(r.kf.post_quant_sse, y_pixels);
        let kf_pr_psnr = psnr_from_sse(r.kf.post_recon_sse, y_pixels);

        eprintln!(
            "  --- INTER aggregate: pred {inter_pred_psnr:.2} dB -> post-quant {inter_pq_psnr:.2} dB -> post-recon {inter_pr_psnr:.2} dB"
        );
        eprintln!(
            "  --- KF    aggregate: pred {kf_pred_psnr:.2} dB -> post-quant {kf_pq_psnr:.2} dB -> post-recon {kf_pr_psnr:.2} dB"
        );
        eprintln!(
            "  --- pred coverage: inter {} / {} px ({:.1}%)",
            r.inter.pred_px,
            n_inter * y_pixels,
            100.0 * r.inter.pred_px as f64 / (n_inter * y_pixels) as f64
        );

        // STAGE TRANSITIONS (the localization): how much SSE does each
        // stage ADD relative to the previous? post-quant - pred is the
        // residual recovery (negative = residual HELPS); post-recon -
        // post-quant is the loop-filter delta.
        let rvp9_all_pr_psnr = psnr_from_sse(r.all_recon_sse, frames.len() as u64 * y_pixels);
        eprintln!(
            "  --- ALL-frame post-recon PSNR = {rvp9_all_pr_psnr:.2} dB  |  total bytes = {}",
            r.total_bytes
        );

        // KEY COMPARISON — rvp9 vs vpxenc post-recon at the SAME qindex.
        if let Some(ref vpxenc) = vpxenc {
            if let Some(lv) = measure_libvpx(vpxenc, q, framerate, &frames) {
                let dpsnr = lv.psnr_all - rvp9_all_pr_psnr;
                let dbytes = lv.total_bytes as i64 - r.total_bytes as i64;
                eprintln!(
                    "  === vpxenc @ same qindex {q}: post-recon PSNR = {:.2} dB  |  bytes = {}",
                    lv.psnr_all, lv.total_bytes
                );
                eprintln!(
                    "  === DELTA (vpxenc - rvp9): post-recon PSNR = {dpsnr:+.2} dB  |  bytes = {dbytes:+}"
                );
                eprintln!(
                    "  === vpxenc all-frame SSE = {}  vs rvp9 = {}  (ratio {:.3})",
                    lv.all_recon_sse,
                    r.all_recon_sse,
                    r.all_recon_sse as f64 / lv.all_recon_sse.max(1) as f64,
                );
            }
        } else {
            eprintln!(
                "  === [vpxenc not found at D:\\language\\libvpx-stock — skipping comparison]"
            );
        }
    }

    // Clear the gate so it doesn't leak into other in-process tests.
    // SAFETY: see the setup block above; this restores the diagnostic env knob.
    unsafe {
        std::env::remove_var("RVP9_E9_PREDSTAT");
    }
    predstat::init_from_env();

    eprintln!();
    eprintln!("LOCALIZATION READING GUIDE:");
    eprintln!(
        "  * pred-PSNR low + post-quant-PSNR >> pred-PSNR -> residual coding works; gap is NOT pred-bound."
    );
    eprintln!(
        "  * post-quant-PSNR << theoretical-for-qstep -> quantization too aggressive (zbin/round/dead-zone)."
    );
    eprintln!(
        "  * post-recon-PSNR << post-quant-PSNR more than libvpx LF -> loop-filter strength."
    );
    eprintln!(
        "  * vpxenc post-recon-PSNR >> rvp9 at same Q with FEWER bytes -> rvp9 dominated on rate-distortion."
    );
}
