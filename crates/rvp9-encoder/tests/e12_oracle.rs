// E12 — search-failure-vs-prediction-ceiling oracle harness. stderr IS
// the output mechanism (this binary's job is to print a report).
#![allow(clippy::print_stderr)]

//! E12 — the decisive oracle. Drives foreman_cif (256×256 crop, 64
//! frames) through rvp9 at a fixed Q (q=120 primary) with
//! `RVP9_E12_ORACLE` on, snapshotting the per-frame oracle aggregate
//! (`e12stat`) after each encoded frame. Reports:
//!
//!   * per-frame prediction PSNR: (0,0) / rvp9-found / exhaustive-oracle,
//!     split FIRST inter frame (frame 1, predicting the fresh keyframe)
//!     vs LATER inter frames — discriminates reference cascade;
//!   * the oracle-best-MV table (sample rows printed by the module);
//!   * how often the oracle-best is sub-pel and beats rvp9-found by ≥2 dB.
//!
//! Verdict logic:
//!   * oracle ≫ rvp9-found  → SEARCH FAILURE (tractable next fix).
//!   * oracle ≈ rvp9-found ≈ 26 dB → PREDICTION CEILING (deeper story).
//!
//! Run:
//! ```text
//! cargo test --release --test e12_oracle -- --ignored --nocapture
//! ```

use rvp9_decoder::Decoder;
use rvp9_encoder::{
    e12stat, AqMode, EncoderConfig, RcMode, StreamEncoder, StreamFrameKind, YuvFrameRef,
};
use rvp9_harness::y4m::{probe_foreman_cif, read_cropped, YuvFrameBuf};

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: u32 = 64;
const Q: u8 = 120;

fn yuv_view(f: &YuvFrameBuf, w: usize) -> YuvFrameRef<'_> {
    YuvFrameRef {
        y: &f.0,
        u: &f.1,
        v: &f.2,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

fn psnr(sse: u64, px: u64) -> f64 {
    if sse == 0 || px == 0 {
        return 100.0;
    }
    let mse = sse as f64 / px as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
#[ignore = "E12 — search-failure-vs-ceiling oracle; run with `--ignored --nocapture`"]
fn e12_oracle_foreman() {
    let Some(path) = probe_foreman_cif() else {
        eprintln!("    [skip] foreman_cif.y4m not found at tests/corpus/foreman_cif.y4m");
        return;
    };
    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES as usize) {
        Ok((f, fr)) if f.len() == N_FRAMES as usize => (f, fr),
        Ok((f, _)) => {
            eprintln!("    [skip] foreman_cif only yielded {}/{} frames", f.len(), N_FRAMES);
            return;
        }
        Err(e) => {
            eprintln!("    [skip] foreman_cif read failed: {e}");
            return;
        }
    };

    unsafe {
        std::env::set_var("RVP9_FIXED_Q", Q.to_string());
        std::env::set_var("RVP9_E12_ORACLE", "1");
    }
    e12stat::init_from_env();

    eprintln!("E12 Oracle — foreman_cif {W}x{H}, {N_FRAMES} frames, fixed q={Q}");
    eprintln!("source: {} @ {framerate} fps", path.display());
    eprintln!("==================================================================");

    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 100_000, // unused under RVP9_FIXED_Q
        gop_length: N_FRAMES + 1,
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

    // Per-frame oracle aggregates (skip the keyframe; oracle only fires
    // on inter leaves).
    let mut first_inter: Option<e12stat::E12Stats> = None;
    let mut later = e12stat::E12Stats::default();
    let mut frame_lines: Vec<String> = Vec::new();

    for (i, f) in frames.iter().enumerate() {
        let _ = e12stat::take_snapshot();
        let pkt = enc.encode_frame(&yuv_view(f, W as usize)).expect("encode_frame");
        let snap = e12stat::take_snapshot();

        // Decode to keep the encoder/decoder ref chain alive (the recon
        // ref the next inter frame predicts from must match the decoded
        // output; the e9 harness does the same).
        let parsed = dec.parse_packet(&pkt.data).expect("parse_packet");
        let pf = parsed.frames();
        assert_eq!(pf.len(), 1);
        let _ = dec.decode_frame(&pf[0]).expect("decode_frame");

        if pkt.frame_kind == StreamFrameKind::Inter && snap.n > 0 {
            frame_lines.push(format!(
                "  frame {:>2}: n={:>3} ZERO={:.2}dB shipped={:.2}dB NEWMVraw={:.2}dB oracle={:.2}dB \
                 | subpel={}/{} oracle-beats-shipped(≥2dB)={}/{}",
                i,
                snap.n,
                psnr(snap.sse_zero, snap.px),
                psnr(snap.sse_rvp9, snap.px),
                psnr(snap.sse_newmv, snap.px),
                psnr(snap.sse_oracle, snap.px),
                snap.oracle_subpel,
                snap.n,
                snap.oracle_beats_rvp9,
                snap.n,
            ));
            if first_inter.is_none() {
                first_inter = Some(snap);
            } else {
                later.n += snap.n;
                later.sse_zero += snap.sse_zero;
                later.sse_rvp9 += snap.sse_rvp9;
                later.sse_newmv += snap.sse_newmv;
                later.sse_oracle_fp += snap.sse_oracle_fp;
                later.sse_oracle += snap.sse_oracle;
                later.px += snap.px;
                later.oracle_subpel += snap.oracle_subpel;
                later.oracle_beats_rvp9 += snap.oracle_beats_rvp9;
            }
        }
    }

    unsafe {
        std::env::remove_var("RVP9_FIXED_Q");
        std::env::remove_var("RVP9_E12_ORACLE");
    }

    eprintln!();
    eprintln!("=== Per-frame prediction PSNR (sampled inter leaves) ===");
    for line in &frame_lines {
        eprintln!("{line}");
    }

    eprintln!();
    eprintln!("=== FIRST inter frame vs LATER inter frames (cascade test) ===");
    if let Some(fi) = first_inter {
        eprintln!(
            "  FIRST (predicts fresh KF): n={} ZERO={:.2}dB shipped={:.2}dB NEWMVraw={:.2}dB oracle={:.2}dB",
            fi.n,
            psnr(fi.sse_zero, fi.px),
            psnr(fi.sse_rvp9, fi.px),
            psnr(fi.sse_newmv, fi.px),
            psnr(fi.sse_oracle, fi.px),
        );
    }
    eprintln!(
        "  LATER (mid-GOP, n={}): ZERO={:.2}dB shipped={:.2}dB NEWMVraw={:.2}dB \
         oracle_fullpel={:.2}dB oracle_subpel={:.2}dB",
        later.n,
        psnr(later.sse_zero, later.px),
        psnr(later.sse_rvp9, later.px),
        psnr(later.sse_newmv, later.px),
        psnr(later.sse_oracle_fp, later.px),
        psnr(later.sse_oracle, later.px),
    );

    eprintln!();
    eprintln!("=== Oracle verdict aggregate (later inter frames) ===");
    let oracle_db = psnr(later.sse_oracle, later.px);
    let oracle_fp_db = psnr(later.sse_oracle_fp, later.px);
    let shipped_db = psnr(later.sse_rvp9, later.px);
    let newmv_db = psnr(later.sse_newmv, later.px);
    eprintln!(
        "  oracle-shipped = {:+.2} dB | oracle-NEWMVraw = {:+.2} dB | NEWMVraw-shipped = {:+.2} dB",
        oracle_db - shipped_db,
        oracle_db - newmv_db,
        newmv_db - shipped_db,
    );
    eprintln!(
        "  full-pel split: oracle_fullpel-NEWMVraw = {:+.2} dB | sub-pel adds = {:+.2} dB \
         (oracle_full+sub - oracle_full)",
        oracle_fp_db - newmv_db,
        oracle_db - oracle_fp_db,
    );
    eprintln!(
        "  oracle sub-pel: {}/{} ({:.0}%) | oracle beats shipped by ≥2dB: {}/{} ({:.0}%)",
        later.oracle_subpel,
        later.n,
        100.0 * later.oracle_subpel as f64 / later.n.max(1) as f64,
        later.oracle_beats_rvp9,
        later.n,
        100.0 * later.oracle_beats_rvp9 as f64 / later.n.max(1) as f64,
    );
    eprintln!();
    eprintln!("=== DECISIVE VERDICT ===");
    if oracle_db < 30.0 {
        eprintln!("  PREDICTION CEILING — even the exhaustive oracle can't beat ~mid-20s dB.");
    } else if oracle_db - shipped_db >= 3.0 {
        // A reachable MV exists; localise WHERE it's lost.
        if newmv_db - shipped_db >= 2.0 {
            eprintln!(
                "  SEARCH-FAILURE / RD-REJECTION: NEWMV's RAW search DOES find a much-better\n  \
                 MV ({:+.2}dB over shipped) but the RD argmin SHIPS the worse (0,0) — the\n  \
                 cost model / mode-RD rejects it. The picker, not the search kernel.",
                newmv_db - shipped_db,
            );
        } else {
            eprintln!(
                "  SEARCH-FAILURE / NON-CONVERGENCE: NEWMV's RAW search output is no better\n  \
                 than shipped ({:+.2}dB) yet the oracle reaches {:+.2}dB — the diamond/sub-pel\n  \
                 search itself fails to converge to the reachable good MV.",
                newmv_db - shipped_db,
                oracle_db - shipped_db,
            );
        }
    } else {
        eprintln!("  MIXED — inspect per-frame rows.");
    }
}
