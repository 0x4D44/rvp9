// F.0.2 — per-block oracle harness binary. stdout/stderr IS the
// output mechanism (artifacts go to target/track_f_oracle/).
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! F.0.2 — per-(frame, block) libvpx-vs-rvp9 picker-decision oracle.
//!
//! Drives the rvp9 leg (via [`StreamEncoder`] with the
//! `RVP9_F02_ORACLE=1` thread-local sink) and the libvpx leg (via the
//! patched `vpxenc.exe` produced by `scripts/build_f02_libvpx.ps1`)
//! over the same foreman_cif input, then emits:
//!
//! - A per-frame summary table (markdown) per
//!   `target/track_f_oracle/<run>/summary.md`.
//! - The raw paired (rvp9, libvpx) records per
//!   `target/track_f_oracle/<run>/foreman_<kbps>kbps_raw.jsonl`.
//!
//! When the patched libvpx binary is missing, the binary errors out
//! loudly with the exact PowerShell command to bring it up (HLD §3.3).
//! When the corpus file is missing, same behaviour.
//!
//! Usage (HLD §4):
//! ```text
//! cargo run -p rvp9-harness --release --bin f02-block-oracle -- \
//!     --foreman tests/corpus/foreman_cif.y4m \
//!     --vpxenc third_party/libvpx/build_f02/vpxenc.exe \
//!     --width 256 --height 256 \
//!     --fps 30 \
//!     --frames 64 \
//!     --cpu-used 6 \
//!     --threads 1 --tile-columns 6 \
//!     --kbps 25,50,100,250 \
//!     --aq-mode 3 \
//!     --vbr \
//!     --fixed-q 120 \
//!     --strict \
//!     --out target/track_f_oracle/
//! ```

use rvp9_encoder::f02_oracle::{self, BlockDecision};
use rvp9_encoder::{
    AqMode, EncoderConfig, EncoderParallelism, RcMode, Stage4aProductSplitOverride, StreamEncoder,
    YuvFrameRef,
};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::f02_schema::{compare_decisions, parse_jsonl_stream};
use rvp9_harness::y4m::read_cropped;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_W: u32 = 256;
const DEFAULT_H: u32 = 256;
const DEFAULT_FRAMES: u32 = 64;
const PRODUCT_GOP_LENGTH: u32 = 60;

#[derive(Copy, Clone)]
struct OracleEncodeConfig {
    width: u32,
    height: u32,
    frames: u32,
    fps: Option<u32>,
    cpu_used: i32,
    threads: u32,
    tile_columns: u8,
    kbps: u32,
    vbr: bool,
    aq_mode: u8,
}

fn oracle_parallelism(encode: OracleEncodeConfig) -> EncoderParallelism {
    EncoderParallelism {
        max_threads: encode.threads,
        requested_log2_tile_cols: encode.tile_columns,
    }
}

fn oracle_fps(encode: OracleEncodeConfig, source_fps: f64) -> f64 {
    encode.fps.map_or(source_fps, f64::from)
}

fn arg(args: &[String], key: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == key {
            return iter.next().cloned();
        }
        if let Some(stripped) = a.strip_prefix(&format!("{key}=")) {
            return Some(stripped.to_string());
        }
    }
    None
}

fn parse_cpu_used(args: &[String]) -> Result<i32, String> {
    let cpu_used = arg(args, "--cpu-used")
        .map(|value| {
            value
                .parse::<i32>()
                .map_err(|_| format!("--cpu-used must be an integer, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(7);
    if !(6..=8).contains(&cpu_used) {
        return Err(format!(
            "--cpu-used must be one of 6, 7, or 8, got {cpu_used}"
        ));
    }
    Ok(cpu_used)
}

fn parse_aq_mode(args: &[String]) -> Result<u8, String> {
    let aq_mode = arg(args, "--aq-mode")
        .map(|value| {
            value
                .parse::<u8>()
                .map_err(|_| format!("--aq-mode must be 0 or 3, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(0);
    if !matches!(aq_mode, 0 | 3) {
        return Err(format!("--aq-mode must be 0 or 3, got {aq_mode}"));
    }
    Ok(aq_mode)
}

fn encoder_aq_mode(aq_mode: u8) -> AqMode {
    if aq_mode == 3 {
        AqMode::CyclicRefresh
    } else {
        AqMode::Off
    }
}

fn rvp9_rate_config(vbr: bool) -> (u32, RcMode) {
    if vbr {
        (PRODUCT_GOP_LENGTH, RcMode::Vbr)
    } else {
        (PRODUCT_GOP_LENGTH, RcMode::Cbr)
    }
}

fn split_override(strict: bool) -> Stage4aProductSplitOverride {
    if strict {
        Stage4aProductSplitOverride::ForceStrict
    } else {
        Stage4aProductSplitOverride::Auto
    }
}

fn fixed_q_cli_range(qindex: u8) -> Option<(u8, u8)> {
    let cli_q = qindex.checked_div(4)?;
    if !qindex.is_multiple_of(4) || cli_q > 55 {
        return None;
    }
    Some((cli_q, cli_q + 8))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let foreman_path = PathBuf::from(
        arg(&argv, "--foreman").unwrap_or_else(|| "tests/corpus/foreman_cif.y4m".into()),
    );
    let vpxenc_path = PathBuf::from(
        arg(&argv, "--vpxenc").unwrap_or_else(|| r"third_party\libvpx\build_f02\vpxenc.exe".into()),
    );
    let frames: u32 = arg(&argv, "--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_FRAMES);
    let width: u32 = arg(&argv, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_W);
    let height: u32 = arg(&argv, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_H);
    let fps: Option<u32> = arg(&argv, "--fps").and_then(|s| s.parse().ok());
    let cpu_used = parse_cpu_used(&argv)?;
    let aq_mode = parse_aq_mode(&argv)?;
    let threads: u32 = arg(&argv, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let tile_columns: u8 = arg(&argv, "--tile-columns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let kbps_list: Vec<u32> = arg(&argv, "--kbps")
        .unwrap_or_else(|| "100".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let out_dir =
        PathBuf::from(arg(&argv, "--out").unwrap_or_else(|| "target/track_f_oracle".into()));
    let no_libvpx = argv.iter().any(|a| a == "--no-libvpx");
    let report_only = argv.iter().any(|a| a == "--report-only");
    let fixed_q = arg(&argv, "--fixed-q")
        .map(|value| {
            value
                .parse::<u8>()
                .map_err(|_| format!("--fixed-q must be an integer qindex, got {value:?}"))
        })
        .transpose()?;
    if let Some(qindex) = fixed_q
        && fixed_q_cli_range(qindex).is_none()
    {
        return Err(
            format!("--fixed-q must be divisible by 4 and in 0..=220, got {qindex}").into(),
        );
    }
    // Fixed-Q diagnostics must retain the production Stage 4a VBR picker
    // calibration. RVP9_FIXED_Q bypasses only the controller's chosen Q.
    let vbr = argv.iter().any(|a| a == "--vbr") || fixed_q.is_some();
    let strict = argv.iter().any(|a| a == "--strict");

    if !foreman_path.exists() {
        eprintln!(
            "ERROR: foreman corpus not found at {}",
            foreman_path.display()
        );
        eprintln!("  Run scripts/fetch-corpus.{{ps1,sh}} to fetch it, then re-run.");
        std::process::exit(1);
    }
    if !no_libvpx && !vpxenc_path.exists() {
        eprintln!(
            "ERROR: patched vpxenc.exe not found at {}",
            vpxenc_path.display()
        );
        eprintln!("  Build it with:");
        eprintln!("    pwsh -File scripts/build_f02_libvpx.ps1");
        eprintln!("  Or pass --no-libvpx to drive only the rvp9 leg.");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&out_dir)?;
    let run_dir = out_dir.join(format!("run_{}", chrono_like_timestamp()));
    std::fs::create_dir_all(&run_dir)?;

    println!("F.0.2 per-block oracle harness");
    println!("  foreman   : {}", foreman_path.display());
    if no_libvpx {
        println!("  libvpx    : SKIPPED (--no-libvpx)");
    } else {
        println!("  libvpx    : {}", vpxenc_path.display());
    }
    println!("  frames    : {}", frames);
    println!("  geometry  : {width}x{height}");
    println!(
        "  fps       : {}",
        fps.map_or_else(|| "source header".to_string(), |value| value.to_string())
    );
    println!("  cpu-used  : {cpu_used}");
    println!("  threads   : {threads}");
    println!("  tile cols : log2 request {tile_columns}");
    println!("  bitrates  : {:?} kbps", kbps_list);
    println!("  rate mode : {}", if vbr { "VBR" } else { "CBR" });
    println!("  AQ mode   : {aq_mode}");
    println!(
        "  fixed Q   : {}",
        fixed_q.map_or_else(|| "off".to_string(), |q| q.to_string())
    );
    println!(
        "  split     : {}",
        if strict { "strict" } else { "product" }
    );
    println!("  out dir   : {}", run_dir.display());
    println!(
        "  policy    : {}",
        if report_only {
            "report-only (mismatches do not fail)"
        } else {
            "strict (mismatches fail)"
        }
    );

    // Enable the rvp9 leg's oracle. Once enabled at process startup
    // for THIS process; the harness driver's encoder runs are the
    // only consumers, so no measurement leakage to other tools.
    // SAFETY: process-local; no other thread races on this var.
    unsafe { std::env::set_var("RVP9_F02_ORACLE", "1") };
    if let Some(qindex) = fixed_q {
        // SAFETY: this single-threaded diagnostic process owns the
        // environment and sets the override before constructing an encoder.
        unsafe { std::env::set_var("RVP9_FIXED_Q", qindex.to_string()) };
    }
    f02_oracle::init_from_env();

    let mut strict_mismatches = 0usize;
    for &kbps in &kbps_list {
        println!("\n=== {} kbps target bitrate ===", kbps);
        let rvp9_ivf = run_dir.join(format!("foreman_{}kbps_rvp9.ivf", kbps));
        let encode = OracleEncodeConfig {
            width,
            height,
            frames,
            fps,
            cpu_used,
            threads,
            tile_columns,
            kbps,
            vbr,
            aq_mode,
        };
        let rvp9_records = drive_rvp9_leg(&foreman_path, encode, strict, &rvp9_ivf)?;
        println!(
            "  rvp9: {} per-block decisions across {} frames",
            rvp9_records.len(),
            frames
        );
        println!("  rvp9 IVF  → {}", rvp9_ivf.display());

        let libvpx_records = if no_libvpx {
            Vec::new()
        } else {
            drive_libvpx_leg(&vpxenc_path, &foreman_path, encode, &run_dir, fixed_q)?
        };
        if !no_libvpx {
            println!("  libvpx: {} per-block decisions", libvpx_records.len());
        }

        let comparison = if no_libvpx {
            None
        } else {
            let comparison = compare_decisions(&rvp9_records, &libvpx_records);
            let mismatches = comparison.mismatch_count();
            println!(
                "  compare: {} matched / {} mismatched frame-block keys",
                comparison.pairs.len() - mismatches,
                mismatches,
            );
            if let Err(error) = comparison.require_match(report_only) {
                strict_mismatches += error.mismatches;
            }

            let paired_path = run_dir.join(format!("foreman_{}kbps_raw.jsonl", kbps));
            let mut paired_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&paired_path)?;
            comparison.write_paired_jsonl(&mut paired_file)?;
            println!("  paired raw → {}", paired_path.display());
            Some(comparison)
        };

        // Per-frame summary table.
        let summary_path = run_dir.join(format!("foreman_{}kbps_summary.md", kbps));
        let mut sf = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&summary_path)?;
        writeln!(sf, "# F.0.2 oracle — {} kbps", kbps)?;
        writeln!(sf)?;
        writeln!(sf, "Schema liveness:")?;
        writeln!(
            sf,
            "- Strict fields: `bsize`, `mode`, `ref_frame`, `mv_row`, `mv_col`, `sse_after_pred`, `skip_flag`, `q_index`, `tx_size`, and `luma_eob_nonzero`."
        )?;
        writeln!(
            sf,
            "- `coef_bits_est`: live on rvp9, **0 on libvpx** (the libvpx-side split remains diagnostic follow-up)."
        )?;
        writeln!(
            sf,
            "- `mode_bits_est`, `mv_bits_est`, `ref_bits_est`, `filter_bits_est`, `skip_bits_est`: live on both legs."
        )?;
        writeln!(
            sf,
            "- `picker_rate`, `picker_rd_cost`, `model_var_y`, `model_tx_size`, `model_skip_txfm`, `sse_after_pred`: live on both legs."
        )?;
        writeln!(
            sf,
            "- `luma_coef_rate`, `luma_coef_dist`, `uv_model_rate`, `uv_model_dist`, `uv_u_sensitive`, `uv_v_sensitive`: live on both legs."
        )?;
        writeln!(
            sf,
            "- `partition_u_sensitive`, `partition_v_sensitive`: SB partition-predictor color-sensitivity oracle on both legs."
        )?;
        writeln!(
            sf,
            "- `last_candidates`: live LAST-reference `[ZERO, NEAREST, NEAR, NEW]` candidate costs on both legs; skip-transform states use the shared `None=0, AcOnly=1, AcDc=2` numbering."
        )?;
        writeln!(
            sf,
            "- `picker_dist`: rvp9 reports decoded-pixel SSE; libvpx reports its coefficient/model-RD ranking distortion."
        )?;
        writeln!(sf)?;
        writeln!(
            sf,
            "| frame_idx | rvp9 n_blocks | libvpx n_blocks | paired matches | mismatches |"
        )?;
        writeln!(sf, "|---|---|---|---|---|")?;
        let mut rvp9_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &rvp9_records {
            *rvp9_by_frame.entry(d.frame_idx).or_default() += 1;
        }
        let mut vpx_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &libvpx_records {
            *vpx_by_frame.entry(d.frame_idx).or_default() += 1;
        }
        let max_frame = rvp9_by_frame
            .keys()
            .chain(vpx_by_frame.keys())
            .copied()
            .max()
            .unwrap_or(0);
        for f in 0..=max_frame {
            let (matches, mismatches) = comparison.as_ref().map_or((0, 0), |comparison| {
                comparison
                    .pairs
                    .iter()
                    .filter(|pair| pair.key.frame_idx == f)
                    .fold((0, 0), |(matches, mismatches), pair| {
                        if pair.is_match() {
                            (matches + 1, mismatches)
                        } else {
                            (matches, mismatches + 1)
                        }
                    })
            });
            writeln!(
                sf,
                "| {} | {} | {} | {} | {} |",
                f,
                rvp9_by_frame.get(&f).copied().unwrap_or(0),
                vpx_by_frame.get(&f).copied().unwrap_or(0),
                matches,
                mismatches,
            )?;
        }
        println!("  summary → {}", summary_path.display());

        // Raw JSONL dump of rvp9-side records.
        let raw_path = run_dir.join(format!("foreman_{}kbps_rvp9.jsonl", kbps));
        let mut rf = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&raw_path)?;
        for d in &rvp9_records {
            d.write_jsonl(&mut rf)?;
        }
        println!("  rvp9 raw  → {}", raw_path.display());
    }

    if strict_mismatches > 0 {
        return Err(format!(
            "F.0.2 strict oracle found {strict_mismatches} mismatched frame/block key(s); \
             inspect the paired raw JSONL artifacts or pass --report-only for diagnostic runs"
        )
        .into());
    }

    println!("\nDone. Artifacts in {}", run_dir.display());
    Ok(())
}

/// Approximation of an ISO-8601-ish stamp without pulling in `chrono`
/// (workspace has no dep on it). Format `yyyymmdd_hhmmss` based on
/// `SystemTime::now()`.
fn chrono_like_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Crude: seconds-since-epoch in hex. Unique enough for the
    // harness's per-run subdir.
    format!("{:x}", now)
}

/// Drive the rvp9 leg: open foreman_cif, encode `frames` frames at
/// the requested target bitrate, drain the per-block sink at end of
/// each frame.
fn drive_rvp9_leg(
    foreman_path: &Path,
    encode: OracleEncodeConfig,
    strict: bool,
    ivf_path: &Path,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let (gop_length, rc_mode) = rvp9_rate_config(encode.vbr);
    let (yuv_frames, source_fps) = read_cropped(
        foreman_path,
        encode.width,
        encode.height,
        encode.frames as usize,
    )?;
    let fps = oracle_fps(encode, source_fps);
    let cfg = EncoderConfig {
        width: encode.width,
        height: encode.height,
        framerate: fps,
        target_bitrate_bps: encode.kbps * 1000,
        gop_length,
        aq_mode: encoder_aq_mode(encode.aq_mode),
        use_prev_frame_mvs: false,
        speed: encode.cpu_used,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: split_override(strict),
        min_qindex: 4,
        max_qindex: 255,
    };
    let parallelism = oracle_parallelism(encode);
    let mut enc = StreamEncoder::new_with_parallelism(cfg, parallelism)?;
    let mut all_records = Vec::new();
    let mut packets = Vec::with_capacity(yuv_frames.len());
    // Drain any prior thread-local state.
    let _ = f02_oracle::drain_frame();
    for (y, u, v) in yuv_frames.iter() {
        let view = YuvFrameRef {
            y,
            u,
            v,
            y_stride: encode.width as usize,
            u_stride: encode.width as usize / 2,
            v_stride: encode.width as usize / 2,
        };
        let packet = enc.encode_frame(&view)?;
        packets.push(packet.data);
        let mut per_frame = f02_oracle::drain_frame();
        all_records.append(&mut per_frame);
    }
    let mut ivf = File::create(ivf_path)?;
    let header = IvfHeader::vp9(
        encode.width as u16,
        encode.height as u16,
        fps.round() as u32,
        1,
    );
    let packet_refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    write_ivf(&mut ivf, &header, &packet_refs)?;
    Ok(all_records)
}

/// Drive the libvpx leg: spawn `vpxenc` with `VPX_F02_ORACLE_OUT`
/// pointed at a sidecar JSONL file under `run_dir`. The harness
/// then re-reads that file.
///
/// The source is center-cropped to the same requested I420 geometry used by
/// the rvp9 leg. The CLI options match the Stage 4a product gate.
fn drive_libvpx_leg(
    vpxenc: &Path,
    foreman: &Path,
    encode: OracleEncodeConfig,
    run_dir: &Path,
    fixed_q: Option<u8>,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let sidecar = run_dir.join(format!("foreman_{}kbps_libvpx.jsonl", encode.kbps));
    let raw = run_dir.join(format!("foreman_{}kbps_input.i420", encode.kbps));
    let (cropped, source_fps) =
        read_cropped(foreman, encode.width, encode.height, encode.frames as usize)?;
    let fps = oracle_fps(encode, source_fps);
    {
        let mut out = File::create(&raw)?;
        for (y, u, v) in &cropped {
            out.write_all(y)?;
            out.write_all(u)?;
            out.write_all(v)?;
        }
    }

    let mut cmd = Command::new(vpxenc);
    cmd.env("VPX_F02_ORACLE_OUT", &sidecar)
        .arg("--codec=vp9")
        .arg("--ivf")
        .arg("--rt")
        .arg(format!("--cpu-used={}", encode.cpu_used))
        .arg("--tune-content=default")
        .arg("--lag-in-frames=0")
        .arg("--passes=1")
        .arg(format!("--threads={}", encode.threads))
        .arg(format!("--tile-columns={}", encode.tile_columns))
        .arg("--i420")
        .arg(format!("--width={}", encode.width))
        .arg(format!("--height={}", encode.height))
        .arg(format!("--fps={}/1", fps.round() as u32))
        .arg(format!("--target-bitrate={}", encode.kbps))
        .arg(format!("--limit={}", encode.frames))
        .arg(format!("--aq-mode={}", encode.aq_mode))
        .arg("-o")
        .arg(run_dir.join(format!("foreman_{}kbps_libvpx.ivf", encode.kbps)))
        .arg(&raw)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(qindex) = fixed_q {
        let (cli_q, max_q) =
            fixed_q_cli_range(qindex).expect("fixed q validated before libvpx invocation");
        cmd.arg("--end-usage=q")
            .arg(format!("--cq-level={cli_q}"))
            .arg(format!("--min-q={cli_q}"))
            .arg(format!("--max-q={max_q}"));
    } else if !encode.vbr {
        cmd.arg("--end-usage=cbr");
    }
    let out = cmd.output()?;
    let _ = std::fs::remove_file(&raw);
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "vpxenc failed (exit {}): {}",
            out.status.code().unwrap_or(-1),
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        )
        .into());
    }
    if !sidecar.exists() {
        return Err(format!(
            "libvpx sidecar at {} was NOT produced — \
             is third_party/libvpx-patches/f02_oracle.patch applied?",
            sidecar.display()
        )
        .into());
    }
    let f = File::open(&sidecar)?;
    let records = parse_jsonl_stream(BufReader::new(f))
        .map_err(|e| format!("libvpx sidecar parse failed: {e}"))?;
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encode_config() -> OracleEncodeConfig {
        OracleEncodeConfig {
            width: 1280,
            height: 704,
            frames: 60,
            fps: Some(30),
            cpu_used: 7,
            threads: 1,
            tile_columns: 6,
            kbps: 1000,
            vbr: true,
            aq_mode: 0,
        }
    }

    #[test]
    fn vbr_oracle_uses_production_gop_and_vbr_rate_control() {
        assert_eq!(rvp9_rate_config(true), (60, RcMode::Vbr));
    }

    #[test]
    fn cbr_oracle_uses_production_gop_and_cbr_rate_control() {
        assert_eq!(rvp9_rate_config(false), (60, RcMode::Cbr));
    }

    #[test]
    fn strict_flag_selects_stage4a_strict_split_policy() {
        assert_eq!(
            split_override(true),
            Stage4aProductSplitOverride::ForceStrict
        );
        assert_eq!(split_override(false), Stage4aProductSplitOverride::Auto);
    }

    #[test]
    fn fixed_q_maps_to_valid_libvpx_cli_range() {
        assert_eq!(fixed_q_cli_range(120), Some((30, 38)));
        assert_eq!(fixed_q_cli_range(121), None);
        assert_eq!(fixed_q_cli_range(224), None);
    }

    #[test]
    fn cpu_used_defaults_to_seven_and_accepts_product_presets() {
        assert_eq!(parse_cpu_used(&[]).unwrap(), 7);
        for preset in 6..=8 {
            let args = vec![format!("--cpu-used={preset}")];
            assert_eq!(parse_cpu_used(&args).unwrap(), preset);
        }
    }

    #[test]
    fn cpu_used_rejects_non_product_presets_and_invalid_values() {
        for value in ["5", "9", "fast"] {
            let args = vec!["--cpu-used".to_string(), value.to_string()];
            assert!(parse_cpu_used(&args).is_err());
        }
    }

    #[test]
    fn aq_mode_accepts_only_off_and_cyclic_refresh() {
        assert_eq!(parse_aq_mode(&[]).unwrap(), 0);
        assert_eq!(parse_aq_mode(&["--aq-mode=3".to_string()]).unwrap(), 3);
        for value in ["1", "2", "4", "cyclic"] {
            let args = vec!["--aq-mode".to_string(), value.to_string()];
            assert!(parse_aq_mode(&args).is_err());
        }
        assert_eq!(encoder_aq_mode(0), AqMode::Off);
        assert_eq!(encoder_aq_mode(3), AqMode::CyclicRefresh);
    }

    #[test]
    fn production_oracle_uses_matched_parallelism() {
        let parallelism = oracle_parallelism(test_encode_config());
        assert_eq!(parallelism.max_threads, 1);
        assert_eq!(parallelism.requested_log2_tile_cols, 6);
        assert_eq!(parallelism.effective_log2_tile_cols(1280), 2);
    }

    #[test]
    fn explicit_fps_overrides_source_header() {
        assert_eq!(oracle_fps(test_encode_config(), 50.0), 30.0);

        let mut source_rate = test_encode_config();
        source_rate.fps = None;
        assert_eq!(oracle_fps(source_rate, 50.0), 50.0);
    }
}
