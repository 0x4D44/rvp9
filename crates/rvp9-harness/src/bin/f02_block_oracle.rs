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
//!
//! Add `--spatial-layers 2|3 --inter-layer-pred 0|3 --fixed-q Q` and point
//! `--vpxenc` at the patched `vp9_spatial_svc_encoder` to drive exact fixed-Q
//! spatial state. Select `--vbr` or `--cbr` to retain the corresponding picker
//! calibration. Omit `--fixed-q` and add `--report-only` for natural-CBR
//! diagnosis. The harness validates every component, then compares
//! `--analysis-spatial-id` (the base layer by default) at its coded geometry.

use rvp9_core::bitstream::ChromaFormat;
use rvp9_core::mode_decision::inter::NewMvGateReason;
use rvp9_encoder::f02_oracle::{self, BlockDecision, MlPartitionTrace};
use rvp9_encoder::svc::{InterLayerPred, SpatialSvcConfig};
use rvp9_encoder::{
    AqMode, EncoderConfig, EncoderParallelism, RcMode, SpatialEncoder, Stage4aProductSplitOverride,
    StreamEncoder, YuvFrameRef,
};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::f02_schema::{
    analyze_newmv_cells, analyze_winner_cells, compare_decisions,
    compare_matching_newmv_candidates, compare_matching_newmv_search_stages, parse_jsonl_stream,
};
use rvp9_harness::y4m::{read_cropped, write_frame, write_header};
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
    spatial_layers: u8,
    inter_layer_pred: u8,
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

fn parse_vbr(args: &[String], fixed_q: Option<u8>) -> Result<bool, String> {
    let explicit_vbr = args.iter().any(|argument| argument == "--vbr");
    let explicit_cbr = args.iter().any(|argument| argument == "--cbr");
    if explicit_vbr && explicit_cbr {
        return Err("--vbr and --cbr are mutually exclusive".to_owned());
    }
    Ok(if explicit_cbr {
        false
    } else {
        explicit_vbr || fixed_q.is_some()
    })
}

fn validate_spatial_oracle_mode(
    spatial_layers: u8,
    vbr: bool,
    fixed_q: Option<u8>,
    aq_mode: u8,
    report_only: bool,
) -> Result<(), String> {
    if spatial_layers <= 1 {
        return Ok(());
    }
    if aq_mode != 0 {
        return Err("the spatial F02 mode requires --aq-mode 0".to_owned());
    }
    if vbr && fixed_q.is_none() {
        return Err("the spatial VBR F02 mode requires --fixed-q QINDEX".to_owned());
    }
    if fixed_q.is_none() && !report_only {
        return Err(
            "natural-CBR spatial analysis is diagnostic only; pass --report-only or select --fixed-q QINDEX"
                .to_owned(),
        );
    }
    Ok(())
}

fn append_libvpx_rate_args(command: &mut Command, vbr: bool, fixed_q: Option<u8>) {
    if let Some(qindex) = fixed_q {
        let (cli_q, max_q) =
            fixed_q_cli_range(qindex).expect("fixed q validated before libvpx invocation");
        if vbr {
            command
                .arg("--end-usage=q")
                .arg(format!("--cq-level={cli_q}"))
                .arg(format!("--min-q={cli_q}"))
                .arg(format!("--max-q={max_q}"));
        } else {
            command
                .arg("--disable-warning-prompt")
                .arg("--end-usage=cbr")
                .arg(format!("--min-q={cli_q}"))
                .arg(format!("--max-q={cli_q}"));
        }
    } else if !vbr {
        command.arg("--end-usage=cbr");
    }
}

fn spatial_layer_bitrates(target_kbps: u32, spatial_layers: u8) -> Result<String, String> {
    let weights: &[u32] = match spatial_layers {
        2 => &[1, 4],
        3 => &[1, 4, 16],
        other => return Err(format!("--spatial-layers must be 2 or 3, got {other}")),
    };
    let total = weights.iter().sum::<u32>();
    if !target_kbps.is_multiple_of(total) {
        return Err(format!(
            "spatial target bitrate {target_kbps} kbps must be divisible by {total}"
        ));
    }
    Ok(weights
        .iter()
        .map(|weight| (target_kbps / total * weight).to_string())
        .collect::<Vec<_>>()
        .join(","))
}

fn spatial_layer_geometry(
    width: u32,
    height: u32,
    spatial_layers: u8,
    spatial_id: u8,
) -> Result<(u32, u32), String> {
    if !matches!(spatial_layers, 1..=3) {
        return Err(format!(
            "--spatial-layers must be 1, 2, or 3, got {spatial_layers}"
        ));
    }
    if spatial_id >= spatial_layers {
        return Err(format!(
            "--analysis-spatial-id must be below --spatial-layers ({spatial_layers}), got {spatial_id}"
        ));
    }
    let divisor = 1u32 << (spatial_layers - spatial_id - 1);
    if !width.is_multiple_of(divisor) || !height.is_multiple_of(divisor) {
        return Err(format!(
            "{width}x{height} is not divisible by spatial layer {spatial_id}'s {divisor}x scale"
        ));
    }
    Ok((width / divisor, height / divisor))
}

fn inter_layer_pred(encode: OracleEncodeConfig) -> Result<InterLayerPred, String> {
    match encode.inter_layer_pred {
        0 => Ok(InterLayerPred::On),
        3 => Ok(InterLayerPred::OnConstrained),
        other => Err(format!("--inter-layer-pred must be 0 or 3, got {other}")),
    }
}

fn append_libvpx_spatial_args(
    command: &mut Command,
    input: &Path,
    output: &Path,
    encode: OracleEncodeConfig,
    fixed_q: Option<u8>,
) -> Result<(), String> {
    let scales = match encode.spatial_layers {
        2 => "1/2,1/1",
        3 => "1/4,1/2,1/1",
        other => return Err(format!("--spatial-layers must be 2 or 3, got {other}")),
    };
    let rates = spatial_layer_bitrates(encode.kbps, encode.spatial_layers)?;
    let (min_q, max_q) = if let Some(qindex) = fixed_q {
        let cli_q = fixed_q_cli_range(qindex)
            .ok_or_else(|| format!("invalid spatial fixed qindex {qindex}"))?
            .0;
        command.env("VPX_F02_PRESERVE_LAYER_QUANTIZERS", "1");
        (cli_q, cli_q)
    } else {
        // Match the production spatial BD-rate harness and pinned example.
        (2, 56)
    };
    let min_quantizers = std::iter::repeat_n(min_q.to_string(), usize::from(encode.spatial_layers))
        .collect::<Vec<_>>()
        .join(",");
    let max_quantizers = std::iter::repeat_n(max_q.to_string(), usize::from(encode.spatial_layers))
        .collect::<Vec<_>>()
        .join(",");
    command
        .arg(format!("--frames={}", encode.frames))
        .arg(format!("--output={}", output.display()))
        .arg(format!("--target-bitrate={}", encode.kbps))
        .arg(format!("--speed={}", encode.cpu_used))
        .arg(format!("--threads={}", encode.threads))
        .arg(format!(
            "--aqmode={}",
            if encode.aq_mode == 3 { 1 } else { 0 }
        ))
        .arg(format!("--rc-end-usage={}", if encode.vbr { 0 } else { 1 }))
        .arg(format!("--spatial-layers={}", encode.spatial_layers))
        .arg("--temporal-layers=1")
        .arg(format!("--kf-dist={PRODUCT_GOP_LENGTH}"))
        .arg("--lag-in-frames=0")
        .arg(format!("--scale-factors={scales}"))
        .arg(format!("--bitrates={rates}"))
        .arg(format!("--min-q={min_quantizers}"))
        .arg(format!("--max-q={max_quantizers}"))
        .arg("--tune-content=0")
        .arg(format!("--inter-layer-pred={}", encode.inter_layer_pred))
        .arg(input);
    Ok(())
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

fn spatial_analysis_records(
    records: &[BlockDecision],
    spatial_layers: u8,
    analysis_spatial_id: u8,
    frames: u32,
    fixed_q: Option<u8>,
    label: &str,
) -> Result<Vec<BlockDecision>, String> {
    if spatial_layers == 1 {
        return Ok(records.to_vec());
    }
    if records.is_empty() {
        return Err(format!("{label} spatial sidecar is empty"));
    }
    let mut seen = vec![false; usize::from(spatial_layers)];
    for record in records {
        if record.spatial_id >= spatial_layers {
            return Err(format!(
                "{label} sidecar has spatial_id {} outside 0..{}",
                record.spatial_id,
                spatial_layers - 1
            ));
        }
        if record.frame_idx >= frames {
            return Err(format!(
                "{label} sidecar has frame_idx {} outside 0..{}",
                record.frame_idx,
                frames - 1
            ));
        }
        if let Some(expected_q) = fixed_q
            && record.q_index != expected_q
        {
            return Err(format!(
                "{label} sidecar is not fixed qindex {expected_q}: spatial {} frame {} uses {}",
                record.spatial_id, record.frame_idx, record.q_index
            ));
        }
        seen[usize::from(record.spatial_id)] = true;
    }
    let missing = seen
        .iter()
        .enumerate()
        .filter_map(|(spatial_id, &present)| (!present).then_some(spatial_id))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "{label} sidecar is missing spatial layer record(s): {missing:?}"
        ));
    }
    Ok(records
        .iter()
        .copied()
        .filter(|record| record.spatial_id == analysis_spatial_id)
        .collect())
}

fn oracle_encoder_config(
    encode: OracleEncodeConfig,
    strict: bool,
    framerate: f64,
    fixed_q: Option<u8>,
) -> EncoderConfig {
    let (gop_length, rc_mode) = rvp9_rate_config(encode.vbr);
    let (min_qindex, max_qindex) = if let Some(qindex) = fixed_q {
        (qindex, qindex)
    } else if encode.spatial_layers > 1 && !encode.vbr {
        // Match the production spatial harness and pinned example's 2..56
        // CLI quantizer bounds, expressed as internal qindices.
        (8, 224)
    } else {
        (4, 255)
    };
    EncoderConfig {
        width: encode.width,
        height: encode.height,
        framerate,
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
        min_qindex,
        max_qindex,
    }
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
    let spatial_layers: u8 = arg(&argv, "--spatial-layers")
        .map(|value| {
            value
                .parse::<u8>()
                .map_err(|_| format!("--spatial-layers must be 1, 2, or 3, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(1);
    if !(1..=3).contains(&spatial_layers) {
        return Err(format!("--spatial-layers must be 1, 2, or 3, got {spatial_layers}").into());
    }
    let analysis_spatial_id: u8 = arg(&argv, "--analysis-spatial-id")
        .map(|value| {
            value
                .parse::<u8>()
                .map_err(|_| format!("--analysis-spatial-id must be an integer, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(0);
    spatial_layer_geometry(width, height, spatial_layers, analysis_spatial_id)?;
    let inter_layer_pred: u8 = arg(&argv, "--inter-layer-pred")
        .map(|value| {
            value
                .parse::<u8>()
                .map_err(|_| format!("--inter-layer-pred must be 0 or 3, got {value:?}"))
        })
        .transpose()?
        .unwrap_or(0);
    if !matches!(inter_layer_pred, 0 | 3) {
        return Err(format!("--inter-layer-pred must be 0 or 3, got {inter_layer_pred}").into());
    }
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
    // Fixed-Q diagnostics default to the production Stage 4a VBR picker
    // calibration. An explicit --cbr retains CBR calibration while bypassing
    // only the controller's chosen Q on both encoder legs.
    let vbr = parse_vbr(&argv, fixed_q)?;
    validate_spatial_oracle_mode(spatial_layers, vbr, fixed_q, aq_mode, report_only)?;
    if spatial_layers > 1 && frames < 2 {
        return Err("the spatial F02 mode requires at least two frames".into());
    }
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
            "ERROR: patched libvpx encoder not found at {}",
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
    println!("  spatial   : {spatial_layers} layer(s)");
    if spatial_layers > 1 {
        println!("  inter pred: {inter_layer_pred}");
        println!("  analysis  : spatial layer {analysis_spatial_id}");
    }
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
            spatial_layers,
            inter_layer_pred,
        };
        let _ = f02_oracle::drain_partition_traces();
        let rvp9_records = if spatial_layers > 1 {
            drive_rvp9_spatial_leg(&foreman_path, encode, strict, &rvp9_ivf, fixed_q)?
        } else {
            drive_rvp9_leg(&foreman_path, encode, strict, &rvp9_ivf)?
        };
        println!(
            "  rvp9: {} per-block decisions across {} frames",
            rvp9_records.len(),
            frames
        );
        println!("  rvp9 IVF  → {}", rvp9_ivf.display());
        let rvp9_partition_traces = f02_oracle::drain_partition_traces();
        let rvp9_partition_path = run_dir.join(format!("foreman_{}kbps_partition_rvp9.tsv", kbps));
        if !rvp9_partition_traces.is_empty() {
            write_partition_traces(&rvp9_partition_path, &rvp9_partition_traces)?;
            println!(
                "  rvp9 ML partition: {} nodes → {}",
                rvp9_partition_traces.len(),
                rvp9_partition_path.display()
            );
        }

        let libvpx_records = if no_libvpx {
            Vec::new()
        } else {
            if spatial_layers > 1 {
                drive_libvpx_spatial_leg(&vpxenc_path, &foreman_path, encode, &run_dir, fixed_q)?
            } else {
                drive_libvpx_leg(&vpxenc_path, &foreman_path, encode, &run_dir, fixed_q)?
            }
        };
        if !no_libvpx {
            println!("  libvpx: {} per-block decisions", libvpx_records.len());
            let libvpx_partition_path =
                run_dir.join(format!("foreman_{}kbps_partition_libvpx.tsv", kbps));
            if !rvp9_partition_traces.is_empty() && libvpx_partition_path.exists() {
                report_partition_trace_comparison(&rvp9_partition_path, &libvpx_partition_path)?;
            }
        }

        let rvp9_analysis_records = spatial_analysis_records(
            &rvp9_records,
            spatial_layers,
            analysis_spatial_id,
            frames,
            fixed_q,
            "rvp9",
        )?;
        let libvpx_analysis_records = if no_libvpx {
            Vec::new()
        } else {
            spatial_analysis_records(
                &libvpx_records,
                spatial_layers,
                analysis_spatial_id,
                frames,
                fixed_q,
                "libvpx",
            )?
        };
        let (analysis_width, analysis_height) = if spatial_layers > 1 {
            spatial_layer_geometry(width, height, spatial_layers, analysis_spatial_id)?
        } else {
            (width, height)
        };
        let newmv_analysis = if no_libvpx {
            None
        } else {
            Some(analyze_newmv_cells(
                &rvp9_analysis_records,
                &libvpx_analysis_records,
                analysis_height.div_ceil(8) as u16,
                analysis_width.div_ceil(8) as u16,
            ))
        };
        let winner_analysis = if no_libvpx {
            None
        } else {
            Some(analyze_winner_cells(
                &rvp9_analysis_records,
                &libvpx_analysis_records,
                analysis_height.div_ceil(8) as u16,
                analysis_width.div_ceil(8) as u16,
            ))
        };
        let matching_newmv_costs = if no_libvpx {
            None
        } else {
            Some(compare_matching_newmv_candidates(
                &rvp9_analysis_records,
                &libvpx_analysis_records,
            ))
        };
        let matching_newmv_search_stages = if no_libvpx {
            None
        } else {
            Some(compare_matching_newmv_search_stages(
                &rvp9_analysis_records,
                &libvpx_analysis_records,
            ))
        };
        if let Some(analysis) = newmv_analysis {
            println!(
                "  NEWMV cells: libvpx LAST/nonzero {}; rvp9 exact selected {}, exact lost {}, near selected {}, near lost {}, different selected {}, different lost {}, zero {}, unavailable {}, missing/ambiguous {}; rejected higher-RD {}, rejected non-higher-RD {}",
                analysis.libvpx_last_nonzero_newmv_cells,
                analysis.rvp9_new_exact_selected_cells,
                analysis.rvp9_new_exact_lost_cells,
                analysis.rvp9_new_near_selected_cells,
                analysis.rvp9_new_near_lost_cells,
                analysis.rvp9_new_different_selected_cells,
                analysis.rvp9_new_different_lost_cells,
                analysis.rvp9_new_zero_cells,
                analysis.rvp9_new_unavailable_cells,
                analysis.rvp9_missing_cells + analysis.rvp9_ambiguous_cells,
                analysis.rvp9_new_rejected_higher_rd_cells,
                analysis.rvp9_new_rejected_not_higher_rd_cells,
            );
            let unavailable = [
                NewMvGateReason::Unknown,
                NewMvGateReason::InterModeMask,
                NewMvGateReason::ScreenContent,
                NewMvGateReason::ReferenceMask,
                NewMvGateReason::LowTemporalVariance,
                NewMvGateReason::AdaptiveRdThreshold,
                NewMvGateReason::SubpelSyntax,
                NewMvGateReason::DuplicateMv,
            ]
            .into_iter()
            .filter_map(|reason| {
                let count = analysis.rvp9_new_unavailable_by_gate[reason as usize];
                (count > 0).then(|| format!("{}={count}", reason.as_tag()))
            })
            .collect::<Vec<_>>()
            .join(", ");
            println!("  unavailable LAST/NEWMV gates: {unavailable}");
        }
        if let Some(analysis) = winner_analysis {
            println!(
                "  winner cells: {} compared; exact {}, reference mismatch {}, same-reference mode mismatch {}, same-reference/mode MV mismatch {}, filter-only mismatch {}; block-size matches {}; missing rvp9/libvpx {}/{}; overlaps rvp9/libvpx {}/{}",
                analysis.compared_cells,
                analysis.exact_winner_matches,
                analysis.ref_frame_mismatch_cells,
                analysis.ref_same_mode_mismatch_cells,
                analysis.ref_mode_same_mv_mismatch_cells,
                analysis.ref_mode_mv_same_filter_mismatch_cells,
                analysis.bsize_matches,
                analysis.rvp9_missing_cells,
                analysis.libvpx_missing_cells,
                analysis.rvp9_overlapped_cells,
                analysis.libvpx_overlapped_cells,
            );
        }
        if let Some(costs) = matching_newmv_costs {
            println!(
                "  matching NEWMV costs: {} exact leaves ({} rejected); rvp9/libvpx rate {:.4} dist {:.4} RD {:.4}",
                costs.records,
                costs.rejected_records,
                ratio(costs.rvp9_rate_sum, costs.libvpx_rate_sum),
                ratio(costs.rvp9_dist_sum, costs.libvpx_dist_sum),
                ratio(costs.rvp9_rd_cost_sum, costs.libvpx_rd_cost_sum),
            );
        }
        if let Some(stages) = matching_newmv_search_stages {
            println!(
                "  matching NEWMV search stages: {} exact leaves; start {}/{}, fullpel {}/{}, final {}/{}; inputs near {}/{}, SB-pred {}/{}, pred-winner {}/{}; same-start/fullpel-diverged {}, same-fullpel/final-diverged {}; same-fullpel inputs ref-mv {}/{}, error-per-bit {}/{}, source {}/{}, reference {}/{}; exact-input final {}/{}",
                stages.records,
                stages.start_matches,
                stages.records,
                stages.fullpel_matches,
                stages.records,
                stages.final_matches,
                stages.records,
                stages.near_mv_matches,
                stages.records,
                stages.sb_pred_mv_matches,
                stages.records,
                stages.mv_pred_mv_matches,
                stages.records,
                stages.same_start_fullpel_diverged,
                stages.same_fullpel_final_diverged,
                stages.same_fullpel_ref_mv_matches,
                stages.same_fullpel_records,
                stages.same_fullpel_error_per_bit_matches,
                stages.same_fullpel_records,
                stages.same_fullpel_source_hash_matches,
                stages.same_fullpel_records,
                stages.same_fullpel_reference_hash_matches,
                stages.same_fullpel_records,
                stages.exact_input_final_matches,
                stages.exact_input_records,
            );
            if stages.exact_input_subpel_records > 0 {
                println!(
                    "  exact-input subpel traces: {} records; limits {}, policy {}, center distortion {}, center MV cost {}; half coords {}, distortion {}, MV cost {}, winner {}; quarter coords {}, distortion {}, MV cost {}, winner {}",
                    stages.exact_input_subpel_records,
                    stages.subpel_limit_matches,
                    stages.subpel_policy_matches,
                    stages.subpel_center_distortion_matches,
                    stages.subpel_center_mv_cost_matches,
                    stages.subpel_half_coordinate_matches,
                    stages.subpel_half_distortion_matches,
                    stages.subpel_half_mv_cost_matches,
                    stages.subpel_half_winner_matches,
                    stages.subpel_quarter_coordinate_matches,
                    stages.subpel_quarter_distortion_matches,
                    stages.subpel_quarter_mv_cost_matches,
                    stages.subpel_quarter_winner_matches,
                );
            }
        }

        let comparison = if no_libvpx {
            None
        } else {
            let comparison = compare_decisions(&rvp9_analysis_records, &libvpx_analysis_records);
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
        if spatial_layers > 1 {
            writeln!(
                sf,
                "Scope: validated {spatial_layers}-layer spatial SVC; comparisons below use only spatial layer {analysis_spatial_id} at {analysis_width}x{analysis_height}."
            )?;
            writeln!(sf)?;
        }
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
            "- `last_candidates`: live LAST-reference `[ZERO, NEAREST, NEAR, NEW]` candidate costs on both legs; skip-transform states use the shared `None=0, AcOnly=1, AcDc=2` numbering. Fresh NEWMV entries append search start, full-pel result, and final fractional result."
        )?;
        writeln!(
            sf,
            "- `picker_dist`: rvp9 reports decoded-pixel SSE; libvpx reports its coefficient/model-RD ranking distortion."
        )?;
        writeln!(sf)?;
        if let Some(analysis) = winner_analysis {
            writeln!(sf, "## Partition-independent winner diagnosis")?;
            writeln!(sf)?;
            writeln!(
                sf,
                "Both partition trees are expanded over visible 8x8 cells. Mismatch rows are hierarchical: the first differing field claims the cell."
            )?;
            writeln!(sf)?;
            writeln!(sf, "| winner outcome | cells |")?;
            writeln!(sf, "|---|---:|")?;
            writeln!(sf, "| compared | {} |", analysis.compared_cells)?;
            writeln!(sf, "| exact winner | {} |", analysis.exact_winner_matches)?;
            writeln!(
                sf,
                "| reference mismatch | {} |",
                analysis.ref_frame_mismatch_cells
            )?;
            writeln!(
                sf,
                "| same-reference mode mismatch | {} |",
                analysis.ref_same_mode_mismatch_cells
            )?;
            writeln!(
                sf,
                "| same-reference/mode MV mismatch | {} |",
                analysis.ref_mode_same_mv_mismatch_cells
            )?;
            writeln!(
                sf,
                "| filter-only mismatch | {} |",
                analysis.ref_mode_mv_same_filter_mismatch_cells
            )?;
            writeln!(
                sf,
                "| covering block-size match | {} |",
                analysis.bsize_matches
            )?;
            writeln!(
                sf,
                "| missing rvp9/libvpx | {}/{} |",
                analysis.rvp9_missing_cells, analysis.libvpx_missing_cells
            )?;
            writeln!(
                sf,
                "| overlaps rvp9/libvpx | {}/{} |",
                analysis.rvp9_overlapped_cells, analysis.libvpx_overlapped_cells
            )?;
            writeln!(sf)?;
        }
        if let Some(analysis) = newmv_analysis {
            writeln!(sf, "## Partition-independent LAST/NEWMV diagnosis")?;
            writeln!(sf)?;
            writeln!(
                sf,
                "Counts are visible 8x8 cells where libvpx selected a non-zero LAST/NEWMV vector. `exact` means the same q3 vector; `near` means each component is within one pixel."
            )?;
            writeln!(sf)?;
            writeln!(sf, "| rvp9 outcome | cells |")?;
            writeln!(sf, "|---|---:|")?;
            writeln!(
                sf,
                "| libvpx LAST/nonzero NEWMV target | {} |",
                analysis.libvpx_last_nonzero_newmv_cells
            )?;
            writeln!(
                sf,
                "| exact candidate selected | {} |",
                analysis.rvp9_new_exact_selected_cells
            )?;
            writeln!(
                sf,
                "| exact candidate lost RD | {} |",
                analysis.rvp9_new_exact_lost_cells
            )?;
            writeln!(
                sf,
                "| near candidate selected | {} |",
                analysis.rvp9_new_near_selected_cells
            )?;
            writeln!(
                sf,
                "| near candidate lost RD | {} |",
                analysis.rvp9_new_near_lost_cells
            )?;
            writeln!(
                sf,
                "| different non-zero candidate selected | {} |",
                analysis.rvp9_new_different_selected_cells
            )?;
            writeln!(
                sf,
                "| different non-zero candidate lost RD | {} |",
                analysis.rvp9_new_different_lost_cells
            )?;
            writeln!(sf, "| zero candidate | {} |", analysis.rvp9_new_zero_cells)?;
            writeln!(
                sf,
                "| unavailable candidate | {} |",
                analysis.rvp9_new_unavailable_cells
            )?;
            for reason in [
                NewMvGateReason::Unknown,
                NewMvGateReason::InterModeMask,
                NewMvGateReason::ScreenContent,
                NewMvGateReason::ReferenceMask,
                NewMvGateReason::LowTemporalVariance,
                NewMvGateReason::AdaptiveRdThreshold,
                NewMvGateReason::SubpelSyntax,
                NewMvGateReason::DuplicateMv,
            ] {
                let count = analysis.rvp9_new_unavailable_by_gate[reason as usize];
                if count > 0 {
                    writeln!(sf, "| unavailable: {} | {count} |", reason.as_tag())?;
                }
            }
            writeln!(
                sf,
                "| missing or ambiguous rvp9 coverage | {} |",
                analysis.rvp9_missing_cells + analysis.rvp9_ambiguous_cells
            )?;
            writeln!(
                sf,
                "| classified target cells | {} |",
                analysis.classified_cells()
            )?;
            writeln!(
                sf,
                "| rejected candidates with higher RD than winner | {} |",
                analysis.rvp9_new_rejected_higher_rd_cells
            )?;
            writeln!(
                sf,
                "| rejected candidates without higher RD than winner | {} |",
                analysis.rvp9_new_rejected_not_higher_rd_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → LAST NEARESTMV | {} |",
                analysis.rvp9_exact_or_near_lost_to_last_nearest_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → LAST NEARMV | {} |",
                analysis.rvp9_exact_or_near_lost_to_last_near_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → LAST ZEROMV | {} |",
                analysis.rvp9_exact_or_near_lost_to_last_zero_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → another LAST NEWMV | {} |",
                analysis.rvp9_exact_or_near_lost_to_last_new_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → another reference | {} |",
                analysis.rvp9_exact_or_near_lost_to_other_ref_cells
            )?;
            writeln!(
                sf,
                "| exact/near rejected → intra | {} |",
                analysis.rvp9_exact_or_near_lost_to_intra_cells
            )?;
            writeln!(sf)?;
        }
        if let Some(costs) = matching_newmv_costs {
            writeln!(sf, "### Exact-leaf, exact-vector NEWMV candidate costs")?;
            writeln!(sf)?;
            writeln!(
                sf,
                "This subset requires equal top-left, block size, and non-zero LAST/NEWMV candidate vector on both legs."
            )?;
            writeln!(sf)?;
            writeln!(sf, "| metric | value |")?;
            writeln!(sf, "|---|---:|")?;
            writeln!(sf, "| matching records | {} |", costs.records)?;
            writeln!(sf, "| rejected by rvp9 | {} |", costs.rejected_records)?;
            writeln!(
                sf,
                "| rvp9/libvpx candidate rate | {:.6} |",
                ratio(costs.rvp9_rate_sum, costs.libvpx_rate_sum)
            )?;
            writeln!(
                sf,
                "| rvp9/libvpx luma model rate | {:.6} |",
                ratio(costs.rvp9_luma_rate_sum, costs.libvpx_luma_rate_sum)
            )?;
            writeln!(
                sf,
                "| rvp9/libvpx chroma model rate | {:.6} |",
                ratio(costs.rvp9_uv_rate_sum, costs.libvpx_uv_rate_sum)
            )?;
            let rvp9_syntax_rate = costs
                .rvp9_rate_sum
                .saturating_sub(costs.rvp9_luma_rate_sum)
                .saturating_sub(costs.rvp9_uv_rate_sum);
            let libvpx_syntax_rate = costs
                .libvpx_rate_sum
                .saturating_sub(costs.libvpx_luma_rate_sum)
                .saturating_sub(costs.libvpx_uv_rate_sum);
            writeln!(
                sf,
                "| rvp9/libvpx syntax-rate remainder | {:.6} |",
                ratio(rvp9_syntax_rate, libvpx_syntax_rate)
            )?;
            writeln!(
                sf,
                "| rvp9/libvpx candidate distortion | {:.6} |",
                ratio(costs.rvp9_dist_sum, costs.libvpx_dist_sum)
            )?;
            writeln!(
                sf,
                "| rvp9/libvpx candidate RD cost | {:.6} |",
                ratio(costs.rvp9_rd_cost_sum, costs.libvpx_rd_cost_sum)
            )?;
            for rvp9_skip in 0..3 {
                for libvpx_skip in 0..3 {
                    let count = costs.skip_pair_counts[rvp9_skip * 3 + libvpx_skip];
                    if count != 0 {
                        writeln!(
                            sf,
                            "| skip-state pair rvp9 {rvp9_skip} / libvpx {libvpx_skip} | {count} |"
                        )?;
                    }
                }
            }
            const BLOCK_SIZE_LABELS: [&str; 13] = [
                "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16", "32x32",
                "32x64", "64x32", "64x64",
            ];
            for (label, count) in BLOCK_SIZE_LABELS
                .iter()
                .zip(costs.exact_input_skip_mismatch_by_bsize)
            {
                if count != 0 {
                    writeln!(sf, "| exact-input skip disagreement {label} | {count} |")?;
                }
            }
            writeln!(
                sf,
                "| exact source/reference inputs | {} |",
                costs.exact_input_records
            )?;
            writeln!(
                sf,
                "| exact-input candidate selected by rvp9 | {} |",
                costs.exact_input_selected_records
            )?;
            writeln!(
                sf,
                "| selected exact-input final filter matches | {} |",
                costs.exact_input_selected_filter_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx model variance | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_var_sum,
                    costs.exact_input_libvpx_var_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx model SSE | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_sse_sum,
                    costs.exact_input_libvpx_sse_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input model TX-size matches | {} |",
                costs.exact_input_tx_size_matches
            )?;
            writeln!(
                sf,
                "| exact-input DC-threshold matches | {} |",
                costs.exact_input_dc_thr_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx DC threshold | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_dc_thr_sum,
                    costs.exact_input_libvpx_dc_thr_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input AC-threshold matches | {} |",
                costs.exact_input_ac_thr_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx AC threshold | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_ac_thr_sum,
                    costs.exact_input_libvpx_ac_thr_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input normalized-variance matches | {} |",
                costs.exact_input_var_tx_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx normalized variance | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_var_tx_sum,
                    costs.exact_input_libvpx_var_tx_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input normalized-SSE matches | {} |",
                costs.exact_input_sse_tx_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx normalized SSE | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_sse_tx_sum,
                    costs.exact_input_libvpx_sse_tx_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input DC-dequant matches | {} |",
                costs.exact_input_dc_quant_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx DC dequant | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_dc_quant_sum,
                    costs.exact_input_libvpx_dc_quant_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input AC-dequant matches | {} |",
                costs.exact_input_ac_quant_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx AC dequant | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_ac_quant_sum,
                    costs.exact_input_libvpx_ac_quant_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input DC round-fp matches | {} |",
                costs.exact_input_dc_round_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx DC round-fp | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_dc_round_sum,
                    costs.exact_input_libvpx_dc_round_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input AC round-fp matches | {} |",
                costs.exact_input_ac_round_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx AC round-fp | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_ac_round_sum,
                    costs.exact_input_libvpx_ac_round_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input DC quant-fp matches | {} |",
                costs.exact_input_fp_dc_quant_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx DC quant-fp | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_fp_dc_quant_sum,
                    costs.exact_input_libvpx_fp_dc_quant_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input AC quant-fp matches | {} |",
                costs.exact_input_fp_ac_quant_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx AC quant-fp | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_fp_ac_quant_sum,
                    costs.exact_input_libvpx_fp_ac_quant_sum
                )
            )?;
            writeln!(
                sf,
                "| exact-input model skip-state matches | {} |",
                costs.exact_input_model_skip_matches
            )?;
            writeln!(
                sf,
                "| exact-input residual-hash matches | {} |",
                costs.exact_input_residual_hash_matches
            )?;
            writeln!(
                sf,
                "| exact-input Hadamard coefficient-hash matches | {} |",
                costs.exact_input_had_coeff_hash_matches
            )?;
            writeln!(
                sf,
                "| exact-input quantized coefficient-hash matches | {} |",
                costs.exact_input_qcoeff_hash_matches
            )?;
            writeln!(
                sf,
                "| exact-input quantized nonzero-count matches | {} |",
                costs.exact_input_qcoeff_nonzero_matches
            )?;
            writeln!(
                sf,
                "| exact-input rvp9/libvpx quantized nonzero count | {:.6} |",
                ratio(
                    costs.exact_input_rvp9_qcoeff_nonzero_sum,
                    costs.exact_input_libvpx_qcoeff_nonzero_sum
                )
            )?;
            const BLOCK_SIZE_TAGS: [&str; 13] = [
                "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16", "32x32",
                "32x64", "64x32", "64x64",
            ];
            for (bsize_idx, ((rvp9_count, libvpx_count), matching_count)) in costs
                .exact_input_rvp9_coeff_trace_by_bsize
                .iter()
                .zip(costs.exact_input_libvpx_coeff_trace_by_bsize)
                .zip(costs.exact_input_matching_coeff_trace_by_bsize)
                .enumerate()
            {
                if *rvp9_count != 0 || libvpx_count != 0 {
                    writeln!(
                        sf,
                        "| exact-input coefficient traces {} rvp9 / libvpx / matching | {} / {} / {} |",
                        BLOCK_SIZE_TAGS[bsize_idx], rvp9_count, libvpx_count, matching_count
                    )?;
                }
            }
            for rvp9_skip in 0..2 {
                for libvpx_skip in 0..2 {
                    let count = costs.exact_input_yrd_skip_pair_counts[rvp9_skip * 2 + libvpx_skip];
                    if count != 0 {
                        writeln!(
                            sf,
                            "| exact-input block-YRD skippable rvp9 {rvp9_skip} / libvpx {libvpx_skip} | {count} |"
                        )?;
                    }
                }
            }
            for rvp9_skip in 0..3 {
                for libvpx_skip in 0..3 {
                    let count = costs.exact_input_skip_pair_counts[rvp9_skip * 3 + libvpx_skip];
                    if count != 0 {
                        writeln!(
                            sf,
                            "| exact-input skip pair rvp9 {rvp9_skip} / libvpx {libvpx_skip} | {count} |"
                        )?;
                    }
                }
            }
            writeln!(sf)?;
        }
        if let Some(stages) = matching_newmv_search_stages {
            writeln!(sf)?;
            writeln!(sf, "### Exact-leaf NEWMV search stages")?;
            writeln!(sf)?;
            writeln!(sf, "| stage | matching leaves | compared leaves |")?;
            writeln!(sf, "|---|---:|---:|")?;
            writeln!(
                sf,
                "| full-pel start | {} | {} |",
                stages.start_matches, stages.records
            )?;
            writeln!(
                sf,
                "| full-pel result | {} | {} |",
                stages.fullpel_matches, stages.records
            )?;
            writeln!(
                sf,
                "| final fractional result | {} | {} |",
                stages.final_matches, stages.records
            )?;
            writeln!(
                sf,
                "| second causal MV reference | {} | {} |",
                stages.near_mv_matches, stages.records
            )?;
            writeln!(
                sf,
                "| superblock predictor input | {} | {} |",
                stages.sb_pred_mv_matches, stages.records
            )?;
            writeln!(
                sf,
                "| predictor-SAD winner | {} | {} |",
                stages.mv_pred_mv_matches, stages.records
            )?;
            writeln!(sf)?;
            writeln!(
                sf,
                "Same-start/full-pel-diverged: {}; same-full-pel/final-diverged: {}.",
                stages.same_start_fullpel_diverged, stages.same_fullpel_final_diverged
            )?;
            writeln!(sf)?;
            writeln!(
                sf,
                "| equal-fullpel input | matching leaves | equal-fullpel leaves |"
            )?;
            writeln!(sf, "|---|---:|---:|")?;
            writeln!(
                sf,
                "| reference MV | {} | {} |",
                stages.same_fullpel_ref_mv_matches, stages.same_fullpel_records
            )?;
            writeln!(
                sf,
                "| error-per-bit | {} | {} |",
                stages.same_fullpel_error_per_bit_matches, stages.same_fullpel_records
            )?;
            writeln!(
                sf,
                "| source-pixel hash | {} | {} |",
                stages.same_fullpel_source_hash_matches, stages.same_fullpel_records
            )?;
            writeln!(
                sf,
                "| reference-footprint hash | {} | {} |",
                stages.same_fullpel_reference_hash_matches, stages.same_fullpel_records
            )?;
            writeln!(
                sf,
                "| final MV on exact complete inputs | {} | {} |",
                stages.exact_input_final_matches, stages.exact_input_records
            )?;
            if stages.exact_input_subpel_records > 0 {
                writeln!(sf)?;
                writeln!(
                    sf,
                    "| exact-input subpel field | matching leaves | traced leaves |"
                )?;
                writeln!(sf, "|---|---:|---:|")?;
                for (name, matches) in [
                    ("derived MV limits", stages.subpel_limit_matches),
                    ("search policy", stages.subpel_policy_matches),
                    (
                        "center pixel distortion",
                        stages.subpel_center_distortion_matches,
                    ),
                    (
                        "center MV syntax cost",
                        stages.subpel_center_mv_cost_matches,
                    ),
                    (
                        "half-pel candidate coordinates",
                        stages.subpel_half_coordinate_matches,
                    ),
                    (
                        "half-pel pixel costs",
                        stages.subpel_half_distortion_matches,
                    ),
                    (
                        "half-pel MV syntax costs",
                        stages.subpel_half_mv_cost_matches,
                    ),
                    ("half-pel winner", stages.subpel_half_winner_matches),
                    (
                        "quarter-pel candidate coordinates",
                        stages.subpel_quarter_coordinate_matches,
                    ),
                    (
                        "quarter-pel pixel costs",
                        stages.subpel_quarter_distortion_matches,
                    ),
                    (
                        "quarter-pel MV syntax costs",
                        stages.subpel_quarter_mv_cost_matches,
                    ),
                    ("quarter-pel winner", stages.subpel_quarter_winner_matches),
                ] {
                    writeln!(
                        sf,
                        "| {name} | {matches} | {} |",
                        stages.exact_input_subpel_records
                    )?;
                }
            }
        }
        writeln!(
            sf,
            "| frame_idx | rvp9 n_blocks | libvpx n_blocks | paired matches | mismatches |"
        )?;
        writeln!(sf, "|---|---|---|---|---|")?;
        let mut rvp9_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &rvp9_analysis_records {
            *rvp9_by_frame.entry(d.frame_idx).or_default() += 1;
        }
        let mut vpx_by_frame = std::collections::BTreeMap::<u32, usize>::new();
        for d in &libvpx_analysis_records {
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

fn write_partition_traces(
    path: &Path,
    traces: &[MlPartitionTrace],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for trace in traces {
        trace.write_tsv(&mut file)?;
    }
    Ok(())
}

fn report_partition_trace_comparison(
    rvp9_path: &Path,
    libvpx_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let rvp9 = std::fs::read_to_string(rvp9_path)?;
    let libvpx = std::fs::read_to_string(libvpx_path)?;
    let rvp9_lines = rvp9.lines().collect::<Vec<_>>();
    let libvpx_lines = libvpx.lines().collect::<Vec<_>>();
    let mut compared = 0usize;
    let mut key = 0usize;
    let mut dc_q = 0usize;
    let mut source = 0usize;
    let mut predictor = 0usize;
    let mut variance = 0usize;
    let mut features = 0usize;
    let mut scores = 0usize;
    let mut decisions = 0usize;
    let mut first_causal_mismatch = None;
    for (rvp9, libvpx) in rvp9_lines.iter().zip(&libvpx_lines) {
        let rvp9 = rvp9.split('\t').collect::<Vec<_>>();
        let libvpx = libvpx.split('\t').collect::<Vec<_>>();
        if rvp9.len() != 21 || libvpx.len() != 21 {
            return Err("ML partition trace record did not contain 21 fields".into());
        }
        compared += 1;
        key += usize::from(rvp9[..5] == libvpx[..5]);
        dc_q += usize::from(rvp9[5] == libvpx[5]);
        source += usize::from(rvp9[6] == libvpx[6]);
        predictor += usize::from(rvp9[7] == libvpx[7]);
        variance += usize::from(rvp9[8..13] == libvpx[8..13]);
        features += usize::from(rvp9[13..19] == libvpx[13..19]);
        scores += usize::from(rvp9[19] == libvpx[19]);
        decisions += usize::from(rvp9[20] == libvpx[20]);
        if first_causal_mismatch.is_none() && (rvp9[..19] != libvpx[..19] || rvp9[20] != libvpx[20])
        {
            first_causal_mismatch = Some((rvp9.join("\t"), libvpx.join("\t")));
        }
    }
    println!(
        "  ML partition trace: {compared} paired nodes; rvp9 {} / libvpx {} total",
        rvp9_lines.len(),
        libvpx_lines.len()
    );
    println!(
        "    key {key}/{compared}, dc_q {dc_q}/{compared}, source {source}/{compared}, predictor {predictor}/{compared}, variance {variance}/{compared}, features {features}/{compared}, score bits {scores}/{compared}, decisions {decisions}/{compared}"
    );
    if let Some((rvp9, libvpx)) = first_causal_mismatch {
        println!("    first causal rvp9 : {rvp9}");
        println!("    first causal libvpx: {libvpx}");
    }
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
    let (yuv_frames, source_fps) = read_cropped(
        foreman_path,
        encode.width,
        encode.height,
        encode.frames as usize,
    )?;
    let fps = oracle_fps(encode, source_fps);
    let cfg = oracle_encoder_config(encode, strict, fps, None);
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

/// Drive the real packet-level spatial encoder and retain all component
/// records. Analysis filters them only after layer/frame/Q validation.
fn drive_rvp9_spatial_leg(
    foreman_path: &Path,
    encode: OracleEncodeConfig,
    strict: bool,
    ivf_path: &Path,
    fixed_q: Option<u8>,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let (yuv_frames, source_fps) = read_cropped(
        foreman_path,
        encode.width,
        encode.height,
        encode.frames as usize,
    )?;
    let fps = oracle_fps(encode, source_fps);
    let config = oracle_encoder_config(encode, strict, fps, fixed_q);
    let spatial = SpatialSvcConfig {
        number_spatial_layers: encode.spatial_layers,
        inter_layer_pred: inter_layer_pred(encode)?,
    };
    let mut encoder =
        SpatialEncoder::new_with_parallelism(config, spatial, oracle_parallelism(encode))?;
    let mut all_records = Vec::new();
    let mut packets = Vec::with_capacity(yuv_frames.len());
    let _ = f02_oracle::drain_frame();
    for (y, u, v) in &yuv_frames {
        let packet = encoder.encode_access_unit(&YuvFrameRef {
            y,
            u,
            v,
            y_stride: encode.width as usize,
            u_stride: encode.width as usize / 2,
            v_stride: encode.width as usize / 2,
        })?;
        if let Some(expected_q) = fixed_q
            && packet
                .layers
                .iter()
                .any(|layer| layer.q_index != expected_q)
        {
            return Err(format!(
                "rvp9 spatial access unit {} did not stay at fixed qindex {expected_q}: {:?}",
                encoder.access_unit_index() - 1,
                packet
                    .layers
                    .iter()
                    .map(|layer| layer.q_index)
                    .collect::<Vec<_>>()
            )
            .into());
        }
        packets.push(packet.data);
        all_records.extend(f02_oracle::drain_frame());
    }
    let mut ivf = File::create(ivf_path)?;
    let header = IvfHeader::vp9(
        encode.width as u16,
        encode.height as u16,
        fps.round() as u32,
        1,
    );
    let packet_refs = packets.iter().map(Vec::as_slice).collect::<Vec<_>>();
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
    append_libvpx_rate_args(&mut cmd, encode.vbr, fixed_q);
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

/// Drive pinned libvpx's spatial example over the same cropped top-layer
/// frames as rvp9. The F02-only environment gate preserves the explicit
/// per-layer quantizer bounds under CBR.
fn drive_libvpx_spatial_leg(
    anchor: &Path,
    foreman: &Path,
    encode: OracleEncodeConfig,
    run_dir: &Path,
    fixed_q: Option<u8>,
) -> Result<Vec<BlockDecision>, Box<dyn std::error::Error>> {
    let sidecar = run_dir.join(format!("foreman_{}kbps_libvpx.jsonl", encode.kbps));
    let input = run_dir.join(format!("foreman_{}kbps_spatial_input.y4m", encode.kbps));
    let output = run_dir.join(format!("foreman_{}kbps_libvpx.ivf", encode.kbps));
    let (cropped, source_fps) =
        read_cropped(foreman, encode.width, encode.height, encode.frames as usize)?;
    let fps = oracle_fps(encode, source_fps).round() as u32;
    {
        let mut file = File::create(&input)?;
        write_header(
            &mut file,
            encode.width,
            encode.height,
            fps,
            1,
            ChromaFormat::Yuv420,
        )?;
        for (y, u, v) in &cropped {
            write_frame(&mut file, y, u, v)?;
        }
    }

    let mut command = Command::new(anchor);
    let partition_sidecar =
        run_dir.join(format!("foreman_{}kbps_partition_libvpx.tsv", encode.kbps));
    command
        .env("VPX_F02_ORACLE_OUT", &sidecar)
        .env("VPX_F02_PARTITION_ORACLE_OUT", &partition_sidecar)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    append_libvpx_spatial_args(&mut command, &input, &output, encode, fixed_q)?;
    let result = command.output()?;
    let _ = std::fs::remove_file(&input);
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!(
            "vp9_spatial_svc_encoder failed (exit {}): {}",
            result.status.code().unwrap_or(-1),
            stderr.lines().take(8).collect::<Vec<_>>().join("\n")
        )
        .into());
    }
    if !sidecar.exists() {
        return Err(format!(
            "libvpx spatial sidecar at {} was not produced; rebuild with scripts/build-f02-libvpx.sh",
            sidecar.display()
        )
        .into());
    }
    let records = parse_jsonl_stream(BufReader::new(File::open(&sidecar)?))
        .map_err(|error| format!("libvpx spatial sidecar parse failed: {error}"))?;
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
            spatial_layers: 1,
            inter_layer_pred: 0,
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
    fn explicit_cbr_retains_fixed_q_calibration() {
        let args = vec!["--cbr".to_owned()];
        assert!(!parse_vbr(&args, Some(148)).unwrap());

        let mut command = Command::new("vpxenc");
        append_libvpx_rate_args(&mut command, false, Some(148));
        let arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            arguments,
            [
                "--disable-warning-prompt",
                "--end-usage=cbr",
                "--min-q=37",
                "--max-q=37"
            ]
        );
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

    #[test]
    fn spatial_oracle_command_matches_canonical_three_layer_anchor() {
        let mut encode = test_encode_config();
        encode.width = 1280;
        encode.height = 720;
        encode.frames = 2;
        encode.cpu_used = 8;
        encode.kbps = 420;
        encode.vbr = false;
        encode.spatial_layers = 3;
        encode.inter_layer_pred = 0;

        assert_eq!(spatial_layer_bitrates(420, 3).unwrap(), "20,80,320");
        assert_eq!(spatial_layer_geometry(1280, 720, 3, 0).unwrap(), (320, 180));

        let mut command = Command::new("vp9_spatial_svc_encoder");
        append_libvpx_spatial_args(
            &mut command,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            encode,
            Some(148),
        )
        .unwrap();
        let arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            arguments,
            [
                "--frames=2",
                "--output=output.ivf",
                "--target-bitrate=420",
                "--speed=8",
                "--threads=1",
                "--aqmode=0",
                "--rc-end-usage=1",
                "--spatial-layers=3",
                "--temporal-layers=1",
                "--kf-dist=60",
                "--lag-in-frames=0",
                "--scale-factors=1/4,1/2,1/1",
                "--bitrates=20,80,320",
                "--min-q=37,37,37",
                "--max-q=37,37,37",
                "--tune-content=0",
                "--inter-layer-pred=0",
                "input.y4m",
            ]
        );
        assert!(command.get_envs().any(|(key, value)| {
            key == "VPX_F02_PRESERVE_LAYER_QUANTIZERS" && value.is_some_and(|value| value == "1")
        }));
        assert!(spatial_analysis_records(&[], 3, 0, 2, Some(148), "empty").is_err());
    }

    #[test]
    fn report_only_spatial_oracle_accepts_natural_cbr() {
        assert!(validate_spatial_oracle_mode(2, false, None, 0, true).is_ok());
        assert!(validate_spatial_oracle_mode(2, false, None, 0, false).is_err());
        assert!(validate_spatial_oracle_mode(2, true, None, 0, true).is_err());
        assert!(validate_spatial_oracle_mode(2, false, None, 3, true).is_err());
    }

    #[test]
    fn fixed_q_spatial_oracle_accepts_vbr_picker_calibration() {
        assert!(validate_spatial_oracle_mode(3, true, Some(220), 0, false).is_ok());
    }

    #[test]
    fn natural_cbr_spatial_anchor_uses_canonical_quantizer_bounds() {
        let mut encode = test_encode_config();
        encode.width = 1920;
        encode.height = 1080;
        encode.frames = 2;
        encode.cpu_used = 6;
        encode.kbps = 7_560;
        encode.vbr = false;
        encode.spatial_layers = 2;

        let mut command = Command::new("vp9_spatial_svc_encoder");
        append_libvpx_spatial_args(
            &mut command,
            Path::new("input.y4m"),
            Path::new("output.ivf"),
            encode,
            None,
        )
        .unwrap();
        let arguments = command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(arguments.contains(&"--min-q=2,2".to_owned()));
        assert!(arguments.contains(&"--max-q=56,56".to_owned()));
        assert!(
            !command
                .get_envs()
                .any(|(key, value)| key == "VPX_F02_PRESERVE_LAYER_QUANTIZERS" && value.is_some())
        );
    }

    #[test]
    fn natural_cbr_spatial_rvp9_uses_canonical_quantizer_bounds() {
        let mut encode = test_encode_config();
        encode.width = 1920;
        encode.height = 1080;
        encode.vbr = false;
        encode.spatial_layers = 2;

        let config = oracle_encoder_config(encode, false, 30.0, None);
        assert_eq!(config.min_qindex, 8);
        assert_eq!(config.max_qindex, 224);

        let fixed = oracle_encoder_config(encode, false, 30.0, Some(4));
        assert_eq!(fixed.min_qindex, 4);
        assert_eq!(fixed.max_qindex, 4);
    }

    #[test]
    fn spatial_analysis_layer_selects_its_coded_geometry() {
        assert_eq!(spatial_layer_geometry(1280, 720, 3, 0).unwrap(), (320, 180));
        assert_eq!(spatial_layer_geometry(1280, 720, 3, 1).unwrap(), (640, 360));
        assert_eq!(
            spatial_layer_geometry(1280, 720, 3, 2).unwrap(),
            (1280, 720)
        );
        assert!(spatial_layer_geometry(1280, 720, 3, 3).is_err());
    }
}
